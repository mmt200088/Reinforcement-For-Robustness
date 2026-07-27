"""4-sub-stage Stage-2 RL orchestrator.

Trains BLB Stage-2 RL as four sequential sub-stages, one per block in the
order ``1 → 2 → 4 → 5``. Block 3 is held frozen at the ``static_skeletons``
baseline for the entire run. Each sub-stage:

  1. Builds a :class:`BLBStage2SubstageEnv` over the shared base env, with
     ``frozen_base_action_vec`` pre-filled with the previous sub-stages'
     validated picks (and baseline values for blocks not yet trained).
  2. Calibrates a noisy preflight against the frozen base to measure
     ``acc_{k-1}``, then sets
     ``acc_threshold = max(acc_{k-1} * (1 - tol/N), acc_orig * (1 - tol))``.
     This is the "progressive re-baseline" budget allocation chosen during
     grilling: earlier sub-stages get a tighter threshold so they leave
     accuracy budget for later sub-stages.
  3. Trains a fresh :class:`BLBStage2SequentialPolicy` (sized to the
     sub-stage horizon, not the legacy 59 steps) via :func:`train_sequential`
     for ``substage_episodes_each`` episodes.
  4. Picks the top-K candidates by hard-priority rank key from training and
     runs ``promotion_validation_trials`` extra K-trial probes per candidate.
     The arg-max (with priority still P3) becomes the sub-stage's validated
     best.
  5. Splices the validated best back into ``frozen_base`` and continues.

After sub-stage 4, the assembled full 577-dim vec is written into
``ev.noise_rl_metadata['blb_v3_best_action_vec']`` so the existing auto
final-eval + GLUE-submission chain (``embedded.run_embedded_final_eval``)
picks it up unchanged.

Resume granularity is "episode within the current sub-stage": the top-level
``substage_progress.json`` records ``current_substage_idx`` and a per-sub-stage
checkpoint lives in ``substage_<k>_block<n>/`` (mirrors the legacy sequential
path's checkpoint layout, so the existing graceful-stop machinery in
``train_sequential`` works as-is).

The legacy single-shot ``BLBStage2RLRunner.run`` and the per-block
``run_sequential_via_runner`` paths are untouched -- this module is opt-in
via ``--blb-v3-substage-mode 1``.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is required at runtime
    torch = None  # type: ignore


SUBSTAGE_PROGRESS_FILENAME = "substage_progress.json"
SUBSTAGE_RL_VARIANT = "blb_v3_substage_gtrxl_v2scale"
DEFAULT_BLOCK_ORDER: Tuple[int, ...] = (1, 2, 4, 5)
DEFAULT_FROZEN_BLOCKS: Tuple[int, ...] = (3,)


# ---------------------------------------------------------------------------
# Progress file (top-level, one JSON for the whole 4-substage run)
# ---------------------------------------------------------------------------
def _substage_progress_path(blb_progress_dir: str) -> str:
    return os.path.join(str(blb_progress_dir), SUBSTAGE_PROGRESS_FILENAME)


def load_substage_progress(blb_progress_dir: str) -> Optional[Dict[str, Any]]:
    path = _substage_progress_path(blb_progress_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        return data
    except Exception:
        return None


def save_substage_progress(blb_progress_dir: str, payload: Mapping[str, Any]) -> str:
    os.makedirs(str(blb_progress_dir), exist_ok=True)
    path = _substage_progress_path(blb_progress_dir)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(dict(payload), f, indent=2, sort_keys=True, ensure_ascii=False)
    os.replace(tmp, path)
    return path


def substage_subdir(blb_progress_dir: str, substage_index: int, block_idx: int) -> str:
    name = f"substage_{int(substage_index) + 1}_block{int(block_idx)}"
    return os.path.join(str(blb_progress_dir), name)


# ---------------------------------------------------------------------------
# Budget allocator: "progressive re-baseline" (chosen during grilling)
# ---------------------------------------------------------------------------
def compute_substage_acc_threshold(
        *,
        acc_orig: float,
        acc_after_prev: float,
        stage2_limit_tolerance: float,
        num_substages: int,
        probe_size: int,
        ) -> float:
    """Threshold for sub-stage k.

    ``threshold = max(acc_after_prev * (1 - tol/N), acc_orig * (1 - tol))``

    ``stage2_limit_tolerance`` is a relative fraction, so 0.001 means 0.1%.
    ``probe_size`` is kept in the signature for compatibility with older
    callers, but it no longer relaxes the gate.
    """
    _ = probe_size
    n = max(1, int(num_substages))
    tol = max(0.0, float(stage2_limit_tolerance))
    per_stage = tol / n
    hard_floor = float(acc_orig) * (1.0 - tol)
    progressive = float(acc_after_prev) * (1.0 - per_stage)
    return float(max(progressive, hard_floor))


# ---------------------------------------------------------------------------
# Rank key (mirrors blb_stage2_rl.reward.priority semantics):
# (invalid_steps == 0, terminal_priority desc, terminal_reward desc)
# Returns a tuple suitable for ``max(...)`` / ``sorted(..., reverse=True)``.
# ---------------------------------------------------------------------------
def _rank_key_from_record(record: Any) -> Tuple[int, int, float, float]:
    invalid_steps = int(getattr(record, "invalid_steps", 0) or 0)
    priority = int(getattr(record, "terminal_priority", 0) or 0)
    reward = float(getattr(record, "total_reward", 0.0) or 0.0)
    cost_rank = float(getattr(record, "terminal_cost_rank_score", 0.0) or 0.0)
    return (-invalid_steps, priority, cost_rank, reward)


# ---------------------------------------------------------------------------
# Top-K promotion validation: re-run each candidate ``num_trials`` times via
# the env's normal step path. Picks the validated best by rank key.
# ---------------------------------------------------------------------------
def promote_top_k(
        *,
        base_env: Any,
        candidates: Sequence[Tuple[Tuple[int, int, float, float], np.ndarray]],
        top_k: int,
        num_trials: int,
        log: Callable[[str], None],
        ) -> Tuple[Optional[np.ndarray], List[Dict[str, Any]]]:
    """Re-evaluate top-K candidates with K-trial probe.

    Args:
        candidates: list of ``(rank_key, action_vec)`` from training.
        top_k: keep only the top-K by training rank key.
        num_trials: probes per candidate. The env's
            ``num_trials_per_step`` is temporarily overridden so each step
            uses ``num_trials`` independent noise seeds.

    Returns:
        ``(best_action_vec or None, validation_records list)``. ``best_action_vec``
        is ``None`` only when zero candidates pass at validation time (very
        unlikely for healthy training).
    """
    if not candidates:
        return None, []
    ranked = sorted(candidates, key=lambda x: x[0], reverse=True)[: int(max(1, top_k))]
    # Snapshot + override base_env's K so we get statistically stable estimates.
    original_k = int(getattr(base_env.env_cfg, "num_trials_per_step", 1) or 1)
    base_env.env_cfg.num_trials_per_step = int(max(1, num_trials))
    validation_records: List[Dict[str, Any]] = []
    try:
        best_vec: Optional[np.ndarray] = None
        best_key: Tuple[int, int, float, float] = (-(10**9), -(10**9), -float("inf"), -float("inf"))
        for ri, (train_key, vec) in enumerate(ranked):
            try:
                _state, reward, _done, info = base_env.step(np.asarray(vec, dtype=np.int64))
            except Exception as exc:
                log(f"  [substage][promote] candidate {ri+1}/{len(ranked)} bridge_error: {exc}")
                validation_records.append({
                    "rank_train": ri,
                    "train_key": list(train_key),
                    "bridge_error": str(exc),
                    "skipped": True,
                })
                continue
            metrics = info.get("metrics") if isinstance(info, Mapping) else None
            priority = int(info.get("priority", 0)) if isinstance(info, Mapping) else 0
            invalid_steps = 0  # full vec; if any block was invalid, env returns priority 1 / penalty
            cost_rank = float(info.get("terminal_cost_rank_score", 0.0) or 0.0)
            validated_key = (-invalid_steps, priority, cost_rank, float(reward))
            log(
                f"  [substage][promote] cand {ri+1}: train_key={train_key} → "
                f"validated_key={validated_key} reward={float(reward):.4f}"
            )
            validation_records.append({
                "rank_train": ri,
                "train_key": list(train_key),
                "validated_key": list(validated_key),
                "reward": float(reward),
                "priority": int(priority),
                "metrics": _safe_metrics_dict(metrics),
                "action_vec_head": [int(x) for x in np.asarray(vec, dtype=int)[:24].tolist()],
            })
            if validated_key > best_key:
                best_key = validated_key
                best_vec = np.asarray(vec, dtype=np.int64).copy()
    finally:
        base_env.env_cfg.num_trials_per_step = int(original_k)
    return best_vec, validation_records


def _safe_metrics_dict(metrics: Any) -> Dict[str, float]:
    if metrics is None:
        return {}
    out: Dict[str, float] = {}
    for attr in ("loss_mean", "loss_std", "metric1_mean", "metric1_std",
                 "metric2_mean", "metric2_std"):
        try:
            out[attr] = float(getattr(metrics, attr, 0.0))
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------------
# Frozen base update: copy active block's slot indices from validated_best
# back into a baseline-derived frozen_base.
# ---------------------------------------------------------------------------
def splice_block_slots_into_base(
        *,
        frozen_base: np.ndarray,
        validated_best: np.ndarray,
        active_block_idx: int,
        substage_schedule: Sequence[Any],
        ) -> np.ndarray:
    """Return a new frozen_base with the active block's slots overwritten
    by ``validated_best``.

    ``substage_schedule`` is the substage env's ``_schedule`` -- each spec's
    ``full_vec_offsets`` already point into the 577-dim layout, so we just
    copy those positions.
    """
    new_base = np.asarray(frozen_base, dtype=np.int64).copy()
    vb = np.asarray(validated_best, dtype=np.int64).reshape(-1)
    for spec in substage_schedule:
        for off in spec.full_vec_offsets:
            new_base[int(off)] = int(vb[int(off)])
    return new_base


# ---------------------------------------------------------------------------
# Orchestrator entry point (called from BLBStage2RLRunner.run when
# train_cfg.substage_mode == True)
# ---------------------------------------------------------------------------
def run_substage_via_runner(
        *,
        runner: Any,
        train_cfg: Any,
        fixed_gelu: np.ndarray,
        fixed_softmax: np.ndarray,
        fixed_label: str,
        fixed_source: str,
        resume_checkpoint_path: Optional[str] = None,
        ) -> Dict[str, Any]:
    """Top-level 4-sub-stage driver. Mirrors the surface of
    :func:`sequential_runner.run_sequential_via_runner` so the existing dispatch
    in :class:`BLBStage2RLRunner` can swap implementations cleanly.
    """
    # Lazy imports to keep the module torch-light at import time.
    from .baseline_bootstrap import (
        load_static_skeletons_baseline,
        static_skeletons_baseline_to_action,
    )
    from .env import BLBStage2Env, BLBStage2EnvConfig, estimate_baseline_cost_stats
    from .reward import BaselineCostStats, RewardWeights, calibrate_weights_from_baseline
    from .persistence import (
        BLBStatusBoard,
        write_training_curves,
    )
    from .runner import resolve_blb_persistence_dir
    from .schedule_geometry import schedule_max_num_levels
    from .sequential_env import SequentialEnvConfig
    from .sequential_policy import (
        BLBStage2SequentialPolicy,
        SequentialPolicyConfig,
        SequentialPPOConfig,
    )
    from .sequential_runner import (
        SequentialTrainConfig,
        train_sequential,
    )
    from .substage_env import BLBStage2SubstageEnv
    from .truncation_levels import (
        CHECKPOINT_K_DOMAIN_KEY,
        K_LEVELS,
        checkpoint_k_domain_contract,
        validate_exact_k_domain,
        validate_checkpoint_k_domain,
    )

    validate_exact_k_domain(K_LEVELS)
    if torch is None:
        raise RuntimeError("torch is required to run the substage path")

    ev = runner.evaluator
    log = runner._make_log_safe(ev.log)
    bullet = "*"

    # ----- Persistent dir + status -----
    blb_progress_dir = resolve_blb_persistence_dir(ev)
    try:
        ev.noise_stage_progress_dir = blb_progress_dir
    except Exception:
        pass
    os.makedirs(blb_progress_dir, exist_ok=True)

    log("\n" + "=" * 80)
    log("【阶段 5：BLB Stage-2 噪声 RL · 4-sub-stage 模式（block 1→2→4→5；block 3 frozen）】")
    log("=" * 80)
    log(f"  {bullet} BLB 持久化目录：{blb_progress_dir}")
    log(f"  {bullet} 固定 GELU 阶数：{np.asarray(fixed_gelu, dtype=int).tolist()}")
    log(f"  {bullet} 固定 Softmax 阶数：{np.asarray(fixed_softmax, dtype=int).tolist()}")
    log(f"  {bullet} profile={train_cfg.profile!r}    seed={int(train_cfg.seed)}")

    # ----- Static skeletons baseline -----
    fixed_gelu_arr = np.asarray(fixed_gelu, dtype=int)
    fixed_softmax_arr = np.asarray(fixed_softmax, dtype=int)
    ev.apply_configuration(fixed_gelu_arr, fixed_softmax_arr)
    try:
        ev.reversible_handler.restore_layer_input_noise(
            layer_indices=list(range(ev.total_layers)),
        )
    except Exception:
        pass
    probe_batches = runner._build_probe_batches(ev, train_cfg)
    train_cfg.probe_batch_count = max(1, int(len(probe_batches) or train_cfg.probe_batch_count))
    rescale_bridge = runner._build_rescale_bridge(train_cfg, log=log)

    ss_baseline_obj = load_static_skeletons_baseline(
        rescale_optimizer_root=str(train_cfg.inproc_rescale_optimizer_root),
        dataset=str(train_cfg.profile),
        num_layers=int(ev.total_layers),
        gelu_per_layer=[int(x) for x in fixed_gelu_arr.reshape(-1)],
        softmax_per_layer=[int(x) for x in fixed_softmax_arr.reshape(-1)],
    )
    ss_action_vec, max_sfs, ss_cost_stats, _ = static_skeletons_baseline_to_action(
        ss_baseline_obj, snap_sf_to_noise_table=False,
    )
    baseline_action_vec = np.asarray(ss_action_vec, dtype=np.int64).reshape(-1)
    log(f"  {bullet} static_skeletons baseline ← {ss_baseline_obj.archive_path}")

    # ----- Base env (shared across sub-stages) -----
    base_env = BLBStage2Env(
        handler=ev.reversible_handler,
        model=ev.model,
        probe_batches=probe_batches,
        rescale_bridge=rescale_bridge,
        baseline=BaselineCostStats(),
        reward_weights=RewardWeights(),
        acc_threshold=train_cfg.acc_threshold,
        stab_threshold=train_cfg.stab_threshold,
        max_sfs=max_sfs,
        num_layers=int(ev.total_layers),
        gelu_degree=fixed_gelu_arr,
        attn_degree=fixed_softmax_arr,
        layers_attribute="model." + ev.layers_attribute,
        is_regression=bool(getattr(ev, "is_regression", False)),
        env_cfg=BLBStage2EnvConfig(
            profile=train_cfg.profile,
            num_trials_per_step=train_cfg.num_trials_per_step,
            probe_batch_count=train_cfg.probe_batch_count,
            truncation_backend=train_cfg.truncation_backend,
            truncation_ring_bits=train_cfg.truncation_ring_bits,
            truncation_source_fractional_bits=(
                train_cfg.truncation_source_fractional_bits
            ),
        ),
    )
    base_env.pareto_cost_archive = None
    base_env.sync_degree_vectors_from_model()

    # Multi-GPU probe runner (mirrors sequential_runner's setup)
    reward_devices = list(getattr(train_cfg, "reward_devices", []) or [])
    if reward_devices and len(reward_devices) >= 2:
        from .probe_runner import build_probe_runner
        log(f"  [multi-gpu] reward probe enabled: devices={reward_devices}")
        base_env.probe_runner = build_probe_runner(
            primary_model=ev.model,
            primary_handler=ev.reversible_handler,
            primary_bridge=base_env.bridge,
            primary_probe_batches=base_env.probe_batches,
            layers_attribute="model." + ev.layers_attribute,
            is_regression=bool(getattr(ev, "is_regression", False)),
            device_ids=reward_devices,
            metric_profile=str(train_cfg.profile),
            log_fn=lambda m: log(f"  [multi-gpu] {m}"),
        )

    # ----- Baseline cost stats + clean preflight -----
    precomputed = {
        "total_bits_sum": int(ss_cost_stats.total_bits_sum),
        "total_fusion_count": int(ss_cost_stats.total_fusion_count),
        "avg_k": float(ss_cost_stats.avg_k),
    }
    baseline = estimate_baseline_cost_stats(
        base_env,
        sample_count=int(train_cfg.calibrate_baseline_samples),
        precomputed_baseline_signals=precomputed,
    )
    base_env.baseline = baseline
    baseline_metrics = runner._estimate_baseline_metrics(base_env)
    baseline.loss_mean = float(baseline_metrics.loss_mean)
    baseline.loss_std = float(baseline_metrics.loss_std)
    baseline.metric1_mean = float(baseline_metrics.metric1_mean)
    baseline.metric2_mean = float(baseline_metrics.metric2_mean)
    baseline.metric1_std = float(getattr(baseline_metrics, "metric1_std", 0.0) or 0.0)
    baseline.metric2_std = float(getattr(baseline_metrics, "metric2_std", 0.0) or 0.0)
    baseline_clean_metric1 = float(baseline_metrics.metric1_mean)
    calibrate_weights_from_baseline(base_env.reward_weights, baseline)
    log(
        f"  {bullet} clean baseline: m1={baseline_clean_metric1:.4f}  "
        f"loss_std={baseline.loss_std:.4f}  total_bits_sum={int(baseline.total_bits_sum)}"
    )

    # ----- Original noisy preflight (acc_orig) ------------------------------
    # Drive one step of base_env at the baseline action vec to measure noisy
    # metrics with this Stage-1 fixed_gelu/softmax. This is acc_orig — the
    # hard floor referenced by the budget allocator.
    log(f"  {bullet} noisy baseline preflight @ static_skeletons baseline …")
    try:
        _s, _r, _d, preflight_info = base_env.step(baseline_action_vec)
        m = preflight_info.get("metrics") if isinstance(preflight_info, Mapping) else None
        acc_orig = float(getattr(m, "metric1_mean", baseline_clean_metric1) if m else baseline_clean_metric1)
        loss_std_orig = float(getattr(m, "loss_std", baseline.loss_std) if m else baseline.loss_std)
    except Exception as exc:
        log(f"  [warn] noisy baseline preflight failed: {exc}; using clean m1 as acc_orig")
        acc_orig = baseline_clean_metric1
        loss_std_orig = baseline.loss_std
    log(f"  {bullet} acc_orig={acc_orig:.4f}  loss_std_orig={loss_std_orig:.4f}")

    # ----- OSR pre-prune (opt-in) --------------------------------------------
    # If train_cfg.osr_results_path is set we either load existing results or
    # run a fresh scan into the same path. The resulting mask is passed to
    # train_sequential alongside the existing 3 masks (see osr.py prologue
    # for the design discussion). ``osr_scan_only=True`` exits after scan.
    osr_mask = None
    osr_results_path = str(getattr(train_cfg, "osr_results_path", "") or "")
    if osr_results_path:
        from .osr import prepare_osr_mask
        stage2_limit_tolerance_pre = float(
            getattr(ev, "stage2_limit_tolerance", 0.05) or 0.05
        )
        stage2_stab_tolerance_pre = float(
            getattr(ev, "stage2_stability_tolerance", 0.05) or 0.05
        )
        osr_mask, scan_only_exit = prepare_osr_mask(
            base_env=base_env,
            results_path=osr_results_path,
            scan_only=bool(getattr(train_cfg, "osr_scan_only", False)),
            num_combo_samples=int(getattr(train_cfg, "osr_num_combo_samples", 300) or 300),
            allow_fingerprint_mismatch=bool(
                getattr(train_cfg, "osr_allow_fingerprint_mismatch", False)
            ),
            num_layers=int(ev.total_layers),
            profile=str(train_cfg.profile),
            stage1_gelu_per_layer=[int(x) for x in fixed_gelu_arr.reshape(-1)],
            stage1_softmax_per_layer=[int(x) for x in fixed_softmax_arr.reshape(-1)],
            baseline_action_vec=baseline_action_vec,
            acc_orig=float(acc_orig),
            loss_std_orig=float(loss_std_orig),
            k_trials=int(train_cfg.num_trials_per_step),
            probe_size=int(getattr(ev, "stage2_probe_size", 256) or 256),
            tol_acc=stage2_limit_tolerance_pre,
            tol_stab=stage2_stab_tolerance_pre,
            log_fn=log,
        )
        if osr_mask is not None:
            log(f"  {bullet} OSR mask active: {osr_mask.summary()}")
        if scan_only_exit:
            log(f"  {bullet} osr_scan_only=True → exiting after scan")
            return {
                "status": "completed",
                "osr_results_path": osr_results_path,
                "osr_summary": osr_mask.summary() if osr_mask else "",
                "blb_v3_best_action_vec": None,
                "rl_variant": SUBSTAGE_RL_VARIANT,
                "substage_diagnostics": {"osr_scan_only": True},
            }

    # ----- Resume vs fresh -----
    progress = load_substage_progress(blb_progress_dir) or {}
    block_order: List[int] = list(
        getattr(train_cfg, "substage_block_order", None) or DEFAULT_BLOCK_ORDER
    )
    frozen_blocks_cfg: List[int] = list(
        getattr(train_cfg, "substage_frozen_blocks", None) or DEFAULT_FROZEN_BLOCKS
    )
    episodes_each: int = int(
        getattr(train_cfg, "substage_episodes_each", 15000) or 15000
    )
    promotion_top_k: int = int(
        getattr(train_cfg, "substage_promotion_top_k", 5) or 5
    )
    promotion_trials: int = int(
        getattr(train_cfg, "substage_promotion_trials", 8) or 8
    )
    stage2_limit_tolerance: float = float(
        getattr(ev, "stage2_limit_tolerance", 0.05) or 0.05
    )

    if not progress:
        progress = {
            "block_order": block_order,
            "frozen_blocks": frozen_blocks_cfg,
            "current_substage_idx": 0,
            "completed_substages": [],
            "acc_orig": float(acc_orig),
            "loss_std_orig": float(loss_std_orig),
            "stage2_limit_tolerance": float(stage2_limit_tolerance),
            "acc_after_each_substage": [float(acc_orig)],
            "episodes_each": int(episodes_each),
            "promotion_top_k": int(promotion_top_k),
            "promotion_trials": int(promotion_trials),
            "frozen_base_action_vec": baseline_action_vec.tolist(),
            "rl_variant": SUBSTAGE_RL_VARIANT,
            "static_skeletons_archive": str(ss_baseline_obj.archive_path),
            "profile": str(train_cfg.profile),
            "seed": int(train_cfg.seed),
            "fixed_label": str(fixed_label),
            "fixed_source": str(fixed_source),
        }
    else:
        log(
            f"  {bullet} resume detected: substage idx={progress.get('current_substage_idx', 0)} "
            f"completed={len(progress.get('completed_substages', []))}"
        )

    frozen_base = np.asarray(
        progress.get("frozen_base_action_vec", baseline_action_vec.tolist()),
        dtype=np.int64,
    ).reshape(-1)
    acc_after_each = list(progress.get("acc_after_each_substage", [float(acc_orig)]))

    save_substage_progress(blb_progress_dir, progress)

    # ----- Sub-stage loop ----------------------------------------------------
    n_substages = len(block_order)
    device = next(iter(ev.model.parameters())).device if hasattr(ev.model, "parameters") else torch.device("cpu")

    for k in range(int(progress.get("current_substage_idx", 0)), n_substages):
        active_block = int(block_order[k])
        subdir = substage_subdir(blb_progress_dir, k, active_block)
        os.makedirs(subdir, exist_ok=True)
        log("\n" + "-" * 70)
        log(f"  [substage {k+1}/{n_substages}] block={active_block}  subdir={subdir}")
        log(f"  [substage] frozen base sum_k=...  (acc_orig={acc_orig:.4f})")

        # ---- 1) Build substage env over the shared base_env ----
        substage_env = BLBStage2SubstageEnv(
            base_env=base_env,
            active_block_idx=active_block,
            frozen_base_action_vec=frozen_base,
            env_cfg=SequentialEnvConfig(),
        )

        # ---- 2) Re-baseline acc & set sub-stage threshold ----
        if k == 0:
            acc_prev = float(acc_orig)
        else:
            # Drive one step at the current frozen_base to measure noisy acc.
            try:
                _s, _r, _d, info = base_env.step(frozen_base)
                m = info.get("metrics") if isinstance(info, Mapping) else None
                acc_prev = float(getattr(m, "metric1_mean", acc_after_each[-1]) if m else acc_after_each[-1])
            except Exception as exc:
                log(f"  [warn] re-baseline preflight failed: {exc}; using prev acc")
                acc_prev = float(acc_after_each[-1])
        new_threshold = compute_substage_acc_threshold(
            acc_orig=acc_orig,
            acc_after_prev=acc_prev,
            stage2_limit_tolerance=stage2_limit_tolerance,
            num_substages=n_substages,
            probe_size=int(train_cfg.probe_batch_count) * int(getattr(train_cfg, "probe_size", 256) or 256),
        )
        base_env.acc_threshold = float(new_threshold)
        log(
            f"  [substage {k+1}] threshold → acc_prev={acc_prev:.4f}  "
            f"tol/{n_substages}={stage2_limit_tolerance/n_substages:.4f}  "
            f"hard_floor={acc_orig * (1.0 - stage2_limit_tolerance):.4f}  "
            f"acc_threshold={new_threshold:.4f}"
        )

        # ---- 3) Fresh GTrXL policy sized to substage horizon ----
        pol_cfg = SequentialPolicyConfig(
            state_dim=int(substage_env.state_dim),
            max_step_dim=int(substage_env._max_step_dim),
            max_num_levels=schedule_max_num_levels(substage_env.schedule),
            horizon=int(substage_env.horizon),
            num_layers=int(ev.total_layers),
            block_count=6,
        )
        policy = BLBStage2SequentialPolicy(pol_cfg).to(device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=float(train_cfg.ppo.lr))

        # ---- 4) Resume sub-stage checkpoint if present ----
        ck_path = os.path.join(subdir, "blb_stage2_rl_checkpoint_live.pt")
        start_ep = 0
        if os.path.exists(ck_path):
            try:
                ckpt = torch.load(ck_path, map_location=device)
            except Exception as exc:
                raise RuntimeError(
                    f"failed to read substage checkpoint {ck_path}; "
                    "a fresh run is required"
                ) from exc
            validate_checkpoint_k_domain(
                ckpt,
                context=f"substage checkpoint {ck_path}",
            )
            try:
                policy.load_state_dict(ckpt["policy_state_dict"])
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as exc:
                raise RuntimeError(
                    f"substage checkpoint {ck_path} contains incompatible policy "
                    "or optimizer state; a fresh run is required"
                ) from exc
            start_ep = int(ckpt.get("completed_episodes", 0) or 0)
            log(f"  [substage {k+1}] resume from ep={start_ep}")

        # ---- 5) Collect candidates via callbacks ----
        candidates: List[Tuple[Tuple[int, int, float, float], np.ndarray]] = []
        episode_returns: List[float] = []
        # Mutable cell: on_step_end captures env._pending_full_vec on the
        # terminal step (EpisodeRecord doesn't carry the action vec itself).
        last_terminal_vec_holder: List[Optional[np.ndarray]] = [None]

        def _on_step_end(_ep_idx: int, step_within: int, _info: Dict[str, Any]) -> None:
            if int(step_within) == int(substage_env.horizon - 1):
                try:
                    last_terminal_vec_holder[0] = np.asarray(
                        substage_env._pending_full_vec, dtype=np.int64,
                    ).copy()
                except Exception:
                    last_terminal_vec_holder[0] = None

        def _on_episode_end(record: Any) -> None:
            episode_returns.append(float(getattr(record, "total_reward", 0.0) or 0.0))
            vec = last_terminal_vec_holder[0]
            last_terminal_vec_holder[0] = None
            if vec is None:
                return
            key = _rank_key_from_record(record)
            candidates.append((key, vec))
            # Avoid unbounded memory growth: only keep top 50 by rank key.
            if len(candidates) > 50:
                candidates.sort(key=lambda x: x[0], reverse=True)
                del candidates[50:]

        def _on_ppo_update_end(_metrics: Dict[str, float], completed_ep: int, _last: Any) -> None:
            try:
                torch.save({
                    "policy_state_dict": policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "completed_episodes": int(completed_ep) + int(start_ep),
                    "substage_index": int(k),
                    "active_block": int(active_block),
                    "rl_variant": SUBSTAGE_RL_VARIANT,
                    CHECKPOINT_K_DOMAIN_KEY: checkpoint_k_domain_contract(),
                }, ck_path)
            except Exception as exc:
                log(f"  [warn] substage {k+1} checkpoint save failed: {exc}")

        # ---- 6) Train ----
        ppo_cfg = SequentialPPOConfig(
            lr=float(train_cfg.ppo.lr),
            clip_range=float(train_cfg.ppo.clip_range),
            n_epochs=int(train_cfg.ppo.n_epochs),
            minibatch_size=int(train_cfg.ppo.minibatch_size),
            ent_coef=float(train_cfg.ppo.ent_coef),
            value_coef=float(train_cfg.ppo.value_coef),
            max_grad_norm=float(train_cfg.ppo.max_grad_norm),
        )
        remaining = max(0, int(episodes_each) - int(start_ep))
        seq_cfg = SequentialTrainConfig(
            total_episodes=int(remaining),
            update_every_n_episodes=max(1, int(train_cfg.rollout_size)),
            log_every_n_episodes=max(1, int(train_cfg.rollout_size)),
            seed=int(train_cfg.seed) + 1001 * int(k),
            ppo=ppo_cfg,
            ent_coef_anchor=float(getattr(train_cfg, "ent_coef_anchor", 0.0)),
            ent_coef_ramp_episodes=int(getattr(train_cfg, "ent_coef_ramp_episodes", 600)),
            absolute_episode_start=int(start_ep),
            # Per-substage anchor: short forced-baseline so the fresh policy
            # has a warm start. Keep modest (default 30) since horizon=12.
            fast_reward_mode_enabled=bool(getattr(train_cfg, "fast_reward_mode_enabled", False)),
            online_num_trials_per_step=int(getattr(train_cfg, "online_num_trials_per_step", 5)),
            terminal_eval_batch_size=int(getattr(train_cfg, "terminal_eval_batch_size", 1)),
            promotion_validation_trials=int(promotion_trials),
        )
        # Anchor: each substage's GTrXL starts fresh, so a short forced anchor
        # gets it past the cold-start collapse seen in the legacy 60k smoke.
        anchor = max(0, int(getattr(train_cfg, "force_baseline_episodes", 30) or 30))
        try:
            train_sequential(
                env=substage_env,
                policy=policy,
                train_cfg=seq_cfg,
                device=device,
                optimizer=optimizer,
                on_episode_end=_on_episode_end,
                on_ppo_update_end=_on_ppo_update_end,
                on_step_end=_on_step_end,
                baseline_action_vec=baseline_action_vec,
                max_rejection_retries=int(getattr(train_cfg, "max_rejection_retries", 32) or 32),
                force_baseline_episodes=anchor,
                osr_mask=osr_mask,
            )
        except KeyboardInterrupt:
            log(f"  [substage {k+1}] KeyboardInterrupt — saving and exiting")
            _on_ppo_update_end({}, len(episode_returns), None)
            save_substage_progress(blb_progress_dir, progress)
            raise

        # ---- 7) Top-K promotion validation ----
        log(f"  [substage {k+1}] training done; running top-{promotion_top_k} promotion validation "
            f"(R={promotion_trials} trials each)")
        validated_best, validation_records = promote_top_k(
            base_env=base_env,
            candidates=candidates,
            top_k=int(promotion_top_k),
            num_trials=int(promotion_trials),
            log=log,
        )
        if validated_best is None:
            # Defensive fallback: use the frozen base itself (no-op for this block).
            log(f"  [substage {k+1}] no validated candidate; keeping frozen base for this block")
            validated_best = frozen_base.copy()

        # ---- 8) Splice validated best into frozen_base ----
        new_frozen = splice_block_slots_into_base(
            frozen_base=frozen_base,
            validated_best=validated_best,
            active_block_idx=active_block,
            substage_schedule=substage_env._schedule,
        )
        frozen_base = new_frozen

        # ---- 9) Measure new acc after this substage ----
        try:
            _s, _r, _d, info = base_env.step(frozen_base)
            m = info.get("metrics") if isinstance(info, Mapping) else None
            acc_after = float(getattr(m, "metric1_mean", acc_prev) if m else acc_prev)
        except Exception as exc:
            log(f"  [warn] post-substage measurement failed: {exc}; using acc_prev")
            acc_after = float(acc_prev)
        acc_after_each.append(float(acc_after))
        log(f"  [substage {k+1}] acc_after = {acc_after:.4f}  (acc_orig={acc_orig:.4f}, "
            f"hard_floor={acc_orig - stage2_limit_tolerance:.4f})")

        # ---- 10) Write per-substage paper curve + diagnostics ----
        try:
            write_training_curves(
                subdir,
                episode_returns=episode_returns,
                log_fn=log,
            )
        except Exception as exc:
            log(f"  [warn] substage {k+1} curve write failed: {exc}")

        # ---- 11) Persist progress ----
        progress["current_substage_idx"] = int(k + 1)
        progress["acc_after_each_substage"] = [float(x) for x in acc_after_each]
        progress["frozen_base_action_vec"] = [int(x) for x in frozen_base.tolist()]
        progress.setdefault("completed_substages", []).append({
            "substage_index": int(k),
            "active_block": int(active_block),
            "subdir": subdir,
            "episodes_completed": int(len(episode_returns) + int(start_ep)),
            "acc_after": float(acc_after),
            "threshold_used": float(new_threshold),
            "validated_best_action_vec_head": [int(x) for x in validated_best[:24].tolist()],
            "validation_records": validation_records,
        })
        save_substage_progress(blb_progress_dir, progress)

    # ----- 12) Done: assemble final action vec ------------------------------
    # The downstream auto-final-eval path reads ``blb_v3_best_action_vec`` from
    # the dict we return below. The legacy single-shot + sequential paths both
    # return additional legacy-compat fields (best_noise_config, fixed_gelu,
    # etc.) so callers don't crash. We mirror that.
    final_action_vec = np.asarray(frozen_base, dtype=np.int64).reshape(-1)
    log(f"  [substage] DONE: assembled final 577-dim vec; len={final_action_vec.size}")

    # ----- 13) Top-level paper curve with all sub-stages combined -----
    try:
        all_returns: List[float] = []
        boundaries: List[int] = []
        labels: List[str] = []
        for entry in progress.get("completed_substages", []):
            sub_subdir = entry.get("subdir") or substage_subdir(
                blb_progress_dir, int(entry["substage_index"]), int(entry["active_block"]),
            )
            npz_path = os.path.join(sub_subdir, "blb_stage2_training_curve.npz")
            if os.path.exists(npz_path):
                try:
                    z = np.load(npz_path)
                    rets = list(np.asarray(z["episode_returns"], dtype=float).tolist())
                except Exception:
                    rets = []
            else:
                rets = []
            if rets:
                if all_returns:
                    boundaries.append(len(all_returns))
                    labels.append(f"sub-stage {int(entry['substage_index']) + 1}: block {int(entry['active_block'])}")
                all_returns.extend(rets)
        write_training_curves(
            blb_progress_dir,
            episode_returns=all_returns,
            substage_boundaries=boundaries,
            substage_labels=labels,
            log_fn=log,
        )
    except Exception as exc:
        log(f"  [warn] top-level paper curve failed: {exc}")

    # ----- 14) Assemble legacy-compatible result dict -----------------------
    from .runner import _build_legacy_compatible_best_noise_config
    cost_reference_noise_config = ev._get_max_noise_configuration()
    try:
        cost_reference_tot_c, _ = ev.get_noise_simulated_cost(**cost_reference_noise_config)
    except Exception:
        cost_reference_tot_c = 0.0
    legacy_best = _build_legacy_compatible_best_noise_config(ev)
    try:
        base_loss, base_p, base_s, _ = ev.evaluate_model(
            fixed_gelu_arr, fixed_softmax_arr, use_train=False,
            split=ev.get_reward_reference_split_name(),
        )
        limit_dict = ev.build_constraint_limits_from_metrics(base_loss, base_p, base_s)
        limit_loss = float(limit_dict["loss"])
        limit_p = float(limit_dict["metric1"])
        limit_s = float(limit_dict["metric2"])
    except Exception as exc:
        log(f"  [warn] post-train evaluate_model failed: {exc}; using zero limits")
        base_loss = base_p = base_s = 0.0
        limit_loss = limit_p = limit_s = 0.0

    return {
        "fixed_gelu": fixed_gelu_arr.copy(),
        "fixed_softmax": fixed_softmax_arr.copy(),
        "baseline_noise_config": {k: v.copy() for k, v in cost_reference_noise_config.items()},
        "baseline_tot_c": float(cost_reference_tot_c),
        "cost_reference_noise_config": {k: v.copy() for k, v in cost_reference_noise_config.items()},
        "cost_reference_source": "max_noise_configuration",
        "performance_baseline_gelu": fixed_gelu_arr.copy(),
        "performance_baseline_softmax": fixed_softmax_arr.copy(),
        "performance_baseline_source": "stage1_fixed_low_risk_noise",
        "k_trials": int(train_cfg.num_trials_per_step),
        "probe_size": int(getattr(ev, "stage2_probe_size", 256)),
        "limit_loss": float(limit_loss),
        "limit_p": float(limit_p),
        "limit_s": float(limit_s),
        "proxy_limit_loss": float(limit_loss),
        "proxy_limit_p": float(limit_p),
        "proxy_limit_s": float(limit_s),
        "proxy_base_loss": float(base_loss),
        "proxy_base_p": float(base_p),
        "proxy_base_s": float(base_s),
        "raw_model_baseline_metrics": {
            "loss": float(base_loss),
            "metric1": float(base_p),
            "metric2": float(base_s),
        },
        "search_limits": {"loss": float(limit_loss), "metric1": float(limit_p), "metric2": float(limit_s)},
        "status": "completed",
        "training_eval_split": str(ev.get_reward_reference_split_name()),
        "best_noise_config": {k: v.copy() for k, v in legacy_best.items()},
        "stable_search_best_noise_config": {k: v.copy() for k, v in legacy_best.items()},
        "stable_joint_best_noise_config": {k: v.copy() for k, v in legacy_best.items()},
        "blb_v3_best_action_vec": [int(x) for x in final_action_vec.tolist()],
        "blb_v3_best_reward": 0.0,
        "blb_v3_profile": str(train_cfg.profile),
        "blb_v3_total_episodes": int(episodes_each) * len(block_order),
        "rl_variant": SUBSTAGE_RL_VARIANT,
        "substage_diagnostics": {
            "acc_orig": float(acc_orig),
            "acc_after_each_substage": [float(x) for x in acc_after_each],
            "stage2_limit_tolerance": float(stage2_limit_tolerance),
            "block_order": list(block_order),
            "frozen_blocks": list(frozen_blocks_cfg),
            "episodes_each": int(episodes_each),
            "promotion_top_k": int(promotion_top_k),
            "promotion_trials": int(promotion_trials),
        },
        "substage_progress": dict(progress),
    }
