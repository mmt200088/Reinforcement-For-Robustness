"""Stage-2 episode-parallel rollout (fusion-count mode only, 2026-06-10).

Mirrors the validated ``stage1_rl/parallel_runner.py`` pattern at the same
granularity: N workers (one per GPU) each run *complete* episodes — GTrXL
policy rollout on their own policy replica, per-step ``ReplanSession`` calls,
and the terminal K-trial reward probe run *serially* on their own model
replica — while one PPO learner on the primary device consumes the rollouts
in GLOBAL episode order. This replaces (for fusion mode) the older K-split
scheme where N GPUs only parallelized the K probe trials of one action
(~2.9x on 5 GPUs); whole-episode parallelism removes the serial GTrXL
rollout + install/bookkeeping from the critical path and decouples K from
the GPU count.

GPU-count-independence contract (the user's hard requirement):

* every random draw is keyed by the GLOBAL episode index via
  ``blb_stage2_rl/seed_utils.py`` — policy sampling per (episode, step,
  attempt) on the worker device's CUDA Philox generator (device-independent
  streams), probe noise per (episode, trial) via
  ``function_handler.reseed_noise_rng_for_device`` (replacing the legacy
  wall-clock/eval-counter seed that made runs irreproducible);
* episode→worker assignment is balanced contiguous chunks of global indices
  (``assign_global_episodes``) and results are returned in GLOBAL order, so
  the PPO update sees the same buffer for any worker count;
* fusion mode has no cross-episode mask / frontier-seed state (the offline
  fusion map is all-valid; per-slot masks are disabled), which is what makes
  strict 1-GPU == N-GPU achievable at this granularity. If the optimizer
  ever reports an invalid fusion action (broken map), the worker falls back
  to baseline and logs loudly — that anomaly is the one case that can break
  bit-identity and must be investigated.

Scope guard: per-slot (non-fusion) mode keeps the legacy serial loop and the
K-split ``--blb-v3-reward-devices`` path untouched.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import contextlib
import hashlib
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from function_handler import ReversibleLayerHandler

# Reusable no-op context for the uncontended (1 worker/device) path.
_NULL_LOCK = contextlib.nullcontext()

from .action_space import K_LEVELS, _baseline_k_index_for_block
from .env import BLBStage2Env
from .fusion_curriculum import (
    build_fusion_step_level_mask,
    fusion_block_curriculum,
    fusion_probe_target_block,
    select_mutable_step_indices,
)
from .probe_runner import (
    _move_probe_batch_to_device,
    enable_cuda_reward_probe_fast_math,
)
from .seed_utils import (
    assign_global_episodes,
    derive_policy_step_seed,
    derive_probe_seed,
)
from .sequential_env import BLBStage2SequentialEnv
from .sequential_policy import step_to_mask_and_levels
from .sequential_runner import (
    EpisodeRecord,
    _open_step_level_mask,
    _resolve_baseline_prior_scale,
)

try:
    from blb_rl_bridge import BLBNoiseRLBridge
except Exception:  # pragma: no cover — torch-free import path
    BLBNoiseRLBridge = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Episode outcome — everything the main thread needs to replay one episode
# ---------------------------------------------------------------------------

@dataclass
class FusionEpisodeOutcome:
    """One worker-collected episode, ready for main-thread assembly.

    ``transitions`` entries are kwargs for ``SequentialRolloutBuffer.add``
    (numpy arrays + python scalars only — no CUDA tensors cross threads).
    """
    rel_ep: int
    absolute_ep: int
    transitions: List[Dict[str, Any]]
    record: EpisodeRecord
    pending_full_vec: Optional[np.ndarray]
    invalid_anomaly: bool = False


# ---------------------------------------------------------------------------
# Terminal-info snapshot (mirror of the inline extraction in train_sequential)
# ---------------------------------------------------------------------------

_BREAKDOWN_FIELDS = (
    ("terminal_priority", "priority", int, 0),
    ("terminal_stab_excess_m1", "stab_excess_m1", float, 0.0),
    ("terminal_stab_excess_m2", "stab_excess_m2", float, 0.0),
    ("terminal_stab_excess_loss", "stab_excess_loss", float, 0.0),
    ("terminal_stab_violation", "stab_violation", float, 0.0),
    ("terminal_bits_gain", "bits_drop", float, 0.0),
    ("terminal_k_gain", "k_drop", float, 0.0),
    ("terminal_fusion_gain", "fusion_gain", float, 0.0),
    ("terminal_cost_score", "cost_score", float, 0.0),
    ("terminal_p3_metric_margin_reward", "p3_metric_margin_reward", float, 0.0),
    # ADR-014 DEBUG: barrier/margin (mirror serial _apply_terminal_info_to_record).
    ("terminal_worst_signed_margin", "worst_signed_margin", float, 0.0),
    ("terminal_acc_barrier_sat", "acc_barrier_sat", float, 0.0),
    ("terminal_acc_barrier_vio", "acc_barrier_vio", float, 0.0),
    ("terminal_near_miss", "near_miss", bool, False),
    ("terminal_margin_m1", "margin_m1", float, 0.0),
    ("terminal_margin_m2", "margin_m2", float, 0.0),
    ("terminal_cost_fusion_bonus", "cost_fusion_bonus", float, 0.0),
    ("terminal_cost_truncation_bonus", "cost_truncation_bonus", float, 0.0),
    ("terminal_cost_bits_tiebreaker", "cost_bits_tiebreaker", float, 0.0),
    ("terminal_cost_truncation_step_gain", "cost_truncation_step_gain", float, 0.0),
    ("terminal_cost_rank_score", "cost_rank_score", float, 0.0),
    ("terminal_cost_rank_fusion", "cost_rank_fusion", float, 0.0),
    ("terminal_cost_rank_truncation", "cost_rank_truncation", float, 0.0),
    ("terminal_cost_rank_bits", "cost_rank_bits", float, 0.0),
    ("terminal_pareto_event_kind", "pareto_event_kind", str, ""),
    ("terminal_pareto_action_hash", "pareto_action_hash", str, ""),
    ("terminal_pareto_frontier_removed", "pareto_frontier_removed", int, 0),
)

_METRIC_FIELDS = (
    ("terminal_loss_mean", "loss_mean"),
    ("terminal_loss_std", "loss_std"),
    ("terminal_metric1_mean", "metric1_mean"),
    ("terminal_metric2_mean", "metric2_mean"),
    ("terminal_metric1_std", "metric1_std"),
    ("terminal_metric2_std", "metric2_std"),
)


def _update_terminal_snapshot(snapshot: Dict[str, Any], info: Dict[str, Any]) -> None:
    """Overwrite ``snapshot`` from ``info['terminal_info']`` when present.

    Field-for-field mirror of the inline extraction in
    ``train_sequential`` (sequential_runner.py) so EpisodeRecords from the
    parallel path are indistinguishable from serial ones.
    """
    term_info_dict = info.get("terminal_info") or {}
    # 2026-06-13: per-block-type fusion split (diagnostic), mirrored into
    # terminal_info by sequential_env so the parallel snapshot matches serial.
    if "fusion_count_b2" in term_info_dict:
        snapshot["fusion_count_b2"] = int(term_info_dict.get("fusion_count_b2", 0) or 0)
        snapshot["fusion_count_b4"] = int(term_info_dict.get("fusion_count_b4", 0) or 0)
        snapshot["fusion_count_b5"] = int(term_info_dict.get("fusion_count_b5", 0) or 0)
    # ADR-014 DEBUG: fusion cost shape (raw vs saturated), mirror of serial path.
    if "fusion_cost_fusion_norm" in term_info_dict:
        snapshot["terminal_fusion_norm_raw"] = float(
            term_info_dict.get("fusion_cost_fusion_norm", 0.0) or 0.0
        )
        snapshot["terminal_fusion_norm_saturated"] = float(
            term_info_dict.get("fusion_cost_fusion_norm_saturated", 0.0) or 0.0
        )
    term_breakdown = term_info_dict.get("reward_breakdown")
    term_metrics = term_info_dict.get("metrics")
    term_probe_diag = term_info_dict.get("probe_diagnostics") or {}
    if term_breakdown is not None:
        for rec_key, attr, cast, default in _BREAKDOWN_FIELDS:
            snapshot[rec_key] = cast(getattr(term_breakdown, attr, default) or default)
    if term_metrics is not None:
        for rec_key, attr in _METRIC_FIELDS:
            snapshot[rec_key] = float(getattr(term_metrics, attr, 0.0) or 0.0)
    if isinstance(term_probe_diag, dict) and term_probe_diag:
        snapshot["terminal_probe_wall_seconds"] = float(
            term_probe_diag.get("wall_seconds", 0.0) or 0.0
        )
        snapshot["terminal_probe_devices"] = [
            str(x) for x in (term_probe_diag.get("devices") or [])
        ]
        snapshot["terminal_probe_trial_counts"] = [
            int(x) for x in (term_probe_diag.get("per_worker_trial_counts") or [])
        ]
        snapshot["terminal_probe_trial_indices"] = [
            [int(y) for y in (x or [])]
            for x in (term_probe_diag.get("per_worker_trial_indices") or [])
        ]
        snapshot["terminal_probe_speedup"] = float(
            term_probe_diag.get("speedup_vs_sequential", 1.0) or 1.0
        )
        snapshot["terminal_cost_eval_wall_seconds"] = float(
            term_probe_diag.get("cost_eval_wall_seconds", 0.0) or 0.0
        )
        snapshot["terminal_probe_install_wall_seconds"] = float(
            term_probe_diag.get("probe_install_wall_seconds", 0.0) or 0.0
        )
        snapshot["terminal_probe_clear_wall_seconds"] = float(
            term_probe_diag.get("probe_clear_wall_seconds", 0.0) or 0.0
        )
        snapshot["terminal_probe_install_skipped"] = bool(
            term_probe_diag.get("probe_install_skipped", False)
        )
        snapshot["terminal_probe_clear_skipped"] = bool(
            term_probe_diag.get("probe_clear_skipped", False)
        )


def _default_terminal_snapshot() -> Dict[str, Any]:
    snap: Dict[str, Any] = {}
    for rec_key, _attr, cast, default in _BREAKDOWN_FIELDS:
        snap[rec_key] = cast(default)
    for rec_key, _attr in _METRIC_FIELDS:
        snap[rec_key] = 0.0
    snap.update(
        terminal_probe_wall_seconds=0.0,
        terminal_probe_devices=[],
        terminal_probe_trial_counts=[],
        terminal_probe_trial_indices=[],
        terminal_probe_speedup=1.0,
        terminal_cost_eval_wall_seconds=0.0,
        terminal_probe_install_wall_seconds=0.0,
        terminal_probe_clear_wall_seconds=0.0,
        terminal_probe_install_skipped=False,
        terminal_probe_clear_skipped=False,
        fusion_count_b2=0,
        fusion_count_b4=0,
        fusion_count_b5=0,
        terminal_fusion_norm_raw=0.0,
        terminal_fusion_norm_saturated=0.0,
    )
    return snap


# ---------------------------------------------------------------------------
# Worker-side fusion episode collection
# ---------------------------------------------------------------------------

def _fusion_episode_generator(
        *,
        seq_env: BLBStage2SequentialEnv,
        policy: Any,
        device: torch.device,
        train_cfg: Any,
        rel_ep: int,
        absolute_ep: int,
        base_seed: int,
        baseline_action_vec: np.ndarray,
        force_baseline_episodes: int,
        forbidden_mask: Any,
        max_rejection_retries: int = 32,
        log_fn: Optional[Callable[[str], None]] = None,
        ):
    """Generator form of ONE fusion-mode episode (2026-06-21 batched rollout).

    Yields the per-step forward inputs ``(obs_t, slot_mask_t, levels_t,
    action_level_mask_t, baseline_prior_scale)`` and receives ``(logits,
    safe_logits, value)`` via ``.send(...)``. Because the GTrXL logits are
    action-INDEPENDENT, ONE yield per step suffices — sampling, the forced
    anchor, the fusion probe, rejection retries and the fallback all reuse the
    same logits (no extra forwards). The DRIVER decides whether that forward runs
    inline (``collect_fusion_episode``, B=1) or batched across episodes
    (``collect_fusion_episodes_batched``, B=W); the per-episode logic here is
    identical either way, which is what makes the two float-equivalent. Sampling
    is the batch-/device-invariant seeded sampler (``policy.sample_from_logits``)
    so no global-RNG device lock is needed; the terminal probe still serializes
    on ``env.probe_device_lock``. Returns the ``FusionEpisodeOutcome`` as the
    generator's ``StopIteration.value``.
    """
    log = log_fn or (lambda _m: None)
    env = seq_env
    assert getattr(env, "_fusion_map", None) is not None, (
        "fusion episode generator requires fusion-count mode"
    )

    obs = env.reset(seed=None)
    # probe_noise_seed is set right before each commit_step (below), NOT here: the
    # batched driver shares ONE base_env across B episodes, so a setup-time write
    # would be clobbered by the last episode's setup. Setting it per-step-before-
    # commit keeps it this episode's seed at the (serial) terminal probe. For the
    # serial driver (dedicated base) the value is identical to the old setup-time
    # write, so serial behavior is unchanged.

    # 2026-06-21: the per-step GTrXL forward is hoisted to the DRIVER (yield
    # below) so it can be batched across episodes. The KV-cache fast path
    # (2026-06-19) is retired — it was the wrong lever (server-measured 0.60x:
    # the H<=59 forward is launch-bound, not FLOP-bound), superseded by the
    # batched forward which amortizes launches across B episodes.

    per_step_sum = 0.0
    terminal_reward = 0.0
    invalid_steps = 0
    valid_step_count = 0
    total_bits_sum = 0
    fusion_count_sum = 0
    first_invalid: Optional[Dict[str, int]] = None
    early_terminated = False
    steps_taken = 0
    invalid_block_details: List[Dict[str, Any]] = []
    per_step_optimizer_wall_seconds_val = 0.0
    policy_rollout_wall_seconds_val = 0.0
    rejection_optimizer_wall_seconds_val = 0.0
    samples_rejected_by_mask = 0
    samples_rejected_by_optimizer = 0
    steps_fallen_back_to_baseline = 0
    invalid_anomaly = False
    snapshot = _default_terminal_snapshot()
    transitions: List[Dict[str, Any]] = []

    baseline_prior_scale = _resolve_baseline_prior_scale(
        int(absolute_ep),
        anchor_episodes=int(force_baseline_episodes),
    )
    force_this_ep = (
        int(force_baseline_episodes) > 0
        and int(absolute_ep) < int(force_baseline_episodes)
    )
    # Scheduled forced-fusion probe (ADR-011, redesigned ADR-012): pure
    # function of the absolute episode index -> identical across workers
    # (1==N preserved). Probe episodes force ONLY the target block type's
    # fusion option to 1; K and all other blocks follow the CURRENT policy
    # under the normal curriculum mask (target option level injected into the
    # mask) — a clean "what if you ALSO fused block-type T" counterfactual.
    fusion_probe_block: Optional[int] = None
    if not force_this_ep:
        fusion_probe_block = fusion_probe_target_block(
            int(absolute_ep),
            anchor_episodes=int(force_baseline_episodes),
            interval=int(getattr(train_cfg, "fusion_probe_interval", 200)),
        )

    # --- fusion block-granularity safe-neighbor curriculum (mirror of the
    # serial loop; fc_rng is already keyed by the absolute episode index) ---
    fusion_curriculum_active = False
    fusion_curriculum_open = False
    fusion_mutable_steps: set = set()
    fusion_neighbor_radius = 1
    if (
            bool(getattr(train_cfg, "fusion_neighbor_curriculum_enabled", True))
            and not force_this_ep
    ):
        fc_seed = int(train_cfg.seed) if train_cfg.seed is not None else 0
        fc_rng = np.random.default_rng(
            int((fc_seed + absolute_ep * 2_654_435_761) % (2**32))
        )
        fc_open, fc_num_mutable, fusion_neighbor_radius = fusion_block_curriculum(
            absolute_episode_idx=int(absolute_ep),
            anchor_episodes=int(force_baseline_episodes),
            ramp_episodes=int(
                getattr(train_cfg, "fusion_neighbor_ramp_episodes", 0)
                or int(train_cfg.total_episodes)
                or 1
            ),
            horizon=int(env.horizon),
            max_radius=int(getattr(train_cfg, "fusion_neighbor_max_radius", 6)),
        )
        fusion_curriculum_active = True
        fusion_curriculum_open = bool(fc_open)
        if not fc_open:
            fusion_mutable_steps = select_mutable_step_indices(
                rng=fc_rng, horizon=int(env.horizon), num_mutable=int(fc_num_mutable),
            )

    while True:
        spec = env.current_spec()
        slot_mask_np, levels_np = step_to_mask_and_levels(
            spec, policy.cfg.max_step_dim, policy.cfg.max_num_levels,
        )
        obs_t = torch.from_numpy(obs).float().to(device).unsqueeze(0)
        slot_mask_t = torch.from_numpy(slot_mask_np).to(device).unsqueeze(0)
        levels_t = torch.from_numpy(levels_np).to(device).unsqueeze(0)
        n_active = int(slot_mask_np.sum())

        if fusion_curriculum_active and not fusion_curriculum_open:
            action_level_mask_np = build_fusion_step_level_mask(
                fusion_num_options=int(spec.fusion_num_options),
                k_num_levels=int(spec.k_num_levels),
                k_level_values=list(K_LEVELS),
                mutable=(int(spec.step_idx) in fusion_mutable_steps),
                radius=int(fusion_neighbor_radius),
                baseline_k_index=int(_baseline_k_index_for_block(int(spec.block_idx))),
                max_step_dim=policy.cfg.max_step_dim,
                max_num_levels=policy.cfg.max_num_levels,
            )
        else:
            action_level_mask_np = _open_step_level_mask(
                spec=spec,
                max_step_dim=policy.cfg.max_step_dim,
                max_num_levels=policy.cfg.max_num_levels,
            )
        if (
                fusion_probe_block is not None
                and int(spec.block_idx) == int(fusion_probe_block)
                and int(spec.fusion_num_options) > 1
        ):
            # ADR-012 probe: make the forced option level legal even when the
            # curriculum pins this block to baseline.
            action_level_mask_np[0, 1] = True
        action_level_mask_t = (
            torch.from_numpy(action_level_mask_np).to(device).unsqueeze(0)
        )

        # ONE action-independent forward per step — the driver runs it (inline
        # for the serial wrapper, batched across episodes for the batched driver)
        # and sends back this row's (masked logits, safe logits, value). All of
        # the sample / forced / probe / rejection / fallback logic below reuses
        # these logits (no further forwards).
        _policy_t0 = time.perf_counter()
        logits_t, safe_logits_t, value_t = yield (
            obs_t, slot_mask_t, levels_t, action_level_mask_t, baseline_prior_scale,
        )
        policy_rollout_wall_seconds_val += float(time.perf_counter() - _policy_t0)

        chosen_action_np: Optional[np.ndarray] = None
        chosen_log_prob = 0.0
        chosen_value = float(value_t.reshape(-1)[0].item())
        chosen_eval_info: Optional[Dict[str, Any]] = None

        if force_this_ep:
            # Forced-baseline anchor: fusion baseline = (option 0, baseline K).
            forced_action = np.asarray(
                [0, int(_baseline_k_index_for_block(spec.block_idx))], dtype=np.int64
            )
            forced_padded = np.zeros(policy.cfg.max_step_dim, dtype=np.int64)
            forced_padded[:n_active] = forced_action[:n_active]
            actions_t = torch.from_numpy(forced_padded).to(device).unsqueeze(0)
            (lp_t,) = policy.logprob_from_logits(
                logits_t, safe_logits_t, slot_mask_t, actions_t,
            )
            chosen_eval_info = env.evaluate_step(forced_action.tolist())
            chosen_action_np = forced_padded
            chosen_log_prob = float(lp_t.item())
        else:
            action_np_try: Optional[np.ndarray] = None
            log_prob_t = value_t = None
            eval_info: Optional[Dict[str, Any]] = None
            for attempt in range(int(max_rejection_retries)):
                seed = derive_policy_step_seed(
                    int(base_seed), int(absolute_ep), int(spec.step_idx), int(attempt),
                )
                # Seeded inverse-CDF sample on the (action-independent) logits —
                # batch-/device-invariant, lock-free (no global manual_seed). A
                # rejection retry just re-draws with attempt+1 from the SAME
                # logits (no re-forward).
                action_t, log_prob_t = policy.sample_from_logits(
                    logits_t, safe_logits_t, slot_mask_t, [int(seed)],
                )
                action_np_try = action_t.squeeze(0).cpu().numpy().astype(np.int64)
                if (
                        fusion_probe_block is not None
                        and int(spec.block_idx) == int(fusion_probe_block)
                        and int(spec.fusion_num_options) > 1
                        and int(action_np_try[0]) != 1
                ):
                    # ADR-012 probe: force ONLY the target block's option to 1;
                    # K keeps the policy's own sample. Re-evaluate
                    # log_prob/value for the modified action under the same
                    # mask so PPO ratios stay well-defined.
                    action_np_try = action_np_try.copy()
                    action_np_try[0] = 1
                    actions_fix_t = (
                        torch.from_numpy(action_np_try).to(device).unsqueeze(0)
                    )
                    (log_prob_t,) = policy.logprob_from_logits(
                        logits_t, safe_logits_t, slot_mask_t, actions_fix_t,
                    )
                step_action_try = action_np_try[:n_active].tolist()
                tup = tuple(int(x) for x in step_action_try)

                if forbidden_mask is not None and forbidden_mask.is_forbidden(
                        spec.layer_idx, spec.block_idx, tup):
                    samples_rejected_by_mask += 1
                    continue

                eval_info = env.evaluate_step(step_action_try)
                if eval_info["valid"]:
                    chosen_action_np = action_np_try
                    chosen_log_prob = float(log_prob_t.item())
                    chosen_eval_info = eval_info
                    break

                # Should be unreachable with an all-valid fusion map: this is
                # the one event that can break 1==N (mask becomes stateful).
                invalid_anomaly = True
                if forbidden_mask is not None:
                    forbidden_mask.add(spec.layer_idx, spec.block_idx, tup)
                samples_rejected_by_optimizer += 1
                rejection_optimizer_wall_seconds_val += float(
                    eval_info.get("optimizer_wall_seconds", 0.0) or 0.0
                )
                log(
                    f"[stage2-parallel][ANOMALY] invalid fusion action at "
                    f"ep={absolute_ep} L{spec.layer_idx}-B{spec.block_idx} "
                    f"tup={tup} — fusion map should be all-valid; "
                    f"1-GPU==N-GPU no longer guaranteed for this run"
                )

            if chosen_eval_info is None:
                fallback_action = np.asarray(
                    [0, int(_baseline_k_index_for_block(spec.block_idx))],
                    dtype=np.int64,
                )
                fallback_padded = np.zeros(policy.cfg.max_step_dim, dtype=np.int64)
                fallback_padded[:n_active] = fallback_action[:n_active]
                actions_t = torch.from_numpy(fallback_padded).to(device).unsqueeze(0)
                (lp_t,) = policy.logprob_from_logits(
                    logits_t, safe_logits_t, slot_mask_t, actions_t,
                )
                chosen_eval_info = env.evaluate_step(fallback_action.tolist())
                chosen_action_np = fallback_padded
                chosen_log_prob = float(lp_t.item())
                steps_fallen_back_to_baseline += 1

        assert chosen_action_np is not None and chosen_eval_info is not None
        step_action_for_env = chosen_action_np[:n_active].tolist()

        # Set this episode's probe noise seed right before commit_step so a
        # SHARED base_env (batched driver) uses THIS row's seed at the (serial)
        # terminal probe — see the note at episode start.
        env.base.probe_noise_seed = derive_probe_seed(int(base_seed), int(absolute_ep))
        next_obs, reward, done, info = env.commit_step(
            chosen_eval_info,
            defer_terminal_forward=False,
        )
        steps_taken += 1
        valid = bool(info.get("valid", True))
        if valid:
            valid_step_count += 1
        else:
            if first_invalid is None:
                first_invalid = {
                    "step": int(info.get("step", steps_taken - 1)),
                    "block_idx": int(info.get("block_idx", 0)),
                    "layer_idx": int(info.get("layer_idx", 0)),
                }
            invalid_block_details.append({
                "step": int(info.get("step", steps_taken - 1)),
                "layer": int(info.get("layer_idx", 0)),
                "block": int(info.get("block_idx", 0)),
                "graph_key": str(info.get("graph_key", "")),
                "reason": "parallel_commit_invalid",
                "rejected_by": "commit",
                "action_tuple": list(int(x) for x in step_action_for_env),
            })
        total_bits_sum += int(info.get("total_bits", 0))
        fusion_count_sum += int(info.get("fusion_count", 0))
        per_step_optimizer_wall_seconds_val += float(
            info.get("optimizer_wall_seconds", 0.0) or 0.0
        )
        if info.get("early_terminated"):
            early_terminated = True

        transitions.append(dict(
            state=obs,
            action=chosen_action_np,
            slot_mask=slot_mask_np,
            per_slot_num_levels=levels_np,
            action_level_mask=action_level_mask_np,
            log_prob=float(chosen_log_prob),
            value=float(chosen_value),
            reward=float(reward),
            done=bool(done),
            baseline_prior_scale=float(baseline_prior_scale),
        ))
        per_step_sum += float(reward)
        if "terminal_reward" in info:
            terminal_reward = float(info["terminal_reward"])
        _update_terminal_snapshot(snapshot, info)

        obs = next_obs
        if done:
            break

    record = EpisodeRecord(
        episode_idx=int(rel_ep),
        total_reward=float(per_step_sum),
        terminal_reward=float(terminal_reward),
        per_step_reward_sum=float(per_step_sum - terminal_reward),
        invalid_steps=int(invalid_steps),
        early_terminated=bool(early_terminated),
        steps_taken=int(steps_taken),
        valid_step_count=int(valid_step_count),
        total_bits_sum_over_steps=int(total_bits_sum),
        fusion_count_sum_over_steps=int(fusion_count_sum),
        fusion_count_b2=int(snapshot["fusion_count_b2"]),
        fusion_count_b4=int(snapshot["fusion_count_b4"]),
        fusion_count_b5=int(snapshot["fusion_count_b5"]),
        first_invalid_step=(int(first_invalid["step"]) if first_invalid else None),
        first_invalid_block=(int(first_invalid["block_idx"]) if first_invalid else None),
        first_invalid_layer=(int(first_invalid["layer_idx"]) if first_invalid else None),
        step_infos=[],
        invalid_block_details=invalid_block_details,
        terminal_priority=int(snapshot["terminal_priority"]),
        terminal_loss_mean=float(snapshot["terminal_loss_mean"]),
        terminal_loss_std=float(snapshot["terminal_loss_std"]),
        terminal_metric1_mean=float(snapshot["terminal_metric1_mean"]),
        terminal_metric2_mean=float(snapshot["terminal_metric2_mean"]),
        terminal_metric1_std=float(snapshot["terminal_metric1_std"]),
        terminal_metric2_std=float(snapshot["terminal_metric2_std"]),
        terminal_stab_excess_m1=float(snapshot["terminal_stab_excess_m1"]),
        terminal_stab_excess_m2=float(snapshot["terminal_stab_excess_m2"]),
        terminal_stab_excess_loss=float(snapshot["terminal_stab_excess_loss"]),
        terminal_stab_violation=float(snapshot["terminal_stab_violation"]),
        terminal_bits_gain=float(snapshot["terminal_bits_gain"]),
        terminal_k_gain=float(snapshot["terminal_k_gain"]),
        terminal_fusion_gain=float(snapshot["terminal_fusion_gain"]),
        terminal_cost_score=float(snapshot["terminal_cost_score"]),
        terminal_p3_metric_margin_reward=float(snapshot["terminal_p3_metric_margin_reward"]),
        terminal_worst_signed_margin=float(snapshot["terminal_worst_signed_margin"]),
        terminal_acc_barrier_sat=float(snapshot["terminal_acc_barrier_sat"]),
        terminal_acc_barrier_vio=float(snapshot["terminal_acc_barrier_vio"]),
        terminal_near_miss=bool(snapshot["terminal_near_miss"]),
        terminal_margin_m1=float(snapshot["terminal_margin_m1"]),
        terminal_margin_m2=float(snapshot["terminal_margin_m2"]),
        terminal_fusion_norm_raw=float(snapshot["terminal_fusion_norm_raw"]),
        terminal_fusion_norm_saturated=float(snapshot["terminal_fusion_norm_saturated"]),
        terminal_cost_fusion_bonus=float(snapshot["terminal_cost_fusion_bonus"]),
        terminal_cost_truncation_bonus=float(snapshot["terminal_cost_truncation_bonus"]),
        terminal_cost_bits_tiebreaker=float(snapshot["terminal_cost_bits_tiebreaker"]),
        terminal_cost_truncation_step_gain=float(snapshot["terminal_cost_truncation_step_gain"]),
        terminal_cost_rank_score=float(snapshot["terminal_cost_rank_score"]),
        terminal_cost_rank_fusion=float(snapshot["terminal_cost_rank_fusion"]),
        terminal_cost_rank_truncation=float(snapshot["terminal_cost_rank_truncation"]),
        terminal_cost_rank_bits=float(snapshot["terminal_cost_rank_bits"]),
        terminal_pareto_event_kind=str(snapshot["terminal_pareto_event_kind"]),
        terminal_pareto_action_hash=str(snapshot["terminal_pareto_action_hash"]),
        terminal_pareto_frontier_removed=int(snapshot["terminal_pareto_frontier_removed"]),
        terminal_probe_wall_seconds=float(snapshot["terminal_probe_wall_seconds"]),
        terminal_probe_devices=list(snapshot["terminal_probe_devices"]),
        terminal_probe_trial_counts=list(snapshot["terminal_probe_trial_counts"]),
        terminal_probe_trial_indices=[list(x) for x in snapshot["terminal_probe_trial_indices"]],
        terminal_probe_speedup=float(snapshot["terminal_probe_speedup"]),
        per_step_optimizer_wall_seconds=float(per_step_optimizer_wall_seconds_val),
        policy_rollout_wall_seconds=float(policy_rollout_wall_seconds_val),
        terminal_cost_eval_wall_seconds=float(snapshot["terminal_cost_eval_wall_seconds"]),
        terminal_probe_install_wall_seconds=float(snapshot["terminal_probe_install_wall_seconds"]),
        terminal_probe_clear_wall_seconds=float(snapshot["terminal_probe_clear_wall_seconds"]),
        terminal_probe_install_skipped=bool(snapshot["terminal_probe_install_skipped"]),
        terminal_probe_clear_skipped=bool(snapshot["terminal_probe_clear_skipped"]),
        safe_neighbor_active=bool(fusion_curriculum_active and not fusion_curriculum_open),
        safe_neighbor_mutation_count=int(len(fusion_mutable_steps)),
        safe_neighbor_radius=int(
            fusion_neighbor_radius
            if (fusion_curriculum_active and not fusion_curriculum_open) else 0
        ),
        exploration_mode=(
            "forced_baseline" if force_this_ep else
            (f"forced_fusion_probe_b{int(fusion_probe_block)}"
             if fusion_probe_block is not None else "radius1")
        ),
        guarded_radius2_active=False,
        guarded_radius2_recent_frontier_expansions=0,
        guarded_radius2_recent_duplicate_rate=0.0,
        guarded_radius2_recent_dominated_rate=0.0,
        guarded_radius2_cooldown_remaining=0,
        guarded_radius2_safe_offset_count=0,
        guarded_radius2_episode_count=0,
        guarded_radius2_failure_count=0,
        guarded_radius2_frontier_expansion_count=0,
        samples_rejected_by_mask=int(samples_rejected_by_mask),
        samples_rejected_by_optimizer=int(samples_rejected_by_optimizer),
        steps_fallen_back_to_baseline=int(steps_fallen_back_to_baseline),
        forbidden_mask_total=int(forbidden_mask.total() if forbidden_mask is not None else 0),
        static_invalid_level_disabled=0,
        static_invalid_level_applied=0,
        static_invalid_level_scan_evaluated=0,
        static_invalid_level_scan_invalid=0,
        empirical_invalid_level_disabled=0,
        empirical_invalid_level_applied=0,
        rejection_optimizer_wall_seconds=float(rejection_optimizer_wall_seconds_val),
        baseline_prior_scale=float(baseline_prior_scale),
        base_action_source=(
            "fusion_probe" if fusion_probe_block is not None else "baseline"
        ),
        proposal_direction=(
            "anchor" if force_this_ep else
            ("fusion_probe" if fusion_probe_block is not None else "none")
        ),
        empirical_offset_success_rate=0.0,
        empirical_offset_failure_rate=0.0,
        frontier_seed_episode=-1,
    )

    pending_full_vec = getattr(env, "_pending_full_vec", None)
    return FusionEpisodeOutcome(
        rel_ep=int(rel_ep),
        absolute_ep=int(absolute_ep),
        transitions=transitions,
        record=record,
        pending_full_vec=(
            None if pending_full_vec is None
            else np.asarray(pending_full_vec, dtype=np.int64).copy()
        ),
        invalid_anomaly=bool(invalid_anomaly),
    )


def collect_fusion_episode(
        *,
        seq_env: BLBStage2SequentialEnv,
        policy: Any,
        device: torch.device,
        train_cfg: Any,
        rel_ep: int,
        absolute_ep: int,
        base_seed: int,
        baseline_action_vec: np.ndarray,
        force_baseline_episodes: int,
        forbidden_mask: Any,
        max_rejection_retries: int = 32,
        log_fn: Optional[Callable[[str], None]] = None,
        device_lock: Optional[threading.Lock] = None,
        profile: bool = False,
        ) -> FusionEpisodeOutcome:
    """Collect ONE fusion-mode episode (serial: the generator with its per-step
    forward run INLINE, B=1). Backward-compatible signature/behavior — the only
    change vs the pre-2026-06-21 path is the batch-invariant seeded sampler.
    ``device_lock`` is retained for call-site compatibility but unused: the seeded
    sampler is lock-free and the terminal probe serializes on
    ``env.probe_device_lock``. ``profile`` adds cuda-synced forward timing so the
    serial ``policy_rollout_wall_seconds`` is measured the SAME way the batched
    driver measures its forward (symmetric OFF-vs-ON A/B; without it OFF would be
    timed async and ON synced).
    """
    del device_lock  # seeded sampling needs no global-RNG lock (see generator)
    gen = _fusion_episode_generator(
        seq_env=seq_env, policy=policy, device=device, train_cfg=train_cfg,
        rel_ep=rel_ep, absolute_ep=absolute_ep, base_seed=base_seed,
        baseline_action_vec=baseline_action_vec,
        force_baseline_episodes=force_baseline_episodes,
        forbidden_mask=forbidden_mask, max_rejection_retries=max_rejection_retries,
        log_fn=log_fn,
    )
    sync = bool(profile) and device.type == "cuda"
    try:
        req = next(gen)
        while True:
            obs_t, slot_mask_t, levels_t, alm_t, prior = req
            if sync:
                torch.cuda.synchronize()
            logits, safe_logits, value = policy.forward_and_mask(
                obs_t, slot_mask_t, levels_t,
                action_level_mask=alm_t, baseline_prior_scale=prior,
                truncate_to_current=True,
            )
            if sync:
                torch.cuda.synchronize()
            req = gen.send((logits, safe_logits, value))
    except StopIteration as stop:
        return stop.value


def collect_fusion_episodes_batched(
        *,
        seq_envs: Sequence[BLBStage2SequentialEnv],
        policy: Any,
        device: torch.device,
        train_cfg: Any,
        window_rel_eps: Sequence[int],
        absolute_eps: Sequence[int],
        base_seed: int,
        baseline_action_vec: np.ndarray,
        force_baseline_episodes: int,
        forbidden_mask: Any,
        max_rejection_retries: int = 32,
        log_fn: Optional[Callable[[str], None]] = None,
        profile: bool = False,
        ) -> List[FusionEpisodeOutcome]:
    """Collect B fusion episodes in LOCKSTEP, batching the per-step GTrXL forward
    across all B (the launch-bound rollout cost amortized ~B×).

    All episodes share the episode-INDEPENDENT step schedule, so they have an
    identical horizon and advance one step together — at each step the B current
    obs are stacked into one ``[B, state_dim]`` forward. Per-row sampling, env
    stepping (replan) and the terminal probe stay per episode (the probe is
    already the optimized K-trial path and cannot share a forward across distinct
    noise configs). Float-equivalent to B serial ``collect_fusion_episode`` calls
    (batched GEMM differs ~1e-6), NOT byte-identical -> gated by the batch-
    invariance self-test, not the old 1==N byte-diff. ``profile`` adds
    cuda-synced forward timing (the authoritative rollout wall for A/B).
    """
    B = len(seq_envs)
    if B == 0:
        return []
    gens = [
        _fusion_episode_generator(
            seq_env=seq_envs[e], policy=policy, device=device, train_cfg=train_cfg,
            rel_ep=int(window_rel_eps[e]), absolute_ep=int(absolute_eps[e]),
            base_seed=base_seed, baseline_action_vec=baseline_action_vec,
            force_baseline_episodes=force_baseline_episodes,
            forbidden_mask=forbidden_mask,
            max_rejection_retries=max_rejection_retries, log_fn=log_fn,
        )
        for e in range(B)
    ]
    outcomes: List[Optional[FusionEpisodeOutcome]] = [None] * B
    pending: Dict[int, Any] = {e: next(g) for e, g in enumerate(gens)}
    active = list(range(B))
    fwd_wall = 0.0
    while active:
        obs_B = torch.cat([pending[e][0] for e in active], dim=0)
        sm_B = torch.cat([pending[e][1] for e in active], dim=0)
        lv_B = torch.cat([pending[e][2] for e in active], dim=0)
        alm_B = torch.cat([pending[e][3] for e in active], dim=0)
        prior_B = torch.tensor(
            [float(pending[e][4]) for e in active],
            device=device, dtype=torch.float32,
        )
        if profile and device.type == "cuda":
            torch.cuda.synchronize()
        _t0 = time.perf_counter()
        logits_B, safe_B, value_B = policy.forward_and_mask(
            obs_B, sm_B, lv_B,
            action_level_mask=alm_B, baseline_prior_scale=prior_B,
            # Lockstep => all B rows share current_step => the causal-prefix
            # truncation is exact AND processes only ~t+1 tokens/step (matching
            # the serial path), which is what actually lets the batched forward
            # amortize the launch/dispatch cost instead of paying full-H work.
            truncate_to_current=True,
        )
        if profile and device.type == "cuda":
            torch.cuda.synchronize()
        fwd_wall += float(time.perf_counter() - _t0)
        nxt: List[int] = []
        for i, e in enumerate(active):
            try:
                pending[e] = gens[e].send(
                    (logits_B[i:i + 1], safe_B[i:i + 1], value_B[i:i + 1])
                )
                nxt.append(e)
            except StopIteration as stop:
                outcomes[e] = stop.value
        active = nxt
    # attribute the (shared) batched forward wall evenly across episodes so
    # episodes.jsonl's policy_rollout_wall_seconds reflects the amortized cost.
    per_ep = fwd_wall / float(B)
    for oc in outcomes:
        if oc is not None:
            oc.record.policy_rollout_wall_seconds = float(per_ep)
    return [oc for oc in outcomes if oc is not None]


# ---------------------------------------------------------------------------
# Worker + runner
# ---------------------------------------------------------------------------

@dataclass
class Stage2FusionWorker:
    """Per-GPU state: replicated model/handler/bridge/env (+ policy replica)."""
    device: torch.device
    seq_env: BLBStage2SequentialEnv
    role: str = "primary"  # "primary" (worker 0, reuses the env) or "replica"
    policy_replica: Optional[Any] = None  # eval-mode copy, refreshed per window
    # Shared per-DEVICE lock (2026-06-12 workers-per-device): two workers on
    # the same GPU share that device's default CUDA generator (policy
    # sampling) and its dedicated noise generator (probe trials), so the two
    # RNG-consuming atomic units — (manual_seed -> sample) and
    # (reseed_noise -> one full probe trial forward) — are serialized through
    # this lock. With one worker per device the lock is uncontended and the
    # behavior is bit-identical to before.
    device_lock: Optional[threading.Lock] = None


class Stage2ParallelRunner:
    """Window-parallel whole-episode collection for fusion-mode Stage-2 RL.

    ``run_window`` collects ``num_episodes`` complete episodes split across
    workers in balanced contiguous chunks of GLOBAL indices and returns the
    outcomes in GLOBAL order, plus a per-window ``rollout_sig`` (sha1 over
    every episode's actions/rewards/terminal metrics) for the 1-vs-N
    determinism harness.
    """

    def __init__(
            self,
            *,
            workers: List[Stage2FusionWorker],
            log_fn: Optional[Callable[[str], None]] = None,
            seq_env_cfg: Any = None,
            fusion_map: Any = None,
            ):
        if not workers:
            raise ValueError("Stage2ParallelRunner requires at least one worker")
        self.workers = workers
        self.log = log_fn or (lambda _m: None)
        # Needed to build B SequentialEnv contexts per worker for batched rollout
        # (all share the worker's base_env / model — no extra model copies).
        self._seq_env_cfg = seq_env_cfg
        self._fusion_map = fusion_map

    @property
    def num_workers(self) -> int:
        return len(self.workers)

    def _sync_policy_replicas(self, policy: Any) -> None:
        primary_device = next(policy.parameters()).device
        for w in self.workers:
            if w.device == primary_device:
                replica = copy.deepcopy(policy)
            else:
                with torch.cuda.device(w.device):
                    replica = copy.deepcopy(policy).to(w.device)
            replica.eval()
            w.policy_replica = replica

    def _worker_batch_envs(
            self,
            worker: Stage2FusionWorker,
            n: int,
            ) -> List[BLBStage2SequentialEnv]:
        """``n`` SequentialEnv rollout contexts for one worker, ALL sharing the
        worker's base_env (model) — NO extra model copies. Row 0 reuses the
        worker's existing env; the rest are fresh accumulators over the same base.
        Sharing is safe because per-step rollout mutates only each env's own
        accumulator, and the terminal probe (the only base mutation) runs serially
        per row within this worker's single thread (probe_noise_seed is set
        per-step-before-commit, so the shared base carries the right seed).
        """
        base = worker.seq_env.base
        envs: List[BLBStage2SequentialEnv] = [worker.seq_env]
        for _ in range(1, int(n)):
            envs.append(BLBStage2SequentialEnv(
                base_env=base,
                env_cfg=self._seq_env_cfg,
                fusion_map=self._fusion_map,
            ))
        return envs

    def run_window(
            self,
            *,
            policy: Any,
            train_cfg: Any,
            window_rel_start: int,
            num_episodes: int,
            absolute_episode_start: int,
            base_seed: int,
            baseline_action_vec: np.ndarray,
            force_baseline_episodes: int,
            forbidden_mask: Any,
            max_rejection_retries: int = 32,
            ) -> List[FusionEpisodeOutcome]:
        self._sync_policy_replicas(policy)
        n = int(num_episodes)
        assignments = assign_global_episodes(n, self.num_workers)
        results: List[Optional[FusionEpisodeOutcome]] = [None] * n
        errors: List[BaseException] = []

        batched = bool(getattr(train_cfg, "batched_rollout_enabled", False))
        profile = bool(getattr(train_cfg, "rollout_profile", False))

        def _worker_main(w_idx: int) -> None:
            worker = self.workers[w_idx]
            try:
                if worker.device.type == "cuda":
                    torch.cuda.set_device(worker.device)
                my_eps = list(assignments[w_idx])
                if not my_eps:
                    return
                if batched and len(my_eps) > 1:
                    # Batched lockstep rollout (2026-06-21): one GTrXL forward per
                    # step across all this worker's episodes (launch amortized).
                    rel_eps = [int(window_rel_start + g) for g in my_eps]
                    abs_eps = [int(absolute_episode_start + r) for r in rel_eps]
                    envs = self._worker_batch_envs(worker, len(my_eps))
                    outs = collect_fusion_episodes_batched(
                        seq_envs=envs,
                        policy=worker.policy_replica,
                        device=worker.device,
                        train_cfg=train_cfg,
                        window_rel_eps=rel_eps,
                        absolute_eps=abs_eps,
                        base_seed=int(base_seed),
                        baseline_action_vec=baseline_action_vec,
                        force_baseline_episodes=int(force_baseline_episodes),
                        forbidden_mask=forbidden_mask,
                        max_rejection_retries=int(max_rejection_retries),
                        log_fn=self.log,
                        profile=profile,
                    )
                    for g, oc in zip(my_eps, outs, strict=True):
                        results[g] = oc
                else:
                    for g in my_eps:
                        rel_ep = int(window_rel_start + g)
                        absolute_ep = int(absolute_episode_start + rel_ep)
                        results[g] = collect_fusion_episode(
                            seq_env=worker.seq_env,
                            policy=worker.policy_replica,
                            device=worker.device,
                            train_cfg=train_cfg,
                            rel_ep=rel_ep,
                            absolute_ep=absolute_ep,
                            base_seed=int(base_seed),
                            baseline_action_vec=baseline_action_vec,
                            force_baseline_episodes=int(force_baseline_episodes),
                            forbidden_mask=forbidden_mask,
                            max_rejection_retries=int(max_rejection_retries),
                            log_fn=self.log,
                            device_lock=worker.device_lock,
                            profile=profile,
                        )
            except BaseException as exc:  # surface worker crashes to the main thread
                errors.append(exc)

        threads = [
            threading.Thread(target=_worker_main, args=(i,), daemon=True)
            for i in range(self.num_workers)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise RuntimeError(
                f"stage2 parallel rollout worker failed: {errors[0]!r}"
            ) from errors[0]
        flat = [r for r in results if r is not None]
        if len(flat) != n:
            raise RuntimeError(
                f"stage2 parallel rollout lost episodes: got {len(flat)}/{n}"
            )

        sig_payload = [
            (
                oc.absolute_ep,
                [t["action"].tolist() for t in oc.transitions],
                [t["reward"] for t in oc.transitions],
                float(oc.record.terminal_reward),
                int(oc.record.terminal_priority),
                float(oc.record.terminal_loss_mean),
                float(oc.record.terminal_metric1_mean),
            )
            for oc in flat
        ]
        sig = hashlib.sha1(repr(sig_payload).encode("utf-8")).hexdigest()[:16]
        # ``workers=`` goes LAST so the 1-vs-N harness can grep the
        # worker-count-independent prefix ``window_start=.. episodes=..
        # rollout_sig=..`` and diff it byte-for-byte across GPU counts.
        self.log(
            f"[stage2-rollout] window_start={int(window_rel_start)} "
            f"episodes={n} rollout_sig={sig} workers={self.num_workers}"
        )
        return flat


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def expand_device_ids_for_workers(
        device_ids: List[int],
        workers_per_device: int,
        ) -> List[int]:
    """Worker -> device assignment list (2026-06-12 workers-per-device).

    Interleaved expansion ([0,1,2] x 2 -> [0,1,2,0,1,2]) so worker indices
    0..D-1 stay on distinct devices (worker 0 remains the primary on
    device_ids[0]). Episode results are functions of the GLOBAL episode index
    alone (ADR-009), so ANY worker count / assignment yields byte-identical
    results — the 1-vs-N gate validates this directly.
    """
    wpd = max(1, int(workers_per_device))
    return [int(d) for _ in range(wpd) for d in device_ids]


def build_stage2_parallel_runner(
        *,
        primary_seq_env: BLBStage2SequentialEnv,
        device_ids: List[int],
        layers_attribute: str,
        is_regression: bool,
        bridge_factory: Callable[[], Any],
        seq_env_cfg: Any,
        fusion_map: Any,
        log_fn: Optional[Callable[[str], None]] = None,
        workers_per_device: int = 1,
        ) -> Stage2ParallelRunner:
    """One worker per assignment entry; worker 0 reuses the primary env stack.

    ``workers_per_device > 1`` runs multiple workers per GPU to overlap one
    worker's CPU-side rollout/bookkeeping with a sibling's GPU-bound probe
    (the 60k profile: probe 2.69s = 78%% of episode wall, rollout 0.74s = 21%%).
    RNG-consuming atomic units are serialized through a shared per-device
    lock, so results stay byte-identical for any worker count.

    Must be called AFTER baseline cost calibration + noisy preflight so the
    replicated ``BLBStage2Env``s inherit the final thresholds / baseline /
    reward weights (shared read-only references).
    """
    if BLBNoiseRLBridge is None:
        raise RuntimeError(
            "blb_rl_bridge import failed earlier; cannot build the Stage-2 "
            "episode-parallel runner."
        )
    if not device_ids:
        raise ValueError("build_stage2_parallel_runner requires at least one device id")

    log = log_fn or (lambda _m: None)
    enable_cuda_reward_probe_fast_math()

    assignment = expand_device_ids_for_workers(device_ids, workers_per_device)
    device_locks: Dict[int, threading.Lock] = {
        int(d): threading.Lock() for d in set(assignment)
    }

    primary_base = primary_seq_env.base
    workers: List[Stage2FusionWorker] = [
        Stage2FusionWorker(
            device=torch.device(f"cuda:{int(assignment[0])}"),
            seq_env=primary_seq_env,
            role="primary",
            device_lock=device_locks[int(assignment[0])],
        )
    ]
    primary_base.probe_device_lock = device_locks[int(assignment[0])]
    log(f"[stage2-parallel] worker 0: cuda:{int(assignment[0])} (primary, reusing env)")

    for d in assignment[1:]:
        device = torch.device(f"cuda:{int(d)}")
        t0 = time.perf_counter()
        with torch.cuda.device(device):
            replica = copy.deepcopy(primary_base.model)
            replica = replica.to(device)
            replica.eval()
            replica_handler = ReversibleLayerHandler(replica)
            replica_batches = [
                _move_probe_batch_to_device(b, device)
                for b in primary_base.probe_batches
            ]
            replica_env = BLBStage2Env(
                handler=replica_handler,
                model=replica,
                probe_batches=replica_batches,
                rescale_bridge=bridge_factory(),
                baseline=primary_base.baseline,
                reward_weights=primary_base.reward_weights,
                acc_threshold=float(primary_base.acc_threshold),
                stab_threshold=float(primary_base.stab_threshold),
                max_sfs=primary_base.max_sfs,
                num_layers=int(primary_base.num_layers),
                gelu_degree=primary_base.gelu_degree,
                attn_degree=primary_base.attn_degree,
                layers_attribute=layers_attribute,
                is_regression=bool(is_regression),
                env_cfg=copy.copy(primary_base.env_cfg),
                acc_threshold_m2=primary_base.acc_threshold_m2,
            )
            replica_env.pareto_cost_archive = None
            replica_env.sync_degree_vectors_from_model()
            replica_seq = BLBStage2SequentialEnv(
                base_env=replica_env, env_cfg=seq_env_cfg, fusion_map=fusion_map,
            )
        replica_env.probe_device_lock = device_locks[int(d)]
        workers.append(Stage2FusionWorker(
            device=device, seq_env=replica_seq, role="replica",
            device_lock=device_locks[int(d)],
        ))
        log(
            f"[stage2-parallel] worker {len(workers) - 1}: {device} "
            f"(deepcopy replica, {time.perf_counter() - t0:.1f}s)"
        )

    per_dev = {d: assignment.count(d) for d in sorted(set(assignment))}
    if int(workers_per_device) > 1:
        log(
            f"[stage2-parallel] workers-per-device={int(workers_per_device)} -> "
            f"{len(assignment)} workers, per-device counts {per_dev} "
            f"(per-device RNG atomic-unit locks active)"
        )
    return Stage2ParallelRunner(
        workers=workers, log_fn=log, seq_env_cfg=seq_env_cfg, fusion_map=fusion_map,
    )
