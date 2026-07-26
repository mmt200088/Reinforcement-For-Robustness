"""Sequential PPO training loop + launcher orchestration.

Two entrypoints:

  * :func:`train_sequential` -- thin standalone driver over
    :class:`BLBStage2SequentialEnv` + :class:`BLBStage2SequentialPolicy` +
    :class:`SequentialRolloutBuffer`. Useful for A/B from a notebook.

  * :func:`run_sequential_via_runner` -- called from
    :meth:`blb_stage2_rl.runner.BLBStage2RLRunner.run` when
    ``train_cfg.sequential_rl`` is True (default on 2026-05-15). Reuses the
    existing runner's setup helpers (env construction, baseline calibration,
    persistent-dir resolution) and assembles a ``noise_stage_result`` dict in
    the same legacy-compatible shape downstream consumers expect.
"""
from __future__ import annotations

import copy
from collections import deque
import json
import logging
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from elastic_gpu import (
    ElasticGPUFailure,
    is_recoverable_gpu_failure,
    raise_if_elastic_gpu_restart_requested,
)
from report_format_utils import format_elapsed as _seq_fmt_elapsed
from report_format_utils import progress_bar as _seq_progress_bar
from rl_data_points import (
    RLDataPointWriter,
    make_unique_run_id,
    write_strict_json_file,
)

from .action_mask import (
    EmpiricalInvalidLevelMask,
    ForbiddenActionMask,
    StaticInvalidLevelMask,
)
from .action_space import K_LEVELS, _baseline_k_index_for_block
from .baseline_bootstrap import resolve_stage2_model_type
from .fusion_curriculum import (
    FUSION_NEIGHBOR_RAMP_FRACTION,
    build_fusion_step_level_mask,
    fusion_block_curriculum,
    fusion_probe_target_block,
    select_mutable_step_indices,
)
from .sequential_env import BLBStage2SequentialEnv, SequentialEnvConfig
from .sequential_policy import (
    BLBStage2SequentialPolicy,
    SequentialPolicyConfig,
    SequentialPPOConfig,
    SequentialRolloutBuffer,
    sequential_ppo_update,
    step_to_mask_and_levels,
)

if TYPE_CHECKING:
    from .statistical_constraints import BaselineReference


CUDA_RNG_ROLE_REGISTRY_VERSION = 1


def merge_cuda_rng_role_registry(
        previous_registry: Optional[Sequence[Any]],
        active_role_states: Sequence[Any],
        ) -> List[Any]:
    """Update active logical roles while retaining temporarily absent roles."""
    registry = list(previous_registry or ())
    for role_index, state in enumerate(active_role_states):
        if role_index < len(registry):
            registry[role_index] = state
        else:
            registry.append(state)
    return registry


def resolve_cuda_rng_role_registry(
        checkpoint: Mapping[str, Any],
        *,
        active_role_count: int,
        new_role_state_factory: Callable[[int], Any],
        ) -> Tuple[List[Any], List[Any]]:
    """Resolve active CUDA RNG streams without tying them to physical GPUs."""
    current_count = int(active_role_count)
    if current_count < 0:
        raise ValueError("active CUDA RNG role count must be non-negative")

    stored_registry = checkpoint.get("cuda_rng_state_by_role")
    if stored_registry is None:
        legacy_states = checkpoint.get("cuda_rng_state_all")
        if legacy_states is None:
            if current_count == 0:
                return [], []
            raise RuntimeError("layerwise checkpoint CUDA RNG state is missing")
        registry = list(legacy_states)
        if len(registry) != current_count:
            raise RuntimeError(
                "legacy layerwise checkpoint GPU count changed: "
                f"checkpoint={len(registry)}, current={current_count}; "
                "exact CUDA RNG role mapping is unavailable"
            )
        return registry, list(registry)

    version = int(checkpoint.get("cuda_rng_role_registry_version", 0) or 0)
    if version != CUDA_RNG_ROLE_REGISTRY_VERSION:
        raise RuntimeError(
            "unsupported layerwise checkpoint CUDA RNG role registry "
            f"version: {version}"
        )
    registry = list(stored_registry)
    saved_active_count = int(checkpoint.get(
        "cuda_rng_active_role_count", len(registry),
    ))
    if saved_active_count < 0 or saved_active_count > len(registry):
        raise RuntimeError(
            "layerwise checkpoint CUDA RNG active role count is invalid"
        )
    if current_count == 0 and saved_active_count > 0:
        raise RuntimeError(
            "layerwise checkpoint requires CUDA but no healthy GPU is visible"
        )
    if current_count > 0 and saved_active_count == 0:
        raise RuntimeError(
            "layerwise checkpoint was created without CUDA; "
            "changing the training backend cannot preserve exact results"
        )
    while len(registry) < current_count:
        registry.append(new_role_state_factory(len(registry)))
    return registry, list(registry[:current_count])


def _normalize_supported_rl_algo(value: Any, *, context: str = "rl_algo") -> str:
    algo = str(value or "ppo").strip().lower()
    if algo != "ppo":
        raise ValueError(
            "GRPO has been disabled for this project after the PPO-vs-GRPO "
            f"MRPC generalization study. {context} must be 'ppo', got {value!r}."
        )
    return "ppo"


def resolve_resumed_best_reward(
        resumed_best: Mapping[str, Any],
        historical_best: Any,
        ) -> float:
    """Return the best finite diagnostic reward available at resume."""
    candidates = (resumed_best.get("reward"), historical_best)
    finite_values: List[float] = []
    for candidate in candidates:
        try:
            value = float(candidate)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(value):
            finite_values.append(value)
    return max(finite_values, default=-math.inf)


@dataclass
class SequentialTrainConfig:
    total_episodes: int = 100
    update_every_n_episodes: int = 4
    log_every_n_episodes: int = 4
    seed: Optional[int] = None
    ppo: SequentialPPOConfig = field(default_factory=SequentialPPOConfig)
    # PPO-only. Legacy fields remain so old configs deserialize, but non-PPO
    # values are rejected before training.
    rl_algo: str = "ppo"
    grpo_kl_beta: float = 0.0

    def __post_init__(self) -> None:
        self.rl_algo = _normalize_supported_rl_algo(
            self.rl_algo, context="SequentialTrainConfig.rl_algo"
        )
        self.grpo_kl_beta = 0.0
    # 2026-05-18 (warmstart-sampling hotfix): the PPO entropy bonus was
    # actively undoing the forced-baseline anchor — entropy rose 6.48 →
    # 9.21 across the 3 anchor PPO updates, so the policy ended *more*
    # diffuse than at init and crashed acc immediately when sampling
    # started. The fix is a schedule:
    #   ep < anchor_episodes                            → ent_coef = ent_coef_anchor (0.0)
    #   anchor_episodes ≤ ep < anchor_eps + ramp_eps    → linear ramp 0 → cfg.ppo.ent_coef
    #   ep ≥ anchor_eps + ramp_eps                      → cfg.ppo.ent_coef (steady)
    # The 0-ent_coef anchor lets the policy gradient cleanly concentrate
    # mass on baseline; the ramp then re-enables exploration gradually.
    ent_coef_anchor: float = 0.0
    ent_coef_ramp_episodes: int = 600
    # ADR-015 (2026-06-14): Stage-1 cosine entropy schedule (default for the
    # rebuild). "cosine" = high→low (start high, plateau, cosine-decay, no
    # anchor) — the Stage-1 exploration the user asked us to port; "anchor_ramp"
    # = the legacy low→high schedule. See _resolve_cosine_ent_coef_schedule.
    ent_coef_schedule: str = "cosine"
    ent_coef_cosine_start: float = 0.05
    ent_coef_cosine_end: float = 0.001
    ent_coef_cosine_plateau: float = 0.25
    ent_coef_cosine_decay_end: float = 1.0
    ent_coef_cosine_lower_bound: float = 0.012
    # ADR-015/Stage-1 alignment: "stage1_aligned" is the active default.
    # "continuous" and "tiered" remain for historical A/B only.
    reward_design: str = "stage1_aligned"
    absolute_episode_start: int = 0
    planned_total_episodes: Optional[int] = None
    convergence_resume_state: Optional[Mapping[str, Any]] = None
    convergence_min_episodes: int = 90_000
    convergence_patience_updates: int = 100
    warmstart_neighbor_sampling: bool = False
    warmstart_neighbor_ramp_episodes: int = 0
    warmstart_neighbor_max_mutations: int = 12
    warmstart_neighbor_max_radius: int = 1
    warmstart_mutable_full_offsets: Optional[List[int]] = None
    # Fusion-mode block-granularity safe-neighbor curriculum (additive 2026-06-05).
    # Per-slot safe-neighbor does not apply in fusion mode (the action is per-block
    # (option, K), not per-SF-slot). This curriculum instead gently widens how many
    # of the H blocks may leave the baseline (option 0, baseline K) each episode.
    # It reaches "all blocks, full radius" (== unrestricted open mask) by the end of
    # the ramp, so it is a pure warmup that dissolves — the full action space stays
    # reachable. ``fusion_neighbor_ramp_episodes`` 0 → derive 0.5 * total_episodes.
    fusion_neighbor_curriculum_enabled: bool = False
    fusion_neighbor_ramp_episodes: int = 0
    fusion_neighbor_max_radius: int = 6
    # Scheduled forced-fusion probes (ADR-011): every N post-anchor episodes,
    # one episode forces fusion option 1 on one block type (rotating
    # block2 -> block5 -> block4) at baseline K so PPO keeps receiving fresh
    # evidence of fusion's true value even after the policy's fusion logits
    # collapse. 0 disables. Decision is a pure function of the absolute
    # episode index (deterministic; identical across episode-parallel workers).
    fusion_probe_interval: int = 0
    # ADR-012 exploration floor (fusion mode): mixture sampling on the fusion
    # OPTION slot (and a smaller floor on the K slot) so the policy can never
    # become deterministic on the 2-way fusion choice — the 2nd 60k ended with
    # entropy 0.000 / clip 0.000 (frozen policy) and zero on-policy fusion
    # samples for the last 43.5k episodes. 0 disables.
    fusion_exploration_epsilon: float = 0.0
    fusion_exploration_epsilon_k: float = 0.0
    guarded_radius2_enabled: bool = False
    guarded_radius2_min_episode: int = 1060
    guarded_radius2_stall_window: int = 600
    guarded_radius2_health_window: int = 100
    guarded_radius2_max_mutations: int = 4
    guarded_radius2_episode_fraction: float = 0.15
    guarded_radius2_cooldown_episodes: int = 300
    guarded_radius2_min_radius1_successes: int = 3
    static_invalid_level_mask_enabled: bool = False
    empirical_invalid_level_mask_enabled: bool = False
    empirical_invalid_level_min_samples: int = 3
    empirical_invalid_level_min_rate: float = 0.80
    empirical_invalid_level_max_valid: int = 0
    fast_reward_mode_enabled: bool = False
    online_num_trials_per_step: int = 5
    terminal_eval_batch_size: int = 4
    protected_k1_enabled: bool = False
    protected_k1_guard_sigma: float = 4.0
    protected_k1_audit_fraction: float = 0.02
    promotion_validation_trials: int = 25
    promotion_margin_window: float = 0.25
    final_selection_top_n: int = 20
    final_selection_validation_trials: int = 25
    online_constraint_probability: float = 0.50
    promotion_constraint_probability: float = 0.80
    final_constraint_probability: float = 0.95


# ---------------------------------------------------------------------------
# v2-style console helpers (rounded box / banner / progress bar / time fmt)
# Ported from noise_rl_module_v2 so the sequential path has the same look-and-feel.
# ---------------------------------------------------------------------------

def _seq_log_major_rule(log_fn, title: str, width: int = 68) -> None:
    """Banner: ═══════ on either side of a title line."""
    bar = "═" * int(width)
    log_fn("")
    log_fn(bar)
    log_fn(title if title.startswith(" ") else f"  {title}")
    log_fn(bar)


def _seq_block_title(log_fn, title: str) -> None:
    """Block subtitle: 【…】 brackets."""
    log_fn("")
    log_fn(f"  【{title}】")


def _seq_display_width(s: str) -> int:
    """Display columns counted by east-asian-width (CJK = 2 cols, else 1).

    This is a noticeable improvement over ``len()`` for our Chinese log lines;
    v2's ``_log_rounded_box`` uses plain ``len()`` and produces a slightly
    misaligned right border. Sequential output uses this helper instead so
    boxes line up cleanly even with mixed CJK / ASCII content.
    """
    import unicodedata
    w = 0
    for ch in str(s):
        ea = unicodedata.east_asian_width(ch)
        w += 2 if ea in ("W", "F") else 1
    return w


def _seq_ljust_display(s: str, width: int) -> str:
    """Pad ``s`` with spaces so its *display width* equals ``width``."""
    pad = max(0, int(width) - _seq_display_width(s))
    return str(s) + (" " * pad)


def _seq_log_rounded_box(log_fn, lines, indent: str = "  ", min_inner_width: int = 8) -> None:
    """Render multi-line content as a plain, border-less indented block.

    Historically this used rounded box-drawing characters (``╭─╮│╰╯``), but
    those broke badly on narrow terminals + CJK mixed content: when a single
    line stretched past the terminal width, the right border wrapped to the
    next line and visually decoupled from the rest of the box. As of 2026-05-17
    we emit a tiny dashed separator + bulleted lines, which scans cleanly at
    any width and survives copy-paste into chat / markdown.
    """
    stripped = [str(x) for x in lines]
    if not stripped:
        return
    sep = "─" * 4
    log_fn(f"{indent}{sep}")
    for s in stripped:
        log_fn(f"{indent}· {s}")
    log_fn(f"{indent}{sep}")


def _seq_fmt_eta_finish(eta_seconds: float) -> str:
    finish_ts = time.time() + max(float(eta_seconds), 0.0)
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(finish_ts))


# How often (in PPO updates) to print the big progress box, matching v2's
# NOISE_RL_PROGRESS_BOX_PPO_INTERVAL.
SEQ_PROGRESS_BOX_PPO_INTERVAL = 5
SEQ_RL_VARIANT = "blb_v3_sequential_gtrxl_v2scale"

# Fusion-count warmstart (2026-06-03): the option space is tiny, so bias the
# baseline (fusion=0 / K=max) prior harder than the per-slot default (1.2) so
# cold-start sits at baseline and explores outward.
FUSION_WARMSTART_BIAS_GAIN = 2.5
# ADR-012: default K-slot exploration floor (option-slot floor is the
# fusion_exploration_epsilon config field; this one is the paired K default).
FUSION_EXPLORATION_EPSILON_K = 0.02


def _resolve_ent_coef_schedule(
        *,
        ep_count_1based: int,
        anchor_episodes: int,
        target_ent_coef: float,
        anchor_ent_coef: float = 0.0,
        ramp_episodes: int = 600,
        ) -> float:
    """Three-stage entropy schedule for sequential PPO.

    Anchor stage (``ep < anchor_episodes``): return ``anchor_ent_coef``
    (default 0.0) so the PPO update on forced-baseline rollouts can
    concentrate policy mass on baseline without the entropy bonus
    pulling it apart.

    Ramp stage (``anchor_episodes ≤ ep < anchor_episodes + ramp_episodes``):
    linearly interpolate from ``anchor_ent_coef`` to ``target_ent_coef``.

    Steady stage (``ep ≥ anchor_episodes + ramp_episodes``): return
    ``target_ent_coef`` (the standard PPO ent_coef value).

    Episode count is 1-based to match ``force_baseline_episodes`` semantics
    (ep_count_1based == anchor_episodes is the FIRST sample-phase episode).
    """
    ep = int(ep_count_1based)
    anchor = max(0, int(anchor_episodes))
    ramp = max(1, int(ramp_episodes))
    target = float(target_ent_coef)
    anchor_val = float(anchor_ent_coef)
    if ep <= anchor:
        return anchor_val
    if ep >= anchor + ramp:
        return target
    progress = (ep - anchor) / float(ramp)   # in (0, 1)
    return anchor_val + (target - anchor_val) * float(progress)


def _resolve_cosine_ent_coef_schedule(
        ep_count_1based: int,
        total_episodes: int,
        *,
        start: float = 0.05,
        end: float = 0.001,
        plateau_ratio: float = 0.25,
        lower_bound: float = 0.012,
        ) -> float:
    """ADR-015 Stage-1-style cosine entropy schedule (port of
    layer_importance_evaluator.update_hyperparameters + RL_OPT_FLAGS).

    Starts HIGH (``start``) and stays there for the first ``plateau_ratio`` of
    training (充分探索), then cosine-decays to ``end``, floored at ``lower_bound``.
    This REPLACES the fusion-mode anchor+ramp schedule (which started at 0 during
    the baseline anchor then ramped UP — the opposite of exploration and a root of
    the "初始策略/探索" problem). No baseline anchor here: the small all-valid
    (option,K) space wants high-entropy exploration from episode 1.
    """
    total = max(1, int(total_episodes))
    progress = float(ep_count_1based) / float(total)
    plateau = min(1.0, max(0.0, float(plateau_ratio)))
    if progress <= plateau:
        val = float(start)
    else:
        t = (progress - plateau) / max(1e-8, 1.0 - plateau)
        t = min(1.0, max(0.0, t))
        val = float(end) + 0.5 * (float(start) - float(end)) * (1.0 + math.cos(math.pi * t))
    return max(float(lower_bound), float(val))


def _resolve_baseline_prior_scale(
        absolute_episode_idx: int,
        *,
        anchor_episodes: int = 60,
        ) -> float:
    """Decaying logit prior toward static baseline.

    This prior is a soft safety rail, not a permanent lock. It starts strong
    enough to keep the fresh GTrXL near the verified baseline, then decays so
    empirical cost-boundary proposals can move away when F1 evidence supports
    them.
    """
    ep = int(absolute_episode_idx)
    anchor = max(0, int(anchor_episodes))
    if ep < anchor:
        return 8.0
    if ep < 1000:
        denom = max(1, 1000 - anchor)
        t = max(0.0, min(1.0, float(ep - anchor) / float(denom)))
        return 8.0 + (6.0 - 8.0) * t
    if ep < 5000:
        t = max(0.0, min(1.0, float(ep - 1000) / 4000.0))
        return 6.0 + (3.0 - 6.0) * t
    if ep < 15000:
        t = max(0.0, min(1.0, float(ep - 5000) / 10000.0))
        return 3.0 + (0.0 - 3.0) * t
    return 0.0


def _compute_per_slot_mode_preferred(
        *,
        schedule: Sequence[Any],
        baseline_action_vec: Optional[np.ndarray],
        max_step_dim: int,
        fallback_idx: int = 4,
        ) -> List[int]:
    """Per-slot mode of the baseline action across all steps in the schedule.

    For each slot position ``p`` in ``[0, max_step_dim)``, collect the
    baseline value at that slot position across every step where ``p`` is
    active (i.e. where ``len(spec.full_vec_offsets) > p``), then return the
    most common value. This gives a per-slot-position preferred index that
    actually matches the baseline distribution — without it, a uniform
    ``preferred=[fallback_idx]*max_step_dim`` is wrong for 8/13 slot positions
    (see 2026-05-18 diagnosis in `reports/stage2_rl/bug_reports/`).

    If ``baseline_action_vec`` is None or the schedule is empty, falls back
    to ``[fallback_idx]*max_step_dim``.
    """
    from collections import Counter

    if baseline_action_vec is None or not schedule:
        return [int(fallback_idx)] * int(max_step_dim)
    bvec = np.asarray(baseline_action_vec, dtype=np.int64).reshape(-1)
    preferred: List[int] = []
    for slot_pos in range(int(max_step_dim)):
        vals: List[int] = []
        for spec in schedule:
            offsets = getattr(spec, "full_vec_offsets", None)
            if offsets is None or slot_pos >= len(offsets):
                continue
            off = int(offsets[slot_pos])
            if 0 <= off < bvec.size:
                vals.append(int(bvec[off]))
        if not vals:
            preferred.append(int(fallback_idx))
            continue
        mode_val, _ = Counter(vals).most_common(1)[0]
        preferred.append(int(mode_val))
    return preferred


def _resolve_sequential_force_baseline_episodes(train_cfg: Any) -> int:
    """Resolve the absolute forced-baseline anchor length for sequential RL."""
    total = max(0, int(getattr(train_cfg, "total_episodes", 0) or 0))
    explicit = int(getattr(train_cfg, "force_baseline_episodes", 0) or 0)
    if explicit > 0:
        return max(0, min(int(explicit), int(total)))
    warmstart_anchor = getattr(train_cfg, "warmstart_anchor_episodes", None)
    if warmstart_anchor is not None:
        return max(0, min(int(warmstart_anchor), int(total)))
    return 0


def _near_baseline_level_indices(
        *,
        kind: str,
        baseline_idx: int,
        dim: int,
        radius: int,
        ) -> List[int]:
    """Local allowed categorical indices around a base value.

    This is deliberately bidirectional. Lower SF is only a proposal direction,
    not a truth about metric/stability boundary movement. K is decoded through
    non-monotonic ``K_LEVELS``, so locality is by distance in truncation bits,
    not categorical-index monotonicity.
    """
    dim = int(dim)
    baseline_idx = int(baseline_idx)
    radius = max(0, int(radius))
    if dim <= 0:
        return []
    if baseline_idx < 0 or baseline_idx >= dim:
        raise ValueError(f"baseline index {baseline_idx} out of width {dim}")
    if str(kind) == "K":
        base_k = int(K_LEVELS[baseline_idx])
        candidates = [
            int(idx) for idx, _value in enumerate(K_LEVELS[:dim])
        ]
        candidates.sort(key=lambda idx: (abs(int(K_LEVELS[idx]) - base_k), int(idx)))
        keep = min(len(candidates), max(1, 2 * int(radius) + 1))
        return sorted(int(idx) for idx in candidates[:keep])
    lo = max(0, baseline_idx - radius)
    hi = min(dim - 1, baseline_idx + radius)
    return [int(idx) for idx in range(lo, hi + 1)]


def _default_step_level_mask(
        *,
        spec: Any,
        baseline_action_vec: Sequence[int],
        max_step_dim: int,
        max_num_levels: int,
        ) -> np.ndarray:
    """Base-action-only per-level mask for one sequential step."""
    baseline = np.asarray(baseline_action_vec, dtype=np.int64).reshape(-1)
    mask = np.zeros((int(max_step_dim), int(max_num_levels)), dtype=bool)
    for slot_idx, (offset, dim) in enumerate(zip(spec.full_vec_offsets, spec.slot_dims)):
        if slot_idx >= int(max_step_dim):
            break
        offset = int(offset)
        dim = int(dim)
        if offset < 0 or offset >= baseline.size:
            continue
        baseline_idx = int(baseline[offset])
        if baseline_idx < 0 or baseline_idx >= dim or baseline_idx >= int(max_num_levels):
            raise ValueError(
                f"baseline action offset {offset} index {baseline_idx} out of width {dim}"
            )
        mask[slot_idx, baseline_idx] = True
    return mask


def _open_step_level_mask(
        *,
        spec: Any,
        max_step_dim: int,
        max_num_levels: int,
        ) -> np.ndarray:
    """Full legal support for one sequential step before pruning layers apply."""
    mask = np.zeros((int(max_step_dim), int(max_num_levels)), dtype=bool)
    # Fusion mode: FusionStepSpec exposes (fusion_num_options, k_num_levels).
    if hasattr(spec, "fusion_num_options"):
        per_slot = [int(spec.fusion_num_options), int(spec.k_num_levels)]
    else:
        per_slot = [int(d) for d in spec.slot_dims]
    for slot_idx, dim in enumerate(per_slot):
        if slot_idx >= int(max_step_dim):
            break
        width = min(int(dim), int(max_num_levels))
        if width > 0:
            mask[slot_idx, :width] = True
    return mask


@dataclass(frozen=True)
class _StepStaticTensors:
    slot_mask_np: np.ndarray
    levels_np: np.ndarray
    slot_mask_t: torch.Tensor
    levels_t: torch.Tensor


@dataclass(frozen=True)
class _CachedFusionActionLevelMask:
    mask_np: np.ndarray
    mask_t: torch.Tensor


def _step_static_cache_key(
        *,
        schedule: Sequence[Any],
        max_step_dim: int,
        max_num_levels: int,
        device: torch.device,
        ) -> Tuple[str, int, int, Tuple[Tuple[Any, ...], ...]]:
    items: List[Tuple[Any, ...]] = []
    for pos, spec in enumerate(schedule):
        step_idx = int(getattr(spec, "step_idx", pos))
        if hasattr(spec, "fusion_num_options"):
            items.append((
                "fusion",
                step_idx,
                int(spec.fusion_num_options),
                int(spec.k_num_levels),
            ))
        else:
            items.append((
                "per_slot",
                step_idx,
                tuple(int(dim) for dim in getattr(spec, "slot_dims", ())),
            ))
    return (
        str(torch.device(device)),
        int(max_step_dim),
        int(max_num_levels),
        tuple(items),
    )


def _get_cached_step_static_tensors(
        env: BLBStage2SequentialEnv,
        *,
        max_step_dim: int,
        max_num_levels: int,
        device: torch.device,
        ) -> List[_StepStaticTensors]:
    """Build immutable per-step NumPy/device tensors once per schedule."""
    schedule = env.schedule
    key = _step_static_cache_key(
        schedule=schedule,
        max_step_dim=int(max_step_dim),
        max_num_levels=int(max_num_levels),
        device=torch.device(device),
    )
    cached = getattr(env, "_stage2_static_step_tensor_cache", None)
    if isinstance(cached, tuple) and len(cached) == 2 and cached[0] == key:
        return cached[1]

    by_step: Dict[int, _StepStaticTensors] = {}
    for pos, spec in enumerate(schedule):
        step_idx = int(getattr(spec, "step_idx", pos))
        if step_idx in by_step:
            raise ValueError(f"duplicate sequential step index {step_idx}")
        slot_mask_np, levels_np = step_to_mask_and_levels(
            spec,
            int(max_step_dim),
            int(max_num_levels),
        )
        slot_mask_np = np.asarray(slot_mask_np, dtype=bool)
        levels_np = np.asarray(levels_np, dtype=np.int64)
        static = _StepStaticTensors(
            slot_mask_np=slot_mask_np,
            levels_np=levels_np,
            slot_mask_t=torch.as_tensor(slot_mask_np, device=device).unsqueeze(0),
            levels_t=torch.as_tensor(levels_np, device=device).unsqueeze(0),
        )
        slot_mask_np.setflags(write=False)
        levels_np.setflags(write=False)
        by_step[step_idx] = static
    step_static_tensors = [by_step[idx] for idx in range(len(schedule))]
    setattr(env, "_stage2_static_step_tensor_cache", (key, step_static_tensors))
    return step_static_tensors


def _get_cached_fusion_action_level_mask(
        env: BLBStage2SequentialEnv,
        *,
        spec: Any,
        mode: str,
        mutable: bool,
        radius: int,
        force_option_one: bool,
        max_step_dim: int,
        max_num_levels: int,
        device: torch.device,
        ) -> _CachedFusionActionLevelMask:
    """Cache repeated fusion support masks and their device copies."""
    mode_key = str(mode)
    if mode_key not in {"curriculum", "open"}:
        raise ValueError(f"unknown fusion action-level mask mode: {mode!r}")
    device = torch.device(device)
    n_opts = int(spec.fusion_num_options)
    n_k = int(spec.k_num_levels)
    baseline_k_index = int(_baseline_k_index_for_block(int(spec.block_idx)))
    effective_mutable = bool(mutable) if mode_key == "curriculum" else True
    effective_radius = max(0, int(radius)) if mode_key == "curriculum" else 0
    effective_force = bool(force_option_one and n_opts > 1)
    key = (
        str(device),
        mode_key,
        int(getattr(spec, "step_idx", 0)),
        int(spec.block_idx),
        n_opts,
        n_k,
        effective_mutable,
        effective_radius,
        baseline_k_index,
        effective_force,
        int(max_step_dim),
        int(max_num_levels),
        tuple(int(value) for value in K_LEVELS[:n_k]),
    )
    cache = getattr(env, "_stage2_fusion_action_level_mask_cache", None)
    if not isinstance(cache, dict):
        cache = {}
    cached = cache.get(key)
    if cached is not None:
        return cached

    if mode_key == "curriculum":
        mask_np = build_fusion_step_level_mask(
            fusion_num_options=n_opts,
            k_num_levels=n_k,
            k_level_values=list(K_LEVELS),
            mutable=effective_mutable,
            radius=effective_radius,
            baseline_k_index=baseline_k_index,
            max_step_dim=int(max_step_dim),
            max_num_levels=int(max_num_levels),
        )
    else:
        mask_np = _open_step_level_mask(
            spec=spec,
            max_step_dim=int(max_step_dim),
            max_num_levels=int(max_num_levels),
        )
    if effective_force:
        mask_np = np.array(mask_np, dtype=bool, copy=True)
        mask_np[0, 1] = True
    else:
        mask_np = np.asarray(mask_np, dtype=bool)
    cached = _CachedFusionActionLevelMask(
        mask_np=mask_np,
        mask_t=torch.as_tensor(mask_np, device=device).unsqueeze(0),
    )
    mask_np.setflags(write=False)
    cache[key] = cached
    setattr(env, "_stage2_fusion_action_level_mask_cache", cache)
    return cached


def _build_step_level_mask(
        *,
        spec: Any,
        baseline_action_vec: Sequence[int],
        selected_full_offsets: Set[int],
        max_step_dim: int,
        max_num_levels: int,
        radius: int,
        ) -> np.ndarray:
    """Near-base per-level mask for one step.

    Slots selected for the current episode may move inside a local
    near-base neighborhood. Every other slot stays base-action-only.
    """
    baseline = np.asarray(baseline_action_vec, dtype=np.int64).reshape(-1)
    mask = _default_step_level_mask(
        spec=spec,
        baseline_action_vec=baseline,
        max_step_dim=int(max_step_dim),
        max_num_levels=int(max_num_levels),
    )
    selected = {int(x) for x in (selected_full_offsets or set())}
    for slot_idx, (offset, dim, kind) in enumerate(
            zip(spec.full_vec_offsets, spec.slot_dims, spec.slot_kinds)
            ):
        if slot_idx >= int(max_step_dim):
            break
        offset = int(offset)
        if offset not in selected:
            continue
        dim = int(dim)
        baseline_idx = int(baseline[offset])
        allowed = _near_baseline_level_indices(
            kind=str(kind),
            baseline_idx=baseline_idx,
            dim=dim,
            radius=int(radius),
        )
        mask[slot_idx, :] = False
        for idx in allowed:
            if 0 <= int(idx) < dim and int(idx) < int(max_num_levels):
                mask[slot_idx, int(idx)] = True
        if not bool(mask[slot_idx, baseline_idx]):
            raise ValueError(f"near-baseline mask lost baseline offset {offset}")
    return mask


def _sequential_neighbor_curriculum(
        *,
        absolute_episode_idx: int,
        anchor_episodes: int,
        ramp_episodes: int,
        max_mutations: int,
        max_radius: int,
        ) -> Tuple[int, int]:
    ramp = max(1, int(ramp_episodes))
    after_anchor = max(0, int(absolute_episode_idx) - int(anchor_episodes))
    progress = min(1.0, max(0.0, float(after_anchor) / float(ramp)))
    mutations = 1 + int(math.floor(progress * max(0, int(max_mutations) - 1)))
    radius = 1 + int(math.floor(progress * max(0, int(max_radius) - 1)))
    return max(1, mutations), max(1, radius)


@dataclass
class GuardedRadius2Decision:
    active: bool = False
    mode: str = "radius1"
    radius: int = 1
    mutation_count: int = 0
    safe_offsets: Tuple[int, ...] = ()
    recent_frontier_expansions: int = 0
    recent_duplicate_rate: float = 0.0
    recent_dominated_rate: float = 0.0
    cooldown_remaining: int = 0
    safe_offset_count: int = 0
    radius2_episode_count: int = 0
    radius2_failure_count: int = 0
    radius2_frontier_expansion_count: int = 0
    reason: str = ""


@dataclass
class OffsetEmpiricalStats:
    seen: int = 0
    p3_success: int = 0
    p1: int = 0
    p2: int = 0
    loss_cap: int = 0
    stab_violation: int = 0
    invalid: int = 0
    frontier_expansion: int = 0
    frontier_member: int = 0
    dominated: int = 0
    duplicate: int = 0
    fusion_gain_sum: float = 0.0
    k_gain_sum: float = 0.0
    bits_gain_sum: float = 0.0

    @property
    def failures(self) -> int:
        return int(self.p1 + self.p2 + self.loss_cap + self.stab_violation + self.invalid)

    @property
    def success_rate(self) -> float:
        return float(self.p3_success) / float(max(1, self.seen))

    @property
    def failure_rate(self) -> float:
        return float(self.failures) / float(max(1, self.seen))

    @property
    def mean_positive_cost_gain(self) -> float:
        denom = float(max(1, self.p3_success))
        return max(0.0, self.fusion_gain_sum / denom) + max(0.0, self.k_gain_sum / denom) + max(0.0, self.bits_gain_sum / denom)


class GuardedRadius2Controller:
    """Open radius2 only after radius1 Pareto search stalls and stays healthy."""

    def __init__(
            self,
            *,
            enabled: bool = False,
            min_episode: int = 1060,
            stall_window: int = 600,
            health_window: int = 100,
            max_mutations: int = 4,
            episode_fraction: float = 0.15,
            cooldown_episodes: int = 300,
            min_radius1_successes: int = 3,
            ) -> None:
        self.enabled = bool(enabled)
        self.min_episode = max(0, int(min_episode))
        self.stall_window = max(1, int(stall_window))
        self.health_window = max(1, int(health_window))
        self.max_mutations = max(1, int(max_mutations))
        self.episode_fraction = float(np.clip(float(episode_fraction), 0.0, 1.0))
        self.cooldown_episodes = max(0, int(cooldown_episodes))
        self.min_radius1_successes = max(1, int(min_radius1_successes))
        self._history: List[Dict[str, Any]] = []
        self._offset_successes: Dict[int, int] = {}
        self._offset_failures: Dict[int, int] = {}
        self._offset_stats: Dict[int, OffsetEmpiricalStats] = {}
        self._cooldown_until_episode = -1
        self.radius2_episode_count = 0
        self.radius2_failure_count = 0
        self.radius2_frontier_expansion_count = 0

    def decide(self, *, absolute_episode_idx: int, rng: Any) -> GuardedRadius2Decision:
        ep = int(absolute_episode_idx)
        recent = self._history[-self.stall_window:]
        recent_frontier = sum(
            1 for row in recent
            if str(row.get("terminal_pareto_event_kind", "")) == "frontier_expansion"
        )
        duplicate_rate = (
            sum(1 for row in recent if str(row.get("terminal_pareto_event_kind", "")) == "duplicate")
            / float(len(recent))
            if recent else 0.0
        )
        dominated_rate = (
            sum(1 for row in recent if str(row.get("terminal_pareto_event_kind", "")) == "dominated")
            / float(len(recent))
            if recent else 0.0
        )
        safe_offsets = self._safe_offsets()
        cooldown_remaining = max(0, int(self._cooldown_until_episode - ep + 1))
        base = {
            "recent_frontier_expansions": int(recent_frontier),
            "recent_duplicate_rate": float(duplicate_rate),
            "recent_dominated_rate": float(dominated_rate),
            "cooldown_remaining": int(cooldown_remaining),
            "safe_offset_count": int(len(safe_offsets)),
            "radius2_episode_count": int(self.radius2_episode_count),
            "radius2_failure_count": int(self.radius2_failure_count),
            "radius2_frontier_expansion_count": int(self.radius2_frontier_expansion_count),
        }
        if not self.enabled:
            return GuardedRadius2Decision(reason="disabled", **base)
        if ep < self.min_episode:
            return GuardedRadius2Decision(reason="before_min_episode", **base)
        if cooldown_remaining > 0:
            return GuardedRadius2Decision(reason="cooldown", **base)
        if len(self._history) < self.stall_window:
            return GuardedRadius2Decision(reason="insufficient_history", **base)
        if recent_frontier >= 1:
            return GuardedRadius2Decision(reason="frontier_not_stalled", **base)
        health_recent = self._history[-self.health_window:]
        if any(bool(row.get("unhealthy", False)) for row in health_recent):
            return GuardedRadius2Decision(reason="recent_unhealthy", **base)
        if not safe_offsets:
            return GuardedRadius2Decision(reason="no_safe_offsets", **base)
        if self.episode_fraction <= 0.0:
            return GuardedRadius2Decision(reason="fraction_zero", **base)
        try:
            draw = float(rng.random())
        except Exception:
            draw = 1.0
        if draw >= self.episode_fraction:
            return GuardedRadius2Decision(reason="fraction_skip", **base)
        return GuardedRadius2Decision(
            active=True,
            mode="guarded_radius2",
            radius=2,
            mutation_count=min(self.max_mutations, len(safe_offsets)),
            safe_offsets=tuple(sorted(int(x) for x in safe_offsets)),
            reason="frontier_stalled_and_healthy",
            **base,
        )

    def record_episode(
            self,
            *,
            absolute_episode_idx: int,
            selected_offsets: Sequence[int],
            radius: int,
            terminal_priority: int,
            invalid_steps: int,
            early_terminated: bool,
            terminal_stab_violation: float,
            terminal_loss_mean: float,
            terminal_pareto_event_kind: str,
            terminal_fusion_gain: float = 0.0,
            terminal_k_gain: float = 0.0,
            terminal_bits_gain: float = 0.0,
            ) -> None:
        offsets = {int(x) for x in (selected_offsets or [])}
        unhealthy = self._is_unhealthy(
            terminal_priority=terminal_priority,
            invalid_steps=invalid_steps,
            early_terminated=early_terminated,
            terminal_stab_violation=terminal_stab_violation,
            terminal_loss_mean=terminal_loss_mean,
        )
        if int(radius) == 1 and offsets:
            target = self._offset_failures if unhealthy else self._offset_successes
            for offset in offsets:
                target[offset] = int(target.get(offset, 0)) + 1
        for offset in offsets:
            stats = self._offset_stats.setdefault(int(offset), OffsetEmpiricalStats())
            stats.seen += 1
            if int(terminal_priority) == 3 and not unhealthy:
                stats.p3_success += 1
                stats.fusion_gain_sum += float(terminal_fusion_gain)
                stats.k_gain_sum += float(terminal_k_gain)
                stats.bits_gain_sum += float(terminal_bits_gain)
                event = str(terminal_pareto_event_kind or "")
                if event == "frontier_expansion":
                    stats.frontier_expansion += 1
                elif event == "frontier_member":
                    stats.frontier_member += 1
                elif event == "dominated":
                    stats.dominated += 1
                elif event == "duplicate":
                    stats.duplicate += 1
            else:
                if int(terminal_priority) == 1:
                    stats.p1 += 1
                if int(terminal_priority) == 2:
                    stats.p2 += 1
                if int(invalid_steps) > 0 or bool(early_terminated):
                    stats.invalid += 1
                if float(terminal_loss_mean) >= 99.0:
                    stats.loss_cap += 1
                if float(terminal_stab_violation) > 0.0:
                    stats.stab_violation += 1
        if int(radius) >= 2:
            self.radius2_episode_count += 1
            if unhealthy:
                self.radius2_failure_count += 1
                self._cooldown_until_episode = max(
                    self._cooldown_until_episode,
                    int(absolute_episode_idx) + self.cooldown_episodes,
                )
            if str(terminal_pareto_event_kind) == "frontier_expansion":
                self.radius2_frontier_expansion_count += 1
        self._history.append({
            "episode": int(absolute_episode_idx),
            "terminal_priority": int(terminal_priority),
            "invalid_steps": int(invalid_steps),
            "early_terminated": bool(early_terminated),
            "terminal_stab_violation": float(terminal_stab_violation),
            "terminal_loss_mean": float(terminal_loss_mean),
            "terminal_pareto_event_kind": str(terminal_pareto_event_kind or ""),
            "unhealthy": bool(unhealthy),
        })

    def _safe_offsets(self) -> List[int]:
        out: List[int] = []
        for offset, stats in self._offset_stats.items():
            if int(stats.p3_success) < self.min_radius1_successes:
                continue
            if int(stats.failures) != 0:
                continue
            out.append(int(offset))
        return sorted(out)

    def offset_rates(self, selected_offsets: Sequence[int]) -> Tuple[float, float]:
        rates: List[Tuple[float, float]] = []
        for offset in selected_offsets or []:
            stats = self._offset_stats.get(int(offset))
            if stats is None:
                continue
            rates.append((float(stats.success_rate), float(stats.failure_rate)))
        if not rates:
            return 0.0, 0.0
        return (
            float(np.mean([x[0] for x in rates])),
            float(np.mean([x[1] for x in rates])),
        )

    def offset_weight(self, offset: int) -> float:
        stats = self._offset_stats.get(int(offset))
        if stats is None:
            return 1.0
        weight = (
            1.0
            + 2.0 * float(stats.p3_success)
            + 3.0 * float(stats.frontier_expansion)
            + 1.0 * float(stats.frontier_member)
            + 0.02 * float(stats.mean_positive_cost_gain)
        )
        if stats.failures > 0:
            weight *= 0.25
        return max(0.05, float(weight))

    @staticmethod
    def _is_unhealthy(
            *,
            terminal_priority: int,
            invalid_steps: int,
            early_terminated: bool,
            terminal_stab_violation: float,
            terminal_loss_mean: float,
            ) -> bool:
        return (
            int(terminal_priority) in (1, 2)
            or int(invalid_steps) > 0
            or bool(early_terminated)
            or float(terminal_stab_violation) > 0.0
            or float(terminal_loss_mean) >= 99.0
        )


def _candidate_neighbor_offsets(
        *,
        schedule: Sequence[Any],
        baseline_action_vec: Sequence[int],
        mutable_full_offsets: Optional[Sequence[int]] = None,
        ) -> List[int]:
    baseline = np.asarray(baseline_action_vec, dtype=np.int64).reshape(-1)
    allowed_set = (
        {int(x) for x in mutable_full_offsets}
        if mutable_full_offsets is not None else None
    )
    out: List[int] = []
    seen: Set[int] = set()
    for spec in schedule:
        for offset, dim, field_name in zip(
                spec.full_vec_offsets,
                spec.slot_dims,
                spec.slot_field_names,
                ):
            offset = int(offset)
            if offset in seen:
                continue
            if str(field_name).startswith("__"):
                continue
            if allowed_set is not None and offset not in allowed_set:
                continue
            if offset < 0 or offset >= baseline.size:
                continue
            if int(dim) <= 1:
                continue
            out.append(offset)
            seen.add(offset)
    return out


def _sample_episode_neighbor_offsets(
        *,
        schedule: Sequence[Any],
        baseline_action_vec: Sequence[int],
        mutable_full_offsets: Optional[Sequence[int]],
        mutation_count: int,
        rng: np.random.Generator,
        empirical_controller: Optional[GuardedRadius2Controller] = None,
        ) -> Set[int]:
    candidates = _candidate_neighbor_offsets(
        schedule=schedule,
        baseline_action_vec=baseline_action_vec,
        mutable_full_offsets=mutable_full_offsets,
    )
    if not candidates:
        return set()
    count = max(0, min(int(mutation_count), len(candidates)))
    if count <= 0:
        return set()
    probs = None
    if empirical_controller is not None:
        weights = np.asarray(
            [empirical_controller.offset_weight(int(x)) for x in candidates],
            dtype=np.float64,
        )
        total_w = float(np.sum(weights))
        if np.isfinite(total_w) and total_w > 0.0:
            probs = weights / total_w
    chosen = rng.choice(
        np.asarray(candidates, dtype=np.int64),
        size=count,
        replace=False,
        p=probs,
    )
    return {int(x) for x in np.asarray(chosen, dtype=np.int64).reshape(-1).tolist()}


def _noisy_metric_threshold_from_baseline(
        *,
        noisy_baseline_metric: float,
        tolerance: float,
        ) -> float:
    """Relative metric gate from the noisy baseline.

    ``tolerance`` is a fraction, so ``0.001`` means a strict 0.1% relative drop
    from the noisy all-max BLB baseline. Older code subtracted one probe sample
    (``1 / probe_size``), which made a configured 0.1% gate materially looser
    on MRPC-sized probes.
    """
    baseline = float(noisy_baseline_metric)
    tol = max(0.0, float(tolerance))
    return max(0.0, baseline * (1.0 - tol))


def _noisy_accuracy_threshold_with_probe_guard(
        *,
        noisy_baseline_metric1: float,
        allowed_acc_drop: float,
        probe_size: int,
        ) -> float:
    """Compatibility wrapper for the old helper name.

    ``probe_size`` is intentionally ignored. The trainer gate now uses the
    strict relative tolerance requested by the CLI/config.
    """
    _ = probe_size
    baseline = float(noisy_baseline_metric1)
    tol = max(0.0, float(allowed_acc_drop))
    return max(0.0, baseline * (1.0 - tol))


def _noisy_std_threshold_from_baseline(
        *,
        noisy_baseline_std: float,
        stability_multiplier: float,
        floor: float,
        ) -> float:
    """Per-channel stability threshold used by reward.py.

    Stability tolerance is a multiplier on the noisy baseline std. The floor is
    only an absolute minimum for degenerate near-zero baseline variance.
    """
    raw = float(noisy_baseline_std)
    if not np.isfinite(raw):
        raw = 0.0
    return float(max(raw * max(0.0, float(stability_multiplier)), float(floor)))


@dataclass
class EpisodeRecord:
    episode_idx: int
    total_reward: float
    terminal_reward: float
    per_step_reward_sum: float
    invalid_steps: int
    early_terminated: bool
    steps_taken: int
    # Enriched fields (added 2026-05-15 for richer console output)
    valid_step_count: int = 0
    total_bits_sum_over_steps: int = 0
    fusion_count_sum_over_steps: int = 0
    # 2026-06-13: per-block-TYPE fusion split (diagnostic only).
    fusion_count_b2: int = 0
    fusion_count_b4: int = 0
    fusion_count_b5: int = 0
    terminal_final_config_fingerprint: str = ""
    terminal_materialization_failure_reason: str = ""
    terminal_model_uses_replan_config: bool = False
    first_invalid_step: Optional[int] = None
    first_invalid_block: Optional[int] = None
    first_invalid_layer: Optional[int] = None
    step_infos: List[Dict[str, Any]] = field(default_factory=list)
    # Per-block invalid_chain details (added 2026-05-17): every step where
    # the optimizer reported ``valid=False`` lands here, with a short reason
    # extracted from ``info["invalid_chain"]``. Surfaced into the details/
    # rollover txt so operators can grep "L11-B3" / "primes_over_q_max" /
    # "fusion cannot reduce" / etc. without having to re-run the optimizer.
    invalid_block_details: List[Dict[str, Any]] = field(default_factory=list)
    # Terminal reward breakdown (added 2026-05-18): the actual reward priority
    # + per-trial metrics from the terminal compute_reward call. Replaces the
    # previously-hardcoded `priority = 1 if invalid_steps > 0 else 3` label in
    # the details file, which had no relation to breakdown.priority and lied
    # whenever P2(stab) tripped on a "0 invalid_steps" episode. 0 = unset.
    terminal_priority: int = 0
    terminal_loss_mean: float = 0.0
    terminal_loss_std: float = 0.0
    terminal_metric1_mean: float = 0.0
    terminal_metric2_mean: float = 0.0
    terminal_metric1_std: float = 0.0
    terminal_metric2_std: float = 0.0
    terminal_stab_excess_m1: float = 0.0
    terminal_stab_excess_m2: float = 0.0
    terminal_stab_excess_loss: float = 0.0
    terminal_stab_violation: float = 0.0
    terminal_bits_gain: float = 0.0
    terminal_k_gain: float = 0.0
    terminal_fusion_gain: float = 0.0
    terminal_cost_score: float = 0.0
    terminal_p3_metric_margin_reward: float = 0.0
    # ADR-014 (2026-06-14) DEBUG: the ADR-013 barrier/margin quantities were
    # computed in RewardBreakdown but never persisted -> the failing mechanism
    # was a black box (had to infer noise>margin from metric_std). Persist them
    # per-episode so the next collapse is READ, not guessed. worst_signed_margin
    # (=mu) is the barrier input; acc_barrier_sat/vio are its outputs;
    # fusion_norm_raw vs _saturated shows the anti-runaway saturation in action.
    terminal_worst_signed_margin: float = 0.0
    terminal_acc_barrier_sat: float = 0.0
    terminal_acc_barrier_vio: float = 0.0
    terminal_near_miss: bool = False
    terminal_margin_m1: float = 0.0
    terminal_margin_m2: float = 0.0
    terminal_fusion_norm_raw: float = 0.0
    terminal_fusion_norm_saturated: float = 0.0
    terminal_cost_fusion_bonus: float = 0.0
    terminal_cost_truncation_bonus: float = 0.0
    terminal_cost_bits_tiebreaker: float = 0.0
    terminal_cost_truncation_step_gain: float = 0.0
    terminal_cost_rank_score: float = 0.0
    terminal_cost_rank_fusion: float = 0.0
    terminal_cost_rank_truncation: float = 0.0
    terminal_cost_rank_bits: float = 0.0
    terminal_pareto_event_kind: str = ""
    terminal_pareto_action_hash: str = ""
    terminal_pareto_frontier_removed: int = 0
    terminal_probe_wall_seconds: float = 0.0
    terminal_probe_devices: List[str] = field(default_factory=list)
    terminal_probe_trial_counts: List[int] = field(default_factory=list)
    terminal_probe_trial_indices: List[List[int]] = field(default_factory=list)
    terminal_probe_speedup: float = 1.0
    fusion_action_steps: List[Dict[str, Any]] = field(default_factory=list)
    per_step_optimizer_wall_seconds: float = 0.0
    policy_rollout_wall_seconds: float = 0.0
    terminal_cost_eval_wall_seconds: float = 0.0
    terminal_probe_install_wall_seconds: float = 0.0
    terminal_probe_clear_wall_seconds: float = 0.0
    terminal_probe_install_skipped: bool = False
    terminal_probe_clear_skipped: bool = False
    safe_neighbor_active: bool = False
    safe_neighbor_mutation_count: int = 0
    safe_neighbor_radius: int = 0
    exploration_mode: str = ""
    guarded_radius2_active: bool = False
    guarded_radius2_recent_frontier_expansions: int = 0
    guarded_radius2_recent_duplicate_rate: float = 0.0
    guarded_radius2_recent_dominated_rate: float = 0.0
    guarded_radius2_cooldown_remaining: int = 0
    guarded_radius2_safe_offset_count: int = 0
    guarded_radius2_episode_count: int = 0
    guarded_radius2_failure_count: int = 0
    guarded_radius2_frontier_expansion_count: int = 0
    samples_rejected_by_mask: int = 0
    samples_rejected_by_optimizer: int = 0
    steps_fallen_back_to_baseline: int = 0
    forbidden_mask_total: int = 0
    static_invalid_level_disabled: int = 0
    static_invalid_level_applied: int = 0
    static_invalid_level_scan_evaluated: int = 0
    static_invalid_level_scan_invalid: int = 0
    empirical_invalid_level_disabled: int = 0
    empirical_invalid_level_applied: int = 0
    rejection_optimizer_wall_seconds: float = 0.0
    baseline_prior_scale: float = 0.0
    base_action_source: str = ""
    proposal_direction: str = ""
    empirical_offset_success_rate: float = 0.0
    empirical_offset_failure_rate: float = 0.0
    frontier_seed_episode: int = -1


def _attach_pending_full_vec_for_callback(
        record: EpisodeRecord,
        pending_full_vec: Optional[np.ndarray],
        ) -> None:
    """Attach the action vector collected with this episode to ``record``.

    Episode-parallel workers have independent ``BLBStage2SequentialEnv``
    instances. The primary env's ``_pending_full_vec`` can therefore be stale
    for records collected by replica workers, so callback/diagnostic code must
    prefer the per-outcome vector attached here.
    """
    if pending_full_vec is None:
        return
    setattr(
        record,
        "_pending_full_vec_for_callback",
        np.asarray(pending_full_vec, dtype=np.int64).copy(),
    )


def _record_full_vec_for_callback(
        record: EpisodeRecord,
        seq_env: BLBStage2SequentialEnv,
        ) -> Optional[np.ndarray]:
    record_full_vec = getattr(record, "_pending_full_vec_for_callback", None)
    if record_full_vec is not None:
        return np.asarray(record_full_vec, dtype=np.int64)
    pending = getattr(seq_env, "_pending_full_vec", None)
    if pending is None:
        return None
    return np.asarray(pending, dtype=np.int64).copy()


def _episode_best_rank_key(record: Any) -> Tuple[float, ...]:
    """Stage-1-aligned best-action ranking.

    Hard gates still order P3 > P2 > P1. Inside P3, use the same bounded
    Stage-1-style reward that PPO optimizes first, then use the unbounded cost
    rank as a deterministic tie-breaker after the log-barrier signal.
    """
    try:
        priority = int(getattr(record, "terminal_priority", 0) or 0)
    except Exception:
        priority = 0
    try:
        invalid_steps = int(getattr(record, "invalid_steps", 0) or 0)
    except Exception:
        invalid_steps = 0

    total_reward = float(getattr(record, "total_reward", 0.0) or 0.0)
    terminal_reward = float(getattr(record, "terminal_reward", 0.0) or 0.0)
    metric1 = float(getattr(record, "terminal_metric1_mean", 0.0) or 0.0)
    metric2 = float(getattr(record, "terminal_metric2_mean", 0.0) or 0.0)
    stab_violation = float(getattr(record, "terminal_stab_violation", 0.0) or 0.0)

    if priority == 3 and invalid_steps == 0:
        return (
            3.0,
            terminal_reward,
            total_reward,
            float(getattr(record, "terminal_cost_rank_score", 0.0) or 0.0),
            float(getattr(record, "terminal_fusion_gain", 0.0) or 0.0),
            float(getattr(record, "terminal_k_gain", 0.0) or 0.0),
            float(getattr(record, "terminal_bits_gain", 0.0) or 0.0),
        )
    if priority == 2:
        return (
            2.0,
            -max(0.0, stab_violation),
            metric1,
            metric2,
            terminal_reward,
            total_reward,
        )
    if priority == 1:
        return (
            1.0,
            -float(max(0, invalid_steps)),
            metric1,
            metric2,
            terminal_reward,
            total_reward,
        )
    return (0.0, terminal_reward, total_reward)


def _stage2_record_loss_ok(record: Any, loss_threshold: Optional[float]) -> bool:
    if loss_threshold is None:
        return True
    try:
        loss = float(getattr(record, "terminal_loss_mean", float("inf")))
        threshold = float(loss_threshold)
    except Exception:
        return False
    return bool(math.isfinite(loss) and loss <= threshold + 1e-12)


def _stage2_record_strict_feasible(
        record: Any,
        loss_threshold: Optional[float],
        ) -> bool:
    try:
        priority = int(getattr(record, "terminal_priority", 0) or 0)
        invalid_steps = int(getattr(record, "invalid_steps", 0) or 0)
    except Exception:
        return False
    return bool(
        priority == 3
        and invalid_steps == 0
        and _stage2_record_loss_ok(record, loss_threshold)
    )


def _select_stage2_strict_feasible_best_record(
        records: Sequence[Any],
        *,
        loss_threshold: Optional[float],
        top_n: int = 20,
        ) -> Optional[Any]:
    """Return the best strict-feasible record among the ranked top-N.

    Ranking remains the training rank key; strict feasibility is applied as a
    final filter. This prevents one slightly loss-failed rank-best from forcing
    a baseline fallback when a lower-ranked candidate satisfies the constraints.
    """
    limit = max(1, int(top_n or 1))
    ranked = sorted(records, key=_episode_best_rank_key, reverse=True)[:limit]
    for record in ranked:
        if _stage2_record_strict_feasible(record, loss_threshold):
            return record
    return None


def _format_invalid_chain_reason(invalid_chain: Any) -> str:
    """One-line summary of an optimizer invalid_chain payload.

    Mirrors :func:`scripts/blb_diagnose_invalid_blocks._invalid_chain_reason`
    so the in-training detail records use the same wording as the offline
    sidecar. Kept tiny and dependency-free to stay safe inside the hot loop.
    """
    if invalid_chain is None:
        return "(none)"
    if not isinstance(invalid_chain, dict):
        return str(invalid_chain)
    parts: List[str] = []
    for k in ("reason", "message", "stage", "primes_over_q_max", "primes_under_q_min"):
        v = invalid_chain.get(k)
        if v not in (None, "", []):
            parts.append(f"{k}={v}")
    if not parts:
        # Fall back to the full dict, but cap length so a giant payload
        # can't bloat one details file.
        try:
            text = json.dumps(invalid_chain, ensure_ascii=False)
        except Exception:
            text = str(invalid_chain)
        return text[:240] + (" …" if len(text) > 240 else "")
    return "; ".join(parts)


def _apply_terminal_info_to_record(
        record: EpisodeRecord,
        terminal_reward: float,
        term_info_dict: Mapping[str, Any],
        *,
        cached_reward_hit: bool = False,
        validation_required: bool = False,
        ) -> None:
    """Populate an EpisodeRecord from the base env terminal info dict."""
    record.terminal_reward = float(terminal_reward)
    record.total_reward = float(record.per_step_reward_sum + float(terminal_reward))
    record.terminal_final_config_fingerprint = str(
        term_info_dict.get("final_config_fingerprint", "") or ""
    )
    record.terminal_materialization_failure_reason = str(
        term_info_dict.get("materialization_failure_reason", "") or ""
    )
    replan_application = term_info_dict.get("replan_application") or {}
    record.terminal_model_uses_replan_config = bool(
        isinstance(replan_application, Mapping)
        and replan_application.get("model_uses_replan_config", False)
    )
    # 2026-06-13: per-block-type fusion split (sequential_env mirrors it into
    # terminal_info). Diagnostic only; absent on non-fusion/legacy paths.
    if "fusion_count_b2" in term_info_dict:
        record.fusion_count_b2 = int(term_info_dict.get("fusion_count_b2", 0) or 0)
        record.fusion_count_b4 = int(term_info_dict.get("fusion_count_b4", 0) or 0)
        record.fusion_count_b5 = int(term_info_dict.get("fusion_count_b5", 0) or 0)
    if isinstance(term_info_dict.get("fusion_action_steps"), list):
        record.fusion_action_steps = [
            dict(x) for x in term_info_dict.get("fusion_action_steps", [])
            if isinstance(x, Mapping)
        ]
    # ADR-014 DEBUG: fusion cost shape (raw vs saturated) mirrored from the env.
    if "fusion_cost_fusion_norm" in term_info_dict:
        record.terminal_fusion_norm_raw = float(term_info_dict.get("fusion_cost_fusion_norm", 0.0) or 0.0)
        record.terminal_fusion_norm_saturated = float(
            term_info_dict.get("fusion_cost_fusion_norm_saturated", 0.0) or 0.0
        )
    term_breakdown = term_info_dict.get("reward_breakdown")
    term_metrics = term_info_dict.get("metrics")
    term_probe_diag = term_info_dict.get("probe_diagnostics") or {}
    if term_breakdown is not None:
        record.terminal_priority = int(getattr(term_breakdown, "priority", 0) or 0)
        record.terminal_stab_excess_m1 = float(getattr(term_breakdown, "stab_excess_m1", 0.0) or 0.0)
        record.terminal_stab_excess_m2 = float(getattr(term_breakdown, "stab_excess_m2", 0.0) or 0.0)
        record.terminal_stab_excess_loss = float(getattr(term_breakdown, "stab_excess_loss", 0.0) or 0.0)
        record.terminal_stab_violation = float(getattr(term_breakdown, "stab_violation", 0.0) or 0.0)
        record.terminal_bits_gain = float(getattr(term_breakdown, "bits_drop", 0.0) or 0.0)
        record.terminal_k_gain = float(getattr(term_breakdown, "k_drop", 0.0) or 0.0)
        record.terminal_fusion_gain = float(getattr(term_breakdown, "fusion_gain", 0.0) or 0.0)
        record.terminal_cost_score = float(getattr(term_breakdown, "cost_score", 0.0) or 0.0)
        record.terminal_p3_metric_margin_reward = float(getattr(term_breakdown, "p3_metric_margin_reward", 0.0) or 0.0)
        # ADR-014 DEBUG: persist the barrier/margin (were a black box).
        record.terminal_worst_signed_margin = float(getattr(term_breakdown, "worst_signed_margin", 0.0) or 0.0)
        record.terminal_acc_barrier_sat = float(getattr(term_breakdown, "acc_barrier_sat", 0.0) or 0.0)
        record.terminal_acc_barrier_vio = float(getattr(term_breakdown, "acc_barrier_vio", 0.0) or 0.0)
        record.terminal_near_miss = bool(getattr(term_breakdown, "near_miss", False))
        record.terminal_margin_m1 = float(getattr(term_breakdown, "margin_m1", 0.0) or 0.0)
        record.terminal_margin_m2 = float(getattr(term_breakdown, "margin_m2", 0.0) or 0.0)
        record.terminal_cost_fusion_bonus = float(getattr(term_breakdown, "cost_fusion_bonus", 0.0) or 0.0)
        record.terminal_cost_truncation_bonus = float(getattr(term_breakdown, "cost_truncation_bonus", 0.0) or 0.0)
        record.terminal_cost_bits_tiebreaker = float(getattr(term_breakdown, "cost_bits_tiebreaker", 0.0) or 0.0)
        record.terminal_cost_truncation_step_gain = float(getattr(term_breakdown, "cost_truncation_step_gain", 0.0) or 0.0)
        record.terminal_cost_rank_score = float(getattr(term_breakdown, "cost_rank_score", 0.0) or 0.0)
        record.terminal_cost_rank_fusion = float(getattr(term_breakdown, "cost_rank_fusion", 0.0) or 0.0)
        record.terminal_cost_rank_truncation = float(getattr(term_breakdown, "cost_rank_truncation", 0.0) or 0.0)
        record.terminal_cost_rank_bits = float(getattr(term_breakdown, "cost_rank_bits", 0.0) or 0.0)
        record.terminal_pareto_event_kind = str(getattr(term_breakdown, "pareto_event_kind", "") or "")
        record.terminal_pareto_action_hash = str(getattr(term_breakdown, "pareto_action_hash", "") or "")
        record.terminal_pareto_frontier_removed = int(getattr(term_breakdown, "pareto_frontier_removed", 0) or 0)
    if term_metrics is not None:
        record.terminal_loss_mean = float(getattr(term_metrics, "loss_mean", 0.0) or 0.0)
        record.terminal_loss_std = float(getattr(term_metrics, "loss_std", 0.0) or 0.0)
        record.terminal_metric1_mean = float(getattr(term_metrics, "metric1_mean", 0.0) or 0.0)
        record.terminal_metric2_mean = float(getattr(term_metrics, "metric2_mean", 0.0) or 0.0)
        record.terminal_metric1_std = float(getattr(term_metrics, "metric1_std", 0.0) or 0.0)
        record.terminal_metric2_std = float(getattr(term_metrics, "metric2_std", 0.0) or 0.0)
    if isinstance(term_probe_diag, dict):
        record.terminal_probe_wall_seconds = float(term_probe_diag.get("wall_seconds", 0.0) or 0.0)
        record.terminal_probe_devices = [str(x) for x in (term_probe_diag.get("devices") or [])]
        record.terminal_probe_trial_counts = [
            int(x) for x in (term_probe_diag.get("per_worker_trial_counts") or [])
        ]
        record.terminal_probe_trial_indices = [
            [int(y) for y in (x or [])]
            for x in (term_probe_diag.get("per_worker_trial_indices") or [])
        ]
        record.terminal_probe_speedup = float(
            term_probe_diag.get("speedup_vs_sequential", 1.0) or 1.0
        )
        record.terminal_cost_eval_wall_seconds = float(
            term_probe_diag.get("cost_eval_wall_seconds", 0.0) or 0.0
        )
        record.terminal_probe_install_wall_seconds = float(
            term_probe_diag.get("probe_install_wall_seconds", 0.0) or 0.0
        )
        record.terminal_probe_clear_wall_seconds = float(
            term_probe_diag.get("probe_clear_wall_seconds", 0.0) or 0.0
        )
        record.terminal_probe_install_skipped = bool(term_probe_diag.get("probe_install_skipped", False))
        record.terminal_probe_clear_skipped = bool(term_probe_diag.get("probe_clear_skipped", False))
    if cached_reward_hit:
        record.exploration_mode = f"{record.exploration_mode}|cached_reward_hit"
    if validation_required:
        record.exploration_mode = f"{record.exploration_mode}|validation_required"


def _protected_step_actions(
        *,
        spec: Any,
        n_active: int,
        baseline_action_vec: Optional[np.ndarray],
        base_action_vec_for_mask: Optional[np.ndarray],
        ) -> List[List[int]]:
    protected_actions: List[List[int]] = []
    offsets = list(spec.full_vec_offsets)
    for vec in (baseline_action_vec, base_action_vec_for_mask):
        if vec is None:
            continue
        arr = np.asarray(vec, dtype=np.int64).reshape(-1)
        protected_actions.append(
            arr[offsets][:int(n_active)].reshape(-1).astype(np.int64).tolist()
        )
    return protected_actions


def _precompute_static_invalid_level_mask(
        *,
        env: BLBStage2SequentialEnv,
        baseline_action_vec: Sequence[int],
        enabled: bool,
        log_fn: Optional[Callable[[str], None]] = None,
        bullet: str = "*",
        ) -> Tuple[StaticInvalidLevelMask, Dict[str, Any]]:
    """Build an aggressive pre-RL level mask from baseline-prefix checks.

    The scan changes one slot at a time, asks the real optimizer whether the
    current block remains chain-valid, and records invalid levels. It advances
    the scan context by committing the baseline action for non-terminal steps
    only, so it never triggers the terminal model-forward reward probe.
    """
    out = StaticInvalidLevelMask()
    summary: Dict[str, Any] = {
        "enabled": bool(enabled),
        "evaluated": 0,
        "invalid": 0,
        "disabled": 0,
        "aborted": False,
        "reason": "",
        "elapsed_seconds": 0.0,
    }
    if not bool(enabled):
        return out, summary

    baseline = np.asarray(baseline_action_vec, dtype=np.int64).reshape(-1)
    t0 = time.perf_counter()
    try:
        env.reset()
        horizon = int(env.horizon)
        for step_idx in range(horizon):
            spec = env.current_spec()
            offsets = list(spec.full_vec_offsets)
            n_active = len(spec.slot_dims)
            baseline_slice = baseline[offsets][:n_active].astype(np.int64).copy()

            for slot_idx, dim in enumerate(spec.slot_dims):
                dim_int = int(dim)
                baseline_idx = int(baseline_slice[slot_idx])
                for level_idx in range(dim_int):
                    if int(level_idx) == baseline_idx:
                        continue
                    candidate = baseline_slice.copy()
                    candidate[int(slot_idx)] = int(level_idx)
                    info = env.evaluate_step(candidate.tolist())
                    summary["evaluated"] = int(summary["evaluated"]) + 1
                    if not bool(info.get("valid", False)):
                        summary["invalid"] = int(summary["invalid"]) + 1
                        out.add_invalid(
                            spec.layer_idx,
                            spec.block_idx,
                            int(slot_idx),
                            int(level_idx),
                            reason=_format_invalid_chain_reason(info.get("invalid_chain")),
                            config_name=str(info.get("config_name", "")),
                        )

            if step_idx >= horizon - 1:
                break
            baseline_info = env.evaluate_step(baseline_slice.tolist())
            if not bool(baseline_info.get("valid", False)):
                summary["aborted"] = True
                summary["reason"] = (
                    f"baseline prefix invalid at step={int(step_idx)} "
                    f"L{int(spec.layer_idx)}-B{int(spec.block_idx)}: "
                    f"{_format_invalid_chain_reason(baseline_info.get('invalid_chain'))}"
                )
                out = StaticInvalidLevelMask()
                break
            env.commit_step(baseline_info)
    except Exception as exc:
        summary["aborted"] = True
        summary["reason"] = f"scan_exception: {exc}"
        out = StaticInvalidLevelMask()
    finally:
        summary["elapsed_seconds"] = float(time.perf_counter() - t0)
        summary["disabled"] = int(out.total_disabled())
        try:
            env.reset()
        except Exception as exc:
            summary["reset_warning"] = str(exc)

    if log_fn is not None:
        if bool(summary.get("aborted")):
            log_fn(
                f"  {bullet} static invalid-level pre-scan aborted: "
                f"{summary.get('reason', '')}"
            )
        else:
            log_fn(
                f"  {bullet} static invalid-level pre-scan: "
                f"evaluated={int(summary['evaluated'])}, "
                f"invalid={int(summary['invalid'])}, "
                f"disabled={int(summary['disabled'])}, "
                f"elapsed={float(summary['elapsed_seconds']):.2f}s"
            )
    return out, summary


def train_sequential(
        *,
        env: BLBStage2SequentialEnv,
        policy: BLBStage2SequentialPolicy,
        train_cfg: Optional[SequentialTrainConfig] = None,
        device: Optional[torch.device] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        on_episode_end: Optional[Callable[[EpisodeRecord], None]] = None,
        on_ppo_update_end: Optional[
            Callable[[Dict[str, float], int, "EpisodeRecord"], None]
        ] = None,
        on_step_end: Optional[
            Callable[[int, int, Dict[str, Any]], None]
        ] = None,
        capture_step_infos: bool = False,
        logger: Optional[logging.Logger] = None,
        forbidden_mask: Optional[ForbiddenActionMask] = None,
        static_invalid_mask: Optional[StaticInvalidLevelMask] = None,
        static_invalid_scan_summary: Optional[Mapping[str, Any]] = None,
        empirical_invalid_mask: Optional[EmpiricalInvalidLevelMask] = None,
        # OSR pre-prune mask (opt-in, 2026-05-27). Lives alongside the three
        # masks above; populated by ``blb_stage2_rl.osr.run_osr_scan`` before
        # RL starts (sidecar preset writes ``osr_results.json``). When None,
        # the training loop behaves exactly as before.
        osr_mask: Optional["OSRPrePruneMask"] = None,
        baseline_action_vec: Optional[np.ndarray] = None,
        max_rejection_retries: int = 32,
        force_baseline_episodes: int = 0,
        # Episode-parallel rollout (fusion mode only): a Stage2ParallelRunner.
        # When set, episodes are collected window-by-window across its workers
        # (global-episode seeding, global-order assembly) and the serial loop
        # below is bypassed. None → legacy behavior bit-for-bit.
        parallel_runner: Optional[Any] = None,
        ) -> Dict[str, object]:
    """Train ``policy`` on ``env`` with sequential PPO.

    Args:
        env:                Sequential env (horizon = 59 for L=12).
        policy:             Sequential actor-critic.
        train_cfg:          Hyper-params; defaults to a conservative SequentialTrainConfig.
        device:             Where to store tensors; defaults to model's parameter device.
        on_episode_end:     Optional callback fired at the end of each episode with
                            the EpisodeRecord. Use it to save checkpoints / status
                            / rich-log per-episode summaries.
        on_ppo_update_end:  Optional callback fired after every PPO update with
                            ``(ppo_metrics_dict, completed_episode_count, last_episode_record)``.
                            Use it to print boxed PPO summary / progress bar / ETA.
        on_step_end:        Optional callback fired after every env.step with
                            ``(episode_idx, step_idx_within_episode, info_dict)``.
                            Use sparingly (firing 59 times per episode); typical
                            pattern is to log only for the first episode.
        capture_step_infos: If True, attach the per-step info dicts to the
                            EpisodeRecord. Default False to save memory.
        logger:             Optional logger; otherwise uses module-level logging.

    Returns dict with episode_rewards / ppo_metrics / final_invalid_rate.
    """
    train_cfg = train_cfg or SequentialTrainConfig()
    train_cfg.rl_algo = _normalize_supported_rl_algo(
        getattr(train_cfg, "rl_algo", "ppo"), context="SequentialTrainConfig.rl_algo"
    )
    train_cfg.grpo_kl_beta = 0.0
    device = device or next(policy.parameters()).device
    log = logger or logging.getLogger(__name__)
    optimizer = optimizer or torch.optim.Adam(policy.parameters(), lr=train_cfg.ppo.lr)
    buffer = SequentialRolloutBuffer()
    # PPO's stochasticity should come from the categorical action distribution
    # and stored log-probs, not from unrecorded dropout masks. Keeping the
    # actor-critic in eval mode also avoids many tiny dropout kernels in the
    # 59-step online rollout path; gradients still flow during PPO updates.
    policy.eval()

    # Per-(layer, block) blacklist of action tuples that produced invalid_chain.
    # Survives across episodes within this train_sequential call. If a caller
    # supplied an existing mask (e.g. resumed from checkpoint), keep its entries
    # so we don't re-discover the same failures.
    if forbidden_mask is None:
        forbidden_mask = ForbiddenActionMask()
    static_invalid_enabled = bool(
        getattr(train_cfg, "static_invalid_level_mask_enabled", False)
    )
    if not static_invalid_enabled:
        static_invalid_mask = None
    static_invalid_scan_summary = dict(static_invalid_scan_summary or {})
    empirical_invalid_enabled = bool(
        getattr(train_cfg, "empirical_invalid_level_mask_enabled", False)
    )
    if empirical_invalid_enabled and empirical_invalid_mask is None:
        empirical_invalid_mask = EmpiricalInvalidLevelMask(
            min_invalid_samples=int(
                getattr(train_cfg, "empirical_invalid_level_min_samples", 3)
            ),
            min_invalid_rate=float(
                getattr(train_cfg, "empirical_invalid_level_min_rate", 0.80)
            ),
            max_valid_samples=int(
                getattr(train_cfg, "empirical_invalid_level_max_valid", 0)
            ),
        )
    if not empirical_invalid_enabled:
        empirical_invalid_mask = None
    if baseline_action_vec is not None:
        baseline_action_vec = np.asarray(baseline_action_vec, dtype=np.int64).reshape(-1)
    force_baseline_episodes = max(0, int(force_baseline_episodes))
    if force_baseline_episodes > 0 and baseline_action_vec is None:
        log.warning(
            "[seqRL] force_baseline_episodes=%d requested but baseline_action_vec is None — "
            "skipping forced-baseline warmstart",
            force_baseline_episodes,
        )
        force_baseline_episodes = 0
    rejection_counters = {
        "samples_rejected_by_mask": 0,   # rejected before optimizer call (cheap)
        "samples_rejected_by_optimizer": 0,  # called optimizer, got invalid → blacklisted
        "steps_fallen_back_to_baseline": 0,  # max_retries exhausted, used baseline_step_action
        "steps_committed_valid": 0,
        "steps_committed_invalid": 0,    # only when no baseline available + retries exhausted
        "steps_forced_to_baseline_anchor": 0,  # forced via force_baseline_episodes
    }

    episode_returns: List[float] = []
    episode_records: List[EpisodeRecord] = []
    ppo_metric_history: List[Dict[str, float]] = []
    ppo_update_wall_seconds_accum = 0.0
    episode_callback_wall_seconds_accum = 0.0
    frontier_seed_actions: List[Tuple[int, np.ndarray]] = []

    if train_cfg.seed is not None:
        torch.manual_seed(int(train_cfg.seed))
        np.random.seed(int(train_cfg.seed) % (2**32))

    absolute_episode_start = max(0, int(getattr(train_cfg, "absolute_episode_start", 0) or 0))
    mutable_full_offsets = getattr(train_cfg, "warmstart_mutable_full_offsets", None)
    if mutable_full_offsets is not None:
        mutable_full_offsets = [int(x) for x in mutable_full_offsets]
    guarded_radius2 = GuardedRadius2Controller(
        enabled=bool(getattr(train_cfg, "guarded_radius2_enabled", False)),
        min_episode=int(getattr(train_cfg, "guarded_radius2_min_episode", 1060)),
        stall_window=int(getattr(train_cfg, "guarded_radius2_stall_window", 600)),
        health_window=int(getattr(train_cfg, "guarded_radius2_health_window", 100)),
        max_mutations=int(getattr(train_cfg, "guarded_radius2_max_mutations", 4)),
        episode_fraction=float(getattr(train_cfg, "guarded_radius2_episode_fraction", 0.15)),
        cooldown_episodes=int(getattr(train_cfg, "guarded_radius2_cooldown_episodes", 300)),
        min_radius1_successes=int(
            getattr(train_cfg, "guarded_radius2_min_radius1_successes", 3)
        ),
    )
    fast_reward_mode_enabled = bool(
        getattr(train_cfg, "fast_reward_mode_enabled", False)
    )
    online_num_trials_per_step = max(
        1, int(getattr(train_cfg, "online_num_trials_per_step", 5) or 5)
    )
    terminal_eval_batch_size = max(
        1, int(getattr(train_cfg, "terminal_eval_batch_size", 1) or 1)
    )
    promotion_validation_trials = max(
        1, int(getattr(train_cfg, "promotion_validation_trials", 1) or 1)
    )
    promotion_margin_window = float(
        getattr(train_cfg, "promotion_margin_window", 0.25) or 0.0
    )
    if fast_reward_mode_enabled and getattr(env.base, "probe_runner", None) is None:
        log.warning(
            "[seqRL] fast_reward_mode_enabled requested without ProbeRunner; "
            "falling back to the standard terminal reward path"
        )
        fast_reward_mode_enabled = False
    if fast_reward_mode_enabled:
        terminal_eval_batch_size = min(
            terminal_eval_batch_size,
            int(getattr(env.base.probe_runner, "num_workers", terminal_eval_batch_size)),
        )
        log.info(
            "[seqRL] fast reward mode enabled: online_num_trials_per_step=%d "
            "terminal_eval_batch_size=%d promotion_validation_trials=%d "
            "promotion_margin_window=%.3f",
            online_num_trials_per_step,
            terminal_eval_batch_size,
            promotion_validation_trials,
            promotion_margin_window,
        )

    pending_terminal_drafts: List[Dict[str, Any]] = []
    terminal_metric_cache: Dict[str, Any] = {}
    validation_metric_cache: Dict[str, Any] = {}
    best_online_reward_seen = -float("inf")
    best_online_cost_rank_seen = -float("inf")

    def _finalize_completed_record(draft: Dict[str, Any]) -> None:
        nonlocal best_online_reward_seen, best_online_cost_rank_seen
        nonlocal ppo_update_wall_seconds_accum
        nonlocal episode_callback_wall_seconds_accum
        record: EpisodeRecord = draft["record"]
        absolute_ep_local = int(draft["absolute_ep"])
        selected_offsets_local = sorted(int(x) for x in draft["selected_offsets"])
        neighbor_radius_local = int(draft["neighbor_radius"])
        neighbor_mask_active_local = bool(draft["neighbor_mask_active"])
        guarded_decision_local: GuardedRadius2Decision = draft["guarded_decision"]
        pending_full_vec_local = draft.get("pending_full_vec")

        episode_returns.append(float(record.total_reward))
        is_new_cost_rank_seed = bool(
            int(record.terminal_priority) == 3
            and int(record.invalid_steps) == 0
            and float(record.terminal_cost_rank_score) > float(best_online_cost_rank_seen)
        )
        best_online_reward_seen = max(best_online_reward_seen, float(record.total_reward))
        if int(record.terminal_priority) == 3 and int(record.invalid_steps) == 0:
            best_online_cost_rank_seen = max(
                best_online_cost_rank_seen,
                float(record.terminal_cost_rank_score),
            )
        guarded_radius2.record_episode(
            absolute_episode_idx=int(absolute_ep_local),
            selected_offsets=selected_offsets_local,
            radius=int(neighbor_radius_local if neighbor_mask_active_local else 0),
            terminal_priority=int(record.terminal_priority),
            invalid_steps=int(record.invalid_steps),
            early_terminated=bool(record.early_terminated),
            terminal_stab_violation=float(record.terminal_stab_violation),
            terminal_loss_mean=float(record.terminal_loss_mean),
            terminal_pareto_event_kind=str(record.terminal_pareto_event_kind),
            terminal_fusion_gain=float(record.terminal_fusion_gain),
            terminal_k_gain=float(record.terminal_k_gain),
            terminal_bits_gain=float(record.terminal_bits_gain),
        )
        if bool(record.guarded_radius2_active):
            after_decision = guarded_radius2.decide(
                absolute_episode_idx=int(absolute_ep_local + 1),
                rng=np.random.default_rng(0),
            )
            record.guarded_radius2_cooldown_remaining = int(
                after_decision.cooldown_remaining
            )
            record.guarded_radius2_episode_count = int(
                guarded_radius2.radius2_episode_count
            )
            record.guarded_radius2_failure_count = int(
                guarded_radius2.radius2_failure_count
            )
            record.guarded_radius2_frontier_expansion_count = int(
                guarded_radius2.radius2_frontier_expansion_count
            )
        episode_records.append(record)
        if (
                int(record.terminal_priority) == 3
                and (
                    str(record.terminal_pareto_event_kind) in {
                        "frontier_expansion",
                        "frontier_member",
                    }
                    or is_new_cost_rank_seed
                )
                and pending_full_vec_local is not None
        ):
            frontier_seed_actions.append((
                int(absolute_ep_local),
                np.asarray(pending_full_vec_local, dtype=np.int64).copy(),
            ))
            if len(frontier_seed_actions) > 64:
                del frontier_seed_actions[:-64]
        _attach_pending_full_vec_for_callback(record, pending_full_vec_local)
        if on_episode_end is not None:
            episode_callback_t0 = time.perf_counter()
            try:
                on_episode_end(record)
            finally:
                episode_callback_wall_seconds_accum += float(
                    time.perf_counter() - episode_callback_t0
                )

        if (int(record.episode_idx) + 1) % int(train_cfg.update_every_n_episodes) == 0:
            _ep_1based = int(absolute_episode_start + int(record.episode_idx) + 1)
            if str(getattr(train_cfg, "ent_coef_schedule", "anchor_ramp")) == "cosine":
                # ADR-015: Stage-1 cosine schedule (high→low, no anchor).
                current_ent_coef = _resolve_cosine_ent_coef_schedule(
                    _ep_1based,
                    int(train_cfg.total_episodes),
                    start=float(getattr(train_cfg, "ent_coef_cosine_start", 0.05)),
                    end=float(getattr(train_cfg, "ent_coef_cosine_end", 0.001)),
                    plateau_ratio=float(getattr(train_cfg, "ent_coef_cosine_plateau", 0.25)),
                    lower_bound=float(getattr(train_cfg, "ent_coef_cosine_lower_bound", 0.012)),
                )
            else:
                current_ent_coef = _resolve_ent_coef_schedule(
                    ep_count_1based=_ep_1based,
                    anchor_episodes=int(force_baseline_episodes),
                    target_ent_coef=float(train_cfg.ppo.ent_coef),
                    anchor_ent_coef=float(getattr(train_cfg, "ent_coef_anchor", 0.0)),
                    ramp_episodes=int(getattr(train_cfg, "ent_coef_ramp_episodes", 600)),
                )
            if parallel_runner is not None:
                # Deterministic pre-update reseed: the minibatch shuffle in
                # sequential_ppo_update consumes the global numpy RNG; keying
                # it by (seed, update_idx) keeps the UPDATE identical for any
                # GPU count (worker rollouts no longer advance these streams).
                from .seed_utils import derive_update_seed
                _update_seed = derive_update_seed(
                    int(train_cfg.seed or 0), len(ppo_metric_history),
                )
                torch.manual_seed(int(_update_seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(_update_seed))
                np.random.seed(int(_update_seed) % (2**32))
            ppo_update_t0 = time.perf_counter()
            metrics = sequential_ppo_update(
                policy, optimizer, buffer, train_cfg.ppo, device,
                ent_coef_override=current_ent_coef,
            )
            ppo_update_wall = float(time.perf_counter() - ppo_update_t0)
            metrics["ppo_update_wall_seconds"] = float(ppo_update_wall)
            ppo_update_wall_seconds_accum += float(ppo_update_wall)
            ppo_metric_history.append(metrics)
            buffer.clear()
            if on_ppo_update_end is not None:
                try:
                    on_ppo_update_end(dict(metrics), int(record.episode_idx) + 1, record)
                except Exception:
                    pass
            if (int(record.episode_idx) + 1) % int(train_cfg.log_every_n_episodes) == 0:
                log.info(
                    "[seqRL] ep=%d return=%.3f invalid_steps=%d ppo: %s",
                    int(record.episode_idx), record.total_reward, record.invalid_steps, metrics,
                )

    def flush_pending_terminal_drafts(reason: str) -> None:
        if not pending_terminal_drafts:
            return
        drafts = list(pending_terminal_drafts)
        pending_terminal_drafts.clear()
        prepared_by_draft: List[Optional[Mapping[str, Any]]] = [None] * len(drafts)
        uncached_indices: List[int] = []
        for idx, draft in enumerate(drafts):
            terminal_vec = np.asarray(draft["terminal_action_vec"], dtype=np.int64)
            action_key = str(draft.get("terminal_action_hash") or "")
            if not action_key:
                from .candidate_store import action_hash as _action_hash
                action_key = _action_hash(terminal_vec)
                draft["terminal_action_hash"] = action_key
            prepared = env.base.prepare_action_for_terminal_probe(terminal_vec)
            prepared_by_draft[idx] = prepared
            cached_metrics = terminal_metric_cache.get(action_key)
            if cached_metrics is not None:
                result = env.base._finish_prepared_terminal_probe(
                    prepared,
                    cached_metrics,
                    probe_diagnostics={
                        "fast_reward_mode": True,
                        "cached_reward_hit": True,
                        "flush_reason": str(reason),
                        "online_num_trials_per_step": int(online_num_trials_per_step),
                        "terminal_eval_batch_size": int(terminal_eval_batch_size),
                        "validation_required": False,
                    },
                    forward_ran=False,
                )
                _state, terminal_reward_local, _done, term_info = result
                draft["terminal_result"] = result
                buffer.add_reward_at(int(draft["terminal_buffer_index"]), terminal_reward_local)
                _apply_terminal_info_to_record(
                    draft["record"], terminal_reward_local, term_info,
                    cached_reward_hit=True,
                    validation_required=False,
                )
            else:
                uncached_indices.append(idx)

        for start in range(0, len(uncached_indices), int(terminal_eval_batch_size)):
            batch_indices = uncached_indices[start:start + int(terminal_eval_batch_size)]
            prepared_batch = [
                prepared_by_draft[i] for i in batch_indices
                if prepared_by_draft[i] is not None
            ]
            results = env.base.evaluate_prepared_terminal_batch(
                prepared_batch,
                num_trials_per_action=int(online_num_trials_per_step),
                validation_required=False,
            )
            for local_idx, result in enumerate(results):
                draft_idx = batch_indices[local_idx]
                draft = drafts[draft_idx]
                _state, terminal_reward_local, _done, term_info = result
                draft["terminal_result"] = result
                buffer.add_reward_at(int(draft["terminal_buffer_index"]), terminal_reward_local)
                _apply_terminal_info_to_record(
                    draft["record"], terminal_reward_local, term_info,
                    cached_reward_hit=False,
                    validation_required=False,
                )
                metrics_obj = term_info.get("metrics")
                action_key = str(draft.get("terminal_action_hash") or "")
                if metrics_obj is not None and action_key:
                    terminal_metric_cache[action_key] = metrics_obj

        for draft_idx, draft in enumerate(drafts):
            record: EpisodeRecord = draft["record"]
            action_key = str(draft.get("terminal_action_hash") or "")
            has_prior_best = bool(math.isfinite(float(best_online_reward_seen)))
            has_prior_cost_rank = bool(math.isfinite(float(best_online_cost_rank_seen)))
            validation_required = bool(
                int(record.terminal_priority) == 3
                and int(promotion_validation_trials) > int(online_num_trials_per_step)
                and (
                    str(record.terminal_pareto_event_kind) in {"frontier_expansion", "frontier_member"}
                    or (
                        has_prior_cost_rank
                        and float(record.terminal_cost_rank_score) > float(best_online_cost_rank_seen)
                    )
                    or (
                        has_prior_best
                        and float(record.total_reward) >= float(best_online_reward_seen - promotion_margin_window)
                    )
                )
            )
            if validation_required:
                cached_validation = validation_metric_cache.get(action_key)
                prepared = prepared_by_draft[draft_idx]
                if prepared is None:
                    prepared = env.base.prepare_action_for_terminal_probe(
                        np.asarray(draft["terminal_action_vec"], dtype=np.int64)
                    )
                if cached_validation is None:
                    validation_results = env.base.evaluate_prepared_terminal_batch(
                        [prepared],
                        num_trials_per_action=int(promotion_validation_trials),
                        validation_required=True,
                    )
                    validation_result = validation_results[0]
                    _state, validation_reward, _done, validation_info = validation_result
                    validation_metrics = validation_info.get("metrics")
                    if validation_metrics is not None and action_key:
                        validation_metric_cache[action_key] = validation_metrics
                else:
                    validation_result = env.base._finish_prepared_terminal_probe(
                        prepared,
                        cached_validation,
                        probe_diagnostics={
                            "fast_reward_mode": True,
                            "cached_reward_hit": True,
                            "validation_required": True,
                            "promotion_validation": True,
                        },
                        forward_ran=False,
                    )
                    _state, validation_reward, _done, validation_info = validation_result

                validation_priority = int(
                    getattr(validation_info.get("reward_breakdown"), "priority", 0) or 0
                )
                if validation_priority < int(record.terminal_priority):
                    delta = float(validation_reward) - float(record.terminal_reward)
                    buffer.add_reward_at(int(draft["terminal_buffer_index"]), delta)
                    _apply_terminal_info_to_record(
                        record, float(validation_reward), validation_info,
                        cached_reward_hit=False,
                        validation_required=True,
                    )
                elif "|validation_required" not in record.exploration_mode:
                    record.exploration_mode = f"{record.exploration_mode}|validation_required"

            _finalize_completed_record(draft)

    # ---------- Episode-parallel branch (fusion mode; bypasses the serial loop) ----------
    if parallel_runner is not None:
        if fast_reward_mode_enabled:
            raise RuntimeError(
                "episode-parallel rollout is incompatible with fast reward mode"
            )
        if baseline_action_vec is None:
            raise RuntimeError(
                "episode-parallel rollout requires baseline_action_vec"
            )
        total = int(train_cfg.total_episodes)
        upd = max(1, int(train_cfg.update_every_n_episodes))
        for window_start in range(0, total, upd):
            n_window = min(upd, total - window_start)
            window_t0 = time.perf_counter()
            ppo_wall_before = float(ppo_update_wall_seconds_accum)
            callback_wall_before = float(episode_callback_wall_seconds_accum)
            outcomes = parallel_runner.run_window(
                policy=policy,
                train_cfg=train_cfg,
                window_rel_start=int(window_start),
                num_episodes=int(n_window),
                absolute_episode_start=int(absolute_episode_start),
                base_seed=int(train_cfg.seed or 0),
                baseline_action_vec=baseline_action_vec,
                force_baseline_episodes=int(force_baseline_episodes),
                forbidden_mask=forbidden_mask,
                max_rejection_retries=int(max_rejection_retries),
            )
            collect_wall = float(time.perf_counter() - window_t0)
            assembly_t0 = time.perf_counter()
            buffer_add_wall = 0.0
            finalize_wall = 0.0
            # Main-thread assembly in GLOBAL episode order: replay transitions
            # into the shared buffer, then run the unchanged finalize path
            # (bookkeeping, callbacks, PPO update at window boundaries).
            for oc in outcomes:
                terminal_buffer_index = -1
                buffer_t0 = time.perf_counter()
                for tr in oc.transitions:
                    buffer_idx = buffer.add(**tr)
                    if bool(tr.get("done")):
                        terminal_buffer_index = int(buffer_idx)
                buffer_add_wall += float(time.perf_counter() - buffer_t0)
                rejection_counters["steps_committed_valid"] += int(oc.record.valid_step_count)
                rejection_counters["samples_rejected_by_mask"] += int(
                    oc.record.samples_rejected_by_mask
                )
                rejection_counters["samples_rejected_by_optimizer"] += int(
                    oc.record.samples_rejected_by_optimizer
                )
                rejection_counters["steps_fallen_back_to_baseline"] += int(
                    oc.record.steps_fallen_back_to_baseline
                )
                if oc.record.exploration_mode == "forced_baseline":
                    rejection_counters["steps_forced_to_baseline_anchor"] += int(
                        oc.record.steps_taken
                    )
                finalize_t0 = time.perf_counter()
                _finalize_completed_record({
                    "record": oc.record,
                    "absolute_ep": int(oc.absolute_ep),
                    "selected_offsets": [],
                    "neighbor_radius": 0,
                    "neighbor_mask_active": False,
                    "guarded_decision": GuardedRadius2Decision(),
                    "pending_full_vec": oc.pending_full_vec,
                    "terminal_buffer_index": int(terminal_buffer_index),
                })
                finalize_wall += float(time.perf_counter() - finalize_t0)
            assembly_wall = float(time.perf_counter() - assembly_t0)
            ppo_update_wall = float(ppo_update_wall_seconds_accum - ppo_wall_before)
            episode_callback_wall = float(
                episode_callback_wall_seconds_accum - callback_wall_before
            )
            finalize_other_wall = max(
                0.0,
                float(finalize_wall) - float(ppo_update_wall) - float(episode_callback_wall),
            )
            log.info(
                f"  [stage2-rollout-timing] window_start={int(window_start)} "
                f"episodes={int(n_window)} collect_total_s={collect_wall:.3f} "
                f"assembly_update_s={assembly_wall:.3f} "
                f"buffer_add_s={buffer_add_wall:.3f} "
                f"finalize_update_s={finalize_wall:.3f} "
                f"episode_callback_s={episode_callback_wall:.3f} "
                f"finalize_other_s={finalize_other_wall:.3f} "
                f"ppo_update_s={ppo_update_wall:.3f} "
                f"window_total_s={float(time.perf_counter() - window_t0):.3f}"
            )
        return {
            "episode_returns": episode_returns,
            "episode_records": episode_records,
            "ppo_metrics": ppo_metric_history,
            "final_invalid_rate": (
                float(np.mean([r.invalid_steps > 0 for r in episode_records[-10:]]))
                if episode_records else 0.0
            ),
            "forbidden_mask": forbidden_mask,
            "static_invalid_mask": static_invalid_mask,
            "static_invalid_scan_summary": dict(static_invalid_scan_summary),
            "empirical_invalid_mask": empirical_invalid_mask,
            "rejection_counters": dict(rejection_counters),
        }

    step_static_tensors = _get_cached_step_static_tensors(
        env,
        max_step_dim=policy.cfg.max_step_dim,
        max_num_levels=policy.cfg.max_num_levels,
        device=device,
    )

    for ep in range(int(train_cfg.total_episodes)):
        absolute_ep = int(absolute_episode_start + ep)
        # Only seed the env's RNG on the very first reset; subsequent resets
        # advance the RNG to avoid identical rollouts every episode.
        seed_for_this_ep = (
            int(train_cfg.seed) + int(absolute_ep)
            if (train_cfg.seed is not None and ep == 0)
            else None
        )
        obs = env.reset(seed=seed_for_this_ep)
        per_step_sum = 0.0
        terminal_reward = 0.0
        invalid_steps = 0
        valid_step_count = 0
        total_bits_sum = 0
        fusion_count_sum = 0
        first_invalid: Optional[Dict[str, int]] = None
        early_terminated = False
        steps_taken = 0
        captured_step_infos: List[Dict[str, Any]] = []
        invalid_block_details: List[Dict[str, Any]] = []
        per_step_optimizer_wall_seconds_val = 0.0
        policy_rollout_wall_seconds_val = 0.0
        # Terminal breakdown extracted from the last commit_step's info dict.
        # 0 / 0.0 means the episode never produced a terminal reward (e.g.,
        # early_terminate_on_invalid fired before the last step).
        terminal_priority_int = 0
        terminal_final_config_fingerprint_val = ""
        terminal_materialization_failure_reason_val = ""
        terminal_model_uses_replan_config_val = False
        terminal_loss_mean_val = 0.0
        terminal_loss_std_val = 0.0
        terminal_metric1_val = 0.0
        terminal_metric2_val = 0.0
        terminal_metric1_std_val = 0.0
        terminal_metric2_std_val = 0.0
        terminal_stab_excess_m1_val = 0.0
        terminal_stab_excess_m2_val = 0.0
        terminal_stab_excess_loss_val = 0.0
        terminal_stab_violation_val = 0.0
        terminal_bits_gain_val = 0.0
        terminal_k_gain_val = 0.0
        terminal_fusion_gain_val = 0.0
        terminal_cost_score_val = 0.0
        terminal_p3_metric_margin_reward_val = 0.0
        terminal_cost_fusion_bonus_val = 0.0
        terminal_cost_truncation_bonus_val = 0.0
        terminal_cost_bits_tiebreaker_val = 0.0
        terminal_cost_truncation_step_gain_val = 0.0
        terminal_cost_rank_score_val = 0.0
        terminal_cost_rank_fusion_val = 0.0
        terminal_cost_rank_truncation_val = 0.0
        terminal_cost_rank_bits_val = 0.0
        terminal_pareto_event_kind_val = ""
        terminal_pareto_action_hash_val = ""
        terminal_pareto_frontier_removed_val = 0
        terminal_probe_wall_seconds_val = 0.0
        terminal_probe_devices_val: List[str] = []
        terminal_probe_trial_counts_val: List[int] = []
        terminal_probe_trial_indices_val: List[List[int]] = []
        terminal_probe_speedup_val = 1.0
        fusion_action_steps_val: List[Dict[str, Any]] = []
        terminal_cost_eval_wall_seconds_val = 0.0
        terminal_probe_install_wall_seconds_val = 0.0
        terminal_probe_clear_wall_seconds_val = 0.0
        terminal_probe_install_skipped_val = False
        terminal_probe_clear_skipped_val = False
        terminal_buffer_index = -1
        terminal_deferred = False
        terminal_deferred_action_vec: Optional[np.ndarray] = None
        rejection_start = dict(rejection_counters)
        rejection_optimizer_wall_seconds_val = 0.0
        static_invalid_level_applied_val = 0
        empirical_invalid_level_applied_val = 0
        _baseline_prior_enabled = (
            bool(getattr(train_cfg, "warmstart_baseline_bias", False))
            or int(force_baseline_episodes) > 0
        )
        baseline_prior_scale = (
            _resolve_baseline_prior_scale(
                int(absolute_ep),
                anchor_episodes=int(force_baseline_episodes),
            )
            if _baseline_prior_enabled else 0.0
        )

        # 2026-05-18 (rdv2 hotfix): forced-baseline anchor episodes. The
        # warmstart bias on the action head alone was inadequate — only 5/13
        # slot positions had a preferred index matching the actual baseline
        # value (the others were either out of range or pointed at a
        # non-baseline K). With ~84% per-slot bias on those 5 slots and
        # uniform sampling on the other 8, virtually no rollout matched
        # baseline closely enough to satisfy acc_threshold, so every
        # candidate landed in metric_ok=False (priority 1) → reward
        # collapsed to -5 to -7. To bootstrap the value function and shift
        # policy mass toward baseline before exploration, the FIRST
        # ``force_baseline_episodes`` episodes execute the baseline action
        # at every step (no sampling, no rejection). PPO still records the
        # state/action/log_prob/value/reward via ``policy.evaluate_action``
        # so the update pushes the policy probability mass toward baseline
        # while the value head learns the +45 baseline reward.
        force_this_ep = (
            force_baseline_episodes > 0
            and int(absolute_ep) < int(force_baseline_episodes)
        )

        neighbor_mask_active = False
        neighbor_selected_offsets: Set[int] = set()
        neighbor_radius = 1
        guarded_decision = GuardedRadius2Decision()
        base_action_vec_for_mask = baseline_action_vec
        base_action_source = "baseline"
        frontier_seed_episode = -1
        proposal_direction = "none"
        if (
                (not force_this_ep)
                and bool(getattr(train_cfg, "warmstart_neighbor_sampling", False))
                and baseline_action_vec is not None
        ):
            neighbor_mutations, neighbor_radius = _sequential_neighbor_curriculum(
                absolute_episode_idx=int(absolute_ep),
                anchor_episodes=int(force_baseline_episodes),
                ramp_episodes=int(
                    getattr(train_cfg, "warmstart_neighbor_ramp_episodes", 0)
                    or int(train_cfg.total_episodes)
                    or 1
                ),
                max_mutations=int(getattr(train_cfg, "warmstart_neighbor_max_mutations", 8)),
                max_radius=int(getattr(train_cfg, "warmstart_neighbor_max_radius", 2)),
            )
            base_neighbor_mutations = int(neighbor_mutations)
            base_neighbor_radius = int(neighbor_radius)
            seed_base = int(train_cfg.seed) if train_cfg.seed is not None else 0
            episode_rng = np.random.default_rng(
                int((seed_base + absolute_ep * 1_000_003) % (2**32))
            )
            if frontier_seed_actions:
                draw = float(episode_rng.random())
                if int(absolute_ep) < 600:
                    use_frontier = draw < 0.30
                else:
                    use_frontier = draw < 0.50
                    if draw >= 0.85:
                        base_action_source = "exploratory"
                if use_frontier:
                    seed_ep, seed_vec = frontier_seed_actions[
                        int(episode_rng.integers(0, len(frontier_seed_actions)))
                    ]
                    base_action_vec_for_mask = np.asarray(seed_vec, dtype=np.int64)
                    base_action_source = "frontier"
                    frontier_seed_episode = int(seed_ep)
            elif int(absolute_ep) >= 600:
                base_action_source = "exploratory"
            neighbor_mask_active = True
            guarded_decision = guarded_radius2.decide(
                absolute_episode_idx=int(absolute_ep),
                rng=episode_rng,
            )
            if guarded_decision.active:
                if frontier_seed_actions and float(episode_rng.random()) < 0.60:
                    seed_ep, seed_vec = frontier_seed_actions[
                        int(episode_rng.integers(0, len(frontier_seed_actions)))
                    ]
                    base_action_vec_for_mask = np.asarray(seed_vec, dtype=np.int64)
                    base_action_source = "frontier"
                    frontier_seed_episode = int(seed_ep)
                guarded_offsets = list(int(x) for x in guarded_decision.safe_offsets)
                neighbor_radius = int(guarded_decision.radius)
                neighbor_mutations = min(
                    int(guarded_decision.mutation_count),
                    int(getattr(train_cfg, "guarded_radius2_max_mutations", 4)),
                )
                neighbor_selected_offsets = _sample_episode_neighbor_offsets(
                    schedule=env.schedule,
                    baseline_action_vec=baseline_action_vec,
                    mutable_full_offsets=guarded_offsets,
                    mutation_count=int(neighbor_mutations),
                    rng=episode_rng,
                    empirical_controller=guarded_radius2,
                )
                if not neighbor_selected_offsets:
                    guarded_decision.active = False
                    guarded_decision.mode = "radius1"
                    guarded_decision.reason = "no_radius2_offsets_sampled"
            if not guarded_decision.active:
                neighbor_mutations = int(base_neighbor_mutations)
                neighbor_radius = int(base_neighbor_radius)
                neighbor_selected_offsets = _sample_episode_neighbor_offsets(
                    schedule=env.schedule,
                    baseline_action_vec=baseline_action_vec,
                    mutable_full_offsets=mutable_full_offsets,
                    mutation_count=int(neighbor_mutations),
                    rng=episode_rng,
                    empirical_controller=guarded_radius2,
                )
            proposal_direction = (
                "empirical_radius2" if guarded_decision.active else
                "empirical_bidirectional_radius1"
            )

        # --- fusion-mode block-granularity safe-neighbor curriculum ---
        # Per-slot safe-neighbor is off in fusion mode. Instead, each episode lets
        # only a growing subset of the H blocks leave the baseline (option 0,
        # baseline K); the rest are pinned. The mutable subset is random per
        # episode (so every block gets explored), and both the subset size and the
        # per-block radius widen to fully-open by the end of the ramp. This gives
        # PPO a discriminating gradient before the 47-block joint space opens,
        # without ever permanently masking a config.
        episode_fusion_mode = getattr(env, "_fusion_map", None) is not None
        # Scheduled forced-fusion probe (ADR-011, redesigned ADR-012): a pure
        # function of the absolute episode index, so it is deterministic and
        # identical in the episode-parallel workers. Probe episodes force ONLY
        # the target block type's fusion option to 1; K and every other block
        # follow the CURRENT policy under the normal curriculum mask (the
        # target option level is injected into the mask), so the probe is a
        # clean "what if you ALSO fused block-type T" counterfactual on top of
        # the policy — the old baseline-K forcing cancelled the fusion gain
        # against learned deep-K savings and made probes useless or negative.
        fusion_probe_block: Optional[int] = None
        if episode_fusion_mode and not force_this_ep:
            fusion_probe_block = fusion_probe_target_block(
                int(absolute_ep),
                anchor_episodes=int(force_baseline_episodes),
                interval=int(getattr(train_cfg, "fusion_probe_interval", 0)),
            )
        fusion_curriculum_active = False
        fusion_curriculum_open = False
        fusion_mutable_steps: Set[int] = set()
        fusion_neighbor_radius = 1
        if (
                episode_fusion_mode
                and bool(getattr(train_cfg, "fusion_neighbor_curriculum_enabled", False))
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
            fusion_mode = hasattr(spec, "fusion_num_options")
            step_static = step_static_tensors[int(spec.step_idx)]
            slot_mask_np = step_static.slot_mask_np
            levels_np = step_static.levels_np
            obs_t = torch.from_numpy(obs).float().to(device).unsqueeze(0)
            slot_mask_t = step_static.slot_mask_t
            levels_t = step_static.levels_t
            n_active = int(slot_mask_np.sum())
            action_level_mask_np: Optional[np.ndarray] = None
            action_level_mask_t = None
            if fusion_mode:
                # Fusion mode: the offline map holds only valid SF configs, so the
                # legal support is the open per-slot mask. Invalid / OSR pruning are
                # disabled. When the block-granularity safe-neighbor curriculum is
                # active (and not yet fully open), most blocks are pinned to baseline
                # and only the episode's selected blocks may move within a widening
                # neighborhood; the curriculum dissolves to the open mask after ramp.
                mask_mode = (
                    "curriculum"
                    if fusion_curriculum_active and not fusion_curriculum_open
                    else "open"
                )
                force_option_one = (
                    fusion_probe_block is not None
                    and int(spec.block_idx) == int(fusion_probe_block)
                    and int(spec.fusion_num_options) > 1
                )
                cached_action_mask = _get_cached_fusion_action_level_mask(
                    env,
                    spec=spec,
                    mode=mask_mode,
                    mutable=(int(spec.step_idx) in fusion_mutable_steps),
                    radius=int(fusion_neighbor_radius),
                    force_option_one=bool(force_option_one),
                    max_step_dim=policy.cfg.max_step_dim,
                    max_num_levels=policy.cfg.max_num_levels,
                    device=device,
                )
                action_level_mask_np = cached_action_mask.mask_np
                action_level_mask_t = cached_action_mask.mask_t
            elif neighbor_mask_active and base_action_vec_for_mask is not None:
                action_level_mask_np = _build_step_level_mask(
                    spec=spec,
                    baseline_action_vec=base_action_vec_for_mask,
                    selected_full_offsets=neighbor_selected_offsets,
                    max_step_dim=policy.cfg.max_step_dim,
                    max_num_levels=policy.cfg.max_num_levels,
                    radius=int(neighbor_radius),
                )
            elif static_invalid_mask is not None or empirical_invalid_mask is not None:
                action_level_mask_np = _open_step_level_mask(
                    spec=spec,
                    max_step_dim=policy.cfg.max_step_dim,
                    max_num_levels=policy.cfg.max_num_levels,
                )
            if action_level_mask_np is not None and not fusion_mode:
                protected_actions = _protected_step_actions(
                    spec=spec,
                    n_active=n_active,
                    baseline_action_vec=baseline_action_vec,
                    base_action_vec_for_mask=base_action_vec_for_mask,
                )
                if static_invalid_mask is not None:
                    before_allowed = int(np.asarray(action_level_mask_np, dtype=bool).sum())
                    action_level_mask_np = static_invalid_mask.apply(
                        spec.layer_idx,
                        spec.block_idx,
                        action_level_mask_np,
                        protected_actions=protected_actions,
                    )
                    after_allowed = int(np.asarray(action_level_mask_np, dtype=bool).sum())
                    static_invalid_level_applied_val += max(0, before_allowed - after_allowed)
                if empirical_invalid_mask is not None:
                    before_allowed = int(np.asarray(action_level_mask_np, dtype=bool).sum())
                    action_level_mask_np = empirical_invalid_mask.apply(
                        spec.layer_idx,
                        spec.block_idx,
                        action_level_mask_np,
                        protected_actions=protected_actions,
                    )
                    after_allowed = int(np.asarray(action_level_mask_np, dtype=bool).sum())
                    empirical_invalid_level_applied_val += max(0, before_allowed - after_allowed)
                if osr_mask is not None:
                    action_level_mask_np = osr_mask.apply_per_slot(
                        spec.layer_idx,
                        spec.block_idx,
                        action_level_mask_np,
                        protected_actions=protected_actions,
                    )
                action_level_mask_t = (
                    torch.from_numpy(action_level_mask_np).to(device).unsqueeze(0)
                )

            # -- Forced-baseline anchor short-circuit --
            # Skip sampling + rejection-loop entirely; commit the baseline
            # action slice. Value/log_prob come from `policy.evaluate_action`
            # against the CURRENT policy so PPO gradients are well-defined.
            if force_this_ep and baseline_action_vec is not None:
                if fusion_mode:
                    # fusion baseline action = (option 0 == all-max baseline, baseline K)
                    forced_action = np.asarray(
                        [0, int(_baseline_k_index_for_block(spec.block_idx))], dtype=np.int64
                    )
                else:
                    baseline_slice = baseline_action_vec[list(spec.full_vec_offsets)][:n_active]
                    forced_action = np.asarray(baseline_slice, dtype=np.int64)
                forced_padded = np.zeros(policy.cfg.max_step_dim, dtype=np.int64)
                forced_padded[:n_active] = forced_action
                policy_t0 = time.perf_counter()
                with torch.inference_mode():
                    actions_t = torch.from_numpy(forced_padded).to(device).unsqueeze(0)
                    lp_t, _, val_t = policy.evaluate_action(
                        obs_t, actions_t, slot_mask_t, levels_t,
                        action_level_mask=action_level_mask_t,
                        baseline_prior_scale=baseline_prior_scale,
                        truncate_to_current=True,
                        truncate_seq_len=int(spec.step_idx) + 1,
                    )
                policy_rollout_wall_seconds_val += float(time.perf_counter() - policy_t0)
                chosen_eval_info = env.evaluate_step(forced_action.tolist())
                if empirical_invalid_mask is not None:
                    if bool(chosen_eval_info.get("valid", False)):
                        empirical_invalid_mask.record_valid(
                            spec.layer_idx, spec.block_idx, forced_action.tolist()
                        )
                    else:
                        empirical_invalid_mask.record_invalid(
                            spec.layer_idx, spec.block_idx, forced_action.tolist()
                        )
                chosen_action_np = forced_padded
                chosen_log_prob = lp_t.detach().reshape(())
                chosen_value = val_t.detach().reshape(())
                rejection_counters["steps_forced_to_baseline_anchor"] += 1

                action_np = chosen_action_np
                log_prob = chosen_log_prob
                value = chosen_value
                step_action_for_env = action_np[:n_active].tolist()

                next_obs, reward, done, info = env.commit_step(
                    chosen_eval_info,
                    defer_terminal_forward=bool(fast_reward_mode_enabled),
                )
                steps_taken += 1
                valid = bool(info.get("valid", True))
                if valid:
                    valid_step_count += 1
                    rejection_counters["steps_committed_valid"] += 1
                total_bits_sum += int(info.get("total_bits", 0))
                fusion_count_sum += int(info.get("fusion_count", 0))
                per_step_optimizer_wall_seconds_val += float(
                    info.get("optimizer_wall_seconds", 0.0) or 0.0
                )
                enriched_info = dict(info)
                enriched_info["action"] = step_action_for_env
                enriched_info["reward"] = float(reward)
                enriched_info["value"] = value
                enriched_info["log_prob"] = log_prob
                enriched_info["forced_baseline"] = True
                enriched_info["exploration_mode"] = "forced_baseline"
                enriched_info["baseline_prior_scale"] = float(baseline_prior_scale)
                enriched_info["base_action_source"] = "baseline"
                enriched_info["proposal_direction"] = "anchor"
                enriched_info["guarded_radius2_active"] = False
                if on_step_end is not None:
                    try:
                        on_step_end(int(ep), int(steps_taken - 1), enriched_info)
                    except Exception:
                        pass
                if capture_step_infos:
                    captured_step_infos.append(enriched_info)
                buffer_idx = buffer.add(
                    state=obs,
                    action=action_np,
                    slot_mask=slot_mask_np,
                    per_slot_num_levels=levels_np,
                    action_level_mask=action_level_mask_np,
                    log_prob=log_prob,
                    value=value,
                    reward=float(reward),
                    done=bool(done),
                    baseline_prior_scale=float(baseline_prior_scale),
                )
                if done:
                    terminal_buffer_index = int(buffer_idx)
                    terminal_deferred = bool(info.get("terminal_deferred", False))
                    if terminal_deferred:
                        terminal_deferred_action_vec = np.asarray(
                            info.get("terminal_action_vec"), dtype=np.int64,
                        ).copy()
                per_step_sum += float(reward)
                if "terminal_reward" in info:
                    terminal_reward = float(info["terminal_reward"])
                term_info_dict = info.get("terminal_info") or {}
                terminal_final_config_fingerprint_val = str(
                    term_info_dict.get("final_config_fingerprint", "") or ""
                )
                terminal_materialization_failure_reason_val = str(
                    term_info_dict.get("materialization_failure_reason", "") or ""
                )
                _terminal_replan = term_info_dict.get("replan_application") or {}
                terminal_model_uses_replan_config_val = bool(
                    isinstance(_terminal_replan, Mapping)
                    and _terminal_replan.get("model_uses_replan_config", False)
                )
                term_breakdown = term_info_dict.get("reward_breakdown")
                term_metrics = term_info_dict.get("metrics")
                term_probe_diag = term_info_dict.get("probe_diagnostics") or {}
                if isinstance(term_info_dict.get("fusion_action_steps"), list):
                    fusion_action_steps_val = [
                        dict(x) for x in term_info_dict.get("fusion_action_steps", [])
                        if isinstance(x, Mapping)
                    ]
                if term_breakdown is not None:
                    terminal_priority_int = int(getattr(term_breakdown, "priority", 0) or 0)
                    terminal_stab_excess_m1_val = float(getattr(term_breakdown, "stab_excess_m1", 0.0) or 0.0)
                    terminal_stab_excess_m2_val = float(getattr(term_breakdown, "stab_excess_m2", 0.0) or 0.0)
                    terminal_stab_excess_loss_val = float(getattr(term_breakdown, "stab_excess_loss", 0.0) or 0.0)
                    terminal_stab_violation_val = float(getattr(term_breakdown, "stab_violation", 0.0) or 0.0)
                    terminal_bits_gain_val = float(getattr(term_breakdown, "bits_drop", 0.0) or 0.0)
                    terminal_k_gain_val = float(getattr(term_breakdown, "k_drop", 0.0) or 0.0)
                    terminal_fusion_gain_val = float(getattr(term_breakdown, "fusion_gain", 0.0) or 0.0)
                    terminal_cost_score_val = float(getattr(term_breakdown, "cost_score", 0.0) or 0.0)
                    terminal_p3_metric_margin_reward_val = float(
                        getattr(term_breakdown, "p3_metric_margin_reward", 0.0) or 0.0
                    )
                    terminal_cost_fusion_bonus_val = float(
                        getattr(term_breakdown, "cost_fusion_bonus", 0.0) or 0.0
                    )
                    terminal_cost_truncation_bonus_val = float(
                        getattr(term_breakdown, "cost_truncation_bonus", 0.0) or 0.0
                    )
                    terminal_cost_bits_tiebreaker_val = float(
                        getattr(term_breakdown, "cost_bits_tiebreaker", 0.0) or 0.0
                    )
                    terminal_cost_truncation_step_gain_val = float(
                        getattr(term_breakdown, "cost_truncation_step_gain", 0.0) or 0.0
                    )
                    terminal_cost_rank_score_val = float(
                        getattr(term_breakdown, "cost_rank_score", 0.0) or 0.0
                    )
                    terminal_cost_rank_fusion_val = float(
                        getattr(term_breakdown, "cost_rank_fusion", 0.0) or 0.0
                    )
                    terminal_cost_rank_truncation_val = float(
                        getattr(term_breakdown, "cost_rank_truncation", 0.0) or 0.0
                    )
                    terminal_cost_rank_bits_val = float(
                        getattr(term_breakdown, "cost_rank_bits", 0.0) or 0.0
                    )
                    terminal_pareto_event_kind_val = str(getattr(term_breakdown, "pareto_event_kind", "") or "")
                    terminal_pareto_action_hash_val = str(getattr(term_breakdown, "pareto_action_hash", "") or "")
                    terminal_pareto_frontier_removed_val = int(getattr(term_breakdown, "pareto_frontier_removed", 0) or 0)
                if term_metrics is not None:
                    terminal_loss_mean_val = float(getattr(term_metrics, "loss_mean", 0.0) or 0.0)
                    terminal_loss_std_val = float(getattr(term_metrics, "loss_std", 0.0) or 0.0)
                    terminal_metric1_val = float(getattr(term_metrics, "metric1_mean", 0.0) or 0.0)
                    terminal_metric2_val = float(getattr(term_metrics, "metric2_mean", 0.0) or 0.0)
                    terminal_metric1_std_val = float(getattr(term_metrics, "metric1_std", 0.0) or 0.0)
                    terminal_metric2_std_val = float(getattr(term_metrics, "metric2_std", 0.0) or 0.0)
                if isinstance(term_probe_diag, dict):
                    terminal_probe_wall_seconds_val = float(term_probe_diag.get("wall_seconds", 0.0) or 0.0)
                    terminal_probe_devices_val = [str(x) for x in (term_probe_diag.get("devices") or [])]
                    terminal_probe_trial_counts_val = [
                        int(x) for x in (term_probe_diag.get("per_worker_trial_counts") or [])
                    ]
                    terminal_probe_trial_indices_val = [
                        [int(y) for y in (x or [])]
                        for x in (term_probe_diag.get("per_worker_trial_indices") or [])
                    ]
                    terminal_probe_speedup_val = float(
                        term_probe_diag.get("speedup_vs_sequential", 1.0) or 1.0
                    )
                    terminal_cost_eval_wall_seconds_val = float(
                        term_probe_diag.get("cost_eval_wall_seconds", 0.0) or 0.0
                    )
                    terminal_probe_install_wall_seconds_val = float(
                        term_probe_diag.get("probe_install_wall_seconds", 0.0) or 0.0
                    )
                    terminal_probe_clear_wall_seconds_val = float(
                        term_probe_diag.get("probe_clear_wall_seconds", 0.0) or 0.0
                    )
                    terminal_probe_install_skipped_val = bool(
                        term_probe_diag.get("probe_install_skipped", False)
                    )
                    terminal_probe_clear_skipped_val = bool(
                        term_probe_diag.get("probe_clear_skipped", False)
                    )
                obs = next_obs
                if done:
                    break
                continue   # advance to next step (forced loop)

            # -- Rejection-sample around the per-(layer, block) blacklist --
            # The policy may sample a tuple that the optimizer previously
            # rejected for this (layer, block). We:
            #   1) Skip it BEFORE calling evaluate_step if it's already in the
            #      mask (no optimizer call → cheap).
            #   2) If it's a fresh tuple but evaluate_step returns invalid,
            #      add it to the mask and re-sample.
            #   3) If max_rejection_retries are exhausted, fall back to the
            #      baseline action sliced from baseline_action_vec for this
            #      step (guaranteed valid by static_skeletons). If no baseline
            #      is available, commit the last sampled action even though
            #      it failed — caller will see invalid=True in the info dict.
            chosen_action_np: Optional[np.ndarray] = None
            chosen_log_prob: Any = 0.0
            chosen_value: Any = 0.0
            chosen_eval_info: Optional[Dict[str, Any]] = None
            attempts_this_step = 0

            for _attempt in range(int(max_rejection_retries)):
                attempts_this_step += 1
                policy_t0 = time.perf_counter()
                with torch.inference_mode():
                    action_t, log_prob_t, value_t = policy.sample_action(
                        obs_t, slot_mask_t, levels_t,
                        deterministic=False,
                        action_level_mask=action_level_mask_t,
                        baseline_prior_scale=baseline_prior_scale,
                        truncate_to_current=True,
                        truncate_seq_len=int(spec.step_idx) + 1,
                    )
                policy_rollout_wall_seconds_val += float(time.perf_counter() - policy_t0)
                action_np_try = action_t.squeeze(0).cpu().numpy().astype(np.int64)
                if (
                        fusion_mode
                        and fusion_probe_block is not None
                        and int(spec.block_idx) == int(fusion_probe_block)
                        and int(spec.fusion_num_options) > 1
                        and int(action_np_try[0]) != 1
                ):
                    # ADR-012 probe: force ONLY the target block's option to 1;
                    # K (and everything else) keeps the policy's own sample.
                    # Re-evaluate log_prob/value for the modified action under
                    # the same mask so PPO ratios stay well-defined.
                    action_np_try = action_np_try.copy()
                    action_np_try[0] = 1
                    policy_t1 = time.perf_counter()
                    with torch.inference_mode():
                        actions_fix_t = (
                            torch.from_numpy(action_np_try).to(device).unsqueeze(0)
                        )
                        log_prob_t, _probe_ent, value_t = policy.evaluate_action(
                            obs_t, actions_fix_t, slot_mask_t, levels_t,
                            action_level_mask=action_level_mask_t,
                            baseline_prior_scale=baseline_prior_scale,
                            truncate_to_current=True,
                            truncate_seq_len=int(spec.step_idx) + 1,
                        )
                    policy_rollout_wall_seconds_val += float(
                        time.perf_counter() - policy_t1
                    )
                step_action_try = action_np_try[:n_active].tolist()
                tup = tuple(int(x) for x in step_action_try)

                if forbidden_mask.is_forbidden(spec.layer_idx, spec.block_idx, tup):
                    rejection_counters["samples_rejected_by_mask"] += 1
                    continue   # cheap re-sample, no optimizer call
                # OSR per-combo blacklist (pre-prune): same short-circuit as
                # ForbiddenActionMask, but populated by the offline OSR scan
                # instead of runtime experience.
                if osr_mask is not None and osr_mask.is_combo_pruned(
                        spec.layer_idx, spec.block_idx, tup):
                    rejection_counters["samples_rejected_by_mask"] += 1
                    continue

                eval_info = env.evaluate_step(step_action_try)
                if eval_info["valid"]:
                    if empirical_invalid_mask is not None:
                        empirical_invalid_mask.record_valid(
                            spec.layer_idx, spec.block_idx, tup
                        )
                    chosen_action_np = action_np_try
                    chosen_log_prob = log_prob_t.detach().reshape(())
                    chosen_value = value_t.detach().reshape(())
                    chosen_eval_info = eval_info
                    break

                # New invalid → blacklist + try again. Per the user's spec
                # ("就好像训练过程中根本不存在这些动作"), the rejected sample
                # is NOT counted toward invalid_steps / invalid_block_details —
                # only ``rejection_counters`` records the diagnostic count.
                forbidden_mask.add(spec.layer_idx, spec.block_idx, tup)
                if empirical_invalid_mask is not None:
                    empirical_invalid_mask.record_invalid(spec.layer_idx, spec.block_idx, tup)
                rejection_counters["samples_rejected_by_optimizer"] += 1
                rejection_optimizer_wall_seconds_val += float(
                    eval_info.get("optimizer_wall_seconds", 0.0) or 0.0
                )

            if chosen_eval_info is None:
                # Exhausted retries — fall back to the baseline action for this
                # specific step. Baseline is the static_skeletons all-max action,
                # which is verified valid across all 59 (layer, block) configs,
                # so committing it is guaranteed to keep this episode usable.
                if baseline_action_vec is not None:
                    baseline_slice = baseline_action_vec[
                        list(spec.full_vec_offsets)
                    ][:n_active]
                    fallback_action = np.asarray(baseline_slice, dtype=np.int64)
                    fallback_padded = np.zeros(policy.cfg.max_step_dim, dtype=np.int64)
                    fallback_padded[:n_active] = fallback_action
                    policy_t0 = time.perf_counter()
                    with torch.inference_mode():
                        actions_t = torch.from_numpy(fallback_padded).to(device).unsqueeze(0)
                        lp_t, _, val_t = policy.evaluate_action(
                            obs_t, actions_t, slot_mask_t, levels_t,
                            action_level_mask=action_level_mask_t,
                            baseline_prior_scale=baseline_prior_scale,
                            truncate_to_current=True,
                            truncate_seq_len=int(spec.step_idx) + 1,
                        )
                    policy_rollout_wall_seconds_val += float(time.perf_counter() - policy_t0)
                    chosen_eval_info = env.evaluate_step(fallback_action.tolist())
                    if empirical_invalid_mask is not None:
                        if bool(chosen_eval_info.get("valid", False)):
                            empirical_invalid_mask.record_valid(
                                spec.layer_idx, spec.block_idx, fallback_action.tolist()
                            )
                        else:
                            empirical_invalid_mask.record_invalid(
                                spec.layer_idx, spec.block_idx, fallback_action.tolist()
                            )
                    chosen_action_np = fallback_padded
                    chosen_log_prob = lp_t.detach().reshape(())
                    chosen_value = val_t.detach().reshape(())
                    rejection_counters["steps_fallen_back_to_baseline"] += 1
                else:
                    # Last-resort: commit the most-recently sampled action even
                    # though it failed. Should not happen in production because
                    # the runner always provides baseline_action_vec.
                    chosen_action_np = action_np_try
                    chosen_log_prob = log_prob_t.detach().reshape(())
                    chosen_value = value_t.detach().reshape(())
                    chosen_eval_info = eval_info

            assert chosen_action_np is not None and chosen_eval_info is not None

            action_np = chosen_action_np
            log_prob = chosen_log_prob
            value = chosen_value
            step_action_for_env = action_np[:n_active].tolist()

            next_obs, reward, done, info = env.commit_step(
                chosen_eval_info,
                defer_terminal_forward=bool(fast_reward_mode_enabled),
            )
            steps_taken += 1
            valid = bool(info.get("valid", True))
            if valid:
                valid_step_count += 1
                rejection_counters["steps_committed_valid"] += 1
            else:
                # Only happens if baseline fallback itself failed (defensive).
                rejection_counters["steps_committed_invalid"] += 1
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
                    "reason": _format_invalid_chain_reason(info.get("invalid_chain")),
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

            # Augment info with the per-step action + reward + value for
            # downstream rich loggers (kept here so callers don't have to
            # re-track state).
            enriched_info = dict(info)
            enriched_info["action"] = step_action_for_env
            enriched_info["reward"] = float(reward)
            enriched_info["value"] = value
            enriched_info["log_prob"] = log_prob
            enriched_info["baseline_prior_scale"] = float(baseline_prior_scale)
            enriched_info["base_action_source"] = str(base_action_source)
            enriched_info["proposal_direction"] = str(proposal_direction)
            enriched_info["frontier_seed_episode"] = int(frontier_seed_episode)
            if action_level_mask_np is not None:
                enriched_info["neighbor_selected_offsets"] = sorted(
                    int(x) for x in neighbor_selected_offsets
                )
                enriched_info["neighbor_radius"] = int(neighbor_radius)
                enriched_info["exploration_mode"] = str(
                    guarded_decision.mode if guarded_decision.active else "radius1"
                )
                enriched_info["guarded_radius2_active"] = bool(guarded_decision.active)

            if on_step_end is not None:
                try:
                    on_step_end(int(ep), int(steps_taken - 1), enriched_info)
                except Exception:
                    pass
            if capture_step_infos:
                captured_step_infos.append(enriched_info)

            buffer_idx = buffer.add(
                state=obs,
                action=action_np,
                slot_mask=slot_mask_np,
                per_slot_num_levels=levels_np,
                action_level_mask=action_level_mask_np,
                log_prob=log_prob,
                value=value,
                reward=float(reward),
                done=bool(done),
                baseline_prior_scale=float(baseline_prior_scale),
            )
            if done:
                terminal_buffer_index = int(buffer_idx)
                terminal_deferred = bool(info.get("terminal_deferred", False))
                if terminal_deferred:
                    terminal_deferred_action_vec = np.asarray(
                        info.get("terminal_action_vec"), dtype=np.int64,
                    ).copy()
            per_step_sum += float(reward)
            if "terminal_reward" in info:
                terminal_reward = float(info["terminal_reward"])
            # Extract terminal breakdown for the final EpisodeRecord. Lives in
            # ``info["terminal_info"]`` (sequential_env.py:commit_step writes it
            # there on the terminal step). Falls back to defaults when this is
            # not the terminal step or when the base env short-circuited (any
            # invalid → no compute_reward call).
            term_info_dict = info.get("terminal_info") or {}
            terminal_final_config_fingerprint_val = str(
                term_info_dict.get("final_config_fingerprint", "") or ""
            )
            terminal_materialization_failure_reason_val = str(
                term_info_dict.get("materialization_failure_reason", "") or ""
            )
            _terminal_replan = term_info_dict.get("replan_application") or {}
            terminal_model_uses_replan_config_val = bool(
                isinstance(_terminal_replan, Mapping)
                and _terminal_replan.get("model_uses_replan_config", False)
            )
            term_breakdown = term_info_dict.get("reward_breakdown")
            term_metrics = term_info_dict.get("metrics")
            term_probe_diag = term_info_dict.get("probe_diagnostics") or {}
            if isinstance(term_info_dict.get("fusion_action_steps"), list):
                fusion_action_steps_val = [
                    dict(x) for x in term_info_dict.get("fusion_action_steps", [])
                    if isinstance(x, Mapping)
                ]
            if term_breakdown is not None:
                terminal_priority_int = int(getattr(term_breakdown, "priority", 0) or 0)
                terminal_stab_excess_m1_val = float(getattr(term_breakdown, "stab_excess_m1", 0.0) or 0.0)
                terminal_stab_excess_m2_val = float(getattr(term_breakdown, "stab_excess_m2", 0.0) or 0.0)
                terminal_stab_excess_loss_val = float(getattr(term_breakdown, "stab_excess_loss", 0.0) or 0.0)
                terminal_stab_violation_val = float(getattr(term_breakdown, "stab_violation", 0.0) or 0.0)
                terminal_bits_gain_val = float(getattr(term_breakdown, "bits_drop", 0.0) or 0.0)
                terminal_k_gain_val = float(getattr(term_breakdown, "k_drop", 0.0) or 0.0)
                terminal_fusion_gain_val = float(getattr(term_breakdown, "fusion_gain", 0.0) or 0.0)
                terminal_cost_score_val = float(getattr(term_breakdown, "cost_score", 0.0) or 0.0)
                terminal_p3_metric_margin_reward_val = float(
                    getattr(term_breakdown, "p3_metric_margin_reward", 0.0) or 0.0
                )
                terminal_cost_fusion_bonus_val = float(
                    getattr(term_breakdown, "cost_fusion_bonus", 0.0) or 0.0
                )
                terminal_cost_truncation_bonus_val = float(
                    getattr(term_breakdown, "cost_truncation_bonus", 0.0) or 0.0
                )
                terminal_cost_bits_tiebreaker_val = float(
                    getattr(term_breakdown, "cost_bits_tiebreaker", 0.0) or 0.0
                )
                terminal_cost_truncation_step_gain_val = float(
                    getattr(term_breakdown, "cost_truncation_step_gain", 0.0) or 0.0
                )
                terminal_cost_rank_score_val = float(
                    getattr(term_breakdown, "cost_rank_score", 0.0) or 0.0
                )
                terminal_cost_rank_fusion_val = float(
                    getattr(term_breakdown, "cost_rank_fusion", 0.0) or 0.0
                )
                terminal_cost_rank_truncation_val = float(
                    getattr(term_breakdown, "cost_rank_truncation", 0.0) or 0.0
                )
                terminal_cost_rank_bits_val = float(
                    getattr(term_breakdown, "cost_rank_bits", 0.0) or 0.0
                )
                terminal_pareto_event_kind_val = str(getattr(term_breakdown, "pareto_event_kind", "") or "")
                terminal_pareto_action_hash_val = str(getattr(term_breakdown, "pareto_action_hash", "") or "")
                terminal_pareto_frontier_removed_val = int(getattr(term_breakdown, "pareto_frontier_removed", 0) or 0)
            if term_metrics is not None:
                terminal_loss_mean_val = float(getattr(term_metrics, "loss_mean", 0.0) or 0.0)
                terminal_loss_std_val = float(getattr(term_metrics, "loss_std", 0.0) or 0.0)
                terminal_metric1_val = float(getattr(term_metrics, "metric1_mean", 0.0) or 0.0)
                terminal_metric2_val = float(getattr(term_metrics, "metric2_mean", 0.0) or 0.0)
                terminal_metric1_std_val = float(getattr(term_metrics, "metric1_std", 0.0) or 0.0)
                terminal_metric2_std_val = float(getattr(term_metrics, "metric2_std", 0.0) or 0.0)
            if isinstance(term_probe_diag, dict):
                terminal_probe_wall_seconds_val = float(term_probe_diag.get("wall_seconds", 0.0) or 0.0)
                terminal_probe_devices_val = [str(x) for x in (term_probe_diag.get("devices") or [])]
                terminal_probe_trial_counts_val = [
                    int(x) for x in (term_probe_diag.get("per_worker_trial_counts") or [])
                ]
                terminal_probe_trial_indices_val = [
                    [int(y) for y in (x or [])]
                    for x in (term_probe_diag.get("per_worker_trial_indices") or [])
                ]
                terminal_probe_speedup_val = float(
                    term_probe_diag.get("speedup_vs_sequential", 1.0) or 1.0
                )
                terminal_cost_eval_wall_seconds_val = float(
                    term_probe_diag.get("cost_eval_wall_seconds", 0.0) or 0.0
                )
                terminal_probe_install_wall_seconds_val = float(
                    term_probe_diag.get("probe_install_wall_seconds", 0.0) or 0.0
                )
                terminal_probe_clear_wall_seconds_val = float(
                    term_probe_diag.get("probe_clear_wall_seconds", 0.0) or 0.0
                )
                terminal_probe_install_skipped_val = bool(
                    term_probe_diag.get("probe_install_skipped", False)
                )
                terminal_probe_clear_skipped_val = bool(
                    term_probe_diag.get("probe_clear_skipped", False)
                )

            obs = next_obs
            if done:
                break

        empirical_success_rate, empirical_failure_rate = guarded_radius2.offset_rates(
            sorted(int(x) for x in neighbor_selected_offsets)
        )
        episode_rejections = {
            key: int(rejection_counters.get(key, 0) - rejection_start.get(key, 0))
            for key in rejection_counters
        }
        record = EpisodeRecord(
            episode_idx=int(ep),
            total_reward=float(per_step_sum),
            terminal_reward=float(terminal_reward),
            per_step_reward_sum=float(per_step_sum - terminal_reward),
            invalid_steps=int(invalid_steps),
            early_terminated=bool(early_terminated),
            steps_taken=int(steps_taken),
            valid_step_count=int(valid_step_count),
            total_bits_sum_over_steps=int(total_bits_sum),
            fusion_count_sum_over_steps=int(fusion_count_sum),
            first_invalid_step=(int(first_invalid["step"]) if first_invalid else None),
            first_invalid_block=(int(first_invalid["block_idx"]) if first_invalid else None),
            first_invalid_layer=(int(first_invalid["layer_idx"]) if first_invalid else None),
            step_infos=captured_step_infos,
            invalid_block_details=invalid_block_details,
            terminal_final_config_fingerprint=str(
                terminal_final_config_fingerprint_val
            ),
            terminal_materialization_failure_reason=str(
                terminal_materialization_failure_reason_val
            ),
            terminal_model_uses_replan_config=bool(
                terminal_model_uses_replan_config_val
            ),
            terminal_priority=int(terminal_priority_int),
            terminal_loss_mean=float(terminal_loss_mean_val),
            terminal_loss_std=float(terminal_loss_std_val),
            terminal_metric1_mean=float(terminal_metric1_val),
            terminal_metric2_mean=float(terminal_metric2_val),
            terminal_metric1_std=float(terminal_metric1_std_val),
            terminal_metric2_std=float(terminal_metric2_std_val),
            terminal_stab_excess_m1=float(terminal_stab_excess_m1_val),
            terminal_stab_excess_m2=float(terminal_stab_excess_m2_val),
            terminal_stab_excess_loss=float(terminal_stab_excess_loss_val),
            terminal_stab_violation=float(terminal_stab_violation_val),
            terminal_bits_gain=float(terminal_bits_gain_val),
            terminal_k_gain=float(terminal_k_gain_val),
            terminal_fusion_gain=float(terminal_fusion_gain_val),
            terminal_cost_score=float(terminal_cost_score_val),
            terminal_p3_metric_margin_reward=float(terminal_p3_metric_margin_reward_val),
            terminal_cost_fusion_bonus=float(terminal_cost_fusion_bonus_val),
            terminal_cost_truncation_bonus=float(terminal_cost_truncation_bonus_val),
            terminal_cost_bits_tiebreaker=float(terminal_cost_bits_tiebreaker_val),
            terminal_cost_truncation_step_gain=float(terminal_cost_truncation_step_gain_val),
            terminal_cost_rank_score=float(terminal_cost_rank_score_val),
            terminal_cost_rank_fusion=float(terminal_cost_rank_fusion_val),
            terminal_cost_rank_truncation=float(terminal_cost_rank_truncation_val),
            terminal_cost_rank_bits=float(terminal_cost_rank_bits_val),
            terminal_pareto_event_kind=str(terminal_pareto_event_kind_val),
            terminal_pareto_action_hash=str(terminal_pareto_action_hash_val),
            terminal_pareto_frontier_removed=int(terminal_pareto_frontier_removed_val),
            terminal_probe_wall_seconds=float(terminal_probe_wall_seconds_val),
            terminal_probe_devices=list(terminal_probe_devices_val),
            terminal_probe_trial_counts=list(terminal_probe_trial_counts_val),
            terminal_probe_trial_indices=[list(x) for x in terminal_probe_trial_indices_val],
            terminal_probe_speedup=float(terminal_probe_speedup_val),
            fusion_action_steps=list(fusion_action_steps_val),
            per_step_optimizer_wall_seconds=float(per_step_optimizer_wall_seconds_val),
            policy_rollout_wall_seconds=float(policy_rollout_wall_seconds_val),
            terminal_cost_eval_wall_seconds=float(terminal_cost_eval_wall_seconds_val),
            terminal_probe_install_wall_seconds=float(terminal_probe_install_wall_seconds_val),
            terminal_probe_clear_wall_seconds=float(terminal_probe_clear_wall_seconds_val),
            terminal_probe_install_skipped=bool(terminal_probe_install_skipped_val),
            terminal_probe_clear_skipped=bool(terminal_probe_clear_skipped_val),
            safe_neighbor_active=bool(
                (fusion_curriculum_active and not fusion_curriculum_open)
                if episode_fusion_mode else neighbor_mask_active
            ),
            safe_neighbor_mutation_count=int(
                len(fusion_mutable_steps) if episode_fusion_mode
                else len(neighbor_selected_offsets)
            ),
            safe_neighbor_radius=int(
                (fusion_neighbor_radius
                 if (fusion_curriculum_active and not fusion_curriculum_open) else 0)
                if episode_fusion_mode
                else (neighbor_radius if neighbor_mask_active else 0)
            ),
            exploration_mode=(
                "forced_baseline" if force_this_ep else
                (f"forced_fusion_probe_b{int(fusion_probe_block)}"
                 if fusion_probe_block is not None else
                 str(guarded_decision.mode if guarded_decision.active else "radius1"))
            ),
            guarded_radius2_active=bool(guarded_decision.active),
            guarded_radius2_recent_frontier_expansions=int(
                guarded_decision.recent_frontier_expansions
            ),
            guarded_radius2_recent_duplicate_rate=float(
                guarded_decision.recent_duplicate_rate
            ),
            guarded_radius2_recent_dominated_rate=float(
                guarded_decision.recent_dominated_rate
            ),
            guarded_radius2_cooldown_remaining=int(guarded_decision.cooldown_remaining),
            guarded_radius2_safe_offset_count=int(guarded_decision.safe_offset_count),
            guarded_radius2_episode_count=int(guarded_decision.radius2_episode_count),
            guarded_radius2_failure_count=int(guarded_decision.radius2_failure_count),
            guarded_radius2_frontier_expansion_count=int(
                guarded_decision.radius2_frontier_expansion_count
            ),
            samples_rejected_by_mask=int(episode_rejections.get("samples_rejected_by_mask", 0)),
            samples_rejected_by_optimizer=int(
                episode_rejections.get("samples_rejected_by_optimizer", 0)
            ),
            steps_fallen_back_to_baseline=int(
                episode_rejections.get("steps_fallen_back_to_baseline", 0)
            ),
            forbidden_mask_total=int(forbidden_mask.total()),
            static_invalid_level_disabled=(
                int(static_invalid_mask.total_disabled())
                if static_invalid_mask is not None else 0
            ),
            static_invalid_level_applied=int(static_invalid_level_applied_val),
            static_invalid_level_scan_evaluated=int(
                static_invalid_scan_summary.get("evaluated", 0) or 0
            ),
            static_invalid_level_scan_invalid=int(
                static_invalid_scan_summary.get("invalid", 0) or 0
            ),
            empirical_invalid_level_disabled=(
                int(empirical_invalid_mask.total_disabled())
                if empirical_invalid_mask is not None else 0
            ),
            empirical_invalid_level_applied=int(empirical_invalid_level_applied_val),
            rejection_optimizer_wall_seconds=float(rejection_optimizer_wall_seconds_val),
            baseline_prior_scale=float(baseline_prior_scale),
            base_action_source=str("baseline" if force_this_ep else base_action_source),
            proposal_direction=str("anchor" if force_this_ep else proposal_direction),
            empirical_offset_success_rate=float(empirical_success_rate),
            empirical_offset_failure_rate=float(empirical_failure_rate),
            frontier_seed_episode=int(frontier_seed_episode),
        )
        pending_full_vec = getattr(env, "_pending_full_vec", None)
        draft = {
            "record": record,
            "absolute_ep": int(absolute_ep),
            "selected_offsets": sorted(int(x) for x in neighbor_selected_offsets),
            "neighbor_radius": int(neighbor_radius if neighbor_mask_active else 0),
            "neighbor_mask_active": bool(neighbor_mask_active),
            "guarded_decision": guarded_decision,
            "pending_full_vec": (
                None if pending_full_vec is None
                else np.asarray(pending_full_vec, dtype=np.int64).copy()
            ),
            "terminal_buffer_index": int(terminal_buffer_index),
        }
        if bool(terminal_deferred):
            if terminal_deferred_action_vec is None or int(terminal_buffer_index) < 0:
                raise RuntimeError("deferred terminal reward missing action vec or buffer index")
            draft["terminal_action_vec"] = np.asarray(
                terminal_deferred_action_vec, dtype=np.int64,
            ).copy()
            from .candidate_store import action_hash as _action_hash
            draft["terminal_action_hash"] = _action_hash(draft["terminal_action_vec"])
            pending_terminal_drafts.append(draft)
            if (
                    len(pending_terminal_drafts) >= int(terminal_eval_batch_size)
                    or (ep + 1) % int(train_cfg.update_every_n_episodes) == 0
                    or (ep + 1) >= int(train_cfg.total_episodes)
            ):
                flush_pending_terminal_drafts("batch_full_or_update_boundary")
        else:
            _finalize_completed_record(draft)

    flush_pending_terminal_drafts("train_end")

    return {
        "episode_returns": episode_returns,
        "episode_records": episode_records,
        "ppo_metrics": ppo_metric_history,
        "final_invalid_rate": (
            float(np.mean([r.invalid_steps > 0 for r in episode_records[-10:]]))
            if episode_records else 0.0
        ),
        "forbidden_mask": forbidden_mask,
        "static_invalid_mask": static_invalid_mask,
        "static_invalid_scan_summary": dict(static_invalid_scan_summary),
        "empirical_invalid_mask": empirical_invalid_mask,
        "rejection_counters": dict(rejection_counters),
    }


# ---------------------------------------------------------------------------
# Launcher integration: drive the sequential loop from BLBStage2RLRunner
# ---------------------------------------------------------------------------
#
# Note on size: ``run_sequential_via_runner`` is ~1000 lines as of 2026-05-16
# (down from ~1018 after this round of extraction). Sections that are good
# candidates for further extraction into private helpers — but each requires
# careful local-variable plumbing because the function shares ~30 locals
# across phases:
#
#   * §0.1 + §1     "setup_evaluator_and_phase"   (persistent dir + degrees)
#   * §2  + §3      "build_env_and_bridge"        (probe / bridge / base env)
#   * §4            "calibrate_baseline_metrics"  (cost stats + reward weights)
#   * §5            "build_sequential_policy"     (env + policy + optimizer + warmstart)
#   * §6            "resume_from_checkpoint"      (already concise; keep inline)
#   * §7            "run_training_loop"           (the PPO loop itself)
#   * §7.5 + §7.6   "persist_action_artifacts"    (action_full + report.md)
#   * §7.7          extracted as _register_run_in_experiments_log (below)
#   * §10           "assemble_legacy_result_dict" (the giant return dict)
#
# When refactoring, write a smoke test FIRST (see tests/test_sequential_smoke.py
# for the artifact-contract pattern) — there's no end-to-end torch-backed test
# yet, so refactors are best done one section at a time with the smoke test
# as the regression net for artifact behavior.

def _register_run_in_experiments_log(
        *,
        run_basename: str,
        profile: str,
        model_type: str,
        preset_label: str,
        seed: int,
        elapsed_sec: float,
        completed_episodes: int,
        total_episodes_planned: int,
        best_reward: float,
        best_action_present: bool,
        episode_count: int,
        blb_progress_dir: str,
        diag_recorder,
        save_path: str,
        best_action_description_paths: Mapping[str, str],
        baseline_action_description_paths: Mapping[str, str],
        log: Callable[[str], None],
        bullet: str,
) -> None:
    """Append a row to ``experiments/registry.jsonl`` for this completed run.

    Best-effort: errors are swallowed (registering must never fail a trained
    model). Set ``BLB_STRICT=1`` to see the actual exception during debug.
    See :mod:`blb_stage2_rl.strict` for the strict-mode pattern.
    """
    try:
        import subprocess as _sp
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        registry_path = os.path.join(repo_root, "experiments", "registry.jsonl")
        os.makedirs(os.path.dirname(registry_path), exist_ok=True)
        artifact_paths = {
            "best_action_full_md": best_action_description_paths.get("md", ""),
            "best_action_full_json": best_action_description_paths.get("json", ""),
            "baseline_action_full_md": baseline_action_description_paths.get("md", ""),
            "report_md": os.path.join(blb_progress_dir, "blb_stage2_report.md"),
            "diagnostics_summary": diag_recorder.summary_md_path,
            "diagnostics_dir": diag_recorder.output_dir,
            "best_action_vec_json": diag_recorder.best_json_path,
            "status_json": os.path.join(blb_progress_dir, "blb_stage2_status.json"),
            "checkpoint_pt": save_path,
        }
        _sp.run(
            [
                "python3", "-m", "tools.experiments_log",
                "register",
                "--run-id", str(run_basename),
                "--dataset", str(profile),
                "--model-type", str(model_type),
                "--algorithm", "rl",
                "--preset", str(preset_label or ""),
                "--rl-variant", SEQ_RL_VARIANT,
                "--seed", str(int(seed)),
                "--status", ("complete" if best_action_present else "training_only"),
                "--elapsed-sec", str(float(elapsed_sec)),
                "--completed-episodes", str(int(completed_episodes)),
                "--total-episodes-planned", str(int(total_episodes_planned)),
                "--best-reward", str(float(best_reward)),
                "--persistent-dir", str(blb_progress_dir),
                "--artifact-paths-json", json.dumps(artifact_paths),
                "--notes", f"auto-registered at training end ({episode_count} ep completed)",
                "--registry-path", registry_path,
            ],
            check=False,
            capture_output=True,
            cwd=repo_root,   # so `-m tools.experiments_log` resolves the package
        )
        log(f"  {bullet} 已登记到 experiments/registry.jsonl（run_id={run_basename}）")
    except Exception as exc:
        log(f"  [experiments][warning] register failed: {exc}")


def _resolve_robust_baseline_config(train_cfg: Any, evaluator: Any) -> Tuple[float, float, int]:
    """Read robust constraint calibration inputs without legacy tolerance math."""
    raw_precision_tolerance = getattr(evaluator, "stage2_limit_tolerance", None)
    precision_tolerance = (
        0.001 if raw_precision_tolerance is None else float(raw_precision_tolerance)
    )
    raw_stability_multiplier = getattr(
        train_cfg,
        "stage2_stability_multiplier",
        getattr(evaluator, "stage2_stability_multiplier", None),
    )
    stability_multiplier = (
        2.0 if raw_stability_multiplier is None else float(raw_stability_multiplier)
    )
    raw_bootstrap_samples = getattr(train_cfg, "constraint_bootstrap_samples", None)
    bootstrap_samples = 4096 if raw_bootstrap_samples is None else int(raw_bootstrap_samples)
    return precision_tolerance, stability_multiplier, bootstrap_samples


def _run_legacy_preflight_if_needed(
        *,
        robust_mode: bool,
        run_legacy_preflight: Callable[[], None],
        ) -> None:
    """Run the legacy one-shot preflight only outside robust mode."""
    if not robust_mode:
        run_legacy_preflight()


def _build_authoritative_validation_env(
        *,
        runner: Any,
        ev: Any,
        base_env: Any,
        train_cfg: Any,
        reward_devices: Sequence[int],
        log: Callable[[str], None],
        ) -> Tuple[Any, int]:
    """Clone the probe shell while preserving the canonical primary bridge."""
    validation_full_batches = runner._build_validation_full_batches(ev)
    validation_full = ev.dataset_splits.get("validation_full")
    example_count = len(validation_full)

    promotion_env = copy.copy(base_env)
    promotion_env.env_cfg = copy.copy(base_env.env_cfg)
    promotion_env.env_cfg.probe_batch_count = len(validation_full_batches)
    promotion_env.env_cfg.persistent_probe_install = False
    promotion_env.probe_batches = list(validation_full_batches)
    promotion_env.probe_runner = None
    promotion_env.baseline = copy.deepcopy(base_env.baseline)
    promotion_env.reward_weights = copy.deepcopy(base_env.reward_weights)
    promotion_env.statistical_reference = None
    promotion_env.probe_noise_seed = None
    promotion_env._installed_config_fingerprint = None
    promotion_env._installed_action_hash = None
    promotion_env._last_probe_diagnostics = {}

    devices = [int(value) for value in reward_devices]
    shared_owner = getattr(base_env, "_shared_probe_runner_owner", None)
    if shared_owner is not None:
        frozen_batches = tuple(validation_full_batches)
        registered_sets = getattr(
            base_env, "_shared_probe_batch_sets", None,
        )
        if registered_sets is None:
            registered_sets = {}
            base_env._shared_probe_batch_sets = registered_sets
        registered_f4 = registered_sets.get("F4")
        if registered_f4 is None:
            shared_owner.register_batch_set("F4", frozen_batches)
            registered_sets["F4"] = frozen_batches
        elif (
                len(registered_f4) != len(frozen_batches)
                or any(
                    previous is not current
                    for previous, current in zip(
                        registered_f4, frozen_batches,
                    )
                )
        ):
            raise ValueError(
                "F4 probe batch set is already registered with different batches"
            )
        promotion_env.probe_runner = shared_owner.view("F4")
    elif len(devices) >= 2:
        raise RuntimeError(
            "authoritative validation requires the shared F1 probe-runner owner"
        )
    log(
        "  * F4 authoritative validation: "
        f"split=validation_full examples={example_count} "
        f"batches={len(validation_full_batches)} devices={devices or ['primary']}"
    )
    return promotion_env, int(example_count)


def _install_robust_baseline_reference(
        base_env: Any,
        baseline: Any,
        weights: Any,
        reference: "BaselineReference",
        ) -> None:
    """Install pooled robust constraints into the reward and environment state."""
    baseline.loss_mean = float(reference.loss_mean)
    baseline.metric1_mean = float(reference.metric1_mean)
    baseline.metric2_mean = float(reference.metric2_mean)
    baseline.loss_std = float(reference.loss_std)
    baseline.metric1_std = float(reference.metric1_std)
    baseline.metric2_std = float(reference.metric2_std)
    weights.baseline_metric1 = float(reference.metric1_mean)
    weights.baseline_metric2 = float(reference.metric2_mean)
    weights.stab_tolerance = float(reference.stability_multiplier)
    weights.stab_floor = 0.0
    base_env.statistical_reference = reference
    base_env.loss_threshold = float(reference.loss_limit)
    base_env.acc_threshold = float(reference.metric1_limit)
    base_env.acc_threshold_m2 = float(reference.metric2_limit)
    base_env.stab_threshold = float(reference.loss_std_limit)


def _collect_robust_baseline_reference(
        *,
        base_env: Any,
        baseline_action_vec: Sequence[int],
        base_seed: int,
        precision_tolerance: float,
        stability_multiplier: float,
        bootstrap_samples: int,
        baseline_groups: int = 5,
        trials_per_group: int = 5,
        max_groups: int = 10,
        group_index_start: int = 0,
        ) -> Tuple["BaselineReference", Dict[str, Any]]:
    """Collect deterministic grouped baseline trials for robust constraints."""
    from .seed_utils import derive_baseline_group_probe_seed
    from .statistical_constraints import (
        DegenerateBaselineVariance,
        TrialSeries,
        build_baseline_reference,
    )

    original_trials = getattr(base_env.env_cfg, "num_trials_per_step", 1)
    original_probe_seed = getattr(base_env, "probe_noise_seed", None)
    reward_weights = getattr(base_env, "reward_weights", None)
    original_reward_design = (
        getattr(reward_weights, "reward_design", None)
        if reward_weights is not None else None
    )
    had_statistical_reference = hasattr(base_env, "statistical_reference")
    original_statistical_reference = getattr(base_env, "statistical_reference", None)
    action = np.asarray(baseline_action_vec, dtype=np.int64).reshape(-1).copy()
    groups: List[Any] = []
    raw_groups: List[Dict[str, Any]] = []

    try:
        if (
                reward_weights is not None
                and str(original_reward_design).strip().lower() == "robust_constrained"
        ):
            reward_weights.reward_design = "stage1_aligned"
        if had_statistical_reference:
            base_env.statistical_reference = None
        required_groups = int(baseline_groups)
        group_trials = int(trials_per_group)
        group_limit = int(max_groups)
        group_start = int(group_index_start)
        if required_groups <= 0 or group_trials <= 0 or group_limit < required_groups:
            raise ValueError("robust baseline group counts must be positive and ordered")
        if group_start < 0:
            raise ValueError("robust baseline group_index_start must be nonnegative")
        if required_groups * group_trials < 25:
            raise ValueError("robust baseline calibration requires at least 25 total trials")
        base_env.env_cfg.num_trials_per_step = group_trials
        for local_group_idx in range(group_limit):
            group_idx = group_start + local_group_idx
            group_probe_seed = derive_baseline_group_probe_seed(base_seed, group_idx)
            base_env.probe_noise_seed = group_probe_seed
            base_env.clear_installed_blb()
            base_env.reset(seed=group_probe_seed)
            _state, _reward, _done, info = base_env.step(action)
            metrics = info.get("metrics") if isinstance(info, Mapping) else None
            if metrics is None:
                raise ValueError("robust baseline probe did not return EpisodeMetrics")

            loss_trials = tuple(float(value) for value in metrics.loss_trials)
            metric1_trials = tuple(float(value) for value in metrics.metric1_trials)
            metric2_trials = tuple(float(value) for value in metrics.metric2_trials)
            trial_seeds = tuple(int(value) for value in metrics.trial_seeds)
            if not (
                    len(loss_trials) == len(metric1_trials) == len(metric2_trials) == group_trials
                    and len(trial_seeds) == group_trials
            ):
                raise ValueError(
                    "robust baseline group raw trials and seeds must match trials_per_group"
                )
            group = TrialSeries(
                loss=loss_trials,
                metric1=metric1_trials,
                metric2=metric2_trials,
                seeds=trial_seeds,
            )
            groups.append(group)
            raw_groups.append({
                "group_index": int(group_idx),
                "group_probe_seed": int(group_probe_seed),
                "trial_seeds": [int(value) for value in trial_seeds],
                "loss_trials": [float(value) for value in loss_trials],
                "metric1_trials": [float(value) for value in metric1_trials],
                "metric2_trials": [float(value) for value in metric2_trials],
            })
            if len(groups) < required_groups:
                continue
            try:
                reference = build_baseline_reference(
                    groups,
                    precision_tolerance=precision_tolerance,
                    stability_multiplier=stability_multiplier,
                    bootstrap_samples=bootstrap_samples,
                    seed=base_seed,
                )
            except DegenerateBaselineVariance as exc:
                if local_group_idx == group_limit - 1:
                    exc.raw_groups = tuple(raw_groups)
                    raise
                continue

            summary = {
                "ok": True,
                "threshold_source": "robust_all_max_blb_baseline",
                "trial_count": int(reference.trial_count),
                "group_count": int(len(groups)),
                "groups": raw_groups,
                "pooled": {
                    "trial_count": int(reference.trial_count),
                    "loss_mean": float(reference.loss_mean),
                    "metric1_mean": float(reference.metric1_mean),
                    "metric2_mean": float(reference.metric2_mean),
                    "loss_std": float(reference.loss_std),
                    "metric1_std": float(reference.metric1_std),
                    "metric2_std": float(reference.metric2_std),
                    "limits": {
                        "loss": float(reference.loss_limit),
                        "metric1": float(reference.metric1_limit),
                        "metric2": float(reference.metric2_limit),
                        "loss_std": float(reference.loss_std_limit),
                        "metric1_std": float(reference.metric1_std_limit),
                        "metric2_std": float(reference.metric2_std_limit),
                    },
                },
                "limits": {
                    "loss": float(reference.loss_limit),
                    "metric1": float(reference.metric1_limit),
                    "metric2": float(reference.metric2_limit),
                    "loss_std": float(reference.loss_std_limit),
                    "metric1_std": float(reference.metric1_std_limit),
                    "metric2_std": float(reference.metric2_std_limit),
                },
                "bootstrap": {
                    "samples": int(reference.bootstrap_samples),
                    "seed": int(reference.bootstrap_seed),
                },
                "precision_tolerance": float(reference.precision_tolerance),
                "stability_multiplier": float(reference.stability_multiplier),
            }
            return reference, summary
    finally:
        try:
            base_env.clear_installed_blb()
        finally:
            base_env.env_cfg.num_trials_per_step = original_trials
            base_env.probe_noise_seed = original_probe_seed
            if reward_weights is not None and original_reward_design is not None:
                reward_weights.reward_design = original_reward_design
            if had_statistical_reference:
                base_env.statistical_reference = original_statistical_reference

    raise AssertionError("robust baseline collection exhausted without a result")


def _build_layerwise_candidate_identity_context(
        *,
        train_cfg: Any,
        evaluator: Any,
        fusion_map: Any,
        max_sfs: Any,
        fixed_gelu: np.ndarray,
        fixed_softmax: np.ndarray,
        robust_reference: Any,
        authoritative_robust_reference: Any,
        validation_banks: Any,
        probe_example_count: int,
        authoritative_example_count: int,
        schedule: Sequence[Any],
        static_skeletons_baseline: Any,
        algorithm_contract: Mapping[str, Any],
        algorithm_contract_hash: str,
        ) -> Dict[str, Any]:
    """Bind layerwise raw evidence to the complete effective run context."""
    from .candidate_store import build_candidate_identity_context, sha256_json
    from .layerwise_action import (
        K_LEVELS,
        LAYERWISE_COST_MODEL_REVISION,
        layerwise_action_space_version,
    )
    from .layerwise_runner import bind_layerwise_candidate_identity

    stage1_degrees = {
        "gelu": [int(value) for value in fixed_gelu.reshape(-1)],
        "softmax": [int(value) for value in fixed_softmax.reshape(-1)],
    }
    num_layers = len(stage1_degrees["gelu"])
    def reference_payload(reference: Any) -> Dict[str, Any]:
        return {
            "precision_tolerance": float(reference.precision_tolerance),
            "stability_multiplier": float(reference.stability_multiplier),
            "bootstrap_seed": int(reference.bootstrap_seed),
            "bootstrap_samples": int(reference.bootstrap_samples),
            "limits": {
                "loss": float(reference.loss_limit),
                "metric1": float(reference.metric1_limit),
                "metric2": float(reference.metric2_limit),
                "loss_std": float(reference.loss_std_limit),
                "metric1_std": float(reference.metric1_std_limit),
                "metric2_std": float(reference.metric2_std_limit),
            },
        }

    threshold_policy = {
        "precision_tolerance": float(robust_reference.precision_tolerance),
        "stability_multiplier": float(robust_reference.stability_multiplier),
        "bootstrap_seed": int(robust_reference.bootstrap_seed),
        "bootstrap_samples": int(robust_reference.bootstrap_samples),
        "online_constraint_probability": float(
            getattr(train_cfg, "online_constraint_probability", 0.50)
        ),
        "promotion_constraint_probability": float(
            getattr(train_cfg, "promotion_constraint_probability", 0.80)
        ),
        "final_constraint_probability": float(
            getattr(train_cfg, "final_constraint_probability", 0.95)
        ),
        "limits": {
            "loss": float(robust_reference.loss_limit),
            "metric1": float(robust_reference.metric1_limit),
            "metric2": float(robust_reference.metric2_limit),
            "loss_std": float(robust_reference.loss_std_limit),
            "metric1_std": float(robust_reference.metric1_std_limit),
            "metric2_std": float(robust_reference.metric2_std_limit),
        },
        "evidence_tiers": {
            "F1": {
                "split": "validation_full_stratified_probe",
                "example_count": int(probe_example_count),
                "reference": reference_payload(robust_reference),
            },
            "F4": {
                "split": "validation_full",
                "example_count": int(authoritative_example_count),
                "reference": reference_payload(authoritative_robust_reference),
                "validation_banks": validation_banks.contract_payload(),
            },
        },
    }
    rescale_root = os.path.realpath(str(train_cfg.inproc_rescale_optimizer_root))
    model_type = resolve_stage2_model_type(
        str(getattr(evaluator, "model_type", "") or ""),
        num_layers=num_layers,
    )
    context = build_candidate_identity_context(
        action_space_version=layerwise_action_space_version(num_layers),
        registry_hash=sha256_json(fusion_map),
        max_sfs_hash=sha256_json(max_sfs),
        stage1_config_content_hash=sha256_json(stage1_degrees),
        stage1_gelu_degrees=stage1_degrees["gelu"],
        stage1_softmax_degrees=stage1_degrees["softmax"],
        profile=str(train_cfg.profile),
        rescale_optimizer_mode="in_process_real",
        rescale_optimizer_root=rescale_root,
        rescale_optimizer_canonical_hash=sha256_json({
            "root": rescale_root,
            "static_skeletons": static_skeletons_baseline,
        }),
        decode_version="layerwise_action_v1",
        dataset=str(train_cfg.profile),
        model=model_type,
        metric_policy_version="robust_bootstrap_5x5_v1",
        threshold_policy_hash=sha256_json(threshold_policy),
        mask_schedule_hash=sha256_json(schedule),
    )
    return bind_layerwise_candidate_identity(
        context,
        K_LEVELS,
        LAYERWISE_COST_MODEL_REVISION,
        {
            "algorithm_contract_hash": str(algorithm_contract_hash),
            "resource_secondary_epsilon": algorithm_contract[
                "resource_secondary_epsilon"
            ],
            "compute_axis_denominator": algorithm_contract[
                "compute_axis_denominator"
            ],
            "communication_axis_denominator": algorithm_contract[
                "communication_axis_denominator"
            ],
            "resource_credit_mode": algorithm_contract[
                "resource_credit_mode"
            ],
            "strict_resource_order": algorithm_contract[
                "strict_resource_order"
            ],
        },
    )


def _run_layerwise_training_branch(
        *,
        train_cfg: Any,
        evaluator: Any,
        base_env: Any,
        fusion_map: Any,
        max_sfs: Any,
        robust_reference: Any,
        promotion_base_env: Any,
        authoritative_robust_reference: Any,
        authoritative_robust_summary: Optional[Mapping[str, Any]],
        authoritative_validation_banks: Any,
        authoritative_validation_example_count: int,
        static_skeletons_baseline: Any,
        baseline_action_vec: Sequence[int],
        fixed_gelu: np.ndarray,
        fixed_softmax: np.ndarray,
        fixed_label: str,
        fixed_source: str,
        blb_progress_dir: str,
        baseline_preflight_metrics: Mapping[str, Any],
        status: Any,
        resume_checkpoint_path: Any,
        run_lock: Any,
        log: Callable[[str], None],
        ) -> Dict[str, Any]:
    """Run Task-7 layerwise PPO without entering legacy block scaffolds."""
    if robust_reference is None:
        raise RuntimeError("layerwise robust PPO requires a calibrated statistical reference")
    if (
            promotion_base_env is None
            or authoritative_robust_reference is None
            or authoritative_validation_banks is None
    ):
        raise RuntimeError(
            "layerwise robust PPO requires an authoritative validation_full evaluator"
        )
    if list(getattr(train_cfg, "stage2_rl_devices", []) or []):
        raise RuntimeError(
            "layerwise PPO does not use the legacy block episode-parallel runner; "
            "configure reward_devices for terminal probe parallelism"
        )
    bullet = "*"

    from .candidate_store import CandidateStore, sha256_json
    from .layerwise_env import BLBStage2LayerwiseEnv
    from .diagnostics import EpisodeStats, PPOUpdateStats, RLDiagnosticsRecorder
    from .network_variants import (
        LEGACY_SHARED_RL_VARIANT,
        bind_policy_network_contract,
        policy_network_architecture,
        normalize_policy_network_variant,
        validate_checkpoint_policy_network_variant,
    )
    from .layerwise_action import (
        K_LEVELS as LAYERWISE_K_LEVELS,
        LAYERWISE_COST_MODEL_REVISION,
        RESOURCE_SECONDARY_EPSILON,
        layerwise_action_space_version,
        max_communication_saving_units,
        max_compute_saving_units,
    )
    from .fusion_fixed_action import build_fusion_fixed_config
    from .layerwise_runner import (
        DEFAULT_CONVERGENCE_PATIENCE_UPDATES,
        DEFAULT_CONVERGENCE_MIN_EPISODES,
        _PROBABILITY_FIELDS,
        _to_plain_mapping,
        build_layerwise_run_context,
        CheckpointFileFingerprintTracker,
        initialize_layerwise_policy,
        normalized_constraint_safety_margins,
        resolve_layerwise_episode_budget,
        StrictSelectionKey,
        strict_selection_key,
        strict_selection_key_from_snapshot,
        train_layerwise,
        validate_fresh_layerwise_run_state,
        validate_layerwise_checkpoint_metadata,
        validate_layerwise_episode_limit_extension,
    )
    from .persistence import write_training_curves
    from .runner import _build_legacy_compatible_best_noise_config
    from json_utils import to_jsonable

    layerwise_manifest_path = os.path.join(
        blb_progress_dir, "layerwise_run_manifest.json",
    )
    requested_total_episodes = int(train_cfg.total_episodes)
    resolve_layerwise_episode_budget(requested_total_episodes, 0)
    convergence_patience_updates = int(getattr(
        train_cfg,
        "convergence_patience_updates",
        DEFAULT_CONVERGENCE_PATIENCE_UPDATES,
    ))
    if convergence_patience_updates <= 0:
        raise ValueError("layerwise convergence patience must be positive")
    convergence_min_episodes = int(getattr(
        train_cfg,
        "convergence_min_episodes",
        DEFAULT_CONVERGENCE_MIN_EPISODES,
    ))
    if convergence_min_episodes < 0:
        raise ValueError("layerwise convergence minimum episodes must be nonnegative")
    protected_k1_enabled = bool(
        getattr(train_cfg, "protected_k1_enabled", False)
    )
    algorithm_revision = (
        "dual_resource_maxmin_shapley_three_bank_convergence_v11_protected_k1"
        if protected_k1_enabled else
        "dual_resource_maxmin_shapley_three_bank_convergence_v10"
    )
    policy_network_variant = normalize_policy_network_variant(
        getattr(train_cfg, "policy_network_variant", None)
    )
    policy_architecture = policy_network_architecture(policy_network_variant)
    rl_variant = LEGACY_SHARED_RL_VARIANT
    layerwise_entropy_regularization = {
        "kind": "disabled",
        "coefficient": 0.0,
        "optimization_role": "monitor_only",
    }
    layerwise_termination = {
        "mode": "convergence_or_max_episodes",
        "episode_limit": (
            None if requested_total_episodes == 0 else requested_total_episodes
        ),
        "minimum_episodes": convergence_min_episodes,
        "patience_updates": convergence_patience_updates,
        "requires_robust_feasible_candidate": True,
        "frontier_stall_update_windows": convergence_patience_updates,
        "selected_action_stable_update_windows": convergence_patience_updates,
        "strict_revalidation_required": True,
        "strict_revalidation_trials": int(
            authoritative_validation_banks.bank_c.trial_count
        ),
        "strict_revalidation_diagnostic_probability": float(
            getattr(train_cfg, "final_constraint_probability", 0.95)
        ),
        "selection_order": (
            "feasible,robust_floor,secondary_progress,confidence_vector,"
            "safety_margin_vector,"
            "action_lexicographic"
        ),
        "entropy_role": "diagnostic_only",
        "validation_banks": authoritative_validation_banks.contract_payload(),
        "counts_only_finite_ppo_updates": True,
    }
    algorithm_termination = dict(layerwise_termination)
    algorithm_termination["episode_limit"] = "runtime_extendable"
    layerwise_ppo_mode = {
        "factorized_actor_clip": True,
        "behavior_log_prob_source": "sampling_time_per_slot_v1",
        "actor_credit_mode": "shared_constraint_plus_own_resource_shapley",
        "actor_advantage_normalization": "per_slot_center_shared_scale_v1",
        "entropy_average_active_slots": True,
        "entropy_normalize_active_slots": True,
    }
    layerwise_env = BLBStage2LayerwiseEnv(
        base_env=base_env,
        fusion_map=fusion_map,
        baseline_action_vec=baseline_action_vec,
        profile=str(train_cfg.profile),
    )
    layerwise_horizon = int(layerwise_env.horizon)
    if layerwise_horizon != int(evaluator.total_layers):
        raise RuntimeError(
            "layerwise environment/model depth mismatch: "
            f"{layerwise_horizon} != {int(evaluator.total_layers)}"
        )
    online_probe_example_count = sum(
        int(batch.labels.numel()) for batch in base_env.probe_batches
    )
    if online_probe_example_count <= 0:
        raise RuntimeError("layerwise F1 probe must contain at least one example")
    if online_probe_example_count != 256:
        raise RuntimeError(
            "layerwise F1 probe must contain exactly 256 stratified examples; "
            f"received {online_probe_example_count}"
        )
    policy_cfg = SequentialPolicyConfig(
        state_dim=int(layerwise_env.state_dim),
        max_step_dim=6,
        max_num_levels=6,
        horizon=layerwise_horizon,
        num_layers=layerwise_horizon,
        metadata_width=0,
        signal_width=4,
        step_layer_indices=tuple(range(layerwise_horizon)),
        step_block_indices=(3,) * layerwise_horizon,
        network_variant=policy_network_variant,
        **policy_architecture,
    )
    ppo = SequentialPPOConfig(
        lr=float(train_cfg.ppo.lr),
        clip_range=float(train_cfg.ppo.clip_range),
        n_epochs=int(train_cfg.ppo.n_epochs),
        minibatch_size=int(train_cfg.ppo.minibatch_size),
        ent_coef=0.0,
        value_coef=float(train_cfg.ppo.value_coef),
        max_grad_norm=float(train_cfg.ppo.max_grad_norm),
        gamma=1.0,
        gae_lambda=1.0,
        per_slot_entropy_recovery=False,
        factorized_actor_clip=True,
        entropy_average_active_slots=True,
        entropy_normalize_active_slots=True,
    )
    algorithm_contract = {
        "schema_version": "stage2_layerwise_algorithm_contract_v5",
        "algorithm_revision": algorithm_revision,
        "rl_variant": rl_variant,
        "action_space_version": layerwise_action_space_version(
            layerwise_horizon
        ),
        "decode_version": "layerwise_action_v1",
        "cost_model_revision": LAYERWISE_COST_MODEL_REVISION,
        "k_levels": [int(value) for value in LAYERWISE_K_LEVELS],
        "resource_secondary_epsilon": float(RESOURCE_SECONDARY_EPSILON),
        "compute_axis_denominator": int(
            max_compute_saving_units(layerwise_horizon)
        ),
        "communication_axis_denominator": int(
            max_communication_saving_units(layerwise_horizon)
        ),
        "resource_credit_mode": "two_family_shapley_per_slot_v1",
        "strict_resource_order": ["robust_floor", "secondary_progress"],
        "resource_objective": {
            "compute_axis": "learnable_block4_fusion_count",
            "communication_axis": "removed_truncation_k_bits",
            "selection": "max_min_then_mean",
            "ppo_surrogate": "(robust_floor+eta*secondary_progress)/(1+eta)",
        },
        "policy": {
            "state_dim": int(policy_cfg.state_dim),
            "horizon": int(policy_cfg.horizon),
            "max_step_dim": int(policy_cfg.max_step_dim),
            "max_num_levels": int(policy_cfg.max_num_levels),
        },
        "ppo": asdict(ppo),
        "rollout_size": int(train_cfg.rollout_size),
        "ppo_mode": layerwise_ppo_mode,
        "entropy_regularization": layerwise_entropy_regularization,
        "termination": algorithm_termination,
        "evidence_tiers": {
            "F1": {
                "split": "validation_full_stratified_probe",
                "example_count": int(online_probe_example_count),
                "trials_per_episode": int(train_cfg.num_trials_per_step),
                "baseline_trial_count": int(
                    getattr(train_cfg, "baseline_groups", 5)
                    * getattr(train_cfg, "baseline_trials_per_group", 5)
                ),
                "roles": ["ppo_reward", "advantage", "promotion_prefilter"],
                "authoritative": False,
            },
            "F4": {
                "split": "validation_full",
                "example_count": int(authoritative_validation_example_count),
                "bank_a_trial_count": int(
                    authoritative_validation_banks.bank_a.trial_count
                ),
                "bank_b_trial_count": int(
                    authoritative_validation_banks.bank_b.trial_count
                ),
                "bank_c_trial_count": int(
                    authoritative_validation_banks.bank_c.trial_count
                ),
                "promotion_pooled_trial_count": int(
                    authoritative_validation_banks.promotion_trial_count
                ),
                "final_pooled_trial_count": int(
                    authoritative_validation_banks.final_trial_count
                ),
                "hard_gate": "six_point_constraints",
                "bootstrap_probability_role": "diagnostic_tiebreak_only",
                "roles": ["strict_frontier", "convergence", "final_selection"],
                "authoritative": True,
            },
        },
        "persistence_protocol": "stable_parent_lock_incremental_fingerprint_v2",
    }
    if protected_k1_enabled:
        algorithm_contract["protected_k1"] = {
            "role": "extreme_precision_reject_only",
            "full_trial_count": 5,
            "guard_sigma": float(
                getattr(train_cfg, "protected_k1_guard_sigma", 4.0)
            ),
            "audit_fraction": float(
                getattr(train_cfg, "protected_k1_audit_fraction", 0.02)
            ),
            "candidate_store_eligible": False,
            "stability_measured": False,
            "reward_p1_probability_ceiling": 0.5,
            "frontier_protection": (
                "compute_communication_nondominated_or_equal"
            ),
            "audit_false_negative_behavior": "fail_open_to_k5",
            "fail_open_persistence": "checkpointed",
        }
    algorithm_contract = bind_policy_network_contract(
        algorithm_contract,
        policy_network_variant,
        policy_shape={
            "state_dim": int(policy_cfg.state_dim),
            "horizon": int(policy_cfg.horizon),
            "max_step_dim": int(policy_cfg.max_step_dim),
            "max_num_levels": int(policy_cfg.max_num_levels),
            "d_model": int(policy_cfg.d_model),
            "n_heads": int(policy_cfg.n_heads),
            "n_layers": int(policy_cfg.n_layers),
            "d_ff": int(policy_cfg.d_ff),
            "dropout": float(policy_cfg.dropout),
            "actor_dim": int(policy_cfg.actor_dim),
            "critic_dim": int(policy_cfg.critic_dim),
            "mlp_critic_hidden": [512, 512, int(policy_cfg.d_model)],
        },
    )
    rl_variant = str(algorithm_contract["rl_variant"])
    algorithm_contract_hash = sha256_json(algorithm_contract)
    identity_context = _build_layerwise_candidate_identity_context(
        train_cfg=train_cfg,
        evaluator=evaluator,
        fusion_map=fusion_map,
        max_sfs=max_sfs,
        fixed_gelu=fixed_gelu,
        fixed_softmax=fixed_softmax,
        robust_reference=robust_reference,
        authoritative_robust_reference=authoritative_robust_reference,
        validation_banks=authoritative_validation_banks,
        probe_example_count=int(online_probe_example_count),
        authoritative_example_count=int(authoritative_validation_example_count),
        schedule=layerwise_env.schedule,
        static_skeletons_baseline=static_skeletons_baseline,
        algorithm_contract=algorithm_contract,
        algorithm_contract_hash=algorithm_contract_hash,
    )
    run_context = build_layerwise_run_context(
        identity_context,
        algorithm_contract_hash,
        {
            "online_trials_per_episode": int(train_cfg.num_trials_per_step),
            "promotion_validation_trials": int(
                getattr(train_cfg, "promotion_validation_trials", 25)
            ),
            "final_selection_validation_trials": int(
                getattr(train_cfg, "final_selection_validation_trials", 25)
            ),
            "baseline_groups": int(getattr(train_cfg, "baseline_groups", 5)),
            "baseline_trials_per_group": int(
                getattr(train_cfg, "baseline_trials_per_group", 5)
            ),
            "constraint_bootstrap_samples": int(
                getattr(train_cfg, "constraint_bootstrap_samples", 4096)
            ),
            "online_constraint_probability": float(
                getattr(train_cfg, "online_constraint_probability", 0.50)
            ),
            "promotion_constraint_probability": float(
                getattr(train_cfg, "promotion_constraint_probability", 0.80)
            ),
            "final_constraint_probability": float(
                getattr(train_cfg, "final_constraint_probability", 0.95)
            ),
            "convergence_min_episodes": int(convergence_min_episodes),
            "convergence_patience_updates": int(convergence_patience_updates),
            **({
                "protected_k1": dict(algorithm_contract["protected_k1"])
            } if protected_k1_enabled else {}),
            "validation_banks": authoritative_validation_banks.contract_payload(),
            "evidence_tiers": {
                "F1": {
                    "split": "validation_full_stratified_probe",
                    "example_count": int(online_probe_example_count),
                    "fidelity": "F1",
                    "baseline_reference": dict(
                        baseline_preflight_metrics.get("robust_reference") or {}
                    ),
                },
                "F4": {
                    "split": "validation_full",
                    "example_count": int(authoritative_validation_example_count),
                    "fidelity": "F4",
                    "baseline_reference": dict(authoritative_robust_summary or {}),
                },
            },
        },
    )
    run_context_hash = sha256_json(run_context)
    run_lock.bind_context(run_context_hash)
    run_manifest = {
        "schema_version": "stage2_layerwise_robust_run_v5",
        "status": "running",
        "rl_variant": rl_variant,
        "policy_network_variant": policy_network_variant,
        "algorithm_revision": algorithm_revision,
        "algorithm_contract": algorithm_contract,
        "algorithm_contract_hash": algorithm_contract_hash,
        "run_context": run_context,
        "run_context_hash": run_context_hash,
        "profile": str(train_cfg.profile),
        "decision_granularity": "layer",
        "reward_design": "robust_constrained",
        "fixed_gelu": [int(value) for value in np.asarray(fixed_gelu).reshape(-1)],
        "fixed_softmax": [int(value) for value in np.asarray(fixed_softmax).reshape(-1)],
        "fixed_label": str(fixed_label),
        "fixed_source": str(fixed_source),
        "planned_episodes": layerwise_termination["episode_limit"],
        "entropy_regularization": layerwise_entropy_regularization,
        "termination": layerwise_termination,
        "evidence_tiers": algorithm_contract["evidence_tiers"],
        "baseline_references": {
            "F1": dict(baseline_preflight_metrics.get("robust_reference") or {}),
            "F4": dict(authoritative_robust_summary or {}),
        },
        "ppo_mode": layerwise_ppo_mode,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    torch.manual_seed(int(train_cfg.seed))
    np.random.seed(int(train_cfg.seed) % (2**32))
    random.seed(int(train_cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = BLBStage2SequentialPolicy(policy_cfg).to(device)
    policy_network_summary = policy.network_parameter_summary()
    run_manifest["policy_network"] = policy_network_summary
    log(
        f"  {bullet} Stage-2 policy network: {policy_network_variant} "
        f"(total={policy_network_summary['total']:,}, "
        f"shared={policy_network_summary['shared']:,}, "
        f"actor_only={policy_network_summary['actor_only']:,}, "
        f"critic_only={policy_network_summary['critic_only']:,})"
    )
    initialize_layerwise_policy(policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(train_cfg.ppo.lr))
    save_path = os.path.join(blb_progress_dir, "blb_stage2_rl_checkpoint_live.pt")
    candidate_store_path = os.path.join(blb_progress_dir, "candidate_store.jsonl")
    effective_resume_path = resume_checkpoint_path
    if not effective_resume_path and os.path.isfile(save_path):
        effective_resume_path = save_path
        log(f"  {bullet} 检测到 layerwise live checkpoint，自动 resume: {save_path}")
    start_episode = 0
    resumed_best: Dict[str, Any] = {}
    resumed_strict_pareto_frontier: List[Dict[str, Any]] = []
    resumed_convergence_state: Dict[str, Any] = {}
    resumed_candidate_store_size: Optional[int] = None
    resumed_diagnostics_jsonl_sizes: Optional[Mapping[str, Any]] = None
    resumed_store_file_fingerprints: Optional[Mapping[str, Any]] = None
    resumed_structured_run_id: Optional[str] = None
    resumed_ppo_update_count = 0
    resume_checkpoint: Optional[Mapping[str, Any]] = None
    cuda_rng_role_registry: List[Any] = []
    resumed_active_cuda_rng_states: Optional[List[Any]] = None
    planned_total_episodes = requested_total_episodes
    if effective_resume_path and os.path.isfile(effective_resume_path):
        try:
            checkpoint = torch.load(
                effective_resume_path, map_location=device, weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(effective_resume_path, map_location=device)
        validate_checkpoint_policy_network_variant(
            checkpoint, policy_network_variant,
        )
        validate_layerwise_checkpoint_metadata(
            checkpoint,
            rl_variant=rl_variant,
            algorithm_revision=algorithm_revision,
            algorithm_contract_hash=algorithm_contract_hash,
            run_context_hash=run_context_hash,
        )
        active_cuda_role_count = (
            int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        )
        cuda_rng_role_registry, resumed_active_cuda_rng_states = (
            resolve_cuda_rng_role_registry(
                checkpoint,
                active_role_count=active_cuda_role_count,
                new_role_state_factory=lambda role_index: (
                    torch.Generator(device=f"cuda:{role_index}")
                    .manual_seed(int(train_cfg.seed))
                    .get_state()
                    .cpu()
                ),
            )
        )
        resume_checkpoint = checkpoint
        start_episode = int(checkpoint.get("episode", 0))
        resumed_best = dict(checkpoint.get("strict_best") or {})
        if checkpoint.get("strict_pareto_frontier") is None:
            raise RuntimeError(
                "layerwise checkpoint strict resource Pareto frontier is missing"
            )
        resumed_strict_pareto_frontier = [
            dict(row) for row in checkpoint["strict_pareto_frontier"]
        ]
        resumed_convergence_state = dict(checkpoint.get("convergence_state") or {})
        resumed_candidate_store_size = checkpoint.get("candidate_store_size")
        resumed_diagnostics_jsonl_sizes = checkpoint.get("diagnostics_jsonl_sizes")
        resumed_store_file_fingerprints = checkpoint.get("store_file_fingerprints")
        resumed_structured_run_id = str(
            checkpoint.get("structured_run_id", "") or ""
        )
        if checkpoint.get("ppo_update_count") is None:
            raise RuntimeError("layerwise checkpoint PPO update count is missing")
        resumed_ppo_update_count = int(checkpoint.get("ppo_update_count"))
        if resumed_ppo_update_count < 0:
            raise RuntimeError("layerwise checkpoint PPO update count is invalid")
        checkpoint_planned_total = int(checkpoint.get(
            "planned_total_episodes", planned_total_episodes,
        ))
        validate_layerwise_episode_limit_extension(
            checkpoint_planned_total, planned_total_episodes,
        )
        log(f"  {bullet} layerwise resume @ episode {start_episode}")
    remaining_episode_budget = resolve_layerwise_episode_budget(
        requested_total_episodes,
        start_episode,
    )
    layerwise_train_cfg = SequentialTrainConfig(
        total_episodes=remaining_episode_budget,
        update_every_n_episodes=max(1, int(train_cfg.rollout_size)),
        log_every_n_episodes=max(1, int(train_cfg.rollout_size)),
        seed=int(train_cfg.seed),
        absolute_episode_start=int(start_episode),
        planned_total_episodes=int(planned_total_episodes),
        convergence_resume_state=resumed_convergence_state,
        convergence_min_episodes=convergence_min_episodes,
        convergence_patience_updates=convergence_patience_updates,
        ppo=ppo,
        rl_algo="ppo",
        online_num_trials_per_step=int(train_cfg.num_trials_per_step),
        terminal_eval_batch_size=int(train_cfg.terminal_eval_batch_size),
        protected_k1_enabled=bool(
            getattr(train_cfg, "protected_k1_enabled", False)
        ),
        protected_k1_guard_sigma=float(
            getattr(train_cfg, "protected_k1_guard_sigma", 4.0)
        ),
        protected_k1_audit_fraction=float(
            getattr(train_cfg, "protected_k1_audit_fraction", 0.02)
        ),
        promotion_validation_trials=int(
            getattr(train_cfg, "promotion_validation_trials", 25)
        ),
        final_selection_validation_trials=int(
            getattr(train_cfg, "final_selection_validation_trials", 25)
        ),
        online_constraint_probability=float(
            getattr(train_cfg, "online_constraint_probability", 0.50)
        ),
        promotion_constraint_probability=float(
            getattr(train_cfg, "promotion_constraint_probability", 0.80)
        ),
        final_constraint_probability=float(
            getattr(train_cfg, "final_constraint_probability", 0.95)
        ),
        reward_design="robust_constrained",
    )
    candidate_store = CandidateStore(candidate_store_path)
    from jsonl_utils import iter_jsonl

    diagnostics_dir = os.path.join(blb_progress_dir, "diagnostics")
    existing_episode_path = os.path.join(diagnostics_dir, "episodes.jsonl")
    existing_update_path = os.path.join(diagnostics_dir, "ppo_updates.jsonl")

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_id_marker = os.path.join(blb_progress_dir, "rl_data_points_run_id.txt")
    if resume_checkpoint is None:
        validate_fresh_layerwise_run_state(
            run_id_marker,
            (
                candidate_store.path,
                existing_episode_path,
                existing_update_path,
                layerwise_manifest_path,
            ),
        )
    if os.path.isfile(run_id_marker):
        with open(run_id_marker, encoding="utf-8") as handle:
            structured_run_id = handle.read().strip()
    elif resume_checkpoint is not None:
        raise RuntimeError(
            "layerwise checkpoint structured run-id marker is missing; "
            "restore the complete run directory or start a fresh run"
        )
    else:
        run_source = (
            str(getattr(evaluator, "run_output_dir", "") or "").strip()
            or os.path.dirname(os.path.normpath(blb_progress_dir))
        )
        try:
            run_id_base = os.path.relpath(run_source, repo_root)
        except ValueError:
            run_id_base = run_source
        structured_run_id = make_unique_run_id(run_id_base)
        marker_tmp = run_id_marker + ".tmp"
        with open(marker_tmp, "w", encoding="utf-8") as handle:
            handle.write(structured_run_id + "\n")
        os.replace(marker_tmp, run_id_marker)
    layerwise_model_type = resolve_stage2_model_type(
        str(getattr(evaluator, "model_type", "") or ""),
        num_layers=layerwise_horizon,
    )
    stage2_data_writer = RLDataPointWriter(
        root_dir=os.path.join(repo_root, "rl_training_data_points"),
        run_id=structured_run_id,
        stage="stage2",
        model_type=layerwise_model_type,
        dataset=str(train_cfg.profile),
    )

    def layerwise_slots_view(action_vec):
        from .action_io import action_vec_to_slots_list

        return action_vec_to_slots_list(
            action_vec,
            max_sfs=max_sfs,
            num_layers=int(evaluator.total_layers),
            gelu_degree=fixed_gelu,
            attn_degree=fixed_softmax,
            profile=str(train_cfg.profile),
        )

    diag_recorder = RLDiagnosticsRecorder(
        output_dir=blb_progress_dir,
        num_layers=int(evaluator.total_layers),
        num_action_slots=int(getattr(base_env, "total_action_dim", 0) or 0),
        max_action_levels=64,
        top_k=20,
        log_fn=log,
        slots_view_builder=layerwise_slots_view,
        schema_version="stage2_layerwise_action_v1",
        data_point_writer=stage2_data_writer,
        strict_writes=True,
        history_window=600,
        ppo_history_window=10,
    )

    def checkpoint_file_specs(
            candidate_size: Any,
            diagnostics_sizes: Any,
            ) -> Dict[str, Tuple[Any, int]]:
        if candidate_size is None:
            raise RuntimeError("layerwise checkpoint candidate_store_size is missing")
        sizes = dict(diagnostics_sizes or {})
        primary = dict(sizes.get("primary") or {})
        structured = dict(sizes.get("structured") or {})
        specs: Dict[str, Tuple[Any, int]] = {
            "candidate_store.jsonl": (candidate_store.path, int(candidate_size)),
            "primary/episodes.jsonl": (
                diag_recorder.episodes_path,
                int(primary.get("episodes.jsonl", 0)),
            ),
            "primary/ppo_updates.jsonl": (
                diag_recorder.ppo_updates_path,
                int(primary.get("ppo_updates.jsonl", 0)),
            ),
            "structured/episodes.jsonl": (
                stage2_data_writer.jsonl_path("episodes.jsonl"),
                int(structured.get("episodes.jsonl", 0)),
            ),
            "structured/ppo_updates.jsonl": (
                stage2_data_writer.jsonl_path("ppo_updates.jsonl"),
                int(structured.get("ppo_updates.jsonl", 0)),
            ),
        }
        return specs

    fingerprint_tracker = CheckpointFileFingerprintTracker()
    if resume_checkpoint is not None:
        if resumed_structured_run_id != structured_run_id:
            raise RuntimeError(
                "layerwise checkpoint structured run-id mismatch; "
                "restore the complete run directory or start a fresh run"
            )
        resume_file_specs = checkpoint_file_specs(
            resumed_candidate_store_size,
            resumed_diagnostics_jsonl_sizes,
        )
        fingerprint_tracker.validate_and_seed(
            dict(resumed_store_file_fingerprints or {}),
            resume_file_specs,
        )
        policy.load_state_dict(resume_checkpoint["policy"])
        if resume_checkpoint.get("policy_ppo_aux") is not None:
            policy.load_ppo_aux_state_dict(resume_checkpoint["policy_ppo_aux"])
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        candidate_store.recover_to_checkpoint_size(
            int(resumed_candidate_store_size),
        )
    diag_recorder.recover_to_checkpoint_sizes(
        resumed_diagnostics_jsonl_sizes,
    )
    if resume_checkpoint is not None:
        if resume_checkpoint.get("torch_rng_state") is not None:
            torch.set_rng_state(resume_checkpoint["torch_rng_state"].cpu())
        if resumed_active_cuda_rng_states is not None:
            for role_index, state in enumerate(resumed_active_cuda_rng_states):
                torch.cuda.set_rng_state(state.cpu(), device=role_index)
        if resume_checkpoint.get("numpy_rng_state") is not None:
            np.random.set_state(resume_checkpoint["numpy_rng_state"])
        if resume_checkpoint.get("python_rng_state") is not None:
            random.setstate(resume_checkpoint["python_rng_state"])
    write_strict_json_file(layerwise_manifest_path, run_manifest)
    diag_recorder.set_baseline_action_vec(layerwise_env.pending_full_vector)
    restored_diagnostics = diag_recorder.restore_existing()
    expected_episode_high_water = int(start_episode) - 1
    if (
            int(restored_diagnostics["episodes"]) != int(start_episode)
            or int(diag_recorder.episode_high_water)
            != expected_episode_high_water
    ):
        raise RuntimeError(
            "layerwise checkpoint episode diagnostics mismatch: "
            f"checkpoint_count={start_episode}, "
            f"restored_count={restored_diagnostics['episodes']}, "
            f"restored_high_water={diag_recorder.episode_high_water}"
        )
    if (
            int(restored_diagnostics["ppo_updates"])
            != int(resumed_ppo_update_count)
            or int(diag_recorder.ppo_update_high_water)
            != int(resumed_ppo_update_count)
    ):
        raise RuntimeError(
            "layerwise checkpoint PPO diagnostics mismatch: "
            f"checkpoint_count={resumed_ppo_update_count}, "
            f"restored_count={restored_diagnostics['ppo_updates']}, "
            f"restored_high_water={diag_recorder.ppo_update_high_water}"
        )
    probability_thresholds = {
        "online": float(getattr(train_cfg, "online_constraint_probability", 0.50)),
        "promotion": float(getattr(train_cfg, "promotion_constraint_probability", 0.80)),
        "final": float(getattr(train_cfg, "final_constraint_probability", 0.95)),
    }
    constraint_limits = {
        "loss": float(robust_reference.loss_limit),
        "metric1": float(robust_reference.metric1_limit),
        "metric2": float(robust_reference.metric2_limit),
        "loss_std": float(robust_reference.loss_std_limit),
        "metric1_std": float(robust_reference.metric1_std_limit),
        "metric2_std": float(robust_reference.metric2_std_limit),
    }
    diag_recorder.set_meta({
        "profile": str(train_cfg.profile),
        "fixed_label": str(fixed_label),
        "fixed_source": str(fixed_source),
        "rl_variant": rl_variant,
        "policy_network_variant": policy_network_variant,
        "policy_network": policy_network_summary,
        "decision_granularity": "layer",
        "reward_design": "robust_constrained",
        "algorithm_revision": algorithm_revision,
        "algorithm_contract_hash": algorithm_contract_hash,
        "run_context_hash": run_context_hash,
        "cost_model_revision": LAYERWISE_COST_MODEL_REVISION,
        "resource_objective": dict(algorithm_contract["resource_objective"]),
        "resource_secondary_epsilon": float(RESOURCE_SECONDARY_EPSILON),
        "compute_axis_denominator": int(
            algorithm_contract["compute_axis_denominator"]
        ),
        "communication_axis_denominator": int(
            algorithm_contract["communication_axis_denominator"]
        ),
        "resource_credit_mode": algorithm_contract["resource_credit_mode"],
        "strict_resource_order": list(
            algorithm_contract["strict_resource_order"]
        ),
        "total_episodes_planned": layerwise_termination["episode_limit"],
        "rollout_size": int(train_cfg.rollout_size),
        "ppo_lr": float(train_cfg.ppo.lr),
        "gamma": 1.0,
        "gae_lambda": 1.0,
        "entropy_regularization": layerwise_entropy_regularization,
        "termination": layerwise_termination,
        "ppo_mode": layerwise_ppo_mode,
        "stage2_k_trials": int(train_cfg.num_trials_per_step),
        "baseline_groups": int(getattr(train_cfg, "baseline_groups", 5)),
        "baseline_trials_per_group": int(
            getattr(train_cfg, "baseline_trials_per_group", 5)
        ),
        "constraint_bootstrap_samples": int(
            getattr(train_cfg, "constraint_bootstrap_samples", 4096)
        ),
        "constraint_probabilities": probability_thresholds,
        "constraint_limits": constraint_limits,
        "baseline_preflight_metrics": dict(baseline_preflight_metrics),
        "borderline_retest_enabled": False,
        "borderline_retest_trials_multiplier": 1,
    })
    log(f"  {bullet} [data-points] layerwise Stage-2 → {stage2_data_writer.run_dir}")

    recent_episode_records = deque(maxlen=max(1, int(train_cfg.rollout_size)))
    completed_episode_count = int(start_episode)
    best_reward_so_far = resolve_resumed_best_reward(
        resumed_best, diag_recorder.best_episode_return
    )
    best_selection_key: Optional[StrictSelectionKey] = None
    strict_best: Dict[str, Any] = dict(resumed_best)
    strict_pareto_frontier: List[Dict[str, Any]] = copy.deepcopy(
        resumed_strict_pareto_frontier
    )
    best_selection_key = strict_selection_key_from_snapshot(resumed_best)
    ppo_update_counter = int(resumed_ppo_update_count)
    started_at = time.time()

    def build_reloadable_best_group(best_payload: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        action_vec = best_payload.get("full_vector")
        if action_vec is None:
            return None
        fixed_config = build_fusion_fixed_config(
            action_vec,
            profile=str(train_cfg.profile),
            num_layers=int(evaluator.total_layers),
            gelu=np.asarray(fixed_gelu, dtype=int),
            softmax=np.asarray(fixed_softmax, dtype=int),
            fusion_map=fusion_map,
            source="stage2_layerwise_strict_best",
        )
        group = dict(fixed_config["group"])
        group["policy_actions"] = best_payload.get("action_matrix")
        overrides = best_payload.get("boosted_overrides") or {}
        group["boosted_overrides"] = [
            {
                "block_idx": int(block_idx),
                "layer_idx": int(layer_idx),
                "field_values": {
                    str(name): int(value) for name, value in values.items()
                },
            }
            for (block_idx, layer_idx), values in sorted(
                overrides.items(),
                key=lambda item: (int(item[0][1]), int(item[0][0])),
            )
        ]
        return group

    def write_strict_best_diagnostics(
            best_payload: Mapping[str, Any],
            *,
            episode: int,
            ) -> None:
        full_vector = best_payload.get("full_vector")
        if full_vector is None or len(full_vector) == 0:
            diag_recorder.clear_best_action_snapshot()
            return
        action_matrix = [list(row) for row in (best_payload.get("action_matrix") or [])]
        b4_count = sum(int(row[0]) for row in action_matrix if row)
        reward = float(best_payload.get("reward") or 0.0)
        variable_cost = float(best_payload.get("variable_cost") or 0.0)
        metrics = dict(best_payload.get("metrics") or {})
        diag_recorder.write_best_action_snapshot(
            episode_stats=EpisodeStats(
                episode=int(episode),
                total_reward=reward,
                terminal_reward=reward,
                per_step_sum=0.0,
                valid_steps=layerwise_horizon,
                invalid_steps=0,
                steps_taken=layerwise_horizon,
                total_bits=0,
                fusion_count=2 * layerwise_horizon + b4_count,
                first_invalid_step=None,
                first_invalid_block=None,
                first_invalid_layer=None,
                early_terminated=False,
                fusion_count_b2=layerwise_horizon,
                fusion_count_b4=b4_count,
                fusion_count_b5=layerwise_horizon,
                terminal_priority=3,
                terminal_loss_mean=float(metrics.get("loss_mean", 0.0)),
                terminal_loss_std=float(metrics.get("loss_std", 0.0)),
                terminal_metric1_mean=float(metrics.get("metric1_mean", 0.0)),
                terminal_metric1_std=float(metrics.get("metric1_std", 0.0)),
                terminal_metric2_mean=float(metrics.get("metric2_mean", 0.0)),
                terminal_metric2_std=float(metrics.get("metric2_std", 0.0)),
                terminal_cost_score=variable_cost,
                terminal_cost_rank_score=variable_cost,
                variable_cost=variable_cost,
                compute_saving=float(best_payload.get("compute_saving") or 0.0),
                communication_saving=float(
                    best_payload.get("communication_saving") or 0.0
                ),
                robust_floor=float(best_payload.get("robust_floor") or 0.0),
                secondary_progress=float(
                    best_payload.get("secondary_progress") or 0.0
                ),
                ppo_resource_score=float(
                    best_payload.get("ppo_resource_score") or variable_cost
                ),
                compute_shapley_credit=float(
                    best_payload.get("compute_shapley_credit") or 0.0
                ),
                communication_shapley_credit=float(
                    best_payload.get("communication_shapley_credit") or 0.0
                ),
                layer_resource_rewards=list(
                    best_payload.get("layer_resource_rewards") or []
                ),
                slot_resource_rewards=list(
                    best_payload.get("slot_resource_rewards") or []
                ),
                layer_action_matrix=action_matrix,
                promotion_status="strict_best_reconciled",
            ),
            full_action_vec=np.asarray(full_vector, dtype=np.int64),
            best_reward_so_far=reward,
        )

    def save_layerwise_checkpoint(
            *,
            completed: int,
            strict_best: Optional[Mapping[str, Any]],
            convergence_state: Optional[Mapping[str, Any]],
            ) -> None:
        nonlocal cuda_rng_role_registry
        best_payload = dict(strict_best or {})
        checkpoint_best_action = best_payload.get("full_vector")
        checkpoint_best_group = build_reloadable_best_group(best_payload)
        candidate_store_size = (
            candidate_store.path.stat().st_size
            if candidate_store.path.exists() else 0
        )
        diagnostics_jsonl_sizes = diag_recorder.committed_jsonl_sizes()
        store_file_fingerprints = fingerprint_tracker.fingerprints(
            checkpoint_file_specs(candidate_store_size, diagnostics_jsonl_sizes)
        )
        active_cuda_rng_states = (
            [
                state.cpu()
                for state in torch.cuda.get_rng_state_all()
            ]
            if torch.cuda.is_available()
            else []
        )
        cuda_rng_role_registry = merge_cuda_rng_role_registry(
            cuda_rng_role_registry,
            active_cuda_rng_states,
        )
        checkpoint = {
            "policy": policy.state_dict(),
            "policy_ppo_aux": policy.ppo_aux_state_dict(),
            "optimizer": optimizer.state_dict(),
            "episode": int(completed),
            "strict_best": best_payload,
            "strict_pareto_frontier": copy.deepcopy(strict_pareto_frontier),
            "best_action": checkpoint_best_action,
            "blb_v3_best_action_vec": checkpoint_best_action,
            "blb_v3_best_action_group": checkpoint_best_group,
            "blb_v3_fusion_count_action": True,
            "profile": str(train_cfg.profile),
            "convergence_state": dict(convergence_state or {}),
            "planned_total_episodes": int(planned_total_episodes),
            "candidate_store_size": int(candidate_store_size),
            "diagnostics_jsonl_sizes": diagnostics_jsonl_sizes,
            "store_file_fingerprints": store_file_fingerprints,
            "structured_run_id": structured_run_id,
            "ppo_update_count": int(ppo_update_counter),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": active_cuda_rng_states or None,
            "cuda_rng_role_registry_version": 1,
            "cuda_rng_state_by_role": cuda_rng_role_registry,
            "cuda_rng_active_role_count": len(active_cuda_rng_states),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "rl_variant": rl_variant,
            "policy_network_variant": policy_network_variant,
            "policy_network": policy_network_summary,
            "algorithm_revision": algorithm_revision,
            "algorithm_contract": algorithm_contract,
            "algorithm_contract_hash": algorithm_contract_hash,
            "run_context": run_context,
            "run_context_hash": run_context_hash,
        }
        tmp_path = save_path + ".tmp"
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, save_path)

    if resume_checkpoint is None:
        save_layerwise_checkpoint(
            completed=0,
            strict_best=strict_best,
            convergence_state={},
        )

    def on_layerwise_episode(record: Any) -> None:
        nonlocal best_selection_key, completed_episode_count, best_reward_so_far
        episode_identity = int(record.episode_index)
        if episode_identity != completed_episode_count:
            raise RuntimeError(
                "layerwise episode callback identity mismatch: "
                f"expected={completed_episode_count}, received={episode_identity}"
            )
        if episode_identity != diag_recorder.episode_high_water + 1:
            raise RuntimeError(
                "layerwise episode diagnostics identity mismatch: "
                f"high_water={diag_recorder.episode_high_water}, "
                f"received={episode_identity}"
            )
        completed_episode_count += 1
        best_reward_so_far = max(best_reward_so_far, float(record.reward))
        recent_episode_records.append(record)
        pooled_assessment = _to_plain_mapping(record.assessment)
        fresh_assessment = _to_plain_mapping(record.fresh_assessment)
        fresh_metrics = dict(record.metrics or {})
        pooled_metrics = dict(record.pooled_metrics or {})
        probe_diagnostics = _to_plain_mapping(record.probe_diagnostics)

        def trials_payload(trials: Any) -> Dict[str, Any]:
            if trials is None:
                return {}
            return {
                "loss": [float(value) for value in trials.loss],
                "metric1": [float(value) for value in trials.metric1],
                "metric2": [float(value) for value in trials.metric2],
                "seeds": [int(value) for value in trials.seeds],
            }

        fresh_trials = trials_payload(record.raw_trials)
        pooled_trials = trials_payload(record.pooled_trials)
        fresh_probabilities = {
            name: float(fresh_assessment[name])
            for name in _PROBABILITY_FIELDS if name in fresh_assessment
        }
        pooled_probabilities = {
            name: float(pooled_assessment[name])
            for name in _PROBABILITY_FIELDS if name in pooled_assessment
        }
        promotion_assessment = _to_plain_mapping(record.promotion_assessment)
        promotion_metrics = _to_plain_mapping(record.promotion_metrics)
        promotion_probabilities = {
            name: float(promotion_assessment[name])
            for name in _PROBABILITY_FIELDS if name in promotion_assessment
        }
        b4_count = sum(int(row[0]) for row in record.action_matrix)
        k_values = []
        for layer_idx, row in enumerate(record.action_matrix):
            for slot_idx in range(1, 6):
                if layer_idx == 0 and slot_idx == 1:
                    continue
                k_values.append(int(LAYERWISE_K_LEVELS[int(row[slot_idx])]))
        avg_k = float(np.mean(k_values)) if k_values else 13.0
        is_new_best = False
        if (
                record.promotion_status in ("promoted", "already_promoted")
                and record.promotion_candidate_key
                and len(promotion_probabilities) == len(_PROBABILITY_FIELDS)
        ):
            selection_key = strict_selection_key(
                record.promotion_candidate_key,
                {
                    "variable_cost": record.variable_cost,
                    "compute_saving": record.compute_saving,
                    "communication_saving": record.communication_saving,
                    "robust_floor": record.robust_floor,
                    "secondary_progress": record.secondary_progress,
                    "action_matrix": record.action_matrix,
                    "assessment": promotion_assessment,
                    "metrics": promotion_metrics,
                    "constraint_safety_margins": (
                        normalized_constraint_safety_margins(
                            promotion_metrics,
                            authoritative_robust_reference,
                        )
                    ),
                    "full_vector": record.pending_full_vector,
                },
            )
            if best_selection_key is None or selection_key < best_selection_key:
                best_selection_key = selection_key
                is_new_best = True
        convergence_state = {
            "stall_update_windows": int(record.stall_update_windows),
            "selected_action_identity": record.selected_action_identity,
            "selected_action_stable_update_windows": int(
                record.selected_action_stable_update_windows
            ),
            "converged": bool(record.converged),
            "extension_required": bool(record.extension_required),
            "plateau_ready": bool(record.plateau_ready),
            "strict_revalidation_passed": bool(
                record.strict_revalidation_passed
            ),
            "strict_revalidation_status": str(
                record.strict_revalidation_status
            ),
            "termination_reason": str(record.termination_reason),
            "best_robust_feasible_cost": record.best_robust_feasible_cost,
            "best_robust_feasible_objective": (
                None
                if record.best_robust_feasible_objective is None
                else list(record.best_robust_feasible_objective)
            ),
        }
        episode_stats = EpisodeStats(
                episode=int(record.episode_index),
                total_reward=float(record.reward),
                terminal_reward=float(record.reward),
                per_step_sum=0.0,
                valid_steps=layerwise_horizon - int(record.invalid_steps),
                invalid_steps=int(record.invalid_steps),
                steps_taken=layerwise_horizon,
                total_bits=0,
                fusion_count=2 * layerwise_horizon + b4_count,
                first_invalid_step=None,
                first_invalid_block=None,
                first_invalid_layer=None,
                early_terminated=False,
                fusion_count_b2=layerwise_horizon,
                fusion_count_b4=b4_count,
                fusion_count_b5=layerwise_horizon,
                terminal_final_config_fingerprint=str(
                    record.final_config_fingerprint
                ),
                terminal_materialization_failure_reason=str(
                    record.materialization_failure_reason
                ),
                terminal_model_uses_replan_config=bool(
                    record.model_uses_replan_config
                ),
                terminal_priority=int(record.priority),
                terminal_loss_mean=float(fresh_metrics.get("loss_mean", 0.0)),
                terminal_loss_std=float(fresh_metrics.get("loss_std", 0.0)),
                terminal_metric1_mean=float(fresh_metrics.get("metric1_mean", 0.0)),
                terminal_metric1_std=float(fresh_metrics.get("metric1_std", 0.0)),
                terminal_metric2_mean=float(fresh_metrics.get("metric2_mean", 0.0)),
                terminal_metric2_std=float(fresh_metrics.get("metric2_std", 0.0)),
                terminal_k_gain=13.0 - avg_k,
                terminal_fusion_gain=(
                    float(b4_count) / float(layerwise_horizon)
                ),
                terminal_cost_score=float(record.variable_cost),
                terminal_cost_rank_score=float(record.variable_cost),
                terminal_probe_wall_seconds=float(
                    probe_diagnostics.get("wall_seconds", 0.0) or 0.0
                ),
                terminal_probe_devices=[str(value) for value in (probe_diagnostics.get("devices") or [])],
                terminal_probe_trial_counts=[
                    int(value) for value in (
                        probe_diagnostics.get("per_worker_trial_counts") or []
                    )
                ],
                terminal_probe_trial_indices=[
                    [int(index) for index in (indices or [])]
                    for indices in (
                        probe_diagnostics.get("per_worker_trial_indices") or []
                    )
                ],
                terminal_probe_speedup=float(
                    probe_diagnostics.get("speedup_vs_sequential", 1.0) or 1.0
                ),
                terminal_cost_eval_wall_seconds=float(
                    probe_diagnostics.get("cost_eval_wall_seconds", 0.0) or 0.0
                ),
                terminal_probe_install_wall_seconds=float(
                    probe_diagnostics.get("probe_install_wall_seconds", 0.0) or 0.0
                ),
                terminal_probe_clear_wall_seconds=float(
                    probe_diagnostics.get("probe_clear_wall_seconds", 0.0) or 0.0
                ),
                terminal_probe_install_skipped=bool(probe_diagnostics.get(
                    "probe_install_skipped", False
                )),
                terminal_probe_clear_skipped=bool(probe_diagnostics.get(
                    "probe_clear_skipped", False
                )),
                protected_k1_enabled=bool(record.protected_k1_enabled),
                protected_k1_screened=bool(record.protected_k1_screened),
                protected_k1_audited=bool(record.protected_k1_audited),
                protected_k1_k1_only_reject=bool(
                    record.protected_k1_k1_only_reject
                ),
                protected_k1_audit_precision_false_negative=bool(
                    record.protected_k1_audit_precision_false_negative
                ),
                protected_k1_audit_p3_false_negative=bool(
                    record.protected_k1_audit_p3_false_negative
                ),
                protected_k1_reason=str(record.protected_k1_reason),
                protected_k1_guard_sigma=float(
                    record.protected_k1_guard_sigma
                ),
                protected_k1_worst_precision_z=(
                    record.protected_k1_worst_precision_z
                ),
                protected_k1_trials_executed=int(
                    record.protected_k1_trials_executed
                ),
                raw_trials=fresh_trials,
                constraint_probabilities=pooled_probabilities,
                fresh_trials=fresh_trials,
                pooled_trials=pooled_trials,
                fresh_metrics={str(k): float(v) for k, v in fresh_metrics.items()},
                pooled_metrics={str(k): float(v) for k, v in pooled_metrics.items()},
                fresh_constraint_probabilities=fresh_probabilities,
                pooled_constraint_probabilities=pooled_probabilities,
                fresh_trial_count=int(record.fresh_trial_count),
                pooled_trial_count=int(record.pooled_trial_count),
                reward_evidence=str(record.reward_evidence),
                ranking_evidence=str(record.ranking_evidence),
                constraint_thresholds={
                    **constraint_limits,
                    **probability_thresholds,
                },
                variable_cost=float(record.variable_cost),
                compute_saving=float(record.compute_saving),
                communication_saving=float(record.communication_saving),
                robust_floor=float(record.robust_floor),
                secondary_progress=float(record.secondary_progress),
                ppo_resource_score=float(record.ppo_resource_score),
                compute_shapley_credit=float(record.compute_shapley_credit),
                communication_shapley_credit=float(
                    record.communication_shapley_credit
                ),
                layer_resource_rewards=[
                    float(value) for value in record.layer_resource_rewards
                ],
                slot_resource_rewards=[
                    [float(value) for value in row]
                    for row in record.slot_resource_rewards
                ],
                layer_action_matrix=[list(row) for row in record.action_matrix],
                block4_entropy=record.block4_entropy,
                k_entropy=record.k_entropy,
                promotion_trial_count=int(record.promoted_trial_count),
                promotion_status=str(record.promotion_status),
                convergence_state=convergence_state,
            )
        diag_recorder.record_episode(
            episode_stats=episode_stats,
            full_action_vec=np.asarray(record.pending_full_vector, dtype=np.int64),
            is_new_best=is_new_best,
            best_reward_so_far=float(best_reward_so_far),
        )
        status.update_after_episode(
            int(record.episode_index) + 1,
            float(record.reward),
            {
                "priority": int(record.priority),
                "variable_cost": float(record.variable_cost),
                "compute_saving": float(record.compute_saving),
                "communication_saving": float(record.communication_saving),
                "robust_floor": float(record.robust_floor),
                "secondary_progress": float(record.secondary_progress),
                "ppo_resource_score": float(record.ppo_resource_score),
                "block4_entropy": record.block4_entropy,
                "k_entropy": record.k_entropy,
                **convergence_state,
            },
        )

    def on_layerwise_update(metrics: Mapping[str, Any], completed: int, record: Any) -> None:
        nonlocal ppo_update_counter, strict_best, strict_pareto_frontier
        nonlocal best_selection_key
        if int(completed) != completed_episode_count:
            raise RuntimeError(
                "layerwise PPO callback episode count mismatch: "
                f"expected={completed_episode_count}, received={completed}"
            )
        ppo_update_counter += 1
        if ppo_update_counter != diag_recorder.ppo_update_high_water + 1:
            raise RuntimeError(
                "layerwise PPO diagnostics identity mismatch: "
                f"high_water={diag_recorder.ppo_update_high_water}, "
                f"received={ppo_update_counter}"
            )
        strict_best = dict(metrics.get("strict_best") or {})
        strict_pareto_frontier = [
            dict(row) for row in metrics.get("strict_pareto_frontier", [])
        ]
        best_selection_key = strict_selection_key_from_snapshot(strict_best)
        write_strict_best_diagnostics(strict_best, episode=int(record.episode_index))
        recent = list(recent_episode_records)
        recent_rewards = [float(item.reward) for item in recent] or [0.0]
        update_stats = PPOUpdateStats(
            update=ppo_update_counter,
            completed_episodes=int(completed),
            policy_loss=float(metrics.get("policy_loss", 0.0)),
            value_loss=float(metrics.get("value_loss", 0.0)),
            entropy=float(metrics.get("entropy", 0.0)),
            clip_fraction=float(metrics.get("clip_fraction", 0.0)),
            n_samples=int(
                metrics.get("n_samples", len(recent) * layerwise_horizon)
            ),
            window_mean_return=float(np.mean(recent_rewards)),
            window_max_return=float(np.max(recent_rewards)),
            window_min_return=float(np.min(recent_rewards)),
            window_mean_invalid=float(np.mean([item.invalid_steps for item in recent])),
            best_reward_so_far=float(best_reward_so_far),
            elapsed_sec=float(time.time() - started_at),
            ent_coef=float(metrics.get("ent_coef", 0.0)),
            approx_kl=float(metrics.get("approx_kl", 0.0)),
            kl_early_stop=bool(metrics.get("kl_early_stop", False)),
            lr=float(metrics.get("lr", train_cfg.ppo.lr)),
            lr_scale=float(metrics.get("lr_scale", 1.0)),
            entropy_recovery_delta=float(
                metrics.get("entropy_recovery_delta", 0.0)
            ),
            nonfinite_minibatches=int(metrics.get("nonfinite_minibatches", 0) or 0),
            nonfinite_update_skipped=bool(
                metrics.get("nonfinite_update_skipped", False)
            ),
            convergence_update_counted=bool(
                metrics.get("convergence_update_counted", True)
            ),
            return_mean=float(metrics.get("return_mean", 0.0)),
            return_std=float(metrics.get("return_std", 1.0)),
            block4_entropy=metrics.get("block4_entropy"),
            k_entropy=metrics.get("k_entropy"),
            stall_update_windows=int(metrics.get("stall_update_windows", 0)),
            selected_action_identity=metrics.get("selected_action_identity"),
            selected_action_stable_update_windows=int(
                metrics.get("selected_action_stable_update_windows", 0)
            ),
            converged=bool(metrics.get("converged", False)),
            extension_required=bool(metrics.get("extension_required", False)),
            plateau_ready=bool(metrics.get("plateau_ready", False)),
            strict_revalidation_passed=bool(
                metrics.get("strict_revalidation_passed", False)
            ),
            strict_revalidation_status=str(
                metrics.get("strict_revalidation_status", "not_due")
            ),
            termination_reason=str(metrics.get("termination_reason", "running")),
            best_robust_feasible_cost=metrics.get("best_robust_feasible_cost"),
            best_robust_feasible_objective=(
                None
                if metrics.get("best_robust_feasible_objective") is None
                else [
                    float(value)
                    for value in metrics["best_robust_feasible_objective"]
                ]
            ),
            strict_pareto_frontier=[
                dict(row) for row in metrics.get("strict_pareto_frontier", [])
            ],
            actor_clip_mode=str(metrics.get("actor_clip_mode", "joint")),
            actor_credit_mode=str(metrics.get("actor_credit_mode", "scalar_gae")),
            entropy_objective_mode=str(
                metrics.get("entropy_objective_mode", "joint_sum")
            ),
            slot_labels=[str(value) for value in metrics.get("slot_labels", [])],
            entropy_per_slot=list(metrics.get("entropy_per_slot", [])),
            approx_kl_per_slot=list(metrics.get("approx_kl_per_slot", [])),
            clip_fraction_per_slot=list(metrics.get("clip_fraction_per_slot", [])),
            raw_advantage_mean_per_slot=list(
                metrics.get("raw_advantage_mean_per_slot", [])
            ),
            raw_advantage_std_per_slot=list(
                metrics.get("raw_advantage_std_per_slot", [])
            ),
            raw_advantage_snr_per_slot=list(
                metrics.get("raw_advantage_snr_per_slot", [])
            ),
            value_explained_variance_pre=metrics.get(
                "value_explained_variance_pre"
            ),
            value_explained_variance_post=metrics.get(
                "value_explained_variance_post"
            ),
            value_rmse_pre=metrics.get("value_rmse_pre"),
            value_rmse_post=metrics.get("value_rmse_post"),
            value_bias_pre=metrics.get("value_bias_pre"),
            value_bias_post=metrics.get("value_bias_post"),
            shared_grad_parameter_count=int(
                metrics.get("shared_grad_parameter_count", 0) or 0
            ),
            actor_shared_grad_norm=metrics.get("actor_shared_grad_norm"),
            critic_shared_grad_norm=metrics.get("critic_shared_grad_norm"),
            actor_critic_shared_grad_cosine=metrics.get(
                "actor_critic_shared_grad_cosine"
            ),
            preclip_grad_norm_mean=metrics.get("preclip_grad_norm_mean"),
            preclip_grad_norm_max=metrics.get("preclip_grad_norm_max"),
        )
        diag_recorder.record_ppo_update(update_stats)
        save_layerwise_checkpoint(
            completed=int(completed),
            strict_best=strict_best,
            convergence_state=metrics.get("convergence_state"),
        )
        shared_probe_runner = getattr(
            base_env,
            "_shared_probe_runner_owner",
            None,
        )
        if shared_probe_runner is not None:
            deferred_gpu_failure = (
                shared_probe_runner.pop_deferred_gpu_failure()
            )
            if deferred_gpu_failure is not None:
                raise deferred_gpu_failure
        raise_if_elastic_gpu_restart_requested()
        status.update_after_ppo_update(
            int(ppo_update_counter),
            {
                "completed_episodes": int(completed),
                "policy_loss": float(update_stats.policy_loss),
                "value_loss": float(update_stats.value_loss),
                "entropy": float(update_stats.entropy),
                "clip_fraction": float(update_stats.clip_fraction),
                "ent_coef": update_stats.ent_coef,
                "approx_kl": float(update_stats.approx_kl),
                "kl_early_stop": update_stats.kl_early_stop,
                "lr": update_stats.lr,
                "lr_scale": update_stats.lr_scale,
                "entropy_recovery_delta": update_stats.entropy_recovery_delta,
                "nonfinite_minibatches": update_stats.nonfinite_minibatches,
                "nonfinite_update_skipped": update_stats.nonfinite_update_skipped,
                "convergence_update_counted": update_stats.convergence_update_counted,
                "return_mean": update_stats.return_mean,
                "return_std": update_stats.return_std,
                "value_explained_variance_post": (
                    update_stats.value_explained_variance_post
                ),
                "value_rmse_post": update_stats.value_rmse_post,
                "actor_critic_shared_grad_cosine": (
                    update_stats.actor_critic_shared_grad_cosine
                ),
                "preclip_grad_norm_mean": update_stats.preclip_grad_norm_mean,
                "entropy_per_slot": update_stats.entropy_per_slot,
                "approx_kl_per_slot": update_stats.approx_kl_per_slot,
                "clip_fraction_per_slot": update_stats.clip_fraction_per_slot,
                "window_mean_return": float(update_stats.window_mean_return),
                "window_max_return": float(update_stats.window_max_return),
                "window_min_return": float(update_stats.window_min_return),
                "window_mean_invalid": float(update_stats.window_mean_invalid),
                "block4_entropy": update_stats.block4_entropy,
                "k_entropy": update_stats.k_entropy,
                "stall_update_windows": int(update_stats.stall_update_windows),
                "selected_action_identity": update_stats.selected_action_identity,
                "selected_action_stable_update_windows": int(
                    update_stats.selected_action_stable_update_windows
                ),
                "converged": bool(update_stats.converged),
                "extension_required": bool(update_stats.extension_required),
                "plateau_ready": bool(update_stats.plateau_ready),
                "strict_revalidation_passed": bool(
                    update_stats.strict_revalidation_passed
                ),
                "strict_revalidation_status": (
                    update_stats.strict_revalidation_status
                ),
                "termination_reason": update_stats.termination_reason,
                "best_robust_feasible_cost": update_stats.best_robust_feasible_cost,
                "best_robust_feasible_objective": (
                    update_stats.best_robust_feasible_objective
                ),
                "strict_pareto_frontier": update_stats.strict_pareto_frontier,
                "actor_clip_mode": update_stats.actor_clip_mode,
                "actor_credit_mode": update_stats.actor_credit_mode,
                "entropy_objective_mode": update_stats.entropy_objective_mode,
            },
        )
        if strict_best.get("reward") is not None and strict_best.get("full_vector"):
            best_full_vector = [int(value) for value in strict_best["full_vector"]]
            current_full_vector = [
                int(value) for value in record.pending_full_vector
            ]
            status.set_best(
                best_reward=float(strict_best["reward"]),
                best_action_vec=best_full_vector,
                best_breakdown={
                    "priority": 3,
                    "variable_cost": strict_best.get("variable_cost"),
                    "resource_objective": {
                        field_name: strict_best.get(field_name)
                        for field_name in (
                            "compute_saving",
                            "communication_saving",
                            "robust_floor",
                            "secondary_progress",
                            "ppo_resource_score",
                            "compute_shapley_credit",
                            "communication_shapley_credit",
                        )
                    },
                    "action_matrix": strict_best.get("action_matrix"),
                    "assessment": strict_best.get("assessment"),
                    "metrics": strict_best.get("metrics"),
                },
                best_episode=(
                    int(record.episode_index) + 1
                    if current_full_vector == best_full_vector else None
                ),
            )
        if int(completed) % max(1, int(train_cfg.save_interval)) == 0:
            diag_recorder.flush_periodic()
    status.set_phase(
        f"PPO training ({layerwise_horizon}-step layerwise robust)"
    )
    training_completed = False
    completion_status = "failed"
    summary: Dict[str, Any]
    try:
        summary = train_layerwise(
            env=layerwise_env,
            promotion_base_env=promotion_base_env,
            validation_banks=authoritative_validation_banks,
            policy=policy,
            train_cfg=layerwise_train_cfg,
            candidate_store=candidate_store,
            identity_context=identity_context,
            device=device,
            optimizer=optimizer,
            on_episode_end=on_layerwise_episode,
            on_ppo_update_end=on_layerwise_update,
            retain_history=False,
        )
        summary_completed_episodes = int(summary.get(
            "completed_episodes", completed_episode_count,
        ))
        if summary_completed_episodes != completed_episode_count:
            raise RuntimeError(
                "layerwise training summary episode count mismatch: "
                f"callbacks={completed_episode_count}, "
                f"summary={summary_completed_episodes}"
            )
        if (
                diag_recorder.episode_count != completed_episode_count
                or diag_recorder.ppo_update_count != ppo_update_counter
        ):
            raise RuntimeError(
                "layerwise diagnostics cumulative count mismatch: "
                f"episodes={diag_recorder.episode_count}/{completed_episode_count}, "
                f"updates={diag_recorder.ppo_update_count}/{ppo_update_counter}"
            )
        strict_best = dict(summary.get("strict_best") or {})
        strict_pareto_frontier = [
            dict(row) for row in summary.get("strict_pareto_frontier", [])
        ]
        write_strict_best_diagnostics(
            strict_best,
            episode=int(completed_episode_count),
        )
        save_layerwise_checkpoint(
            completed=int(completed_episode_count),
            strict_best=summary.get("strict_best"),
            convergence_state=summary.get("convergence_state"),
        )
        if summary.get("converged", False):
            completion_status = "converged"
        elif requested_total_episodes > 0:
            completion_status = "max_episodes_reached"
        else:
            raise RuntimeError(
                "unbounded layerwise training stopped without strict convergence"
            )
        training_completed = True
    except ElasticGPUFailure:
        raise
    except Exception as exc:
        if not is_recoverable_gpu_failure(exc):
            raise
        raise ElasticGPUFailure(
            device="cuda:0",
            role="learner-primary",
            operation="stage2_layerwise_training",
            cause=exc,
        ) from exc
    finally:
        try:
            if not training_completed:
                status.set_phase("failed")
            run_manifest.update({
                "status": completion_status,
                "completed_episodes": int(completed_episode_count),
                "ppo_update_count": int(ppo_update_counter),
                "best_resource_objective": (
                    None
                    if not strict_best
                    else {
                        field_name: copy.deepcopy(strict_best.get(field_name))
                        for field_name in (
                            "compute_saving",
                            "communication_saving",
                            "robust_floor",
                            "secondary_progress",
                            "ppo_resource_score",
                        )
                    }
                ),
                "strict_pareto_frontier": copy.deepcopy(
                    strict_pareto_frontier
                ),
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
            write_strict_json_file(layerwise_manifest_path, run_manifest)
        finally:
            try:
                diag_recorder.finalize(status=completion_status)
            finally:
                try:
                    base_env.clear_installed_blb()
                finally:
                    try:
                        for restore_name in (
                                "restore_layer_block5_noise", "restore_layer_block4_noise",
                                "restore_layer_block3_noise", "restore_layer_block2_noise",
                                "restore_layer_block1_noise", "restore_blb_first_input_noise",
                        ):
                            method = getattr(
                                evaluator.reversible_handler, restore_name, None,
                            )
                            if method is None:
                                continue
                            try:
                                method(layer_indices=list(range(evaluator.total_layers)))
                            except Exception:
                                pass
                        evaluator.apply_configuration(fixed_gelu, fixed_softmax)
                    finally:
                        shared_owner = getattr(
                            base_env, "_shared_probe_runner_owner", None,
                        )
                        if shared_owner is not None:
                            shared_owner.close()
                        else:
                            promotion_runner = getattr(
                                promotion_base_env, "probe_runner", None,
                            )
                            if (
                                    promotion_runner is not None
                                    and promotion_runner is not getattr(
                                        base_env, "probe_runner", None,
                                    )
                            ):
                                promotion_runner.close()
    status.set_phase(completion_status)

    bank_b_best = dict(summary.get("bank_b_best") or {})
    compact_summary = {
        "schema_version": "stage2_layerwise_robust_summary_v5",
        "status": completion_status,
        "rl_variant": rl_variant,
        "policy_network_variant": policy_network_variant,
        "policy_network": policy_network_summary,
        "algorithm_revision": algorithm_revision,
        "algorithm_contract_hash": algorithm_contract_hash,
        "run_context_hash": run_context_hash,
        "completed_episodes": int(summary.get("completed_episodes", start_episode)),
        "best_action_matrix": summary.get("best_action_matrix"),
        "best_full_vector": summary.get("best_full_vector"),
        "best_assessment": summary.get("best_assessment"),
        "strict_best_assessment": summary.get("best_assessment"),
        "best_metrics": summary.get("best_metrics"),
        "best_resource_objective": summary.get("best_resource_objective"),
        "strict_pareto_frontier": summary.get("strict_pareto_frontier", []),
        # Read-only compatibility alias for report consumers predating v4.
        "best_variable_cost": summary.get("best_variable_cost"),
        "best_reward": summary.get("best_reward"),
        "best_promotion_evidence": summary.get("best_promotion_evidence"),
        "bank_b_best": bank_b_best or None,
        "protected_k1": dict(summary.get("protected_k1") or {}),
        "final_evidence": {
            "status": (
                "strict_revalidation_passed"
                if summary.get("strict_revalidation_passed", False)
                else "bank_b_confirmed_not_final_certified"
                if bank_b_best
                else "no_candidate"
            ),
            "diagnostic_probability": float(
                getattr(train_cfg, "final_constraint_probability", 0.95)
            ),
            "hard_gate": "six_point_constraints",
            "bank_a_trial_count": int(
                authoritative_validation_banks.bank_a.trial_count
            ),
            "bank_b_trial_count": int(
                authoritative_validation_banks.bank_b.trial_count
            ),
            "bank_c_trial_count": int(
                authoritative_validation_banks.bank_c.trial_count
            ),
            "pooled_final_trial_count": int(
                authoritative_validation_banks.final_trial_count
            ),
            "current_assessment": (
                summary.get("best_assessment")
                or bank_b_best.get("assessment")
            ),
            "note": (
                "Bank A qualifies a candidate, independent Bank B confirms "
                "the pooled AB point gate, and held-out Bank C certifies the "
                "pooled ABC point gate; probabilities are diagnostics only."
            ),
        },
        "block4_entropy": summary.get("block4_entropy"),
        "k_entropy": summary.get("k_entropy"),
        "stall_update_windows": summary.get("stall_update_windows"),
        "selected_action_identity": summary.get("selected_action_identity"),
        "selected_action_stable_update_windows": summary.get(
            "selected_action_stable_update_windows"
        ),
        "converged": bool(summary.get("converged", False)),
        "extension_required": bool(summary.get("extension_required", False)),
        "plateau_ready": bool(summary.get("plateau_ready", False)),
        "strict_revalidation_passed": bool(
            summary.get("strict_revalidation_passed", False)
        ),
        "strict_revalidation_status": str(
            summary.get("strict_revalidation_status", "not_due")
        ),
        "recommended_extension_episodes": int(
            summary.get("recommended_extension_episodes", 0) or 0
        ),
        "entropy_regularization": layerwise_entropy_regularization,
        "termination": layerwise_termination,
        "termination_reason": str(
            summary.get("termination_reason") or completion_status
        ),
        "evidence_tiers": algorithm_contract["evidence_tiers"],
        "constraint_probability_thresholds": probability_thresholds,
        "constraint_limits": constraint_limits,
        "baseline_reference": dict(baseline_preflight_metrics),
        "ppo_update_count": int(ppo_update_counter),
        "candidate_store": candidate_store.path,
        "checkpoint": save_path,
        "structured_data_dir": stage2_data_writer.run_dir,
    }
    compact_summary = to_jsonable(compact_summary, stringify_unknown=True)
    write_strict_json_file(
        os.path.join(blb_progress_dir, "layerwise_summary.json"),
        compact_summary,
    )
    stage2_data_writer.write_summary(compact_summary)
    stage2_data_writer.close()

    curve_series = {
        "returns": [], "loss": [], "metric1": [], "metric2": [],
        "fusion": [], "avg_k": [], "entropy": [], "entropy_episode": [],
    }
    if os.path.isfile(existing_episode_path):
        for row in iter_jsonl(existing_episode_path, errors="raise"):
            curve_series["returns"].append(float(row.get("total_reward", 0.0)))
            curve_series["loss"].append(float(row.get("terminal_loss_mean", 0.0)))
            curve_series["metric1"].append(float(row.get("terminal_metric1_mean", 0.0)))
            curve_series["metric2"].append(float(row.get("terminal_metric2_mean", 0.0)))
            curve_series["fusion"].append(int(row.get("fusion_count", 0)))
            curve_series["avg_k"].append(
                13.0 - float(row.get("terminal_k_gain", 0.0))
            )
    if os.path.isfile(existing_update_path):
        for row in iter_jsonl(existing_update_path, errors="raise"):
            curve_series["entropy"].append(float(row.get("entropy", 0.0)))
            curve_series["entropy_episode"].append(
                int(row.get("completed_episodes", 0))
            )
    if curve_series["returns"]:
        write_training_curves(
            blb_progress_dir,
            episode_returns=curve_series["returns"],
            episode_losses=curve_series["loss"],
            episode_metric1s=curve_series["metric1"],
            episode_metric2s=curve_series["metric2"],
            episode_fusion_counts=curve_series["fusion"],
            episode_avg_ks=curve_series["avg_k"],
            baselines={
                "loss": float(robust_reference.loss_mean),
                "metric1": float(robust_reference.metric1_mean),
                "metric2": float(robust_reference.metric2_mean),
                "avg_k": 13.0,
            },
            entropy_series=curve_series["entropy"],
            entropy_episodes=curve_series["entropy_episode"],
            metric1_name="metric1",
            metric2_name="metric2",
            log_fn=log,
        )

    best_full_vector = summary.get("best_full_vector")
    best_action_matrix = summary.get("best_action_matrix")
    best_action_group = build_reloadable_best_group({
        "full_vector": best_full_vector,
        "action_matrix": best_action_matrix,
        "boosted_overrides": summary.get("best_boosted_overrides") or {},
    }) if best_full_vector is not None else None
    cost_reference_noise_config = evaluator._get_max_noise_configuration()
    cost_reference_tot_c, _ = evaluator.get_noise_simulated_cost(
        **cost_reference_noise_config
    )
    legacy_best = _build_legacy_compatible_best_noise_config(evaluator)
    best_reward = summary.get("best_reward")
    if best_reward is None:
        best_reward = -float("inf")
    limits = {
        "loss": float(robust_reference.loss_limit),
        "metric1": float(robust_reference.metric1_limit),
        "metric2": float(robust_reference.metric2_limit),
        "loss_std": float(robust_reference.loss_std_limit),
        "metric1_std": float(robust_reference.metric1_std_limit),
        "metric2_std": float(robust_reference.metric2_std_limit),
    }
    return {
        "fixed_gelu": fixed_gelu.copy(),
        "fixed_softmax": fixed_softmax.copy(),
        "baseline_noise_config": {
            key: value.copy() for key, value in cost_reference_noise_config.items()
        },
        "baseline_tot_c": float(cost_reference_tot_c),
        "best_noise_config": {key: value.copy() for key, value in legacy_best.items()},
        "stable_search_best_noise_config": {
            key: value.copy() for key, value in legacy_best.items()
        },
        "stable_joint_best_noise_config": {
            key: value.copy() for key, value in legacy_best.items()
        },
        "limit_loss": limits["loss"],
        "limit_p": limits["metric1"],
        "limit_s": limits["metric2"],
        "proxy_limit_loss": limits["loss"],
        "proxy_limit_p": limits["metric1"],
        "proxy_limit_s": limits["metric2"],
        "search_limits": limits,
        "all_max_blb_baseline_metrics": dict(baseline_preflight_metrics),
        "status": completion_status,
        "blb_v3_best_action_vec": best_full_vector,
        "blb_v3_best_action_group": best_action_group,
        "blb_v3_layerwise_best_action_group": best_action_group,
        "blb_v3_best_reward": float(best_reward),
        "blb_v3_profile": str(train_cfg.profile),
        "blb_v3_fusion_count_action": True,
        "blb_v3_total_episodes": int(summary.get("completed_episodes", 0)),
        "rl_variant": rl_variant,
        "policy_network_variant": policy_network_variant,
        "policy_network": policy_network_summary,
        "algorithm_revision": algorithm_revision,
        "algorithm_contract_hash": algorithm_contract_hash,
        "run_context_hash": run_context_hash,
        "selection_diagnostics": {
            "selection_mode": "layerwise_robust_dual_resource_strict",
            "best_action_vec": best_full_vector,
            "best_action_matrix": best_action_matrix,
            "best_assessment": summary.get("best_assessment"),
            "best_metrics": summary.get("best_metrics"),
            "best_resource_objective": summary.get("best_resource_objective"),
            "strict_pareto_frontier": summary.get(
                "strict_pareto_frontier", []
            ),
            # Read-only compatibility alias for report consumers predating v4.
            "best_variable_cost": summary.get("best_variable_cost"),
            "best_promotion_evidence": summary.get("best_promotion_evidence"),
            "final_evidence": compact_summary["final_evidence"],
        },
        "sequential_diagnostics": {
            "horizon": layerwise_horizon,
            "max_step_dim": 6,
            "state_dim": int(layerwise_env.state_dim),
            "episode_count": int(summary.get(
                "completed_episodes", completed_episode_count,
            )),
            "ppo_metric_count": int(ppo_update_counter),
            "block4_entropy": summary.get("block4_entropy"),
            "k_entropy": summary.get("k_entropy"),
            "selected_action_identity": summary.get("selected_action_identity"),
            "selected_action_stable_update_windows": summary.get(
                "selected_action_stable_update_windows"
            ),
            "converged": bool(summary.get("converged", False)),
            "extension_required": bool(summary.get("extension_required", False)),
            "plateau_ready": bool(summary.get("plateau_ready", False)),
            "strict_revalidation_passed": bool(
                summary.get("strict_revalidation_passed", False)
            ),
            "strict_revalidation_status": str(
                summary.get("strict_revalidation_status", "not_due")
            ),
            "termination_reason": str(
                summary.get("termination_reason") or completion_status
            ),
        },
        "layerwise_summary": summary,
    }


class _ProbeRunnerOwnerHolder:
    """Own one shared probe pool across every Stage-2 exit path."""

    def __init__(self) -> None:
        self._owner: Optional[Any] = None
        self._closed = False

    def bind(self, owner: Any) -> None:
        if owner is None:
            raise ValueError("probe runner owner must not be None")
        if self._closed:
            raise RuntimeError("probe runner owner holder is already closed")
        if self._owner is None:
            self._owner = owner
            return
        if self._owner is not owner:
            raise RuntimeError("probe runner owner holder is already bound")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owner is not None:
            self._owner.close()


def run_sequential_via_runner(
        *,
        runner,
        train_cfg,
        fixed_gelu,
        fixed_softmax,
        fixed_label,
        fixed_source,
        resume_checkpoint_path=None,
        ) -> Dict[str, Any]:
    """Lock the complete Stage-2 run before any probe or persistent write."""
    from .layerwise_runner import LayerwiseRunLock
    from .runner import resolve_blb_persistence_dir

    blb_progress_dir = resolve_blb_persistence_dir(runner.evaluator)
    probe_runner_owner_holder = _ProbeRunnerOwnerHolder()
    with LayerwiseRunLock(blb_progress_dir) as run_lock:
        try:
            return _run_sequential_via_runner_locked(
                runner=runner,
                train_cfg=train_cfg,
                fixed_gelu=fixed_gelu,
                fixed_softmax=fixed_softmax,
                fixed_label=fixed_label,
                fixed_source=fixed_source,
                resume_checkpoint_path=resume_checkpoint_path,
                run_lock=run_lock,
                probe_runner_owner_holder=probe_runner_owner_holder,
            )
        finally:
            probe_runner_owner_holder.close()


def _run_sequential_via_runner_locked(
        *,
        runner,                           # BLBStage2RLRunner (avoid circular import)
        train_cfg,                        # BLBStage2TrainConfig
        fixed_gelu,
        fixed_softmax,
        fixed_label,
        fixed_source,
        resume_checkpoint_path=None,
        run_lock: Any,
        probe_runner_owner_holder: _ProbeRunnerOwnerHolder,
        ) -> Dict[str, Any]:
    """Drive the sequential RL pipeline using BLBStage2RLRunner's setup helpers.

    Reuses ``runner._build_probe_batches``, ``runner._build_rescale_bridge``,
    ``runner._estimate_baseline_metrics``, and ``resolve_blb_persistence_dir``
    so the env / model / persistence / baseline-cost story is identical to the
    legacy single-shot path. The rollout loop is replaced with
    :func:`train_sequential` and a small bookkeeping shell saves a curve / a
    final-report markdown / a checkpoint in the same persistent directory.

    Returns a noise_stage_result dict matching the keys downstream consumers
    (UnifiedFinalEvaluationModule, BLBActionFinalEvaluationModule) read from
    the single-shot path: ``blb_v3_best_action_vec``, ``blb_v3_profile``,
    ``best_noise_config`` (legacy-compat all-max), ``limit_loss`` /
    ``limit_p`` / ``limit_s``, ``baseline_tot_c``.
    """
    import pickle
    train_cfg.rl_algo = _normalize_supported_rl_algo(
        getattr(train_cfg, "rl_algo", "ppo"), context="BLBStage2TrainConfig.rl_algo"
    )
    train_cfg.grpo_kl_beta = 0.0

    from .baseline_bootstrap import (
        load_calibrated_stage2_action_context,
        validate_calibrated_stage2_action_context,
    )
    from .diagnostics import (
        EpisodeStats,
        PPOUpdateStats,
        RLDiagnosticsRecorder,
    )
    from .env import BLBStage2Env, BLBStage2EnvConfig
    from .reward import BaselineCostStats, ParetoCostArchive, RewardWeights
    from .persistence import (
        BLBRewardCrashWatcher,
        BLBStatusBoard,
        BLBStepDetailsWriter,
        write_diagnostic_curves,
        write_training_curves,
    )
    from .runner import (
        _build_legacy_compatible_best_noise_config,
        _selection_float,
        resolve_blb_persistence_dir,
    )
    from .action_space import action_dims_for_config, describe_action_vector
    from .action_io import action_vec_to_slots_list
    from .probe_runner import enable_cuda_reward_probe_fast_math

    # Keep baseline and reward-probe kernels in the same mode for every GPU count.
    enable_cuda_reward_probe_fast_math()
    ev = runner.evaluator
    stage2_model_type = resolve_stage2_model_type(
        str(getattr(ev, "model_type", "") or ""),
        num_layers=int(ev.total_layers),
    )
    robust_mode = (
        str(getattr(train_cfg, "reward_design", "")).strip().lower()
        == "robust_constrained"
    )
    from .layerwise_runner import resolve_decision_path
    decision_path = resolve_decision_path(
        fusion_count_action=bool(getattr(train_cfg, "fusion_count_action", False)),
        decision_granularity=str(
            getattr(train_cfg, "decision_granularity", "block")
        ),
        reward_design=str(getattr(train_cfg, "reward_design", "stage1_aligned")),
    )
    bullet = "*"
    log = runner._make_log_safe(ev.log)
    active_rl_mode = (
        "layerwise_robust" if decision_path == "layerwise"
        else "sequential_per_block"
    )

    # ---------- 0.1) Persistent dir ----------
    legacy_progress_dir = str(getattr(ev, "noise_stage_progress_dir", "") or "")
    blb_progress_dir = resolve_blb_persistence_dir(ev)
    try:
        ev.noise_stage_progress_dir = blb_progress_dir
    except Exception:
        pass

    _seq_log_major_rule(
        log,
        (
            "阶段 5 · 二阶段噪声强化学习"
            f"（BLB v3 · {int(ev.total_layers)}-step layerwise robust）"
            if decision_path == "layerwise"
            else "阶段 5 · 二阶段噪声强化学习（BLB v3 · per-block sequential）"
        ),
    )
    log(
        f"  {bullet} 模式（mode）："
        + (
            f"horizon={int(ev.total_layers)} layerwise，max_step_dim=6"
            if decision_path == "layerwise"
            else "horizon=59 per-block sequential"
        )
    )
    log(f"  {bullet} 固定 GELU/Softmax 来源（source）：{fixed_source}    标签（label）：{fixed_label}")
    log(f"  {bullet} GELU 离散阶数向量:   {np.asarray(fixed_gelu, dtype=int).tolist()}")
    log(f"  {bullet} Softmax 离散阶数向量: {np.asarray(fixed_softmax, dtype=int).tolist()}")
    log(
        f"  {bullet} 训练概览：profile={train_cfg.profile!r}    "
        f"total_episodes={train_cfg.total_episodes}    "
        f"PPO 更新间隔（rollout_size）= {max(1, int(train_cfg.rollout_size))}    "
        f"seed={int(train_cfg.seed)}"
    )
    log(f"  {bullet} BLB 持久化目录：{blb_progress_dir}")

    run_basename = os.path.basename(os.path.normpath(str(getattr(ev, "run_output_dir", "") or ""))) \
        or "blb_stage2_default_run"
    status = BLBStatusBoard(
        blb_progress_dir,
        total_episodes=int(train_cfg.total_episodes),
        profile=str(train_cfg.profile),
        run_basename=run_basename,
        extra_meta={
            "fixed_label": str(fixed_label),
            "fixed_source": str(fixed_source),
            "rl_mode": active_rl_mode,
            "rescale_optimizer": "in_process_real",
            "rescale_optimizer_root": str(train_cfg.inproc_rescale_optimizer_root),
        },
        log_fn=log,
    )
    status.set_phase("装载 stage1 GELU/Softmax 多项式近似")

    # Stage2 root (parent of progress/) — host of details/ + warning.txt so the
    # layout matches what legacy noise_rl_module_v2 produced. The legacy single-
    # shot BLBStage2RLRunner.run() already wires these into its loop; the
    # sequential path missed them before 2026-05-17 and the user noticed.
    blb_stage2_root = os.path.dirname(os.path.normpath(blb_progress_dir))
    details_batch_size = max(int(train_cfg.rollout_size) * 3, 360)
    details_writer = BLBStepDetailsWriter(
        blb_stage2_root,
        batch_size=details_batch_size,
        log_fn=log,
    )
    crash_watcher = BLBRewardCrashWatcher(
        blb_stage2_root,
        drop_threshold=0.3,
        log_fn=log,
    )
    log(
        f"  {bullet} 详细诊断：{os.path.join(blb_stage2_root, 'details')}/ "
        f"（每 {details_batch_size} 回合一文件，记录每回合错误/动作变化）"
    )
    log(
        f"  {bullet} 奖励暴跌警告：{os.path.join(blb_stage2_root, 'warning.txt')} "
        f"（PPO rollout 平均奖励较上一次跌幅 > {crash_watcher._drop_threshold:.2f} 时记录）"
    )

    # ---------- 1) apply stage1 polynomial degrees ----------
    fixed_gelu = np.asarray(fixed_gelu, dtype=int)
    fixed_softmax = np.asarray(fixed_softmax, dtype=int)
    ev.apply_configuration(fixed_gelu, fixed_softmax)
    try:
        ev.reversible_handler.restore_layer_input_noise(
            layer_indices=list(range(ev.total_layers)),
        )
    except Exception:
        pass

    # ---------- 2) probe + bridge + baseline ----------
    probe_batches = runner._build_probe_batches(
        ev,
        train_cfg,
        probe_size_override=(256 if decision_path == "layerwise" else None),
    )
    train_cfg.probe_batch_count = max(1, int(len(probe_batches) or train_cfg.probe_batch_count))
    log(f"  {bullet} 评估子集：batch 数 = {len(probe_batches)}")

    rescale_bridge = runner._build_rescale_bridge(train_cfg, log=log)

    calibrated_action_context = load_calibrated_stage2_action_context(
        rescale_optimizer_root=str(train_cfg.inproc_rescale_optimizer_root),
        dataset=str(train_cfg.profile),
        num_layers=int(ev.total_layers),
        gelu_per_layer=[int(x) for x in fixed_gelu.reshape(-1)],
        softmax_per_layer=[int(x) for x in fixed_softmax.reshape(-1)],
        snap_sf_to_noise_table=False,
    )
    validate_calibrated_stage2_action_context(
        calibrated_action_context,
        dataset=str(train_cfg.profile),
        num_layers=int(ev.total_layers),
        gelu_per_layer=[int(x) for x in fixed_gelu.reshape(-1)],
        softmax_per_layer=[int(x) for x in fixed_softmax.reshape(-1)],
        snap_sf_to_noise_table=False,
    )
    ss_baseline_obj = calibrated_action_context.baseline
    ss_action_vec = calibrated_action_context.baseline_action_vec
    max_sfs = calibrated_action_context.max_sfs
    ss_cost_stats = calibrated_action_context.cost_stats
    _ss_diag = calibrated_action_context.diagnostics
    baseline_action_vec = np.asarray(ss_action_vec, dtype=np.int64).reshape(-1)
    log(
        f"  {bullet} calibrated static_skeletons baseline loaded from "
        f"{ss_baseline_obj.archive_path} "
        f"(sha256={calibrated_action_context.provenance['archive_sha256']})"
    )

    # ---------- 3) base env ----------
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
        gelu_degree=fixed_gelu,
        attn_degree=fixed_softmax,
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
            borderline_retest_enabled=False,
            borderline_retest_trials_multiplier=1,
        ),
    )
    base_env.pareto_cost_archive = None
    base_env.sync_degree_vectors_from_model()

    # ---------- 3.5) Multi-GPU reward-probe runner (opt-in) ----------
    reward_devices = list(getattr(train_cfg, "reward_devices", []) or [])
    if reward_devices and len(reward_devices) >= 2:
        from .probe_runner import build_probe_runner
        log(f"  [multi-gpu] reward probe enabled: devices={reward_devices}")
        shared_probe_runner_owner = build_probe_runner(
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
        probe_runner_owner_holder.bind(shared_probe_runner_owner)
        base_env._shared_probe_runner_owner = shared_probe_runner_owner
        base_env._shared_probe_batch_sets = {
            "F1": tuple(base_env.probe_batches),
        }
        base_env.probe_runner = shared_probe_runner_owner.view("F1")

    # ---------- 4) baseline cost / reward weights ----------
    from .env import estimate_baseline_cost_stats
    from .reward import calibrate_weights_from_baseline
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

    # baseline accuracy/stability (CLEAN model — used for the cost-side
    # baseline metric1 reference; loss_std here is 0 since no noise is installed
    # and we use a deterministic forward path. We deliberately do NOT use this
    # value to set stab_threshold — see noisy preflight below.)
    baseline_metrics = runner._estimate_baseline_metrics(base_env)
    baseline.loss_mean = float(baseline_metrics.loss_mean)
    baseline.loss_std = float(baseline_metrics.loss_std)
    baseline.metric1_mean = float(baseline_metrics.metric1_mean)
    baseline.metric2_mean = float(baseline_metrics.metric2_mean)
    # v3 stability path: copy per-trial stds for m1 / m2 too — combined_stab_excess
    # in compute_reward needs baseline.metric{1,2}_std to derive the per-channel
    # thresholds and normalize the excess. Clean preflight stds are typically 0
    # (deterministic forward), so the noisy preflight below also writes them.
    baseline.metric1_std = float(getattr(baseline_metrics, "metric1_std", 0.0) or 0.0)
    baseline.metric2_std = float(getattr(baseline_metrics, "metric2_std", 0.0) or 0.0)
    baseline_clean_metric1 = float(baseline_metrics.metric1_mean)
    baseline_clean_metric2 = float(baseline_metrics.metric2_mean)

    # Adaptive scalar cost uses structural normalizers. Fusion and K/truncation
    # get interval bonuses; total_bits stays a weak linear term.
    baseline.typical_bits_drop = float(
        max(baseline.total_bits_sum / max(int(base_env.num_layers), 1), 1.0)
    )
    baseline.typical_fusion_count = float(base_env.num_layers)
    baseline.typical_k_drop = 5.0

    # Now baseline is fully populated; calibrate reward weights (v2-style
    # `calibrate_weights_from_baseline` writes baseline_metric1 into the
    # weights so margin_acc has the right denominator).
    weights = calibrate_weights_from_baseline(baseline)
    base_env.reward_weights = weights

    # ---------- 4.5) NOISY baseline preflight: calibrate acc/stab gates ----------
    # Before this preflight the sequential path used to derive
    #   stab_threshold = baseline.loss_std * 1.5 + 1e-3
    # but baseline.loss_std comes from the *clean* model (no BLB noise installed,
    # K trials produce identical losses → std = 0), so the threshold ended up at
    # 0.001 — below the per-trial noise floor of every real candidate. Every
    # episode would then trip priority-2, the reward fell into the inf-fallback
    # branch (terminal_reward = -priority2_penalty - 1.0 * priority2_scale = -150
    # exactly), and PPO got essentially zero gradient signal across action space.
    # See diagnostics/diagnostics_summary.md from the s1t0.005 run for evidence.
    #
    # Fix: install the all-max baseline action with real BLB noise, run K trials,
    # and read the *noisy* probe metrics. Calibrate the gates from these so
    # candidates can be ranked by accuracy/stability deltas rather than all
    # collapsing into the same fallback.
    noisy_baseline_metric1 = baseline_clean_metric1
    noisy_baseline_metric2 = baseline_clean_metric2
    noisy_baseline_loss_std = 0.0
    noisy_baseline_metric1_std = 0.0
    noisy_baseline_metric2_std = 0.0
    noisy_baseline_loss_mean = float(baseline.loss_mean)
    preflight_ok = False
    # Episode-parallel deterministic mode: key the preflight probe noise too
    # (reserved pseudo-episode -1) so the calibrated acc/stab thresholds are
    # identical for any GPU count and across reruns. Legacy mode (flag unset)
    # keeps the true-random preflight bit-for-bit.
    def run_legacy_preflight() -> None:
        nonlocal noisy_baseline_metric1
        nonlocal noisy_baseline_metric2
        nonlocal noisy_baseline_loss_std
        nonlocal noisy_baseline_metric1_std
        nonlocal noisy_baseline_metric2_std
        nonlocal noisy_baseline_loss_mean
        nonlocal preflight_ok
        if list(getattr(train_cfg, "stage2_rl_devices", []) or []):
            from .seed_utils import PREFLIGHT_EPISODE, derive_probe_seed
            base_env.probe_noise_seed = derive_probe_seed(
                int(getattr(train_cfg, "seed", 42) or 42), PREFLIGHT_EPISODE,
            )
            log(
                f"  {bullet} [stage2-parallel] deterministic preflight probe seed = "
                f"{base_env.probe_noise_seed}"
            )
        try:
            base_env.reset(seed=int(train_cfg.seed))
            _, _preflight_reward, _, preflight_info = base_env.step(baseline_action_vec)
            noisy_metrics = preflight_info.get("metrics")
            if noisy_metrics is not None:
                noisy_baseline_metric1 = float(getattr(noisy_metrics, "metric1_mean", baseline_clean_metric1))
                noisy_baseline_metric2 = float(getattr(noisy_metrics, "metric2_mean", baseline_clean_metric2))
                raw_std = float(getattr(noisy_metrics, "loss_std", 0.0))
                noisy_baseline_loss_std = raw_std if np.isfinite(raw_std) else 0.0
                raw_m1_std = float(getattr(noisy_metrics, "metric1_std", 0.0))
                noisy_baseline_metric1_std = raw_m1_std if np.isfinite(raw_m1_std) else 0.0
                raw_m2_std = float(getattr(noisy_metrics, "metric2_std", 0.0))
                noisy_baseline_metric2_std = raw_m2_std if np.isfinite(raw_m2_std) else 0.0
                raw_mean = float(getattr(noisy_metrics, "loss_mean", baseline.loss_mean))
                noisy_baseline_loss_mean = raw_mean if np.isfinite(raw_mean) else float(baseline.loss_mean)
                # Overwrite baseline std fields with the noisy preflight values —
                # these feed v3 combined_stab_excess thresholds. Keep means tied to
                # the clean reference so rank/report code has a stable frame.
                baseline.loss_std = noisy_baseline_loss_std
                baseline.metric1_std = noisy_baseline_metric1_std
                baseline.metric2_std = noisy_baseline_metric2_std
                preflight_ok = True
        except Exception as exc:
            log(f"  [baseline-preflight][warning] noisy probe failed: {exc}")

    _run_legacy_preflight_if_needed(
        robust_mode=robust_mode,
        run_legacy_preflight=run_legacy_preflight,
    )

    # Resolve gates from the noisy preflight + the user's tolerances.
    # tolerances come from rl_tune.py CLI (stage2_limit_tolerance,
    # stage2_stability_tolerance), which the launcher feeds from the preset
    # (defaults 0.005 / 0.005 in mrpc-blb-stage2-rl.conf).
    allowed_acc_drop = max(0.0, float(getattr(ev, "stage2_limit_tolerance", 0.05)))
    stability_tol = max(0.0, float(getattr(ev, "stage2_stability_tolerance", 1.2)))
    # 2026-06-11 fix: the v3 per-channel stability gates (m1_std / m2_std /
    # loss_std inside compute_reward) derive their thresholds from
    # weights.stab_tolerance, which silently stayed at the dataclass default
    # (0.5) regardless of --stage2-stability-tolerance. Wire the CLI tolerance
    # through so relaxing stability actually relaxes ALL stability channels,
    # not just the env-level loss_std gate below.
    weights.stab_tolerance = float(stability_tol)

    # ADR-015/Stage-1 alignment: Stage-1 reward shape plus std stability. The
    # active default gates off the ADR-011/012 exploration patches that were
    # tuned for the old tiered reward. Saturation is already off (tau=0).
    weights.reward_design = str(getattr(train_cfg, "reward_design", "stage1_aligned"))
    _continuous = weights.reward_design in ("continuous", "stage1_aligned")
    log(
        f"  {bullet} [ADR-015] reward_design={weights.reward_design}"
        + ("（连续有界 reward + 严格稳定性刹车 + Stage-1 cosine 熵 + 严格可行性选择；"
           "已关 anchor/warmstart/probe/ε/curriculum）" if _continuous else "（tiered 回滚路径）")
    )

    user_acc_threshold = float(base_env.acc_threshold)
    if not (np.isfinite(user_acc_threshold) and user_acc_threshold > 0.0):
        # Default: floor the gate at noisy_baseline × (1 - tolerance), so a
        # configured 0.001 is exactly a 0.1% relative drop. Do not subtract a
        # one-sample probe guard here: that made the true trainer gate looser
        # than the CLI/config value.
        new_acc_threshold = _noisy_metric_threshold_from_baseline(
            noisy_baseline_metric=float(noisy_baseline_metric1),
            tolerance=float(allowed_acc_drop),
        )
        base_env.acc_threshold = new_acc_threshold

    # v3: derive a separate m2 threshold from the noisy m2 baseline. Same
    # relative tolerance as m1; the thresholds differ only because
    # baseline.m1 != baseline.m2.
    if base_env.acc_threshold_m2 is None:
        base_env.acc_threshold_m2 = _noisy_metric_threshold_from_baseline(
            noisy_baseline_metric=float(noisy_baseline_metric2),
            tolerance=float(allowed_acc_drop),
        )

    # 2026-06-15 (user spec): loss_mean is also a hard constraint (LOWER-better),
    # aligning with Stage-1's loss/m1/m2 joint gate. Threshold lets the noisy
    # baseline loss RISE by the SAME limit_tolerance the accuracy gate allows m1/m2
    # to DROP — i.e. "loss 允许上浮 0.5%". Relative form (loss has no discrete
    # probe-quantization, so no one-sample guard). Only consumed when
    # reward_design="continuous"; the tiered rollback ignores it.
    if base_env.loss_threshold is None:
        base_env.loss_threshold = float(noisy_baseline_loss_mean) * (1.0 + float(allowed_acc_drop))

    stab_floor = float(getattr(weights, "stab_floor", 0.01) or 0.01)
    stab_threshold_m1 = _noisy_std_threshold_from_baseline(
        noisy_baseline_std=float(noisy_baseline_metric1_std),
        stability_multiplier=float(stability_tol),
        floor=stab_floor,
    )
    stab_threshold_m2 = _noisy_std_threshold_from_baseline(
        noisy_baseline_std=float(noisy_baseline_metric2_std),
        stability_multiplier=float(stability_tol),
        floor=stab_floor,
    )
    stab_threshold_loss = _noisy_std_threshold_from_baseline(
        noisy_baseline_std=float(noisy_baseline_loss_std),
        stability_multiplier=float(stability_tol),
        floor=stab_floor,
    )

    user_stab_threshold = float(base_env.stab_threshold)
    stab_calib_summary = ""
    if not np.isfinite(user_stab_threshold):
        # Loss channel is still passed as the legacy env-level override; m1/m2
        # derive from baseline.metric{1,2}_std inside compute_reward through the
        # same weights.stab_tolerance multiplier.
        base_env.stab_threshold = float(stab_threshold_loss)
        stab_calib_summary = (
            f"multiplier formula: "
            f"loss_std={noisy_baseline_loss_std:.4f} × tol={stability_tol:.4f} "
            f"→ loss_std_threshold={base_env.stab_threshold:.4f}; "
            f"m1_std={noisy_baseline_metric1_std:.4f} × tol={stability_tol:.4f} "
            f"→ m1_std_threshold={stab_threshold_m1:.4f}; "
            f"m2_std={noisy_baseline_metric2_std:.4f} × tol={stability_tol:.4f} "
            f"→ m2_std_threshold={stab_threshold_m2:.4f} "
            f"(floor={stab_floor:.4f})"
        )
    else:
        stab_threshold_loss = float(base_env.stab_threshold)

    baseline_preflight_metrics = {
        "ok": bool(preflight_ok),
        "trial_count": int(getattr(train_cfg, "num_trials_per_step", 1) or 1),
        "metric1_mean": float(noisy_baseline_metric1),
        "metric2_mean": float(noisy_baseline_metric2),
        "loss_mean": float(noisy_baseline_loss_mean),
        "metric1_std": float(noisy_baseline_metric1_std),
        "metric2_std": float(noisy_baseline_metric2_std),
        "loss_std": float(noisy_baseline_loss_std),
        "metric1_threshold": float(base_env.acc_threshold),
        "metric2_threshold": float(base_env.acc_threshold_m2),
        "loss_threshold": (
            float(base_env.loss_threshold) if base_env.loss_threshold is not None else None
        ),
        "metric1_std_threshold": float(stab_threshold_m1),
        "metric2_std_threshold": float(stab_threshold_m2),
        "loss_std_threshold": float(stab_threshold_loss),
        "limit_tolerance": float(allowed_acc_drop),
        "stability_tolerance": float(stability_tol),
        "stability_floor": float(stab_floor),
        "threshold_source": "noisy_all_max_blb_baseline",
    }

    robust_reference = None
    promotion_base_env = None
    authoritative_robust_reference = None
    authoritative_robust_summary = None
    authoritative_validation_banks = None
    authoritative_validation_example_count = 0
    if robust_mode:
        precision_tolerance, stability_multiplier, bootstrap_samples = (
            _resolve_robust_baseline_config(train_cfg, ev)
        )
        if decision_path == "layerwise":
            from .layerwise_runner import (
                validate_layerwise_three_bank_convergence_config,
                validate_layerwise_validation_bank_config,
            )

            configured_baseline_groups, configured_baseline_trials = (
                validate_layerwise_validation_bank_config(train_cfg)
            )
            validate_layerwise_three_bank_convergence_config(train_cfg)
        else:
            configured_baseline_groups = int(
                getattr(train_cfg, "baseline_groups", 5)
            )
            configured_baseline_trials = int(
                getattr(train_cfg, "baseline_trials_per_group", 5)
            )
        robust_reference, robust_summary = _collect_robust_baseline_reference(
            base_env=base_env,
            baseline_action_vec=baseline_action_vec,
            base_seed=int(train_cfg.seed),
            precision_tolerance=precision_tolerance,
            stability_multiplier=stability_multiplier,
            bootstrap_samples=bootstrap_samples,
            baseline_groups=configured_baseline_groups,
            trials_per_group=configured_baseline_trials,
            max_groups=max(10, 2 * configured_baseline_groups),
        )
        _install_robust_baseline_reference(
            base_env, baseline, weights, robust_reference,
        )
        base_env.statistical_gate_probability = float(
            getattr(train_cfg, "online_constraint_probability", 0.50)
        )
        stab_floor = float(weights.stab_floor)
        noisy_baseline_loss_mean = float(robust_reference.loss_mean)
        noisy_baseline_metric1 = float(robust_reference.metric1_mean)
        noisy_baseline_metric2 = float(robust_reference.metric2_mean)
        noisy_baseline_loss_std = float(robust_reference.loss_std)
        noisy_baseline_metric1_std = float(robust_reference.metric1_std)
        noisy_baseline_metric2_std = float(robust_reference.metric2_std)
        allowed_acc_drop = float(robust_reference.precision_tolerance)
        stability_tol = float(robust_reference.stability_multiplier)
        stab_threshold_loss = float(robust_reference.loss_std_limit)
        stab_threshold_m1 = float(robust_reference.metric1_std_limit)
        stab_threshold_m2 = float(robust_reference.metric2_std_limit)
        baseline_preflight_metrics["robust_reference"] = robust_summary
        baseline_preflight_metrics.update(robust_summary)
        baseline_preflight_metrics.update({
            "metric1_mean": noisy_baseline_metric1,
            "metric2_mean": noisy_baseline_metric2,
            "loss_mean": noisy_baseline_loss_mean,
            "metric1_std": noisy_baseline_metric1_std,
            "metric2_std": noisy_baseline_metric2_std,
            "loss_std": noisy_baseline_loss_std,
            "metric1_threshold": float(robust_reference.metric1_limit),
            "metric2_threshold": float(robust_reference.metric2_limit),
            "loss_threshold": float(robust_reference.loss_limit),
            "metric1_std_threshold": float(robust_reference.metric1_std_limit),
            "metric2_std_threshold": float(robust_reference.metric2_std_limit),
            "loss_std_threshold": float(robust_reference.loss_std_limit),
            "limit_tolerance": float(robust_reference.precision_tolerance),
            "stability_tolerance": float(robust_reference.stability_multiplier),
            "stability_floor": float(weights.stab_floor),
        })
        if decision_path == "layerwise":
            (
                promotion_base_env,
                authoritative_validation_example_count,
            ) = _build_authoritative_validation_env(
                runner=runner,
                ev=ev,
                base_env=base_env,
                train_cfg=train_cfg,
                reward_devices=reward_devices,
                log=log,
            )
            from .layerwise_runner import (
                LayerwiseValidationBank,
                LayerwiseValidationBanks,
            )
            from .statistical_constraints import build_baseline_reference

            bank_references = {}
            bank_summaries = {}
            bank_group_starts = {"A": 1_000, "B": 2_000, "C": 3_000}
            trials_per_bank_group = configured_baseline_trials
            for bank_label in ("A", "B", "C"):
                bank_reference, bank_summary = _collect_robust_baseline_reference(
                    base_env=promotion_base_env,
                    baseline_action_vec=baseline_action_vec,
                    base_seed=int(train_cfg.seed),
                    precision_tolerance=precision_tolerance,
                    stability_multiplier=stability_multiplier,
                    bootstrap_samples=bootstrap_samples,
                    baseline_groups=configured_baseline_groups,
                    trials_per_group=trials_per_bank_group,
                    max_groups=configured_baseline_groups,
                    group_index_start=bank_group_starts[bank_label],
                )
                bank_references[bank_label] = bank_reference
                bank_summaries[bank_label] = bank_summary

            promotion_reference = build_baseline_reference(
                [bank_references["A"].trials, bank_references["B"].trials],
                precision_tolerance=precision_tolerance,
                stability_multiplier=stability_multiplier,
                bootstrap_samples=bootstrap_samples,
                seed=int(train_cfg.seed) + 10_001,
            )
            final_reference = build_baseline_reference(
                [
                    bank_references["A"].trials,
                    bank_references["B"].trials,
                    bank_references["C"].trials,
                ],
                precision_tolerance=precision_tolerance,
                stability_multiplier=stability_multiplier,
                bootstrap_samples=bootstrap_samples,
                seed=int(train_cfg.seed) + 10_002,
            )

            def build_bank(label):
                summary = bank_summaries[label]
                return LayerwiseValidationBank(
                    label=label,
                    reference=bank_references[label],
                    probe_seeds=tuple(
                        int(group["group_probe_seed"])
                        for group in summary["groups"]
                    ),
                    trials_per_probe=trials_per_bank_group,
                )

            authoritative_validation_banks = LayerwiseValidationBanks(
                bank_a=build_bank("A"),
                bank_b=build_bank("B"),
                bank_c=build_bank("C"),
                promotion_reference=promotion_reference,
                final_reference=final_reference,
            )
            authoritative_robust_reference = promotion_reference

            def pooled_reference_summary(reference):
                return {
                    "trial_count": int(reference.trial_count),
                    "loss_mean": float(reference.loss_mean),
                    "metric1_mean": float(reference.metric1_mean),
                    "metric2_mean": float(reference.metric2_mean),
                    "loss_std": float(reference.loss_std),
                    "metric1_std": float(reference.metric1_std),
                    "metric2_std": float(reference.metric2_std),
                    "limits": {
                        "loss": float(reference.loss_limit),
                        "metric1": float(reference.metric1_limit),
                        "metric2": float(reference.metric2_limit),
                        "loss_std": float(reference.loss_std_limit),
                        "metric1_std": float(reference.metric1_std_limit),
                        "metric2_std": float(reference.metric2_std_limit),
                    },
                }

            authoritative_robust_summary = {
                "ok": True,
                "schema_version": "stage2_validation_banks_v1",
                "hard_gate": "six_point_constraints",
                "bootstrap_probability_role": "diagnostic_tiebreak_only",
                "banks": bank_summaries,
                "promotion_reference_ab": pooled_reference_summary(
                    promotion_reference,
                ),
                "final_reference_abc": pooled_reference_summary(
                    final_reference,
                ),
                "contract": authoritative_validation_banks.contract_payload(),
            }
            _install_robust_baseline_reference(
                promotion_base_env,
                promotion_base_env.baseline,
                promotion_base_env.reward_weights,
                authoritative_robust_reference,
            )
            baseline_preflight_metrics["authoritative_validation_full"] = {
                **dict(authoritative_robust_summary),
                "split": "validation_full",
                "example_count": int(authoritative_validation_example_count),
                "fidelity": "F4",
            }

    log(
        f"  {bullet} 基线噪声预热（noisy baseline preflight）："
        f"K={baseline_preflight_metrics['trial_count']}  "
        f"m1(noisy)={noisy_baseline_metric1:.4f}  "
        f"m2(noisy)={noisy_baseline_metric2:.4f}  "
        f"loss_mean(noisy)={noisy_baseline_loss_mean:.4f}  "
        f"std(loss/m1/m2)="
        f"{noisy_baseline_loss_std:.4f}/"
        f"{noisy_baseline_metric1_std:.4f}/"
        f"{noisy_baseline_metric2_std:.4f}"
    )
    _loss_thr_disp = (
        f"{base_env.loss_threshold:.4f}" if base_env.loss_threshold is not None else "None"
    )
    log(
        f"  {bullet} 校准后硬约束阈值（calibrated gates）："
        f"m1_threshold={base_env.acc_threshold:.4f}  "
        f"m2_threshold={float(base_env.acc_threshold_m2):.4f}  "
        f"loss_threshold={_loss_thr_disp}  "
        f"std_thresholds(loss/m1/m2)="
        f"{stab_threshold_loss:.4f}/"
        f"{stab_threshold_m1:.4f}/"
        f"{stab_threshold_m2:.4f}  "
        f"(limit_tol={allowed_acc_drop:.4f}, stab_tol={stability_tol:.4f}; "
        f"loss 越低越好，允许相对上浮 limit_tol；"
        f"m1/m2 越高越好，允许相对下降 limit_tol；std 越低越好，× stab_tol)"
    )
    if stab_calib_summary:
        log(f"  {bullet} 稳定阈值校准来源（stab calibration source）：{stab_calib_summary}")

    if reward_devices and len(reward_devices) >= 2:
        try:
            base_env.clear_installed_blb()
        except Exception:
            pass
        base_env.env_cfg.persistent_probe_install = True
        log(
            f"  {bullet} Multi-GPU BLB install cache：enabled "
            f"(devices={reward_devices}; wrappers/hooks stay installed and cfgs update in-place)"
        )

    _seq_block_title(log, "基线信号（baseline cost / reward / metrics）")
    _seq_log_rounded_box(log, [
        f"成本基线（baseline cost）："
        f"total_bits={baseline.total_bits_sum}, "
        f"fusion={baseline.total_fusion_count}, "
        f"avg_k={baseline.avg_k:.2f}",
        f"指标基线（baseline metrics）："
        f"loss={baseline.loss_mean:.4f}, m1={baseline.metric1_mean:.4f}, m2={baseline.metric2_mean:.4f}",
        f"奖励权重（reward weights, v2-style rdv2）："
        f"cost_weight={weights.cost_weight:.4g}, lambda_stab={weights.lambda_stab:.4g}, "
        f"invalid_penalty={weights.invalid_penalty:.4g}, "
        f"clip=[{weights.reward_clip_min:.1f}, {weights.reward_clip_max:.1f}], "
        f"tier=[{weights.tier_metric_bonus:.1f}, +{weights.tier_stability_bonus:.1f}], "
        f"baseline_metric1={weights.baseline_metric1:.4f}",
        f"硬约束阈值（acc=hard, stab=soft cap for excess penalty）："
        f"m1_threshold={base_env.acc_threshold:.4f}, "
        f"m2_threshold={float(base_env.acc_threshold_m2):.4f}, "
        f"loss_threshold={_loss_thr_disp}, "
        f"std_thresholds(loss/m1/m2)="
        f"{stab_threshold_loss:.4f}/{stab_threshold_m1:.4f}/{stab_threshold_m2:.4f}",
        f"static_skeletons archive：{ss_baseline_obj.archive_path}",
    ])

    # Keep the Pareto archive for diagnostics and empirical exploration stats.
    # It no longer supplies the PPO scalar cost reward unless
    # RewardWeights.cost_reward_mode is explicitly set to "pareto_only".
    base_env.pareto_cost_archive = ParetoCostArchive(baseline=baseline)
    log(
        f"  {bullet} Adaptive scalar cost reward：P1/P2 不吃 cost；P3 中 "
        f"fusion/K 使用区间式 boost，total_bits 使用弱线性项；Pareto frontier 仅用于诊断/探索统计。"
    )

    # ---------- 5) sequential env + policy ----------
    seq_env_cfg = SequentialEnvConfig(
        invalid_penalty=float(getattr(train_cfg, "sequential_invalid_penalty", 1.0)),
        cost_shaping_coeff=float(getattr(train_cfg, "sequential_cost_shaping_coeff", 0.0)),
        fusion_shaping_coeff=float(getattr(train_cfg, "sequential_fusion_shaping_coeff", 0.0)),
        early_terminate_on_invalid=bool(getattr(train_cfg, "sequential_early_terminate_on_invalid", False)),
    )
    # Fusion-count action mode (opt-in, 2026-06-03): each step decides
    # (fusion_option, K) via the offline map instead of all per-slot SF heads.
    # Disables safe-neighbor / guarded-radius2 (no SF-locality in option space);
    # the map holds only valid configs so invalid masks are unnecessary.
    # Resolve the fusion block-curriculum ramp once (0 → 0.5 * total_episodes) so
    # both the console banner and seq_train_cfg below use the same concrete value.
    _fc_curriculum_on = False if _continuous else bool(
        getattr(train_cfg, "fusion_neighbor_curriculum_enabled", False)
    )
    _fc_ramp = int(getattr(train_cfg, "fusion_neighbor_ramp_episodes", 0) or 0)
    if _fc_curriculum_on and _fc_ramp <= 0:
        _fc_ramp = max(1, int(FUSION_NEIGHBOR_RAMP_FRACTION * int(train_cfg.total_episodes)))
    fusion_map = None
    if bool(getattr(train_cfg, "fusion_count_action", False)):
        from .fusion_count_map import FusionCountMap
        fusion_map = FusionCountMap.load(str(train_cfg.profile))
        # Per-slot safe-neighbor / guarded-radius2 stay off in fusion mode; the
        # block-granularity curriculum below is the fusion-mode replacement.
        train_cfg.warmstart_neighbor_sampling = False
        log(
            f"  {bullet} Fusion-count action ENABLED：map graphs={len(fusion_map.graphs)}, "
            f"max_options={fusion_map.max_num_options()}；per-slot radius2 停用，"
            f"block 粒度 safe-neighbor curriculum "
            f"{'启用（ramp=' + str(_fc_ramp) + ' ep 后全开，全空间可达）' if _fc_curriculum_on else '停用（对照组：全开）'}"
        )
        # 2026-06-03 fusion reward: the P2 stability gate needs the per-episode std,
        # which requires >=2 trials. Warn on K<2 and force the fast-reward online-K=1
        # deferral off so each episode (and the noisy baseline preflight) runs a real
        # multi-trial probe (use --stage2-k-trials 4 + --blb-v3-reward-devices 0,1,2,3).
        if int(getattr(train_cfg, "num_trials_per_step", 4)) < 2:
            log(
                f"  {bullet} [fusion][warning] num_trials_per_step="
                f"{int(getattr(train_cfg, 'num_trials_per_step', 0))} < 2 — stability std "
                "gate needs >=2 trials; pass --stage2-k-trials 4 for the 4-trial probe."
            )
        if bool(getattr(train_cfg, "fast_reward_mode_enabled", False)):
            train_cfg.fast_reward_mode_enabled = False
            log(
                f"  {bullet} [fusion] fast-reward (online-K=1) disabled so every episode "
                "gets a real K-trial stability std."
            )
    if decision_path == "layerwise":
        return _run_layerwise_training_branch(
            train_cfg=train_cfg,
            evaluator=ev,
            base_env=base_env,
            fusion_map=fusion_map,
            max_sfs=max_sfs,
            robust_reference=robust_reference,
            promotion_base_env=promotion_base_env,
            authoritative_robust_reference=authoritative_robust_reference,
            authoritative_robust_summary=authoritative_robust_summary,
            authoritative_validation_banks=authoritative_validation_banks,
            authoritative_validation_example_count=(
                authoritative_validation_example_count
            ),
            static_skeletons_baseline=ss_baseline_obj,
            baseline_action_vec=ss_action_vec,
            fixed_gelu=fixed_gelu,
            fixed_softmax=fixed_softmax,
            fixed_label=fixed_label,
            fixed_source=fixed_source,
            blb_progress_dir=blb_progress_dir,
            baseline_preflight_metrics=baseline_preflight_metrics,
            status=status,
            resume_checkpoint_path=resume_checkpoint_path,
            run_lock=run_lock,
            log=log,
        )
    seq_env = BLBStage2SequentialEnv(base_env=base_env, env_cfg=seq_env_cfg, fusion_map=fusion_map)

    # ---------- 5.05) Episode-parallel rollout (fusion mode, opt-in 2026-06-10) ----------
    # --stage2-rl-devices "0,1,2,3,4" → N workers each run COMPLETE episodes
    # (policy rollout + per-step replan + serial K-trial probe) on their own
    # model replica; mirrors Stage-1's validated data-parallel pattern and
    # replaces the K-split probe parallelism for fusion mode. Empty → legacy
    # serial loop unchanged. Even a single id ("0") routes through the new
    # deterministic path so a 1-GPU run reproduces an N-GPU run bit-for-bit.
    stage2_parallel_runner = None
    stage2_rl_devices = [int(x) for x in (getattr(train_cfg, "stage2_rl_devices", []) or [])]
    if stage2_rl_devices:
        if fusion_map is None:
            raise RuntimeError(
                "--stage2-rl-devices requires the fusion-count action "
                "(--blb-v3-fusion-count-action 1); the per-slot path keeps the "
                "legacy loop / --blb-v3-reward-devices K-split."
            )
        if reward_devices and len(reward_devices) >= 2:
            raise RuntimeError(
                "--stage2-rl-devices and --blb-v3-reward-devices are mutually "
                "exclusive: episode-parallel runs the K trials serially on each "
                "worker's own GPU."
            )
        if bool(getattr(train_cfg, "fast_reward_mode_enabled", False)):
            raise RuntimeError(
                "--stage2-rl-devices is incompatible with fast reward mode "
                "(online-K=1 deferral); fusion mode needs the real K-trial std."
            )
        from .parallel_runner import build_stage2_parallel_runner
        stage2_parallel_runner = build_stage2_parallel_runner(
            primary_seq_env=seq_env,
            device_ids=stage2_rl_devices,
            layers_attribute="model." + ev.layers_attribute,
            is_regression=bool(getattr(ev, "is_regression", False)),
            bridge_factory=lambda: runner._build_rescale_bridge(train_cfg, log=log),
            seq_env_cfg=seq_env_cfg,
            fusion_map=fusion_map,
            log_fn=lambda m: log(f"  {m}"),
            workers_per_device=int(getattr(train_cfg, "stage2_workers_per_device", 1)),
        )
        # Persistent install per worker model: hooks stay installed across
        # episodes (cfg updates in place) — same optimization the K-split path
        # enables, now worker-local. Safe here because every worker deepcopied
        # a CLEAN model (preflight ran install→clear before this point).
        for _w in stage2_parallel_runner.workers:
            _w.seq_env.base.env_cfg.persistent_probe_install = True
        log(
            f"  {bullet} [stage2-parallel] episode-parallel rollout ENABLED: "
            f"devices={stage2_rl_devices} workers={stage2_parallel_runner.num_workers} "
            f"K={int(train_cfg.num_trials_per_step)} trials serial per worker; "
            f"global-episode seeding (policy/probe/update streams)"
        )

    # Checkpoint variant: fusion-mode policies have a different shape (max_step_dim=2),
    # so tag them so a per-slot checkpoint sharing the same persistent dir is rejected
    # cleanly on resume instead of failing with a torch shape-mismatch.
    seq_rl_variant = SEQ_RL_VARIANT + ("_fusioncount_v1" if fusion_map is not None else "")

    torch.manual_seed(int(train_cfg.seed))
    np.random.seed(int(train_cfg.seed) % (2**32))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_cfg = SequentialPolicyConfig(
        state_dim=int(seq_env.state_dim),
        max_step_dim=int(seq_env.max_step_dim),
        max_num_levels=(max(6, int(fusion_map.max_num_options())) if fusion_map is not None else 6),
        horizon=int(seq_env.horizon),
        num_layers=int(ev.total_layers),
        network_variant=getattr(train_cfg, "policy_network_variant", None),
    )
    policy = BLBStage2SequentialPolicy(policy_cfg).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(train_cfg.ppo.lr))

    if fusion_map is not None:
        # ADR-012 exploration floor: slot 0 = fusion option, slot 1 = K.
        # ADR-015: OFF under the continuous reward (cosine entropy is the
        # exploration mechanism now; the ε floor was a tiered-reward patch).
        _eps_opt = 0.0 if _continuous else float(
            getattr(train_cfg, "fusion_exploration_epsilon", 0.0) or 0.0
        )
        _eps_k = 0.0 if _continuous else float(
            getattr(train_cfg, "fusion_exploration_epsilon_k", 0.0)
            or 0.0
        )
        if _eps_opt > 0.0 or _eps_k > 0.0:
            policy.set_slot_exploration_epsilon(
                [_eps_opt, _eps_k] + [0.0] * (policy_cfg.max_step_dim - 2)
            )
            log(
                f"  [fusion][adr-012] exploration floor installed: "
                f"option eps={_eps_opt:g}, K eps={_eps_k:g} "
                f"(mixture replayed identically in PPO update)"
            )

    # Warmstart: bias every action head row toward the BASELINE-indexed slot.
    # 2026-05-18 (rdv2 hotfix): the earlier ``[LEVELS_F - 1] * max_step_dim
    # = [4]*13`` formula was wrong for 8/13 slot positions. The per-step slot
    # kinds vary per (layer, block) — slot 2 in some steps is M (3 levels,
    # baseline idx 2), in others R (4 levels, baseline idx 3); slot 6 is
    # K-in-B1/3/5 (baseline idx 3) for many steps but K-in-B2/4 (idx 4) for
    # others. Setting all to 4 either masked out (for slots with <5 active
    # levels) or actively biased AWAY from baseline (K-B1/3/5: preferred 4
    # = K_LEVELS[4]=10 vs baseline 3 = K=13).
    #
    # Fix: compute the MODE of baseline_action_vec values across all 59 steps
    # for each of the 13 slot positions, and use that as the preferred index.
    # This matches the most common baseline value for each slot position and
    # makes the bias actually push toward baseline rather than away from it.
    # Falls back to LEVELS_F-1 if step schedule isn't available (defensive).
    #
    # NOTE: the forced-baseline anchor (force_baseline_episodes, applied per
    # episode in the inner loop below) is the primary warmstart mechanism;
    # this bias is the soft prior that takes over after the anchor ends.
    warmstart_applied = False
    preferred_summary = ""
    if bool(train_cfg.warmstart_baseline_bias):
        try:
            from .action_space import LEVELS_F
            if fusion_map is not None:
                # fusion: slot 0 = option, slot 1 = K (baseline index).
                # Cold-start near the verified baseline, then decay the prior to
                # zero via _resolve_baseline_prior_scale. Leaving the option slot
                # unbiased makes a fresh 47-step policy sample too many fusion=1
                # choices before it has evidence, which collapses the first
                # post-anchor window to loss caps.
                preferred = [0, int(_baseline_k_index_for_block(1))]
            else:
                preferred = _compute_per_slot_mode_preferred(
                    schedule=seq_env.schedule,
                    baseline_action_vec=baseline_action_vec,
                    max_step_dim=policy_cfg.max_step_dim,
                    fallback_idx=int(LEVELS_F) - 1,
                )
            warmstart_gain = float(train_cfg.warmstart_bias_gain)
            if _continuous:
                # Stage-1-aligned reward keeps the policy algorithm simple, but
                # still needs a schedulable cold-start prior when the user enables
                # warmstart_baseline_bias. The explicit baseline_prior_scale stored
                # per episode controls the decay and reaches zero, so this default
                # gain is only a fallback for calls that omit the explicit scale.
                warmstart_gain = max(warmstart_gain, 1.0)
            elif fusion_map is not None:
                # Tiny fusion action space (<=2 options x 6 K per block): pull the
                # baseline (fusion=0 / K=max) prior up so cold-start sits at baseline
                # and explores outward (user spec 2026-06-03).
                warmstart_gain = max(warmstart_gain, FUSION_WARMSTART_BIAS_GAIN)
            policy.apply_preferred_per_step_bias(
                preferred,
                gain=warmstart_gain,
            )
            warmstart_applied = True
            preferred_summary = (
                f"preferred per slot (mode over {len(seq_env.schedule)} steps) = "
                + str(preferred)
            )
        except Exception as exc:
            log(f"  [warmstart][warning] preferred-per-step bias failed: {exc}")

    action_dim_by_index = action_dims_for_config(int(ev.total_layers))
    mutable_neighbor_offsets: List[int] = []
    try:
        baseline_desc_for_curriculum = describe_action_vector(
            baseline_action_vec,
            max_sfs=max_sfs,
            num_layers=int(ev.total_layers),
            gelu_degree=fixed_gelu,
            attn_degree=fixed_softmax,
            profile=str(train_cfg.profile),
        )
        for record in baseline_desc_for_curriculum.get("records", []) or []:
            if not isinstance(record, Mapping):
                continue
            gi = int(record.get("global_index", -1))
            if gi < 0 or gi >= len(action_dim_by_index):
                continue
            if int(action_dim_by_index[gi]) <= 1:
                continue
            if not bool(record.get("effective", True)):
                continue
            if record.get("effective_value") is None:
                continue
            mutable_neighbor_offsets.append(int(gi))
    except Exception as exc:
        mutable_neighbor_offsets = []
        log(f"  [warmstart][warning] failed to build sequential mutable slots: {exc}")

    # Preview values for the startup hyperparameter box. The actual
    # values used during train_sequential are computed later (in the
    # SequentialTrainConfig construction + force_baseline_episodes resolve)
    # — these mirror that logic so the box shows the eventual config.
    _preview_force_baseline_episodes = _resolve_sequential_force_baseline_episodes(train_cfg)
    _preview_ent_coef_anchor = float(getattr(train_cfg, "ent_coef_anchor", 0.0))
    _preview_ramp = int(getattr(train_cfg, "ent_coef_ramp_episodes", 600))

    _seq_block_title(log, "训练超参与环境设置（Training hyperparameters · sequential per-block）")
    _seq_log_rounded_box(log, [
        f"Sequential env：horizon={seq_env.horizon}    "
        f"max_step_dim={seq_env.max_step_dim}    "
        f"state_dim={seq_env.state_dim}    "
        f"device={str(device)}",
        f"Policy：GTrXL d_model={policy_cfg.d_model}, heads={policy_cfg.n_heads}, "
        f"layers={policy_cfg.n_layers}, d_ff={policy_cfg.d_ff}, dropout={policy_cfg.dropout:.2f}, "
        f"per-slot heads=[{policy_cfg.max_step_dim}×{policy_cfg.max_num_levels}]    "
        f"env_layers={policy_cfg.num_layers}",
        f"PPO：lr={train_cfg.ppo.lr:.6g}    "
        f"clip={train_cfg.ppo.clip_range:.3f}    "
        f"n_epochs={train_cfg.ppo.n_epochs}    "
        f"mb={train_cfg.ppo.minibatch_size}    "
        f"ent_coef={train_cfg.ppo.ent_coef:.4g}    "
        f"value_coef={train_cfg.ppo.value_coef:.4g}",
        f"训练规模：total_episodes={train_cfg.total_episodes}    "
        f"rollout_size={max(1, int(train_cfg.rollout_size))}    "
        f"save_interval={int(train_cfg.save_interval)}    "
        f"calibrate_baseline_samples={int(train_cfg.calibrate_baseline_samples)}",
        f"Reward shaping：invalid_penalty={seq_env_cfg.invalid_penalty:.3g}    "
        f"cost_coeff={seq_env_cfg.cost_shaping_coeff:.3g}    "
        f"fusion_coeff={seq_env_cfg.fusion_shaping_coeff:.3g}    "
        f"early_term_on_invalid={seq_env_cfg.early_terminate_on_invalid}",
        f"Warmstart：decaying_logit_prior={warmstart_applied}    "
        f"initial_gain={float(train_cfg.warmstart_bias_gain):.3g}    "
        f"force_baseline_episodes={int(_preview_force_baseline_episodes)}    "
        f"{preferred_summary}",
        f"Baseline prior schedule：anchor=1.20; ep60..600: 1.00→0.45; "
        f"ep600..2000: 0.45→0.15; after ep2000: 0.15",
        f"Safe neighbor curriculum：enabled={bool(getattr(train_cfg, 'warmstart_neighbor_sampling', False))}    "
        f"mutable_offsets={len(mutable_neighbor_offsets)}    "
        f"ramp={int(getattr(train_cfg, 'warmstart_neighbor_ramp_episodes', 0) or train_cfg.total_episodes)}    "
        f"max_mutations={int(getattr(train_cfg, 'warmstart_neighbor_max_mutations', 8))}    "
        f"max_radius={int(getattr(train_cfg, 'warmstart_neighbor_max_radius', 2))}",
        f"Guarded radius2：enabled={bool(getattr(train_cfg, 'guarded_radius2_enabled', False))}    "
        f"min_episode={int(getattr(train_cfg, 'guarded_radius2_min_episode', 1060))}    "
        f"stall_window={int(getattr(train_cfg, 'guarded_radius2_stall_window', 600))}    "
        f"max_mutations={int(getattr(train_cfg, 'guarded_radius2_max_mutations', 4))}    "
        f"fraction={float(getattr(train_cfg, 'guarded_radius2_episode_fraction', 0.15)):.3g}    "
        f"cooldown={int(getattr(train_cfg, 'guarded_radius2_cooldown_episodes', 300))}",
        f"Static invalid-level pre-mask：enabled={bool(getattr(train_cfg, 'static_invalid_level_mask_enabled', False))}    "
        "scan=baseline-prefix one-slot optimizer feasibility",
        f"Empirical invalid-level mask：enabled={bool(getattr(train_cfg, 'empirical_invalid_level_mask_enabled', False))}    "
        f"min_invalid={int(getattr(train_cfg, 'empirical_invalid_level_min_samples', 3))}    "
        f"min_rate={float(getattr(train_cfg, 'empirical_invalid_level_min_rate', 0.80)):.2f}    "
        f"max_valid={int(getattr(train_cfg, 'empirical_invalid_level_max_valid', 0))}",
        f"Fast reward mode：enabled={bool(getattr(train_cfg, 'fast_reward_mode_enabled', False))}    "
        f"online_k={int(getattr(train_cfg, 'online_num_trials_per_step', 5))}    "
        f"terminal_eval_batch_size={int(getattr(train_cfg, 'terminal_eval_batch_size', 4))}    "
        f"promotion_validation_trials={int(getattr(train_cfg, 'promotion_validation_trials', 4))}    "
        f"final_selection_top_n={int(getattr(train_cfg, 'final_selection_top_n', 20))}    "
        f"final_selection_validation_trials={int(getattr(train_cfg, 'final_selection_validation_trials', 20))}    "
        f"promotion_margin_window={float(getattr(train_cfg, 'promotion_margin_window', 0.25)):.3g}",
        "Non-monotonic cost-boundary exploration：SF/K move 是 proposal；真实方向只由 F1 metric/stability、"
        "Rescale_optimizer cost signals、adaptive scalar cost 和 diagnostic archive 确认。",
        f"Entropy schedule (anchor → ramp → steady)："
        f"anchor[0..{int(_preview_force_baseline_episodes)}]ep_coef={float(_preview_ent_coef_anchor):.4g} → "
        f"ramp[{int(_preview_force_baseline_episodes)}..{int(_preview_force_baseline_episodes)+int(_preview_ramp)}]ep_coef→{float(train_cfg.ppo.ent_coef):.4g} → "
        f"steady[{int(_preview_force_baseline_episodes)+int(_preview_ramp)}+]={float(train_cfg.ppo.ent_coef):.4g}",
    ])

    # ---------- 6) optional resume ----------
    # ---------- 6) optional resume ----------
    # Persistent dir's standard live checkpoint name. If the caller didn't pass
    # an explicit resume_checkpoint_path, but a live ckpt exists at the
    # standard location, prefer that automatically (matches the launcher's
    # "same params → auto-resume" contract).
    save_path = os.path.join(blb_progress_dir, "blb_stage2_rl_checkpoint_live.pt")
    effective_resume_path = resume_checkpoint_path
    if (not effective_resume_path) and os.path.isfile(save_path):
        effective_resume_path = save_path
        log(f"  {bullet} 检测到已存在 live checkpoint，自动 resume: {save_path}")

    start_episode = 0
    best_reward = -float("inf")
    best_rank_key: Tuple[float, ...] = tuple()
    best_action_vec: Optional[np.ndarray] = None
    best_record: Optional[EpisodeRecord] = None
    if effective_resume_path and os.path.isfile(effective_resume_path):
        try:
            ckpt = torch.load(effective_resume_path, map_location=device)
            ckpt_variant = str(ckpt.get("rl_variant", "") or "")
            if ckpt_variant and ckpt_variant != seq_rl_variant:
                log(
                    f"  [resume][warning] checkpoint at {effective_resume_path} "
                    f"has rl_variant={ckpt_variant!r} (expected {seq_rl_variant!r}); "
                    f"skipping load to avoid policy-shape mismatch. Training will "
                    f"start fresh."
                )
            else:
                if "policy" in ckpt:
                    policy.load_state_dict(ckpt["policy"])
                if "policy_ppo_aux" in ckpt:
                    policy.load_ppo_aux_state_dict(ckpt.get("policy_ppo_aux"))
                if "optimizer" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer"])
                start_episode = int(ckpt.get("episode", 0))
                if "best_reward" in ckpt:
                    try:
                        best_reward = float(ckpt["best_reward"])
                    except Exception:
                        best_reward = -float("inf")
                saved_rank_key = ckpt.get("best_rank_key")
                if isinstance(saved_rank_key, (list, tuple)):
                    try:
                        best_rank_key = tuple(float(x) for x in saved_rank_key)
                    except Exception:
                        best_rank_key = tuple()
                if ckpt.get("best_action") is not None:
                    try:
                        best_action_vec = np.asarray(
                            ckpt["best_action"], dtype=np.int64
                        )
                    except Exception:
                        best_action_vec = None
                log(
                    f"  {bullet} resumed from {effective_resume_path} @ "
                    f"ep={start_episode}    best_reward="
                    f"{('+%.4f' % best_reward) if np.isfinite(best_reward) else 'N/A'}"
                )
        except Exception as exc:
            log(f"  [resume][warning] failed to resume from {effective_resume_path}: {exc}")

    status.set_phase("PPO 训练 (sequential per-block)")

    # ---------- 7) sequential training loop ----------
    ppo = SequentialPPOConfig(
        lr=float(train_cfg.ppo.lr),
        clip_range=float(train_cfg.ppo.clip_range),
        n_epochs=int(train_cfg.ppo.n_epochs),
        minibatch_size=int(train_cfg.ppo.minibatch_size),
        ent_coef=float(train_cfg.ppo.ent_coef),
        value_coef=float(train_cfg.ppo.value_coef),
        max_grad_norm=float(train_cfg.ppo.max_grad_norm),
    )
    # Clamp remaining episodes ≥ 0 — if start_episode already exceeds the
    # total, the loop simply runs 0 episodes and the assembly path emits
    # the final report against the existing best_action_vec from the
    # resumed checkpoint.
    remaining_episodes = max(0, int(train_cfg.total_episodes) - int(start_episode))
    seq_train_cfg = SequentialTrainConfig(
        total_episodes=int(remaining_episodes),
        update_every_n_episodes=max(1, int(train_cfg.rollout_size)),
        log_every_n_episodes=max(1, int(train_cfg.rollout_size)),
        seed=int(train_cfg.seed),
        ppo=ppo,
        rl_algo="ppo",
        grpo_kl_beta=0.0,
        ent_coef_anchor=float(getattr(train_cfg, "ent_coef_anchor", 0.0)),
        ent_coef_ramp_episodes=int(getattr(train_cfg, "ent_coef_ramp_episodes", 600)),
        absolute_episode_start=int(start_episode),
        warmstart_neighbor_sampling=bool(getattr(train_cfg, "warmstart_neighbor_sampling", False)),
        warmstart_neighbor_ramp_episodes=(
            int(getattr(train_cfg, "warmstart_neighbor_ramp_episodes", 0) or 0)
            if bool(getattr(train_cfg, "warmstart_neighbor_sampling", False))
            else 0
        ),
        warmstart_neighbor_max_mutations=int(
            getattr(train_cfg, "warmstart_neighbor_max_mutations", 8)
        ),
        warmstart_neighbor_max_radius=int(
            getattr(train_cfg, "warmstart_neighbor_max_radius", 2)
        ),
        warmstart_mutable_full_offsets=list(mutable_neighbor_offsets),
        fusion_neighbor_curriculum_enabled=bool(_fc_curriculum_on),
        fusion_neighbor_ramp_episodes=int(_fc_ramp),
        fusion_neighbor_max_radius=int(getattr(train_cfg, "fusion_neighbor_max_radius", 6)),
        # ADR-015: probes / ε floor OFF under the continuous reward (tiered patches).
        fusion_probe_interval=(0 if _continuous else int(getattr(train_cfg, "fusion_probe_interval", 0))),
        fusion_exploration_epsilon=(0.0 if _continuous else float(
            getattr(train_cfg, "fusion_exploration_epsilon", 0.0)
        )),
        fusion_exploration_epsilon_k=(0.0 if _continuous else float(
            getattr(train_cfg, "fusion_exploration_epsilon_k", 0.0)
        )),
        # ADR-015: Stage-1 cosine entropy schedule + continuous reward design.
        ent_coef_schedule=str(getattr(train_cfg, "ent_coef_schedule", "cosine")),
        ent_coef_cosine_start=float(getattr(train_cfg, "ent_coef_cosine_start", 0.05)),
        ent_coef_cosine_end=float(getattr(train_cfg, "ent_coef_cosine_end", 0.001)),
        ent_coef_cosine_plateau=float(getattr(train_cfg, "ent_coef_cosine_plateau", 0.25)),
        ent_coef_cosine_lower_bound=float(getattr(train_cfg, "ent_coef_cosine_lower_bound", 0.012)),
        reward_design=str(getattr(train_cfg, "reward_design", "stage1_aligned")),
        guarded_radius2_enabled=bool(getattr(train_cfg, "guarded_radius2_enabled", False)),
        guarded_radius2_min_episode=int(getattr(train_cfg, "guarded_radius2_min_episode", 1060)),
        guarded_radius2_stall_window=int(getattr(train_cfg, "guarded_radius2_stall_window", 600)),
        guarded_radius2_health_window=int(getattr(train_cfg, "guarded_radius2_health_window", 100)),
        guarded_radius2_max_mutations=int(getattr(train_cfg, "guarded_radius2_max_mutations", 4)),
        guarded_radius2_episode_fraction=float(
            getattr(train_cfg, "guarded_radius2_episode_fraction", 0.15)
        ),
        guarded_radius2_cooldown_episodes=int(
            getattr(train_cfg, "guarded_radius2_cooldown_episodes", 300)
        ),
        guarded_radius2_min_radius1_successes=int(
            getattr(train_cfg, "guarded_radius2_min_radius1_successes", 3)
        ),
        static_invalid_level_mask_enabled=bool(
            getattr(train_cfg, "static_invalid_level_mask_enabled", False)
        ),
        fast_reward_mode_enabled=bool(getattr(train_cfg, "fast_reward_mode_enabled", False)),
        online_num_trials_per_step=int(getattr(
            train_cfg,
            "online_num_trials_per_step",
            getattr(train_cfg, "num_trials_per_step", 5),
        )),
        terminal_eval_batch_size=int(getattr(train_cfg, "terminal_eval_batch_size", 4)),
        protected_k1_enabled=bool(
            getattr(train_cfg, "protected_k1_enabled", False)
        ),
        protected_k1_guard_sigma=float(
            getattr(train_cfg, "protected_k1_guard_sigma", 4.0)
        ),
        protected_k1_audit_fraction=float(
            getattr(train_cfg, "protected_k1_audit_fraction", 0.02)
        ),
        promotion_validation_trials=int(getattr(train_cfg, "promotion_validation_trials", 4)),
        promotion_margin_window=float(getattr(train_cfg, "promotion_margin_window", 0.25)),
        final_selection_top_n=int(getattr(train_cfg, "final_selection_top_n", 20)),
        final_selection_validation_trials=int(
            getattr(train_cfg, "final_selection_validation_trials", 20)
        ),
    )

    episode_returns: List[float] = []
    live_episode_records: List[EpisodeRecord] = []
    live_ppo_metrics: List[Dict[str, Any]] = []
    total_episodes_planned = int(train_cfg.total_episodes)
    rollout_avg_window: List[float] = []
    rollout_invalid_window: List[int] = []
    rollout_valid_window: List[int] = []
    rollout_terminal_window: List[float] = []
    ppo_update_counter = [0]   # mutable closure cell

    # ---------- 6.5) Long-term diagnostics recorder ----------
    # Writes JSONL / NPZ / Markdown to <blb_progress_dir>/diagnostics/. The
    # summary.md is regenerated each save_interval and is the entry point for
    # debugging a paused / finished run.
    num_action_slots = len(action_dims_for_config(int(ev.total_layers)))

    # slots_view_builder converts a raw RL action_vec into the human-readable
    # "slot list" form (slot_label + scaling_factor / truncation_bits). The
    # recorder calls this lazily when writing best_action_vec.json + top_candidates.
    def _slots_view_builder(vec):
        return action_vec_to_slots_list(
            vec,
            max_sfs=max_sfs,
            num_layers=int(ev.total_layers),
            gelu_degree=fixed_gelu,
            attn_degree=fixed_softmax,
            profile=str(train_cfg.profile),
        )

    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _stage2_run_source = (
        str(getattr(ev, "run_output_dir", "") or "").strip()
        or os.path.dirname(os.path.normpath(blb_progress_dir))
        or f"stage2-{train_cfg.profile}"
    )
    try:
        _stage2_run_id_base = os.path.relpath(_stage2_run_source, _repo_root)
    except ValueError:
        _stage2_run_id_base = str(_stage2_run_source)
    _stage2_run_id = make_unique_run_id(_stage2_run_id_base)
    stage2_data_writer = RLDataPointWriter(
        root_dir=os.path.join(_repo_root, "rl_training_data_points"),
        run_id=_stage2_run_id,
        stage="stage2",
        model_type=stage2_model_type,
        dataset=str(train_cfg.profile),
    )
    log(f"  {bullet} [data-points] Stage-2 structured RL data → {stage2_data_writer.run_dir}")

    diag_recorder = RLDiagnosticsRecorder(
        output_dir=blb_progress_dir,
        num_layers=int(ev.total_layers),
        num_action_slots=int(num_action_slots),
        max_action_levels=6,
        top_k=20,
        log_fn=log,
        slots_view_builder=_slots_view_builder,
        data_point_writer=stage2_data_writer,
    )
    # Provide the static_skeletons baseline so top-K rows in the summary
    # can show *diffs* against it (which slots actually changed vs baseline).
    try:
        diag_recorder.set_baseline_action_vec(baseline_action_vec)
    except Exception as exc:
        log(f"  [diag][warning] set_baseline_action_vec failed: {exc}")
    diag_recorder.set_meta({
        "profile": str(train_cfg.profile),
        "source_data_run_id_base": _stage2_run_id_base,
        "fixed_label": str(fixed_label),
        "fixed_source": str(fixed_source),
        "rl_variant": seq_rl_variant,
        "reward_design": str(getattr(train_cfg, "reward_design", "stage1_aligned")),
        "total_episodes_planned": int(total_episodes_planned),
        "rollout_size": int(train_cfg.rollout_size),
        "save_interval": int(train_cfg.save_interval),
        "stage2_k_trials": int(getattr(train_cfg, "num_trials_per_step", 0) or 0),
        "stage2_limit_tolerance": float(allowed_acc_drop),
        "stage2_stability_tolerance": float(stability_tol),
        "ppo_lr": float(train_cfg.ppo.lr),
        "ppo_clip_range": float(train_cfg.ppo.clip_range),
        "ppo_ent_coef": float(train_cfg.ppo.ent_coef),
        "ppo_value_coef": float(train_cfg.ppo.value_coef),
        "invalid_penalty": float(seq_env_cfg.invalid_penalty),
        "cost_shaping_coeff": float(seq_env_cfg.cost_shaping_coeff),
        "fusion_shaping_coeff": float(seq_env_cfg.fusion_shaping_coeff),
        "early_terminate_on_invalid": bool(seq_env_cfg.early_terminate_on_invalid),
        "static_invalid_level_mask_enabled": bool(
            getattr(train_cfg, "static_invalid_level_mask_enabled", False)
        ),
        "acc_threshold": float(base_env.acc_threshold),
        "acc_threshold_m2": float(base_env.acc_threshold_m2),
        "loss_threshold": (
            float(base_env.loss_threshold) if base_env.loss_threshold is not None else None
        ),
        "stab_threshold": float(base_env.stab_threshold),
        "stab_threshold_loss": float(stab_threshold_loss),
        "stab_threshold_m1": float(stab_threshold_m1),
        "stab_threshold_m2": float(stab_threshold_m2),
        "baseline_preflight_metrics": dict(baseline_preflight_metrics),
        "borderline_retest_enabled": bool(base_env.env_cfg.borderline_retest_enabled),
        "borderline_retest_trials_multiplier": int(
            base_env.env_cfg.borderline_retest_trials_multiplier
        ),
        "static_skeletons_archive": str(ss_baseline_obj.archive_path),
        "fast_reward_mode_enabled": bool(getattr(train_cfg, "fast_reward_mode_enabled", False)),
        "online_num_trials_per_step": int(getattr(train_cfg, "online_num_trials_per_step", 5)),
        "terminal_eval_batch_size": int(getattr(train_cfg, "terminal_eval_batch_size", 4)),
        "protected_k1_enabled": bool(
            getattr(train_cfg, "protected_k1_enabled", False)
        ),
        "protected_k1_guard_sigma": float(
            getattr(train_cfg, "protected_k1_guard_sigma", 4.0)
        ),
        "protected_k1_audit_fraction": float(
            getattr(train_cfg, "protected_k1_audit_fraction", 0.02)
        ),
        "promotion_validation_trials": int(getattr(train_cfg, "promotion_validation_trials", 4)),
        "final_selection_top_n": int(getattr(train_cfg, "final_selection_top_n", 20)),
        "final_selection_validation_trials": int(
            getattr(train_cfg, "final_selection_validation_trials", 20)
        ),
        "promotion_margin_window": float(getattr(train_cfg, "promotion_margin_window", 0.25)),
    })
    try:
        diag_recorder.set_baseline_avg_k(float(baseline.avg_k))
    except Exception:
        pass

    _seq_log_major_rule(log, "开始 PPO 训练（PPO training start）")
    _seq_log_rounded_box(log, [
        f"总回合：{total_episodes_planned}    "
        f"PPO 窗口：每 {max(1, int(train_cfg.rollout_size))} 回合更新一次    "
        f"checkpoint 间隔：每 {int(train_cfg.save_interval)} 回合",
        f"进度文件：{os.path.join(blb_progress_dir, 'blb_stage2_status.json')}",
        f"checkpoint：{save_path}",
        f"训练曲线：{os.path.join(blb_progress_dir, 'blb_stage2_training_curve.png')}",
        f"详细 trace：第 1 个 episode 会逐 step 打印；之后按窗口汇总",
    ])

    def _format_best_action_slots(action_vec_arr: Optional[np.ndarray]) -> List[str]:
        """Best-action decoded slot view, one slot per line.

        Layout (chosen 2026-05-17 after user reported the old `; `-joined form
        was unreadable on terminals < 200 cols):

            [L00.B1]   (block scope subtitle on its own line)
              · F.gelu_out_sf            scaling_factor=22     [op=ctpt_gelu_out, dist=fresh, N=8192] [inactive]
              · W.wffn2_sf               scaling_factor=12     [op=ctpt_ffn2, dist=encoding, N=8192]
              · K.output_truncation_k    truncation_bits=8     [op=block1_output_truncation, dist=truncation, N=8192]

        Each slot lands on its own line so terminals don't wrap mid-record, and
        each block gets a `[L<i>.B<n>]` header so the eye can quickly scan
        which layer/block a slot belongs to. ``scaling_factor`` / ``truncation_bits``
        is column-aligned with a fixed minimum width on the field-name column so
        values stack vertically.
        """
        if action_vec_arr is None:
            return ["best action 尚未产生（episode_count=0）"]
        try:
            arr = np.asarray(action_vec_arr, dtype=int).reshape(-1)
            slots = _slots_view_builder(arr)

            def _slot_label(row: Mapping[str, Any]) -> str:
                kind = str(row.get("kind", ""))
                field_name = str(row.get("field_name", ""))
                return f"{kind}.{field_name}"

            def _slot_value(row: Mapping[str, Any]) -> str:
                kind = str(row.get("kind", ""))
                if kind == "K":
                    value = row.get("truncation_bits")
                    return f"truncation_bits={value}"
                value = row.get("scaling_factor")
                return "scaling_factor=off" if value is None else f"scaling_factor={value}"

            def _slot_meta(row: Mapping[str, Any]) -> str:
                meta = (
                    f"[op={row.get('operation')}, "
                    f"dist={row.get('distribution')}, "
                    f"N={row.get('N')}]"
                )
                if not bool(row.get("effective", True)):
                    meta += " [inactive]"
                return meta

            grouped: Dict[str, List[Mapping[str, Any]]] = {}
            first_input_rows: List[Mapping[str, Any]] = []
            for row in slots:
                block = row.get("block")
                if block is None:
                    first_input_rows.append(row)
                    continue
                layer = int(row.get("layer", 0))
                key = f"L{layer:02d}.B{int(block)}"
                grouped.setdefault(key, []).append(row)

            lines: List[str] = []

            def _emit_block(header: str, rows: List[Mapping[str, Any]]) -> None:
                if not rows:
                    return
                lines.append(f"[{header}]")
                # Column widths derived from the actual rows so blocks with
                # longer field names don't bleed into the value column.
                label_w = max(len(_slot_label(r)) for r in rows)
                value_w = max(len(_slot_value(r)) for r in rows)
                label_w = max(label_w, 18)
                value_w = max(value_w, 22)
                for r in rows:
                    label = _slot_label(r).ljust(label_w)
                    value = _slot_value(r).ljust(value_w)
                    lines.append(f"  · {label}  {value}  {_slot_meta(r)}")

            for key in sorted(grouped):
                _emit_block(key, grouped[key])
            _emit_block("first_input", first_input_rows)
            return lines
        except Exception as exc:
            return [f"<format_best_action_slots failed: {exc}>"]

    def _step_callback(episode_idx: int, step_within: int, info: Dict[str, Any]) -> None:
        """Per-step trace — only for the first episode of the WHOLE run. Helps
        verify wiring and convergence on the very first rollout, then
        gracefully stays silent thereafter (the periodic boxes carry summary)."""
        if int(episode_idx) != 0:
            return
        valid = bool(info.get("valid", True))
        flag = "✓" if valid else "✗"
        step = int(info.get("step", step_within))
        block_idx = int(info.get("block_idx", 0))
        layer_idx = int(info.get("layer_idx", 0))
        total_bits = int(info.get("total_bits", 0))
        fusion = int(info.get("fusion_count", 0))
        reward = float(info.get("reward", 0.0))
        value = float(info.get("value", 0.0))
        action = info.get("action", [])
        action_str = ",".join(str(int(a)) for a in (action[:8] if isinstance(action, list) else []))
        if isinstance(action, list) and len(action) > 8:
            action_str += "…"
        log(
            f"    [ep0 step {step:02d}] {flag} L{layer_idx:02d}-B{block_idx} "
            f"a=[{action_str}] r={reward:+.4f} V={value:+.3f} "
            f"bits={total_bits} fusion={fusion}"
        )

    def _episode_callback(record: EpisodeRecord) -> None:
        nonlocal best_reward, best_rank_key, best_action_vec, best_record
        episode_returns.append(float(record.total_reward))
        live_episode_records.append(record)
        rollout_avg_window.append(float(record.total_reward))
        rollout_invalid_window.append(int(record.invalid_steps))
        rollout_valid_window.append(int(record.valid_step_count))
        rollout_terminal_window.append(float(record.terminal_reward))
        record_full_vec = _record_full_vec_for_callback(record, seq_env)

        # Per-episode details file (details/noise_ppo_step_info_<start>-<end>.txt).
        # Mirrors what legacy noise_rl_module_v2 wrote: one record per episode with
        # return / priority / cost signals / invalidity. Auto-rolls every
        # ``details_batch_size`` episodes — the writer owns the buffer + flush.
        #
        # 2026-05-17: rich invalid-block enumeration. Each invalid sub-step is
        # listed on its own line as `L<i>-B<n> graph=<key> reason=<reason>`,
        # so operators can grep "L11-B3" / "fusion cannot reduce" / etc. without
        # re-running the optimizer. ``first_invalid`` is kept as a sticky
        # summary on top of that list for fast scanning.
        try:
            # 2026-05-18: use the real breakdown.priority surfaced via
            # EpisodeRecord.terminal_priority. Fallback to the legacy
            # "1 if invalid_steps>0 else 3" only when terminal_priority is
            # unset (very short or early-terminated episodes that never
            # reached compute_reward).
            if int(record.terminal_priority) > 0:
                priority = int(record.terminal_priority)
            else:
                priority = 1 if record.invalid_steps > 0 else 3
            extra_lines = [
                (
                    f"valid_steps={int(record.valid_step_count)}/{int(record.steps_taken)}, "
                    f"invalid_steps={int(record.invalid_steps)}, "
                    f"terminal_reward={float(record.terminal_reward):+.4f}"
                ),
                (
                    f"terminal_metrics: loss_mean={float(record.terminal_loss_mean):.4f}  "
                    f"loss_std={float(record.terminal_loss_std):.4f}  "
                    f"m1={float(record.terminal_metric1_mean):.4f}  "
                    f"m2={float(record.terminal_metric2_mean):.4f}  "
                    f"m1_std={float(record.terminal_metric1_std):.4f}  "
                    f"m2_std={float(record.terminal_metric2_std):.4f}"
                ),
                (
                    f"terminal_stab_excess: "
                    f"m1={float(record.terminal_stab_excess_m1):.6f}  "
                    f"m2={float(record.terminal_stab_excess_m2):.6f}  "
                    f"loss={float(record.terminal_stab_excess_loss):.6f}  "
                    f"combined={float(record.terminal_stab_violation):.6f}"
                ),
                (
                    f"probe_runner: wall={float(record.terminal_probe_wall_seconds):.4f}s  "
                    f"devices={list(record.terminal_probe_devices)}  "
                    f"trial_counts={list(record.terminal_probe_trial_counts)}  "
                    f"trial_indices={list(record.terminal_probe_trial_indices)}  "
                    f"speedup={float(record.terminal_probe_speedup):.3f}x"
                ),
                (
                    f"adaptive_cost: score={float(record.terminal_cost_score):+.6f}  "
                    f"p3_margin={float(record.terminal_p3_metric_margin_reward):+.6f}  "
                    f"fusion_bonus={float(record.terminal_cost_fusion_bonus):+.6f}  "
                    f"trunc_bonus={float(record.terminal_cost_truncation_bonus):+.6f}  "
                    f"bits_tiebreaker={float(record.terminal_cost_bits_tiebreaker):+.6f}  "
                    f"trunc_units={float(record.terminal_cost_truncation_step_gain):+.3f}"
                ),
                (
                    f"cost_rank: unbounded={float(record.terminal_cost_rank_score):+.6f}  "
                    f"fusion={float(record.terminal_cost_rank_fusion):+.6f}  "
                    f"trunc={float(record.terminal_cost_rank_truncation):+.6f}  "
                    f"bits={float(record.terminal_cost_rank_bits):+.6f}"
                ),
                (
                    f"pareto_diag: event={record.terminal_pareto_event_kind or 'none'}  "
                    f"score={float(record.terminal_cost_score):+.6f}  "
                    f"fusion_gain={float(record.terminal_fusion_gain):+.3f}  "
                    f"k_gain={float(record.terminal_k_gain):+.3f}  "
                    f"bits_gain={float(record.terminal_bits_gain):+.3f}  "
                    f"removed={int(record.terminal_pareto_frontier_removed)}"
                ),
                (
                    f"safe_neighbor: active={bool(record.safe_neighbor_active)}  "
                    f"mutated_offsets={int(record.safe_neighbor_mutation_count)}  "
                    f"radius={int(record.safe_neighbor_radius)}"
                ),
                (
                    f"prior/proposal: baseline_prior_scale={float(record.baseline_prior_scale):.4f}  "
                    f"base_action_source={record.base_action_source or 'none'}  "
                    f"proposal_direction={record.proposal_direction or 'none'}  "
                    f"frontier_seed_episode={int(record.frontier_seed_episode)}  "
                    f"offset_success_rate={float(record.empirical_offset_success_rate):.3f}  "
                    f"offset_failure_rate={float(record.empirical_offset_failure_rate):.3f}"
                ),
                (
                    f"exploration: mode={record.exploration_mode or 'none'}  "
                    f"guarded_r2={bool(record.guarded_radius2_active)}  "
                    f"frontier_expansions@stall={int(record.guarded_radius2_recent_frontier_expansions)}  "
                    f"dup_rate={float(record.guarded_radius2_recent_duplicate_rate):.3f}  "
                    f"dom_rate={float(record.guarded_radius2_recent_dominated_rate):.3f}  "
                    f"cooldown={int(record.guarded_radius2_cooldown_remaining)}"
                ),
                (
                    f"invalid_masks: static_disabled={int(record.static_invalid_level_disabled)}  "
                    f"static_applied={int(record.static_invalid_level_applied)}  "
                    f"static_scan={int(record.static_invalid_level_scan_invalid)}/"
                    f"{int(record.static_invalid_level_scan_evaluated)} invalid/evaluated  "
                    f"empirical_disabled={int(record.empirical_invalid_level_disabled)}  "
                    f"empirical_applied={int(record.empirical_invalid_level_applied)}"
                ),
                (
                    "first_invalid="
                    + (
                        "none"
                        if record.first_invalid_step is None
                        else f"step{record.first_invalid_step} (L{record.first_invalid_layer}-B{record.first_invalid_block})"
                    )
                ),
            ]
            if record.invalid_block_details:
                extra_lines.append(
                    f"invalid_blocks ({len(record.invalid_block_details)}):"
                )
                for d in record.invalid_block_details:
                    extra_lines.append(
                        f"  · step{int(d.get('step', -1)):02d} "
                        f"L{int(d.get('layer', -1)):02d}-B{int(d.get('block', -1))} "
                        f"graph={d.get('graph_key', '')}  "
                        f"reason={d.get('reason', '(none)')}"
                    )
            details_writer.append_episode(
                episode=int(start_episode + record.episode_idx + 1),
                episode_return=float(record.total_reward),
                priority=int(priority),
                invalid=bool(record.invalid_steps > 0),
                opt_signals={
                    "total_bits_sum": int(record.total_bits_sum_over_steps),
                    "total_fusion_count": int(record.fusion_count_sum_over_steps),
                    "any_invalid": bool(record.invalid_steps > 0),
                },
                extra_lines=extra_lines,
            )
        except Exception as exc:
            log(f"  [details][warning] append_episode failed: {exc}")

        # First episode summary on its own line
        if record.episode_idx == 0:
            log(
                f"  [ep0 summary] total_return={record.total_reward:+.4f}  "
                f"terminal_reward={record.terminal_reward:+.4f}  "
                f"per_step_sum={record.per_step_reward_sum:+.4f}  "
                f"valid={record.valid_step_count}/{record.steps_taken}  "
                f"invalid={record.invalid_steps}  "
                f"first_invalid="
                f"{'-' if record.first_invalid_step is None else f'step{record.first_invalid_step} (L{record.first_invalid_layer}-B{record.first_invalid_block})'}"
            )

        # Track best with the same bounded Stage-1-style reward PPO optimizes.
        # Hard gates still keep P3 > P2 > P1; unbounded cost rank is only the
        # tie-breaker inside P3.
        current_rank_key = _episode_best_rank_key(record)
        is_new_best = (not best_rank_key) or current_rank_key > best_rank_key
        if is_new_best:
            best_rank_key = tuple(current_rank_key)
            best_reward = float(record.total_reward)
            best_action_vec = (
                None if record_full_vec is None
                else np.asarray(record_full_vec, dtype=np.int64).copy()
            )
            best_record = record
            # Episode-new-best banner (v2-style: "回合 N · 训练过程新高")
            _seq_block_title(
                log,
                f"回合 {start_episode + record.episode_idx + 1} · 训练过程新高（episode new best）",
            )
            log(
                f"  {bullet} 总回报 total_reward={record.total_reward:+.4f}  │  "
                f"终局奖励 terminal_reward={record.terminal_reward:+.4f}  │  "
                f"valid_steps={record.valid_step_count}/{record.steps_taken}  │  "
                f"invalid_steps={record.invalid_steps}"
            )
            # 2026-05-18 (rdv2 hotfix): surface inference test metrics on
            # every new best so the user can verify acc/stab gates directly
            # without grepping details/ files. m1/m2 for MRPC are accuracy/F1.
            _prio_label = (
                "P1(acc/invalid)" if int(record.terminal_priority) == 1
                else "P2(stab)" if int(record.terminal_priority) == 2
                else "P3(cost)" if int(record.terminal_priority) == 3
                else "P?"
            )
            log(
                f"  {bullet} 推理指标（inference test metrics）："
                f"loss_mean={record.terminal_loss_mean:.4f}  "
                f"loss_std={record.terminal_loss_std:.4f}  "
                f"m1(metric1)={record.terminal_metric1_mean:.4f}  "
                f"m2(metric2)={record.terminal_metric2_mean:.4f}  "
                f"m1_std={record.terminal_metric1_std:.4f}  "
                f"m2_std={record.terminal_metric2_std:.4f}  "
                f"stab_excess={record.terminal_stab_violation:.6f}  "
                f"priority={_prio_label}  "
                f"total_bits={record.total_bits_sum_over_steps}  "
                f"fusion={record.fusion_count_sum_over_steps}"
            )
            if best_action_vec is not None:
                log("  " + bullet + " 当前 best action（decoded slots；不输出 action index）：")
                for snippet_line in _format_best_action_slots(best_action_vec):
                    log("      " + snippet_line)

        # Periodic checkpoint
        if (record.episode_idx + 1) % int(train_cfg.save_interval) == 0:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                payload = {
                    "policy": policy.state_dict(),
                    "policy_ppo_aux": policy.ppo_aux_state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "episode": int(start_episode + record.episode_idx + 1),
                    "best_reward": float(best_reward),
                    "best_action": (
                        best_action_vec.tolist() if best_action_vec is not None else None
                    ),
                    "best_rank_key": [float(x) for x in best_rank_key],
                    "rl_variant": seq_rl_variant,
                    # Persist the forbidden-action mask so the next resume
                    # doesn't have to re-discover the same invalid tuples.
                    "forbidden_mask_records": forbidden_mask.to_json_records(),
                    "static_invalid_level_mask_records": (
                        static_invalid_mask.to_json_records()
                        if static_invalid_mask is not None else []
                    ),
                    "static_invalid_level_scan_summary": dict(static_invalid_scan_summary),
                    "empirical_invalid_level_mask_records": (
                        empirical_invalid_mask.to_json_records()
                        if empirical_invalid_mask is not None else []
                    ),
                }
                tmp = save_path + ".tmp"
                torch.save(payload, tmp)
                os.replace(tmp, save_path)
                log(
                    f"  [checkpoint] 已保存 · 回合 {start_episode + record.episode_idx + 1} "
                    f"→ {save_path}  ·  {forbidden_mask.summary()}  ·  "
                    f"{static_invalid_mask.summary() if static_invalid_mask is not None else 'static_invalid_level_mask=disabled'}  ·  "
                    f"{empirical_invalid_mask.summary() if empirical_invalid_mask is not None else 'empirical_invalid_level_mask=disabled'}"
                )
            except Exception as exc:
                log(f"  [save_checkpoint][警告] 保存 {save_path} 失败: {exc}")

        try:
            status.update_after_episode(
                int(start_episode + record.episode_idx + 1),
                float(record.total_reward),
                {
                    "invalid_steps": int(record.invalid_steps),
                    "valid_step_count": int(record.valid_step_count),
                    "early_terminated": bool(record.early_terminated),
                    "steps_taken": int(record.steps_taken),
                    "terminal_reward": float(record.terminal_reward),
                    "terminal_metric2_mean": float(record.terminal_metric2_mean),
                    "terminal_metric1_std": float(record.terminal_metric1_std),
                    "terminal_metric2_std": float(record.terminal_metric2_std),
                    "terminal_stab_violation": float(record.terminal_stab_violation),
                    "terminal_bits_gain": float(record.terminal_bits_gain),
                    "terminal_k_gain": float(record.terminal_k_gain),
                    "terminal_fusion_gain": float(record.terminal_fusion_gain),
                    "terminal_cost_score": float(record.terminal_cost_score),
                    "terminal_p3_metric_margin_reward": float(
                        record.terminal_p3_metric_margin_reward
                    ),
                    "terminal_cost_fusion_bonus": float(record.terminal_cost_fusion_bonus),
                    "terminal_cost_truncation_bonus": float(
                        record.terminal_cost_truncation_bonus
                    ),
                    "terminal_cost_bits_tiebreaker": float(
                        record.terminal_cost_bits_tiebreaker
                    ),
                    "terminal_cost_truncation_step_gain": float(
                        record.terminal_cost_truncation_step_gain
                    ),
                    "terminal_cost_rank_score": float(record.terminal_cost_rank_score),
                    "terminal_cost_rank_fusion": float(record.terminal_cost_rank_fusion),
                    "terminal_cost_rank_truncation": float(
                        record.terminal_cost_rank_truncation
                    ),
                    "terminal_cost_rank_bits": float(record.terminal_cost_rank_bits),
                    "terminal_pareto_event_kind": str(record.terminal_pareto_event_kind),
                    "best_rank_key": [float(x) for x in best_rank_key],
                    "best_reward": float(best_reward),
                    "safe_neighbor_active": bool(record.safe_neighbor_active),
                    "safe_neighbor_mutation_count": int(record.safe_neighbor_mutation_count),
                    "safe_neighbor_radius": int(record.safe_neighbor_radius),
                    "exploration_mode": str(record.exploration_mode),
                    "guarded_radius2_active": bool(record.guarded_radius2_active),
                    "guarded_radius2_recent_frontier_expansions": int(
                        record.guarded_radius2_recent_frontier_expansions
                    ),
                    "guarded_radius2_cooldown_remaining": int(
                        record.guarded_radius2_cooldown_remaining
                    ),
                    "baseline_prior_scale": float(record.baseline_prior_scale),
                    "base_action_source": str(record.base_action_source),
                    "proposal_direction": str(record.proposal_direction),
                    "empirical_offset_success_rate": float(record.empirical_offset_success_rate),
                    "empirical_offset_failure_rate": float(record.empirical_offset_failure_rate),
                    "samples_rejected_by_mask": int(record.samples_rejected_by_mask),
                    "samples_rejected_by_optimizer": int(record.samples_rejected_by_optimizer),
                    "steps_fallen_back_to_baseline": int(record.steps_fallen_back_to_baseline),
                    "forbidden_mask_total": int(record.forbidden_mask_total),
                    "static_invalid_level_disabled": int(record.static_invalid_level_disabled),
                    "static_invalid_level_applied": int(record.static_invalid_level_applied),
                    "static_invalid_level_scan_evaluated": int(
                        record.static_invalid_level_scan_evaluated
                    ),
                    "static_invalid_level_scan_invalid": int(
                        record.static_invalid_level_scan_invalid
                    ),
                    "empirical_invalid_level_disabled": int(record.empirical_invalid_level_disabled),
                    "rejection_optimizer_wall_seconds": float(record.rejection_optimizer_wall_seconds),
                },
            )
        except Exception:
            pass

        # --- Long-term diagnostics: per-episode JSONL + top-K + heatmap update.
        # The recorder owns its own try/except internally; we still wrap to keep
        # training resilient if the dataclass schema ever drifts.
        try:
            diag_recorder.record_episode(
                episode_stats=EpisodeStats(
                    episode=int(start_episode + record.episode_idx),
                    total_reward=float(record.total_reward),
                    terminal_reward=float(record.terminal_reward),
                    per_step_sum=float(record.per_step_reward_sum),
                    valid_steps=int(record.valid_step_count),
                    invalid_steps=int(record.invalid_steps),
                    steps_taken=int(record.steps_taken),
                    total_bits=int(record.total_bits_sum_over_steps),
                    fusion_count=int(record.fusion_count_sum_over_steps),
                    fusion_count_b2=int(record.fusion_count_b2),
                    fusion_count_b4=int(record.fusion_count_b4),
                    fusion_count_b5=int(record.fusion_count_b5),
                    terminal_final_config_fingerprint=str(
                        record.terminal_final_config_fingerprint
                    ),
                    terminal_materialization_failure_reason=str(
                        record.terminal_materialization_failure_reason
                    ),
                    terminal_model_uses_replan_config=bool(
                        record.terminal_model_uses_replan_config
                    ),
                    first_invalid_step=(
                        int(record.first_invalid_step)
                        if record.first_invalid_step is not None else None
                    ),
                    first_invalid_block=(
                        int(record.first_invalid_block)
                        if record.first_invalid_block is not None else None
                    ),
                    first_invalid_layer=(
                        int(record.first_invalid_layer)
                        if record.first_invalid_layer is not None else None
                    ),
                    early_terminated=bool(record.early_terminated),
                    terminal_priority=int(record.terminal_priority),
                    terminal_loss_mean=float(record.terminal_loss_mean),
                    terminal_loss_std=float(record.terminal_loss_std),
                    terminal_metric1_mean=float(record.terminal_metric1_mean),
                    terminal_metric2_mean=float(record.terminal_metric2_mean),
                    terminal_metric1_std=float(record.terminal_metric1_std),
                    terminal_metric2_std=float(record.terminal_metric2_std),
                    terminal_stab_excess_m1=float(record.terminal_stab_excess_m1),
                    terminal_stab_excess_m2=float(record.terminal_stab_excess_m2),
                    terminal_stab_excess_loss=float(record.terminal_stab_excess_loss),
                    terminal_stab_violation=float(record.terminal_stab_violation),
                    terminal_bits_gain=float(record.terminal_bits_gain),
                    terminal_k_gain=float(record.terminal_k_gain),
                    terminal_fusion_gain=float(record.terminal_fusion_gain),
                    terminal_cost_score=float(record.terminal_cost_score),
                    terminal_p3_metric_margin_reward=float(
                        record.terminal_p3_metric_margin_reward
                    ),
                    terminal_worst_signed_margin=float(record.terminal_worst_signed_margin),
                    terminal_acc_barrier_sat=float(record.terminal_acc_barrier_sat),
                    terminal_acc_barrier_vio=float(record.terminal_acc_barrier_vio),
                    terminal_near_miss=bool(record.terminal_near_miss),
                    terminal_margin_m1=float(record.terminal_margin_m1),
                    terminal_margin_m2=float(record.terminal_margin_m2),
                    terminal_fusion_norm_raw=float(record.terminal_fusion_norm_raw),
                    terminal_fusion_norm_saturated=float(record.terminal_fusion_norm_saturated),
                    terminal_cost_fusion_bonus=float(record.terminal_cost_fusion_bonus),
                    terminal_cost_truncation_bonus=float(record.terminal_cost_truncation_bonus),
                    terminal_cost_bits_tiebreaker=float(record.terminal_cost_bits_tiebreaker),
                    terminal_cost_truncation_step_gain=float(
                        record.terminal_cost_truncation_step_gain
                    ),
                    terminal_cost_rank_score=float(record.terminal_cost_rank_score),
                    terminal_cost_rank_fusion=float(record.terminal_cost_rank_fusion),
                    terminal_cost_rank_truncation=float(record.terminal_cost_rank_truncation),
                    terminal_cost_rank_bits=float(record.terminal_cost_rank_bits),
                    terminal_pareto_event_kind=str(record.terminal_pareto_event_kind),
                    terminal_pareto_action_hash=str(record.terminal_pareto_action_hash),
                    terminal_pareto_frontier_removed=int(record.terminal_pareto_frontier_removed),
                    terminal_probe_wall_seconds=float(record.terminal_probe_wall_seconds),
                    terminal_probe_devices=list(record.terminal_probe_devices),
                    terminal_probe_trial_counts=list(record.terminal_probe_trial_counts),
                    terminal_probe_trial_indices=[
                        list(x) for x in record.terminal_probe_trial_indices
                    ],
                    terminal_probe_speedup=float(record.terminal_probe_speedup),
                    fusion_action_steps=[
                        dict(x) for x in (record.fusion_action_steps or [])
                        if isinstance(x, Mapping)
                    ],
                    per_step_optimizer_wall_seconds=float(record.per_step_optimizer_wall_seconds),
                    policy_rollout_wall_seconds=float(record.policy_rollout_wall_seconds),
                    terminal_cost_eval_wall_seconds=float(record.terminal_cost_eval_wall_seconds),
                    terminal_probe_install_wall_seconds=float(record.terminal_probe_install_wall_seconds),
                    terminal_probe_clear_wall_seconds=float(record.terminal_probe_clear_wall_seconds),
                    terminal_probe_install_skipped=bool(record.terminal_probe_install_skipped),
                    terminal_probe_clear_skipped=bool(record.terminal_probe_clear_skipped),
                    safe_neighbor_active=bool(record.safe_neighbor_active),
                    safe_neighbor_mutation_count=int(record.safe_neighbor_mutation_count),
                    safe_neighbor_radius=int(record.safe_neighbor_radius),
                    exploration_mode=str(record.exploration_mode),
                    guarded_radius2_active=bool(record.guarded_radius2_active),
                    guarded_radius2_recent_frontier_expansions=int(record.guarded_radius2_recent_frontier_expansions),
                    guarded_radius2_recent_duplicate_rate=float(record.guarded_radius2_recent_duplicate_rate),
                    guarded_radius2_recent_dominated_rate=float(record.guarded_radius2_recent_dominated_rate),
                    guarded_radius2_cooldown_remaining=int(record.guarded_radius2_cooldown_remaining),
                    guarded_radius2_safe_offset_count=int(record.guarded_radius2_safe_offset_count),
                    guarded_radius2_episode_count=int(record.guarded_radius2_episode_count),
                    guarded_radius2_failure_count=int(record.guarded_radius2_failure_count),
                    guarded_radius2_frontier_expansion_count=int(record.guarded_radius2_frontier_expansion_count),
                    samples_rejected_by_mask=int(record.samples_rejected_by_mask),
                    samples_rejected_by_optimizer=int(record.samples_rejected_by_optimizer),
                    steps_fallen_back_to_baseline=int(record.steps_fallen_back_to_baseline),
                    forbidden_mask_total=int(record.forbidden_mask_total),
                    static_invalid_level_disabled=int(record.static_invalid_level_disabled),
                    static_invalid_level_applied=int(record.static_invalid_level_applied),
                    static_invalid_level_scan_evaluated=int(
                        record.static_invalid_level_scan_evaluated
                    ),
                    static_invalid_level_scan_invalid=int(
                        record.static_invalid_level_scan_invalid
                    ),
                    empirical_invalid_level_disabled=int(record.empirical_invalid_level_disabled),
                    empirical_invalid_level_applied=int(record.empirical_invalid_level_applied),
                    rejection_optimizer_wall_seconds=float(record.rejection_optimizer_wall_seconds),
                    baseline_prior_scale=float(record.baseline_prior_scale),
                    base_action_source=str(record.base_action_source),
                    proposal_direction=str(record.proposal_direction),
                    empirical_offset_success_rate=float(record.empirical_offset_success_rate),
                    empirical_offset_failure_rate=float(record.empirical_offset_failure_rate),
                    frontier_seed_episode=int(record.frontier_seed_episode),
                ),
                full_action_vec=(
                    None if record_full_vec is None
                    else np.asarray(record_full_vec, dtype=np.int64)
                ),
                is_new_best=bool(is_new_best),
                best_reward_so_far=float(best_reward),
            )
        except Exception as exc:
            log(f"  [diag][warning] record_episode failed: {exc}")
        # Flush the heavy artifacts (summary.md / npz / top jsonl) on the same
        # cadence as the model checkpoint — cheap enough at 200-episode intervals.
        if (record.episode_idx + 1) % int(train_cfg.save_interval) == 0:
            try:
                diag_recorder.flush_periodic()
                log(
                    f"  [diag] 诊断摘要已刷新 → "
                    f"{diag_recorder.summary_md_path}"
                )
            except Exception as exc:
                log(f"  [diag][warning] flush_periodic failed: {exc}")

    def _write_live_training_curves(
            *,
            completed_episodes: int,
            force: bool = False,
            ) -> None:
        # live_curve_refresh: keep Stage-2's in-flight curve in the same
        # Stage-1-style format as the final curve, without redrawing every PPO
        # update. Default cadence: every progress box (5 PPO updates) and final.
        if not force and ppo_update_counter[0] % SEQ_PROGRESS_BOX_PPO_INTERVAL != 0:
            return
        records = list(live_episode_records)
        if not records:
            return
        try:
            ep_returns = [
                float(getattr(r, "total_reward", 0.0) or 0.0)
                for r in records
            ]
            ep_losses = [
                float(getattr(r, "terminal_loss_mean", 0.0) or 0.0)
                for r in records
            ]
            ep_m1 = [
                float(getattr(r, "terminal_metric1_mean", 0.0) or 0.0)
                for r in records
            ]
            ep_m2 = [
                float(getattr(r, "terminal_metric2_mean", 0.0) or 0.0)
                for r in records
            ]
            ep_fusion = [
                float(getattr(r, "fusion_count_sum_over_steps", 0) or 0)
                for r in records
            ]
            base_avg_k = float(getattr(baseline, "avg_k", 13.0) or 13.0)
            ep_avgk = [
                base_avg_k - float(getattr(r, "terminal_k_gain", 0.0) or 0.0)
                for r in records
            ]
            ent = [
                float(m.get("entropy"))
                for m in live_ppo_metrics
                if m.get("entropy") is not None
            ]
            ent_eps = [
                float(m.get("completed_episodes", 0) or 0)
                for m in live_ppo_metrics
                if m.get("entropy") is not None
            ]
            curve_paths = write_training_curves(
                blb_progress_dir,
                episode_returns=ep_returns,
                episode_losses=ep_losses or None,
                episode_metric1s=ep_m1 or None,
                episode_metric2s=ep_m2 or None,
                episode_fusion_counts=ep_fusion or None,
                episode_avg_ks=ep_avgk or None,
                baselines={
                    "loss": float(getattr(baseline, "loss_mean", 0.0) or 0.0),
                    "metric1": float(getattr(baseline, "metric1_mean", 0.0) or 0.0),
                    "metric2": float(getattr(baseline, "metric2_mean", 0.0) or 0.0),
                    "avg_k": base_avg_k,
                },
                entropy_series=ent or None,
                entropy_episodes=ent_eps or None,
                log_fn=log,
            )
            if curve_paths.get("png"):
                log(
                    f"  [live_curve_refresh] Stage-1-style curve refreshed "
                    f"@ episode {int(completed_episodes)} → {curve_paths['png']}"
                )
        except Exception as exc:
            log(f"  [live_curve_refresh][warning] failed: {exc}")

    def _ppo_update_end_callback(
            metrics: Dict[str, float],
            completed_episodes: int,
            last_record: EpisodeRecord,
            ) -> None:
        ppo_update_counter[0] += 1
        # Per-PPO-update window summary box
        win_n = max(1, len(rollout_avg_window))
        avg_ret = float(np.mean(rollout_avg_window)) if rollout_avg_window else 0.0
        max_ret = float(np.max(rollout_avg_window)) if rollout_avg_window else 0.0
        min_ret = float(np.min(rollout_avg_window)) if rollout_avg_window else 0.0
        avg_inv = float(np.mean(rollout_invalid_window)) if rollout_invalid_window else 0.0
        avg_valid = float(np.mean(rollout_valid_window)) if rollout_valid_window else 0.0
        avg_term = float(np.mean(rollout_terminal_window)) if rollout_terminal_window else 0.0
        is_last_update = (
            start_episode + completed_episodes >= total_episodes_planned
        )
        # Wall-clock timestamp helps correlate window summaries with ops events
        # (kill -USR1, OOM, GPU thermals) when reading the log days later. Cost
        # is negligible compared to the PPO update itself.
        now_ts = time.strftime("%Y-%m-%d %H:%M:%S")
        summary_lines = [
            f"PPO 窗口摘要 · 截至回合（through episode） "
            f"{start_episode + completed_episodes}  ·  时间 {now_ts}",
            f"窗口 N={win_n} 回合 ·  "
            f"平均回报 mean return={avg_ret:+.4f} (min={min_ret:+.4f}, max={max_ret:+.4f})  ·  "
            f"平均终局 mean terminal={avg_term:+.4f}",
            f"平均 valid steps={avg_valid:.2f}/{last_record.steps_taken}  ·  "
            f"平均 invalid={avg_inv:.2f}  ·  "
            f"当前 rank-best reward={best_reward:+.4f}  ·  "
            f"rank_key={list(best_rank_key) if best_rank_key else []}",
            f"policy_loss={metrics.get('policy_loss', 0.0):+.4f}  ·  "
            f"value_loss={metrics.get('value_loss', 0.0):+.4f}  ·  "
            f"entropy={metrics.get('entropy', 0.0):+.4f}  ·  "
            f"clip_fraction={metrics.get('clip_fraction', 0.0):.3f}  ·  "
            f"ent_coef={metrics.get('ent_coef', 0.0):.5f}",
            f"approx_kl={metrics.get('approx_kl', 0.0):.5f}  ·  "
            f"kl_stop={bool(metrics.get('kl_early_stop', False))}  ·  "
            f"entropy_recovery={metrics.get('entropy_recovery_delta', 0.0):.5f}  ·  "
            f"return_norm=({metrics.get('return_mean', 0.0):+.3f}, {metrics.get('return_std', 1.0):.3f})",
            f"LR={metrics.get('lr', optimizer.param_groups[0]['lr']):.6f}  ·  "
            f"lr_scale={metrics.get('lr_scale', 1.0):.3f}  ·  "
            f"更新序号 update#{ppo_update_counter[0]}  ·  "
            f"PPO 样本数={int(metrics.get('n_samples', 0))}",
        ]
        log("")
        for idx, line in enumerate(summary_lines):
            log(("  " if idx == 0 else "    ") + line)
        # Persist PPO-update diagnostics before clearing the rolling window.
        try:
            diag_recorder.record_ppo_update(PPOUpdateStats(
                update=int(ppo_update_counter[0]),
                completed_episodes=int(start_episode + completed_episodes),
                policy_loss=float(metrics.get("policy_loss", 0.0)),
                value_loss=float(metrics.get("value_loss", 0.0)),
                entropy=float(metrics.get("entropy", 0.0)),
                clip_fraction=float(metrics.get("clip_fraction", 0.0)),
                n_samples=int(metrics.get("n_samples", 0)),
                window_mean_return=float(avg_ret),
                window_max_return=float(max_ret),
                window_min_return=float(min_ret),
                window_mean_invalid=float(avg_inv),
                best_reward_so_far=float(best_reward),
                elapsed_sec=float(time.time() - t_start),
                ent_coef=float(metrics.get("ent_coef", 0.0)),
                approx_kl=float(metrics.get("approx_kl", 0.0)),
                kl_early_stop=bool(metrics.get("kl_early_stop", False)),
                lr=float(metrics.get("lr", optimizer.param_groups[0]["lr"])),
                lr_scale=float(metrics.get("lr_scale", 1.0)),
                entropy_recovery_delta=float(metrics.get("entropy_recovery_delta", 0.0)),
                return_mean=float(metrics.get("return_mean", 0.0)),
                return_std=float(metrics.get("return_std", 1.0)),
            ))
        except Exception as exc:
            log(f"  [diag][warning] record_ppo_update failed: {exc}")

        try:
            status.update_after_ppo_update(
                int(ppo_update_counter[0]),
                {
                    "completed_episodes": int(start_episode + completed_episodes),
                    "policy_loss": float(metrics.get("policy_loss", 0.0)),
                    "value_loss": float(metrics.get("value_loss", 0.0)),
                    "entropy": float(metrics.get("entropy", 0.0)),
                    "clip_fraction": float(metrics.get("clip_fraction", 0.0)),
                    "n_samples": int(metrics.get("n_samples", 0)),
                    "window_mean_return": float(avg_ret),
                    "window_max_return": float(max_ret),
                    "window_min_return": float(min_ret),
                    "window_mean_invalid": float(avg_inv),
                    "best_reward_so_far": float(best_reward),
                    "elapsed_sec": float(time.time() - t_start),
                    "ent_coef": float(metrics.get("ent_coef", 0.0)),
                    "approx_kl": float(metrics.get("approx_kl", 0.0)),
                    "kl_early_stop": bool(metrics.get("kl_early_stop", False)),
                    "lr": float(metrics.get("lr", optimizer.param_groups[0]["lr"])),
                    "lr_scale": float(metrics.get("lr_scale", 1.0)),
                    "entropy_recovery_delta": float(
                        metrics.get("entropy_recovery_delta", 0.0)
                    ),
                    "return_mean": float(metrics.get("return_mean", 0.0)),
                    "return_std": float(metrics.get("return_std", 1.0)),
                },
            )
        except Exception as exc:
            log(f"  [status][warning] update_after_ppo_update failed: {exc}")

        live_ppo_metrics.append({
            "completed_episodes": int(start_episode + completed_episodes),
            "policy_loss": float(metrics.get("policy_loss", 0.0)),
            "value_loss": float(metrics.get("value_loss", 0.0)),
            "entropy": float(metrics.get("entropy", 0.0)),
            "clip_fraction": float(metrics.get("clip_fraction", 0.0)),
            "window_mean_return": float(avg_ret),
            "window_mean_invalid": float(avg_inv),
        })
        _write_live_training_curves(
            completed_episodes=int(start_episode + completed_episodes),
            force=bool(is_last_update),
        )

        # Reward-crash watcher: compare this PPO rollout's mean return to the
        # previous rollout's. If it dropped > drop_threshold (default 0.3), an
        # entry is appended to <noise_root>/warning.txt pointing at the current
        # details batch file. Matches legacy noise_rl_module_v2 warning.txt
        # semantics so the user can still grep for collapse events.
        try:
            crash_watcher.observe_rollout(
                rollout_mean=float(avg_ret),
                episode_start=int(start_episode + completed_episodes - win_n + 1),
                episode_end=int(start_episode + completed_episodes),
                details_path=str(details_writer.current_batch_path),
                phase_label="BLB Stage-2 RL · sequential（v3）",
            )
        except Exception as exc:
            log(f"  [crash-watch][warning] observe_rollout failed: {exc}")

        # clear the rolling window for the next PPO interval
        rollout_avg_window.clear()
        rollout_invalid_window.clear()
        rollout_valid_window.clear()
        rollout_terminal_window.clear()

        # Periodic big progress box (every SEQ_PROGRESS_BOX_PPO_INTERVAL updates,
        # or on the last update of the run).
        if (
            ppo_update_counter[0] % SEQ_PROGRESS_BOX_PPO_INTERVAL == 0
            or is_last_update
        ):
            elapsed_now = time.time() - t_start
            done_this_run = float(completed_episodes)
            avg_ep_time = elapsed_now / max(done_this_run, 1.0)
            remaining_ep = float(total_episodes_planned - (start_episode + completed_episodes))
            eta_seconds = avg_ep_time * max(remaining_ep, 0.0)
            progress_lines = [
                f"BLB Stage-2 噪声 RL 进度 · 回合 "
                f"{start_episode + completed_episodes} / {total_episodes_planned}",
                _seq_progress_bar(
                    int(start_episode + completed_episodes),
                    int(total_episodes_planned),
                ),
                f"训练期 rank-best 得分: {best_reward:+.4f}",
                f"已用时: {_seq_fmt_elapsed(elapsed_now)}    "
                f"预计剩余: {_seq_fmt_elapsed(eta_seconds)}    "
                f"预计完成: {_seq_fmt_eta_finish(eta_seconds)}    "
                f"PPO 更新: {ppo_update_counter[0]} 次",
            ]
            _seq_log_rounded_box(log, progress_lines)

    t_start = time.time()
    # Forbidden-action mask: starts empty (or rehydrated from checkpoint
    # `forbidden_mask_records` if present in the resumed checkpoint).
    forbidden_mask = ForbiddenActionMask()
    # Fusion mode: the offline map holds only valid configs, so invalid-level
    # masks (and their per-slot feasibility scan, which assumes BlockStepSpec) are
    # both unnecessary and incompatible — disable them.
    static_invalid_mask = (
        StaticInvalidLevelMask()
        if (bool(getattr(seq_train_cfg, "static_invalid_level_mask_enabled", False)) and fusion_map is None)
        else None
    )
    static_invalid_scan_summary: Dict[str, Any] = {
        "enabled": bool(static_invalid_mask is not None),
        "evaluated": 0,
        "invalid": 0,
        "disabled": 0,
        "aborted": False,
        "reason": "",
        "elapsed_seconds": 0.0,
        "source": "none",
    }
    empirical_invalid_mask = EmpiricalInvalidLevelMask(
        min_invalid_samples=int(
            getattr(seq_train_cfg, "empirical_invalid_level_min_samples", 3)
        ),
        min_invalid_rate=float(
            getattr(seq_train_cfg, "empirical_invalid_level_min_rate", 0.80)
        ),
        max_valid_samples=int(
            getattr(seq_train_cfg, "empirical_invalid_level_max_valid", 0)
        ),
    ) if (bool(getattr(seq_train_cfg, "empirical_invalid_level_mask_enabled", False)) and fusion_map is None) else None
    if effective_resume_path and os.path.isfile(effective_resume_path):
        try:
            _ckpt = torch.load(effective_resume_path, map_location=device)
            rec = _ckpt.get("forbidden_mask_records") if isinstance(_ckpt, dict) else None
            if rec:
                forbidden_mask = ForbiddenActionMask.from_json_records(rec)
                log(
                    f"  {bullet} 已从 checkpoint 恢复 forbidden_action_mask: "
                    f"{forbidden_mask.summary()}"
                )
            static_rec = (
                _ckpt.get("static_invalid_level_mask_records")
                if isinstance(_ckpt, dict) else None
            )
            if static_rec and static_invalid_mask is not None:
                static_invalid_mask = StaticInvalidLevelMask.from_json_records(static_rec)
                static_invalid_scan_summary = dict(
                    _ckpt.get("static_invalid_level_scan_summary") or {}
                )
                static_invalid_scan_summary["source"] = "checkpoint"
                log(
                    f"  {bullet} 已从 checkpoint 恢复 static_invalid_level_mask: "
                    f"{static_invalid_mask.summary()}"
                )
            empirical_rec = (
                _ckpt.get("empirical_invalid_level_mask_records")
                if isinstance(_ckpt, dict) else None
            )
            if empirical_rec and empirical_invalid_mask is not None:
                empirical_invalid_mask = EmpiricalInvalidLevelMask.from_json_records(empirical_rec)
                empirical_invalid_mask.min_invalid_samples = int(
                    getattr(seq_train_cfg, "empirical_invalid_level_min_samples", 3)
                )
                empirical_invalid_mask.min_invalid_rate = float(
                    getattr(seq_train_cfg, "empirical_invalid_level_min_rate", 0.80)
                )
                empirical_invalid_mask.max_valid_samples = int(
                    getattr(seq_train_cfg, "empirical_invalid_level_max_valid", 0)
                )
                log(
                    f"  {bullet} 已从 checkpoint 恢复 empirical_invalid_level_mask: "
                    f"{empirical_invalid_mask.summary()}"
                )
        except Exception as exc:
            log(f"  [resume][warning] failed to restore forbidden_mask: {exc}")

    if static_invalid_mask is not None and static_invalid_mask.total_disabled() == 0:
        static_invalid_mask, static_invalid_scan_summary = _precompute_static_invalid_level_mask(
            env=seq_env,
            baseline_action_vec=baseline_action_vec,
            enabled=True,
            log_fn=log,
            bullet=bullet,
        )
        static_invalid_scan_summary["source"] = "baseline_prefix_scan"

    # 2026-05-18 (rdv2 hotfix): force first N episodes to use the baseline
    # action so the value function calibrates around +45 (baseline reward)
    # and PPO pushes policy mass toward baseline before exploration starts.
    # Without this, the warmstart-biased policy still samples ~80% of slots
    # uniformly at random for kinds whose baseline index is not the bias
    # target — virtually no rollout matches baseline closely enough to
    # satisfy acc_threshold, and every reward collapses to ~-7. See
    # `reports/stage2_rl/bug_reports/2026-05-18_stage2_rl_rdv2_negative_reward_startup/`.
    # The fallback is now exactly 60 episodes, matching the non-monotonic
    # boundary-search curriculum.
    # Continuous reward still needs the configured baseline anchor. The
    # fusion-count action space can otherwise spend the first PPO windows in
    # P1/loss-cap candidates before the policy has seen a feasible return.
    _force_baseline_episodes = _resolve_sequential_force_baseline_episodes(train_cfg)
    log(
        f"  {bullet} 强制 baseline 锚点（forced-baseline anchor）: "
        f"前 {_force_baseline_episodes} 个 episode 直接执行 baseline action，"
        f"让 value head 学到 +45 基线 reward，policy 概率质量先聚到 baseline 附近，"
        f"之后再切到 PPO sample。"
    )

    seq_result = train_sequential(
        env=seq_env,
        policy=policy,
        train_cfg=seq_train_cfg,
        device=device,
        optimizer=optimizer,
        on_episode_end=_episode_callback,
        on_ppo_update_end=_ppo_update_end_callback,
        on_step_end=_step_callback,
        capture_step_infos=False,  # save memory; we surface aggregates instead
        logger=logging.getLogger("blb_stage2_rl.sequential"),
        forbidden_mask=forbidden_mask,
        static_invalid_mask=static_invalid_mask,
        static_invalid_scan_summary=static_invalid_scan_summary,
        empirical_invalid_mask=empirical_invalid_mask,
        baseline_action_vec=baseline_action_vec,
        max_rejection_retries=32,
        force_baseline_episodes=_force_baseline_episodes,
        parallel_runner=stage2_parallel_runner,
    )
    elapsed = float(time.time() - t_start)
    status.set_phase("已完成")

    # Flush any remaining buffered episode records into the current details/
    # batch file before reporting the final summary — otherwise the last
    # ``len(episode_returns) % details_batch_size`` records would never land
    # on disk (the writer auto-rolls on full batches, not on shutdown).
    try:
        flushed_path = details_writer.flush()
        if flushed_path:
            log(f"  {bullet} 详细诊断 final flush → {flushed_path}")
    except Exception as exc:
        log(f"  [details][warning] final flush failed: {exc}")

    _seq_log_major_rule(log, "PPO 训练结束（Sequential PPO training completed）")
    final_invalid_rate = float(seq_result.get("final_invalid_rate", 0.0))
    _seq_log_rounded_box(log, [
        f"总耗时（elapsed）：{_seq_fmt_elapsed(elapsed)}",
        f"完成回合数：{len(episode_returns)}（计划 {total_episodes_planned}）",
        f"PPO 更新次数：{ppo_update_counter[0]}",
        f"最近 10 回合 invalid 率：{final_invalid_rate*100:.1f}%",
        f"训练期 rank-best reward：{best_reward:+.4f}",
        (
            "训练期最优 best_action vec 已确定（写入 result['blb_v3_best_action_vec']，"
            "下游 final-eval 会安装该动作）"
            if best_action_vec is not None
            else "训练期最优 best_action vec 尚未产生"
        ),
    ])

    # Final flush of the diagnostics recorder — leaves summary.md in its
    # final form for post-hoc inspection.
    try:
        diag_recorder.finalize()
        log(f"  {bullet} 诊断目录已完成 → {diag_recorder.output_dir}")
        log(f"  {bullet} 诊断汇总 → {diag_recorder.summary_md_path}")
        if best_action_vec is not None:
            log(
                f"  {bullet} 手动 final-eval 可调用：\n"
                f"      bash Paean/run_final_eval.sh "
                f"--preset mrpc-final-eval-only "
                f"--action-config {diag_recorder.best_json_path}"
            )
    except Exception as exc:
        log(f"  [diag][warning] finalize failed: {exc}")

    # ---------- 7.5) Human-readable best/baseline action description files ----------
    # These are *separate* from the diagnostics/ dir: they live at the
    # blb_progress_dir root so legacy tooling that looks for
    # ``blb_stage2_best_action_full.{json,md}`` still finds them — but now
    # both files use the SF/K-first schema.
    from .action_space import describe_action_vector as _describe_action_vector
    from .persistence import write_action_description_files as _write_action_description_files
    # ADR-015: STRICT feasibility selection. Revalidate the ranked top-N before
    # publishing the best action, then choose the first strict-feasible candidate.
    # This keeps a slightly loss-failed rank-best from forcing a baseline fallback
    # when a lower-ranked candidate satisfies the constraints.
    best_fallback_to_baseline = False
    _loss_thr = getattr(base_env, "loss_threshold", None)
    _final_top_n = max(1, int(getattr(train_cfg, "final_selection_top_n", 20) or 20))
    _final_trials = max(
        int(getattr(train_cfg, "promotion_validation_trials", 1) or 1),
        int(getattr(train_cfg, "final_selection_validation_trials", 20) or 20),
    )
    _selection_records = list(live_episode_records)
    if best_record is not None and all(r is not best_record for r in _selection_records):
        _selection_records.append(best_record)
    def _final_selection_record_vec(record: Any) -> Optional[np.ndarray]:
        raw = getattr(record, "_pending_full_vec_for_callback", None)
        if raw is None:
            return None
        return np.asarray(raw, dtype=np.int64).copy()

    _selection_records = [
        r for r in _selection_records
        if _final_selection_record_vec(r) is not None
    ]
    _selection_candidates = sorted(
        _selection_records, key=_episode_best_rank_key, reverse=True,
    )[:_final_top_n]
    if _selection_candidates:
        log(
            f"  {bullet} [ADR-015] final strict selection: "
            f"revalidate top-{len(_selection_candidates)} with K={_final_trials} trials"
        )

    def _fusion_group_from_record_or_vec(record: Any, action_vec: np.ndarray) -> Optional[Dict[str, Any]]:
        if fusion_map is None:
            return None
        raw_steps = getattr(record, "fusion_action_steps", None) or []
        option_by_step: Dict[str, int] = {}
        choices_by_step: List[Dict[str, Any]] = []
        for item in raw_steps:
            if not isinstance(item, Mapping):
                continue
            if "step_idx" not in item or "option_id" not in item:
                continue
            step_key = str(int(item["step_idx"]))
            option_by_step[step_key] = int(item["option_id"])
            choices_by_step.append(dict(item))
        if option_by_step:
            return {
                "option_by_step": option_by_step,
                "choices_by_step": choices_by_step,
                "source": "episode_record_fusion_action_steps",
            }

        from .fusion_fixed_action import build_fusion_fixed_config
        cfg = build_fusion_fixed_config(
            np.asarray(action_vec, dtype=int),
            profile=str(train_cfg.profile),
            num_layers=int(ev.total_layers),
            gelu=np.asarray(fixed_gelu, dtype=int),
            softmax=np.asarray(fixed_softmax, dtype=int),
            fusion_map=fusion_map,
            source="final_strict_revalidation",
        )
        group = cfg.get("group")
        return group if isinstance(group, Mapping) else None

    for _rank, _candidate in enumerate(_selection_candidates, start=1):
        _candidate_vec = _final_selection_record_vec(_candidate)
        if _candidate_vec is None:
            log(
                f"  {bullet} [ADR-015] final candidate rank {_rank} "
                f"ep={getattr(_candidate, 'episode_idx', '?')} skipped: no action vec"
            )
            continue
        try:
            _candidate_boosted_overrides = None
            if fusion_map is not None:
                from .fusion_fixed_action import build_boosted_overrides_from_group
                _candidate_group = _fusion_group_from_record_or_vec(_candidate, _candidate_vec)
                if _candidate_group is not None:
                    _candidate_boosted_overrides = (
                        build_boosted_overrides_from_group(
                            np.asarray(_candidate_vec, dtype=np.int64),
                            group=_candidate_group,
                            fusion_map=fusion_map,
                            num_layers=int(ev.total_layers),
                            profile=str(train_cfg.profile),
                            gelu=np.asarray(fixed_gelu, dtype=int),
                            softmax=np.asarray(fixed_softmax, dtype=int),
                        ) or None
                    )
            _prepared = base_env.prepare_action_for_terminal_probe(
                np.asarray(_candidate_vec, dtype=np.int64),
                boosted_overrides=_candidate_boosted_overrides,
            )
            _result = base_env.evaluate_prepared_terminal_batch(
                [_prepared],
                num_trials_per_action=int(_final_trials),
                validation_required=True,
            )[0]
            _state, _terminal_reward, _done, _term_info = _result
            _apply_terminal_info_to_record(
                _candidate,
                float(_terminal_reward),
                _term_info,
                cached_reward_hit=False,
                validation_required=True,
            )
            log(
                f"  {bullet} [ADR-015] final candidate rank {_rank} "
                f"ep={int(start_episode + int(getattr(_candidate, 'episode_idx', 0)) + 1)} "
                f"P{int(getattr(_candidate, 'terminal_priority', 0) or 0)} "
                f"loss={float(getattr(_candidate, 'terminal_loss_mean', float('nan'))):.6f} "
                f"m1={float(getattr(_candidate, 'terminal_metric1_mean', float('nan'))):.6f} "
                f"m2={float(getattr(_candidate, 'terminal_metric2_mean', float('nan'))):.6f} "
                f"loss_std={float(getattr(_candidate, 'terminal_loss_std', float('nan'))):.6f} "
                f"rank_score={float(getattr(_candidate, 'terminal_cost_rank_score', 0.0) or 0.0):+.4f}"
            )
        except Exception as exc:
            log(
                f"  [ADR-015][warning] final candidate rank {_rank} "
                f"revalidation failed: {exc}"
            )

    _selected_record = _select_stage2_strict_feasible_best_record(
        _selection_candidates,
        loss_threshold=_loss_thr,
        top_n=len(_selection_candidates) if _selection_candidates else 1,
    )
    if _selected_record is not None:
        best_record = _selected_record
        best_rank_key = tuple(_episode_best_rank_key(best_record))
        best_reward = float(getattr(best_record, "total_reward", 0.0) or 0.0)
        _selected_vec = _final_selection_record_vec(best_record)
        best_action_vec = (
            None if _selected_vec is None
            else np.asarray(_selected_vec, dtype=np.int64).copy()
        )
        _loss_thr_disp = "None" if _loss_thr is None else f"{float(_loss_thr):.6f}"
        log(
            f"  {bullet} [ADR-015] final strict best selected: "
            f"ep={int(start_episode + int(getattr(best_record, 'episode_idx', 0)) + 1)} "
            f"reward={best_reward:+.4f} "
            f"loss={float(getattr(best_record, 'terminal_loss_mean', float('nan'))):.6f} "
            f"threshold={_loss_thr_disp} "
            f"rank_score={float(getattr(best_record, 'terminal_cost_rank_score', 0.0) or 0.0):+.4f}"
        )
    else:
        best_fallback_to_baseline = True
        log(
            f"  {bullet} [ADR-015] top-{_final_top_n} after K={_final_trials} "
            f"contains no strict-feasible candidate → best 回退 baseline"
        )
        if baseline_action_vec is not None:
            best_action_vec = np.asarray(baseline_action_vec, dtype=np.int64).copy()
        best_record = None
        best_rank_key = tuple()
        best_reward = 0.0  # baseline harvests no cost saving in the continuous reward

    if best_action_vec is not None:
        try:
            _final_best_vec = np.asarray(best_action_vec, dtype=np.int64)
            _final_slots = (
                None if diag_recorder._slots_view_builder is None
                else list(diag_recorder._slots_view_builder(_final_best_vec))
            )
            _final_best_payload = {
                "schema_version": diag_recorder.schema_version,
                "num_layers": int(ev.total_layers),
                "source": "blb_v3_sequential_final_strict_best",
                "selection": {
                    "method": "top_n_strict_feasible_after_revalidation",
                    "top_n": int(_final_top_n),
                    "validation_trials": int(_final_trials),
                    "fallback_to_baseline": bool(best_fallback_to_baseline),
                },
                "best_reward": float(best_reward),
                "best_rank_key": [float(x) for x in best_rank_key],
                "meta": dict(getattr(diag_recorder, "_meta", {}) or {}),
                "slots": _final_slots,
                "diff_vs_baseline": diag_recorder._diff_against_baseline(_final_slots),
                "action_vec": _final_best_vec.tolist(),
            }
            _tmp_best = diag_recorder.best_json_path + ".tmp"
            with open(_tmp_best, "w", encoding="utf-8") as _fh:
                json.dump(_final_best_payload, _fh, ensure_ascii=False, indent=2)
            os.replace(_tmp_best, diag_recorder.best_json_path)
            log(
                f"  {bullet} [ADR-015] diagnostics best_action_vec.json updated "
                f"to final strict best → {diag_recorder.best_json_path}"
            )
        except Exception as exc:
            log(f"  [diag][warning] final strict best JSON update failed: {exc}")

    best_action_description_paths: Dict[str, str] = {}
    baseline_action_description_paths: Dict[str, str] = {}
    # 加大精度 handoff: in fusion-count mode the persisted best_action_vec is a flat
    # grid-index vector that CANNOT carry the precision boost (boosted SFs live in
    # the option's explicit_field_values, above the grid). Reconstruct the per-step
    # fusion (option, K) selection from the flat vec + the in-memory training map so
    # the standalone validation-set final eval and the GLUE submission replay the
    # EXACT boosted config the RL search selected (the same config the training
    # terminal probe installed via _boosted_overrides). Without this, both install
    # PRE-boost (noisier) noise — a config RL never optimized.
    best_fusion_group: Optional[Dict[str, Any]] = None
    best_fusion_fixed_path: str = ""
    try:
        baseline_desc = _describe_action_vector(
            baseline_action_vec,
            max_sfs=max_sfs,
            num_layers=int(ev.total_layers),
            gelu_degree=fixed_gelu,
            attn_degree=fixed_softmax,
            profile=str(train_cfg.profile),
        )
        # Also carry the raw int vec inside the JSON so it's a true drop-in.
        baseline_desc = dict(baseline_desc)
        baseline_desc["action_vec"] = np.asarray(baseline_action_vec, dtype=int).tolist()
        baseline_desc["source"] = "static_skeletons_baseline"
        baseline_action_description_paths = _write_action_description_files(
            blb_progress_dir, baseline_desc, label="baseline", log_fn=log,
        )
        if baseline_action_description_paths.get("md"):
            log(f"  {bullet} 基线动作可读说明 → {baseline_action_description_paths['md']}")
    except Exception as exc:
        log(f"  [persist][warning] baseline action description write failed: {exc}")
    if best_action_vec is not None:
        try:
            best_desc = _describe_action_vector(
                best_action_vec,
                max_sfs=max_sfs,
                num_layers=int(ev.total_layers),
                gelu_degree=fixed_gelu,
                attn_degree=fixed_softmax,
                profile=str(train_cfg.profile),
            )
            best_desc = dict(best_desc)
            best_desc["action_vec"] = np.asarray(best_action_vec, dtype=int).tolist()
            best_desc["source"] = "blb_v3_sequential_runtime_best"
            best_desc["best_reward"] = float(best_reward)
            best_desc["best_rank_key"] = [float(x) for x in best_rank_key]
            best_action_description_paths = _write_action_description_files(
                blb_progress_dir, best_desc, label="best", log_fn=log,
            )
            if best_action_description_paths.get("md"):
                log(f"  {bullet} 最优动作可读说明 → {best_action_description_paths['md']}")
        except Exception as exc:
            log(f"  [persist][warning] best action description write failed: {exc}")

    # 加大精度 handoff (fusion-count mode only): persist a fusion_count_fixed_action_v1
    # config so downstream eval/submission replay the boosted option, not pre-boost.
    if fusion_map is not None and best_action_vec is not None:
        try:
            from .fusion_fixed_action import build_fusion_fixed_config
            best_fusion_cfg = build_fusion_fixed_config(
                np.asarray(best_action_vec, dtype=int),
                profile=str(train_cfg.profile),
                num_layers=int(ev.total_layers),
                gelu=np.asarray(fixed_gelu, dtype=int),
                softmax=np.asarray(fixed_softmax, dtype=int),
                fusion_map=fusion_map,
                source="blb_v3_sequential_runtime_best",
            )
            best_fusion_group = best_fusion_cfg.get("group")
            best_fusion_fixed_path = os.path.join(
                blb_progress_dir, "blb_stage2_best_action_fusion_fixed.json",
            )
            with open(best_fusion_fixed_path, "w", encoding="utf-8") as _fh:
                json.dump(best_fusion_cfg, _fh, ensure_ascii=False, indent=2)
            _summ = best_fusion_cfg.get("summary", {})
            log(
                f"  {bullet} 最优动作 fusion-fixed 配置（含 boost 还原）→ {best_fusion_fixed_path}"
                f"  (fusion={_summ.get('total_fusion_count')}, "
                f"boosted_options={_summ.get('boosted_option_count')})"
            )
        except Exception as exc:
            log(f"  [persist][warning] fusion-fixed best action write failed: {exc}")

    # Refresh status board's best block with the human-readable slot view too.
    try:
        best_slots_view = None
        best_slots_grouped = None
        if best_action_vec is not None and diag_recorder._slots_view_builder is not None:
            best_slots_view = list(diag_recorder._slots_view_builder(
                np.asarray(best_action_vec, dtype=int),
            ))
            from .action_io import group_slots_by_layer_block
            best_slots_grouped = group_slots_by_layer_block(best_slots_view)
        if best_action_vec is not None:
            status.set_best(
                float(best_reward),
                best_action_vec=np.asarray(best_action_vec, dtype=int).tolist(),
                best_slots=best_slots_view,
                best_slots_by_layer=best_slots_grouped,
            )
    except Exception as exc:
        log(f"  [status][warning] best block refresh failed: {exc}")

    # ---------- 7.6) Final training report markdown ----------
    try:
        from .persistence import write_blb_final_report
        best_slots_for_report = None
        baseline_slots_for_report = None
        slot_diff_for_report = None
        if diag_recorder._slots_view_builder is not None and best_action_vec is not None:
            best_slots_for_report = list(diag_recorder._slots_view_builder(
                np.asarray(best_action_vec, dtype=int),
            ))
            baseline_slots_for_report = diag_recorder._baseline_slots
            slot_diff_for_report = diag_recorder._diff_against_baseline(best_slots_for_report)
        baseline_summary: Dict[str, Any] = {
            "total_bits_sum": int(getattr(baseline, "total_bits_sum", 0) or 0),
            "total_fusion_count": int(getattr(baseline, "total_fusion_count", 0) or 0),
            "avg_k": float(getattr(baseline, "avg_k", 0.0) or 0.0),
            "loss_mean": float(getattr(baseline, "loss_mean", 0.0) or 0.0),
            "metric1_mean": float(getattr(baseline, "metric1_mean", 0.0) or 0.0),
            "metric2_mean": float(getattr(baseline, "metric2_mean", 0.0) or 0.0),
        }
        weights_summary: Dict[str, Any] = {
            "design": "budgeted_adaptive_scalar_p3_cost",
            "cost_weight": float(getattr(weights, "cost_weight", 0.0) or 0.0),
            "lambda_stab": float(getattr(weights, "lambda_stab", 0.0) or 0.0),
            "invalid_penalty": float(getattr(weights, "invalid_penalty", 0.0) or 0.0),
            "reward_clip_min": float(getattr(weights, "reward_clip_min", -5.0)),
            "reward_clip_max": float(getattr(weights, "reward_clip_max", 5.0)),
            "tier_metric_bonus": float(getattr(weights, "tier_metric_bonus", 0.0) or 0.0),
            "tier_stability_bonus": float(getattr(weights, "tier_stability_bonus", 0.0) or 0.0),
            "baseline_metric1": float(getattr(weights, "baseline_metric1", 0.0) or 0.0),
            "baseline_metric2": float(getattr(weights, "baseline_metric2", 0.0) or 0.0),
            "cost_reward_mode": str(getattr(weights, "cost_reward_mode", "")),
            "p3_metric_margin_budget": float(
                getattr(weights, "p3_metric_margin_budget", 0.0) or 0.0
            ),
            "p3_cost_budget": float(getattr(weights, "p3_cost_budget", 0.0) or 0.0),
            "cost_fusion_step_bonus": float(
                getattr(weights, "cost_fusion_step_bonus", 0.0) or 0.0
            ),
            "cost_k_step_bonus": float(getattr(weights, "cost_k_step_bonus", 0.0) or 0.0),
            "cost_k_step_size": float(getattr(weights, "cost_k_step_size", 0.0) or 0.0),
            "cost_bits_linear_scale": float(
                getattr(weights, "cost_bits_linear_scale", 0.0) or 0.0
            ),
            "cost_bits_tiebreaker_clip": float(
                getattr(weights, "cost_bits_tiebreaker_clip", 0.0) or 0.0
            ),
            "cost_score_clip_min": float(getattr(weights, "cost_score_clip_min", 0.0) or 0.0),
            "cost_score_clip_max": float(getattr(weights, "cost_score_clip_max", 0.0) or 0.0),
        }
        best_breakdown_for_report: Optional[Dict[str, Any]] = None
        if best_record is not None:
            best_breakdown_for_report = {
                "terminal_priority": int(best_record.terminal_priority),
                "terminal_reward": float(best_record.terminal_reward),
                "terminal_cost_score": float(best_record.terminal_cost_score),
                "terminal_cost_rank_score": float(best_record.terminal_cost_rank_score),
                "terminal_cost_rank_fusion": float(best_record.terminal_cost_rank_fusion),
                "terminal_cost_rank_truncation": float(
                    best_record.terminal_cost_rank_truncation
                ),
                "terminal_cost_rank_bits": float(best_record.terminal_cost_rank_bits),
                "terminal_p3_metric_margin_reward": float(
                    best_record.terminal_p3_metric_margin_reward
                ),
                "terminal_cost_fusion_bonus": float(best_record.terminal_cost_fusion_bonus),
                "terminal_cost_truncation_bonus": float(
                    best_record.terminal_cost_truncation_bonus
                ),
                "terminal_cost_bits_tiebreaker": float(
                    best_record.terminal_cost_bits_tiebreaker
                ),
                "terminal_cost_truncation_step_gain": float(
                    best_record.terminal_cost_truncation_step_gain
                ),
                "terminal_fusion_gain": float(best_record.terminal_fusion_gain),
                "terminal_k_gain": float(best_record.terminal_k_gain),
                "terminal_bits_gain": float(best_record.terminal_bits_gain),
                "terminal_metric1_mean": float(best_record.terminal_metric1_mean),
                "terminal_metric2_mean": float(best_record.terminal_metric2_mean),
                "terminal_stab_violation": float(best_record.terminal_stab_violation),
            }
        report_path = write_blb_final_report(
            blb_progress_dir,
            run_basename=run_basename,
            profile=str(train_cfg.profile),
            total_episodes=int(total_episodes_planned),
            completed_episodes=int(start_episode + len(episode_returns)),
            elapsed_sec=float(elapsed),
            best_reward=float(best_reward),
            best_breakdown=best_breakdown_for_report,
            best_action_vec=(
                np.asarray(best_action_vec, dtype=int).tolist()
                if best_action_vec is not None else None
            ),
            baseline=baseline_summary,
            reward_weights=weights_summary,
            episode_returns=episode_returns,
            rescale_invoker_kind="in_process_real",
            log_fn=log,
            best_slots=best_slots_for_report,
            baseline_slots=baseline_slots_for_report,
            slot_diff_vs_baseline=slot_diff_for_report,
            best_action_full_md_path=best_action_description_paths.get("md", ""),
            best_action_full_json_path=best_action_description_paths.get("json", ""),
        )
        log(f"  {bullet} 最终训练报告 → {report_path}")
    except Exception as exc:
        log(f"  [persist][warning] final report write failed: {exc}")

    # ---------- 7.7) Register run into experiments/registry.jsonl ----------
    _register_run_in_experiments_log(
        run_basename=run_basename,
        profile=str(train_cfg.profile),
        model_type=stage2_model_type,
        preset_label=str(fixed_label or ""),
        seed=int(train_cfg.seed),
        elapsed_sec=float(elapsed),
        completed_episodes=int(start_episode + len(episode_returns)),
        total_episodes_planned=int(total_episodes_planned),
        best_reward=float(best_reward) if best_action_vec is not None else 0.0,
        best_action_present=best_action_vec is not None,
        episode_count=len(episode_returns),
        blb_progress_dir=blb_progress_dir,
        diag_recorder=diag_recorder,
        save_path=save_path,
        best_action_description_paths=best_action_description_paths,
        baseline_action_description_paths=baseline_action_description_paths,
        log=log,
        bullet=bullet,
    )

    # ---------- 8) Training curves (Stage-1 风格) + entropy + 健康检测报告 ----------
    # 每回合并列序列从 in-memory ``episode_records`` 取（与 diagnostics/episodes.jsonl
    # 逐字一致：record_episode(fusion_count=record.fusion_count_sum_over_steps)）。
    # 跨 resume 的完整历史曲线用 scripts/blb_regen_stage2_outputs.py 离线重建。
    _records = seq_result.get("episode_records", []) or []
    _ep_returns = [float(getattr(r, "total_reward", 0.0) or 0.0) for r in _records] or list(episode_returns)
    _ep_losses = [float(getattr(r, "terminal_loss_mean", 0.0) or 0.0) for r in _records]
    _ep_m1 = [float(getattr(r, "terminal_metric1_mean", 0.0) or 0.0) for r in _records]
    _ep_m2 = [float(getattr(r, "terminal_metric2_mean", 0.0) or 0.0) for r in _records]
    _ep_fusion = [float(getattr(r, "fusion_count_sum_over_steps", 0) or 0) for r in _records]
    _base_avg_k = float(getattr(baseline, "avg_k", 13.0) or 13.0)
    _ep_avgk = [_base_avg_k - float(getattr(r, "terminal_k_gain", 0.0) or 0.0) for r in _records]
    _pri = [int(getattr(r, "terminal_priority", 0) or 0) for r in _records]
    _ppo_metrics = seq_result.get("ppo_metrics", []) or []
    _ent = [float(m.get("entropy")) for m in _ppo_metrics if m.get("entropy") is not None]
    _ent_eps = [float(m.get("completed_episodes", 0) or 0) for m in _ppo_metrics if m.get("entropy") is not None]
    try:
        curve_paths = write_training_curves(
            blb_progress_dir,
            episode_returns=_ep_returns,
            episode_losses=_ep_losses or None,
            episode_metric1s=_ep_m1 or None,
            episode_metric2s=_ep_m2 or None,
            episode_fusion_counts=_ep_fusion or None,
            episode_avg_ks=_ep_avgk or None,
            baselines={
                "loss": float(getattr(baseline, "loss_mean", 0.0) or 0.0),
                "metric1": float(getattr(baseline, "metric1_mean", 0.0) or 0.0),
                "metric2": float(getattr(baseline, "metric2_mean", 0.0) or 0.0),
                "avg_k": _base_avg_k,
            },
            entropy_series=_ent or None,
            entropy_episodes=_ent_eps or None,
            log_fn=log,
        )
        if curve_paths.get("png"):
            log(f"  {bullet} 训练曲线 PNG → {curve_paths['png']}")
        if curve_paths.get("entropy_png"):
            log(f"  {bullet} 熵曲线 PNG → {curve_paths['entropy_png']}")
    except Exception as exc:
        log(f"  [警告] 写训练曲线失败：{exc}")

    # 8.1) 局部最优 / 健康检测报告（Stage-1 同款 pruning_search_log.txt 版式）。
    try:
        from rl_local_optimum import write_local_optimum_report
        from .persistence import BLB_SEARCH_LOG_TXT
        write_local_optimum_report(
            os.path.join(blb_progress_dir, BLB_SEARCH_LOG_TXT),
            episode_returns=_ep_returns,
            episode_entropies=_ent or None,
            best_score_history=None,
            completed_episodes=int(start_episode + len(episode_returns)),
            title="BLB Stage-2 RL",
            extra_lines=[
                "",
                "--- 优先级分布（priority histogram）---",
                f"  P1(acc):  {sum(1 for p in _pri if p == 1)}",
                f"  P2(stab): {sum(1 for p in _pri if p == 2)}",
                f"  P3(cost): {sum(1 for p in _pri if p == 3)}",
            ],
            priority=_pri,
            fusion_count=[float(getattr(r, "fusion_count_sum_over_steps", 0) or 0) for r in _records],
            worst_signed_margin=[float(getattr(r, "terminal_worst_signed_margin", 0.0) or 0.0) for r in _records],
            log_fn=log,
        )
    except Exception as exc:
        log(f"  [警告] 写检测报告失败：{exc}")

    # 8.2) 崩溃诊断曲线（ADR-014）：reward 分解 / fusion-vs-feasibility / 噪声 vs 余量。
    try:
        def _rc(attr):
            return [float(getattr(r, attr, 0.0) or 0.0) for r in _records]
        diag_curve = write_diagnostic_curves(
            blb_progress_dir,
            priority=_pri,
            fusion_count=[float(getattr(r, "fusion_count_sum_over_steps", 0) or 0) for r in _records],
            fusion_b2=[float(getattr(r, "fusion_count_b2", 0) or 0) for r in _records],
            fusion_b4=[float(getattr(r, "fusion_count_b4", 0) or 0) for r in _records],
            fusion_b5=[float(getattr(r, "fusion_count_b5", 0) or 0) for r in _records],
            worst_signed_margin=_rc("terminal_worst_signed_margin"),
            acc_barrier_sat=_rc("terminal_acc_barrier_sat"),
            acc_barrier_vio=_rc("terminal_acc_barrier_vio"),
            cost_score=_rc("terminal_cost_score"),
            p3_metric_margin=_rc("terminal_p3_metric_margin_reward"),
            metric1_std=_rc("terminal_metric1_std"),
            log_fn=log,
        )
        if diag_curve.get("diagnostics_png"):
            log(f"  {bullet} 崩溃诊断曲线 PNG → {diag_curve['diagnostics_png']}")
    except Exception as exc:
        log(f"  [警告] 写诊断曲线失败：{exc}")

    # ---------- 8.5) 解耦归档 + best_policy（对齐 Stage-1）----------
    # 2026-06-01 解耦：sequential stage2-only 完成 → 归档进 stage2/record/{combo N date}/
    # 并打 COMPLETED；同时把最优 policy 汇总到 best_policy/。这段以前只存在于 legacy
    # 单发路径（runner.py），sequential 提前 return 永远到不了，导致 Stage-2 工作目录
    # 缺 record/ / COMPLETED / final_config / final_eval（与 Stage-1 不对齐）。
    # best-effort：任何异常只记日志，绝不让收尾处崩训练。
    if getattr(ev, "decoupled_layout", False) and best_action_vec is not None:
        try:
            import datetime as _dt
            import json as _json
            import shutil as _shutil
            from config import run_layout as _rl
            from .persistence import (
                BLB_TRAINING_CURVE_PNG,
                BLB_ENTROPY_CURVE_PNG,
                BLB_DIAGNOSTIC_CURVE_PNG,
                BLB_REWARD_PAPER_PNG,
                BLB_FINAL_REPORT_MD,
                BLB_SEARCH_LOG_TXT,
            )

            _wd = os.path.normpath(str(getattr(ev, "run_output_dir", "") or ""))  # <root>/stage2/<combo>
            if _wd and _wd != ".":
                _combo = os.path.basename(_wd)
                _root = os.path.dirname(os.path.dirname(_wd))
                _bd = best_breakdown_for_report if isinstance(best_breakdown_for_report, dict) else {}
                _paths = best_action_description_paths or {}

                def _bk(attr, default=None):
                    return (float(getattr(best_record, attr)) if best_record is not None
                            and getattr(best_record, attr, None) is not None else default)

                _final_config = {
                    "stage": 2,
                    "combo": _combo,
                    "profile": str(train_cfg.profile),
                    "num_layers": int(ev.total_layers),
                    "blb_v3_best_action_vec": np.asarray(best_action_vec, dtype=int).tolist(),
                    # 前置 Stage-1（一个 stage2 严格绑定一个 stage1）。
                    "gelu_degree_per_layer": np.asarray(fixed_gelu, dtype=int).tolist(),
                    "softmax_degree_per_layer": np.asarray(fixed_softmax, dtype=int).tolist(),
                    "best_action_readable_json": _paths.get("json", ""),
                    "best_action_readable_md": _paths.get("md", ""),
                    # 加大精度: fusion (option, K) per step so a re-eval from this
                    # archive replays the boosted config (None ⇒ per-slot run).
                    "blb_v3_fusion_count_action": bool(best_fusion_group is not None),
                    "blb_v3_best_action_group": best_fusion_group,
                    "blb_v3_best_action_fusion_fixed_path": best_fusion_fixed_path,
                }
                _final_eval = {
                    "source": "training_best_mean_of_K_trials",
                    "note": "basic snapshot (训练记录的 K 次 MC 噪声 trial 最优档); "
                            "重型同-cost 组对比见独立 final-eval 工具。",
                    "best_reward": float(best_reward) if np.isfinite(best_reward) else None,
                    "loss": _bk("terminal_loss_mean"),
                    "metric1": _bk("terminal_metric1_mean"),
                    "metric2": _bk("terminal_metric2_mean"),
                    "cost": {
                        "total_bits_sum": (int(getattr(best_record, "total_bits_sum_over_steps", 0) or 0)
                                           if best_record is not None else None),
                        "total_fusion_count": (int(getattr(best_record, "fusion_count_sum_over_steps", 0) or 0)
                                               if best_record is not None else None),
                        "avg_k": ((_base_avg_k - _bk("terminal_k_gain", 0.0))
                                  if best_record is not None else None),
                    },
                    "baseline_cost": {
                        "total_bits_sum": int(baseline.total_bits_sum),
                        "total_fusion_count": int(baseline.total_fusion_count),
                        "avg_k": float(baseline.avg_k),
                        "loss_mean": float(getattr(baseline, "loss_mean", 0.0)),
                        "metric1_mean": float(getattr(baseline, "metric1_mean", 0.0)),
                        "metric2_mean": float(getattr(baseline, "metric2_mean", 0.0)),
                        "loss_std": float(getattr(baseline, "loss_std", 0.0)),
                        "metric1_std": float(getattr(baseline, "metric1_std", 0.0)),
                        "metric2_std": float(getattr(baseline, "metric2_std", 0.0)),
                    },
                    "trainer_gate_baseline": dict(baseline_preflight_metrics),
                    "breakdown": _bd,
                }
                _metadata = {
                    "stage": 2,
                    "combo": _combo,
                    "profile": str(train_cfg.profile),
                    "data_path": getattr(ev, "data_path", ""),
                    "completed_at": _dt.datetime.now().isoformat(),
                    "episodes": int(start_episode + len(episode_returns)),
                    "stage1_run_id": getattr(ev, "stage1_run_id", ""),
                    "stage2_limit_tolerance": getattr(ev, "stage2_limit_tolerance", None),
                    "stage2_stability_tolerance": getattr(ev, "stage2_stability_tolerance", None),
                    "stage2_k_trials": int(getattr(train_cfg, "num_trials_per_step", 0) or 0),
                    "trainer_gate_baseline": dict(baseline_preflight_metrics),
                }
                _report_md = (
                    f"# Stage-2 record: {_combo}\n\n"
                    f"- profile: {train_cfg.profile}, num_layers: {ev.total_layers}\n"
                    f"- best_reward: {best_reward}\n"
                    f"- prerequisite Stage-1 gelu: {np.asarray(fixed_gelu, dtype=int).tolist()}\n"
                    f"- prerequisite Stage-1 softmax: {np.asarray(fixed_softmax, dtype=int).tolist()}\n"
                    f"- best action readable: {_paths.get('md', '')}\n"
                )
                _curves = [
                    os.path.join(blb_progress_dir, BLB_TRAINING_CURVE_PNG),
                    os.path.join(blb_progress_dir, BLB_ENTROPY_CURVE_PNG),
                    os.path.join(blb_progress_dir, BLB_DIAGNOSTIC_CURVE_PNG),
                    os.path.join(blb_progress_dir, BLB_REWARD_PAPER_PNG),
                    os.path.join(blb_progress_dir, BLB_FINAL_REPORT_MD),
                    os.path.join(blb_progress_dir, BLB_SEARCH_LOG_TXT),
                    _paths.get("json", ""),
                    _paths.get("md", ""),
                ]
                _rdir, _rid, _n = _rl.snapshot_decoupled_record(
                    2, _combo, _wd,
                    final_config=_final_config,
                    final_eval=_final_eval,
                    metadata=_metadata,
                    curve_paths=[p for p in _curves if p],
                    report_md=_report_md,
                    root=_root,
                )
                log(f"  {bullet} [解耦] Stage-2 已归档进 record → {_rdir}（COMPLETED 已标记）")

                # 工作目录 metadata.json（对齐 Stage-1：算法 / 约束 / 阶段状态 / run_count）。
                # Stage-1 由 launcher 在启动时建基础字段、完成时更新 stage_status；解耦
                # sequential Stage-2 走不到那条 launcher 分支，这里自给自足地建/并入一份。
                try:
                    _meta_path = os.path.join(_wd, "metadata.json")
                    _existing = {}
                    if os.path.isfile(_meta_path):
                        try:
                            with open(_meta_path, encoding="utf-8") as _ef:
                                _existing = _json.load(_ef) or {}
                        except Exception:
                            _existing = {}
                    _now = _dt.datetime.now().isoformat()
                    _existing.setdefault("algorithm", "rl")
                    _existing.setdefault("model_type", stage2_model_type)
                    _existing.setdefault("dataset", str(train_cfg.profile))
                    _existing.setdefault("created_at", _now)
                    _existing.setdefault("run_count", 1)
                    _existing["last_updated_at"] = _now
                    _existing["stage2_limit_tolerance"] = getattr(ev, "stage2_limit_tolerance", None)
                    _existing["stage2_stability_tolerance"] = getattr(ev, "stage2_stability_tolerance", None)
                    _existing["stage2_k_trials"] = int(getattr(train_cfg, "num_trials_per_step", 0) or 0)
                    _existing["stage2_probe_size"] = int(getattr(ev, "stage2_probe_size", 256) or 256)
                    _existing["trainer_gate_baseline"] = dict(baseline_preflight_metrics)
                    _ss = _existing.setdefault("stage_status", {})
                    _ss["stage2_search"] = "completed"
                    _sd = _existing.setdefault("stage_detail", {})
                    _sd.setdefault("stage2_search", {}).update({
                        "episodes": int(start_episode + len(episode_returns)),
                        "best_reward": float(best_reward) if np.isfinite(best_reward) else None,
                        "record_id": _rid,
                    })
                    with open(_meta_path, "w", encoding="utf-8") as _mf:
                        _json.dump(_existing, _mf, ensure_ascii=False, indent=2)
                    log(f"  {bullet} [解耦] 工作目录 metadata.json 已更新 → {_meta_path}")
                except Exception as _me:
                    log(f"  [解耦][警告] 工作目录 metadata.json 写入失败：{_me}")

                # best_policy/ 目录（对齐 Stage-1：policy + 约束元数据）。
                try:
                    _bp_dir = os.path.join(_wd, "best_policy")
                    os.makedirs(_bp_dir, exist_ok=True)
                    if save_path and os.path.isfile(save_path):
                        _shutil.copy2(save_path, os.path.join(_bp_dir, "blb_stage2_policy.pt"))
                    with open(os.path.join(_bp_dir, "constraint_metadata.json"), "w",
                              encoding="utf-8") as _bf:
                        _json.dump({
                            "profile": str(train_cfg.profile),
                            "stage2_limit_tolerance": getattr(ev, "stage2_limit_tolerance", None),
                            "stage2_stability_tolerance": getattr(ev, "stage2_stability_tolerance", None),
                            "stage2_k_trials": int(getattr(train_cfg, "num_trials_per_step", 0) or 0),
                            "stage2_probe_size": int(getattr(ev, "stage2_probe_size", 256) or 256),
                            "trainer_gate_baseline": dict(baseline_preflight_metrics),
                            "search_algorithm": "rl",
                            "rl_variant": str(getattr(train_cfg, "rl_variant", "") or ""),
                        }, _bf, ensure_ascii=False, indent=2)
                    log(f"  {bullet} [best_policy] 已汇总 → {_bp_dir}")
                except Exception as _bpe:
                    log(f"  [best_policy][警告] 汇总失败：{_bpe}")
        except Exception as _snap_exc:
            log(f"  [解耦][警告] Stage-2 record 归档失败（不影响训练结果）：{_snap_exc}")

    # ---------- 9) Restore handler to clean polynomial-only state ----------
    try:
        for restore_name in (
                "restore_layer_block5_noise", "restore_layer_block4_noise",
                "restore_layer_block3_noise", "restore_layer_block2_noise",
                "restore_layer_block1_noise", "restore_blb_first_input_noise",
        ):
            method = getattr(ev.reversible_handler, restore_name, None)
            if method is None:
                continue
            try:
                method(layer_indices=list(range(ev.total_layers)))
            except Exception:
                pass
    finally:
        ev.apply_configuration(fixed_gelu, fixed_softmax)

    # ---------- 10) Assemble legacy-compat result dict ----------
    cost_reference_noise_config = ev._get_max_noise_configuration()
    cost_reference_tot_c, _ = ev.get_noise_simulated_cost(**cost_reference_noise_config)
    legacy_best = _build_legacy_compatible_best_noise_config(ev)

    base_loss, base_p, base_s, _ = ev.evaluate_model(
        fixed_gelu, fixed_softmax, use_train=False,
        split=ev.get_reward_reference_split_name(),
    )
    limit_dict = ev.build_constraint_limits_from_metrics(base_loss, base_p, base_s)
    clean_limit_loss = float(limit_dict["loss"])
    clean_limit_p = float(limit_dict["metric1"])
    clean_limit_s = float(limit_dict["metric2"])
    search_limit_loss = (
        float(base_env.loss_threshold)
        if base_env.loss_threshold is not None
        else float(clean_limit_loss)
    )
    search_limit_p = float(base_env.acc_threshold)
    search_limit_s = float(
        base_env.acc_threshold_m2
        if base_env.acc_threshold_m2 is not None
        else base_env.acc_threshold
    )

    result: Dict[str, Any] = {
        "fixed_gelu": fixed_gelu.copy(),
        "fixed_softmax": fixed_softmax.copy(),
        "baseline_noise_config": {k: v.copy() for k, v in cost_reference_noise_config.items()},
        "baseline_tot_c": float(cost_reference_tot_c),
        "cost_reference_noise_config": {k: v.copy() for k, v in cost_reference_noise_config.items()},
        "cost_reference_source": "max_noise_configuration",
        "performance_baseline_gelu": fixed_gelu.copy(),
        "performance_baseline_softmax": fixed_softmax.copy(),
        "performance_baseline_source": "stage1_fixed_low_risk_noise",
        "k_trials": int(train_cfg.num_trials_per_step),
        "probe_size": int(getattr(ev, "stage2_probe_size", 256)),
        "limit_loss": float(search_limit_loss),
        "limit_p": float(search_limit_p),
        "limit_s": float(search_limit_s),
        "proxy_limit_loss": float(search_limit_loss),
        "proxy_limit_p": float(search_limit_p),
        "proxy_limit_s": float(search_limit_s),
        "proxy_base_loss": float(base_loss),
        "proxy_base_p": float(base_p),
        "proxy_base_s": float(base_s),
        "raw_model_baseline_metrics": {
            "loss": float(base_loss),
            "metric1": float(base_p),
            "metric2": float(base_s),
        },
        "raw_model_constraint_limits": {
            "loss": float(clean_limit_loss),
            "metric1": float(clean_limit_p),
            "metric2": float(clean_limit_s),
        },
        "all_max_blb_baseline_metrics": dict(baseline_preflight_metrics),
        "search_limits": {
            "loss": float(search_limit_loss),
            "metric1": float(search_limit_p),
            "metric2": float(search_limit_s),
            "loss_std": float(stab_threshold_loss),
            "metric1_std": float(stab_threshold_m1),
            "metric2_std": float(stab_threshold_m2),
        },
        "status": "completed",
        "training_eval_split": str(ev.get_reward_reference_split_name()),
        "best_noise_config": {k: v.copy() for k, v in legacy_best.items()},
        "stable_search_best_noise_config": {k: v.copy() for k, v in legacy_best.items()},
        "stable_joint_best_noise_config": {k: v.copy() for k, v in legacy_best.items()},
        "selection_diagnostics": {
            "selection_mode": "blb_v3_sequential_stage1_reward_then_cost_rank",
            "best_reward": float(best_reward),
            "best_rank_key": [float(x) for x in best_rank_key],
            "best_action_vec": (
                best_action_vec.tolist() if best_action_vec is not None else None
            ),
        },
        "blb_v3_best_action_vec": (
            best_action_vec.tolist() if best_action_vec is not None else None
        ),
        "blb_v3_best_reward": float(best_reward),
        "blb_v3_profile": str(train_cfg.profile),
        # 加大精度 handoff: fusion-count mode carries the reconstructed per-step
        # (option, K) selection so the embedded/standalone final eval replays the
        # boosted config (None for per-slot runs ⇒ the default index decode path).
        "blb_v3_fusion_count_action": bool(fusion_map is not None),
        "blb_v3_best_action_group": best_fusion_group,
        "blb_v3_best_action_fusion_fixed_path": best_fusion_fixed_path,
        "blb_v3_total_episodes": int(train_cfg.total_episodes),
        "rl_variant": seq_rl_variant,
        "sequential_diagnostics": {
            "horizon": int(seq_env.horizon),
            "max_step_dim": int(seq_env.max_step_dim),
            "state_dim": int(seq_env.state_dim),
            "episode_count": int(len(episode_returns)),
            "final_invalid_rate": float(seq_result.get("final_invalid_rate", 0.0)),
            "elapsed_sec": float(elapsed),
            "ppo_metric_count": int(len(seq_result.get("ppo_metrics", []))),
        },
        "training_hparams": {
            "blb_v3_total_episodes": int(train_cfg.total_episodes),
            "blb_v3_rollout_size": int(train_cfg.rollout_size),
            "blb_v3_ppo_lr": float(train_cfg.ppo.lr),
            "blb_v3_sequential_rl": True,
            "blb_v3_sequential_invalid_penalty": float(seq_env_cfg.invalid_penalty),
            "blb_v3_sequential_cost_shaping_coeff": float(seq_env_cfg.cost_shaping_coeff),
            "blb_v3_sequential_fusion_shaping_coeff": float(seq_env_cfg.fusion_shaping_coeff),
            "blb_v3_sequential_early_terminate_on_invalid": bool(
                seq_env_cfg.early_terminate_on_invalid
            ),
            "blb_v3_guarded_radius2_enabled": bool(
                getattr(train_cfg, "guarded_radius2_enabled", False)
            ),
            "blb_v3_guarded_radius2_min_episode": int(
                getattr(train_cfg, "guarded_radius2_min_episode", 1060)
            ),
            "blb_v3_guarded_radius2_stall_window": int(
                getattr(train_cfg, "guarded_radius2_stall_window", 600)
            ),
            "blb_v3_guarded_radius2_max_mutations": int(
                getattr(train_cfg, "guarded_radius2_max_mutations", 4)
            ),
            "blb_v3_guarded_radius2_episode_fraction": float(
                getattr(train_cfg, "guarded_radius2_episode_fraction", 0.15)
            ),
            "blb_v3_guarded_radius2_cooldown_episodes": int(
                getattr(train_cfg, "guarded_radius2_cooldown_episodes", 300)
            ),
            "k_trials": int(train_cfg.num_trials_per_step),
            "protected_k1_enabled": bool(
                getattr(train_cfg, "protected_k1_enabled", False)
            ),
            "protected_k1_guard_sigma": float(
                getattr(train_cfg, "protected_k1_guard_sigma", 4.0)
            ),
            "protected_k1_audit_fraction": float(
                getattr(train_cfg, "protected_k1_audit_fraction", 0.02)
            ),
            "probe_size": int(getattr(ev, "stage2_probe_size", 256)),
            "baseline_preflight_trial_count": int(
                baseline_preflight_metrics.get("trial_count", train_cfg.num_trials_per_step)
            ),
            "stage2_limit_tolerance": float(allowed_acc_drop),
            "stage2_stability_tolerance": float(stability_tol),
            "metric1_threshold": float(search_limit_p),
            "metric2_threshold": float(search_limit_s),
            "loss_threshold": float(search_limit_loss),
            "std_threshold_loss": float(stab_threshold_loss),
            "std_threshold_metric1": float(stab_threshold_m1),
            "std_threshold_metric2": float(stab_threshold_m2),
        },
        "reward_diagnostics": {
            "episode_return_mean": (
                float(np.mean(episode_returns)) if episode_returns else None
            ),
            "episode_return_max": (
                float(np.max(episode_returns)) if episode_returns else None
            ),
            "best_reward": float(best_reward),
        },
    }
    return result
