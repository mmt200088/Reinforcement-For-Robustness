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

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from .action_mask import ForbiddenActionMask
from .action_space import K_LEVELS
from .sequential_env import BLBStage2SequentialEnv, SequentialEnvConfig
from .sequential_policy import (
    BLBStage2SequentialPolicy,
    SequentialPolicyConfig,
    SequentialPPOConfig,
    SequentialRolloutBuffer,
    sequential_ppo_update,
    step_to_mask_and_levels,
)


@dataclass
class SequentialTrainConfig:
    total_episodes: int = 100
    update_every_n_episodes: int = 4
    log_every_n_episodes: int = 4
    seed: Optional[int] = None
    ppo: SequentialPPOConfig = field(default_factory=SequentialPPOConfig)
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
    ent_coef_ramp_episodes: int = 240
    absolute_episode_start: int = 0
    warmstart_neighbor_sampling: bool = True
    warmstart_neighbor_ramp_episodes: int = 0
    warmstart_neighbor_max_mutations: int = 8
    warmstart_neighbor_max_radius: int = 2
    warmstart_mutable_full_offsets: Optional[List[int]] = None


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


def _seq_progress_bar(current: int, total: int, width: int = 30) -> str:
    ratio = min(float(current) / max(float(total), 1.0), 1.0)
    filled = int(round(ratio * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {ratio:6.1%}"


def _seq_fmt_elapsed(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


def _seq_fmt_eta_finish(eta_seconds: float) -> str:
    finish_ts = time.time() + max(float(eta_seconds), 0.0)
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(finish_ts))


# How often (in PPO updates) to print the big progress box, matching v2's
# NOISE_RL_PROGRESS_BOX_PPO_INTERVAL.
SEQ_PROGRESS_BOX_PPO_INTERVAL = 5


def _resolve_ent_coef_schedule(
        *,
        ep_count_1based: int,
        anchor_episodes: int,
        target_ent_coef: float,
        anchor_ent_coef: float = 0.0,
        ramp_episodes: int = 240,
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
    rollout = int(
        getattr(
            train_cfg,
            "rollout_size",
            getattr(train_cfg, "update_every_n_episodes", 1),
        ) or 1
    )
    return max(0, min(max(60, int(rollout) * 2), int(total)))


def _near_baseline_level_indices(
        *,
        kind: str,
        baseline_idx: int,
        dim: int,
        radius: int,
        ) -> List[int]:
    """Local allowed categorical indices around the baseline value.

    K is decoded through non-monotonic ``K_LEVELS``, so locality is by K value
    order rather than by categorical index.
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
            idx for idx, value in enumerate(K_LEVELS[:dim])
            if int(value) <= base_k
        ]
        candidates.sort(key=lambda idx: int(K_LEVELS[idx]), reverse=True)
        return [int(idx) for idx in candidates[: radius + 1]]
    lo = max(0, baseline_idx - radius)
    hi = min(dim - 1, baseline_idx)
    return [int(idx) for idx in range(lo, hi + 1)]


def _default_step_level_mask(
        *,
        spec: Any,
        baseline_action_vec: Sequence[int],
        max_step_dim: int,
        max_num_levels: int,
        ) -> np.ndarray:
    """Baseline-only per-level mask for one sequential step."""
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


def _build_step_level_mask(
        *,
        spec: Any,
        baseline_action_vec: Sequence[int],
        selected_full_offsets: Set[int],
        max_step_dim: int,
        max_num_levels: int,
        radius: int,
        ) -> np.ndarray:
    """Near-baseline per-level mask for one step.

    Slots selected for the current episode may move inside a local
    near-baseline neighborhood. Every other slot stays baseline-only.
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
    chosen = rng.choice(np.asarray(candidates, dtype=np.int64), size=count, replace=False)
    return {int(x) for x in np.asarray(chosen, dtype=np.int64).reshape(-1).tolist()}


def _noisy_accuracy_threshold_with_probe_guard(
        *,
        noisy_baseline_metric1: float,
        allowed_acc_drop: float,
        probe_size: int,
        ) -> float:
    """Accuracy gate with one-probe-sample guard for noisy online probes.

    MRPC probe accuracy is discrete. With the default 256-example online probe,
    one example is ~0.0039 accuracy. A K=5 noisy baseline can therefore jitter
    just below ``baseline - tolerance`` even when the action is exactly the
    static-skeleton baseline. The guard prevents false P1(acc) points for the
    baseline while leaving real collapses (e.g. m1≈0.31) far below threshold.
    """
    baseline = float(noisy_baseline_metric1)
    drop = max(0.0, float(allowed_acc_drop))
    sample_guard = 1.0 / float(max(1, int(probe_size)))
    return max(0.0, baseline - drop - sample_guard)


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
    safe_neighbor_active: bool = False
    safe_neighbor_mutation_count: int = 0
    safe_neighbor_radius: int = 0


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


def train_sequential(
        *,
        env: BLBStage2SequentialEnv,
        policy: BLBStage2SequentialPolicy,
        train_cfg: Optional[SequentialTrainConfig] = None,
        device: Optional[torch.device] = None,
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
        baseline_action_vec: Optional[np.ndarray] = None,
        max_rejection_retries: int = 32,
        force_baseline_episodes: int = 0,
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
    device = device or next(policy.parameters()).device
    log = logger or logging.getLogger(__name__)
    optimizer = torch.optim.Adam(policy.parameters(), lr=train_cfg.ppo.lr)
    buffer = SequentialRolloutBuffer()

    # Per-(layer, block) blacklist of action tuples that produced invalid_chain.
    # Survives across episodes within this train_sequential call. If a caller
    # supplied an existing mask (e.g. resumed from checkpoint), keep its entries
    # so we don't re-discover the same failures.
    if forbidden_mask is None:
        forbidden_mask = ForbiddenActionMask()
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

    if train_cfg.seed is not None:
        torch.manual_seed(int(train_cfg.seed))
        np.random.seed(int(train_cfg.seed) % (2**32))

    absolute_episode_start = max(0, int(getattr(train_cfg, "absolute_episode_start", 0) or 0))
    mutable_full_offsets = getattr(train_cfg, "warmstart_mutable_full_offsets", None)
    if mutable_full_offsets is not None:
        mutable_full_offsets = [int(x) for x in mutable_full_offsets]

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
        # Terminal breakdown extracted from the last commit_step's info dict.
        # 0 / 0.0 means the episode never produced a terminal reward (e.g.,
        # early_terminate_on_invalid fired before the last step).
        terminal_priority_int = 0
        terminal_loss_mean_val = 0.0
        terminal_loss_std_val = 0.0
        terminal_metric1_val = 0.0

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
        if (
                (not force_this_ep)
                and bool(getattr(train_cfg, "warmstart_neighbor_sampling", True))
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
            seed_base = int(train_cfg.seed) if train_cfg.seed is not None else 0
            episode_rng = np.random.default_rng(
                int((seed_base + absolute_ep * 1_000_003) % (2**32))
            )
            neighbor_mask_active = True
            neighbor_selected_offsets = _sample_episode_neighbor_offsets(
                schedule=env.schedule,
                baseline_action_vec=baseline_action_vec,
                mutable_full_offsets=mutable_full_offsets,
                mutation_count=int(neighbor_mutations),
                rng=episode_rng,
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
            action_level_mask_np: Optional[np.ndarray] = None
            action_level_mask_t = None
            if neighbor_mask_active and baseline_action_vec is not None:
                action_level_mask_np = _build_step_level_mask(
                    spec=spec,
                    baseline_action_vec=baseline_action_vec,
                    selected_full_offsets=neighbor_selected_offsets,
                    max_step_dim=policy.cfg.max_step_dim,
                    max_num_levels=policy.cfg.max_num_levels,
                    radius=int(neighbor_radius),
                )
                action_level_mask_t = (
                    torch.from_numpy(action_level_mask_np).to(device).unsqueeze(0)
                )

            # -- Forced-baseline anchor short-circuit --
            # Skip sampling + rejection-loop entirely; commit the baseline
            # action slice. Value/log_prob come from `policy.evaluate_action`
            # against the CURRENT policy so PPO gradients are well-defined.
            if force_this_ep and baseline_action_vec is not None:
                baseline_slice = baseline_action_vec[list(spec.full_vec_offsets)][:n_active]
                forced_action = np.asarray(baseline_slice, dtype=np.int64)
                forced_padded = np.zeros(policy.cfg.max_step_dim, dtype=np.int64)
                forced_padded[:n_active] = forced_action
                with torch.no_grad():
                    actions_t = torch.from_numpy(forced_padded).to(device).unsqueeze(0)
                    lp_t, _, val_t = policy.evaluate_action(
                        obs_t, actions_t, slot_mask_t, levels_t,
                    )
                chosen_eval_info = env.evaluate_step(forced_action.tolist())
                chosen_action_np = forced_padded
                chosen_log_prob = float(lp_t.item())
                chosen_value = float(val_t.item())
                rejection_counters["steps_forced_to_baseline_anchor"] += 1

                action_np = chosen_action_np
                log_prob = chosen_log_prob
                value = chosen_value
                step_action_for_env = action_np[:n_active].tolist()

                next_obs, reward, done, info = env.commit_step(chosen_eval_info)
                steps_taken += 1
                valid = bool(info.get("valid", True))
                if valid:
                    valid_step_count += 1
                    rejection_counters["steps_committed_valid"] += 1
                total_bits_sum += int(info.get("total_bits", 0))
                fusion_count_sum += int(info.get("fusion_count", 0))
                enriched_info = dict(info)
                enriched_info["action"] = step_action_for_env
                enriched_info["reward"] = float(reward)
                enriched_info["value"] = value
                enriched_info["log_prob"] = log_prob
                enriched_info["forced_baseline"] = True
                if on_step_end is not None:
                    try:
                        on_step_end(int(ep), int(steps_taken - 1), enriched_info)
                    except Exception:
                        pass
                if capture_step_infos:
                    captured_step_infos.append(enriched_info)
                buffer.add(
                    state=obs,
                    action=action_np,
                    slot_mask=slot_mask_np,
                    per_slot_num_levels=levels_np,
                    action_level_mask=None,
                    log_prob=log_prob,
                    value=value,
                    reward=float(reward),
                    done=bool(done),
                )
                per_step_sum += float(reward)
                if "terminal_reward" in info:
                    terminal_reward = float(info["terminal_reward"])
                term_info_dict = info.get("terminal_info") or {}
                term_breakdown = term_info_dict.get("reward_breakdown")
                term_metrics = term_info_dict.get("metrics")
                if term_breakdown is not None:
                    terminal_priority_int = int(getattr(term_breakdown, "priority", 0) or 0)
                if term_metrics is not None:
                    terminal_loss_mean_val = float(getattr(term_metrics, "loss_mean", 0.0) or 0.0)
                    terminal_loss_std_val = float(getattr(term_metrics, "loss_std", 0.0) or 0.0)
                    terminal_metric1_val = float(getattr(term_metrics, "metric1_mean", 0.0) or 0.0)
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
            chosen_log_prob: float = 0.0
            chosen_value: float = 0.0
            chosen_eval_info: Optional[Dict[str, Any]] = None
            attempts_this_step = 0

            for _attempt in range(int(max_rejection_retries)):
                attempts_this_step += 1
                with torch.no_grad():
                    action_t, log_prob_t, value_t = policy.sample_action(
                        obs_t, slot_mask_t, levels_t,
                        deterministic=False,
                        action_level_mask=action_level_mask_t,
                    )
                action_np_try = action_t.squeeze(0).cpu().numpy().astype(np.int64)
                step_action_try = action_np_try[:n_active].tolist()
                tup = tuple(int(x) for x in step_action_try)

                if forbidden_mask.is_forbidden(spec.layer_idx, spec.block_idx, tup):
                    rejection_counters["samples_rejected_by_mask"] += 1
                    continue   # cheap re-sample, no optimizer call

                eval_info = env.evaluate_step(step_action_try)
                if eval_info["valid"]:
                    chosen_action_np = action_np_try
                    chosen_log_prob = float(log_prob_t.item())
                    chosen_value = float(value_t.item())
                    chosen_eval_info = eval_info
                    break

                # New invalid → blacklist + try again. Per the user's spec
                # ("就好像训练过程中根本不存在这些动作"), the rejected sample
                # is NOT counted toward invalid_steps / invalid_block_details —
                # only ``rejection_counters`` records the diagnostic count.
                forbidden_mask.add(spec.layer_idx, spec.block_idx, tup)
                rejection_counters["samples_rejected_by_optimizer"] += 1

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
                    with torch.no_grad():
                        actions_t = torch.from_numpy(fallback_padded).to(device).unsqueeze(0)
                        lp_t, _, val_t = policy.evaluate_action(
                            obs_t, actions_t, slot_mask_t, levels_t,
                            action_level_mask=action_level_mask_t,
                        )
                    chosen_eval_info = env.evaluate_step(fallback_action.tolist())
                    chosen_action_np = fallback_padded
                    chosen_log_prob = float(lp_t.item())
                    chosen_value = float(val_t.item())
                    rejection_counters["steps_fallen_back_to_baseline"] += 1
                else:
                    # Last-resort: commit the most-recently sampled action even
                    # though it failed. Should not happen in production because
                    # the runner always provides baseline_action_vec.
                    chosen_action_np = action_np_try
                    chosen_log_prob = float(log_prob_t.item())
                    chosen_value = float(value_t.item())
                    chosen_eval_info = eval_info

            assert chosen_action_np is not None and chosen_eval_info is not None

            action_np = chosen_action_np
            log_prob = chosen_log_prob
            value = chosen_value
            step_action_for_env = action_np[:n_active].tolist()

            next_obs, reward, done, info = env.commit_step(chosen_eval_info)
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
            if action_level_mask_np is not None:
                enriched_info["neighbor_selected_offsets"] = sorted(
                    int(x) for x in neighbor_selected_offsets
                )
                enriched_info["neighbor_radius"] = int(neighbor_radius)

            if on_step_end is not None:
                try:
                    on_step_end(int(ep), int(steps_taken - 1), enriched_info)
                except Exception:
                    pass
            if capture_step_infos:
                captured_step_infos.append(enriched_info)

            buffer.add(
                state=obs,
                action=action_np,
                slot_mask=slot_mask_np,
                per_slot_num_levels=levels_np,
                action_level_mask=action_level_mask_np,
                log_prob=log_prob,
                value=value,
                reward=float(reward),
                done=bool(done),
            )
            per_step_sum += float(reward)
            if "terminal_reward" in info:
                terminal_reward = float(info["terminal_reward"])
            # Extract terminal breakdown for the final EpisodeRecord. Lives in
            # ``info["terminal_info"]`` (sequential_env.py:commit_step writes it
            # there on the terminal step). Falls back to defaults when this is
            # not the terminal step or when the base env short-circuited (any
            # invalid → no compute_reward call).
            term_info_dict = info.get("terminal_info") or {}
            term_breakdown = term_info_dict.get("reward_breakdown")
            term_metrics = term_info_dict.get("metrics")
            if term_breakdown is not None:
                terminal_priority_int = int(getattr(term_breakdown, "priority", 0) or 0)
            if term_metrics is not None:
                terminal_loss_mean_val = float(getattr(term_metrics, "loss_mean", 0.0) or 0.0)
                terminal_loss_std_val = float(getattr(term_metrics, "loss_std", 0.0) or 0.0)
                terminal_metric1_val = float(getattr(term_metrics, "metric1_mean", 0.0) or 0.0)

            obs = next_obs
            if done:
                break

        episode_returns.append(per_step_sum)
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
            terminal_priority=int(terminal_priority_int),
            terminal_loss_mean=float(terminal_loss_mean_val),
            terminal_loss_std=float(terminal_loss_std_val),
            terminal_metric1_mean=float(terminal_metric1_val),
            safe_neighbor_active=bool(neighbor_mask_active),
            safe_neighbor_mutation_count=int(len(neighbor_selected_offsets)),
            safe_neighbor_radius=int(neighbor_radius if neighbor_mask_active else 0),
        )
        episode_records.append(record)
        if on_episode_end is not None:
            on_episode_end(record)

        if (ep + 1) % int(train_cfg.update_every_n_episodes) == 0:
            # 2026-05-18 hotfix: entropy schedule (see SequentialTrainConfig
            # docstring). ``ep`` is 0-indexed; the schedule uses 1-indexed
            # episode count to match the anchor boundary semantics in
            # ``force_baseline_episodes``.
            current_ent_coef = _resolve_ent_coef_schedule(
                ep_count_1based=int(absolute_episode_start + ep + 1),
                anchor_episodes=int(force_baseline_episodes),
                target_ent_coef=float(train_cfg.ppo.ent_coef),
                anchor_ent_coef=float(getattr(train_cfg, "ent_coef_anchor", 0.0)),
                ramp_episodes=int(getattr(train_cfg, "ent_coef_ramp_episodes", 240)),
            )
            metrics = sequential_ppo_update(
                policy, optimizer, buffer, train_cfg.ppo, device,
                ent_coef_override=current_ent_coef,
            )
            ppo_metric_history.append(metrics)
            buffer.clear()
            if on_ppo_update_end is not None:
                try:
                    on_ppo_update_end(dict(metrics), int(ep + 1), record)
                except Exception:
                    pass
            if (ep + 1) % int(train_cfg.log_every_n_episodes) == 0:
                log.info(
                    "[seqRL] ep=%d return=%.3f invalid_steps=%d ppo: %s",
                    ep, record.total_reward, record.invalid_steps, metrics,
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
                "--rl-variant", "blb_v3_sequential",
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


def run_sequential_via_runner(
        *,
        runner,                           # BLBStage2RLRunner (avoid circular import)
        train_cfg,                        # BLBStage2TrainConfig
        fixed_gelu,
        fixed_softmax,
        fixed_label,
        fixed_source,
        resume_checkpoint_path=None,
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

    from .baseline_bootstrap import (
        load_static_skeletons_baseline,
        static_skeletons_baseline_to_action,
    )
    from .diagnostics import (
        EpisodeStats,
        PPOUpdateStats,
        RLDiagnosticsRecorder,
    )
    from .env import BLBStage2Env, BLBStage2EnvConfig
    from .reward import BaselineCostStats, RewardWeights
    from .persistence import (
        BLBRewardCrashWatcher,
        BLBStatusBoard,
        BLBStepDetailsWriter,
        write_training_curves,
    )
    from .runner import (
        _build_legacy_compatible_best_noise_config,
        _selection_float,
        resolve_blb_persistence_dir,
    )
    from .action_space import action_dims_for_config, describe_action_vector
    from .action_io import action_vec_to_slots_list

    ev = runner.evaluator
    bullet = "*"
    log = runner._make_log_safe(ev.log)

    # ---------- 0.1) Persistent dir ----------
    legacy_progress_dir = str(getattr(ev, "noise_stage_progress_dir", "") or "")
    blb_progress_dir = resolve_blb_persistence_dir(ev)
    try:
        ev.noise_stage_progress_dir = blb_progress_dir
    except Exception:
        pass

    _seq_log_major_rule(
        log,
        "阶段 5 · 二阶段噪声强化学习（BLB v3 · per-block sequential）",
    )
    log(f"  {bullet} 模式（mode）：horizon=59 per-block sequential（自 2026-05-15 起为默认）")
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
            "rl_mode": "sequential_per_block",
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
    probe_batches = runner._build_probe_batches(ev, train_cfg)
    train_cfg.probe_batch_count = max(1, int(len(probe_batches) or train_cfg.probe_batch_count))
    log(f"  {bullet} 评估子集：batch 数 = {len(probe_batches)}")

    rescale_bridge = runner._build_rescale_bridge(train_cfg, log=log)

    ss_baseline_obj = load_static_skeletons_baseline(
        rescale_optimizer_root=str(train_cfg.inproc_rescale_optimizer_root),
        dataset=str(train_cfg.profile),
        num_layers=int(ev.total_layers),
        gelu_per_layer=[int(x) for x in fixed_gelu.reshape(-1)],
        softmax_per_layer=[int(x) for x in fixed_softmax.reshape(-1)],
    )
    ss_action_vec, max_sfs, ss_cost_stats, _ss_diag = static_skeletons_baseline_to_action(
        ss_baseline_obj,
        snap_sf_to_noise_table=False,
    )
    baseline_action_vec = np.asarray(ss_action_vec, dtype=np.int64).reshape(-1)
    log(f"  {bullet} static_skeletons baseline loaded from {ss_baseline_obj.archive_path}")

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
        ),
    )
    base_env.sync_degree_vectors_from_model()

    # ---------- 3.5) Multi-GPU reward-probe runner (opt-in) ----------
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
            log_fn=lambda m: log(f"  [multi-gpu] {m}"),
        )

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

    # v3 cost path: the user spec says typical_bits_drop ≈ baseline / num_layers
    # (saving "one layer's worth of bits" → bits_norm ≈ 1.0). Override the
    # random-sample estimate with this structural normalizer so bits / fusion / k
    # weights sit at the user-specified 1 / 30 / 30 ratio. typical_fusion (12) and
    # typical_k_drop (5 = K_LEVELS range 8→13) are static structural maxima.
    baseline.typical_bits_drop = float(
        max(baseline.total_bits_sum / max(int(base_env.num_layers), 1), 1.0)
    )
    baseline.typical_fusion_count = 12.0
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

    # Resolve gates from the noisy preflight + the user's tolerances.
    # tolerances come from rl_tune.py CLI (stage2_limit_tolerance,
    # stage2_stability_tolerance), which the launcher feeds from the preset
    # (defaults 0.005 / 0.005 in mrpc-blb-stage2-rl.conf).
    allowed_acc_drop = max(0.0, float(getattr(ev, "stage2_limit_tolerance", 0.05)))
    stability_tol = max(0.0, float(getattr(ev, "stage2_stability_tolerance", 0.05)))

    user_acc_threshold = float(base_env.acc_threshold)
    if not (np.isfinite(user_acc_threshold) and user_acc_threshold > 0.0):
        # Default: floor the gate at (noisy baseline accuracy − tolerance) so
        # actions that wreck accuracy get caught by priority 1 instead of
        # masquerading as cost-priority candidates. The probe-size guard avoids
        # false P1(acc) episodes where the all-max baseline itself lands one
        # discrete probe sample below the nominal threshold.
        new_acc_threshold = _noisy_accuracy_threshold_with_probe_guard(
            noisy_baseline_metric1=float(noisy_baseline_metric1),
            allowed_acc_drop=float(allowed_acc_drop),
            probe_size=int(getattr(ev, "stage2_probe_size", 256)),
        )
        base_env.acc_threshold = new_acc_threshold

    # v3: derive a separate m2 threshold from the noisy m2 baseline. Same
    # tolerance / probe-size guard as m1 — user spec (2026-05-20) confirms the
    # per-metric thresholds differ only because baseline.m1 ≠ baseline.m2.
    if base_env.acc_threshold_m2 is None:
        base_env.acc_threshold_m2 = _noisy_accuracy_threshold_with_probe_guard(
            noisy_baseline_metric1=float(noisy_baseline_metric2),
            allowed_acc_drop=float(allowed_acc_drop),
            probe_size=int(getattr(ev, "stage2_probe_size", 256)),
        )

    user_stab_threshold = float(base_env.stab_threshold)
    stab_calib_summary = ""
    if not np.isfinite(user_stab_threshold):
        # 2026-05-18 (rdv2): v2 公式 = baseline_loss_std × (1 + tolerance).
        # 之前一版尝试用 5 个均匀随机 action 的 loss_std P90 做 dynamic
        # threshold，但 577-dim 均匀随机几乎必然 invalid_chain（report
        # `2026-05-18_dynamic_stab_calibration_fallback` 印证：25 次 0 个 valid），
        # 校准失败回落到 0.05，问题没改。
        #
        # 现在 reward 是 v2-style **soft** penalty + clipped + tier_bonus
        # （见 reward.py / ADR-007），即使 stab_excess 很大也只贡献 -5 给 shaping，
        # 不会让 reward 跌到 -150。所以 stab_threshold 设紧一点没关系，stab_ok
        # 偶尔达到就拿 +20 额外 tier_bonus，达不到就拿 +20 (metric_ok)，cost
        # 信号继续可见。
        derived = noisy_baseline_loss_std * (1.0 + stability_tol)
        base_env.stab_threshold = float(max(derived, 0.01))
        stab_calib_summary = (
            f"v2 formula: noisy_baseline_loss_std={noisy_baseline_loss_std:.4f} × "
            f"(1 + tol={stability_tol:.4f}) → stab_threshold = {base_env.stab_threshold:.4f}  "
            f"(soft cap under clipped+tier reward; ADR-007)"
        )

    log(
        f"  {bullet} 基线噪声预热（noisy baseline preflight）："
        f"acc(noisy)={noisy_baseline_metric1:.4f}  "
        f"loss_std(noisy)={noisy_baseline_loss_std:.4f}  "
        f"loss_mean(noisy)={noisy_baseline_loss_mean:.4f}"
    )
    log(
        f"  {bullet} 校准后硬约束阈值（calibrated gates）："
        f"acc_threshold={base_env.acc_threshold:.4f}  "
        f"stab_threshold={base_env.stab_threshold:.4f}  "
        f"(limit_tol={allowed_acc_drop:.4f}, stab_tol={stability_tol:.4f})"
    )
    if stab_calib_summary:
        log(f"  {bullet} 稳定阈值校准来源（stab calibration source）：{stab_calib_summary}")

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
        f"acc_threshold={base_env.acc_threshold:.4f}, stab_threshold={base_env.stab_threshold:.4f}",
        f"static_skeletons archive：{ss_baseline_obj.archive_path}",
    ])

    # ---------- 5) sequential env + policy ----------
    seq_env_cfg = SequentialEnvConfig(
        invalid_penalty=float(getattr(train_cfg, "sequential_invalid_penalty", 1.0)),
        cost_shaping_coeff=float(getattr(train_cfg, "sequential_cost_shaping_coeff", 0.05)),
        fusion_shaping_coeff=float(getattr(train_cfg, "sequential_fusion_shaping_coeff", 0.0)),
        early_terminate_on_invalid=bool(getattr(train_cfg, "sequential_early_terminate_on_invalid", False)),
    )
    seq_env = BLBStage2SequentialEnv(base_env=base_env, env_cfg=seq_env_cfg)

    torch.manual_seed(int(train_cfg.seed))
    np.random.seed(int(train_cfg.seed) % (2**32))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_cfg = SequentialPolicyConfig(
        state_dim=int(seq_env.state_dim),
        max_step_dim=int(seq_env.max_step_dim),
        max_num_levels=6,
        horizon=int(seq_env.horizon),
        num_layers=int(ev.total_layers),
    )
    policy = BLBStage2SequentialPolicy(policy_cfg).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(train_cfg.ppo.lr))

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
            preferred = _compute_per_slot_mode_preferred(
                schedule=seq_env.schedule,
                baseline_action_vec=baseline_action_vec,
                max_step_dim=policy_cfg.max_step_dim,
                fallback_idx=int(LEVELS_F) - 1,
            )
            policy.apply_preferred_per_step_bias(
                preferred,
                gain=float(train_cfg.warmstart_bias_gain),
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
    _preview_ramp = int(getattr(train_cfg, "ent_coef_ramp_episodes", 240))

    _seq_block_title(log, "训练超参与环境设置（Training hyperparameters · sequential per-block）")
    _seq_log_rounded_box(log, [
        f"Sequential env：horizon={seq_env.horizon}    "
        f"max_step_dim={seq_env.max_step_dim}    "
        f"state_dim={seq_env.state_dim}    "
        f"device={str(device)}",
        f"Policy：state_dim={policy_cfg.state_dim}, d_hidden={policy_cfg.d_hidden}, "
        f"head=[{policy_cfg.max_step_dim}×{policy_cfg.max_num_levels}]    "
        f"num_layers={policy_cfg.num_layers}",
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
        f"Warmstart：bias={warmstart_applied}    "
        f"gain={float(train_cfg.warmstart_bias_gain):.3g}    "
        f"force_baseline_episodes={int(_preview_force_baseline_episodes)}    "
        f"{preferred_summary}",
        f"Safe neighbor curriculum：enabled={bool(getattr(train_cfg, 'warmstart_neighbor_sampling', True))}    "
        f"mutable_offsets={len(mutable_neighbor_offsets)}    "
        f"ramp={int(getattr(train_cfg, 'warmstart_neighbor_ramp_episodes', 0) or train_cfg.total_episodes)}    "
        f"max_mutations={int(getattr(train_cfg, 'warmstart_neighbor_max_mutations', 8))}    "
        f"max_radius={int(getattr(train_cfg, 'warmstart_neighbor_max_radius', 2))}",
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
    best_action_vec: Optional[np.ndarray] = None
    if effective_resume_path and os.path.isfile(effective_resume_path):
        try:
            ckpt = torch.load(effective_resume_path, map_location=device)
            ckpt_variant = str(ckpt.get("rl_variant", "") or "")
            if ckpt_variant and ckpt_variant != "blb_v3_sequential":
                log(
                    f"  [resume][warning] checkpoint at {effective_resume_path} "
                    f"has rl_variant={ckpt_variant!r} (expected 'blb_v3_sequential'); "
                    f"skipping load to avoid policy-shape mismatch. Training will "
                    f"start fresh."
                )
            else:
                if "policy" in ckpt:
                    policy.load_state_dict(ckpt["policy"])
                if "optimizer" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer"])
                start_episode = int(ckpt.get("episode", 0))
                if "best_reward" in ckpt:
                    try:
                        best_reward = float(ckpt["best_reward"])
                    except Exception:
                        best_reward = -float("inf")
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
        ent_coef_anchor=float(getattr(train_cfg, "ent_coef_anchor", 0.0)),
        ent_coef_ramp_episodes=int(getattr(train_cfg, "ent_coef_ramp_episodes", 240)),
        absolute_episode_start=int(start_episode),
        warmstart_neighbor_sampling=bool(getattr(train_cfg, "warmstart_neighbor_sampling", True)),
        warmstart_neighbor_ramp_episodes=int(
            getattr(train_cfg, "warmstart_neighbor_ramp_episodes", 0)
            or int(train_cfg.total_episodes)
            or 1
        ),
        warmstart_neighbor_max_mutations=int(
            getattr(train_cfg, "warmstart_neighbor_max_mutations", 8)
        ),
        warmstart_neighbor_max_radius=int(
            getattr(train_cfg, "warmstart_neighbor_max_radius", 2)
        ),
        warmstart_mutable_full_offsets=list(mutable_neighbor_offsets),
    )

    episode_returns: List[float] = []
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

    diag_recorder = RLDiagnosticsRecorder(
        output_dir=blb_progress_dir,
        num_layers=int(ev.total_layers),
        num_action_slots=int(num_action_slots),
        max_action_levels=6,
        top_k=20,
        log_fn=log,
        slots_view_builder=_slots_view_builder,
    )
    # Provide the static_skeletons baseline so top-K rows in the summary
    # can show *diffs* against it (which slots actually changed vs baseline).
    try:
        diag_recorder.set_baseline_action_vec(baseline_action_vec)
    except Exception as exc:
        log(f"  [diag][warning] set_baseline_action_vec failed: {exc}")
    diag_recorder.set_meta({
        "profile": str(train_cfg.profile),
        "fixed_label": str(fixed_label),
        "fixed_source": str(fixed_source),
        "rl_variant": "blb_v3_sequential",
        "total_episodes_planned": int(total_episodes_planned),
        "rollout_size": int(train_cfg.rollout_size),
        "save_interval": int(train_cfg.save_interval),
        "ppo_lr": float(train_cfg.ppo.lr),
        "ppo_clip_range": float(train_cfg.ppo.clip_range),
        "ppo_ent_coef": float(train_cfg.ppo.ent_coef),
        "ppo_value_coef": float(train_cfg.ppo.value_coef),
        "invalid_penalty": float(seq_env_cfg.invalid_penalty),
        "cost_shaping_coeff": float(seq_env_cfg.cost_shaping_coeff),
        "fusion_shaping_coeff": float(seq_env_cfg.fusion_shaping_coeff),
        "early_terminate_on_invalid": bool(seq_env_cfg.early_terminate_on_invalid),
        "acc_threshold": float(base_env.acc_threshold),
        "stab_threshold": float(base_env.stab_threshold),
        "static_skeletons_archive": str(ss_baseline_obj.archive_path),
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
        nonlocal best_reward, best_action_vec
        episode_returns.append(float(record.total_reward))
        rollout_avg_window.append(float(record.total_reward))
        rollout_invalid_window.append(int(record.invalid_steps))
        rollout_valid_window.append(int(record.valid_step_count))
        rollout_terminal_window.append(float(record.terminal_reward))

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
                    f"m1={float(record.terminal_metric1_mean):.4f}"
                ),
                (
                    f"safe_neighbor: active={bool(record.safe_neighbor_active)}  "
                    f"mutated_offsets={int(record.safe_neighbor_mutation_count)}  "
                    f"radius={int(record.safe_neighbor_radius)}"
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

        # Track best
        is_new_best = float(record.total_reward) > best_reward
        if is_new_best:
            best_reward = float(record.total_reward)
            best_action_vec = np.asarray(seq_env._pending_full_vec, dtype=np.int64).copy()
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
            # without grepping details/ files. m1 for MRPC = accuracy; the
            # second metric (m2) isn't currently captured in EpisodeRecord —
            # add it here once env.py threads it through.
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
                f"priority={_prio_label}  "
                f"total_bits={record.total_bits_sum_over_steps}  "
                f"fusion={record.fusion_count_sum_over_steps}"
            )
            log("  " + bullet + " 当前 best action（decoded slots；不输出 action index）：")
            for snippet_line in _format_best_action_slots(best_action_vec):
                log("      " + snippet_line)

        # Periodic checkpoint
        if (record.episode_idx + 1) % int(train_cfg.save_interval) == 0:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                payload = {
                    "policy": policy.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "episode": int(start_episode + record.episode_idx + 1),
                    "best_reward": float(best_reward),
                    "best_action": (
                        best_action_vec.tolist() if best_action_vec is not None else None
                    ),
                    "rl_variant": "blb_v3_sequential",
                    # Persist the forbidden-action mask so the next resume
                    # doesn't have to re-discover the same invalid tuples.
                    "forbidden_mask_records": forbidden_mask.to_json_records(),
                }
                tmp = save_path + ".tmp"
                torch.save(payload, tmp)
                os.replace(tmp, save_path)
                log(
                    f"  [checkpoint] 已保存 · 回合 {start_episode + record.episode_idx + 1} "
                    f"→ {save_path}  ·  {forbidden_mask.summary()}"
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
                    "best_reward": float(best_reward),
                    "safe_neighbor_active": bool(record.safe_neighbor_active),
                    "safe_neighbor_mutation_count": int(record.safe_neighbor_mutation_count),
                    "safe_neighbor_radius": int(record.safe_neighbor_radius),
                },
            )
        except Exception:
            pass

        # --- Long-term diagnostics: per-episode JSONL + top-K + heatmap update.
        # The recorder owns its own try/except internally; we still wrap to keep
        # training resilient if the dataclass schema ever drifts.
        try:
            full_vec_now = (
                np.asarray(seq_env._pending_full_vec, dtype=np.int64).copy()
                if getattr(seq_env, "_pending_full_vec", None) is not None
                else None
            )
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
                    safe_neighbor_active=bool(record.safe_neighbor_active),
                    safe_neighbor_mutation_count=int(record.safe_neighbor_mutation_count),
                    safe_neighbor_radius=int(record.safe_neighbor_radius),
                ),
                full_action_vec=full_vec_now,
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
            f"当前 best={best_reward:+.4f}",
            f"policy_loss={metrics.get('policy_loss', 0.0):+.4f}  ·  "
            f"value_loss={metrics.get('value_loss', 0.0):+.4f}  ·  "
            f"entropy={metrics.get('entropy', 0.0):+.4f}  ·  "
            f"clip_fraction={metrics.get('clip_fraction', 0.0):.3f}  ·  "
            f"ent_coef={metrics.get('ent_coef', 0.0):.5f}",
            f"LR={optimizer.param_groups[0]['lr']:.6f}  ·  "
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
            ))
        except Exception as exc:
            log(f"  [diag][warning] record_ppo_update failed: {exc}")

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
        is_last_update = (
            start_episode + completed_episodes >= total_episodes_planned
        )
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
                f"训练期最优（train best）得分: {best_reward:+.4f}",
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
        except Exception as exc:
            log(f"  [resume][warning] failed to restore forbidden_mask: {exc}")

    # 2026-05-18 (rdv2 hotfix): force first N episodes to use the baseline
    # action so the value function calibrates around +45 (baseline reward)
    # and PPO pushes policy mass toward baseline before exploration starts.
    # Without this, the warmstart-biased policy still samples ~80% of slots
    # uniformly at random for kinds whose baseline index is not the bias
    # target — virtually no rollout matches baseline closely enough to
    # satisfy acc_threshold, and every reward collapses to ~-7. See
    # `reports/stage2_rl/bug_reports/2026-05-18_stage2_rl_rdv2_negative_reward_startup/`.
    # The fallback ``rollout_size * 2`` mirrors warmstart_anchor_episodes
    # in the legacy single-shot runner.
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
        on_episode_end=_episode_callback,
        on_ppo_update_end=_ppo_update_end_callback,
        on_step_end=_step_callback,
        capture_step_infos=False,  # save memory; we surface aggregates instead
        logger=logging.getLogger("blb_stage2_rl.sequential"),
        forbidden_mask=forbidden_mask,
        baseline_action_vec=baseline_action_vec,
        max_rejection_retries=32,
        force_baseline_episodes=_force_baseline_episodes,
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
        f"训练期最优 best_reward：{best_reward:+.4f}",
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
    best_action_description_paths: Dict[str, str] = {}
    baseline_action_description_paths: Dict[str, str] = {}
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
            best_action_description_paths = _write_action_description_files(
                blb_progress_dir, best_desc, label="best", log_fn=log,
            )
            if best_action_description_paths.get("md"):
                log(f"  {bullet} 最优动作可读说明 → {best_action_description_paths['md']}")
        except Exception as exc:
            log(f"  [persist][warning] best action description write failed: {exc}")

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
            "design": "v2-style rdv2",
            "cost_weight": float(getattr(weights, "cost_weight", 0.0) or 0.0),
            "lambda_stab": float(getattr(weights, "lambda_stab", 0.0) or 0.0),
            "invalid_penalty": float(getattr(weights, "invalid_penalty", 0.0) or 0.0),
            "reward_clip_min": float(getattr(weights, "reward_clip_min", -5.0)),
            "reward_clip_max": float(getattr(weights, "reward_clip_max", 5.0)),
            "tier_metric_bonus": float(getattr(weights, "tier_metric_bonus", 0.0) or 0.0),
            "tier_stability_bonus": float(getattr(weights, "tier_stability_bonus", 0.0) or 0.0),
            "baseline_metric1": float(getattr(weights, "baseline_metric1", 0.0) or 0.0),
        }
        report_path = write_blb_final_report(
            blb_progress_dir,
            run_basename=run_basename,
            profile=str(train_cfg.profile),
            total_episodes=int(total_episodes_planned),
            completed_episodes=int(start_episode + len(episode_returns)),
            elapsed_sec=float(elapsed),
            best_reward=float(best_reward),
            best_breakdown=None,
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
        model_type=str(getattr(ev, "model_type", "bert-base")),
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

    # ---------- 8) Training curve PNG/NPZ ----------
    try:
        curve_paths = write_training_curves(
            blb_progress_dir,
            episode_returns=episode_returns,
            best_reward_curve=[float(best_reward)] * len(episode_returns) if episode_returns else [],
            ppo_loss_curve=[float(m.get("policy_loss", 0.0)) for m in seq_result.get("ppo_metrics", [])],
            log_fn=log,
        )
        if curve_paths.get("png"):
            log(f"  {bullet} 训练曲线 PNG → {curve_paths['png']}")
    except Exception as exc:
        log(f"  [警告] 写训练曲线失败：{exc}")

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
    limit_loss = float(limit_dict["loss"])
    limit_p = float(limit_dict["metric1"])
    limit_s = float(limit_dict["metric2"])

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
        "selection_diagnostics": {
            "selection_mode": "blb_v3_sequential_runtime_best",
            "best_reward": float(best_reward),
            "best_action_vec": (
                best_action_vec.tolist() if best_action_vec is not None else None
            ),
        },
        "blb_v3_best_action_vec": (
            best_action_vec.tolist() if best_action_vec is not None else None
        ),
        "blb_v3_best_reward": float(best_reward),
        "blb_v3_profile": str(train_cfg.profile),
        "blb_v3_total_episodes": int(train_cfg.total_episodes),
        "rl_variant": "blb_v3_sequential",
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
            "k_trials": int(train_cfg.num_trials_per_step),
            "probe_size": int(getattr(ev, "stage2_probe_size", 256)),
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
