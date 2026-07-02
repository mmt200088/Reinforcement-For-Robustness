"""Fusion-mode block-granularity safe-neighbor curriculum (torch-free).

Pure scheduling + per-step level-mask logic for the fusion-count action path
(``--blb-v3-fusion-count-action 1``). Kept torch-free (no ``action_space`` /
torch import — K-level values are passed in) so the **"the full action space
stays reachable"** invariant can be unit-tested locally. ``sequential_runner``
wires the real ``FusionStepSpec`` fields + ``K_LEVELS`` into these helpers.

Why this exists (2026-06-05): in fusion mode every episode decides ``(option, K)``
for all H blocks (47 for mrpc) jointly. With an unrestricted (open) mask from the
first post-anchor episode, even a warmstart-biased policy fuses ~6-10 blocks per
episode just by independent compounding, the accuracy collapses, every episode
lands in the same failure tier, and PPO loses a discriminating gradient (the
2026-06-04 600-ep smoke collapsed to 511/600 P1 with no recovery). This curriculum
pins most blocks to the baseline action ``(option 0, baseline K)`` early and widens
the mutable subset + per-block radius until, by the end of the ramp, the mask is
identical to the open mask. It is therefore a pure warmup that dissolves: nothing
is permanently masked, and the optimum is never hidden.
"""
from __future__ import annotations

from functools import lru_cache
import math
from typing import List, Sequence, Tuple

import numpy as np

# Fraction of total episodes over which the per-block neighborhood widens from
# {1 block, radius 1} to fully open. After the ramp the per-block mask equals the
# unrestricted open mask, so no config is ever permanently hidden.
FUSION_NEIGHBOR_RAMP_FRACTION = 0.5


def fusion_block_curriculum(
        *,
        absolute_episode_idx: int,
        anchor_episodes: int,
        ramp_episodes: int,
        horizon: int,
        max_radius: int,
        ) -> Tuple[bool, int, int]:
    """Block-granularity neighborhood schedule for fusion mode.

    Returns ``(fully_open, num_mutable_blocks, radius)``. Both the number of
    mutable blocks (1 -> ``horizon``) and the per-block radius (1 -> ``max_radius``)
    widen linearly with post-anchor progress. ``fully_open`` becomes True once the
    ramp completes; from that point the mask equals the open mask, which proves the
    full action space remains reachable (nothing is permanently masked).
    """
    horizon = max(1, int(horizon))
    ramp = max(1, int(ramp_episodes))
    after_anchor = max(0, int(absolute_episode_idx) - int(anchor_episodes))
    progress = float(after_anchor) / float(ramp)
    if progress >= 1.0:
        return True, horizon, max(1, int(max_radius))
    num_mutable = 1 + int(math.floor(progress * max(0, horizon - 1)))
    radius = 1 + int(math.floor(progress * max(0, int(max_radius) - 1)))
    return False, max(1, min(num_mutable, horizon)), max(1, radius)


@lru_cache(maxsize=256)
def _cached_near_baseline_k_indices(
        k_level_values: Tuple[int, ...],
        baseline_idx: int,
        radius: int,
        ) -> Tuple[int, ...]:
    dim = len(k_level_values)
    base_k = int(k_level_values[int(baseline_idx)])
    candidates = list(range(dim))
    candidates.sort(key=lambda idx: (abs(int(k_level_values[idx]) - base_k), int(idx)))
    keep = min(len(candidates), max(1, 2 * int(radius) + 1))
    return tuple(sorted(int(idx) for idx in candidates[:keep]))


def near_baseline_k_indices(
        *,
        k_level_values: Sequence[int],
        baseline_idx: int,
        dim: int,
        radius: int,
        ) -> List[int]:
    """K indices within ``radius`` truncation-bit steps of baseline.

    K is decoded through non-monotonic ``K_LEVELS``, so locality is by distance in
    truncation bits, not categorical index. Mirrors the per-slot path's K branch.
    """
    dim = int(dim)
    radius = max(0, int(radius))
    if dim <= 0:
        return []
    baseline_idx = int(baseline_idx)
    if baseline_idx < 0 or baseline_idx >= dim:
        raise ValueError(f"baseline K index {baseline_idx} out of width {dim}")
    k_values = tuple(int(k_level_values[idx]) for idx in range(dim))
    return list(_cached_near_baseline_k_indices(k_values, baseline_idx, radius))


def build_fusion_step_level_mask(
        *,
        fusion_num_options: int,
        k_num_levels: int,
        k_level_values: Sequence[int],
        mutable: bool,
        radius: int,
        baseline_k_index: int,
        max_step_dim: int,
        max_num_levels: int,
        ) -> np.ndarray:
    """Near-baseline per-level mask for one fusion step (slot0=option, slot1=K).

    A non-mutable block is pinned to its baseline action ``(option 0, baseline K)``.
    A mutable block may move option within ``[0, radius]`` and K within ``radius``
    truncation-bit steps of baseline. The baseline action is always allowed so every
    step stays feasible (and the forced anchor matches). When ``mutable`` is True for
    every block and ``radius`` covers all option/K widths, the result equals the
    unrestricted open mask — so the curriculum never permanently hides a config.
    """
    mask = np.zeros((int(max_step_dim), int(max_num_levels)), dtype=bool)
    n_opts = min(int(fusion_num_options), int(max_num_levels))
    n_k = min(int(k_num_levels), int(max_num_levels))
    base_k = int(baseline_k_index)
    if n_opts <= 0 or n_k <= 0:
        raise ValueError("fusion step has no option/K levels")
    if base_k < 0 or base_k >= n_k:
        raise ValueError(f"baseline K index {base_k} out of width {n_k}")
    # Baseline action — always feasible (option 0 is the all-max baseline option).
    mask[0, 0] = True
    mask[1, base_k] = True
    if not mutable:
        return mask
    opt_hi = min(n_opts - 1, max(0, int(radius)))
    for o in range(0, opt_hi + 1):
        mask[0, o] = True
    for k in near_baseline_k_indices(
            k_level_values=k_level_values, baseline_idx=base_k, dim=n_k, radius=int(radius)):
        if 0 <= int(k) < n_k:
            mask[1, int(k)] = True
    return mask


def select_mutable_step_indices(
        *,
        rng: np.random.Generator,
        horizon: int,
        num_mutable: int,
        ) -> set:
    """Random subset of step indices allowed to leave baseline this episode."""
    horizon = max(1, int(horizon))
    k = max(1, min(int(num_mutable), horizon))
    sel = rng.choice(horizon, size=k, replace=False)
    return {int(x) for x in np.atleast_1d(sel)}


# ---- scheduled forced-fusion probes (ADR-011, redesigned by ADR-012) --------
# Standing re-exploration: every ``interval`` post-anchor episodes, ONE episode
# forces fusion option 1 on every block of ONE rotating block type, scored
# normally, so PPO keeps receiving fresh evidence of fusion's value even after
# the policy's fusion logits collapse. The decision is a pure function of the
# absolute episode index -> deterministic and identical across episode-parallel
# workers (1==N preserved).
#
# ADR-012 (2026-06-12) — two corrections from the 2nd 60k run's forensics:
# (a) probes no longer force baseline K (the collector samples K and all
#     non-target blocks from the CURRENT policy, only the target block's
#     option is forced): forcing baseline K cancelled the fusion gain against
#     the policy's learned deep-K savings (b2 probe net +0.07, b5 probe net
#     -0.86 — the b5 probe was teaching that fusion is BAD);
# (b) block4 is dropped from the rotation: a 12-layer block4 fusion probe is
#     a guaranteed accuracy fail (-46, observed 100/100 in the 2nd 60k) and
#     only taught anti-fusion generalization. Selective per-layer block4
#     fusion is left to on-policy epsilon exploration under the graded
#     near-miss boundary (reward.near_miss_*).
FUSION_PROBE_BLOCK_ROTATION: Tuple[int, ...] = (2, 5)


def fusion_probe_target_block(
        absolute_episode_idx: int,
        *,
        anchor_episodes: int,
        interval: int,
        rotation: Sequence[int] = FUSION_PROBE_BLOCK_ROTATION,
        ) -> int | None:
    """Block type forced to fusion option 1 this episode, or None (normal ep).

    ``interval <= 0`` disables probes. The first post-anchor episode (rel == 0)
    is a probe so evidence starts flowing immediately after the anchor.
    """
    if int(interval) <= 0 or not rotation:
        return None
    rel = int(absolute_episode_idx) - int(anchor_episodes)
    if rel < 0 or rel % int(interval) != 0:
        return None
    return int(rotation[(rel // int(interval)) % len(rotation)])
