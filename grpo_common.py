"""Torch-free GRPO helpers shared by Stage-1 and Stage-2.

GRPO (Group Relative Policy Optimization) replaces PPO's critic-based advantage
with a *group-relative* one: within a group of trajectories sampled from the same
state, each trajectory's advantage is its outcome return normalized against the
group. Here the "group" is the batch of episodes collected in one PPO/GRPO update
window — every episode starts from the same frozen model + static_skeletons
baseline, so the whole window is one group (see the 2026-05-31 PPO→GRPO design).

This module is deliberately torch-free (numpy only) so the normalization math is
unit-testable without a torch install (the rest of the RL stack imports torch).
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

GRPO_DEFAULT_EPS = 1e-4


def grpo_group_normalize(
        returns: Sequence[float],
        *,
        eps: float = GRPO_DEFAULT_EPS,
        ) -> np.ndarray:
    """Group-relative normalization of per-trajectory (episode) returns.

    ``advantage_i = (R_i - mean(R)) / (std(R) + eps)`` using the population
    (biased) std, matching the standard GRPO formulation. Degenerate groups are
    handled gracefully: a single member, a zero-spread group, or any non-finite
    value all collapse to advantage 0.0 (no within-group signal) rather than
    dividing by ~eps and exploding.
    """
    arr = np.asarray(returns, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return np.zeros(0, dtype=np.float32)
    mean = float(np.mean(arr))
    std = float(np.std(arr))  # population std (matches GRPO)
    adv = (arr - mean) / (std + float(eps))
    adv = np.where(np.isfinite(adv), adv, 0.0)
    return adv.astype(np.float32)


def segment_episode_returns(
        rewards: Sequence[float],
        dones: Sequence[bool],
        ) -> tuple[np.ndarray, np.ndarray]:
    """Segment a flat reward/done stream into per-episode total returns.

    Episodes end ON the transition whose ``done`` is True (the buffer layout
    used by ``SequentialRolloutBuffer``). Returns ``(episode_id_per_step,
    episode_total_returns)`` where ``episode_id_per_step[t]`` indexes into
    ``episode_total_returns``. A trailing episode without a closing ``done`` is
    still counted. Outcome return = undiscounted sum of the episode's rewards.
    """
    r = np.asarray(rewards, dtype=np.float64).reshape(-1)
    d = np.asarray(dones, dtype=bool).reshape(-1)
    n = r.size
    ep_id = np.zeros(n, dtype=np.int64)
    ep_returns: List[float] = []
    cur = 0.0
    cid = 0
    opened = False
    for t in range(n):
        cur += float(r[t])
        ep_id[t] = cid
        opened = True
        if d[t]:
            ep_returns.append(cur)
            cur = 0.0
            cid += 1
            opened = False
    if opened:  # trailing episode with no closing done marker
        ep_returns.append(cur)
    return ep_id, np.asarray(ep_returns, dtype=np.float64)


def grpo_per_step_advantages(
        rewards: Sequence[float],
        dones: Sequence[bool],
        *,
        eps: float = GRPO_DEFAULT_EPS,
        outlier_clip: float = 0.0,
        ) -> np.ndarray:
    """Outcome-supervised group-relative advantage, broadcast to every step.

    For each episode i: ``R_i`` = sum of its rewards; ``A_i`` = group-normalized
    ``R_i`` over all episodes in the window; every step of episode i gets ``A_i``.
    Returns a per-step float32 array aligned with the input order. ``outlier_clip``
    > 0 clamps advantages to ``[-clip, +clip]`` for stability.
    """
    ep_id, ep_returns = segment_episode_returns(rewards, dones)
    if ep_returns.size == 0:
        return np.zeros(0, dtype=np.float32)
    ep_adv = grpo_group_normalize(ep_returns, eps=eps)
    adv = ep_adv[ep_id]
    if outlier_clip and outlier_clip > 0:
        adv = np.clip(adv, -float(outlier_clip), float(outlier_clip))
    return adv.astype(np.float32)
