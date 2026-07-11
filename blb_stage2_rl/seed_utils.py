"""Torch-free seed derivation for the Stage-2 episode-parallel rollout.

GPU-count-independence contract (mirrors ``stage1_rl/seed_utils.py``): every
random draw in the episode-parallel path is keyed by the *global* episode
index (plus step / trial / attempt), never by ``(worker_idx, local_idx)`` or
by wall clock, so the same ``(base_seed, global_episode)`` produces the same
stream no matter how many workers/GPUs split the window.

Three independent streams are salted apart so policy sampling, probe noise,
and the PPO-update shuffle can never alias each other:

* policy  — per ``(global_episode, step, attempt)``; seeds the worker
  device's CUDA Philox generator right before ``policy.sample_action``.
* probe   — per ``global_episode``; the per-trial offset is the same
  ``seed XOR trial_idx * KNUTH`` mix as ``probe_runner._trial_seed`` and
  reseeds the worker-local scoped noise generator
  (``function_handler.noise_rng_scope`` +
  ``function_handler.reseed_noise_rng_for_device``).
* update  — per PPO update index; reseeds the global numpy / torch RNGs
  before ``sequential_ppo_update`` (its minibatch shuffle uses
  ``np.random.shuffle``).

``PREFLIGHT_EPISODE = -1`` reserves a probe stream for the noisy baseline
preflight so threshold calibration is reproducible too.
"""

from __future__ import annotations

import operator
from typing import List

# Knuth multiplicative-hash constant — same as stage1_rl/seed_utils.py and
# probe_runner._TRIAL_SEED_MULTIPLIER, so the trial mix stays consistent.
_KNUTH = 2654435761
_MASK = 0x7FFFFFFFFFFFFFFF

# Stream salts (arbitrary fixed constants; only need to be distinct).
_POLICY_SALT = 0x515AC0DE
_PROBE_SALT = 0x09E3779B9
_UPDATE_SALT = 0x2545F4914F6CDD1D

#: Reserved pseudo-episode index for the noisy baseline preflight probe.
PREFLIGHT_EPISODE = -1


def derive_policy_step_seed(
        base_seed: int,
        global_episode_idx: int,
        step_idx: int,
        attempt_idx: int = 0,
        ) -> int:
    """Seed for one policy-sampling draw (one rejection-loop attempt)."""
    h = int(base_seed) & _MASK
    h ^= _POLICY_SALT
    h ^= (int(global_episode_idx) * _KNUTH) & _MASK
    h ^= (int(step_idx) * (_KNUTH + 1)) & _MASK
    h ^= (int(attempt_idx) * (_KNUTH + 2)) & _MASK
    return h & _MASK


def derive_probe_seed(base_seed: int, global_episode_idx: int) -> int:
    """Per-episode base seed for the terminal K-trial probe noise."""
    h = int(base_seed) & _MASK
    h ^= _PROBE_SALT
    h ^= (int(global_episode_idx) * _KNUTH) & _MASK
    return h & _MASK


def derive_baseline_group_probe_seed(base_seed: int, group_idx: int) -> int:
    """Deterministic probe seed for one robust-baseline evidence group."""
    if isinstance(group_idx, bool):
        raise TypeError("group_idx must be an integer, not bool")
    try:
        normalized_group_idx = operator.index(group_idx)
    except TypeError as exc:
        raise TypeError("group_idx must be an integer") from exc
    if normalized_group_idx < 0:
        raise ValueError("group_idx must be non-negative")
    return derive_probe_seed(
        base_seed,
        PREFLIGHT_EPISODE - 1 - int(normalized_group_idx),
    )


def derive_probe_trial_seed(probe_seed: int, trial_idx: int) -> int:
    """Per-trial offset — identical mix to ``probe_runner._trial_seed``."""
    return int((int(probe_seed) ^ (int(trial_idx) * _KNUTH)) & _MASK)


def derive_update_seed(base_seed: int, update_idx: int) -> int:
    """Deterministic pre-PPO-update reseed (np shuffle + torch)."""
    h = int(base_seed) & _MASK
    h ^= _UPDATE_SALT & _MASK
    h ^= (int(update_idx) * _KNUTH) & _MASK
    return h & _MASK


def assign_global_episodes(total_episodes: int, num_workers: int) -> List[List[int]]:
    """Balanced contiguous chunks of global episode indices.

    Covers ``range(total_episodes)`` exactly once for any worker count
    (counts differ by at most 1; remainder goes to the lowest workers).
    Same contract as ``stage1_rl.seed_utils.assign_global_episodes``.
    """
    total = max(0, int(total_episodes))
    n = max(1, int(num_workers))
    base = total // n
    rem = total % n
    out: List[List[int]] = []
    cursor = 0
    for w in range(n):
        count = base + (1 if w < rem else 0)
        out.append(list(range(cursor, cursor + count)))
        cursor += count
    return out


def assign_global_episodes_interleaved(total_episodes: int, num_workers: int) -> List[List[int]]:
    """Deterministic round-robin episode assignment for parallel rollout.

    This covers ``range(total_episodes)`` exactly once while preserving the
    global episode index as the sole RNG key.  Unlike contiguous chunks, it
    spreads deterministic episode-to-episode probe latency variance across
    workers, reducing window long-tail stalls without changing PPO assembly
    order.
    """
    total = max(0, int(total_episodes))
    n = max(1, int(num_workers))
    return [list(range(w, total, n)) for w in range(n)]
