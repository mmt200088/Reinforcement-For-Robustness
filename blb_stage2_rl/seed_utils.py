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

import numpy as np

# Knuth multiplicative-hash constant — same as stage1_rl/seed_utils.py and
# probe_runner._TRIAL_SEED_MULTIPLIER, so the trial mix stays consistent.
_KNUTH = 2654435761
_MASK = 0x7FFFFFFFFFFFFFFF

# Stream salts (arbitrary fixed constants; only need to be distinct).
_POLICY_SALT = 0x515AC0DE
_PROBE_SALT = 0x09E3779B9
_UPDATE_SALT = 0x2545F4914F6CDD1D

# Layerwise robust evidence reserves the low 40 bits for the existing trial
# mixer. For the supported 256-trial maximum, ``trial_idx * KNUTH`` never
# reaches bit 40, so the domain and episode bits cannot be changed by trial
# derivation. This gives a proof of disjointness across adjacent episodes.
_LAYERWISE_TRIAL_BITS = 40
_LAYERWISE_TRIAL_MASK = (1 << _LAYERWISE_TRIAL_BITS) - 1
_LAYERWISE_EPISODE_BITS = 21
_LAYERWISE_MAX_EPISODE = (1 << _LAYERWISE_EPISODE_BITS) - 1
_LAYERWISE_MAX_TRIALS = 256
_LAYERWISE_ONLINE_DOMAIN = 1
_LAYERWISE_PROMOTION_DOMAIN = 2
_LAYERWISE_DOMAIN_SHIFT = _LAYERWISE_TRIAL_BITS + _LAYERWISE_EPISODE_BITS
_LAYERWISE_LOW_SALT = 0x6C6179657277697365

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
    def nonnegative_integer(name: str, value: int) -> int:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be an integer, not bool")
        try:
            normalized = operator.index(value)
        except TypeError as exc:
            raise TypeError(f"{name} must be an integer") from exc
        if normalized < 0:
            raise ValueError(f"{name} must be non-negative")
        return int(normalized)

    normalized_base_seed = nonnegative_integer("base_seed", base_seed)
    normalized_group_idx = nonnegative_integer("group_idx", group_idx)
    return derive_probe_seed(
        normalized_base_seed,
        PREFLIGHT_EPISODE - 1 - normalized_group_idx,
    )


def derive_probe_trial_seed(probe_seed: int, trial_idx: int) -> int:
    """Per-trial offset — identical mix to ``probe_runner._trial_seed``."""
    return int((int(probe_seed) ^ (int(trial_idx) * _KNUTH)) & _MASK)


def _derive_layerwise_probe_seed(
        base_seed: int,
        stream_index: int,
        *,
        trial_count: int,
        domain: int,
        ) -> int:
    if isinstance(stream_index, (bool, np.bool_)):
        raise TypeError("stream_index must be an integer, not bool")
    try:
        index = operator.index(stream_index)
    except TypeError as exc:
        raise TypeError("stream_index must be an integer") from exc
    if index < 0 or index > _LAYERWISE_MAX_EPISODE:
        raise ValueError(
            f"stream_index must be in [0, {_LAYERWISE_MAX_EPISODE}], got {index}"
        )
    if isinstance(trial_count, (bool, np.bool_)):
        raise TypeError("trial_count must be an integer, not bool")
    try:
        count = operator.index(trial_count)
    except TypeError as exc:
        raise TypeError("trial_count must be an integer") from exc
    if count < 1 or count > _LAYERWISE_MAX_TRIALS:
        raise ValueError(
            f"trial_count must be in [1, {_LAYERWISE_MAX_TRIALS}], got {count}"
        )
    if (count - 1) * _KNUTH > _LAYERWISE_TRIAL_MASK:
        raise ValueError("trial_count exceeds the reserved layerwise trial domain")

    low = (int(base_seed) ^ _LAYERWISE_LOW_SALT) & _LAYERWISE_TRIAL_MASK
    return int(
        ((int(domain) << _LAYERWISE_DOMAIN_SHIFT)
         | (int(index) << _LAYERWISE_TRIAL_BITS)
         | low)
        & _MASK
    )


def derive_layerwise_episode_probe_seed(
        base_seed: int,
        global_episode_idx: int,
        *,
        trial_count: int = 5,
        ) -> int:
    """Base seed for one layerwise online evidence group.

    The episode occupies bits 40..60 and trial mixing is confined to bits
    0..39. Therefore trial sets for distinct supported episodes are disjoint,
    including adjacent episodes, rather than merely collision-resistant.
    """
    return _derive_layerwise_probe_seed(
        base_seed,
        global_episode_idx,
        trial_count=trial_count,
        domain=_LAYERWISE_ONLINE_DOMAIN,
    )


def derive_layerwise_promotion_probe_seed(
        base_seed: int,
        attempt_idx: int,
        *,
        trial_count: int,
        ) -> int:
    """Base seed for a domain-separated layerwise promotion attempt."""
    return _derive_layerwise_probe_seed(
        base_seed,
        attempt_idx,
        trial_count=trial_count,
        domain=_LAYERWISE_PROMOTION_DOMAIN,
    )


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
