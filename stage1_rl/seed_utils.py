"""Torch-free Stage-1 rollout seeding + episode assignment.

These two helpers define the **GPU-count-independence contract** for the
Stage-1 multi-GPU rollout, and they are deliberately torch-free so the contract
is unit-testable without a GPU / torch install (``parallel_runner`` imports
torch, so it can't be imported on a torch-free box).

Contract: episode ``g`` in PPO window ``w`` is seeded by
``derive_episode_seed(base, w, g)`` keyed on the GLOBAL episode index ``g``
(not on which worker runs it), and the runner reassembles rollouts in global
order. So a window runs the identical seeded episodes in the identical order
for any worker / GPU count — Stage-1 results don't change with #GPUs. This
mirrors Stage-2's ProbeRunner, which seeds by global trial index.
"""
from __future__ import annotations

from typing import List

_WORKER_SEED_MULTIPLIER = 2654435761


def derive_episode_seed(base_seed: int, window_idx: int, global_episode_idx: int) -> int:
    """Per-(window, GLOBAL episode index) seed — independent of worker/GPU count.

    Depends only on ``base_seed``, the PPO ``window_idx`` and the episode's
    GLOBAL position in the window, so episode ``g`` samples the same actions
    whether one worker runs the whole window or N workers split it.
    """
    h = int(base_seed) & 0x7FFFFFFFFFFFFFFF
    h ^= int(window_idx) * (_WORKER_SEED_MULTIPLIER + 1)
    h ^= int(global_episode_idx) * (_WORKER_SEED_MULTIPLIER + 2)
    return int(h & 0x7FFFFFFFFFFFFFFF)


def assign_global_episodes(total_episodes: int, num_workers: int) -> List[List[int]]:
    """Split global episode indices ``[0, total_episodes)`` into ``num_workers``
    balanced, contiguous chunks; return the list of global indices per worker.

    The split only balances LOAD — it never changes results, because every
    episode is seeded by its GLOBAL index and the caller reassembles rollouts
    in global order. So 1 worker (one chunk) and N workers (N chunks) run the
    identical seeded episode set. Remainder episodes go to the lowest-indexed
    workers (counts differ by at most 1), so any worker/GPU count is supported
    without dropping episodes.
    """
    if total_episodes <= 0 or num_workers <= 0:
        return [[] for _ in range(max(0, num_workers))]
    base = total_episodes // num_workers
    rem = total_episodes % num_workers
    out: List[List[int]] = []
    start = 0
    for w in range(num_workers):
        n = base + (1 if w < rem else 0)
        out.append(list(range(start, start + n)))
        start += n
    return out
