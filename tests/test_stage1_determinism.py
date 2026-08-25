"""GPU-count-independence contract for the Stage-1 multi-GPU rollout (torch-free).

Proves the property the user requires: the seeded episode sequence a PPO window
produces is IDENTICAL no matter how many workers / GPUs split it (1, 4, 5, ...),
so Stage-1 results don't change with the server's GPU count. The torch-free
``seed_utils`` core is what makes this checkable without a GPU box.
"""
import pathlib
import sys
import unittest

_REPO = pathlib.Path(__file__).resolve().parents[1]


if str(_REPO / "stage1_rl") not in sys.path:
    sys.path.insert(0, str(_REPO / "stage1_rl"))

import seed_utils

assign_global_episodes = seed_utils.assign_global_episodes
derive_episode_seed = seed_utils.derive_episode_seed


def _window_seed_sequence(base, window, total, num_workers):
    """Reproduce ``run_window``'s per-episode seeds, reassembled in GLOBAL order
    (exactly what the runner stores at ``results[g]``)."""
    assignments = assign_global_episodes(total, num_workers)
    seeds = [None] * total
    for chunk in assignments:
        for g in chunk:
            seeds[g] = derive_episode_seed(base, window, g)
    return seeds


class AssignGlobalEpisodesTest(unittest.TestCase):
    def test_covers_every_index_exactly_once_in_order(self):
        for total in (0, 1, 24, 120, 121):
            for nw in (1, 2, 4, 5, 7):
                chunks = assign_global_episodes(total, nw)
                self.assertEqual(len(chunks), nw)
                flat = [g for c in chunks for g in c]
                self.assertEqual(
                    flat, list(range(total)),
                    f"total={total} nw={nw}: not a clean [0,total) cover",
                )

    def test_balanced_within_one(self):
        chunks = assign_global_episodes(120, 7)
        counts = [len(c) for c in chunks]
        self.assertEqual(sum(counts), 120)
        self.assertLessEqual(max(counts) - min(counts), 1)

    def test_more_workers_than_episodes_leaves_empties(self):
        chunks = assign_global_episodes(3, 5)
        self.assertEqual([len(c) for c in chunks], [1, 1, 1, 0, 0])
        self.assertEqual([g for c in chunks for g in c], [0, 1, 2])


class GpuCountIndependenceTest(unittest.TestCase):
    def test_seed_sequence_identical_across_worker_counts(self):
        base, window, total = 42, 3, 120
        ref = _window_seed_sequence(base, window, total, 1)
        self.assertTrue(all(s is not None for s in ref))
        for nw in (2, 4, 5, 7, 120):
            self.assertEqual(
                _window_seed_sequence(base, window, total, nw), ref,
                f"window seed sequence differs for {nw} workers — would change results",
            )

    def test_episode_seed_is_pure_function_of_global_index(self):
        self.assertEqual(derive_episode_seed(7, 2, 33), derive_episode_seed(7, 2, 33))
        self.assertNotEqual(derive_episode_seed(7, 2, 33), derive_episode_seed(7, 2, 34))
        self.assertNotEqual(derive_episode_seed(7, 2, 33), derive_episode_seed(7, 3, 33))

    def test_different_windows_do_not_collide_sequences(self):
        base, total = 42, 120
        self.assertNotEqual(
            _window_seed_sequence(base, 0, total, 4),
            _window_seed_sequence(base, 1, total, 4),
        )


if __name__ == "__main__":
    unittest.main()
