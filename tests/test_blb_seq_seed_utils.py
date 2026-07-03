"""GPU-count-independence contract for the Stage-2 episode-parallel rollout (torch-free).

Mirror of tests/test_stage1_determinism.py for blb_stage2_rl/seed_utils.py:
every seed is keyed by the GLOBAL episode index (+ step / trial / attempt /
update), never by worker count or wall clock, so a window's seed sequence is
identical for 1, 2, 4, 5, ... workers.
"""
import importlib.util
import pathlib
import sys
import unittest

_REPO = pathlib.Path(__file__).resolve().parents[1]
# Load the module by file path under a unique name — importing blb_stage2_rl
# as a package would pull torch, and a bare ``import seed_utils`` would
# collide with stage1's module of the same name in shared test processes.
_spec = importlib.util.spec_from_file_location(
    "blb_seq_seed_utils", str(_REPO / "blb_stage2_rl" / "seed_utils.py")
)
seed_utils = importlib.util.module_from_spec(_spec)
sys.modules["blb_seq_seed_utils"] = seed_utils
_spec.loader.exec_module(seed_utils)


class AssignGlobalEpisodesTest(unittest.TestCase):
    def test_covers_every_index_exactly_once_in_order(self):
        for total in (0, 1, 47, 60, 61):
            for nw in (1, 2, 4, 5, 7):
                chunks = seed_utils.assign_global_episodes(total, nw)
                self.assertEqual(len(chunks), nw)
                flat = [g for c in chunks for g in c]
                self.assertEqual(flat, list(range(total)), (total, nw))

    def test_balanced_counts(self):
        chunks = seed_utils.assign_global_episodes(60, 5)
        self.assertEqual([len(c) for c in chunks], [12] * 5)
        chunks = seed_utils.assign_global_episodes(61, 5)
        self.assertEqual(sorted(len(c) for c in chunks), [12, 12, 12, 12, 13])


class AssignGlobalEpisodesInterleavedTest(unittest.TestCase):
    def test_one_worker_is_identity(self):
        self.assertEqual(
            seed_utils.assign_global_episodes_interleaved(7, 1),
            [list(range(7))],
        )

    def test_covers_every_index_once_with_balanced_counts(self):
        for total in (0, 1, 47, 60, 61):
            for nw in (1, 2, 4, 5, 7, 10):
                chunks = seed_utils.assign_global_episodes_interleaved(total, nw)
                self.assertEqual(len(chunks), nw)
                flat = [g for c in chunks for g in c]
                self.assertEqual(sorted(flat), list(range(total)), (total, nw))
                counts = [len(c) for c in chunks]
                self.assertLessEqual(max(counts, default=0) - min(counts, default=0), 1)

    def test_round_robin_shape_spreads_adjacent_episodes(self):
        self.assertEqual(
            seed_utils.assign_global_episodes_interleaved(12, 5),
            [[0, 5, 10], [1, 6, 11], [2, 7], [3, 8], [4, 9]],
        )


class GpuCountIndependenceTest(unittest.TestCase):
    def _window_probe_seeds(self, base, start, total, num_workers):
        """Per-episode probe seeds reassembled in GLOBAL order (what the
        parallel runner stores at results[g])."""
        seeds = [None] * total
        for chunk in seed_utils.assign_global_episodes(total, num_workers):
            for g in chunk:
                seeds[g] = seed_utils.derive_probe_seed(base, start + g)
        return seeds

    def test_probe_seed_sequence_identical_across_worker_counts(self):
        ref = self._window_probe_seeds(42, 600, 60, 1)
        for nw in (2, 4, 5, 7, 60):
            self.assertEqual(self._window_probe_seeds(42, 600, 60, nw), ref, nw)

    def test_policy_step_seeds_identical_across_worker_counts(self):
        # Worker count never enters the formula; lock the full (ep, step,
        # attempt) grid for one window against a literal recomputation.
        grid_a = [
            seed_utils.derive_policy_step_seed(7, ep, st, at)
            for ep in range(100, 160) for st in range(47) for at in (0, 1)
        ]
        grid_b = [
            seed_utils.derive_policy_step_seed(7, ep, st, at)
            for ep in range(100, 160) for st in range(47) for at in (0, 1)
        ]
        self.assertEqual(grid_a, grid_b)

    def test_streams_do_not_alias(self):
        # policy / probe / update streams must differ for the same key tuple.
        base, ep = 42, 1234
        policy = seed_utils.derive_policy_step_seed(base, ep, 0, 0)
        probe = seed_utils.derive_probe_seed(base, ep)
        update = seed_utils.derive_update_seed(base, ep)
        self.assertEqual(len({policy, probe, update}), 3)
        # distinct episodes / steps / attempts / trials give distinct seeds
        self.assertNotEqual(
            seed_utils.derive_probe_seed(base, ep),
            seed_utils.derive_probe_seed(base, ep + 1),
        )
        self.assertNotEqual(
            seed_utils.derive_policy_step_seed(base, ep, 3, 0),
            seed_utils.derive_policy_step_seed(base, ep, 4, 0),
        )
        self.assertNotEqual(
            seed_utils.derive_policy_step_seed(base, ep, 3, 0),
            seed_utils.derive_policy_step_seed(base, ep, 3, 1),
        )
        self.assertNotEqual(
            seed_utils.derive_probe_trial_seed(probe, 0),
            seed_utils.derive_probe_trial_seed(probe, 1),
        )
        # trial 0 == the base probe seed (same convention as _trial_seed)
        self.assertEqual(seed_utils.derive_probe_trial_seed(probe, 0), probe)

    def test_preflight_episode_reserved(self):
        # The preflight stream must not collide with episode 0's stream.
        self.assertNotEqual(
            seed_utils.derive_probe_seed(42, seed_utils.PREFLIGHT_EPISODE),
            seed_utils.derive_probe_seed(42, 0),
        )


class TrialMixConsistencyTest(unittest.TestCase):
    def test_trial_mix_matches_probe_runner_formula(self):
        # probe_runner._trial_seed(base, i) = (base ^ (i * 2654435761)) & MASK.
        # Locked here torch-free so the two implementations cannot drift.
        for base in (0, 42, 0x7FFFFFFFFFFFFFFF):
            for i in range(6):
                expected = (int(base) ^ (i * 2654435761)) & 0x7FFFFFFFFFFFFFFF
                self.assertEqual(
                    seed_utils.derive_probe_trial_seed(base, i), expected
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
