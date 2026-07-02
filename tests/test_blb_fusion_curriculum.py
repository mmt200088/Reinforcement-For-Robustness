"""Torch-free tests for the fusion-mode block-granularity safe-neighbor curriculum.

The curriculum (``blb_stage2_rl/fusion_curriculum.py``) gently widens, per episode,
how many of the H blocks may leave the baseline action ``(option 0, baseline K)``,
and by how much. The hard requirement these tests lock is the user's constraint:

  *the curriculum must never permanently hide a config — RL must still be able to
   search the full action space; the restriction is only gradually opened.*

This is proven two ways: (1) ``fusion_block_curriculum`` reports ``fully_open`` once
the ramp completes (every block mutable, full radius); (2) at that point the
per-block level mask is byte-identical to the unrestricted open mask.
"""
from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_BLB_DIR = _REPO_ROOT / "blb_stage2_rl"
for _p in (str(_REPO_ROOT), str(_BLB_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fusion_curriculum as fcur  # bare import; blb_stage2_rl on sys.path

# Default K table (action_space.DEFAULT_K_LEVELS_LEGACY_COMPAT) — intentionally
# non-monotonic so the K-distance locality is exercised, not categorical index.
K_LEVELS = (8, 9, 11, 13, 10, 12)
N_K = len(K_LEVELS)
BASELINE_K_IDX = 3  # value 13 (the max-K baseline)


def _open_mask(n_opts: int, n_k: int, max_step_dim: int = 2, max_num_levels: int = 6) -> np.ndarray:
    """Replica of the unrestricted open mask for one fusion step."""
    mask = np.zeros((max_step_dim, max_num_levels), dtype=bool)
    mask[0, : min(n_opts, max_num_levels)] = True
    mask[1, : min(n_k, max_num_levels)] = True
    return mask


class FusionBlockCurriculumScheduleTest(unittest.TestCase):
    H = 47
    ANCHOR = 80
    RAMP = 300
    MAXR = 6

    def _curr(self, ep):
        return fcur.fusion_block_curriculum(
            absolute_episode_idx=ep, anchor_episodes=self.ANCHOR,
            ramp_episodes=self.RAMP, horizon=self.H, max_radius=self.MAXR,
        )

    def test_starts_minimal_at_anchor(self):
        # During the anchor and at its boundary only one block may move, radius 1.
        for ep in (0, 40, self.ANCHOR):
            fully_open, num_mutable, radius = self._curr(ep)
            self.assertFalse(fully_open)
            self.assertEqual(num_mutable, 1)
            self.assertEqual(radius, 1)

    def test_monotonic_nondecreasing(self):
        prev_m, prev_r = 0, 0
        for ep in range(self.ANCHOR, self.ANCHOR + self.RAMP + 50):
            _fully_open, num_mutable, radius = self._curr(ep)
            self.assertGreaterEqual(num_mutable, prev_m)
            self.assertGreaterEqual(radius, prev_r)
            self.assertLessEqual(num_mutable, self.H)
            self.assertLessEqual(radius, self.MAXR)
            prev_m, prev_r = num_mutable, radius

    def test_fully_opens_after_ramp(self):
        # THE core guarantee: once the ramp completes the schedule reports fully
        # open with every block mutable and full radius — nothing stays masked.
        for ep in (self.ANCHOR + self.RAMP, self.ANCHOR + self.RAMP + 1, 10 * self.RAMP):
            fully_open, num_mutable, radius = self._curr(ep)
            self.assertTrue(fully_open)
            self.assertEqual(num_mutable, self.H)
            self.assertEqual(radius, self.MAXR)

    def test_ramp_actually_reaches_open(self):
        # The open phase is reachable at a finite episode (not asymptotic).
        opened_at = next(
            ep for ep in range(self.ANCHOR, self.ANCHOR + 5 * self.RAMP)
            if self._curr(ep)[0]
        )
        self.assertLessEqual(opened_at, self.ANCHOR + self.RAMP)


class FusionStepLevelMaskTest(unittest.TestCase):
    def _mask(self, *, n_opts, mutable, radius):
        return fcur.build_fusion_step_level_mask(
            fusion_num_options=n_opts, k_num_levels=N_K, k_level_values=K_LEVELS,
            mutable=mutable, radius=radius, baseline_k_index=BASELINE_K_IDX,
            max_step_dim=2, max_num_levels=6,
        )

    def test_nonmutable_block_pinned_to_baseline(self):
        # A pinned block exposes exactly the baseline action and nothing else.
        for n_opts in (1, 2, 3):
            mask = self._mask(n_opts=n_opts, mutable=False, radius=5)
            self.assertTrue(mask[0, 0])
            self.assertTrue(mask[1, BASELINE_K_IDX])
            self.assertEqual(int(mask.sum()), 2, f"n_opts={n_opts}: only baseline cells")

    def test_baseline_always_allowed(self):
        for n_opts in (1, 2, 3):
            for mutable in (True, False):
                for radius in range(0, 8):
                    mask = self._mask(n_opts=n_opts, mutable=mutable, radius=radius)
                    self.assertTrue(mask[0, 0], "option 0 (baseline) must stay allowed")
                    self.assertTrue(mask[1, BASELINE_K_IDX], "baseline K must stay allowed")

    def test_mutable_radius1_widens_locally(self):
        mask = self._mask(n_opts=3, mutable=True, radius=1)
        # options [0, 1] only (radius 1)
        self.assertListEqual([bool(x) for x in mask[0]], [True, True, False, False, False, False])
        # K = the 3 nearest-by-bit indices to baseline 13: {idx3=13, idx5=12, idx2=11}
        self.assertEqual({int(i) for i in np.where(mask[1])[0]}, {2, 3, 5})

    def test_mutable_full_radius_equals_open_mask(self):
        # THE core guarantee at the mask level: a mutable block at the fully-open
        # radius is byte-identical to the unrestricted open mask, so every option
        # and every K is reachable — no config is permanently hidden.
        for n_opts in (1, 2, 3):
            mask = self._mask(n_opts=n_opts, mutable=True, radius=6)
            np.testing.assert_array_equal(mask, _open_mask(n_opts, N_K))

    def test_invalid_baseline_k_raises(self):
        with self.assertRaises(ValueError):
            fcur.build_fusion_step_level_mask(
                fusion_num_options=2, k_num_levels=N_K, k_level_values=K_LEVELS,
                mutable=False, radius=1, baseline_k_index=99,
                max_step_dim=2, max_num_levels=6,
            )

    def test_reuses_cached_mask_template_without_sharing_returned_array(self):
        cache = fcur._cached_fusion_step_level_mask
        cache.cache_clear()

        first = self._mask(n_opts=3, mutable=True, radius=1)
        first[0, 0] = False
        second = self._mask(n_opts=3, mutable=True, radius=1)

        self.assertTrue(second[0, 0])
        self.assertEqual({int(i) for i in np.where(second[1])[0]}, {2, 3, 5})
        self.assertGreaterEqual(cache.cache_info().hits, 1)


class NearBaselineKTest(unittest.TestCase):
    def test_includes_baseline_and_size(self):
        for radius in range(0, 6):
            idxs = fcur.near_baseline_k_indices(
                k_level_values=K_LEVELS, baseline_idx=BASELINE_K_IDX, dim=N_K, radius=radius,
            )
            self.assertIn(BASELINE_K_IDX, idxs)
            self.assertEqual(len(idxs), min(N_K, 2 * radius + 1))
            self.assertEqual(idxs, sorted(set(idxs)))

    def test_large_radius_keeps_all(self):
        idxs = fcur.near_baseline_k_indices(
            k_level_values=K_LEVELS, baseline_idx=BASELINE_K_IDX, dim=N_K, radius=10,
        )
        self.assertEqual(idxs, list(range(N_K)))

    def test_orders_by_truncation_bit_distance(self):
        # radius 0 keeps only the baseline; radius 1 adds the single closest by bits.
        r0 = fcur.near_baseline_k_indices(
            k_level_values=K_LEVELS, baseline_idx=BASELINE_K_IDX, dim=N_K, radius=0)
        self.assertEqual(r0, [BASELINE_K_IDX])
        r1 = fcur.near_baseline_k_indices(
            k_level_values=K_LEVELS, baseline_idx=BASELINE_K_IDX, dim=N_K, radius=1)
        # value 13 nearest neighbours by |Δbits|: 12 (idx5) at distance 1.
        self.assertEqual(set(r1), {2, 3, 5})

    def test_reuses_cached_near_k_ordering_without_exposing_cached_tuple(self):
        cache = fcur._cached_near_baseline_k_indices
        cache.cache_clear()

        first = fcur.near_baseline_k_indices(
            k_level_values=K_LEVELS, baseline_idx=BASELINE_K_IDX, dim=N_K, radius=1)
        first.append(99)
        second = fcur.near_baseline_k_indices(
            k_level_values=K_LEVELS, baseline_idx=BASELINE_K_IDX, dim=N_K, radius=1)

        self.assertEqual(set(second), {2, 3, 5})
        self.assertNotIn(99, second)
        self.assertGreaterEqual(cache.cache_info().hits, 1)


class SelectMutableStepIndicesTest(unittest.TestCase):
    def test_single_mutable_step_uses_integer_draw_without_choice_array(self):
        class OneDrawRng:
            def __init__(self):
                self.integer_calls = 0

            def integers(self, high):
                self.integer_calls += 1
                self.high = high
                return 7

            def choice(self, *_args, **_kwargs):
                raise AssertionError("single mutable step should not allocate choice array")

        rng = OneDrawRng()
        sel = fcur.select_mutable_step_indices(rng=rng, horizon=47, num_mutable=1)

        self.assertEqual(sel, {7})
        self.assertEqual(rng.high, 47)
        self.assertEqual(rng.integer_calls, 1)

    def test_distinct_in_range_and_count(self):
        rng = np.random.default_rng(0)
        for num in (1, 5, 47):
            sel = fcur.select_mutable_step_indices(rng=rng, horizon=47, num_mutable=num)
            self.assertEqual(len(sel), num)
            self.assertTrue(all(0 <= i < 47 for i in sel))

    def test_reproducible_per_seed(self):
        a = fcur.select_mutable_step_indices(
            rng=np.random.default_rng(123), horizon=47, num_mutable=8)
        b = fcur.select_mutable_step_indices(
            rng=np.random.default_rng(123), horizon=47, num_mutable=8)
        self.assertEqual(a, b)

    def test_caps_at_horizon(self):
        sel = fcur.select_mutable_step_indices(
            rng=np.random.default_rng(1), horizon=47, num_mutable=999)
        self.assertEqual(len(sel), 47)


class FullSpaceReachabilityTest(unittest.TestCase):
    """End-to-end: over the curriculum's lifetime every (option, K) of every block
    is reachable, i.e. the safe-neighbor restriction never permanently masks any
    config (it is a warmup that dissolves to the open mask)."""

    def test_union_over_curriculum_covers_full_space(self):
        H, ANCHOR, RAMP, MAXR = 12, 10, 40, 6
        n_opts, n_k = 3, N_K
        # Accumulate, per "block", the union of allowed (option, K) cells seen as a
        # mutable block across the whole schedule (anchor → well past the ramp).
        union = np.zeros((2, 6), dtype=bool)
        for ep in range(0, ANCHOR + RAMP + 5):
            fully_open, _num_mutable, radius = fcur.fusion_block_curriculum(
                absolute_episode_idx=ep, anchor_episodes=ANCHOR,
                ramp_episodes=RAMP, horizon=H, max_radius=MAXR,
            )
            if fully_open:
                union |= _open_mask(n_opts, n_k)
            else:
                union |= fcur.build_fusion_step_level_mask(
                    fusion_num_options=n_opts, k_num_levels=n_k, k_level_values=K_LEVELS,
                    mutable=True, radius=radius, baseline_k_index=BASELINE_K_IDX,
                    max_step_dim=2, max_num_levels=6,
                )
        np.testing.assert_array_equal(union, _open_mask(n_opts, n_k))


class FusionProbeScheduleTest(unittest.TestCase):
    """Scheduled forced-fusion probes (ADR-011) — determinism + rotation.

    The probe decision must be a pure function of the absolute episode index so
    that episode-parallel workers reach identical decisions (1==N invariant),
    and the rotation must visit every fusable block type.
    """

    def test_anchor_episodes_never_probe(self):
        for ep in range(0, 60):
            # callers only consult the helper post-anchor, but rel<0 must be None
            self.assertIsNone(
                fcur.fusion_probe_target_block(ep - 60, anchor_episodes=60, interval=200)
            )

    def test_rotation_block2_block5(self):
        # ADR-012: block4 dropped from the rotation (a 12-layer block4 fusion
        # probe is a guaranteed accuracy fail and only taught anti-fusion).
        got = [
            fcur.fusion_probe_target_block(60 + i * 200, anchor_episodes=60, interval=200)
            for i in range(6)
        ]
        self.assertEqual(got, [2, 5, 2, 5, 2, 5])

    def test_non_multiple_episodes_are_none(self):
        for off in (1, 7, 199, 201):
            self.assertIsNone(
                fcur.fusion_probe_target_block(60 + off, anchor_episodes=60, interval=200)
            )

    def test_interval_zero_disables(self):
        self.assertIsNone(
            fcur.fusion_probe_target_block(60, anchor_episodes=60, interval=0)
        )

    def test_pure_function_of_episode_index(self):
        # determinism across "workers": same args -> same answer, every call.
        for ep in range(0, 2000, 37):
            a = fcur.fusion_probe_target_block(ep, anchor_episodes=60, interval=200)
            b = fcur.fusion_probe_target_block(ep, anchor_episodes=60, interval=200)
            self.assertEqual(a, b)

    def test_probe_frequency_is_sparse(self):
        # 60k episodes / interval 200 -> 300 probes (0.5% overhead), 150 per type.
        probes = [
            fcur.fusion_probe_target_block(ep, anchor_episodes=60, interval=200)
            for ep in range(60, 60060)
        ]
        hits = [x for x in probes if x is not None]
        self.assertEqual(len(hits), 300)
        self.assertEqual(hits.count(2), 150)
        self.assertEqual(hits.count(5), 150)
        self.assertEqual(hits.count(4), 0)


if __name__ == "__main__":
    unittest.main()
