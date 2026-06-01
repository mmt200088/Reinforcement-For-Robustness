"""Torch-free unit tests for the shared GRPO advantage helpers (grpo_common).

These exercise the group-relative normalization math directly (numpy only), so
they run on a torch-free dev box as well as in CI.
"""
import math
from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import grpo_common as G


class GroupNormalizeTest(unittest.TestCase):
    def test_known_values(self):
        adv = G.grpo_group_normalize([1.0, 2.0, 3.0], eps=0.0)
        # mean=2, population std=sqrt(2/3); adv = (R-2)/std
        std = math.sqrt(2.0 / 3.0)
        self.assertAlmostEqual(adv[0], -1.0 / std, places=4)
        self.assertAlmostEqual(adv[1], 0.0, places=5)
        self.assertAlmostEqual(adv[2], 1.0 / std, places=4)

    def test_zero_mean_and_order_preserved(self):
        adv = G.grpo_group_normalize([0.5, 10.0, -3.0, 4.0])
        self.assertAlmostEqual(float(adv.mean()), 0.0, places=4)
        # higher return => higher advantage
        order = sorted(range(4), key=lambda i: [0.5, 10.0, -3.0, 4.0][i])
        adv_sorted = [adv[i] for i in order]
        self.assertEqual(adv_sorted, sorted(adv_sorted))

    def test_degenerate_groups_give_zero(self):
        self.assertEqual(list(G.grpo_group_normalize([])), [])
        self.assertEqual(list(G.grpo_group_normalize([7.0])), [0.0])
        # zero spread -> all zero (no within-group signal)
        self.assertEqual(list(G.grpo_group_normalize([5.0, 5.0, 5.0])), [0.0, 0.0, 0.0])

    def test_nonfinite_collapses_to_zero(self):
        adv = G.grpo_group_normalize([1.0, float("nan"), 3.0])
        self.assertTrue(all(math.isfinite(x) for x in adv))


class SegmentAndBroadcastTest(unittest.TestCase):
    def test_segment_episode_returns(self):
        # two episodes: [1,2 | 3] then [10 | 20]  (done on 2nd and 4th? define explicitly)
        rewards = [1.0, 2.0, 3.0, 10.0, 20.0]
        dones = [False, False, True, False, True]
        ep_id, ep_ret = G.segment_episode_returns(rewards, dones)
        self.assertEqual(list(ep_id), [0, 0, 0, 1, 1])
        self.assertEqual(list(ep_ret), [6.0, 30.0])

    def test_trailing_episode_without_done(self):
        rewards = [1.0, 2.0, 5.0]
        dones = [True, False, False]  # second episode never closes
        ep_id, ep_ret = G.segment_episode_returns(rewards, dones)
        self.assertEqual(list(ep_id), [0, 1, 1])
        self.assertEqual(list(ep_ret), [1.0, 7.0])

    def test_per_step_advantages_broadcast(self):
        # ep0 return=6, ep1 return=30 -> group-normalized, broadcast to steps
        rewards = [1.0, 2.0, 3.0, 10.0, 20.0]
        dones = [False, False, True, False, True]
        adv = G.grpo_per_step_advantages(rewards, dones)
        # all steps of an episode share the same advantage
        self.assertAlmostEqual(adv[0], adv[1], places=6)
        self.assertAlmostEqual(adv[1], adv[2], places=6)
        self.assertAlmostEqual(adv[3], adv[4], places=6)
        # lower-return episode gets the lower advantage
        self.assertLess(adv[0], adv[3])

    def test_outlier_clip(self):
        rewards = [0.0, 1000.0]
        dones = [True, True]
        adv = G.grpo_per_step_advantages(rewards, dones, outlier_clip=1.0)
        self.assertTrue(all(abs(x) <= 1.0 + 1e-6 for x in adv))


if __name__ == "__main__":
    unittest.main()
