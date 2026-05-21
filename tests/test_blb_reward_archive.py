import importlib.util
import sys
from pathlib import Path
import unittest


def load_reward_module():
    path = Path(__file__).resolve().parents[1] / "blb_stage2_rl" / "reward.py"
    spec = importlib.util.spec_from_file_location("reward_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ParetoCostArchiveTests(unittest.TestCase):
    def _p3(self, *, fusion_gain=0.0, k_gain=0.0, bits_gain=0.0):
        reward = load_reward_module()

        return reward.RewardBreakdown(
            reward=40.0,
            priority=3,
            invalid=False,
            metric_ok=True,
            stab_ok=True,
            fusion_gain=float(fusion_gain),
            k_drop=float(k_gain),
            bits_drop=float(bits_gain),
        )

    def test_excludes_p1_and_p2_candidates(self):
        reward = load_reward_module()

        archive = reward.ParetoCostArchive()

        p1 = archive.add(
            "p1",
            reward.RewardBreakdown(
                reward=0.0,
                priority=1,
                invalid=True,
                fusion_gain=10.0,
                k_drop=10.0,
                bits_drop=10.0,
            ),
        )
        p2 = archive.add(
            "p2",
            reward.RewardBreakdown(
                reward=20.0,
                priority=2,
                invalid=False,
                metric_ok=True,
                stab_ok=False,
                fusion_gain=10.0,
                k_drop=10.0,
                bits_drop=10.0,
            ),
        )

        self.assertEqual(p1.kind, "excluded")
        self.assertEqual(p2.kind, "excluded")
        self.assertEqual(archive.frontier, ())

    def test_replaces_dominated_frontier_members(self):
        reward = load_reward_module()

        archive = reward.ParetoCostArchive()
        first = archive.add("a", self._p3(fusion_gain=1.0, k_gain=1.0, bits_gain=1.0))
        second = archive.add("b", self._p3(fusion_gain=2.0, k_gain=1.0, bits_gain=1.0))

        self.assertEqual(first.kind, "frontier_expansion")
        self.assertGreater(first.shaping, 0.0)
        self.assertEqual(second.kind, "frontier_expansion")
        self.assertGreater(second.shaping, first.shaping)
        self.assertEqual([entry.action_hash for entry in archive.frontier], ["b"])

    def test_keeps_mutually_non_dominated_candidates(self):
        reward = load_reward_module()

        archive = reward.ParetoCostArchive()
        archive.add("fusion", self._p3(fusion_gain=3.0, k_gain=1.0, bits_gain=1.0))
        event = archive.add("k", self._p3(fusion_gain=1.0, k_gain=3.0, bits_gain=1.0))

        self.assertEqual(event.kind, "frontier_member")
        self.assertGreater(event.shaping, 0.0)
        self.assertLess(event.shaping, 0.10)
        self.assertEqual({entry.action_hash for entry in archive.frontier}, {"fusion", "k"})

    def test_dominated_and_duplicate_events_are_bounded(self):
        reward = load_reward_module()

        archive = reward.ParetoCostArchive(max_abs_shaping=0.35)
        archive.add("strong", self._p3(fusion_gain=2.0, k_gain=2.0, bits_gain=2.0))
        dominated = archive.add("weak", self._p3(fusion_gain=1.0, k_gain=1.0, bits_gain=1.0))
        duplicate = archive.add("strong", self._p3(fusion_gain=100.0, k_gain=100.0, bits_gain=100.0))

        self.assertEqual(dominated.kind, "dominated")
        self.assertAlmostEqual(dominated.shaping, -0.10, places=6)
        self.assertEqual(duplicate.kind, "duplicate")
        self.assertAlmostEqual(duplicate.shaping, -0.025, places=6)
        self.assertEqual(len(archive.frontier), 1)

    def test_default_pareto_event_shaping_is_stronger_but_tier_safe(self):
        reward = load_reward_module()

        archive = reward.ParetoCostArchive()
        first = archive.add("a", self._p3(fusion_gain=1.0, k_gain=1.0, bits_gain=1.0))
        member = archive.add("b", self._p3(fusion_gain=2.0, k_gain=0.0, bits_gain=1.0))
        dominated = archive.add("c", self._p3(fusion_gain=0.0, k_gain=0.0, bits_gain=0.0))
        duplicate = archive.add("a", self._p3(fusion_gain=9.0, k_gain=9.0, bits_gain=9.0))

        self.assertEqual(first.kind, "frontier_expansion")
        self.assertEqual(member.kind, "frontier_member")
        self.assertEqual(dominated.kind, "dominated")
        self.assertEqual(duplicate.kind, "duplicate")
        self.assertAlmostEqual(first.shaping, 0.20, places=6)
        self.assertAlmostEqual(member.shaping, 0.05, places=6)
        self.assertAlmostEqual(dominated.shaping, -0.10, places=6)
        self.assertAlmostEqual(duplicate.shaping, -0.025, places=6)
        for event in (first, member, dominated, duplicate):
            self.assertLessEqual(abs(event.shaping), 0.35)

    def test_ignores_typical_normalizers_for_ranking(self):
        reward = load_reward_module()

        archive = reward.ParetoCostArchive(
            baseline=reward.BaselineCostStats(
                typical_bits_drop=1_000_000.0,
                typical_fusion_count=0.001,
                typical_k_drop=0.001,
            )
        )
        archive.add("base", self._p3(fusion_gain=1.0, k_gain=1.0, bits_gain=1.0))
        event = archive.add("raw-dominates", self._p3(fusion_gain=2.0, k_gain=1.0, bits_gain=1.0))

        self.assertEqual(event.kind, "frontier_expansion")
        self.assertEqual([entry.action_hash for entry in archive.frontier], ["raw-dominates"])

    def test_compute_reward_uses_pareto_event_for_p3_cost(self):
        reward = load_reward_module()

        class Signals:
            any_invalid = False
            total_bits_sum = 90
            total_fusion_count = 2

        archive = reward.ParetoCostArchive(max_abs_shaping=0.25)
        baseline = reward.BaselineCostStats(
            total_bits_sum=100,
            total_fusion_count=0,
            avg_k=13.0,
            metric1_mean=0.90,
            metric2_mean=0.90,
            metric1_std=0.0,
            metric2_std=0.0,
            loss_std=0.0,
            typical_bits_drop=1_000_000.0,
            typical_fusion_count=0.001,
            typical_k_drop=0.001,
        )
        weights = reward.calibrate_weights_from_baseline(baseline)
        breakdown = reward.compute_reward(
            reward.EpisodeMetrics(
                loss_mean=0.30,
                loss_std=0.0,
                metric1_mean=0.90,
                metric2_mean=0.90,
                metric1_std=0.0,
                metric2_std=0.0,
            ),
            Signals(),
            action_avg_k=12.0,
            baseline=baseline,
            weights=weights,
            pareto_archive=archive,
            action_hash="p3-a",
        )

        self.assertEqual(breakdown.priority, 3)
        self.assertEqual(breakdown.pareto_event_kind, "frontier_expansion")
        self.assertEqual(breakdown.cost_score, 0.20)
        self.assertEqual([entry.action_hash for entry in archive.frontier], ["p3-a"])

    def test_compute_reward_excludes_p2_from_cost_and_archive(self):
        reward = load_reward_module()

        class Signals:
            any_invalid = False
            total_bits_sum = 90
            total_fusion_count = 2

        archive = reward.ParetoCostArchive()
        baseline = reward.BaselineCostStats(
            total_bits_sum=100,
            total_fusion_count=0,
            avg_k=13.0,
            metric1_mean=0.90,
            metric2_mean=0.90,
            metric1_std=0.0,
            metric2_std=0.0,
            loss_std=0.0,
        )
        weights = reward.calibrate_weights_from_baseline(baseline)
        breakdown = reward.compute_reward(
            reward.EpisodeMetrics(
                loss_mean=0.30,
                loss_std=1.0,
                metric1_mean=0.90,
                metric2_mean=0.90,
                metric1_std=0.0,
                metric2_std=0.0,
            ),
            Signals(),
            action_avg_k=12.0,
            baseline=baseline,
            weights=weights,
            pareto_archive=archive,
            action_hash="p2-a",
        )

        self.assertEqual(breakdown.priority, 2)
        self.assertEqual(breakdown.cost_score, 0.0)
        self.assertEqual(breakdown.pareto_event_kind, "excluded")
        self.assertEqual(archive.frontier, ())


if __name__ == "__main__":
    unittest.main()
