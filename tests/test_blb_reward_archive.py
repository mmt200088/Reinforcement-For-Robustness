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

    def test_compute_reward_uses_adaptive_scalar_for_p3_cost_while_recording_pareto(self):
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
        self.assertGreater(breakdown.cost_score, 0.20)
        self.assertAlmostEqual(breakdown.r_fusion, 0.70, places=6)
        self.assertGreaterEqual(breakdown.r_k, 4.0)
        self.assertGreater(breakdown.r_bits, 0.0)
        self.assertLess(breakdown.r_bits, reward.DEFAULT_COST_FUSION_STEP_BONUS)
        self.assertAlmostEqual(
            breakdown.cost_score,
            reward.DEFAULT_P3_COST_BUDGET,
            places=6,
        )
        self.assertEqual([entry.action_hash for entry in archive.frontier], ["p3-a"])

    def test_adaptive_scalar_cost_has_fusion_and_truncation_step_boosts(self):
        reward = load_reward_module()

        class Signals:
            any_invalid = False
            total_bits_sum = 100
            total_fusion_count = 1

        baseline = reward.BaselineCostStats(
            total_bits_sum=120,
            total_fusion_count=0,
            avg_k=13.0,
            metric1_mean=0.90,
            metric2_mean=0.90,
            metric1_std=0.0,
            metric2_std=0.0,
            loss_std=0.0,
            typical_bits_drop=120.0,
            typical_fusion_count=12.0,
            typical_k_drop=5.0,
        )
        weights = reward.calibrate_weights_from_baseline(baseline)
        # ADR-013: this asserts the legacy linear P3 metric-margin term, which
        # the log-barrier supersedes by default — test it with the barrier off.
        weights.acc_barrier_enabled = False
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
            action_avg_k=13.0 - (1.0 / 12.0),
            baseline=baseline,
            weights=weights,
            action_hash="scalar-a",
        )

        self.assertEqual(breakdown.priority, 3)
        self.assertAlmostEqual(breakdown.r_fusion, 0.35, places=6)
        self.assertAlmostEqual(breakdown.r_k, 0.35, places=6)
        self.assertAlmostEqual(breakdown.cost_truncation_step_gain, 1.0, places=6)
        self.assertGreater(breakdown.r_bits, 0.0)
        self.assertAlmostEqual(
            breakdown.cost_score,
            breakdown.r_fusion + breakdown.r_k + breakdown.r_bits,
            places=6,
        )
        self.assertGreater(breakdown.p3_metric_margin_reward, 0.0)

    def test_p3_cost_rank_remains_unbounded_after_ppo_cost_score_clips(self):
        reward = load_reward_module()

        baseline = reward.BaselineCostStats(
            total_bits_sum=2000,
            total_fusion_count=0,
            avg_k=13.0,
            metric1_mean=0.90,
            metric2_mean=0.90,
            metric1_std=0.0,
            metric2_std=0.0,
            loss_std=0.0,
            typical_bits_drop=1000.0,
        )
        weights = reward.calibrate_weights_from_baseline(baseline)
        metrics = reward.EpisodeMetrics(
            loss_mean=0.30,
            loss_std=0.0,
            metric1_mean=0.90,
            metric2_mean=0.90,
            metric1_std=0.0,
            metric2_std=0.0,
        )

        def run(fusion_count, action_avg_k, total_bits_sum):
            signals = type(
                "Signals", (),
                {
                    "any_invalid": False,
                    "total_bits_sum": float(total_bits_sum),
                    "total_fusion_count": float(fusion_count),
                },
            )()
            return reward.compute_reward(
                metrics,
                signals,
                action_avg_k=float(action_avg_k),
                baseline=baseline,
                weights=weights,
            )

        capped_low = run(8, 12.50, 1600)
        capped_high = run(14, 11.80, 1500)

        self.assertAlmostEqual(capped_low.cost_score, reward.DEFAULT_P3_COST_BUDGET)
        self.assertAlmostEqual(capped_high.cost_score, reward.DEFAULT_P3_COST_BUDGET)
        self.assertAlmostEqual(capped_low.reward, capped_high.reward)
        self.assertGreater(capped_high.cost_rank_score, capped_low.cost_rank_score)
        self.assertGreater(capped_high.cost_rank_fusion, capped_low.cost_rank_fusion)
        self.assertGreater(capped_high.cost_rank_truncation, capped_low.cost_rank_truncation)

    def test_p1_p2_do_not_receive_p3_cost_rank(self):
        reward = load_reward_module()

        baseline = reward.BaselineCostStats(
            total_bits_sum=2000,
            total_fusion_count=0,
            avg_k=13.0,
            metric1_mean=0.90,
            metric2_mean=0.90,
            metric1_std=0.0,
            metric2_std=0.0,
            loss_std=0.0,
            typical_bits_drop=1000.0,
        )
        weights = reward.calibrate_weights_from_baseline(baseline)
        huge_cost_signals = type(
            "Signals", (),
            {
                "any_invalid": False,
                "total_bits_sum": 1000.0,
                "total_fusion_count": 99.0,
            },
        )()

        p1 = reward.compute_reward(
            reward.EpisodeMetrics(
                loss_mean=0.30,
                loss_std=0.0,
                metric1_mean=0.10,
                metric2_mean=0.10,
                metric1_std=0.0,
                metric2_std=0.0,
            ),
            huge_cost_signals,
            action_avg_k=1.0,
            baseline=baseline,
            weights=weights,
        )
        p2 = reward.compute_reward(
            reward.EpisodeMetrics(
                loss_mean=0.30,
                loss_std=1.0,
                metric1_mean=0.90,
                metric2_mean=0.90,
                metric1_std=1.0,
                metric2_std=1.0,
            ),
            huge_cost_signals,
            action_avg_k=1.0,
            baseline=baseline,
            weights=weights,
        )

        self.assertEqual(p1.priority, 1)
        self.assertEqual(p1.cost_score, 0.0)
        self.assertEqual(p1.cost_rank_score, 0.0)
        self.assertLessEqual(p1.reward, 5.0)

        self.assertEqual(p2.priority, 2)
        self.assertEqual(p2.cost_score, 0.0)
        self.assertEqual(p2.cost_rank_score, 0.0)
        self.assertLess(p2.reward, 35.0)

    def test_adaptive_scalar_cost_step_boundaries_and_bits_tiebreaker(self):
        reward = load_reward_module()

        class Signals:
            any_invalid = False
            total_bits_sum = 100
            total_fusion_count = 0

        baseline = reward.BaselineCostStats(
            total_bits_sum=200,
            total_fusion_count=0,
            avg_k=13.0,
            metric1_mean=0.90,
            metric2_mean=0.90,
            metric1_std=0.0,
            metric2_std=0.0,
            loss_std=0.0,
            typical_bits_drop=100.0,
        )
        weights = reward.calibrate_weights_from_baseline(baseline)
        metrics = reward.EpisodeMetrics(
            loss_mean=0.30,
            loss_std=0.0,
            metric1_mean=0.90,
            metric2_mean=0.90,
            metric1_std=0.0,
            metric2_std=0.0,
        )

        def run(fusion_gain, k_gain):
            dynamic_signals = type(
                "DynamicSignals",
                (),
                {
                    "any_invalid": False,
                    "total_bits_sum": 100,
                    "total_fusion_count": float(fusion_gain),
                },
            )()
            return reward.compute_reward(
                metrics,
                dynamic_signals,
                action_avg_k=float(baseline.avg_k) - float(k_gain),
                baseline=baseline,
                weights=weights,
            )

        self.assertAlmostEqual(run(0.99, 0.0).r_fusion, 0.0, places=6)
        self.assertAlmostEqual(run(1.0, 0.0).r_fusion, 0.35, places=6)
        self.assertAlmostEqual(run(1.99, 0.0).r_fusion, 0.35, places=6)
        self.assertAlmostEqual(run(2.0, 0.0).r_fusion, 0.70, places=6)

        one_k = 1.0 / 12.0
        self.assertAlmostEqual(run(0.0, one_k * 0.99).r_k, 0.0, places=6)
        self.assertAlmostEqual(run(0.0, one_k).r_k, 0.35, places=6)
        self.assertAlmostEqual(run(0.0, one_k * 1.99).r_k, 0.35, places=6)
        self.assertAlmostEqual(run(0.0, one_k * 2.0).r_k, 0.70, places=6)

        bits_only = run(0.0, 0.0)
        self.assertLessEqual(
            abs(bits_only.r_bits),
            reward.DEFAULT_COST_BITS_TIEBREAKER_CLIP,
        )
        self.assertLess(
            abs(bits_only.r_bits),
            reward.DEFAULT_COST_FUSION_STEP_BONUS,
        )

    def test_p3_high_accuracy_margin_does_not_hide_cost_ordering(self):
        reward = load_reward_module()

        baseline = reward.BaselineCostStats(
            total_bits_sum=100,
            total_fusion_count=0,
            avg_k=13.0,
            metric1_mean=0.90,
            metric2_mean=0.90,
            metric1_std=0.0,
            metric2_std=0.0,
            loss_std=0.0,
            typical_bits_drop=100.0,
        )
        weights = reward.calibrate_weights_from_baseline(baseline)
        # ADR-013: legacy linear P3 metric-margin path (barrier off); the
        # cost-ordering invariant this test checks is unchanged by the barrier.
        weights.acc_barrier_enabled = False
        metrics = reward.EpisodeMetrics(
            loss_mean=0.30,
            loss_std=0.0,
            metric1_mean=1.00,
            metric2_mean=1.00,
            metric1_std=0.0,
            metric2_std=0.0,
        )

        class NoFusion:
            any_invalid = False
            total_bits_sum = 100
            total_fusion_count = 0

        class OneFusion(NoFusion):
            total_fusion_count = 1

        no_fusion = reward.compute_reward(
            metrics,
            NoFusion(),
            action_avg_k=13.0,
            baseline=baseline,
            weights=weights,
        )
        one_fusion = reward.compute_reward(
            metrics,
            OneFusion(),
            action_avg_k=13.0,
            baseline=baseline,
            weights=weights,
        )

        self.assertAlmostEqual(
            no_fusion.p3_metric_margin_reward,
            reward.DEFAULT_P3_METRIC_MARGIN_BUDGET,
            places=6,
        )
        self.assertAlmostEqual(
            one_fusion.reward - no_fusion.reward,
            reward.DEFAULT_COST_FUSION_STEP_BONUS,
            places=6,
        )

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
        self.assertEqual(breakdown.r_fusion, 0.0)
        self.assertEqual(breakdown.r_k, 0.0)
        self.assertEqual(breakdown.r_bits, 0.0)
        self.assertEqual(archive.frontier, ())

    def test_compute_reward_excludes_p1_from_adaptive_scalar_cost(self):
        reward = load_reward_module()

        class Signals:
            any_invalid = False
            total_bits_sum = 1
            total_fusion_count = 99

        archive = reward.ParetoCostArchive()
        baseline = reward.BaselineCostStats(
            total_bits_sum=1000,
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
                loss_std=0.0,
                metric1_mean=0.10,
                metric2_mean=0.10,
                metric1_std=0.0,
                metric2_std=0.0,
            ),
            Signals(),
            action_avg_k=1.0,
            baseline=baseline,
            weights=weights,
            pareto_archive=archive,
            action_hash="p1-big-cost",
        )

        self.assertEqual(breakdown.priority, 1)
        self.assertFalse(breakdown.metric_ok)
        self.assertEqual(breakdown.cost_score, 0.0)
        self.assertEqual(breakdown.r_fusion, 0.0)
        self.assertEqual(breakdown.r_k, 0.0)
        self.assertEqual(breakdown.r_bits, 0.0)
        self.assertEqual(breakdown.pareto_event_kind, "excluded")
        self.assertEqual(archive.frontier, ())

    def test_compute_reward_excludes_optimizer_invalid_from_adaptive_scalar_cost(self):
        reward = load_reward_module()

        class Signals:
            any_invalid = True
            total_bits_sum = 1
            total_fusion_count = 99

        archive = reward.ParetoCostArchive()
        baseline = reward.BaselineCostStats(
            total_bits_sum=1000,
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
                loss_std=0.0,
                metric1_mean=0.90,
                metric2_mean=0.90,
                metric1_std=0.0,
                metric2_std=0.0,
            ),
            Signals(),
            action_avg_k=1.0,
            baseline=baseline,
            weights=weights,
            pareto_archive=archive,
            action_hash="p1-invalid-big-cost",
        )

        self.assertEqual(breakdown.priority, 1)
        self.assertTrue(breakdown.invalid)
        self.assertEqual(breakdown.cost_score, 0.0)
        self.assertEqual(breakdown.p3_metric_margin_reward, 0.0)
        self.assertEqual(breakdown.r_fusion, 0.0)
        self.assertEqual(breakdown.r_k, 0.0)
        self.assertEqual(breakdown.r_bits, 0.0)
        self.assertEqual(breakdown.pareto_event_kind, "excluded")
        self.assertEqual(archive.frontier, ())


if __name__ == "__main__":
    unittest.main()
