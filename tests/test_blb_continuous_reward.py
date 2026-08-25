"""Bounded Stage-2 reward invariants."""
import types
import unittest

from rfr.preparation.fusion import cost as FC
from blb_stage2_rl import reward as R

THR = 0.858
BASELINE_M = 0.871


def _weights(**kw):
    base = dict(baseline_metric1=BASELINE_M, baseline_metric2=BASELINE_M,
                stab_tolerance=1.2)
    base.update(kw)
    return R.RewardWeights(**base)


def _baseline():
    return R.BaselineCostStats(
        total_bits_sum=11285, total_fusion_count=0, avg_k=13.0,
        loss_mean=0.37, loss_std=0.01, metric1_mean=BASELINE_M, metric2_mean=BASELINE_M,
        metric1_std=0.002, metric2_std=0.002, typical_bits_drop=1000,
        typical_fusion_count=24, typical_k_drop=5,
    )


def _reward(m1, *, std=0.002, fusion=0, cost=0.0, invalid=False, k=13.0, weights=None):
    w = weights or _weights()
    base = _baseline()
    met = R.EpisodeMetrics(loss_mean=0.37, loss_std=std, metric1_mean=m1, metric2_mean=m1,
                           metric1_std=std, metric2_std=std)

    class _OB:
        any_invalid = invalid
        total_bits_sum = 11285 - 30 * fusion
        total_fusion_count = fusion

    return R.compute_reward(
        met, _OB(), action_avg_k=k, baseline=base, weights=w,
        acc_threshold=THR, acc_threshold_m2=THR,
        external_cost_score=cost, external_cost_rank=float(fusion),
    )


class ContinuousDefaultTest(unittest.TestCase):
    def test_default_is_stage1_aligned_but_continuous_remains_available(self):
        self.assertEqual(R.RewardWeights().reward_design, "stage1_aligned")
        self.assertEqual(R.RewardWeights(reward_design="continuous").reward_design, "continuous")


class BoundedAndContinuousTest(unittest.TestCase):
    def test_all_bounded_in_clip_range(self):
        rs = [
            _reward(0.871, fusion=8, cost=2.0).reward,
            _reward(THR + 0.001, fusion=8, cost=2.0).reward,
            _reward(THR - 0.001, fusion=8, cost=2.0).reward,
            _reward(0.70, fusion=20, cost=4.5).reward,
            _reward(0.871, std=0.05, fusion=8, cost=2.0).reward,
            _reward(0.871, invalid=True).reward,
        ]
        for r in rs:
            self.assertLessEqual(r, 5.0 + 1e-9)
            self.assertGreaterEqual(r, -5.0 - 1e-9)

    def test_continuous_across_feasibility_boundary(self):


        feas = _reward(THR + 0.0005, fusion=6, cost=1.5).reward
        viol = _reward(THR - 0.0005, fusion=6, cost=1.5).reward
        self.assertLess(abs(feas - viol), 8.0)

    def test_tiered_still_jumps(self):

        w = _weights(reward_design="tiered")
        feas = _reward(THR + 0.0005, fusion=6, cost=1.5, weights=w).reward
        viol = _reward(THR - 0.0005, fusion=6, cost=1.5, weights=w).reward
        self.assertGreater(abs(feas - viol), 20.0)


class HardPriorityItem7Test(unittest.TestCase):
    def test_p1_with_huge_cost_below_p3(self):
        p3 = _reward(0.871, fusion=4, cost=1.0).reward
        p1_bigcost = _reward(0.70, fusion=24, cost=4.5).reward
        self.assertLess(p1_bigcost, p3)

    def test_p2_gets_no_cost_below_p3(self):
        p3 = _reward(0.871, std=0.002, fusion=8, cost=2.0).reward
        p2 = _reward(0.871, std=0.05, fusion=8, cost=4.5).reward
        self.assertLess(p2, p3)


class StabilityBrakeTest(unittest.TestCase):
    def test_high_std_is_p2_not_p3(self):
        b = _reward(0.871, std=0.05, fusion=8, cost=2.0)
        self.assertEqual(b.priority, 2)
        self.assertFalse(b.stab_ok)
        self.assertTrue(b.metric_ok)

    def test_low_std_feasible_is_p3(self):
        b = _reward(0.871, std=0.002, fusion=8, cost=2.0)
        self.assertEqual(b.priority, 3)
        self.assertTrue(b.stab_ok and b.metric_ok)

    def test_strict_tolerance_tightens_gate(self):


        big_std_baseline = R.BaselineCostStats(
            total_bits_sum=11285, total_fusion_count=0, avg_k=13.0,
            loss_mean=0.37, loss_std=0.01, metric1_mean=BASELINE_M, metric2_mean=BASELINE_M,
            metric1_std=0.01, metric2_std=0.01, typical_bits_drop=1000,
            typical_fusion_count=24, typical_k_drop=5,
        )

        def _stab_at(tol, obs):
            met = R.EpisodeMetrics(loss_mean=0.37, loss_std=obs, metric1_mean=0.871,
                                   metric2_mean=0.871, metric1_std=obs, metric2_std=obs)

            class _OB:
                any_invalid = False
                total_bits_sum = 11285
                total_fusion_count = 8

            return R.compute_reward(
                met, _OB(), action_avg_k=13.0, baseline=big_std_baseline,
                weights=_weights(stab_tolerance=tol),
                acc_threshold=THR, acc_threshold_m2=THR,
            )


        self.assertTrue(_stab_at(5.0, 0.03).stab_ok)
        self.assertFalse(_stab_at(1.2, 0.03).stab_ok)

        self.assertFalse(_stab_at(5.0, 0.06).stab_ok)


class PriorityLabelTest(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(_reward(0.871, std=0.002).priority, 3)
        self.assertEqual(_reward(0.871, std=0.05).priority, 2)
        self.assertEqual(_reward(0.70, std=0.002).priority, 1)
        self.assertEqual(_reward(0.871, invalid=True).priority, 1)


class DeterminismTest(unittest.TestCase):
    def test_same_inputs_same_reward(self):
        a = _reward(0.865, std=0.003, fusion=5, cost=1.7).reward
        b = _reward(0.865, std=0.003, fusion=5, cost=1.7).reward
        self.assertEqual(a, b)


class RobustConstrainedResourceIsolationTest(unittest.TestCase):
    @staticmethod
    def _assessment(*, precision, stability):
        return types.SimpleNamespace(
            loss_precision_probability=precision,
            metric1_precision_probability=precision,
            metric2_precision_probability=precision,
            loss_stability_probability=stability,
            metric1_stability_probability=stability,
            metric2_stability_probability=stability,
        )

    def test_invalid_p1_and_p2_ignore_resource_score(self):
        cases = (
            (None, True),
            (self._assessment(precision=0.49, stability=0.99), False),
            (self._assessment(precision=0.99, stability=0.49), False),
        )
        for assessment, invalid in cases:
            with self.subTest(assessment=assessment, invalid=invalid):
                low = R.robust_constrained_reward(assessment, invalid, 0.0)
                high = R.robust_constrained_reward(assessment, invalid, 1.0)
                self.assertEqual(low, high)

    def test_p3_adds_packed_resource_score_exactly_once(self):
        assessment = self._assessment(precision=0.9, stability=0.8)
        low = R.robust_constrained_reward(assessment, False, 0.0)
        high = R.robust_constrained_reward(assessment, False, 0.75)
        self.assertEqual(low[1], 3)
        self.assertEqual(high[1], 3)
        self.assertAlmostEqual(high[0] - low[0], 0.75)

    def test_reward_breakdown_preserves_dual_resource_diagnostics(self):
        objective = {
            "compute_saving": 0.25,
            "communication_saving": 0.5,
            "robust_floor": 0.25,
            "secondary_progress": 0.375,
            "ppo_resource_score": 0.250012498750125,
            "compute_shapley_credit": 0.1250062493750625,
            "communication_shapley_credit": 0.1250062493750625,
            "layer_resource_rewards": [0.250012498750125],
            "slot_resource_rewards": [[0.1250062493750625, 0.1250062493750625]],
        }
        breakdown = R.compute_reward(
            R.EpisodeMetrics(
                loss_mean=0.37,
                loss_std=0.002,
                metric1_mean=BASELINE_M,
                metric2_mean=BASELINE_M,
                metric1_std=0.002,
                metric2_std=0.002,
            ),
            types.SimpleNamespace(any_invalid=False),
            action_avg_k=12.0,
            baseline=_baseline(),
            weights=_weights(reward_design="robust_constrained"),
            acc_threshold=THR,
            acc_threshold_m2=THR,
            external_cost_score=objective["ppo_resource_score"],
            external_cost_rank=objective["ppo_resource_score"],
            external_resource_objective=objective,
            constraint_assessment=self._assessment(precision=0.9, stability=0.8),
        )
        for field_name in (
            "compute_saving",
            "communication_saving",
            "robust_floor",
            "secondary_progress",
            "ppo_resource_score",
            "compute_shapley_credit",
            "communication_shapley_credit",
        ):
            self.assertEqual(getattr(breakdown, field_name), objective[field_name])
        self.assertEqual(
            breakdown.layer_resource_rewards,
            objective["layer_resource_rewards"],
        )
        self.assertEqual(
            breakdown.slot_resource_rewards,
            objective["slot_resource_rewards"],
        )


class LossMeanGateTest(unittest.TestCase):
    """2026-06-15 (determinism): loss_mean is a LOWER-better diagnostic in the reward
    (``loss_ok``, computed against the CLEAN deterministic baseline so it is
    byte-identical across GPU counts) but it does NOT feed metric_ok / priority /
    the reward scalar — the noisy loss reference is not cross-GPU-deterministic
    (unlike DISCRETE accuracy), so a continuous loss term in the per-episode reward
    breaks 1==N. The HARD loss constraint is enforced at strict feasibility SELECTION
    instead (sequential_runner). The tiered rollback never computes loss_ok."""

    def _r(self, design, loss_mean, std=0.002):
        w = _weights(reward_design=design)
        met = R.EpisodeMetrics(loss_mean=loss_mean, loss_std=std,
                               metric1_mean=BASELINE_M, metric2_mean=BASELINE_M,
                               metric1_std=std, metric2_std=std)

        class _OB:
            any_invalid = False
            total_bits_sum = 11285
            total_fusion_count = 8

        return R.compute_reward(met, _OB(), action_avg_k=13.0, baseline=_baseline(),
                                weights=w, acc_threshold=THR, acc_threshold_m2=THR)

    def test_loss_does_NOT_affect_priority_or_metric_ok(self):

        bad_loss = self._r("continuous", 0.40)
        self.assertTrue(bad_loss.metric_ok)
        self.assertEqual(bad_loss.priority, 3)

    def test_loss_ok_diagnostic_tracks_clean_baseline(self):

        self.assertTrue(self._r("continuous", 0.370).loss_ok)
        self.assertFalse(self._r("continuous", 0.40).loss_ok)
        self.assertTrue(self._r("continuous", 0.10).loss_ok)

    def test_loss_ok_same_reward_regardless_of_loss(self):


        self.assertEqual(self._r("continuous", 0.370).reward,
                         self._r("continuous", 0.40).reward)

    def test_tiered_loss_ok_default_true(self):

        r = self._r("tiered", 5.0)
        self.assertTrue(r.loss_ok)
        self.assertEqual(r.priority, 3)


class FusionSweepBrakeTest(unittest.TestCase):
    """As fusion rises, std rises (fusion adds noise); the reward must peak at a
    feasible LOW-noise fusion level and the high-fusion (high-std) tail must be
    pinned at the floor (P2, no cost) — the principled anti-runaway brake."""

    def test_peak_is_feasible_not_max_fusion(self):
        max_f = 35
        rewards = []
        for f in range(0, max_f + 1):

            std = 0.002 + 0.0007 * f
            cost = min(4.5, 4.5 * f / max_f)
            rewards.append((f, _reward(0.871, std=std, fusion=f, cost=cost)))
        peak_f, peak_b = max(rewards, key=lambda fr: fr[1].reward)
        self.assertEqual(peak_b.priority, 3, "peak must be strictly feasible (P3)")
        self.assertLess(peak_f, max_f, "max fusion must not be optimal")
        self.assertLessEqual(rewards[-1][1].reward, peak_b.reward)

        self.assertEqual(rewards[-1][1].priority, 2)


class ADR016LandscapeTest(unittest.TestCase):
    """2026-06-16 ADR-016: the continuous reward must have (1) a recovery gradient in
    the violated region (a milder violation scores strictly HIGHER than a deeper one
    — the old -VIO*exp form clipped both to a flat -5 → the 5th 60k froze), and (2) an
    INTERIOR peak at a feasible moderate fusion with a restoring force past it (the
    old cost lure dominated → ran fusion to max), via the headroom-scaled cost (cost
    fades smoothly to 0 as the margin → 0, so there is no P3-gate cliff)."""

    def _cont(self, margin, cost_frac, w):

        eff = cost_frac * w.p3_cost_budget if margin >= 0.0 else 0.0
        scalar, _acc_b, _stab_b = R._continuous_reward(
            acc_margins=[margin], std_margins=[10.0],
            effective_cost_score=eff, invalid=False, weights=w,
        )
        return scalar

    def test_violated_region_has_recovery_gradient(self):
        w = _weights()
        mild = self._cont(-2.0, 0.0, w)
        deep = self._cont(-20.0, 0.0, w)
        self.assertGreater(mild, deep + 0.5, "milder violation must score strictly higher (recovery gradient)")
        self.assertGreaterEqual(deep, -5.0001)

        self.assertNotAlmostEqual(mild, deep, places=2)

    def test_interior_peak_not_max_fusion(self):


        w = _weights()
        rewards = []
        for f in range(0, 37):
            cost_frac = min(1.0, f / 30.0)
            margin = 3.0 - 0.18 * f
            rewards.append((f, self._cont(margin, cost_frac, w)))
        peak_f, peak_r = max(rewards, key=lambda fr: fr[1])
        self.assertLess(peak_f, 30, "peak must be interior, not max fusion")
        self.assertGreater(peak_f, 2, "peak must not collapse to ~zero fusion")
        self.assertLess(rewards[-1][1], peak_r - 1.0, "max fusion must be clearly worse than the peak")

        self.assertLess(rewards[34][1], rewards[18][1])

    def test_headroom_removes_the_boundary_cliff(self):


        w = _weights()
        just_in = self._cont(0.05, 1.0, w)
        just_out = self._cont(-0.05, 1.0, w)
        self.assertLess(abs(just_in - just_out), 1.0, "no knife-edge cliff at the boundary")

    def test_bounded_and_item7_preserved(self):
        w = _weights()
        vals = [self._cont(m, c, w) for m in (-20, -2, 0.0, 1.0, 3.0) for c in (0.0, 1.0)]
        self.assertTrue(all(-5.0001 <= v <= 5.0001 for v in vals))

        self.assertLess(self._cont(-2.0, 1.0, w), self._cont(2.0, 0.5, w))


if __name__ == "__main__":
    unittest.main(verbosity=2)
