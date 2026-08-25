"""Fusion-cost saturation invariants."""
import unittest

from rfr.preparation.fusion import cost as FC
from blb_stage2_rl import reward as RW


class SaturateShapeTest(unittest.TestCase):
    def test_identity_when_off(self):
        for x in (0.0, 0.1, 0.37, 0.5, 1.0):
            self.assertAlmostEqual(FC.saturate_fusion(x, 0.0), x, places=12)
            self.assertAlmostEqual(FC.saturate_fusion(x, -1.0), x, places=12)

    def test_endpoints(self):
        self.assertAlmostEqual(FC.saturate_fusion(0.0, 0.15), 0.0, places=9)
        self.assertAlmostEqual(FC.saturate_fusion(1.0, 0.15), 1.0, places=9)

    def test_monotone_increasing(self):
        ys = [FC.saturate_fusion(x / 50.0, 0.15) for x in range(51)]
        self.assertTrue(all(ys[i] <= ys[i + 1] + 1e-12 for i in range(len(ys) - 1)))

    def test_concave_diminishing_marginal(self):

        xs = [i / 40.0 for i in range(41)]
        ys = [FC.saturate_fusion(x, 0.15) for x in xs]
        slopes = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
        self.assertTrue(all(slopes[i] >= slopes[i + 1] - 1e-12 for i in range(len(slopes) - 1)))

        self.assertLess(slopes[-1], 0.1 * slopes[0])

    def test_knee_lifts_low_values(self):

        self.assertGreater(FC.saturate_fusion(0.23, 0.15), 0.7)
        self.assertGreater(FC.saturate_fusion(0.1, 0.15), 0.1)


class ComputeFusionCostSaturationTest(unittest.TestCase):
    def _choices(self, n_fused):

        ch = []
        for i in range(12):
            ch.append(FC.BlockChoice(block_idx=2, graph_key=f"block2_L{i}",
                                     fusion_count=(1 if i < n_fused else 0),
                                     max_fusion=1, k_value=13))
        return ch

    def test_saturated_field_present_and_raw_unchanged(self):
        res0 = FC.compute_fusion_cost_saving(self._choices(6), fusion_w={2: 150.0},
                                             trunc_w=50.0, fusion_saturation_tau=0.0)
        res = FC.compute_fusion_cost_saving(self._choices(6), fusion_w={2: 150.0},
                                            trunc_w=50.0, fusion_saturation_tau=0.15)

        self.assertAlmostEqual(res0.fusion_norm, res.fusion_norm, places=9)

        self.assertAlmostEqual(res0.fusion_norm_saturated, res0.fusion_norm, places=9)

        self.assertGreaterEqual(res.fusion_norm_saturated, res.fusion_norm - 1e-9)

    def test_marginal_saturated_reward_decays(self):

        def sat(n):
            return FC.compute_fusion_cost_saving(
                self._choices(n), fusion_w={2: 150.0}, trunc_w=50.0,
                fusion_saturation_tau=0.15).fusion_norm_saturated
        gain_low = sat(2) - sat(1)
        gain_high = sat(10) - sat(9)
        self.assertGreater(gain_low, gain_high)


class InteriorPeakTest(unittest.TestCase):
    """``cost(fusion) + barrier(margin(fusion))`` peaks at a moderate POSITIVE
    margin — max fusion is never optimal (the property the runaway violated)."""

    def _reward_curve(self, tau):
        w = RW.RewardWeights()
        max_f = 35
        budget = float(w.p3_cost_budget)
        fusion_budget = budget * float(RW.FUSION_COST_BUDGET_FRACTION)

        def margin(f):
            return (13.0 - f) * 0.1

        rewards = []
        for f in range(0, max_f + 1):
            fn = f / float(max_f)
            cost = FC.saturate_fusion(fn, tau) * fusion_budget
            mu = margin(f)
            bar = RW.accuracy_margin_barrier(mu, w)
            if mu >= 0.0:
                rewards.append(40.0 + cost + bar)
            else:
                rewards.append(max(float(w.acc_barrier_floor), bar))
        return rewards, margin

    def test_peak_at_positive_margin_not_max(self):
        rewards, margin = self._reward_curve(RW.FUSION_SATURATION_TAU)
        peak = max(range(len(rewards)), key=lambda i: rewards[i])
        self.assertGreater(margin(peak), 0.0, "optimum must sit at positive margin")
        self.assertLess(peak, 13, "optimum must be below the feasibility boundary")
        self.assertLess(rewards[-1], rewards[peak], "max fusion must not be optimal")

    def test_no_runaway_incentive_past_knee(self):


        rewards, _ = self._reward_curve(RW.FUSION_SATURATION_TAU)
        peak = max(range(len(rewards)), key=lambda i: rewards[i])
        self.assertLessEqual(rewards[11], rewards[peak])

    def test_margin_ref_raised(self):
        self.assertGreaterEqual(RW.DEFAULT_ACC_BARRIER_MARGIN_REF, 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
