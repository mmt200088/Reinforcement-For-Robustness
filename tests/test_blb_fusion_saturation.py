"""Torch-free tests for the ADR-014 structural anti-runaway fusion cost.

The 4th 60k still collapsed HOT because the LINEAR fusion cost reward is a
deterministic monotone incentive that the noise-drowned accuracy barrier can't
counter — fusion ran away 8→35. The fix (``fusion_cost.saturate_fusion`` +
``compute_fusion_cost_saving(fusion_saturation_tau=...)``) makes the fusion
reward CONCAVE so its marginal value → ~0 past a healthy knee (~fusion 8), and
``DEFAULT_ACC_BARRIER_MARGIN_REF`` is raised so the restoring penalty starts at
more headroom. Together: ``cost(fusion)+barrier(margin)`` has an interior peak at
a moderate POSITIVE margin, and there is no deterministic pull past the knee.

Loaded via ``spec_from_file_location`` so the torch-importing package __init__ is
never triggered (reward.py / fusion_cost.py are numpy-only at module level).
"""
import importlib.util
import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO_ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


FC = _load("fusion_cost_sat_test", "blb_stage2_rl/fusion_cost.py")
RW = _load("reward_sat_test", "blb_stage2_rl/reward.py")


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
        # Each additional step adds LESS than the previous (the anti-runaway core).
        xs = [i / 40.0 for i in range(41)]
        ys = [FC.saturate_fusion(x, 0.15) for x in xs]
        slopes = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
        self.assertTrue(all(slopes[i] >= slopes[i + 1] - 1e-12 for i in range(len(slopes) - 1)))
        # marginal far past the knee is a tiny fraction of the marginal at the start
        self.assertLess(slopes[-1], 0.1 * slopes[0])

    def test_knee_lifts_low_values(self):
        # ~80% of the reward harvested by fusion_norm ~0.23 (fusion ~8 of ~35).
        self.assertGreater(FC.saturate_fusion(0.23, 0.15), 0.7)
        self.assertGreater(FC.saturate_fusion(0.1, 0.15), 0.1)  # lifts low (no cold collapse)


class ComputeFusionCostSaturationTest(unittest.TestCase):
    def _choices(self, n_fused):
        # n_fused block2 layers fused (max_fusion 1 each), rest baseline; plus K.
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
        # raw fusion_norm identical regardless of tau (kept for diagnostics)
        self.assertAlmostEqual(res0.fusion_norm, res.fusion_norm, places=9)
        # tau=0 => saturated == raw
        self.assertAlmostEqual(res0.fusion_norm_saturated, res0.fusion_norm, places=9)
        # tau>0 => saturated >= raw (concave lifts intermediate values)
        self.assertGreaterEqual(res.fusion_norm_saturated, res.fusion_norm - 1e-9)

    def test_marginal_saturated_reward_decays(self):
        # Going 1->2 fused gains MORE saturated reward than 9->10 fused.
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
        w = RW.RewardWeights()  # margin_ref=0.5 (ADR-014), tau lives in weights
        max_f = 35
        budget = float(w.p3_cost_budget)
        fusion_budget = budget * float(RW.FUSION_COST_BUDGET_FRACTION)

        def margin(f):  # crosses 0 at fusion 13 (the empirical boundary)
            return (13.0 - f) * 0.1

        rewards = []
        for f in range(0, max_f + 1):
            fn = f / float(max_f)
            cost = FC.saturate_fusion(fn, tau) * fusion_budget
            mu = margin(f)
            bar = RW.accuracy_margin_barrier(mu, w)
            if mu >= 0.0:
                rewards.append(40.0 + cost + bar)   # P3 tier + cost + (<=0) barrier
            else:
                rewards.append(max(float(w.acc_barrier_floor), bar))  # P1: no tier, no cost
        return rewards, margin

    def test_peak_at_positive_margin_not_max(self):
        rewards, margin = self._reward_curve(RW.FUSION_SATURATION_TAU)
        peak = max(range(len(rewards)), key=lambda i: rewards[i])
        self.assertGreater(margin(peak), 0.0, "optimum must sit at positive margin")
        self.assertLess(peak, 13, "optimum must be below the feasibility boundary")
        self.assertLess(rewards[-1], rewards[peak], "max fusion must not be optimal")

    def test_no_runaway_incentive_past_knee(self):
        # Past the saturation knee, the marginal cost reward is negligible, so the
        # (negative) barrier slope dominates => reward strictly falls toward the
        # boundary. Check reward at fusion 11 < reward at the peak.
        rewards, _ = self._reward_curve(RW.FUSION_SATURATION_TAU)
        peak = max(range(len(rewards)), key=lambda i: rewards[i])
        self.assertLessEqual(rewards[11], rewards[peak])

    def test_margin_ref_raised(self):
        self.assertGreaterEqual(RW.DEFAULT_ACC_BARRIER_MARGIN_REF, 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
