"""Torch-free tests for the ADR-015 CONTINUOUS bounded Stage-2 reward.

The 4 fusion collapses shared two diseases the user named from the curves: the
reward amplitude was too large (the tier 0/+20/+40 structure swings ±40 at the
feasibility boundary) and the stability constraint was vacuous (500% tolerance).
ADR-015 ports Stage-1's design (continuous log-barrier + cost, normalized +
clipped to [-5,+5]) plus the original Stage-2's std stability constraint, and
makes the strict std gate the anti-runaway brake. These tests lock:

  * the reward is bounded to ~[-5,+5] AND continuous across the feasibility
    boundary (no ±40 jump);
  * hard priority / item 7 (a violated barrier dwarfs any P3-gated cost);
  * the strict std gate rejects high-variance (high-fusion) configs → P2 → no
    cost reward (the brake);
  * priority labels and determinism (1==N) are correct.

Loaded via spec_from_file_location (reward.py / fusion_cost.py are numpy-only).
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


R = _load("reward_cont_test", "blb_stage2_rl/reward.py")
FC = _load("fusion_cost_cont_test", "blb_stage2_rl/fusion_cost.py")

THR = 0.858
BASELINE_M = 0.871


def _weights(**kw):
    base = dict(baseline_metric1=BASELINE_M, baseline_metric2=BASELINE_M,
                stab_tolerance=1.2)  # MULTIPLIER on baseline std (2026-06-15): thr = baseline.X_std × tol; 1.2 = original Stage-2's 1.2×
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
    def test_default_is_continuous(self):
        self.assertEqual(R.RewardWeights().reward_design, "continuous")


class BoundedAndContinuousTest(unittest.TestCase):
    def test_all_bounded_in_clip_range(self):
        rs = [
            _reward(0.871, fusion=8, cost=2.0).reward,    # P3 feasible
            _reward(THR + 0.001, fusion=8, cost=2.0).reward,  # just feasible
            _reward(THR - 0.001, fusion=8, cost=2.0).reward,  # just violated
            _reward(0.70, fusion=20, cost=4.5).reward,    # deep violation + big cost
            _reward(0.871, std=0.05, fusion=8, cost=2.0).reward,  # hi-std
            _reward(0.871, invalid=True).reward,
        ]
        for r in rs:
            self.assertLessEqual(r, 5.0 + 1e-9)
            self.assertGreaterEqual(r, -5.0 - 1e-9)

    def test_continuous_across_feasibility_boundary(self):
        # The whole point: crossing the accuracy boundary must NOT swing ±40 like
        # the tiers did — the gap is bounded by the clip width (10), and in
        # practice far smaller near the boundary.
        feas = _reward(THR + 0.0005, fusion=6, cost=1.5).reward
        viol = _reward(THR - 0.0005, fusion=6, cost=1.5).reward
        self.assertLess(abs(feas - viol), 8.0)  # nothing like the old ±40

    def test_tiered_still_jumps(self):
        # Control: the tiered path (rollback) DOES jump ~40 at the boundary.
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
        p2 = _reward(0.871, std=0.05, fusion=8, cost=4.5).reward  # acc ok, std ↑ → P2
        self.assertLess(p2, p3)


class StabilityBrakeTest(unittest.TestCase):
    def test_high_std_is_p2_not_p3(self):
        b = _reward(0.871, std=0.05, fusion=8, cost=2.0)   # std 25x baseline 0.002
        self.assertEqual(b.priority, 2)
        self.assertFalse(b.stab_ok)
        self.assertTrue(b.metric_ok)

    def test_low_std_feasible_is_p3(self):
        b = _reward(0.871, std=0.002, fusion=8, cost=2.0)
        self.assertEqual(b.priority, 3)
        self.assertTrue(b.stab_ok and b.metric_ok)

    def test_strict_tolerance_tightens_gate(self):
        # 2026-06-15 (user spec): stab_tolerance is a MULTIPLIER on baseline std —
        # thr = max(baseline.X_std × tol, stab_floor). Use baseline std=0.01 so the
        # multiplier (not the 0.01 floor) decides the gate, then show 5.0(=5×) is a
        # LENIENT but real gate while 1.2(=1.2×) is strict. Crucially 5.0 is NOT
        # vacuous: a high-enough std still fails it.
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

        # std=0.03: 5.0×(thr=0.05) passes, 1.2×(thr=0.012) rejects.
        self.assertTrue(_stab_at(5.0, 0.03).stab_ok)    # 0.03 < 0.01*5 = 0.05
        self.assertFalse(_stab_at(1.2, 0.03).stab_ok)   # 0.03 > 0.01*1.2 = 0.012
        # 5.0 is a real gate, not vacuous: std=0.06 > 0.05 still fails.
        self.assertFalse(_stab_at(5.0, 0.06).stab_ok)   # 0.06 > 0.01*5 = 0.05


class PriorityLabelTest(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(_reward(0.871, std=0.002).priority, 3)         # feasible
        self.assertEqual(_reward(0.871, std=0.05).priority, 2)          # stab fail
        self.assertEqual(_reward(0.70, std=0.002).priority, 1)          # acc fail
        self.assertEqual(_reward(0.871, invalid=True).priority, 1)      # invalid


class DeterminismTest(unittest.TestCase):
    def test_same_inputs_same_reward(self):
        a = _reward(0.865, std=0.003, fusion=5, cost=1.7).reward
        b = _reward(0.865, std=0.003, fusion=5, cost=1.7).reward
        self.assertEqual(a, b)


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
        # A high loss with OK accuracy + stability stays P3 (loss not in the gate).
        bad_loss = self._r("continuous", 0.40)   # ~8% loss rise
        self.assertTrue(bad_loss.metric_ok)      # accuracy gate unaffected by loss
        self.assertEqual(bad_loss.priority, 3)   # still P3 (loss is selection-only)

    def test_loss_ok_diagnostic_tracks_clean_baseline(self):
        # loss_ok is a deterministic diagnostic vs CLEAN baseline (0.37) × (1+0.5%).
        self.assertTrue(self._r("continuous", 0.370).loss_ok)   # within tol
        self.assertFalse(self._r("continuous", 0.40).loss_ok)   # exceeds clean+0.5%
        self.assertTrue(self._r("continuous", 0.10).loss_ok)    # lower is fine

    def test_loss_ok_same_reward_regardless_of_loss(self):
        # Because loss is out of the scalar, the reward is identical for two configs
        # that differ ONLY in loss_mean (the determinism-relevant property).
        self.assertEqual(self._r("continuous", 0.370).reward,
                         self._r("continuous", 0.40).reward)

    def test_tiered_loss_ok_default_true(self):
        # tiered rollback: loss gate inactive, loss_ok defaults True, priority by m1/m2/std.
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
            # std grows with fusion: low fusion stable, high fusion unstable.
            std = 0.002 + 0.0007 * f
            cost = min(4.5, 4.5 * f / max_f)
            rewards.append((f, _reward(0.871, std=std, fusion=f, cost=cost)))
        peak_f, peak_b = max(rewards, key=lambda fr: fr[1].reward)
        self.assertEqual(peak_b.priority, 3, "peak must be strictly feasible (P3)")
        self.assertLess(peak_f, max_f, "max fusion must not be optimal")
        self.assertLessEqual(rewards[-1][1].reward, peak_b.reward)
        # the high-fusion tail is stability-violated (the brake fired)
        self.assertEqual(rewards[-1][1].priority, 2)


class ADR016LandscapeTest(unittest.TestCase):
    """2026-06-16 ADR-016: the continuous reward must have (1) a recovery gradient in
    the violated region (a milder violation scores strictly HIGHER than a deeper one
    — the old -VIO*exp form clipped both to a flat -5 → the 5th 60k froze), and (2) an
    INTERIOR peak at a feasible moderate fusion with a restoring force past it (the
    old cost lure dominated → ran fusion to max), via the headroom-scaled cost (cost
    fades smoothly to 0 as the margin → 0, so there is no P3-gate cliff)."""

    def _cont(self, margin, cost_frac, w):
        # P3-gating upstream: cost only enters when the config is feasible (margin>=0).
        eff = cost_frac * w.p3_cost_budget if margin >= 0.0 else 0.0
        scalar, _acc_b, _stab_b = R._continuous_reward(
            acc_margins=[margin], std_margins=[10.0],  # std comfortably satisfied
            effective_cost_score=eff, invalid=False, weights=w,
        )
        return scalar

    def test_violated_region_has_recovery_gradient(self):
        w = _weights()
        mild = self._cont(-2.0, 0.0, w)   # ~m1=0.84 in the real run
        deep = self._cont(-20.0, 0.0, w)  # ~m1=0.63 (catastrophic) in the real run
        self.assertGreater(mild, deep + 0.5, "milder violation must score strictly higher (recovery gradient)")
        self.assertGreaterEqual(deep, -5.0001)  # still bounded
        # the OLD -VIO*exp form would clip BOTH to -5 (no gradient) — guard against regress
        self.assertNotAlmostEqual(mild, deep, places=2)

    def test_interior_peak_not_max_fusion(self):
        # Synthetic fusion sweep: cost_frac rises with fusion; margin falls and crosses
        # 0 near f~17 (mirrors the real run). Reward must PEAK at a feasible moderate
        # fusion and be clearly LOWER at max fusion (the anti-runaway restoring force).
        w = _weights()
        rewards = []
        for f in range(0, 37):
            cost_frac = min(1.0, f / 30.0)
            margin = 3.0 - 0.18 * f          # >0 for f<~16.7, <0 beyond
            rewards.append((f, self._cont(margin, cost_frac, w)))
        peak_f, peak_r = max(rewards, key=lambda fr: fr[1])
        self.assertLess(peak_f, 30, "peak must be interior, not max fusion")
        self.assertGreater(peak_f, 2, "peak must not collapse to ~zero fusion")
        self.assertLess(rewards[-1][1], peak_r - 1.0, "max fusion must be clearly worse than the peak")
        # monotone decline from the peak to max fusion (a real restoring force)
        self.assertLess(rewards[34][1], rewards[18][1])

    def test_headroom_removes_the_boundary_cliff(self):
        # Crossing the feasibility boundary must NOT be a big reward cliff: the
        # headroom fades the cost to ~0 just inside, so reward(+) ≈ reward(-).
        w = _weights()
        just_in = self._cont(0.05, 1.0, w)
        just_out = self._cont(-0.05, 1.0, w)
        self.assertLess(abs(just_in - just_out), 1.0, "no knife-edge cliff at the boundary")

    def test_bounded_and_item7_preserved(self):
        w = _weights()
        vals = [self._cont(m, c, w) for m in (-20, -2, 0.0, 1.0, 3.0) for c in (0.0, 1.0)]
        self.assertTrue(all(-5.0001 <= v <= 5.0001 for v in vals))
        # any violated config (cost forced full but gated off) < a feasible one
        self.assertLess(self._cont(-2.0, 1.0, w), self._cont(2.0, 0.5, w))


if __name__ == "__main__":
    unittest.main(verbosity=2)
