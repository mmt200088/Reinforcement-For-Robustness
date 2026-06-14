"""ADR-013 Stage-1-style log-barrier accuracy term — torch-free locks.

The 3rd 60k fusion run (stage2_grid_gate_60k_20260612_191530) collapsed HOT:
fusion marched monotonically 1.4 -> 35, metric1 0.866 -> 0.690, and the back
half froze flat at reward -6.95 (every episode catastrophic P1) because nothing
created a restoring force at the feasible frontier and the cliff had no gradient
to climb back from. The log-barrier (mirroring Stage-1's proven
``log_barrier_reward``) replaces both the ADR-012 near-miss tier (P1) and the
linear P3 metric-margin (P3). These tests lock the four properties that make it
work, plus the invariants it must NOT break (selection / item 7 / determinism).

``blb_stage2_rl/reward.py`` is torch-free but ``blb_stage2_rl/__init__`` pulls
torch, so we import the module by file path (same trick as
``test_stage1_eval_accel.py``).
"""
import importlib.util
import math
import pathlib
import sys
import unittest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_reward_module():
    path = _REPO_ROOT / "blb_stage2_rl" / "reward.py"
    spec = importlib.util.spec_from_file_location("_reward_logbarrier_solo", path)
    mod = importlib.util.module_from_spec(spec)
    # dataclasses need the module registered before exec (ClassVar lookup).
    sys.modules["_reward_logbarrier_solo"] = mod
    spec.loader.exec_module(mod)
    return mod


rwd = _load_reward_module()


def _weights(**kw):
    # ADR-013/014 tests exercise the TIERED reward path; ADR-015 flipped the
    # default to "continuous", so pin tiered here (the continuous path has its
    # own coverage in test_blb_continuous_reward.py).
    base = dict(baseline_metric1=0.871, baseline_metric2=0.871, acc_tolerance=0.005,
                reward_design="tiered")
    base.update(kw)
    return rwd.RewardWeights(**base)


class _Opt:
    def __init__(self, fusion, bits):
        self.total_fusion_count = int(fusion)
        self.total_bits_sum = int(bits)
        self.any_invalid = False


def _baseline():
    return rwd.BaselineCostStats(
        total_bits_sum=11285, total_fusion_count=0, avg_k=13.0,
        metric1_mean=0.871, metric2_mean=0.871, loss_mean=0.37,
        metric1_std=0.002, metric2_std=0.002, loss_std=0.01,
        typical_bits_drop=1000, typical_fusion_count=24, typical_k_drop=5,
    )


THR = 0.858


def _reward(m1, *, fusion=0, ext=0.0, weights=None, invalid=False, m1_std=0.002):
    w = weights if weights is not None else _weights()
    met = rwd.EpisodeMetrics(
        loss_mean=0.37, loss_std=0.01, metric1_mean=m1, metric2_mean=m1,
        metric1_std=m1_std, metric2_std=m1_std,
    )
    opt = _Opt(fusion, 11285 - 30 * fusion)
    opt.any_invalid = invalid
    return rwd.compute_reward(
        met, opt, 13.0, _baseline(), weights=w,
        acc_threshold=THR, acc_threshold_m2=THR,
        external_cost_score=float(ext), external_cost_rank=float(fusion),
        any_invalid=invalid,
    )


class BarrierHelperShapeTest(unittest.TestCase):
    def test_zero_beyond_headroom(self):
        w = _weights()
        ref = w.acc_barrier_margin_ref
        self.assertEqual(rwd.accuracy_margin_barrier(ref, w), 0.0)
        self.assertEqual(rwd.accuracy_margin_barrier(ref + 0.5, w), 0.0)
        self.assertEqual(rwd.accuracy_margin_barrier(2.0, w), 0.0)

    def test_restoring_force_below_headroom(self):
        """Satisfied side: strictly decreasing toward the boundary (mu -> 0)."""
        w = _weights()
        ref = w.acc_barrier_margin_ref
        xs = [ref * f for f in (0.99, 0.75, 0.5, 0.25, 0.1, 0.02)]
        ys = [rwd.accuracy_margin_barrier(x, w) for x in xs]
        for y in ys:
            self.assertLessEqual(y, 0.0)
        # decreasing as margin shrinks
        self.assertEqual(ys, sorted(ys, reverse=True))
        self.assertLess(ys[-1], ys[0])

    def test_violated_monotone_no_plateau(self):
        """Violated side keeps a gradient over the whole realistic depth — the
        missing recovery path of the 3rd 60k (flat -6.95)."""
        w = _weights()
        xs = [(-0.05) - 0.1 * i for i in range(120)]  # mu from -0.05 down to ~-12
        ys = [rwd.accuracy_margin_barrier(x, w) for x in xs]
        # strictly decreasing until the floor; no flat plateau in the body
        body = [y for y in ys if y > w.acc_barrier_floor + 1e-9]
        self.assertTrue(all(body[i] > body[i + 1] for i in range(len(body) - 1)))
        self.assertGreater(len(set(round(y, 6) for y in body)), 50)

    def test_floor_clamped(self):
        w = _weights()
        self.assertEqual(rwd.accuracy_margin_barrier(-1000.0, w), w.acc_barrier_floor)
        self.assertGreaterEqual(rwd.accuracy_margin_barrier(-8.0, w), w.acc_barrier_floor)

    def test_continuous_at_boundary(self):
        w = _weights()
        left = rwd.accuracy_margin_barrier(-1e-6, w)
        right = rwd.accuracy_margin_barrier(1e-6, w)
        self.assertLess(abs(left - right), 1e-2)

    def test_margin_ref_moves_the_penalty_onset(self):
        loose = _weights(acc_barrier_margin_ref=0.15)
        strict = _weights(acc_barrier_margin_ref=0.50)
        mu = 0.30
        # at mu=0.30: loose REF=0.15 -> beyond headroom -> 0; strict REF=0.50 -> penalized
        self.assertEqual(rwd.accuracy_margin_barrier(mu, loose), 0.0)
        self.assertLess(rwd.accuracy_margin_barrier(mu, strict), 0.0)


class InteriorPeakTest(unittest.TestCase):
    """The defining property: cost(fusion) + barrier(margin(fusion)) peaks at a
    POSITIVE-margin interior point, and reward DECLINES on both sides."""

    @staticmethod
    def _margin_model(f):
        # convex accuracy drop: early fusions ~free, later ones expensive
        # (matches the 3rd-60k curve qualitatively).
        return 0.871 - 0.0052 * f * (0.6 + 0.4 * f / 35.0)

    def _sweep(self, weights=None):
        out = []
        for f in range(0, 36):
            m1 = self._margin_model(f)
            ext = min(3.0, 3.0 * f / 36.0)   # P3 fusion cost grows with fusion
            rb = _reward(m1, fusion=f, ext=ext, weights=weights)
            out.append((f, rb.reward, rb.priority, rb.worst_signed_margin))
        return out

    def test_peak_is_interior_and_positive_margin(self):
        sweep = self._sweep()
        peak = max(sweep, key=lambda r: r[1])
        f, reward, prio, mu = peak
        self.assertGreater(f, 0, "peak must not be the zero-fusion baseline")
        self.assertLess(f, 35, "peak must not be max fusion (runaway)")
        self.assertEqual(prio, 3, "peak must be a valid P3 config")
        self.assertGreater(mu, 0.0, "peak must sit at positive accuracy margin (headroom)")

    def test_peak_beats_baseline(self):
        sweep = self._sweep()
        baseline_reward = sweep[0][1]
        peak_reward = max(r[1] for r in sweep)
        self.assertGreater(peak_reward, baseline_reward,
                           "fusion must be adopted (peak > zero-fusion baseline)")

    def test_reward_declines_past_peak(self):
        sweep = self._sweep()
        peak_f = max(sweep, key=lambda r: r[1])[0]
        past = [r[1] for r in sweep if r[0] >= peak_f]
        # monotone non-increasing after the peak -> restoring force toward it
        self.assertTrue(all(past[i] >= past[i + 1] - 1e-9 for i in range(len(past) - 1)))

    def test_violated_region_has_recovery_gradient(self):
        sweep = self._sweep()
        viol = [r[1] for r in sweep if r[2] == 1]  # P1 episodes
        self.assertGreater(len(viol), 3)
        # deeper into collapse (more fusion) => strictly lower reward => a
        # gradient pointing back toward feasibility (the 3rd-60k had none).
        self.assertTrue(all(viol[i] > viol[i + 1] for i in range(len(viol) - 1)))


class HardPriorityPreservedTest(unittest.TestCase):
    """The barrier rewrites only the PPO scalar; priority / item 7 unchanged."""

    def test_priority_unchanged_by_barrier(self):
        # margin just below threshold is P1 with or without the barrier.
        on = _reward(THR - 0.004, weights=_weights(acc_barrier_enabled=True))
        off = _reward(THR - 0.004, weights=_weights(acc_barrier_enabled=False))
        self.assertEqual(on.priority, 1)
        self.assertEqual(off.priority, 1)
        # a comfortable config is P3 either way
        on3 = _reward(0.871, ext=1.0, weights=_weights(acc_barrier_enabled=True))
        off3 = _reward(0.871, ext=1.0, weights=_weights(acc_barrier_enabled=False))
        self.assertEqual(on3.priority, 3)
        self.assertEqual(off3.priority, 3)

    def test_item7_violation_below_every_p3(self):
        """No accuracy-violated (P1) episode can ever out-score ANY P3, so cost
        cannot offset an accuracy failure even in the scalar."""
        w = _weights()
        worst_p3 = _reward(THR + 1e-4, fusion=0, ext=0.0, weights=w)   # thinnest P3, no cost
        # a P1 with the maximum possible cost handed in (should be ignored in P1)
        best_p1 = _reward(THR - 1e-4, fusion=24, ext=4.5, weights=w)
        self.assertEqual(worst_p3.priority, 3)
        self.assertEqual(best_p1.priority, 1)
        self.assertLess(best_p1.reward, worst_p3.reward)

    def test_p1_never_gets_cost(self):
        w = _weights()
        no_cost = _reward(THR - 0.01, fusion=0, ext=0.0, weights=w)
        big_cost = _reward(THR - 0.01, fusion=24, ext=4.5, weights=w)
        # same margin -> identical barrier -> cost did not leak into P1
        self.assertAlmostEqual(no_cost.reward, big_cost.reward, places=6)

    def test_invalid_uses_legacy_term_not_barrier(self):
        b = _reward(0.80, fusion=10, ext=3.0, invalid=True)
        self.assertEqual(b.priority, 1)
        self.assertEqual(b.acc_barrier_vio, 0.0)   # barrier skipped on invalid
        self.assertEqual(b.acc_barrier_sat, 0.0)
        self.assertLess(b.reward, 0.0)


class DeterminismTest(unittest.TestCase):
    def test_barrier_is_pure_function_of_metrics(self):
        a = _reward(0.864, fusion=8, ext=1.2)
        b = _reward(0.864, fusion=8, ext=1.2)
        self.assertEqual(a.reward, b.reward)
        self.assertEqual(a.worst_signed_margin, b.worst_signed_margin)
        self.assertEqual(a.acc_barrier_sat, b.acc_barrier_sat)


if __name__ == "__main__":
    unittest.main(verbosity=2)
