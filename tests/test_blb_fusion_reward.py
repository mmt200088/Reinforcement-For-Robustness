"""Torch-free tests for the Stage-2 fusion-count reward redesign (2026-06-03).

Covers the pure per-block weighted cost helper (``fusion_cost``) and the
``external_cost`` threading in ``reward.compute_reward``. Both modules are torch-free
and imported by bare name with ``blb_stage2_rl/`` on ``sys.path`` (the package
``__init__`` pulls torch, which the local box lacks).
"""
from __future__ import annotations

import pathlib
import sys
import unittest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_BLB_DIR = _REPO_ROOT / "blb_stage2_rl"
for _p in (str(_REPO_ROOT), str(_BLB_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fusion_cost
import fusion_count_map as fcm
import reward as rwd

try:  # action_space transitively imports torch (blb_rl_bridge) — skip locally if absent
    import action_space as _asp
    _HAS_ASP = True
except Exception:
    _HAS_ASP = False

# Spec weights: block1:block2:block4:block5:truncation = 80:150:130:40:50.
FW = {1: 80.0, 2: 150.0, 4: 130.0, 5: 40.0}
TW = 50.0


class _Sig:
    """Minimal opt_signals stand-in."""
    any_invalid = False
    total_bits_sum = 100
    total_fusion_count = 0


def _baseline():
    return rwd.BaselineCostStats(
        total_bits_sum=200, total_fusion_count=0, avg_k=13.0,
        loss_mean=0.3, loss_std=0.01,
        metric1_mean=0.85, metric2_mean=0.85,
        metric1_std=0.01, metric2_std=0.01,
        typical_bits_drop=50, typical_fusion_count=1, typical_k_drop=2,
    )


def _weights():
    return rwd.RewardWeights(baseline_metric1=0.85, baseline_metric2=0.85)


def _bc(block_idx, fusion_count, max_fusion, k_value, graph_key="g"):
    return fusion_cost.BlockChoice(
        block_idx=block_idx,
        graph_key=graph_key,
        fusion_count=fusion_count,
        max_fusion=max_fusion,
        k_value=k_value,
    )


class FusionCostSavingTest(unittest.TestCase):
    def test_baseline_zero_saving(self):
        # All option 0 (fusion_count=0) + K=max(13) => zero saving.
        choices = [
            _bc(2, 0, 1, 13),
            _bc(5, 0, 1, 13),
            _bc(1, 0, 0, 13),
            _bc(4, 0, 0, 13),
        ]
        res = fusion_cost.compute_fusion_cost_saving(choices, fusion_w=FW, trunc_w=TW)
        self.assertEqual(res.cost_norm, 0.0)
        self.assertEqual(res.cost_rank, 0.0)
        # denom = block2 fusion 150 + block5 fusion 40 + 4 * trunc 50 = 390.
        self.assertAlmostEqual(res.max_actual, 390.0)

    def test_full_saving_normalizes_to_one(self):
        # fusion_count == max_fusion AND K=min(8) on every fusable lever.
        choices = [
            _bc(2, 1, 1, 8),
            _bc(5, 1, 1, 8),
            _bc(1, 0, 0, 8),
            _bc(4, 0, 0, 8),
        ]
        res = fusion_cost.compute_fusion_cost_saving(choices, fusion_w=FW, trunc_w=TW)
        self.assertAlmostEqual(res.cost_norm, 1.0)
        self.assertAlmostEqual(res.cost_rank, 390.0)

    def test_block1_block4_fusion_inert(self):
        # max_fusion==0 => fusion weight never contributes (even with a bogus count),
        # and the 80/130 fusion weights are absent from the normalizer.
        choices = [_bc(1, 99, 0, 8), _bc(4, 99, 0, 8)]
        res = fusion_cost.compute_fusion_cost_saving(choices, fusion_w=FW, trunc_w=TW)
        self.assertAlmostEqual(res.max_actual, 100.0)  # 2 * trunc only, no 80/130
        self.assertAlmostEqual(res.cost_rank, 100.0)   # 2 * (50 * trunc_saving=1)
        for pb in res.per_block:
            self.assertEqual(pb["fusion_contrib"], 0.0)
            self.assertEqual(pb["fusion_saving"], 0.0)

    def test_trunc_saving_linear(self):
        # K=13 -> 0, K=8 -> 1, midpoint K=10.5 not in levels; check K=11 -> (13-11)/5.
        choices = [_bc(2, 0, 1, 11)]
        res = fusion_cost.compute_fusion_cost_saving(choices, fusion_w=FW, trunc_w=TW)
        # actual = 50 * (13-11)/5 = 50 * 0.4 = 20 ; denom = 150 (fusion) + 50 = 200.
        self.assertAlmostEqual(res.cost_rank, 20.0)
        self.assertAlmostEqual(res.max_actual, 200.0)
        self.assertAlmostEqual(res.cost_norm, 0.1)

    def test_rank_monotonic_in_fusion(self):
        base = [_bc(2, 0, 1, 13)]
        more = [_bc(2, 1, 1, 13)]
        r0 = fusion_cost.compute_fusion_cost_saving(base, fusion_w=FW, trunc_w=TW)
        r1 = fusion_cost.compute_fusion_cost_saving(more, fusion_w=FW, trunc_w=TW)
        self.assertGreater(r1.cost_rank, r0.cost_rank)
        self.assertAlmostEqual(r1.cost_rank - r0.cost_rank, 150.0)

    def test_precomputed_max_actual_used(self):
        choices = [_bc(2, 1, 1, 8)]
        res = fusion_cost.compute_fusion_cost_saving(
            choices, fusion_w=FW, trunc_w=TW, max_actual=400.0
        )
        # actual = 150 + 50 = 200 ; norm = 200/400 = 0.5.
        self.assertAlmostEqual(res.cost_norm, 0.5)
        self.assertAlmostEqual(res.max_actual, 400.0)


class ExternalCostThreadingTest(unittest.TestCase):
    def test_weight_constants_match_spec(self):
        self.assertEqual(rwd.FUSION_COST_W, {1: 80.0, 2: 150.0, 4: 130.0, 5: 40.0})
        self.assertEqual(rwd.TRUNC_COST_W, 50.0)
        self.assertEqual((rwd.K_MAX_BITS, rwd.K_MIN_BITS), (13, 8))

    def test_p3_uses_external_cost(self):
        # accuracy ok (m == baseline) + stability ok (std 0) => P3; external cost used.
        metrics = rwd.EpisodeMetrics(
            loss_mean=0.3, loss_std=0.0,
            metric1_mean=0.85, metric2_mean=0.85,
            metric1_std=0.0, metric2_std=0.0,
        )
        bd = rwd.compute_reward(
            metrics, _Sig(), action_avg_k=13.0, baseline=_baseline(), weights=_weights(),
            external_cost_score=3.0, external_cost_rank=999.0,
        )
        self.assertEqual(bd.priority, 3)
        self.assertAlmostEqual(bd.cost_score, 3.0)
        self.assertAlmostEqual(bd.cost_rank_score, 999.0)
        self.assertGreater(bd.reward, 40.0)  # tier +40 floor + margin + cost

    def test_external_cost_clipped_to_budget(self):
        metrics = rwd.EpisodeMetrics(
            loss_mean=0.3, loss_std=0.0,
            metric1_mean=0.85, metric2_mean=0.85,
            metric1_std=0.0, metric2_std=0.0,
        )
        w = _weights()
        bd = rwd.compute_reward(
            metrics, _Sig(), action_avg_k=13.0, baseline=_baseline(), weights=w,
            external_cost_score=999.0, external_cost_rank=999.0,
        )
        self.assertAlmostEqual(bd.cost_score, float(w.p3_cost_budget))  # clipped

    def test_p1_ignores_external_cost(self):
        # accuracy fail (m below threshold) => P1; external cost must not contribute.
        metrics = rwd.EpisodeMetrics(
            loss_mean=2.0, loss_std=0.0,
            metric1_mean=0.50, metric2_mean=0.50,
            metric1_std=0.0, metric2_std=0.0,
        )
        bd = rwd.compute_reward(
            metrics, _Sig(), action_avg_k=13.0, baseline=_baseline(), weights=_weights(),
            external_cost_score=3.0, external_cost_rank=999.0,
        )
        self.assertEqual(bd.priority, 1)
        self.assertEqual(bd.cost_score, 0.0)
        self.assertEqual(bd.cost_rank_score, 0.0)
        self.assertLessEqual(bd.reward, 0.0)

    def test_none_external_cost_preserves_legacy_path(self):
        # Without external cost the old aggregate path still produces a P3 reward.
        metrics = rwd.EpisodeMetrics(
            loss_mean=0.3, loss_std=0.0,
            metric1_mean=0.85, metric2_mean=0.85,
            metric1_std=0.0, metric2_std=0.0,
        )
        bd = rwd.compute_reward(
            metrics, _Sig(), action_avg_k=13.0, baseline=_baseline(), weights=_weights(),
        )
        self.assertEqual(bd.priority, 3)
        # legacy path: cost_score comes from the aggregate scalar (no exception).
        self.assertIsInstance(bd.cost_score, float)


class RealMapIntegrationTest(unittest.TestCase):
    """End-to-end against the committed mrpc maps + reward.FUSION_COST_W weights.

    Validates the block1/block4 fusion-degeneracy that the design relies on is real
    (max_fusion==0 from the actual maps), and that a realistic 47-block episode
    normalizes to [0, 1] with MAX_ACTUAL == 4630.
    """

    @classmethod
    def setUpClass(cls):
        cls.fmap = fcm.FusionCountMap.load("mrpc")

    def _max_fusion(self, graph_key):
        return max((int(o.fusion_count) for o in self.fmap.options(graph_key)), default=0)

    def test_block1_block4_degenerate_block2_block5_fusable(self):
        self.assertEqual(self._max_fusion("block1_mrpc"), 0)   # degenerate (K-only)
        self.assertEqual(self._max_fusion("block4"), 0)        # degenerate (K-only)
        self.assertEqual(self._max_fusion("block2_mrpc"), 1)   # fusable
        self.assertEqual(self._max_fusion("block5_n4"), 1)     # fusable

    def _schedule_choices(self, *, fusion_count_of, k_value):
        """47-block mrpc schedule: 11 block1 (L1-11) + 12 each of block2/4/5."""
        choices = []
        for blk, gk, n in (
            (1, "block1_mrpc", 11), (2, "block2_mrpc", 12),
            (4, "block4", 12), (5, "block5_n4", 12),
        ):
            mf = self._max_fusion(gk)
            for _ in range(n):
                choices.append(fusion_cost.BlockChoice(
                    block_idx=blk, graph_key=gk,
                    fusion_count=fusion_count_of(mf), max_fusion=mf, k_value=k_value,
                ))
        return choices

    def test_baseline_episode_zero_saving(self):
        choices = self._schedule_choices(fusion_count_of=lambda mf: 0, k_value=13)
        res = fusion_cost.compute_fusion_cost_saving(
            choices, fusion_w=rwd.FUSION_COST_W, trunc_w=rwd.TRUNC_COST_W,
        )
        self.assertEqual(len(choices), 47)
        self.assertEqual(res.cost_norm, 0.0)
        self.assertAlmostEqual(res.max_actual, 4630.0)

    def test_max_saving_episode_normalizes_to_one(self):
        choices = self._schedule_choices(fusion_count_of=lambda mf: mf, k_value=8)
        res = fusion_cost.compute_fusion_cost_saving(
            choices, fusion_w=rwd.FUSION_COST_W, trunc_w=rwd.TRUNC_COST_W,
        )
        self.assertAlmostEqual(res.cost_norm, 1.0)
        self.assertAlmostEqual(res.max_actual, 4630.0)


@unittest.skipUnless(_HAS_ASP, "action_space requires torch (server contract gate)")
class RescaleDecodeExpansionTest(unittest.TestCase):
    """2026-06-04: rescale slots deepened to 15 levels, step-1, snap-floored at SF=10;
    non-rescale decode (step-2) unchanged; baseline (max idx -> max_sf) unchanged."""

    def test_levels_r_is_15(self):
        self.assertEqual(_asp.LEVELS_R, 15)

    def test_rescale_step1_deep_sweep_raw(self):
        # idx 0 = None (drop, never enumerated); idx 14 (max) -> max_sf; step-1.
        self.assertIsNone(_asp._rescale_sf_from_index(0, 30))
        self.assertEqual(_asp._rescale_sf_from_index(14, 30), 30)   # max idx -> max_sf
        self.assertEqual(_asp._rescale_sf_from_index(1, 30), 17)    # 30 - 1*(15-1-1)=30-13
        self.assertEqual(_asp._rescale_sf_from_index(7, 30), 23)    # 30 - (14-7)=23

    def test_field_level_values_rescale_snapped_to_floor(self):
        vals = _asp._field_level_values(kind="R", levels=_asp.LEVELS_R, max_sf=30, N=8192)
        self.assertEqual(len(vals), 15)
        self.assertIsNone(vals[0])
        self.assertEqual(vals[14], 30)
        self.assertEqual(vals[1], 17)
        # low max_sf: idx 1..4 decode below 10 -> snapped to the table floor (10).
        vlow = _asp._field_level_values(kind="R", levels=_asp.LEVELS_R, max_sf=20, N=8192)
        self.assertTrue(all((v is None) or (int(v) >= 10) for v in vlow))
        self.assertEqual(vlow[1], 10)    # 20-13=7 -> snap 10
        self.assertEqual(vlow[14], 20)   # max idx -> max_sf

    def test_non_rescale_decode_unchanged_step2(self):
        # F kind keeps step-2 at its current 5 levels: max=30 -> {22,24,26,28,30}.
        vals = _asp._field_level_values(kind="F", levels=_asp.LEVELS_F, max_sf=30, N=8192)
        self.assertEqual([int(v) for v in vals], [22, 24, 26, 28, 30])

    def test_baseline_max_index_still_decodes_to_max_sf(self):
        # make_all_max sets R slots to LEVELS_R-1 -> must still decode to max_sf
        # so option0 == baseline holds in the fusion-map builder.
        self.assertEqual(_asp._rescale_sf_from_index(_asp.LEVELS_R - 1, 27), 27)


if __name__ == "__main__":
    unittest.main()
