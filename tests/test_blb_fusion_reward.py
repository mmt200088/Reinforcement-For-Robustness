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
import fusion_enum
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

    Map-AGNOSTIC: block1/block4 are fusion-degenerate under the old shallow maps
    (max_fusion==0) but fusable under the deeper 10-level maps (max_fusion==1).
    MAX_ACTUAL is computed dynamically from each block's actual max_fusion, so these
    tests assert the self-normalizing invariants (cost_norm 0 at baseline, 1 at max
    saving) + a dynamically-expected MAX_ACTUAL, holding for BOTH map versions.
    """

    @classmethod
    def setUpClass(cls):
        cls.fmap = fcm.FusionCountMap.load("mrpc")

    def _max_fusion(self, graph_key):
        return max((int(o.fusion_count) for o in self.fmap.options(graph_key)), default=0)

    _SCHED = ((1, "block1_mrpc", 11), (2, "block2_mrpc", 12), (4, "block4", 12), (5, "block5_n4", 12))

    def test_block2_block5_fusable_and_all_maps_load(self):
        # block2 / block5_n4 fuse under any map version; block1/block4 are {0 or 1}.
        self.assertGreaterEqual(self._max_fusion("block2_mrpc"), 1)
        self.assertGreaterEqual(self._max_fusion("block5_n4"), 1)
        self.assertIn(self._max_fusion("block1_mrpc"), (0, 1))
        self.assertIn(self._max_fusion("block4"), (0, 1))

    def _expected_max_actual(self):
        total = 0.0
        for blk, gk, n in self._SCHED:
            if self._max_fusion(gk) > 0:
                total += rwd.FUSION_COST_W[blk] * n
            total += rwd.TRUNC_COST_W * n
        return total

    def _schedule_choices(self, *, fusion_count_of, k_value):
        """47-block mrpc schedule: 11 block1 (L1-11) + 12 each of block2/4/5."""
        choices = []
        for blk, gk, n in self._SCHED:
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
        self.assertAlmostEqual(res.max_actual, self._expected_max_actual())

    def test_max_saving_episode_normalizes_to_one(self):
        choices = self._schedule_choices(fusion_count_of=lambda mf: mf, k_value=8)
        res = fusion_cost.compute_fusion_cost_saving(
            choices, fusion_w=rwd.FUSION_COST_W, trunc_w=rwd.TRUNC_COST_W,
        )
        self.assertAlmostEqual(res.cost_norm, 1.0)
        self.assertAlmostEqual(res.max_actual, self._expected_max_actual())


class PinClassificationCriterionTest(unittest.TestCase):
    """Locks the (fusion_count, total_bits) pin criterion in fusion_enum.

    Fusion is driven by the JOINT lowering of several non-rescale encode SFs
    (committed maps: block2 fc=1 = inv_std_fresh+gamma+wk lowered together). Probed
    ALONE from baseline, each such encode keeps fusion_count but lowers total_bits.
    A fusion-only criterion pins all of them and the map collapses to fusion={0};
    the (fusion, bits) criterion keeps them enumerated. Regression guard against the
    2026-06-04 relaxation that was reverted after server ground-truth proved the
    joint structure.
    """

    def test_joint_encode_kept_by_bits_proxy(self):
        base_key = (0, 100)
        # an encode lowered alone: fusion UNCHANGED (it is only a joint lever), but
        # total_bits drops -> the (fusion, bits) probe must force enumeration.
        probe = {"valid": True, "fusion_count": 0, "total_bits": 96}
        self.assertTrue(fusion_enum._level_breaks_pin(probe, base_key))
        # the latent bug it guards: fusion alone is unchanged, so a fusion-only
        # predicate would (wrongly) keep the slot pinnable.
        self.assertEqual(int(probe["fusion_count"]), base_key[0])

    def test_truly_inert_slot_is_pinnable(self):
        # moves neither fusion nor bits -> safe to pin at baseline (min noise).
        self.assertFalse(fusion_enum._level_breaks_pin({"valid": True, "fusion_count": 0, "total_bits": 100}, (0, 100)))

    def test_fusion_change_breaks_pin(self):
        self.assertTrue(fusion_enum._level_breaks_pin({"valid": True, "fusion_count": 1, "total_bits": 100}, (0, 100)))

    def test_invalid_level_breaks_pin(self):
        self.assertTrue(fusion_enum._level_breaks_pin({"valid": False}, (0, 100)))


class ShardedReduceEquivalenceTest(unittest.TestCase):
    """The streaming per-shard `_MinNoiseReducer` (OOM fix for block4's ~3e8 valid
    configs) must produce the SAME options as batch grouping over every valid config.
    """

    def _make_configs(self):
        import random

        rng = random.Random(1234)
        EC = fusion_enum.EvaluatedConfig
        cfgs = [EC((9, 9, 9), 0, 100, 1.0, (-1,), {})]  # baseline: fc0, global-min var
        # explicit fc=1 ties at the fc=1 minimum (distinct signatures, all kept).
        cfgs += [EC((1, 2, 3), 1, 90, 1.2, (-2,), {}),
                 EC((3, 2, 1), 1, 95, 1.2, (-3,), {}),
                 EC((2, 1, 3), 1, 88, 1.2, (-4,), {})]
        for i in range(3000):
            ai = tuple(rng.randint(0, 9) for _ in range(3))
            fc = rng.choice([0, 1, 1, 2])
            cfgs.append(EC(ai, fc, rng.randint(80, 120), round(rng.uniform(1.5, 5.0), 6), (i,), {}))
        return cfgs[0].action_indices, cfgs

    @staticmethod
    def _keys(options):
        return [(o["fusion_count"], round(o["total_variance"], 9), tuple(o["action_indices"])) for o in options]

    def test_sharded_matches_batch(self):
        baseline, cfgs = self._make_configs()
        batch = fusion_enum.group_min_noise_options(cfgs, baseline)
        k = 7
        reducers = [fusion_enum._MinNoiseReducer() for _ in range(k)]
        for i, ec in enumerate(cfgs):
            reducers[i % k].add(ec)
        merged = []
        for r in reducers:
            merged.extend(r.results())
        sharded = fusion_enum.group_min_noise_options(merged, baseline)
        self.assertEqual(self._keys(batch), self._keys(sharded))
        # the fc=1 minimum has 3 tied members; all must survive both paths.
        fc1_batch = [o for o in batch if o["fusion_count"] == 1]
        self.assertEqual(len(fc1_batch), 3)

    def test_reducer_counts_and_shrinks(self):
        _baseline, cfgs = self._make_configs()
        r = fusion_enum._MinNoiseReducer()
        for ec in cfgs:
            r.add(ec)
        self.assertEqual(r.num_valid, len(cfgs))  # true count preserved
        self.assertLess(len(r.results()), len(cfgs))  # kept set is small


@unittest.skipUnless(_HAS_ASP, "action_space requires torch (server contract gate)")
class HybridDecodeTest(unittest.TestCase):
    """2026-06-04 rule (restored 2026-06-10 after the uniform/floor-12 grid was
    reverted same-day): all SF kinds use a fixed 10-level hybrid sweep from the
    baseline SF (top 5 step-2, bottom 5 step-1, reaching baseline-14); N forced
    to 16384. The fusion builder skips duplicate-value (snap-floored) levels
    pre-enumeration — a result-equivalent acceleration only."""

    def test_all_sf_kinds_are_10_levels(self):
        self.assertEqual(_asp.LEVELS_F, 10)
        self.assertEqual(_asp.LEVELS_W, 10)
        self.assertEqual(_asp.LEVELS_MS, 10)
        self.assertEqual(_asp.LEVELS_R, 10)
        self.assertFalse(hasattr(_asp, "MIN_SF_FLOOR"))  # floor-12 fully reverted

    def test_hybrid_sweep_matches_user_example(self):
        # baseline 30 -> 30,28,26,24,22,20,19,18,17,16 (idx 9..0).
        got = [_asp.sf_from(i, 30, 10) for i in range(9, -1, -1)]
        self.assertEqual(got, [30, 28, 26, 24, 22, 20, 19, 18, 17, 16])

    def test_rescale_idx0_none_max_idx_is_baseline(self):
        self.assertIsNone(_asp._rescale_sf_from_index(0, 30))
        self.assertEqual(_asp._rescale_sf_from_index(9, 30), 30)   # max idx -> baseline SF
        self.assertEqual(_asp._rescale_sf_from_index(1, 30), 17)   # offset 13
        # baseline invariant: max idx decodes to max_sf so option0 == baseline.
        self.assertEqual(_asp._rescale_sf_from_index(_asp.LEVELS_R - 1, 27), 27)

    def test_field_level_values_rescale_and_fresh(self):
        r = _asp._field_level_values(kind="R", levels=_asp.LEVELS_R, max_sf=30, N=16384)
        self.assertEqual(len(r), 10)
        self.assertIsNone(r[0])
        self.assertEqual([int(v) for v in r[1:]], [17, 18, 19, 20, 22, 24, 26, 28, 30])
        f = _asp._field_level_values(kind="F", levels=_asp.LEVELS_F, max_sf=30, N=16384)
        self.assertEqual([int(v) for v in f], [16, 17, 18, 19, 20, 22, 24, 26, 28, 30])

    def test_low_baseline_sf_snaps_to_floor(self):
        # baseline SF 14 (mask): the deep levels fall below 10 -> snapped to 10.
        vlow = _asp._field_level_values(kind="F", levels=_asp.LEVELS_F, max_sf=14, N=16384)
        self.assertTrue(all(int(v) >= 10 for v in vlow))
        self.assertEqual(int(vlow[-1]), 14)   # max idx -> baseline SF

    def test_distinct_level_indices_skip_duplicate_values_only(self):
        # baseline 30: all 10 hybrid values distinct -> all 10 enumerable.
        self.assertEqual(
            _asp.distinct_sf_level_indices(kind="F", levels=10, max_sf=30, N=16384),
            list(range(10)),
        )
        # baseline 20: hybrid values 20,18,16,14,12,10 then snap-duplicates of 10
        # at idx 0..4 -> keep the LOWEST index per value (lex-min representative
        # the post-eval signature dedup would keep): [0, 5, 6, 7, 8, 9].
        d20 = _asp.distinct_sf_level_indices(kind="F", levels=10, max_sf=20, N=16384)
        self.assertEqual(d20, [0, 5, 6, 7, 8, 9])
        vals20 = _asp._field_level_values(kind="F", levels=10, max_sf=20, N=16384)
        self.assertEqual(sorted(int(vals20[i]) for i in d20), [10, 12, 14, 16, 18, 20])
        # baseline 14: distinct values {14, 12, 10} -> [0, 8, 9].
        self.assertEqual(
            _asp.distinct_sf_level_indices(kind="F", levels=10, max_sf=14, N=16384),
            [0, 8, 9],
        )
        # R: idx0 (None/drop) never enumerable; baseline 30 keeps idx 1..9.
        dr = _asp.distinct_sf_level_indices(kind="R", levels=10, max_sf=30, N=16384)
        self.assertEqual(dr, list(range(1, 10)))

    def test_block_default_N_forced_to_16384(self):
        for b in (1, 2, 3, 4, 5):
            self.assertEqual(_asp._block_default_N(b, gelu_degree=0, attn_degree=2), 16384)
            self.assertEqual(_asp._block_default_N(b, gelu_degree=4, attn_degree=4), 16384)


if __name__ == "__main__":
    unittest.main()
