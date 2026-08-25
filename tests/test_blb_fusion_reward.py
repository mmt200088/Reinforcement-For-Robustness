"""Stage-2 fusion-count reward invariants."""
from __future__ import annotations

import unittest

from blb_stage2_rl import action_space as _asp
from rfr.preparation.fusion import cost as fusion_cost
from rfr.preparation.fusion import count_map as fcm
from rfr.preparation.fusion import enumeration as fusion_enum
from blb_stage2_rl import layerwise_action
from blb_stage2_rl import reward as rwd


FW = {1: 80.0, 2: 150.0, 4: 130.0, 5: 40.0}
TW = 50.0


class _Sig:
    """Minimal opt_signals stand-in."""
    any_invalid = False
    total_bits_sum = 100
    total_fusion_count = 0


def _layerwise_actions(*, block4_fusion=0, preset_index=0):
    from blb_stage2_rl.precision_presets import PRECISION_PRESETS

    preset = PRECISION_PRESETS[preset_index]
    actions = []
    for _layer_idx in range(12):
        k_by_block = {
            block_idx: preset.k_by_block[block_idx - 1]
            for block_idx in range(1, 6)
        }
        actions.append(layerwise_action.LayerwiseDecodedAction(
            block4_fusion, k_by_block, preset_index,
        ))
    return actions


class LayerwiseVariableCostContractTest(unittest.TestCase):
    def test_higher_block4_fusion_increases_only_fusion_units(self):
        baseline = layerwise_action.compute_variable_cost(
            _layerwise_actions(block4_fusion=0, preset_index=0)
        )
        fused = layerwise_action.compute_variable_cost(
            _layerwise_actions(block4_fusion=1, preset_index=0)
        )

        self.assertEqual(baseline.normalized, 0.0)
        self.assertEqual(fused.fusion_saving, 1.0)
        self.assertEqual(fused.truncation_saving, baseline.truncation_saving)
        self.assertEqual(fused.fusion_units, 12.0)
        self.assertEqual(fused.truncation_units, baseline.truncation_units)
        self.assertEqual(fused.robust_floor, 0.0)
        self.assertEqual(fused.secondary_progress, 0.5)
        self.assertEqual(fused.normalized, 0.5)

    def test_fusion_and_precision_preset_update_independent_axes(self):
        baseline_actions = _layerwise_actions(block4_fusion=0, preset_index=0)
        fusion_actions = _layerwise_actions(block4_fusion=0, preset_index=0)
        fusion_actions[3] = layerwise_action.LayerwiseDecodedAction(
            1, dict(fusion_actions[3].k_by_block), 0,
        )
        communication_actions = _layerwise_actions(
            block4_fusion=0, preset_index=0,
        )
        low = _layerwise_actions(block4_fusion=0, preset_index=2)[3]
        communication_actions[3] = low

        baseline = layerwise_action.compute_variable_cost(baseline_actions)
        fusion = layerwise_action.compute_variable_cost(fusion_actions)
        communication = layerwise_action.compute_variable_cost(
            communication_actions,
        )

        self.assertEqual(baseline.ppo_resource_score, 0.0)
        self.assertAlmostEqual(fusion.compute_saving, 1.0 / 12.0)
        self.assertEqual(fusion.communication_saving, 0.0)
        self.assertEqual(communication.compute_saving, 0.0)
        self.assertAlmostEqual(communication.communication_saving, 1.0 / 12.0)
        self.assertEqual(fusion.robust_floor, 0.0)
        self.assertEqual(communication.robust_floor, 0.0)
        self.assertAlmostEqual(
            fusion.ppo_resource_score,
            communication.ppo_resource_score,
        )

    def test_equal_weighted_sum_rewards_both_resource_axes(self):
        compute_only = layerwise_action.compute_variable_cost(
            _layerwise_actions(block4_fusion=1, preset_index=0)
        )
        communication_only = layerwise_action.compute_variable_cost(
            _layerwise_actions(block4_fusion=0, preset_index=2)
        )
        balanced = layerwise_action.compute_variable_cost(
            _layerwise_actions(block4_fusion=1, preset_index=2)
        )

        self.assertEqual(compute_only.ppo_resource_score, 0.5)
        self.assertEqual(communication_only.ppo_resource_score, 0.5)
        self.assertEqual(balanced.compute_saving, 1.0)
        self.assertEqual(balanced.communication_saving, 1.0)
        self.assertEqual(balanced.ppo_resource_score, 1.0)

    def test_preset_utility_not_individual_k_bits_drives_cost(self):
        high = layerwise_action.compute_variable_cost(
            _layerwise_actions(preset_index=0),
        )
        medium = layerwise_action.compute_variable_cost(
            _layerwise_actions(preset_index=1),
        )
        low = layerwise_action.compute_variable_cost(
            _layerwise_actions(preset_index=2),
        )
        self.assertEqual(high.communication_saving, 0.0)
        self.assertEqual(medium.communication_saving, 0.5)
        self.assertEqual(low.communication_saving, 1.0)

    def test_fixed_block2_and_block5_fusion_are_not_variable_cost_inputs(self):
        with self.assertRaises(TypeError):
            layerwise_action.LayerwiseDecodedAction(
                block4_fusion=0,
                k_by_block={2: 10, 3: 10, 4: 12, 5: 11},
                block2_fusion=1,
            )
        self.assertEqual(
            set(layerwise_action.LayerwiseDecodedAction.__dataclass_fields__),
            {"block4_fusion", "k_by_block", "precision_preset_index"},
        )


def _baseline():
    return rwd.BaselineCostStats(
        total_bits_sum=200, total_fusion_count=0, avg_k=13.0,
        loss_mean=0.3, loss_std=0.01,
        metric1_mean=0.85, metric2_mean=0.85,
        metric1_std=0.01, metric2_std=0.01,
        typical_bits_drop=50, typical_fusion_count=1, typical_k_drop=2,
    )


def _weights():


    return rwd.RewardWeights(baseline_metric1=0.85, baseline_metric2=0.85,
                             reward_design="tiered")


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

        choices = [
            _bc(2, 0, 1, 13),
            _bc(5, 0, 1, 13),
            _bc(1, 0, 0, 13),
            _bc(4, 0, 0, 13),
        ]
        res = fusion_cost.compute_fusion_cost_saving(choices, fusion_w=FW, trunc_w=TW)
        self.assertEqual(res.cost_norm, 0.0)
        self.assertEqual(res.cost_rank, 0.0)

        self.assertAlmostEqual(res.max_actual, 390.0)

    def test_full_saving_normalizes_to_one(self):

        choices = [
            _bc(2, 1, 1, 6),
            _bc(5, 1, 1, 6),
            _bc(1, 0, 0, 6),
            _bc(4, 0, 0, 6),
        ]
        res = fusion_cost.compute_fusion_cost_saving(choices, fusion_w=FW, trunc_w=TW)
        self.assertAlmostEqual(res.cost_norm, 1.0)
        self.assertAlmostEqual(res.cost_rank, 390.0)

    def test_block1_block4_fusion_inert(self):


        choices = [_bc(1, 99, 0, 6), _bc(4, 99, 0, 6)]
        res = fusion_cost.compute_fusion_cost_saving(choices, fusion_w=FW, trunc_w=TW)
        self.assertAlmostEqual(res.max_actual, 100.0)
        self.assertAlmostEqual(res.cost_rank, 100.0)
        for pb in res.per_block:
            self.assertEqual(pb["fusion_contrib"], 0.0)
            self.assertEqual(pb["fusion_saving"], 0.0)

    def test_trunc_saving_linear(self):

        choices = [_bc(2, 0, 1, 11)]
        res = fusion_cost.compute_fusion_cost_saving(choices, fusion_w=FW, trunc_w=TW)
        trunc_saving = (13.0 - 11.0) / 7.0
        self.assertAlmostEqual(res.cost_rank, 50.0 * trunc_saving)
        self.assertAlmostEqual(res.max_actual, 200.0)
        self.assertAlmostEqual(res.cost_norm, 50.0 * trunc_saving / 200.0)

    def test_rank_monotonic_in_fusion(self):
        base = [_bc(2, 0, 1, 13)]
        more = [_bc(2, 1, 1, 13)]
        r0 = fusion_cost.compute_fusion_cost_saving(base, fusion_w=FW, trunc_w=TW)
        r1 = fusion_cost.compute_fusion_cost_saving(more, fusion_w=FW, trunc_w=TW)
        self.assertGreater(r1.cost_rank, r0.cost_rank)
        self.assertAlmostEqual(r1.cost_rank - r0.cost_rank, 150.0)

    def test_precomputed_max_actual_used(self):
        choices = [_bc(2, 1, 1, 6)]
        res = fusion_cost.compute_fusion_cost_saving(
            choices, fusion_w=FW, trunc_w=TW, max_actual=400.0
        )

        self.assertAlmostEqual(res.cost_norm, 0.5)
        self.assertAlmostEqual(res.max_actual, 400.0)

    def test_default_max_actual_reuses_single_pass_accumulators(self):
        choices = [_bc(2, 1, 1, 8), _bc(5, 0, 1, 13), _bc(4, 0, 0, 8)]
        original = fusion_cost.max_actual_for_choices

        def fail_second_pass(*_args, **_kwargs):
            raise AssertionError("compute_fusion_cost_saving should not rescan choices for max_actual")

        fusion_cost.max_actual_for_choices = fail_second_pass
        try:
            res = fusion_cost.compute_fusion_cost_saving(choices, fusion_w=FW, trunc_w=TW)
        finally:
            fusion_cost.max_actual_for_choices = original


        self.assertAlmostEqual(res.max_actual, 150.0 + 40.0 + 3 * 50.0)


class NearMissGradedTierTest(unittest.TestCase):
    """ADR-012: graded near-miss tier replaces the P1 cliff near the threshold.

    2nd-60k forensics: ALL 1226 on-policy fusion P1s were borderline
    (m1 in [0.833, 0.858], zero catastrophic) — each ate the full -46 cliff,
    making expected fusion advantage ~-3.8 despite a +0.1 true P3 advantage.
    The graded band keeps priority=1 (selection/rank unchanged) but the PPO
    scalar slopes from cap(35) at deficit->0 to floor(15) at deficit=band,
    then falls back to the old cliff beyond the band.
    """

    BASE_M1 = 0.8672
    THR = 0.858

    def _reward(self, m1):


        w = rwd.RewardWeights(
            baseline_metric1=self.BASE_M1, baseline_metric2=self.BASE_M1,
            acc_barrier_enabled=False, reward_design="tiered",
        )
        base = rwd.BaselineCostStats(
            total_bits_sum=1000, total_fusion_count=0, avg_k=13.0,
            loss_mean=0.34, loss_std=0.002, metric1_mean=self.BASE_M1,
            metric2_mean=self.BASE_M1, metric1_std=0.001, metric2_std=0.001,
        )

        class _Opt:
            any_invalid = False
            total_bits_sum = 1000
            total_fusion_count = 0

        m = rwd.EpisodeMetrics(
            loss_mean=0.34, loss_std=0.002, metric1_mean=m1, metric2_mean=m1,
            metric1_std=0.001, metric2_std=0.001,
        )
        return rwd.compute_reward(
            m, _Opt(), action_avg_k=13.0, baseline=base, weights=w,
            acc_threshold=self.THR, acc_threshold_m2=self.THR, stab_threshold=0.05,
            external_cost_score=0.0, external_cost_rank=0.0,
        )

    def test_pass_unaffected(self):
        b = self._reward(self.BASE_M1)
        self.assertEqual(b.priority, 3)
        self.assertFalse(b.near_miss)
        self.assertGreaterEqual(b.reward, 40.0)

    def test_tiny_miss_graded_not_cliff(self):
        b = self._reward(0.8570)
        self.assertEqual(b.priority, 1)
        self.assertTrue(b.near_miss)
        self.assertGreater(b.reward, 25.0)
        self.assertLess(b.reward, 40.0)

    def test_grading_monotonic_in_deficit(self):
        rewards = [self._reward(m1).reward for m1 in (0.857, 0.854, 0.851, 0.849)]
        self.assertEqual(rewards, sorted(rewards, reverse=True))

    def test_beyond_band_keeps_cliff(self):
        b = self._reward(0.830)
        self.assertEqual(b.priority, 1)
        self.assertFalse(b.near_miss)
        self.assertLess(b.reward, 0.0)

    def test_catastrophic_keeps_cliff(self):
        b = self._reward(0.32)
        self.assertFalse(b.near_miss)
        self.assertLess(b.reward, -4.0)

    def test_invalid_never_near_miss(self):
        w = rwd.RewardWeights(
            baseline_metric1=self.BASE_M1, baseline_metric2=self.BASE_M1,
            acc_barrier_enabled=False, reward_design="tiered",
        )
        base = rwd.BaselineCostStats(
            total_bits_sum=1000, total_fusion_count=0, avg_k=13.0,
            loss_mean=0.34, loss_std=0.002, metric1_mean=self.BASE_M1,
            metric2_mean=self.BASE_M1, metric1_std=0.001, metric2_std=0.001,
        )

        class _Opt:
            any_invalid = True
            total_bits_sum = 1000
            total_fusion_count = 0

        m = rwd.EpisodeMetrics(
            loss_mean=0.34, loss_std=0.002, metric1_mean=0.857, metric2_mean=0.857,
            metric1_std=0.001, metric2_std=0.001,
        )
        b = rwd.compute_reward(
            m, _Opt(), action_avg_k=13.0, baseline=base, weights=w,
            acc_threshold=self.THR, acc_threshold_m2=self.THR, stab_threshold=0.05,
        )
        self.assertFalse(b.near_miss)
        self.assertLess(b.reward, 0.0)

    def test_near_miss_below_every_p3(self):

        nm = self._reward(self.THR - 1e-5)
        p3_floor = self._reward(self.THR + 1e-5)
        self.assertTrue(nm.near_miss)
        self.assertLess(nm.reward, p3_floor.reward)


class BudgetSplitComponentsTest(unittest.TestCase):
    """ADR-011: fusion / truncation components normalized over their OWN maxima.

    The 60k fusion run collapsed to fusion=0 partly because the shared
    normalization let the K pot (47 x 50) dilute one block5 fusion to +0.029
    reward. The split components let the env budget them separately.
    """

    def test_components_zero_at_baseline(self):
        choices = [_bc(2, 0, 1, 13), _bc(5, 0, 1, 13), _bc(4, 0, 0, 13)]
        res = fusion_cost.compute_fusion_cost_saving(choices, fusion_w=FW, trunc_w=TW)
        self.assertEqual(res.fusion_norm, 0.0)
        self.assertEqual(res.trunc_norm, 0.0)

        self.assertAlmostEqual(res.fusion_max_actual, 190.0)
        self.assertAlmostEqual(res.trunc_max_actual, 150.0)

    def test_k6_is_full_truncation_saving_and_k13_is_zero(self):
        low = [_bc(2, 0, 1, 6), _bc(5, 0, 1, 6)]
        baseline = [_bc(2, 0, 1, 13), _bc(5, 0, 1, 13)]
        low_result = fusion_cost.compute_fusion_cost_saving(
            low, fusion_w=FW, trunc_w=TW,
        )
        baseline_result = fusion_cost.compute_fusion_cost_saving(
            baseline, fusion_w=FW, trunc_w=TW,
        )
        self.assertAlmostEqual(low_result.trunc_norm, 1.0)
        self.assertAlmostEqual(baseline_result.trunc_norm, 0.0)

    def test_fusion_only_moves_fusion_norm(self):
        choices = [_bc(2, 1, 1, 13), _bc(5, 0, 1, 13)]
        res = fusion_cost.compute_fusion_cost_saving(choices, fusion_w=FW, trunc_w=TW)
        self.assertAlmostEqual(res.fusion_actual, 150.0)
        self.assertAlmostEqual(res.fusion_norm, 150.0 / 190.0)
        self.assertEqual(res.trunc_norm, 0.0)

    def test_k_only_moves_trunc_norm(self):
        choices = [_bc(2, 0, 1, 8), _bc(5, 0, 1, 13)]
        res = fusion_cost.compute_fusion_cost_saving(choices, fusion_w=FW, trunc_w=TW)
        self.assertEqual(res.fusion_norm, 0.0)
        self.assertAlmostEqual(res.trunc_actual, 50.0 * (13.0 - 8.0) / 7.0)
        self.assertAlmostEqual(res.trunc_norm, (13.0 - 8.0) / 7.0 / 2.0)

    def test_both_full_saturate_to_one(self):
        choices = [_bc(2, 1, 1, 6), _bc(5, 1, 1, 6)]
        res = fusion_cost.compute_fusion_cost_saving(choices, fusion_w=FW, trunc_w=TW)
        self.assertAlmostEqual(res.fusion_norm, 1.0)
        self.assertAlmostEqual(res.trunc_norm, 1.0)

    def test_fusion_degenerate_schedule_has_zero_fusion_max(self):


        choices = [_bc(1, 0, 0, 6), _bc(4, 0, 0, 6)]
        res = fusion_cost.compute_fusion_cost_saving(choices, fusion_w=FW, trunc_w=TW)
        self.assertEqual(res.fusion_max_actual, 0.0)
        self.assertEqual(res.fusion_norm, 0.0)
        self.assertAlmostEqual(res.trunc_norm, 1.0)

    def test_split_budget_marginal_fusion_visible(self):


        baseline = (
            [_bc(2, 0, 1, 13) for _ in range(12)]
            + [_bc(4, 0, 1, 13) for _ in range(12)]
            + [_bc(5, 0, 1, 13) for _ in range(12)]
            + [_bc(1, 0, 0, 13) for _ in range(11)]
        )
        one_b5 = list(baseline)
        one_b5[24] = _bc(5, 1, 1, 13)
        one_b2 = list(baseline)
        one_b2[0] = _bc(2, 1, 1, 13)
        budget = 4.5
        frac = rwd.FUSION_COST_BUDGET_FRACTION
        self.assertAlmostEqual(frac, 0.5)
        def score(ch):
            r = fusion_cost.compute_fusion_cost_saving(ch, fusion_w=FW, trunc_w=TW)
            return r.fusion_norm * budget * frac + r.trunc_norm * budget * (1 - frac)
        self.assertAlmostEqual(score(baseline), 0.0)

        self.assertAlmostEqual(score(one_b5), 40.0 / 3840.0 * 2.25)
        self.assertGreaterEqual(score(one_b5), 0.02)
        self.assertAlmostEqual(score(one_b2), 150.0 / 3840.0 * 2.25)
        self.assertGreater(score(one_b2), score(one_b5))

    def test_fusion_and_truncation_have_equal_budget(self):
        budget = 4.5
        frac = rwd.FUSION_COST_BUDGET_FRACTION
        self.assertAlmostEqual(frac, 0.5)

        full_fusion = [_bc(2, 1, 1, 13), _bc(5, 1, 1, 13)]
        full_trunc = [_bc(2, 0, 1, 6), _bc(5, 0, 1, 6)]

        def score(ch):
            r = fusion_cost.compute_fusion_cost_saving(ch, fusion_w=FW, trunc_w=TW)
            return r.fusion_norm * budget * frac + r.trunc_norm * budget * (1 - frac)

        self.assertAlmostEqual(score(full_fusion), budget / 2.0)
        self.assertAlmostEqual(score(full_trunc), budget / 2.0)


class ExternalCostThreadingTest(unittest.TestCase):
    def test_weight_constants_match_spec(self):
        self.assertEqual(rwd.FUSION_COST_W, {1: 80.0, 2: 150.0, 4: 130.0, 5: 40.0})
        self.assertEqual(rwd.TRUNC_COST_W, 50.0)
        self.assertEqual((rwd.K_MAX_BITS, rwd.K_MIN_BITS), (13, 6))

    def test_p3_uses_external_cost(self):

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
        self.assertGreater(bd.reward, 40.0)

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
        self.assertAlmostEqual(bd.cost_score, float(w.p3_cost_budget))

    def test_p1_ignores_external_cost(self):

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

        metrics = rwd.EpisodeMetrics(
            loss_mean=0.3, loss_std=0.0,
            metric1_mean=0.85, metric2_mean=0.85,
            metric1_std=0.0, metric2_std=0.0,
        )
        bd = rwd.compute_reward(
            metrics, _Sig(), action_avg_k=13.0, baseline=_baseline(), weights=_weights(),
        )
        self.assertEqual(bd.priority, 3)

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
        choices = self._schedule_choices(fusion_count_of=lambda mf: mf, k_value=6)
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


        probe = {"valid": True, "fusion_count": 0, "total_bits": 96}
        self.assertTrue(fusion_enum._level_breaks_pin(probe, base_key))


        self.assertEqual(int(probe["fusion_count"]), base_key[0])

    def test_truly_inert_slot_is_pinnable(self):

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
        cfgs = [EC((9, 9, 9), 0, 100, 1.0, (-1,), {})]

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

        fc1_batch = [o for o in batch if o["fusion_count"] == 1]
        self.assertEqual(len(fc1_batch), 3)

    def test_reducer_counts_and_shrinks(self):
        _baseline, cfgs = self._make_configs()
        r = fusion_enum._MinNoiseReducer()
        for ec in cfgs:
            r.add(ec)
        self.assertEqual(r.num_valid, len(cfgs))
        self.assertLess(len(r.results()), len(cfgs))


class UniformStep1DecodeTest(unittest.TestCase):
    """2026-06-11 rule (supersedes the 2026-06-04 hybrid 2/1 sweep): all SF kinds
    use a UNIFORM step-1 downward sweep from the baseline SF, 15 levels max, no
    extra floor (noise-table min 10 via _snap_to_table); N forced to 16384. The
    fusion builder skips duplicate-value (snap-floored) levels pre-enumeration —
    a result-equivalent acceleration only."""

    def test_all_sf_kinds_are_15_levels(self):
        self.assertEqual(_asp.LEVELS_F, 15)
        self.assertEqual(_asp.LEVELS_W, 15)
        self.assertEqual(_asp.LEVELS_MS, 15)
        self.assertEqual(_asp.LEVELS_R, 15)
        self.assertFalse(hasattr(_asp, "MIN_SF_FLOOR"))

    def test_step1_sweep_full_integer_range(self):

        got = [_asp.sf_from(i, 30, 15) for i in range(14, -1, -1)]
        self.assertEqual(got, list(range(30, 15, -1)))

    def test_rescale_idx0_none_max_idx_is_baseline(self):
        self.assertIsNone(_asp._rescale_sf_from_index(0, 30))
        self.assertEqual(_asp._rescale_sf_from_index(14, 30), 30)
        self.assertEqual(_asp._rescale_sf_from_index(1, 30), 17)

        self.assertEqual(_asp._rescale_sf_from_index(_asp.LEVELS_R - 1, 27), 27)

    def test_field_level_values_rescale_and_fresh(self):
        r = _asp._field_level_values(kind="R", levels=_asp.LEVELS_R, max_sf=30, N=16384)
        self.assertEqual(len(r), 15)
        self.assertIsNone(r[0])
        self.assertEqual([int(v) for v in r[1:]], list(range(17, 31)))
        f = _asp._field_level_values(kind="F", levels=_asp.LEVELS_F, max_sf=30, N=16384)
        self.assertEqual([int(v) for v in f], list(range(16, 31)))

    def test_low_baseline_sf_snaps_to_floor(self):

        vlow = _asp._field_level_values(kind="F", levels=_asp.LEVELS_F, max_sf=14, N=16384)
        self.assertTrue(all(int(v) >= 10 for v in vlow))
        self.assertEqual(int(vlow[-1]), 14)

    def test_distinct_level_indices_skip_duplicate_values_only(self):

        self.assertEqual(
            _asp.distinct_sf_level_indices(kind="F", levels=15, max_sf=30, N=16384),
            list(range(15)),
        )


        d20 = _asp.distinct_sf_level_indices(kind="F", levels=15, max_sf=20, N=16384)
        self.assertEqual(d20, [0] + list(range(5, 15)))
        vals20 = _asp._field_level_values(kind="F", levels=15, max_sf=20, N=16384)
        self.assertEqual(sorted(int(vals20[i]) for i in d20), list(range(10, 21)))

        self.assertEqual(
            _asp.distinct_sf_level_indices(kind="F", levels=15, max_sf=14, N=16384),
            [0, 11, 12, 13, 14],
        )

        dr = _asp.distinct_sf_level_indices(kind="R", levels=15, max_sf=30, N=16384)
        self.assertEqual(dr, list(range(1, 15)))

    def test_block_default_N_forced_to_16384(self):
        for b in (1, 2, 3, 4, 5):
            self.assertEqual(_asp._block_default_N(b, gelu_degree=4, attn_degree=4), 16384)


if __name__ == "__main__":
    unittest.main()
