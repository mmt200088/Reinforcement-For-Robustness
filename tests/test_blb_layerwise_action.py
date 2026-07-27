"""Torch-free contract tests for the layerwise Stage-2 codec."""

from __future__ import annotations

from dataclasses import replace
import pathlib
import sys
import unittest
from unittest import mock

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_BLB_DIR = _REPO_ROOT / "blb_stage2_rl"
for _path in (str(_REPO_ROOT), str(_BLB_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import fusion_count_map as fcm
import layerwise_action as layerwise

try:
    from action_space import make_all_max_action_vector as _make_all_max_action_vector
except ImportError:
    _make_all_max_action_vector = None


_BLOCK_SLOT_COUNTS = {1: 9, 2: 23, 3: 8, 4: 17, 5: 16}
_BLOCK_OFFSETS = {1: 0, 2: 9, 3: 32, 4: 40, 5: 57}
_LAYER_WIDTH = 73


def _legacy_all_max(num_layers: int = 12) -> np.ndarray:
    """Use action_space's baseline when torch is available, otherwise mirror it."""
    if _make_all_max_action_vector is not None:
        return _make_all_max_action_vector(num_layers)
    vector = np.full(num_layers * _LAYER_WIDTH + 1, 14, dtype=int)
    k_baseline_idx = layerwise.K_LEVELS.index(13)
    for layer_idx in range(num_layers):
        base = layer_idx * _LAYER_WIDTH
        for block_idx, count in _BLOCK_SLOT_COUNTS.items():
            vector[base + _BLOCK_OFFSETS[block_idx] + count - 1] = k_baseline_idx
    vector[-1] = 4
    return vector


def _option(graph, fusion_count: int):
    matches = [option for option in graph.options if option.fusion_count == fusion_count]
    assert len(matches) == 1
    return matches[0]


def _non_contiguous_fusion_map() -> fcm.FusionCountMap:
    def option(option_id: int, fusion_count: int, slots: int, marker: int, *, boosted: bool = False):
        payload = {
            "option_id": option_id,
            "fusion_count": fusion_count,
            "tie_index": 0,
            "total_variance": 1.0,
            "total_bits": 100,
            "slots": {},
            "action_indices": [marker] * slots,
        }
        if boosted:
            payload["boosted"] = True
            payload["explicit_field_values"] = {"field": marker, "output_truncation_k": 13}
        return payload

    def graph(graph_key: str, slots: int, options: list[dict]):
        return {
            "graph_key": graph_key,
            "k_slot_index": slots - 1,
            "block_num_slots": slots,
            "options": options,
        }

    return fcm.FusionCountMap.from_payload({
        "profile": "mrpc",
        "graphs": {
            "block1_mrpc": graph("block1_mrpc", 9, [
                option(0, 2, 9, 14), option(10, 0, 9, 12),
            ]),
            "block2_mrpc": graph("block2_mrpc", 23, [
                option(0, 0, 23, 14), option(88, 1, 23, 9, boosted=True),
            ]),
            "block4": graph("block4", 17, [
                option(0, 2, 17, 14), option(10, 0, 17, 12), option(88, 1, 17, 8, boosted=True),
            ]),
            "block5_n4": graph("block5_n4", 16, [
                option(0, 0, 16, 14), option(88, 1, 16, 7, boosted=True),
            ]),
        },
    })


class LayerwiseScheduleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.map = fcm.FusionCountMap.load("mrpc")
        cls.specs = layerwise.layerwise_schedule(12, cls.map)

    def test_has_canonical_twelve_step_geometry(self):
        self.assertEqual(layerwise.LAYERWISE_SLOT_NAMES, (
            "block4_fusion", "truncation_precision",
        ))
        self.assertEqual(len(self.specs), 12)
        self.assertEqual([spec.step_idx for spec in self.specs], list(range(12)))
        self.assertEqual([spec.layer_idx for spec in self.specs], list(range(12)))
        self.assertEqual(
            self.specs[0].slot_dims,
            (2, 3),
        )
        self.assertTrue(
            all(
                spec.slot_mask == (True, True)
                for spec in self.specs
            )
        )
        self.assertEqual([spec.terminal for spec in self.specs], [False] * 11 + [True])
        self.assertEqual(dict(self.specs[0].graph_keys_by_block)[5], "block5_n4")

    def test_schedule_rejects_invalid_layers_and_missing_graph(self):
        with self.assertRaises(ValueError):
            layerwise.layerwise_schedule(0, self.map)
        missing = fcm.FusionCountMap.load("mrpc")
        del missing.graphs["block4"]
        with self.assertRaisesRegex(KeyError, "block4"):
            layerwise.layerwise_schedule(12, missing)


class LayerwiseApplicationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.map = fcm.FusionCountMap.load("mrpc")
        cls.specs = layerwise.layerwise_schedule(12, cls.map, gelu_degrees=[4] * 12)

    def test_each_layer_splices_maps_and_preserves_block3_scaling_factors(self):
        full = _legacy_all_max()
        baseline = full.copy()
        expected_k_values = (9, 8, 8, 10, 9)
        k_indices = [
            layerwise.K_LEVELS.index(value) for value in expected_k_values
        ]

        for spec in self.specs:
            action = [spec.layer_idx % 2, 1]
            result = layerwise.apply_layer_action(full, action, spec, self.map)
            full = result.full_vector
            layer_base = spec.layer_idx * _LAYER_WIDTH

            self.assertEqual(result.decoded.block4_fusion, spec.layer_idx % 2)
            expected_blocks = {1, 2, 3, 4, 5}
            self.assertEqual(set(result.decoded.k_by_block), expected_blocks)
            for block_idx, expected_k in zip(
                    (1, 2, 3, 4, 5), expected_k_values,
            ):
                if block_idx in expected_blocks:
                    self.assertEqual(result.decoded.k_by_block[block_idx], expected_k)

            block1_start = layer_base + _BLOCK_OFFSETS[1]
            np.testing.assert_array_equal(
                full[block1_start:block1_start + _BLOCK_SLOT_COUNTS[1] - 1],
                baseline[block1_start:block1_start + _BLOCK_SLOT_COUNTS[1] - 1],
            )
            self.assertEqual(
                int(full[block1_start + _BLOCK_SLOT_COUNTS[1] - 1]),
                k_indices[0],
            )
            self.assertNotIn(1, result.fusion_option_ids)

            for block_idx, graph_key, count in ((2, "block2_mrpc", 1), (4, "block4", spec.layer_idx % 2), (5, "block5_n4", 1)):
                option = _option(self.map.graphs[graph_key], count)
                self.assertEqual(result.fusion_option_ids[block_idx], option.option_id)
                start = layer_base + _BLOCK_OFFSETS[block_idx]
                expected = self.map.expand(graph_key, option.option_id, k_indices[block_idx - 1])
                np.testing.assert_array_equal(full[start:start + _BLOCK_SLOT_COUNTS[block_idx]], expected)

            block3_start = layer_base + _BLOCK_OFFSETS[3]
            np.testing.assert_array_equal(full[block3_start:block3_start + 7], baseline[block3_start:block3_start + 7])
            self.assertEqual(full[block3_start + 7], k_indices[2])
            self.assertNotIn(3, result.fusion_option_ids)

            for block_idx in (2, 4, 5):
                option = next(o for o in self.map.graphs[dict(spec.graph_keys_by_block)[block_idx]].options if o.option_id == result.fusion_option_ids[block_idx])
                if option.boosted:
                    self.assertEqual(
                        result.boosted_field_values_by_block[block_idx]["output_truncation_k"],
                        result.decoded.k_by_block[block_idx],
                    )

    def test_decodes_shared_k6_k13_domain_and_preserves_k_indices(self):
        self.assertEqual(layerwise.K_LEVELS[:6], (8, 9, 11, 13, 10, 12))
        self.assertEqual(layerwise.K_LEVELS[6:], (6, 7))

        action = [0, 2]
        result = layerwise.apply_layer_action(
            _legacy_all_max(), action, self.specs[1], self.map,
        )

        self.assertEqual(
            result.decoded.k_by_block,
            {1: 7, 2: 6, 3: 6, 4: 8, 5: 7},
        )
        layer_base = _LAYER_WIDTH
        for block_idx, expected_k in result.decoded.k_by_block.items():
            k_offset = (
                layer_base
                + _BLOCK_OFFSETS[block_idx]
                + _BLOCK_SLOT_COUNTS[block_idx]
                - 1
            )
            self.assertEqual(
                int(result.full_vector[k_offset]),
                layerwise.K_LEVELS.index(expected_k),
            )
        self.assertEqual(set(result.boosted_field_values_by_block), {2, 5})
        self.assertEqual(
            result.boosted_field_values_by_block[2]["output_truncation_k"],
            6,
        )
        for block_idx, values in result.boosted_field_values_by_block.items():
            self.assertEqual(
                values["output_truncation_k"],
                result.decoded.k_by_block[block_idx],
            )

    def test_resolves_actual_non_contiguous_option_ids(self):
        fusion_map = _non_contiguous_fusion_map()
        with self.assertRaises(IndexError):
            fusion_map.expand("block2_mrpc", 88, 0)
        spec = layerwise.layerwise_schedule(12, fusion_map)[1]
        result = layerwise.apply_layer_action(
            _legacy_all_max(), [1, 1], spec, fusion_map,
        )
        self.assertEqual(result.fusion_option_ids, {2: 88, 4: 88, 5: 88})
        layer_base = _LAYER_WIDTH
        self.assertEqual(int(result.full_vector[layer_base + _BLOCK_OFFSETS[2]]), 9)
        self.assertEqual(int(result.full_vector[layer_base + _BLOCK_OFFSETS[4]]), 8)
        self.assertEqual(result.boosted_field_values_by_block[2]["output_truncation_k"], 8)


class VariableCostTest(unittest.TestCase):
    def _actions(self, fusion: int, preset_index: int):
        preset = layerwise.precision_preset(preset_index)
        return [
            layerwise.LayerwiseDecodedAction(
                fusion,
                {
                    block_idx: preset.k_by_block[block_idx - 1]
                    for block_idx in range(1, 6)
                },
                preset_index,
            )
            for _layer_idx in range(12)
        ]

    def test_independent_resource_axes_and_monotonicity(self):
        baseline = layerwise.compute_variable_cost(self._actions(0, 0))
        both = layerwise.compute_variable_cost(self._actions(1, 2))
        fusion_only = layerwise.compute_variable_cost(self._actions(1, 0))
        communication_only = layerwise.compute_variable_cost(self._actions(0, 2))

        self.assertEqual(
            (
                baseline.compute_saving,
                baseline.communication_saving,
                baseline.ppo_resource_score,
            ),
            (0.0, 0.0, 0.0),
        )
        self.assertEqual(
            (both.compute_saving, both.communication_saving, both.ppo_resource_score),
            (1.0, 1.0, 1.0),
        )
        self.assertEqual((fusion_only.compute_saving, fusion_only.communication_saving), (1.0, 0.0))
        self.assertEqual((communication_only.compute_saving, communication_only.communication_saving), (0.0, 1.0))
        self.assertEqual((baseline.fusion_count, baseline.removed_k_bits), (0, 132))
        self.assertEqual((both.fusion_count, both.removed_k_bits), (12, 372))
        self.assertEqual(len(both.layer_resource_rewards), 12)
        self.assertEqual(len(both.slot_resource_rewards), 12)
        self.assertTrue(all(len(row) == 2 for row in both.slot_resource_rewards))
        self.assertAlmostEqual(sum(both.layer_resource_rewards), both.ppo_resource_score)
        for layer_reward, slot_rewards in zip(
                both.layer_resource_rewards, both.slot_resource_rewards,
        ):
            self.assertAlmostEqual(sum(slot_rewards), layer_reward)
        self.assertGreater(
            fusion_only.ppo_resource_score, baseline.ppo_resource_score,
        )
        self.assertGreater(
            communication_only.ppo_resource_score, baseline.ppo_resource_score,
        )
        self.assertLess(fusion_only.ppo_resource_score, both.ppo_resource_score)
        self.assertLess(
            communication_only.ppo_resource_score, both.ppo_resource_score,
        )

    def test_one_axis_cannot_modify_the_other_axis(self):
        baseline_actions = self._actions(0, 0)
        baseline = layerwise.compute_variable_cost(baseline_actions)

        fused_actions = self._actions(0, 0)
        fused_actions[7] = layerwise.LayerwiseDecodedAction(
            1, dict(fused_actions[7].k_by_block), 0,
        )
        fused = layerwise.compute_variable_cost(fused_actions)

        lower_k_actions = self._actions(0, 0)
        medium = layerwise.precision_preset(1)
        lower_k_actions[7] = layerwise.LayerwiseDecodedAction(
            0,
            {
                block_idx: medium.k_by_block[block_idx - 1]
                for block_idx in range(1, 6)
            },
            1,
        )
        lower_k = layerwise.compute_variable_cost(lower_k_actions)

        self.assertAlmostEqual(fused.compute_saving - baseline.compute_saving, 1.0 / 12.0)
        self.assertEqual(fused.communication_saving, baseline.communication_saving)
        self.assertEqual(lower_k.compute_saving, baseline.compute_saving)
        self.assertAlmostEqual(
            lower_k.communication_saving - baseline.communication_saving,
            0.5 / 12.0,
        )

    def test_separable_slot_credits_cover_each_resource_family_once(self):
        result = layerwise.compute_variable_cost(self._actions(1, 1))
        self.assertEqual(result.fusion_count, 12)
        self.assertEqual(result.removed_k_bits, 252)
        self.assertAlmostEqual(result.compute_saving, 1.0)
        self.assertAlmostEqual(result.communication_saving, 0.5)
        self.assertAlmostEqual(
            result.compute_shapley_credit + result.communication_shapley_credit,
            result.ppo_resource_score,
        )
        self.assertGreater(result.slot_resource_rewards[0][1], 0.0)
        compute_slot_total = sum(row[0] for row in result.slot_resource_rewards)
        communication_slot_total = sum(
            sum(row[1:]) for row in result.slot_resource_rewards
        )
        self.assertAlmostEqual(compute_slot_total, result.compute_shapley_credit)
        self.assertAlmostEqual(
            communication_slot_total, result.communication_shapley_credit,
        )
        self.assertAlmostEqual(
            sum(map(sum, result.slot_resource_rewards)), result.ppo_resource_score,
        )

    def test_network_weighted_score_is_monotonic_on_each_resource_axis(self):
        for ratio in (0.25, 1.0, 4.0):
            with self.subTest(ratio=ratio):
                previous = -1.0
                for compute in np.linspace(0.0, 1.0, 13):
                    score = layerwise.dual_resource_score(
                        compute, 0.5, ratio,
                    )[2]
                    self.assertGreater(score, previous)
                    previous = score
                previous = -1.0
                for communication in (0.0, 0.5, 1.0):
                    score = layerwise.dual_resource_score(
                        0.5, communication, ratio,
                    )[2]
                    self.assertGreater(score, previous)
                    previous = score

    def test_cost_uses_decoded_k_values_not_category_order(self):
        actions = self._actions(0, 2)
        result = layerwise.compute_variable_cost(actions)
        self.assertEqual(result.communication_saving, 1.0)
        actions[0] = layerwise.LayerwiseDecodedAction(
            0, {1: 7, 2: 6, 3: 6, 4: 8, 5: 8}, 2,
        )
        with self.assertRaises(ValueError):
            layerwise.compute_variable_cost(actions)

    def test_bert_large_uses_all_24_layers_and_dynamic_resource_denominators(self):
        low = layerwise.precision_preset(2)
        actions = [
            layerwise.LayerwiseDecodedAction(
                1,
                {
                    block_idx: low.k_by_block[block_idx - 1]
                    for block_idx in range(1, 6)
                },
                2,
            )
            for _layer_idx in range(24)
        ]

        result = layerwise.compute_variable_cost(actions)

        self.assertEqual(result.fusion_count, 24)
        self.assertEqual(result.removed_k_bits, 31 * 24)
        self.assertEqual(result.compute_saving, 1.0)
        self.assertEqual(result.communication_saving, 1.0)
        self.assertEqual(result.ppo_resource_score, 1.0)
        self.assertEqual(len(result.layer_resource_rewards), 24)
        self.assertEqual(len(result.slot_resource_rewards), 24)

        matrix = [[1, 2] for _ in range(24)]
        decoded = layerwise.compute_variable_cost_from_action_matrix(matrix)
        self.assertEqual(decoded.fusion_count, 24)
        self.assertEqual(len(decoded.layer_resource_rewards), 24)

    def test_layer_count_is_bound_into_identity_and_resource_denominators(self):
        self.assertEqual(
            layerwise.layerwise_action_space_version(12),
            "stage2_layerwise_12x2_hml_v3",
        )
        self.assertEqual(
            layerwise.layerwise_action_space_version(24),
            "stage2_layerwise_24x2_hml_v3",
        )
        self.assertEqual(layerwise.max_compute_saving_units(12), 12.0)
        self.assertEqual(layerwise.max_compute_saving_units(24), 24.0)
        self.assertEqual(layerwise.max_communication_saving_units(12), 12.0)
        self.assertEqual(layerwise.max_communication_saving_units(24), 24.0)

        for helper in (
            layerwise.layerwise_action_space_version,
            layerwise.max_compute_saving_units,
            layerwise.max_communication_saving_units,
        ):
            with self.subTest(helper=helper.__name__):
                with self.assertRaises(ValueError):
                    helper(0)


class BertLargeLayerwiseScheduleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.map = fcm.FusionCountMap.load("mrpc_large")

    def test_24_layer_schedule_does_not_require_a_block1_fusion_map(self):
        gelu = [1, 2] * 12

        specs = layerwise.layerwise_schedule(
            24,
            self.map,
            profile="mrpc_large",
            gelu_degrees=gelu,
        )

        self.assertEqual(len(specs), 24)
        self.assertEqual([spec.layer_idx for spec in specs], list(range(24)))
        self.assertEqual([spec.terminal for spec in specs], [False] * 23 + [True])
        self.assertEqual(dict(specs[0].graph_keys_by_block)[5], "block5_n1")
        self.assertEqual(dict(specs[1].graph_keys_by_block)[5], "block5_n2")

        baseline = _legacy_all_max(24)
        block1_before = baseline[_LAYER_WIDTH:_LAYER_WIDTH + _BLOCK_SLOT_COUNTS[1] - 1].copy()
        applied = layerwise.apply_layer_action(
            baseline,
            [1, 1],
            specs[1],
            self.map,
        )
        np.testing.assert_array_equal(
            applied.full_vector[
                _LAYER_WIDTH:_LAYER_WIDTH + _BLOCK_SLOT_COUNTS[1] - 1
            ],
            block1_before,
        )
        self.assertNotIn(1, applied.fusion_option_ids)


class KLevelsContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.map = fcm.FusionCountMap.load("mrpc")
        cls.spec = layerwise.layerwise_schedule(12, cls.map)[1]

    def test_rejects_unsupported_k_levels_at_every_public_entry_point(self):
        actions = [
            layerwise.LayerwiseDecodedAction(0, {1: 13, 2: 13, 3: 13, 4: 13, 5: 13})
            for _ in range(12)
        ]
        baseline = _legacy_all_max()
        matrix = [[0, 0] for _ in range(12)]
        with mock.patch.object(layerwise, "K_LEVELS", (6, 7, 8, 9, 10, 11, 12, 14)):
            with self.assertRaisesRegex(ValueError, "K_LEVELS"):
                layerwise.layerwise_schedule(12, self.map)
            with self.assertRaisesRegex(ValueError, "K_LEVELS"):
                layerwise.apply_layer_action(
                    baseline, [0, 0], self.spec, self.map,
                )
            with self.assertRaisesRegex(ValueError, "K_LEVELS"):
                layerwise.compute_variable_cost(actions)
            with self.assertRaisesRegex(ValueError, "K_LEVELS"):
                list(layerwise.one_coordinate_neighbors(matrix))

    def test_reordered_supported_k_levels_decode_and_cost_by_value(self):
        reordered = (13, 6, 8, 10, 7, 9, 11, 12)
        self.assertEqual(len(reordered), 8)
        self.assertEqual(set(reordered), set(range(6, 14)))
        with mock.patch.object(layerwise, "K_LEVELS", reordered):
            spec = layerwise.layerwise_schedule(12, self.map)[1]
            action = [0, 0]
            result = layerwise.apply_layer_action(_legacy_all_max(), action, spec, self.map)
            self.assertEqual(
                result.decoded.k_by_block,
                {1: 11, 2: 10, 3: 10, 4: 12, 5: 11},
            )

            low = layerwise.precision_preset(2)
            actions = [
                layerwise.LayerwiseDecodedAction(
                    0,
                    {
                        block_idx: low.k_by_block[block_idx - 1]
                        for block_idx in range(1, 6)
                    },
                    2,
                )
                for _ in range(12)
            ]
            self.assertAlmostEqual(
                layerwise.compute_variable_cost(actions).communication_saving,
                1.0,
            )


class LayerwiseOwnershipTest(unittest.TestCase):
    def test_decoded_action_owns_an_immutable_mapping(self):
        source = {2: 8}
        decoded = layerwise.LayerwiseDecodedAction(0, source)
        source[2] = 13
        self.assertEqual(decoded.k_by_block[2], 8)
        with self.assertRaises(TypeError):
            decoded.k_by_block[2] = 13

    def test_application_owns_read_only_vector_and_frozen_metadata(self):
        source_vector = np.arange(4, dtype=int)
        source_options = {2: 88}
        source_boosted = {2: {"output_truncation_k": 8}}
        application = layerwise.LayerActionApplication(
            source_vector,
            layerwise.LayerwiseDecodedAction(0, {2: 8}),
            source_options,
            source_boosted,
        )
        source_vector[0] = 99
        source_options[2] = 0
        source_boosted[2]["output_truncation_k"] = 13
        self.assertEqual(int(application.full_vector[0]), 0)
        self.assertEqual(application.fusion_option_ids[2], 88)
        self.assertEqual(application.boosted_field_values_by_block[2]["output_truncation_k"], 8)
        with self.assertRaises(ValueError):
            application.full_vector[0] = 99
        with self.assertRaises(TypeError):
            application.fusion_option_ids[2] = 0
        with self.assertRaises(TypeError):
            application.boosted_field_values_by_block[2]["output_truncation_k"] = 13


class LayerwiseNeighborTest(unittest.TestCase):
    def test_neighbors_are_complete_unique_and_non_mutating(self):
        action = [[0, 0] for _ in range(12)]
        original = [row[:] for row in action]
        neighbors = list(layerwise.one_coordinate_neighbors(action))
        self.assertEqual(len(neighbors), 36)
        self.assertEqual(
            len({tuple(value for row in neighbor for value in row) for neighbor in neighbors}),
            36,
        )
        layer_zero_precision_candidates = {
            neighbor[0][1]
            for neighbor in neighbors
            if neighbor[0][1] != action[0][1]
            and all(
                neighbor[row][col] == action[row][col]
                for row in range(12)
                for col in range(2)
                if (row, col) != (0, 1)
            )
        }
        self.assertEqual(layer_zero_precision_candidates, {1, 2})
        self.assertEqual(action, original)
        for neighbor in neighbors:
            changes = [
                (row, col)
                for row in range(12)
                for col in range(2)
                if neighbor[row][col] != action[row][col]
            ]
            self.assertEqual(len(changes), 1)
            self.assertIsNot(neighbor[0], action[0])

    def test_validates_late_coordinates_before_first_yield(self):
        action = [[0, 0] for _ in range(12)]
        action[11][1] = 3
        iterator = layerwise.one_coordinate_neighbors(action)
        with self.assertRaises(ValueError):
            next(iterator)

    def test_layer_zero_precision_preset_installs_block1_k(self):
        fusion_map = fcm.FusionCountMap.load("mrpc")
        spec = layerwise.layerwise_schedule(12, fusion_map)[0]
        k_index = layerwise.K_LEVELS.index(7)
        result = layerwise.apply_layer_action(
            _legacy_all_max(), [0, 2], spec, fusion_map,
        )
        self.assertEqual(result.decoded.k_by_block[1], 7)
        self.assertEqual(
            int(result.full_vector[_BLOCK_OFFSETS[1] + _BLOCK_SLOT_COUNTS[1] - 1]),
            k_index,
        )

        action = [[0, 0] for _ in range(12)]
        neighbors = list(layerwise.one_coordinate_neighbors(action))
        self.assertEqual(len(neighbors), 36)
        self.assertEqual(
            sum(neighbor[0][1] != action[0][1] for neighbor in neighbors),
            2,
        )


class LayerwiseValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.map = fcm.FusionCountMap.load("mrpc")
        cls.spec = layerwise.layerwise_schedule(12, cls.map)[1]

    def test_invalid_action_shapes_and_indices_fail_loudly(self):
        with self.assertRaises(ValueError):
            layerwise.apply_layer_action(_legacy_all_max(), [0], self.spec, self.map)
        with self.assertRaises(ValueError):
            layerwise.apply_layer_action(
                _legacy_all_max(), [2, 0], self.spec, self.map,
            )
        with self.assertRaises(ValueError):
            list(layerwise.one_coordinate_neighbors([]))

    def test_duplicate_fixed_fusion_options_fail_loudly(self):
        duplicate = fcm.FusionCountMap.load("mrpc")
        graph = duplicate.graphs["block2_mrpc"]
        graph.options = graph.options + [replace(
            next(option for option in graph.options if option.fusion_count == 1),
            option_id=99,
        )]
        with self.assertRaisesRegex(ValueError, "exactly one"):
            layerwise.layerwise_schedule(12, duplicate)

    def test_map_structure_validation_fails_loudly(self):
        cases = []

        missing_graph = _non_contiguous_fusion_map()
        del missing_graph.graphs["block4"]
        cases.append(("missing graph", missing_graph, KeyError, "block4"))

        for graph_key, fusion_count in (
            ("block2_mrpc", 1),
            ("block4", 0),
            ("block4", 1),
            ("block5_n4", 1),
        ):
            missing_required = _non_contiguous_fusion_map()
            missing_required.graphs[graph_key].options = [
                option for option in missing_required.graphs[graph_key].options
                if option.fusion_count != fusion_count
            ]
            cases.append((
                f"missing {graph_key} fusion count {fusion_count}",
                missing_required,
                ValueError,
                f"fusion_count={fusion_count}",
            ))

        duplicate_count = _non_contiguous_fusion_map()
        duplicate_count.graphs["block4"].options.append(replace(
            duplicate_count.graphs["block4"].options[-1],
            option_id=99,
        ))
        cases.append(("duplicate required count", duplicate_count, ValueError, "fusion_count=1"))

        duplicate_id = _non_contiguous_fusion_map()
        duplicate_id.graphs["block2_mrpc"].options.append(duplicate_id.graphs["block2_mrpc"].options[-1])
        cases.append(("duplicate option id", duplicate_id, ValueError, "duplicate option_id"))

        malformed_vector = _non_contiguous_fusion_map()
        malformed_vector.graphs["block2_mrpc"].options[0].action_indices.pop()
        cases.append(("malformed action vector", malformed_vector, ValueError, "action_indices"))

        invalid_k_slot = _non_contiguous_fusion_map()
        invalid_k_slot.graphs["block2_mrpc"].k_slot_index = 23
        cases.append(("out of range K slot", invalid_k_slot, ValueError, "K slot"))

        boosted_without_values = _non_contiguous_fusion_map()
        graph = boosted_without_values.graphs["block2_mrpc"]
        graph.options[1] = replace(graph.options[1], explicit_field_values=None)
        cases.append(("boosted option lacks explicit values", boosted_without_values, ValueError, "boosted"))

        for name, fusion_map, error_type, message in cases:
            with self.subTest(name=name), self.assertRaisesRegex(error_type, message):
                layerwise.layerwise_schedule(12, fusion_map)


if __name__ == "__main__":
    unittest.main()
