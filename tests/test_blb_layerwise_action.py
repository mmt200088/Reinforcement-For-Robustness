"""Torch-free contract tests for the 12-step layerwise Stage-2 codec."""

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
            "block4_fusion", "block1_k", "block2_k", "block3_k", "block4_k", "block5_k",
        ))
        self.assertEqual(len(self.specs), 12)
        self.assertEqual([spec.step_idx for spec in self.specs], list(range(12)))
        self.assertEqual([spec.layer_idx for spec in self.specs], list(range(12)))
        self.assertEqual(
            self.specs[0].slot_dims,
            (2, len(layerwise.K_LEVELS), len(layerwise.K_LEVELS), len(layerwise.K_LEVELS), len(layerwise.K_LEVELS), len(layerwise.K_LEVELS)),
        )
        self.assertEqual(self.specs[0].slot_mask, (True, False, True, True, True, True))
        self.assertTrue(all(spec.slot_mask == (True, True, True, True, True, True) for spec in self.specs[1:]))
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
        k_indices = [layerwise.K_LEVELS.index(value) for value in (8, 9, 11, 12, 10)]

        for spec in self.specs:
            action = [spec.layer_idx % 2, *k_indices]
            result = layerwise.apply_layer_action(full, action, spec, self.map)
            full = result.full_vector
            layer_base = spec.layer_idx * _LAYER_WIDTH

            self.assertEqual(result.decoded.block4_fusion, spec.layer_idx % 2)
            expected_blocks = {2, 3, 4, 5} if spec.layer_idx == 0 else {1, 2, 3, 4, 5}
            self.assertEqual(set(result.decoded.k_by_block), expected_blocks)
            for block_idx, expected_k in zip((1, 2, 3, 4, 5), (8, 9, 11, 12, 10)):
                if block_idx in expected_blocks:
                    self.assertEqual(result.decoded.k_by_block[block_idx], expected_k)

            if spec.layer_idx == 0:
                np.testing.assert_array_equal(full[layer_base:layer_base + 9], baseline[layer_base:layer_base + 9])
                self.assertNotIn(1, result.fusion_option_ids)
            else:
                block1 = _option(self.map.graphs["block1_mrpc"], 0)
                self.assertEqual(result.fusion_option_ids[1], block1.option_id)

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

    def test_resolves_actual_non_contiguous_option_ids(self):
        fusion_map = _non_contiguous_fusion_map()
        with self.assertRaises(IndexError):
            fusion_map.expand("block2_mrpc", 88, 0)
        spec = layerwise.layerwise_schedule(12, fusion_map)[1]
        result = layerwise.apply_layer_action(_legacy_all_max(), [1, 0, 0, 0, 0, 0], spec, fusion_map)
        self.assertEqual(result.fusion_option_ids, {1: 10, 2: 88, 4: 88, 5: 88})
        layer_base = _LAYER_WIDTH
        self.assertEqual(int(result.full_vector[layer_base + _BLOCK_OFFSETS[2]]), 9)
        self.assertEqual(int(result.full_vector[layer_base + _BLOCK_OFFSETS[4]]), 8)
        self.assertEqual(result.boosted_field_values_by_block[2]["output_truncation_k"], 8)


class VariableCostTest(unittest.TestCase):
    def _actions(self, fusion: int, k: int):
        actions = []
        for layer_idx in range(12):
            blocks = {2: k, 3: k, 4: k, 5: k}
            if layer_idx:
                blocks[1] = k
            actions.append(layerwise.LayerwiseDecodedAction(fusion, blocks))
        return actions

    def test_exact_cost_formula_and_monotonicity(self):
        low = layerwise.compute_variable_cost(self._actions(0, 13))
        high = layerwise.compute_variable_cost(self._actions(1, 8))
        middle = layerwise.compute_variable_cost(self._actions(1, 13))
        self.assertEqual((low.fusion_saving, low.truncation_saving, low.normalized), (0.0, 0.0, 0.0))
        self.assertEqual((high.fusion_saving, high.truncation_saving, high.normalized), (1.0, 1.0, 1.0))
        self.assertEqual((low.fusion_units, low.truncation_units, low.total_units), (0.0, 0.0, 0.0))
        self.assertEqual((high.fusion_units, high.truncation_units), (12.0, 147.5))
        self.assertEqual(high.total_units, high.max_units)
        self.assertEqual(len(high.layer_cost_rewards), 12)
        self.assertEqual(len(high.slot_cost_rewards), 12)
        self.assertTrue(all(len(row) == 6 for row in high.slot_cost_rewards))
        self.assertAlmostEqual(sum(high.layer_cost_rewards), high.normalized)
        for layer_cost, slot_costs in zip(
                high.layer_cost_rewards, high.slot_cost_rewards,
        ):
            self.assertAlmostEqual(sum(slot_costs), layer_cost)
        self.assertGreater(middle.normalized, low.normalized)
        self.assertLess(middle.normalized, high.normalized)
        self.assertAlmostEqual(middle.normalized, 12.0 / 159.5)

    def test_k_drop_two_equals_one_block4_fusion_toggle(self):
        baseline_actions = self._actions(0, 13)
        baseline = layerwise.compute_variable_cost(baseline_actions)

        fused_actions = self._actions(0, 13)
        fused_actions[7] = layerwise.LayerwiseDecodedAction(
            1, dict(fused_actions[7].k_by_block),
        )
        fused = layerwise.compute_variable_cost(fused_actions)

        lower_k_actions = self._actions(0, 13)
        changed_k = dict(lower_k_actions[7].k_by_block)
        changed_k[3] = 11
        lower_k_actions[7] = layerwise.LayerwiseDecodedAction(0, changed_k)
        lower_k = layerwise.compute_variable_cost(lower_k_actions)

        fusion_delta = fused.normalized - baseline.normalized
        k_delta = lower_k.normalized - baseline.normalized
        self.assertAlmostEqual(fusion_delta, 1.0 / 159.5)
        self.assertAlmostEqual(k_delta, fusion_delta)
        self.assertAlmostEqual(
            fused.layer_cost_rewards[7], lower_k.layer_cost_rewards[7],
        )
        self.assertAlmostEqual(fused.slot_cost_rewards[7][0], fusion_delta)
        self.assertAlmostEqual(lower_k.slot_cost_rewards[7][3], k_delta)

    def test_layer_cost_terms_cover_every_active_k_once(self):
        result = layerwise.compute_variable_cost(self._actions(1, 11))
        expected_layer0_units = 1.0 + 4 * 1.0
        expected_other_units = 1.0 + 5 * 1.0
        self.assertAlmostEqual(result.layer_cost_rewards[0], expected_layer0_units / 159.5)
        self.assertEqual(result.slot_cost_rewards[0][1], 0.0)
        self.assertAlmostEqual(result.slot_cost_rewards[0][0], 1.0 / 159.5)
        for value in result.layer_cost_rewards[1:]:
            self.assertAlmostEqual(value, expected_other_units / 159.5)
        self.assertEqual(result.fusion_units, 12.0)
        self.assertEqual(result.truncation_units, 59.0)
        self.assertEqual(result.total_units, 71.0)

    def test_cost_uses_decoded_k_values_not_category_order(self):
        actions = self._actions(0, 8)
        result = layerwise.compute_variable_cost(actions)
        self.assertEqual(result.truncation_saving, 1.0)
        actions[0] = layerwise.LayerwiseDecodedAction(0, {2: 8, 3: 8, 4: 8, 5: 8, 1: 8})
        with self.assertRaises(ValueError):
            layerwise.compute_variable_cost(actions)


class KLevelsContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.map = fcm.FusionCountMap.load("mrpc")
        cls.spec = layerwise.layerwise_schedule(12, cls.map)[1]

    def test_rejects_unsupported_k_levels_at_every_public_entry_point(self):
        actions = [layerwise.LayerwiseDecodedAction(0, {2: 13, 3: 13, 4: 13, 5: 13})]
        actions.extend(
            layerwise.LayerwiseDecodedAction(0, {1: 13, 2: 13, 3: 13, 4: 13, 5: 13})
            for _ in range(11)
        )
        baseline = _legacy_all_max()
        matrix = [[0, 0, 0, 0, 0, 0] for _ in range(12)]
        with mock.patch.object(layerwise, "K_LEVELS", (8, 9, 10, 11, 12, 14)):
            with self.assertRaisesRegex(ValueError, "K_LEVELS"):
                layerwise.layerwise_schedule(12, self.map)
            with self.assertRaisesRegex(ValueError, "K_LEVELS"):
                layerwise.apply_layer_action(baseline, [0, 0, 0, 0, 0, 0], self.spec, self.map)
            with self.assertRaisesRegex(ValueError, "K_LEVELS"):
                layerwise.compute_variable_cost(actions)
            with self.assertRaisesRegex(ValueError, "K_LEVELS"):
                list(layerwise.one_coordinate_neighbors(matrix))

    def test_reordered_supported_k_levels_decode_and_cost_by_value(self):
        reordered = (13, 8, 10, 9, 11, 12)
        with mock.patch.object(layerwise, "K_LEVELS", reordered):
            spec = layerwise.layerwise_schedule(12, self.map)[1]
            action = [0, reordered.index(8), reordered.index(9), reordered.index(10), reordered.index(11), reordered.index(12)]
            result = layerwise.apply_layer_action(_legacy_all_max(), action, spec, self.map)
            self.assertEqual(result.decoded.k_by_block, {1: 8, 2: 9, 3: 10, 4: 11, 5: 12})

            actions = [layerwise.LayerwiseDecodedAction(0, {2: 8, 3: 8, 4: 8, 5: 8})]
            actions.extend(
                layerwise.LayerwiseDecodedAction(0, {1: 8, 2: 8, 3: 8, 4: 8, 5: 8})
                for _ in range(11)
            )
            self.assertAlmostEqual(
                layerwise.compute_variable_cost(actions).normalized,
                147.5 / 159.5,
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
        action = [[0, layerwise.K_LEVELS.index(13), layerwise.K_LEVELS.index(13), layerwise.K_LEVELS.index(13), layerwise.K_LEVELS.index(13), layerwise.K_LEVELS.index(13)] for _ in range(12)]
        original = [row[:] for row in action]
        neighbors = list(layerwise.one_coordinate_neighbors(action))
        self.assertEqual(len(neighbors), 307)
        self.assertEqual(len({tuple(value for row in neighbor for value in row) for neighbor in neighbors}), 307)
        self.assertEqual(action, original)
        for neighbor in neighbors:
            changes = [(row, col) for row in range(12) for col in range(6) if neighbor[row][col] != action[row][col]]
            self.assertEqual(len(changes), 1)
            self.assertNotEqual(changes[0], (0, 1))
            self.assertIsNot(neighbor[0], action[0])

    def test_validates_late_coordinates_before_first_yield(self):
        action = [[0, 0, 0, 0, 0, 0] for _ in range(12)]
        action[11][5] = len(layerwise.K_LEVELS)
        iterator = layerwise.one_coordinate_neighbors(action)
        with self.assertRaises(ValueError):
            next(iterator)

    def test_masked_layer_zero_block1_k_is_ignored_everywhere(self):
        fusion_map = fcm.FusionCountMap.load("mrpc")
        spec = layerwise.layerwise_schedule(12, fusion_map)[0]
        result = layerwise.apply_layer_action(
            _legacy_all_max(), [0, 999, 0, 0, 0, 0], spec, fusion_map,
        )
        self.assertNotIn(1, result.decoded.k_by_block)

        action = [[0, 0, 0, 0, 0, 0] for _ in range(12)]
        action[0][1] = 999
        neighbors = list(layerwise.one_coordinate_neighbors(action))
        self.assertEqual(len(neighbors), 307)
        self.assertTrue(all(neighbor[0][1] == 999 for neighbor in neighbors))


class LayerwiseValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.map = fcm.FusionCountMap.load("mrpc")
        cls.spec = layerwise.layerwise_schedule(12, cls.map)[1]

    def test_invalid_action_shapes_and_indices_fail_loudly(self):
        with self.assertRaises(ValueError):
            layerwise.apply_layer_action(_legacy_all_max(), [0] * 5, self.spec, self.map)
        with self.assertRaises(ValueError):
            layerwise.apply_layer_action(_legacy_all_max(), [2, 0, 0, 0, 0, 0], self.spec, self.map)
        with self.assertRaises(ValueError):
            list(layerwise.one_coordinate_neighbors([[0] * 6 for _ in range(11)]))

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
            ("block1_mrpc", 0),
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
        malformed_vector.graphs["block1_mrpc"].options[0].action_indices.pop()
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
