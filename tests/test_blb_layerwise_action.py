"""Torch-free contract tests for the 12-step layerwise Stage-2 codec."""

from __future__ import annotations

from dataclasses import dataclass
import pathlib
import sys
import unittest

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
        @dataclass
        class Option:
            option_id: int
            fusion_count: int
            action_indices: list[int]
            boosted: bool = False
            explicit_field_values: dict[str, int] | None = None

        @dataclass
        class Graph:
            graph_key: str
            k_slot_index: int
            block_num_slots: int
            options: list[Option]

        class Map:
            def __init__(self):
                self.graphs = {}
                for key, size in (("block1_mrpc", 9), ("block2_mrpc", 23), ("block4", 17), ("block5_n4", 16)):
                    options = [Option(41, 0, [14] * size), Option(88, 1, [13] * size)]
                    if key == "block1_mrpc":
                        options = [Option(41, 0, [14] * size)]
                    self.graphs[key] = Graph(key, size - 1, size, options)

            def options(self, graph_key):
                return self.graphs[graph_key].options

            def expand(self, graph_key, option_id, k_index):
                option = next(option for option in self.options(graph_key) if option.option_id == option_id)
                result = np.asarray(option.action_indices, dtype=int).copy()
                result[self.graphs[graph_key].k_slot_index] = k_index
                return result

        fake = Map()
        spec = layerwise.layerwise_schedule(12, fake)[1]
        result = layerwise.apply_layer_action(_legacy_all_max(), [1, 0, 0, 0, 0, 0], spec, fake)
        self.assertEqual(result.fusion_option_ids, {1: 41, 2: 88, 4: 88, 5: 88})


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
        self.assertGreater(middle.normalized, low.normalized)
        self.assertLess(middle.normalized, high.normalized)

    def test_cost_uses_decoded_k_values_not_category_order(self):
        actions = self._actions(0, 8)
        result = layerwise.compute_variable_cost(actions)
        self.assertEqual(result.truncation_saving, 1.0)
        actions[0] = layerwise.LayerwiseDecodedAction(0, {2: 8, 3: 8, 4: 8, 5: 8, 1: 8})
        with self.assertRaises(ValueError):
            layerwise.compute_variable_cost(actions)


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
        graph.options = graph.options + [next(option for option in graph.options if option.fusion_count == 1)]
        with self.assertRaisesRegex(ValueError, "exactly one"):
            layerwise.layerwise_schedule(12, duplicate)


if __name__ == "__main__":
    unittest.main()
