"""Contracts for the two-slot Stage-2 layerwise action."""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from rfr.preparation.fusion.count_map import FusionCountMap
from rfr.search.common.layerwise_action import (
    LAYERWISE_SLOT_NAMES,
    apply_layer_action,
    compute_variable_cost_from_action_matrix,
    describe_layerwise_action_matrix,
    materialize_layerwise_counterfactuals,
    layerwise_schedule,
)
from rfr.search.common.precision_presets import (
    PRECISION_PRESETS,
    allocated_precision_tolerances,
    network_axis_weights,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


class LayerwisePrecisionPresetContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fusion_map = FusionCountMap.load("mrpc")
        cls.schedule = layerwise_schedule(
            12,
            cls.fusion_map,
            profile="mrpc",
            gelu_degrees=[4] * 12,
        )

    def test_formal_mrpc_preset_pins_hml_action_and_equal_network_weights(self):
        preset_path = _REPO_ROOT / "presets" / "bert-base-mrpc-stage2-rl.conf"
        text = preset_path.read_text(encoding="utf-8")

        self.assertIn("--stage2-stability-multiplier 2.0", text)
        self.assertEqual(tuple(preset.name for preset in PRECISION_PRESETS), (
            "high", "medium", "low",
        ))
        self.assertEqual(network_axis_weights(1.0), (0.5, 0.5))

    def test_policy_has_only_fusion_and_precision_preset_slots(self):
        self.assertEqual(
            LAYERWISE_SLOT_NAMES,
            ("block4_fusion", "truncation_precision"),
        )
        self.assertEqual(self.schedule[0].slot_dims, (2, 3))
        self.assertEqual(self.schedule[0].slot_mask, (True, True))

    def test_readable_action_description_lists_every_layer_and_block_k(self):
        rows = [[0, 0], [1, 1], [0, 2]]

        description = describe_layerwise_action_matrix(rows)

        self.assertEqual(
            description,
            [
                {
                    "layer_idx": 0,
                    "block4_fusion_count": 0,
                    "precision_preset_index": 0,
                    "precision_preset_name": "high",
                    "truncation_k_by_block": {
                        "block1": 11,
                        "block2": 10,
                        "block3": 10,
                        "block4": 12,
                        "block5": 11,
                    },
                    "cleartext_simulation_k_by_block": {
                        "block1": 11,
                        "block2": 10,
                        "block3": 10,
                        "block4": 12,
                        "block5": 11,
                    },
                    "ciphertext_truncation_k_by_block": {
                        "block1": 13,
                        "block2": 13,
                        "block3": 13,
                        "block4": 13,
                        "block5": 13,
                    },
                    "reserve_bits_by_block": {
                        "block1": 2,
                        "block2": 3,
                        "block3": 3,
                        "block4": 1,
                        "block5": 2,
                    },
                    "ciphertext_ring_bits": 40,
                },
                {
                    "layer_idx": 1,
                    "block4_fusion_count": 1,
                    "precision_preset_index": 1,
                    "precision_preset_name": "medium",
                    "truncation_k_by_block": {
                        "block1": 9,
                        "block2": 8,
                        "block3": 8,
                        "block4": 10,
                        "block5": 9,
                    },
                    "cleartext_simulation_k_by_block": {
                        "block1": 9,
                        "block2": 8,
                        "block3": 8,
                        "block4": 10,
                        "block5": 9,
                    },
                    "ciphertext_truncation_k_by_block": {
                        "block1": 12,
                        "block2": 12,
                        "block3": 12,
                        "block4": 12,
                        "block5": 12,
                    },
                    "reserve_bits_by_block": {
                        "block1": 3,
                        "block2": 4,
                        "block3": 4,
                        "block4": 2,
                        "block5": 3,
                    },
                    "ciphertext_ring_bits": 39,
                },
                {
                    "layer_idx": 2,
                    "block4_fusion_count": 0,
                    "precision_preset_index": 2,
                    "precision_preset_name": "low",
                    "truncation_k_by_block": {
                        "block1": 7,
                        "block2": 6,
                        "block3": 6,
                        "block4": 8,
                        "block5": 7,
                    },
                    "cleartext_simulation_k_by_block": {
                        "block1": 7,
                        "block2": 6,
                        "block3": 6,
                        "block4": 8,
                        "block5": 7,
                    },
                    "ciphertext_truncation_k_by_block": {
                        "block1": 11,
                        "block2": 11,
                        "block3": 11,
                        "block4": 12,
                        "block5": 11,
                    },
                    "reserve_bits_by_block": {
                        "block1": 4,
                        "block2": 5,
                        "block3": 5,
                        "block4": 4,
                        "block5": 4,
                    },
                    "ciphertext_ring_bits": 38,
                },
            ],
        )

    def test_preset_table_is_exact_and_has_requested_utility(self):
        self.assertEqual(
            [
                (preset.name, preset.k_by_block, preset.communication_utility)
                for preset in PRECISION_PRESETS
            ],
            [
                ("high", (11, 10, 10, 12, 11), 0.0),
                ("medium", (9, 8, 8, 10, 9), 0.5),
                ("low", (7, 6, 6, 8, 7), 1.0),
            ],
        )

    def test_preset_table_exposes_paper_semantics_without_changing_simulation_k(self):
        self.assertEqual(
            [
                (
                    preset.name,
                    preset.ciphertext_k_by_block,
                    preset.simulation_k_by_block,
                    preset.reserve_bits_by_block,
                    preset.ciphertext_ring_bits,
                    preset.k_by_block,
                    preset.communication_utility,
                )
                for preset in PRECISION_PRESETS
            ],
            [
                (
                    "high",
                    (13, 13, 13, 13, 13),
                    (11, 10, 10, 12, 11),
                    (2, 3, 3, 1, 2),
                    40,
                    (11, 10, 10, 12, 11),
                    0.0,
                ),
                (
                    "medium",
                    (12, 12, 12, 12, 12),
                    (9, 8, 8, 10, 9),
                    (3, 4, 4, 2, 3),
                    39,
                    (9, 8, 8, 10, 9),
                    0.5,
                ),
                (
                    "low",
                    (11, 11, 11, 12, 11),
                    (7, 6, 6, 8, 7),
                    (4, 5, 5, 4, 4),
                    38,
                    (7, 6, 6, 8, 7),
                    1.0,
                ),
            ],
        )

    def test_every_preset_reaches_legacy_vector_and_boosted_overrides(self):
        from rfr.search.common.truncation_levels import K_LEVELS

        baseline = np.full(12 * 73 + 1, 14, dtype=int)
        for layer_idx in range(12):
            for block_start, block_width in (
                (0, 9), (9, 23), (32, 8), (40, 17), (57, 16),
            ):
                baseline[layer_idx * 73 + block_start + block_width - 1] = (
                    K_LEVELS.index(13)
                )
        baseline[-1] = 4
        for preset_index, preset in enumerate(PRECISION_PRESETS):
            application = apply_layer_action(
                baseline,
                (1, preset_index),
                self.schedule[0],
                self.fusion_map,
            )
            self.assertEqual(
                tuple(application.decoded.k_by_block[block] for block in range(1, 6)),
                preset.k_by_block,
            )
            for block_idx, k_value in enumerate(preset.k_by_block, start=1):
                block_starts = {1: 0, 2: 9, 3: 32, 4: 40, 5: 57}
                block_widths = {1: 9, 2: 23, 3: 8, 4: 17, 5: 16}
                k_offset = block_starts[block_idx] + block_widths[block_idx] - 1
                k_index = int(application.full_vector[k_offset])
                self.assertEqual(K_LEVELS[k_index], k_value)
                if block_idx in (2, 4, 5):
                    self.assertEqual(
                        application.boosted_field_values_by_block[block_idx][
                            "output_truncation_k"
                        ],
                        k_value,
                    )

    def test_counterfactuals_isolate_compute_and_communication_axes(self):
        from rfr.search.common.truncation_levels import K_LEVELS

        baseline = np.full(12 * 73 + 1, 14, dtype=int)
        baseline_k_index = K_LEVELS.index(13)
        for layer_idx in range(12):
            for block_start, block_width in (
                (0, 9), (9, 23), (32, 8), (40, 17), (57, 16),
            ):
                baseline[layer_idx * 73 + block_start + block_width - 1] = (
                    baseline_k_index
                )
        baseline[-1] = 4
        action_matrix = [
            [layer_idx % 2, layer_idx % 3]
            for layer_idx in range(12)
        ]

        materialized = materialize_layerwise_counterfactuals(
            baseline,
            action_matrix,
            self.schedule,
            self.fusion_map,
        )

        self.assertEqual(
            set(materialized),
            {"joint", "compute_only", "communication_only"},
        )
        self.assertEqual(
            materialized["joint"].action_matrix,
            tuple(tuple(row) for row in action_matrix),
        )
        for layer_idx, row in enumerate(action_matrix):
            preset = PRECISION_PRESETS[row[1]]
            for block_idx, (block_start, block_width) in enumerate(
                    ((0, 9), (9, 23), (32, 8), (40, 17), (57, 16)),
                    start=1,
            ):
                k_offset = layer_idx * 73 + block_start + block_width - 1
                self.assertEqual(
                    materialized["compute_only"].full_vector[k_offset],
                    baseline_k_index,
                )
                self.assertEqual(
                    materialized["communication_only"].full_vector[k_offset],
                    K_LEVELS.index(preset.k_by_block[block_idx - 1]),
                )
        for fields in materialized["compute_only"].boosted_overrides.values():
            self.assertEqual(fields["output_truncation_k"], 13)
        self.assertEqual(
            dict(materialized["communication_only"].boosted_overrides),
            {},
        )
        baseline_without_k = baseline.copy()
        communication_without_k = (
            materialized["communication_only"].full_vector.copy()
        )
        for layer_idx in range(12):
            for block_start, block_width in (
                (0, 9), (9, 23), (32, 8), (40, 17), (57, 16),
            ):
                k_offset = layer_idx * 73 + block_start + block_width - 1
                baseline_without_k[k_offset] = 0
                communication_without_k[k_offset] = 0
        np.testing.assert_array_equal(
            communication_without_k,
            baseline_without_k,
        )

    def test_retargeted_axis_reference_splits_only_precision_limits(self):
        from rfr.search.common.statistical_constraints import (
            TrialSeries,
            build_baseline_reference,
            retarget_precision_tolerance,
        )

        groups = []
        for group_idx in range(5):
            groups.append(TrialSeries(
                loss=[0.30 + 0.001 * group_idx + 0.0001 * i for i in range(5)],
                metric1=[0.90 - 0.001 * group_idx - 0.0001 * i for i in range(5)],
                metric2=[0.80 - 0.001 * group_idx - 0.0001 * i for i in range(5)],
            ))
        reference = build_baseline_reference(
            groups,
            precision_tolerance=0.001,
            stability_multiplier=2.0,
            bootstrap_samples=128,
            seed=41,
        )

        compute_reference = retarget_precision_tolerance(reference, 0.0005)

        self.assertEqual(compute_reference.precision_tolerance, 0.0005)
        self.assertAlmostEqual(
            compute_reference.loss_limit,
            reference.loss_mean * 1.0005,
        )
        self.assertAlmostEqual(
            compute_reference.metric1_limit,
            reference.metric1_mean * 0.9995,
        )
        self.assertAlmostEqual(
            compute_reference.metric2_limit,
            reference.metric2_mean * 0.9995,
        )
        self.assertEqual(
            (
                compute_reference.loss_std_limit,
                compute_reference.metric1_std_limit,
                compute_reference.metric2_std_limit,
            ),
            (
                reference.loss_std_limit,
                reference.metric1_std_limit,
                reference.metric2_std_limit,
            ),
        )
        for channel in ("loss", "metric1", "metric2"):
            np.testing.assert_array_equal(
                compute_reference.bootstrap_means[channel],
                reference.bootstrap_means[channel],
            )
            np.testing.assert_array_equal(
                compute_reference.bootstrap_stds[channel],
                reference.bootstrap_stds[channel],
            )

    def test_equal_network_weights_honor_requested_cost_ratio(self):
        high = compute_variable_cost_from_action_matrix([[0, 0]])
        medium = compute_variable_cost_from_action_matrix([[0, 1]])
        low = compute_variable_cost_from_action_matrix([[0, 2]])
        fusion = compute_variable_cost_from_action_matrix([[1, 0]])

        self.assertEqual(high.ppo_resource_score, 0.0)
        self.assertEqual(medium.ppo_resource_score, 0.25)
        self.assertEqual(low.ppo_resource_score, 0.5)
        self.assertEqual(fusion.ppo_resource_score, 0.5)
        self.assertEqual(
            compute_variable_cost_from_action_matrix([[1, 2]]).ppo_resource_score,
            1.0,
        )

    def test_reward_facing_costs_remain_pre_change_goldens(self):
        observed = [
            compute_variable_cost_from_action_matrix([[0, preset_index]])
            for preset_index in range(3)
        ]

        self.assertEqual(
            [result.removed_k_bits for result in observed],
            [11, 21, 31],
        )
        self.assertEqual(
            [result.ppo_resource_score for result in observed],
            [0.0, 0.25, 0.5],
        )

    def test_network_ratio_controls_score_and_precision_budget(self):
        compute_weight, communication_weight = network_axis_weights(3.0)
        self.assertEqual((compute_weight, communication_weight), (0.25, 0.75))
        compute_tolerance, communication_tolerance = (
            allocated_precision_tolerances(0.001, 3.0)
        )
        self.assertAlmostEqual(compute_tolerance, 0.00025)
        self.assertAlmostEqual(communication_tolerance, 0.00075)
        self.assertAlmostEqual(
            compute_tolerance + communication_tolerance,
            0.001,
        )

        compute_only = compute_variable_cost_from_action_matrix(
            [[1, 0]], communication_importance_ratio=3.0,
        )
        communication_only = compute_variable_cost_from_action_matrix(
            [[0, 2]], communication_importance_ratio=3.0,
        )
        self.assertEqual(compute_only.ppo_resource_score, 0.25)
        self.assertEqual(communication_only.ppo_resource_score, 0.75)

    def test_action_matrix_requires_two_slots(self):
        with self.assertRaisesRegex(ValueError, "num_layers x 2"):
            compute_variable_cost_from_action_matrix(
                np.zeros((12, 6), dtype=int),
            )


if __name__ == "__main__":
    unittest.main()
