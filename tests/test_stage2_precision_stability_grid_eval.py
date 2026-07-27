"""Contracts for the 27-group Stage-2 precision/stability evaluation."""
from __future__ import annotations

import ast
import unittest
from types import SimpleNamespace

import numpy as np

from scripts.run_stage2_precision_stability_grid_eval import (
    FUSION_PROFILES,
    K_LEVELS,
    REPO_ROOT,
    TRUNCATION_PROFILES,
    apply_k_profile_to_full_vector,
    apply_k_schedule_to_full_vector,
    build_group_specs,
    build_policy_action_matrix,
    installed_k_evidence,
)


class Stage2PrecisionStabilityGridTest(unittest.TestCase):
    def test_action_space_imports_os_for_max_sf_materialization(self):
        source = (REPO_ROOT / "blb_stage2_rl" / "action_space.py").read_text(
            encoding="utf-8",
        )
        tree = ast.parse(source)
        imported_names = {
            alias.name
            for node in tree.body
            if isinstance(node, ast.Import)
            for alias in node.names
        }

        self.assertIn("os", imported_names)

    def test_repo_root_contains_the_runtime_dependencies(self):
        self.assertTrue((REPO_ROOT / "Rescale_optimizer").is_dir())
        self.assertTrue((REPO_ROOT / "blb_stage2_rl").is_dir())

    def test_builds_requested_three_by_nine_grid(self):
        groups = build_group_specs(num_layers=12)

        self.assertEqual(len(FUSION_PROFILES), 3)
        self.assertEqual(len(TRUNCATION_PROFILES), 9)
        self.assertEqual(len(groups), 27)
        self.assertEqual(
            [group.fusion_by_block for group in groups[::9]],
            [(0, 0, 0), (1, 0, 1), (1, 1, 1)],
        )
        self.assertEqual(
            [group.k_by_layer[0] for group in groups[:9]],
            [
                (13, 13, 13, 13, 13),
                (8, 8, 8, 8, 8),
                (7, 7, 7, 7, 7),
                (6, 6, 6, 6, 6),
                (11, 10, 10, 12, 11),
                (9, 8, 8, 10, 9),
                (7, 6, 6, 8, 7),
                (11, 10, 10, 12, 11),
                (9, 8, 8, 10, 9),
            ],
        )
        self.assertEqual(
            groups[7].k_by_layer[1],
            (7, 6, 6, 8, 7),
        )
        self.assertEqual(
            groups[8].k_by_layer[1],
            (7, 6, 6, 8, 7),
        )
        self.assertEqual(len({group.name for group in groups}), 27)

    def test_policy_matrix_uses_exact_k6_k7_indices(self):
        group = next(
            item for item in build_group_specs(num_layers=12)
            if item.fusion_by_block == (1, 1, 1)
            and item.name == "f111_low"
        )

        matrix = build_policy_action_matrix(group, num_layers=12)

        self.assertEqual(len(matrix), 12)
        expected = (
            1,
            K_LEVELS.index(7),
            K_LEVELS.index(6),
            K_LEVELS.index(6),
            K_LEVELS.index(8),
            K_LEVELS.index(7),
        )
        self.assertTrue(all(tuple(row) == expected for row in matrix))

    def test_policy_matrix_uses_one_based_odd_even_k_schedule(self):
        group = next(
            item for item in build_group_specs(num_layers=12)
            if item.name == "f111_odd_high_even_low"
        )

        matrix = build_policy_action_matrix(group, num_layers=12)

        high = (11, 10, 10, 12, 11)
        low = (7, 6, 6, 8, 7)
        self.assertEqual(group.k_by_layer[0], high)
        self.assertEqual(group.k_by_layer[1], low)
        self.assertEqual(group.k_by_layer[10], high)
        self.assertEqual(group.k_by_layer[11], low)
        self.assertEqual(
            matrix[0],
            (1, *(K_LEVELS.index(value) for value in high)),
        )
        self.assertEqual(
            matrix[1],
            (1, *(K_LEVELS.index(value) for value in low)),
        )

    def test_control_vector_sets_all_five_blocks_including_layer0_block1(self):
        fields = [
            (1, "field_b1", "F"),
            (1, "output_truncation_k", "K"),
            (2, "output_truncation_k", "K"),
            (3, "output_truncation_k", "K"),
            (4, "output_truncation_k", "K"),
            (5, "output_truncation_k", "K"),
        ]
        baseline = np.zeros(12 * len(fields) + 1, dtype=int)
        profile = (11, 10, 10, 12, 11)

        updated = apply_k_profile_to_full_vector(
            baseline,
            k_by_block=profile,
            num_layers=12,
            per_layer_fields=fields,
        )

        for layer_idx in range(12):
            offset = layer_idx * len(fields)
            self.assertEqual(updated[offset], 0)
            self.assertEqual(
                updated[offset + 1:offset + 6].tolist(),
                [K_LEVELS.index(value) for value in profile],
            )
        self.assertEqual(updated[-1], 0)
        self.assertEqual(
            int(np.count_nonzero(updated == K_LEVELS.index(11))),
            24,
        )

    def test_control_vector_supports_one_based_odd_even_schedule(self):
        fields = [
            (1, "output_truncation_k", "K"),
            (2, "output_truncation_k", "K"),
            (3, "output_truncation_k", "K"),
            (4, "output_truncation_k", "K"),
            (5, "output_truncation_k", "K"),
        ]
        baseline = np.zeros(12 * len(fields) + 1, dtype=int)
        high = (11, 10, 10, 12, 11)
        low = (7, 6, 6, 8, 7)
        schedule = tuple(
            high if layer_idx % 2 == 0 else low
            for layer_idx in range(12)
        )

        updated = apply_k_schedule_to_full_vector(
            baseline,
            k_by_layer=schedule,
            num_layers=12,
            per_layer_fields=fields,
        )

        for layer_idx, expected in enumerate(schedule):
            offset = layer_idx * len(fields)
            self.assertEqual(
                updated[offset:offset + 5].tolist(),
                [K_LEVELS.index(value) for value in expected],
            )

    def test_policy_matrix_rejects_control_fusion_profile(self):
        control = build_group_specs(num_layers=12)[0]
        with self.assertRaisesRegex(ValueError, "not policy-representable"):
            build_policy_action_matrix(control, num_layers=12)

    def test_installed_k_audit_reads_every_post_materialization_cfg(self):
        profile = (7, 6, 6, 8, 7)
        decoded = SimpleNamespace(**{
            f"block{block_idx}_cfgs": {
                layer_idx: SimpleNamespace(
                    output_truncation_k=profile[block_idx - 1],
                    output_truncation_mode="binary",
                )
                for layer_idx in range(12)
            }
            for block_idx in range(1, 6)
        })

        rows = installed_k_evidence(
            decoded,
            expected_k_by_block=profile,
            num_layers=12,
            expected_backend="binary",
        )

        self.assertEqual(len(rows), 60)
        self.assertEqual(
            rows[0],
            {"layer": 0, "block": 1, "k_value": 7, "backend": "binary"},
        )
        self.assertEqual(rows[-1]["k_value"], 7)

        decoded.block3_cfgs[4].output_truncation_k = 13
        with self.assertRaisesRegex(RuntimeError, "installed K mismatch"):
            installed_k_evidence(
                decoded,
                expected_k_by_block=profile,
                num_layers=12,
                expected_backend="binary",
            )

    def test_installed_k_audit_validates_odd_even_schedule(self):
        high = (11, 10, 10, 12, 11)
        low = (7, 6, 6, 8, 7)
        schedule = tuple(
            high if layer_idx % 2 == 0 else low
            for layer_idx in range(12)
        )
        decoded = SimpleNamespace(**{
            f"block{block_idx}_cfgs": {
                layer_idx: SimpleNamespace(
                    output_truncation_k=schedule[layer_idx][block_idx - 1],
                    output_truncation_mode="binary",
                )
                for layer_idx in range(12)
            }
            for block_idx in range(1, 6)
        })

        rows = installed_k_evidence(
            decoded,
            expected_k_by_layer=schedule,
            expected_backend="binary",
        )

        self.assertEqual(len(rows), 60)
        self.assertEqual(
            [row["k_value"] for row in rows[:10]],
            [*high, *low],
        )


if __name__ == "__main__":
    unittest.main()
