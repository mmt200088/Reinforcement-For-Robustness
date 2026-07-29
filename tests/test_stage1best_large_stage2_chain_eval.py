from __future__ import annotations

import json
from pathlib import Path
import unittest

from blb_stage2_rl.fusion_count_map import FusionCountMap
from blb_stage2_rl.layerwise_action import layerwise_schedule
from scripts import run_stage1best_large_stage2_chain_eval as experiment


REPO_ROOT = Path(__file__).resolve().parents[1]


class Stage1BestLargeStage2ChainEvalTest(unittest.TestCase):
    def test_six_groups_and_layerwise_actions_match_requested_contract(self):
        groups = experiment.build_group_specs(num_layers=24, k_value=13)

        self.assertEqual(
            [group.name for group in groups],
            [
                "original_plaintext",
                "stage1_best_plaintext",
                "gelu4_stage2_b4_f0",
                "gelu4_stage2_b4_f1",
                "stage1_best_stage2_b4_f0",
                "stage1_best_stage2_b4_f1",
            ],
        )
        self.assertEqual(sum(group.stage2_enabled for group in groups), 4)
        for group in groups:
            if not group.stage2_enabled:
                self.assertIsNone(group.action_matrix)
                continue
            self.assertEqual(len(group.action_matrix), 24)
            self.assertTrue(all(len(row) == 6 for row in group.action_matrix))
            self.assertTrue(all(row[0] == group.block4_fusion for row in group.action_matrix))
            self.assertTrue(all(row[1:] == (3, 3, 3, 3, 3) for row in group.action_matrix))

    def test_stage1_record_is_the_provenanced_large_mrpc_winner(self):
        path = (
            REPO_ROOT
            / "Parting Chapter"
            / "stage1"
            / "record"
            / experiment.STAGE1_RECORD_ID
            / "final_config.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["combo"], "bert large mrpc")
        self.assertEqual(payload["gelu_degree_per_layer"], list(experiment.STAGE1_BEST_GELU))
        self.assertEqual(payload["softmax_degree_per_layer"], [6] * 24)
        self.assertEqual(
            payload["selection"]["source_run"],
            "bert-large-mrpc-stage1-ppo-entropy0p1-20260626",
        )
        self.assertEqual(
            payload["selection"]["source_config_commit"],
            "237314146003f68c82830abfa886cf2ef7086baf",
        )
        approx = json.loads(
            (REPO_ROOT / "Model_analysis" / "configs"
             / "approx_per_dataset.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            approx["mrpc_large"]["stage1"]["gelu"],
            list(experiment.STAGE1_BEST_GELU),
        )

    def test_experiment_uses_the_production_resolver_and_runtime_chain(self):
        source = (
            REPO_ROOT / "scripts" / "run_stage1best_large_stage2_chain_eval.py"
        ).read_text(encoding="utf-8")

        for token in (
            "_resolve_stage2_fixed_stage1_config",
            "resolve_stage2_profile",
            "load_calibrated_stage2_action_context",
            "validate_calibrated_stage2_action_context",
            'deps["FusionCountMap"].load',
            "BLBStage2LayerwiseEnv",
            "_build_validation_full_batches",
            "pack_repeat_evaluation",
        ):
            self.assertIn(token, source)
        self.assertNotIn("BLBStage2SequentialEnv", source)
        self.assertIn("model_uses_replan_config", source)
        self.assertIn("final_config_fingerprint", source)
        self.assertIn('if str(profile) != "mrpc_large"', source)
        self.assertIn("expected exactly fusion counts", source)
        self.assertNotIn("stage1_rl_episodes=0", source)
        self.assertIn("stage2_rl_episodes_specified=False", source)
        self.assertIn(
            'layer_name="model." + evaluator.layers_attribute',
            source,
        )
        self.assertNotIn(
            "try:\n        evaluator.reversible_handler."
            "restore_layer_input_noise",
            source,
        )

    def test_report_provenance_matches_the_user_supplied_historical_report(self):
        provenance = experiment._stage1_record_provenance(
            experiment.STAGE1_RECORD_ID
        )

        self.assertEqual(
            provenance["selection"]["source_validation_report_sha256"],
            "e9fab28cdc4ae68eb5b9031030108a7bcd6faf40c387245f72ee171b01434271",
        )
        independent = provenance["independent_validation_evidence"]
        self.assertEqual(independent["validation_full_size"], 408)
        self.assertAlmostEqual(independent["loss"], 1.421497045)
        self.assertAlmostEqual(independent["metric1_accuracy"], 0.889706)
        self.assertAlmostEqual(independent["metric2_weighted_f1"], 0.885077)

    def test_stage1_degrees_select_the_expected_real_maps_and_ro_skeletons(self):
        fusion_map = FusionCountMap.load("mrpc_large")
        expected_graphs = {
            "block2_mrpc_large",
            "block4",
            "block5_n1",
            "block5_n2",
            "block5_n4",
        }

        self.assertEqual(set(fusion_map.graphs), expected_graphs)
        for graph_key, graph in fusion_map.graphs.items():
            with self.subTest(graph_key=graph_key):
                self.assertEqual(
                    [option.fusion_count for option in graph.options],
                    [0, 1],
                )

        all4_schedule = layerwise_schedule(
            24,
            fusion_map,
            profile="mrpc_large",
            gelu_degrees=[4] * 24,
        )
        stage1_schedule = layerwise_schedule(
            24,
            fusion_map,
            profile="mrpc_large",
            gelu_degrees=experiment.STAGE1_BEST_GELU,
        )
        all4_block5 = [
            dict(spec.graph_keys_by_block)[5] for spec in all4_schedule
        ]
        stage1_block5 = [
            dict(spec.graph_keys_by_block)[5] for spec in stage1_schedule
        ]

        self.assertEqual(all4_block5, ["block5_n4"] * 24)
        self.assertEqual(stage1_block5.count("block5_n1"), 22)
        self.assertEqual(stage1_block5.count("block5_n2"), 2)
        self.assertEqual(
            [index for index, graph_key in enumerate(stage1_block5)
             if graph_key == "block5_n2"],
            [10, 11],
        )

        archive = json.loads(
            (
                REPO_ROOT
                / "Rescale_optimizer"
                / "configs"
                / "mrpc_large"
                / "static_skeletons_mrpc_large.json"
            ).read_text(encoding="utf-8")
        )
        config_names = {
            str(result["config_name"]) for result in archive["results"]
        }
        self.assertIn("block3_exp_n6", config_names)
        self.assertTrue(
            {"block5_n1", "block5_n2", "block5_n4"}.issubset(config_names)
        )


if __name__ == "__main__":
    unittest.main()
