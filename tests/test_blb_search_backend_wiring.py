from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SearchBackendWiringTests(unittest.TestCase):
    def test_python_entrypoint_forwards_search_backend_and_budget(self):
        tree = ast.parse((ROOT / "rl_tune.py").read_text(encoding="utf-8"))
        train = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "train"
        )
        arguments = {argument.arg for argument in train.args.args}
        self.assertIn("blb_v3_search_backend", arguments)
        self.assertIn("blb_v3_search_evaluation_budget", arguments)
        self.assertIn("blb_v3_search_full_validation", arguments)
        evaluator_call = next(
            node for node in ast.walk(train)
            if isinstance(node, ast.Call)
            and (
                getattr(node.func, "id", "")
                or getattr(node.func, "attr", "")
            ) == "LayerImportanceEvaluator"
        )
        keywords = {keyword.arg for keyword in evaluator_call.keywords}
        self.assertIn("blb_v3_search_backend", keywords)
        self.assertIn("blb_v3_search_evaluation_budget", keywords)
        self.assertIn("blb_v3_search_full_validation", keywords)

    def test_evaluator_and_train_config_preserve_ppo_default(self):
        evaluator_tree = ast.parse(
            (ROOT / "layer_importance_evaluator.py").read_text(encoding="utf-8")
        )
        evaluator_class = next(
            node for node in evaluator_tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "LayerImportanceEvaluator"
        )
        init = next(
            node for node in evaluator_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        defaults = dict(zip(
            [argument.arg for argument in init.args.args[-len(init.args.defaults):]],
            init.args.defaults,
        ))
        self.assertEqual(
            ast.literal_eval(defaults["blb_v3_search_backend"]),
            "ppo",
        )

        runner_tree = ast.parse(
            (ROOT / "blb_stage2_rl" / "runner.py").read_text(encoding="utf-8")
        )
        config_class = next(
            node for node in runner_tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "BLBStage2TrainConfig"
        )
        assignments = {
            node.target.id: node.value
            for node in config_class.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.value is not None
        }
        self.assertEqual(
            ast.literal_eval(assignments["search_backend"]),
            "ppo",
        )

    def test_launcher_and_layerwise_runner_expose_all_three_baselines(self):
        launcher = (ROOT / "llama_7B_LayerImportance.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("--blb-v3-search-backend", launcher)
        self.assertIn("--blb-v3-search-evaluation-budget", launcher)
        self.assertIn("--blb-v3-search-rf-n-estimators", launcher)
        self.assertIn(
            "--blb-v3-search-mutation-max-coordinates", launcher,
        )
        self.assertIn("ppo|bo_rf|greedy|coinn_ga", launcher)
        self.assertIn("--blb_v3_search_backend", launcher)
        self.assertIn("--blb_v3_search_rf_n_estimators", launcher)
        self.assertIn(
            "--blb_v3_search_mutation_max_coordinates", launcher,
        )

        runner = (
            ROOT / "blb_stage2_rl" / "sequential_runner.py"
        ).read_text(encoding="utf-8")
        self.assertIn("run_layerwise_search_baseline", runner)
        self.assertIn("search_backend != \"ppo\"", runner)
        self.assertIn("search_full_validation", runner)
        self.assertIn(
            'if not bool(search_run["scientific_export_allowed"])',
            runner,
        )
        self.assertIn("smoke_only_complete", runner)

        baseline_runner = (
            ROOT / "blb_stage2_rl" / "search_baseline_runner.py"
        ).read_text(encoding="utf-8")
        self.assertIn("promote_candidate_if_eligible", baseline_runner)
        self.assertIn("certify_candidate_with_bank_c", baseline_runner)
        self.assertIn(
            "joint_six_point_plus_compute_and_communication_",
            baseline_runner,
        )

        evaluator = (ROOT / "layer_importance_evaluator.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "smoke-only Stage-2 search cannot be handed to final evaluation",
            evaluator,
        )


if __name__ == "__main__":
    unittest.main()
