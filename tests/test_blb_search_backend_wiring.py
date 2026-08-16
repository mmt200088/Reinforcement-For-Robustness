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
        self.assertIn("comparator_smoke", arguments)
        self.assertIn("stage2_inference_batch_size", arguments)
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
        self.assertIn("comparator_smoke", keywords)
        self.assertIn("stage2_inference_batch_size", keywords)

    def test_comparator_smoke_is_parsed_and_validated_at_evaluator_boundary(self):
        entrypoint = (ROOT / "rl_tune.py").read_text(encoding="utf-8")
        self.assertIn(
            'comparator_smoke = parse_bool_flag(\n'
            '        comparator_smoke, "comparator_smoke",\n'
            '    )',
            entrypoint,
        )

        evaluator_source = (
            ROOT / "layer_importance_evaluator.py"
        ).read_text(encoding="utf-8")
        evaluator_tree = ast.parse(evaluator_source)
        evaluator_class = next(
            node for node in evaluator_tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "LayerImportanceEvaluator"
        )
        init = next(
            node for node in evaluator_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        defaults = dict(zip(  # noqa: B905 - local test env is Python 3.9
            [argument.arg for argument in init.args.args[-len(init.args.defaults):]],
            init.args.defaults,
        ))
        self.assertFalse(ast.literal_eval(defaults["comparator_smoke"]))
        self.assertIn(
            'self.comparator_smoke = self._coerce_bool_flag(\n'
            '            comparator_smoke, "comparator_smoke",\n'
            '        )',
            evaluator_source,
        )
        for contract in (
            "comparator smoke requires Stage-2 evaluation budget 1",
            "comparator smoke disables Stage-2 strict validation",
            "comparator smoke requires Stage-2 online trial count 3",
            "comparator smoke requires final evaluation to be skipped",
        ):
            with self.subTest(contract=contract):
                self.assertIn(contract, evaluator_source)
        self.assertIn(
            "two-stage comparator requires full Stage-2 strict ",
            evaluator_source,
        )

    def test_outer_two_stage_result_is_plain_and_atomic(self):
        source = (ROOT / "layer_importance_evaluator.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", "")
            == "_build_ordinary_two_stage_result"
        ]
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            {keyword.arg for keyword in calls[0].keywords},
            {
                "backend",
                "stage1_best_config",
                "stage2_result",
                "final_eval_result",
                "final_eval_status",
                "final_eval_ineligible_reason",
                "final_eval_error",
            },
        )
        self.assertIn('"two_stage_result.json"', source)
        self.assertIn("_atomic_json(", source)
        for removed in (
            "build_two_stage_authority_predicate",
            "build_authenticated_two_stage_projection",
            "load_completed_paean_final_eval_result",
            "load_completed_stage2_search_authority",
            "publish_two_stage_search_final_result",
            "two_stage_search_final.json",
        ):
            self.assertNotIn(removed, source)

    def test_outer_two_stage_result_keeps_direct_scientific_binding(self):
        source = (ROOT / "layer_importance_evaluator.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        helper = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_build_ordinary_two_stage_result"
        )
        helper_source = ast.get_source_segment(source, helper)

        self.assertIn('get("selection_binding")', helper_source)
        self.assertIn('get("stage1_consumed_binding")', helper_source)
        self.assertIn("final_config_fingerprint", helper_source)
        self.assertIn("strict_feasible", helper_source)
        self.assertNotIn("formal_feasible", helper_source)
        self.assertNotIn("authority_predicate", helper_source)
        self.assertNotIn("scientific_export_allowed", helper_source)

    def test_strict_infeasible_comparator_skips_optional_final_eval_only(self):
        source = (ROOT / "layer_importance_evaluator.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)

        def _is_result_get(node, key):
            return bool(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "noise_stage_result"
                and node.func.attr == "get"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == key
            )

        comparisons = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and len(node.comparators) == 1
        ]
        self.assertTrue(any(
            isinstance(node.ops[0], ast.NotEq)
            and _is_result_get(node.left, "search_backend")
            and isinstance(node.comparators[0], ast.Attribute)
            and isinstance(node.comparators[0].value, ast.Name)
            and node.comparators[0].value.id == "self"
            and node.comparators[0].attr == "blb_v3_search_backend"
            for node in comparisons
        ))
        self.assertTrue(any(
            isinstance(node.ops[0], ast.NotEq)
            and _is_result_get(node.left, "status")
            and isinstance(node.comparators[0], ast.Constant)
            and node.comparators[0].value == "completed"
            for node in comparisons
        ))
        self.assertIn("final_eval_ineligible_reason = None", source)
        self.assertIn(
            "final_eval_ineligible_reason is not None", source,
        )
        self.assertIn('"skipped_ineligible"', source)
        self.assertLess(
            source.index("final_eval_ineligible_reason is not None"),
            source.index("self.run_unified_final_eval("),
        )

    def test_one_backend_drives_serial_stage1_and_stage2_comparators(self):
        evaluator = (ROOT / "layer_importance_evaluator.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("from stage1_rl.search_runner import (", evaluator)
        self.assertIn("run_stage1_search,", evaluator)
        self.assertIn("backend=backend", evaluator)
        self.assertIn('self.stage2_fixed_config_source = "stage1_result"', evaluator)
        self.assertIn(
            'f"stage1_{self.blb_v3_search_backend}_result"',
            evaluator,
        )
        self.assertIn("stage1_bound_into_stage2", evaluator)
        self.assertIn("stage1_selection_binding", evaluator)
        self.assertIn('"result_path": stage1_result_path', evaluator)
        self.assertIn("build_stage1_search_accounting(", evaluator)
        self.assertIn(
            "two-stage comparator must run both Stage-1 and Stage-2",
            evaluator,
        )
        self.assertIn("comparator_stage1_only", evaluator)
        self.assertIn(
            '"stage1_bound_into_stage2": '
            'not self.comparator_stage1_only',
            evaluator,
        )
        self.assertIn(
            "comparator Stage-1-only mode must run Stage-1 and ",
            evaluator,
        )
        self.assertIn(
            "skip Stage-2/final evaluation",
            evaluator,
        )
        self.assertIn(
            "two-stage comparator requires full Stage-2 strict",
            evaluator,
        )
        self.assertIn(
            "two-stage comparator must strictly validate top 5",
            evaluator,
        )
        self.assertNotIn(
            "avg_loss, avg_time, metric1, metric2 = 0.0, 0.0, 0.0, 0.0",
            evaluator,
        )

        runner = (
            ROOT / "blb_stage2_rl" / "sequential_runner.py"
        ).read_text(encoding="utf-8")
        self.assertIn("expected_stage1_source", runner)
        self.assertIn(
            "two-stage comparator must bind its own Stage-1 result", runner,
        )
        self.assertIn("search_backend = normalize_search_backend(", runner)
        self.assertIn("backend=search_backend", runner)
        self.assertNotIn(
            "non-PPO Stage-2 search does not support checkpoint resume",
            runner,
        )

    def test_stage1_handoff_reopens_completed_result_before_stage2_binding(self):
        evaluator = (ROOT / "layer_importance_evaluator.py").read_text(
            encoding="utf-8"
        )
        reopen_call = "load_completed_search_result("
        self.assertIn(reopen_call, evaluator)
        run_index = evaluator.index("in_memory_stage1_result = run_stage1_search(")
        reopen_index = evaluator.index(reopen_call, run_index)
        selected_index = evaluator.index(
            "selected_stage1 = stage1_comparator_result.best"
        )
        self.assertLess(run_index, reopen_index)
        self.assertLess(reopen_index, selected_index)
        self.assertIn(
            "in-memory Stage-1 result does not match its completed ",
            evaluator,
        )

    def test_stage1_ga_outer_gate_requires_full_200_generation_completion(self):
        evaluator = (ROOT / "layer_importance_evaluator.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            '!= "completed_generations"',
            evaluator,
        )
        self.assertIn(
            "stage1_comparator_result.evaluation_count",
            evaluator,
        )

    def test_stage1_producer_uses_ordinary_result_path(self):
        evaluator = (ROOT / "layer_importance_evaluator.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("stage1_rl.provenance", evaluator)
        self.assertIn(
            'stage1_result_path = os.path.join(\n'
            '                stage1_output_dir, "result.json",\n'
            '            )',
            evaluator,
        )
        self.assertGreaterEqual(
            evaluator.count('"result_path": stage1_result_path'),
            2,
        )

        runner = (
            ROOT / "blb_stage2_rl" / "sequential_runner.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("stage1_rl.provenance", runner)
        self.assertIn(
            "from stage1_rl.search_runner import load_completed_search_result",
            runner,
        )
        self.assertIn(
            "completed_stage1 = load_completed_search_result(", runner,
        )
        self.assertIn("os.path.dirname(stage1_result_path)", runner)

    def test_evaluator_validates_reproducible_mrpc_setup(self):
        evaluator = (ROOT / "layer_importance_evaluator.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("validate_mrpc_evaluation_setup(", evaluator)
        for expected in (
            "model=self.model",
            'tokenizer=getattr(data_collator, "tokenizer", None)',
            "collator=data_collator",
            "full_validation=test_data",
            "self.mrpc_reproducibility.stability_probe",
            "batch_size=self.batch_size",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, evaluator)
        self.assertNotIn("validate_formal_mrpc_batch_size", evaluator)
        self.assertNotIn("validate_formal_mrpc_run_identity", evaluator)
        self.assertNotIn("hash_formal_mrpc_tokenized_view", evaluator)

    def test_evaluator_pins_comparator_stage2_to_historical_rl_batch(self):
        evaluator = (ROOT / "layer_importance_evaluator.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "MRPC_STAGE2_RL_ALIGNMENT_BATCH_SIZE", evaluator,
        )
        self.assertIn(
            "two-stage comparators require Stage-2 inference batch size",
            evaluator,
        )
        self.assertIn(
            "def activate_stage2_inference_batch_size(self)", evaluator,
        )

        runner = (ROOT / "blb_stage2_rl" / "runner.py").read_text(
            encoding="utf-8"
        )
        activation = "ev.activate_stage2_inference_batch_size()"
        self.assertIn(activation, runner)
        self.assertLess(
            runner.index(activation),
            runner.index("run_sequential_via_runner("),
        )

    def test_evaluator_revalidates_stage2_scientific_parameters(self):
        evaluator = (ROOT / "layer_importance_evaluator.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(evaluator)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", "")
            == "validate_comparator_scientific_parameters"
        ]
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            {keyword.arg for keyword in calls[0].keywords},
            {
                "communication_importance_ratio",
                "truncation_backend",
                "truncation_ring_bits",
                "truncation_source_fractional_bits",
            },
        )
        self.assertIn("if self.mrpc_reproducibility is None:", evaluator)
        self.assertIn(
            "two-stage comparators require the MRPC reproducibility fixture",
            evaluator,
        )
        self.assertIn(
            'self.stage2_rl_variant != "blb_v3"',
            evaluator,
        )

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
        self.assertIn("run bo_rf", launcher)
        self.assertIn("run coinn_ga", launcher)
        self.assertIn(
            'ga|genetic) SUBCOMMAND_ARGS=(--search-algorithm ga', launcher,
        )
        self.assertIn("--stage1-accuracy-tolerance 0.001", launcher)
        self.assertIn("--stage2-limit-tolerance 0.001", launcher)
        self.assertIn("--stage2-stability-multiplier 2.0", launcher)
        self.assertIn("--blb-v3-final-selection-top-n 5", launcher)
        self.assertIn("--blb-v3-search-mutation-max-coordinates 4", launcher)
        self.assertIn("--blb-v3-search-patience-generations 5", launcher)
        self.assertIn("_PERSISTENT_ALGORITHM=\"$BLB_V3_SEARCH_BACKEND\"", launcher)
        self.assertIn("--blb_v3_search_rf_n_estimators", launcher)
        self.assertIn(
            "--blb_v3_search_mutation_max_coordinates", launcher,
        )

        evaluator = (ROOT / "layer_importance_evaluator.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'self.blb_v3_search_backend == "coinn_ga" and (', evaluator,
        )
        self.assertIn(
            "or int(self.blb_v3_search_patience_generations) != 5",
            evaluator,
        )
        self.assertIn(
            '10_000 if self.comparator_stage1_only else 50_000',
            evaluator,
        )
        self.assertIn(
            '1_000 if self.comparator_stage1_only else 100',
            evaluator,
        )
        self.assertIn("200-generation ", evaluator)
        self.assertIn("11,464-inference full-run contract", evaluator)
        self.assertIn("800-generation safety cap", evaluator)
        self.assertIn("45,664-inference safety cap", evaluator)
        self.assertIn("Stage1SearchGracefulStop", evaluator)
        self.assertIn("NOISE_STAGE_STOP_FLAG_FILENAME", evaluator)
        self.assertIn("install_graceful_stop_handler", evaluator)
        self.assertIn("is_graceful_stop_requested", evaluator)
        self.assertIn("stop_requested=stage1_comparator_stop_requested", evaluator)
        self.assertIn('"stopped_by": "graceful_stop"', evaluator)

        runner = (
            ROOT / "blb_stage2_rl" / "sequential_runner.py"
        ).read_text(encoding="utf-8")
        self.assertIn("run_layerwise_search_baseline", runner)
        self.assertIn("search_backend != \"ppo\"", runner)
        self.assertIn("search_full_validation", runner)
        self.assertIn(
            'if search_run["strict_validation"] is None',
            runner,
        )
        self.assertIn("smoke_only_complete", runner)
        self.assertIn("completed_infeasible", runner)
        self.assertIn("full_search_strict_least_violating", runner)

        baseline_runner = (
            ROOT / "blb_stage2_rl" / "search_baseline_runner.py"
        ).read_text(encoding="utf-8")
        self.assertIn("promote_candidate_if_eligible", baseline_runner)
        self.assertIn("certify_candidate_with_bank_c", baseline_runner)
        self.assertIn(
            "joint_six_point_plus_compute_and_communication_",
            baseline_runner,
        )
        self.assertIn(
            "_prepare_strict_materialization_fingerprints(",
            baseline_runner,
        )
        self.assertIn("prepare_action_for_terminal_probe", baseline_runner)
        self.assertNotIn("expected_identity_context_hash=", baseline_runner)
        self.assertNotIn("expected_formal_run_identity=", baseline_runner)

        evaluator = (ROOT / "layer_importance_evaluator.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "smoke-only Stage-2 search cannot be handed to final evaluation",
            evaluator,
        )
        self.assertIn(
            "two-stage comparators require Stage-1 seed 42",
            evaluator,
        )
        self.assertIn(
            "two-stage comparators require Stage-2 seed 42",
            evaluator,
        )
        self.assertIn(
            "BO-RF comparator requires 128 estimators and leaf size 2",
            evaluator,
        )
        self.assertNotIn("formal BO-RF comparator", evaluator)


if __name__ == "__main__":
    unittest.main()
