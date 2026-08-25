"""Ordinary Stage-1 to Stage-2 comparator binding contracts."""

from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from rfr.common.json_utils import stable_json_hash

ROOT = Path(__file__).resolve().parents[1]


class EmbeddedFinalEvaluationProtocolSourceTests(unittest.TestCase):
    def test_embedded_entrypoint_validates_protocol_before_runner_dispatch(self):
        source = (ROOT / "Paean" / "embedded.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "run_embedded_final_eval"
        )
        function_source = ast.get_source_segment(source, function)

        self.assertIn("require_final_evaluation_protocol(", function_source)
        self.assertLess(
            function_source.index("require_final_evaluation_protocol("),
            function_source.index("should_run_blb_action_eval ="),
        )


def _load_function(rel_path: str, name: str, **globals_):
    path = ROOT / rel_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.Module(
        body=[ast.parse("from __future__ import annotations\n").body[0], function],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {
        "__name__": "blb_stage2_rl._ordinary_binding_test",
        "__package__": "blb_stage2_rl",
        **globals_,
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def _completed_stage1_result():
    return SimpleNamespace(
        algorithm="bo_rf",
        config=SimpleNamespace(seed=1729),
        best=SimpleNamespace(
            action=(0, 1),
            gelu_degrees=(4, 2),
            softmax_degrees=(6, 6),
        ),
    )


def _selection_binding(result_path, *, gelu_degrees=(4, 2)):
    selection = {
        "backend": "bo_rf",
        "seed": 1729,
        "action": [0, 1],
        "gelu_degrees": list(gelu_degrees),
        "softmax_degrees": [6, 6],
        "num_layers": 2,
        "dataset_protocol_hash": "probe-a",
    }
    return {
        **selection,
        "result_path": result_path,
        "result_sha256": hashlib.sha256(
            Path(result_path).read_bytes()
        ).hexdigest(),
        "selection_hash": stable_json_hash(selection),
    }


class OrdinaryTwoStageBindingTest(unittest.TestCase):
    def test_stage1_producer_puts_result_path_in_plain_binding(self):
        source = (ROOT / "layer_importance_evaluator.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        assignments = {
            target.id: node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
            and target.id in {
                "stage1_selection_payload", "stage1_selection_binding",
            }
        }
        selection = assignments["stage1_selection_payload"]
        binding = assignments["stage1_selection_binding"]
        self.assertIsInstance(selection, ast.Dict)
        self.assertIsInstance(binding, ast.Dict)
        selection_keys = {
            key.value for key in selection.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        keys = {key.value for key in binding.keys if isinstance(key, ast.Constant) and isinstance(key.value, str)}
        self.assertIn("result_path", keys)
        self.assertIn("seed", selection_keys)
        self.assertIn("result_sha256", keys)
        self.assertIn("selection_hash", keys)

    def test_invocation_loads_completed_stage1_result_and_emits_plain_binding(self):
        build = _load_function(
            "blb_stage2_rl/sequential_runner.py",
            "_build_search_invocation_contract",
            Any=object,
            Mapping=dict,
            np=np,
            os=os,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = os.path.join(tmpdir, "result.json")
            Path(result_path).write_text("{}\n", encoding="utf-8")
            binding = _selection_binding(result_path)
            evaluator = SimpleNamespace(
                model_type="bert-base",
                total_layers=2,
                batch_size=16,
                dataset_protocol_hash="probe-a",
                stage1_comparator_selection_binding=binding,
            )
            runner = SimpleNamespace(evaluator=evaluator)
            train_cfg = SimpleNamespace(
                search_backend="bo_rf",
                profile="mrpc",
                seed=1729,
                stage2_inference_batch_size=64,
            )
            with mock.patch(
                "stage1_rl.search_runner.load_completed_search_result",
                return_value=_completed_stage1_result(),
            ) as load_completed:
                invocation = build(
                    runner=runner,
                    train_cfg=train_cfg,
                    fixed_gelu=np.asarray([4, 2], dtype=int),
                    fixed_softmax=np.asarray([6, 6], dtype=int),
                    fixed_label="BO-RF Stage-1",
                    fixed_source="stage1_bo_rf_result",
                )

        load_completed.assert_called_once_with(os.path.dirname(result_path))
        self.assertEqual(
            invocation["schema_version"],
            "stage2_search_train_probe_invocation_v1",
        )
        self.assertEqual(invocation["search_backend"], "bo_rf")
        self.assertEqual(invocation["seed"], 1729)
        self.assertEqual(invocation["num_layers"], 2)
        self.assertEqual(invocation["fixed_gelu"], [4, 2])
        self.assertEqual(invocation["fixed_softmax"], [6, 6])
        self.assertEqual(invocation["stage1_result_path"], result_path)
        self.assertEqual(invocation["stage1_selection_binding"], binding)
        self.assertEqual(
            invocation["dataset_protocol_schema"],
            "glue_train_probe_protocol_v1",
        )
        self.assertEqual(invocation["dataset_protocol_hash"], "probe-a")
        self.assertEqual(
            invocation["scientific_parameters"]["stage2_inference_batch_size"],
            64,
        )
        self.assertEqual(
            invocation["scientific_parameters"]["stage1_inference_batch_size"],
            16,
        )
        self.assertEqual(
            invocation["scientific_parameters"]["stage2_probe_size"],
            256,
        )
        self.assertEqual(
            invocation["scientific_parameters"]["stage2_stability_tolerance"],
            1.2,
        )
        self.assertEqual(
            invocation["scientific_parameters"]["truncation_backend"],
            "binary",
        )
        self.assertEqual(
            invocation["scientific_parameters"]["terminal_eval_batch_size"],
            4,
        )
        self.assertEqual(
            invocation["rng_contract"]["online_stream"],
            "ppo_global_evaluation_index_v1",
        )
        self.assertNotIn("mrpc_reproducibility_identity", invocation)
        for removed in (
            "invocation_hash",
            "stage1_result_sha256",
            "stage1_selection_provenance",
            "formal_dataset_protocol",
            "formal_dataset_identity",
            "formal_run_identity",
            "formal_run_identity_hash",
        ):
            self.assertNotIn(removed, invocation)

    def test_invocation_rejects_stage1_result_that_disagrees_with_binding(self):
        build = _load_function(
            "blb_stage2_rl/sequential_runner.py",
            "_build_search_invocation_contract",
            Any=object,
            Mapping=dict,
            np=np,
            os=os,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = os.path.join(tmpdir, "result.json")
            Path(result_path).write_text("{}\n", encoding="utf-8")
            binding = _selection_binding(
                result_path, gelu_degrees=(4, 1),
            )
            evaluator = SimpleNamespace(
                model_type="bert-base",
                total_layers=2,
                dataset_protocol_hash="probe-a",
                stage1_comparator_selection_binding=binding,
            )
            with mock.patch(
                "stage1_rl.search_runner.load_completed_search_result",
                return_value=_completed_stage1_result(),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "does not match the completed Stage-1 result",
                ):
                    build(
                        runner=SimpleNamespace(evaluator=evaluator),
                        train_cfg=SimpleNamespace(
                            search_backend="bo_rf",
                            profile="mrpc",
                            seed=1729,
                        ),
                        fixed_gelu=np.asarray([4, 1], dtype=int),
                        fixed_softmax=np.asarray([6, 6], dtype=int),
                        fixed_label="BO-RF Stage-1",
                        fixed_source="stage1_bo_rf_result",
                    )

    def test_invocation_rejects_stage1_result_byte_drift(self):
        build = _load_function(
            "blb_stage2_rl/sequential_runner.py",
            "_build_search_invocation_contract",
            Any=object,
            Mapping=dict,
            np=np,
            os=os,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = os.path.join(tmpdir, "result.json")
            Path(result_path).write_text("{}\n", encoding="utf-8")
            binding = _selection_binding(result_path)
            Path(result_path).write_text('{"changed": true}\n', encoding="utf-8")
            evaluator = SimpleNamespace(
                model_type="bert-base",
                total_layers=2,
                batch_size=16,
                dataset_protocol_hash="probe-a",
                stage1_comparator_selection_binding=binding,
            )
            with mock.patch(
                "stage1_rl.search_runner.load_completed_search_result",
                return_value=_completed_stage1_result(),
            ), self.assertRaisesRegex(
                RuntimeError, "does not match the completed Stage-1 result",
            ):
                build(
                    runner=SimpleNamespace(evaluator=evaluator),
                    train_cfg=SimpleNamespace(
                        search_backend="bo_rf",
                        profile="mrpc",
                        seed=1729,
                        stage2_inference_batch_size=64,
                    ),
                    fixed_gelu=np.asarray([4, 2], dtype=int),
                    fixed_softmax=np.asarray([6, 6], dtype=int),
                    fixed_label="BO-RF Stage-1",
                    fixed_source="stage1_bo_rf_result",
                )

    def test_active_stage2_comparator_branch_consumes_plain_binding_only(self):
        source = (ROOT / "blb_stage2_rl/sequential_runner.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        branch = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "search_backend != 'ppo'"
            and any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "run_layerwise_search_baseline"
                for child in ast.walk(node)
            )
        )
        branch_source = ast.get_source_segment(source, branch)

        self.assertIn("stage1_selection_binding", branch_source)
        self.assertNotIn("stage1_comparator_selection_provenance", branch_source)
        self.assertNotIn("stage1_selection_provenance", branch_source)
        self.assertNotIn("formal_run_identity", branch_source)
        self.assertNotIn("outer_invocation_hash", branch_source)

    def test_active_comparator_uses_direct_materialization_without_authority(self):
        source = (ROOT / "blb_stage2_rl/sequential_runner.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_run_layerwise_training_branch"
        )
        function_source = ast.get_source_segment(source, function)

        self.assertIn("run_layerwise_search_baseline", function_source)
        self.assertIn("final_config_fingerprint", function_source)
        self.assertNotIn("materialization_context", function_source)
        self.assertNotIn("axis_materialization_authority", function_source)
        self.assertNotIn("build_runtime_stage2_materialization_context", function_source)
        self.assertNotIn("build_stage2_axis_materialization_authority", function_source)

    def test_candidate_identity_context_binds_plain_stage1_selection_only(self):
        build = _load_function(
            "blb_stage2_rl/sequential_runner.py",
            "_build_layerwise_candidate_identity_context",
            Mapping=dict,
            np=np,
            os=os,
            resolve_stage2_model_type=(lambda model_type, *, num_layers: model_type),
        )
        reference = SimpleNamespace(
            precision_tolerance=0.001,
            stability_multiplier=2.0,
            bootstrap_seed=42,
            bootstrap_samples=100,
            loss_limit=1.0,
            metric1_limit=0.8,
            metric2_limit=0.8,
            loss_std_limit=0.1,
            metric1_std_limit=0.1,
            metric2_std_limit=0.1,
        )
        validation_banks = SimpleNamespace(contract_payload=lambda: {"bank_a_trials": 15})
        algorithm_contract = {
            "communication_importance_ratio": 1.0,
            "compute_axis_denominator": 12,
            "communication_axis_denominator": 12,
            "resource_credit_mode": "separable_weighted_per_slot_v1",
            "strict_resource_order": ["weighted_score", "balance_tiebreak"],
        }
        stage1_binding = {
            "backend": "bo_rf",
            "seed": 42,
            "action": [0, 1],
            "gelu_degrees": [4, 2],
            "softmax_degrees": [6, 6],
            "num_layers": 2,
            "result_path": "stage1/result.json",
            "result_sha256": "a" * 64,
            "selection_hash": "b" * 64,
            "dataset_protocol_hash": "probe-a",
        }
        common = {
            "train_cfg": SimpleNamespace(
                inproc_rescale_optimizer_root="Rescale_optimizer",
                profile="mrpc",
                stage2_inference_batch_size=64,
            ),
            "evaluator": SimpleNamespace(
                model_type="bert-base",
                batch_size=16,
                dataset_protocol_hash="probe-a",
            ),
            "fusion_map": {"block4": []},
            "max_sfs": {"block4": []},
            "fixed_gelu": np.asarray([4, 2], dtype=int),
            "fixed_softmax": np.asarray([6, 6], dtype=int),
            "robust_reference": reference,
            "authoritative_robust_reference": reference,
            "validation_banks": validation_banks,
            "probe_example_count": 256,
            "authoritative_example_count": 408,
            "schedule": [],
            "static_skeletons_baseline": {"profile": "mrpc"},
            "algorithm_contract": algorithm_contract,
            "algorithm_contract_hash": "algorithm-v9",
        }

        comparator_context = build(
            **common,
            stage1_selection_binding=stage1_binding,
        )
        ppo_context = build(**common)

        expected_binding = {
            **stage1_binding,
            "result_path": os.path.abspath(stage1_binding["result_path"]),
        }
        self.assertEqual(
            comparator_context["stage1_selection_binding"],
            expected_binding,
        )
        self.assertEqual(comparator_context["stage2_inference_batch_size"], 64)
        self.assertNotIn("stage2_inference_batch_size", ppo_context)
        self.assertNotIn("stage1_selection_binding", ppo_context)
        for removed in (
            "stage1_selection_provenance",
            "formal_run_identity",
            "formal_stage1_contract_hash",
        ):
            self.assertNotIn(removed, comparator_context)

    def test_candidate_identity_rejects_missing_stage1_content_hashes(self):
        build = _load_function(
            "blb_stage2_rl/sequential_runner.py",
            "_build_layerwise_candidate_identity_context",
            Mapping=dict,
            np=np,
            os=os,
            resolve_stage2_model_type=(lambda model_type, *, num_layers: model_type),
        )
        reference = SimpleNamespace(
            precision_tolerance=0.001,
            stability_multiplier=2.0,
            bootstrap_seed=42,
            bootstrap_samples=100,
            loss_limit=1.0,
            metric1_limit=0.8,
            metric2_limit=0.8,
            loss_std_limit=0.1,
            metric1_std_limit=0.1,
            metric2_std_limit=0.1,
        )
        common = {
            "train_cfg": SimpleNamespace(
                inproc_rescale_optimizer_root="Rescale_optimizer",
                profile="mrpc",
                stage2_inference_batch_size=64,
            ),
            "evaluator": SimpleNamespace(
                model_type="bert-base",
                batch_size=16,
                dataset_protocol_hash="probe-a",
            ),
            "fusion_map": {"block4": []},
            "max_sfs": {"block4": []},
            "fixed_gelu": np.asarray([4, 2], dtype=int),
            "fixed_softmax": np.asarray([6, 6], dtype=int),
            "robust_reference": reference,
            "authoritative_robust_reference": reference,
            "validation_banks": SimpleNamespace(
                contract_payload=lambda: {"bank_a_trials": 15}
            ),
            "probe_example_count": 256,
            "authoritative_example_count": 408,
            "schedule": [],
            "static_skeletons_baseline": {"profile": "mrpc"},
            "algorithm_contract": {
                "communication_importance_ratio": 1.0,
                "compute_axis_denominator": 12,
                "communication_axis_denominator": 12,
                "resource_credit_mode": "separable_weighted_per_slot_v1",
                "strict_resource_order": [
                    "weighted_score", "balance_tiebreak",
                ],
            },
            "algorithm_contract_hash": "algorithm-v9",
        }
        binding = {
            "backend": "bo_rf",
            "seed": 42,
            "action": [0, 1],
            "gelu_degrees": [4, 2],
            "softmax_degrees": [6, 6],
            "num_layers": 2,
            "result_path": "stage1/result.json",
            "result_sha256": "",
            "selection_hash": "b" * 64,
        }

        with self.assertRaisesRegex(
            RuntimeError,
            "content hashes",
        ):
            build(**common, stage1_selection_binding=binding)

    def test_completed_resume_returns_before_model_or_materialization_setup(self):
        run = _load_function(
            "blb_stage2_rl/sequential_runner.py",
            "run_sequential_via_runner",
            Any=object,
            Dict=dict,
            K_LEVELS=(8,),
            validate_exact_k_domain=lambda _levels: None,
            _ProbeRunnerOwnerHolder=_FakeProbeOwnerHolder,
            _preflight_completed_search_resume=lambda **_kwargs: {
                "status": "completed",
                "search_backend": "bo_rf",
            },
            _preflight_pending_strict_search_resume=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("completed resume checked pending strict state")
            ),
            _run_sequential_via_runner_locked=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("completed resume entered Stage-2 runtime")
            ),
            _build_stage2_materialization_env=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("completed resume built materialization environment")
            ),
            _reauthenticate_completed_strict_materializations=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("completed resume reauthenticated materialization")
            ),
        )
        layerwise_runner = ModuleType("blb_stage2_rl.layerwise_runner")
        layerwise_runner.LayerwiseRunLock = _FakeRunLock
        runner_module = ModuleType("blb_stage2_rl.training")
        runner_module.resolve_blb_persistence_dir = lambda _evaluator: "/unused"
        evaluator = SimpleNamespace()
        runner = SimpleNamespace(evaluator=evaluator)

        with mock.patch.dict(
            sys.modules,
            {
                "blb_stage2_rl.layerwise_runner": layerwise_runner,
                "blb_stage2_rl.training": runner_module,
            },
        ):
            resumed = run(
                runner=runner,
                train_cfg=SimpleNamespace(),
                fixed_gelu=np.asarray([4, 2]),
                fixed_softmax=np.asarray([6, 6]),
                fixed_label="BO-RF Stage-1",
                fixed_source="stage1_bo_rf_result",
            )

        self.assertEqual(resumed["status"], "completed")
        self.assertTrue(_FakeProbeOwnerHolder.last.closed)

    def test_completed_resume_preflight_uses_plain_inner_artifacts(self):
        source = (ROOT / "blb_stage2_rl/sequential_runner.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_preflight_completed_search_resume"
        )
        function_source = ast.get_source_segment(source, function)

        self.assertIn("_load_plain_completed_search_run", function_source)
        self.assertNotIn("authority_io", function_source)
        self.assertNotIn("completion_seal", function_source)
        self.assertNotIn("sequential_completion_seal", function_source)
        self.assertNotIn("_verify_inner_completed_search_artifacts", function_source)
        self.assertNotIn("_verify_search_resume_file_descriptor", function_source)

    def test_completed_resume_writer_is_plain_atomic_json(self):
        source = (ROOT / "blb_stage2_rl/sequential_runner.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_write_completed_search_resume"
        )
        function_source = ast.get_source_segment(source, function)

        self.assertIn("resume_result.json", function_source)
        self.assertIn("_atomic_json", function_source)
        self.assertNotIn("completion_seal", function_source)
        self.assertNotIn("sequential_completion_seal", function_source)
        self.assertNotIn("_search_resume_file_descriptor", function_source)
        self.assertNotIn("_verify_search_resume_file_descriptor", function_source)

    def test_completed_resume_result_uses_scientific_fields_only(self):
        source = (ROOT / "blb_stage2_rl/sequential_runner.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_build_completed_search_resume_result"
        )
        function_source = ast.get_source_segment(source, function)

        self.assertIn("strict_feasible", function_source)
        self.assertIn("stage1_consumed_binding", function_source)
        self.assertNotIn("formal_feasible", function_source)
        self.assertNotIn("scientific_export_allowed", function_source)
        self.assertNotIn("stage1_consumed_provenance", function_source)
        self.assertNotIn("formal_run_identity", function_source)
        self.assertNotIn("inner_completion_identity", function_source)
        self.assertNotIn("materialization_context", function_source)
        self.assertNotIn("axis_materialization_authority", function_source)

    def test_completed_resume_restores_stage2_inference_batch_size(self):
        build = _load_function(
            "blb_stage2_rl/sequential_runner.py",
            "_build_completed_search_resume_result",
            Any=object,
            Mapping=dict,
            np=np,
            _selected_action_identity_payload=lambda _selected: {
                "action_matrix": [[0, 0], [0, 0]],
                "full_vector": [0, 0, 0, 0],
                "final_config_fingerprint": "fixed-config",
            },
        )
        stage1_binding = {"backend": "bo_rf", "selection_hash": "binding"}
        invocation_contract = {
            "search_backend": "bo_rf",
            "profile": "mrpc",
            "num_layers": 2,
            "stage1_selection_binding": stage1_binding,
            "scientific_parameters": {"stage2_inference_batch_size": 64},
        }
        selected = SimpleNamespace(
            metadata={"pending_full_vector": [0, 0, 0, 0]},
            action_matrix=((0, 0), (0, 0)),
            limits=SimpleNamespace(
                as_dict=lambda: {
                    "loss_max": 1.0,
                    "metric1_min": 0.5,
                    "metric2_min": 0.5,
                },
            ),
            as_dict=lambda: {"reward": 0.0},
        )
        inner_run = {
            "manifest": {
                "status": "smoke_only_complete",
                "search_backend": "bo_rf",
                "profile": "mrpc",
                "num_layers": 2,
                "fixed_gelu": [4, 4],
                "fixed_softmax": [6, 6],
                "stage1_backend": "bo_rf",
                "stage1_bound_into_stage2": True,
                "stage1_selection_binding": stage1_binding,
                "strict_identity_context_hash": "strict-context",
                "resume_contract": {
                    "requested_manifest": {
                        "stage2_invocation": invocation_contract,
                    },
                },
            },
            "selected": selected,
            "result": SimpleNamespace(
                evaluation_count=1,
                as_dict=lambda: {"evaluation_count": 1},
            ),
            "strict_feasible": False,
            "strict_validation": None,
            "artifact_paths": {},
        }
        fusion_module = ModuleType("rfr.preparation.fusion.fixed_action")
        fusion_module.build_fusion_fixed_config = lambda *_args, **_kwargs: None
        action_module = ModuleType("blb_stage2_rl.layerwise_action")
        action_module.describe_layerwise_action_matrix = lambda value: list(value)
        runner_module = ModuleType("blb_stage2_rl.training")
        runner_module._build_best_noise_config = lambda _value: {}

        with mock.patch.dict(
            sys.modules,
            {
                "rfr.preparation.fusion.fixed_action": fusion_module,
                "blb_stage2_rl.layerwise_action": action_module,
                "blb_stage2_rl.training": runner_module,
            },
        ):
            resumed = build(
                runner=SimpleNamespace(evaluator=SimpleNamespace()),
                fixed_gelu=np.asarray([4, 4]),
                fixed_softmax=np.asarray([6, 6]),
                invocation_contract=invocation_contract,
                inner_run=inner_run,
            )

        self.assertEqual(resumed["stage2_inference_batch_size"], 64)

    def test_completed_resume_legacy_authority_helpers_are_removed(self):
        source = (ROOT / "blb_stage2_rl/sequential_runner.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        function_names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        removed_helpers = {
            "_search_resume_file_descriptor",
            "_verify_search_resume_file_descriptor",
            "_inner_completion_identity_payload",
            "_completed_inner_search_authority_projection",
            "_completed_inner_search_authority_with_completion_identity",
            "_completed_search_resume_authority_projection",
            "_verify_inner_completed_search_artifacts",
            "_reauthenticate_completed_strict_materializations",
        }

        self.assertTrue(removed_helpers.isdisjoint(function_names))

    def test_outer_two_stage_result_uses_plain_stage_binding(self):
        source_path = ROOT / "layer_importance_evaluator.py"
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        function_names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        self.assertIn("_build_ordinary_two_stage_result", function_names)
        build = _load_function(
            "layer_importance_evaluator.py",
            "_build_ordinary_two_stage_result",
            Any=object,
            Mapping=dict,
            os=os,
        )
        binding = {
            "backend": "bo_rf",
            "seed": 1729,
            "action": [0, 1],
            "gelu_degrees": [4, 2],
            "softmax_degrees": [6, 6],
            "num_layers": 2,
            "result_path": "/runs/stage1/result.json",
        }
        stage1 = {
            "selection_binding": binding,
            "evaluation": {"action": [0, 1]},
            "search_accounting": {"evaluation_count": 64},
        }
        stage2 = {
            "status": "completed",
            "strict_feasible": True,
            "search_backend": "bo_rf",
            "stage1_consumed_binding": binding,
            "blb_v3_best_action_vec": [0, 1, 2, 3],
            "blb_v3_best_action_group": {
                "policy_actions": [[0, 0], [1, 2]],
                "boosted_overrides": [],
            },
            "blb_v3_layerwise_best_configuration": [
                {"layer": 0},
                {"layer": 1},
            ],
            "final_config_fingerprint": "cfg-plain",
            "search_accounting": {"observation_count": 12},
            "selection_diagnostics": {
                "artifact_paths": {
                    "manifest": "/runs/stage2/manifest.json",
                },
                "strict_validation": {
                    "selection_status": "strict_feasible",
                },
            },
        }

        payload = build(
            backend="bo_rf",
            stage1_best_config=stage1,
            stage2_result=stage2,
            final_eval_result={"summary_path": "/runs/final/results.json"},
            final_eval_status="completed",
            final_eval_ineligible_reason=None,
            final_eval_error=None,
        )

        self.assertEqual(payload["schema_version"], "two_stage_search_result_v1")
        self.assertEqual(payload["status"], "complete_strict_feasible")
        self.assertTrue(payload["strict_feasible"])
        self.assertTrue(payload["stage1_bound_into_stage2"])
        self.assertEqual(payload["stage1"]["selection_binding"], binding)
        self.assertEqual(payload["stage2"]["consumed_stage1_binding"], binding)
        self.assertEqual(
            payload["stage2"]["manifest_path"],
            "/runs/stage2/manifest.json",
        )
        self.assertEqual(payload["stage2"]["final_config_fingerprint"], "cfg-plain")
        self.assertEqual(payload["final_eval"]["status"], "completed")
        self.assertIsNone(payload["final_eval"]["ineligible_reason"])
        self.assertNotIn("not_authorized_reason", payload["final_eval"])
        self.assertNotIn("formal_feasible", payload)
        self.assertNotIn("authority_predicate", payload)

        mismatched = dict(stage2)
        mismatched["stage1_consumed_binding"] = {
            **binding,
            "seed": 99,
        }
        with self.assertRaisesRegex(
            RuntimeError,
            "does not match the Stage-1 selection binding",
        ):
            build(
                backend="bo_rf",
                stage1_best_config=stage1,
                stage2_result=mismatched,
                final_eval_result=None,
                final_eval_status="skipped_by_request",
                final_eval_ineligible_reason=None,
                final_eval_error=None,
            )

    def test_optional_paean_failure_preserves_strict_stage2_result(self):
        build = _load_function(
            "layer_importance_evaluator.py",
            "_build_ordinary_two_stage_result",
            Any=object,
            Mapping=dict,
            os=os,
        )
        binding = {
            "backend": "bo_rf",
            "seed": 1729,
            "action": [0, 1],
            "gelu_degrees": [4, 2],
            "softmax_degrees": [6, 6],
            "num_layers": 2,
            "result_path": "/runs/stage1/result.json",
        }
        stage2 = {
            "status": "completed",
            "strict_feasible": True,
            "search_backend": "bo_rf",
            "stage1_consumed_binding": binding,
            "blb_v3_best_action_vec": [0, 1, 2, 3],
            "blb_v3_best_action_group": {
                "policy_actions": [[0, 0], [1, 2]],
            },
            "final_config_fingerprint": "strict-f4-fingerprint",
            "selection_diagnostics": {
                "strict_validation": {
                    "selection_status": "strict_feasible",
                },
            },
        }

        completed = build(
            backend="bo_rf",
            stage1_best_config={"selection_binding": binding},
            stage2_result=stage2,
            final_eval_result={"summary_path": "/runs/paean/result.json"},
            final_eval_status="completed",
            final_eval_ineligible_reason=None,
            final_eval_error=None,
        )
        failed_optional = build(
            backend="bo_rf",
            stage1_best_config={"selection_binding": binding},
            stage2_result=stage2,
            final_eval_result=None,
            final_eval_status="failed_optional",
            final_eval_ineligible_reason=None,
            final_eval_error="RuntimeError('mock Paean failure')",
        )

        self.assertEqual(
            stable_json_hash(failed_optional["stage2"]),
            stable_json_hash(completed["stage2"]),
        )
        self.assertEqual(
            failed_optional["status"], "complete_strict_feasible",
        )
        self.assertTrue(failed_optional["strict_feasible"])
        self.assertEqual(
            failed_optional["stage2"]["final_config_fingerprint"],
            "strict-f4-fingerprint",
        )
        self.assertEqual(
            failed_optional["final_eval"]["status"], "failed_optional",
        )
        self.assertFalse(failed_optional["final_eval"]["executed"])
        self.assertEqual(
            failed_optional["final_eval"]["error"],
            "RuntimeError('mock Paean failure')",
        )

    def test_outer_two_stage_result_accepts_relative_and_absolute_result_path(self):
        build = _load_function(
            "layer_importance_evaluator.py",
            "_build_ordinary_two_stage_result",
            Any=object,
            Mapping=dict,
            os=os,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            absolute_result_path = os.path.join(
                tmpdir,
                "stage1_comparator",
                "bo_rf",
                "result.json",
            )
            relative_result_path = os.path.relpath(absolute_result_path)
            stage1_binding = {
                "backend": "bo_rf",
                "seed": 42,
                "action": [0, 0],
                "gelu_degrees": [4, 4],
                "softmax_degrees": [6, 6],
                "num_layers": 2,
                "result_path": relative_result_path,
            }
            consumed_binding = {
                **stage1_binding,
                "result_path": absolute_result_path,
            }

            payload = build(
                backend="bo_rf",
                stage1_best_config={
                    "selection_binding": stage1_binding,
                },
                stage2_result={
                    "status": "smoke_only_complete",
                    "strict_feasible": False,
                    "search_backend": "bo_rf",
                    "stage1_consumed_binding": consumed_binding,
                    "selection_diagnostics": {},
                },
                final_eval_result=None,
                final_eval_status="skipped_by_request",
                final_eval_ineligible_reason=None,
                final_eval_error=None,
            )

        expected_path = os.path.abspath(relative_result_path)
        self.assertEqual(
            payload["stage1"]["selection_binding"]["result_path"],
            expected_path,
        )
        self.assertEqual(
            payload["stage2"]["consumed_stage1_binding"]["result_path"],
            expected_path,
        )

    def test_outer_two_stage_result_rejects_final_eval_for_infeasible_selection(self):
        build = _load_function(
            "layer_importance_evaluator.py",
            "_build_ordinary_two_stage_result",
            Any=object,
            Mapping=dict,
            os=os,
        )
        binding = {
            "backend": "bo_rf",
            "seed": 1729,
            "action": [0, 1],
            "gelu_degrees": [4, 2],
            "softmax_degrees": [6, 6],
            "num_layers": 2,
            "result_path": "/runs/stage1/result.json",
        }
        stage2 = {
            "status": "completed_infeasible",
            "strict_feasible": False,
            "search_backend": "bo_rf",
            "stage1_consumed_binding": binding,
            "blb_v3_best_action_group": {
                "policy_actions": [[0, 0], [1, 2]],
            },
        }

        with self.assertRaisesRegex(
            RuntimeError,
            "strict-infeasible selection cannot include final evaluation",
        ):
            build(
                backend="bo_rf",
                stage1_best_config={"selection_binding": binding},
                stage2_result=stage2,
                final_eval_result={"summary_path": "/runs/final/results.json"},
                final_eval_status="completed",
                final_eval_ineligible_reason=None,
                final_eval_error=None,
            )

    def test_outer_two_stage_publication_has_no_authority_layer(self):
        source = (ROOT / "layer_importance_evaluator.py").read_text(encoding="utf-8")
        block_start = source.index("ordinary_two_stage_payload = None")
        block_end = source.index(
            "        if ordinary_two_stage_payload is not None:",
            block_start,
        )
        block = source[block_start:block_end]

        self.assertIn("_build_ordinary_two_stage_result", block)
        self.assertIn('"two_stage_result.json"', block)
        for removed in (
            "load_completed_stage2_search_authority",
            "load_completed_paean_final_eval_result",
            "build_two_stage_authority_predicate",
            "build_authenticated_two_stage_projection",
            "publish_two_stage_search_final_result",
            "scientific_export_allowed",
            "stage1_consumed_provenance",
        ):
            self.assertNotIn(removed, block)

    def test_pending_strict_preflight_needs_no_online_completion_seal(self):
        preflight = _load_function(
            "blb_stage2_rl/sequential_runner.py",
            "_preflight_pending_strict_search_resume",
            Any=object,
            Mapping=dict,
            os=os,
            _build_search_invocation_contract=lambda **_kwargs: {
                "schema_version": "stage2_search_train_probe_invocation_v1",
                "dataset_protocol_schema": "glue_train_probe_protocol_v1",
                "dataset_protocol_hash": "probe-a",
                "search_backend": "bo_rf",
            },
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "search_bo_rf")
            os.makedirs(output_dir)
            invocation = {
                "schema_version": "stage2_search_train_probe_invocation_v1",
                "dataset_protocol_schema": "glue_train_probe_protocol_v1",
                "dataset_protocol_hash": "probe-a",
                "search_backend": "bo_rf",
            }
            resume_contract = {
                "requested_manifest": {"stage2_invocation": invocation},
            }
            context = {
                "schema_version": "stage2_pending_strict_resume_context_v2",
                "invocation_contract": invocation,
                "resume_contract": resume_contract,
                "clean_baseline_metrics": {"loss_mean": 0.4},
                "robust_reference": {"loss_limit": 0.5},
                "baseline_preflight_metrics": {"loss_mean": 0.4},
                "validation_banks": {"bank_a": {}},
                "authoritative_robust_summary": {"loss_mean": 0.4},
                "authoritative_validation_example_count": 408,
            }
            for name, payload in (
                ("invocation.json", invocation),
                (
                    "manifest.json",
                    {
                        "status": "search_complete_pending_strict",
                        "dataset_protocol_schema": (
                            "glue_train_probe_protocol_v1"
                        ),
                        "dataset_protocol_hash": "probe-a",
                        "resume_contract": resume_contract,
                    },
                ),
                ("pending_strict_resume_context.json", context),
            ):
                with open(
                    os.path.join(output_dir, name),
                    "w",
                    encoding="utf-8",
                ) as handle:
                    import json

                    json.dump(payload, handle)

            restored = preflight(
                runner=SimpleNamespace(evaluator=SimpleNamespace(
                    dataset_protocol_hash="probe-a",
                )),
                train_cfg=SimpleNamespace(
                    search_backend="bo_rf",
                    search_full_validation=True,
                ),
                fixed_gelu=np.asarray([4, 2]),
                fixed_softmax=np.asarray([6, 6]),
                fixed_label="BO-RF Stage-1",
                fixed_source="stage1_bo_rf_result",
                blb_progress_dir=tmpdir,
            )

        self.assertEqual(restored, context)

    def test_pending_strict_context_is_plain_atomic_json(self):
        writer = _load_function(
            "blb_stage2_rl/sequential_runner.py",
            "_write_pending_strict_resume_context",
            Any=object,
            Mapping=dict,
            os=os,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            writer(
                search_output_dir=tmpdir,
                invocation_contract={"search_backend": "bo_rf"},
                resume_contract={"evaluation_budget": 50_000},
                clean_baseline_metrics={"loss_mean": 0.4},
                robust_reference={"loss_limit": 0.5},
                baseline_preflight_metrics={"loss_mean": 0.4},
                validation_banks={"bank_a": {}},
                authoritative_robust_summary={"loss_mean": 0.4},
                authoritative_validation_example_count=408,
            )
            import json

            with open(
                os.path.join(tmpdir, "pending_strict_resume_context.json"),
                encoding="utf-8",
            ) as handle:
                payload = json.load(handle)

        self.assertEqual(
            payload["schema_version"],
            "stage2_pending_strict_resume_context_v2",
        )
        self.assertNotIn("context_hash", payload)


class _FakeRunLock:
    def __init__(self, _path):
        pass

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        return False


class _FakeProbeOwnerHolder:
    last = None

    def __init__(self):
        self.closed = False
        type(self).last = self

    def close(self):
        self.closed = True


if __name__ == "__main__":
    unittest.main()
