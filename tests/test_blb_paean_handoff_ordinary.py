"""Ordinary trusted-environment handoff contracts for comparator Paean F4."""

from __future__ import annotations

import ast
from collections.abc import Mapping
import copy
import json
import os
import pathlib
import tempfile
import types
import typing
import unittest
from unittest import mock

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parents[1]
_EVALUATOR_PATH = _REPO / "layer_importance_evaluator.py"
_PAEAN_PATH = _REPO / "Paean" / "blb_action_eval.py"


class FinalEvaluationSplitSourceTests(unittest.TestCase):
    def test_blb_action_run_requires_protocol_and_resolved_final_split(self):
        source = _PAEAN_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        class_node = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "BLBActionFinalEvaluationModule"
        )
        run_node = next(
            node for node in class_node.body
            if isinstance(node, ast.FunctionDef) and node.name == "run"
        )
        run_source = ast.get_source_segment(source, run_node)

        self.assertIn("require_final_evaluation_protocol(", run_source)
        self.assertIn(
            'self.final_eval_split = protocol["split_name"]', run_source
        )


def _load_class_method(path, class_name, method_name, **runtime_globals):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(
        (node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == method_name),
        None,
    )
    if method is None:
        raise AssertionError(f"{class_name}.{method_name} is missing")
    future = ast.parse("from __future__ import annotations\n").body[0]
    module = ast.Module(body=[future, method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = dict(runtime_globals)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[method_name]


def _load_evaluator_method(name, **runtime_globals):
    return _load_class_method(
        _EVALUATOR_PATH,
        "LayerImportanceEvaluator",
        name,
        **runtime_globals,
    )


def _load_paean_method(name, **runtime_globals):
    runtime_globals.setdefault("FINAL_EVAL_SPLIT", "validation_full")
    return _load_class_method(
        _PAEAN_PATH,
        "BLBActionFinalEvaluationModule",
        name,
        **runtime_globals,
    )


def _ordinary_handoff(*, backend="bo_rf", profile="mrpc"):
    action_matrix = [[0, 1], [1, 2]]
    full_vector = [0] * 147
    overrides = [
        {
            "block_idx": 2,
            "layer_idx": 0,
            "field_values": {"qk_merge_sf": 47},
        }
    ]
    return {
        "status": "completed",
        "strict_feasible": True,
        "search_backend": backend,
        "rl_variant": f"blb_v3_layerwise_search_{backend}",
        "blb_v3_profile": profile,
        "blb_v3_fusion_count_action": True,
        "blb_v3_best_action_vec": full_vector,
        "blb_v3_best_action_group": {
            "policy_actions": action_matrix,
            "option_by_step": {"0": 1},
            "boosted_overrides": overrides,
        },
        "final_config_fingerprint": "e" * 64,
    }


class Stage2OrdinaryFinalEvalHandoffTest(unittest.TestCase):
    def test_final_eval_eligibility_requires_completed_strict_feasible_result(self):
        ineligible_reason_for = _load_evaluator_method(
            "_stage2_final_eval_ineligible_reason",
            Mapping=Mapping,
        )
        evaluator = types.SimpleNamespace(blb_v3_search_backend="greedy")

        self.assertIsNone(ineligible_reason_for(evaluator, _ordinary_handoff(backend="greedy")))
        reason = ineligible_reason_for(
            evaluator,
            {
                **_ordinary_handoff(backend="greedy"),
                "status": "completed_infeasible",
                "strict_feasible": False,
            },
        )

        self.assertIn("not strict-feasible", reason)

    def test_handoff_copies_direct_selected_configuration_without_authority(self):
        build_handoff = _load_evaluator_method(
            "_build_stage2_final_eval_handoff",
            copy=copy,
            Mapping=Mapping,
            np=np,
        )
        source = _ordinary_handoff()
        evaluator = types.SimpleNamespace(dataset_key="mrpc")

        handoff = build_handoff(evaluator, source)

        self.assertEqual(handoff["status"], "completed")
        self.assertEqual(handoff["search_backend"], "bo_rf")
        self.assertEqual(handoff["rl_variant"], source["rl_variant"])
        self.assertIs(handoff["strict_feasible"], True)
        self.assertEqual(
            handoff["final_config_fingerprint"],
            source["final_config_fingerprint"],
        )
        np.testing.assert_array_equal(
            handoff["blb_v3_best_action_vec"],
            source["blb_v3_best_action_vec"],
        )
        self.assertEqual(
            handoff["blb_v3_best_action_group"],
            source["blb_v3_best_action_group"],
        )
        self.assertIsNot(
            handoff["blb_v3_best_action_group"],
            source["blb_v3_best_action_group"],
        )
        for forbidden in (
            "formal_comparator_handoff",
            "stage2_final_eval_handoff_identity",
            "materialization_context",
            "materialization_context_hash",
            "scientific_export_allowed",
        ):
            self.assertNotIn(forbidden, handoff)

    def test_handoff_rejects_backend_variant_and_profile_drift(self):
        build_handoff = _load_evaluator_method(
            "_build_stage2_final_eval_handoff",
            copy=copy,
            Mapping=Mapping,
            np=np,
        )
        evaluator = types.SimpleNamespace(dataset_key="mrpc")

        with self.assertRaisesRegex(ValueError, "rl_variant backend"):
            build_handoff(
                evaluator,
                {
                    **_ordinary_handoff(),
                    "rl_variant": "blb_v3_layerwise_search_greedy",
                },
            )
        with self.assertRaisesRegex(ValueError, "profile"):
            build_handoff(
                evaluator,
                _ordinary_handoff(profile="rte"),
            )


class PaeanOrdinaryFinalEvalHandoffTest(unittest.TestCase):
    def test_preflight_validates_plain_handoff(self):
        validate = _load_paean_method(
            "_validate_stage2_final_eval_handoff",
            Mapping=Mapping,
        )
        handoff = _ordinary_handoff()

        contract = validate(
            types.SimpleNamespace(),
            handoff,
            expected_profile="mrpc",
        )

        self.assertEqual(contract["status"], "completed")
        self.assertEqual(contract["search_backend"], "bo_rf")
        self.assertEqual(
            contract["blb_v3_best_action_vec"],
            handoff["blb_v3_best_action_vec"],
        )
        self.assertEqual(
            contract["blb_v3_best_action_group"],
            handoff["blb_v3_best_action_group"],
        )
        self.assertEqual(
            contract["final_config_fingerprint"],
            handoff["final_config_fingerprint"],
        )
        self.assertNotIn("stage2_final_eval_handoff_identity", contract)

    def test_preflight_rejects_non_strict_backend_profile_and_fingerprint_drift(self):
        validate = _load_paean_method(
            "_validate_stage2_final_eval_handoff",
            Mapping=Mapping,
        )
        module = types.SimpleNamespace()

        with self.assertRaisesRegex(ValueError, "completed"):
            validate(
                module,
                {**_ordinary_handoff(), "status": "smoke_only_complete"},
                expected_profile="mrpc",
            )
        with self.assertRaisesRegex(ValueError, "strict-feasible"):
            validate(
                module,
                {**_ordinary_handoff(), "strict_feasible": False},
                expected_profile="mrpc",
            )
        with self.assertRaisesRegex(ValueError, "backend"):
            validate(
                module,
                {
                    **_ordinary_handoff(),
                    "rl_variant": "blb_v3_layerwise_search_greedy",
                },
                expected_profile="mrpc",
            )
        with self.assertRaisesRegex(ValueError, "profile"):
            validate(
                module,
                _ordinary_handoff(profile="rte"),
                expected_profile="mrpc",
            )
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            validate(
                module,
                {**_ordinary_handoff(), "final_config_fingerprint": "bad"},
                expected_profile="mrpc",
            )

    def test_selected_candidate_must_match_plain_action_and_group(self):
        validate_handoff = _load_paean_method(
            "_validate_stage2_final_eval_handoff",
            Mapping=Mapping,
        )
        validate_candidate = _load_paean_method(
            "_validate_selected_candidate_handoff",
            Mapping=Mapping,
            np=np,
        )
        handoff = _ordinary_handoff()
        contract = validate_handoff(
            types.SimpleNamespace(),
            handoff,
            expected_profile="mrpc",
        )
        candidate = types.SimpleNamespace(
            action_vec=np.asarray(handoff["blb_v3_best_action_vec"], dtype=int),
            metadata={
                "schema_version": "fusion_count_fixed_action_v1",
                "group": copy.deepcopy(handoff["blb_v3_best_action_group"]),
            },
        )

        validate_candidate(types.SimpleNamespace(), [candidate], contract)

        candidate.metadata["group"]["boosted_overrides"] = []
        with self.assertRaisesRegex(ValueError, "boosted overrides"):
            validate_candidate(types.SimpleNamespace(), [candidate], contract)

    def test_prepared_materialization_requires_exact_selected_fingerprint(self):
        validate = _load_paean_method(
            "_validate_prepared_materialization",
            Mapping=Mapping,
        )
        contract = _ordinary_handoff()
        materialized = types.SimpleNamespace(
            model_ready=True,
            failure_reason=None,
            final_config_fingerprint="e" * 64,
        )

        evidence = validate(
            types.SimpleNamespace(),
            contract,
            materialized=materialized,
        )

        self.assertTrue(evidence["checked_before_forward"])
        self.assertTrue(evidence["final_config_fingerprint_exact_match"])
        self.assertNotIn("materialization_context", evidence)

        materialized.final_config_fingerprint = "0" * 64
        with self.assertRaisesRegex(ValueError, "config fingerprint"):
            validate(
                types.SimpleNamespace(),
                contract,
                materialized=materialized,
            )

    def test_prepared_materialization_is_reused_without_redecode_or_replan(self):
        evaluate = _load_paean_method(
            "_evaluate_action_candidate",
            np=np,
            to_jsonable=lambda value: value,
            avg_truncation_k_in_action=lambda _action, _layers: 9.0,
        )
        signals = types.SimpleNamespace(
            total_bits_sum=100,
            total_fusion_count=2,
            invalid_block_count=1,
            valid_block_count=58,
            any_invalid=True,
        )
        materialized = types.SimpleNamespace(
            action_indices=[3, 1, 4],
            decoded=types.SimpleNamespace(),
            outputs={},
            signals=signals,
            replan_application={"model_uses_replan_config": False},
            optimizer_invalid=True,
            model_ready=False,
            failure_reason="optimizer_invalid_chain",
            final_config_fingerprint="",
        )
        module = types.SimpleNamespace(
            evaluator=types.SimpleNamespace(
                total_layers=1,
                dataset_key="mrpc",
                get_simulated_cost=lambda _gelu, _softmax: (1.0, 0.5, 0.5),
            ),
            rescale_backend="in_process",
            rescale_optimizer_root="/repo/Rescale_optimizer",
            rescale_optimizer_mode="cfg_derived",
            _decode_action_candidate=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("prepared action must not be decoded again")
            ),
            _optimizer_outputs=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("prepared action must not be replanned again")
            ),
            _materialize_decoded_action=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("prepared action must not be materialized again")
            ),
            _fusion_group_diagnostics=lambda **_kwargs: {},
            _config_details=lambda *_args, **_kwargs: {},
            _build_feasibility_report=lambda **_kwargs: {
                "feasible": False,
                "diagnostic_feasible": False,
                "strict_feasible": False,
            },
        )
        consistency = {"final_config_fingerprint_exact_match": True}

        result = evaluate(
            module,
            name="selected",
            action_vec=np.asarray([3, 1, 4], dtype=int),
            overrides={},
            gelu=np.asarray([4], dtype=int),
            softmax=np.asarray([6], dtype=int),
            report_constraints={},
            max_sfs=object(),
            metadata={},
            prepared_materialized=materialized,
            materialization_consistency=consistency,
        )

        self.assertTrue(result["skipped_forward"])
        self.assertEqual(result["materialization_consistency"], consistency)
        self.assertNotIn("materialization_authority", result)

    def test_run_checks_fingerprint_before_artifacts_and_reuses_prepared_object(self):
        source = _PAEAN_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        module_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "BLBActionFinalEvaluationModule"
        )
        run_method = next(
            node for node in module_class.body if isinstance(node, ast.FunctionDef) and node.name == "run"
        )
        run_source = ast.get_source_segment(source, run_method)

        preflight_index = run_source.index("_validate_stage2_final_eval_handoff")
        fingerprint_index = run_source.index("_validate_prepared_materialization")
        artifact_index = run_source.index("os.makedirs")
        baseline_index = run_source.index("_evaluate_clean_baseline")
        self.assertLess(preflight_index, artifact_index)
        self.assertLess(fingerprint_index, artifact_index)
        self.assertLess(fingerprint_index, baseline_index)
        self.assertIn(
            "require_in_process=(final_eval_handoff is not None)",
            run_source,
        )
        self.assertIn(
            "prepared_materialized=prepared_selected_materialized",
            run_source,
        )
        self.assertNotIn("build_runtime_stage2_materialization_context", run_source)
        self.assertNotIn("load_completed_paean_final_eval_result", run_source)
        self.assertNotIn("formal_noise_seed_authority", run_source)

    def test_results_json_persists_plain_handoff_without_seal(self):
        save_results = _load_paean_method(
            "_save_results_json",
            Any=typing.Any,
            Dict=typing.Dict,
            Mapping=Mapping,
            Optional=typing.Optional,
            np=np,
            os=os,
            to_jsonable=lambda value: value,
            _atomic_json=lambda path, payload: pathlib.Path(path).write_text(json.dumps(payload), encoding="utf-8"),
        )
        handoff = _ordinary_handoff()
        with tempfile.TemporaryDirectory() as td:
            module = types.SimpleNamespace(
                evaluator=types.SimpleNamespace(dataset_key="mrpc"),
                results_dir=td,
                cost_match_count=0,
                cost_match_max_attempts=0,
                action_ranges=[],
                action_fixed=[],
                repeat_n=50,
                random_seed=42,
            )
            path = save_results(
                module,
                selected_source="selected",
                baseline_stage1_gelu=[4, 4],
                baseline_stage1_softmax=[6, 6],
                opt_gelu=[4, 4],
                opt_softmax=[6, 6],
                baseline_result={},
                candidate_results=[],
                selection_constraints={},
                final_eval_handoff=handoff,
            )
            payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))

        self.assertEqual(payload["stage2_final_eval_handoff"], handoff)
        self.assertNotIn("stage2_final_eval_handoff_identity", payload)
        self.assertNotIn("stage2_final_eval_handoff_identity_hash", payload)

    def test_install_verification_failure_aborts_before_model_forward(self):
        class FakeBridge:
            def __init__(self, _handler, *, layers_attribute):
                self.layers_attribute = layers_attribute

            def apply(self, **_kwargs):
                return None

            def clear(self):
                return None

        class FakeEvaluator:
            reversible_handler = object()
            layers_attribute = "encoder.layer"
            dataloaders = {"validation_full": object()}

            def __init__(self):
                self.eval_calls = 0

            def apply_configuration(self, _gelu, _softmax):
                return None

            def _resolve_eval_split(self, *, use_train, split):
                raise AssertionError("split resolution must not run after failed installation verification")

            def _run_evaluation(self, *_args, **_kwargs):
                self.eval_calls += 1
                raise AssertionError("model forward must not run after failed installation verification")

        run_trials = _load_paean_method(
            "_run_blb_eval_trials",
            BLBNoiseRLBridge=FakeBridge,
        )
        evaluator = FakeEvaluator()
        module = types.SimpleNamespace(
            evaluator=evaluator,
            _clear_legacy_noise=lambda: None,
            _clear_all_noise=lambda: None,
            _verify_model_installation=lambda _bridge, _decoded: {
                "checked_before_forward": True,
                "handler_cfg_objects_match_decoded_cfgs": False,
                "model_will_use_selected_cfg": False,
            },
        )
        decoded = types.SimpleNamespace(
            block1_cfgs={},
            block2_cfgs={},
            block3_cfgs={},
            block4_cfgs={},
            block5_cfgs={},
        )

        with self.assertRaisesRegex(RuntimeError, "installation verification"):
            run_trials(
                module,
                decoded,
                gelu=np.asarray([4], dtype=int),
                softmax=np.asarray([6], dtype=int),
                repeats=1,
            )

        self.assertEqual(evaluator.eval_calls, 0)


if __name__ == "__main__":
    unittest.main()
