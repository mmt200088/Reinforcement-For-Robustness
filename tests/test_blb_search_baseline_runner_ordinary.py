"""Ordinary trusted-environment contracts for the Stage-2 search runner."""

from __future__ import annotations

import inspect
import json
import os
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from blb_stage2_rl import search_baseline_runner
from blb_stage2_rl.search_baselines import (
    ConstraintLimits,
    SearchEvaluation,
    SearchMetrics,
    SearchResult,
)
from rfr.common.json_utils import stable_json_hash


def _reference():
    return SimpleNamespace(
        loss_limit=1.0,
        metric1_limit=0.5,
        metric2_limit=0.5,
        loss_std_limit=0.2,
        metric1_std_limit=0.2,
        metric2_std_limit=0.2,
    )


def _layerwise_env():
    return SimpleNamespace(
        horizon=1,
        base=SimpleNamespace(
            env_cfg=SimpleNamespace(num_trials_per_step=3),
        ),
    )


def _result():
    evaluations = tuple(
        SearchEvaluation(
            action_matrix=((index // 3, index % 3),),
            metrics=SearchMetrics(
                loss_mean=0.5,
                metric1_mean=0.8,
                metric2_mean=0.7,
                loss_std=0.1,
                metric1_std=0.1,
                metric2_std=0.1,
            ),
            limits=ConstraintLimits(
                loss_max=1.0,
                metric1_min=0.5,
                metric2_min=0.5,
                loss_std_max=0.2,
                metric1_std_max=0.2,
                metric2_std_max=0.2,
            ),
            reward=float(index),
            communication_importance_ratio=1.0,
            metadata={
                "inference_performed": True,
                "materializable": True,
                "pending_full_vector": [index + 1],
                "boosted_overrides": [],
                "final_config_fingerprint": f"{index + 1:064x}",
                "statistical_assessment": {"bootstrap_seed": 42 + index},
                "trial_seeds": [1, 2, 3],
            },
        )
        for index in range(5)
    )
    return SearchResult(
        algorithm="greedy",
        best=max(
            evaluations,
            key=search_baseline_runner.candidate_rank_key,
        ),
        observations=evaluations,
        history=(),
        termination_reason="verified_local_optima",
    )


def _fake_run_search(_backend, _space, evaluator, _config, *, preload=()):
    result = _result()
    if preload:
        return result
    if evaluator.on_evaluation is not None:
        for item in result.observations:
            evaluator.on_evaluation(item.as_dict())
    return result


def _strict_payload(result):
    violations = {
        "families": {
            name: {
                "available": True,
                "point_pass": True,
                "trial_count": 45,
                "banks_run": ["A", "B", "C"],
                "not_run_banks": [],
            }
            for name in ("joint", "compute_only", "communication_only")
        },
        "aggregate": {
            "failed_constraint_count": 0,
            "total_normalized_violation": 0.0,
            "worst_normalized_violation": 0.0,
            "unavailable_family_count": 0,
        },
    }
    ranked = sorted(
        result.observations,
        key=search_baseline_runner.candidate_rank_key,
        reverse=True,
    )[:5]
    records = []
    strict_evaluations = []
    for online in ranked:
        strict_evaluation = SearchEvaluation.from_dict(
            {
                **online.as_dict(),
                "metadata": {
                    **online.metadata,
                    "strict_trial_count": 45,
                    "strict_final_assessment": {
                        name: 0.99
                        for name in (
                            "loss_precision_probability",
                            "metric1_precision_probability",
                            "metric2_precision_probability",
                            "loss_stability_probability",
                            "metric1_stability_probability",
                            "metric2_stability_probability",
                        )
                    },
                    "strict_candidate_key": (
                        f"{int(online.metadata['pending_full_vector'][0]):064x}"
                    ),
                    "strict_violations": violations,
                },
            }
        )
        strict_evaluations.append(strict_evaluation)
        records.append(
            {
                "online_candidate": online.as_dict(),
                "strict_evaluated": True,
                "selection_eligible": True,
                "strict_point_pass": True,
                "strict_feasible": True,
                "strict_trial_count": 45,
                "strict_evaluation": strict_evaluation.as_dict(),
                "violations": violations,
            }
        )
    selected_evaluation = min(
        strict_evaluations,
        key=search_baseline_runner._strict_selected_rank,
    )
    selected = {
        **selected_evaluation.as_dict(),
        "selection_status": "strict_feasible",
        "strict_feasible": True,
        "violations": violations,
    }
    return {
        "schema_version": "stage2_search_strict_validation_v3",
        "requested_top_n": 5,
        "strict_evaluated_candidate_count": 5,
        "selection_status": "strict_feasible",
        "strict_feasible": True,
        "selected_violations": violations,
        "selected": selected,
        "records": records,
    }


def _run_kwargs(output_dir, **overrides):
    values = {
        "backend": "greedy",
        "layerwise_env": _layerwise_env(),
        "robust_reference": _reference(),
        "output_dir": output_dir,
        "evaluation_budget": 6,
        "seed": 42,
        "initial_design_size": 64,
        "candidate_pool_size": 2048,
        "population_size": 64,
        "patience_generations": 100,
        "mutation_max_coordinates": 4,
        "rf_n_estimators": 128,
        "rf_min_samples_leaf": 2,
        "communication_importance_ratio": 1.0,
        "manifest": {
            "backend": "greedy",
            "dataset_protocol_schema": "glue_train_probe_protocol_v1",
            "dataset_protocol_hash": "probe-a",
            "search_split": "train_probe",
        },
    }
    values.update(overrides)
    return values


def _protocol_manifest(**overrides):
    payload = {
        "dataset_protocol_schema": "glue_train_probe_protocol_v1",
        "dataset_protocol_hash": "probe-a",
        "search_split": "train_probe",
    }
    payload.update(overrides)
    return payload


class OrdinaryRunnerApiTest(unittest.TestCase):
    def test_resume_rejects_missing_protocol_before_search_replay(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(
                search_baseline_runner,
                "run_search",
                side_effect=_fake_run_search,
            ):
                search_baseline_runner.run_layerwise_search_baseline(
                    **_run_kwargs(tmpdir)
                )
            manifest_path = os.path.join(tmpdir, "manifest.json")
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
            manifest.pop("dataset_protocol_schema", None)
            manifest.pop("dataset_protocol_hash", None)
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle)

            with mock.patch.object(
                search_baseline_runner,
                "run_search",
                side_effect=AssertionError("stale run must not replay"),
            ), self.assertRaisesRegex(RuntimeError, "train-probe protocol"):
                search_baseline_runner.run_layerwise_search_baseline(
                    **_run_kwargs(tmpdir)
                )

    def test_runtime_evaluator_has_no_physical_trial_wal_parameters(self):
        parameters = inspect.signature(search_baseline_runner.LayerwiseRuntimeEvaluator.__init__).parameters

        self.assertNotIn("candidate_store", parameters)
        self.assertNotIn("identity_context", parameters)
        self.assertNotIn("physical_trial_invocation_hash", parameters)
        self.assertNotIn("materialization_context_hash", parameters)

    def test_selected_action_identity_uses_direct_materialized_configuration(self):
        evaluation = SearchEvaluation(
            action_matrix=((1, 2),),
            metrics=SearchMetrics(
                loss_mean=0.5,
                metric1_mean=0.8,
                metric2_mean=0.7,
                loss_std=0.1,
                metric1_std=0.1,
                metric2_std=0.1,
            ),
            limits=ConstraintLimits(
                loss_max=1.0,
                metric1_min=0.5,
                metric2_min=0.5,
                loss_std_max=0.2,
                metric1_std_max=0.2,
                metric2_std_max=0.2,
            ),
            reward=1.0,
            metadata={
                "pending_full_vector": [3, 4],
                "boosted_overrides": [{"layer": 0, "block": 4}],
                "final_config_fingerprint": "a" * 64,
            },
        )

        try:
            identity = search_baseline_runner._selected_action_identity_payload(evaluation)
        except RuntimeError as exc:
            self.fail(f"direct selected configuration was rejected: {exc}")

        self.assertEqual(identity["action_matrix"], [[1, 2]])
        self.assertEqual(identity["full_vector"], [3, 4])
        self.assertEqual(identity["final_config_fingerprint"], "a" * 64)
        self.assertNotIn("materialization_context_hash", identity)
        self.assertNotIn("axis_materialization_authority", identity)

    def test_strict_materialization_uses_direct_counterfactual_path_only(self):
        module_source = inspect.getsource(search_baseline_runner)
        direct_source = inspect.getsource(search_baseline_runner._prepare_strict_materialization_fingerprints)

        self.assertNotIn("materialization_context", module_source)
        for name in (
            "_authenticated_strict_materialization_inputs",
            "_canonical_strict_record_materializations",
            "_reauthenticate_completed_strict_materializations",
            "_validate_strict_physical_trial_coverage",
        ):
            self.assertFalse(hasattr(search_baseline_runner, name))
        self.assertIn("materialize_layerwise_counterfactuals", direct_source)
        self.assertIn("prepare_action_for_terminal_probe", direct_source)

    def test_strict_validation_has_no_physical_invocation_parameter(self):
        parameters = inspect.signature(search_baseline_runner.canonical_strict_validation).parameters

        self.assertNotIn("physical_trial_invocation_hash", parameters)

    def test_strict_validation_requires_exact_top_five(self):
        with self.assertRaisesRegex(ValueError, "exactly top 5"):
            search_baseline_runner.canonical_strict_validation(
                result=None,
                layerwise_env=None,
                promotion_base_env=None,
                candidate_store=None,
                identity_context={},
                validation_banks=None,
                top_n=4,
                communication_importance_ratio=1.0,
                promotion_probability=0.95,
                final_probability=0.99,
            )

    def test_search_runner_has_no_online_physical_wal_parameters(self):
        parameters = inspect.signature(search_baseline_runner.run_layerwise_search_baseline).parameters

        self.assertNotIn("online_candidate_store", parameters)
        self.assertNotIn("online_identity_context", parameters)
        self.assertNotIn("physical_trial_invocation_hash", parameters)

    def test_rejects_non_five_candidate_payload(self):
        result = _result()
        payload = _strict_payload(result)
        payload["requested_top_n"] = 4

        with self.assertRaisesRegex(RuntimeError, "exactly five"):
            search_baseline_runner._validate_strict_validation_payload(
                result=result,
                payload=payload,
                communication_importance_ratio=1.0,
            )

    def test_rejects_selected_candidate_that_is_not_the_scientific_rank_winner(self):
        result = _result()
        payload = _strict_payload(result)
        payload["selected"] = {
            **payload["records"][-1]["strict_evaluation"],
            "selection_status": "strict_feasible",
            "strict_feasible": True,
            "violations": payload["selected_violations"],
        }

        with self.assertRaisesRegex(RuntimeError, "selected strict candidate"):
            search_baseline_runner._validate_strict_validation_payload(
                result=result,
                payload=payload,
                communication_importance_ratio=1.0,
            )

    def test_rejects_missing_counterfactual_family(self):
        result = _result()
        payload = _strict_payload(result)
        del payload["records"][0]["violations"]["families"]["communication_only"]

        with self.assertRaisesRegex(RuntimeError, "constraint families"):
            search_baseline_runner._validate_strict_validation_payload(
                result=result,
                payload=payload,
                communication_importance_ratio=1.0,
            )

    def test_rejects_selected_materialized_configuration_drift(self):
        result = _result()
        payload = _strict_payload(result)
        payload["selected"]["metadata"]["pending_full_vector"] = [999]

        with self.assertRaisesRegex(RuntimeError, "selected strict candidate"):
            search_baseline_runner._validate_strict_validation_payload(
                result=result,
                payload=payload,
                communication_importance_ratio=1.0,
            )


class OrdinaryRunnerPersistenceTest(unittest.TestCase):
    def test_smoke_run_uses_plain_artifacts_without_seals_or_wal_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(
                search_baseline_runner,
                "run_search",
                side_effect=_fake_run_search,
            ):
                completed = search_baseline_runner.run_layerwise_search_baseline(**_run_kwargs(tmpdir))

            self.assertEqual(completed["manifest"]["status"], "smoke_only_complete")
            self.assertFalse(os.path.exists(os.path.join(tmpdir, "completion_seal.json")))
            self.assertFalse(os.path.exists(os.path.join(tmpdir, "online_completion_seal.json")))
            self.assertFalse(os.path.exists(os.path.join(tmpdir, "strict_attempts.jsonl")))
            manifest_keys = set(completed["manifest"])
            self.assertFalse(any("physical_trial" in key for key in manifest_keys))
            self.assertNotIn("formal_feasible", manifest_keys)
            self.assertNotIn("scientific_export_allowed", manifest_keys)

    def test_completed_plain_resume_runs_no_search_or_strict_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(
                search_baseline_runner,
                "run_search",
                side_effect=_fake_run_search,
            ):
                first = search_baseline_runner.run_layerwise_search_baseline(**_run_kwargs(tmpdir))

            with mock.patch.object(
                search_baseline_runner,
                "run_search",
                side_effect=AssertionError("completed resume reran search"),
            ):
                resumed = search_baseline_runner.run_layerwise_search_baseline(**_run_kwargs(tmpdir))

            self.assertEqual(resumed["result"].as_dict(), first["result"].as_dict())
            self.assertEqual(resumed["manifest"]["status"], "smoke_only_complete")

    def test_strict_completion_uses_scientific_feasibility_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(
                search_baseline_runner,
                "run_search",
                side_effect=_fake_run_search,
            ):
                completed = search_baseline_runner.run_layerwise_search_baseline(
                    **_run_kwargs(tmpdir, strict_validator=_strict_payload)
                )

            self.assertEqual(
                completed["manifest"]["status"],
                "complete_strict_feasible",
            )
            self.assertTrue(completed["manifest"]["strict_feasible"])
            self.assertNotIn("formal_feasible", completed["manifest"])
            with open(
                os.path.join(tmpdir, "strict_validation.json"),
                encoding="utf-8",
            ) as handle:
                strict_payload = json.load(handle)
            self.assertTrue(strict_payload["strict_feasible"])
            self.assertNotIn("formal_feasible", strict_payload)

    def test_completed_resume_rejects_stale_strict_selection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(
                search_baseline_runner,
                "run_search",
                side_effect=_fake_run_search,
            ):
                search_baseline_runner.run_layerwise_search_baseline(
                    **_run_kwargs(
                        tmpdir,
                        strict_validator=_strict_payload,
                    )
                )

            selected_path = os.path.join(
                tmpdir,
                "final_selected_configuration.json",
            )
            with open(selected_path, encoding="utf-8") as handle:
                selected = json.load(handle)
            selected["reward"] = float(selected["reward"]) + 1.0
            with open(selected_path, "w", encoding="utf-8") as handle:
                json.dump(selected, handle)

            with mock.patch.object(
                search_baseline_runner,
                "run_search",
                side_effect=AssertionError("stale resume reran search"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "selected configuration does not match strict validation",
                ):
                    search_baseline_runner.run_layerwise_search_baseline(
                        **_run_kwargs(
                            tmpdir,
                            strict_validator=_strict_payload,
                        )
                    )

    def test_completed_resume_rejects_strict_verdict_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(
                search_baseline_runner,
                "run_search",
                side_effect=_fake_run_search,
            ):
                search_baseline_runner.run_layerwise_search_baseline(
                    **_run_kwargs(
                        tmpdir,
                        strict_validator=_strict_payload,
                    )
                )

            manifest_path = os.path.join(tmpdir, "manifest.json")
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
            manifest["strict_feasible"] = False
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle)

            with mock.patch.object(
                search_baseline_runner,
                "run_search",
                side_effect=AssertionError("verdict resume reran search"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "strict verdict does not match manifest",
                ):
                    search_baseline_runner.run_layerwise_search_baseline(
                        **_run_kwargs(
                            tmpdir,
                            strict_validator=_strict_payload,
                        )
                    )

    def test_failed_strict_attempt_rolls_back_store_before_full_retry(self):
        recover_calls = []

        class FakeCandidateStore:
            def __init__(self, path):
                self.path = os.fspath(path)

            def recover_to_checkpoint_size(self, committed_size):
                recover_calls.append(int(committed_size))
                with open(self.path, "r+b") as handle:
                    handle.truncate(int(committed_size))

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = os.path.join(tmpdir, "strict_candidate_store.jsonl")
            with open(store_path, "wb") as handle:
                handle.write(b"base\n")

            def failing_strict_validator(_result):
                with open(store_path, "ab") as handle:
                    handle.write(b"partial-strict-evidence\n")
                raise RuntimeError("strict interruption")

            manifest = _protocol_manifest(
                backend="greedy",
                strict_candidate_store=store_path,
            )
            with (
                mock.patch.object(
                    search_baseline_runner,
                    "CandidateStore",
                    FakeCandidateStore,
                ),
                mock.patch.object(
                    search_baseline_runner,
                    "run_search",
                    side_effect=_fake_run_search,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "strict interruption"):
                    search_baseline_runner.run_layerwise_search_baseline(
                        **_run_kwargs(
                            tmpdir,
                            manifest=manifest,
                            strict_validator=failing_strict_validator,
                        )
                    )

            with open(store_path, "rb") as handle:
                self.assertEqual(handle.read(), b"base\n")
            with open(
                os.path.join(tmpdir, "manifest.json"),
                encoding="utf-8",
            ) as handle:
                failed_manifest = json.load(handle)
            self.assertEqual(
                failed_manifest["status"],
                "search_complete_pending_strict",
            )
            self.assertEqual(
                failed_manifest["strict_candidate_store_checkpoint_size"],
                len(b"base\n"),
            )

            with (
                mock.patch.object(
                    search_baseline_runner,
                    "CandidateStore",
                    FakeCandidateStore,
                ),
                mock.patch.object(
                    search_baseline_runner,
                    "run_search",
                    side_effect=AssertionError("strict retry reran online search"),
                ),
            ):
                completed = search_baseline_runner.run_layerwise_search_baseline(
                    **_run_kwargs(
                        tmpdir,
                        manifest=manifest,
                        strict_validator=_strict_payload,
                    )
                )

            self.assertEqual(
                completed["manifest"]["status"],
                "complete_strict_feasible",
            )
            self.assertGreaterEqual(len(recover_calls), 2)
            self.assertTrue(all(value == len(b"base\n") for value in recover_calls))

    def test_complete_strict_artifact_republishes_without_revalidation(self):
        original_atomic_json = search_baseline_runner._atomic_json
        crashed = False

        def crash_before_completed_manifest(path, payload):
            nonlocal crashed
            if (
                not crashed
                and os.path.basename(path) == "manifest.json"
                and payload.get("status") == "complete_strict_feasible"
            ):
                crashed = True
                raise RuntimeError("crash before completed manifest")
            return original_atomic_json(path, payload)

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                mock.patch.object(
                    search_baseline_runner,
                    "run_search",
                    side_effect=_fake_run_search,
                ),
                mock.patch.object(
                    search_baseline_runner,
                    "_atomic_json",
                    side_effect=crash_before_completed_manifest,
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "crash before completed manifest",
                ):
                    search_baseline_runner.run_layerwise_search_baseline(
                        **_run_kwargs(
                            tmpdir,
                            strict_validator=_strict_payload,
                        )
                    )

            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "strict_validation.json")))
            with mock.patch.object(
                search_baseline_runner,
                "run_search",
                side_effect=AssertionError("strict republish reran search"),
            ):
                completed = search_baseline_runner.run_layerwise_search_baseline(
                    **_run_kwargs(
                        tmpdir,
                        strict_validator=lambda _result: (_ for _ in ()).throw(
                            AssertionError("complete strict artifact was rerun")
                        ),
                    )
                )

            self.assertEqual(
                completed["manifest"]["status"],
                "complete_strict_feasible",
            )
            self.assertTrue(completed["strict_feasible"])


if __name__ == "__main__":
    unittest.main()
