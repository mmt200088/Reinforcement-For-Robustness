from __future__ import annotations

import json
import os
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from blb_stage2_rl.reward import EpisodeMetrics
from blb_stage2_rl.search_baseline_runner import (
    LayerwiseRuntimeEvaluator,
    canonical_strict_validation,
    limits_from_reference,
    persist_search_result,
    run_layerwise_search_baseline,
)
from blb_stage2_rl.search_baselines import (
    ConstraintLimits,
    LayerwiseSearchSpace,
    SearchConfig,
    SearchEvaluation,
    SearchMetrics,
    SearchResult,
    candidate_rank_key,
    run_search,
)


class _Reference:
    loss_limit = 1.01
    metric1_limit = 0.89
    metric2_limit = 0.84
    loss_std_limit = 0.02
    metric1_std_limit = 0.015
    metric2_std_limit = 0.018


class _BaseEnv:
    def __init__(self):
        self.probe_noise_seed = None
        self.clear_count = 0
        self.env_cfg = SimpleNamespace(num_trials_per_step=3)

    def clear_installed_blb(self):
        self.clear_count += 1


class _LayerwiseEnv:
    horizon = 2
    communication_importance_ratio = 1.0

    def __init__(self, *, forward_ran=True, model_uses_replan=True):
        self.base = _BaseEnv()
        self.forward_ran = bool(forward_ran)
        self.model_uses_replan = bool(model_uses_replan)
        self.rows = []
        self.runtime_terminal_info = None
        self.boosted_overrides = {(4, 0): {"slot": 47}}
        self.pending_full_vector = [1, 2, 3]

    def reset(self, *, seed=None):
        self.rows = []
        self.runtime_terminal_info = None
        self.reset_seed = seed
        return [0.0]

    def step(self, row):
        self.rows.append([int(value) for value in row])
        done = len(self.rows) == self.horizon
        if not done:
            return [0.0], 0.0, False, {"layer_idx": len(self.rows) - 1}
        self.runtime_terminal_info = {
            "metrics": EpisodeMetrics(
                loss_mean=1.0,
                metric1_mean=0.90,
                metric2_mean=0.85,
                loss_std=0.01,
                metric1_std=0.01,
                metric2_std=0.01,
                loss_trials=(0.99, 1.01, 1.0),
                metric1_trials=(0.89, 0.90, 0.91),
                metric2_trials=(0.84, 0.85, 0.86),
                trial_seeds=(11, 12, 13),
            ),
            "invalid": False,
            "forward_ran": self.forward_ran,
            "replan_application": {
                "model_uses_replan_config": self.model_uses_replan,
            },
            "statistical_assessment": {
                "loss_precision_probability": 0.91,
                "metric1_precision_probability": 0.92,
                "metric2_precision_probability": 0.93,
                "loss_stability_probability": 0.81,
                "metric1_stability_probability": 0.82,
                "metric2_stability_probability": 0.83,
                "precision_probability": 0.9,
                "stability_probability": 0.8,
                "gate_probability": 0.5,
                "online_precision_pass": True,
                "online_stability_pass": True,
                "bootstrap_seed": 1234,
            },
            "reward_breakdown": {"priority": 3},
        }
        return [0.0], 1.25, True, {
            "pending_full_vector": [1, 2, 3],
        }


class _SeededLayerwiseEnv(_LayerwiseEnv):
    def step(self, row):
        state, reward, done, info = super().step(row)
        if done:
            probe_seed = int(self.base.probe_noise_seed)
            metrics = self.runtime_terminal_info["metrics"]
            self.runtime_terminal_info["metrics"] = EpisodeMetrics(
                loss_mean=metrics.loss_mean,
                metric1_mean=metrics.metric1_mean,
                metric2_mean=metrics.metric2_mean,
                loss_std=metrics.loss_std,
                metric1_std=metrics.metric1_std,
                metric2_std=metrics.metric2_std,
                loss_trials=metrics.loss_trials,
                metric1_trials=metrics.metric1_trials,
                metric2_trials=metrics.metric2_trials,
                trial_seeds=tuple(probe_seed + index for index in range(3)),
            )
        return state, reward, done, info


class _InvalidCandidateLayerwiseEnv(_LayerwiseEnv):
    def step(self, row):
        state, reward, done, info = super().step(row)
        if done and self.rows[0][0] == 1:
            self.runtime_terminal_info.update({
                "invalid": True,
                "forward_ran": False,
                "materialization_failure_reason": "optimizer_invalid_chain",
                "forward_skipped_reason": "optimizer_invalid_chain",
                "optimizer_invalid_summary": "block4 invalid_chain",
            })
        return state, reward, done, info


def _search_evaluation(
        action_matrix,
        *,
        metric1_mean=0.90,
        valid=True,
        materializable=True,
        ):
    return SearchEvaluation(
        action_matrix=action_matrix,
        metrics=SearchMetrics(
            loss_mean=1.0,
            metric1_mean=metric1_mean,
            metric2_mean=0.85,
            loss_std=0.01,
            metric1_std=0.01,
            metric2_std=0.01,
        ),
        limits=ConstraintLimits(
            loss_max=1.01,
            metric1_min=0.89,
            metric2_min=0.84,
            loss_std_max=0.02,
            metric1_std_max=0.015,
            metric2_std_max=0.018,
        ),
        valid=valid,
        reward=1.0,
        metadata={
            "pending_full_vector": [
                value for row in action_matrix for value in row
            ] if materializable else [],
            "boosted_overrides": [],
            "statistical_assessment": {"bootstrap_seed": 1234},
            "materializable": bool(materializable),
        },
    )


def _search_result(*evaluations):
    return SearchResult(
        algorithm="greedy",
        best=max(evaluations, key=candidate_rank_key),
        observations=tuple(evaluations),
        history=(),
        termination_reason="test",
    )


def _promotion_result(
        *, status, trial_count, metrics, fresh_trial_count=0,
        assessment=None, axis_counterfactuals=None,
        ):
    return SimpleNamespace(
        status=status,
        trial_count=trial_count,
        fresh_trial_count=fresh_trial_count,
        metrics=metrics,
        assessment=assessment,
        axis_counterfactuals=axis_counterfactuals,
    )


def _validation_banks():
    return SimpleNamespace(
        bank_a=SimpleNamespace(trial_count=15),
        promotion_trial_count=30,
        final_trial_count=45,
        final_reference=_Reference(),
        contract_payload=lambda: {"hard_gate": "canonical"},
    )


class RuntimeEvaluatorTests(unittest.TestCase):
    def test_real_layerwise_path_yields_all_six_metrics_and_audit_fields(self):
        env = _LayerwiseEnv()
        callback_rows = []
        evaluator = LayerwiseRuntimeEvaluator(
            env=env,
            reference=_Reference(),
            base_seed=17,
            expected_trials=3,
            on_evaluation=callback_rows.append,
        )

        result = evaluator(((1, 2), (0, 1)))

        self.assertTrue(result.feasible)
        self.assertEqual(env.rows, [[1, 2], [0, 1]])
        self.assertEqual(result.metrics.metric2_mean, 0.85)
        self.assertEqual(result.metrics.metric2_std, 0.01)
        self.assertTrue(result.metadata["forward_ran"])
        self.assertTrue(result.metadata["model_uses_replan_config"])
        self.assertEqual(result.metadata["trial_seeds"], [11, 12, 13])
        self.assertEqual(
            result.constraint_probabilities,
            (0.91, 0.92, 0.93, 0.81, 0.82, 0.83),
        )
        self.assertEqual(result.gate_probability, 0.5)
        self.assertEqual(result.metadata["bootstrap_seed"], 1234)
        self.assertEqual(
            result.metadata["boosted_overrides"],
            [{"block_idx": 4, "layer_idx": 0, "field_values": {"slot": 47}}],
        )
        self.assertEqual(len(callback_rows), 1)
        self.assertGreaterEqual(env.base.clear_count, 1)

    def test_missing_real_forward_fails_closed(self):
        evaluator = LayerwiseRuntimeEvaluator(
            env=_LayerwiseEnv(forward_ran=False),
            reference=_Reference(),
            base_seed=17,
            expected_trials=3,
        )

        with self.assertRaisesRegex(RuntimeError, "forward"):
            evaluator(((0, 0), (0, 0)))

    def test_missing_replan_install_fails_closed(self):
        evaluator = LayerwiseRuntimeEvaluator(
            env=_LayerwiseEnv(model_uses_replan=False),
            reference=_Reference(),
            base_seed=17,
            expected_trials=3,
        )

        with self.assertRaisesRegex(RuntimeError, "replan"):
            evaluator(((0, 0), (0, 0)))

    def test_action_keyed_probe_seed_is_order_independent(self):
        env = _SeededLayerwiseEnv()
        evaluator = LayerwiseRuntimeEvaluator(
            env=env,
            reference=_Reference(),
            base_seed=17,
            expected_trials=3,
        )
        action_a = ((0, 0), (0, 0))
        action_b = ((1, 2), (1, 2))

        first_a = evaluator(action_a)
        observed_b = evaluator(action_b)
        second_a = evaluator(action_a)

        self.assertEqual(
            first_a.metadata["trial_seeds"],
            second_a.metadata["trial_seeds"],
        )
        self.assertEqual(
            first_a.metadata["probe_seed"],
            second_a.metadata["probe_seed"],
        )
        self.assertNotEqual(
            first_a.metadata["trial_seeds"],
            observed_b.metadata["trial_seeds"],
        )

    def test_optimizer_invalid_candidate_returns_invalid_and_search_continues(self):
        env = _InvalidCandidateLayerwiseEnv()
        callback_rows = []
        evaluator = LayerwiseRuntimeEvaluator(
            env=env,
            reference=_Reference(),
            base_seed=17,
            expected_trials=3,
            on_evaluation=callback_rows.append,
        )

        invalid = evaluator(((1, 0), (0, 0)))
        valid = evaluator(((0, 0), (0, 0)))

        self.assertFalse(invalid.valid)
        self.assertFalse(invalid.feasible)
        self.assertFalse(invalid.metadata["inference_performed"])
        self.assertFalse(invalid.metadata["materializable"])
        self.assertTrue(valid.valid)
        self.assertEqual(evaluator.evaluation_count, 2)
        self.assertEqual(len(callback_rows), 2)

    def test_invalid_candidate_with_eval_failure_is_infrastructure_error(self):
        env = _InvalidCandidateLayerwiseEnv()
        original_step = env.step

        def failed_step(row):
            state, reward, done, info = original_step(row)
            if done:
                env.runtime_terminal_info["eval_failed"] = True
                env.runtime_terminal_info["error"] = "cuda failure"
            return state, reward, done, info

        env.step = failed_step
        evaluator = LayerwiseRuntimeEvaluator(
            env=env,
            reference=_Reference(),
            base_seed=17,
            expected_trials=3,
        )

        with self.assertRaisesRegex(RuntimeError, "infrastructure"):
            evaluator(((1, 0), (0, 0)))

    def test_persistence_keeps_manifest_observations_history_and_summary(self):
        env = _LayerwiseEnv()
        evaluation_rows = []
        evaluator = LayerwiseRuntimeEvaluator(
            env=env,
            reference=_Reference(),
            base_seed=17,
            expected_trials=3,
            on_evaluation=evaluation_rows.append,
        )
        result = run_search(
            "greedy",
            LayerwiseSearchSpace(2),
            evaluator,
            SearchConfig(evaluation_budget=3, seed=17),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = persist_search_result(
                output_dir=tmpdir,
                result=result,
                manifest={"profile": "mrpc", "scientific_status": "smoke"},
                observation_rows=evaluation_rows,
            )

            self.assertEqual(
                set(paths),
                {"manifest", "observations", "history", "summary"},
            )
            for path in paths.values():
                self.assertTrue(os.path.isfile(path), path)
            with open(paths["observations"], encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            self.assertEqual(len(rows), result.evaluation_count)
            with open(paths["summary"], encoding="utf-8") as handle:
                summary = json.load(handle)
            self.assertEqual(summary["best"]["metrics"]["metric2_mean"], 0.85)
            self.assertEqual(
                summary["best"]["metadata"]["installed_action"]["layers"][0][
                    "precision_preset_name"
                ],
                result.best.metadata["installed_action"]["layers"][0][
                    "precision_preset_name"
                ],
            )

    def test_reference_limits_include_all_six_channels(self):
        limits = limits_from_reference(_Reference())

        self.assertEqual(limits.loss_max, 1.01)
        self.assertEqual(limits.metric2_min, 0.84)
        self.assertEqual(limits.metric2_std_max, 0.018)

    def test_complete_greedy_run_persists_crash_recoverable_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run = run_layerwise_search_baseline(
                backend="greedy",
                layerwise_env=_LayerwiseEnv(),
                robust_reference=_Reference(),
                output_dir=tmpdir,
                evaluation_budget=2,
                seed=17,
                initial_design_size=2,
                candidate_pool_size=8,
                population_size=4,
                patience_generations=5,
                mutation_max_coordinates=1,
                rf_n_estimators=8,
                rf_min_samples_leaf=1,
                communication_importance_ratio=1.0,
                manifest={"profile": "mrpc", "scientific_status": "smoke"},
            )

            self.assertEqual(run["result"].evaluation_count, 2)
            self.assertEqual(
                run["manifest"]["status"], "smoke_only_complete",
            )
            self.assertFalse(run["scientific_export_allowed"])
            self.assertEqual(
                run["manifest"]["scientific_status"],
                "smoke_only_no_validation_full_gate",
            )
            self.assertEqual(
                run["manifest"]["search_config"]["population_size"], 4,
            )
            with open(
                    run["artifact_paths"]["observations"],
                    encoding="utf-8",
            ) as handle:
                self.assertEqual(
                    sum(1 for line in handle if line.strip()),
                    2,
                )

    def test_completed_run_resumes_without_reinference(self):
        kwargs = {
            "backend": "greedy",
            "robust_reference": _Reference(),
            "evaluation_budget": 2,
            "seed": 17,
            "initial_design_size": 2,
            "candidate_pool_size": 8,
            "population_size": 4,
            "patience_generations": 5,
            "mutation_max_coordinates": 1,
            "rf_n_estimators": 8,
            "rf_min_samples_leaf": 1,
            "communication_importance_ratio": 1.0,
            "manifest": {"profile": "mrpc"},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            first_env = _LayerwiseEnv()
            first = run_layerwise_search_baseline(
                layerwise_env=first_env,
                output_dir=tmpdir,
                **kwargs,
            )
            self.assertGreater(first_env.base.clear_count, 0)

            second_env = _LayerwiseEnv()
            second = run_layerwise_search_baseline(
                layerwise_env=second_env,
                output_dir=tmpdir,
                **kwargs,
            )
            self.assertTrue(second["resumed_completed_run"])
            self.assertEqual(second_env.base.clear_count, 0)
            self.assertEqual(
                second["selected"].action_matrix,
                first["selected"].action_matrix,
            )
            self.assertEqual(
                second["result"].evaluation_count,
                first["result"].evaluation_count,
            )

    def test_strict_phase_resume_does_not_repeat_online_search(self):
        def completed_strict(result):
            selected = result.best.as_dict()
            selected["metadata"] = {
                **selected["metadata"],
                "strict_trial_count": 45,
            }
            return {
                "schema_version": "stage2_search_strict_validation_v2",
                "selection_status": "strict_feasible",
                "formal_feasible": True,
                "selected": selected,
                "records": [{
                    "strict_evaluated": True,
                    "strict_trial_count": 45,
                }],
            }

        kwargs = {
            "backend": "greedy",
            "robust_reference": _Reference(),
            "evaluation_budget": 36,
            "seed": 17,
            "initial_design_size": 2,
            "candidate_pool_size": 8,
            "population_size": 4,
            "patience_generations": 5,
            "mutation_max_coordinates": 1,
            "rf_n_estimators": 8,
            "rf_min_samples_leaf": 1,
            "communication_importance_ratio": 1.0,
            "manifest": {"profile": "mrpc"},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            first_env = _LayerwiseEnv()
            with self.assertRaisesRegex(RuntimeError, "strict interrupted"):
                run_layerwise_search_baseline(
                    layerwise_env=first_env,
                    output_dir=tmpdir,
                    strict_validator=lambda _result: (_ for _ in ()).throw(
                        RuntimeError("strict interrupted")
                    ),
                    **kwargs,
                )
            self.assertGreater(first_env.base.clear_count, 0)
            with open(
                    os.path.join(tmpdir, "manifest.json"),
                    encoding="utf-8",
            ) as handle:
                self.assertEqual(
                    json.load(handle)["status"],
                    "search_complete_pending_strict",
                )

            second_env = _LayerwiseEnv()
            resumed = run_layerwise_search_baseline(
                layerwise_env=second_env,
                output_dir=tmpdir,
                strict_validator=completed_strict,
                **kwargs,
            )
            self.assertEqual(second_env.base.clear_count, 0)
            self.assertTrue(resumed["scientific_export_allowed"])
            self.assertEqual(
                resumed["manifest"]["strict_trial_count"], 45,
            )

    def test_partial_bo_and_ga_resume_fail_closed_without_exact_state(self):
        for backend, budget in (("bo_rf", 2), ("coinn_ga", 804)):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as tmpdir:
                kwargs = {
                    "backend": backend,
                    "layerwise_env": _LayerwiseEnv(),
                    "robust_reference": _Reference(),
                    "output_dir": tmpdir,
                    "evaluation_budget": budget,
                    "seed": 17,
                    "initial_design_size": 2,
                    "candidate_pool_size": 8,
                    "population_size": 4,
                    "patience_generations": 5,
                    "mutation_max_coordinates": 4,
                    "rf_n_estimators": 8,
                    "rf_min_samples_leaf": 1,
                    "communication_importance_ratio": 1.0,
                    "manifest": {"profile": "mrpc"},
                }

                def interrupted_search(_backend, space, evaluator, _config, **_kwargs):
                    evaluator(space.safe_action)
                    raise RuntimeError("online interrupted")

                with patch(
                        "blb_stage2_rl.search_baseline_runner.run_search",
                        side_effect=interrupted_search,
                ):
                    with self.assertRaisesRegex(RuntimeError, "online interrupted"):
                        run_layerwise_search_baseline(**kwargs)

                with self.assertRaisesRegex(
                        RuntimeError, "partial .* resume is disabled",
                ):
                    run_layerwise_search_baseline(
                        **{**kwargs, "layerwise_env": _LayerwiseEnv()}
                    )

    def test_resume_rejects_changed_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            common = {
                "backend": "greedy",
                "layerwise_env": _LayerwiseEnv(),
                "robust_reference": _Reference(),
                "output_dir": tmpdir,
                "evaluation_budget": 2,
                "seed": 17,
                "initial_design_size": 2,
                "candidate_pool_size": 8,
                "population_size": 4,
                "patience_generations": 5,
                "mutation_max_coordinates": 1,
                "rf_n_estimators": 8,
                "rf_min_samples_leaf": 1,
                "communication_importance_ratio": 1.0,
                "manifest": {"profile": "mrpc"},
            }
            run_layerwise_search_baseline(**common)
            with self.assertRaisesRegex(RuntimeError, "resume contract"):
                run_layerwise_search_baseline(
                    **{
                        **common,
                        "layerwise_env": _LayerwiseEnv(),
                        "seed": 18,
                    }
                )

    def test_no_strict_evaluated_candidate_fails_after_persisting(self):
        strict_result = {
            "schema_version": "stage2_search_strict_validation_v1",
            "selected": None,
            "records": [{"strict_point_pass": False}],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(
                    RuntimeError, "no evaluated materializable candidate",
            ):
                run_layerwise_search_baseline(
                    backend="greedy",
                    layerwise_env=_LayerwiseEnv(),
                    robust_reference=_Reference(),
                    output_dir=tmpdir,
                    evaluation_budget=36,
                    seed=17,
                    initial_design_size=2,
                    candidate_pool_size=8,
                    population_size=4,
                    patience_generations=5,
                    mutation_max_coordinates=1,
                    rf_n_estimators=8,
                    rf_min_samples_leaf=1,
                    communication_importance_ratio=1.0,
                    manifest={
                        "profile": "mrpc",
                        "scientific_status": "full_search",
                    },
                    strict_validator=lambda _result: strict_result,
                )

            with open(
                    os.path.join(tmpdir, "manifest.json"),
                    encoding="utf-8",
            ) as handle:
                manifest = json.load(handle)
            self.assertEqual(
                manifest["status"], "failed_no_strict_materializable_candidate",
            )
            self.assertFalse(manifest["strict_validation_passed"])

    def test_no_strict_feasible_returns_materializable_fallback_and_artifacts(self):
        def strict_validator(result):
            selected = result.best.as_dict()
            selected["metadata"] = {
                **selected["metadata"],
                "strict_trial_count": 15,
            }
            return {
                "schema_version": "stage2_search_strict_validation_v2",
                "selection_status": "strict_least_violating",
                "formal_feasible": False,
                "selected": selected,
                "online_best": result.best.as_dict(),
                "records": [{"strict_evaluated": True}],
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            run = run_layerwise_search_baseline(
                backend="greedy",
                layerwise_env=_LayerwiseEnv(),
                robust_reference=_Reference(),
                output_dir=tmpdir,
                evaluation_budget=36,
                seed=17,
                initial_design_size=2,
                candidate_pool_size=8,
                population_size=4,
                patience_generations=5,
                mutation_max_coordinates=1,
                rf_n_estimators=8,
                rf_min_samples_leaf=1,
                communication_importance_ratio=1.0,
                manifest={"profile": "mrpc"},
                strict_validator=strict_validator,
            )

            self.assertIsNotNone(run["selected"])
            self.assertFalse(run["scientific_export_allowed"])
            self.assertEqual(
                run["manifest"]["status"],
                "complete_no_strict_feasible",
            )
            self.assertEqual(
                run["manifest"]["selection_status"],
                "strict_least_violating",
            )
            self.assertTrue(os.path.isfile(
                run["artifact_paths"]["final_selected_configuration"]
            ))
            self.assertTrue(os.path.isfile(
                run["artifact_paths"]["online_best"]
            ))
            self.assertEqual(
                run["online_best"].action_matrix,
                run["result"].best.action_matrix,
            )

    def test_canonical_strict_validation_reuses_shared_bank_gates(self):
        evaluator = LayerwiseRuntimeEvaluator(
            env=_LayerwiseEnv(),
            reference=_Reference(),
            base_seed=17,
            expected_trials=3,
        )
        result = run_search(
            "greedy",
            LayerwiseSearchSpace(2),
            evaluator,
            SearchConfig(evaluation_budget=1, seed=17),
        )
        promotion = SimpleNamespace(
            status="promoted",
            trial_count=30,
            fresh_trial_count=30,
            metrics=None,
            assessment=None,
            axis_counterfactuals={"compute": {}, "communication": {}},
        )
        certification = SimpleNamespace(
            status="final_revalidation_passed",
            trial_count=45,
            fresh_trial_count=15,
            metrics={
                "loss_mean": 1.0,
                "metric1_mean": 0.90,
                "metric2_mean": 0.85,
                "loss_std": 0.01,
                "metric1_std": 0.01,
                "metric2_std": 0.01,
            },
            assessment={
                "loss_precision_probability": 0.99,
                "metric1_precision_probability": 0.99,
                "metric2_precision_probability": 0.99,
                "loss_stability_probability": 0.99,
                "metric1_stability_probability": 0.99,
                "metric2_stability_probability": 0.99,
            },
            axis_counterfactuals={"compute": {}, "communication": {}},
        )
        banks = SimpleNamespace(
            final_reference=_Reference(),
            contract_payload=lambda: {"hard_gate": "canonical"},
        )
        store = SimpleNamespace(path="/tmp/search_strict_candidates.jsonl")

        with patch(
                "blb_stage2_rl.layerwise_runner."
                "promote_candidate_if_eligible",
                return_value=promotion,
        ) as promote_mock, patch(
                "blb_stage2_rl.layerwise_runner."
                "certify_candidate_with_bank_c",
                return_value=certification,
        ) as certify_mock:
            strict = canonical_strict_validation(
                result=result,
                layerwise_env=object(),
                promotion_base_env=object(),
                candidate_store=store,
                identity_context={"profile": "mrpc"},
                validation_banks=banks,
                top_n=5,
                communication_importance_ratio=1.0,
                promotion_probability=0.8,
                final_probability=0.95,
            )

        promote_mock.assert_called_once()
        certify_mock.assert_called_once()
        self.assertTrue(strict["records"][0]["strict_point_pass"])
        self.assertIsNotNone(strict["selected"])
        self.assertEqual(
            strict["selected"]["metrics"]["metric2_mean"], 0.85,
        )

    def test_canonical_validation_processes_all_top_n_feasible_candidates(self):
        first = _search_evaluation(((0, 0), (0, 0)))
        result = _search_result(
            first,
            _search_evaluation(((1, 0), (0, 0))),
            _search_evaluation(((1, 2), (1, 2))),
            first,
        )
        metrics = {
            "loss_mean": 1.0,
            "metric1_mean": 0.90,
            "metric2_mean": 0.85,
            "loss_std": 0.01,
            "metric1_std": 0.01,
            "metric2_std": 0.01,
        }
        promotion = _promotion_result(
            status="promoted", trial_count=30, metrics=metrics,
        )
        certification = _promotion_result(
            status="final_revalidation_passed",
            trial_count=45,
            fresh_trial_count=15,
            metrics=metrics,
            assessment={name: 0.99 for name in (
                "loss_precision_probability",
                "metric1_precision_probability",
                "metric2_precision_probability",
                "loss_stability_probability",
                "metric1_stability_probability",
                "metric2_stability_probability",
            )},
        )

        with patch(
                "blb_stage2_rl.layerwise_runner."
                "promote_candidate_if_eligible",
                return_value=promotion,
        ) as promote_mock, patch(
                "blb_stage2_rl.layerwise_runner."
                "certify_candidate_with_bank_c",
                return_value=certification,
        ) as certify_mock:
            strict = canonical_strict_validation(
                result=result,
                layerwise_env=object(),
                promotion_base_env=object(),
                candidate_store=SimpleNamespace(path="/tmp/candidates.jsonl"),
                identity_context={"profile": "mrpc"},
                validation_banks=_validation_banks(),
                top_n=3,
                communication_importance_ratio=1.0,
                promotion_probability=0.8,
                final_probability=0.95,
            )

        self.assertEqual(promote_mock.call_count, 3)
        self.assertEqual(certify_mock.call_count, 3)
        self.assertEqual(len(strict["records"]), 3)
        self.assertTrue(all(
            record["strict_evaluated"] for record in strict["records"]
        ))
        self.assertEqual(strict["selection_status"], "strict_feasible")
        self.assertTrue(strict["formal_feasible"])

    def test_top_n_includes_online_infeasible_and_strict_fallback(self):
        online_feasible = _search_evaluation(((0, 0), (0, 0)))
        online_infeasible = _search_evaluation(
            ((1, 2), (1, 2)), metric1_mean=0.70,
        )
        result = _search_result(online_infeasible, online_feasible)

        def promotion_side_effect(**kwargs):
            action = tuple(tuple(row) for row in kwargs["action_matrix"])
            metric1 = 0.88 if action == online_infeasible.action_matrix else 0.80
            return _promotion_result(
                status="bank_a_point_failed",
                trial_count=15,
                fresh_trial_count=15,
                metrics={
                    "loss_mean": 1.0,
                    "metric1_mean": metric1,
                    "metric2_mean": 0.85,
                    "loss_std": 0.01,
                    "metric1_std": 0.01,
                    "metric2_std": 0.01,
                },
            )

        with patch(
                "blb_stage2_rl.layerwise_runner."
                "promote_candidate_if_eligible",
                side_effect=promotion_side_effect,
        ) as promote_mock, patch(
                "blb_stage2_rl.layerwise_runner."
                "certify_candidate_with_bank_c",
        ) as certify_mock:
            strict = canonical_strict_validation(
                result=result,
                layerwise_env=object(),
                promotion_base_env=object(),
                candidate_store=SimpleNamespace(path="/tmp/candidates.jsonl"),
                identity_context={"profile": "mrpc"},
                validation_banks=_validation_banks(),
                top_n=2,
                communication_importance_ratio=1.0,
                promotion_probability=0.8,
                final_probability=0.95,
            )

        self.assertEqual(promote_mock.call_count, 2)
        certify_mock.assert_not_called()
        self.assertTrue(any(
            not record["online_candidate"]["feasible"]
            for record in strict["records"]
        ))
        self.assertEqual(strict["selection_status"], "strict_least_violating")
        self.assertFalse(strict["formal_feasible"])
        self.assertIsNotNone(strict["selected"])
        self.assertEqual(
            strict["selected"]["action_matrix"],
            [list(row) for row in online_infeasible.action_matrix],
        )
        selected_record = next(
            record for record in strict["records"]
            if record.get("strict_evaluation", {}).get("action_matrix")
            == strict["selected"]["action_matrix"]
        )
        self.assertEqual(selected_record["strict_trial_count"], 15)
        self.assertEqual(
            strict["selected"]["metadata"]["strict_trial_count"], 15,
        )
        self.assertIn(
            "metric1_mean",
            strict["selected_violations"]["families"]["joint"][
                "constraints"
            ],
        )
        self.assertEqual(
            strict["selected_violations"]["families"]["compute_only"][
                "status"
            ],
            "not_run",
        )

    def test_strict_fallback_ranks_joint_and_axis_violation_families(self):
        mild_axis = _search_evaluation(((0, 0), (0, 0)))
        severe_axis = _search_evaluation(((1, 0), (0, 0)))
        two_axis_failures = _search_evaluation(((1, 2), (1, 2)))
        result = _search_result(
            mild_axis, severe_axis, two_axis_failures,
        )
        joint_metrics = {
            "loss_mean": 1.0,
            "metric1_mean": 0.90,
            "metric2_mean": 0.85,
            "loss_std": 0.01,
            "metric1_std": 0.01,
            "metric2_std": 0.01,
        }

        def axis_payload(*, metric1_mean, point_pass):
            metrics = {**joint_metrics, "metric1_mean": metric1_mean}
            return {
                "loss_limit": 1.01,
                "metric1_limit": 0.89,
                "metric2_limit": 0.84,
                "loss_std_limit": 0.02,
                "metric1_std_limit": 0.015,
                "metric2_std_limit": 0.018,
                "banks": {
                    "A": {
                        "trial_count": 15,
                        "fresh_trial_count": 15,
                        "metrics": metrics,
                        "point_pass": point_pass,
                    },
                    "B": {
                        "trial_count": 30,
                        "fresh_trial_count": 15,
                        "metrics": metrics,
                        "point_pass": point_pass,
                    },
                },
                "point_pass": point_pass,
                "metrics": metrics,
            }

        def promotion_side_effect(**kwargs):
            action = tuple(tuple(row) for row in kwargs["action_matrix"])
            if action == mild_axis.action_matrix:
                compute = axis_payload(metric1_mean=0.88, point_pass=False)
                communication = axis_payload(
                    metric1_mean=0.90, point_pass=True,
                )
            elif action == severe_axis.action_matrix:
                compute = axis_payload(metric1_mean=0.70, point_pass=False)
                communication = axis_payload(
                    metric1_mean=0.90, point_pass=True,
                )
            else:
                compute = axis_payload(metric1_mean=0.80, point_pass=False)
                communication = axis_payload(
                    metric1_mean=0.80, point_pass=False,
                )
            return _promotion_result(
                status="axis_counterfactual_point_failed",
                trial_count=30,
                fresh_trial_count=30,
                metrics=joint_metrics,
                axis_counterfactuals={
                    "compute": compute,
                    "communication": communication,
                },
            )

        with patch(
                "blb_stage2_rl.layerwise_runner."
                "promote_candidate_if_eligible",
                side_effect=promotion_side_effect,
        ), patch(
                "blb_stage2_rl.layerwise_runner."
                "certify_candidate_with_bank_c",
        ) as certify_mock:
            strict = canonical_strict_validation(
                result=result,
                layerwise_env=object(),
                promotion_base_env=object(),
                candidate_store=SimpleNamespace(path="/tmp/candidates.jsonl"),
                identity_context={"profile": "mrpc"},
                validation_banks=_validation_banks(),
                top_n=3,
                communication_importance_ratio=1.0,
                promotion_probability=0.8,
                final_probability=0.95,
            )

        certify_mock.assert_not_called()
        self.assertEqual(
            strict["selected"]["action_matrix"],
            [list(row) for row in mild_axis.action_matrix],
        )
        aggregate = strict["selected_violations"]["aggregate"]
        self.assertEqual(aggregate["failed_constraint_count"], 1)
        selected_compute = strict["selected_violations"]["families"][
            "compute_only"
        ]
        self.assertEqual(selected_compute["trial_count"], 30)
        self.assertEqual(selected_compute["banks_run"], ["A", "B"])
        self.assertEqual(selected_compute["not_run_banks"], ["C"])
        self.assertEqual(selected_compute["status"], "partial_early_stopped")
        self.assertEqual(
            strict["selected"]["metadata"]["strict_trial_count"], 30,
        )

    def test_invalid_and_nonmaterializable_candidates_cannot_be_selected(self):
        invalid = _search_evaluation(
            ((1, 2), (1, 2)), valid=False, materializable=False,
        )
        materializable = _search_evaluation(((0, 0), (0, 0)))
        result = _search_result(invalid, materializable)
        promotion = _promotion_result(
            status="bank_a_point_failed",
            trial_count=15,
            metrics={
                "loss_mean": 1.0,
                "metric1_mean": 0.88,
                "metric2_mean": 0.85,
                "loss_std": 0.01,
                "metric1_std": 0.01,
                "metric2_std": 0.01,
            },
        )

        with patch(
                "blb_stage2_rl.layerwise_runner."
                "promote_candidate_if_eligible",
                return_value=promotion,
        ) as promote_mock:
            strict = canonical_strict_validation(
                result=result,
                layerwise_env=object(),
                promotion_base_env=object(),
                candidate_store=SimpleNamespace(path="/tmp/candidates.jsonl"),
                identity_context={"profile": "mrpc"},
                validation_banks=_validation_banks(),
                top_n=2,
                communication_importance_ratio=1.0,
                promotion_probability=0.8,
                final_probability=0.95,
            )

        promote_mock.assert_called_once()
        self.assertEqual(
            strict["selected"]["action_matrix"],
            [list(row) for row in materializable.action_matrix],
        )
        skipped = next(
            record for record in strict["records"]
            if not record["materializable"]
        )
        self.assertFalse(skipped["selection_eligible"])

    def test_point_gated_strict_selection_round_trips_without_probabilities(self):
        def strict_validator(result):
            selected = result.best.as_dict()
            selected["constraint_probabilities"] = {}
            selected["gate_probability"] = None
            return {
                "schema_version": "stage2_search_strict_validation_v1",
                "selected": selected,
                "records": [{"strict_point_pass": True}],
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            run = run_layerwise_search_baseline(
                backend="greedy",
                layerwise_env=_LayerwiseEnv(),
                robust_reference=_Reference(),
                output_dir=tmpdir,
                evaluation_budget=36,
                seed=17,
                initial_design_size=2,
                candidate_pool_size=8,
                population_size=4,
                patience_generations=5,
                mutation_max_coordinates=1,
                rf_n_estimators=8,
                rf_min_samples_leaf=1,
                communication_importance_ratio=1.0,
                manifest={"profile": "mrpc"},
                strict_validator=strict_validator,
            )

        self.assertTrue(run["manifest"]["strict_validation_passed"])
        self.assertTrue(run["scientific_export_allowed"])
        self.assertEqual(run["selected"].constraint_probabilities, ())
        self.assertIsNone(run["selected"].gate_probability)


if __name__ == "__main__":
    unittest.main()
