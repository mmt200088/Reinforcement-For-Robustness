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
    LayerwiseSearchSpace,
    SearchConfig,
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
                "high",
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

    def test_full_validation_without_strict_candidate_fails_after_persisting(self):
        strict_result = {
            "schema_version": "stage2_search_strict_validation_v1",
            "selected": None,
            "records": [{"strict_point_pass": False}],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(
                    RuntimeError, "no search candidate passed",
            ):
                run_layerwise_search_baseline(
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
                manifest["status"], "complete_no_strict_feasible",
            )
            self.assertFalse(manifest["strict_validation_passed"])
            self.assertTrue(os.path.isfile(
                os.path.join(tmpdir, "strict_validation.json")
            ))

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
                evaluation_budget=1,
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
