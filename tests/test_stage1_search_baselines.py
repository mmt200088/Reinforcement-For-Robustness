from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest

import numpy as np


# Load the two torch-free modules without executing stage1_rl/__init__.py, whose
# existing multi-GPU exports intentionally import torch.  This focused test must
# remain runnable in the lightweight CPU test environment.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_NAME = "_stage1_search_testpkg"
_package = ModuleType(_PACKAGE_NAME)
_package.__path__ = [str(_REPO_ROOT / "stage1_rl")]
sys.modules[_PACKAGE_NAME] = _package


def _load_module(short_name):
    full_name = f"{_PACKAGE_NAME}.{short_name}"
    spec = importlib.util.spec_from_file_location(
        full_name,
        _REPO_ROOT / "stage1_rl" / f"{short_name}.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


_search_baselines = _load_module("search_baselines")
_search_runner = _load_module("search_runner")
SearchConfig = _search_baselines.SearchConfig
SearchEvaluation = _search_baselines.SearchEvaluation
SearchResult = _search_baselines.SearchResult
Stage1Constraints = _search_baselines.Stage1Constraints
Stage1SearchSpace = _search_baselines.Stage1SearchSpace
candidate_rank_key = _search_baselines.candidate_rank_key
run_search = _search_baselines.run_search
structured_maximin_initial_design = _search_baselines.structured_maximin_initial_design
_bo_acquisition_key = _search_baselines._bo_acquisition_key
_select_hamming_diverse_elites = (
    _search_baselines._select_hamming_diverse_elites
)
_tournament = _search_baselines._tournament
Stage1EvaluatorAdapter = _search_runner.Stage1EvaluatorAdapter
Stage1SearchRunner = _search_runner.Stage1SearchRunner
load_search_preload = _search_runner.load_search_preload


TWO_METRIC_CONSTRAINTS = Stage1Constraints(
    baseline_loss=1.0,
    baseline_metrics=(0.90, 0.85),
    loss_max=1.01,
    metric_mins=(0.891, 0.8415),
    metric_names=("Accuracy", "F1"),
)


def _evaluation(
        action,
        *,
        loss=1.0,
        metrics=(0.90, 0.85),
        cost=None,
        constraints=TWO_METRIC_CONSTRAINTS,
        valid=True,
        ):
    if cost is None:
        cost = 100.0 - sum(action)
    return SearchEvaluation(
        action=tuple(action),
        loss=loss,
        metrics=tuple(metrics),
        cost=cost,
        constraints=constraints,
        valid=valid,
    )


class _MeanTree:
    def __init__(self, feature_shapes):
        self.mean = None
        self.feature_shapes = feature_shapes

    def fit(self, features, targets):
        self.feature_shapes.append(tuple(np.asarray(features).shape))
        self.mean = np.asarray(targets, dtype=float).mean(axis=0)
        return self

    def predict(self, features):
        return np.repeat(
            self.mean.reshape(1, -1),
            np.asarray(features).shape[0],
            axis=0,
        )


class _FakeForest(_MeanTree):
    def fit(self, features, targets):
        super().fit(features, targets)
        self.estimators_ = [
            _MeanTree([]).fit(features, targets),
            _MeanTree([]).fit(features, targets),
        ]
        return self


class _ScriptedConstraintForest:
    def fit(self, _features, targets):
        self.output_count = np.asarray(targets).shape[1]
        self.estimators_ = [self]
        return self

    def predict(self, features):
        rows = np.asarray(features).reshape(-1, 2, 3)
        genes = np.argmax(rows, axis=2)
        result = []
        for action in genes:
            margins = np.full(self.output_count, 0.1, dtype=float)
            if tuple(action) == (0, 1):
                margins[0] = -2.0
            elif tuple(action) == (0, 2):
                margins[:2] = -0.01
            else:
                margins[:] = -1.0
            result.append(margins)
        return np.asarray(result)


class _AllContestantsRng:
    def choice(self, population_size, *, size, replace):
        self.last_call = (population_size, size, replace)
        return np.arange(size)


class SearchSpaceAndConstraintTests(unittest.TestCase):
    def test_immutable_categories_decode_to_gelu_and_fixed_softmax(self):
        space = Stage1SearchSpace(4)
        source = [0, 1, 2, 0]

        action = space.validate(source)
        source[0] = 2

        self.assertEqual(action, (0, 1, 2, 0))
        self.assertEqual(space.decode_gelu(action), (4, 2, 1, 4))
        self.assertEqual(space.fixed_softmax(), (6, 6, 6, 6))
        with self.assertRaisesRegex(ValueError, "category"):
            space.validate((0, 1, 3, 0))

    def test_exact_baseline_constraints_support_one_metric(self):
        constraints = Stage1Constraints.from_baseline(
            baseline_loss=0.5,
            baseline_metrics=(0.8,),
            loss_relative_tolerance=0.005,
            metric_relative_tolerance=0.005,
            metric_names=("Accuracy",),
        )
        evaluation = _evaluation(
            (0,),
            loss=0.5025,
            metrics=(0.796,),
            constraints=constraints,
        )

        self.assertEqual(constraints.loss_max, 0.5025)
        self.assertEqual(constraints.metric_mins, (0.796,))
        self.assertTrue(evaluation.feasible)
        self.assertIsNone(evaluation.metric2)

    def test_valid_fallback_rank_uses_required_violation_order(self):
        constraints = Stage1Constraints(
            baseline_loss=1.0,
            baseline_metrics=(1.0, 1.0),
            loss_max=1.0,
            metric_mins=(1.0, 1.0),
        )
        fewer_failed = _evaluation(
            (0, 0), loss=1.2, metrics=(1.0, 1.0), constraints=constraints,
        )
        more_failed = _evaluation(
            (2, 2), loss=1.1, metrics=(0.95, 1.0), constraints=constraints,
        )
        lower_total = _evaluation(
            (1, 0), loss=1.1, metrics=(1.0, 1.0), constraints=constraints,
        )
        higher_total = _evaluation(
            (1, 1), loss=1.2, metrics=(1.0, 1.0), constraints=constraints,
        )
        invalid = _evaluation(
            (0, 1), loss=1.01, metrics=(1.0, 1.0), constraints=constraints,
            valid=False,
        )

        self.assertGreater(candidate_rank_key(fewer_failed), candidate_rank_key(more_failed))
        self.assertGreater(candidate_rank_key(lower_total), candidate_rank_key(higher_total))
        self.assertGreater(candidate_rank_key(higher_total), candidate_rank_key(invalid))

    def test_evaluation_and_result_round_trip(self):
        observed = _evaluation((0, 1))
        result = SearchResult(
            algorithm="greedy",
            config=SearchConfig(evaluation_cap=9),
            best=observed,
            observations=(observed,),
            history=({"phase": "test"},),
            termination_reason="verified_local_optimum",
        )

        restored = SearchResult.from_dict(result.as_dict())

        self.assertEqual(restored.as_dict(), result.as_dict())


class StructuredDesignAndAlgorithmTests(unittest.TestCase):
    def test_l12_initial_population_has_exact_canonical_composition(self):
        space = Stage1SearchSpace(12)

        design = structured_maximin_initial_design(space, count=48, seed=7)

        self.assertEqual(len(design), 48)
        self.assertEqual(len(set(design)), 48)
        self.assertEqual(design[:3], space.anchors)
        reductions = design[3:27]
        self.assertEqual(len(reductions), 24)
        for action in reductions:
            self.assertEqual(sum(value != 0 for value in action), 1)
        self.assertEqual(len(design[27:]), 21)

    def test_greedy_accepts_two_opt_then_returns_to_one_opt(self):
        space = Stage1SearchSpace(3)

        def evaluator(action):
            total = sum(action)
            changed_layers = sum(value != 0 for value in action)
            feasible = changed_layers == 0 or changed_layers >= 2
            return _evaluation(
                action,
                loss=1.0 if feasible else 1.2,
                cost=10.0 - total,
            )

        result = run_search(
            "greedy",
            space,
            evaluator,
            SearchConfig(evaluation_cap=space.cardinality),
        )

        self.assertEqual(result.best.action, (2, 2, 2))
        self.assertTrue(any(
            row["phase"] == "two_opt"
            and row.get("accepted")
            and row.get("return_to_one_opt")
            for row in result.history
        ))
        self.assertTrue(any(
            row["phase"] == "verified_local_optimum"
            and row["one_opt_verified"]
            and row["two_opt_verified"]
            for row in result.history
        ))

    def test_legacy_checkpoint_callback_still_receives_observation_tuple(self):
        snapshots = []
        run_search(
            "greedy",
            Stage1SearchSpace(1),
            lambda action: _evaluation(action),
            SearchConfig(evaluation_cap=3),
            checkpoint_callback=snapshots.append,
        )

        self.assertEqual([len(row) for row in snapshots], [1, 2, 3])
        self.assertTrue(all(isinstance(row, tuple) for row in snapshots))

    def test_bo_uses_one_hot_features_and_hard_cap(self):
        space = Stage1SearchSpace(2)
        shapes = []
        calls = []

        def evaluator(action):
            calls.append(action)
            return _evaluation(action)

        result = run_search(
            "bo_rf",
            space,
            evaluator,
            SearchConfig(
                evaluation_cap=6,
                bo_initial_design_size=3,
                bo_candidate_pool_size=9,
                bo_no_improvement_patience=20,
            ),
            surrogate_factory=lambda _seed: _FakeForest(shapes),
        )

        self.assertEqual(result.evaluation_count, 6)
        self.assertEqual(len(calls), len(set(calls)))
        self.assertTrue(shapes)
        self.assertTrue(all(shape[1] == 6 for shape in shapes))
        self.assertEqual(result.termination_reason, "evaluation_cap")

    def test_bo_acquisition_preserves_constraint_lexicographic_order(self):
        fewer_failed = _bo_acquisition_key(
            has_feasible_incumbent=False,
            probability_of_feasibility=0.0,
            expected_improvement=0.0,
            expected_failed_constraints=1.0,
            expected_total_violation=2.0,
            expected_worst_violation=2.0,
            exploration_tiebreak=0.0,
            objective_tiebreak=-100.0,
            deterministic_tiebreak=(0,),
        )
        more_failed = _bo_acquisition_key(
            has_feasible_incumbent=False,
            probability_of_feasibility=1.0,
            expected_improvement=0.0,
            expected_failed_constraints=2.0,
            expected_total_violation=0.02,
            expected_worst_violation=0.01,
            exploration_tiebreak=100.0,
            objective_tiebreak=100.0,
            deterministic_tiebreak=(0,),
        )
        self.assertGreater(fewer_failed, more_failed)

        pof_zero = _bo_acquisition_key(
            has_feasible_incumbent=True,
            probability_of_feasibility=0.0,
            expected_improvement=100.0,
            expected_failed_constraints=0.0,
            expected_total_violation=0.0,
            expected_worst_violation=0.0,
            exploration_tiebreak=0.0,
            objective_tiebreak=100.0,
            deterministic_tiebreak=(0,),
        )
        positive_pof = _bo_acquisition_key(
            has_feasible_incumbent=True,
            probability_of_feasibility=0.5,
            expected_improvement=1.0,
            expected_failed_constraints=0.0,
            expected_total_violation=0.0,
            expected_worst_violation=0.0,
            exploration_tiebreak=0.0,
            objective_tiebreak=0.0,
            deterministic_tiebreak=(0,),
        )
        self.assertGreater(positive_pof, pof_zero)

    def test_bo_uses_lexicographic_prediction_in_full_search_loop(self):
        space = Stage1SearchSpace(2)
        result = run_search(
            "bo_rf",
            space,
            lambda action: _evaluation(action, loss=1.2),
            SearchConfig(
                evaluation_cap=4,
                bo_initial_design_size=3,
                bo_candidate_pool_size=9,
                bo_no_improvement_patience=10,
            ),
            surrogate_factory=lambda _seed: _ScriptedConstraintForest(),
        )

        self.assertEqual(result.observations[-1].action, (0, 1))

    def test_elites_keep_incumbent_and_prefer_hamming_distance_two(self):
        space = Stage1SearchSpace(3)
        incumbent = _evaluation((0, 0, 0), cost=1.0)
        close = _evaluation((0, 0, 1), cost=2.0)
        diverse = _evaluation((1, 1, 0), cost=3.0)

        elites = _select_hamming_diverse_elites(
            space, [close, diverse, incumbent], 2,
        )

        self.assertEqual(elites[0].action, incumbent.action)
        self.assertEqual(elites[1].action, diverse.action)

    def test_parent_b_prefers_distance_two_then_relaxes_to_distance_one(self):
        parent_a = (0, 0, 0)
        close_but_better = _evaluation((0, 0, 1), cost=1.0)
        distance_two = _evaluation((1, 1, 0), cost=10.0)
        same = _evaluation(parent_a, cost=20.0)

        selected = _tournament(
            [close_but_better, distance_two, same],
            _AllContestantsRng(),
            3,
            diverse_from=parent_a,
        )
        relaxed = _tournament(
            [close_but_better, same],
            _AllContestantsRng(),
            2,
            diverse_from=parent_a,
        )

        self.assertEqual(selected.action, distance_two.action)
        self.assertEqual(relaxed.action, close_but_better.action)

    def test_ga_refills_unique_offspring_each_generation(self):
        space = Stage1SearchSpace(4)
        calls = []
        config = SearchConfig(
            evaluation_cap=42,
            ga_population_size=12,
            ga_elite_count=2,
            ga_update_generations=3,
            ga_tournament_size=3,
            ga_duplicate_attempts=64,
            maximin_candidate_pool_size=128,
        )

        def evaluator(action):
            calls.append(action)
            return _evaluation(action)

        result = run_search("coinn_ga", space, evaluator, config)

        self.assertEqual(result.evaluation_count, 12 + 3 * 10)
        self.assertEqual(result.unique_evaluation_count, result.evaluation_count)
        self.assertEqual(len(calls), len(set(calls)))
        self.assertEqual(result.termination_reason, "completed_generations")
        updates = [row for row in result.history if row["phase"] == "elitist_update"]
        self.assertEqual(len(updates), 3)
        self.assertTrue(all(row["new_unique_evaluations"] == 10 for row in updates))

    def test_canonical_ga_defaults_target_34448_unique_evaluations(self):
        config = SearchConfig()

        self.assertEqual(config.ga_population_size, 48)
        self.assertEqual(config.ga_elite_count, 5)
        self.assertEqual(config.ga_update_generations, 800)
        self.assertEqual(config.canonical_ga_target_evaluations, 34_448)
        self.assertEqual(config.evaluation_cap, 34_448)


class _FakeRealEvaluator:
    def __init__(self):
        self.calls = []

    def stage1_evaluate(self, gelu, softmax, *, use_train, split):
        self.calls.append((tuple(gelu), tuple(softmax), use_train, split))
        return 0.5, 0.8, 123.0

    def get_simulated_cost(self, gelu, softmax):
        return float(sum(gelu) + sum(softmax)), float(sum(gelu)), float(sum(softmax))


class AdapterAndPersistenceTests(unittest.TestCase):
    def test_real_adapter_always_uses_validation_full_and_exact_limits(self):
        constraints = Stage1Constraints(
            baseline_loss=0.5,
            baseline_metrics=(0.8,),
            loss_max=0.505,
            metric_mins=(0.79,),
            metric_names=("Accuracy",),
        )
        real = _FakeRealEvaluator()
        adapter = Stage1EvaluatorAdapter(
            evaluator=real,
            num_layers=2,
            constraints=constraints,
        )

        evaluation = adapter((0, 2))

        self.assertEqual(real.calls, [((4, 1), (6, 6), False, "validation_full")])
        self.assertEqual(evaluation.constraints, constraints)
        self.assertEqual(evaluation.metrics, (0.8,))
        self.assertEqual(evaluation.metadata["split"], "validation_full")
        self.assertEqual(evaluation.metadata["loss_limit"], 0.505)

    def test_runner_persists_and_preloads_without_re_evaluating(self):
        constraints = Stage1Constraints(
            baseline_loss=0.5,
            baseline_metrics=(0.8,),
            loss_max=0.505,
            metric_mins=(0.79,),
        )
        real = _FakeRealEvaluator()
        adapter = Stage1EvaluatorAdapter(
            evaluator=real,
            num_layers=1,
            constraints=constraints,
        )
        config = SearchConfig(evaluation_cap=3, greedy_max_starts=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoints = []
            runner = Stage1SearchRunner(
                adapter=adapter,
                config=config,
                output_dir=tmpdir,
                manifest={"task": "synthetic"},
                checkpoint_callback=checkpoints.append,
                checkpoint_interval=1,
            )
            result = runner.run("greedy")

            self.assertEqual(result.evaluation_count, 3)
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "result.json")))
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "observations.jsonl")))
            preload = load_search_preload(os.path.join(tmpdir, "checkpoint.json"))
            self.assertEqual(len(preload), 3)
            with open(os.path.join(tmpdir, "manifest.json"), encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertEqual(manifest["split"], "validation_full")
            self.assertEqual(manifest["softmax_degrees"], [6])
            self.assertGreaterEqual(len(checkpoints), 4)
            self.assertTrue(all("observations" not in row for row in checkpoints))
            self.assertTrue(all(
                not row.get("result") or "observations" not in row["result"]
                for row in checkpoints
            ))
            with open(os.path.join(tmpdir, "checkpoint.json"), encoding="utf-8") as handle:
                checkpoint = json.load(handle)
            with open(os.path.join(tmpdir, "result.json"), encoding="utf-8") as handle:
                compact_result = json.load(handle)
            self.assertNotIn("observations", checkpoint)
            self.assertNotIn("observations", checkpoint["result"])
            self.assertNotIn("observations", compact_result)
            self.assertNotIn("history", compact_result)
            self.assertEqual(
                compact_result["schema_version"],
                "stage1_gelu_search_compact_result_v2",
            )
            self.assertEqual(checkpoint["observation_store"]["observation_count"], 3)
            self.assertFalse(checkpoint["optimizer_state_restored"])
            self.assertIn("replay_only", checkpoint["resume_semantics"])
            with open(os.path.join(tmpdir, "observations.jsonl"), encoding="utf-8") as handle:
                self.assertEqual(sum(1 for line in handle if line.strip()), 3)

            second_real = _FakeRealEvaluator()
            second = Stage1SearchRunner(
                adapter=Stage1EvaluatorAdapter(
                    evaluator=second_real,
                    num_layers=1,
                    constraints=constraints,
                ),
                config=config,
            ).run("greedy", preload=preload)
            self.assertEqual(second.evaluation_count, 3)
            self.assertEqual(second_real.calls, [])

            auto_resume_real = _FakeRealEvaluator()
            auto_resumed = Stage1SearchRunner(
                adapter=Stage1EvaluatorAdapter(
                    evaluator=auto_resume_real,
                    num_layers=1,
                    constraints=constraints,
                ),
                config=config,
                output_dir=tmpdir,
                manifest={"task": "synthetic"},
            ).run("greedy")
            self.assertEqual(auto_resumed.evaluation_count, 3)
            self.assertEqual(auto_resumed.history, result.history)
            self.assertEqual(
                auto_resumed.termination_reason,
                result.termination_reason,
            )
            self.assertEqual(auto_resume_real.calls, [])

            with self.assertRaisesRegex(
                    RuntimeError, "search configuration does not match",
            ):
                Stage1SearchRunner(
                    adapter=Stage1EvaluatorAdapter(
                        evaluator=_FakeRealEvaluator(),
                        num_layers=1,
                        constraints=constraints,
                    ),
                    config=SearchConfig(
                        evaluation_cap=2,
                        greedy_max_starts=2,
                    ),
                    output_dir=tmpdir,
                ).run("greedy")

    def test_partial_bo_preload_is_rejected_without_surrogate_state(self):
        constraints = Stage1Constraints(
            baseline_loss=0.5,
            baseline_metrics=(0.8,),
            loss_max=0.505,
            metric_mins=(0.79,),
        )
        with self.assertRaisesRegex(RuntimeError, "partial bo_rf resume"):
            Stage1SearchRunner(
                adapter=Stage1EvaluatorAdapter(
                    evaluator=_FakeRealEvaluator(),
                    num_layers=1,
                    constraints=constraints,
                ),
                config=SearchConfig(evaluation_cap=3),
            ).run("bo_rf", preload=[_evaluation((0,))])

    def test_checkpoint_recovers_fsynced_suffix_after_stale_metadata(self):
        constraints = Stage1Constraints(
            baseline_loss=0.5,
            baseline_metrics=(0.8,),
            loss_max=0.505,
            metric_mins=(0.79,),
        )
        real = _FakeRealEvaluator()
        original = real.stage1_evaluate

        def fail_on_third(*args, **kwargs):
            if len(real.calls) >= 2:
                raise RuntimeError("interrupted")
            return original(*args, **kwargs)

        real.stage1_evaluate = fail_on_third
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                Stage1SearchRunner(
                    adapter=Stage1EvaluatorAdapter(
                        evaluator=real,
                        num_layers=1,
                        constraints=constraints,
                    ),
                    config=SearchConfig(
                        evaluation_cap=3, greedy_max_starts=3,
                    ),
                    output_dir=tmpdir,
                    manifest={"task": "stale-checkpoint"},
                    checkpoint_interval=50,
                ).run("greedy")

            with open(
                    os.path.join(tmpdir, "checkpoint.json"),
                    encoding="utf-8",
            ) as handle:
                checkpoint = json.load(handle)
            self.assertEqual(checkpoint["observation_count"], 1)
            recovered = load_search_preload(os.path.join(
                tmpdir, "checkpoint.json",
            ))
            self.assertEqual(len(recovered), 2)
            self.assertGreaterEqual(
                recovered[-1].metadata["search_cumulative_wall_seconds"],
                float(checkpoint.get("search_wall_seconds", 0.0) or 0.0),
            )

    def test_completed_resume_rejects_mismatched_backend_metadata(self):
        constraints = Stage1Constraints(
            baseline_loss=0.5,
            baseline_metrics=(0.8,),
            loss_max=0.505,
            metric_mins=(0.79,),
        )
        config = SearchConfig(evaluation_cap=3, greedy_max_starts=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            Stage1SearchRunner(
                adapter=Stage1EvaluatorAdapter(
                    evaluator=_FakeRealEvaluator(),
                    num_layers=1,
                    constraints=constraints,
                ),
                config=config,
                output_dir=tmpdir,
                manifest={"task": "integrity-backend"},
            ).run("greedy")
            result_path = os.path.join(tmpdir, "result.json")
            with open(result_path, encoding="utf-8") as handle:
                payload = json.load(handle)
            payload["algorithm"] = "bo_rf"
            with open(result_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)

            with self.assertRaisesRegex(
                    RuntimeError, "backend does not match manifest",
            ):
                Stage1SearchRunner(
                    adapter=Stage1EvaluatorAdapter(
                        evaluator=_FakeRealEvaluator(),
                        num_layers=1,
                        constraints=constraints,
                    ),
                    config=config,
                    output_dir=tmpdir,
                    manifest={"task": "integrity-backend"},
                ).run("greedy")

    def test_completed_resume_rejects_truncated_observation_journal(self):
        constraints = Stage1Constraints(
            baseline_loss=0.5,
            baseline_metrics=(0.8,),
            loss_max=0.505,
            metric_mins=(0.79,),
        )
        config = SearchConfig(evaluation_cap=3, greedy_max_starts=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            Stage1SearchRunner(
                adapter=Stage1EvaluatorAdapter(
                    evaluator=_FakeRealEvaluator(),
                    num_layers=1,
                    constraints=constraints,
                ),
                config=config,
                output_dir=tmpdir,
                manifest={"task": "integrity"},
            ).run("greedy")
            observation_path = os.path.join(tmpdir, "observations.jsonl")
            with open(observation_path, encoding="utf-8") as handle:
                first = next(handle)
            with open(observation_path, "w", encoding="utf-8") as handle:
                handle.write(first)

            with self.assertRaisesRegex(
                    RuntimeError, "observation count does not match manifest",
            ):
                Stage1SearchRunner(
                    adapter=Stage1EvaluatorAdapter(
                        evaluator=_FakeRealEvaluator(),
                        num_layers=1,
                        constraints=constraints,
                    ),
                    config=config,
                    output_dir=tmpdir,
                    manifest={"task": "integrity"},
                ).run("greedy")


if __name__ == "__main__":
    unittest.main()
