from __future__ import annotations

import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

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
load_completed_search_result = _search_runner.load_completed_search_result
load_search_preload = _search_runner.load_search_preload
Stage1SearchGracefulStop = _search_runner.Stage1SearchGracefulStop
_validate_completed_search_contract = (
    _search_runner._validate_completed_search_contract
)


TWO_METRIC_CONSTRAINTS = Stage1Constraints(
    baseline_loss=1.0,
    baseline_metrics=(0.90, 0.85),
    loss_max=1.01,
    metric_mins=(0.891, 0.8415),
    metric_names=("Accuracy", "F1"),
)


def _comparator_constraints():
    return Stage1Constraints.from_baseline(
        baseline_loss=0.5,
        baseline_metrics=(0.8, 0.8),
        loss_relative_tolerance=0.001,
        metric_relative_tolerance=0.001,
        metric_names=("accuracy", "weighted_f1"),
    )


def _comparator_config(backend):
    ga_update_generations = 600 if backend == "coinn_ga" else 800
    evaluation_caps = {
        "bo_rf": 50_000,
        "greedy": 3 ** 12,
        "coinn_ga": 64 + ga_update_generations * (64 - 7),
    }
    return SearchConfig(
        evaluation_cap=evaluation_caps[backend],
        seed=42,
        bo_initial_design_size=64,
        bo_candidate_pool_size=2_048,
        bo_no_improvement_patience=100,
        rf_n_estimators=128,
        rf_min_samples_leaf=2,
        acquisition_exploration=0.05,
        greedy_max_starts=3,
        ga_population_size=64,
        ga_elite_count=7,
        ga_update_generations=ga_update_generations,
        ga_no_improvement_patience=5,
        ga_stop_on_no_improvement=backend != "coinn_ga",
        ga_tournament_size=3,
        ga_crossover_probability=0.0,
        ga_mutation_max_layers=4,
        ga_duplicate_attempts=64,
        ga_unique_ratio_threshold=0.60,
        ga_mean_distance_threshold=2.0,
        ga_immigrant_fraction=0.0,
        maximin_candidate_pool_size=1_024,
    )


def _scientific_evaluation_dict(evaluation):
    payload = evaluation.as_dict()
    metadata = dict(payload.get("metadata") or {})
    metadata.pop("wall_seconds", None)
    metadata.pop("search_cumulative_wall_seconds", None)
    payload["metadata"] = metadata
    return payload


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

        design = structured_maximin_initial_design(space, count=64, seed=7)

        self.assertEqual(len(design), 64)
        self.assertEqual(len(set(design)), 64)
        self.assertEqual(design[:3], space.anchors)
        reductions = design[3:27]
        self.assertEqual(len(reductions), 24)
        for action in reductions:
            self.assertEqual(sum(value != 0 for value in action), 1)
        self.assertEqual(len(design[27:]), 37)

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

    def test_greedy_replays_full_cap_preload_without_new_inference(self):
        space = Stage1SearchSpace(2)
        config = SearchConfig(
            evaluation_cap=space.cardinality,
            greedy_max_starts=3,
        )
        reference = run_search(
            "greedy", space, lambda action: _evaluation(action), config,
        )

        result = run_search(
            "greedy",
            space,
            lambda _action: self.fail("replay must not call evaluator"),
            config,
            preload=reference.observations,
        )

        self.assertEqual(result.as_dict(), reference.as_dict())
        self.assertEqual(result.evaluation_count, space.cardinality)

    def test_direct_greedy_preload_replays_exact_history(self):
        space = Stage1SearchSpace(2)
        config = SearchConfig(
            evaluation_cap=space.cardinality,
            greedy_max_starts=3,
        )
        reference = run_search(
            "greedy", space, lambda action: _evaluation(action), config,
        )
        resumed_calls = []

        def resumed_evaluator(action):
            resumed_calls.append(action)
            return _evaluation(action)

        resumed = run_search(
            "greedy",
            space,
            resumed_evaluator,
            config,
            preload=reference.observations[:2],
        )

        self.assertEqual(resumed.as_dict(), reference.as_dict())
        self.assertEqual(
            len(resumed_calls), reference.evaluation_count - 2,
        )

    def test_run_search_rejects_duplicate_preload_for_every_backend(self):
        space = Stage1SearchSpace(1)
        duplicate = _evaluation((0,))
        for backend in ("greedy", "bo_rf", "coinn_ga"):
            with self.subTest(backend=backend):
                with self.assertRaisesRegex(ValueError, "duplicate action"):
                    run_search(
                        backend,
                        space,
                        lambda _action: self.fail(
                            "duplicate preload must fail before evaluation"
                        ),
                        SearchConfig(evaluation_cap=1),
                        preload=(duplicate, duplicate),
                        surrogate_factory=(
                            (lambda _seed: _FakeForest([]))
                            if backend == "bo_rf" else None
                        ),
                    )

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

    def test_bo_partial_preload_replays_exact_trajectory(self):
        space = Stage1SearchSpace(3)
        config = SearchConfig(
            seed=17,
            evaluation_cap=10,
            bo_initial_design_size=4,
            bo_candidate_pool_size=7,
            bo_no_improvement_patience=20,
            maximin_candidate_pool_size=32,
        )
        reference = run_search(
            "bo_rf",
            space,
            lambda action: _evaluation(action),
            config,
            surrogate_factory=lambda _seed: _FakeForest([]),
        )
        resumed_calls = []

        def resumed_evaluator(action):
            resumed_calls.append(action)
            return _evaluation(action)

        resumed = run_search(
            "bo_rf",
            space,
            resumed_evaluator,
            config,
            surrogate_factory=lambda _seed: _FakeForest([]),
            preload=reference.observations[:6],
        )

        self.assertEqual(resumed.as_dict(), reference.as_dict())
        self.assertEqual(
            len(resumed_calls), reference.evaluation_count - 6,
        )

    def test_bo_completion_contract_requires_every_acquisition(self):
        space = Stage1SearchSpace(2)
        result = run_search(
            "bo_rf",
            space,
            lambda action: _evaluation(action),
            SearchConfig(
                evaluation_cap=6,
                bo_initial_design_size=3,
                bo_candidate_pool_size=9,
                bo_no_improvement_patience=20,
            ),
            surrogate_factory=lambda _seed: _FakeForest([]),
        )
        _validate_completed_search_contract(result)
        forged = SearchResult(
            algorithm=result.algorithm,
            config=result.config,
            best=result.best,
            observations=result.observations,
            history=tuple(
                row for row in result.history
                if row.get("iteration") != 2
            ),
            termination_reason=result.termination_reason,
        )

        with self.assertRaisesRegex(RuntimeError, "BO-RF completion contract"):
            _validate_completed_search_contract(forged)

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

    def test_ga_parent_selection_is_feasibility_aware_fitness_proportional(self):
        cheap = _evaluation((0, 0, 0), cost=1.0)
        expensive = _evaluation((1, 1, 1), cost=4.0)
        infeasible = _evaluation(
            (2, 2, 2), cost=0.5, loss=1.2,
        )

        class RecordingRng:
            def __init__(self):
                self.probabilities = None

            def choice(self, count, *args, **kwargs):
                self.probabilities = kwargs.get("p")
                if self.probabilities is None:
                    return np.arange(count)
                return 0

        rng = RecordingRng()
        selected = _tournament(
            [cheap, expensive, infeasible], rng, 3,
        )

        self.assertEqual(selected.action, cheap.action)
        self.assertGreater(rng.probabilities[2], 0.0)
        self.assertAlmostEqual(sum(rng.probabilities[:2]), 0.90)
        self.assertAlmostEqual(rng.probabilities[2], 0.10)
        self.assertAlmostEqual(
            rng.probabilities[0] / rng.probabilities[1], 4.0,
        )

    def test_ga_mixed_population_preserves_subnormal_infeasible_stratum_mass(self):
        zero_constraints = Stage1Constraints(
            baseline_loss=0.0,
            baseline_metrics=(0.0, 0.0),
            loss_max=0.0,
            metric_mins=(0.0, 0.0),
        )
        feasible = _evaluation(
            (0,),
            loss=0.0,
            metrics=(0.0, 0.0),
            cost=1.0,
            constraints=zero_constraints,
        )
        extreme_infeasible = _evaluation(
            (1,),
            loss=1.0e308,
            metrics=(-1.0e308, -1.0e308),
            cost=1.0,
            constraints=zero_constraints,
        )
        self.assertEqual(
            _search_baselines._ga_parent_weights((extreme_infeasible,))[0],
            np.nextafter(0.0, 1.0),
        )

        class RecordingRng:
            def __init__(self):
                self.probabilities = None

            def choice(self, _count, *args, **kwargs):
                self.probabilities = kwargs["p"]
                return 0

        rng = RecordingRng()
        _tournament((feasible, extreme_infeasible), rng, 2)

        self.assertAlmostEqual(rng.probabilities[0], 0.90)
        self.assertAlmostEqual(rng.probabilities[1], 0.10)

    def test_ga_all_infeasible_fitness_decreases_with_violation(self):
        mild = _evaluation((0,), loss=1.02)
        severe = _evaluation((1,), loss=1.20)
        weights = getattr(_search_baselines, "_ga_parent_weights", None)

        self.assertIsNotNone(weights)
        mild_weight, severe_weight = weights((mild, severe))
        self.assertGreater(mild_weight, severe_weight)
        self.assertGreater(severe_weight, 0.0)

    def test_ga_all_infeasible_extreme_finite_metrics_keep_positive_weights(self):
        extreme_loss = _evaluation(
            (0,), loss=1.0e308, metrics=(-1.0e308, -1.0e308),
        )
        extreme_metrics = _evaluation(
            (1,), loss=9.0e307, metrics=(-9.0e307, -9.0e307),
        )

        weights = np.asarray(
            _search_baselines._ga_parent_weights(
                (extreme_loss, extreme_metrics),
            ),
            dtype=float,
        )

        self.assertTrue(np.all(np.isfinite(weights)))
        self.assertTrue(np.all(weights > 0.0))

    def test_ga_mutation_changes_each_selected_gene_only_to_adjacent_category(self):
        space = Stage1SearchSpace(8)
        base = (0,) * space.num_layers
        for seed in range(100):
            mutated = _search_baselines._mutate_action(
                space,
                base,
                np.random.default_rng(seed),
                max_layers=4,
                force=True,
            )
            changed_layers = sum(
                before != after
                for before, after in zip(base, mutated)  # noqa: B905 - Python 3.9
            )
            self.assertGreaterEqual(changed_layers, 1)
            self.assertLessEqual(changed_layers, 4)
            for index, before in enumerate(base):
                self.assertLessEqual(abs(before - mutated[index]), 1)

    def test_coinn_ga_never_invokes_crossover(self):
        space = Stage1SearchSpace(4)
        first = _evaluation((0, 0, 0, 0), cost=1.0)
        mutation_child = (1, 0, 0, 0)
        cache = SimpleNamespace(
            observations=(),
            contains=lambda _action: False,
        )
        config = SearchConfig(
            evaluation_cap=48,
            ga_crossover_probability=1.0,
            ga_duplicate_attempts=1,
            ga_mutation_max_layers=4,
        )

        with (
            mock.patch.object(
                _search_baselines,
                "_tournament",
                return_value=first,
            ) as select_parent,
            mock.patch.object(
                _search_baselines,
                "_crossover",
                side_effect=AssertionError("COINN-GA crossover is forbidden"),
            ) as crossover,
            mock.patch.object(
                _search_baselines,
                "_mutate_action",
                return_value=mutation_child,
            ) as mutate,
        ):
            child, immigrant = _search_baselines._breed_unique_child(
                space=space,
                population=(first,),
                cache=cache,
                blocked=set(),
                rng=np.random.default_rng(17),
                config=config,
            )

        self.assertEqual(child, mutation_child)
        self.assertFalse(immigrant)
        select_parent.assert_called_once()
        crossover.assert_not_called()
        self.assertEqual(mutate.call_args.args[1], first.action)
        self.assertTrue(mutate.call_args.kwargs["force"])

    def test_duplicate_repair_restarts_from_same_selected_parent(self):
        space = Stage1SearchSpace(6)
        first = _evaluation((0, 0, 0, 0, 0, 0), cost=1.0)
        duplicate = (1, 0, 0, 0, 0, 0)
        repaired = (0, 1, 0, 0, 0, 0)
        cache = SimpleNamespace(
            observations=(),
            contains=lambda _action: False,
        )
        config = SearchConfig(
            evaluation_cap=48,
            ga_duplicate_attempts=2,
            ga_mutation_max_layers=4,
        )

        with (
            mock.patch.object(
                _search_baselines,
                "_tournament",
                return_value=first,
            ),
            mock.patch.object(
                _search_baselines,
                "_crossover",
                side_effect=AssertionError("COINN-GA crossover is forbidden"),
            ),
            mock.patch.object(
                _search_baselines,
                "_mutate_action",
                side_effect=(duplicate, repaired),
            ) as mutate,
        ):
            child, immigrant = _search_baselines._breed_unique_child(
                space=space,
                population=(first,),
                cache=cache,
                blocked={duplicate},
                rng=np.random.default_rng(17),
                config=config,
            )

        self.assertEqual(child, repaired)
        self.assertFalse(immigrant)
        self.assertEqual(mutate.call_args_list[0].args[1], first.action)
        self.assertEqual(mutate.call_args_list[1].args[1], first.action)

    def test_duplicate_retry_exhaustion_uses_unseen_adjacent_mutation(self):
        space = Stage1SearchSpace(4)
        first = _evaluation((0, 0, 0, 0), cost=1.0)
        duplicate = (1, 0, 0, 0)
        cache = SimpleNamespace(
            observations=(_evaluation(duplicate, cost=1.0),),
            contains=lambda action: action == duplicate,
        )
        config = SearchConfig(
            evaluation_cap=48,
            ga_duplicate_attempts=1,
            ga_mutation_max_layers=4,
        )

        with (
            mock.patch.object(
                _search_baselines,
                "_tournament",
                return_value=first,
            ),
            mock.patch.object(
                _search_baselines,
                "_mutate_action",
                return_value=duplicate,
            ),
        ):
            child, immigrant = _search_baselines._breed_unique_child(
                space=space,
                population=(first,),
                cache=cache,
                blocked=set(),
                rng=np.random.default_rng(17),
                config=config,
            )

        self.assertEqual(child, (0, 1, 0, 0))
        self.assertFalse(immigrant)

    def test_ga_mixed_population_completes_five_full_unique_generations(self):
        space = Stage1SearchSpace(12)
        config = SearchConfig(
            seed=42,
            evaluation_cap=349,
            ga_population_size=64,
            ga_elite_count=7,
            ga_update_generations=800,
            ga_no_improvement_patience=5,
            ga_duplicate_attempts=64,
            maximin_candidate_pool_size=1024,
        )

        def evaluator(action):
            return _evaluation(
                action,
                loss=1.0 if action == space.all4_action else 1.2,
            )

        result = run_search("coinn_ga", space, evaluator, config)
        updates = [
            row for row in result.history
            if row["phase"] == "elitist_update"
        ]

        self.assertEqual(result.evaluation_count, 349)
        self.assertEqual(result.unique_evaluation_count, 349)
        self.assertEqual(
            [row["new_unique_evaluations"] for row in updates],
            [57, 57, 57, 57, 57],
        )
        self.assertEqual(
            result.termination_reason,
            "ga_no_incumbent_improvement",
        )

    def test_ga_does_not_start_generation_without_full_offspring_budget(self):
        space = Stage1SearchSpace(12)
        result = run_search(
            "coinn_ga",
            space,
            lambda action: _evaluation(action, cost=1.0),
            SearchConfig(
                seed=42,
                evaluation_cap=64 + 56,
                ga_population_size=64,
                ga_elite_count=7,
                ga_update_generations=800,
                ga_no_improvement_patience=5,
                ga_duplicate_attempts=64,
                maximin_candidate_pool_size=1024,
            ),
        )

        updates = [
            row for row in result.history
            if row["phase"] == "elitist_update"
        ]
        self.assertEqual(result.evaluation_count, 64)
        self.assertEqual(updates, [])
        self.assertEqual(result.termination_reason, "evaluation_cap")
        _validate_completed_search_contract(result)

    def test_ga_completion_contract_rejects_unused_full_generation_budget(self):
        space = Stage1SearchSpace(12)
        result = run_search(
            "coinn_ga",
            space,
            lambda action: _evaluation(action, cost=1.0),
            SearchConfig(
                seed=42,
                evaluation_cap=64 + 56,
                ga_population_size=64,
                ga_elite_count=7,
                ga_update_generations=800,
                ga_no_improvement_patience=5,
                ga_duplicate_attempts=64,
                maximin_candidate_pool_size=1024,
            ),
        )
        forged_config = SearchConfig.from_dict({
            **result.config.as_dict(),
            "evaluation_cap": 64 + 57,
        })
        forged = SearchResult(
            algorithm=result.algorithm,
            config=forged_config,
            best=result.best,
            observations=result.observations,
            history=result.history,
            termination_reason=result.termination_reason,
        )

        with self.assertRaisesRegex(RuntimeError, "evaluation cap"):
            _validate_completed_search_contract(forged)

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

    def test_ga_completion_contract_rejects_evaluation_cap_after_all_generations(self):
        space = Stage1SearchSpace(4)
        result = run_search(
            "coinn_ga",
            space,
            lambda action: _evaluation(action),
            SearchConfig(
                evaluation_cap=42,
                ga_population_size=12,
                ga_elite_count=2,
                ga_update_generations=3,
                maximin_candidate_pool_size=128,
            ),
        )
        forged = SearchResult(
            algorithm=result.algorithm,
            config=result.config,
            best=result.best,
            observations=result.observations,
            history=result.history,
            termination_reason="evaluation_cap",
        )

        with self.assertRaisesRegex(RuntimeError, "evaluation cap"):
            _validate_completed_search_contract(forged)

    def test_ga_completion_contract_rejects_more_updates_than_configured(self):
        space = Stage1SearchSpace(4)
        result = run_search(
            "coinn_ga",
            space,
            lambda action: _evaluation(action),
            SearchConfig(
                evaluation_cap=42,
                ga_population_size=12,
                ga_elite_count=2,
                ga_update_generations=3,
                maximin_candidate_pool_size=128,
            ),
        )
        forged_config = SearchConfig.from_dict({
            **result.config.as_dict(),
            "ga_update_generations": 2,
        })
        forged = SearchResult(
            algorithm=result.algorithm,
            config=forged_config,
            best=result.best,
            observations=result.observations,
            history=result.history,
            termination_reason="evaluation_cap",
        )

        with self.assertRaisesRegex(RuntimeError, "configured generation"):
            _validate_completed_search_contract(forged)

    def test_ga_stops_after_five_generations_without_incumbent_improvement(self):
        space = Stage1SearchSpace(5)
        config = SearchConfig(
            evaluation_cap=100,
            ga_population_size=12,
            ga_elite_count=2,
            ga_update_generations=10,
            ga_duplicate_attempts=64,
            maximin_candidate_pool_size=128,
        )
        result = run_search(
            "coinn_ga",
            space,
            lambda action: _evaluation(action, cost=1.0),
            config,
        )
        updates = [
            row for row in result.history
            if row["phase"] == "elitist_update"
        ]

        self.assertEqual(
            result.termination_reason,
            "ga_no_incumbent_improvement",
        )
        self.assertEqual(len(updates), 5)
        self.assertEqual(
            [row["no_improvement_generations"] for row in updates],
            [1, 2, 3, 4, 5],
        )

    def test_ga_can_disable_stagnation_stop_and_complete_every_generation(self):
        space = Stage1SearchSpace(5)
        config = SearchConfig(
            evaluation_cap=42,
            ga_population_size=12,
            ga_elite_count=2,
            ga_update_generations=3,
            ga_stop_on_no_improvement=False,
            ga_duplicate_attempts=64,
            maximin_candidate_pool_size=128,
        )

        result = run_search(
            "coinn_ga",
            space,
            lambda action: _evaluation(action, cost=1.0),
            config,
        )
        updates = [
            row for row in result.history
            if row["phase"] == "elitist_update"
        ]

        self.assertEqual(result.termination_reason, "completed_generations")
        self.assertEqual(len(updates), 3)
        self.assertEqual(result.evaluation_count, 42)
        self.assertEqual(
            [row["no_improvement_generations"] for row in updates],
            [1, 2, 3],
        )
        _validate_completed_search_contract(result)

    def test_ga_completion_contract_accepts_exact_five_generation_stagnation(self):
        space = Stage1SearchSpace(5)
        result = run_search(
            "coinn_ga",
            space,
            lambda action: _evaluation(action, cost=1.0),
            SearchConfig(
                evaluation_cap=100,
                ga_population_size=12,
                ga_elite_count=2,
                ga_update_generations=10,
                ga_duplicate_attempts=64,
                maximin_candidate_pool_size=128,
            ),
        )

        _validate_completed_search_contract(result)

    def test_ga_completion_contract_rejects_stagnation_when_stop_is_disabled(self):
        space = Stage1SearchSpace(5)
        result = run_search(
            "coinn_ga",
            space,
            lambda action: _evaluation(action, cost=1.0),
            SearchConfig(
                evaluation_cap=100,
                ga_population_size=12,
                ga_elite_count=2,
                ga_update_generations=10,
                ga_duplicate_attempts=64,
                maximin_candidate_pool_size=128,
            ),
        )
        forged = SearchResult(
            algorithm=result.algorithm,
            config=SearchConfig.from_dict({
                **result.config.as_dict(),
                "ga_stop_on_no_improvement": False,
            }),
            best=result.best,
            observations=result.observations,
            history=result.history,
            termination_reason=result.termination_reason,
        )

        with self.assertRaisesRegex(RuntimeError, "configured-patience"):
            _validate_completed_search_contract(forged)

    def test_ga_completion_contract_accepts_configured_three_generation_stagnation(self):
        space = Stage1SearchSpace(5)
        result = run_search(
            "coinn_ga",
            space,
            lambda action: _evaluation(action, cost=1.0),
            SearchConfig(
                evaluation_cap=100,
                ga_population_size=12,
                ga_elite_count=2,
                ga_update_generations=10,
                ga_no_improvement_patience=3,
                ga_duplicate_attempts=64,
                maximin_candidate_pool_size=128,
            ),
        )
        updates = [
            row for row in result.history
            if row["phase"] == "elitist_update"
        ]

        self.assertEqual(
            result.termination_reason,
            "ga_no_incumbent_improvement",
        )
        self.assertEqual(len(updates), 3)
        self.assertEqual(
            [row["no_improvement_generations"] for row in updates],
            [1, 2, 3],
        )
        _validate_completed_search_contract(result)

    def test_ga_completion_contract_rejects_partial_final_offspring_generation(self):
        space = Stage1SearchSpace(5)
        result = run_search(
            "coinn_ga",
            space,
            lambda action: _evaluation(action, cost=1.0),
            SearchConfig(
                evaluation_cap=100,
                ga_population_size=12,
                ga_elite_count=2,
                ga_update_generations=10,
                ga_duplicate_attempts=64,
                maximin_candidate_pool_size=128,
            ),
        )
        updates = [
            dict(row) for row in result.history
            if row["phase"] == "elitist_update"
        ]
        forged_observations = result.observations[:-1]
        updates[-1]["evaluations"] -= 1
        updates[-1]["new_unique_evaluations"] -= 1
        forged = SearchResult(
            algorithm=result.algorithm,
            config=result.config,
            best=max(forged_observations, key=candidate_rank_key),
            observations=forged_observations,
            history=(dict(result.history[0]), *updates),
            termination_reason=result.termination_reason,
        )

        with self.assertRaisesRegex(RuntimeError, "generation evaluation accounting"):
            _validate_completed_search_contract(forged)

    def test_ga_completion_contract_rejects_stagnation_before_five_generations(self):
        space = Stage1SearchSpace(5)
        result = run_search(
            "coinn_ga",
            space,
            lambda action: _evaluation(action, cost=1.0),
            SearchConfig(
                evaluation_cap=100,
                ga_population_size=12,
                ga_elite_count=2,
                ga_update_generations=10,
                ga_duplicate_attempts=64,
                maximin_candidate_pool_size=128,
            ),
        )
        updates = [
            row for row in result.history
            if row["phase"] == "elitist_update"
        ]
        fourth_evaluation_count = int(updates[3]["evaluations"])
        forged_observations = result.observations[:fourth_evaluation_count]
        forged = SearchResult(
            algorithm=result.algorithm,
            config=result.config,
            best=max(forged_observations, key=candidate_rank_key),
            observations=forged_observations,
            history=tuple(
                row for row in result.history
                if row.get("generation", 0) <= 4
            ),
            termination_reason="ga_no_incumbent_improvement",
        )

        with self.assertRaisesRegex(RuntimeError, "configured-patience"):
            _validate_completed_search_contract(forged)

    def test_ga_partial_generation_preload_replays_exact_trajectory(self):
        space = Stage1SearchSpace(4)
        config = SearchConfig(
            seed=23,
            evaluation_cap=18,
            ga_population_size=6,
            ga_elite_count=2,
            ga_update_generations=3,
            ga_duplicate_attempts=64,
            maximin_candidate_pool_size=64,
        )
        reference = run_search(
            "coinn_ga", space, lambda action: _evaluation(action), config,
        )
        resumed_calls = []

        def resumed_evaluator(action):
            resumed_calls.append(action)
            return _evaluation(action)

        resumed = run_search(
            "coinn_ga",
            space,
            resumed_evaluator,
            config,
            preload=reference.observations[:8],
        )

        self.assertEqual(resumed.as_dict(), reference.as_dict())
        self.assertEqual(
            len(resumed_calls), reference.evaluation_count - 8,
        )

    def test_ga_does_not_bypass_adjacent_mutation_with_immigrants(self):
        space = Stage1SearchSpace(4)
        config = SearchConfig(
            evaluation_cap=22,
            ga_population_size=12,
            ga_elite_count=2,
            ga_update_generations=1,
            ga_mean_distance_threshold=100.0,
            ga_immigrant_fraction=0.25,
            ga_duplicate_attempts=64,
            maximin_candidate_pool_size=128,
        )

        result = run_search(
            "coinn_ga", space, lambda action: _evaluation(action), config,
        )
        update = next(
            row for row in result.history if row["phase"] == "elitist_update"
        )

        self.assertFalse(update["diversity_triggered"])
        self.assertEqual(update["scheduled_immigrants"], 0)
        self.assertEqual(update["fallback_immigrants"], 0)
        self.assertEqual(update["replaced_worst_nonelite_actions"], [])
        self.assertEqual(update["immigrant_actions"], [])
        self.assertEqual(update["new_unique_evaluations"], 10)

    def test_ga_completion_contract_requires_every_generation(self):
        space = Stage1SearchSpace(4)
        config = SearchConfig(
            evaluation_cap=42,
            ga_population_size=12,
            ga_elite_count=2,
            ga_update_generations=3,
            maximin_candidate_pool_size=128,
        )
        result = run_search(
            "coinn_ga", space, lambda action: _evaluation(action), config,
        )
        _validate_completed_search_contract(result)
        forged = SearchResult(
            algorithm=result.algorithm,
            config=result.config,
            best=result.best,
            observations=result.observations,
            history=tuple(
                row for row in result.history
                if row.get("generation") != 2
            ),
            termination_reason=result.termination_reason,
        )

        with self.assertRaisesRegex(RuntimeError, "GA completion contract"):
            _validate_completed_search_contract(forged)

    def test_greedy_completion_contract_rebuilds_local_neighborhood_proof(self):
        observations = tuple(_evaluation((value,)) for value in range(3))
        forged = SearchResult(
            algorithm="greedy",
            config=SearchConfig(evaluation_cap=3, greedy_max_starts=1),
            best=observations[-1],
            observations=observations,
            history=(
                {
                    "phase": "start",
                    "iteration": 0,
                    "start_index": 0,
                    "current_action": [0],
                },
                {
                    "phase": "verified_local_optimum",
                    "iteration": 1,
                    "start_index": 0,
                    "one_opt_verified": True,
                    "two_opt_verified": True,
                    "current_action": [0],
                },
            ),
            termination_reason="verified_local_optimum",
        )

        with self.assertRaisesRegex(RuntimeError, "neighborhood proof"):
            _validate_completed_search_contract(forged)

    def test_canonical_ga_defaults_target_45664_unique_evaluations(self):
        config = SearchConfig()

        self.assertEqual(config.bo_initial_design_size, 64)
        self.assertEqual(config.ga_population_size, 64)
        self.assertEqual(config.ga_elite_count, 7)
        self.assertEqual(config.ga_update_generations, 800)
        self.assertEqual(config.ga_no_improvement_patience, 5)
        self.assertTrue(config.ga_stop_on_no_improvement)
        self.assertEqual(config.ga_mutation_max_layers, 4)
        self.assertEqual(config.ga_crossover_probability, 0.0)
        self.assertEqual(config.ga_immigrant_fraction, 0.0)
        self.assertEqual(config.canonical_ga_target_evaluations, 45_664)
        self.assertEqual(config.evaluation_cap, 45_664)

    def test_legacy_search_config_defaults_to_stagnation_stop(self):
        payload = SearchConfig().as_dict()
        del payload["ga_stop_on_no_improvement"]

        restored = SearchConfig.from_dict(payload)

        self.assertTrue(restored.ga_stop_on_no_improvement)

    def test_stage1_ga_full_600_generation_contract_has_exact_budget(self):
        config = _search_baselines.stage1_comparator_search_config("coinn_ga")

        self.assertEqual(config.ga_update_generations, 600)
        self.assertFalse(config.ga_stop_on_no_improvement)
        self.assertEqual(config.evaluation_cap, 64 + 600 * (64 - 7))
        self.assertEqual(config.canonical_ga_target_evaluations, 34_264)

    def test_stage1_ga_executes_all_600_update_generations(self):
        space = Stage1SearchSpace(12)
        config = _search_baselines.stage1_comparator_search_config("coinn_ga")
        category_cost = {0: 3.0, 1: 2.5, 2: 1.0}

        result = run_search(
            "coinn_ga",
            space,
            lambda action: _evaluation(
                action,
                cost=36.0 + sum(category_cost[value] for value in action),
            ),
            config,
        )
        updates = [
            row for row in result.history
            if row["phase"] == "elitist_update"
        ]

        self.assertEqual(result.termination_reason, "completed_generations")
        self.assertEqual(len(updates), 600)
        self.assertEqual(updates[-1]["generation"], 600)
        self.assertEqual(result.evaluation_count, 34_264)
        self.assertEqual(result.unique_evaluation_count, 34_264)
        _validate_completed_search_contract(result)


class _FakeRealEvaluator:
    def __init__(self):
        self.calls = []

    def stage1_evaluate(self, gelu, softmax, *, use_train, split):
        self.calls.append((tuple(gelu), tuple(softmax), use_train, split))
        return 0.5, 0.8, 123.0

    def get_simulated_cost(self, gelu, softmax):
        return float(sum(gelu) + sum(softmax)), float(sum(gelu)), float(sum(softmax))


class AdapterAndPersistenceTests(unittest.TestCase):
    def _constraints(self):
        return Stage1Constraints(
            baseline_loss=0.5,
            baseline_metrics=(0.8,),
            loss_max=0.505,
            metric_mins=(0.79,),
        )

    def _ordinary_runner(
            self,
            output_dir,
            evaluator=None,
            *,
            config=None,
            num_layers=1,
            checkpoint_callback=None,
            checkpoint_interval=1,
            ):
        return Stage1SearchRunner(
            adapter=Stage1EvaluatorAdapter(
                evaluator=evaluator or _FakeRealEvaluator(),
                num_layers=num_layers,
                constraints=self._constraints(),
            ),
            config=config or SearchConfig(
                evaluation_cap=3,
                greedy_max_starts=3,
            ),
            output_dir=output_dir,
            manifest={"task": "ordinary-persistence"},
            checkpoint_callback=checkpoint_callback,
            checkpoint_interval=checkpoint_interval,
        )

    @staticmethod
    def _surrogate_factory(backend):
        if backend == "bo_rf":
            return lambda _seed: _FakeForest([])
        return None

    def test_adapter_has_no_physical_attempt_callback(self):
        parameters = inspect.signature(Stage1EvaluatorAdapter).parameters

        self.assertNotIn("on_evaluation_started", parameters)

    def test_public_api_exposes_ordinary_accounting(self):
        self.assertIn("build_stage1_search_accounting", _search_runner.__all__)
        self.assertIn("load_completed_search_result", _search_runner.__all__)
        for removed in (
                "load_completed_search_authority",
                "_verify_completion_seal",
                "_load_inference_attempts",
        ):
            self.assertFalse(hasattr(_search_runner, removed), removed)

    def test_runner_preserves_trusted_experiment_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = Stage1SearchRunner(
                adapter=Stage1EvaluatorAdapter(
                    evaluator=_FakeRealEvaluator(),
                    num_layers=1,
                    constraints=self._constraints(),
                ),
                config=SearchConfig(
                    evaluation_cap=3,
                    greedy_max_starts=3,
                ),
                output_dir=Path(tmpdir) / "stage1",
                manifest={"experiment_tag": "ordinary-mrpc-comparator"},
            )

            self.assertEqual(
                runner.manifest["experiment_tag"],
                "ordinary-mrpc-comparator",
            )

    def test_comparator_smoke_budget_limit_round_trips_without_new_forward(self):
        class NoForwardEvaluator(_FakeRealEvaluator):
            def stage1_evaluate(self, gelu, softmax, *, use_train, split):
                raise AssertionError("completed smoke resume must not evaluate")

        for backend in ("greedy", "coinn_ga"):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as tmpdir:
                config = SearchConfig(
                    evaluation_cap=1,
                    greedy_max_starts=3,
                )
                first_evaluator = _FakeRealEvaluator()
                first = Stage1SearchRunner(
                    adapter=Stage1EvaluatorAdapter(
                        evaluator=first_evaluator,
                        num_layers=1,
                        constraints=self._constraints(),
                    ),
                    config=config,
                    output_dir=tmpdir,
                    manifest={"comparator_smoke": True},
                ).run(backend)

                self.assertEqual(first.termination_reason, "evaluation_cap")
                self.assertEqual(first.evaluation_count, 1)
                self.assertEqual(len(first_evaluator.calls), 1)
                self.assertEqual(
                    load_completed_search_result(tmpdir).as_dict(),
                    first.as_dict(),
                )

                resumed = Stage1SearchRunner(
                    adapter=Stage1EvaluatorAdapter(
                        evaluator=NoForwardEvaluator(),
                        num_layers=1,
                        constraints=self._constraints(),
                    ),
                    config=config,
                    output_dir=tmpdir,
                    manifest={"comparator_smoke": True},
                ).run(backend)
                self.assertEqual(resumed.as_dict(), first.as_dict())

    def test_completed_run_writes_minimum_artifacts_and_round_trips(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self._ordinary_runner(tmpdir).run("greedy")
            output = Path(tmpdir)

            for name in (
                    "observations.jsonl",
                    "checkpoint.json",
                    "history.json",
                    "result.json",
                    "summary.json",
                    "manifest.json",
                    "COMPLETED",
            ):
                self.assertTrue((output / name).exists(), name)
            self.assertFalse((output / "inference_attempts.jsonl").exists())
            self.assertFalse((output / "completion_seal.json").exists())

            manifest = _search_runner.read_json_file(output / "manifest.json")
            checkpoint = _search_runner.read_json_file(output / "checkpoint.json")
            self.assertEqual(checkpoint["status"], "complete")
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["backend"], result.algorithm)
            self.assertEqual(manifest["config"], result.config.as_dict())
            self.assertEqual(manifest["evaluation_count"], result.evaluation_count)
            self.assertEqual(
                manifest["unique_evaluation_count"],
                result.unique_evaluation_count,
            )
            self.assertEqual(
                manifest["termination_reason"], result.termination_reason,
            )
            for removed in (
                    "sealed_artifacts",
                    "formal_run_identity",
                    "formal_stage1_contract",
                    "scientific_export_allowed",
                    "inference_attempt_store",
                    "model_inference_attempt_count",
            ):
                self.assertNotIn(removed, manifest)
                self.assertNotIn(removed, checkpoint)
            rows = _search_runner._read_strict_object_jsonl(
                output / "observations.jsonl"
            )
            self.assertEqual(len(rows), result.evaluation_count)
            self.assertEqual(
                load_completed_search_result(output).as_dict(),
                result.as_dict(),
            )

    def test_complete_checkpoint_resume_finalizes_without_forward(self):
        class NoForwardEvaluator(_FakeRealEvaluator):
            def stage1_evaluate(self, gelu, softmax, *, use_train, split):
                raise AssertionError("completed resume must not evaluate")

        with tempfile.TemporaryDirectory() as tmpdir:
            expected = self._ordinary_runner(tmpdir).run("greedy")
            output = Path(tmpdir)
            for name in (
                    "result.json",
                    "summary.json",
                    "manifest.json",
                    "COMPLETED",
            ):
                (output / name).unlink(missing_ok=True)

            evaluator = NoForwardEvaluator()
            resumed = self._ordinary_runner(tmpdir, evaluator=evaluator).run(
                "greedy"
            )

            self.assertEqual(evaluator.calls, [])
            self.assertEqual(resumed.as_dict(), expected.as_dict())
            for name in (
                    "result.json", "summary.json", "manifest.json", "COMPLETED",
            ):
                self.assertTrue((output / name).exists(), name)

    def test_complete_checkpoint_missing_history_fails_before_forward(self):
        class NoForwardEvaluator(_FakeRealEvaluator):
            def stage1_evaluate(self, gelu, softmax, *, use_train, split):
                raise AssertionError("incomplete publication must not evaluate")

        with tempfile.TemporaryDirectory() as tmpdir:
            self._ordinary_runner(tmpdir).run("greedy")
            output = Path(tmpdir)
            (output / "history.json").unlink()
            (output / "manifest.json").unlink()
            (output / "COMPLETED").unlink()
            evaluator = NoForwardEvaluator()

            with self.assertRaisesRegex(RuntimeError, "history.json"):
                self._ordinary_runner(tmpdir, evaluator=evaluator).run("greedy")
            self.assertEqual(evaluator.calls, [])

    def test_atomic_json_publishes_complete_readable_json(self):
        replace_calls = []
        real_replace = os.replace

        def tracked_replace(source, target):
            replace_calls.append((Path(source).name, Path(target).name))
            return real_replace(source, target)

        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.object(
                _search_runner.os,
                "replace",
                side_effect=tracked_replace,
        ):
            target = Path(tmpdir) / "artifact.json"
            _search_runner._atomic_json(target, {"status": "complete"})
            self.assertEqual(
                _search_runner.read_json_file(target), {"status": "complete"}
            )
            self.assertFalse(Path(str(target) + ".tmp").exists())

        self.assertEqual(replace_calls, [("artifact.json.tmp", "artifact.json")])

    def test_forward_failure_retries_missing_action_without_attempt_wal(self):
        class FailOnceEvaluator(_FakeRealEvaluator):
            def stage1_evaluate(self, gelu, softmax, *, use_train, split):
                self.calls.append((tuple(gelu), tuple(softmax), use_train, split))
                raise RuntimeError("forward interrupted")

        config = SearchConfig(
            evaluation_cap=1,
            bo_initial_design_size=1,
            bo_candidate_pool_size=2,
            maximin_candidate_pool_size=4,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            failed = FailOnceEvaluator()
            with self.assertRaisesRegex(RuntimeError, "forward interrupted"):
                self._ordinary_runner(
                    tmpdir,
                    evaluator=failed,
                    config=config,
                ).run("bo_rf", surrogate_factory=lambda _seed: _FakeForest([]))

            output = Path(tmpdir)
            checkpoint = _search_runner.read_json_file(output / "checkpoint.json")
            self.assertEqual(checkpoint["status"], "failed")
            self.assertEqual(checkpoint["observation_count"], 0)
            self.assertEqual(load_search_preload(output / "checkpoint.json"), ())
            self.assertFalse((output / "inference_attempts.jsonl").exists())

            recovered = _FakeRealEvaluator()
            result = self._ordinary_runner(
                tmpdir,
                evaluator=recovered,
                config=config,
            ).run("bo_rf", surrogate_factory=lambda _seed: _FakeForest([]))

            self.assertEqual(result.evaluation_count, 1)
            self.assertEqual(recovered.calls[0], failed.calls[0])

    def test_resume_repairs_truncated_observation_tail_and_preserves_complete_rows(self):
        config = SearchConfig(evaluation_cap=3, greedy_max_starts=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            interrupted = _FakeRealEvaluator()

            def stop_after_two(payload):
                if (
                        payload.get("status") == "running"
                        and int(payload.get("observation_count", 0)) == 2
                ):
                    raise RuntimeError("publication interrupted")

            with self.assertRaisesRegex(RuntimeError, "publication interrupted"):
                self._ordinary_runner(
                    tmpdir,
                    evaluator=interrupted,
                    config=config,
                    checkpoint_callback=stop_after_two,
                ).run("greedy")

            output = Path(tmpdir)
            checkpoint_path = output / "checkpoint.json"
            checkpoint = _search_runner.read_json_file(checkpoint_path)
            checkpoint["observation_count"] = 1
            checkpoint["observation_store"]["observation_count"] = 1
            _search_runner._atomic_json(checkpoint_path, checkpoint)
            with (output / "observations.jsonl").open("ab") as handle:
                handle.write(b'{"action":')

            resumed_evaluator = _FakeRealEvaluator()
            result = self._ordinary_runner(
                tmpdir,
                evaluator=resumed_evaluator,
                config=config,
            ).run("greedy")

            self.assertEqual(result.evaluation_count, 3)
            self.assertEqual(len(resumed_evaluator.calls), 1)
            rows = _search_runner._read_strict_object_jsonl(
                output / "observations.jsonl"
            )
            self.assertEqual(len(rows), 3)
            self.assertEqual(
                [tuple(row["action"]) for row in rows[:2]],
                [item.action for item in result.observations[:2]],
            )

    def test_resume_rejects_config_mismatch_before_forward(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._ordinary_runner(tmpdir).run("greedy")
            output = Path(tmpdir)
            before = {
                name: (output / name).read_bytes()
                for name in (
                    "checkpoint.json", "observations.jsonl", "history.json",
                    "result.json", "summary.json", "manifest.json", "COMPLETED",
                )
            }
            evaluator = _FakeRealEvaluator()
            with self.assertRaisesRegex(
                    RuntimeError, "search configuration does not match",
            ):
                self._ordinary_runner(
                    tmpdir,
                    evaluator=evaluator,
                    config=SearchConfig(
                        evaluation_cap=2,
                        greedy_max_starts=2,
                    ),
                ).run("greedy")
            self.assertEqual(evaluator.calls, [])
            self.assertEqual(
                before,
                {name: (output / name).read_bytes() for name in before},
            )

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

    def test_duplicate_direct_preload_fails_before_forward_for_every_backend(self):
        constraints = self._constraints()
        duplicate = _evaluation(
            (0,),
            loss=0.5,
            metrics=(0.8,),
            constraints=constraints,
        )
        for backend in ("greedy", "bo_rf", "coinn_ga"):
            with self.subTest(backend=backend):
                real = _FakeRealEvaluator()
                with self.assertRaisesRegex(ValueError, "duplicate preload"):
                    Stage1SearchRunner(
                        adapter=Stage1EvaluatorAdapter(
                            evaluator=real,
                            num_layers=1,
                            constraints=constraints,
                        ),
                        config=SearchConfig(evaluation_cap=3),
                    ).run(
                        backend,
                        preload=(duplicate, duplicate),
                        surrogate_factory=self._surrogate_factory(backend),
                    )
                self.assertEqual(real.calls, [])

    def test_direct_preload_constraint_mismatch_fails_before_forward(self):
        current_constraints = self._constraints()
        stale = _evaluation(
            (0,),
            loss=0.508,
            metrics=(0.785,),
            constraints=Stage1Constraints(
                baseline_loss=0.5,
                baseline_metrics=(0.8,),
                loss_max=0.51,
                metric_mins=(0.78,),
            ),
        )
        real = _FakeRealEvaluator()

        with self.assertRaisesRegex(ValueError, "preload constraints"):
            Stage1SearchRunner(
                adapter=Stage1EvaluatorAdapter(
                    evaluator=real,
                    num_layers=1,
                    constraints=current_constraints,
                ),
                config=SearchConfig(evaluation_cap=3, greedy_max_starts=3),
            ).run("greedy", preload=(stale,))
        self.assertEqual(real.calls, [])

    def test_partial_bo_and_ga_preload_replay_exactly(self):
        cases = (
            (
                "bo_rf",
                SearchConfig(
                    seed=17,
                    evaluation_cap=8,
                    bo_initial_design_size=4,
                    bo_candidate_pool_size=8,
                    bo_no_improvement_patience=20,
                    maximin_candidate_pool_size=32,
                ),
                5,
            ),
            (
                "coinn_ga",
                SearchConfig(
                    seed=23,
                    evaluation_cap=10,
                    ga_population_size=4,
                    ga_elite_count=1,
                    ga_update_generations=2,
                    ga_duplicate_attempts=64,
                    maximin_candidate_pool_size=64,
                ),
                5,
            ),
        )
        for backend, config, prefix_count in cases:
            with self.subTest(backend=backend):
                reference = Stage1SearchRunner(
                    adapter=Stage1EvaluatorAdapter(
                        evaluator=_FakeRealEvaluator(),
                        num_layers=3,
                        constraints=self._constraints(),
                    ),
                    config=config,
                ).run(
                    backend,
                    surrogate_factory=self._surrogate_factory(backend),
                )
                resumed_real = _FakeRealEvaluator()
                resumed = Stage1SearchRunner(
                    adapter=Stage1EvaluatorAdapter(
                        evaluator=resumed_real,
                        num_layers=3,
                        constraints=self._constraints(),
                    ),
                    config=config,
                ).run(
                    backend,
                    surrogate_factory=self._surrogate_factory(backend),
                    preload=reference.observations[:prefix_count],
                )

                self.assertEqual(resumed.history, reference.history)
                self.assertEqual(
                    tuple(
                        _scientific_evaluation_dict(item)
                        for item in resumed.observations
                    ),
                    tuple(
                        _scientific_evaluation_dict(item)
                        for item in reference.observations
                    ),
                )
                self.assertEqual(resumed.best.action, reference.best.action)
                self.assertEqual(
                    resumed.termination_reason,
                    reference.termination_reason,
                )
                self.assertEqual(
                    len(resumed_real.calls),
                    reference.evaluation_count - prefix_count,
                )

    def test_greedy_checkpoint_resume_rebuilds_identical_history(self):
        config = SearchConfig(evaluation_cap=3, greedy_max_starts=3)
        with tempfile.TemporaryDirectory() as reference_dir:
            reference = self._ordinary_runner(
                reference_dir,
                config=config,
            ).run("greedy")

        with tempfile.TemporaryDirectory() as resumed_dir:
            interrupted_real = _FakeRealEvaluator()

            def interrupt_after_two(payload):
                if (
                        payload.get("status") == "running"
                        and int(payload.get("observation_count", 0)) == 2
                ):
                    raise RuntimeError("publication interrupted")

            with self.assertRaisesRegex(
                    RuntimeError, "publication interrupted",
            ):
                self._ordinary_runner(
                    resumed_dir,
                    evaluator=interrupted_real,
                    config=config,
                    checkpoint_callback=interrupt_after_two,
                ).run("greedy")
            self.assertEqual(len(interrupted_real.calls), 2)

            resumed_real = _FakeRealEvaluator()
            resumed = self._ordinary_runner(
                resumed_dir,
                evaluator=resumed_real,
                config=config,
            ).run("greedy")

        self.assertEqual(len(resumed_real.calls), 1)
        self.assertEqual(resumed.history, reference.history)
        self.assertEqual(
            tuple(item.action for item in resumed.observations),
            tuple(item.action for item in reference.observations),
        )

    def test_bo_and_ga_checkpoint_resume_rebuild_identical_history(self):
        cases = (
            (
                "bo_rf",
                SearchConfig(
                    seed=17,
                    evaluation_cap=8,
                    bo_initial_design_size=4,
                    bo_candidate_pool_size=8,
                    bo_no_improvement_patience=20,
                    maximin_candidate_pool_size=32,
                ),
                5,
            ),
            (
                "coinn_ga",
                SearchConfig(
                    seed=23,
                    evaluation_cap=10,
                    ga_population_size=4,
                    ga_elite_count=1,
                    ga_update_generations=2,
                    ga_duplicate_attempts=64,
                    maximin_candidate_pool_size=64,
                ),
                5,
            ),
        )
        for backend, config, prefix_count in cases:
            with self.subTest(backend=backend):
                with tempfile.TemporaryDirectory() as reference_dir:
                    reference = Stage1SearchRunner(
                        adapter=Stage1EvaluatorAdapter(
                            evaluator=_FakeRealEvaluator(),
                            num_layers=3,
                            constraints=self._constraints(),
                        ),
                        config=config,
                        output_dir=reference_dir,
                        manifest={"task": f"{backend}-reference"},
                        checkpoint_interval=1,
                    ).run(
                        backend,
                        surrogate_factory=self._surrogate_factory(backend),
                    )

                with tempfile.TemporaryDirectory() as resumed_dir:
                    interrupted_real = _FakeRealEvaluator()

                    def interrupt(payload):
                        if (
                                payload.get("status") == "running"
                                and int(payload.get("observation_count", 0))
                                == prefix_count
                        ):
                            raise RuntimeError("publication interrupted")

                    with self.assertRaisesRegex(
                            RuntimeError, "publication interrupted",
                    ):
                        Stage1SearchRunner(
                            adapter=Stage1EvaluatorAdapter(
                                evaluator=interrupted_real,
                                num_layers=3,
                                constraints=self._constraints(),
                            ),
                            config=config,
                            output_dir=resumed_dir,
                            manifest={"task": f"{backend}-resume"},
                            checkpoint_callback=interrupt,
                            checkpoint_interval=1,
                        ).run(
                            backend,
                            surrogate_factory=self._surrogate_factory(backend),
                        )
                    self.assertEqual(len(interrupted_real.calls), prefix_count)

                    resumed_real = _FakeRealEvaluator()
                    resumed = Stage1SearchRunner(
                        adapter=Stage1EvaluatorAdapter(
                            evaluator=resumed_real,
                            num_layers=3,
                            constraints=self._constraints(),
                        ),
                        config=config,
                        output_dir=resumed_dir,
                        manifest={"task": f"{backend}-resume"},
                        checkpoint_interval=1,
                    ).run(
                        backend,
                        surrogate_factory=self._surrogate_factory(backend),
                    )

                self.assertEqual(resumed.history, reference.history)
                self.assertEqual(
                    tuple(item.action for item in resumed.observations),
                    tuple(item.action for item in reference.observations),
                )
                self.assertEqual(
                    len(resumed_real.calls),
                    reference.evaluation_count - prefix_count,
                )

    def test_ga_graceful_stop_persists_boundary_and_resumes_exactly(self):
        config = SearchConfig(
            seed=23,
            evaluation_cap=18,
            ga_population_size=6,
            ga_elite_count=2,
            ga_update_generations=3,
            ga_stop_on_no_improvement=False,
            ga_duplicate_attempts=64,
            maximin_candidate_pool_size=64,
        )
        with tempfile.TemporaryDirectory() as reference_dir:
            reference = Stage1SearchRunner(
                adapter=Stage1EvaluatorAdapter(
                    evaluator=_FakeRealEvaluator(),
                    num_layers=4,
                    constraints=self._constraints(),
                ),
                config=config,
                output_dir=reference_dir,
                checkpoint_interval=50,
            ).run("coinn_ga")

        with tempfile.TemporaryDirectory() as resumed_dir:
            interrupted_real = _FakeRealEvaluator()
            runner = Stage1SearchRunner(
                adapter=Stage1EvaluatorAdapter(
                    evaluator=interrupted_real,
                    num_layers=4,
                    constraints=self._constraints(),
                ),
                config=config,
                output_dir=resumed_dir,
                checkpoint_interval=50,
                stop_requested=lambda: len(interrupted_real.calls) >= 8,
            )

            with self.assertRaises(Stage1SearchGracefulStop) as stopped:
                runner.run("coinn_ga")

            self.assertEqual(stopped.exception.observation_count, 8)
            checkpoint = _search_runner.read_json_file(
                Path(resumed_dir) / "checkpoint.json"
            )
            self.assertEqual(checkpoint["status"], "stopped")
            self.assertEqual(checkpoint["observation_count"], 8)
            self.assertEqual(
                len(load_search_preload(Path(resumed_dir) / "checkpoint.json")),
                8,
            )

            resumed_real = _FakeRealEvaluator()
            resumed = Stage1SearchRunner(
                adapter=Stage1EvaluatorAdapter(
                    evaluator=resumed_real,
                    num_layers=4,
                    constraints=self._constraints(),
                ),
                config=config,
                output_dir=resumed_dir,
                checkpoint_interval=50,
            ).run("coinn_ga")

        self.assertEqual(resumed.history, reference.history)
        self.assertEqual(
            tuple(
                _scientific_evaluation_dict(item)
                for item in resumed.observations
            ),
            tuple(
                _scientific_evaluation_dict(item)
                for item in reference.observations
            ),
        )
        self.assertEqual(resumed.best.action, reference.best.action)
        self.assertEqual(len(resumed_real.calls), 10)

    def test_exception_checkpoint_preserves_complete_observations(self):
        real = _FakeRealEvaluator()
        original = real.stage1_evaluate

        def fail_on_third(*args, **kwargs):
            if len(real.calls) >= 2:
                raise RuntimeError("interrupted")
            return original(*args, **kwargs)

        real.stage1_evaluate = fail_on_third
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                self._ordinary_runner(
                    tmpdir,
                    evaluator=real,
                    checkpoint_interval=50,
                ).run("greedy")

            checkpoint_path = Path(tmpdir) / "checkpoint.json"
            checkpoint = _search_runner.read_json_file(checkpoint_path)
            self.assertEqual(checkpoint["status"], "failed")
            self.assertEqual(checkpoint["observation_count"], 2)
            self.assertIn("interrupted", checkpoint["error"])
            recovered = load_search_preload(checkpoint_path)
            self.assertEqual(len(recovered), 2)

    def test_all_invalid_stage1_results_are_not_published_complete(self):
        class InvalidEvaluator(_FakeRealEvaluator):
            def stage1_evaluate(self, gelu, softmax, *, use_train, split):
                self.calls.append((tuple(gelu), tuple(softmax), use_train, split))
                return {
                    "loss": 0.5,
                    "metrics": [0.8],
                    "time_ms": 1.0,
                    "valid": False,
                }

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(
                    RuntimeError, "no valid model-forward evaluation",
            ):
                self._ordinary_runner(
                    tmpdir,
                    evaluator=InvalidEvaluator(),
                ).run("greedy")
            output = Path(tmpdir)
            checkpoint = _search_runner.read_json_file(output / "checkpoint.json")
            self.assertEqual(checkpoint["status"], "failed")
            self.assertEqual(checkpoint["observation_count"], 3)
            self.assertFalse((output / "manifest.json").exists())
            self.assertFalse((output / "COMPLETED").exists())

    def test_incomplete_greedy_contract_is_never_published_complete(self):
        original_run_search = _search_runner.run_search

        def omit_completion_proof(*args, **kwargs):
            result = original_run_search(*args, **kwargs)
            return SearchResult(
                algorithm=result.algorithm,
                config=result.config,
                best=result.best,
                observations=result.observations,
                history=tuple(
                    row for row in result.history
                    if row.get("phase") != "verified_local_optimum"
                ),
                termination_reason=result.termination_reason,
            )

        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.object(
                _search_runner,
                "run_search",
                side_effect=omit_completion_proof,
        ):
            with self.assertRaisesRegex(RuntimeError, "completion contract"):
                self._ordinary_runner(tmpdir).run("greedy")
            output = Path(tmpdir)
            checkpoint = _search_runner.read_json_file(output / "checkpoint.json")
            self.assertEqual(checkpoint["status"], "failed")
            self.assertIn("completion contract", checkpoint["error"])
            self.assertFalse((output / "manifest.json").exists())
            self.assertFalse((output / "COMPLETED").exists())

    def test_same_runner_retry_does_not_repeat_completed_observations(self):
        interrupted = False

        def interrupt_once(payload):
            nonlocal interrupted
            if (
                    not interrupted
                    and payload.get("status") == "running"
                    and int(payload.get("observation_count", 0)) == 2
            ):
                interrupted = True
                raise RuntimeError("publication interrupted")

        with tempfile.TemporaryDirectory() as tmpdir:
            real = _FakeRealEvaluator()
            runner = self._ordinary_runner(
                tmpdir,
                evaluator=real,
                checkpoint_callback=interrupt_once,
            )
            with self.assertRaisesRegex(
                    RuntimeError, "publication interrupted",
            ):
                runner.run("greedy")

            result = runner.run("greedy")

            self.assertEqual(result.evaluation_count, 3)
            self.assertEqual(len(real.calls), 3)
            self.assertFalse(
                (Path(tmpdir) / "inference_attempts.jsonl").exists()
            )

    def test_comparator_config_table_is_exact_for_each_backend(self):
        for backend in ("bo_rf", "greedy", "coinn_ga"):
            with self.subTest(backend=backend):
                actual = _search_baselines.stage1_comparator_search_config(
                    backend
                )
                self.assertEqual(
                    actual.as_dict(),
                    _comparator_config(backend).as_dict(),
                )

    def test_comparator_bo_rf_uses_locked_random_forest_parameters(self):
        captured = {}
        sklearn = ModuleType("sklearn")
        sklearn.__path__ = []
        ensemble = ModuleType("sklearn.ensemble")

        class CapturingRandomForest:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        ensemble.RandomForestRegressor = CapturingRandomForest
        sklearn.ensemble = ensemble
        with mock.patch.dict(
                sys.modules,
                {"sklearn": sklearn, "sklearn.ensemble": ensemble},
        ):
            _search_baselines._default_surrogate_factory(
                _comparator_config("bo_rf")
            )(42)

        self.assertEqual(
            captured,
            {
                "n_estimators": 128,
                "min_samples_leaf": 2,
                "max_features": 0.75,
                "bootstrap": True,
                "random_state": 42,
                "n_jobs": -1,
            },
        )

    def test_comparator_setup_rejects_config_drift(self):
        drifted = _comparator_config("bo_rf").as_dict()
        drifted["evaluation_cap"] = 1
        drifted["bo_initial_design_size"] = 1

        with self.assertRaisesRegex(RuntimeError, "comparator protocol"):
            _search_baselines.validate_stage1_comparator_setup(
                backend="bo_rf",
                config=SearchConfig(**drifted),
                num_layers=12,
                constraints=_comparator_constraints(),
            )

    def test_comparator_setup_rejects_layer_count(self):
        with self.assertRaisesRegex(RuntimeError, "12 layers"):
            _search_baselines.validate_stage1_comparator_setup(
                backend="greedy",
                config=_comparator_config("greedy"),
                num_layers=1,
                constraints=_comparator_constraints(),
            )

    def test_comparator_setup_rejects_constraint_drift(self):
        relaxed = Stage1Constraints.from_baseline(
            baseline_loss=0.5,
            baseline_metrics=(0.8, 0.8),
            loss_relative_tolerance=0.01,
            metric_relative_tolerance=0.01,
            metric_names=("accuracy", "weighted_f1"),
        )

        with self.assertRaisesRegex(RuntimeError, "0.1%"):
            _search_baselines.validate_stage1_comparator_setup(
                backend="greedy",
                config=_comparator_config("greedy"),
                num_layers=12,
                constraints=relaxed,
            )

    def test_completed_resume_rejects_mismatched_backend_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._ordinary_runner(tmpdir).run("greedy")
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest = _search_runner.read_json_file(manifest_path)
            manifest["backend"] = "bo_rf"
            _search_runner._atomic_json(manifest_path, manifest)

            with self.assertRaisesRegex(RuntimeError, "backend"):
                load_completed_search_result(tmpdir)

    def test_completed_resume_rejects_missing_contract_constraints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._ordinary_runner(tmpdir).run("greedy")
            checkpoint_path = Path(tmpdir) / "checkpoint.json"
            checkpoint = _search_runner.read_json_file(checkpoint_path)
            del checkpoint["contract"]["constraints"]
            _search_runner._atomic_json(checkpoint_path, checkpoint)

            with self.assertRaisesRegex(RuntimeError, "contract constraints"):
                load_completed_search_result(tmpdir)

    def test_completed_resume_rejects_mixed_observation_constraints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._ordinary_runner(tmpdir).run("greedy")
            observation_path = Path(tmpdir) / "observations.jsonl"
            rows = _search_runner._read_strict_object_jsonl(observation_path)
            rows[0]["constraints"]["baseline_loss"] = 0.4
            _search_runner.write_jsonl_rows(
                observation_path,
                rows,
                sort_keys=True,
            )

            with self.assertRaisesRegex(RuntimeError, "observation constraints"):
                load_completed_search_result(tmpdir)

    def test_completed_resume_rejects_stale_but_observed_best(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self._ordinary_runner(tmpdir).run("greedy")
            stale = result.observations[0]
            if stale.action == result.best.action:
                stale = result.observations[-1]
            self.assertNotEqual(stale.action, result.best.action)
            for name in ("result.json", "summary.json"):
                path = Path(tmpdir) / name
                payload = _search_runner.read_json_file(path)
                payload["best"] = stale.as_dict()
                _search_runner._atomic_json(path, payload)

            with self.assertRaisesRegex(RuntimeError, "result|best"):
                load_completed_search_result(tmpdir)

    def test_completed_resume_rejects_lost_complete_observation_row(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._ordinary_runner(tmpdir).run("greedy")
            path = Path(tmpdir) / "observations.jsonl"
            rows = path.read_bytes().splitlines(keepends=True)
            path.write_bytes(b"".join(rows[:-1]))

            with self.assertRaisesRegex(RuntimeError, "observation count"):
                load_completed_search_result(tmpdir)

    def test_observation_jsonl_rejects_non_object_and_middle_corruption(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "observations.jsonl"
            path.write_text("[]\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected JSON object"):
                _search_runner._read_strict_object_jsonl(path)

            path.write_text('{"ok": 1}\nnot-json\n{"ok": 2}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                _search_runner._read_strict_object_jsonl(path)

if __name__ == "__main__":
    unittest.main()
