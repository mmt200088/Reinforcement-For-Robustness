from __future__ import annotations

import unittest

import numpy as np

from blb_stage2_rl.layerwise_action import (
    decode_layer_gene,
    decode_layerwise_action_genes,
    encode_layer_gene,
    encode_layerwise_action_matrix,
)
from blb_stage2_rl.search_baselines import (
    ConstraintLimits,
    LayerwiseSearchSpace,
    SearchConfig,
    SearchEvaluation,
    SearchMetrics,
    _bo_acquisition_key,
    _diverse_second_parent,
    _hamming_distance,
    _replacement_mutation,
    _select_hamming_diverse_elites,
    _structured_initial_design,
    candidate_rank_key,
    normalize_search_backend,
    run_search,
)


LIMITS = ConstraintLimits(
    loss_max=1.01,
    metric1_min=0.89,
    metric2_min=0.84,
    loss_std_max=0.020,
    metric1_std_max=0.015,
    metric2_std_max=0.018,
)


def _evaluation(
        action,
        *,
        loss=1.0,
        metric1=0.90,
        metric2=0.85,
        loss_std=0.010,
        metric1_std=0.010,
        metric2_std=0.010,
        valid=True,
        probabilities=(),
        gate_probability=None,
        inference_performed=True,
        ):
    return SearchEvaluation(
        action_matrix=action,
        metrics=SearchMetrics(
            loss_mean=loss,
            metric1_mean=metric1,
            metric2_mean=metric2,
            loss_std=loss_std,
            metric1_std=metric1_std,
            metric2_std=metric2_std,
        ),
        limits=LIMITS,
        valid=valid,
        constraint_probabilities=tuple(probabilities),
        gate_probability=gate_probability,
        metadata={"inference_performed": bool(inference_performed)},
    )


def _probability_evaluation(action, margins, *, valid=True):
    return _evaluation(
        action,
        valid=valid,
        probabilities=tuple(0.5 + float(value) for value in margins),
        gate_probability=0.5,
    )


class _MeanTree:
    def __init__(self):
        self._mean = None

    def fit(self, features, targets):
        del features
        self._mean = np.asarray(targets, dtype=float).mean(axis=0)
        return self

    def predict(self, features):
        return np.repeat(
            self._mean.reshape(1, -1),
            np.asarray(features).shape[0],
            axis=0,
        )


class _TinyForest(_MeanTree):
    fitted_features = []

    def fit(self, features, targets):
        array = np.asarray(features, dtype=float)
        type(self).fitted_features.append(array.copy())
        super().fit(array, targets)
        self.estimators_ = [
            _MeanTree().fit(array, targets),
            _MeanTree().fit(array, targets),
        ]
        return self


class _ScriptedConstraintForest:
    def fit(self, _features, _targets):
        self.estimators_ = [self]
        return self

    def predict(self, features):
        rows = np.asarray(features).reshape(-1, 2, 6)
        genes = np.argmax(rows, axis=2)
        result = []
        for action in genes:
            margins = np.full(6, 0.1, dtype=float)
            if tuple(action) == (0, 1):
                margins[0] = -2.0
            elif tuple(action) == (0, 2):
                margins[:2] = -0.01
            else:
                margins[:] = -1.0
            result.append(margins)
        return np.asarray(result)


class AtomicLayerGeneTests(unittest.TestCase):
    def test_six_valued_codec_is_exact_and_round_trips_matrix(self):
        expected = {
            (0, 0): 0,
            (0, 1): 1,
            (0, 2): 2,
            (1, 0): 3,
            (1, 1): 4,
            (1, 2): 5,
        }
        for row, gene in expected.items():
            with self.subTest(row=row):
                self.assertEqual(encode_layer_gene(row), gene)
                self.assertEqual(decode_layer_gene(gene), row)

        matrix = ((0, 2), (1, 0), (1, 2))
        genes = encode_layerwise_action_matrix(matrix)
        self.assertEqual(genes, (2, 3, 5))
        self.assertEqual(decode_layerwise_action_genes(genes), matrix)

    def test_runtime_flatten_stays_compatible_but_search_neighbors_are_atomic(self):
        space = LayerwiseSearchSpace(num_layers=2)
        action = ((0, 1), (1, 2))

        self.assertEqual(space.dimensions, (2, 3, 2, 3))
        self.assertEqual(space.gene_dimensions, (6, 6))
        self.assertEqual(space.flatten(action), (0, 1, 1, 2))
        self.assertEqual(space.unflatten((0, 1, 1, 2)), action)
        self.assertEqual(space.unflatten(space.genes(action)), action)
        self.assertEqual(space.cardinality, 36)

        before_genes = space.genes(action)
        neighbors = tuple(space.neighbors(action))
        self.assertEqual(len(neighbors), 10)
        for neighbor in neighbors:
            changed_layers = [
                index
                for index, (before, after) in enumerate(
                    zip(before_genes, space.genes(neighbor))
                )
                if before != after
            ]
            self.assertEqual(len(changed_layers), 1)

    def test_two_opt_changes_exactly_two_whole_layers(self):
        space = LayerwiseSearchSpace(num_layers=3)
        action = space.from_genes((0, 1, 2))
        neighbors = tuple(space.two_opt_neighbors(action))

        self.assertEqual(len(neighbors), 3 * 25)
        for neighbor in neighbors:
            self.assertEqual(
                sum(
                    left != right
                    for left, right in zip(
                        space.genes(action), space.genes(neighbor)
                    )
                ),
                2,
            )

    def test_crossover_never_splits_a_layer_gene(self):
        space = LayerwiseSearchSpace(num_layers=6)
        first = space.from_genes((0, 1, 2, 3, 4, 5))
        second = space.from_genes((5, 4, 3, 2, 1, 0))

        for mode in ("two_point", "uniform"):
            with self.subTest(mode=mode):
                child = space.crossover(
                    first,
                    second,
                    np.random.default_rng(17),
                    mode=mode,
                )
                for layer_idx, gene in enumerate(space.genes(child)):
                    self.assertIn(
                        gene,
                        (space.genes(first)[layer_idx],
                         space.genes(second)[layer_idx]),
                    )

    def test_malformed_actions_and_genes_are_rejected(self):
        space = LayerwiseSearchSpace(num_layers=2)
        with self.assertRaisesRegex(ValueError, "2 layers"):
            space.validate(((0, 0),))
        with self.assertRaisesRegex(ValueError, "two coordinates"):
            space.validate(((0, 0, 0), (0, 0)))
        with self.assertRaisesRegex(ValueError, "outside"):
            space.validate(((0, 3), (0, 0)))
        with self.assertRaisesRegex(ValueError, "layer gene"):
            decode_layer_gene(6)


class ConstraintAndRankingTests(unittest.TestCase):
    def test_all_six_precision_and_stability_constraints_are_required(self):
        self.assertTrue(_evaluation(((0, 0),)).feasible)

        violations = (
            {"loss": 1.02},
            {"metric1": 0.88},
            {"metric2": 0.83},
            {"loss_std": 0.021},
            {"metric1_std": 0.016},
            {"metric2_std": 0.019},
        )
        for override in violations:
            with self.subTest(override=override):
                self.assertFalse(_evaluation(((0, 0),), **override).feasible)

    def test_infeasible_rank_is_valid_then_fail_count_total_worst_resource(self):
        invalid = _probability_evaluation(
            ((1, 2),), (-0.001, 0.1, 0.1, 0.1, 0.1, 0.1), valid=False,
        )
        two_failures = _probability_evaluation(
            ((1, 2),), (-0.01, -0.01, 0.1, 0.1, 0.1, 0.1),
        )
        one_failure = _probability_evaluation(
            ((0, 0),), (-0.40, 0.1, 0.1, 0.1, 0.1, 0.1),
        )
        lower_total = _probability_evaluation(
            ((0, 1),), (-0.20, -0.10, 0.1, 0.1, 0.1, 0.1),
        )
        higher_total = _probability_evaluation(
            ((1, 2),), (-0.25, -0.10, 0.1, 0.1, 0.1, 0.1),
        )
        lower_worst = _probability_evaluation(
            ((0, 1),), (-0.25, -0.25, 0.1, 0.1, 0.1, 0.1),
        )
        higher_worst = _probability_evaluation(
            ((1, 2),), (-0.375, -0.125, 0.1, 0.1, 0.1, 0.1),
        )

        self.assertGreater(candidate_rank_key(two_failures), candidate_rank_key(invalid))
        self.assertGreater(candidate_rank_key(one_failure), candidate_rank_key(two_failures))
        self.assertGreater(candidate_rank_key(lower_total), candidate_rank_key(higher_total))
        self.assertGreater(candidate_rank_key(lower_worst), candidate_rank_key(higher_worst))

    def test_feasible_resource_and_communication_weighting_are_preserved(self):
        feasible = _evaluation(((1, 1),))
        infeasible = _evaluation(((1, 2),), loss=1.02)
        self.assertGreater(candidate_rank_key(feasible), candidate_rank_key(infeasible))

        communication_heavy = SearchEvaluation(
            action_matrix=((0, 2),),
            metrics=feasible.metrics,
            limits=LIMITS,
            communication_importance_ratio=3.0,
        )
        compute_weighted = SearchEvaluation(
            action_matrix=((1, 0),),
            metrics=feasible.metrics,
            limits=LIMITS,
            communication_importance_ratio=3.0,
        )
        self.assertGreater(
            communication_heavy.resource.ppo_resource_score,
            compute_weighted.resource.ppo_resource_score,
        )

    def test_invalid_candidate_is_never_fallback_when_any_valid_exists(self):
        space = LayerwiseSearchSpace(1)

        def evaluator(action):
            gene = space.genes(action)[0]
            if gene == 0:
                return _evaluation(action, valid=False)
            margin = -0.05 if gene == 3 else -0.20 - gene * 0.01
            return _probability_evaluation(
                action, (margin, 0.1, 0.1, 0.1, 0.1, 0.1),
            )

        result = run_search(
            "greedy",
            space,
            evaluator,
            SearchConfig(evaluation_budget=space.cardinality),
        )
        self.assertTrue(result.best.valid)
        self.assertEqual(space.genes(result.best.action_matrix), (3,))


class GreedySearchTests(unittest.TestCase):
    def test_legacy_checkpoint_callback_still_receives_observation_tuple(self):
        space = LayerwiseSearchSpace(1)
        snapshots = []
        run_search(
            "greedy",
            space,
            lambda action: _evaluation(action),
            SearchConfig(evaluation_budget=space.cardinality),
            checkpoint_callback=snapshots.append,
        )

        self.assertEqual([len(row) for row in snapshots], list(range(1, 7)))
        self.assertTrue(all(isinstance(row, tuple) for row in snapshots))

    def test_greedy_uses_six_anchors_accepts_two_opt_then_returns_to_one_opt(self):
        space = LayerwiseSearchSpace(3)
        target_genes = (0, 1, 2)
        calls = []

        def evaluator(action):
            calls.append(action)
            genes = space.genes(action)
            if genes == target_genes:
                return _evaluation(action)
            if len(set(genes)) == 1:
                violation = -0.05
            else:
                violation = -0.20
            return _probability_evaluation(
                action, (violation, 0.1, 0.1, 0.1, 0.1, 0.1),
            )

        result = run_search(
            "greedy",
            space,
            evaluator,
            SearchConfig(evaluation_budget=space.cardinality, seed=17),
        )

        self.assertEqual(space.genes(result.best.action_matrix), target_genes)
        self.assertEqual(len(calls), len(set(calls)))
        phases = [row["phase"] for row in result.history]
        accepted_two_opt = next(
            index for index, row in enumerate(result.history)
            if row["phase"] == "greedy_2opt" and row["accepted"]
        )
        self.assertIn("greedy_1opt", phases[accepted_two_opt + 1:])
        self.assertIn("greedy_final_verification", phases)
        self.assertEqual(
            [space.genes(item.action_matrix) for item in result.observations[:6]],
            [(gene, gene, gene) for gene in range(6)],
        )


class BayesianSearchTests(unittest.TestCase):
    def test_bo_uses_categorical_one_hot_and_obeys_hard_budget(self):
        _TinyForest.fitted_features = []
        space = LayerwiseSearchSpace(num_layers=2)
        calls = []

        def evaluator(action):
            calls.append(action)
            return _evaluation(action)

        result = run_search(
            "Bayesian-Optimization",
            space,
            evaluator,
            SearchConfig(
                evaluation_budget=12,
                seed=17,
                initial_design_size=8,
                candidate_pool_size=16,
                patience_generations=20,
            ),
            surrogate_factory=lambda _seed: _TinyForest(),
        )

        self.assertEqual(result.evaluation_count, 12)
        self.assertEqual(len(calls), len(set(calls)))
        self.assertTrue(_TinyForest.fitted_features)
        for features in _TinyForest.fitted_features:
            self.assertEqual(features.shape[1], 12)
            reshaped = features.reshape(features.shape[0], 2, 6)
            np.testing.assert_allclose(reshaped.sum(axis=2), 1.0)
            self.assertTrue(np.all((reshaped == 0.0) | (reshaped == 1.0)))

    def test_bo_native_no_improvement_convergence_stops_before_hard_budget(self):
        _TinyForest.fitted_features = []
        space = LayerwiseSearchSpace(2)

        result = run_search(
            "bo_rf",
            space,
            lambda action: _evaluation(action, valid=False),
            SearchConfig(
                evaluation_budget=20,
                seed=17,
                initial_design_size=6,
                candidate_pool_size=12,
                patience_generations=2,
            ),
            surrogate_factory=lambda _seed: _TinyForest(),
        )

        self.assertEqual(result.termination_reason, "bo_no_improvement")
        self.assertEqual(result.evaluation_count, 8)
        self.assertLess(result.evaluation_count, 20)

    def test_bo_acquisition_preserves_constraint_lexicographic_order(self):
        fewer_failed = _bo_acquisition_key(
            has_feasible_incumbent=False,
            probability_of_feasibility=0.0,
            expected_improvement=0.0,
            expected_failed_constraints=1.0,
            expected_total_violation=2.0,
            expected_worst_violation=2.0,
            exploration_tiebreak=0.0,
            objective_tiebreak=0.0,
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
        space = LayerwiseSearchSpace(2)
        result = run_search(
            "bo_rf",
            space,
            lambda action: _probability_evaluation(
                action, (-0.1,) * 6,
            ),
            SearchConfig(
                evaluation_budget=7,
                seed=17,
                initial_design_size=6,
                candidate_pool_size=36,
                patience_generations=10,
            ),
            surrogate_factory=lambda _seed: _ScriptedConstraintForest(),
        )

        self.assertEqual(
            space.genes(result.observations[-1].action_matrix), (0, 1),
        )

    def test_structured_p64_design_is_six_anchors_thirty_balanced_then_maximin(self):
        space = LayerwiseSearchSpace(7)
        design = _structured_initial_design(
            space, np.random.default_rng(17), 64,
        )

        self.assertEqual(len(design), 64)
        self.assertEqual(len(set(design)), 64)
        self.assertEqual(
            [space.genes(action) for action in design[:6]],
            [(gene,) * 7 for gene in range(6)],
        )
        one_layer = [space.genes(action) for action in design[6:36]]
        self.assertTrue(all(sum(gene != 0 for gene in genes) == 1 for genes in one_layer))
        layer_counts = [
            sum(genes[layer] != 0 for genes in one_layer)
            for layer in range(7)
        ]
        self.assertLessEqual(max(layer_counts) - min(layer_counts), 1)
        alternative_counts = [
            sum(gene == alternative for genes in one_layer for gene in genes)
            for alternative in range(1, 6)
        ]
        self.assertLessEqual(max(alternative_counts) - min(alternative_counts), 1)
        self.assertEqual(len(design[36:]), 28)


class GeneticSearchTests(unittest.TestCase):
    def test_replacement_mutation_honors_configured_layer_cap(self):
        space = LayerwiseSearchSpace(12)
        base = space.from_genes((0,) * 12)
        for seed in range(100):
            mutated = _replacement_mutation(
                space,
                base,
                np.random.default_rng(seed),
                force=True,
                max_layers=4,
            )
            changed = _hamming_distance(space, base, mutated)
            self.assertGreaterEqual(changed, 1)
            self.assertLessEqual(changed, 4)

    def test_elites_keep_incumbent_and_prefer_hamming_distance_two(self):
        space = LayerwiseSearchSpace(3)
        incumbent = _evaluation(space.from_genes((5, 5, 5)))
        close = _evaluation(space.from_genes((5, 5, 4)))
        diverse = _evaluation(space.from_genes((4, 4, 5)))

        elites = _select_hamming_diverse_elites(
            space, [close, diverse, incumbent], 2,
        )

        self.assertEqual(elites[0].action_matrix, incumbent.action_matrix)
        self.assertEqual(elites[1].action_matrix, diverse.action_matrix)

    def test_ga_retains_seven_elites_and_has_exact_update_accounting(self):
        space = LayerwiseSearchSpace(3)
        calls = []

        def evaluator(action):
            calls.append(action)
            return _evaluation(action)

        result = run_search(
            "COINN-style-GA",
            space,
            evaluator,
            SearchConfig(
                evaluation_budget=64 + 2 * 57,
                seed=17,
                ga_population_size=64,
                ga_elite_count=7,
                ga_generations=2,
            ),
        )

        self.assertEqual(result.termination_reason, "generation_limit")
        self.assertEqual(result.evaluation_count, 64 + 2 * 57)
        self.assertEqual(len(calls), len(set(calls)))
        updates = [
            row for row in result.history
            if row["phase"] == "ga_update_generation"
        ]
        self.assertEqual(len(updates), 2)
        initial_elites = _select_hamming_diverse_elites(
            space, result.observations[:64], 7,
        )
        expected_first = [
            [list(layer) for layer in item.action_matrix]
            for item in initial_elites
        ]
        self.assertEqual(updates[0]["elite_actions"], expected_first)
        first_updated_population = [
            *initial_elites,
            *result.observations[64:64 + 57],
        ]
        second_elites = _select_hamming_diverse_elites(
            space, first_updated_population, 7,
        )
        expected_second = [
            [list(layer) for layer in item.action_matrix]
            for item in second_elites
        ]
        self.assertEqual(updates[1]["elite_actions"], expected_second)
        for generation, row in enumerate(updates, start=1):
            self.assertEqual(row["elite_count"], 7)
            self.assertEqual(len(row["elite_actions"]), 7)
            self.assertEqual(row["offspring_evaluated"], 57)
            self.assertEqual(row["expected_evaluations"], 64 + generation * 57)
            self.assertEqual(row["evaluations"], 64 + generation * 57)

    def test_ga_small_space_still_returns_best_observed_feasible_action(self):
        space = LayerwiseSearchSpace(1)
        calls = []

        def evaluator(action):
            calls.append(action)
            return _evaluation(
                action,
                loss=(1.02 if space.genes(action) == (5,) else 1.0),
            )

        result = run_search(
            "ga",
            space,
            evaluator,
            SearchConfig(
                evaluation_budget=space.cardinality,
                seed=17,
                ga_generations=1,
            ),
        )
        self.assertEqual(result.best.action_matrix, ((1, 1),))
        self.assertEqual(len(calls), 6)
        self.assertEqual(len(calls), len(set(calls)))


class InferenceAccountingTests(unittest.TestCase):
    def test_greedy_and_bo_do_not_charge_non_inference_invalid_observations(self):
        for backend in ("greedy", "bo_rf"):
            with self.subTest(backend=backend):
                space = LayerwiseSearchSpace(1)
                calls = []

                def evaluator(action):
                    calls.append(action)
                    gene = space.genes(action)[0]
                    if gene < 2:
                        return _evaluation(
                            action,
                            valid=False,
                            inference_performed=False,
                        )
                    return _evaluation(action)

                result = run_search(
                    backend,
                    space,
                    evaluator,
                    SearchConfig(
                        evaluation_budget=2,
                        seed=17,
                        initial_design_size=6,
                        candidate_pool_size=6,
                        patience_generations=10,
                        observation_attempt_limit=20,
                    ),
                    surrogate_factory=(
                        (lambda _seed: _TinyForest())
                        if backend == "bo_rf" else None
                    ),
                )
                self.assertEqual(result.evaluation_count, 2)
                self.assertEqual(result.observation_count, 4)
                self.assertEqual(len(result.observations), 4)
                self.assertEqual(len(calls), len(set(calls)))
                self.assertEqual(
                    sum(item.inference_performed for item in result.observations),
                    2,
                )

    def test_ga_refills_each_generation_with_inference_reaching_offspring(self):
        space = LayerwiseSearchSpace(3)
        calls = []

        def evaluator(action):
            calls.append(action)
            inference_performed = sum(space.genes(action)) % 4 != 0
            return _evaluation(
                action,
                valid=inference_performed,
                inference_performed=inference_performed,
            )

        result = run_search(
            "coinn_ga",
            space,
            evaluator,
            SearchConfig(
                evaluation_budget=8 + 2 * 6,
                seed=17,
                ga_population_size=8,
                ga_elite_count=2,
                ga_generations=2,
                observation_attempt_limit=200,
            ),
        )

        self.assertEqual(result.evaluation_count, 20)
        self.assertGreater(result.observation_count, result.evaluation_count)
        self.assertEqual(len(calls), len(set(calls)))
        updates = [
            row for row in result.history
            if row["phase"] == "ga_update_generation"
        ]
        self.assertEqual(len(updates), 2)
        for row in updates:
            self.assertEqual(row["offspring_evaluated"], 6)
            self.assertGreaterEqual(row["offspring_observations"], 6)
        self.assertEqual(updates[-1]["expected_evaluations"], 20)

    def test_observation_attempt_guard_bounds_all_non_inference_search(self):
        space = LayerwiseSearchSpace(3)
        calls = []

        def evaluator(action):
            calls.append(action)
            return _evaluation(
                action,
                valid=False,
                inference_performed=False,
            )

        result = run_search(
            "greedy",
            space,
            evaluator,
            SearchConfig(
                evaluation_budget=2,
                observation_attempt_limit=5,
            ),
        )
        self.assertEqual(result.evaluation_count, 0)
        self.assertEqual(result.observation_count, 5)
        self.assertEqual(result.termination_reason, "observation_attempt_guard")
        self.assertEqual(len(calls), 5)


class DiverseSecondParentTests(unittest.TestCase):
    def test_distance_two_pool_is_formed_before_rank_tournament(self):
        space = LayerwiseSearchSpace(3)
        first = _evaluation(space.from_genes((0, 0, 0)))
        distance_one = _evaluation(space.from_genes((5, 0, 0)))
        best_distance_two = _evaluation(space.from_genes((5, 5, 0)))
        other_distance_two = _evaluation(space.from_genes((4, 4, 0)))
        distance_three = _evaluation(space.from_genes((1, 1, 1)))
        population = [
            first,
            distance_one,
            best_distance_two,
            other_distance_two,
            distance_three,
        ]

        selected = _diverse_second_parent(
            space, population, first, np.random.default_rng(17),
        )

        self.assertGreaterEqual(
            _hamming_distance(
                space, first.action_matrix, selected.action_matrix,
            ),
            2,
        )
        self.assertEqual(selected.action_matrix, best_distance_two.action_matrix)

    def test_second_parent_relaxes_to_distance_one_only_when_needed(self):
        space = LayerwiseSearchSpace(3)
        first = _evaluation(space.from_genes((0, 0, 0)))
        candidates = [
            _evaluation(space.from_genes((gene, 0, 0)))
            for gene in (1, 4, 5)
        ]

        selected = _diverse_second_parent(
            space, [first, *candidates], first, np.random.default_rng(17),
        )

        self.assertEqual(
            _hamming_distance(
                space, first.action_matrix, selected.action_matrix,
            ),
            1,
        )
        self.assertEqual(
            selected.action_matrix,
            max(candidates, key=candidate_rank_key).action_matrix,
        )


class BackendNormalizationTests(unittest.TestCase):
    def test_backend_aliases_normalize(self):
        aliases = {
            "SMAC-RF": "bo_rf",
            "Bayesian Optimization": "bo_rf",
            "hill-climbing": "greedy",
            "COINN-style-GA": "coinn_ga",
            "genetic_algorithm": "coinn_ga",
        }
        for alias, expected in aliases.items():
            with self.subTest(alias=alias):
                self.assertEqual(normalize_search_backend(alias), expected)

    def test_unknown_backend_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            run_search(
                "random_guess",
                LayerwiseSearchSpace(1),
                lambda action: _evaluation(action),
                SearchConfig(evaluation_budget=1),
            )


if __name__ == "__main__":
    unittest.main()
