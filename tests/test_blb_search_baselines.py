from __future__ import annotations

import unittest

import numpy as np

from blb_stage2_rl.search_baselines import (
    ConstraintLimits,
    LayerwiseSearchSpace,
    SearchConfig,
    SearchEvaluation,
    SearchMetrics,
    candidate_rank_key,
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
    def fit(self, features, targets):
        super().fit(features, targets)
        self.estimators_ = [
            _MeanTree().fit(features, targets),
            _MeanTree().fit(features, targets),
        ]
        return self


class LayerwiseSearchSpaceTests(unittest.TestCase):
    def test_space_uses_one_fusion_and_one_hml_coordinate_per_layer(self):
        space = LayerwiseSearchSpace(num_layers=2)

        self.assertEqual(space.dimensions, (2, 3, 2, 3))
        self.assertEqual(space.cardinality, 36)
        self.assertEqual(space.safe_action, ((0, 0), (0, 0)))
        self.assertEqual(space.max_resource_action, ((1, 2), (1, 2)))

    def test_neighbors_are_legal_single_coordinate_mesh_moves(self):
        space = LayerwiseSearchSpace(num_layers=2)
        action = ((0, 1), (1, 2))

        neighbors = tuple(space.neighbors(action))

        self.assertEqual(len(neighbors), 5)
        for neighbor in neighbors:
            flat_before = space.flatten(action)
            flat_after = space.flatten(neighbor)
            changed = [
                index
                for index, (before, after) in enumerate(
                    zip(flat_before, flat_after)
                )
                if before != after
            ]
            self.assertEqual(len(changed), 1)
            index = changed[0]
            self.assertLess(
                flat_after[index],
                space.dimensions[index],
            )
            if index % 2 == 1:
                self.assertEqual(
                    abs(flat_after[index] - flat_before[index]),
                    1,
                )

    def test_malformed_actions_are_rejected(self):
        space = LayerwiseSearchSpace(num_layers=2)

        with self.assertRaisesRegex(ValueError, "2 layers"):
            space.validate(((0, 0),))
        with self.assertRaisesRegex(ValueError, "two coordinates"):
            space.validate(((0, 0, 0), (0, 0)))
        with self.assertRaisesRegex(ValueError, "outside"):
            space.validate(((0, 3), (0, 0)))


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
                self.assertFalse(
                    _evaluation(((0, 0),), **override).feasible
                )

    def test_feasible_candidate_outranks_higher_cost_infeasible_candidate(self):
        feasible = _evaluation(((1, 1),))
        infeasible = _evaluation(((1, 2),), loss=1.02)

        self.assertGreater(
            candidate_rank_key(feasible),
            candidate_rank_key(infeasible),
        )

    def test_resource_score_respects_communication_importance_ratio(self):
        compute_only = _evaluation(((1, 0),))
        communication_heavy = SearchEvaluation(
            action_matrix=((0, 2),),
            metrics=compute_only.metrics,
            limits=LIMITS,
            communication_importance_ratio=3.0,
        )
        compute_weighted = SearchEvaluation(
            action_matrix=((1, 0),),
            metrics=compute_only.metrics,
            limits=LIMITS,
            communication_importance_ratio=3.0,
        )

        self.assertGreater(
            communication_heavy.resource.ppo_resource_score,
            compute_weighted.resource.ppo_resource_score,
        )

    def test_bootstrap_probabilities_override_point_only_feasibility(self):
        point_feasible = _evaluation(((0, 0),))
        statistically_infeasible = SearchEvaluation(
            action_matrix=point_feasible.action_matrix,
            metrics=point_feasible.metrics,
            limits=point_feasible.limits,
            constraint_probabilities=(0.9, 0.9, 0.9, 0.9, 0.49, 0.9),
            gate_probability=0.5,
        )

        self.assertTrue(point_feasible.feasible)
        self.assertFalse(statistically_infeasible.feasible)
        self.assertGreater(
            statistically_infeasible.normalized_violation, 0.0,
        )

    def test_lexicographic_tie_break_is_deterministic(self):
        smaller = _evaluation(((0, 1), (1, 0)))
        larger = _evaluation(((1, 0), (0, 1)))

        self.assertGreater(
            candidate_rank_key(smaller),
            candidate_rank_key(larger),
        )


class SearchAlgorithmTests(unittest.TestCase):
    def _run(self, backend):
        space = LayerwiseSearchSpace(num_layers=1)
        calls = []

        def evaluator(action):
            calls.append(action)
            fusion, precision = action[0]
            return _evaluation(
                action,
                loss=(1.02 if (fusion, precision) == (1, 2) else 1.0),
            )

        config = SearchConfig(
            evaluation_budget=space.cardinality,
            seed=17,
            initial_design_size=2,
            candidate_pool_size=32,
            population_size=space.cardinality,
            patience_generations=5,
        )
        result = run_search(
            backend,
            space,
            evaluator,
            config,
            surrogate_factory=(
                (lambda _seed: _TinyForest())
                if backend == "bo_rf"
                else None
            ),
        )
        return result, calls

    def test_greedy_finds_best_observed_feasible_configuration(self):
        result, calls = self._run("greedy")

        self.assertEqual(result.best.action_matrix, ((1, 1),))
        self.assertEqual(len(calls), len(set(calls)))
        self.assertLessEqual(len(calls), 6)

    def test_bo_rf_finds_best_observed_feasible_configuration(self):
        result, calls = self._run("bo_rf")

        self.assertEqual(result.best.action_matrix, ((1, 1),))
        self.assertEqual(len(calls), len(set(calls)))
        self.assertEqual(len(calls), 6)

    def test_coinn_ga_finds_best_observed_feasible_configuration(self):
        result, calls = self._run("coinn_ga")

        self.assertEqual(result.best.action_matrix, ((1, 1),))
        self.assertEqual(len(calls), len(set(calls)))
        self.assertLessEqual(len(calls), 6)

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
