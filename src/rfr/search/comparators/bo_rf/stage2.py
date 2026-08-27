"""BO-RF operators for the Stage-2 layerwise search."""

from __future__ import annotations

import heapq
from typing import Any, Iterable, Optional

import numpy as np

from rfr.search.comparators.common.stage2_core import (
    ActionMatrix,
    CheckpointCallback,
    EvaluationFn,
    LayerwiseSearchSpace,
    SearchConfig,
    SearchEvaluation,
    SearchResult,
    SurrogateFactory,
    _EvaluationCache,
    _best_history_row,
    _resource_score,
    _structured_initial_design,
    candidate_rank_key,
)

def _default_surrogate_factory(config: SearchConfig) -> SurrogateFactory:
    def factory(seed: int) -> Any:
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError as exc:  # pragma: no cover - exercised on server
            raise RuntimeError(
                "BO-RF requires scikit-learn; install it in the runtime "
                "environment or provide surrogate_factory"
            ) from exc
        return RandomForestRegressor(
            n_estimators=int(config.rf_n_estimators),
            min_samples_leaf=int(config.rf_min_samples_leaf),
            max_features=0.75,
            bootstrap=True,
            random_state=int(seed),
            n_jobs=-1,
        )
    return factory

def _candidate_pool(
        space: LayerwiseSearchSpace,
        cache: _EvaluationCache,
        rng: np.random.Generator,
        *,
        pool_size: int,
        ) -> list[ActionMatrix]:
    unseen_count = int(space.cardinality) - len(cache.observations)
    target = min(int(pool_size), unseen_count)
    if target <= 0:
        return []
    if unseen_count <= target and space.cardinality <= 100_000:
        pool = [
            action for action in space.all_actions()
            if not cache.contains(action)
        ]
        order = np.asarray(rng.permutation(len(pool)), dtype=int).reshape(-1)
        return [pool[int(index)] for index in order]

    pool: list[ActionMatrix] = []
    seen: set[ActionMatrix] = set()

    def add(action: ActionMatrix) -> None:
        owned = space.validate(action)
        if owned not in seen and not cache.contains(owned) and len(pool) < target:
            pool.append(owned)
            seen.add(owned)


    global_target = max(1, target // 2)
    attempts = 0
    while len(pool) < global_target and attempts < max(1000, 40 * target):
        attempts += 1
        add(space.random_action(rng))


    ranked = heapq.nlargest(
        min(16, len(cache.observations)),
        cache.observations,
        key=candidate_rank_key,
    )
    local_candidates: list[ActionMatrix] = []
    for observation in ranked:
        local_candidates.extend(space.neighbors(observation.action_matrix))
    if local_candidates:
        order = np.asarray(
            rng.permutation(len(local_candidates)), dtype=int,
        ).reshape(-1)
        local_limit = min(target, global_target + max(1, target // 4))
        for index in order:
            add(local_candidates[int(index)])
            if len(pool) >= local_limit:
                break


    attempts = 0
    while len(pool) < target and attempts < max(2000, 80 * target):
        attempts += 1
        add(space.random_action(rng))
    if len(pool) < target and space.cardinality <= 100_000:
        remaining = [
            action for action in space.all_actions()
            if action not in seen and not cache.contains(action)
        ]
        order = np.asarray(rng.permutation(len(remaining)), dtype=int).reshape(-1)
        for index in order:
            add(remaining[int(index)])
            if len(pool) >= target:
                break
    return pool

def _tree_predictions(model: Any, features: np.ndarray) -> np.ndarray:
    estimators = tuple(getattr(model, "estimators_", ()) or ())
    if estimators:
        predictions = [
            np.asarray(estimator.predict(features), dtype=float)
            for estimator in estimators
        ]
        return np.stack(predictions, axis=0)
    prediction = np.asarray(model.predict(features), dtype=float)
    return prediction.reshape(1, prediction.shape[0], -1)

def _bo_acquisition_key(
        *,
        has_feasible_incumbent: bool,
        probability_of_feasibility: float,
        expected_improvement: float,
        expected_failed_constraints: float,
        expected_total_violation: float,
        expected_worst_violation: float,
        exploration_tiebreak: float,
        objective_tiebreak: float,
        deterministic_tiebreak: tuple[int, ...],
        ) -> tuple[Any, ...]:
    if has_feasible_incumbent:
        return (
            float(probability_of_feasibility) * float(expected_improvement),
            float(probability_of_feasibility),
            float(exploration_tiebreak),
            float(objective_tiebreak),
            deterministic_tiebreak,
        )
    return (
        -float(expected_failed_constraints),
        -float(expected_total_violation),
        -float(expected_worst_violation),
        float(probability_of_feasibility),
        float(exploration_tiebreak),
        float(objective_tiebreak),
        deterministic_tiebreak,
    )

def _run_bo_rf(
        space: LayerwiseSearchSpace,
        evaluator: EvaluationFn,
        config: SearchConfig,
        surrogate_factory: Optional[SurrogateFactory],
        *,
        preload: Iterable[SearchEvaluation] = (),
        checkpoint_callback: Optional[CheckpointCallback] = None,
        ) -> SearchResult:
    rng = np.random.default_rng(int(config.seed))
    cache = _EvaluationCache(
        space,
        evaluator,
        preload=preload,
        checkpoint_callback=checkpoint_callback,
    )
    history: list[dict[str, Any]] = []
    design = _structured_initial_design(
        space,
        rng,
        min(config.initial_design_size, space.cardinality),
    )
    for action in design:
        if not cache.can_observe:
            break
        cache.evaluate(action)
    history.append(_best_history_row(
        cache, phase="structured_initial_design", iteration=0,
        design_size=int(cache.observation_count),
        design_inferences=int(cache.evaluation_count),
    ))

    factory = surrogate_factory or _default_surrogate_factory(config)
    iteration = 0
    no_improvement = 0
    termination = "consecutive_no_improvement"
    while cache.can_observe:
        iteration += 1
        observations = cache.observations
        features = np.asarray(
            [space.one_hot(item.action_matrix) for item in observations],
            dtype=float,
        )
        targets = np.asarray(
            [item.constraint_margins for item in observations],
            dtype=float,
        )
        model = factory(int(config.seed) + iteration)
        model.fit(features, targets)
        pool = _candidate_pool(
            space,
            cache,
            rng,
            pool_size=config.candidate_pool_size,
        )
        if not pool:
            raise RuntimeError(
                "Stage-2 BO-RF exhausted the candidate space before reaching "
                "its consecutive no-improvement limit"
            )
        pool_features = np.asarray(
            [space.one_hot(action) for action in pool],
            dtype=float,
        )
        tree_predictions = _tree_predictions(model, pool_features)
        if tree_predictions.ndim != 3 or tree_predictions.shape[2] != 6:
            raise RuntimeError(
                "BO-RF surrogate must predict six constraint margins"
            )
        feasible_probability = np.mean(
            np.all(tree_predictions >= 0.0, axis=2), axis=0,
        )
        predicted_std = tree_predictions.std(axis=0)
        predicted_violations = np.maximum(0.0, -tree_predictions)
        predicted_failed = np.mean(
            np.sum(tree_predictions < 0.0, axis=2), axis=0,
        )
        predicted_total_violation = np.mean(
            np.sum(predicted_violations, axis=2), axis=0,
        )
        predicted_worst_violation = np.mean(
            np.max(predicted_violations, axis=2), axis=0,
        )
        feasible_observations = [item for item in observations if item.feasible]
        incumbent_resource = max(
            (_resource_score(
                item.action_matrix, config.communication_importance_ratio,
            ) for item in feasible_observations),
            default=0.0,
        )
        acquisition: list[float] = []
        acquisition_keys: list[tuple[Any, ...]] = []
        for index, action in enumerate(pool):
            resource = _resource_score(
                action, config.communication_importance_ratio,
            )
            probability = float(feasible_probability[index])
            uncertainty = float(np.mean(predicted_std[index]))
            deterministic = tuple(-value for value in space.genes(action))
            improvement = max(0.0, resource - incumbent_resource)
            value = (
                probability * improvement
                if feasible_observations
                else -float(predicted_failed[index])
            )
            key = _bo_acquisition_key(
                has_feasible_incumbent=bool(feasible_observations),
                probability_of_feasibility=probability,
                expected_improvement=improvement,
                expected_failed_constraints=float(predicted_failed[index]),
                expected_total_violation=float(
                    predicted_total_violation[index]
                ),
                expected_worst_violation=float(
                    predicted_worst_violation[index]
                ),
                exploration_tiebreak=(
                    float(config.acquisition_exploration) * uncertainty
                ),
                objective_tiebreak=resource,
                deterministic_tiebreak=deterministic,
            )
            acquisition.append(float(value))
            acquisition_keys.append(key)
        selected_index = max(
            range(len(pool)), key=acquisition_keys.__getitem__,
        )
        previous_best = cache.best()
        selected = pool[selected_index]
        observed = cache.evaluate(selected)
        improved = (
            candidate_rank_key(cache.best()) > candidate_rank_key(previous_best)
        )
        no_improvement = 0 if improved else no_improvement + 1
        history.append(_best_history_row(
            cache,
            phase="feasibility_aware_acquisition",
            iteration=iteration,
            acquisition=float(acquisition[selected_index]),
            acquisition_mode=(
                "probability_of_feasibility_times_expected_improvement"
                if feasible_observations
                else "lexicographic_predicted_violation"
            ),
            predicted_feasibility=float(feasible_probability[selected_index]),
            predicted_failed_constraints=float(predicted_failed[selected_index]),
            predicted_total_violation=float(
                predicted_total_violation[selected_index]
            ),
            predicted_worst_violation=float(
                predicted_worst_violation[selected_index]
            ),
            improved=bool(improved),
            inference_performed=bool(observed.inference_performed),
            no_improvement_evaluations=int(no_improvement),
        ))
        if no_improvement >= int(config.bo_no_improvement_patience):
            break
    if no_improvement < int(config.bo_no_improvement_patience):
        raise RuntimeError(
            "Stage-2 BO-RF exhausted the candidate space before reaching its "
            "consecutive no-improvement limit"
        )
    cache.assert_replay_consumed()
    return SearchResult(
        algorithm="bo_rf",
        best=cache.best(),
        observations=cache.observations,
        history=tuple(history),
        termination_reason=termination,
    )


run = _run_bo_rf
