"""BO-RF operators for the Stage-1 categorical search."""

from __future__ import annotations

import heapq
import math
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from rfr.search.comparators.common.stage1_core import (
    CheckpointCallback,
    EvaluationFn,
    IncrementalCheckpointCallback,
    SearchConfig,
    SearchEvaluation,
    SearchResult,
    Stage1Action,
    Stage1SearchSpace,
    SurrogateFactory,
    _EvaluationCache,
    _candidate_stream,
    _history_row,
    candidate_rank_key,
    structured_maximin_initial_design,
)

def _default_surrogate_factory(config: SearchConfig) -> SurrogateFactory:
    def factory(seed: int) -> Any:
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError as exc:  # pragma: no cover - depends on runtime image
            raise RuntimeError(
                "Stage-1 BO-RF requires scikit-learn or an injected "
                "surrogate_factory"
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

def _tree_predictions(model: Any, features: np.ndarray) -> np.ndarray:
    estimators = tuple(getattr(model, "estimators_", ()) or ())
    if estimators:
        predictions = [
            np.asarray(estimator.predict(features), dtype=float)
            for estimator in estimators
        ]
        stacked = np.stack(predictions, axis=0)
    else:
        prediction = np.asarray(model.predict(features), dtype=float)
        if prediction.ndim == 1:
            prediction = prediction.reshape(-1, 1)
        stacked = prediction.reshape(1, prediction.shape[0], prediction.shape[1])
    if stacked.ndim == 2:
        stacked = stacked[:, :, np.newaxis]
    return stacked

def _bo_candidate_pool(
        *,
        space: Stage1SearchSpace,
        cache: _EvaluationCache,
        rng: np.random.Generator,
        pool_size: int,
        ) -> list[Stage1Action]:
    remaining_space = int(space.cardinality) - cache.observation_count
    target = min(int(pool_size), remaining_space)
    if target <= 0:
        return []
    if space.cardinality <= max(10_000, 4 * target):
        return [
            action
            for action in space.all_actions(max_cardinality=max(10_000, space.cardinality))
            if not cache.contains(action)
        ]

    pool: list[Stage1Action] = []
    seen: set[Stage1Action] = set(item.action for item in cache.observations)

    def add(action: Sequence[int]) -> None:
        owned = space.validate(action)
        if owned not in seen and len(pool) < target:
            pool.append(owned)
            seen.add(owned)

    ranked = heapq.nlargest(
        min(16, cache.observation_count),
        cache.observations,
        key=candidate_rank_key,
    )
    for observation in ranked:
        for neighbor in space.one_opt_neighbors(observation.action):
            add(neighbor)
        for neighbor in space.two_opt_neighbors(observation.action):
            add(neighbor)
            if len(pool) >= target // 2:
                break
        if len(pool) >= target // 2:
            break
    for action in _candidate_stream(
            space,
            rng,
            target=target - len(pool),
            excluded=seen,
    ):
        add(action)
    return pool

def _cost_hint(evaluator: EvaluationFn, action: Stage1Action) -> float:
    """Return exact adapter cost when available, otherwise a monotonic proxy."""

    cost_for_action = getattr(evaluator, "cost_for_action", None)
    if callable(cost_for_action):
        return float(cost_for_action(action))


    return -float(sum(action))

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
        *,
        space: Stage1SearchSpace,
        evaluator: EvaluationFn,
        config: SearchConfig,
        surrogate_factory: Optional[SurrogateFactory],
        preload: Iterable[SearchEvaluation],
        checkpoint_callback: Optional[CheckpointCallback],
        incremental_checkpoint_callback: Optional[
            IncrementalCheckpointCallback
        ],
        ) -> SearchResult:
    rng = np.random.default_rng(int(config.seed))
    cache = _EvaluationCache(
        space=space,
        evaluator=evaluator,
        evaluation_cap=config.evaluation_cap,
        preload=preload,
        checkpoint_callback=checkpoint_callback,
        incremental_checkpoint_callback=incremental_checkpoint_callback,
        replay_preload_in_order=True,
    )
    history: list[dict[str, Any]] = []
    design = structured_maximin_initial_design(
        space,
        count=min(int(config.bo_initial_design_size), cache.cap),
        seed=int(config.seed),
        maximin_candidate_pool_size=int(config.maximin_candidate_pool_size),
    )
    for action in design:
        if cache.get(action) is None and cache.remaining <= 0:
            break
        cache.evaluate(action)
    history.append(_history_row(cache, phase="structured_maximin_initial_design", iteration=0))

    factory = surrogate_factory or _default_surrogate_factory(config)
    incumbent_key = candidate_rank_key(cache.best())
    no_improvement = 0
    iteration = 0
    termination = "candidate_space_exhausted"
    while cache.remaining > 0:
        if no_improvement >= int(config.bo_no_improvement_patience):
            termination = "no_improvement_convergence"
            break
        iteration += 1
        observations = cache.observations
        features = space.one_hot([item.action for item in observations])
        targets = np.asarray(
            [item.constraint_margins for item in observations],
            dtype=float,
        )
        model = factory(int(config.seed) + iteration)
        model.fit(features, targets)
        pool = _bo_candidate_pool(
            space=space,
            cache=cache,
            rng=rng,
            pool_size=int(config.bo_candidate_pool_size),
        )
        if not pool:
            termination = "candidate_space_exhausted"
            break
        tree_predictions = _tree_predictions(model, space.one_hot(pool))
        if tree_predictions.ndim != 3 or tree_predictions.shape[2] != targets.shape[1]:
            raise RuntimeError(
                "Stage-1 BO-RF surrogate must predict one margin for loss and "
                "each active metric"
            )
        feasibility_probability = np.mean(
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
        incumbent_cost = min(
            (_cost_hint(evaluator, item.action) for item in feasible_observations),
            default=math.inf,
        )
        candidate_costs = [_cost_hint(evaluator, action) for action in pool]
        acquisition: list[float] = []
        acquisition_keys: list[tuple[Any, ...]] = []
        for index, action in enumerate(pool):
            probability = float(feasibility_probability[index])
            uncertainty = float(np.mean(predicted_std[index]))
            candidate_cost = float(candidate_costs[index])
            deterministic = tuple(-value for value in action)
            improvement = max(0.0, incumbent_cost - candidate_cost)
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
                objective_tiebreak=-candidate_cost,
                deterministic_tiebreak=deterministic,
            )
            acquisition.append(float(value))
            acquisition_keys.append(key)
        selected_index = max(
            range(len(pool)), key=acquisition_keys.__getitem__,
        )
        selected = cache.evaluate(pool[selected_index])
        new_key = candidate_rank_key(cache.best())
        improved = new_key > incumbent_key
        if improved:
            incumbent_key = new_key
            no_improvement = 0
        else:
            no_improvement += 1
        history.append(_history_row(
            cache,
            phase="feasibility_aware_acquisition",
            iteration=iteration,
            selected_action=list(selected.action),
            acquisition=float(acquisition[selected_index]),
            acquisition_mode=(
                "probability_of_feasibility_times_expected_improvement"
                if feasible_observations
                else "lexicographic_predicted_violation"
            ),
            predicted_feasibility=float(feasibility_probability[selected_index]),
            predicted_failed_constraints=float(predicted_failed[selected_index]),
            predicted_total_violation=float(
                predicted_total_violation[selected_index]
            ),
            predicted_worst_violation=float(
                predicted_worst_violation[selected_index]
            ),
            no_improvement=int(no_improvement),
            improved=bool(improved),
        ))
    if cache.remaining <= 0:
        termination = "evaluation_cap"
    cache.assert_replay_consumed()
    return SearchResult(
        algorithm="bo_rf",
        config=config,
        best=cache.best(),
        observations=cache.observations,
        history=tuple(history),
        termination_reason=termination,
    )


run = _run_bo_rf
