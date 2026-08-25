"""COINN-GA operators for the Stage-1 categorical search."""

from __future__ import annotations

import itertools
import math
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from rfr.search.comparators.common.stage1_core import (
    CheckpointCallback,
    EvaluationFn,
    GENE_CATEGORIES,
    IncrementalCheckpointCallback,
    SearchConfig,
    SearchEvaluation,
    SearchResult,
    Stage1Action,
    Stage1SearchSpace,
    _EPS,
    _EvaluationCache,
    _history_row,
    _maximin_fill,
    candidate_rank_key,
    structured_maximin_initial_design,
)

def _population_diversity(
        space: Stage1SearchSpace,
        population: Sequence[SearchEvaluation],
        ) -> tuple[float, float]:
    if not population:
        return 0.0, 0.0
    unique_ratio = len({item.action for item in population}) / float(len(population))
    distances = [
        space.hamming_distance(population[first].action, population[second].action)
        for first in range(len(population))
        for second in range(first + 1, len(population))
    ]
    mean_distance = float(np.mean(distances)) if distances else 0.0
    return float(unique_ratio), mean_distance

def _select_hamming_diverse_elites(
        space: Stage1SearchSpace,
        population: Sequence[SearchEvaluation],
        elite_count: int,
        ) -> list[SearchEvaluation]:
    """Keep the best incumbent, then prefer feasible elites at distance >= 2."""

    target = min(int(elite_count), len(population))
    if target <= 0:
        return []
    ranked = sorted(population, key=candidate_rank_key, reverse=True)
    feasible = [item for item in ranked if item.feasible]
    pools = (feasible, ranked)
    selected: list[SearchEvaluation] = []
    selected_actions: set[Stage1Action] = set()
    for pool in pools:
        remaining = [
            item for item in pool if item.action not in selected_actions
        ]
        while remaining and len(selected) < target:
            if not selected:
                chosen = remaining[0]
            else:
                distance_two = [
                    item for item in remaining
                    if all(
                        space.hamming_distance(item.action, owned.action) >= 2
                        for owned in selected
                    )
                ]
                distance_one = [
                    item for item in remaining
                    if all(
                        space.hamming_distance(item.action, owned.action) >= 1
                        for owned in selected
                    )
                ]
                chosen = (distance_two or distance_one or remaining)[0]
            selected.append(chosen)
            selected_actions.add(chosen.action)
            remaining = [
                item for item in remaining if item.action != chosen.action
            ]
        if len(selected) >= target:
            break
    return selected

def _ga_parent_weights(
        population: Sequence[SearchEvaluation],
        ) -> tuple[float, ...]:
    """Return feasibility-aware positive COINN parent fitness weights."""

    if not population:
        raise ValueError("GA parent population must not be empty")

    def inverse_penalty(item: SearchEvaluation) -> float:
        components = (
            1.0,
            float(not item.valid),
            float(item.failed_constraint_count),
            *item.violations,
            float(item.worst_violation),
        )
        scale = max(components)
        if not math.isfinite(scale):
            return math.nextafter(0.0, 1.0)
        scaled_total = sum(value / scale for value in components)
        return max(
            (1.0 / scale) / scaled_total,
            math.nextafter(0.0, 1.0),
        )

    feasible_weights = tuple(
        1.0 / max(float(item.cost), _EPS) if item.feasible else 0.0
        for item in population
    )
    infeasible_weights = tuple(
        0.0 if item.feasible else inverse_penalty(item)
        for item in population
    )
    feasible_total = float(sum(feasible_weights))
    infeasible_total = float(sum(infeasible_weights))
    if feasible_total > 0.0 and infeasible_total > 0.0:
        return tuple(
            0.90 * (feasible / feasible_total)
            + 0.10 * (infeasible / infeasible_total)
            for feasible, infeasible in zip(  # noqa: B905 - Python 3.9
                feasible_weights, infeasible_weights,
            )
        )
    if feasible_total > 0.0:
        return feasible_weights
    return infeasible_weights

def _tournament(
        population: Sequence[SearchEvaluation],
        rng: np.random.Generator,
        tournament_size: int,
        *,
        diverse_from: Optional[Stage1Action] = None,
        ) -> SearchEvaluation:
    """Compatibility-named fitness-proportional COINN parent selector."""

    del tournament_size, diverse_from
    weights = np.asarray(_ga_parent_weights(population), dtype=float)
    probabilities = weights / float(np.sum(weights))
    index = int(rng.choice(len(population), p=probabilities))
    return population[index]

def _mutate_action(
        space: Stage1SearchSpace,
        action: Stage1Action,
        rng: np.random.Generator,
        *,
        max_layers: int,
        force: bool,
        ) -> Stage1Action:
    candidate = list(space.validate(action))
    probability = 1.0 / float(space.num_layers)
    selected = [
        layer_idx
        for layer_idx in range(space.num_layers)
        if float(rng.random()) < probability
    ]
    if len(selected) > int(max_layers):
        selected = sorted(
            int(value)
            for value in rng.choice(selected, size=int(max_layers), replace=False)
        )
    if force and not selected:
        selected = [int(rng.integers(space.num_layers))]
    for layer_idx in selected:
        current = candidate[layer_idx]
        alternatives = tuple(
            value for value in GENE_CATEGORIES
            if value != current and abs(value - current) == 1
        )
        candidate[layer_idx] = alternatives[
            int(rng.integers(len(alternatives)))
        ]
    return tuple(candidate)

def _crossover(
        space: Stage1SearchSpace,
        first: Stage1Action,
        second: Stage1Action,
        rng: np.random.Generator,
        ) -> Stage1Action:
    lhs = space.validate(first)
    rhs = space.validate(second)
    if space.num_layers == 1:
        return lhs if float(rng.random()) < 0.5 else rhs
    if float(rng.random()) < 0.5:

        boundaries = sorted(
            int(value)
            for value in rng.choice(space.num_layers + 1, size=2, replace=False)
        )
        start, stop = boundaries
        child = list(lhs)
        child[start:stop] = rhs[start:stop]
        return tuple(child)
    mask = rng.random(space.num_layers) < 0.5
    return tuple(rhs[index] if mask[index] else lhs[index] for index in range(space.num_layers))

def _maximin_immigrant(
        *,
        space: Stage1SearchSpace,
        cache: _EvaluationCache,
        references: Sequence[Stage1Action],
        blocked: set[Stage1Action],
        rng: np.random.Generator,
        pool_size: int,
        ) -> Optional[Stage1Action]:
    excluded = set(blocked)
    excluded.update(item.action for item in cache.observations)
    immigrants = _maximin_fill(
        space,
        references,
        count=1,
        rng=rng,
        excluded=excluded,
        pool_size=int(pool_size),
    )
    return None if not immigrants else immigrants[0]

def _breed_unique_child(
        *,
        space: Stage1SearchSpace,
        population: Sequence[SearchEvaluation],
        cache: _EvaluationCache,
        blocked: set[Stage1Action],
        rng: np.random.Generator,
        config: SearchConfig,
        ) -> tuple[Optional[Stage1Action], bool]:
    for _attempt in range(int(config.ga_duplicate_attempts)):
        parent = _tournament(
            population, rng, int(config.ga_tournament_size),
        )
        child = _mutate_action(
            space,
            parent.action,
            rng,
            max_layers=int(config.ga_mutation_max_layers),
            force=True,
        )
        if child in blocked or cache.contains(child):
            child = _mutate_action(
                space,
                parent.action,
                rng,
                max_layers=int(config.ga_mutation_max_layers),
                force=True,
            )
        if child not in blocked and not cache.contains(child):
            return child, False


    max_changed_layers = min(
        int(config.ga_mutation_max_layers),
        int(space.num_layers),
    )
    for changed_layer_count in range(1, max_changed_layers + 1):
        for parent in population:
            parent_action = space.validate(parent.action)
            for layer_indices in itertools.combinations(
                    range(space.num_layers), changed_layer_count,
            ):
                adjacent_values = [
                    tuple(
                        value for value in GENE_CATEGORIES
                        if abs(value - parent_action[layer_idx]) == 1
                    )
                    for layer_idx in layer_indices
                ]
                for replacements in itertools.product(*adjacent_values):
                    candidate = list(parent_action)
                    for layer_idx, replacement in zip(  # noqa: B905 - Python 3.9
                            layer_indices, replacements,
                    ):
                        candidate[layer_idx] = replacement
                    child = tuple(candidate)
                    if child not in blocked and not cache.contains(child):
                        return child, False
    return None, False

def _run_coinn_ga(
        *,
        space: Stage1SearchSpace,
        evaluator: EvaluationFn,
        config: SearchConfig,
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
    population_size = min(
        int(config.ga_population_size),
        int(space.cardinality),
        int(cache.cap),
    )
    elite_count = min(int(config.ga_elite_count), max(0, population_size - 1))
    initial_actions = structured_maximin_initial_design(
        space,
        count=population_size,
        seed=int(config.seed),
        maximin_candidate_pool_size=int(config.maximin_candidate_pool_size),
    )
    population: list[SearchEvaluation] = []
    for action in initial_actions:
        if cache.get(action) is None and cache.remaining <= 0:
            break
        population.append(cache.evaluate(action))
    if not population:
        raise RuntimeError("Stage-1 GA could not evaluate its initial population")
    history: list[dict[str, Any]] = []
    unique_ratio, mean_distance = _population_diversity(space, population)
    history.append(_history_row(
        cache,
        phase="initial_population",
        iteration=0,
        generation=0,
        population_size=len(population),
        anchors=min(3, len(initial_actions)),
        one_layer_reductions=min(2 * space.num_layers, max(0, len(initial_actions) - 3)),
        maximin_count=max(0, len(initial_actions) - 3 - 2 * space.num_layers),
        unique_ratio=float(unique_ratio),
        mean_pairwise_distance=float(mean_distance),
    ))

    termination = "completed_generations"
    completed_generations = 0
    incumbent_key = candidate_rank_key(max(
        population, key=candidate_rank_key,
    ))
    no_improvement_generations = 0
    offspring_target = population_size - elite_count
    if config.ga_require_full_generations:
        required_evaluations = (
            population_size
            + int(config.ga_update_generations) * offspring_target
        )
        if int(cache.cap) < required_evaluations:
            raise RuntimeError(
                "Stage-1 GA full-generation contract has insufficient "
                "evaluation budget"
            )
    for generation in range(1, int(config.ga_update_generations) + 1):
        if cache.remaining < offspring_target:
            if config.ga_require_full_generations:
                raise RuntimeError(
                    "Stage-1 GA full-generation contract reached an "
                    "insufficient evaluation budget"
                )
            termination = "evaluation_cap"
            break
        elites = _select_hamming_diverse_elites(
            space, population, elite_count,
        )
        actions: list[Stage1Action] = []
        blocked: set[Stage1Action] = set()
        while len(actions) < offspring_target:
            child, _used_immigrant = _breed_unique_child(
                space=space,
                population=population,
                cache=cache,
                blocked=blocked,
                rng=rng,
                config=config,
            )
            if child is None:
                break
            actions.append(child)
            blocked.add(child)
        if len(actions) < offspring_target:
            if config.ga_require_full_generations:
                raise RuntimeError(
                    "Stage-1 GA full-generation contract could not produce "
                    "a full set of unique offspring"
                )
            termination = "candidate_space_exhausted"
            break
        offspring = [cache.evaluate(action) for action in actions]
        population = elites + offspring
        completed_generations = generation
        next_incumbent_key = candidate_rank_key(max(
            population, key=candidate_rank_key,
        ))
        improved = next_incumbent_key > incumbent_key
        no_improvement_generations = (
            0 if improved else no_improvement_generations + 1
        )
        incumbent_key = next_incumbent_key
        next_unique_ratio, next_mean_distance = _population_diversity(space, population)
        history.append(_history_row(
            cache,
            phase="elitist_update",
            iteration=generation,
            generation=generation,
            elite_count=len(elites),
            feasible_elite_count=sum(item.feasible for item in elites),
            elite_actions=[list(item.action) for item in elites],
            elite_policy="best_incumbent_then_hamming_distance_2",
            new_unique_evaluations=len(offspring),
            improved=bool(improved),
            no_improvement_generations=int(no_improvement_generations),
            diversity_triggered=False,
            scheduled_immigrants=0,
            fallback_immigrants=0,
            replaced_worst_nonelite_actions=[],
            immigrant_actions=[],
            unique_ratio=float(next_unique_ratio),
            mean_pairwise_distance=float(next_mean_distance),
            post_update_unique_ratio=float(next_unique_ratio),
            post_update_mean_pairwise_distance=float(next_mean_distance),
        ))
        if (
            config.ga_stop_on_no_improvement
            and no_improvement_generations >= int(
                config.ga_no_improvement_patience
            )
        ):
            termination = "ga_no_incumbent_improvement"
            break
    if (
            termination == "completed_generations"
            and completed_generations < int(config.ga_update_generations)
            and cache.remaining <= 0
    ):
        termination = "evaluation_cap"
    if config.ga_require_full_generations and (
            termination != "completed_generations"
            or completed_generations != int(config.ga_update_generations)
    ):
        raise RuntimeError(
            "Stage-1 GA full-generation contract did not complete every "
            "configured generation"
        )
    cache.assert_replay_consumed()
    return SearchResult(
        algorithm="coinn_ga",
        config=config,
        best=cache.best(),
        observations=cache.observations,
        history=tuple(history),
        termination_reason=termination,
    )


run = _run_coinn_ga
