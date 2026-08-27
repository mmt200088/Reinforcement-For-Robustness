"""COINN-GA operators for the Stage-2 layerwise search."""

from __future__ import annotations

import itertools
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from rfr.search.comparators.common.stage2_core import (
    ActionMatrix,
    CheckpointCallback,
    EvaluationFn,
    LAYER_GENE_CARDINALITY,
    LayerwiseSearchSpace,
    SearchConfig,
    SearchEvaluation,
    SearchResult,
    _EvaluationCache,
    _best_history_row,
    _hamming_distance,
    _maximin_candidate,
    _structured_initial_design,
    candidate_rank_key,
    decode_layer_gene,
)


_GA_OFFSPRING_ATTEMPTS = 64


def _ga_parent_weights(
        population: Sequence[SearchEvaluation],
        ) -> tuple[float, ...]:
    """Return feasibility-aware positive COINN parent fitness weights."""

    if not population:
        raise ValueError("GA parent population must not be empty")
    feasible_scores = [
        float(item.resource.ppo_resource_score)
        for item in population
        if item.feasible
    ]
    if feasible_scores:
        minimum = min(feasible_scores)
        return tuple(
            float(item.resource.ppo_resource_score) - minimum + 1.0
            if item.feasible else 0.0
            for item in population
        )
    return tuple(
        1.0 / (
            1.0
            + float(not item.valid)
            + float(item.failed_constraint_count)
            + float(item.normalized_violation)
            + float(item.worst_normalized_violation)
        )
        for item in population
    )

def _tournament_parent(
        population: Sequence[SearchEvaluation],
        rng: np.random.Generator,
        ) -> SearchEvaluation:
    """Select a COINN parent with feasibility-aware proportional fitness."""

    weights = np.asarray(_ga_parent_weights(population), dtype=float)
    probabilities = weights / float(np.sum(weights))
    index = int(rng.choice(len(population), p=probabilities))
    return population[index]

def _diverse_second_parent(
        space: LayerwiseSearchSpace,
        population: Sequence[SearchEvaluation],
        first: SearchEvaluation,
        rng: np.random.Generator,
        ) -> SearchEvaluation:
    distances = [
        (
            item,
            _hamming_distance(
                space, first.action_matrix, item.action_matrix,
            ),
        )
        for item in population
        if item.action_matrix != first.action_matrix
    ]
    eligible = [item for item, distance in distances if distance >= 2]
    if not eligible:
        eligible = [item for item, distance in distances if distance >= 1]
    if not eligible:
        return first
    size = min(3, len(eligible))
    indices = np.asarray(
        rng.choice(len(eligible), size=size, replace=False), dtype=int,
    ).reshape(-1)
    sampled = [eligible[int(index)] for index in indices]
    return max(sampled, key=candidate_rank_key)

def _mesh_adjacent_gene_values(gene: int) -> tuple[int, ...]:
    current = decode_layer_gene(gene)
    return tuple(
        candidate
        for candidate in range(LAYER_GENE_CARDINALITY)
        if candidate != gene
        and max(
            abs(current[0] - decode_layer_gene(candidate)[0]),
            abs(current[1] - decode_layer_gene(candidate)[1]),
        ) <= 1
    )

def _replacement_mutation(
        space: LayerwiseSearchSpace,
        action: ActionMatrix,
        rng: np.random.Generator,
        *,
        force: bool,
        max_layers: int,
        ) -> ActionMatrix:
    genes = list(space.genes(action))
    probability = 1.0 / float(space.num_layers)
    selected = [
        index for index in range(space.num_layers)
        if float(rng.random()) < probability
    ]
    maximum = min(space.num_layers, max(1, int(max_layers)))
    if len(selected) > maximum:
        chosen = np.asarray(
            rng.choice(
                selected, size=maximum, replace=False,
            ),
            dtype=int,
        ).reshape(-1)
        selected = [int(value) for value in chosen]
    if force and not selected:
        selected = [int(rng.integers(space.num_layers))]
    for layer_idx in selected:
        alternatives = _mesh_adjacent_gene_values(genes[layer_idx])
        genes[layer_idx] = alternatives[int(rng.integers(len(alternatives)))]
    return space.from_genes(genes)

def _mean_pairwise_distance(
        space: LayerwiseSearchSpace,
        actions: Sequence[ActionMatrix],
        ) -> float:
    if len(actions) < 2:
        return 0.0
    total = 0
    count = 0
    genes = [space.genes(action) for action in actions]
    for left, right in itertools.combinations(genes, 2):
        total += sum(a != b for a, b in zip(left, right))
        count += 1
    return float(total) / float(count)

def _population_diversity(
        space: LayerwiseSearchSpace,
        population: Sequence[SearchEvaluation],
        ) -> tuple[float, float]:
    actions = [item.action_matrix for item in population]
    unique_ratio = float(len(set(actions))) / float(max(1, len(actions)))
    return unique_ratio, _mean_pairwise_distance(space, actions)

def _select_hamming_diverse_elites(
        space: LayerwiseSearchSpace,
        population: Sequence[SearchEvaluation],
        elite_count: int,
        ) -> list[SearchEvaluation]:
    """Keep the best incumbent, then prefer feasible elites at distance >= 2."""

    target = min(int(elite_count), len(population))
    if target <= 0:
        return []
    ranked = sorted(population, key=candidate_rank_key, reverse=True)
    feasible = [item for item in ranked if item.feasible]
    selected: list[SearchEvaluation] = []
    selected_actions: set[ActionMatrix] = set()
    for pool in (feasible, ranked):
        remaining = [
            item for item in pool
            if item.action_matrix not in selected_actions
        ]
        while remaining and len(selected) < target:
            if not selected:
                chosen = remaining[0]
            else:
                distance_two = [
                    item for item in remaining
                    if all(
                        _hamming_distance(
                            space, item.action_matrix, owned.action_matrix,
                        ) >= 2
                        for owned in selected
                    )
                ]
                distance_one = [
                    item for item in remaining
                    if all(
                        _hamming_distance(
                            space, item.action_matrix, owned.action_matrix,
                        ) >= 1
                        for owned in selected
                    )
                ]
                chosen = (distance_two or distance_one or remaining)[0]
            selected.append(chosen)
            selected_actions.add(chosen.action_matrix)
            remaining = [
                item for item in remaining
                if item.action_matrix != chosen.action_matrix
            ]
        if len(selected) >= target:
            break
    return selected

def _make_ga_child(
        space: LayerwiseSearchSpace,
        population: Sequence[SearchEvaluation],
        rng: np.random.Generator,
        forbidden: set[ActionMatrix],
        *,
        mutation_max_layers: int,
        ) -> tuple[Optional[ActionMatrix], bool]:
    for _attempt in range(_GA_OFFSPRING_ATTEMPTS):
        parent = _tournament_parent(population, rng)
        child = _replacement_mutation(
            space,
            parent.action_matrix,
            rng,
            force=True,
            max_layers=int(mutation_max_layers),
        )
        if child in forbidden:
            child = _replacement_mutation(
                space,
                parent.action_matrix,
                rng,
                force=True,
                max_layers=int(mutation_max_layers),
            )
        if child not in forbidden:
            return child, False
    maximum = min(space.num_layers, max(1, int(mutation_max_layers)))
    parents = sorted(population, key=candidate_rank_key, reverse=True)
    for parent in parents:
        parent_genes = space.genes(parent.action_matrix)
        for changed_count in range(1, maximum + 1):
            for coordinates in itertools.combinations(
                    range(space.num_layers), changed_count,
            ):
                alternatives = tuple(
                    _mesh_adjacent_gene_values(parent_genes[index])
                    for index in coordinates
                )
                for replacements in itertools.product(*alternatives):
                    genes = list(parent_genes)
                    for index, replacement in zip(coordinates, replacements):
                        genes[index] = int(replacement)
                    child = space.from_genes(genes)
                    if child not in forbidden:
                        return child, False
    return None, False

def _run_coinn_ga(
        space: LayerwiseSearchSpace,
        evaluator: EvaluationFn,
        config: SearchConfig,
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
    population_target = min(
        int(config.ga_population_size),
        int(space.cardinality),
    )
    elite_count = min(
        int(config.ga_elite_count), max(0, population_target - 1),
    )
    structured = _structured_initial_design(
        space, rng, population_target,
    )
    population: list[SearchEvaluation] = []
    structured_index = 0
    while len(population) < population_target and cache.can_observe:
        if structured_index < len(structured):
            action = structured[structured_index]
            structured_index += 1
        else:
            forbidden = {
                item.action_matrix for item in cache.observations
            }
            action = _maximin_candidate(
                space,
                [item.action_matrix for item in population],
                forbidden,
                rng,
            )
            if action is None:
                break
        observed = cache.evaluate(action)
        if observed.inference_performed:
            population.append(observed)

    history: list[dict[str, Any]] = [_best_history_row(
        cache,
        phase="ga_initial_population",
        iteration=0,
        population_size=len(population),
        population_target=int(population_target),
        elite_count=int(elite_count),
        structured_observations=int(min(structured_index, len(structured))),
        initialization_provenance=[
            {
                "action_matrix": [list(row) for row in action],
                "source": (
                    "uniform_anchor"
                    if index < 6
                    else (
                        "balanced_one_layer_from_all_0H"
                        if index < 36
                        else "categorical_maximin"
                    )
                ),
            }
            for index, action in enumerate(structured)
        ],
        non_inference_observations=int(
            cache.observation_count - cache.evaluation_count
        ),
    )]
    observed_actions = {
        item.action_matrix for item in cache.observations
    }
    completed_generations = 0
    if len(population) < population_target:
        raise RuntimeError(
            "Stage-2 GA could not build its complete inference-reaching "
            "initial population"
        )

    while (
            len(population) == population_target
            and completed_generations < int(config.ga_generations)
    ):
        elites = _select_hamming_diverse_elites(
            space, population, elite_count,
        )
        offspring_target = population_target - elite_count
        if offspring_target <= 0:
            raise RuntimeError("Stage-2 GA requires at least one offspring")
        if cache.remaining < offspring_target:
            raise RuntimeError(
                "Stage-2 GA exhausted the action space before every configured "
                "generation completed"
            )
        if space.cardinality - cache.observation_count < offspring_target:
            raise RuntimeError(
                "Stage-2 GA exhausted the candidate space before every configured "
                "generation completed"
            )

        unique_ratio, mean_distance = _population_diversity(space, population)
        offspring: list[SearchEvaluation] = []
        forbidden = observed_actions
        observation_start = cache.observation_count
        previous_best = cache.best()

        while len(offspring) < offspring_target:
            if not cache.can_observe:
                raise RuntimeError(
                    "Stage-2 GA exhausted the action space while constructing "
                    "offspring"
                )
            child, _used_immigrant = _make_ga_child(
                space,
                population,
                rng,
                forbidden,
                mutation_max_layers=int(config.mutation_max_coordinates),
            )
            if child is None:
                raise RuntimeError(
                    "Stage-2 GA could not produce a unique offspring action"
                )
            forbidden.add(child)
            observed = cache.evaluate(child)
            if observed.inference_performed:
                offspring.append(observed)

        population = [*elites, *offspring]
        if len(population) != population_target:
            raise RuntimeError("GA population refill did not preserve its size")
        if any(not item.inference_performed for item in population):
            raise RuntimeError("GA parent population contains a non-inference candidate")
        completed_generations += 1
        improved = (
            candidate_rank_key(cache.best()) > candidate_rank_key(previous_best)
        )
        post_unique_ratio, post_mean_distance = _population_diversity(
            space, population,
        )
        generation_observations = cache.observation_count - observation_start
        history.append(_best_history_row(
            cache,
            phase="ga_update_generation",
            iteration=completed_generations,
            generation=int(completed_generations),
            population_size=int(population_target),
            elite_count=int(elite_count),
            feasible_elite_count=sum(item.feasible for item in elites),
            elite_policy="best_incumbent_then_hamming_distance_2",
            elite_actions=[
                [list(row) for row in item.action_matrix]
                for item in elites
            ],
            offspring_evaluated=int(len(offspring)),
            offspring_observations=int(generation_observations),
            non_inference_offspring_observations=int(
                generation_observations - len(offspring)
            ),
            expected_evaluations=int(
                population_target + completed_generations * offspring_target
            ),
            improved=bool(improved),
            unique_ratio=float(unique_ratio),
            mean_pairwise_distance=float(mean_distance),
            diversity_triggered=False,
            diversity_immigrants=0,
            fallback_immigrants=0,
            replaced_worst_nonelite_actions=[],
            immigrant_actions=[],
            post_update_unique_ratio=float(post_unique_ratio),
            post_update_mean_pairwise_distance=float(post_mean_distance),
        ))
    if completed_generations != int(config.ga_generations):
        raise RuntimeError(
            "Stage-2 GA did not complete every configured generation"
        )
    cache.assert_replay_consumed()
    return SearchResult(
        algorithm="coinn_ga",
        best=cache.best(),
        observations=cache.observations,
        history=tuple(history),
        termination_reason="completed_generations",
    )


run = _run_coinn_ga
