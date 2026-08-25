"""Greedy operators for the Stage-2 layerwise search."""

from __future__ import annotations

from typing import Any, Iterable, Optional

from rfr.search.comparators.common.stage2_core import (
    ActionMatrix,
    CheckpointCallback,
    EvaluationFn,
    LayerwiseSearchSpace,
    SearchConfig,
    SearchEvaluation,
    SearchResult,
    _EvaluationCache,
    _best_history_row,
    _cache_stop_reason,
    candidate_rank_key,
)

def _evaluate_full_neighborhood(
        cache: _EvaluationCache,
        actions: Iterable[ActionMatrix],
        ) -> tuple[list[SearchEvaluation], bool]:
    neighborhood = [cache.space.validate(action) for action in actions]
    pending = [action for action in neighborhood if not cache.contains(action)]
    for action in pending:
        if not cache.can_observe:
            break
        cache.evaluate(action)
    complete = all(cache.contains(action) for action in neighborhood)
    observed = [
        item
        for action in neighborhood
        for item in (cache.get(action),)
        if item is not None
    ]
    return observed, complete

def _run_greedy(
        space: LayerwiseSearchSpace,
        evaluator: EvaluationFn,
        config: SearchConfig,
        *,
        preload: Iterable[SearchEvaluation] = (),
        checkpoint_callback: Optional[CheckpointCallback] = None,
        ) -> SearchResult:
    cache = _EvaluationCache(
        space,
        evaluator,
        config.evaluation_budget,
        config.observation_attempt_limit,
        preload=preload,
        checkpoint_callback=checkpoint_callback,
    )
    history: list[dict[str, Any]] = []
    anchors: list[SearchEvaluation] = []
    for anchor_idx, action in enumerate(space.uniform_anchors):
        cached = cache.get(action)
        if cached is None:
            if not cache.can_observe:
                break
            cached = cache.evaluate(action)
        anchors.append(cached)
        history.append(_best_history_row(
            cache,
            phase="uniform_anchor",
            iteration=anchor_idx,
            anchor_gene=int(anchor_idx),
        ))
    if not anchors:
        raise RuntimeError("greedy search could not evaluate an anchor")

    termination = "verified_local_optima"
    scan_index = 0
    for start_index, start in enumerate(anchors):
        current = start
        while True:
            scan_index += 1
            one_evaluated, one_complete = _evaluate_full_neighborhood(
                cache, space.neighbors(current.action_matrix),
            )
            one_candidates = [current, *one_evaluated]
            one_best = max(one_candidates, key=candidate_rank_key)
            one_improved = (
                candidate_rank_key(one_best) > candidate_rank_key(current)
            )
            history.append(_best_history_row(
                cache,
                phase="greedy_1opt",
                iteration=scan_index,
                start_index=int(start_index),
                neighborhood_complete=bool(one_complete),
                accepted=bool(one_improved and one_complete),
            ))
            if not one_complete:
                termination = _cache_stop_reason(cache)
                break
            if one_improved:
                current = one_best
                continue

            two_evaluated, two_complete = _evaluate_full_neighborhood(
                cache, space.two_opt_neighbors(current.action_matrix),
            )
            two_candidates = [current, *two_evaluated]
            two_best = max(two_candidates, key=candidate_rank_key)
            two_improved = (
                candidate_rank_key(two_best) > candidate_rank_key(current)
            )
            history.append(_best_history_row(
                cache,
                phase="greedy_2opt",
                iteration=scan_index,
                start_index=int(start_index),
                neighborhood_complete=bool(two_complete),
                accepted=bool(two_improved and two_complete),
            ))
            if not two_complete:
                termination = _cache_stop_reason(cache)
                break
            if two_improved:
                current = two_best

                continue


            history.append(_best_history_row(
                cache,
                phase="greedy_final_verification",
                iteration=scan_index,
                start_index=int(start_index),
                one_opt_improvement=False,
                two_opt_improvement=False,
            ))
            break
        if termination in ("evaluation_budget", "observation_attempt_guard"):
            break
    if termination != "verified_local_optima" and not cache.can_observe:
        termination = _cache_stop_reason(cache)
    cache.assert_replay_consumed()
    return SearchResult(
        algorithm="greedy",
        best=cache.best(),
        observations=cache.observations,
        history=tuple(history),
        termination_reason=termination,
    )


run = _run_greedy
