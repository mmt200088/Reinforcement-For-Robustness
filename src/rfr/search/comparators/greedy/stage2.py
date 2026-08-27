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
        preload=preload,
        checkpoint_callback=checkpoint_callback,
    )
    history: list[dict[str, Any]] = []
    anchors: list[SearchEvaluation] = []
    for anchor_idx, action in enumerate(space.uniform_anchors):
        cached = cache.get(action)
        if cached is None:
            if not cache.can_observe:
                raise RuntimeError(
                    "Stage-2 Greedy exhausted the action space before "
                    "evaluating every anchor"
                )
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

    scan_index = 0
    for start_index, start in enumerate(anchors):
        current = start
        no_improvement_rounds = 0
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
                raise RuntimeError(
                    "Stage-2 Greedy exhausted the action space during a 1-opt "
                    "neighborhood scan"
                )
            if one_improved:
                current = one_best
                no_improvement_rounds = 0
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
                raise RuntimeError(
                    "Stage-2 Greedy exhausted the action space during a 2-opt "
                    "neighborhood scan"
                )
            if two_improved:
                current = two_best
                no_improvement_rounds = 0
                continue

            no_improvement_rounds += 1
            history.append(_best_history_row(
                cache,
                phase="greedy_no_improvement_round",
                iteration=scan_index,
                start_index=int(start_index),
                one_opt_improvement=False,
                two_opt_improvement=False,
                no_improvement_rounds=int(no_improvement_rounds),
            ))
            if no_improvement_rounds >= int(
                    config.greedy_no_improvement_rounds
            ):
                break
    cache.assert_replay_consumed()
    return SearchResult(
        algorithm="greedy",
        best=cache.best(),
        observations=cache.observations,
        history=tuple(history),
        termination_reason="consecutive_no_improvement_rounds",
    )


run = _run_greedy
