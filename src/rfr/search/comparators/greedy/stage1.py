"""Greedy operators for the Stage-1 categorical search."""

from __future__ import annotations

from typing import Any, Iterable, Optional

from rfr.search.comparators.common.stage1_core import (
    CheckpointCallback,
    EvaluationFn,
    IncrementalCheckpointCallback,
    SearchConfig,
    SearchEvaluation,
    SearchResult,
    Stage1Action,
    Stage1SearchSpace,
    _EvaluationCache,
    _history_row,
    candidate_rank_key,
)

def _scan_neighborhood(
        cache: _EvaluationCache,
        actions: Iterable[Stage1Action],
        ) -> tuple[list[SearchEvaluation], bool]:
    evaluations: list[SearchEvaluation] = []
    complete = True
    for action in actions:
        cached = cache.get(action)
        if cached is not None:
            evaluations.append(cached)
            continue
        if cache.remaining <= 0:
            complete = False
            break
        evaluations.append(cache.evaluate(action))
    return evaluations, complete

def _run_greedy(
        *,
        space: Stage1SearchSpace,
        evaluator: EvaluationFn,
        config: SearchConfig,
        preload: Iterable[SearchEvaluation],
        checkpoint_callback: Optional[CheckpointCallback],
        incremental_checkpoint_callback: Optional[
            IncrementalCheckpointCallback
        ],
        replay_preload_in_order: bool = True,
        ) -> SearchResult:
    cache = _EvaluationCache(
        space=space,
        evaluator=evaluator,
        evaluation_cap=config.evaluation_cap,
        preload=preload,
        checkpoint_callback=checkpoint_callback,
        incremental_checkpoint_callback=incremental_checkpoint_callback,
        replay_preload_in_order=replay_preload_in_order,
    )
    history: list[dict[str, Any]] = []
    all_verified = True
    starts = space.anchors[:min(int(config.greedy_max_starts), len(space.anchors))]
    for start_idx, start in enumerate(starts):
        if cache.get(start) is None and cache.remaining <= 0:
            all_verified = False
            break
        current = cache.evaluate(start)
        iteration = 0
        history.append(_history_row(
            cache,
            phase="start",
            iteration=iteration,
            start_index=int(start_idx),
            current_action=list(current.action),
        ))
        while True:
            iteration += 1
            one_evaluations, complete = _scan_neighborhood(
                cache, space.one_opt_neighbors(current.action),
            )
            if not complete:
                all_verified = False
                history.append(_history_row(
                    cache,
                    phase="one_opt",
                    iteration=iteration,
                    start_index=int(start_idx),
                    neighborhood_complete=False,
                    accepted=False,
                ))
                break
            one_best = max(one_evaluations, key=candidate_rank_key, default=current)
            if candidate_rank_key(one_best) > candidate_rank_key(current):
                current = one_best
                history.append(_history_row(
                    cache,
                    phase="one_opt",
                    iteration=iteration,
                    start_index=int(start_idx),
                    neighborhood_complete=True,
                    accepted=True,
                    current_action=list(current.action),
                ))
                continue

            two_evaluations, complete = _scan_neighborhood(
                cache, space.two_opt_neighbors(current.action),
            )
            if not complete:
                all_verified = False
                history.append(_history_row(
                    cache,
                    phase="two_opt",
                    iteration=iteration,
                    start_index=int(start_idx),
                    neighborhood_complete=False,
                    accepted=False,
                ))
                break
            two_best = max(two_evaluations, key=candidate_rank_key, default=current)
            if candidate_rank_key(two_best) > candidate_rank_key(current):
                current = two_best
                history.append(_history_row(
                    cache,
                    phase="two_opt",
                    iteration=iteration,
                    start_index=int(start_idx),
                    neighborhood_complete=True,
                    accepted=True,
                    return_to_one_opt=True,
                    current_action=list(current.action),
                ))
                continue
            history.append(_history_row(
                cache,
                phase="verified_local_optimum",
                iteration=iteration,
                start_index=int(start_idx),
                one_opt_verified=True,
                two_opt_verified=True,
                current_action=list(current.action),
            ))
            break
        if not all_verified and cache.remaining <= 0:
            break
    cache.assert_replay_consumed()
    termination = (
        "verified_local_optimum"
        if all_verified
        else "evaluation_cap"
    )
    return SearchResult(
        algorithm="greedy",
        config=config,
        best=cache.best(),
        observations=cache.observations,
        history=tuple(history),
        termination_reason=termination,
    )


run = _run_greedy
