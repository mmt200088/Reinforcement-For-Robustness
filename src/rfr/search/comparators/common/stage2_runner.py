"""Real-model adapter and evidence writer for Stage-2 search baselines."""

from __future__ import annotations

import heapq
import json
import os
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from rfr.preparation.data.protocol import (
    TRAIN_PROBE_SPLIT,
    validate_dataset_protocol_binding,
)
from rfr.common.json_utils import read_json_file, stable_json_hash, to_jsonable
from rfr.common.jsonl_utils import read_jsonl, recover_jsonl_file

from rfr.search.common.candidate_store import CandidateStore, candidate_key
from rfr.search.common.layerwise_action import describe_layerwise_action_matrix
from .stage2_core import (
    CONSTRAINT_NAMES,
    CONSTRAINT_PROBABILITY_NAMES,
    ActionMatrix,
    ConstraintLimits,
    LayerwiseSearchSpace,
    SearchConfig,
    SearchEvaluation,
    SearchMetrics,
    SearchResult,
    _select_hamming_diverse_elites,
    candidate_rank_key,
    normalize_search_backend,
    run_search,
)
from rfr.search.rl.stage2.seed_utils import derive_layerwise_online_evaluation_seeds


STAGE2_FORMAL_GA_GENERATIONS = 200
STAGE2_FORMAL_GA_POPULATION_SIZE = 64
STAGE2_FORMAL_GA_ELITE_COUNT = 7
STAGE2_FORMAL_GA_EVALUATIONS = (
    STAGE2_FORMAL_GA_POPULATION_SIZE
    + STAGE2_FORMAL_GA_GENERATIONS
    * (STAGE2_FORMAL_GA_POPULATION_SIZE - STAGE2_FORMAL_GA_ELITE_COUNT)
)
SEARCH_EVIDENCE_SPLIT = TRAIN_PROBE_SPLIT

def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def limits_from_reference(reference: Any) -> ConstraintLimits:
    """Translate the canonical robust baseline reference into six hard limits."""
    return ConstraintLimits(
        loss_max=float(_field(reference, "loss_limit")),
        metric1_min=float(_field(reference, "metric1_limit")),
        metric2_min=float(_field(reference, "metric2_limit")),
        loss_std_max=float(_field(reference, "loss_std_limit")),
        metric1_std_max=float(_field(reference, "metric1_std_limit")),
        metric2_std_max=float(_field(reference, "metric2_std_limit")),
    )


def _metrics_from_runtime(value: Any) -> SearchMetrics:
    if value is None:
        raise RuntimeError("real Stage-2 evaluation returned no metrics")
    return SearchMetrics(
        loss_mean=float(_field(value, "loss_mean")),
        metric1_mean=float(_field(value, "metric1_mean")),
        metric2_mean=float(_field(value, "metric2_mean")),
        loss_std=float(_field(value, "loss_std")),
        metric1_std=float(_field(value, "metric1_std")),
        metric2_std=float(_field(value, "metric2_std")),
    )


def _serialize_boosted_overrides(value: Any) -> list[dict[str, Any]]:
    rows = []
    for key, field_values in dict(value or {}).items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise RuntimeError(
                "boosted override keys must be (block_idx, layer_idx) tuples"
            )
        block_idx, layer_idx = key
        rows.append({
            "block_idx": int(block_idx),
            "layer_idx": int(layer_idx),
            "field_values": {
                str(name): int(sf)
                for name, sf in dict(field_values or {}).items()
            },
        })
    rows.sort(key=lambda row: (row["layer_idx"], row["block_idx"]))
    return rows


def _atomic_json(path: str, payload: Any) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(
            to_jsonable(payload, stringify_unknown=True),
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)
    directory_fd = os.open(
        directory,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _invalid_metrics(limits: ConstraintLimits, runtime_metrics: Any) -> SearchMetrics:
    if runtime_metrics is not None:
        return _metrics_from_runtime(runtime_metrics)
    return SearchMetrics(
        loss_mean=float(limits.loss_max + max(abs(limits.loss_max), 1.0)),
        metric1_mean=float(limits.metric1_min - max(abs(limits.metric1_min), 1.0)),
        metric2_mean=float(limits.metric2_min - max(abs(limits.metric2_min), 1.0)),
        loss_std=float(limits.loss_std_max + max(abs(limits.loss_std_max), 1.0)),
        metric1_std=float(
            limits.metric1_std_max + max(abs(limits.metric1_std_max), 1.0)
        ),
        metric2_std=float(
            limits.metric2_std_max + max(abs(limits.metric2_std_max), 1.0)
        ),
    )


class LayerwiseRuntimeEvaluator:
    """Evaluate one canonical action through the exact RL-to-model path."""

    def __init__(
            self,
            *,
            env: Any,
            reference: Any,
            base_seed: int,
            expected_trials: int,
            on_evaluation: Callable[[Mapping[str, Any]], None] | None = None,
            ):
        self.env = env
        self.limits = limits_from_reference(reference)
        self.base_seed = int(base_seed)
        self.expected_trials = int(expected_trials)
        if self.expected_trials < 2:
            raise ValueError(
                "Stage-2 search stability evaluation requires at least two trials"
            )
        self.on_evaluation = on_evaluation
        self.evaluation_count = 0

    def __call__(self, action_matrix: ActionMatrix) -> SearchEvaluation:
        evaluation_index = int(self.evaluation_count)
        started = time.perf_counter()
        base_env = getattr(self.env, "base", None)
        if base_env is None:
            raise RuntimeError("layerwise search environment has no base env")
        clear = getattr(base_env, "clear_installed_blb", None)
        if callable(clear):
            clear()
        reset_seed, probe_seed = derive_layerwise_online_evaluation_seeds(
            int(self.base_seed),
            evaluation_index,
            trial_count=self.expected_trials,
        )
        try:
            state = self.env.reset(seed=reset_seed)
            del state
            base_env.probe_noise_seed = probe_seed
            terminal_reward = 0.0
            terminal_info: Mapping[str, Any] = {}
            for layer_idx, row in enumerate(action_matrix):
                _state, reward, done, info = self.env.step(row)
                expected_done = layer_idx == len(action_matrix) - 1
                if bool(done) != bool(expected_done):
                    raise RuntimeError(
                        "layerwise environment terminated at the wrong layer"
                    )
                if expected_done:
                    terminal_reward = float(reward)
                    terminal_info = (
                        info if isinstance(info, Mapping) else {}
                    )

            runtime_info = getattr(self.env, "runtime_terminal_info", None)
            if not isinstance(runtime_info, Mapping):
                raise RuntimeError(
                    "layerwise search completed without runtime terminal info"
                )
            runtime_invalid = bool(runtime_info.get("invalid", False))
            if runtime_invalid:
                candidate_invalid = bool(
                    runtime_info.get("optimizer_invalid_summary")
                    or runtime_info.get("materialization_failure_reason")
                    or runtime_info.get("forward_skipped_reason")
                )
                if (
                        not candidate_invalid
                        or bool(runtime_info.get("apply_failed", False))
                        or bool(runtime_info.get("eval_failed", False))
                ):
                    raise RuntimeError(
                        "real Stage-2 infrastructure evaluation failure outside "
                        "expected optimizer/materialization candidate invalidity: "
                        f"{runtime_info.get('error', 'unknown error')}"
                    )
                metadata = {
                    "evaluation_index": evaluation_index,
                    "wall_seconds": float(time.perf_counter() - started),
                    "inference_performed": False,
                    "forward_ran": False,
                    "model_uses_replan_config": False,
                    "materializable": False,
                    "candidate_invalid": True,
                    "invalid_reason": str(
                        runtime_info.get("materialization_failure_reason")
                        or runtime_info.get("forward_skipped_reason")
                        or runtime_info.get("optimizer_invalid_summary")
                    ),
                    "optimizer_invalid_summary": runtime_info.get(
                        "optimizer_invalid_summary"
                    ),
                    "terminal_info": to_jsonable(
                        terminal_info, stringify_unknown=True,
                    ),
                    "runtime_terminal_info": to_jsonable(
                        runtime_info, stringify_unknown=True,
                    ),
                    "online_stream_index": evaluation_index,
                    "probe_seed": int(probe_seed),
                    "installed_action": {
                        "layers": describe_layerwise_action_matrix(action_matrix),
                    },
                }
                evaluation = SearchEvaluation(
                    action_matrix=action_matrix,
                    metrics=_invalid_metrics(
                        self.limits, runtime_info.get("metrics"),
                    ),
                    limits=self.limits,
                    valid=False,
                    reward=terminal_reward,
                    communication_importance_ratio=float(
                        getattr(
                            self.env, "communication_importance_ratio", 1.0,
                        )
                    ),
                    metadata=metadata,
                )
                self.evaluation_count += 1
                if self.on_evaluation is not None:
                    self.on_evaluation(evaluation.as_dict())
                return evaluation
            if runtime_info.get("forward_ran") is not True:
                raise RuntimeError(
                    "real Stage-2 evaluation did not execute model forward"
                )
            replan = runtime_info.get("replan_application")
            if not (
                    isinstance(replan, Mapping)
                    and replan.get("model_uses_replan_config") is True
            ):
                raise RuntimeError(
                    "real Stage-2 evaluation did not install the replan "
                    "configuration into the model"
                )
            final_config_fingerprint = runtime_info.get(
                "final_config_fingerprint"
            )
            if not (
                    isinstance(final_config_fingerprint, str)
                    and len(final_config_fingerprint) == 64
                    and all(
                        character in "0123456789abcdef"
                        for character in final_config_fingerprint
                    )
            ):
                raise RuntimeError(
                    "real Stage-2 evaluation returned no valid final config "
                    "fingerprint"
                )
            runtime_metrics = runtime_info.get("metrics")
            metrics = _metrics_from_runtime(runtime_metrics)
            assessment = runtime_info.get("statistical_assessment")
            if not isinstance(assessment, Mapping):
                raise RuntimeError(
                    "real Stage-2 evaluation returned no statistical assessment"
                )
            try:
                constraint_probabilities = tuple(
                    float(assessment[name])
                    for name in CONSTRAINT_PROBABILITY_NAMES
                )
                gate_probability = float(assessment["gate_probability"])
                bootstrap_seed = int(assessment["bootstrap_seed"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    "real Stage-2 evaluation statistical assessment does not "
                    "contain the six PPO constraint probabilities, gate, and "
                    "bootstrap seed"
                ) from exc
            precision_pass = min(constraint_probabilities[:3]) >= gate_probability
            stability_pass = min(constraint_probabilities[3:]) >= gate_probability
            expected_priority = (
                1 if not precision_pass else (2 if not stability_pass else 3)
            )
            reward_breakdown = runtime_info.get("reward_breakdown")
            runtime_priority = int(_field(
                reward_breakdown,
                "priority",
                runtime_info.get("priority", 0),
            ))
            if runtime_priority != expected_priority:
                raise RuntimeError(
                    "real Stage-2 reward priority does not match its six "
                    "statistical constraint probabilities"
                )
            trial_seeds = tuple(
                int(value)
                for value in (_field(runtime_metrics, "trial_seeds", ()) or ())
            )
            trial_results = {
                name: tuple(
                    float(value)
                    for value in (
                        _field(runtime_metrics, field_name, ()) or ()
                    )
                )
                for name, field_name in (
                    ("loss", "loss_trials"),
                    ("metric1", "metric1_trials"),
                    ("metric2", "metric2_trials"),
                )
            }
            if (
                    len(trial_seeds) != self.expected_trials
                    or any(
                        len(values) != self.expected_trials
                        for values in trial_results.values()
                    )
            ):
                raise RuntimeError(
                    "real Stage-2 evaluation returned an unexpected trial count: "
                    f"{len(trial_seeds)} != {self.expected_trials}"
                )
            pending_vector = getattr(self.env, "pending_full_vector", ())
            if callable(pending_vector):
                pending_vector = pending_vector()
            metadata = {
                "evaluation_index": evaluation_index,
                "wall_seconds": float(time.perf_counter() - started),
                "inference_performed": True,
                "forward_ran": True,
                "model_uses_replan_config": True,
                "materializable": True,
                "final_config_fingerprint": final_config_fingerprint,
                "online_stream_index": evaluation_index,
                "probe_seed": int(probe_seed),
                "trial_seeds": [int(value) for value in trial_seeds],
                "trial_results": {
                    name: [float(value) for value in values]
                    for name, values in trial_results.items()
                },
                "bootstrap_seed": bootstrap_seed,
                "statistical_assessment": to_jsonable(
                    runtime_info.get("statistical_assessment"),
                    stringify_unknown=True,
                ),
                "replan_application": to_jsonable(
                    replan, stringify_unknown=True,
                ),
                "terminal_info": to_jsonable(
                    terminal_info, stringify_unknown=True,
                ),
                "pending_full_vector": [
                    int(value)
                    for value in np.asarray(
                        pending_vector, dtype=np.int64,
                    ).reshape(-1)
                ],
                "boosted_overrides": _serialize_boosted_overrides(
                    getattr(self.env, "boosted_overrides", {}) or {}
                ),
                "installed_action": {
                    "layers": describe_layerwise_action_matrix(action_matrix),
                },
            }
            evaluation = SearchEvaluation(
                action_matrix=action_matrix,
                metrics=metrics,
                limits=self.limits,
                valid=True,
                reward=terminal_reward,
                communication_importance_ratio=float(
                    getattr(
                        self.env, "communication_importance_ratio", 1.0,
                    )
                ),
                constraint_probabilities=constraint_probabilities,
                gate_probability=gate_probability,
                metadata=metadata,
            )
            self.evaluation_count += 1
            if self.on_evaluation is not None:
                self.on_evaluation(evaluation.as_dict())
            return evaluation
        finally:
            if callable(clear):
                clear()


def persist_search_result(
        *,
        output_dir: str,
        result: SearchResult,
        manifest: Mapping[str, Any],
        observation_rows: Sequence[Mapping[str, Any]] | None = None,
        ) -> dict[str, str]:
    """Write the complete compact evidence needed to replay search figures."""
    os.makedirs(output_dir, exist_ok=True)
    paths = {
        "manifest": os.path.join(output_dir, "manifest.json"),
        "observations": os.path.join(output_dir, "observations.jsonl"),
        "history": os.path.join(output_dir, "history.json"),
        "summary": os.path.join(output_dir, "summary.json"),
    }
    _atomic_json(paths["manifest"], dict(manifest))
    if observation_rows is not None:
        observations_tmp = paths["observations"] + ".tmp"
        with open(observations_tmp, "w", encoding="utf-8") as handle:
            for row in observation_rows:
                handle.write(json.dumps(
                    to_jsonable(row, stringify_unknown=True),
                    ensure_ascii=False,
                    sort_keys=True,
                ))
                handle.write("\n")
        os.replace(observations_tmp, paths["observations"])
    elif not os.path.isfile(paths["observations"]):
        raise RuntimeError(
            "Stage-2 observation journal is missing at result persistence"
        )
    recover_jsonl_file(paths["observations"])
    _atomic_json(paths["history"], list(result.history))
    _atomic_json(paths["summary"], result.as_dict())
    return paths


def _write_observation_row(path: str, row: Mapping[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(
            to_jsonable(row, stringify_unknown=True),
            ensure_ascii=False,
            sort_keys=True,
        ))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _search_preload_from_rows(
        rows: Sequence[Mapping[str, Any]],
        ) -> tuple[SearchEvaluation, ...]:
    ordered: list[SearchEvaluation] = []
    by_action: dict[ActionMatrix, SearchEvaluation] = {}
    for row in rows:
        evaluation = SearchEvaluation.from_dict(row)
        previous = by_action.get(evaluation.action_matrix)
        if previous is not None:
            if previous.as_dict() != evaluation.as_dict():
                raise ValueError(
                    "conflicting Stage-2 observations for one action"
                )
            raise ValueError(
                "duplicate Stage-2 observation for one action"
            )
        by_action[evaluation.action_matrix] = evaluation
        ordered.append(evaluation)
    return tuple(ordered)


def load_search_preload(path: str) -> tuple[SearchEvaluation, ...]:
    """Recover and deserialize crash-safe Stage-2 observation rows."""

    recover_jsonl_file(path)
    return _search_preload_from_rows(read_jsonl(
        path,
        errors="raise",
        dict_only=True,
        missing_ok=False,
    ))


def _without_search_runtime_marker(
        evaluation: SearchEvaluation,
        ) -> dict[str, Any]:
    payload = evaluation.as_dict()
    metadata = dict(payload.get("metadata") or {})
    metadata.pop("search_cumulative_wall_seconds", None)
    payload["metadata"] = metadata
    return payload


def _selected_action_identity_payload(
        evaluation: SearchEvaluation,
        ) -> dict[str, Any]:
    metadata = dict(evaluation.metadata)
    full_vector = tuple(
        int(value)
        for value in metadata.get("pending_full_vector", ())
    )
    if not full_vector:
        raise RuntimeError(
            "Stage-2 selected action identity has no materialized full vector"
        )
    final_config_fingerprint = metadata.get("final_config_fingerprint")
    if not _is_sha256(final_config_fingerprint):
        raise RuntimeError(
            "Stage-2 selected action has no valid final configuration fingerprint"
        )
    payload = {
        "schema_version": "stage2_selected_action_identity_v3",
        "action_matrix": [
            [int(value) for value in row]
            for row in evaluation.action_matrix
        ],
        "full_vector": list(full_vector),
        "boosted_overrides": to_jsonable(
            metadata.get("boosted_overrides", []),
            stringify_unknown=True,
        ),
        "final_config_fingerprint": final_config_fingerprint,
    }
    payload["action_identity_hash"] = stable_json_hash(payload)
    return payload


def _load_persisted_search_result(output_dir: str) -> SearchResult:
    observations = load_search_preload(os.path.join(
        output_dir, "observations.jsonl",
    ))
    summary = read_json_file(os.path.join(output_dir, "summary.json"))
    return SearchResult.from_dict(summary, observations=observations)


def _validate_persisted_online_result(
        *,
        output_dir: str,
        manifest: Mapping[str, Any],
        result: SearchResult,
        ) -> None:
    if result.algorithm != normalize_search_backend(
            manifest.get("search_backend")
    ):
        raise RuntimeError(
            "Stage-2 completed algorithm does not match manifest"
        )
    if str(result.termination_reason) != str(
            manifest.get("termination_reason")
    ):
        raise RuntimeError(
            "Stage-2 completed termination reason does not match manifest"
        )
    if result.observation_count != int(manifest.get("observation_count", -1)):
        raise RuntimeError(
            "Stage-2 completed observation count does not match manifest"
        )
    if result.evaluation_count != int(manifest.get("evaluation_count", -1)):
        raise RuntimeError(
            "Stage-2 completed inference count does not match manifest"
        )
    authoritative_best = max(
        result.observations, key=candidate_rank_key,
    )
    if (
            _without_search_runtime_marker(authoritative_best)
            != _without_search_runtime_marker(result.best)
    ):
        raise RuntimeError(
            "Stage-2 completed online best is not the authoritative JSONL winner"
        )
    matching_best = [
        item for item in result.observations
        if item.action_matrix == result.best.action_matrix
    ]
    if (
            len(matching_best) != 1
            or _without_search_runtime_marker(matching_best[0])
            != _without_search_runtime_marker(result.best)
    ):
        raise RuntimeError(
            "Stage-2 completed best evaluation is absent or stale in JSONL"
        )
    online_best_payload = read_json_file(os.path.join(
        output_dir, "online_best.json",
    ))
    if online_best_payload != result.best.as_dict():
        raise RuntimeError(
            "Stage-2 completed online best does not match persisted result"
        )


def _promotion_payload(value: Any) -> dict[str, Any]:
    return {
        "status": str(_field(value, "status", "")),
        "trial_count": int(_field(value, "trial_count", 0)),
        "fresh_trial_count": int(_field(value, "fresh_trial_count", 0)),
        "metrics": to_jsonable(
            _field(value, "metrics"), stringify_unknown=True,
        ),
        "assessment": to_jsonable(
            _field(value, "assessment"), stringify_unknown=True,
        ),
        "axis_counterfactuals": to_jsonable(
            _field(value, "axis_counterfactuals"),
            stringify_unknown=True,
        ),
    }


def _strict_selected_rank(evaluation: SearchEvaluation) -> tuple[Any, ...]:
    """Return the same ascending strict-selection key used by PPO."""
    from rfr.search.rl.stage2.layerwise_runner import strict_selection_key

    metadata = dict(evaluation.metadata)
    full_vector = tuple(
        int(value) for value in metadata.get("pending_full_vector", ())
    )
    candidate_key_value = metadata.get("strict_candidate_key")
    if not full_vector or not _is_sha256(candidate_key_value):
        raise RuntimeError(
            "strict comparator ranking requires the canonical F4 candidate identity"
        )
    return strict_selection_key(candidate_key_value, {
        "assessment": dict(metadata.get("strict_final_assessment") or {}),
        "constraint_safety_margins": evaluation.normalized_margins,
        "action_matrix": evaluation.action_matrix,
        "full_vector": full_vector,
        "communication_importance_ratio": (
            evaluation.communication_importance_ratio
        ),
    })


def _strict_metrics(value: Any) -> SearchMetrics | None:
    if value is None:
        return None
    try:
        return _metrics_from_runtime(value)
    except (RuntimeError, TypeError, ValueError):
        return None


def _constraint_family_violations(
        evaluation: SearchEvaluation,
        *,
        status: str,
        trial_count: int,
        banks_run: Sequence[str],
        not_run_banks: Sequence[str],
        point_pass: bool | None,
        ) -> dict[str, Any]:
    constraints = {}
    violated = []
    violations = []
    for index, name in enumerate(CONSTRAINT_NAMES):
        raw_margin = evaluation.raw_margins[index]
        normalized_margin = evaluation.normalized_margins[index]
        violation = max(0.0, -float(normalized_margin))
        constraints[name] = {
            "raw_margin": float(raw_margin),
            "normalized_margin": float(normalized_margin),
            "normalized_violation": float(violation),
        }
        violations.append(violation)
        if violation > 0.0:
            violated.append(name)
    return {
        "status": str(status),
        "available": True,
        "trial_count": int(trial_count),
        "banks_run": [str(value) for value in banks_run],
        "not_run_banks": [str(value) for value in not_run_banks],
        "point_pass": None if point_pass is None else bool(point_pass),
        "metrics": evaluation.metrics.as_dict(),
        "limits": evaluation.limits.as_dict(),
        "constraints": constraints,
        "violated_constraints": violated,
        "failed_constraint_count": len(violated),
        "total_normalized_violation": float(sum(violations)),
        "worst_normalized_violation": float(max(violations, default=0.0)),
    }


def _unavailable_constraint_family(status: str) -> dict[str, Any]:
    return {
        "status": str(status),
        "available": False,
        "trial_count": 0,
        "banks_run": [],
        "not_run_banks": ["A", "B", "C"],
        "point_pass": None,
        "metrics": None,
        "limits": None,
        "constraints": {},
        "violated_constraints": [],
        "failed_constraint_count": 0,
        "total_normalized_violation": 0.0,
        "worst_normalized_violation": 0.0,
    }


def _axis_constraint_family(
        evaluation: SearchEvaluation,
        payload: Any,
        ) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return _unavailable_constraint_family("not_run")
    metrics = _strict_metrics(payload.get("metrics"))
    if metrics is None:
        return _unavailable_constraint_family("unavailable_no_metrics")
    try:
        limits = ConstraintLimits(
            loss_max=float(payload["loss_limit"]),
            metric1_min=float(payload["metric1_limit"]),
            metric2_min=float(payload["metric2_limit"]),
            loss_std_max=float(payload["loss_std_limit"]),
            metric1_std_max=float(payload["metric1_std_limit"]),
            metric2_std_max=float(payload["metric2_std_limit"]),
        )
    except (KeyError, TypeError, ValueError):
        return _unavailable_constraint_family("unavailable_no_limits")
    banks = dict(payload.get("banks") or {})
    banks_run = [label for label in ("A", "B", "C") if label in banks]
    not_run_banks = [
        label for label in ("A", "B", "C") if label not in banks
    ]
    trial_count = max(
        (
            int(_field(bank_payload, "trial_count", 0))
            for bank_payload in banks.values()
        ),
        default=0,
    )
    point_pass = payload.get("point_pass")
    if not banks_run:
        status = "available_unknown_bank"
    elif not_run_banks:
        status = (
            "partial_early_stopped"
            if point_pass is False else "partial"
        )
    else:
        status = "complete_passed" if point_pass else "complete_failed"
    axis_evaluation = SearchEvaluation(
        action_matrix=evaluation.action_matrix,
        metrics=metrics,
        limits=limits,
        valid=True,
        reward=evaluation.reward,
        communication_importance_ratio=(
            evaluation.communication_importance_ratio
        ),
    )
    return _constraint_family_violations(
        axis_evaluation,
        status=status,
        trial_count=trial_count,
        banks_run=banks_run,
        not_run_banks=not_run_banks,
        point_pass=(None if point_pass is None else bool(point_pass)),
    )


def _strict_violations(
        evaluation: SearchEvaluation,
        axis_counterfactuals: Any = None,
        *,
        joint_banks_run: Sequence[str] = (),
        joint_not_run_banks: Sequence[str] = (),
        ) -> dict[str, Any]:
    axes = (
        dict(axis_counterfactuals)
        if isinstance(axis_counterfactuals, Mapping) else {}
    )
    families = {
        "joint": _constraint_family_violations(
            evaluation,
            status=(
                "complete" if not joint_not_run_banks else "partial"
            ),
            trial_count=int(
                evaluation.metadata.get("strict_trial_count", 0)
            ),
            banks_run=joint_banks_run,
            not_run_banks=joint_not_run_banks,
            point_pass=evaluation.feasible,
        ),
        "compute_only": _axis_constraint_family(
            evaluation, axes.get("compute"),
        ),
        "communication_only": _axis_constraint_family(
            evaluation, axes.get("communication"),
        ),
    }
    families["joint"]["metrics_source"] = evaluation.metadata.get(
        "strict_metrics_source", "unknown",
    )
    for axis_name in ("compute_only", "communication_only"):
        families[axis_name]["metrics_source"] = evaluation.metadata.get(
            "strict_axis_metrics_source", "not_run",
        )
    available = [
        payload for payload in families.values() if payload["available"]
    ]
    aggregate = {
        "failed_constraint_count": int(sum(
            payload["failed_constraint_count"] for payload in available
        )),
        "total_normalized_violation": float(sum(
            payload["total_normalized_violation"] for payload in available
        )),
        "worst_normalized_violation": float(max(
            (
                payload["worst_normalized_violation"]
                for payload in available
            ),
            default=0.0,
        )),
        "available_families": [
            name for name, payload in families.items()
            if payload["available"]
        ],
        "unavailable_families": [
            name for name, payload in families.items()
            if not payload["available"]
        ],
    }
    aggregate["available_family_count"] = len(
        aggregate["available_families"]
    )
    aggregate["unavailable_family_count"] = len(
        aggregate["unavailable_families"]
    )
    return {"families": families, "aggregate": aggregate}


def _strict_hard_gate_incomplete_families(
        violations: Mapping[str, Any],
        ) -> list[str]:
    families = dict(violations.get("families") or {})
    return [
        name
        for name in ("joint", "compute_only", "communication_only")
        if (
            not isinstance(families.get(name), Mapping)
            or not bool(families[name].get("available", False))
            or families[name].get("point_pass") is not True
            or list(families[name].get("not_run_banks") or [])
        )
    ]


def _strict_fallback_rank(evaluation: SearchEvaluation) -> tuple[float, ...]:
    resource = evaluation.resource
    violations = dict(
        evaluation.metadata.get("strict_violations") or {}
    )
    aggregate = dict(violations.get("aggregate") or {})
    lexicographic = tuple(
        -float(value)
        for value in LayerwiseSearchSpace(
            len(evaluation.action_matrix)
        ).flatten(evaluation.action_matrix)
    )
    return (
        -float(aggregate.get("failed_constraint_count", 0)),
        -float(aggregate.get("total_normalized_violation", 0.0)),
        -float(aggregate.get("worst_normalized_violation", 0.0)),
        -float(aggregate.get("unavailable_family_count", 3)),
        float(resource.ppo_resource_score),
        float(resource.robust_floor),
        *lexicographic,
    )


def _validate_strict_validation_payload(
        *,
        result: SearchResult,
        payload: Mapping[str, Any],
        communication_importance_ratio: float,
        ) -> None:
    """Validate the ordinary scientific structure of one strict result."""
    if payload.get("schema_version") != "stage2_search_strict_validation_v3":
        raise RuntimeError("strict validation has an unsupported schema")
    if type(payload.get("requested_top_n")) is not int or int(
            payload["requested_top_n"]
    ) != 5:
        raise RuntimeError("strict validation must contain exactly five candidates")
    if type(payload.get("strict_evaluated_candidate_count")) is not int or int(
            payload["strict_evaluated_candidate_count"]
    ) != 5:
        raise RuntimeError("strict validation must evaluate exactly five candidates")

    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 5:
        raise RuntimeError("strict validation must contain exactly five records")
    eligible = [
        candidate
        for candidate in result.observations
        if (
            candidate.valid
            and candidate.inference_performed
            and bool(candidate.metadata.get("materializable", False))
            and bool(candidate.metadata.get("pending_full_vector"))
        )
    ]
    if len(eligible) < 5:
        raise RuntimeError(
            "strict validation result has fewer than five eligible candidates"
        )
    expected_online = heapq.nlargest(5, eligible, key=candidate_rank_key)

    strict_rows: list[tuple[SearchEvaluation, bool]] = []
    required_families = {
        "joint", "compute_only", "communication_only",
    }
    for index, (record, expected) in enumerate(zip(records, expected_online)):  # noqa: B905 - Python 3.9
        if not isinstance(record, Mapping):
            raise RuntimeError(f"strict record {index} is not an object")
        online_payload = record.get("online_candidate")
        strict_payload = record.get("strict_evaluation")
        if not isinstance(online_payload, Mapping) or not isinstance(
                strict_payload, Mapping
        ):
            raise RuntimeError(
                f"strict record {index} has no online/strict evaluation"
            )
        online = _evaluation_from_payload(
            online_payload, communication_importance_ratio,
        )
        strict = _evaluation_from_payload(
            strict_payload, communication_importance_ratio,
        )
        if (
                _without_search_runtime_marker(online)
                != _without_search_runtime_marker(expected)
        ):
            raise RuntimeError(
                "strict records do not match the authoritative online top five"
            )
        if strict.action_matrix != online.action_matrix:
            raise RuntimeError(
                f"strict record {index} changed the candidate action"
            )
        if (
                record.get("strict_evaluated") is not True
                or record.get("selection_eligible") is not True
                or type(record.get("strict_feasible")) is not bool
        ):
            raise RuntimeError(
                f"strict record {index} is not a completed eligible evaluation"
            )
        violations = record.get("violations")
        if not isinstance(violations, Mapping):
            raise RuntimeError(f"strict record {index} has no violations")
        families = violations.get("families")
        if (
                not isinstance(families, Mapping)
                or not required_families.issubset(families)
                or any(
                    not isinstance(families[name], Mapping)
                    for name in required_families
                )
        ):
            raise RuntimeError(
                f"strict record {index} is missing constraint families"
            )
        if strict.metadata.get("strict_violations") != violations:
            raise RuntimeError(
                f"strict record {index} has inconsistent violation evidence"
            )
        _selected_action_identity_payload(strict)
        strict_rows.append((strict, bool(record["strict_feasible"])))

    verdict = payload.get("strict_feasible")
    if type(verdict) is not bool:
        raise RuntimeError("strict validation has no boolean strict verdict")
    status = str(payload.get("selection_status") or "")
    expected_status = (
        "strict_feasible" if verdict else "strict_least_violating"
    )
    if status != expected_status:
        raise RuntimeError("strict validation has an inconsistent selection status")
    strict_passes = [row for row, passed in strict_rows if passed]
    if verdict:
        if not strict_passes:
            raise RuntimeError("strict feasible verdict has no feasible candidate")
        expected_selected = min(strict_passes, key=_strict_selected_rank)
    else:
        if strict_passes:
            raise RuntimeError("least-violating verdict contains a feasible candidate")
        expected_selected = max(
            (row for row, _passed in strict_rows),
            key=_strict_fallback_rank,
        )

    selected_payload = payload.get("selected")
    if not isinstance(selected_payload, Mapping):
        raise RuntimeError("strict validation has no selected candidate")
    selected = _evaluation_from_payload(
        selected_payload, communication_importance_ratio,
    )
    if (
            selected_payload.get("strict_feasible") is not verdict
            or str(selected_payload.get("selection_status") or "") != status
            or _selected_action_identity_payload(selected)
            != _selected_action_identity_payload(expected_selected)
            or selected.as_dict() != expected_selected.as_dict()
    ):
        raise RuntimeError(
            "selected strict candidate does not match the scientific ranking"
        )
    if payload.get("selected_violations") != expected_selected.metadata.get(
            "strict_violations"
    ):
        raise RuntimeError(
            "selected strict candidate has inconsistent violation evidence"
        )


_STRICT_MATERIALIZATION_FAMILIES = (
    "joint",
    "compute_only",
    "communication_only",
)


def _prepare_strict_materialization_fingerprints(
        *,
        layerwise_env: Any,
        base_env: Any,
        action_matrix: Sequence[Sequence[int]],
        joint_action_indices: Sequence[int],
        joint_boosted_overrides: Mapping[Any, Any],
        ) -> dict[str, str]:
    from rfr.search.common.layerwise_action import materialize_layerwise_counterfactuals

    materializations = materialize_layerwise_counterfactuals(
        layerwise_env.baseline_full_vector,
        action_matrix,
        layerwise_env.schedule,
        layerwise_env.fusion_map,
    )
    joint = materializations["joint"]
    if (
            tuple(int(value) for value in joint.full_vector)
            != tuple(int(value) for value in joint_action_indices)
            or _serialize_boosted_overrides(joint.boosted_overrides)
            != _serialize_boosted_overrides(joint_boosted_overrides)
    ):
        raise RuntimeError(
            "Stage-2 strict candidate does not match canonical joint materialization"
        )

    fingerprints = {}
    for family in _STRICT_MATERIALIZATION_FAMILIES:
        materialization = materializations[family]
        prepared = base_env.prepare_action_for_terminal_probe(
            [int(value) for value in materialization.full_vector],
            boosted_overrides={
                (int(block_idx), int(layer_idx)): {
                    str(name): int(value)
                    for name, value in fields.items()
                }
                for (block_idx, layer_idx), fields in (
                    materialization.boosted_overrides.items()
                )
            },
        )
        if (
                not isinstance(prepared, Mapping)
                or bool(prepared.get("any_invalid", False))
                or prepared.get("requires_forward") is not True
        ):
            raise RuntimeError(
                f"Stage-2 strict {family} action is not materializable"
            )
        fingerprint = str(prepared.get("final_config_fingerprint") or "")
        if not _is_sha256(fingerprint):
            raise RuntimeError(
                f"Stage-2 strict {family} final config fingerprint is invalid"
            )
        fingerprints[family] = fingerprint
    return fingerprints


def _strict_reference_for_trial_count(
        validation_banks: Any,
        trial_count: int,
        ) -> tuple[str, Any]:
    """Return the reference whose limits produced the pooled strict metrics."""
    bank_a = _field(validation_banks, "bank_a")
    candidates = (
        (
            "bank_a",
            int(_field(bank_a, "trial_count", 0)),
            _field(bank_a, "reference"),
        ),
        (
            "promotion",
            int(_field(validation_banks, "promotion_trial_count", 0)),
            _field(validation_banks, "promotion_reference"),
        ),
        (
            "final",
            int(_field(validation_banks, "final_trial_count", 0)),
            _field(validation_banks, "final_reference"),
        ),
    )
    available = [
        (label, count, reference)
        for label, count, reference in candidates
        if count > 0 and reference is not None and int(trial_count) >= count
    ]
    if not available:
        raise RuntimeError(
            "strict pooled metrics have no same-source validation reference: "
            f"trial_count={int(trial_count)}"
        )
    label, _count, reference = max(available, key=lambda item: item[1])
    return label, reference


def _completed_validation_banks(
        validation_banks: Any,
        trial_count: int,
        ) -> tuple[list[str], list[str]]:
    bank_a = int(_field(
        _field(validation_banks, "bank_a"), "trial_count", 0,
    ))
    promotion = int(_field(
        validation_banks, "promotion_trial_count", 0,
    ))
    final = int(_field(validation_banks, "final_trial_count", 0))
    thresholds = (("A", bank_a), ("B", promotion), ("C", final))
    known = [(label, count) for label, count in thresholds if count > 0]
    if not known:
        return [], []
    completed = [
        label for label, count in known if int(trial_count) >= count
    ]
    not_run = [label for label, _count in known if label not in completed]
    return completed, not_run


def _evaluation_from_payload(
        payload: Mapping[str, Any],
        communication_importance_ratio: float,
        ) -> SearchEvaluation:
    return SearchEvaluation.from_dict({
        **dict(payload),
        "communication_importance_ratio": float(
            communication_importance_ratio
        ),
    })


def canonical_strict_validation(
        *,
        result: SearchResult,
        layerwise_env: Any,
        promotion_base_env: Any,
        candidate_store: Any,
        identity_context: Mapping[str, Any],
        validation_banks: Any,
        top_n: int,
        communication_importance_ratio: float,
        promotion_probability: float,
        final_probability: float,
        ) -> dict[str, Any]:
    """Reuse the canonical A/B/C point gates and axis counterfactuals."""
    from rfr.search.rl.stage2.layerwise_runner import (
        _FINAL_REVALIDATION_PASSED,
        _deserialize_boosted_overrides,
        _strict_evidence_final_config_fingerprint,
        certify_candidate_with_bank_c,
        evidence_identity_context,
        promote_candidate_if_eligible,
    )

    requested_top_n = int(top_n)
    if requested_top_n != 5:
        raise ValueError("strict validation requires exactly top 5 candidates")
    eligible = [
        candidate
        for candidate in result.observations
        if (
            candidate.valid
            and candidate.inference_performed
            and bool(candidate.metadata.get("materializable", False))
            and bool(candidate.metadata.get("pending_full_vector"))
        )
    ]
    if len(eligible) < requested_top_n:
        raise RuntimeError(
            "strict validation requires top-N optimizer-valid, materializable, "
            "model-forward candidates: "
            f"eligible={len(eligible)} requested={requested_top_n}"
        )
    ranked = heapq.nlargest(
        requested_top_n, eligible, key=candidate_rank_key,
    )
    records: list[dict[str, Any]] = []
    strict_evaluations: list[tuple[SearchEvaluation, bool]] = []
    for online in ranked:
        resource = online.resource
        metadata = dict(online.metadata)
        full_vector = tuple(
            int(value)
            for value in metadata.get("pending_full_vector", ())
        )
        materializable = bool(
            full_vector and metadata.get("materializable", True)
        )
        if not online.valid or not materializable:
            records.append({
                "online_candidate": online.as_dict(),
                "materializable": False,
                "strict_evaluated": False,
                "promotion": None,
                "certification": None,
                "strict_point_pass": False,
                "strict_feasible": False,
                "selection_eligible": False,
                "skip_reason": (
                    "online_candidate_invalid"
                    if not online.valid else "no_materialized_full_vector"
                ),
            })
            continue
        boosted_overrides = _deserialize_boosted_overrides(
            metadata.get("boosted_overrides", ())
        )
        assessment = dict(metadata.get("statistical_assessment") or {})
        if "bootstrap_seed" not in assessment:
            raise RuntimeError(
                "strict validation candidate has no bootstrap seed"
            )
        bootstrap_seed = int(assessment["bootstrap_seed"])
        promotion = promote_candidate_if_eligible(
            env=layerwise_env,
            promotion_base_env=promotion_base_env,
            candidate_store=candidate_store,
            action_indices=full_vector,
            identity_context=identity_context,
            action_matrix=online.action_matrix,
            assessment=assessment,
            priority=3,
            variable_cost=float(resource.ppo_resource_score),
            frontier_cost=None,
            frontier_candidates=None,
            boosted_overrides=boosted_overrides,
            bootstrap_seed=bootstrap_seed,
            episode_reward=online.reward,
            promotion_probability=float(promotion_probability),
            validation_banks=validation_banks,
        )
        promotion_record = _promotion_payload(promotion)
        if promotion_record["status"] == "failed_evaluation":
            raise RuntimeError(
                "strict promotion infrastructure evaluation failed; "
                "preserving search_complete_pending_strict for retry"
            )
        record = {
            "online_candidate": online.as_dict(),
            "materializable": True,
            "strict_evaluated": False,
            "promotion": promotion_record,
            "certification": None,
            "strict_point_pass": False,
            "strict_feasible": False,
            "selection_eligible": False,
        }
        records.append(record)
        certification = None
        certification_record = None
        if promotion_record["status"] in (
                "promoted", "already_promoted",
        ):
            candidate = {
                "action_matrix": online.action_matrix,
                "full_vector": full_vector,
                "boosted_overrides": boosted_overrides,
                "reward": online.reward,
            }
            certification = certify_candidate_with_bank_c(
                env=layerwise_env,
                promotion_base_env=promotion_base_env,
                candidate_store=candidate_store,
                identity_context=identity_context,
                candidate=candidate,
                bootstrap_seed=bootstrap_seed,
                final_probability=float(final_probability),
                validation_banks=validation_banks,
                )
            certification_record = _promotion_payload(certification)
            if certification_record["status"] == "failed_evaluation":
                raise RuntimeError(
                    "strict certification infrastructure evaluation failed; "
                    "preserving search_complete_pending_strict for retry"
                )
            record["certification"] = certification_record

        strict_metrics = _strict_metrics(
            None if certification is None else certification.metrics
        )
        strict_evidence = (
            None if certification is None else certification.evidence
        )
        metrics_source = "certification"
        strict_trial_count = (
            0 if certification_record is None
            else int(certification_record["trial_count"])
        )
        strict_assessment = (
            None if certification is None else certification.assessment
        )
        certification_axes = (
            None if certification_record is None
            else certification_record["axis_counterfactuals"]
        )
        promotion_axes = promotion_record["axis_counterfactuals"]
        if isinstance(certification_axes, Mapping) and certification_axes:
            strict_axis_counterfactuals = certification_axes
            strict_axis_metrics_source = "certification"
        elif isinstance(promotion_axes, Mapping) and promotion_axes:
            strict_axis_counterfactuals = promotion_axes
            strict_axis_metrics_source = "promotion"
        else:
            strict_axis_counterfactuals = None
            strict_axis_metrics_source = "not_run"
        strict_status = (
            promotion_record["status"]
            if certification_record is None
            else certification_record["status"]
        )
        if strict_metrics is None:
            strict_metrics = _strict_metrics(promotion.metrics)
            strict_evidence = promotion.evidence
            metrics_source = "promotion"
            strict_trial_count = int(promotion_record["trial_count"])
            strict_assessment = promotion.assessment
        if strict_metrics is None or strict_trial_count <= 0:
            record["skip_reason"] = "no_strict_pooled_metrics"
            continue
        strict_candidate_key = _field(strict_evidence, "candidate_key")
        if not _is_sha256(strict_candidate_key):
            raise RuntimeError(
                "canonical strict validation has no F4 candidate key"
            )
        strict_action_indices = tuple(
            int(value)
            for value in (_field(strict_evidence, "action_indices") or ())
        )
        expected_candidate_key = candidate_key(
            full_vector,
            evidence_identity_context(identity_context, "F4"),
        )
        if (
                strict_action_indices != full_vector
                or strict_candidate_key != expected_candidate_key
        ):
            raise RuntimeError(
                "canonical strict validation F4 candidate identity mismatch"
            )

        strict_final_config_fingerprint = (
            _strict_evidence_final_config_fingerprint(
                strict_evidence,
                context="canonical strict validation",
            )
        )
        if strict_final_config_fingerprint is None:
            raise RuntimeError(
                "canonical strict validation has no final config fingerprint"
            )
        strict_materialization_fingerprints = (
            _prepare_strict_materialization_fingerprints(
                layerwise_env=layerwise_env,
                base_env=promotion_base_env,
                action_matrix=online.action_matrix,
                joint_action_indices=full_vector,
                joint_boosted_overrides=boosted_overrides,
            )
        )
        if strict_materialization_fingerprints["joint"] != (
                strict_final_config_fingerprint
        ):
            raise RuntimeError(
                "canonical strict joint materialization fingerprint changed"
            )
        if isinstance(strict_axis_counterfactuals, Mapping):
            for axis_name, family in (
                    ("compute", "compute_only"),
                    ("communication", "communication_only"),
            ):
                axis_payload = strict_axis_counterfactuals.get(axis_name)
                if (
                        isinstance(axis_payload, Mapping)
                        and axis_payload.get("final_config_fingerprint")
                        != strict_materialization_fingerprints[family]
                ):
                    raise RuntimeError(
                        f"canonical strict {family} materialization fingerprint "
                        "changed"
                    )

        passed = bool(
            certification_record is not None
            and certification_record["status"] in (
                _FINAL_REVALIDATION_PASSED,
                "already_final_certified",
            )
        )
        strict_limits_source, strict_reference = (
            _strict_reference_for_trial_count(
                validation_banks,
                strict_trial_count,
            )
        )
        strict_evaluation = SearchEvaluation(
            action_matrix=online.action_matrix,
            metrics=strict_metrics,
            limits=limits_from_reference(strict_reference),
            valid=True,
            reward=online.reward,
            communication_importance_ratio=float(
                communication_importance_ratio
            ),
            metadata={
                **metadata,
                "final_config_fingerprint": strict_final_config_fingerprint,
                "strict_validation_status": strict_status,
                "strict_metrics_source": metrics_source,
                "strict_limits_source": strict_limits_source,
                "strict_trial_count": int(strict_trial_count),
                "strict_materialization_fingerprints": dict(
                    strict_materialization_fingerprints
                ),
                "strict_final_assessment": to_jsonable(
                    strict_assessment, stringify_unknown=True,
                ),
                "strict_axis_counterfactuals": strict_axis_counterfactuals,
                "strict_axis_metrics_source": strict_axis_metrics_source,
                "strict_candidate_store": os.fspath(candidate_store.path),
                "strict_candidate_key": str(strict_candidate_key),
            },
        )
        strict_point_feasible = bool(
            passed and strict_evaluation.feasible
        )
        if passed and not strict_evaluation.feasible:
            raise RuntimeError(
                "canonical final certification passed but pooled six-point "
                "limits do not pass"
            )
        joint_banks_run, joint_not_run_banks = _completed_validation_banks(
            validation_banks, strict_trial_count,
        )
        violations = _strict_violations(
            strict_evaluation,
            strict_axis_counterfactuals,
            joint_banks_run=joint_banks_run,
            joint_not_run_banks=joint_not_run_banks,
        )
        if passed:
            incomplete_families = _strict_hard_gate_incomplete_families(
                violations
            )
            if incomplete_families:
                raise RuntimeError(
                    "final strict certification is missing complete passing "
                    "joint/axis evidence for "
                    f"{incomplete_families}"
                )
        strict_evaluation.metadata["strict_violations"] = violations
        record.update({
            "strict_evaluated": True,
            "selection_eligible": True,
            "strict_trial_count": int(strict_trial_count),
            "strict_metrics_source": metrics_source,
            "strict_point_pass": bool(passed),
            "strict_feasible": strict_point_feasible,
            "violations": violations,
            "strict_evaluation": strict_evaluation.as_dict(),
        })
        strict_evaluations.append((strict_evaluation, strict_point_feasible))
    if len(strict_evaluations) != requested_top_n:
        raise RuntimeError(
            "strict validation did not produce complete pooled metrics for every "
            f"top-N candidate: {len(strict_evaluations)} != {requested_top_n}"
        )
    strict_passes = [
        evaluation
        for evaluation, strict_point_feasible in strict_evaluations
        if strict_point_feasible
    ]
    if strict_passes:
        selected = min(strict_passes, key=_strict_selected_rank)
        selection_status = "strict_feasible"
        strict_feasible = True
    elif strict_evaluations:
        selected = max(
            (evaluation for evaluation, _strict in strict_evaluations),
            key=_strict_fallback_rank,
        )
        selection_status = "strict_least_violating"
        strict_feasible = False
    else:
        selected = None
        selection_status = "no_strict_evaluated_materializable_candidate"
        strict_feasible = False
    selected_violations = (
        None
        if selected is None
        else dict(selected.metadata.get("strict_violations") or {})
    )
    selected_payload = None if selected is None else {
        **selected.as_dict(),
        "selection_status": selection_status,
        "strict_feasible": bool(strict_feasible),
        "violations": selected_violations,
    }
    return {
        "schema_version": "stage2_search_strict_validation_v3",
        "split": SEARCH_EVIDENCE_SPLIT,
        "dataset_protocol_hash": identity_context.get(
            "dataset_protocol_hash"
        ),
        "validation_banks": validation_banks.contract_payload(),
        "joint_and_axis_counterfactual_gate": True,
        "hard_gate": (
            "joint_six_point_plus_compute_and_communication_"
            "counterfactual_six_point_v1"
        ),
        "bootstrap_probability_role": "diagnostic_tiebreak_only",
        "candidate_store": os.fspath(candidate_store.path),
        "requested_top_n": int(requested_top_n),
        "eligible_online_candidate_count": len(eligible),
        "strict_evaluated_candidate_count": len(strict_evaluations),
        "online_best": result.best.as_dict(),
        "selection_status": selection_status,
        "strict_feasible": bool(strict_feasible),
        "selected_violations": selected_violations,
        "selected": selected_payload,
        "records": records,
    }


def _validate_ga_completion_proof(
        result: SearchResult,
        *,
        patience_generations: int,
        generation_cap: int,
        maximum_evaluations: int,
        stop_on_no_improvement: bool = True,
        require_full_generations: bool = False,
        ) -> None:
    """Validate COINN-GA completion from persisted search evidence."""

    patience = int(patience_generations)
    generation_limit = int(generation_cap)
    evaluation_limit = int(maximum_evaluations)
    if (
            patience <= 0
            or generation_limit <= 0
            or evaluation_limit <= 0
            or type(stop_on_no_improvement) is not bool
            or type(require_full_generations) is not bool
            or (require_full_generations and stop_on_no_improvement)
    ):
        raise RuntimeError(
            "COINN-GA completion proof has inconsistent safety caps"
        )
    if require_full_generations and result.termination_reason != "generation_limit":
        raise RuntimeError(
            "COINN-GA full-generation completion proof requires the exact cap"
        )
    if (
            not require_full_generations
            and result.termination_reason not in {
                "ga_no_incumbent_improvement", "generation_limit",
            }
    ):
        raise RuntimeError(
            "COINN-GA lacks native incumbent stagnation or "
            "generation-cap completion proof"
        )
    if result.evaluation_count > evaluation_limit:
        raise RuntimeError(
            "COINN-GA exceeded its maximum inference budget"
        )

    initial_rows = [
        row for row in result.history
        if isinstance(row, Mapping) and row.get("phase") == "ga_initial_population"
    ]
    update_rows = [
        row for row in result.history
        if isinstance(row, Mapping) and row.get("phase") == "ga_update_generation"
    ]
    if (
            len(initial_rows) != 1
            or not result.history
            or result.history[0] is not initial_rows[0]
            or len(result.history) != 1 + len(update_rows)
    ):
        raise RuntimeError(
            "COINN-GA completion proof has invalid generation history"
        )

    initial = initial_rows[0]
    population_size = initial.get("population_target")
    elite_count = initial.get("elite_count")
    if (
            type(population_size) is not int
            or type(elite_count) is not int
            or population_size <= 0
            or elite_count < 0
            or elite_count >= population_size
    ):
        raise RuntimeError(
            "COINN-GA completion proof has invalid population settings"
        )
    offspring_count = population_size - elite_count
    initial_observation_count = initial.get("observations")
    if (
            type(initial_observation_count) is not int
            or initial_observation_count < population_size
            or initial_observation_count > result.observation_count
            or int(initial.get("evaluations", -1)) != population_size
            or int(initial.get("population_size", -1)) != population_size
    ):
        raise RuntimeError(
            "COINN-GA completion proof has invalid initial population"
        )
    population = [
        item
        for item in result.observations[:initial_observation_count]
        if item.inference_performed
    ]
    if len(population) != population_size:
        raise RuntimeError(
            "COINN-GA initial population is not backed by real "
            "evaluation evidence"
        )

    cumulative_observations = initial_observation_count
    cumulative_evaluations = population_size
    no_improvement_generations = 0
    expected_generation = 1
    for row in update_rows:
        if (
                int(row.get("generation", -1)) != expected_generation
                or int(row.get("iteration", -1)) != expected_generation
        ):
            raise RuntimeError(
                "COINN-GA generations are not contiguous"
            )
        expected_elites = _select_hamming_diverse_elites(
            LayerwiseSearchSpace(len(result.best.action_matrix)),
            population,
            elite_count,
        )
        recorded_elites = row.get("elite_actions")
        expected_elite_actions = [
            [list(layer) for layer in item.action_matrix]
            for item in expected_elites
        ]
        if (
                recorded_elites != expected_elite_actions
                or int(row.get("elite_count", -1)) != elite_count
        ):
            raise RuntimeError(
                "COINN-GA does not prove exact elite retention"
            )

        next_observation_count = row.get("observations")
        next_evaluation_count = row.get("evaluations")
        if (
                type(next_observation_count) is not int
                or type(next_evaluation_count) is not int
                or next_observation_count <= cumulative_observations
                or next_observation_count > result.observation_count
                or next_evaluation_count
                != cumulative_evaluations + offspring_count
                or int(row.get("offspring_evaluated", -1)) != offspring_count
                or int(row.get("expected_evaluations", -1))
                != next_evaluation_count
        ):
            raise RuntimeError(
                "COINN-GA has invalid generation accounting"
            )
        generation_observations = result.observations[
            cumulative_observations:next_observation_count
        ]
        offspring = [
            item for item in generation_observations
            if item.inference_performed
        ]
        if len(offspring) != offspring_count:
            raise RuntimeError(
                "COINN-GA generation lacks real offspring evaluation evidence"
            )

        previous_best = max(
            result.observations[:cumulative_observations],
            key=candidate_rank_key,
        )
        current_best = max(
            result.observations[:next_observation_count],
            key=candidate_rank_key,
        )
        improved = (
            candidate_rank_key(current_best) > candidate_rank_key(previous_best)
        )
        no_improvement_generations = (
            0 if improved else no_improvement_generations + 1
        )
        recorded_stagnation = row.get("no_improvement_generations")
        if (
                row.get("improved") is not improved
                or type(recorded_stagnation) is not int
                or recorded_stagnation != no_improvement_generations
        ):
            raise RuntimeError(
                "COINN-GA has inconsistent incumbent stagnation evidence"
            )

        population = [*expected_elites, *offspring]
        cumulative_observations = next_observation_count
        cumulative_evaluations = next_evaluation_count
        expected_generation += 1

    if (
            cumulative_observations != result.observation_count
            or cumulative_evaluations != result.evaluation_count
    ):
        raise RuntimeError(
            "COINN-GA completion proof leaves observations unassigned"
        )
    completed_generations = expected_generation - 1
    if require_full_generations:
        if (
                completed_generations != generation_limit
                or result.evaluation_count
                != population_size + generation_limit * offspring_count
        ):
            raise RuntimeError(
                "COINN-GA full-generation completion proof is incomplete"
            )
    elif result.termination_reason == "ga_no_incumbent_improvement":
        if (
                not stop_on_no_improvement
                or completed_generations >= generation_limit
                or no_improvement_generations != patience
        ):
            raise RuntimeError(
                "COINN-GA lacks a five-generation incumbent stagnation proof"
            )
    elif (
            completed_generations != generation_limit
            or result.evaluation_count
            != population_size + generation_limit * offspring_count
    ):
        raise RuntimeError(
            "COINN-GA generation safety-cap proof is incomplete"
        )


def _load_plain_completed_search_run(
        *,
        output_dir: str,
        manifest: Mapping[str, Any],
        communication_importance_ratio: float,
        ) -> dict[str, Any]:
    result = _load_persisted_search_result(output_dir)
    _validate_persisted_online_result(
        output_dir=output_dir,
        manifest=manifest,
        result=result,
    )
    paths = {
        "manifest": os.path.join(output_dir, "manifest.json"),
        "observations": os.path.join(output_dir, "observations.jsonl"),
        "history": os.path.join(output_dir, "history.json"),
        "summary": os.path.join(output_dir, "summary.json"),
        "online_best": os.path.join(output_dir, "online_best.json"),
        "final_selected_configuration": os.path.join(
            output_dir, "final_selected_configuration.json",
        ),
    }
    selected_payload = read_json_file(paths["final_selected_configuration"])
    if selected_payload is None:
        selected = None
    elif isinstance(selected_payload, Mapping):
        selected = _evaluation_from_payload(
            selected_payload,
            communication_importance_ratio,
        )
    else:
        raise RuntimeError(
            "completed Stage-2 selected configuration must be an object or null"
        )

    strict_validation = None
    status = str(manifest.get("status") or "")
    if status != "smoke_only_complete":
        strict_path = os.path.join(output_dir, "strict_validation.json")
        if not os.path.isfile(strict_path):
            raise RuntimeError(
                "completed Stage-2 strict validation artifact is missing"
            )
        payload = read_json_file(strict_path)
        if not isinstance(payload, Mapping):
            raise RuntimeError(
                "completed Stage-2 strict validation artifact is invalid"
            )
        strict_validation = dict(payload)
        _validate_strict_validation_payload(
            result=result,
            payload=strict_validation,
            communication_importance_ratio=communication_importance_ratio,
        )
        verdict = strict_validation.get("strict_feasible")
        if type(verdict) is not bool:
            raise RuntimeError(
                "completed strict validation has no boolean strict verdict"
            )
        if bool(manifest.get("strict_feasible", False)) != verdict:
            raise RuntimeError(
                "completed strict verdict does not match manifest"
            )
        expected_status = (
            "complete_strict_feasible"
            if verdict else "complete_least_violating"
        )
        if status != expected_status:
            raise RuntimeError(
                "completed strict status does not match strict verdict"
            )
        strict_selection_status = str(
            strict_validation.get("selection_status") or ""
        )
        expected_selection_status = (
            "strict_feasible" if verdict else "strict_least_violating"
        )
        if (
                strict_selection_status != expected_selection_status
                or str(manifest.get("selection_status") or "")
                != expected_selection_status
        ):
            raise RuntimeError(
                "completed strict selection status is inconsistent"
            )
        strict_selected_payload = strict_validation.get("selected")
        if not isinstance(strict_selected_payload, Mapping):
            raise RuntimeError(
                "completed strict validation has no selected configuration"
            )
        strict_selected = _evaluation_from_payload(
            strict_selected_payload,
            communication_importance_ratio,
        )
        if (
                selected is None
                or selected.as_dict() != strict_selected.as_dict()
        ):
            raise RuntimeError(
                "completed selected configuration does not match strict validation"
            )
        if (
                not isinstance(selected_payload, Mapping)
                or selected_payload.get("strict_feasible") is not verdict
                or str(selected_payload.get("selection_status") or "")
                != expected_selection_status
        ):
            raise RuntimeError(
                "completed selected configuration has inconsistent strict status"
            )
        paths["strict_validation"] = strict_path
        candidate_store_path = strict_validation.get("candidate_store")
        if candidate_store_path:
            owned_path = os.fspath(candidate_store_path)
            if not os.path.isabs(owned_path):
                owned_path = os.path.join(output_dir, owned_path)
            if not os.path.isfile(owned_path):
                raise RuntimeError(
                    "completed Stage-2 strict candidate store is missing"
                )
            paths["strict_candidate_store"] = owned_path

    return {
        "result": result,
        "online_best": result.best,
        "selected": selected,
        "strict_validation": strict_validation,
        "artifact_paths": paths,
        "manifest": dict(manifest),
        "strict_feasible": bool(manifest.get("strict_feasible", False)),
    }


def run_layerwise_search_baseline(
        *,
        backend: str,
        layerwise_env: Any,
        robust_reference: Any,
        output_dir: str,
        evaluation_budget: int,
        seed: int,
        initial_design_size: int,
        candidate_pool_size: int,
        population_size: int,
        patience_generations: int,
        mutation_max_coordinates: int,
        rf_n_estimators: int,
        rf_min_samples_leaf: int,
        communication_importance_ratio: float,
        manifest: Mapping[str, Any],
        strict_validator: Callable[[SearchResult], Mapping[str, Any]] | None = None,
        pending_strict_context_writer: Callable[[Mapping[str, Any]], None] | None = None,
        resume: bool = True,
        ) -> dict[str, Any]:
    """Run one non-RL baseline with ordinary crash-recoverable artifacts."""
    normalized_backend = normalize_search_backend(backend)
    run_started_monotonic = time.perf_counter()
    requested_manifest = dict(manifest)
    requested_protocol_hash = str(
        requested_manifest.get("dataset_protocol_hash") or ""
    )
    validate_dataset_protocol_binding(
        requested_manifest,
        expected_hash=requested_protocol_hash,
        artifact="Stage-2 search manifest",
    )
    if requested_manifest.get("search_split") != SEARCH_EVIDENCE_SPLIT:
        raise RuntimeError(
            "Stage-2 search manifest train-probe protocol mismatch; "
            "start a fresh run"
        )
    if normalized_backend == "ppo":
        raise ValueError("run_layerwise_search_baseline requires a non-PPO backend")
    budget = int(evaluation_budget)
    if budget <= 0:
        raise ValueError("search evaluation budget must be positive")

    os.makedirs(output_dir, exist_ok=True)
    observation_path = os.path.join(output_dir, "observations.jsonl")
    manifest_path = os.path.join(output_dir, "manifest.json")
    strict_validation_path = os.path.join(
        output_dir, "strict_validation.json",
    )
    strict_run = strict_validator is not None
    ga_elite_count = min(7, max(1, int(population_size) - 1))
    formal_ga_full_run = bool(
        strict_run and normalized_backend == "coinn_ga"
    )
    ga_generations = STAGE2_FORMAL_GA_GENERATIONS
    ga_maximum_evaluations = int(
        int(population_size)
        + ga_generations * (int(population_size) - int(ga_elite_count))
    )
    ga_stop_on_no_improvement = False
    ga_require_full_generations = formal_ga_full_run
    if strict_run and int(seed) != 42:
        raise ValueError("Stage-2 comparators with strict validation require seed 42")
    if (
            strict_run
            and normalized_backend == "coinn_ga"
            and (
                int(population_size) != 64
                or int(ga_elite_count) != 7
                or int(mutation_max_coordinates) != 4
                or int(patience_generations) != 5
                or budget != STAGE2_FORMAL_GA_EVALUATIONS
            )
    ):
        raise ValueError(
            "strict Stage-2 COINN-GA requires P64/E7, patience 5 as a "
            "diagnostic counter, the 200-generation full-run contract, a "
            "four-layer mutation cap, and the 11,464-inference full-run "
            "contract"
        )
    if (
            strict_run
            and normalized_backend == "bo_rf"
            and (
                budget != 50_000
                or int(initial_design_size) != 64
                or int(candidate_pool_size) != 2_048
                or int(patience_generations) != 2_000
                or int(rf_n_estimators) != 128
                or int(rf_min_samples_leaf) != 2
            )
    ):
        raise ValueError(
            "strict Stage-2 Bayesian RF requires evaluation cap 50,000, "
            "initial design 64, candidate pool 2,048, patience 2,000, "
            "128 trees, and minimum leaf size 2"
        )
    if strict_run and normalized_backend == "greedy":
        expected_greedy_cap = 6 ** int(layerwise_env.horizon)
        if budget != expected_greedy_cap:
            raise ValueError(
                "strict Stage-2 Greedy requires the full action-space safety "
                f"cap {expected_greedy_cap}"
            )

    search_config_payload = {
        "initial_design_size": int(initial_design_size),
        "candidate_pool_size": int(candidate_pool_size),
        "population_size": int(population_size),
        "ga_population_size": int(population_size),
        "ga_elite_count": int(ga_elite_count),
        "ga_generations": int(ga_generations),
        "ga_maximum_evaluations": ga_maximum_evaluations,
        "ga_stop_on_no_improvement": bool(ga_stop_on_no_improvement),
        "ga_require_full_generations": bool(ga_require_full_generations),
        "patience_generations": int(patience_generations),
        "mutation_max_coordinates": int(mutation_max_coordinates),
        "rf_n_estimators": int(rf_n_estimators),
        "rf_min_samples_leaf": int(rf_min_samples_leaf),
    }
    expected_trials = int(
        getattr(layerwise_env.base.env_cfg, "num_trials_per_step", 0)
    )
    resume_contract = to_jsonable({
        "requested_manifest": dict(manifest),
        "num_layers": int(layerwise_env.horizon),
        "constraint_limits": limits_from_reference(robust_reference).as_dict(),
        "trials_per_action": expected_trials,
        "search_backend": normalized_backend,
        "evaluation_budget": budget,
        "seed": int(seed),
        "communication_importance_ratio": float(
            communication_importance_ratio
        ),
        "search_config": search_config_payload,
        "strict_validation_requested": bool(strict_run),
    }, stringify_unknown=True)

    preload: tuple[SearchEvaluation, ...] = ()
    persisted_search_result: SearchResult | None = None
    existing_manifest = (
        read_json_file(manifest_path)
        if os.path.exists(manifest_path)
        else None
    )
    if (
            existing_manifest is None
            and os.path.exists(observation_path)
            and os.path.getsize(observation_path) > 0
    ):
        raise RuntimeError(
            "Stage-2 observations exist without a manifest; use a fresh output "
            "directory or restore the matching manifest"
        )
    if existing_manifest is not None:
        validate_dataset_protocol_binding(
            existing_manifest,
            expected_hash=requested_protocol_hash,
            artifact="Stage-2 resume manifest",
        )
        if not resume:
            raise RuntimeError(
                "search baseline output already exists and resume is disabled: "
                f"{output_dir}"
            )
        existing_status = str(existing_manifest.get("status") or "")
        completed_statuses = {
            "complete_strict_feasible",
            "complete_least_violating",
            "smoke_only_complete",
        }
        resume_contract_matches = (
            existing_manifest.get("resume_contract") == resume_contract
        )
        if not resume_contract_matches:
            raise RuntimeError(
                "search baseline resume contract does not match the existing run"
            )
        if existing_status in completed_statuses:
            return _load_plain_completed_search_run(
                output_dir=output_dir,
                manifest=existing_manifest,
                communication_importance_ratio=float(
                    communication_importance_ratio
                ),
            )
        if existing_status == "search_complete_pending_strict":
            persisted_search_result = _load_persisted_search_result(output_dir)
            _validate_persisted_online_result(
                output_dir=output_dir,
                manifest=existing_manifest,
                result=persisted_search_result,
            )
        elif os.path.exists(observation_path):
            preload = load_search_preload(observation_path)

    strict_candidate_store = None
    strict_candidate_store_checkpoint_size = None
    strict_candidate_store_path = (
        manifest.get("strict_candidate_store") if strict_run else None
    )
    if strict_candidate_store_path:
        owned_store_path = os.fspath(strict_candidate_store_path)
        if not os.path.isabs(owned_store_path):
            owned_store_path = os.path.join(output_dir, owned_store_path)
        strict_candidate_store = CandidateStore(owned_store_path)
        if existing_manifest is not None:
            checkpoint_value = existing_manifest.get(
                "strict_candidate_store_checkpoint_size"
            )
            if checkpoint_value is not None:
                if (
                        isinstance(checkpoint_value, bool)
                        or not isinstance(checkpoint_value, int)
                        or checkpoint_value < 0
                ):
                    raise RuntimeError(
                        "strict candidate-store checkpoint size is invalid"
                    )
                strict_candidate_store_checkpoint_size = int(
                    checkpoint_value
                )

    if pending_strict_context_writer is not None:
        if not strict_run:
            raise ValueError(
                "pending strict context writer requires strict validation"
            )
        pending_strict_context_writer(dict(resume_contract))

    recovered_online_wall_seconds = float(
        0.0
        if existing_manifest is None
        else existing_manifest.get(
            "online_search_wall_seconds",
            existing_manifest.get("search_wall_seconds", 0.0),
        )
    )
    if preload:
        recovered_online_wall_seconds = max(
            recovered_online_wall_seconds,
            max(
                float(item.metadata.get(
                    "search_cumulative_wall_seconds", 0.0,
                ) or 0.0)
                for item in preload
            ),
        )
    run_manifest = {
        **dict(manifest),
        "schema_version": "stage2_layerwise_search_baseline_v3",
        "search_backend": normalized_backend,
        "status": (
            "search_complete_pending_strict"
            if persisted_search_result is not None
            else "running"
        ),
        "scientific_status": (
            "full_search_with_strict_train_probe_gate"
            if strict_run else "smoke_only_no_strict_search_gate"
        ),
        "evaluation_budget": budget,
        "seed": int(seed),
        "communication_importance_ratio": float(
            communication_importance_ratio
        ),
        "search_config": search_config_payload,
        "strict_validation_requested": bool(strict_run),
        "resume_contract": resume_contract,
        "resume_semantics": (
            "deterministic observation replay; completed observations are not "
            "re-inferred"
        ),
        "preloaded_observation_count": len(preload),
        "online_search_wall_seconds": recovered_online_wall_seconds,
        "strict_attempt_count": int(
            0
            if existing_manifest is None
            else existing_manifest.get("strict_attempt_count", 0)
        ),
        "strict_attempt_wall_seconds_total": float(
            0.0
            if existing_manifest is None
            else existing_manifest.get(
                "strict_attempt_wall_seconds_total",
                existing_manifest.get("last_strict_attempt_wall_seconds", 0.0),
            )
        ),
        "started_at": (
            existing_manifest.get("started_at")
            if existing_manifest is not None
            else time.strftime("%Y-%m-%dT%H:%M:%S")
        ),
        "resumed_at": (
            time.strftime("%Y-%m-%dT%H:%M:%S")
            if existing_manifest is not None else None
        ),
    }
    if strict_candidate_store_checkpoint_size is not None:
        run_manifest["strict_candidate_store_checkpoint_size"] = int(
            strict_candidate_store_checkpoint_size
        )
    _atomic_json(manifest_path, run_manifest)

    config = SearchConfig(
        evaluation_budget=budget,
        seed=int(seed),
        initial_design_size=int(initial_design_size),
        candidate_pool_size=int(candidate_pool_size),
        population_size=int(population_size),
        ga_population_size=int(population_size),
        ga_elite_count=int(ga_elite_count),
        ga_generations=int(ga_generations),
        ga_stop_on_no_improvement=bool(ga_stop_on_no_improvement),
        ga_require_full_generations=bool(ga_require_full_generations),
        patience_generations=int(patience_generations),
        mutation_max_coordinates=int(mutation_max_coordinates),
        rf_n_estimators=int(rf_n_estimators),
        rf_min_samples_leaf=int(rf_min_samples_leaf),
        communication_importance_ratio=float(
            communication_importance_ratio
        ),
    )
    if persisted_search_result is None:
        def on_evaluation(row: Mapping[str, Any]) -> None:
            owned = dict(row)
            owned_metadata = dict(owned.get("metadata") or {})
            owned_metadata["search_cumulative_wall_seconds"] = float(
                recovered_online_wall_seconds
                + time.perf_counter() - run_started_monotonic
            )
            owned["metadata"] = owned_metadata
            _write_observation_row(observation_path, owned)

        runtime_evaluator = LayerwiseRuntimeEvaluator(
            env=layerwise_env,
            reference=robust_reference,
            base_seed=int(seed),
            expected_trials=expected_trials,
            on_evaluation=on_evaluation,
        )
        runtime_evaluator.evaluation_count = len(preload)
        result = run_search(
            normalized_backend,
            LayerwiseSearchSpace(int(layerwise_env.horizon)),
            runtime_evaluator,
            config,
            preload=preload,
        )
        paths = persist_search_result(
            output_dir=output_dir,
            result=result,
            manifest=run_manifest,
            observation_rows=None,
        )
        contract_error = None
        if strict_run and normalized_backend == "greedy":
            if result.termination_reason != "verified_local_optima":
                contract_error = (
                    "strict Greedy search stopped before verifying complete "
                    "1-opt and 2-opt neighborhoods"
                )
        elif strict_run and normalized_backend == "coinn_ga":
            try:
                _validate_ga_completion_proof(
                    result,
                    patience_generations=int(patience_generations),
                    generation_cap=int(ga_generations),
                    maximum_evaluations=ga_maximum_evaluations,
                    stop_on_no_improvement=bool(
                        ga_stop_on_no_improvement
                    ),
                    require_full_generations=bool(
                        ga_require_full_generations
                    ),
                )
            except RuntimeError as exc:
                contract_error = str(exc)
        elif (
                strict_run
                and normalized_backend == "bo_rf"
                and result.termination_reason not in {
                    "bo_no_improvement",
                    "evaluation_budget",
                    "candidate_space_exhausted",
                }
        ):
            contract_error = (
                "strict BO-RF stopped outside native convergence or its "
                "evaluation safety cap"
            )
        if contract_error is not None:
            incomplete_manifest = {
                **run_manifest,
                "status": "incomplete_search_contract",
                "evaluation_count": int(result.evaluation_count),
                "observation_count": int(result.observation_count),
                "termination_reason": str(result.termination_reason),
            }
            _atomic_json(manifest_path, incomplete_manifest)
            raise RuntimeError(contract_error)

        online_best_path = os.path.join(output_dir, "online_best.json")
        _atomic_json(online_best_path, result.best.as_dict())
        paths["online_best"] = online_best_path
        online_elapsed = float(
            recovered_online_wall_seconds
            + time.perf_counter() - run_started_monotonic
        )
        run_manifest = {
            **run_manifest,
            "status": "search_complete_pending_strict",
            "search_completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "evaluation_count": int(result.evaluation_count),
            "observation_count": int(result.observation_count),
            "termination_reason": str(result.termination_reason),
            "online_search_wall_seconds": online_elapsed,
            "search_wall_seconds": online_elapsed,
        }
        _atomic_json(manifest_path, run_manifest)
    else:
        result = persisted_search_result
        paths = {
            "manifest": manifest_path,
            "observations": observation_path,
            "history": os.path.join(output_dir, "history.json"),
            "summary": os.path.join(output_dir, "summary.json"),
            "online_best": os.path.join(output_dir, "online_best.json"),
        }

    online_best = result.best
    online_best_path = os.path.join(output_dir, "online_best.json")
    _atomic_json(online_best_path, online_best.as_dict())
    paths["online_best"] = online_best_path

    strict_validation = None
    selected: SearchEvaluation | None = online_best
    selection_status = "online_best_smoke_only"
    strict_feasible = False
    strict_validation_wall_seconds = 0.0
    if strict_run:
        strict_payload_from_artifact = None
        if (
                existing_manifest is not None
                and str(existing_manifest.get("status") or "")
                == "search_complete_pending_strict"
                and os.path.isfile(strict_validation_path)
        ):
            completed_strict_payload = read_json_file(
                strict_validation_path
            )
            if not isinstance(completed_strict_payload, Mapping):
                raise RuntimeError(
                    "completed strict validation artifact is invalid"
                )
            strict_payload_from_artifact = dict(
                completed_strict_payload
            )
        if (
                strict_payload_from_artifact is None
                and strict_candidate_store is not None
        ):
            if strict_candidate_store_checkpoint_size is None:
                store_path = os.fspath(strict_candidate_store.path)
                strict_candidate_store_checkpoint_size = (
                    os.path.getsize(store_path)
                    if os.path.exists(store_path) else 0
                )
                run_manifest = {
                    **run_manifest,
                    "strict_candidate_store_checkpoint_size": int(
                        strict_candidate_store_checkpoint_size
                    ),
                }
                _atomic_json(manifest_path, run_manifest)
            else:
                strict_candidate_store.recover_to_checkpoint_size(
                    strict_candidate_store_checkpoint_size
                )

        strict_started_monotonic = time.perf_counter()
        try:
            strict_payload = (
                strict_payload_from_artifact
                if strict_payload_from_artifact is not None
                else strict_validator(result)
            )
            if not isinstance(strict_payload, Mapping):
                raise RuntimeError(
                    "strict validation must return a mapping"
                )
            strict_validation = dict(strict_payload)
            _validate_strict_validation_payload(
                result=result,
                payload=strict_validation,
                communication_importance_ratio=communication_importance_ratio,
            )
            verdict = strict_validation.get("strict_feasible")
            if type(verdict) is not bool:
                raise RuntimeError(
                    "strict validation must provide a boolean strict_feasible verdict"
                )
            strict_feasible = bool(verdict)
            selected_payload = strict_validation.get("selected")
            if selected_payload is None:
                selected = None
            elif isinstance(selected_payload, Mapping):
                selected = _evaluation_from_payload(
                    selected_payload,
                    communication_importance_ratio,
                )
            else:
                raise RuntimeError(
                    "strict selected candidate must be an object or null"
                )
            selection_status = str(strict_validation.get(
                "selection_status",
                (
                    "strict_feasible"
                    if strict_feasible else "strict_least_violating"
                ),
            ))
            if strict_feasible and selected is None:
                raise RuntimeError(
                    "strict feasible validation must select a candidate"
                )
            if strict_feasible and selection_status != "strict_feasible":
                raise RuntimeError(
                    "strict feasible verdict has an inconsistent selection status"
                )
            if (
                    not strict_feasible
                    and selected is not None
                    and selection_status != "strict_least_violating"
            ):
                raise RuntimeError(
                    "least-violating selection has an inconsistent status"
                )
        except Exception:
            failed_seconds = float(
                time.perf_counter() - strict_started_monotonic
            )
            if (
                    strict_payload_from_artifact is None
                    and strict_candidate_store is not None
                    and strict_candidate_store_checkpoint_size is not None
            ):
                strict_candidate_store.recover_to_checkpoint_size(
                    strict_candidate_store_checkpoint_size
                )
            failed_manifest = {
                **run_manifest,
                "status": "search_complete_pending_strict",
                "strict_attempt_count": int(
                    run_manifest.get("strict_attempt_count", 0)
                ) + 1,
                "last_strict_attempt_wall_seconds": failed_seconds,
                "strict_attempt_wall_seconds_total": float(
                    run_manifest.get(
                        "strict_attempt_wall_seconds_total", 0.0,
                    ) + failed_seconds
                ),
            }
            _atomic_json(manifest_path, failed_manifest)
            raise
        strict_validation_wall_seconds = (
            0.0
            if strict_payload_from_artifact is not None
            else float(time.perf_counter() - strict_started_monotonic)
        )
        if strict_payload_from_artifact is None:
            _atomic_json(strict_validation_path, strict_validation)
        paths["strict_validation"] = strict_validation_path
        candidate_store_path = strict_validation.get("candidate_store")
        if candidate_store_path:
            owned_candidate_store_path = os.fspath(candidate_store_path)
            if not os.path.isabs(owned_candidate_store_path):
                owned_candidate_store_path = os.path.join(
                    output_dir, owned_candidate_store_path,
                )
            if not os.path.isfile(owned_candidate_store_path):
                raise RuntimeError(
                    "strict validation candidate store is missing at completion"
                )
            paths["strict_candidate_store"] = owned_candidate_store_path
        if selected is None:
            failed_manifest = {
                **run_manifest,
                "status": "search_complete_pending_strict",
            }
            _atomic_json(manifest_path, failed_manifest)
            raise RuntimeError(
                "strict validation produced no materializable candidate"
            )

    selected_path = os.path.join(
        output_dir, "final_selected_configuration.json",
    )
    final_selected_payload = None if selected is None else {
        **selected.as_dict(),
        "selection_status": selection_status,
        "strict_feasible": bool(strict_feasible),
        "violations": (
            None
            if strict_validation is None
            else strict_validation.get("selected_violations")
        ),
    }
    _atomic_json(selected_path, final_selected_payload)
    paths["final_selected_configuration"] = selected_path

    strict_records = (
        []
        if strict_validation is None
        else list(strict_validation.get("records") or [])
    )
    strict_evaluated_records = [
        row for row in strict_records if bool(row.get("strict_evaluated", False))
    ]
    online_trial_count = int(sum(
        len(tuple(item.metadata.get("trial_seeds", ()) or ()))
        for item in result.observations
        if item.inference_performed
    ))
    strict_joint_trial_count = int(sum(
        int(row.get("strict_trial_count", 0) or 0)
        for row in strict_evaluated_records
    ))
    strict_compute_trial_count = int(sum(
        int(
            dict(
                dict(row.get("violations") or {}).get("families") or {}
            ).get("compute_only", {}).get("trial_count", 0)
            or 0
        )
        for row in strict_evaluated_records
    ))
    strict_communication_trial_count = int(sum(
        int(
            dict(
                dict(row.get("violations") or {}).get("families") or {}
            ).get("communication_only", {}).get("trial_count", 0)
            or 0
        )
        for row in strict_evaluated_records
    ))
    strict_fresh_trial_count = int(sum(
        int(dict(row.get("promotion") or {}).get("fresh_trial_count", 0) or 0)
        + int(dict(row.get("certification") or {}).get(
            "fresh_trial_count", 0,
        ) or 0)
        for row in strict_evaluated_records
    ))
    online_search_wall_seconds = float(
        run_manifest.get(
            "online_search_wall_seconds",
            run_manifest.get("search_wall_seconds", 0.0),
        )
    )
    strict_attempt_count = int(
        run_manifest.get("strict_attempt_count", 0)
        + (1 if strict_run else 0)
    )
    strict_attempt_wall_seconds_total = float(
        run_manifest.get("strict_attempt_wall_seconds_total", 0.0)
        + strict_validation_wall_seconds
    )
    completed_manifest = {
        **run_manifest,
        "status": (
            "smoke_only_complete"
            if not strict_run
            else (
                "complete_strict_feasible"
                if strict_feasible else "complete_least_violating"
            )
        ),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "evaluation_count": int(result.evaluation_count),
        "inference_reaching_candidate_count": int(result.evaluation_count),
        "model_inference_count": int(result.evaluation_count),
        "model_inference_count_semantics": (
            "legacy_alias_of_inference_reaching_candidate_count"
        ),
        "online_candidate_trial_count": int(online_trial_count),
        "observation_count": int(result.observation_count),
        "non_inference_observation_count": int(
            result.observation_count - result.evaluation_count
        ),
        "online_search_wall_seconds": online_search_wall_seconds,
        "search_wall_seconds": online_search_wall_seconds,
        "strict_attempt_count": strict_attempt_count,
        "last_strict_attempt_wall_seconds": float(
            strict_validation_wall_seconds
        ),
        "strict_attempt_wall_seconds_total": (
            strict_attempt_wall_seconds_total
        ),
        "strict_validation_wall_seconds": (
            strict_attempt_wall_seconds_total
        ),
        "total_wall_seconds": float(
            online_search_wall_seconds + strict_attempt_wall_seconds_total
        ),
        "strict_evaluated_candidate_count": len(strict_evaluated_records),
        "strict_trial_count": strict_joint_trial_count,
        "strict_joint_trial_count": strict_joint_trial_count,
        "strict_compute_trial_count": strict_compute_trial_count,
        "strict_communication_trial_count": (
            strict_communication_trial_count
        ),
        "strict_total_evidence_trial_count": int(
            strict_joint_trial_count
            + strict_compute_trial_count
            + strict_communication_trial_count
        ),
        "strict_fresh_trial_count": strict_fresh_trial_count,
        "total_candidate_trial_count": int(
            online_trial_count
            + strict_joint_trial_count
            + strict_compute_trial_count
            + strict_communication_trial_count
        ),
        "model_forward_trial_count": int(
            online_trial_count
            + strict_joint_trial_count
            + strict_compute_trial_count
            + strict_communication_trial_count
        ),
        "model_forward_trial_count_semantics": (
            "pooled_candidate_evidence_trials_across_online_joint_compute_"
            "communication"
        ),
        "termination_reason": str(result.termination_reason),
        "strict_validation_enabled": bool(strict_run),
        "strict_validation_passed": bool(strict_feasible),
        "strict_feasible": bool(strict_feasible),
        "selection_status": selection_status,
    }
    _atomic_json(paths["manifest"], completed_manifest)
    return {
        "result": result,
        "online_best": online_best,
        "selected": selected,
        "strict_validation": strict_validation,
        "artifact_paths": paths,
        "manifest": completed_manifest,
        "strict_feasible": bool(strict_feasible),
    }
