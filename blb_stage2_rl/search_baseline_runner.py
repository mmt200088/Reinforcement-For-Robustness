"""Real-model adapter and evidence writer for Stage-2 search baselines."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from json_utils import read_json_file, to_jsonable
from jsonl_utils import read_jsonl, recover_jsonl_file

from .candidate_store import action_hash
from .layerwise_action import describe_layerwise_action_matrix
from .search_baselines import (
    ActionMatrix,
    CONSTRAINT_NAMES,
    CONSTRAINT_PROBABILITY_NAMES,
    ConstraintLimits,
    LayerwiseSearchSpace,
    SearchEvaluation,
    SearchConfig,
    SearchMetrics,
    SearchResult,
    candidate_rank_key,
    normalize_search_backend,
    run_search,
)
from .seed_utils import derive_layerwise_episode_probe_seed


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


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
    os.makedirs(os.path.dirname(path), exist_ok=True)
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
    os.replace(temp_path, path)


def _action_seed_key(action_matrix: ActionMatrix) -> int:
    return int(action_hash(action_matrix)[:16], 16)


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
            on_evaluation: Optional[Callable[[Mapping[str, Any]], None]] = None,
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
        action_seed_key = _action_seed_key(action_matrix)
        action_base_seed = int(self.base_seed) ^ int(action_seed_key)
        probe_seed = derive_layerwise_episode_probe_seed(
            action_base_seed,
            0,
            trial_count=self.expected_trials,
        )
        try:
            state = self.env.reset(seed=action_base_seed & 0x7FFFFFFFFFFFFFFF)
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
                    "action_seed_key": int(action_seed_key),
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
            if not bool(runtime_info.get("forward_ran", False)):
                raise RuntimeError(
                    "real Stage-2 evaluation did not execute model forward"
                )
            replan = runtime_info.get("replan_application")
            if not (
                    isinstance(replan, Mapping)
                    and bool(replan.get("model_uses_replan_config", False))
            ):
                raise RuntimeError(
                    "real Stage-2 evaluation did not install the replan "
                    "configuration into the model"
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
            if len(trial_seeds) != self.expected_trials:
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
                "action_seed_key": int(action_seed_key),
                "probe_seed": int(probe_seed),
                "trial_seeds": [int(value) for value in trial_seeds],
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
        observation_rows: Sequence[Mapping[str, Any]],
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


def load_search_preload(path: str) -> tuple[SearchEvaluation, ...]:
    """Recover and deserialize crash-safe Stage-2 observation rows."""

    recover_jsonl_file(path)
    ordered: list[SearchEvaluation] = []
    by_action: dict[ActionMatrix, SearchEvaluation] = {}
    for row in read_jsonl(
            path,
            errors="raise",
            dict_only=True,
            missing_ok=False,
    ):
        evaluation = SearchEvaluation.from_dict(row)
        previous = by_action.get(evaluation.action_matrix)
        if previous is not None:
            if previous.as_dict() != evaluation.as_dict():
                raise ValueError(
                    "conflicting Stage-2 observations for one action"
                )
            continue
        by_action[evaluation.action_matrix] = evaluation
        ordered.append(evaluation)
    return tuple(ordered)


def _load_persisted_search_result(output_dir: str) -> SearchResult:
    observations = load_search_preload(os.path.join(
        output_dir, "observations.jsonl",
    ))
    return SearchResult.from_dict(
        read_json_file(os.path.join(output_dir, "summary.json")),
        observations=observations,
    )


def _completed_search_run(
        *,
        output_dir: str,
        manifest: Mapping[str, Any],
        communication_importance_ratio: float,
        ) -> dict[str, Any]:
    observations_path = os.path.join(output_dir, "observations.jsonl")
    summary_path = os.path.join(output_dir, "summary.json")
    strict_path = os.path.join(output_dir, "strict_validation.json")
    selected_path = os.path.join(
        output_dir, "final_selected_configuration.json",
    )
    result = _load_persisted_search_result(output_dir)
    strict_validation = (
        read_json_file(strict_path)
        if os.path.exists(strict_path)
        else None
    )
    selected_payload = read_json_file(selected_path)
    selected = (
        None
        if selected_payload is None
        else _evaluation_from_payload(
            selected_payload,
            communication_importance_ratio,
        )
    )
    paths = {
        "manifest": os.path.join(output_dir, "manifest.json"),
        "observations": observations_path,
        "history": os.path.join(output_dir, "history.json"),
        "summary": summary_path,
        "online_best": os.path.join(output_dir, "online_best.json"),
        "final_selected_configuration": selected_path,
    }
    if strict_validation is not None:
        paths["strict_validation"] = strict_path
    return {
        "result": result,
        "online_best": result.best,
        "selected": selected,
        "strict_validation": strict_validation,
        "artifact_paths": paths,
        "manifest": dict(manifest),
        "scientific_export_allowed": bool(
            manifest.get("scientific_export_allowed", False)
        ),
        "resumed_completed_run": True,
    }


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


def _strict_selected_rank(evaluation: SearchEvaluation) -> tuple[float, ...]:
    assessment = dict(
        evaluation.metadata.get("strict_final_assessment") or {}
    )
    confidence = tuple(sorted(
        float(assessment.get(name, 0.0))
        for name in CONSTRAINT_PROBABILITY_NAMES
    ))
    point_margins = tuple(sorted(evaluation.normalized_margins))
    resource = evaluation.resource
    lexicographic = tuple(
        -float(value)
        for value in LayerwiseSearchSpace(
            len(evaluation.action_matrix)
        ).flatten(evaluation.action_matrix)
    )
    return (
        float(resource.ppo_resource_score),
        float(resource.robust_floor),
        *confidence,
        *point_margins,
        *lexicographic,
    )


def _strict_metrics(value: Any) -> Optional[SearchMetrics]:
    if value is None:
        return None
    try:
        return SearchMetrics(**{
            name: float(_field(value, name))
            for name in CONSTRAINT_NAMES
        })
    except (TypeError, ValueError):
        return None


def _constraint_family_violations(
        evaluation: SearchEvaluation,
        *,
        status: str,
        trial_count: int,
        banks_run: Sequence[str],
        not_run_banks: Sequence[str],
        point_pass: Optional[bool],
        ) -> dict[str, Any]:
    constraints = {}
    violated = []
    violations = []
    for name, raw_margin, normalized_margin in zip(
            CONSTRAINT_NAMES,
            evaluation.raw_margins,
            evaluation.normalized_margins,
    ):
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
    return {"families": families, "aggregate": aggregate}


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
        float(resource.ppo_resource_score),
        float(resource.robust_floor),
        *lexicographic,
    )


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
    probability_payload = dict(payload.get("constraint_probabilities") or {})
    return SearchEvaluation(
        action_matrix=tuple(
            tuple(int(value) for value in row)
            for row in payload["action_matrix"]
        ),
        metrics=SearchMetrics(**payload["metrics"]),
        limits=ConstraintLimits(
            loss_max=float(payload["limits"]["loss_max"]),
            metric1_min=float(payload["limits"]["metric1_min"]),
            metric2_min=float(payload["limits"]["metric2_min"]),
            loss_std_max=float(payload["limits"]["loss_std_max"]),
            metric1_std_max=float(payload["limits"]["metric1_std_max"]),
            metric2_std_max=float(payload["limits"]["metric2_std_max"]),
        ),
        valid=bool(payload["valid"]),
        reward=payload.get("reward"),
        communication_importance_ratio=float(communication_importance_ratio),
        constraint_probabilities=(
            tuple(
                float(probability_payload[name])
                for name in CONSTRAINT_PROBABILITY_NAMES
            )
            if probability_payload else ()
        ),
        gate_probability=(
            payload.get("gate_probability")
            if probability_payload else None
        ),
        metadata=payload.get("metadata") or {},
    )


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
    from .layerwise_runner import (
        _FINAL_REVALIDATION_PASSED,
        _deserialize_boosted_overrides,
        certify_candidate_with_bank_c,
        promote_candidate_if_eligible,
    )

    final_reference = validation_banks.final_reference
    ranked: list[SearchEvaluation] = []
    seen_actions: set[ActionMatrix] = set()
    for candidate in sorted(
            result.observations,
            key=candidate_rank_key,
            reverse=True,
    ):
        if candidate.action_matrix in seen_actions:
            continue
        seen_actions.add(candidate.action_matrix)
        ranked.append(candidate)
        if len(ranked) >= max(1, int(top_n)):
            break
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
                "formal_feasible": False,
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
        record = {
            "online_candidate": online.as_dict(),
            "materializable": True,
            "strict_evaluated": False,
            "promotion": promotion_record,
            "certification": None,
            "strict_point_pass": False,
            "formal_feasible": False,
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
            record["certification"] = certification_record

        strict_metrics = _strict_metrics(
            None if certification is None else certification.metrics
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
            metrics_source = "promotion"
            strict_trial_count = int(promotion_record["trial_count"])
            strict_assessment = promotion.assessment
        if strict_metrics is None or strict_trial_count <= 0:
            record["skip_reason"] = "no_strict_pooled_metrics"
            continue

        passed = bool(
            certification_record is not None
            and certification_record["status"] in (
                _FINAL_REVALIDATION_PASSED,
                "already_final_certified",
            )
        )
        strict_evaluation = SearchEvaluation(
            action_matrix=online.action_matrix,
            metrics=strict_metrics,
            limits=limits_from_reference(final_reference),
            valid=True,
            reward=online.reward,
            communication_importance_ratio=float(
                communication_importance_ratio
            ),
            metadata={
                **metadata,
                "strict_validation_status": strict_status,
                "strict_metrics_source": metrics_source,
                "strict_trial_count": int(strict_trial_count),
                "strict_final_assessment": to_jsonable(
                    strict_assessment, stringify_unknown=True,
                ),
                "strict_axis_counterfactuals": strict_axis_counterfactuals,
                "strict_axis_metrics_source": strict_axis_metrics_source,
                "strict_candidate_store": os.fspath(candidate_store.path),
            },
        )
        formal_feasible = bool(passed and strict_evaluation.feasible)
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
        strict_evaluation.metadata["strict_violations"] = violations
        record.update({
            "strict_evaluated": True,
            "selection_eligible": True,
            "strict_trial_count": int(strict_trial_count),
            "strict_metrics_source": metrics_source,
            "strict_point_pass": bool(passed),
            "formal_feasible": formal_feasible,
            "violations": violations,
            "strict_evaluation": strict_evaluation.as_dict(),
        })
        strict_evaluations.append((strict_evaluation, formal_feasible))
    strict_passes = [
        evaluation
        for evaluation, formal_feasible in strict_evaluations
        if formal_feasible
    ]
    if strict_passes:
        selected = max(strict_passes, key=_strict_selected_rank)
        selection_status = "strict_feasible"
        formal_feasible = True
    elif strict_evaluations:
        selected = max(
            (evaluation for evaluation, _formal in strict_evaluations),
            key=_strict_fallback_rank,
        )
        selection_status = "strict_least_violating"
        formal_feasible = False
    else:
        selected = None
        selection_status = "no_strict_evaluated_materializable_candidate"
        formal_feasible = False
    selected_violations = (
        None
        if selected is None
        else dict(selected.metadata.get("strict_violations") or {})
    )
    selected_payload = None if selected is None else {
        **selected.as_dict(),
        "selection_status": selection_status,
        "formal_feasible": bool(formal_feasible),
        "violations": selected_violations,
    }
    return {
        "schema_version": "stage2_search_strict_validation_v1",
        "split": "validation_full",
        "validation_banks": validation_banks.contract_payload(),
        "joint_and_axis_counterfactual_gate": True,
        "hard_gate": (
            "joint_six_point_plus_compute_and_communication_"
            "counterfactual_six_point_v1"
        ),
        "bootstrap_probability_role": "diagnostic_tiebreak_only",
        "candidate_store": os.fspath(candidate_store.path),
        "online_best": result.best.as_dict(),
        "selection_status": selection_status,
        "formal_feasible": bool(formal_feasible),
        "selected_violations": selected_violations,
        "selected": selected_payload,
        "records": records,
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
        strict_validator: Optional[
            Callable[[SearchResult], Mapping[str, Any]]
        ] = None,
        resume: bool = True,
        ) -> dict[str, Any]:
    """Run one non-RL baseline and persist crash-recoverable observations."""
    normalized_backend = normalize_search_backend(backend)
    run_started_monotonic = time.perf_counter()
    if normalized_backend == "ppo":
        raise ValueError("run_layerwise_search_baseline requires a non-PPO backend")
    budget = int(evaluation_budget)
    if budget <= 0:
        raise ValueError("search evaluation budget must be positive")
    os.makedirs(output_dir, exist_ok=True)
    observation_path = os.path.join(output_dir, "observations.jsonl")
    manifest_path = os.path.join(output_dir, "manifest.json")
    ga_elite_count = min(7, max(1, int(population_size) - 1))
    ga_expected_evaluations = int(
        int(population_size)
        + 800 * (int(population_size) - int(ga_elite_count))
    )
    if normalized_backend == "coinn_ga" and budget < ga_expected_evaluations:
        raise ValueError(
            "Stage-2 COINN-GA requires enough inference budget for exactly "
            f"800 update generations: {budget} < {ga_expected_evaluations}"
        )
    search_config_payload = {
        "initial_design_size": int(initial_design_size),
        "candidate_pool_size": int(candidate_pool_size),
        "population_size": int(population_size),
        "ga_population_size": int(population_size),
        "ga_elite_count": int(ga_elite_count),
        "ga_generations": 800,
        "ga_expected_evaluations": ga_expected_evaluations,
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
        "strict_validation_requested": bool(strict_validator is not None),
    }, stringify_unknown=True)
    preload: tuple[SearchEvaluation, ...] = ()
    persisted_search_result: Optional[SearchResult] = None
    existing_manifest = (
        read_json_file(manifest_path)
        if os.path.exists(manifest_path)
        else None
    )
    if existing_manifest is not None:
        if not resume:
            raise RuntimeError(
                "search baseline output already exists and resume is disabled: "
                f"{output_dir}"
            )
        if existing_manifest.get("resume_contract") != resume_contract:
            raise RuntimeError(
                "search baseline resume contract does not match the existing run"
            )
        completed_statuses = {
            "complete",
            "complete_no_strict_feasible",
            "smoke_only_complete",
        }
        existing_status = str(existing_manifest.get("status"))
        if existing_status in completed_statuses:
            return _completed_search_run(
                output_dir=output_dir,
                manifest=existing_manifest,
                communication_importance_ratio=float(
                    communication_importance_ratio
                ),
            )
        if existing_status == "search_complete_pending_strict":
            persisted_search_result = _load_persisted_search_result(output_dir)
        elif os.path.exists(observation_path):
            preload = load_search_preload(observation_path)
            if normalized_backend in ("bo_rf", "coinn_ga") and preload:
                raise RuntimeError(
                    f"partial {normalized_backend} resume is disabled because "
                    "exact surrogate/population state is not available; "
                    "restart fresh"
                )

    run_manifest = {
        **dict(manifest),
        "schema_version": "stage2_layerwise_search_baseline_v2",
        "search_backend": normalized_backend,
        "status": (
            "search_complete_pending_strict"
            if persisted_search_result is not None
            else "running"
        ),
        "scientific_status": (
            "full_search_with_validation_full_gate"
            if strict_validator is not None
            else "smoke_only_no_validation_full_gate"
        ),
        "evaluation_budget": budget,
        "seed": int(seed),
        "communication_importance_ratio": float(
            communication_importance_ratio
        ),
        "search_config": search_config_payload,
        "strict_validation_requested": bool(strict_validator is not None),
        "resume_contract": resume_contract,
        "resume_semantics": (
            "deterministic observation replay; completed observations are not "
            "re-inferred"
        ),
        "preloaded_observation_count": len(preload),
        "started_at": (
            existing_manifest.get("started_at")
            if existing_manifest is not None
            else time.strftime("%Y-%m-%dT%H:%M:%S")
        ),
        "resumed_at": (
            time.strftime("%Y-%m-%dT%H:%M:%S")
            if existing_manifest is not None
            else None
        ),
    }
    _atomic_json(manifest_path, run_manifest)
    config = SearchConfig(
        evaluation_budget=budget,
        seed=int(seed),
        initial_design_size=int(initial_design_size),
        candidate_pool_size=int(candidate_pool_size),
        population_size=int(population_size),
        ga_population_size=int(population_size),
        ga_elite_count=int(ga_elite_count),
        ga_generations=800,
        patience_generations=int(patience_generations),
        mutation_max_coordinates=int(mutation_max_coordinates),
        rf_n_estimators=int(rf_n_estimators),
        rf_min_samples_leaf=int(rf_min_samples_leaf),
        communication_importance_ratio=float(
            communication_importance_ratio
        ),
    )
    if persisted_search_result is None:
        observation_rows: list[Mapping[str, Any]] = [
            item.as_dict() for item in preload
        ]

        def on_evaluation(row: Mapping[str, Any]) -> None:
            owned = dict(row)
            observation_rows.append(owned)
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
            observation_rows=observation_rows,
        )
        if (
                normalized_backend == "greedy"
                and strict_validator is not None
                and result.termination_reason != "verified_local_optima"
        ):
            incomplete_manifest = {
                **run_manifest,
                "status": "incomplete_unverified_local_search",
                "evaluation_count": int(result.evaluation_count),
                "observation_count": int(result.observation_count),
                "termination_reason": str(result.termination_reason),
            }
            _atomic_json(manifest_path, incomplete_manifest)
            raise RuntimeError(
                "formal Greedy search exhausted its guard before verifying "
                "complete 1-opt and 2-opt neighborhoods"
            )
        run_manifest = {
            **run_manifest,
            "status": "search_complete_pending_strict",
            "search_completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "evaluation_count": int(result.evaluation_count),
            "observation_count": int(result.observation_count),
            "termination_reason": str(result.termination_reason),
            "search_wall_seconds": float(
                time.perf_counter() - run_started_monotonic
            ),
        }
        _atomic_json(manifest_path, run_manifest)
    else:
        result = persisted_search_result
        paths = {
            "manifest": manifest_path,
            "observations": observation_path,
            "history": os.path.join(output_dir, "history.json"),
            "summary": os.path.join(output_dir, "summary.json"),
        }
    online_best = result.best
    online_best_path = os.path.join(output_dir, "online_best.json")
    _atomic_json(online_best_path, online_best.as_dict())
    paths["online_best"] = online_best_path

    strict_validation = None
    selected: Optional[SearchEvaluation] = online_best
    selection_status = "online_best_smoke_only"
    strict_validation_passed = False
    if strict_validator is not None:
        strict_validation = dict(strict_validator(result))
        strict_path = os.path.join(output_dir, "strict_validation.json")
        _atomic_json(strict_path, strict_validation)
        paths["strict_validation"] = strict_path
        selected_payload = strict_validation.get("selected")
        if selected_payload is None:
            selected = None
            selection_status = str(strict_validation.get(
                "selection_status",
                "no_strict_evaluated_materializable_candidate",
            ))
        else:
            selected = _evaluation_from_payload(
                selected_payload,
                communication_importance_ratio,
            )
            selection_status = str(strict_validation.get(
                "selection_status",
                (
                    "strict_feasible"
                    if bool(selected_payload.get("feasible", False))
                    else "strict_least_violating"
                ),
            ))
        strict_validation_passed = bool(strict_validation.get(
            "formal_feasible",
            selected_payload is not None
            and bool(selected_payload.get("feasible", False)),
        ))

    selected_path = os.path.join(
        output_dir, "final_selected_configuration.json",
    )
    final_selected_payload = None if selected is None else {
        **selected.as_dict(),
        "selection_status": selection_status,
        "formal_feasible": bool(strict_validation_passed),
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
    completed_manifest = {
        **run_manifest,
        "status": (
            "smoke_only_complete"
            if strict_validator is None
            else (
                "complete"
                if strict_validation_passed
                else (
                    "complete_no_strict_feasible"
                    if selected is not None
                    else "failed_no_strict_materializable_candidate"
                )
            )
        ),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "evaluation_count": int(result.evaluation_count),
        "model_inference_count": int(result.evaluation_count),
        "observation_count": int(result.observation_count),
        "non_inference_observation_count": int(
            result.observation_count - result.evaluation_count
        ),
        "search_wall_seconds": float(
            time.perf_counter() - run_started_monotonic
            + (
                float(existing_manifest.get("search_wall_seconds", 0.0))
                if persisted_search_result is not None
                and existing_manifest is not None
                else 0.0
            )
        ),
        "strict_evaluated_candidate_count": len(strict_evaluated_records),
        "strict_trial_count": int(sum(
            int(row.get("strict_trial_count", 0) or 0)
            for row in strict_evaluated_records
        )),
        "termination_reason": str(result.termination_reason),
        "strict_validation_enabled": bool(strict_validator is not None),
        "strict_validation_passed": strict_validation_passed,
        "selection_status": selection_status,
        "formal_feasible": bool(strict_validation_passed),
        "scientific_export_allowed": bool(
            strict_validator is not None and strict_validation_passed
        ),
    }
    _atomic_json(paths["manifest"], completed_manifest)
    if strict_validator is not None and selected is None:
        raise RuntimeError(
            "strict validation produced no evaluated materializable candidate; "
            "the run failed closed after preserving all evidence under "
            f"{output_dir}"
        )
    return {
        "result": result,
        "online_best": online_best,
        "selected": selected,
        "strict_validation": strict_validation,
        "artifact_paths": paths,
        "manifest": completed_manifest,
        "scientific_export_allowed": bool(
            strict_validator is not None and strict_validation_passed
        ),
    }
