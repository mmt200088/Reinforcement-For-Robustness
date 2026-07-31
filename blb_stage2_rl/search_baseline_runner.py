"""Real-model adapter and evidence writer for Stage-2 search baselines."""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from json_utils import to_jsonable

from .layerwise_action import describe_layerwise_action_matrix
from .search_baselines import (
    ActionMatrix,
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
        try:
            state = self.env.reset(seed=self.base_seed + evaluation_index)
            del state
            base_env.probe_noise_seed = derive_layerwise_episode_probe_seed(
                self.base_seed,
                evaluation_index,
                trial_count=self.expected_trials,
            )
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
            if bool(runtime_info.get("invalid", False)):
                raise RuntimeError(
                    "real Stage-2 evaluation produced an invalid configuration"
                )
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
                "forward_ran": True,
                "model_uses_replan_config": True,
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
    ranked = [
        candidate
        for candidate in sorted(
            result.observations,
            key=candidate_rank_key,
            reverse=True,
        )
        if candidate.feasible
    ][:max(1, int(top_n))]
    records: list[dict[str, Any]] = []
    strict_passes: list[SearchEvaluation] = []
    passing_resource: Optional[tuple[float, float]] = None
    for online in ranked:
        resource = online.resource
        resource_key = (
            float(resource.ppo_resource_score),
            float(resource.robust_floor),
        )
        if passing_resource is not None and (
                resource_key[0] < passing_resource[0] - 1.0e-12
                or (
                    math.isclose(
                        resource_key[0], passing_resource[0],
                        rel_tol=0.0, abs_tol=1.0e-12,
                    )
                    and resource_key[1] < passing_resource[1] - 1.0e-12
                )
        ):
            break
        metadata = dict(online.metadata)
        full_vector = tuple(
            int(value)
            for value in metadata.get("pending_full_vector", ())
        )
        if not full_vector:
            raise RuntimeError(
                "strict validation candidate has no materialized full vector"
            )
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
            "promotion": promotion_record,
            "certification": None,
            "strict_point_pass": False,
        }
        records.append(record)
        if promotion_record["status"] not in (
                "promoted", "already_promoted",
        ):
            continue
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
        passed = certification_record["status"] in (
            _FINAL_REVALIDATION_PASSED,
            "already_final_certified",
        )
        record["strict_point_pass"] = bool(passed)
        if not passed:
            continue
        strict_metrics = SearchMetrics(**dict(certification.metrics))
        strict_assessment = to_jsonable(
            certification.assessment, stringify_unknown=True,
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
                "strict_validation_status": certification_record["status"],
                "strict_final_assessment": strict_assessment,
                "strict_axis_counterfactuals": certification_record[
                    "axis_counterfactuals"
                ],
                "strict_candidate_store": os.fspath(candidate_store.path),
            },
        )
        if not strict_evaluation.feasible:
            raise RuntimeError(
                "canonical final certification passed but pooled six-point "
                "limits do not pass"
            )
        strict_passes.append(strict_evaluation)
        passing_resource = resource_key
    selected = (
        None
        if not strict_passes
        else max(strict_passes, key=_strict_selected_rank)
    )
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
        "selected": None if selected is None else selected.as_dict(),
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
        ) -> dict[str, Any]:
    """Run one non-RL baseline and persist crash-recoverable observations."""
    normalized_backend = normalize_search_backend(backend)
    if normalized_backend == "ppo":
        raise ValueError("run_layerwise_search_baseline requires a non-PPO backend")
    budget = int(evaluation_budget)
    if budget <= 0:
        raise ValueError("search evaluation budget must be positive")
    os.makedirs(output_dir, exist_ok=True)
    observation_path = os.path.join(output_dir, "observations.jsonl")
    if os.path.exists(observation_path) and os.path.getsize(observation_path):
        raise RuntimeError(
            "search baseline output already contains observations; use a fresh "
            f"output directory: {output_dir}"
        )
    run_manifest = {
        **dict(manifest),
        "schema_version": "stage2_layerwise_search_baseline_v1",
        "search_backend": normalized_backend,
        "status": "running",
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
        "search_config": {
            "initial_design_size": int(initial_design_size),
            "candidate_pool_size": int(candidate_pool_size),
            "population_size": int(population_size),
            "patience_generations": int(patience_generations),
            "mutation_max_coordinates": int(mutation_max_coordinates),
            "rf_n_estimators": int(rf_n_estimators),
            "rf_min_samples_leaf": int(rf_min_samples_leaf),
        },
        "strict_validation_requested": bool(strict_validator is not None),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _atomic_json(os.path.join(output_dir, "manifest.json"), run_manifest)
    observation_rows: list[Mapping[str, Any]] = []

    def on_evaluation(row: Mapping[str, Any]) -> None:
        owned = dict(row)
        observation_rows.append(owned)
        _write_observation_row(observation_path, owned)

    runtime_evaluator = LayerwiseRuntimeEvaluator(
        env=layerwise_env,
        reference=robust_reference,
        base_seed=int(seed),
        expected_trials=int(
            getattr(layerwise_env.base.env_cfg, "num_trials_per_step", 0)
        ),
        on_evaluation=on_evaluation,
    )
    config = SearchConfig(
        evaluation_budget=budget,
        seed=int(seed),
        initial_design_size=int(initial_design_size),
        candidate_pool_size=int(candidate_pool_size),
        population_size=int(population_size),
        patience_generations=int(patience_generations),
        mutation_max_coordinates=int(mutation_max_coordinates),
        rf_n_estimators=int(rf_n_estimators),
        rf_min_samples_leaf=int(rf_min_samples_leaf),
        communication_importance_ratio=float(
            communication_importance_ratio
        ),
    )
    result = run_search(
        normalized_backend,
        LayerwiseSearchSpace(int(layerwise_env.horizon)),
        runtime_evaluator,
        config,
    )
    paths = persist_search_result(
        output_dir=output_dir,
        result=result,
        manifest=run_manifest,
        observation_rows=observation_rows,
    )
    strict_validation = None
    selected = result.best
    if strict_validator is not None:
        strict_validation = dict(strict_validator(result))
        strict_path = os.path.join(output_dir, "strict_validation.json")
        _atomic_json(strict_path, strict_validation)
        paths["strict_validation"] = strict_path
        if strict_validation["selected"] is not None:
            selected_payload = strict_validation["selected"]
            selected_probability_payload = dict(
                selected_payload.get("constraint_probabilities") or {}
            )
            selected = SearchEvaluation(
                action_matrix=tuple(
                    tuple(int(value) for value in row)
                    for row in selected_payload["action_matrix"]
                ),
                metrics=SearchMetrics(**selected_payload["metrics"]),
                limits=ConstraintLimits(
                    loss_max=float(selected_payload["limits"]["loss_max"]),
                    metric1_min=float(
                        selected_payload["limits"]["metric1_min"]
                    ),
                    metric2_min=float(
                        selected_payload["limits"]["metric2_min"]
                    ),
                    loss_std_max=float(
                        selected_payload["limits"]["loss_std_max"]
                    ),
                    metric1_std_max=float(
                        selected_payload["limits"]["metric1_std_max"]
                    ),
                    metric2_std_max=float(
                        selected_payload["limits"]["metric2_std_max"]
                    ),
                ),
                valid=bool(selected_payload["valid"]),
                reward=selected_payload.get("reward"),
                communication_importance_ratio=float(
                    communication_importance_ratio
                ),
                constraint_probabilities=(
                    tuple(
                        float(selected_probability_payload[name])
                        for name in CONSTRAINT_PROBABILITY_NAMES
                    )
                    if selected_probability_payload else ()
                ),
                gate_probability=(
                    selected_payload.get("gate_probability")
                    if selected_probability_payload else None
                ),
                metadata=selected_payload.get("metadata") or {},
            )
    strict_validation_passed = bool(
        strict_validation is not None
        and strict_validation.get("selected") is not None
    )
    completed_manifest = {
        **run_manifest,
        "status": (
            "smoke_only_complete"
            if strict_validator is None
            else (
                "complete"
                if strict_validation_passed
                else "complete_no_strict_feasible"
            )
        ),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "evaluation_count": int(result.evaluation_count),
        "termination_reason": str(result.termination_reason),
        "strict_validation_enabled": bool(strict_validator is not None),
        "strict_validation_passed": strict_validation_passed,
    }
    _atomic_json(paths["manifest"], completed_manifest)
    if strict_validator is not None and not strict_validation_passed:
        raise RuntimeError(
            "no search candidate passed the validation_full joint and "
            "resource-axis gates; online search and strict-validation evidence "
            f"were preserved under {output_dir}"
        )
    return {
        "result": result,
        "selected": selected,
        "strict_validation": strict_validation,
        "artifact_paths": paths,
        "manifest": completed_manifest,
        "scientific_export_allowed": bool(
            strict_validator is not None and strict_validation_passed
        ),
    }
