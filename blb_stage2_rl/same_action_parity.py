"""Exact fixed-action parity gate for PPO and Stage-2 comparators."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from json_utils import stable_json_hash, to_jsonable

from .layerwise_action import describe_layerwise_action_matrix
from .layerwise_runner import _collect_layerwise_episode
from .search_baseline_runner import (
    LayerwiseRuntimeEvaluator,
    _serialize_boosted_overrides,
    limits_from_reference,
)
from .search_baselines import (
    CONSTRAINT_PROBABILITY_NAMES,
    ActionMatrix,
    SearchEvaluation,
    SearchResult,
)
from .seed_utils import derive_layerwise_online_evaluation_seeds
from .sequential_policy import step_to_mask_and_levels


_METRIC_FIELDS = (
    "loss_mean",
    "metric1_mean",
    "metric2_mean",
    "loss_std",
    "metric1_std",
    "metric2_std",
)
_TRIAL_FIELDS = (
    ("loss", "loss_trials"),
    ("metric1", "metric1_trials"),
    ("metric2", "metric2_trials"),
)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _canonical_action(
        action_matrix: Sequence[Sequence[int]],
        *,
        horizon: int,
        ) -> ActionMatrix:
    action = tuple(
        tuple(int(value) for value in row)
        for row in action_matrix
    )
    if len(action) != int(horizon):
        raise ValueError(
            "fixed action row count does not match the layerwise horizon: "
            f"{len(action)} != {int(horizon)}"
        )
    if any(len(row) != 2 for row in action):
        raise ValueError("each fixed Stage-2 action row must have two slots")
    return action


class _FixedActionPolicy:
    def __init__(self, action_matrix: ActionMatrix):
        self._rows = iter(action_matrix)
        self.consumed = 0

    def sample_action(self, *_args: Any, **_kwargs: Any) -> tuple[np.ndarray, ...]:
        try:
            row = np.asarray(next(self._rows), dtype=np.int64)
        except StopIteration as exc:
            raise RuntimeError("fixed action policy was sampled past its horizon") from exc
        self.consumed += 1
        return (
            row[None, :],
            np.asarray([0.0], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
            np.zeros((1, row.size), dtype=np.float32),
        )


class _RecordingBuffer:
    def __init__(self) -> None:
        self.rows: list[Mapping[str, Any]] = []

    def add(self, **payload: Any) -> int:
        self.rows.append(payload)
        return len(self.rows) - 1


def _metrics_projection(value: Any) -> dict[str, float]:
    return {
        name: float(_field(value, name))
        for name in _METRIC_FIELDS
    }


def _trial_projection(
        value: Any,
        *,
        expected_trials: int,
        ) -> tuple[list[int], dict[str, list[float]]]:
    seeds = [int(item) for item in (_field(value, "trial_seeds", ()) or ())]
    trials = {
        name: [
            float(item)
            for item in (_field(value, field_name, ()) or ())
        ]
        for name, field_name in _TRIAL_FIELDS
    }
    if len(seeds) != int(expected_trials) or any(
            len(items) != int(expected_trials)
            for items in trials.values()
            ):
        raise RuntimeError(
            "same-action parity received an unexpected trial count"
        )
    return seeds, trials


def _probability_projection(value: Mapping[str, Any]) -> dict[str, float]:
    try:
        return {
            name: float(value[name])
            for name in CONSTRAINT_PROBABILITY_NAMES
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "same-action parity is missing a constraint probability"
        ) from exc


def _pending_vector(layerwise_env: Any) -> list[int]:
    value = getattr(layerwise_env, "pending_full_vector", ())
    if callable(value):
        value = value()
    vector = [
        int(item)
        for item in np.asarray(value, dtype=np.int64).reshape(-1)
    ]
    if not vector:
        raise RuntimeError("same-action parity materialized no full vector")
    return vector


def _ppo_projection(
        *,
        action_matrix: ActionMatrix,
        runtime_info: Mapping[str, Any],
        boosted_overrides: Mapping[Any, Any],
        pending_full_vector: Sequence[int],
        reset_seed: int,
        probe_seed: int,
        expected_trials: int,
        episode_reward: float,
        robust_reference: Any,
        ) -> dict[str, Any]:
    metrics = runtime_info.get("metrics")
    if metrics is None:
        raise RuntimeError("PPO same-action route returned no runtime metrics")
    trial_seeds, trial_results = _trial_projection(
        metrics,
        expected_trials=expected_trials,
    )
    assessment = runtime_info.get("statistical_assessment")
    if not isinstance(assessment, Mapping):
        raise RuntimeError("PPO same-action route returned no assessment")
    replan = runtime_info.get("replan_application")
    if not isinstance(replan, Mapping):
        raise RuntimeError("PPO same-action route returned no replan evidence")
    return {
        "action_matrix": [list(row) for row in action_matrix],
        "reset_seed": int(reset_seed),
        "probe_seed": int(probe_seed),
        "trial_seeds": trial_seeds,
        "trial_results": trial_results,
        "metrics": _metrics_projection(metrics),
        "limits": limits_from_reference(robust_reference).as_dict(),
        "constraint_probabilities": _probability_projection(assessment),
        "gate_probability": float(assessment["gate_probability"]),
        "bootstrap_seed": int(assessment["bootstrap_seed"]),
        "reward": float(episode_reward),
        "pending_full_vector": [int(value) for value in pending_full_vector],
        "final_config_fingerprint": str(
            runtime_info.get("final_config_fingerprint", "")
        ),
        "boosted_overrides": _serialize_boosted_overrides(boosted_overrides),
        "installed_action": {
            "layers": describe_layerwise_action_matrix(action_matrix),
        },
        "forward_ran": bool(runtime_info.get("forward_ran", False)),
        "model_uses_replan_config": bool(
            replan.get("model_uses_replan_config", False)
        ),
        "replan_application": to_jsonable(
            replan,
            stringify_unknown=True,
        ),
    }


def _comparator_projection(
        *,
        evaluation: SearchEvaluation,
        reset_seed: int,
        ) -> dict[str, Any]:
    metadata = dict(evaluation.metadata)
    probabilities = {
        name: float(value)
        for name, value in zip(
            CONSTRAINT_PROBABILITY_NAMES,
            evaluation.constraint_probabilities,
        )
    }
    if len(probabilities) != len(CONSTRAINT_PROBABILITY_NAMES):
        raise RuntimeError(
            "comparator same-action route returned incomplete probabilities"
        )
    return {
        "action_matrix": [list(row) for row in evaluation.action_matrix],
        "reset_seed": int(reset_seed),
        "probe_seed": int(metadata["probe_seed"]),
        "trial_seeds": [int(value) for value in metadata["trial_seeds"]],
        "trial_results": {
            str(name): [float(value) for value in values]
            for name, values in dict(metadata["trial_results"]).items()
        },
        "metrics": evaluation.metrics.as_dict(),
        "limits": evaluation.limits.as_dict(),
        "constraint_probabilities": probabilities,
        "gate_probability": float(evaluation.gate_probability),
        "bootstrap_seed": int(metadata["bootstrap_seed"]),
        "reward": float(evaluation.reward),
        "pending_full_vector": [
            int(value) for value in metadata["pending_full_vector"]
        ],
        "final_config_fingerprint": str(
            metadata["final_config_fingerprint"]
        ),
        "boosted_overrides": to_jsonable(
            metadata["boosted_overrides"],
            stringify_unknown=True,
        ),
        "installed_action": to_jsonable(
            metadata["installed_action"],
            stringify_unknown=True,
        ),
        "forward_ran": bool(metadata["forward_ran"]),
        "model_uses_replan_config": bool(
            metadata["model_uses_replan_config"]
        ),
        "replan_application": to_jsonable(
            metadata["replan_application"],
            stringify_unknown=True,
        ),
    }


def _first_difference(left: Any, right: Any, path: str = "$") -> str:
    if type(left) is not type(right):
        return f"{path}: type {type(left).__name__} != {type(right).__name__}"
    if isinstance(left, Mapping):
        left_keys = set(left)
        right_keys = set(right)
        if left_keys != right_keys:
            return (
                f"{path}: keys {sorted(left_keys)} != {sorted(right_keys)}"
            )
        for key in sorted(left_keys):
            difference = _first_difference(
                left[key],
                right[key],
                f"{path}.{key}",
            )
            if difference:
                return difference
        return ""
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return f"{path}: length {len(left)} != {len(right)}"
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            difference = _first_difference(
                left_item,
                right_item,
                f"{path}[{index}]",
            )
            if difference:
                return difference
        return ""
    if left != right:
        return f"{path}: {left!r} != {right!r}"
    return ""


def assert_same_action_projection_equal(
        ppo_projection: Mapping[str, Any],
        comparator_projection: Mapping[str, Any],
        ) -> None:
    """Reject any bit-visible semantic drift between the two routes."""
    ppo_owned = to_jsonable(dict(ppo_projection), stringify_unknown=True)
    comparator_owned = to_jsonable(
        dict(comparator_projection),
        stringify_unknown=True,
    )
    if stable_json_hash(ppo_owned) == stable_json_hash(comparator_owned):
        return
    difference = _first_difference(ppo_owned, comparator_owned)
    raise RuntimeError(
        "Stage-2 same-action parity failed"
        + (f" at {difference}" if difference else "")
    )


def run_same_action_parity_gate(
        *,
        layerwise_env: Any,
        robust_reference: Any,
        action_matrix: Sequence[Sequence[int]],
        base_seed: int,
        stream_index: int = 0,
        expected_trials: int | None = None,
        device: Any = "cpu",
        strict_validator: Callable[[SearchResult], Mapping[str, Any]] | None = None,
        ) -> dict[str, Any]:
    """Run one fixed action through the production PPO and search adapters."""
    horizon = int(getattr(layerwise_env, "horizon", 0))
    action = _canonical_action(action_matrix, horizon=horizon)
    trials = int(
        expected_trials
        if expected_trials is not None
        else getattr(layerwise_env.base.env_cfg, "num_trials_per_step", 0)
    )
    if trials < 2:
        raise ValueError("same-action parity requires at least two trials")
    reset_seed, probe_seed = derive_layerwise_online_evaluation_seeds(
        int(base_seed),
        int(stream_index),
        trial_count=trials,
    )
    configure_deferral = getattr(
        layerwise_env,
        "configure_terminal_probe_deferral",
        None,
    )
    if callable(configure_deferral):
        configure_deferral(False)
    clear = getattr(layerwise_env.base, "clear_installed_blb", None)
    if callable(clear):
        clear()

    policy = _FixedActionPolicy(action)
    draft = _collect_layerwise_episode(
        env=layerwise_env,
        policy=policy,
        rollout_buffer=_RecordingBuffer(),
        entropy_samples=[],
        absolute_episode=int(stream_index),
        base_seed=int(base_seed),
        expected_online_trials=trials,
        horizon=horizon,
        device=device,
        step_adapter_fn=step_to_mask_and_levels,
    )
    if policy.consumed != horizon:
        raise RuntimeError("fixed action policy did not cover the full horizon")
    if draft.prepared_terminal_probe is not None:
        raise RuntimeError(
            "same-action parity unexpectedly received a deferred terminal probe"
        )
    ppo_projection = _ppo_projection(
        action_matrix=action,
        runtime_info=draft.runtime_info,
        boosted_overrides=draft.boosted_overrides,
        pending_full_vector=_pending_vector(layerwise_env),
        reset_seed=reset_seed,
        probe_seed=probe_seed,
        expected_trials=trials,
        episode_reward=draft.episode_reward,
        robust_reference=robust_reference,
    )
    if callable(clear):
        clear()

    evaluator = LayerwiseRuntimeEvaluator(
        env=layerwise_env,
        reference=robust_reference,
        base_seed=int(base_seed),
        expected_trials=trials,
    )
    evaluator.evaluation_count = int(stream_index)
    comparator = evaluator(action)
    comparator_projection = _comparator_projection(
        evaluation=comparator,
        reset_seed=reset_seed,
    )
    assert_same_action_projection_equal(
        ppo_projection,
        comparator_projection,
    )

    evidence: dict[str, Any] = {
        "schema_version": "stage2_same_action_parity_v1",
        "passed": True,
        "base_seed": int(base_seed),
        "stream_index": int(stream_index),
        "expected_trials": trials,
        "projection_sha256": stable_json_hash(ppo_projection),
        "ppo_projection": ppo_projection,
        "comparator_projection": comparator_projection,
    }
    if strict_validator is not None:
        strict_result = SearchResult(
            algorithm="bo_rf",
            best=comparator,
            observations=(comparator,),
            history=(),
            termination_reason="same_action_parity_gate",
        )
        strict_payload = strict_validator(strict_result)
        if not isinstance(strict_payload, Mapping):
            raise RuntimeError("same-action strict validation returned no mapping")
        selected = strict_payload.get("selected")
        if not isinstance(selected, Mapping):
            raise RuntimeError("same-action strict validation selected no candidate")
        selected_action = selected.get("action_matrix")
        if selected_action != [list(row) for row in action]:
            raise RuntimeError(
                "same-action strict validation selected a different action"
            )
        evidence["strict_validation"] = to_jsonable(
            strict_payload,
            stringify_unknown=True,
        )
        evidence["strict_validation_sha256"] = stable_json_hash(
            evidence["strict_validation"]
        )
    return evidence
