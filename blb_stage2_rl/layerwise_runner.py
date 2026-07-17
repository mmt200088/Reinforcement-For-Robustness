"""Twelve-step robust PPO training for the Stage-2 layerwise action space."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, is_dataclass, replace
import fcntl
import hashlib
import json
import math
import os
import time
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from .candidate_store import (
    CandidateStore,
    CandidateTrialEvidence,
    candidate_key,
    sha256_json,
)
from .layerwise_action import compute_variable_cost_from_action_matrix
from .statistical_constraints import (
    TrialSeries,
    assess_candidate,
)


_PROBABILITY_FIELDS = (
    "loss_precision_probability",
    "metric1_precision_probability",
    "metric2_precision_probability",
    "loss_stability_probability",
    "metric1_stability_probability",
    "metric2_stability_probability",
)
_DECISION_GRANULARITIES = frozenset(("layer", "block"))
_REWARD_DESIGNS = frozenset((
    "robust_constrained", "stage1_aligned", "continuous", "tiered",
))
_LAUNCHER_LOCK_FD_ENV = "BLB_STAGE2_RUN_LOCK_FD"
_LAUNCHER_LOCK_PATH_ENV = "BLB_STAGE2_RUN_LOCK_PATH"
DEFAULT_CONVERGENCE_PATIENCE_UPDATES = 100
StrictSelectionKey = tuple[tuple[float, ...], tuple[int, ...], str]
ResourceObjective = tuple[float, float]
_FINAL_REVALIDATION_PASSED = "final_revalidation_passed"
_FINAL_REVALIDATION_FAILED = "final_revalidation_failed"
_UNSET = object()


def evidence_identity_context(
        identity_context: Mapping[str, Any],
        fidelity: str,
        ) -> dict[str, Any]:
    """Return an immutable-base candidate identity for one evidence tier."""
    value = str(fidelity or "").strip().upper()
    if value not in ("F1", "F4"):
        raise ValueError(f"layerwise evidence fidelity must be F1 or F4, got {fidelity!r}")
    context = dict(identity_context)
    context["fidelity"] = value
    return context


def stage2_run_lock_path(progress_dir: Any) -> str:
    """Return a stable lock path outside the deletable run directory."""
    path = os.path.realpath(os.fspath(progress_dir))
    if os.path.basename(path) == "progress":
        parent = os.path.dirname(path)
        run_dir = (
            os.path.dirname(parent)
            if os.path.basename(parent) == "stage2_noise"
            else parent
        )
    else:
        run_dir = path
    lock_parent = os.path.dirname(run_dir)
    lock_name = f".{os.path.basename(run_dir)}.stage2_rl.lock"
    return os.path.join(lock_parent, lock_name)


class LayerwiseRunLock:
    """Hold one writer lock for a Stage-2 persistent run directory."""

    def __init__(self, run_dir: Any) -> None:
        self.run_dir = os.fspath(run_dir)
        self.path = stage2_run_lock_path(self.run_dir)
        self._handle = None
        self._inherited_launcher_lock = False
        self._started_at = time.time()
        self._run_context_hash: Optional[str] = None

    def _write_metadata(self, *, active: bool) -> None:
        if self._handle is None:
            raise RuntimeError("layerwise run lock is not held")
        payload = {
            "schema_version": "stage2_run_lock_v2",
            "pid": int(os.getpid()),
            "active": bool(active),
            "started_at_unix": float(self._started_at),
            "run_context_hash": self._run_context_hash,
        }
        self._handle.seek(0)
        self._handle.truncate()
        self._handle.write(json.dumps(payload, sort_keys=True) + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def __enter__(self) -> "LayerwiseRunLock":
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        inherited_fd_raw = str(os.environ.get(_LAUNCHER_LOCK_FD_ENV, "")).strip()
        inherited_path_raw = str(os.environ.get(_LAUNCHER_LOCK_PATH_ENV, "")).strip()
        if inherited_fd_raw or inherited_path_raw:
            if not inherited_fd_raw or not inherited_path_raw:
                raise RuntimeError("incomplete inherited Stage-2 launcher lock metadata")
            inherited_path = os.path.realpath(inherited_path_raw)
            if inherited_path != os.path.realpath(self.path):
                raise RuntimeError(
                    f"inherited Stage-2 launcher lock path {inherited_path!r} "
                    f"does not match expected {self.path!r}"
                )
            try:
                inherited_fd = int(inherited_fd_raw)
                fd_stat = os.fstat(inherited_fd)
                path_stat = os.stat(self.path)
            except (OSError, ValueError) as exc:
                raise RuntimeError("inherited Stage-2 launcher lock is invalid") from exc
            if (fd_stat.st_dev, fd_stat.st_ino) != (path_stat.st_dev, path_stat.st_ino):
                raise RuntimeError("inherited Stage-2 launcher lock inode mismatch")
            handle = os.fdopen(os.dup(inherited_fd), "r+", encoding="utf-8")
            self._inherited_launcher_lock = True
        else:
            handle = open(self.path, "a+", encoding="utf-8")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                handle.seek(0)
                owner = handle.read().strip()
                handle.close()
                raise RuntimeError(
                    "Stage-2 persistent run is already active"
                    + (f": {owner}" if owner else "")
                ) from exc
        self._handle = handle
        self._write_metadata(active=True)
        return self

    def bind_context(self, run_context_hash: str) -> None:
        value = str(run_context_hash or "").strip()
        if not value:
            raise ValueError("run_context_hash must be non-empty")
        self._run_context_hash = value
        self._write_metadata(active=True)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        if self._handle is None:
            return
        try:
            self._write_metadata(active=False)
        finally:
            if not self._inherited_launcher_lock:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None


def validate_fresh_layerwise_run_state(
        run_id_marker: Any,
        checkpoint_coupled_paths: Sequence[Any],
        ) -> None:
    """Refuse to mix orphaned run data with a new episode-zero policy."""
    stale = []
    marker = os.fspath(run_id_marker)
    if os.path.exists(marker):
        stale.append(marker)
    for raw_path in checkpoint_coupled_paths:
        path = os.fspath(raw_path)
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            stale.append(path)
    if stale:
        raise RuntimeError(
            "layerwise persistent data exists without a checkpoint; "
            "restore the checkpoint or start with a fresh run directory: "
            + ", ".join(stale)
        )


class CheckpointFileFingerprintTracker:
    """Incrementally hash checkpoint-owned file prefixes within one process."""

    def __init__(self) -> None:
        self._states: dict[str, dict[str, Any]] = {}
        self._bytes_hashed = 0

    @property
    def bytes_hashed(self) -> int:
        return int(self._bytes_hashed)

    def fingerprints(
            self,
            file_specs: Mapping[str, tuple[Any, int]],
            ) -> dict[str, str]:
        names = {str(name) for name in file_specs}
        if self._states and names != set(self._states):
            raise RuntimeError("checkpoint fingerprint file set changed")
        for raw_name, (raw_path, raw_size) in sorted(file_specs.items()):
            name = str(raw_name)
            path = os.path.realpath(os.fspath(raw_path))
            size = int(raw_size)
            if size < 0:
                raise ValueError(f"committed size must be non-negative, got {size}")
            state = self._states.get(name)
            if state is None:
                state = {"path": path, "size": 0, "digest": hashlib.sha256()}
                self._states[name] = state
            if state["path"] != path:
                raise RuntimeError(f"checkpoint fingerprint path changed for {name}")
            previous_size = int(state["size"])
            if size < previous_size:
                raise RuntimeError(
                    f"checkpoint fingerprint size regressed for {name}: "
                    f"{size} < {previous_size}"
                )
            remaining = size - previous_size
            if remaining:
                try:
                    with open(path, "rb") as handle:
                        handle.seek(previous_size)
                        while remaining:
                            chunk = handle.read(min(1024 * 1024, remaining))
                            if not chunk:
                                raise RuntimeError(
                                    f"checkpoint file {path!r} is shorter than {size} bytes"
                                )
                            state["digest"].update(chunk)
                            state["size"] = int(state["size"]) + len(chunk)
                            self._bytes_hashed += len(chunk)
                            remaining -= len(chunk)
                except FileNotFoundError as exc:
                    raise RuntimeError(f"checkpoint file {path!r} is missing") from exc
        return {
            name: str(state["digest"].hexdigest())
            for name, state in sorted(self._states.items())
        }

    def validate_and_seed(
            self,
            expected: Mapping[str, Any],
            file_specs: Mapping[str, tuple[Any, int]],
            ) -> None:
        if self._states:
            raise RuntimeError("checkpoint fingerprint tracker is already seeded")
        expected_plain = {
            str(name): str(value) for name, value in dict(expected).items()
        }
        if self.fingerprints(file_specs) != expected_plain:
            raise RuntimeError(
                "layerwise checkpoint store fingerprint mismatch; start a fresh run"
            )


def bind_layerwise_candidate_identity(
        identity_context: Mapping[str, Any],
        k_levels: Sequence[int],
        cost_model_revision: str,
        resource_contract: Mapping[str, Any],
        ) -> dict[str, Any]:
    """Bind persisted evidence to decoded actions and resource semantics."""
    levels = tuple(int(value) for value in k_levels)
    if not levels or len(set(levels)) != len(levels):
        raise ValueError(f"k_levels must be non-empty and unique, got {levels}")
    contract = dict(resource_contract)
    required = (
        "algorithm_contract_hash",
        "resource_secondary_epsilon",
        "compute_axis_denominator",
        "communication_axis_denominator",
        "resource_credit_mode",
        "strict_resource_order",
    )
    missing = [field_name for field_name in required if field_name not in contract]
    if missing:
        raise ValueError(
            f"resource_contract is missing required fields: {missing}"
        )
    algorithm_hash = str(contract["algorithm_contract_hash"] or "").strip()
    credit_mode = str(contract["resource_credit_mode"] or "").strip()
    order_raw = contract["strict_resource_order"]
    if not algorithm_hash or not credit_mode:
        raise ValueError(
            "resource contract hash and credit mode must be non-empty"
        )
    if not isinstance(order_raw, Sequence) or isinstance(order_raw, (str, bytes)):
        raise ValueError("strict_resource_order must be a non-empty sequence")
    strict_order = [str(value).strip() for value in order_raw]
    if not strict_order or any(not value for value in strict_order):
        raise ValueError("strict_resource_order must contain non-empty fields")
    epsilon = float(contract["resource_secondary_epsilon"])
    compute_denominator = float(contract["compute_axis_denominator"])
    communication_denominator = float(
        contract["communication_axis_denominator"]
    )
    if not math.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("resource_secondary_epsilon must be finite and nonnegative")
    for field_name, value in (
            ("compute_axis_denominator", compute_denominator),
            ("communication_axis_denominator", communication_denominator),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{field_name} must be finite and positive")
    context = dict(identity_context)
    context["k_levels"] = list(levels)
    context["cost_model_revision"] = str(cost_model_revision)
    context["resource_objective_contract"] = {
        "algorithm_contract_hash": algorithm_hash,
        "resource_secondary_epsilon": epsilon,
        "compute_axis_denominator": compute_denominator,
        "communication_axis_denominator": communication_denominator,
        "resource_credit_mode": credit_mode,
        "strict_resource_order": strict_order,
    }
    return context


def build_layerwise_run_context(
        candidate_identity_context: Mapping[str, Any],
        algorithm_contract_hash: str,
        training_settings: Mapping[str, Any],
        ) -> dict[str, Any]:
    """Build the complete experiment context accepted by a live checkpoint."""
    return {
        "schema_version": "stage2_layerwise_run_context_v1",
        "algorithm_contract_hash": str(algorithm_contract_hash),
        "candidate_identity_context": dict(candidate_identity_context),
        "training_settings": dict(training_settings),
    }


def validate_layerwise_checkpoint_metadata(
        checkpoint: Mapping[str, Any],
        *,
        rl_variant: str,
        algorithm_revision: str,
        algorithm_contract_hash: str,
        run_context_hash: str,
        ) -> None:
    """Reject checkpoints from a different algorithm or experiment context."""
    checks = (
        ("variant", "rl_variant", rl_variant),
        ("algorithm revision", "algorithm_revision", algorithm_revision),
        ("algorithm contract", "algorithm_contract_hash", algorithm_contract_hash),
        ("run context", "run_context_hash", run_context_hash),
    )
    for label, field, expected in checks:
        actual = str(checkpoint.get(field, "") or "")
        if actual != str(expected):
            raise RuntimeError(
                f"layerwise checkpoint {label} {actual!r} != {str(expected)!r}; "
                "start a fresh run"
            )


def checkpoint_file_fingerprints(
        file_specs: Mapping[str, tuple[Any, int]],
        ) -> dict[str, str]:
    """Hash each checkpoint-owned file prefix at its committed byte boundary."""
    return CheckpointFileFingerprintTracker().fingerprints(file_specs)


def validate_checkpoint_file_fingerprints(
        expected: Mapping[str, Any],
        file_specs: Mapping[str, tuple[Any, int]],
        ) -> None:
    """Verify persisted stores before any checkpoint-driven truncation."""
    CheckpointFileFingerprintTracker().validate_and_seed(expected, file_specs)


def normalize_decision_granularity(value: Any) -> str:
    normalized = str(value or "block").strip().lower()
    if normalized not in _DECISION_GRANULARITIES:
        raise ValueError(
            "decision_granularity must be 'layer' or 'block', "
            f"got {value!r}"
        )
    return normalized


def normalize_reward_design(value: Any) -> str:
    normalized = str(value or "stage1_aligned").strip().lower()
    if normalized not in _REWARD_DESIGNS:
        allowed = ", ".join(sorted(_REWARD_DESIGNS))
        raise ValueError(f"reward_design must be one of {allowed}; got {value!r}")
    return normalized


def apply_public_stage2_decision_config(evaluator: Any, config: Any) -> Any:
    """Apply and validate the public evaluator fields used for dispatch."""
    granularity = getattr(evaluator, "blb_v3_decision_granularity", None)
    reward_design = getattr(evaluator, "blb_v3_reward_design", None)
    config.decision_granularity = normalize_decision_granularity(
        config.decision_granularity if granularity in (None, "") else granularity
    )
    config.reward_design = normalize_reward_design(
        config.reward_design if reward_design in (None, "") else reward_design
    )
    return config


def resolve_decision_path(
        *,
        fusion_count_action: bool,
        decision_granularity: str,
        reward_design: str,
        ) -> str:
    """Validate Stage-2 decision granularity and return the training path."""
    granularity = normalize_decision_granularity(decision_granularity)
    normalized_reward = normalize_reward_design(reward_design)
    robust = normalized_reward == "robust_constrained"
    if robust and granularity != "layer":
        raise ValueError(
            "robust_constrained Stage-2 training requires decision_granularity='layer'"
        )
    if granularity == "layer" and not robust:
        raise ValueError(
            "decision_granularity='layer' requires reward_design='robust_constrained'"
        )
    if granularity == "layer" and not bool(fusion_count_action):
        raise ValueError(
            "layer decision granularity requires the fusion-count action map"
        )
    return "layerwise" if granularity == "layer" else "block"


def validate_stage2_episode_limit_mode(
        episode_limit: int,
        *,
        fusion_count_action: bool,
        decision_granularity: str,
        reward_design: str,
        sequential_rl: bool,
        substage_mode: bool,
        stage2_rl_variant: str,
        ) -> int:
    """Reserve zero-budget semantics for natural-convergence layerwise PPO."""
    limit = int(episode_limit)
    if limit < 0:
        raise ValueError("Stage-2 episode limit must be nonnegative")
    if limit == 0:
        granularity = normalize_decision_granularity(decision_granularity)
        normalized_reward = normalize_reward_design(reward_design)
        variant = str(stage2_rl_variant or "").strip().lower().replace("-", "_")
        if not (
                variant in ("blb_v3", "blb", "v3", "blb_stage2_rl")
                and bool(sequential_rl)
                and not bool(substage_mode)
                and bool(fusion_count_action)
                and granularity == "layer"
                and normalized_reward == "robust_constrained"
        ):
            raise ValueError(
                "Stage-2 episode limit 0 is supported only by layerwise robust "
                "constrained PPO"
            )
    return limit


def initialize_layerwise_policy(policy: Any) -> None:
    """Install the accepted decoded-value priors on all six slot heads."""
    from .layerwise_action import K_LEVELS

    k_probabilities = {
        13: 0.50,
        12: 0.20,
        11: 0.12,
        10: 0.08,
        9: 0.06,
        8: 0.04,
    }
    policy.set_initial_slot_probabilities(
        [{0: 0.60, 1: 0.40}] + [dict(k_probabilities) for _ in range(5)],
        [(0, 1)] + [tuple(K_LEVELS) for _ in range(5)],
    )


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _finite(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _diagnostic_entropy(value: Any) -> Optional[float]:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _assessment_probabilities(assessment: Any) -> tuple[float, ...]:
    if assessment is None:
        raise ValueError("strict ranking requires an assessment")
    probabilities = tuple(
        _finite(_field(assessment, name), name=name)
        for name in _PROBABILITY_FIELDS
    )
    if any(value < 0.0 or value > 1.0 for value in probabilities):
        raise ValueError("assessment probabilities must be in [0, 1]")
    return probabilities


def normalized_constraint_safety_margins(
        metrics: Any,
        statistical_reference: Any,
        ) -> tuple[float, ...]:
    """Return positive-is-safer margins for all six strict constraints."""
    specifications = (
        ("loss_mean", "loss_limit", -1.0),
        ("metric1_mean", "metric1_limit", 1.0),
        ("metric2_mean", "metric2_limit", 1.0),
        ("loss_std", "loss_std_limit", -1.0),
        ("metric1_std", "metric1_std_limit", -1.0),
        ("metric2_std", "metric2_std_limit", -1.0),
    )
    margins = []
    for metric_name, limit_name, direction in specifications:
        observed = _finite(_field(metrics, metric_name), name=metric_name)
        limit = _finite(
            _field(statistical_reference, limit_name), name=limit_name,
        )
        scale = max(abs(limit), 1.0e-12)
        margins.append(direction * (observed - limit) / scale)
    return tuple(margins)


def _constraint_safety_margins(candidate: Any) -> tuple[float, ...]:
    raw = _field(candidate, "constraint_safety_margins")
    if raw is None:
        raise ValueError("strict ranking requires six constraint safety margins")
    margins = tuple(
        _finite(value, name=f"constraint_safety_margins[{index}]")
        for index, value in enumerate(raw)
    )
    if len(margins) != len(_PROBABILITY_FIELDS):
        raise ValueError("strict ranking requires six constraint safety margins")
    return margins


def _resource_fields_from_action_matrix(
        action_matrix: Sequence[Sequence[int]],
        ) -> dict[str, Any]:
    objective = compute_variable_cost_from_action_matrix(action_matrix)
    return {
        "compute_saving": float(objective.compute_saving),
        "communication_saving": float(objective.communication_saving),
        "robust_floor": float(objective.robust_floor),
        "secondary_progress": float(objective.secondary_progress),
        "ppo_resource_score": float(objective.ppo_resource_score),
        "compute_shapley_credit": float(objective.compute_shapley_credit),
        "communication_shapley_credit": float(
            objective.communication_shapley_credit
        ),
        "fusion_count": int(objective.fusion_count),
        "removed_k_bits": int(objective.removed_k_bits),
        "layer_resource_rewards": [
            float(value) for value in objective.layer_resource_rewards
        ],
        "slot_resource_rewards": [
            [float(value) for value in row]
            for row in objective.slot_resource_rewards
        ],
    }


def _candidate_resource_fields(candidate: Any) -> dict[str, Any]:
    action_matrix = _field(candidate, "action_matrix")
    if action_matrix is not None:
        return _resource_fields_from_action_matrix(action_matrix)
    compute = _field(candidate, "compute_saving")
    communication = _field(candidate, "communication_saving")
    if compute is None or communication is None:
        # Read-only compatibility for pre-v9 unit tests/report fixtures.  Live
        # candidates always carry an action matrix and are recomputed above.
        legacy = _finite(_field(candidate, "variable_cost"), name="variable_cost")
        compute = communication = legacy
    compute_value = _finite(compute, name="compute_saving")
    communication_value = _finite(communication, name="communication_saving")
    if not 0.0 <= compute_value <= 1.0 or not 0.0 <= communication_value <= 1.0:
        raise ValueError("resource savings must be in [0, 1]")
    return {
        "compute_saving": compute_value,
        "communication_saving": communication_value,
        "robust_floor": min(compute_value, communication_value),
        "secondary_progress": 0.5 * (compute_value + communication_value),
    }


def strict_rank_key(candidate: Any) -> tuple[float, ...]:
    """Ascending sort key for robust-feasible layerwise candidates."""
    assessment = _field(candidate, "assessment")
    probabilities = _assessment_probabilities(assessment)
    margins = _constraint_safety_margins(candidate)
    resource = _candidate_resource_fields(candidate)
    confidence_order = tuple(-value for value in sorted(probabilities))
    margin_order = tuple(-value for value in sorted(margins))
    return (
        -resource["robust_floor"],
        -resource["secondary_progress"],
        *confidence_order,
        *margin_order,
    )


def strict_selection_key(
        candidate_key_value: Any,
        candidate: Any,
        ) -> StrictSelectionKey:
    """Rank a candidate with action lexicographic order as the final tie-break."""
    identity = str(candidate_key_value).strip()
    if not identity:
        raise ValueError("candidate_key cannot be empty")
    full_vector = _field(candidate, "full_vector")
    if not isinstance(full_vector, Sequence) or isinstance(full_vector, (str, bytes)):
        raise ValueError("strict ranking requires a full action vector")
    action_identity = tuple(int(value) for value in full_vector)
    if not action_identity:
        raise ValueError("strict ranking requires a non-empty full action vector")
    return strict_rank_key(candidate), action_identity, identity


def strict_selection_key_from_snapshot(
        snapshot: Mapping[str, Any],
        ) -> Optional[StrictSelectionKey]:
    """Rebuild the live selection-key shape from a persisted strict snapshot."""
    if not snapshot.get("rank_key") or not snapshot.get("candidate_key"):
        return None
    return strict_selection_key(snapshot["candidate_key"], snapshot)


def strict_resource_pareto_frontier(
        candidates: Mapping[str, Mapping[str, Any]],
        ) -> dict[str, Mapping[str, Any]]:
    """Return deterministic strict-feasible non-dominated F/C candidates."""
    rows = []
    for identity, candidate in candidates.items():
        resource = _candidate_resource_fields(candidate)
        rows.append((str(identity), candidate, resource))
    frontier = []
    for identity, candidate, resource in rows:
        dominated = any(
            other_resource["compute_saving"] >= resource["compute_saving"] - 1.0e-12
            and other_resource["communication_saving"]
            >= resource["communication_saving"] - 1.0e-12
            and (
                other_resource["compute_saving"]
                > resource["compute_saving"] + 1.0e-12
                or other_resource["communication_saving"]
                > resource["communication_saving"] + 1.0e-12
            )
            for other_identity, _other, other_resource in rows
            if other_identity != identity
        )
        if not dominated:
            frontier.append((identity, candidate))
    frontier.sort(key=lambda item: strict_selection_key(item[0], item[1]))
    return {identity: candidate for identity, candidate in frontier}


def _strict_best_snapshot(
        accepted_candidates: Mapping[str, Mapping[str, Any]],
        ) -> Optional[dict[str, Any]]:
    """Return a detached snapshot of the current feasible frontier winner."""
    if not accepted_candidates:
        return None
    candidate_key_value, best = min(
        accepted_candidates.items(),
        key=lambda item: strict_selection_key(item[0], item[1]),
    )
    resource = _candidate_resource_fields(best)
    promotion_trials = best.get("promotion_trials")
    promotion_evidence = None
    if isinstance(promotion_trials, TrialSeries):
        promotion_evidence = {
            "status": "promoted",
            "trial_count": len(promotion_trials.loss),
            "trials": {
                "loss": list(promotion_trials.loss),
                "metric1": list(promotion_trials.metric1),
                "metric2": list(promotion_trials.metric2),
                "seeds": list(promotion_trials.seeds),
            },
        }
    return {
        "candidate_key": str(candidate_key_value),
        "rank_key": list(strict_rank_key(best)),
        "action_matrix": [list(row) for row in best["action_matrix"]],
        "full_vector": list(best["full_vector"]),
        "assessment": _to_plain_mapping(best["assessment"]),
        "metrics": dict(best["metrics"]),
        **resource,
        "variable_cost": float(resource.get(
            "ppo_resource_score", _field(best, "variable_cost", 0.0),
        )),
        "constraint_safety_margins": list(
            _constraint_safety_margins(best)
        ),
        "reward": (
            None if best.get("reward") is None else float(best["reward"])
        ),
        "boosted_overrides": copy.deepcopy(best["boosted_overrides"]),
        "promotion_evidence": promotion_evidence,
    }


def _strict_pareto_snapshots(
        accepted_candidates: Mapping[str, Mapping[str, Any]],
        ) -> list[dict[str, Any]]:
    frontier = strict_resource_pareto_frontier(accepted_candidates)
    return [
        _strict_best_snapshot({identity: candidate})
        for identity, candidate in frontier.items()
    ]


def normalized_entropy_snapshot(
        entropy_per_slot: Any,
        slot_masks: Any,
        per_slot_num_levels: Any,
        ) -> dict[str, float | int | None]:
    """Normalize active per-slot entropy and split Block4 from K slots."""
    entropy = np.asarray(entropy_per_slot, dtype=np.float64)
    masks = np.asarray(slot_masks, dtype=bool)
    levels = np.asarray(per_slot_num_levels, dtype=np.int64)
    if entropy.ndim != 2 or entropy.shape != masks.shape or entropy.shape != levels.shape:
        raise ValueError("entropy, masks, and levels must be aligned 2-D arrays")
    if entropy.shape[1] != 6:
        raise ValueError("layerwise entropy requires exactly six slots")

    active = masks & (levels > 1)
    normalized = np.zeros_like(entropy, dtype=np.float64)
    normalized[active] = entropy[active] / np.log(levels[active].astype(np.float64))
    block4_values = normalized[:, 0][active[:, 0]]
    k_values = normalized[:, 1:][active[:, 1:]]
    return {
        "block4": (
            float(np.mean(block4_values)) if block4_values.size else None
        ),
        "k": float(np.mean(k_values)) if k_values.size else None,
        "block4_slot_count": int(block4_values.size),
        "k_slot_count": int(k_values.size),
    }


def redistribute_layerwise_rewards(
        *,
        terminal_reward: float,
        priority: int,
        ppo_resource_score: float,
        layer_resource_rewards: Sequence[float],
        ) -> tuple[float, ...]:
    """Move P3 resource credit to its twelve source layer transitions.

    The returned rewards always sum to ``terminal_reward``. Precision/stability
    failures retain terminal-only credit, so resources cannot leak into P1/P2.
    """
    reward = _finite(terminal_reward, name="terminal_reward")
    resource_score = _finite(ppo_resource_score, name="ppo_resource_score")
    layer_resources = tuple(
        _finite(value, name=f"layer_resource_rewards[{index}]")
        for index, value in enumerate(layer_resource_rewards)
    )
    if len(layer_resources) != 12:
        raise ValueError(
            "layer_resource_rewards must contain 12 values, got "
            f"{len(layer_resources)}"
        )
    if resource_score < 0.0 or resource_score > 1.0:
        raise ValueError(
            f"ppo_resource_score must be in [0, 1], got {resource_score}"
        )
    if not math.isclose(
            sum(layer_resources), resource_score, rel_tol=0.0, abs_tol=1.0e-9,
    ):
        raise ValueError(
            "layer_resource_rewards sum must equal ppo_resource_score: "
            f"{sum(layer_resources)} != {resource_score}"
        )
    if int(priority) != 3:
        return (0.0,) * 11 + (reward,)

    redistributed = list(layer_resources)
    redistributed[-1] += reward - resource_score
    if not math.isclose(sum(redistributed), reward, rel_tol=0.0, abs_tol=1.0e-9):
        raise RuntimeError("layerwise reward redistribution changed episode return")
    return tuple(float(value) for value in redistributed)


@dataclass(frozen=True)
class LayerwiseConvergenceState:
    completed_episodes: int
    block4_entropy: Optional[float]
    k_entropy: Optional[float]
    stall_update_windows: int
    converged: bool
    extension_required: bool
    best_robust_feasible_objective: Optional[ResourceObjective]
    selected_action_identity: Optional[str]
    selected_action_stable_update_windows: int
    plateau_ready: bool
    strict_revalidation_passed: bool
    termination_reason: str

    @property
    def best_robust_feasible_cost(self) -> Optional[float]:
        """Read-only compatibility alias for older report fixtures."""
        if self.best_robust_feasible_objective is None:
            return None
        return float(self.best_robust_feasible_objective[0])


def _normalize_resource_objective(
        value: Optional[Sequence[float]],
        *,
        name: str,
        ) -> Optional[ResourceObjective]:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        legacy = _finite(value, name=name)
        value = (legacy, legacy)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must contain robust_floor and secondary_progress")
    values = tuple(
        _finite(item, name=f"{name}[{index}]")
        for index, item in enumerate(value)
    )
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    robust_floor, secondary_progress = values
    if not 0.0 <= robust_floor <= secondary_progress <= 1.0:
        raise ValueError(
            f"{name} must satisfy 0 <= robust_floor <= secondary_progress <= 1"
        )
    return float(robust_floor), float(secondary_progress)


def _resolve_resource_objective(
        objective: Optional[Sequence[float]],
        legacy_cost: Any,
        ) -> Optional[ResourceObjective]:
    if legacy_cost is not _UNSET:
        if objective is not None:
            raise ValueError(
                "provide robust_feasible_objective, not robust_feasible_cost"
            )
        if legacy_cost is None:
            return None
        cost = _finite(legacy_cost, name="robust_feasible_cost")
        objective = (cost, cost)
    return _normalize_resource_objective(
        objective, name="robust_feasible_objective",
    )


def _objective_compare(left: ResourceObjective, right: ResourceObjective) -> int:
    for left_value, right_value in zip(left, right):
        if left_value > right_value + 1.0e-12:
            return 1
        if left_value < right_value - 1.0e-12:
            return -1
    return 0


class LayerwiseConvergenceTracker:
    """Track robust frontier and exact selected-action stability."""

    def __init__(
            self,
            *,
            patience_updates: int = DEFAULT_CONVERGENCE_PATIENCE_UPDATES,
            ) -> None:
        self.patience_updates = int(patience_updates)
        if self.patience_updates <= 0:
            raise ValueError("patience_updates must be positive")
        self._best_objective: Optional[ResourceObjective] = None
        self._current_frontier_objective: Optional[ResourceObjective] = None
        self._stall_windows = 0
        self._selected_action_identity: Optional[str] = None
        self._selected_action_stable_windows = 0

    def state_dict(self) -> dict[str, Any]:
        best = (
            None if self._best_objective is None else list(self._best_objective)
        )
        current = (
            None
            if self._current_frontier_objective is None
            else list(self._current_frontier_objective)
        )
        return {
            "patience_updates": int(self.patience_updates),
            "best_robust_feasible_objective": best,
            "current_robust_feasible_objective": current,
            # Read-only aliases retained until report fixtures migrate to v9.
            "best_robust_feasible_cost": (
                None if best is None else float(best[0])
            ),
            "current_robust_feasible_cost": (
                None if current is None else float(current[0])
            ),
            "stall_update_windows": int(self._stall_windows),
            "selected_action_identity": self._selected_action_identity,
            "selected_action_stable_update_windows": int(
                self._selected_action_stable_windows
            ),
        }

    def load_state_dict(self, state: Optional[Mapping[str, Any]]) -> None:
        if not isinstance(state, Mapping):
            return
        expected_contract = {
            "patience_updates": int(self.patience_updates),
        }
        for field_name, expected in expected_contract.items():
            if field_name not in state:
                continue
            observed = state.get(field_name)
            observed = None if observed is None else int(observed)
            if observed != expected:
                raise ValueError(
                    "layerwise convergence contract mismatch: "
                    f"{field_name} checkpoint={observed!r}, requested={expected!r}"
                )
        raw_best = state.get("best_robust_feasible_objective", _UNSET)
        if raw_best is _UNSET:
            legacy_best = state.get("best_robust_feasible_cost")
            raw_best = None if legacy_best is None else (legacy_best, legacy_best)
        self._best_objective = _normalize_resource_objective(
            raw_best, name="best_robust_feasible_objective",
        )
        raw_current = state.get("current_robust_feasible_objective", _UNSET)
        if raw_current is _UNSET:
            legacy_current = state.get("current_robust_feasible_cost")
            raw_current = (
                None if legacy_current is None else (legacy_current, legacy_current)
            )
        self._current_frontier_objective = _normalize_resource_objective(
            raw_current, name="current_robust_feasible_objective",
        )
        stall_windows = int(state.get("stall_update_windows", 0))
        if stall_windows < 0:
            raise ValueError("stall_update_windows must be nonnegative")
        self._stall_windows = stall_windows

        selected_identity = state.get("selected_action_identity")
        if selected_identity is not None:
            selected_identity = str(selected_identity).strip()
            if not selected_identity:
                raise ValueError("selected_action_identity cannot be empty")
        selected_windows = int(state.get("selected_action_stable_update_windows", 0))
        if selected_windows < 0:
            raise ValueError("selected_action_stable_update_windows must be nonnegative")
        if selected_identity is None:
            selected_windows = 0
        self._selected_action_identity = selected_identity
        self._selected_action_stable_windows = selected_windows

    def reconcile_frontier(
            self,
            robust_feasible_objective: Optional[Sequence[float]] = None,
            robust_feasible_action_identity: Optional[str] = None,
            *,
            robust_feasible_cost: Any = _UNSET,
            ) -> None:
        """Align restored convergence state with the revalidated frontier."""
        objective = _resolve_resource_objective(
            robust_feasible_objective, robust_feasible_cost,
        )
        if objective is None:
            if robust_feasible_action_identity is not None:
                raise ValueError("selected action identity requires a feasible frontier")
            self._current_frontier_objective = None
            self._stall_windows = 0
            self._selected_action_identity = None
            self._selected_action_stable_windows = 0
            return
        if (
                self._current_frontier_objective is None
                or _objective_compare(
                    objective, self._current_frontier_objective,
                ) != 0
        ):
            self._best_objective = objective
            self._stall_windows = 0
        self._current_frontier_objective = objective

        selected_identity = (
            None
            if robust_feasible_action_identity is None
            else str(robust_feasible_action_identity).strip()
        )
        if selected_identity == "":
            raise ValueError("robust_feasible_action_identity cannot be empty")
        if selected_identity != self._selected_action_identity:
            self._selected_action_identity = selected_identity
            self._selected_action_stable_windows = 0

    def observe_update(
            self,
            *,
            completed_episodes: int,
            block4_entropy: Optional[float],
            k_entropy: Optional[float],
            robust_feasible_objective: Optional[Sequence[float]] = None,
            robust_feasible_action_identity: Optional[str] = None,
            count_patience: bool = True,
            strict_revalidation_passed: bool = False,
            robust_feasible_cost: Any = _UNSET,
            ) -> LayerwiseConvergenceState:
        episodes = int(completed_episodes)
        objective = _resolve_resource_objective(
            robust_feasible_objective, robust_feasible_cost,
        )
        if objective is None:
            if robust_feasible_action_identity is not None:
                raise ValueError("selected action identity requires a feasible frontier")
            self._current_frontier_objective = None
            self._stall_windows = 0
            self._selected_action_identity = None
            self._selected_action_stable_windows = 0
        else:
            frontier_restarted = self._current_frontier_objective is None
            frontier_retracted = bool(
                self._current_frontier_objective is not None
                and _objective_compare(
                    objective, self._current_frontier_objective,
                ) < 0
            )
            if frontier_restarted or frontier_retracted:
                self._best_objective = objective
                self._stall_windows = 0
            elif (
                    self._best_objective is None
                    or _objective_compare(objective, self._best_objective) > 0
            ):
                self._best_objective = objective
                self._stall_windows = 0
            elif count_patience:
                self._stall_windows += 1
            self._current_frontier_objective = objective

            selected_identity = (
                None
                if robust_feasible_action_identity is None
                else str(robust_feasible_action_identity).strip()
            )
            if selected_identity == "":
                raise ValueError("robust_feasible_action_identity cannot be empty")
            if selected_identity != self._selected_action_identity:
                self._selected_action_identity = selected_identity
                self._selected_action_stable_windows = 0
            elif selected_identity is not None and count_patience:
                self._selected_action_stable_windows += 1

        b4 = _diagnostic_entropy(block4_entropy)
        k_value = _diagnostic_entropy(k_entropy)
        plateau_ready = bool(
            objective is not None
            and self._best_objective is not None
            and self._stall_windows >= self.patience_updates
            and self._selected_action_identity is not None
            and self._selected_action_stable_windows >= self.patience_updates
        )
        revalidation_passed = bool(strict_revalidation_passed)
        converged = bool(plateau_ready and revalidation_passed)
        termination_reason = "converged" if converged else "running"
        return LayerwiseConvergenceState(
            completed_episodes=episodes,
            block4_entropy=b4,
            k_entropy=k_value,
            stall_update_windows=int(self._stall_windows),
            best_robust_feasible_objective=self._best_objective,
            selected_action_identity=self._selected_action_identity,
            selected_action_stable_update_windows=int(
                self._selected_action_stable_windows
            ),
            converged=converged,
            extension_required=False,
            plateau_ready=plateau_ready,
            strict_revalidation_passed=revalidation_passed,
            termination_reason=termination_reason,
        )


def is_unbounded_layerwise_training(
        remaining_episodes: int,
        planned_total_episodes: Optional[int],
        ) -> bool:
    """Disambiguate an unbounded run from an exhausted bounded resume."""
    remaining = int(remaining_episodes)
    planned = (
        remaining
        if planned_total_episodes is None else int(planned_total_episodes)
    )
    if remaining < 0 or planned < 0:
        raise ValueError("layerwise episode counts must be nonnegative")
    return planned == 0


def resolve_layerwise_episode_budget(
        requested_total_episodes: int,
        completed_episodes: int,
        ) -> int:
    """Return the remaining bounded budget, preserving zero as unbounded."""
    requested = int(requested_total_episodes)
    completed = int(completed_episodes)
    if requested < 0 or completed < 0:
        raise ValueError("layerwise episode counts must be nonnegative")
    if requested == 0:
        return 0
    if completed > requested:
        raise ValueError(
            f"layerwise checkpoint episode {completed} exceeds requested total {requested}"
        )
    return requested - completed


@dataclass(frozen=True)
class PromotionResult:
    status: str
    trial_count: int
    fresh_trial_count: int
    evidence: Optional[CandidateTrialEvidence]
    assessment: Optional[Any]
    metrics: Optional[Mapping[str, float]]


@dataclass(frozen=True)
class LayerwiseEpisodeRecord:
    episode_index: int
    reward: float
    priority: int
    action_matrix: tuple[tuple[int, ...], ...]
    pending_full_vector: tuple[int, ...]
    variable_cost: float
    compute_saving: float
    communication_saving: float
    robust_floor: float
    secondary_progress: float
    ppo_resource_score: float
    compute_shapley_credit: float
    communication_shapley_credit: float
    layer_resource_rewards: tuple[float, ...]
    slot_resource_rewards: tuple[tuple[float, ...], ...]
    raw_trials: Optional[TrialSeries]
    pooled_trials: Optional[TrialSeries]
    fresh_trial_count: int
    pooled_trial_count: int
    reward_evidence: str
    ranking_evidence: str
    fresh_assessment: Optional[Mapping[str, Any]]
    assessment: Optional[Any]
    metrics: Mapping[str, float]
    pooled_metrics: Optional[Mapping[str, float]]
    probe_diagnostics: Mapping[str, Any]
    promoted_trial_count: int
    promotion_status: str
    promotion_candidate_key: Optional[str]
    promotion_assessment: Optional[Mapping[str, Any]]
    promotion_metrics: Optional[Mapping[str, float]]
    invalid_steps: int
    step_count: int
    block4_entropy: Optional[float]
    k_entropy: Optional[float]
    stall_update_windows: int
    selected_action_identity: Optional[str]
    selected_action_stable_update_windows: int
    converged: bool
    extension_required: bool
    plateau_ready: bool
    strict_revalidation_passed: bool
    strict_revalidation_status: str
    termination_reason: str
    best_robust_feasible_cost: Optional[float]
    best_robust_feasible_objective: Optional[ResourceObjective]


def _to_plain_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def _trial_series_from_info(
        info: Any,
        *,
        required: bool = False,
        expected_count: Optional[int] = None,
        context: str = "terminal",
        ) -> Optional[TrialSeries]:
    if not isinstance(info, Mapping):
        if required:
            raise RuntimeError(f"{context} info is required for raw evidence")
        return None
    raw = info.get("statistical_trials")
    if not isinstance(raw, Mapping):
        if required:
            raise RuntimeError(f"{context} statistical_trials are required")
        return None
    trials = TrialSeries(
        loss=raw.get("loss", ()),
        metric1=raw.get("metric1", ()),
        metric2=raw.get("metric2", ()),
        seeds=raw.get("seeds", ()),
    )
    if not trials.seeds or len(trials.seeds) != len(trials.loss):
        raise ValueError(f"{context} requires nonempty aligned trial seeds")
    if expected_count is not None and len(trials.loss) != int(expected_count):
        raise ValueError(
            f"{context} expected exactly {int(expected_count)} raw trials; "
            f"received {len(trials.loss)}"
        )
    return trials


def _metrics_from_trials(trials: TrialSeries) -> dict[str, float]:
    loss = np.asarray(trials.loss, dtype=np.float64)
    metric1 = np.asarray(trials.metric1, dtype=np.float64)
    metric2 = np.asarray(trials.metric2, dtype=np.float64)
    return {
        "loss_mean": float(np.mean(loss)),
        "loss_std": float(np.std(loss, ddof=1)) if loss.size > 1 else 0.0,
        "metric1_mean": float(np.mean(metric1)),
        "metric1_std": float(np.std(metric1, ddof=1)) if metric1.size > 1 else 0.0,
        "metric2_mean": float(np.mean(metric2)),
        "metric2_std": float(np.std(metric2, ddof=1)) if metric2.size > 1 else 0.0,
    }


def _assessment_passes(assessment: Any, threshold: float) -> bool:
    try:
        return min(_assessment_probabilities(assessment)) >= float(threshold)
    except (TypeError, ValueError):
        return False


def _append_promotion_status(
        store: CandidateStore,
        action_indices: Sequence[int],
        identity_context: Mapping[str, Any],
        *,
        status: str,
        metadata: Optional[Mapping[str, Any]] = None,
        ) -> None:
    store.append({
        "record_type": "candidate_promotion_status_v1",
        "action_indices": list(action_indices),
        "effective_action_indices": list(action_indices),
        "identity_context": dict(identity_context),
        "fidelity": "F4",
        "valid": status in ("promoted", _FINAL_REVALIDATION_PASSED),
        "promotion_status": str(status),
        "promotion_metadata": dict(metadata or {}),
    })


def _serialize_boosted_overrides(overrides: Mapping[Any, Any]) -> list[dict[str, Any]]:
    rows = []
    for key, values in sorted(
            dict(overrides).items(),
            key=lambda item: (int(item[0][1]), int(item[0][0])),
    ):
        block_idx, layer_idx = key
        rows.append({
            "block_idx": int(block_idx),
            "layer_idx": int(layer_idx),
            "field_values": {
                str(name): int(value) for name, value in dict(values).items()
            },
        })
    return rows


def _deserialize_boosted_overrides(rows: Any) -> dict[tuple[int, int], dict[str, int]]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("boosted_overrides must be a sequence of rows")
    overrides: dict[tuple[int, int], dict[str, int]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("boosted_overrides rows must be mappings")
        values = row.get("field_values")
        if not isinstance(values, Mapping):
            raise ValueError("boosted_overrides field_values must be a mapping")
        key = (int(row["block_idx"]), int(row["layer_idx"]))
        overrides[key] = {str(name): int(value) for name, value in values.items()}
    return overrides


def _record_final_revalidation_outcome(
        *,
        candidate_store: CandidateStore,
        identity_context: Mapping[str, Any],
        revalidation_identity_context: Mapping[str, Any],
        candidate: Mapping[str, Any],
        passed: bool,
        revalidation_status: str,
        bootstrap_seed: int,
        final_probability: float,
        final_trial_count: int,
        ) -> None:
    """Persist the final verdict on the base F4 identity for honest resume."""
    action_indices = tuple(int(value) for value in candidate["full_vector"])
    resource = _resource_fields_from_action_matrix(candidate["action_matrix"])
    metadata = {
        "final_revalidation_identity_context": evidence_identity_context(
            revalidation_identity_context, "F4",
        ),
        "final_revalidation_probability": float(final_probability),
        "final_revalidation_trial_count": int(final_trial_count),
        "revalidation_status": str(revalidation_status),
        "assessment_bootstrap_seed": int(bootstrap_seed),
        **resource,
        "variable_cost": float(resource["ppo_resource_score"]),
        "action_matrix": [
            list(map(int, row)) for row in candidate["action_matrix"]
        ],
        "boosted_overrides": _serialize_boosted_overrides(
            candidate["boosted_overrides"],
        ),
    }
    if candidate.get("reward") is not None:
        metadata["episode_reward"] = _finite(
            candidate["reward"], name="episode_reward",
        )
    _append_promotion_status(
        candidate_store,
        action_indices,
        evidence_identity_context(identity_context, "F4"),
        status=(
            _FINAL_REVALIDATION_PASSED
            if passed else _FINAL_REVALIDATION_FAILED
        ),
        metadata=metadata,
    )


def restore_promoted_candidates(
        *,
        candidate_store: CandidateStore,
        identity_context: Mapping[str, Any],
        statistical_reference: Any,
        assess_candidate_fn: Callable[..., Any] = assess_candidate,
        promotion_probability: float = 0.80,
        assessment_trial_limit: int = 25,
        final_probability: float = 0.95,
        final_assessment_trial_limit: int = 25,
        ) -> dict[str, dict[str, Any]]:
    """Rebuild the current promoted frontier from append-only raw evidence."""
    full_identity_context = evidence_identity_context(identity_context, "F4")
    latest_status: dict[
        str, tuple[str, tuple[int, ...], dict[str, Any]]
    ] = {}
    wanted_context_hash = sha256_json(full_identity_context)
    for record in candidate_store.iter_active_records():
        if record.get("record_type") != "candidate_promotion_status_v1":
            continue
        if str(record.get("identity_context_hash", "")) != wanted_context_hash:
            continue
        key = str(record.get("candidate_key", ""))
        if not key:
            continue
        latest_status[key] = (
            str(record.get("promotion_status", "")),
            tuple(int(value) for value in record.get("action_indices", ())),
            dict(record.get("promotion_metadata") or {}),
        )

    restored: dict[str, dict[str, Any]] = {}
    for key, (status, action_indices, promotion_metadata) in latest_status.items():
        if status not in ("promoted", _FINAL_REVALIDATION_PASSED) or not action_indices:
            continue
        final_revalidated = status == _FINAL_REVALIDATION_PASSED
        evidence_context = full_identity_context
        gate_probability = float(promotion_probability)
        trial_limit = int(assessment_trial_limit)
        if final_revalidated:
            raw_context = promotion_metadata.get(
                "final_revalidation_identity_context",
            )
            if not isinstance(raw_context, Mapping):
                continue
            evidence_context = dict(raw_context)
            gate_probability = float(final_probability)
            trial_limit = int(final_assessment_trial_limit)
            stored_probability = float(
                promotion_metadata.get(
                    "final_revalidation_probability", gate_probability,
                )
            )
            stored_trial_count = int(
                promotion_metadata.get(
                    "final_revalidation_trial_count", trial_limit,
                )
            )
            if not math.isclose(
                    stored_probability, gate_probability,
                    rel_tol=0.0, abs_tol=1.0e-12,
            ):
                raise ValueError("final revalidation probability changed across resume")
            if stored_trial_count != trial_limit:
                raise ValueError("final revalidation trial count changed across resume")
        evidence = candidate_store.trial_evidence_for_action(
            action_indices, evidence_context,
            max_trials=trial_limit,
        )
        if evidence is None or not evidence.promoted:
            continue
        metadata: dict[str, Any] = {}
        for group in evidence.groups:
            for name in (
                    "action_matrix", "variable_cost", "episode_reward",
                    "assessment_bootstrap_seed", "boosted_overrides",
            ):
                if name in group:
                    metadata[name] = group[name]
        for name in (
                "action_matrix", "variable_cost", "episode_reward",
                "assessment_bootstrap_seed", "boosted_overrides",
        ):
            if name in promotion_metadata:
                metadata[name] = promotion_metadata[name]
        if not all(name in metadata for name in (
                "action_matrix", "boosted_overrides",
        )):
            continue
        assessment = assess_candidate_fn(
            evidence.trials,
            statistical_reference,
            gate_probability=gate_probability,
            bootstrap_seed=int(metadata.get("assessment_bootstrap_seed", 0)),
        )
        if not _assessment_passes(assessment, gate_probability):
            continue
        action_matrix = tuple(
            tuple(int(value) for value in row)
            for row in metadata["action_matrix"]
        )
        if len(action_matrix) != 12 or any(len(row) != 6 for row in action_matrix):
            raise ValueError("persisted layerwise action_matrix must be 12x6")
        resource = _resource_fields_from_action_matrix(action_matrix)
        reward = metadata.get("episode_reward")
        restored_metrics = _metrics_from_trials(evidence.trials)
        restored[key] = {
            **resource,
            "variable_cost": float(resource["ppo_resource_score"]),
            "assessment": assessment,
            "metrics": restored_metrics,
            "constraint_safety_margins": normalized_constraint_safety_margins(
                restored_metrics, statistical_reference,
            ),
            "action_matrix": action_matrix,
            "full_vector": tuple(action_indices),
            "boosted_overrides": _deserialize_boosted_overrides(
                metadata["boosted_overrides"]
            ),
            "reward": None if reward is None else _finite(reward, name="episode_reward"),
            "promotion_trials": evidence.trials,
            "final_revalidation_status": (
                "passed" if final_revalidated else "not_run"
            ),
        }
    return restored


def _promotion_probe_seed(
        candidate_key_value: str,
        bootstrap_seed: int,
        existing_trial_count: int,
        existing_seeds: Sequence[int],
        fresh_trial_count: int,
        ) -> tuple[int, tuple[int, ...]]:
    from .seed_utils import (
        derive_layerwise_promotion_probe_seed,
        derive_probe_trial_seed,
    )

    material = (
        f"layerwise-promotion:{candidate_key_value}:{int(bootstrap_seed)}:"
        f"{int(existing_trial_count)}"
    ).encode("utf-8")
    seed_material = (
        int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
        & 0x7FFFFFFFFFFFFFFF
    )
    occupied = {int(seed) for seed in existing_seeds}
    # Attempt domains are pairwise disjoint. With N occupied seeds, N+1
    # attempts guarantee at least one trial set has no overlap.
    for attempt_idx in range(len(occupied) + 1):
        probe_seed = derive_layerwise_promotion_probe_seed(
            seed_material,
            attempt_idx,
            trial_count=int(fresh_trial_count),
        )
        predicted = tuple(
            derive_probe_trial_seed(probe_seed, trial_idx)
            for trial_idx in range(int(fresh_trial_count))
        )
        if occupied.isdisjoint(predicted):
            return probe_seed, predicted
    raise RuntimeError("could not allocate disjoint layerwise promotion trial seeds")


def promote_candidate_if_eligible(
        *,
        env: Any,
        promotion_base_env: Optional[Any] = None,
        candidate_store: CandidateStore,
        action_indices: Sequence[int],
        identity_context: Mapping[str, Any],
        action_matrix: Sequence[Sequence[int]],
        assessment: Any,
        priority: int,
        variable_cost: Optional[float],
        frontier_cost: Optional[float],
        frontier_candidates: Optional[Mapping[str, Mapping[str, Any]]] = None,
        boosted_overrides: Mapping[Any, Any],
        bootstrap_seed: int,
        episode_reward: Optional[float] = None,
        assess_candidate_fn: Callable[..., Any] = assess_candidate,
        prefilter_probability: Optional[float] = None,
        promotion_probability: float = 0.80,
        target_trial_count: int = 25,
        ) -> PromotionResult:
    """Promote one robust frontier improvement using fresh real probes."""
    full_identity_context = evidence_identity_context(identity_context, "F4")
    full_base_env = promotion_base_env or env.base
    evidence = candidate_store.trial_evidence_for_action(
        action_indices, full_identity_context,
        max_trials=int(target_trial_count),
    )
    trial_count = candidate_store.trial_count_for_action(
        action_indices, full_identity_context,
    )
    pooled_metrics = _metrics_from_trials(evidence.trials) if evidence is not None else None
    if evidence is not None and evidence.promoted:
        authoritative_assessment = assess_candidate_fn(
            evidence.trials,
            full_base_env.statistical_reference,
            gate_probability=float(promotion_probability),
            bootstrap_seed=int(bootstrap_seed),
        )
        return PromotionResult(
            "already_promoted",
            trial_count,
            0,
            evidence,
            authoritative_assessment,
            pooled_metrics,
        )
    if int(priority) != 3:
        return PromotionResult(
            "priority_not_p3", trial_count, 0, evidence, assessment, pooled_metrics,
        )
    prefilter_gate = (
        float(promotion_probability)
        if prefilter_probability is None else float(prefilter_probability)
    )
    if not 0.0 < prefilter_gate <= float(promotion_probability):
        raise ValueError(
            "prefilter_probability must be in (0, promotion_probability]"
        )
    if not _assessment_passes(assessment, prefilter_gate):
        return PromotionResult(
            "promotion_probability_below_gate", trial_count, 0,
            evidence, assessment, pooled_metrics,
        )
    resource = _resource_fields_from_action_matrix(action_matrix)
    cost = float(resource["ppo_resource_score"])
    dominated = False
    if frontier_candidates is not None:
        compute = float(resource["compute_saving"])
        communication = float(resource["communication_saving"])
        dominated = any(
            other["compute_saving"] >= compute - 1.0e-12
            and other["communication_saving"] >= communication - 1.0e-12
            and (
                other["compute_saving"] > compute + 1.0e-12
                or other["communication_saving"] > communication + 1.0e-12
            )
            for other in (
                _candidate_resource_fields(candidate)
                for candidate in frontier_candidates.values()
            )
        )
    elif frontier_cost is not None:
        # Compatibility path for isolated pre-v9 test fixtures only.  The live
        # trainer always supplies frontier_candidates and uses F/C dominance.
        legacy_cost = _finite(variable_cost, name="variable_cost")
        dominated = legacy_cost < float(frontier_cost) - 1.0e-12
    if dominated:
        return PromotionResult(
            (
                "resource_dominated"
                if frontier_candidates is not None
                else "not_frontier_improvement"
            ),
            trial_count, 0,
            evidence, assessment, pooled_metrics,
        )
    target = int(target_trial_count)
    if target <= 0:
        raise ValueError("target_trial_count must be positive")
    pending_reassessment = bool(
        evidence is not None
        and evidence.promotion_attempted
        and not evidence.promotion_status
        and trial_count >= target
    )
    if evidence is not None and evidence.promotion_attempted and not pending_reassessment:
        return PromotionResult(
            "promotion_already_attempted", trial_count, 0,
            evidence, assessment, pooled_metrics,
        )

    fresh_count = max(0, target - trial_count)
    status_metadata = {
        "existing_trial_count": int(trial_count),
        "requested_fresh_trial_count": int(fresh_count),
        **resource,
        "variable_cost": float(cost),
        "assessment_bootstrap_seed": int(bootstrap_seed),
        "action_matrix": [list(map(int, row)) for row in action_matrix],
        "boosted_overrides": _serialize_boosted_overrides(boosted_overrides),
    }
    if episode_reward is not None:
        status_metadata["episode_reward"] = float(episode_reward)
    promotion_probe_seed: Optional[int] = None
    predicted_trial_seeds: tuple[int, ...] = ()
    if fresh_count:
        promotion_probe_seed, predicted_trial_seeds = _promotion_probe_seed(
            (
                evidence.candidate_key
                if evidence is not None
                else candidate_key(action_indices, full_identity_context)
            ),
            bootstrap_seed,
            trial_count,
            (() if evidence is None else evidence.trials.seeds),
            fresh_count,
        )
    try:
        if fresh_count:
            online_clear = getattr(env.base, "clear_installed_blb", None)
            if full_base_env is not env.base and callable(online_clear):
                online_clear()
                env.base._installed_action_hash = None
            previous_probe_seed = getattr(full_base_env, "probe_noise_seed", None)
            full_base_env.probe_noise_seed = promotion_probe_seed
            try:
                prepared = full_base_env.prepare_action_for_terminal_probe(
                    list(action_indices),
                    external_cost_score=cost,
                    external_cost_rank=cost,
                    external_resource_objective=resource,
                    boosted_overrides=copy.deepcopy(dict(boosted_overrides)),
                )
                evaluated = full_base_env.evaluate_prepared_terminal_batch(
                    [prepared],
                    num_trials_per_action=fresh_count,
                    validation_required=True,
                )
            finally:
                full_base_env.probe_noise_seed = previous_probe_seed
                if full_base_env is not env.base:
                    full_clear = getattr(full_base_env, "clear_installed_blb", None)
                    if callable(full_clear):
                        full_clear()
                    if callable(online_clear):
                        online_clear()
                    env.base._installed_action_hash = None
            if len(evaluated) != 1:
                raise RuntimeError(
                    f"promotion expected one terminal result, received {len(evaluated)}"
                )
            terminal_info = evaluated[0][3]
            if not isinstance(terminal_info, Mapping) or bool(terminal_info.get("invalid", False)):
                raise RuntimeError("promotion terminal evaluation was invalid")
            fresh_trials = _trial_series_from_info(
                terminal_info,
                required=True,
                expected_count=fresh_count,
                context="promotion terminal",
            )
            if tuple(fresh_trials.seeds) != predicted_trial_seeds:
                raise RuntimeError(
                    "promotion terminal trial seeds did not match the predicted fresh set"
                )
            candidate_store.append_trial_group(
                action_indices,
                fresh_trials,
                {
                    "identity_context": full_identity_context,
                    "fidelity": "F4",
                    **resource,
                    "variable_cost": float(cost),
                    "action_matrix": [list(map(int, row)) for row in action_matrix],
                    "boosted_overrides_hash": sha256_json(boosted_overrides),
                    "boosted_overrides": _serialize_boosted_overrides(boosted_overrides),
                    "boosted_overrides_provenance": "layerwise_env",
                    "assessment_bootstrap_seed": int(bootstrap_seed),
                    "promotion_marker": "fresh_top_up",
                    "promotion_status": "pending_reassessment",
                },
            )
        evidence = candidate_store.trial_evidence_for_action(
            action_indices, full_identity_context,
            max_trials=target,
        )
        trial_count = candidate_store.trial_count_for_action(
            action_indices, full_identity_context,
        )
        if evidence is None or trial_count < target:
            raise RuntimeError(
                f"promotion evidence count {trial_count} "
                f"is below target {target}"
            )
        pooled_assessment = assess_candidate_fn(
            evidence.trials,
            full_base_env.statistical_reference,
            gate_probability=float(promotion_probability),
            bootstrap_seed=int(bootstrap_seed),
        )
        promotion_status = (
            "promoted" if _assessment_passes(pooled_assessment, promotion_probability)
            else "failed_probability_gate"
        )
    except Exception as exc:
        promotion_status = "failed_evaluation"
        pooled_assessment = assessment
        status_metadata["error"] = str(exc)

    _append_promotion_status(
        candidate_store,
        action_indices,
        full_identity_context,
        status=promotion_status,
        metadata=status_metadata,
    )
    evidence = candidate_store.trial_evidence_for_action(
        action_indices, full_identity_context,
        max_trials=target,
    )
    total_trial_count = candidate_store.trial_count_for_action(
        action_indices, full_identity_context,
    )
    return PromotionResult(
        status=promotion_status,
        trial_count=total_trial_count,
        fresh_trial_count=(fresh_count if evidence is not None and total_trial_count >= target else 0),
        evidence=evidence,
        assessment=pooled_assessment,
        metrics=(_metrics_from_trials(evidence.trials) if evidence is not None else pooled_metrics),
    )


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def _first_detached_scalar(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.reshape(-1)[0].detach().reshape(())
    return float(np.asarray(value).reshape(-1)[0])


def _policy_input(value: np.ndarray, device: Any) -> Any:
    try:
        import torch
    except ImportError:
        return value
    return torch.as_tensor(value, device=device)


def _current_policy_entropy(
        policy: Any,
        samples: Sequence[Mapping[str, np.ndarray]],
        device: Any,
        ) -> dict[str, float | int | None]:
    if not samples or not hasattr(policy, "evaluate_action"):
        return {
            "block4": None, "k": None,
            "block4_slot_count": 0, "k_slot_count": 0,
        }
    import torch

    states = torch.as_tensor(np.stack([row["state"] for row in samples]), device=device)
    actions = torch.as_tensor(np.stack([row["action"] for row in samples]), device=device)
    masks = torch.as_tensor(np.stack([row["slot_mask"] for row in samples]), device=device)
    levels = torch.as_tensor(np.stack([row["levels"] for row in samples]), device=device)
    with torch.no_grad():
        evaluated = policy.evaluate_action(
            states,
            actions,
            masks,
            levels,
            return_per_slot_entropy=True,
        )
    entropy_per_slot = _as_numpy(evaluated[3])
    return normalized_entropy_snapshot(
        entropy_per_slot,
        _as_numpy(masks),
        _as_numpy(levels),
    )


def train_layerwise(
        *,
        env: Any,
        policy: Any,
        train_cfg: Any,
        candidate_store: CandidateStore,
        identity_context: Optional[Mapping[str, Any]] = None,
        promotion_base_env: Optional[Any] = None,
        on_episode_end: Optional[Callable[[LayerwiseEpisodeRecord], None]] = None,
        on_ppo_update_end: Optional[
            Callable[[Mapping[str, Any], int, LayerwiseEpisodeRecord], None]
        ] = None,
        device: Any = None,
        optimizer: Any = None,
        rollout_buffer: Any = None,
        ppo_update_fn: Optional[Callable[..., Mapping[str, Any]]] = None,
        assess_candidate_fn: Callable[..., Any] = assess_candidate,
        step_adapter_fn: Optional[Callable[[Any, int, int], tuple[np.ndarray, np.ndarray]]] = None,
        ) -> dict[str, Any]:
    """Collect 12-step layerwise episodes and update the shared PPO policy."""
    if identity_context is None:
        raise ValueError("layerwise training requires a CandidateStore identity_context")
    probe_identity_context = evidence_identity_context(identity_context, "F1")
    authoritative_base_env = promotion_base_env or env.base
    if int(getattr(env, "horizon", 0)) != 12 or int(getattr(env, "max_step_dim", 0)) != 6:
        raise ValueError("layerwise training requires horizon=12 and max_step_dim=6")
    if device is None:
        try:
            device = next(policy.parameters()).device
        except (AttributeError, StopIteration, TypeError):
            device = None
    if optimizer is None:
        import torch

        optimizer = torch.optim.Adam(
            policy.parameters(), lr=float(getattr(train_cfg.ppo, "lr", 5.0e-5)),
        )
    if rollout_buffer is None:
        from .sequential_policy import SequentialRolloutBuffer

        rollout_buffer = SequentialRolloutBuffer()
    if ppo_update_fn is None:
        from .sequential_policy import sequential_ppo_update

        ppo_update_fn = sequential_ppo_update
    if step_adapter_fn is None:
        from .sequential_policy import step_to_mask_and_levels

        step_adapter_fn = step_to_mask_and_levels

    ppo_cfg = copy.copy(train_cfg.ppo)
    ppo_cfg.gamma = 1.0
    ppo_cfg.gae_lambda = 1.0
    update_window = max(1, int(getattr(train_cfg, "update_every_n_episodes", 120)))
    total_episodes = int(getattr(train_cfg, "total_episodes", 0))
    planned_total_episodes = getattr(train_cfg, "planned_total_episodes", None)
    unbounded_training = is_unbounded_layerwise_training(
        total_episodes,
        planned_total_episodes,
    )
    convergence_patience_updates = int(getattr(
        train_cfg,
        "convergence_patience_updates",
        DEFAULT_CONVERGENCE_PATIENCE_UPDATES,
    ))
    absolute_start = int(getattr(train_cfg, "absolute_episode_start", 0))
    base_seed = getattr(train_cfg, "seed", None)
    expected_online_trials = int(
        getattr(train_cfg, "online_num_trials_per_step", 5)
    )
    if expected_online_trials <= 0:
        raise ValueError("online_num_trials_per_step must be positive")
    online_probability = float(
        getattr(train_cfg, "online_constraint_probability", 0.50)
    )
    promotion_probability = float(
        getattr(train_cfg, "promotion_constraint_probability", 0.80)
    )
    final_probability = float(
        getattr(train_cfg, "final_constraint_probability", 0.95)
    )
    promotion_trials = int(getattr(train_cfg, "promotion_validation_trials", 25))
    final_validation_trials = int(
        getattr(train_cfg, "final_selection_validation_trials", 25)
    )
    if not 0.0 < online_probability <= promotion_probability <= final_probability <= 1.0:
        raise ValueError(
            "constraint probabilities must satisfy "
            "0 < online <= promotion <= final <= 1"
        )
    if promotion_trials < expected_online_trials:
        raise ValueError("promotion_validation_trials must cover the online trial group")
    if final_validation_trials < promotion_trials:
        raise ValueError(
            "final_selection_validation_trials must be at least "
            "promotion_validation_trials"
        )
    policy.eval()

    records: list[LayerwiseEpisodeRecord] = []
    rewards: list[float] = []
    ppo_diagnostics: list[dict[str, Any]] = []
    accepted_candidates = restore_promoted_candidates(
        candidate_store=candidate_store,
        identity_context=identity_context,
        statistical_reference=authoritative_base_env.statistical_reference,
        assess_candidate_fn=assess_candidate_fn,
        promotion_probability=promotion_probability,
        assessment_trial_limit=promotion_trials,
        final_probability=final_probability,
        final_assessment_trial_limit=final_validation_trials,
    )
    accepted_candidates = dict(
        strict_resource_pareto_frontier(accepted_candidates)
    )
    convergence_resume_state = getattr(train_cfg, "convergence_resume_state", None)
    convergence_resume_state = (
        dict(convergence_resume_state)
        if isinstance(convergence_resume_state, Mapping) else {}
    )
    convergence_tracker = LayerwiseConvergenceTracker(
        patience_updates=convergence_patience_updates,
    )
    convergence_tracker.load_state_dict(convergence_resume_state)
    restored_strict_best = _strict_best_snapshot(accepted_candidates)
    restored_frontier_objective = (
        None if restored_strict_best is None
        else (
            float(restored_strict_best["robust_floor"]),
            float(restored_strict_best["secondary_progress"]),
        )
    )
    restored_selected_identity = (
        None if restored_strict_best is None
        else str(restored_strict_best["candidate_key"])
    )
    convergence_tracker.reconcile_frontier(
        restored_frontier_objective,
        restored_selected_identity,
    )
    restored_tracker_state = convergence_tracker.state_dict()
    restored_block4_entropy = convergence_resume_state.get("block4_entropy")
    restored_k_entropy = convergence_resume_state.get("k_entropy")
    restored_converged = bool(
        unbounded_training
        and restored_frontier_objective is not None
        and restored_selected_identity is not None
        and restored_tracker_state["selected_action_identity"]
        == restored_selected_identity
        and int(restored_tracker_state["stall_update_windows"])
        >= convergence_patience_updates
        and int(
            restored_tracker_state["selected_action_stable_update_windows"]
        ) >= convergence_patience_updates
        and convergence_resume_state.get("strict_revalidation_passed", False)
        and convergence_resume_state.get("converged", False)
    )
    restored_plateau_ready = bool(
        restored_frontier_objective is not None
        and restored_selected_identity is not None
        and int(restored_tracker_state["stall_update_windows"])
        >= convergence_patience_updates
        and int(
            restored_tracker_state["selected_action_stable_update_windows"]
        ) >= convergence_patience_updates
    )
    restored_revalidation_passed = bool(
        restored_converged
        and convergence_resume_state.get("strict_revalidation_passed", False)
    )
    strict_revalidation_status = str(
        convergence_resume_state.get("strict_revalidation_status", "not_due")
    )
    convergence_state = LayerwiseConvergenceState(
        completed_episodes=absolute_start,
        block4_entropy=_diagnostic_entropy(restored_block4_entropy),
        k_entropy=_diagnostic_entropy(restored_k_entropy),
        stall_update_windows=int(restored_tracker_state["stall_update_windows"]),
        best_robust_feasible_objective=(
            None
            if restored_tracker_state["best_robust_feasible_objective"] is None
            else tuple(restored_tracker_state["best_robust_feasible_objective"])
        ),
        selected_action_identity=restored_tracker_state["selected_action_identity"],
        selected_action_stable_update_windows=int(
            restored_tracker_state["selected_action_stable_update_windows"]
        ),
        converged=restored_converged,
        extension_required=False,
        plateau_ready=restored_plateau_ready,
        strict_revalidation_passed=restored_revalidation_passed,
        termination_reason=("converged" if restored_converged else "running"),
    )
    entropy_samples: list[dict[str, np.ndarray]] = []

    local_episode = 0
    while not convergence_state.converged and (
            unbounded_training or local_episode < total_episodes
    ):
        absolute_episode = absolute_start + local_episode
        state = env.reset(
            seed=(None if base_seed is None else int(base_seed) + absolute_episode)
        )
        if base_seed is not None and hasattr(env, "base"):
            from .seed_utils import derive_layerwise_episode_probe_seed

            env.base.probe_noise_seed = derive_layerwise_episode_probe_seed(
                int(base_seed),
                absolute_episode,
                trial_count=expected_online_trials,
            )
        step_infos: list[Mapping[str, Any]] = []
        transition_indices: list[int] = []
        terminal_info: Optional[Mapping[str, Any]] = None
        episode_reward = 0.0
        for step_idx in range(12):
            spec = env.current_spec()
            slot_mask, levels = step_adapter_fn(spec, 6, 6)
            state_np = np.asarray(state, dtype=np.float32)
            sample_out = policy.sample_action(
                _policy_input(state_np[None, ...], device),
                _policy_input(slot_mask[None, ...], device),
                _policy_input(levels[None, ...], device),
                deterministic=False,
                baseline_prior_scale=0.0,
                return_per_slot_log_prob=True,
            )
            if len(sample_out) != 4:
                raise RuntimeError(
                    "layerwise factorized PPO requires sampling-time per-slot log probabilities"
                )
            actions_raw, log_prob_raw, value_raw, log_prob_per_slot_raw = sample_out
            action = _as_numpy(actions_raw).reshape(-1).astype(np.int64)
            action[~slot_mask] = 0
            log_prob = _first_detached_scalar(log_prob_raw)
            value = _first_detached_scalar(value_raw)
            log_prob_per_slot = (
                log_prob_per_slot_raw.detach().reshape(-1)
                if hasattr(log_prob_per_slot_raw, "detach")
                else np.asarray(log_prob_per_slot_raw, dtype=np.float32).reshape(-1)
            )
            next_state, reward, done, info = env.step(action.tolist())
            expected_done = step_idx == 11
            if bool(done) != expected_done:
                raise RuntimeError(
                    f"layerwise episode termination mismatch at step {step_idx}: done={done}"
                )
            transition_index = rollout_buffer.add(
                state=state_np,
                action=action,
                slot_mask=slot_mask,
                per_slot_num_levels=levels,
                action_level_mask=None,
                log_prob=log_prob,
                log_prob_per_slot=log_prob_per_slot,
                value=value,
                reward=0.0,
                done=bool(done),
                baseline_prior_scale=0.0,
            )
            transition_indices.append(int(transition_index))
            entropy_samples.append({
                "state": state_np.copy(),
                "action": action.copy(),
                "slot_mask": slot_mask.copy(),
                "levels": levels.copy(),
            })
            step_infos.append(info if isinstance(info, Mapping) else {})
            state = next_state
            if expected_done:
                terminal_info = info
                episode_reward = float(reward)

        if terminal_info is None:
            raise RuntimeError("layerwise episode completed without terminal info")
        runtime_info = getattr(env, "runtime_terminal_info", None)
        runtime_info = runtime_info if isinstance(runtime_info, Mapping) else {}
        action_matrix = tuple(
            tuple(int(value) for value in row)
            for row in terminal_info.get("policy_actions", ())
        )
        if len(action_matrix) != 12 or any(len(row) != 6 for row in action_matrix):
            raise RuntimeError("layerwise terminal policy_actions must be a 12x6 matrix")
        full_vector = tuple(
            int(value) for value in terminal_info.get("pending_full_vector", ())
        )
        if not full_vector:
            raise RuntimeError("layerwise terminal pending_full_vector is required")
        resource_objective = terminal_info.get("resource_objective", {})
        ppo_resource_score = _finite(
            _field(resource_objective, "ppo_resource_score"),
            name="ppo_resource_score",
        )
        compute_shapley_credit = _finite(
            _field(resource_objective, "compute_shapley_credit"),
            name="compute_shapley_credit",
        )
        communication_shapley_credit = _finite(
            _field(resource_objective, "communication_shapley_credit"),
            name="communication_shapley_credit",
        )
        layer_resource_rewards = tuple(
            float(value) for value in _field(
                resource_objective,
                "layer_resource_rewards",
                (),
            )
        )
        slot_resource_rewards = tuple(
            tuple(float(value) for value in row)
            for row in _field(
                resource_objective,
                "slot_resource_rewards",
                (),
            )
        )
        if len(layer_resource_rewards) != 12:
            raise RuntimeError(
                "layerwise terminal layer_resource_rewards must contain 12 values"
            )
        if len(slot_resource_rewards) != 12 or any(
                len(row) != 6 for row in slot_resource_rewards
        ):
            raise RuntimeError(
                "layerwise terminal slot_resource_rewards must be a 12x6 matrix"
            )
        for layer_idx, (layer_resource, slot_resources) in enumerate(zip(
                layer_resource_rewards, slot_resource_rewards,
        )):
            if not math.isclose(
                    sum(slot_resources), layer_resource,
                    rel_tol=0.0, abs_tol=1.0e-9,
            ):
                raise RuntimeError(
                    f"layer {layer_idx} slot resource sum does not match layer resource"
                )
        if not math.isclose(
                compute_shapley_credit + communication_shapley_credit,
                ppo_resource_score,
                rel_tol=0.0,
                abs_tol=1.0e-9,
        ):
            raise RuntimeError("resource-family credits do not sum to PPO resource score")
        if not math.isclose(
                sum(row[0] for row in slot_resource_rewards),
                compute_shapley_credit,
                rel_tol=0.0,
                abs_tol=1.0e-9,
        ):
            raise RuntimeError("fusion slots do not sum to compute Shapley credit")
        if not math.isclose(
                sum(sum(row[1:]) for row in slot_resource_rewards),
                communication_shapley_credit,
                rel_tol=0.0,
                abs_tol=1.0e-9,
        ):
            raise RuntimeError("K slots do not sum to communication Shapley credit")
        exact_resource = _resource_fields_from_action_matrix(action_matrix)
        for field_name in (
                "compute_saving",
                "communication_saving",
                "robust_floor",
                "secondary_progress",
                "ppo_resource_score",
                "compute_shapley_credit",
                "communication_shapley_credit",
        ):
            observed = _finite(
                _field(resource_objective, field_name), name=field_name,
            )
            if not math.isclose(
                    observed, float(exact_resource[field_name]),
                    rel_tol=0.0, abs_tol=1.0e-9,
            ):
                raise RuntimeError(
                    f"terminal {field_name} does not match action-matrix objective"
                )
        if not np.allclose(
                np.asarray(slot_resource_rewards, dtype=np.float64),
                np.asarray(exact_resource["slot_resource_rewards"], dtype=np.float64),
                rtol=0.0,
                atol=1.0e-9,
        ):
            raise RuntimeError(
                "terminal slot_resource_rewards do not match action-matrix objective"
            )
        ppo_resource_score = float(exact_resource["ppo_resource_score"])
        variable_cost = ppo_resource_score
        breakdown = runtime_info.get("reward_breakdown")
        priority = int(_field(breakdown, "priority", runtime_info.get("priority", 0)))
        invalid_terminal = bool(runtime_info.get("invalid", False))
        if invalid_terminal and not math.isclose(episode_reward, -5.0, abs_tol=1.0e-9):
            raise RuntimeError(
                f"invalid layerwise terminal reward must be -5, got {episode_reward}"
            )
        reward_priority = 0 if invalid_terminal else priority
        redistributed_rewards = redistribute_layerwise_rewards(
            terminal_reward=episode_reward,
            priority=reward_priority,
            ppo_resource_score=ppo_resource_score,
            layer_resource_rewards=layer_resource_rewards,
        )
        zero_slot_resources = ((0.0,) * 6,) * 12
        actor_slot_resources = (
            slot_resource_rewards if reward_priority == 3 else zero_slot_resources
        )
        actor_shared_return = (
            episode_reward - ppo_resource_score
            if reward_priority == 3 else episode_reward
        )
        for transition_index, reward_delta, per_slot_resource in zip(
                transition_indices, redistributed_rewards, actor_slot_resources,
        ):
            rollout_buffer.add_reward_at(transition_index, reward_delta)
            rollout_buffer.set_actor_cost_at(transition_index, per_slot_resource)
            rollout_buffer.set_actor_shared_return_at(
                transition_index, actor_shared_return,
            )
        raw_trials = (
            None
            if invalid_terminal
            else _trial_series_from_info(
                runtime_info,
                required=True,
                expected_count=expected_online_trials,
                context="valid robust terminal",
            )
        )
        fresh_assessment = _to_plain_mapping(runtime_info.get("statistical_assessment"))
        metrics = _to_plain_mapping(runtime_info.get("metrics"))
        bootstrap_seed = int(fresh_assessment.get("bootstrap_seed", 0))
        pooled_assessment = None
        pooled_metrics = None
        pooled_trials = None
        promotion = PromotionResult(
            "invalid_terminal" if invalid_terminal else "not_evaluated",
            0, 0, None, None, None,
        )
        if raw_trials is not None:
            candidate_store.append_trial_group(
                full_vector,
                raw_trials,
                {
                    "identity_context": probe_identity_context,
                    "fidelity": "F1",
                    "episode_index": int(absolute_episode),
                    **exact_resource,
                    "variable_cost": float(variable_cost),
                    "action_matrix": [list(row) for row in action_matrix],
                    "boosted_overrides_hash": sha256_json(
                        getattr(env, "boosted_overrides", {})
                    ),
                    "boosted_overrides": _serialize_boosted_overrides(
                        getattr(env, "boosted_overrides", {})
                    ),
                    "boosted_overrides_provenance": "layerwise_env",
                    "assessment_bootstrap_seed": int(bootstrap_seed),
                    "episode_reward": float(episode_reward),
                    "promotion_marker": "online_group",
                },
            )
            evidence = candidate_store.trial_evidence_for_action(
                full_vector, probe_identity_context,
                max_trials=promotion_trials,
            )
            if evidence is None:
                raise RuntimeError("candidate evidence append was not readable")
            pooled_assessment = assess_candidate_fn(
                evidence.trials,
                env.base.statistical_reference,
                gate_probability=promotion_probability,
                bootstrap_seed=bootstrap_seed,
            )
            pooled_metrics = _metrics_from_trials(evidence.trials)
            pooled_trials = evidence.trials
            promotion = promote_candidate_if_eligible(
                env=env,
                promotion_base_env=authoritative_base_env,
                candidate_store=candidate_store,
                action_indices=full_vector,
                identity_context=identity_context,
                action_matrix=action_matrix,
                assessment=pooled_assessment,
                priority=priority,
                variable_cost=variable_cost,
                frontier_cost=None,
                frontier_candidates=accepted_candidates,
                boosted_overrides=getattr(env, "boosted_overrides", {}),
                bootstrap_seed=bootstrap_seed,
                episode_reward=episode_reward,
                assess_candidate_fn=assess_candidate_fn,
                prefilter_probability=online_probability,
                promotion_probability=promotion_probability,
                target_trial_count=promotion_trials,
            )
            promotion_evidence = promotion.evidence or evidence
            candidate_key_value = promotion_evidence.candidate_key
            if (
                    promotion.evidence is not None
                    and promotion.evidence.promoted
                    and promotion_evidence.trial_count >= promotion_trials
                    and _assessment_passes(
                        promotion.assessment, promotion_probability,
                    )
            ):
                existing_candidate = accepted_candidates.get(candidate_key_value)
                accepted_candidates[candidate_key_value] = {
                    **exact_resource,
                    "variable_cost": float(variable_cost),
                    "assessment": promotion.assessment,
                    "metrics": dict(promotion.metrics or {}),
                    "constraint_safety_margins": (
                        normalized_constraint_safety_margins(
                            promotion.metrics or {},
                            authoritative_base_env.statistical_reference,
                        )
                    ),
                    "action_matrix": action_matrix,
                    "full_vector": full_vector,
                    "boosted_overrides": copy.deepcopy(
                        getattr(env, "boosted_overrides", {})
                    ),
                    "reward": (
                        float(existing_candidate["reward"])
                        if existing_candidate is not None
                        and existing_candidate.get("reward") is not None
                        else float(episode_reward)
                    ),
                    "promotion_trials": promotion.evidence.trials,
                }
                accepted_candidates = dict(
                    strict_resource_pareto_frontier(accepted_candidates)
                )

        local_episode += 1
        completed = local_episode
        # The environment's direct terminal return is the PPO source of truth.
        rewards.append(episode_reward)
        entropy_snapshot = {
            "block4": None, "k": None,
            "block4_slot_count": 0, "k_slot_count": 0,
        }
        update_due = completed % update_window == 0 or (
            not unbounded_training and completed == total_episodes
        )
        ppo_metrics: Optional[dict[str, Any]] = None
        if update_due:
            ppo_metrics = dict(ppo_update_fn(
                policy,
                optimizer,
                rollout_buffer,
                ppo_cfg,
                device,
                ent_coef_override=0.0,
            ))
            convergence_update_counted = bool(
                int(ppo_metrics.get("nonfinite_minibatches", 0) or 0) == 0
                and not bool(ppo_metrics.get("nonfinite_update_skipped", False))
            )
            entropy_snapshot = _current_policy_entropy(policy, entropy_samples, device)
            strict_best_snapshot = _strict_best_snapshot(accepted_candidates)
            best_objective = (
                None if strict_best_snapshot is None
                else (
                    float(strict_best_snapshot["robust_floor"]),
                    float(strict_best_snapshot["secondary_progress"]),
                )
            )
            best_action_identity = (
                None if strict_best_snapshot is None
                else str(strict_best_snapshot["candidate_key"])
            )
            convergence_state = convergence_tracker.observe_update(
                completed_episodes=absolute_start + completed,
                block4_entropy=entropy_snapshot["block4"],
                k_entropy=entropy_snapshot["k"],
                robust_feasible_objective=best_objective,
                robust_feasible_action_identity=best_action_identity,
                count_patience=convergence_update_counted,
            )
            strict_revalidation_status = "not_due"
            if (
                    unbounded_training
                    and convergence_state.plateau_ready
                    and strict_best_snapshot is not None
            ):
                revalidation_context = {
                    **dict(identity_context),
                    "convergence_revalidation_update": int(
                        absolute_start + completed
                    ),
                    "convergence_revalidation_candidate": str(
                        strict_best_snapshot["candidate_key"]
                    ),
                }
                revalidation_bootstrap_seed = int(base_seed or 0) + int(
                    absolute_start + completed
                )
                revalidation = promote_candidate_if_eligible(
                    env=env,
                    promotion_base_env=authoritative_base_env,
                    candidate_store=candidate_store,
                    action_indices=strict_best_snapshot["full_vector"],
                    identity_context=revalidation_context,
                    action_matrix=strict_best_snapshot["action_matrix"],
                    assessment=strict_best_snapshot["assessment"],
                    priority=3,
                    variable_cost=float(strict_best_snapshot["variable_cost"]),
                    frontier_cost=None,
                    frontier_candidates=None,
                    boosted_overrides=strict_best_snapshot["boosted_overrides"],
                    bootstrap_seed=revalidation_bootstrap_seed,
                    episode_reward=strict_best_snapshot.get("reward"),
                    assess_candidate_fn=assess_candidate_fn,
                    prefilter_probability=promotion_probability,
                    promotion_probability=final_probability,
                    target_trial_count=final_validation_trials,
                )
                revalidation_passed = bool(
                    revalidation.evidence is not None
                    and revalidation.evidence.promoted
                    and int(revalidation.trial_count) >= final_validation_trials
                    and _assessment_passes(
                        revalidation.assessment, final_probability,
                    )
                )
                selected_key = str(strict_best_snapshot["candidate_key"])
                revalidation_status = str(revalidation.status)
                completed_probability_verdict = bool(
                    revalidation_passed
                    or revalidation_status == "failed_probability_gate"
                )
                if completed_probability_verdict:
                    _record_final_revalidation_outcome(
                        candidate_store=candidate_store,
                        identity_context=identity_context,
                        revalidation_identity_context=revalidation_context,
                        candidate=strict_best_snapshot,
                        passed=revalidation_passed,
                        revalidation_status=revalidation_status,
                        bootstrap_seed=revalidation_bootstrap_seed,
                        final_probability=final_probability,
                        final_trial_count=final_validation_trials,
                    )
                if revalidation_passed:
                    selected = accepted_candidates[selected_key]
                    selected["assessment"] = revalidation.assessment
                    selected["metrics"] = dict(revalidation.metrics or {})
                    selected["constraint_safety_margins"] = (
                        normalized_constraint_safety_margins(
                            selected["metrics"],
                            authoritative_base_env.statistical_reference,
                        )
                    )
                    selected["promotion_trials"] = revalidation.evidence.trials
                    selected["final_revalidation_status"] = "passed"
                    strict_best_snapshot = _strict_best_snapshot(accepted_candidates)
                    if (
                            strict_best_snapshot is not None
                            and strict_best_snapshot["candidate_key"] == selected_key
                    ):
                        strict_revalidation_status = "passed"
                        convergence_state = convergence_tracker.observe_update(
                            completed_episodes=absolute_start + completed,
                            block4_entropy=entropy_snapshot["block4"],
                            k_entropy=entropy_snapshot["k"],
                            robust_feasible_objective=(
                                float(strict_best_snapshot["robust_floor"]),
                                float(strict_best_snapshot["secondary_progress"]),
                            ),
                            robust_feasible_action_identity=selected_key,
                            count_patience=False,
                            strict_revalidation_passed=True,
                        )
                    else:
                        strict_revalidation_status = (
                            "winner_changed_after_revalidation"
                        )
                else:
                    strict_revalidation_status = revalidation_status
                    if revalidation_status == "failed_probability_gate":
                        accepted_candidates.pop(selected_key, None)
                    strict_best_snapshot = _strict_best_snapshot(accepted_candidates)

                if not convergence_state.converged:
                    best_objective = (
                        None if strict_best_snapshot is None
                        else (
                            float(strict_best_snapshot["robust_floor"]),
                            float(strict_best_snapshot["secondary_progress"]),
                        )
                    )
                    best_action_identity = (
                        None if strict_best_snapshot is None
                        else str(strict_best_snapshot["candidate_key"])
                    )
                    convergence_tracker.reconcile_frontier(
                        best_objective, best_action_identity,
                    )
                    convergence_state = convergence_tracker.observe_update(
                        completed_episodes=absolute_start + completed,
                        block4_entropy=entropy_snapshot["block4"],
                        k_entropy=entropy_snapshot["k"],
                        robust_feasible_objective=best_objective,
                        robust_feasible_action_identity=best_action_identity,
                        count_patience=False,
                    )
            if (
                    not unbounded_training
                    and completed >= total_episodes
                    and not convergence_state.converged
            ):
                strict_revalidation_status = "not_applicable_bounded"
                convergence_state = replace(
                    convergence_state,
                    strict_revalidation_passed=False,
                    termination_reason="bounded_budget_exhausted",
                )
            persisted_convergence_state = {
                **convergence_tracker.state_dict(),
                "block4_entropy": convergence_state.block4_entropy,
                "k_entropy": convergence_state.k_entropy,
                "converged": convergence_state.converged,
                "extension_required": convergence_state.extension_required,
                "plateau_ready": convergence_state.plateau_ready,
                "strict_revalidation_passed": (
                    convergence_state.strict_revalidation_passed
                ),
                "strict_revalidation_status": strict_revalidation_status,
                "termination_reason": convergence_state.termination_reason,
            }
            ppo_metrics.update({
                "completed_episodes": absolute_start + completed,
                "block4_entropy": entropy_snapshot["block4"],
                "k_entropy": entropy_snapshot["k"],
                "stall_update_windows": convergence_state.stall_update_windows,
                "selected_action_identity": convergence_state.selected_action_identity,
                "selected_action_stable_update_windows": (
                    convergence_state.selected_action_stable_update_windows
                ),
                "converged": convergence_state.converged,
                "extension_required": convergence_state.extension_required,
                "plateau_ready": convergence_state.plateau_ready,
                "strict_revalidation_passed": (
                    convergence_state.strict_revalidation_passed
                ),
                "strict_revalidation_status": strict_revalidation_status,
                "termination_reason": convergence_state.termination_reason,
                "best_robust_feasible_cost": convergence_state.best_robust_feasible_cost,
                "best_robust_feasible_objective": (
                    None
                    if convergence_state.best_robust_feasible_objective is None
                    else list(convergence_state.best_robust_feasible_objective)
                ),
                "convergence_update_counted": convergence_update_counted,
                "convergence_state": persisted_convergence_state,
                "strict_best": strict_best_snapshot,
                "strict_pareto_frontier": _strict_pareto_snapshots(
                    accepted_candidates
                ),
            })
            ppo_diagnostics.append(ppo_metrics)

        invalid_steps = sum(
            not bool(_field(info.get("layer_summary", {}), "all_valid", True))
            for info in step_infos
        )
        record = LayerwiseEpisodeRecord(
            episode_index=absolute_episode,
            reward=float(rewards[-1]),
            priority=priority,
            action_matrix=action_matrix,
            pending_full_vector=full_vector,
            variable_cost=float(variable_cost),
            compute_saving=float(exact_resource["compute_saving"]),
            communication_saving=float(exact_resource["communication_saving"]),
            robust_floor=float(exact_resource["robust_floor"]),
            secondary_progress=float(exact_resource["secondary_progress"]),
            ppo_resource_score=float(exact_resource["ppo_resource_score"]),
            compute_shapley_credit=float(
                exact_resource["compute_shapley_credit"]
            ),
            communication_shapley_credit=float(
                exact_resource["communication_shapley_credit"]
            ),
            layer_resource_rewards=tuple(
                float(value) for value in exact_resource["layer_resource_rewards"]
            ),
            slot_resource_rewards=tuple(
                tuple(float(value) for value in row)
                for row in exact_resource["slot_resource_rewards"]
            ),
            raw_trials=raw_trials,
            pooled_trials=pooled_trials,
            fresh_trial_count=(0 if raw_trials is None else len(raw_trials.loss)),
            pooled_trial_count=(0 if pooled_trials is None else len(pooled_trials.loss)),
            reward_evidence="fresh_trials",
            ranking_evidence=(
                "F4_validation_full"
                if promotion.evidence is not None else "F1_prefilter_only"
            ),
            fresh_assessment=fresh_assessment or None,
            assessment=pooled_assessment,
            metrics={
                name: float(value)
                for name, value in metrics.items()
                if name in (
                    "loss_mean", "loss_std", "metric1_mean", "metric1_std",
                    "metric2_mean", "metric2_std",
                )
            },
            pooled_metrics=pooled_metrics,
            probe_diagnostics=_to_plain_mapping(
                runtime_info.get("probe_diagnostics")
            ),
            promoted_trial_count=int(promotion.trial_count),
            promotion_status=str(promotion.status),
            promotion_candidate_key=(
                None if promotion.evidence is None
                else str(promotion.evidence.candidate_key)
            ),
            promotion_assessment=(
                None if promotion.assessment is None
                else _to_plain_mapping(promotion.assessment)
            ),
            promotion_metrics=(
                None if promotion.metrics is None
                else dict(promotion.metrics)
            ),
            invalid_steps=int(invalid_steps),
            step_count=12,
            block4_entropy=entropy_snapshot["block4"],
            k_entropy=entropy_snapshot["k"],
            stall_update_windows=int(convergence_state.stall_update_windows),
            selected_action_identity=convergence_state.selected_action_identity,
            selected_action_stable_update_windows=int(
                convergence_state.selected_action_stable_update_windows
            ),
            converged=bool(convergence_state.converged),
            extension_required=bool(convergence_state.extension_required),
            plateau_ready=bool(convergence_state.plateau_ready),
            strict_revalidation_passed=bool(
                convergence_state.strict_revalidation_passed
            ),
            strict_revalidation_status=str(strict_revalidation_status),
            termination_reason=str(convergence_state.termination_reason),
            best_robust_feasible_cost=convergence_state.best_robust_feasible_cost,
            best_robust_feasible_objective=(
                None
                if convergence_state.best_robust_feasible_objective is None
                else tuple(convergence_state.best_robust_feasible_objective)
            ),
        )
        records.append(record)
        if on_episode_end is not None:
            on_episode_end(record)
        if update_due:
            if on_ppo_update_end is not None and ppo_metrics is not None:
                on_ppo_update_end(ppo_metrics, absolute_start + completed, record)
            rollout_buffer.clear()
            entropy_samples.clear()

    if not unbounded_training and not convergence_state.converged:
        strict_revalidation_status = "not_applicable_bounded"
        convergence_state = replace(
            convergence_state,
            strict_revalidation_passed=False,
            termination_reason="bounded_budget_exhausted",
        )
    strict_best = _strict_best_snapshot(accepted_candidates)
    strict_pareto_frontier = _strict_pareto_snapshots(accepted_candidates)
    final_convergence_state = {
        **convergence_tracker.state_dict(),
        "block4_entropy": convergence_state.block4_entropy,
        "k_entropy": convergence_state.k_entropy,
        "converged": convergence_state.converged,
        "extension_required": convergence_state.extension_required,
        "plateau_ready": convergence_state.plateau_ready,
        "strict_revalidation_passed": convergence_state.strict_revalidation_passed,
        "strict_revalidation_status": strict_revalidation_status,
        "termination_reason": convergence_state.termination_reason,
    }
    return {
        "strict_best": strict_best,
        "strict_pareto_frontier": strict_pareto_frontier,
        "convergence_state": final_convergence_state,
        "best_action": (
            list(strict_best["full_vector"]) if strict_best is not None else None
        ),
        "best_action_matrix": (
            [list(row) for row in strict_best["action_matrix"]]
            if strict_best is not None else None
        ),
        "best_full_vector": (
            list(strict_best["full_vector"]) if strict_best is not None else None
        ),
        "best_assessment": (
            dict(strict_best["assessment"]) if strict_best is not None else None
        ),
        "best_metrics": (
            dict(strict_best["metrics"]) if strict_best is not None else None
        ),
        "best_resource_objective": (
            None
            if strict_best is None
            else {
                field_name: copy.deepcopy(strict_best[field_name])
                for field_name in (
                    "compute_saving",
                    "communication_saving",
                    "robust_floor",
                    "secondary_progress",
                    "ppo_resource_score",
                    "compute_shapley_credit",
                    "communication_shapley_credit",
                    "fusion_count",
                    "removed_k_bits",
                    "layer_resource_rewards",
                    "slot_resource_rewards",
                )
            }
        ),
        "best_variable_cost": (
            float(strict_best["variable_cost"]) if strict_best is not None else None
        ),
        "best_reward": (
            None if strict_best is None or strict_best.get("reward") is None
            else float(strict_best["reward"])
        ),
        "best_boosted_overrides": (
            copy.deepcopy(strict_best["boosted_overrides"])
            if strict_best is not None else None
        ),
        "best_promotion_evidence": (
            copy.deepcopy(strict_best.get("promotion_evidence"))
            if strict_best is not None else None
        ),
        "episode_rewards": rewards,
        "ppo_metrics": ppo_diagnostics,
        "episode_records": records,
        "block4_entropy": convergence_state.block4_entropy,
        "k_entropy": convergence_state.k_entropy,
        "stall_update_windows": convergence_state.stall_update_windows,
        "selected_action_identity": convergence_state.selected_action_identity,
        "selected_action_stable_update_windows": (
            convergence_state.selected_action_stable_update_windows
        ),
        "converged": convergence_state.converged,
        "extension_required": convergence_state.extension_required,
        "plateau_ready": convergence_state.plateau_ready,
        "strict_revalidation_passed": convergence_state.strict_revalidation_passed,
        "strict_revalidation_status": strict_revalidation_status,
        "termination_reason": convergence_state.termination_reason,
        "recommended_extension_episodes": 0,
        "completed_episodes": absolute_start + local_episode,
    }
