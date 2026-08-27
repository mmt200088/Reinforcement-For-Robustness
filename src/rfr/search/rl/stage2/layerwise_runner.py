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
import torch
from rfr.preparation.data.protocol import validate_dataset_protocol_binding

from rfr.search.common.candidate_store import (
    CandidateStore,
    CandidateTrialEvidence,
    candidate_key,
    sha256_json,
)
from rfr.search.common.layerwise_action import (
    LAYERWISE_SLOT_NAMES,
    compute_variable_cost_from_action_matrix,
    describe_layerwise_action_matrix,
    materialize_layerwise_counterfactuals,
)
from rfr.search.common.precision_presets import (
    allocated_precision_tolerances,
    network_axis_weights,
    validate_communication_importance_ratio,
)
from rfr.search.common.statistical_constraints import (
    ConstraintAssessment,
    TrialSeries,
    assess_candidate,
    baseline_reference_from_resume_payload,
    baseline_reference_resume_payload,
    retarget_constraint_assessment,
    retarget_precision_tolerance,
)
_PROBABILITY_FIELDS = (
    "loss_precision_probability",
    "metric1_precision_probability",
    "metric2_precision_probability",
    "loss_stability_probability",
    "metric1_stability_probability",
    "metric2_stability_probability",
)
_LAUNCHER_LOCK_FD_ENV = "BLB_STAGE2_RUN_LOCK_FD"
_LAUNCHER_LOCK_PATH_ENV = "BLB_STAGE2_RUN_LOCK_PATH"
LAYERWISE_VALIDATION_BANK_GROUPS = 5
LAYERWISE_VALIDATION_TRIALS_PER_GROUP = 3
LAYERWISE_VALIDATION_BANK_TRIALS = (
    LAYERWISE_VALIDATION_BANK_GROUPS
    * LAYERWISE_VALIDATION_TRIALS_PER_GROUP
)
StrictSelectionKey = tuple[tuple[float, ...], tuple[int, ...], str]
_FINAL_REVALIDATION_PASSED = "final_revalidation_passed"
_FINAL_REVALIDATION_FAILED = "final_revalidation_failed"
_FINAL_REVALIDATION_RETRYABLE = "failed_evaluation"
_STRICT_GATE_CONTRACT = (
    "joint_six_point_plus_compute_and_communication_"
    "counterfactual_six_point_v1"
)


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
            if os.path.basename(parent) == "stage2"
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
        "communication_importance_ratio",
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
    communication_importance_ratio = validate_communication_importance_ratio(
        contract["communication_importance_ratio"],
    )
    compute_weight, communication_weight = network_axis_weights(
        communication_importance_ratio,
    )
    compute_denominator = float(contract["compute_axis_denominator"])
    communication_denominator = float(
        contract["communication_axis_denominator"]
    )
    for field_name, value in (
            ("compute_axis_denominator", compute_denominator),
            ("communication_axis_denominator", communication_denominator),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{field_name} must be finite and positive")
    context = dict(identity_context)
    stage1_binding = contract.get("stage1_selection_binding")
    if stage1_binding is not None:
        if not isinstance(stage1_binding, Mapping) or not stage1_binding:
            raise ValueError(
                "stage1_selection_binding must be a non-empty mapping"
            )
        context["stage1_selection_binding"] = dict(stage1_binding)
    if "stage2_inference_batch_size" in contract:
        stage2_inference_batch_size = int(
            contract["stage2_inference_batch_size"]
        )
        if stage2_inference_batch_size <= 0:
            raise ValueError("stage2_inference_batch_size must be positive")
        context["stage2_inference_batch_size"] = stage2_inference_batch_size
    context["k_levels"] = list(levels)
    context["cost_model_revision"] = str(cost_model_revision)
    context["resource_objective_contract"] = {
        "algorithm_contract_hash": algorithm_hash,
        "communication_importance_ratio": communication_importance_ratio,
        "compute_weight": compute_weight,
        "communication_weight": communication_weight,
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
        dataset_protocol_schema: str,
        dataset_protocol_hash: str,
        ) -> None:
    """Reject checkpoints from a different algorithm or experiment context."""
    validate_dataset_protocol_binding(
        checkpoint,
        expected_hash=dataset_protocol_hash,
        artifact="layerwise checkpoint",
    )
    if str(checkpoint.get("dataset_protocol_schema") or "") != str(
        dataset_protocol_schema
    ):
        raise RuntimeError(
            "layerwise checkpoint train-probe protocol mismatch; start a fresh run"
        )
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


def initialize_layerwise_policy(policy: Any) -> None:
    """Install conservative but fully exploratory priors on both slot heads."""
    policy.set_initial_slot_probabilities(
        [
            {0: 0.60, 1: 0.40},
            {0: 0.60, 1: 0.27, 2: 0.13},
        ],
        [
            (0, 1),
            (0, 1, 2),
        ],
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


def point_constraints_pass(
        metrics: Any,
        statistical_reference: Any,
        *,
        tolerance: float = 1.0e-12,
        ) -> bool:
    """Return whether all loss/m1/m2 mean and std point limits pass."""
    slack = _finite(tolerance, name="point_constraint_tolerance")
    if slack < 0.0:
        raise ValueError("point constraint tolerance must be nonnegative")
    return all(
        margin >= -slack
        for margin in normalized_constraint_safety_margins(
            metrics, statistical_reference,
        )
    )


def validate_layerwise_validation_bank_config(train_cfg: Any) -> tuple[int, int]:
    """Fail before calibration unless the fixed A/B/C 15-trial contract is used."""
    baseline_groups = int(getattr(
        train_cfg, "baseline_groups", LAYERWISE_VALIDATION_BANK_GROUPS,
    ))
    trials_per_group = int(getattr(
        train_cfg,
        "baseline_trials_per_group",
        LAYERWISE_VALIDATION_TRIALS_PER_GROUP,
    ))
    promotion_trials = int(getattr(
        train_cfg,
        "promotion_validation_trials",
        LAYERWISE_VALIDATION_BANK_TRIALS,
    ))
    final_trials = int(
        getattr(
            train_cfg,
            "final_selection_validation_trials",
            LAYERWISE_VALIDATION_BANK_TRIALS,
        )
    )
    if (
            baseline_groups != LAYERWISE_VALIDATION_BANK_GROUPS
            or trials_per_group != LAYERWISE_VALIDATION_TRIALS_PER_GROUP
            or promotion_trials != LAYERWISE_VALIDATION_BANK_TRIALS
            or final_trials != LAYERWISE_VALIDATION_BANK_TRIALS
    ):
        raise ValueError(
            "layerwise validation requires fixed A=15, B=15, C=15 banks "
            "(baseline_groups=5, baseline_trials_per_group=3, "
            "promotion_validation_trials=15, "
            "final_selection_validation_trials=15)"
        )
    return baseline_groups, trials_per_group


def validate_layerwise_episode_limit_extension(
        checkpoint_limit: int,
        requested_limit: int,
        ) -> int:
    """Allow an equal/larger runtime cap without changing search identity."""
    previous = int(checkpoint_limit)
    requested = int(requested_limit)
    if previous <= 0 or requested <= 0:
        raise ValueError("layerwise episode limits must be positive")
    if requested < previous:
        raise RuntimeError(
            "layerwise resume cannot shrink the episode limit: "
            f"checkpoint={previous}, requested={requested}"
        )
    return requested


def _exact_resume_mapping(
        name: str,
        payload: Mapping[str, Any],
        keys: Sequence[str],
        ) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{name} must be a mapping")
    copied = dict(payload)
    required = set(keys)
    actual = set(copied)
    if actual != required:
        missing = sorted(required - actual)
        extra = sorted(repr(key) for key in actual - required)
        raise ValueError(
            f"{name} must contain exactly {tuple(keys)}; "
            f"missing={missing}, extra={extra}"
        )
    return copied


@dataclass(frozen=True)
class LayerwiseValidationBank:
    """One fixed common-random-number validation bank."""

    label: str
    reference: Any
    probe_seeds: Sequence[int]
    trials_per_probe: int

    def __post_init__(self) -> None:
        from rfr.search.rl.stage2.seed_utils import derive_probe_trial_seed

        label = str(self.label).strip().upper()
        if label not in ("A", "B", "C"):
            raise ValueError(f"validation bank label must be A, B, or C, got {label!r}")
        probe_seeds = tuple(int(value) for value in self.probe_seeds)
        trials_per_probe = int(self.trials_per_probe)
        if not probe_seeds or len(set(probe_seeds)) != len(probe_seeds):
            raise ValueError("validation bank probe seeds must be nonempty and unique")
        if trials_per_probe <= 0:
            raise ValueError("validation bank trials_per_probe must be positive")
        reference_trials = _field(self.reference, "trials")
        if not isinstance(reference_trials, TrialSeries):
            raise TypeError("validation bank reference must carry TrialSeries trials")
        expected_trial_seeds = tuple(
            derive_probe_trial_seed(probe_seed, trial_idx)
            for probe_seed in probe_seeds
            for trial_idx in range(trials_per_probe)
        )
        if tuple(reference_trials.seeds) != expected_trial_seeds:
            raise ValueError(
                f"validation bank {label} reference seeds do not match its probe bank"
            )
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "probe_seeds", probe_seeds)
        object.__setattr__(self, "trials_per_probe", trials_per_probe)

    @property
    def trial_seeds(self) -> tuple[int, ...]:
        from rfr.search.rl.stage2.seed_utils import derive_probe_trial_seed

        return tuple(
            derive_probe_trial_seed(probe_seed, trial_idx)
            for probe_seed in self.probe_seeds
            for trial_idx in range(self.trials_per_probe)
        )

    @property
    def trial_count(self) -> int:
        return len(self.probe_seeds) * self.trials_per_probe


@dataclass(frozen=True)
class LayerwiseValidationBanks:
    """Independent A/B/C banks plus their pooled baseline references."""

    bank_a: LayerwiseValidationBank
    bank_b: LayerwiseValidationBank
    bank_c: LayerwiseValidationBank
    promotion_reference: Any
    final_reference: Any

    def __post_init__(self) -> None:
        banks = (self.bank_a, self.bank_b, self.bank_c)
        if tuple(bank.label for bank in banks) != ("A", "B", "C"):
            raise ValueError("validation banks must be ordered A, B, C")
        trial_counts = {bank.trial_count for bank in banks}
        if len(trial_counts) != 1:
            raise ValueError(
                "validation banks A, B, and C must contain equal trial counts"
            )
        all_probe_seeds: set[int] = set()
        all_trial_seeds: set[int] = set()
        for bank in banks:
            probe_seeds = set(bank.probe_seeds)
            trial_seeds = set(bank.trial_seeds)
            if all_probe_seeds.intersection(probe_seeds):
                raise ValueError("validation bank probe seeds must be pairwise disjoint")
            if all_trial_seeds.intersection(trial_seeds):
                raise ValueError("validation bank trial seeds must be pairwise disjoint")
            all_probe_seeds.update(probe_seeds)
            all_trial_seeds.update(trial_seeds)
        expected_ab = self.bank_a.trial_seeds + self.bank_b.trial_seeds
        expected_abc = expected_ab + self.bank_c.trial_seeds
        promotion_trials = _field(self.promotion_reference, "trials")
        final_trials = _field(self.final_reference, "trials")
        if not isinstance(promotion_trials, TrialSeries):
            raise TypeError("promotion reference must carry TrialSeries trials")
        if not isinstance(final_trials, TrialSeries):
            raise TypeError("final reference must carry TrialSeries trials")
        if tuple(promotion_trials.seeds) != expected_ab:
            raise ValueError("promotion reference must pool Bank A then Bank B")
        if tuple(final_trials.seeds) != expected_abc:
            raise ValueError("final reference must pool Bank A, Bank B, then Bank C")
        for channel in ("loss", "metric1", "metric2"):
            expected_promotion = tuple(
                value
                for bank in (self.bank_a, self.bank_b)
                for value in getattr(_field(bank.reference, "trials"), channel)
            )
            if tuple(getattr(promotion_trials, channel)) != expected_promotion:
                raise ValueError(
                    "promotion reference must contain the exact Bank A then "
                    f"Bank B trials for {channel}"
                )
            expected_final = tuple(
                value
                for bank in banks
                for value in getattr(_field(bank.reference, "trials"), channel)
            )
            if tuple(getattr(final_trials, channel)) != expected_final:
                raise ValueError(
                    "final reference must contain the exact Bank A, Bank B, "
                    f"then Bank C trials for {channel}"
                )

    @property
    def promotion_trial_count(self) -> int:
        return self.bank_a.trial_count + self.bank_b.trial_count

    @property
    def final_trial_count(self) -> int:
        return self.promotion_trial_count + self.bank_c.trial_count

    def contract_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "layerwise_validation_banks_v1",
            "banks": {
                bank.label: {
                    "probe_seeds": list(bank.probe_seeds),
                    "trial_seeds": list(bank.trial_seeds),
                    "trials_per_probe": int(bank.trials_per_probe),
                    "trial_count": int(bank.trial_count),
                }
                for bank in (self.bank_a, self.bank_b, self.bank_c)
            },
            "promotion_trial_count": int(self.promotion_trial_count),
            "final_trial_count": int(self.final_trial_count),
            "hard_gate": _STRICT_GATE_CONTRACT,
            "bootstrap_probability_role": "diagnostic_tiebreak_only",
        }

    def resume_payload(self) -> dict[str, Any]:
        """Return all raw references required for zero-baseline resume."""
        return {
            "schema_version": "layerwise_validation_banks_resume_v1",
            "banks": {
                bank.label: {
                    "label": bank.label,
                    "probe_seeds": list(bank.probe_seeds),
                    "trials_per_probe": int(bank.trials_per_probe),
                    "reference": baseline_reference_resume_payload(bank.reference),
                }
                for bank in (self.bank_a, self.bank_b, self.bank_c)
            },
            "promotion_reference": baseline_reference_resume_payload(
                self.promotion_reference
            ),
            "final_reference": baseline_reference_resume_payload(
                self.final_reference
            ),
            "contract": self.contract_payload(),
        }

    @classmethod
    def from_resume_payload(
            cls,
            payload: Mapping[str, Any],
            ) -> LayerwiseValidationBanks:
        """Restore validation banks and revalidate pooling and seed contracts."""
        copied = _exact_resume_mapping(
            "layerwise validation banks resume payload",
            payload,
            (
                "schema_version",
                "banks",
                "promotion_reference",
                "final_reference",
                "contract",
            ),
        )
        if copied["schema_version"] != "layerwise_validation_banks_resume_v1":
            raise ValueError(
                "layerwise validation banks resume payload.schema_version "
                "must be 'layerwise_validation_banks_resume_v1'"
            )
        bank_payloads = _exact_resume_mapping(
            "layerwise validation banks resume payload.banks",
            copied["banks"],
            ("A", "B", "C"),
        )
        restored_banks = {}
        for label in ("A", "B", "C"):
            bank_payload = _exact_resume_mapping(
                f"layerwise validation bank {label} resume payload",
                bank_payloads[label],
                ("label", "probe_seeds", "trials_per_probe", "reference"),
            )
            if bank_payload["label"] != label:
                raise ValueError(
                    f"layerwise validation bank {label} label mismatch"
                )
            restored_banks[label] = LayerwiseValidationBank(
                label=label,
                reference=baseline_reference_from_resume_payload(
                    bank_payload["reference"]
                ),
                probe_seeds=bank_payload["probe_seeds"],
                trials_per_probe=bank_payload["trials_per_probe"],
            )
        restored = cls(
            bank_a=restored_banks["A"],
            bank_b=restored_banks["B"],
            bank_c=restored_banks["C"],
            promotion_reference=baseline_reference_from_resume_payload(
                copied["promotion_reference"]
            ),
            final_reference=baseline_reference_from_resume_payload(
                copied["final_reference"]
            ),
        )
        if copied["contract"] != restored.contract_payload():
            raise ValueError(
                "layerwise validation banks resume payload contract mismatch"
            )
        return restored


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
        communication_importance_ratio: float = 1.0,
        ) -> dict[str, Any]:
    objective = compute_variable_cost_from_action_matrix(
        action_matrix,
        communication_importance_ratio=communication_importance_ratio,
    )
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
        "compute_weight": float(objective.compute_weight),
        "communication_weight": float(objective.communication_weight),
        "communication_importance_ratio": float(
            objective.communication_importance_ratio
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
        return _resource_fields_from_action_matrix(
            action_matrix,
            _finite(
                _field(candidate, "communication_importance_ratio", 1.0),
                name="communication_importance_ratio",
            ),
        )
    compute = _field(candidate, "compute_saving")
    communication = _field(candidate, "communication_saving")
    if compute is None or communication is None:


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
        "ppo_resource_score": _finite(
            _field(candidate, "ppo_resource_score", 0.5 * (
                compute_value + communication_value
            )),
            name="ppo_resource_score",
        ),
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
        -resource["ppo_resource_score"],
        -resource["robust_floor"],
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
        "layer_configurations": describe_layerwise_action_matrix(
            best["action_matrix"]
        ),
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
        "boosted_overrides": _copy_boosted_overrides(
            best["boosted_overrides"]
        ),
        "promotion_evidence": promotion_evidence,
        "axis_counterfactuals": copy.deepcopy(
            best.get("axis_counterfactuals")
        ),
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
    """Normalize entropy for Block4 fusion and the precision-preset slot."""
    entropy = np.asarray(entropy_per_slot, dtype=np.float64)
    masks = np.asarray(slot_masks, dtype=bool)
    levels = np.asarray(per_slot_num_levels, dtype=np.int64)
    if entropy.ndim != 2 or entropy.shape != masks.shape or entropy.shape != levels.shape:
        raise ValueError("entropy, masks, and levels must be aligned 2-D arrays")
    if entropy.shape[1] != len(LAYERWISE_SLOT_NAMES):
        raise ValueError("layerwise entropy requires exactly two slots")

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
    """Move P3 resource credit to its source layer transitions.

    The returned rewards always sum to ``terminal_reward``. Precision/stability
    failures retain terminal-only credit, so resources cannot leak into P1/P2.
    """
    reward = _finite(terminal_reward, name="terminal_reward")
    resource_score = _finite(ppo_resource_score, name="ppo_resource_score")
    layer_resources = tuple(
        _finite(value, name=f"layer_resource_rewards[{index}]")
        for index, value in enumerate(layer_resource_rewards)
    )
    if not layer_resources:
        raise ValueError("layer_resource_rewards must contain at least one value")
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
        return (0.0,) * (len(layer_resources) - 1) + (reward,)

    redistributed = list(layer_resources)
    redistributed[-1] += reward - resource_score
    if not math.isclose(sum(redistributed), reward, rel_tol=0.0, abs_tol=1.0e-9):
        raise RuntimeError("layerwise reward redistribution changed episode return")
    return tuple(float(value) for value in redistributed)


def resolve_layerwise_episode_budget(
        requested_total_episodes: int,
        completed_episodes: int,
        ) -> int:
    """Return the remaining configured maximum-episode budget."""
    requested = int(requested_total_episodes)
    completed = int(completed_episodes)
    if requested <= 0 or completed < 0:
        raise ValueError(
            "requested layerwise episodes must be positive and completed "
            "episodes must be nonnegative"
        )
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
    axis_counterfactuals: Optional[Mapping[str, Any]] = None


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
    final_config_fingerprint: str
    materialization_failure_reason: str
    model_uses_replan_config: bool
    promoted_trial_count: int
    promotion_status: str
    promotion_candidate_key: Optional[str]
    promotion_assessment: Optional[Mapping[str, Any]]
    promotion_metrics: Optional[Mapping[str, float]]
    invalid_steps: int
    step_count: int
    block4_entropy: Optional[float]
    k_entropy: Optional[float]
    strict_revalidation_status: str
    termination_reason: str


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


def _constraint_assessment_from_mapping(
        value: Any,
        ) -> Optional[ConstraintAssessment]:
    mapping = _to_plain_mapping(value)
    field_names = tuple(ConstraintAssessment.__dataclass_fields__)
    if any(name not in mapping for name in field_names):
        return None
    try:
        return ConstraintAssessment(**{
            name: mapping[name] for name in field_names
        })
    except (TypeError, ValueError):
        return None


def _assess_pooled_online_trials(
        *,
        raw_trials: TrialSeries,
        pooled_trials: TrialSeries,
        fresh_assessment: Any,
        reference: Any,
        gate_probability: float,
        bootstrap_seed: int,
        assess_candidate_fn: Callable[..., Any],
        ) -> Any:
    assessment_mapping = _to_plain_mapping(fresh_assessment)
    if (
            assess_candidate_fn is assess_candidate
            and pooled_trials == raw_trials
            and assessment_mapping.get("bootstrap_seed") == int(bootstrap_seed)
    ):
        normalized = _constraint_assessment_from_mapping(assessment_mapping)
        if normalized is not None:
            return retarget_constraint_assessment(
                normalized,
                gate_probability=float(gate_probability),
            )
    return assess_candidate_fn(
        pooled_trials,
        reference,
        gate_probability=float(gate_probability),
        bootstrap_seed=int(bootstrap_seed),
    )


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
    store.append_promotion_status(
        action_indices,
        identity_context,
        status=status,
        metadata=dict(metadata or {}),
    )


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


def _copy_boosted_overrides(
        overrides: Mapping[Any, Any],
        ) -> dict[tuple[int, int], dict[str, int]]:
    return {
        (int(block_idx), int(layer_idx)): {
            str(field_name): int(field_value)
            for field_name, field_value in fields.items()
        }
        for (block_idx, layer_idx), fields in overrides.items()
    }


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
    resource = _resource_fields_from_action_matrix(
        candidate["action_matrix"],
        float(candidate.get("communication_importance_ratio", 1.0)),
    )
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


def _restore_three_bank_candidates(
        *,
        candidate_store: CandidateStore,
        identity_context: Mapping[str, Any],
        assess_candidate_fn: Callable[..., Any],
        promotion_probability: float,
        final_probability: float,
        validation_banks: LayerwiseValidationBanks,
        ) -> dict[str, dict[str, Any]]:
    full_identity_context = evidence_identity_context(identity_context, "F4")
    wanted_context_hash = sha256_json(full_identity_context)
    latest_status: dict[
        str, tuple[str, tuple[int, ...], dict[str, Any]]
    ] = {}
    for record in candidate_store.iter_active_records():
        if record.get("record_type") not in (
                "candidate_promotion_status_v1",
                "candidate_promotion_status_v2",
        ):
            continue
        if str(record.get("identity_context_hash", "")) != wanted_context_hash:
            continue
        key = str(record.get("candidate_key", ""))
        action_indices = tuple(
            int(value) for value in record.get("action_indices", ())
        )
        if key and action_indices:
            latest_status[key] = (
                str(record.get("promotion_status", "")),
                action_indices,
                dict(record.get("promotion_metadata") or {}),
            )

    restored: dict[str, dict[str, Any]] = {}
    for key, (status, action_indices, status_metadata) in latest_status.items():
        if status not in (
                "promoted",
                _FINAL_REVALIDATION_PASSED,
                _FINAL_REVALIDATION_RETRYABLE,
        ):
            continue
        final_certified = status == _FINAL_REVALIDATION_PASSED
        trial_limit = (
            validation_banks.final_trial_count
            if final_certified else validation_banks.promotion_trial_count
        )
        reference = (
            validation_banks.final_reference
            if final_certified else validation_banks.promotion_reference
        )
        expected_seeds = (
            validation_banks.bank_a.trial_seeds
            + validation_banks.bank_b.trial_seeds
            + (validation_banks.bank_c.trial_seeds if final_certified else ())
        )
        evidence = candidate_store.trial_evidence_for_action(
            action_indices, full_identity_context, max_trials=trial_limit,
        )
        observed_count = _validate_bank_evidence_prefix(
            evidence, expected_seeds, context="restored validation banks",
        )
        if evidence is None or observed_count != trial_limit:
            continue
        final_config_fingerprint = (
            _strict_evidence_final_config_fingerprint(
                evidence,
                context="restored validation banks",
            )
        )
        if final_config_fingerprint is None:
            raise RuntimeError(
                "restored validation banks have no final config fingerprint"
            )
        metrics = _metrics_from_trials(evidence.trials)
        if not point_constraints_pass(metrics, reference):
            continue
        metadata = dict(status_metadata)
        for group in evidence.groups:
            for name in (
                    "action_matrix", "episode_reward",
                    "assessment_bootstrap_seed", "boosted_overrides",
            ):
                if name in group and name not in metadata:
                    metadata[name] = group[name]
        if not all(name in metadata for name in (
                "action_matrix", "boosted_overrides",
        )):
            continue
        action_matrix = tuple(
            tuple(int(value) for value in row)
            for row in metadata["action_matrix"]
        )
        if not action_matrix or any(
                len(row) != len(LAYERWISE_SLOT_NAMES) for row in action_matrix
        ):
            raise ValueError(
                "persisted layerwise action_matrix must be a nonempty Nx2 matrix"
            )
        assessment = assess_candidate_fn(
            evidence.trials,
            reference,
            gate_probability=(
                float(final_probability)
                if final_certified else float(promotion_probability)
            ),
            bootstrap_seed=int(metadata.get("assessment_bootstrap_seed", 0)),
        )
        resource = _resource_fields_from_action_matrix(
            action_matrix,
            float(metadata.get("communication_importance_ratio", 1.0)),
        )
        reward = metadata.get("episode_reward")
        restored[key] = {
            **resource,
            "variable_cost": float(resource["ppo_resource_score"]),
            "assessment": assessment,
            "metrics": metrics,
            "constraint_safety_margins": (
                normalized_constraint_safety_margins(metrics, reference)
            ),
            "action_matrix": action_matrix,
            "full_vector": tuple(action_indices),
            "boosted_overrides": _deserialize_boosted_overrides(
                metadata["boosted_overrides"],
            ),
            "final_config_fingerprint": final_config_fingerprint,
            "reward": (
                None if reward is None else _finite(reward, name="episode_reward")
            ),
            "promotion_trials": evidence.trials,
            "final_revalidation_status": (
                "passed" if final_certified else "not_run"
            ),
            "validation_evidence": (
                f"ABC_{evidence.trial_count}"
                if final_certified else f"AB_{evidence.trial_count}"
            ),
            "axis_counterfactuals": copy.deepcopy(
                metadata.get("axis_counterfactuals")
            ),
        }
    return restored


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
        validation_banks: Optional[LayerwiseValidationBanks] = None,
        ) -> dict[str, dict[str, Any]]:
    """Rebuild the current promoted frontier from append-only raw evidence."""
    if validation_banks is not None:
        return _restore_three_bank_candidates(
            candidate_store=candidate_store,
            identity_context=identity_context,
            assess_candidate_fn=assess_candidate_fn,
            promotion_probability=promotion_probability,
            final_probability=final_probability,
            validation_banks=validation_banks,
        )
    full_identity_context = evidence_identity_context(identity_context, "F4")
    latest_status: dict[
        str, tuple[str, tuple[int, ...], dict[str, Any]]
    ] = {}
    wanted_context_hash = sha256_json(full_identity_context)
    for record in candidate_store.iter_active_records():
        if record.get("record_type") not in (
                "candidate_promotion_status_v1",
                "candidate_promotion_status_v2",
        ):
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
        if not action_matrix or any(
                len(row) != len(LAYERWISE_SLOT_NAMES) for row in action_matrix
        ):
            raise ValueError(
                "persisted layerwise action_matrix must be a nonempty Nx2 matrix"
            )
        resource = _resource_fields_from_action_matrix(
            action_matrix,
            float(metadata.get("communication_importance_ratio", 1.0)),
        )
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
            "axis_counterfactuals": copy.deepcopy(
                promotion_metadata.get("axis_counterfactuals")
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
    from rfr.search.rl.stage2.seed_utils import (
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


def _latest_promotion_status(
        candidate_store: CandidateStore,
        action_indices: Sequence[int],
        identity_context: Mapping[str, Any],
        ) -> tuple[str, dict[str, Any]]:
    return candidate_store.latest_promotion_status_for_action(
        action_indices, identity_context,
    )


def _validation_bank_prefix(
        validation_banks: LayerwiseValidationBanks,
        label: str,
        ) -> tuple[tuple[int, ...], int, int, LayerwiseValidationBank]:
    normalized = str(label).strip().upper()
    if normalized == "A":
        bank = validation_banks.bank_a
        before: tuple[int, ...] = ()
    elif normalized == "B":
        bank = validation_banks.bank_b
        before = validation_banks.bank_a.trial_seeds
    elif normalized == "C":
        bank = validation_banks.bank_c
        before = (
            validation_banks.bank_a.trial_seeds
            + validation_banks.bank_b.trial_seeds
        )
    else:
        raise ValueError(f"unknown validation bank {label!r}")
    expected = before + bank.trial_seeds
    return expected, len(before), len(expected), bank


def _validate_bank_evidence_prefix(
        evidence: Optional[CandidateTrialEvidence],
        expected_seeds: Sequence[int],
        *,
        context: str,
        ) -> int:
    if evidence is None:
        return 0
    observed = tuple(int(value) for value in evidence.trials.seeds)
    expected = tuple(int(value) for value in expected_seeds)
    if len(observed) > len(expected) or observed != expected[:len(observed)]:
        raise RuntimeError(
            f"{context} evidence does not match the fixed validation-bank seed prefix"
        )
    return len(observed)


def _strict_final_config_fingerprint(
        value: Any,
        *,
        context: str,
        ) -> str:
    fingerprint = str(value or "")
    if len(fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in fingerprint
    ):
        raise RuntimeError(f"{context} final config fingerprint is invalid")
    return fingerprint


def _strict_evidence_final_config_fingerprint(
        evidence: CandidateTrialEvidence | None,
        *,
        expected: str | None = None,
        context: str,
        ) -> str | None:
    if evidence is None:
        return None
    fingerprints = {
        _strict_final_config_fingerprint(
            group.get("final_config_fingerprint"),
            context=context,
        )
        for group in evidence.groups
    }
    if len(fingerprints) != 1:
        raise RuntimeError(
            f"{context} evidence has inconsistent final config fingerprints"
        )
    observed = next(iter(fingerprints))
    if expected is not None and observed != expected:
        raise RuntimeError(
            f"{context} evidence final config fingerprint does not match "
            "the prepared action"
        )
    return observed


def _collect_fixed_validation_bank(
        *,
        env: Any,
        full_base_env: Any,
        candidate_store: CandidateStore,
        action_indices: Sequence[int],
        full_identity_context: Mapping[str, Any],
        action_matrix: Sequence[Sequence[int]],
        boosted_overrides: Mapping[Any, Any],
        bootstrap_seed: int,
        episode_reward: Optional[float],
        validation_banks: LayerwiseValidationBanks,
        bank_label: str,
        ) -> tuple[CandidateTrialEvidence, int]:
    expected_seeds, start_count, target_count, bank = _validation_bank_prefix(
        validation_banks, bank_label,
    )
    resource = _resource_fields_from_action_matrix(
        action_matrix,
        float(getattr(env, "communication_importance_ratio", 1.0)),
    )
    cost = float(resource["ppo_resource_score"])
    online_clear = getattr(env.base, "clear_installed_blb", None)
    if full_base_env is not env.base and callable(online_clear):
        online_clear()
        env.base._installed_config_fingerprint = None
        env.base._installed_action_hash = None
    previous_probe_seed = getattr(full_base_env, "probe_noise_seed", None)
    fresh_count = 0
    try:
        prepared = full_base_env.prepare_action_for_terminal_probe(
            list(action_indices),
            external_cost_score=cost,
            external_cost_rank=cost,
            external_resource_objective=resource,
            boosted_overrides=_copy_boosted_overrides(boosted_overrides),
        )
        final_config_fingerprint = _strict_final_config_fingerprint(
            prepared.get("final_config_fingerprint"),
            context=f"Bank {bank.label} prepared action",
        )
        evidence = candidate_store.trial_evidence_for_action(
            action_indices, full_identity_context, max_trials=target_count,
        )
        existing_count = _validate_bank_evidence_prefix(
            evidence, expected_seeds, context=f"Bank {bank.label}",
        )
        _strict_evidence_final_config_fingerprint(
            evidence,
            expected=final_config_fingerprint,
            context=f"Bank {bank.label}",
        )
        if existing_count < start_count:
            raise RuntimeError(
                f"Bank {bank.label} cannot start before "
                f"{start_count} earlier-bank trials"
            )
        if (existing_count - start_count) % bank.trials_per_probe:
            raise RuntimeError(
                f"Bank {bank.label} evidence ends inside a probe group"
            )
        if existing_count >= target_count:
            if evidence is None:
                raise AssertionError("complete validation bank has no evidence")
            return evidence, 0
        next_group = (existing_count - start_count) // bank.trials_per_probe
        remaining_groups = [
            (group_index, int(bank.probe_seeds[group_index]))
            for group_index in range(next_group, len(bank.probe_seeds))
        ]
        probe_runner = getattr(full_base_env, "probe_runner", None)
        grouped_bank_capable = bool(
            len(remaining_groups) > 1
            and probe_runner is not None
            and hasattr(probe_runner, "run_action_trial_groups")
        )
        if grouped_bank_capable:
            prepared_groups = []
            for _group_index, probe_seed in remaining_groups:
                prepared_group = dict(prepared)
                prepared_group["probe_base_seed"] = int(probe_seed)
                prepared_groups.append(prepared_group)
            evaluated_groups = full_base_env.evaluate_prepared_terminal_batch(
                prepared_groups,
                num_trials_per_action=bank.trials_per_probe,
                validation_required=True,
            )
        else:
            evaluated_groups = []
            for _group_index, probe_seed in remaining_groups:
                full_base_env.probe_noise_seed = int(probe_seed)
                evaluated_groups.extend(
                    full_base_env.evaluate_prepared_terminal_batch(
                        [prepared],
                        num_trials_per_action=bank.trials_per_probe,
                        validation_required=True,
                    )
                )
        if len(evaluated_groups) != len(remaining_groups):
            raise RuntimeError(
                f"Bank {bank.label} expected {len(remaining_groups)} "
                f"terminal results, received {len(evaluated_groups)}"
            )
        for (
                (group_index, probe_seed),
                evaluated,
        ) in zip(remaining_groups, evaluated_groups):
            terminal_info = evaluated[3]
            if (
                    not isinstance(terminal_info, Mapping)
                    or bool(terminal_info.get("invalid", False))
            ):
                raise RuntimeError(
                    f"Bank {bank.label} terminal evaluation was invalid"
                )
            fresh_trials = _trial_series_from_info(
                terminal_info,
                required=True,
                expected_count=bank.trials_per_probe,
                context=f"Bank {bank.label} terminal",
            )
            expected_group_seeds = bank.trial_seeds[
                group_index * bank.trials_per_probe:
                (group_index + 1) * bank.trials_per_probe
            ]
            if tuple(fresh_trials.seeds) != expected_group_seeds:
                raise RuntimeError(
                    f"Bank {bank.label} terminal trial seeds did not match "
                    "the fixed common-random-number group"
                )
            metadata = {
                "identity_context": dict(full_identity_context),
                "fidelity": "F4",
                "validation_bank": bank.label,
                "validation_bank_group_index": int(group_index),
                "validation_bank_probe_seed": probe_seed,
                "validation_bank_trials_per_probe": int(bank.trials_per_probe),
                "hard_gate": _STRICT_GATE_CONTRACT,
                "bootstrap_probability_role": "diagnostic_tiebreak_only",
                **resource,
                "variable_cost": cost,
                "action_matrix": [list(map(int, row)) for row in action_matrix],
                "boosted_overrides_hash": sha256_json(boosted_overrides),
                "boosted_overrides": _serialize_boosted_overrides(
                    boosted_overrides,
                ),
                "final_config_fingerprint": final_config_fingerprint,
                "boosted_overrides_provenance": "layerwise_env",
                "assessment_bootstrap_seed": int(bootstrap_seed),
                "promotion_marker": f"validation_bank_{bank.label.lower()}",
                "promotion_status": "pending_reassessment",
            }
            if episode_reward is not None:
                metadata["episode_reward"] = float(episode_reward)
            candidate_store.append_trial_group(
                action_indices, fresh_trials, metadata,
                compact=True,
            )
            fresh_count += len(fresh_trials.loss)
    finally:
        full_base_env.probe_noise_seed = previous_probe_seed
        if full_base_env is not env.base:
            full_clear = getattr(full_base_env, "clear_installed_blb", None)
            if callable(full_clear):
                full_clear()
            full_base_env._installed_config_fingerprint = None
            full_base_env._installed_action_hash = None
            if callable(online_clear):
                online_clear()
            env.base._installed_config_fingerprint = None
            env.base._installed_action_hash = None

    evidence = candidate_store.trial_evidence_for_action(
        action_indices, full_identity_context, max_trials=target_count,
    )
    observed_count = _validate_bank_evidence_prefix(
        evidence, expected_seeds, context=f"Bank {bank.label}",
    )
    _strict_evidence_final_config_fingerprint(
        evidence,
        expected=final_config_fingerprint,
        context=f"Bank {bank.label}",
    )
    if evidence is None or observed_count != target_count:
        raise RuntimeError(
            f"Bank {bank.label} evidence count {observed_count} != {target_count}"
        )
    return evidence, fresh_count


def _retarget_validation_banks_for_axis(
        validation_banks: LayerwiseValidationBanks,
        *,
        precision_tolerance: float,
        ) -> LayerwiseValidationBanks:
    """Keep seeds/stability fixed while assigning one axis a mean budget."""
    return LayerwiseValidationBanks(
        bank_a=replace(
            validation_banks.bank_a,
            reference=retarget_precision_tolerance(
                validation_banks.bank_a.reference,
                precision_tolerance,
            ),
        ),
        bank_b=replace(
            validation_banks.bank_b,
            reference=retarget_precision_tolerance(
                validation_banks.bank_b.reference,
                precision_tolerance,
            ),
        ),
        bank_c=replace(
            validation_banks.bank_c,
            reference=retarget_precision_tolerance(
                validation_banks.bank_c.reference,
                precision_tolerance,
            ),
        ),
        promotion_reference=retarget_precision_tolerance(
            validation_banks.promotion_reference,
            precision_tolerance,
        ),
        final_reference=retarget_precision_tolerance(
            validation_banks.final_reference,
            precision_tolerance,
        ),
    )


def _axis_proxy_action_matrix(
        action_matrix: Sequence[Sequence[int]],
        axis: str,
        ) -> tuple[tuple[int, int], ...]:
    rows = tuple(tuple(int(value) for value in row) for row in action_matrix)
    if not rows or any(len(row) != len(LAYERWISE_SLOT_NAMES) for row in rows):
        raise ValueError("action_matrix must have shape num_layers x 2")
    if axis == "compute":
        return tuple((row[0], 0) for row in rows)
    if axis == "communication":
        return tuple((0, row[1]) for row in rows)
    raise ValueError(f"unknown resource axis {axis!r}")


def _evaluate_axis_counterfactual_banks(
        *,
        env: Any,
        full_base_env: Any,
        candidate_store: CandidateStore,
        joint_action_indices: Sequence[int],
        joint_boosted_overrides: Mapping[Any, Any],
        identity_context: Mapping[str, Any],
        action_matrix: Sequence[Sequence[int]],
        bootstrap_seed: int,
        episode_reward: Optional[float],
        assess_candidate_fn: Callable[..., Any],
        gate_probability: float,
        validation_banks: LayerwiseValidationBanks,
        bank_labels: Sequence[str],
        ) -> tuple[dict[str, Any], int]:
    """Evaluate compute-only and communication-only strict counterfactuals."""
    ratio = validate_communication_importance_ratio(
        getattr(env, "communication_importance_ratio", 1.0),
    )
    total_tolerance = float(
        validation_banks.final_reference.precision_tolerance
    )
    compute_tolerance, communication_tolerance = (
        allocated_precision_tolerances(total_tolerance, ratio)
    )
    materialized = materialize_layerwise_counterfactuals(
        env.baseline_full_vector,
        action_matrix,
        env.schedule,
        env.fusion_map,
    )
    observed_joint = tuple(int(value) for value in joint_action_indices)
    expected_joint = tuple(
        int(value) for value in materialized["joint"].full_vector
    )
    if observed_joint != expected_joint:
        raise RuntimeError(
            "layerwise joint action does not match canonical counterfactual "
            "materialization"
        )
    if _serialize_boosted_overrides(joint_boosted_overrides) != (
            _serialize_boosted_overrides(
                materialized["joint"].boosted_overrides,
            )
    ):
        raise RuntimeError(
            "layerwise joint boosted overrides do not match canonical "
            "counterfactual materialization"
        )

    labels = tuple(str(label).strip().upper() for label in bank_labels)
    if not labels or labels not in (("A", "B"), ("A", "B", "C")):
        raise ValueError("axis counterfactual banks must be A+B or A+B+C")
    results: dict[str, Any] = {}
    total_fresh = 0
    for axis, materialization, tolerance in (
            ("compute", materialized["compute_only"], compute_tolerance),
            (
                "communication",
                materialized["communication_only"],
                communication_tolerance,
            ),
    ):
        axis_banks = _retarget_validation_banks_for_axis(
            validation_banks,
            precision_tolerance=tolerance,
        )
        axis_identity = evidence_identity_context({
            **dict(identity_context),
            "counterfactual_axis": axis,
            "counterfactual_contract": (
                "compute_k13_or_communication_fusion000_v1"
            ),
            "axis_precision_tolerance": float(tolerance),
            "communication_importance_ratio": float(ratio),
        }, "F4")
        proxy_matrix = _axis_proxy_action_matrix(action_matrix, axis)
        evidence = None
        metrics = None
        assessment = None
        reference = None
        passed = True
        bank_payloads: dict[str, Any] = {}
        for label in labels:
            evidence, fresh_count = _collect_fixed_validation_bank(
                env=env,
                full_base_env=full_base_env,
                candidate_store=candidate_store,
                action_indices=materialization.full_vector,
                full_identity_context=axis_identity,
                action_matrix=proxy_matrix,
                boosted_overrides=materialization.boosted_overrides,
                bootstrap_seed=bootstrap_seed,
                episode_reward=episode_reward,
                validation_banks=axis_banks,
                bank_label=label,
            )
            total_fresh += int(fresh_count)
            metrics = _metrics_from_trials(evidence.trials)
            if label == "A":
                reference = axis_banks.bank_a.reference
            elif label == "B":
                reference = axis_banks.promotion_reference
            else:
                reference = axis_banks.final_reference
            assessment = assess_candidate_fn(
                evidence.trials,
                reference,
                gate_probability=float(gate_probability),
                bootstrap_seed=int(bootstrap_seed),
            )
            current_pass = point_constraints_pass(metrics, reference)
            bank_payloads[label] = {
                "trial_count": int(evidence.trial_count),
                "fresh_trial_count": int(fresh_count),
                "metrics": dict(metrics),
                "assessment": _to_plain_mapping(assessment),
                "point_pass": bool(current_pass),
            }
            if not current_pass:
                passed = False
                break
        if reference is None:
            raise RuntimeError("axis counterfactual produced no validation reference")
        final_config_fingerprint = _strict_evidence_final_config_fingerprint(
            evidence,
            context=f"{axis} axis strict validation",
        )
        if final_config_fingerprint is None:
            raise RuntimeError(
                f"{axis} axis strict validation has no final config fingerprint"
            )
        results[axis] = {
            "mode": materialization.mode,
            "precision_tolerance": float(tolerance),
            "stability_multiplier": float(reference.stability_multiplier),
            "loss_limit": float(reference.loss_limit),
            "metric1_limit": float(reference.metric1_limit),
            "metric2_limit": float(reference.metric2_limit),
            "loss_std_limit": float(reference.loss_std_limit),
            "metric1_std_limit": float(reference.metric1_std_limit),
            "metric2_std_limit": float(reference.metric2_std_limit),
            "full_vector": [
                int(value) for value in materialization.full_vector
            ],
            "action_hash": sha256_json(
                [int(value) for value in materialization.full_vector]
            ),
            "boosted_overrides": _serialize_boosted_overrides(
                materialization.boosted_overrides,
            ),
            "final_config_fingerprint": final_config_fingerprint,
            "banks": bank_payloads,
            "point_pass": bool(passed and len(bank_payloads) == len(labels)),
            "metrics": None if metrics is None else dict(metrics),
            "assessment": (
                None if assessment is None
                else _to_plain_mapping(assessment)
            ),
        }
    return results, total_fresh


def _promote_candidate_through_validation_banks(
        *,
        env: Any,
        promotion_base_env: Optional[Any],
        candidate_store: CandidateStore,
        action_indices: Sequence[int],
        identity_context: Mapping[str, Any],
        action_matrix: Sequence[Sequence[int]],
        assessment: Any,
        priority: int,
        variable_cost: Optional[float],
        frontier_cost: Optional[float],
        frontier_candidates: Optional[Mapping[str, Mapping[str, Any]]],
        boosted_overrides: Mapping[Any, Any],
        bootstrap_seed: int,
        episode_reward: Optional[float],
        assess_candidate_fn: Callable[..., Any],
        promotion_probability: float,
        validation_banks: LayerwiseValidationBanks,
        ) -> PromotionResult:
    full_identity_context = evidence_identity_context(identity_context, "F4")
    full_base_env = promotion_base_env or env.base
    evidence = candidate_store.trial_evidence_for_action(
        action_indices,
        full_identity_context,
        max_trials=validation_banks.final_trial_count,
    )
    trial_count = 0 if evidence is None else evidence.trial_count
    latest_status, latest_metadata = _latest_promotion_status(
        candidate_store, action_indices, full_identity_context,
    )
    terminal_failures = {
        "bank_a_point_failed",
        "bank_b_point_failed",
        "bank_c_point_failed",
        "axis_counterfactual_point_failed",
        _FINAL_REVALIDATION_FAILED,
    }
    if latest_status in ("promoted", _FINAL_REVALIDATION_PASSED):
        trial_limit = (
            validation_banks.final_trial_count
            if latest_status == _FINAL_REVALIDATION_PASSED
            else validation_banks.promotion_trial_count
        )
        expected = (
            validation_banks.bank_a.trial_seeds
            + validation_banks.bank_b.trial_seeds
            + (
                validation_banks.bank_c.trial_seeds
                if latest_status == _FINAL_REVALIDATION_PASSED else ()
            )
        )
        evidence = candidate_store.trial_evidence_for_action(
            action_indices, full_identity_context, max_trials=trial_limit,
        )
        _validate_bank_evidence_prefix(
            evidence, expected, context="restored validation banks",
        )
        reference = (
            validation_banks.final_reference
            if latest_status == _FINAL_REVALIDATION_PASSED
            else validation_banks.promotion_reference
        )
        metrics = None if evidence is None else _metrics_from_trials(evidence.trials)
        diagnostic = (
            None if evidence is None else assess_candidate_fn(
                evidence.trials,
                reference,
                gate_probability=float(promotion_probability),
                bootstrap_seed=int(bootstrap_seed),
            )
        )
        return PromotionResult(
            "already_promoted",
            trial_count,
            0,
            evidence,
            diagnostic,
            metrics,
            latest_metadata.get("axis_counterfactuals"),
        )
    if latest_status in terminal_failures:
        return PromotionResult(
            "promotion_already_attempted", trial_count, 0,
            evidence, assessment,
            None if evidence is None else _metrics_from_trials(evidence.trials),
            latest_metadata.get("axis_counterfactuals"),
        )
    if int(priority) != 3:
        return PromotionResult(
            "priority_not_p3", trial_count, 0, evidence, assessment, None,
        )

    resource = _resource_fields_from_action_matrix(
        action_matrix,
        float(getattr(env, "communication_importance_ratio", 1.0)),
    )
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
        legacy_cost = _finite(variable_cost, name="variable_cost")
        dominated = legacy_cost < float(frontier_cost) - 1.0e-12
    if dominated:
        return PromotionResult(
            (
                "resource_dominated"
                if frontier_candidates is not None
                else "not_frontier_improvement"
            ),
            trial_count, 0, evidence, assessment, None,
        )

    status_metadata = {
        **resource,
        "variable_cost": float(resource["ppo_resource_score"]),
        "assessment_bootstrap_seed": int(bootstrap_seed),
        "action_matrix": [list(map(int, row)) for row in action_matrix],
        "boosted_overrides": _serialize_boosted_overrides(boosted_overrides),
        "hard_gate": _STRICT_GATE_CONTRACT,
        "bootstrap_probability_role": "diagnostic_tiebreak_only",
        "validation_bank_contract": validation_banks.contract_payload(),
    }
    if episode_reward is not None:
        status_metadata["episode_reward"] = float(episode_reward)
    fresh_count = 0
    pooled_assessment = assessment
    pooled_metrics: Optional[Mapping[str, float]] = None
    axis_counterfactuals: Optional[Mapping[str, Any]] = None
    promotion_status = "failed_evaluation"
    try:
        bank_a_evidence, bank_a_fresh = _collect_fixed_validation_bank(
            env=env,
            full_base_env=full_base_env,
            candidate_store=candidate_store,
            action_indices=action_indices,
            full_identity_context=full_identity_context,
            action_matrix=action_matrix,
            boosted_overrides=boosted_overrides,
            bootstrap_seed=bootstrap_seed,
            episode_reward=episode_reward,
            validation_banks=validation_banks,
            bank_label="A",
        )
        fresh_count += bank_a_fresh
        bank_a_metrics = _metrics_from_trials(bank_a_evidence.trials)
        bank_a_assessment = assess_candidate_fn(
            bank_a_evidence.trials,
            validation_banks.bank_a.reference,
            gate_probability=float(promotion_probability),
            bootstrap_seed=int(bootstrap_seed),
        )
        status_metadata["bank_a_metrics"] = bank_a_metrics
        status_metadata["bank_a_assessment"] = _to_plain_mapping(
            bank_a_assessment,
        )
        status_metadata["bank_a_point_pass"] = point_constraints_pass(
            bank_a_metrics, validation_banks.bank_a.reference,
        )
        if not status_metadata["bank_a_point_pass"]:
            promotion_status = "bank_a_point_failed"
            pooled_assessment = bank_a_assessment
            pooled_metrics = bank_a_metrics
        else:
            bank_b_evidence, bank_b_fresh = _collect_fixed_validation_bank(
                env=env,
                full_base_env=full_base_env,
                candidate_store=candidate_store,
                action_indices=action_indices,
                full_identity_context=full_identity_context,
                action_matrix=action_matrix,
                boosted_overrides=boosted_overrides,
                bootstrap_seed=bootstrap_seed,
                episode_reward=episode_reward,
                validation_banks=validation_banks,
                bank_label="B",
            )
            fresh_count += bank_b_fresh
            pooled_metrics = _metrics_from_trials(bank_b_evidence.trials)
            pooled_assessment = assess_candidate_fn(
                bank_b_evidence.trials,
                validation_banks.promotion_reference,
                gate_probability=float(promotion_probability),
                bootstrap_seed=int(bootstrap_seed),
            )
            status_metadata["bank_ab_metrics"] = pooled_metrics
            status_metadata["bank_ab_assessment"] = _to_plain_mapping(
                pooled_assessment,
            )
            status_metadata["bank_ab_point_pass"] = point_constraints_pass(
                pooled_metrics, validation_banks.promotion_reference,
            )
            if not status_metadata["bank_ab_point_pass"]:
                promotion_status = "bank_b_point_failed"
            else:
                axis_counterfactuals, axis_fresh = (
                    _evaluate_axis_counterfactual_banks(
                        env=env,
                        full_base_env=full_base_env,
                        candidate_store=candidate_store,
                        joint_action_indices=action_indices,
                        joint_boosted_overrides=boosted_overrides,
                        identity_context=identity_context,
                        action_matrix=action_matrix,
                        bootstrap_seed=bootstrap_seed,
                        episode_reward=episode_reward,
                        assess_candidate_fn=assess_candidate_fn,
                        gate_probability=promotion_probability,
                        validation_banks=validation_banks,
                        bank_labels=("A", "B"),
                    )
                )
                fresh_count += int(axis_fresh)
                status_metadata["axis_counterfactuals"] = (
                    axis_counterfactuals
                )
                axis_pass = all(
                    bool(payload.get("point_pass", False))
                    for payload in axis_counterfactuals.values()
                )
                status_metadata["axis_counterfactual_point_pass"] = bool(
                    axis_pass
                )
                promotion_status = (
                    "promoted"
                    if axis_pass else "axis_counterfactual_point_failed"
                )
    except Exception as exc:
        status_metadata["error"] = str(exc)

    _append_promotion_status(
        candidate_store,
        action_indices,
        full_identity_context,
        status=promotion_status,
        metadata=status_metadata,
    )
    target = (
        validation_banks.promotion_trial_count
        if promotion_status in (
            "promoted",
            "bank_b_point_failed",
            "axis_counterfactual_point_failed",
        )
        else validation_banks.bank_a.trial_count
    )
    evidence = candidate_store.trial_evidence_for_action(
        action_indices, full_identity_context, max_trials=target,
    )
    trial_count = candidate_store.trial_count_for_action(
        action_indices, full_identity_context,
    )
    return PromotionResult(
        status=promotion_status,
        trial_count=trial_count,
        fresh_trial_count=fresh_count,
        evidence=evidence,
        assessment=pooled_assessment,
        metrics=pooled_metrics,
        axis_counterfactuals=axis_counterfactuals,
    )


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
        validation_banks: Optional[LayerwiseValidationBanks] = None,
        ) -> PromotionResult:
    """Promote one robust frontier improvement using fresh real probes."""
    if validation_banks is not None:
        return _promote_candidate_through_validation_banks(
            env=env,
            promotion_base_env=promotion_base_env,
            candidate_store=candidate_store,
            action_indices=action_indices,
            identity_context=identity_context,
            action_matrix=action_matrix,
            assessment=assessment,
            priority=priority,
            variable_cost=variable_cost,
            frontier_cost=frontier_cost,
            frontier_candidates=frontier_candidates,
            boosted_overrides=boosted_overrides,
            bootstrap_seed=bootstrap_seed,
            episode_reward=episode_reward,
            assess_candidate_fn=assess_candidate_fn,
            promotion_probability=promotion_probability,
            validation_banks=validation_banks,
        )
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
    resource = _resource_fields_from_action_matrix(
        action_matrix,
        float(getattr(env, "communication_importance_ratio", 1.0)),
    )
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
                env.base._installed_config_fingerprint = None
                env.base._installed_action_hash = None
            previous_probe_seed = getattr(full_base_env, "probe_noise_seed", None)
            full_base_env.probe_noise_seed = promotion_probe_seed
            try:
                prepared = full_base_env.prepare_action_for_terminal_probe(
                    list(action_indices),
                    external_cost_score=cost,
                    external_cost_rank=cost,
                    external_resource_objective=resource,
                    boosted_overrides=_copy_boosted_overrides(
                        boosted_overrides
                    ),
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
                    full_base_env._installed_config_fingerprint = None
                    full_base_env._installed_action_hash = None
                    if callable(online_clear):
                        online_clear()
                    env.base._installed_config_fingerprint = None
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
                compact=True,
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


def certify_candidate_with_bank_c(
        *,
        env: Any,
        promotion_base_env: Optional[Any],
        candidate_store: CandidateStore,
        identity_context: Mapping[str, Any],
        candidate: Mapping[str, Any],
        bootstrap_seed: int,
        assess_candidate_fn: Callable[..., Any] = assess_candidate,
        final_probability: float = 0.95,
        validation_banks: LayerwiseValidationBanks,
        ) -> PromotionResult:
    """Run the held-out C bank and certify the pooled A+B+C point estimate."""
    full_identity_context = evidence_identity_context(identity_context, "F4")
    action_indices = tuple(int(value) for value in candidate["full_vector"])
    action_matrix = tuple(
        tuple(int(value) for value in row)
        for row in candidate["action_matrix"]
    )
    boosted_overrides = dict(candidate.get("boosted_overrides") or {})
    full_base_env = promotion_base_env or env.base
    latest_status, latest_metadata = _latest_promotion_status(
        candidate_store, action_indices, full_identity_context,
    )
    if latest_status == _FINAL_REVALIDATION_PASSED:
        evidence = candidate_store.trial_evidence_for_action(
            action_indices,
            full_identity_context,
            max_trials=validation_banks.final_trial_count,
        )
        observed_count = _validate_bank_evidence_prefix(
            evidence,
            validation_banks.bank_a.trial_seeds
            + validation_banks.bank_b.trial_seeds
            + validation_banks.bank_c.trial_seeds,
            context="final validation banks",
        )
        if evidence is None:
            raise RuntimeError("final certification status has no raw evidence")
        if observed_count != validation_banks.final_trial_count:
            raise RuntimeError(
                "final certification status has incomplete raw evidence"
            )
        evidence, fresh_count = _collect_fixed_validation_bank(
            env=env,
            full_base_env=full_base_env,
            candidate_store=candidate_store,
            action_indices=action_indices,
            full_identity_context=full_identity_context,
            action_matrix=action_matrix,
            boosted_overrides=boosted_overrides,
            bootstrap_seed=bootstrap_seed,
            episode_reward=(
                None if candidate.get("reward") is None
                else float(candidate["reward"])
            ),
            validation_banks=validation_banks,
            bank_label="C",
        )
        if fresh_count:
            raise AssertionError(
                "completed final certification unexpectedly ran fresh trials"
            )
        metrics = _metrics_from_trials(evidence.trials)
        diagnostic = assess_candidate_fn(
            evidence.trials,
            validation_banks.final_reference,
            gate_probability=float(final_probability),
            bootstrap_seed=int(bootstrap_seed),
        )
        return PromotionResult(
            "already_final_certified",
            evidence.trial_count,
            0,
            evidence,
            diagnostic,
            metrics,
            latest_metadata.get("axis_counterfactuals"),
        )
    if latest_status in (
            "bank_c_point_failed",
            "axis_counterfactual_point_failed",
            _FINAL_REVALIDATION_FAILED,
    ):
        evidence = candidate_store.trial_evidence_for_action(
            action_indices,
            full_identity_context,
            max_trials=validation_banks.final_trial_count,
        )
        return PromotionResult(
            "final_certification_already_attempted",
            0 if evidence is None else evidence.trial_count,
            0,
            evidence,
            None,
            None if evidence is None else _metrics_from_trials(evidence.trials),
            latest_metadata.get("axis_counterfactuals"),
        )
    retryable_ab_evidence = False
    if latest_status == _FINAL_REVALIDATION_RETRYABLE:
        ab_evidence = candidate_store.trial_evidence_for_action(
            action_indices,
            full_identity_context,
            max_trials=validation_banks.promotion_trial_count,
        )
        observed_count = _validate_bank_evidence_prefix(
            ab_evidence,
            validation_banks.bank_a.trial_seeds
            + validation_banks.bank_b.trial_seeds,
            context="retryable final validation",
        )
        retryable_ab_evidence = bool(
            ab_evidence is not None
            and observed_count == validation_banks.promotion_trial_count
            and point_constraints_pass(
                _metrics_from_trials(ab_evidence.trials),
                validation_banks.promotion_reference,
            )
        )
    if latest_status != "promoted" and not retryable_ab_evidence:
        evidence = candidate_store.trial_evidence_for_action(
            action_indices,
            full_identity_context,
            max_trials=validation_banks.promotion_trial_count,
        )
        return PromotionResult(
            "candidate_not_bank_b_confirmed",
            0 if evidence is None else evidence.trial_count,
            0,
            evidence,
            None,
            None if evidence is None else _metrics_from_trials(evidence.trials),
        )

    resource = _resource_fields_from_action_matrix(
        action_matrix,
        float(getattr(env, "communication_importance_ratio", 1.0)),
    )
    status_metadata = {
        **resource,
        "variable_cost": float(resource["ppo_resource_score"]),
        "assessment_bootstrap_seed": int(bootstrap_seed),
        "action_matrix": [list(row) for row in action_matrix],
        "boosted_overrides": _serialize_boosted_overrides(boosted_overrides),
        "hard_gate": _STRICT_GATE_CONTRACT,
        "bootstrap_probability_role": "diagnostic_tiebreak_only",
        "validation_bank_contract": validation_banks.contract_payload(),
    }
    prior_axis_counterfactuals = latest_metadata.get("axis_counterfactuals")
    if isinstance(prior_axis_counterfactuals, Mapping):
        prior_axis_counterfactuals = copy.deepcopy(
            prior_axis_counterfactuals
        )
        status_metadata["axis_counterfactuals"] = prior_axis_counterfactuals
    else:
        prior_axis_counterfactuals = None
    reward = candidate.get("reward")
    if reward is not None:
        status_metadata["episode_reward"] = _finite(
            reward, name="episode_reward",
        )
    fresh_count = 0
    assessment = None
    metrics = None
    axis_counterfactuals: Mapping[str, Any] | None = (
        prior_axis_counterfactuals
    )
    status = "failed_evaluation"
    try:
        evidence, fresh_count = _collect_fixed_validation_bank(
            env=env,
            full_base_env=full_base_env,
            candidate_store=candidate_store,
            action_indices=action_indices,
            full_identity_context=full_identity_context,
            action_matrix=action_matrix,
            boosted_overrides=boosted_overrides,
            bootstrap_seed=bootstrap_seed,
            episode_reward=(None if reward is None else float(reward)),
            validation_banks=validation_banks,
            bank_label="C",
        )
        metrics = _metrics_from_trials(evidence.trials)
        assessment = assess_candidate_fn(
            evidence.trials,
            validation_banks.final_reference,
            gate_probability=float(final_probability),
            bootstrap_seed=int(bootstrap_seed),
        )
        passed = point_constraints_pass(
            metrics, validation_banks.final_reference,
        )
        status_metadata.update({
            "bank_abc_metrics": metrics,
            "bank_abc_assessment": _to_plain_mapping(assessment),
            "bank_abc_point_pass": bool(passed),
        })
        if not passed:
            status = "bank_c_point_failed"
        else:
            axis_counterfactuals, axis_fresh = (
                _evaluate_axis_counterfactual_banks(
                    env=env,
                    full_base_env=full_base_env,
                    candidate_store=candidate_store,
                    joint_action_indices=action_indices,
                    joint_boosted_overrides=boosted_overrides,
                    identity_context=identity_context,
                    action_matrix=action_matrix,
                    bootstrap_seed=bootstrap_seed,
                    episode_reward=(None if reward is None else float(reward)),
                    assess_candidate_fn=assess_candidate_fn,
                    gate_probability=final_probability,
                    validation_banks=validation_banks,
                    bank_labels=("A", "B", "C"),
                )
            )
            fresh_count += int(axis_fresh)
            status_metadata["axis_counterfactuals"] = axis_counterfactuals
            axis_pass = all(
                bool(payload.get("point_pass", False))
                for payload in axis_counterfactuals.values()
            )
            status_metadata["axis_counterfactual_point_pass"] = bool(
                axis_pass
            )
            status = (
                _FINAL_REVALIDATION_PASSED
                if axis_pass else "axis_counterfactual_point_failed"
            )
    except Exception as exc:
        status_metadata["error"] = str(exc)

    _append_promotion_status(
        candidate_store,
        action_indices,
        full_identity_context,
        status=status,
        metadata=status_metadata,
    )
    evidence = candidate_store.trial_evidence_for_action(
        action_indices,
        full_identity_context,
        max_trials=validation_banks.final_trial_count,
    )
    return PromotionResult(
        status=status,
        trial_count=(0 if evidence is None else evidence.trial_count),
        fresh_trial_count=int(fresh_count),
        evidence=evidence,
        assessment=assessment,
        metrics=metrics,
        axis_counterfactuals=axis_counterfactuals,
    )


def _certify_strict_best_candidates(
        *,
        env: Any,
        promotion_base_env: Optional[Any],
        candidate_store: CandidateStore,
        identity_context: Mapping[str, Any],
        accepted_candidates: dict[str, dict[str, Any]],
        bootstrap_seed: int,
        assess_candidate_fn: Callable[..., Any],
        final_probability: float,
        validation_banks: LayerwiseValidationBanks,
        exhaustive_fallback: bool,
        ) -> tuple[str, Optional[dict[str, Any]]]:
    """Certify the deterministic winner, falling back only at the max cap."""
    attempts_remaining = len(accepted_candidates) if exhaustive_fallback else 1
    status = "no_bank_b_confirmed_candidate"
    while attempts_remaining > 0:
        selected = _strict_best_snapshot(accepted_candidates)
        if selected is None:
            return "no_bank_b_confirmed_candidate", None
        attempts_remaining -= 1
        selected_key = str(selected["candidate_key"])
        result = certify_candidate_with_bank_c(
            env=env,
            promotion_base_env=promotion_base_env,
            candidate_store=candidate_store,
            identity_context=identity_context,
            candidate=selected,
            bootstrap_seed=int(bootstrap_seed),
            assess_candidate_fn=assess_candidate_fn,
            final_probability=final_probability,
            validation_banks=validation_banks,
        )
        passed = bool(
            result.status in (
                _FINAL_REVALIDATION_PASSED,
                "already_final_certified",
            )
            and result.evidence is not None
            and result.evidence.trial_count >= validation_banks.final_trial_count
            and point_constraints_pass(
                result.metrics or {}, validation_banks.final_reference,
            )
        )
        if not passed:
            status = str(result.status)
            if result.status in (
                    "bank_c_point_failed",
                    "axis_counterfactual_point_failed",
            ):
                accepted_candidates.pop(selected_key, None)
                survivor = _strict_best_snapshot(accepted_candidates)
                if survivor is not None:
                    survivor_candidate = accepted_candidates[
                        str(survivor["candidate_key"])
                    ]
                    if (
                            survivor_candidate.get("final_revalidation_status")
                            == "passed"
                    ):
                        return "passed", survivor
                if exhaustive_fallback:
                    continue
            return status, _strict_best_snapshot(accepted_candidates)

        candidate = accepted_candidates[selected_key]
        candidate["assessment"] = result.assessment
        candidate["metrics"] = dict(result.metrics or {})
        candidate["constraint_safety_margins"] = (
            normalized_constraint_safety_margins(
                candidate["metrics"], validation_banks.final_reference,
            )
        )
        candidate["promotion_trials"] = result.evidence.trials
        candidate["final_revalidation_status"] = "passed"
        candidate["validation_evidence"] = (
            f"ABC_{result.evidence.trial_count}"
        )
        candidate["axis_counterfactuals"] = copy.deepcopy(
            getattr(result, "axis_counterfactuals", None)
        )
        winner = _strict_best_snapshot(accepted_candidates)
        if winner is None:
            return "no_bank_b_confirmed_candidate", None
        winner_candidate = accepted_candidates[str(winner["candidate_key"])]
        if winner_candidate.get("final_revalidation_status") == "passed":
            return "passed", winner
        status = "winner_changed_after_bank_c_certification"
        if not exhaustive_fallback:
            return status, winner
    winner = _strict_best_snapshot(accepted_candidates)
    if winner is not None:
        winner_candidate = accepted_candidates[str(winner["candidate_key"])]
        if winner_candidate.get("final_revalidation_status") == "passed":
            return "passed", winner
    return status, winner


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
    return torch.as_tensor(value, device=device)


def resolve_exact_terminal_batch_size(
        requested_batch_size: int,
        trial_count: int,
        worker_count: int,
        ) -> int:
    """Return the smallest useful exact cross-episode scheduling window."""
    requested = max(1, int(requested_batch_size))
    trials = max(1, int(trial_count))
    workers = max(1, int(worker_count))
    if requested == 1 or workers == 1 or trials % workers == 0:
        return 1
    balance_period = workers // math.gcd(trials, workers)
    return min(requested, balance_period)


def _probe_pool_state(probe_runner: Any) -> tuple[int, int]:
    generation = max(
        0, int(getattr(probe_runner, "pool_generation", 0) or 0)
    )
    worker_count = max(
        1, int(getattr(probe_runner, "num_workers", 1) or 1)
    )
    return generation, worker_count


@dataclass
class _LayerwiseEpisodeDraft:
    absolute_episode: int
    terminal_info: Mapping[str, Any]
    runtime_info: Mapping[str, Any]
    episode_reward: float
    step_infos: list[Mapping[str, Any]]
    transition_indices: list[int]
    entropy_start_index: int
    boosted_overrides: Mapping[Any, Any]
    prepared_terminal_probe: Optional[Mapping[str, Any]] = None


def _collect_layerwise_episode(
        *,
        env: Any,
        policy: Any,
        rollout_buffer: Any,
        entropy_samples: list[dict[str, np.ndarray]],
        absolute_episode: int,
        base_seed: Optional[int],
        expected_online_trials: int,
        horizon: int,
        device: Any,
        step_adapter_fn: Callable[[Any], tuple[np.ndarray, np.ndarray]],
        ) -> _LayerwiseEpisodeDraft:
    reset_seed = None
    probe_seed = None
    if base_seed is not None:
        from rfr.search.rl.stage2.seed_utils import derive_layerwise_online_evaluation_seeds

        reset_seed, probe_seed = derive_layerwise_online_evaluation_seeds(
            int(base_seed),
            int(absolute_episode),
            trial_count=int(expected_online_trials),
        )
    state = env.reset(seed=reset_seed)
    if probe_seed is not None and hasattr(env, "base"):
        env.base.probe_noise_seed = probe_seed
    step_infos: list[Mapping[str, Any]] = []
    transition_indices: list[int] = []
    entropy_start_index = len(entropy_samples)
    terminal_info: Optional[Mapping[str, Any]] = None
    episode_reward = 0.0
    for step_idx in range(horizon):
        spec = env.current_spec()
        slot_mask, levels = step_adapter_fn(spec)
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
                "layerwise factorized PPO requires sampling-time "
                "per-slot log probabilities"
            )
        actions_raw, log_prob_raw, value_raw, log_prob_per_slot_raw = sample_out
        action = _as_numpy(actions_raw).reshape(-1).astype(np.int64)
        action[~slot_mask] = 0
        log_prob = _first_detached_scalar(log_prob_raw)
        value = _first_detached_scalar(value_raw)
        log_prob_per_slot = (
            log_prob_per_slot_raw.detach().reshape(-1)
            if hasattr(log_prob_per_slot_raw, "detach")
            else np.asarray(
                log_prob_per_slot_raw, dtype=np.float32,
            ).reshape(-1)
        )
        next_state, reward, done, info = env.step(action.tolist())
        expected_done = step_idx == horizon - 1
        if bool(done) != expected_done:
            raise RuntimeError(
                "layerwise episode termination mismatch at step "
                f"{step_idx}: done={done}"
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
            terminal_info = info if isinstance(info, Mapping) else {}
            episode_reward = float(reward)

    if terminal_info is None:
        raise RuntimeError("layerwise episode completed without terminal info")
    pending_probe = getattr(env, "pending_terminal_probe", None)
    prepared = None
    if bool(terminal_info.get("terminal_probe_deferred", False)):
        if not isinstance(pending_probe, Mapping):
            raise RuntimeError(
                "deferred layerwise terminal is missing its prepared probe"
            )
        prepared = pending_probe.get("prepared")
        if not isinstance(prepared, Mapping):
            raise RuntimeError(
                "deferred layerwise terminal prepared probe is invalid"
            )
        runtime_info: Mapping[str, Any] = {}
        boosted_overrides = _copy_boosted_overrides(
            pending_probe.get("boosted_overrides") or {}
        )
    else:
        runtime_value = getattr(env, "runtime_terminal_info", None)
        runtime_info = (
            runtime_value if isinstance(runtime_value, Mapping) else {}
        )
        boosted_overrides = _copy_boosted_overrides(
            getattr(env, "boosted_overrides", {}) or {}
        )
    return _LayerwiseEpisodeDraft(
        absolute_episode=int(absolute_episode),
        terminal_info=terminal_info,
        runtime_info=runtime_info,
        episode_reward=float(episode_reward),
        step_infos=step_infos,
        transition_indices=transition_indices,
        entropy_start_index=entropy_start_index,
        boosted_overrides=boosted_overrides,
        prepared_terminal_probe=prepared,
    )


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
        validation_banks: Optional[LayerwiseValidationBanks] = None,
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
        stop_requested: Optional[Callable[[], bool]] = None,
        retain_history: bool = True,
        ) -> dict[str, Any]:
    """Collect layerwise episodes and update the shared PPO policy."""
    if identity_context is None:
        raise ValueError("layerwise training requires a CandidateStore identity_context")
    probe_identity_context = evidence_identity_context(identity_context, "F1")
    authoritative_base_env = promotion_base_env or env.base
    horizon = int(getattr(env, "horizon", 0))
    if (
            horizon <= 0
            or int(getattr(env, "max_step_dim", 0)) != len(LAYERWISE_SLOT_NAMES)
    ):
        raise ValueError(
            "layerwise training requires a positive horizon and max_step_dim=2"
        )
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
        from rfr.search.rl.stage2.policy import LayerwiseRolloutBuffer

        rollout_buffer = LayerwiseRolloutBuffer()
    if ppo_update_fn is None:
        from rfr.search.rl.stage2.policy import layerwise_ppo_update

        ppo_update_fn = layerwise_ppo_update
    if step_adapter_fn is None:
        from rfr.search.rl.stage2.policy import layer_action_mask_and_levels

        step_adapter_fn = layer_action_mask_and_levels

    ppo_cfg = copy.copy(train_cfg.ppo)
    ppo_cfg.gamma = 1.0
    ppo_cfg.gae_lambda = 1.0
    update_window = max(1, int(getattr(train_cfg, "update_every_n_episodes", 120)))
    total_episodes = int(getattr(train_cfg, "total_episodes", 0))
    if total_episodes <= 0:
        raise ValueError("layerwise training requires a positive episode limit")
    planned_total_episodes = getattr(train_cfg, "planned_total_episodes", None)
    absolute_start = int(getattr(train_cfg, "absolute_episode_start", 0))
    base_seed = getattr(train_cfg, "seed", None)
    expected_online_trials = int(
        getattr(train_cfg, "online_num_trials_per_step", 3)
    )
    if expected_online_trials <= 0:
        raise ValueError("online_num_trials_per_step must be positive")
    requested_terminal_batch_size = max(
        1, int(getattr(train_cfg, "terminal_eval_batch_size", 1) or 1)
    )
    probe_runner = getattr(getattr(env, "base", None), "probe_runner", None)
    probe_pool_generation, probe_worker_count = _probe_pool_state(
        probe_runner
    )
    exact_batch_capable = bool(
        base_seed is not None
        and probe_runner is not None
        and hasattr(probe_runner, "run_action_trial_groups")
        and hasattr(env, "configure_terminal_probe_deferral")
        and hasattr(getattr(env, "base", None), "evaluate_prepared_terminal_batch")
        and bool(getattr(
            getattr(getattr(env, "base", None), "env_cfg", None),
            "persistent_probe_install",
            False,
        ))
    )
    terminal_batch_size = (
        resolve_exact_terminal_batch_size(
            requested_terminal_batch_size,
            expected_online_trials,
            probe_worker_count,
        )
        if exact_batch_capable else 1
    )
    if hasattr(env, "configure_terminal_probe_deferral"):
        env.configure_terminal_probe_deferral(terminal_batch_size > 1)
    probe_pool_schedule = [{
        "first_episode": int(absolute_start),
        "pool_generation": int(probe_pool_generation),
        "worker_count": int(probe_worker_count),
        "terminal_batch_size": int(terminal_batch_size),
    }]
    online_probability = float(
        getattr(train_cfg, "online_constraint_probability", 0.50)
    )
    promotion_probability = float(
        getattr(train_cfg, "promotion_constraint_probability", 0.80)
    )
    final_probability = float(
        getattr(train_cfg, "final_constraint_probability", 0.95)
    )
    promotion_trials = int(getattr(train_cfg, "promotion_validation_trials", 15))
    final_validation_trials = int(
        getattr(train_cfg, "final_selection_validation_trials", 15)
    )
    if not 0.0 < online_probability <= promotion_probability <= final_probability <= 1.0:
        raise ValueError(
            "constraint probabilities must satisfy "
            "0 < online <= promotion <= final <= 1"
        )
    if validation_banks is not None:
        if promotion_trials != validation_banks.bank_a.trial_count:
            raise ValueError(
                "promotion_validation_trials must equal each A/B bank trial count"
            )
        if final_validation_trials != validation_banks.bank_c.trial_count:
            raise ValueError(
                "final_selection_validation_trials must equal Bank C trial count"
            )
    else:
        if promotion_trials < expected_online_trials:
            raise ValueError(
                "promotion_validation_trials must cover the online trial group"
            )
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
        validation_banks=validation_banks,
    )
    strict_revalidation_status = "not_due"
    latest_block4_entropy: Optional[float] = None
    latest_k_entropy: Optional[float] = None
    entropy_samples: list[dict[str, np.ndarray]] = []
    local_episode = 0
    finalized_drafts: list[_LayerwiseEpisodeDraft] = []
    graceful_stopped = False
    while local_episode < total_episodes:
        if not finalized_drafts:
            current_pool_generation, current_worker_count = (
                _probe_pool_state(probe_runner)
            )
            if (
                    exact_batch_capable
                    and (
                        current_pool_generation != probe_pool_generation
                        or current_worker_count != probe_worker_count
                    )
            ):
                probe_pool_generation = current_pool_generation
                probe_worker_count = current_worker_count
                terminal_batch_size = resolve_exact_terminal_batch_size(
                    requested_terminal_batch_size,
                    expected_online_trials,
                    probe_worker_count,
                )
                if hasattr(env, "configure_terminal_probe_deferral"):
                    env.configure_terminal_probe_deferral(terminal_batch_size > 1)
                probe_pool_schedule.append({
                    "first_episode": int(
                        absolute_start + local_episode
                    ),
                    "pool_generation": int(probe_pool_generation),
                    "worker_count": int(probe_worker_count),
                    "terminal_batch_size": int(terminal_batch_size),
                })
            episodes_to_update = update_window - (
                local_episode % update_window
            )
            episodes_to_end = total_episodes - local_episode
            collect_count = min(
                terminal_batch_size,
                episodes_to_update,
                episodes_to_end,
            )
            collected = [
                _collect_layerwise_episode(
                    env=env,
                    policy=policy,
                    rollout_buffer=rollout_buffer,
                    entropy_samples=entropy_samples,
                    absolute_episode=(
                        absolute_start + local_episode + batch_offset
                    ),
                    base_seed=base_seed,
                    expected_online_trials=expected_online_trials,
                    horizon=horizon,
                    device=device,
                    step_adapter_fn=step_adapter_fn,
                )
                for batch_offset in range(collect_count)
            ]
            if terminal_batch_size > 1:
                prepared_batch = [
                    draft.prepared_terminal_probe for draft in collected
                ]
                if any(item is None for item in prepared_batch):
                    raise RuntimeError(
                        "exact terminal scheduling collected an "
                        "unprepared layerwise episode"
                    )
                terminal_results = env.base.evaluate_prepared_terminal_batch(
                    prepared_batch,
                    num_trials_per_action=expected_online_trials,
                    validation_required=False,
                )
                if len(terminal_results) != len(collected):
                    raise RuntimeError(
                        "exact terminal scheduling returned "
                        f"{len(terminal_results)} results for "
                        f"{len(collected)} episodes"
                    )
                for draft, result in zip(collected, terminal_results):
                    _terminal_state, reward, done, runtime_info = result
                    if not bool(done) or not isinstance(runtime_info, Mapping):
                        raise RuntimeError(
                            "exact terminal scheduling returned an invalid "
                            "base-env terminal result"
                        )
                    draft.episode_reward = float(reward)
                    draft.runtime_info = runtime_info
            finalized_drafts.extend(collected)

        draft = finalized_drafts.pop(0)
        absolute_episode = int(draft.absolute_episode)
        terminal_info = draft.terminal_info
        runtime_info = draft.runtime_info
        episode_reward = float(draft.episode_reward)
        step_infos = draft.step_infos
        transition_indices = draft.transition_indices
        episode_boosted_overrides = draft.boosted_overrides
        action_matrix = tuple(
            tuple(int(value) for value in row)
            for row in terminal_info.get("policy_actions", ())
        )
        if len(action_matrix) != horizon or any(
                len(row) != len(LAYERWISE_SLOT_NAMES) for row in action_matrix
        ):
            raise RuntimeError(
                "layerwise terminal policy_actions must match the "
                f"{horizon}x2 environment contract"
            )
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
        if len(layer_resource_rewards) != horizon:
            raise RuntimeError(
                "layerwise terminal layer_resource_rewards must contain "
                f"{horizon} values"
            )
        if len(slot_resource_rewards) != horizon or any(
                len(row) != len(LAYERWISE_SLOT_NAMES)
                for row in slot_resource_rewards
        ):
            raise RuntimeError(
                "layerwise terminal slot_resource_rewards must match the "
                f"{horizon}x2 environment contract"
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
            raise RuntimeError(
                "fusion slots do not sum to direct compute credit"
            )
        if not math.isclose(
                sum(sum(row[1:]) for row in slot_resource_rewards),
                communication_shapley_credit,
                rel_tol=0.0,
                abs_tol=1.0e-9,
        ):
            raise RuntimeError(
                "precision-preset slots do not sum to direct communication "
                "credit"
            )
        exact_resource = _resource_fields_from_action_matrix(
            action_matrix,
            float(getattr(env, "communication_importance_ratio", 1.0)),
        )
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
        if (
                bool(runtime_info.get("apply_failed", False))
                or bool(runtime_info.get("eval_failed", False))
        ):
            truncate = getattr(rollout_buffer, "truncate", None)
            if not callable(truncate):
                raise RuntimeError(
                    "rollout buffer does not support transactional truncation"
                )
            truncate(transition_indices[0])
            del entropy_samples[draft.entropy_start_index:]
            raise RuntimeError(
                "layerwise terminal infrastructure evaluation failed; "
                "the episode must not enter PPO rollout state"
            )
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
        zero_slot_resources = (
            (0.0,) * len(LAYERWISE_SLOT_NAMES),
        ) * horizon
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
        candidate_trials = raw_trials
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
        if candidate_trials is not None:
            candidate_store.append_trial_group(
                full_vector,
                candidate_trials,
                {
                    "identity_context": probe_identity_context,
                    "fidelity": "F1",
                    "episode_index": int(absolute_episode),
                    **exact_resource,
                    "variable_cost": float(variable_cost),
                    "action_matrix": [list(row) for row in action_matrix],
                    "boosted_overrides_hash": sha256_json(
                        episode_boosted_overrides
                    ),
                    "boosted_overrides_provenance": "layerwise_env",
                    "assessment_bootstrap_seed": int(bootstrap_seed),
                    "episode_reward": float(episode_reward),
                    "promotion_marker": "online_group",
                },
                compact=True,
            )
            evidence = candidate_store.trial_evidence_for_action(
                full_vector, probe_identity_context,
                max_trials=promotion_trials,
            )
            if evidence is None:
                raise RuntimeError("candidate evidence append was not readable")
            pooled_assessment = _assess_pooled_online_trials(
                raw_trials=candidate_trials,
                pooled_trials=evidence.trials,
                fresh_assessment=fresh_assessment,
                reference=env.base.statistical_reference,
                gate_probability=promotion_probability,
                bootstrap_seed=bootstrap_seed,
                assess_candidate_fn=assess_candidate_fn,
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
                boosted_overrides=episode_boosted_overrides,
                bootstrap_seed=bootstrap_seed,
                episode_reward=episode_reward,
                assess_candidate_fn=assess_candidate_fn,
                prefilter_probability=online_probability,
                promotion_probability=promotion_probability,
                target_trial_count=promotion_trials,
                validation_banks=validation_banks,
            )
            promotion_evidence = promotion.evidence or evidence
            candidate_key_value = promotion_evidence.candidate_key
            promotion_passed = bool(
                promotion.evidence is not None
                and promotion.evidence.promoted
                and (
                    (
                        validation_banks is not None
                        and promotion_evidence.trial_count
                        >= validation_banks.promotion_trial_count
                        and point_constraints_pass(
                            promotion.metrics or {},
                            validation_banks.promotion_reference,
                        )
                    )
                    or (
                        validation_banks is None
                        and promotion_evidence.trial_count >= promotion_trials
                        and _assessment_passes(
                            promotion.assessment, promotion_probability,
                        )
                    )
                )
            )
            if promotion_passed:
                existing_candidate = accepted_candidates.get(candidate_key_value)
                accepted_candidates[candidate_key_value] = {
                    **exact_resource,
                    "variable_cost": float(variable_cost),
                    "assessment": promotion.assessment,
                    "metrics": dict(promotion.metrics or {}),
                    "constraint_safety_margins": (
                        normalized_constraint_safety_margins(
                            promotion.metrics or {},
                            (
                                validation_banks.promotion_reference
                                if validation_banks is not None
                                else authoritative_base_env.statistical_reference
                            ),
                        )
                    ),
                    "action_matrix": action_matrix,
                    "full_vector": full_vector,
                    "boosted_overrides": _copy_boosted_overrides(
                        episode_boosted_overrides
                    ),
                    "reward": (
                        float(existing_candidate["reward"])
                        if existing_candidate is not None
                        and existing_candidate.get("reward") is not None
                        else float(episode_reward)
                    ),
                    "promotion_trials": promotion.evidence.trials,
                    "axis_counterfactuals": copy.deepcopy(
                        getattr(promotion, "axis_counterfactuals", None)
                    ),
                }

        local_episode += 1
        completed = local_episode

        if retain_history:
            rewards.append(episode_reward)
        entropy_snapshot = {
            "block4": None, "k": None,
            "block4_slot_count": 0, "k_slot_count": 0,
        }
        update_due = completed % update_window == 0 or completed == total_episodes
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
            entropy_snapshot = _current_policy_entropy(
                policy, entropy_samples, device,
            )
            latest_block4_entropy = entropy_snapshot["block4"]
            latest_k_entropy = entropy_snapshot["k"]
            strict_best_snapshot = _strict_best_snapshot(accepted_candidates)
            strict_revalidation_status = "not_due"
            maximum_reached = bool(
                absolute_start + completed
                >= (
                    total_episodes
                    if planned_total_episodes is None
                    else int(planned_total_episodes)
                )
            )
            if validation_banks is not None and maximum_reached:
                strict_revalidation_status, strict_best_snapshot = (
                    _certify_strict_best_candidates(
                        env=env,
                        promotion_base_env=authoritative_base_env,
                        candidate_store=candidate_store,
                        identity_context=identity_context,
                        accepted_candidates=accepted_candidates,
                        bootstrap_seed=(
                            int(base_seed or 0) + absolute_start + completed
                        ),
                        assess_candidate_fn=assess_candidate_fn,
                        final_probability=final_probability,
                        validation_banks=validation_banks,
                        exhaustive_fallback=True,
                    )
                )
            elif maximum_reached:
                strict_revalidation_status = "not_applicable"
            ppo_metrics.update({
                "completed_episodes": absolute_start + completed,
                "block4_entropy": latest_block4_entropy,
                "k_entropy": latest_k_entropy,
                "strict_revalidation_status": strict_revalidation_status,
                "termination_reason": (
                    "maximum_episodes" if maximum_reached else "running"
                ),
                "strict_best": strict_best_snapshot,
                "strict_pareto_frontier": _strict_pareto_snapshots(
                    accepted_candidates
                ),
            })
            if retain_history:
                ppo_diagnostics.append(ppo_metrics)

        invalid_steps = sum(
            not bool(_field(info.get("layer_summary", {}), "all_valid", True))
            for info in step_infos
        )
        episode_limit_reached = bool(
            absolute_start + completed
            >= (
                total_episodes
                if planned_total_episodes is None
                else int(planned_total_episodes)
            )
        )
        record = LayerwiseEpisodeRecord(
            episode_index=absolute_episode,
            reward=float(episode_reward),
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
                "F4_train_probe"
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
            final_config_fingerprint=str(
                runtime_info.get("final_config_fingerprint", "") or ""
            ),
            materialization_failure_reason=str(
                runtime_info.get("materialization_failure_reason", "") or ""
            ),
            model_uses_replan_config=bool(
                _to_plain_mapping(runtime_info.get("replan_application")).get(
                    "model_uses_replan_config", False
                )
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
            step_count=horizon,
            block4_entropy=entropy_snapshot["block4"],
            k_entropy=entropy_snapshot["k"],
            strict_revalidation_status=str(strict_revalidation_status),
            termination_reason=(
                "maximum_episodes" if episode_limit_reached else "running"
            ),
        )
        if retain_history:
            records.append(record)
        if on_episode_end is not None:
            on_episode_end(record)
        if update_due:
            if on_ppo_update_end is not None and ppo_metrics is not None:
                on_ppo_update_end(ppo_metrics, absolute_start + completed, record)
            rollout_buffer.clear()
            entropy_samples.clear()
            if stop_requested is not None and stop_requested():
                graceful_stopped = True
                break

    maximum_boundary_reached = bool(
        not graceful_stopped
        and absolute_start + local_episode
        >= (
            total_episodes
            if planned_total_episodes is None
            else int(planned_total_episodes)
        )
    )
    if (
            validation_banks is not None
            and maximum_boundary_reached
            and strict_revalidation_status != "passed"
    ):
        strict_revalidation_status, _strict_best = (
            _certify_strict_best_candidates(
                env=env,
                promotion_base_env=authoritative_base_env,
                candidate_store=candidate_store,
                identity_context=identity_context,
                accepted_candidates=accepted_candidates,
                bootstrap_seed=(
                    int(base_seed or 0) + absolute_start + local_episode
                ),
                assess_candidate_fn=assess_candidate_fn,
                final_probability=final_probability,
                validation_banks=validation_banks,
                exhaustive_fallback=True,
            )
        )
    elif validation_banks is None:
        strict_revalidation_status = "not_applicable"

    termination_reason = (
        "graceful_stop" if graceful_stopped else "maximum_episodes"
    )
    bank_b_best = _strict_best_snapshot(accepted_candidates)
    final_candidates = accepted_candidates
    if maximum_boundary_reached:
        final_candidates = {
            key: candidate
            for key, candidate in accepted_candidates.items()
            if candidate.get("final_revalidation_status") == "passed"
        }
    strict_best = _strict_best_snapshot(final_candidates)
    strict_pareto_frontier = _strict_pareto_snapshots(accepted_candidates)
    return {
        "strict_best": strict_best,
        "bank_b_best": bank_b_best,
        "strict_pareto_frontier": strict_pareto_frontier,
        "best_action": (
            list(strict_best["full_vector"]) if strict_best is not None else None
        ),
        "best_action_matrix": (
            [list(row) for row in strict_best["action_matrix"]]
            if strict_best is not None else None
        ),
        "best_layer_configurations": (
            copy.deepcopy(strict_best["layer_configurations"])
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
            _copy_boosted_overrides(strict_best["boosted_overrides"])
            if strict_best is not None else None
        ),
        "best_promotion_evidence": (
            copy.deepcopy(strict_best.get("promotion_evidence"))
            if strict_best is not None else None
        ),
        "best_axis_counterfactuals": (
            copy.deepcopy(strict_best.get("axis_counterfactuals"))
            if strict_best is not None else None
        ),
        "episode_rewards": rewards,
        "ppo_metrics": ppo_diagnostics,
        "episode_records": records,
        "probe_pool_schedule": probe_pool_schedule,
        "block4_entropy": latest_block4_entropy,
        "k_entropy": latest_k_entropy,
        "strict_revalidation_status": strict_revalidation_status,
        "termination_reason": termination_reason,
        "graceful_stopped": graceful_stopped,
        "completed_episodes": absolute_start + local_episode,
    }
