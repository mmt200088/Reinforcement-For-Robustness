"""Deterministic bootstrap constraints for robust Stage-2 evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import operator
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

_CHANNELS = ("loss", "metric1", "metric2")
_MINIMUM_BASELINE_TRIALS = 25
_MAX_BOOTSTRAP_INDEX_ELEMENTS = 1_000_000


class InsufficientBaselineTrials(ValueError):
    """Raised when baseline calibration has fewer than 25 pooled trials."""

    def __init__(self, trial_count: int) -> None:
        self.trial_count = int(trial_count)
        self.required_trial_count = _MINIMUM_BASELINE_TRIALS
        super().__init__(
            f"baseline calibration requires at least {self.required_trial_count} "
            f"pooled trials; received {self.trial_count}"
        )


class DegenerateBaselineVariance(ValueError):
    """Raised when one or more pooled baseline channels have invalid variance."""

    def __init__(self, channels: Sequence[str]) -> None:
        self.channels = tuple(str(channel) for channel in channels)
        super().__init__(
            "baseline variance must be finite and nonzero for channels: "
            + ", ".join(self.channels)
        )


@dataclass(frozen=True)
class TrialSeries:
    """Aligned raw loss and metric trials, optionally carrying their seeds."""

    loss: Sequence[float]
    metric1: Sequence[float]
    metric2: Sequence[float]
    seeds: Sequence[int] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        normalized = {}
        for channel in _CHANNELS:
            try:
                values = tuple(float(value) for value in getattr(self, channel))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{channel} must be a finite numeric sequence") from exc
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{channel} must contain only finite values")
            normalized[channel] = values

        lengths = {len(normalized[channel]) for channel in _CHANNELS}
        if len(lengths) != 1:
            raise ValueError("loss, metric1, and metric2 must have equal lengths")
        trial_count = len(normalized["loss"])
        if trial_count == 0:
            raise ValueError("trial channels must contain at least one value")

        try:
            normalized_seeds = []
            for seed in self.seeds:
                if isinstance(seed, (bool, np.bool_)):
                    raise TypeError
                normalized_seeds.append(operator.index(seed))
        except (TypeError, ValueError) as exc:
            raise ValueError("seeds must be an integer sequence") from exc
        seeds = tuple(normalized_seeds)
        if seeds and len(seeds) != trial_count:
            raise ValueError(
                f"seeds length {len(seeds)} does not match trial count {trial_count}"
            )

        for channel, values in normalized.items():
            object.__setattr__(self, channel, values)
        object.__setattr__(self, "seeds", seeds)


@dataclass(frozen=True)
class BaselineReference:
    """Pooled baseline point estimates and precomputed bootstrap rows."""

    trials: TrialSeries
    trial_count: int
    precision_tolerance: float
    stability_multiplier: float
    bootstrap_seed: int
    bootstrap_samples: int
    loss_mean: float
    metric1_mean: float
    metric2_mean: float
    loss_std: float
    metric1_std: float
    metric2_std: float
    loss_limit: float
    metric1_limit: float
    metric2_limit: float
    loss_std_limit: float
    metric1_std_limit: float
    metric2_std_limit: float
    bootstrap_means: Mapping[str, np.ndarray]
    bootstrap_stds: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        sample_count = _positive_integer("bootstrap_samples", self.bootstrap_samples)
        object.__setattr__(self, "bootstrap_samples", sample_count)
        object.__setattr__(
            self,
            "bootstrap_means",
            _validated_bootstrap_mapping(
                "bootstrap_means",
                self.bootstrap_means,
                sample_count,
            ),
        )
        object.__setattr__(
            self,
            "bootstrap_stds",
            _validated_bootstrap_mapping(
                "bootstrap_stds",
                self.bootstrap_stds,
                sample_count,
            ),
        )


@dataclass(frozen=True)
class ConstraintAssessment:
    """Six independent feasibility probabilities and their online gates."""

    loss_precision_probability: float
    metric1_precision_probability: float
    metric2_precision_probability: float
    loss_stability_probability: float
    metric1_stability_probability: float
    metric2_stability_probability: float
    precision_probability: float
    stability_probability: float
    gate_probability: float
    online_precision_pass: bool
    online_stability_pass: bool


def _finite_float(name: str, value: float) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite number") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _nonnegative_seed(name: str, value: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a non-negative integer")
    try:
        normalized = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a non-negative integer") from exc
    if normalized < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return normalized


def _positive_integer(name: str, value: int) -> int:
    normalized = _nonnegative_seed(name, value)
    if normalized == 0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _readonly_array(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


def _validated_bootstrap_mapping(
    name: str,
    mapping: Mapping[str, np.ndarray],
    sample_count: int,
) -> Mapping[str, np.ndarray]:
    if not isinstance(mapping, Mapping):
        raise TypeError(f"{name} must be a mapping")
    copied_mapping = dict(mapping)
    required_keys = set(_CHANNELS)
    actual_keys = set(copied_mapping)
    if actual_keys != required_keys:
        missing = sorted(required_keys - actual_keys)
        extra = sorted(repr(key) for key in actual_keys - required_keys)
        raise ValueError(
            f"{name} must contain exactly {_CHANNELS}; "
            f"missing={missing}, extra={extra}"
        )

    normalized = {}
    for channel in _CHANNELS:
        try:
            values = np.asarray(copied_mapping[channel], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}[{channel!r}] must be a finite numeric array") from exc
        expected_shape = (sample_count,)
        if values.shape != expected_shape:
            raise ValueError(
                f"{name}[{channel!r}] shape {values.shape} != {expected_shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name}[{channel!r}] must contain only finite values")
        normalized[channel] = _readonly_array(values)
    return MappingProxyType(normalized)


def _bootstrap_summaries(
    arrays: Mapping[str, np.ndarray],
    indices: np.ndarray,
) -> tuple[Mapping[str, np.ndarray], Mapping[str, np.ndarray]]:
    means = {}
    stds = {}
    with np.errstate(over="ignore", invalid="ignore"):
        for channel in _CHANNELS:
            samples = arrays[channel][indices]
            means[channel] = np.mean(samples, axis=1)
            stds[channel] = np.std(samples, axis=1, ddof=1)
    return means, stds


def build_baseline_reference(
    groups: Sequence[TrialSeries],
    *,
    precision_tolerance: float,
    stability_multiplier: float,
    bootstrap_samples: int,
    seed: int,
) -> BaselineReference:
    """Pool baseline groups and precompute deterministic bootstrap summaries."""

    tolerance = _finite_float("precision_tolerance", precision_tolerance)
    if not 0.0 <= tolerance < 1.0:
        raise ValueError("precision_tolerance must be in [0, 1)")
    multiplier = _finite_float("stability_multiplier", stability_multiplier)
    if multiplier <= 0.0:
        raise ValueError("stability_multiplier must be positive")
    sample_count = _positive_integer("bootstrap_samples", bootstrap_samples)
    bootstrap_seed = _nonnegative_seed("seed", seed)

    normalized_groups = tuple(groups)
    if not all(isinstance(group, TrialSeries) for group in normalized_groups):
        raise TypeError("groups must contain only TrialSeries values")
    trial_count = sum(len(group.loss) for group in normalized_groups)
    if trial_count < _MINIMUM_BASELINE_TRIALS:
        raise InsufficientBaselineTrials(trial_count)

    groups_with_seeds = sum(bool(group.seeds) for group in normalized_groups)
    if groups_with_seeds not in (0, len(normalized_groups)):
        raise ValueError("baseline groups must either all provide seeds or all omit them")

    pooled_values = {
        channel: tuple(
            value
            for group in normalized_groups
            for value in getattr(group, channel)
        )
        for channel in _CHANNELS
    }
    pooled_seeds = (
        tuple(seed_value for group in normalized_groups for seed_value in group.seeds)
        if groups_with_seeds
        else ()
    )
    pooled_trials = TrialSeries(**pooled_values, seeds=pooled_seeds)
    arrays = {
        channel: np.asarray(getattr(pooled_trials, channel), dtype=np.float64)
        for channel in _CHANNELS
    }

    with np.errstate(over="ignore", invalid="ignore"):
        means = {channel: float(np.mean(arrays[channel])) for channel in _CHANNELS}
        stds = {
            channel: float(np.std(arrays[channel], ddof=1))
            for channel in _CHANNELS
        }
    degenerate_channels = tuple(
        channel
        for channel in _CHANNELS
        if (
            np.all(arrays[channel] == arrays[channel][0])
            or not math.isfinite(stds[channel])
            or stds[channel] <= 0.0
        )
    )
    if degenerate_channels:
        raise DegenerateBaselineVariance(degenerate_channels)
    nonfinite_means = tuple(
        channel for channel in _CHANNELS if not math.isfinite(means[channel])
    )
    if nonfinite_means:
        raise ValueError(
            "baseline means must be finite for channels: "
            + ", ".join(nonfinite_means)
        )

    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, trial_count, size=(sample_count, trial_count))
    bootstrap_means, bootstrap_stds = _bootstrap_summaries(arrays, indices)

    return BaselineReference(
        trials=pooled_trials,
        trial_count=trial_count,
        precision_tolerance=tolerance,
        stability_multiplier=multiplier,
        bootstrap_seed=bootstrap_seed,
        bootstrap_samples=sample_count,
        loss_mean=means["loss"],
        metric1_mean=means["metric1"],
        metric2_mean=means["metric2"],
        loss_std=stds["loss"],
        metric1_std=stds["metric1"],
        metric2_std=stds["metric2"],
        loss_limit=means["loss"] * (1.0 + tolerance),
        metric1_limit=means["metric1"] * (1.0 - tolerance),
        metric2_limit=means["metric2"] * (1.0 - tolerance),
        loss_std_limit=stds["loss"] * multiplier,
        metric1_std_limit=stds["metric1"] * multiplier,
        metric2_std_limit=stds["metric2"] * multiplier,
        bootstrap_means=bootstrap_means,
        bootstrap_stds=bootstrap_stds,
    )


def _probability(pass_rows: np.ndarray) -> float:
    return float(np.clip(np.mean(pass_rows), 0.0, 1.0))


def assess_candidate(
    trials: TrialSeries,
    reference: BaselineReference,
    *,
    gate_probability: float,
    bootstrap_seed: int,
) -> ConstraintAssessment:
    """Estimate six candidate pass probabilities against a baseline reference."""

    if not isinstance(trials, TrialSeries):
        raise TypeError("trials must be a TrialSeries")
    if not isinstance(reference, BaselineReference):
        raise TypeError("reference must be a BaselineReference")
    trial_count = len(trials.loss)
    if trial_count < 2:
        raise ValueError("candidate assessment requires at least two trials")

    gate = _finite_float("gate_probability", gate_probability)
    if not 0.0 < gate <= 1.0:
        raise ValueError("gate_probability must be in (0, 1]")
    candidate_seed = _nonnegative_seed("bootstrap_seed", bootstrap_seed)

    arrays = {
        channel: np.asarray(getattr(trials, channel), dtype=np.float64)
        for channel in _CHANNELS
    }
    tolerance = reference.precision_tolerance
    multiplier = reference.stability_multiplier
    sample_count = reference.bootstrap_samples
    rows_per_chunk = max(1, _MAX_BOOTSTRAP_INDEX_ELEMENTS // trial_count)
    pass_counts = {
        "loss_precision": 0,
        "metric1_precision": 0,
        "metric2_precision": 0,
        "loss_stability": 0,
        "metric1_stability": 0,
        "metric2_stability": 0,
    }
    rng = np.random.default_rng(candidate_seed)
    for start in range(0, sample_count, rows_per_chunk):
        stop = min(sample_count, start + rows_per_chunk)
        indices = rng.integers(0, trial_count, size=(stop - start, trial_count))
        candidate_means, candidate_stds = _bootstrap_summaries(arrays, indices)
        pass_counts["loss_precision"] += int(np.count_nonzero(
            candidate_means["loss"]
            <= reference.bootstrap_means["loss"][start:stop] * (1.0 + tolerance)
        ))
        pass_counts["metric1_precision"] += int(np.count_nonzero(
            candidate_means["metric1"]
            >= reference.bootstrap_means["metric1"][start:stop] * (1.0 - tolerance)
        ))
        pass_counts["metric2_precision"] += int(np.count_nonzero(
            candidate_means["metric2"]
            >= reference.bootstrap_means["metric2"][start:stop] * (1.0 - tolerance)
        ))
        pass_counts["loss_stability"] += int(np.count_nonzero(
            candidate_stds["loss"]
            <= multiplier * reference.bootstrap_stds["loss"][start:stop]
        ))
        pass_counts["metric1_stability"] += int(np.count_nonzero(
            candidate_stds["metric1"]
            <= multiplier * reference.bootstrap_stds["metric1"][start:stop]
        ))
        pass_counts["metric2_stability"] += int(np.count_nonzero(
            candidate_stds["metric2"]
            <= multiplier * reference.bootstrap_stds["metric2"][start:stop]
        ))
    probabilities = {
        name: float(np.clip(count / sample_count, 0.0, 1.0))
        for name, count in pass_counts.items()
    }
    precision_probability = min(
        probabilities["loss_precision"],
        probabilities["metric1_precision"],
        probabilities["metric2_precision"],
    )
    stability_probability = min(
        probabilities["loss_stability"],
        probabilities["metric1_stability"],
        probabilities["metric2_stability"],
    )

    return ConstraintAssessment(
        loss_precision_probability=probabilities["loss_precision"],
        metric1_precision_probability=probabilities["metric1_precision"],
        metric2_precision_probability=probabilities["metric2_precision"],
        loss_stability_probability=probabilities["loss_stability"],
        metric1_stability_probability=probabilities["metric1_stability"],
        metric2_stability_probability=probabilities["metric2_stability"],
        precision_probability=precision_probability,
        stability_probability=stability_probability,
        gate_probability=gate,
        online_precision_pass=precision_probability >= gate,
        online_stability_pass=stability_probability >= gate,
    )


def retarget_constraint_assessment(
    assessment: ConstraintAssessment,
    *,
    gate_probability: float,
) -> ConstraintAssessment:
    """Apply a new gate to already-computed bootstrap probabilities."""
    if not isinstance(assessment, ConstraintAssessment):
        raise TypeError("assessment must be a ConstraintAssessment")
    gate = _finite_float("gate_probability", gate_probability)
    if not 0.0 < gate <= 1.0:
        raise ValueError("gate_probability must be in (0, 1]")
    return replace(
        assessment,
        gate_probability=gate,
        online_precision_pass=assessment.precision_probability >= gate,
        online_stability_pass=assessment.stability_probability >= gate,
    )


__all__ = [
    "BaselineReference",
    "ConstraintAssessment",
    "DegenerateBaselineVariance",
    "InsufficientBaselineTrials",
    "TrialSeries",
    "assess_candidate",
    "build_baseline_reference",
    "retarget_constraint_assessment",
]
