"""Conservative one-trial screening for Stage-2 terminal probes.

K=1 is only allowed to reject an extreme precision failure. It never estimates
stability and never authorizes candidate promotion or final selection.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Sequence

import numpy as np

from .statistical_constraints import BaselineReference, ConstraintAssessment


@dataclass(frozen=True)
class ProtectedK1Config:
    enabled: bool = False
    guard_sigma: float = 4.0
    audit_fraction: float = 0.02

    def __post_init__(self) -> None:
        guard = float(self.guard_sigma)
        audit = float(self.audit_fraction)
        if not math.isfinite(guard) or guard <= 0.0:
            raise ValueError("guard_sigma must be finite and positive")
        if not math.isfinite(audit) or not 0.0 <= audit <= 1.0:
            raise ValueError("audit_fraction must be in [0, 1]")
        object.__setattr__(self, "guard_sigma", guard)
        object.__setattr__(self, "audit_fraction", audit)


@dataclass(frozen=True)
class ProtectedK1Decision:
    screened: bool
    reason: str
    violating_channels: tuple[str, ...]
    worst_precision_z: float
    assessment: ConstraintAssessment


def _normalize_trial(
        trial: Sequence[float],
        ) -> tuple[float, float, float]:
    if len(trial) != 3:
        raise ValueError("protected K=1 trial must contain loss, metric1, metric2")
    values = tuple(float(value) for value in trial)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("protected K=1 trial values must be finite")
    return values


def _gate_probability(value: float) -> float:
    gate = float(value)
    if not math.isfinite(gate) or not 0.0 < gate <= 1.0:
        raise ValueError("gate_probability must be in (0, 1]")
    return gate


def assess_single_trial_precision(
        trial: Sequence[float],
        reference: BaselineReference,
        *,
        gate_probability: float,
        ) -> ConstraintAssessment:
    """Score only precision against baseline bootstrap rows.

    Stability probabilities are neutral placeholders. Callers must retain the
    explicit ``protected_k1_stability_measured=False`` provenance and may use
    this assessment only for an already-screened P1 reward.
    """
    if not isinstance(reference, BaselineReference):
        raise TypeError("reference must be a BaselineReference")
    loss, metric1, metric2 = _normalize_trial(trial)
    gate = _gate_probability(gate_probability)
    tolerance = float(reference.precision_tolerance)

    loss_probability = float(np.mean(
        loss <= reference.bootstrap_means["loss"] * (1.0 + tolerance)
    ))
    metric1_probability = float(np.mean(
        metric1 >= reference.bootstrap_means["metric1"] * (1.0 - tolerance)
    ))
    metric2_probability = float(np.mean(
        metric2 >= reference.bootstrap_means["metric2"] * (1.0 - tolerance)
    ))
    precision_probability = min(
        loss_probability, metric1_probability, metric2_probability,
    )
    return ConstraintAssessment(
        loss_precision_probability=loss_probability,
        metric1_precision_probability=metric1_probability,
        metric2_precision_probability=metric2_probability,
        loss_stability_probability=1.0,
        metric1_stability_probability=1.0,
        metric2_stability_probability=1.0,
        precision_probability=precision_probability,
        stability_probability=1.0,
        gate_probability=gate,
        online_precision_pass=precision_probability >= gate,
        online_stability_pass=True,
    )


def decide_protected_k1(
        trial: Sequence[float],
        reference: BaselineReference,
        *,
        guard_sigma: float,
        gate_probability: float,
        force_k5: bool,
        ) -> ProtectedK1Decision:
    """Return whether one extreme precision failure may stop before K=5."""
    if not isinstance(reference, BaselineReference):
        raise TypeError("reference must be a BaselineReference")
    guard = float(guard_sigma)
    if not math.isfinite(guard) or guard <= 0.0:
        raise ValueError("guard_sigma must be finite and positive")
    loss, metric1, metric2 = _normalize_trial(trial)
    assessment = assess_single_trial_precision(
        (loss, metric1, metric2),
        reference,
        gate_probability=gate_probability,
    )
    # robust_constrained_reward assigns P1 only below probability 0.5. If an
    # experiment raises its online gate above 0.5, screening a probability in
    # [0.5, gate) would manufacture a K=1 "reject" that the reward labels P3.
    reject_probability = min(float(gate_probability), 0.5)
    z_by_channel = {
        "loss": (loss - float(reference.loss_limit)) / float(reference.loss_std),
        "metric1": (
            float(reference.metric1_limit) - metric1
        ) / float(reference.metric1_std),
        "metric2": (
            float(reference.metric2_limit) - metric2
        ) / float(reference.metric2_std),
    }
    violating_channels = tuple(
        channel for channel in ("loss", "metric1", "metric2")
        if z_by_channel[channel] > guard
    )
    worst_z = max(z_by_channel.values())
    if force_k5:
        reason = "protected_by_frontier"
        screened = False
    elif not violating_channels:
        reason = "within_guard"
        screened = False
    elif assessment.precision_probability >= reject_probability:
        reason = "bootstrap_fail_open"
        screened = False
    else:
        reason = "extreme_precision_failure"
        screened = True
    return ProtectedK1Decision(
        screened=screened,
        reason=reason,
        violating_channels=violating_channels,
        worst_precision_z=float(worst_z),
        assessment=assessment,
    )


def should_audit_protected_k1(
        base_seed: int,
        absolute_episode: int,
        audit_fraction: float,
        ) -> bool:
    """Select a deterministic fraction of screened episodes for exact K=5."""
    fraction = float(audit_fraction)
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("audit_fraction must be in [0, 1]")
    if fraction <= 0.0:
        return False
    if fraction >= 1.0:
        return True
    episode = int(absolute_episode)
    if episode < 0:
        raise ValueError("absolute_episode must be nonnegative")
    payload = (
        f"protected-k1-audit:{int(base_seed)}:{episode}".encode("ascii")
    )
    sample = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return sample < int(fraction * (1 << 64))


__all__ = [
    "ProtectedK1Config",
    "ProtectedK1Decision",
    "assess_single_trial_precision",
    "decide_protected_k1",
    "should_audit_protected_k1",
]
