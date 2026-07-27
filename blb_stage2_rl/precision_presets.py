"""Canonical layerwise truncation presets and network trade-off helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

try:
    from .truncation_levels import K_LEVELS
except ImportError:  # pragma: no cover - legacy top-level import compatibility
    from truncation_levels import K_LEVELS


@dataclass(frozen=True)
class PrecisionPreset:
    name: str
    k_by_block: Tuple[int, int, int, int, int]
    communication_utility: float

    def __post_init__(self) -> None:
        if len(self.k_by_block) != 5:
            raise ValueError(f"{self.name}: precision preset must contain five K values")
        unsupported = [value for value in self.k_by_block if int(value) not in K_LEVELS]
        if unsupported:
            raise ValueError(f"{self.name}: unsupported K values {unsupported}")
        utility = float(self.communication_utility)
        if not math.isfinite(utility) or not 0.0 <= utility <= 1.0:
            raise ValueError(f"{self.name}: communication utility must be in [0, 1]")


PRECISION_PRESETS = (
    PrecisionPreset("high", (11, 10, 10, 12, 11), 0.0),
    PrecisionPreset("medium", (9, 8, 8, 10, 9), 0.5),
    PrecisionPreset("low", (7, 6, 6, 8, 7), 1.0),
)
PRECISION_PRESET_NAMES = tuple(preset.name for preset in PRECISION_PRESETS)
PRECISION_PRESET_VERSION = "stage2_precision_presets_hml_v1"


def precision_preset(index: int) -> PrecisionPreset:
    normalized = int(index)
    if not 0 <= normalized < len(PRECISION_PRESETS):
        raise ValueError(
            f"precision preset index {normalized} outside [0, {len(PRECISION_PRESETS)})"
        )
    return PRECISION_PRESETS[normalized]


def validate_communication_importance_ratio(value: float) -> float:
    ratio = float(value)
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("communication importance ratio must be finite and positive")
    return ratio


def network_axis_weights(
        communication_importance_ratio: float,
        ) -> Tuple[float, float]:
    """Return normalized compute and communication weights."""
    ratio = validate_communication_importance_ratio(
        communication_importance_ratio,
    )
    denominator = 1.0 + ratio
    return 1.0 / denominator, ratio / denominator


def allocated_precision_tolerances(
        total_precision_tolerance: float,
        communication_importance_ratio: float,
        ) -> Tuple[float, float]:
    """Allocate the total one-sided precision budget between resource axes."""
    tolerance = float(total_precision_tolerance)
    if not math.isfinite(tolerance) or not 0.0 <= tolerance < 1.0:
        raise ValueError("total precision tolerance must be finite and in [0, 1)")
    compute_weight, communication_weight = network_axis_weights(
        communication_importance_ratio,
    )
    return tolerance * compute_weight, tolerance * communication_weight


__all__ = [
    "PRECISION_PRESETS",
    "PRECISION_PRESET_NAMES",
    "PRECISION_PRESET_VERSION",
    "PrecisionPreset",
    "allocated_precision_tolerances",
    "network_axis_weights",
    "precision_preset",
    "validate_communication_importance_ratio",
]
