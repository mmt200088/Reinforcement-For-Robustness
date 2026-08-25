"""Canonical layerwise truncation presets and network trade-off helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

from .truncation_levels import K_LEVELS


@dataclass(frozen=True)
class PrecisionPreset:
    # Ciphertext K is report metadata; simulation K is installed in the model.
    name: str
    ciphertext_k_by_block: Tuple[int, int, int, int, int]
    simulation_k_by_block: Tuple[int, int, int, int, int]
    ciphertext_ring_bits: int
    communication_utility: float

    def __post_init__(self) -> None:
        for field_name, values in (
                ("ciphertext", self.ciphertext_k_by_block),
                ("simulation", self.simulation_k_by_block),
        ):
            if len(values) != 5:
                raise ValueError(
                    f"{self.name}: {field_name} precision preset must contain five K values"
                )
            unsupported = [value for value in values if int(value) not in K_LEVELS]
            if unsupported:
                raise ValueError(
                    f"{self.name}: unsupported {field_name} K values {unsupported}"
                )
        if any(value < 0 for value in self.reserve_bits_by_block):
            raise ValueError(
                f"{self.name}: simulation K cannot exceed ciphertext K"
            )
        if int(self.ciphertext_ring_bits) <= 0:
            raise ValueError(
                f"{self.name}: ciphertext ring bits must be positive"
            )
        utility = float(self.communication_utility)
        if not math.isfinite(utility) or not 0.0 <= utility <= 1.0:
            raise ValueError(f"{self.name}: communication utility must be in [0, 1]")

    @property
    def reserve_bits_by_block(self) -> Tuple[int, int, int, int, int]:
        return tuple(
            int(ciphertext_k) - int(simulation_k)
            for ciphertext_k, simulation_k in zip(
                self.ciphertext_k_by_block,
                self.simulation_k_by_block,
            )
        )

    @property
    def k_by_block(self) -> Tuple[int, int, int, int, int]:
        """Compatibility alias for the unchanged executable simulation K."""
        return self.simulation_k_by_block


PRECISION_PRESETS = (
    PrecisionPreset(
        "high",
        (13, 13, 13, 13, 13),
        (11, 10, 10, 12, 11),
        40,
        0.0,
    ),
    PrecisionPreset(
        "medium",
        (12, 12, 12, 12, 12),
        (9, 8, 8, 10, 9),
        39,
        0.5,
    ),
    PrecisionPreset(
        "low",
        (11, 11, 11, 12, 11),
        (7, 6, 6, 8, 7),
        38,
        1.0,
    ),
)
PRECISION_PRESET_NAMES = tuple(preset.name for preset in PRECISION_PRESETS)
PRECISION_PRESET_VERSION = "stage2_precision_presets_hml_v1"
PRECISION_PRESET_METADATA_VERSION = "stage2_precision_presets_paper_semantics_v1"


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
    "PRECISION_PRESET_METADATA_VERSION",
    "PRECISION_PRESET_NAMES",
    "PRECISION_PRESET_VERSION",
    "PrecisionPreset",
    "allocated_precision_tolerances",
    "network_axis_weights",
    "precision_preset",
    "validate_communication_importance_ratio",
]
