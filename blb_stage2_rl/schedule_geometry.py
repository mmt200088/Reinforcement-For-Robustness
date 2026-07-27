"""Torch-free schedule geometry shared by Stage-2 policy entry points."""
from __future__ import annotations

from typing import Any, Iterable


def schedule_max_num_levels(schedule: Iterable[Any]) -> int:
    """Return the widest categorical slot in a sequential policy schedule."""
    try:
        specs = list(schedule)
    except TypeError as exc:
        raise ValueError("schedule must be an iterable of step specs") from exc
    if not specs:
        raise ValueError("schedule must not be empty")

    widest = 0
    for step_idx, spec in enumerate(specs):
        has_fusion_width = hasattr(spec, "fusion_num_options")
        has_k_width = hasattr(spec, "k_num_levels")
        if has_fusion_width or has_k_width:
            if not (has_fusion_width and has_k_width):
                raise ValueError(
                    f"schedule step {step_idx} has incomplete fusion geometry"
                )
            raw_dims = (
                getattr(spec, "fusion_num_options"),
                getattr(spec, "k_num_levels"),
            )
        elif hasattr(spec, "slot_dims"):
            raw_dims = getattr(spec, "slot_dims")
        else:
            raise ValueError(
                f"schedule step {step_idx} exposes neither slot_dims nor "
                "fusion_num_options/k_num_levels"
            )

        try:
            dims = tuple(int(value) for value in raw_dims)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"schedule step {step_idx} has invalid categorical widths"
            ) from exc
        if not dims:
            raise ValueError(f"schedule step {step_idx} has no categorical slots")
        for slot_idx, width in enumerate(dims):
            if width <= 0:
                raise ValueError(
                    f"schedule step {step_idx} slot {slot_idx} width must be "
                    f"positive, got {width}"
                )
        widest = max(widest, max(dims))
    return int(widest)
