"""Torch-free introspection of a Rescale_optimizer replan output.

Small pure helpers that read the ``new_compact_config`` a replan returns, so
verification tooling can reason about the modulus chain without importing torch
(``rescale_optimizer_bridge`` pulls in ``function_handler`` → torch).
"""
from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Tuple


def fused_skeleton_positions(
        compact: Mapping[str, Any],
        baseline_skeleton: Sequence[int],
        skel_field_specs: Sequence[Tuple[str, Optional[int]]],
        ) -> List[Tuple[str, Optional[int]]]:
    """Return ``[(cfg_field, tuple_index), ...]`` for each baseline-skeleton
    RESCALE position (``r >= 1``) the optimizer FUSED AWAY.

    Mirrors ``apply_optimizer_output_to_cfg``'s fused detection exactly: a
    baseline rescale position is fused when its graph node (``baseline_skeleton[r]``,
    matched against the compact ``cut_point_sf`` index ``i``) is either ABSENT from
    ``cut_point_sf`` or kept as a PASSTHROUGH (present with ``sf`` but no ``sf_post``
    key — the accumulated scale flowing through, no drop). ``r == 0`` is the source
    fresh position and is never a rescale.

    ``compact`` is the replan ``new_compact_config`` dict; ``baseline_skeleton`` the
    baseline node-id sequence; ``skel_field_specs`` the ``(cfg_field, tuple_index)``
    per skeleton position (``DEFAULT_CFG_TO_T_NEW_MAP`` / the bridge's derived table,
    converted to plain tuples so this stays torch-free).
    """
    cut_points = {}
    for entry in (compact.get("cut_point_sf") or []):
        if isinstance(entry, Mapping) and "i" in entry:
            cut_points[int(entry["i"])] = entry


    if not cut_points:
        return []
    out: List[Tuple[str, Optional[int]]] = []
    for r, (cfg_field, tuple_index) in enumerate(skel_field_specs):
        if r == 0 or r >= len(baseline_skeleton):
            continue
        cpt = cut_points.get(int(baseline_skeleton[r]))
        if cpt is None or cpt.get("sf_post") is None:
            out.append((str(cfg_field), tuple_index))
    return out
