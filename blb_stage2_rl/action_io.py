"""Human-readable action ↔ action_vec converters.

The RL policy outputs an integer ``action_vec`` (877 dims for L=12) where each
slot's value is an *action index* into per-slot level table. That's the right
representation for the policy but is opaque for humans: index 5 in block 2's
``wq_encode`` slot vs. block 3's ``square_rescale_sf_0`` slot mean completely
different *scaling factors* — and humans think in scaling factors, not indices.

This module converts between the two views:

* :func:`action_vec_to_slots_list` — given an ``action_vec`` (RL's view),
  produce a list of records, one per slot, each labelled with its model
  location and carrying the *decoded* ``scaling_factor`` (or
  ``truncation_bits`` for K slots). This is the format the diagnostics
  recorder writes to ``best_action_vec.json``.

* :func:`slots_list_to_action_vec` — inverse: given a list of slot overrides
  (label + scaling_factor/truncation_bits), produce an ``action_vec``. Used by
  Paean's ``load_action_grid_config`` so the user can write a JSON that says
  ``{"label":"L05.B3.K","truncation_bits":10}`` instead of ``[3,4,5,...]``.

Schema (one entry in the slots list)
------------------------------------

For SF kinds (F, W, M, S):
::

  {
    "label":          "L05.B5.W.wffn1",       // unique, model-location-aware
    "layer":          5,
    "block":          5,
    "kind":           "W",
    "field_name":     "wffn1_sf",             // exact action_space field name
    "operation":      "block5_wffn1_encode",  // graph-node-ish name
    "location":       "layer5.block5.wffn1_sf",
    "distribution":   "weight",
    "scaling_factor": 14,                     // PRIMARY user-facing value
    "action_index":   5,                      // back-compat / sanity check
    "level_values":   [8, 10, 12, 14],        // what other choices look like
    "max_sf":         14,
    "N":              14,
    "effective":      true,
    "note":           ""
  }

For R (rescale) kind: same shape, but ``scaling_factor`` can be ``null`` →
that rescale point is *off* at this action (action_index==0).

For K kind:
::

  {
    "label":           "L05.B3.K",
    "layer":           5,
    "block":           3,
    "kind":            "K",
    "operation":       "block3_output_truncation",
    "truncation_bits": 13,                    // PRIMARY user-facing value
    "action_index":    3,
    "level_values":    [8, 9, 11, 13, 10, 12, 6, 7]
  }

The legacy ``first_input`` slot (last element of action_vec, layer 0, no block)
is deprecated. It may appear in old descriptions for compatibility, but new
slot-list overrides must not select it.
::

  {
    "label":          "L0.first_input.F",
    "layer":          0,
    "block":          null,
    "kind":           "F",
    "scaling_factor": 26,
    ...
  }

Robustness
----------
The ``_value_to_action_index`` parser is **forgiving**: if the user passes a
scaling_factor that snapped to a different value (e.g. user wrote 13 but the
table contains only {8,10,12,14}), we pick the *closest available* level and
warn via ``_ValueCoerced`` in the returned tuple. This lets non-RL operators
edit configs by approximation instead of looking up the table for every slot.
"""
from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .action_space import (
    BLB_FIRST_INPUT_N,
    K_LEVELS,
    LEVELS_FIRST_INPUT,
    LEVELS_R,
    MaxSFsTable,
    NOISE_TABLE_ALLOWED_SCALING_FACTORS_BY_N,
    NUM_LEVELS_PER_DIM_BY_BLOCK_KIND,
    describe_action_vector,
    load_max_sfs,
    make_all_max_action_vector,
    make_all_min_action_vector,
    per_layer_field_offsets,
    sf_from,
    _rescale_sf_from_index,
    _snap_to_table,
    _block_default_N,
    _degree_for_layer,
    validate_action_vector,
)


SCHEMA_VERSION = "blb_v3_slots_human_v1"

# ---------------------------------------------------------------------------
# action_vec → human-readable slots list (output direction)
# ---------------------------------------------------------------------------

def action_vec_to_slots_list(
        action_vec: Sequence[int],
        *,
        max_sfs: MaxSFsTable,
        num_layers: int,
        gelu_degree: object = 4,
        attn_degree: object = 4,
        profile: str = "default",
        ) -> List[Dict[str, Any]]:
    """Return a list of human-readable slot dicts for ``action_vec``.

    Wraps :func:`blb_stage2_rl.action_space.describe_action_vector` and
    renames its fields for end-user clarity (``value`` → ``scaling_factor``
    or ``truncation_bits`` depending on kind).
    """
    description = describe_action_vector(
        action_vec,
        max_sfs=max_sfs,
        num_layers=int(num_layers),
        gelu_degree=gelu_degree,
        attn_degree=attn_degree,
        profile=str(profile),
    )
    records = description.get("records") or []
    out: List[Dict[str, Any]] = []
    for rec in records:
        kind = str(rec.get("kind", ""))
        entry: Dict[str, Any] = {
            "label": str(rec.get("slot_label", "")),
            "global_index": int(rec.get("global_index", -1)),
            "layer": int(rec.get("layer", 0)),
            "block": rec.get("block_index"),
            "kind": kind,
            "field_name": str(rec.get("field", "")),
            "operation": str(rec.get("operation", "")),
            "location": str(rec.get("location", "")),
            "distribution": str(rec.get("distribution", "")),
            "action_index": int(rec.get("action_index", 0)),
            "level_values": list(rec.get("level_values") or []),
            "N": rec.get("N"),
            "max_sf": rec.get("max_sf"),
            "effective": bool(rec.get("effective", True)),
        }
        if kind == "K":
            # PRIMARY value: truncation_bits
            entry["truncation_bits"] = rec.get("value")
        else:
            # PRIMARY value: scaling_factor (or None for "rescale off")
            entry["scaling_factor"] = rec.get("value")
        if rec.get("note"):
            entry["note"] = str(rec.get("note"))
        out.append(entry)
    return out


def action_vec_to_slots_dict(
        action_vec: Sequence[int],
        **kwargs,
        ) -> Dict[str, Dict[str, Any]]:
    """Same as :func:`action_vec_to_slots_list` but keyed by ``label``.

    Convenient when the user wants ``slots["L05.B3.K"]["truncation_bits"]``
    lookups in the JSON. Loses ordering but order is recoverable via
    ``global_index``.
    """
    return {row["label"]: row for row in action_vec_to_slots_list(action_vec, **kwargs)}


def group_slots_by_layer_block(
        slots_list: Sequence[Mapping[str, Any]],
        ) -> Dict[str, Dict[str, Any]]:
    """Group flat slots list into a layer→block nested view for printing.

    Output shape::

      {
        "L00": {
          "B2": {"F.qk_bsgs": 16, "F.ln_bias": 14, ..., "K": 10},
          "B3": {...},
          ...
        },
        "L01": {...},
        ...,
        "first_input": 26
      }

    Strictly for human display in the markdown summary; ``Paean`` reads the
    flat list/dict.
    """
    layers: Dict[str, Dict[str, Any]] = {}
    first_input_value = None
    for row in slots_list:
        layer_idx = int(row.get("layer", 0))
        block_idx = row.get("block")
        kind = str(row.get("kind", ""))
        # First-input slot: stash separately
        if block_idx is None:
            first_input_value = row.get("scaling_factor")
            continue
        label_key_short = row["label"].split(".", 2)[-1]  # everything after L00.B2.
        layer_key = f"L{layer_idx:02d}"
        block_key = f"B{int(block_idx)}"
        per_layer = layers.setdefault(layer_key, {})
        per_block = per_layer.setdefault(block_key, {})
        if kind == "K":
            per_block["K"] = row.get("truncation_bits")
        else:
            per_block[label_key_short] = row.get("scaling_factor")
    out: Dict[str, Any] = dict(sorted(layers.items()))
    if first_input_value is not None:
        out["first_input"] = first_input_value
    return out


# ---------------------------------------------------------------------------
# Human-readable slots list → action_vec (input direction)
# ---------------------------------------------------------------------------

@dataclass
class SlotOverrideError(ValueError):
    """Raised when a slot override cannot be reconciled to an action index."""
    label: str
    reason: str
    def __str__(self) -> str:
        return f"slot {self.label!r}: {self.reason}"


def _coerce_action_index_from_sf(
        *,
        kind: str,
        sf_value: Optional[int],
        max_sf: int,
        N: int,
        ) -> int:
    """Pick the action_index whose decoded SF is closest to ``sf_value``.

    Closest in *snapped* SF (post ``_snap_to_table``), with ties broken toward
    the larger index (= safer/larger SF). For R kind, ``sf_value is None`` →
    action_index 0 (rescale off).
    """
    if kind == "R" and sf_value in (None, "off", "OFF", ""):
        return 0
    levels = int(NUM_LEVELS_PER_DIM_BY_BLOCK_KIND[str(kind)])
    candidate_sfs: List[Optional[int]] = []
    for idx in range(levels):
        if kind == "R":
            raw_sf = _rescale_sf_from_index(idx, int(max_sf))
            if raw_sf is None:
                candidate_sfs.append(None)
                continue
        else:
            raw_sf = sf_from(idx, int(max_sf), levels)
        candidate_sfs.append(int(_snap_to_table(raw_sf, int(N))))
    if sf_value is None:
        # User asked for "any" but slot is not R kind — fall back to max idx.
        return levels - 1
    target = int(sf_value)
    best_idx = -1
    best_dist = float("inf")
    for idx, sf in enumerate(candidate_sfs):
        if sf is None:
            continue
        dist = abs(int(sf) - target)
        # Tie-break toward larger SF (= larger idx for monotone SF tables).
        if dist < best_dist or (dist == best_dist and idx > best_idx):
            best_dist = dist
            best_idx = idx
    if best_idx < 0:
        # All candidates None (shouldn't happen for non-R kinds).
        return levels - 1
    return best_idx


def _coerce_action_index_from_k(value: object) -> int:
    """Pick the K_LEVELS index closest to ``value``."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"truncation_bits must be an integer, got {value!r}")
    try:
        target = operator.index(value)
    except TypeError as exc:
        raise ValueError(
            f"truncation_bits must be an integer, got {value!r}"
        ) from exc
    levels = list(K_LEVELS)
    best_idx = 0
    best_dist = abs(int(levels[0]) - target)
    for idx, k in enumerate(levels[1:], start=1):
        d = abs(int(k) - target)
        if d < best_dist:
            best_dist = d
            best_idx = idx
    return best_idx


def _coerce_first_input_index(value: int) -> int:
    """Pick the first-input-fresh SF level index closest to ``value``.

    First-input table is fixed: max_sf=30, levels=5 → {22, 24, 26, 28, 30}.
    Snapped to N=BLB_FIRST_INPUT_N.
    """
    levels = int(LEVELS_FIRST_INPUT)
    max_sf = 30
    target = int(value)
    candidates = [int(_snap_to_table(sf_from(i, max_sf, levels), BLB_FIRST_INPUT_N)) for i in range(levels)]
    best_idx = 0
    best_dist = abs(candidates[0] - target)
    for i, sf in enumerate(candidates[1:], start=1):
        d = abs(int(sf) - target)
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx


def parse_slot_label(label: str) -> Dict[str, Any]:
    """Parse ``L{i}.B{n}.{kind}[.{short}]`` (or ``L{i}.first_input.{kind}``).

    Returns a dict with keys: ``layer``, ``block`` (None for first_input),
    ``kind``, ``short_field`` (or "" for K). The caller can resolve
    ``field_name`` by walking the per-layer offset table because the short
    label is a forward-only abbreviation.
    """
    text = str(label or "").strip()
    parts = text.split(".")
    if len(parts) < 3:
        raise ValueError(f"slot label {label!r} too short; expected L<i>.B<n>.<kind>[.<short>]")
    layer_part = parts[0]
    block_part = parts[1]
    kind = parts[2]
    short = ".".join(parts[3:]) if len(parts) > 3 else ""
    if not layer_part.startswith("L"):
        raise ValueError(f"slot label {label!r}: expected layer prefix 'L<n>'")
    try:
        layer = int(layer_part[1:])
    except ValueError as exc:
        raise ValueError(f"slot label {label!r}: cannot parse layer index: {exc}") from exc
    if block_part == "first_input":
        return {"layer": layer, "block": None, "kind": kind, "short_field": short}
    if not block_part.startswith("B"):
        raise ValueError(f"slot label {label!r}: expected block prefix 'B<n>' or 'first_input'")
    try:
        block = int(block_part[1:])
    except ValueError as exc:
        raise ValueError(f"slot label {label!r}: cannot parse block index: {exc}") from exc
    return {"layer": layer, "block": block, "kind": kind, "short_field": short}


def _build_label_to_offset_map(num_layers: int) -> Dict[str, Tuple[int, int, str, str]]:
    """Map every canonical slot ``label`` to ``(global_index, layer, kind, field_name)``.

    Built from the same source-of-truth used by the policy (``per_layer_field_offsets``
    and ``make_slot_label``). Idempotent: same num_layers always yields the same map.
    """
    from .action_space import make_slot_label
    fields = per_layer_field_offsets()
    layer_dim = len(fields)
    out: Dict[str, Tuple[int, int, str, str]] = {}
    for li in range(int(num_layers)):
        for field_offset, (block_idx, field_name, kind) in enumerate(fields):
            global_index = li * layer_dim + field_offset
            label = make_slot_label(li, int(block_idx), str(kind), str(field_name))
            out[label] = (int(global_index), int(li), str(kind), str(field_name))
    # First-input slot at very end
    first_offset = int(num_layers) * layer_dim
    first_label = make_slot_label(0, None, "F", "first_input_sf")
    out[first_label] = (int(first_offset), 0, "F", "first_input_sf")
    return out


def slots_list_to_action_vec(
        slots: Sequence[Mapping[str, Any]],
        *,
        max_sfs: MaxSFsTable,
        num_layers: int,
        gelu_degree: object = 4,
        attn_degree: object = 4,
        base_action_vec: Optional[Sequence[int]] = None,
        base: str = "max",
        ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Convert a user-supplied slot list to a complete ``action_vec``.

    Args:
        slots:              List of dicts. Each must have a ``label`` plus
                            either ``scaling_factor`` (for F/W/M/S/R kinds)
                            or ``truncation_bits`` (for K). Missing slots
                            inherit from ``base_action_vec`` / ``base``.
        max_sfs:            MaxSFsTable used by the RL run (must match what
                            the policy was trained with).
        num_layers:         Number of layers (e.g. 12 for bert-base).
        gelu_degree, attn_degree: Polynomial degrees (per layer or scalar).
        base_action_vec:    Pre-existing vec to layer overrides onto. Takes
                            precedence over ``base`` if both given.
        base:               Fallback if ``base_action_vec`` is None: one of
                            ``"max"``, ``"min"``. Default: ``"max"``.

    Returns:
        ``(action_vec, coercion_notes)``. ``coercion_notes`` is a list of
        dicts describing slots where the user's value didn't exactly match a
        table level (e.g. requested SF=13 → snapped to 14). Empty if all
        values matched exactly.
    """
    # Build base
    if base_action_vec is not None:
        vec = validate_action_vector(base_action_vec, int(num_layers)).copy()
    else:
        base_choice = str(base or "max").strip().lower()
        if base_choice in ("max", "all-max", "all_max"):
            vec = make_all_max_action_vector(int(num_layers)).astype(np.int64)
        elif base_choice in ("min", "all-min", "all_min"):
            vec = make_all_min_action_vector(int(num_layers)).astype(np.int64)
        else:
            raise ValueError(f"unknown base {base!r}; expected 'max' or 'min'")

    label_map = _build_label_to_offset_map(int(num_layers))
    coercion_notes: List[Dict[str, Any]] = []

    for entry in slots:
        if not isinstance(entry, Mapping):
            raise ValueError(f"slot entry must be a mapping, got {type(entry).__name__}: {entry!r}")
        label = str(entry.get("label", "")).strip()
        if not label:
            raise ValueError(f"slot entry missing 'label': {dict(entry)!r}")
        if label not in label_map:
            raise ValueError(
                f"unknown slot label {label!r}; not produced by this num_layers={num_layers} "
                f"action space. Use the labels from best_action_vec.json's `slots` list."
            )
        global_index, layer_idx, kind, field_name = label_map[label]

        # Effective gelu/attn degree for this layer (for table N lookup).
        li_gelu = _degree_for_layer(gelu_degree, layer_idx, int(num_layers), default=4, name="gelu_degree")
        li_attn = _degree_for_layer(attn_degree, layer_idx, int(num_layers), default=4, name="attn_degree")
        # Block index from label_map for non-first-input slots.
        block_idx = None
        if "first_input" not in label:
            block_idx = int(label.split(".")[1][1:])

        if label == "L0.first_input.F" or field_name == "first_input_sf":
            if entry.get("effective") is False:
                continue
            raise ValueError(
                f"slot {label}: first_input is deprecated and is not selectable; "
                "the first HE config is treated as lossless and no first_input "
                "noise is installed"
            )

        if kind == "K":
            if "truncation_bits" not in entry:
                raise ValueError(
                    f"slot {label}: K (truncation) slot requires field 'truncation_bits'"
                )
            if entry["truncation_bits"] is None and entry.get("effective") is False:
                continue
            requested_k = entry["truncation_bits"]
            new_idx = _coerce_action_index_from_k(requested_k)
            requested_k = operator.index(requested_k)
            old_idx = int(vec[global_index])
            vec[global_index] = int(new_idx)
            decoded_after = int(K_LEVELS[new_idx])
            if decoded_after != requested_k:
                coercion_notes.append({
                    "label": label,
                    "requested_truncation_bits": requested_k,
                    "applied_truncation_bits": decoded_after,
                    "old_action_index": old_idx,
                    "new_action_index": new_idx,
                    "reason": "snapped to nearest K_LEVELS entry",
                })
            continue

        if "scaling_factor" not in entry:
            raise ValueError(
                f"slot {label}: kind={kind!r} requires field 'scaling_factor'"
            )
        sf_value = entry["scaling_factor"]
        max_sf = int(max_sfs.get(int(block_idx), str(field_name), layer_idx=layer_idx))
        N = int(_block_default_N(int(block_idx), gelu_degree=li_gelu, attn_degree=li_attn))
        new_idx = _coerce_action_index_from_sf(
            kind=kind, sf_value=(None if sf_value in (None, "off") else int(sf_value)),
            max_sf=int(max_sf), N=int(N),
        )
        old_idx = int(vec[global_index])
        vec[global_index] = int(new_idx)
        if sf_value is not None and sf_value not in ("off",):
            if kind == "R":
                if new_idx == 0:
                    applied_sf = None
                else:
                    applied_sf = int(_snap_to_table(_rescale_sf_from_index(new_idx, max_sf), N))
            else:
                applied_sf = int(_snap_to_table(
                    sf_from(new_idx, max_sf, int(NUM_LEVELS_PER_DIM_BY_BLOCK_KIND[kind])), N,
                ))
            if applied_sf != int(sf_value):
                coercion_notes.append({
                    "label": label,
                    "requested_scaling_factor": int(sf_value),
                    "applied_scaling_factor": applied_sf,
                    "old_action_index": old_idx,
                    "new_action_index": new_idx,
                    "reason": "snapped to nearest table level",
                })
    return vec.astype(np.int64), coercion_notes


def slots_payload_to_action_vec(
        payload: Mapping[str, Any],
        *,
        max_sfs: MaxSFsTable,
        num_layers: int,
        gelu_degree: object = 4,
        attn_degree: object = 4,
        ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Read a JSON payload (one of the supported shapes) → action_vec.

    Supported payload shapes (all keys are optional unless noted):

    A) **Full slots list** — what the recorder writes::

         {"schema_version": "blb_v3_slots_human_v1",
          "num_layers": 12,
          "slots": [ {"label": "...", "scaling_factor": 14}, ... ]}

    B) **Slots dict** — keyed by label::

         {"num_layers": 12,
          "slots": {"L05.B3.K": {"truncation_bits": 10}, ...}}

    C) **base + overrides** — start from baseline, list only changes::

         {"num_layers": 12,
          "base": "max",
          "overrides": [{"label": "L05.B3.K", "truncation_bits": 10}, ...]}

    D) **Old action_vec format** — back-compat::

         {"num_layers": 12, "action_vec": [3, 4, 5, ...]}
    """
    if not isinstance(payload, Mapping):
        raise ValueError("slots payload must be a JSON object")

    # Shape D (back-compat) — flat action_vec
    av = payload.get("action_vec") or payload.get("base_action_vec") or payload.get("base_action")
    if av is not None and isinstance(av, (list, tuple)):
        return validate_action_vector(av, int(num_layers)).copy(), []

    # base / base_action_vec
    base_action_vec = None
    base_str = "max"
    base_field = payload.get("base")
    if isinstance(base_field, str):
        base_str = base_field
    elif isinstance(base_field, (list, tuple)):
        base_action_vec = base_field

    raw_slots = payload.get("slots")
    overrides = payload.get("overrides")

    slot_entries: List[Mapping[str, Any]] = []
    if isinstance(raw_slots, list):
        slot_entries = list(raw_slots)
    elif isinstance(raw_slots, Mapping):
        for label, value in raw_slots.items():
            if isinstance(value, Mapping):
                entry = dict(value)
                entry.setdefault("label", str(label))
                slot_entries.append(entry)
            elif isinstance(value, (int, float, type(None))):
                # Bare value: infer kind from the label (K vs SF).
                lbl = str(label)
                kind_part = lbl.split(".", 2)[-1].split(".")[0] if "." in lbl else ""
                # heuristic: if label ends in ".K" or contains ".K" alone → K
                if lbl.endswith(".K") or kind_part == "K":
                    slot_entries.append({"label": lbl, "truncation_bits": int(value)})
                else:
                    slot_entries.append({"label": lbl, "scaling_factor": value})
            else:
                raise ValueError(
                    f"slots[{label!r}] must be a mapping or numeric, got {type(value).__name__}"
                )

    if isinstance(overrides, list):
        slot_entries.extend(overrides)
    elif isinstance(overrides, Mapping):
        for label, value in overrides.items():
            if isinstance(value, Mapping):
                entry = dict(value)
                entry.setdefault("label", str(label))
                slot_entries.append(entry)
            elif isinstance(value, (int, float, type(None))):
                lbl = str(label)
                if lbl.endswith(".K"):
                    slot_entries.append({"label": lbl, "truncation_bits": int(value)})
                else:
                    slot_entries.append({"label": lbl, "scaling_factor": value})

    return slots_list_to_action_vec(
        slot_entries,
        max_sfs=max_sfs,
        num_layers=int(num_layers),
        gelu_degree=gelu_degree,
        attn_degree=attn_degree,
        base_action_vec=base_action_vec,
        base=base_str,
    )


__all__ = [
    "SCHEMA_VERSION",
    "action_vec_to_slots_list",
    "action_vec_to_slots_dict",
    "group_slots_by_layer_block",
    "slots_list_to_action_vec",
    "slots_payload_to_action_vec",
    "parse_slot_label",
]
