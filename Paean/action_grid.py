from __future__ import annotations

import itertools
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from json_utils import read_json_file
from blb_stage2_rl.action_space import (
    K_LEVELS,
    NUM_LEVELS_PER_DIM_BY_BLOCK_KIND,
    action_dims_for_config,
    action_vector_to_cfgs,
    build_optimizer_requests,
    layer_dims,
    load_max_sfs,
    make_all_max_action_vector,
    make_all_min_action_vector,
    per_layer_field_offsets,
    sf_from,
    sum_truncation_k_in_action,
)

_K_LEVEL_INDEX: Dict[int, int] = {int(value): idx for idx, value in enumerate(K_LEVELS)}
_SORTED_K_LEVEL_CHOICES: Tuple[int, ...] = tuple(sorted(int(value) for value in K_LEVELS))
_SELECTOR_SLOT_CACHE: Dict[Tuple[int, str, int], List[Dict[str, object]]] = {}
_SF_CHOICE_CACHE: Dict[
    Tuple[int, str, int, int],
    Tuple[Dict[int, int], Tuple[int, ...]],
] = {}
_MAX_SFS_PROFILE_CACHE: Dict[str, Any] = {}


def _load_max_sfs_cached(profile: str) -> Any:
    key = str(profile)
    cached = _MAX_SFS_PROFILE_CACHE.get(key)
    if cached is None:
        cached = load_max_sfs(key)
        _MAX_SFS_PROFILE_CACHE[key] = cached
    return cached


@dataclass(frozen=True)
class ActionCandidate:
    name: str
    action_vec: np.ndarray
    overrides: Dict[str, int]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionGridConfig:
    base_action_vec: Optional[np.ndarray]
    fixed_specs: Tuple[str, ...]
    range_specs: Tuple[str, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)


def coerce_spec_list(raw_value) -> Tuple[str, ...]:
    if raw_value in (None, ""):
        return ()
    if isinstance(raw_value, (list, tuple)):
        out: List[str] = []
        for item in raw_value:
            if item in (None, ""):
                continue
            if isinstance(item, str):
                out.extend(_split_spec_string(item))
            elif isinstance(item, Mapping):
                out.extend(_mapping_to_specs(item))
            else:
                out.append(str(item))
        return tuple(out)

    text = str(raw_value).strip()
    if not text:
        return ()
    if text.startswith("["):
        parsed = json.loads(text)
        return coerce_spec_list(parsed)
    if text.startswith("{"):
        parsed = json.loads(text)
        if not isinstance(parsed, Mapping):
            raise ValueError("action spec JSON object must be a mapping")
        return tuple(_mapping_to_specs(parsed))
    return tuple(_split_spec_string(text))


def load_action_grid_config(
        path_value: str,
        *,
        num_layers_hint: int = 0,
        profile: str = "default",
        gelu_degree: object = 4,
        attn_degree: object = 4,
        ) -> ActionGridConfig:
    """Load an action-config JSON file. Accepts four schemas:

    1. **Human-readable slots list / dict** (preferred — recorder writes this)::

        {"schema_version": "blb_v3_slots_human_v1",
         "num_layers": 12,
         "slots": [
           {"label": "L05.B3.K", "truncation_bits": 10},
           {"label": "L05.B5.W.wffn1", "scaling_factor": 14},
           ...
         ]}

       Or keyed by label::

        {"slots": {"L05.B3.K": {"truncation_bits": 10}, ...}}

    2. **base + overrides** (sparse — start from baseline, only list changes)::

        {"num_layers": 12,
         "base": "max",
         "overrides": [{"label": "L05.B3.K", "truncation_bits": 10}]}

    3. **Old action_vec list** (back-compat with existing presets)::

        {"action_vec": [3, 4, 5, ...], "num_layers": 12}

    4. **fixed / ranges only** (back-compat with cartesian-sweep presets)::

        {"fixed": {"layer2.block5.wffn1_sf": 18},
         "ranges": {"block3.truncation": [8, 9, 10, 11, 12, 13]}}

    All four shapes also support the optional top-level ``fixed`` / ``ranges``
    fields which apply *after* the base/slots/action_vec is decoded.
    """
    path = Path(str(path_value or "").strip())
    if not path.is_file():
        raise FileNotFoundError(f"final_eval action config does not exist: {path}")
    payload = read_json_file(path, encoding="utf-8-sig")
    if not isinstance(payload, Mapping):
        raise ValueError("--action-config JSON must be an object")

    num_layers = int(payload.get("num_layers", 0) or num_layers_hint or 0)

    # Detect shape: slot-form (1) / base+overrides (2) takes precedence over
    # the legacy action_vec form (3); falling through both leaves the legacy
    # path (just fixed/ranges, no base) (4).
    has_slots = (
        isinstance(payload.get("slots"), (list, Mapping))
        and bool(payload.get("slots"))
    )
    has_overrides = (
        isinstance(payload.get("overrides"), (list, Mapping))
        and bool(payload.get("overrides"))
    )
    has_action_vec = (
        payload.get("action_vec") is not None
        or payload.get("base_action_vec") is not None
        or payload.get("base_action") is not None
    )

    base_action_vec: Optional[np.ndarray | str] = None
    coercion_notes: List[Dict[str, object]] = []
    if (has_slots or has_overrides) and num_layers > 0:
        # New schema — convert via blb_stage2_rl.action_io.
        try:
            from blb_stage2_rl.action_io import slots_payload_to_action_vec
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                f"slot-form action-config requires blb_stage2_rl.action_io; import failed: {exc}"
            )
        cfg_profile = str(payload.get("profile") or profile or "default")
        cfg_gelu = payload.get("gelu_degree", gelu_degree)
        cfg_attn = payload.get("attn_degree", attn_degree)
        max_sfs = _load_max_sfs_cached(cfg_profile)
        slot_payload = dict(payload)
        if "base" not in slot_payload:
            for base_key in ("base_action_vec", "base_action"):
                base_value = payload.get(base_key)
                if isinstance(base_value, (list, tuple, str)):
                    slot_payload["base"] = base_value
                    break
        # ``slots_payload_to_action_vec`` keeps legacy ``action_vec`` support
        # for old configs, but this branch has already selected the slot-form
        # schema.  Remove flat-vector fallbacks so real slots cannot be shadowed
        # by stale map action indices.
        for stale_key in ("action_vec", "base_action_vec", "base_action"):
            slot_payload.pop(stale_key, None)
        vec, coercion_notes = slots_payload_to_action_vec(
            slot_payload,
            max_sfs=max_sfs,
            num_layers=int(num_layers),
            gelu_degree=cfg_gelu,
            attn_degree=cfg_attn,
        )
        base_action_vec = np.asarray(vec, dtype=int)
    elif has_action_vec:
        # Legacy shape — flat action_vec.
        base_raw = (
            payload.get("action_vec")
            or payload.get("base_action_vec")
            or payload.get("base_action")
        )
        base_action_vec = _parse_base_action_vec(base_raw, int(num_layers))
    # else: no base at all → caller decides (typically "max" inside build_action_candidates).

    if coercion_notes:
        # Surface coercions to the operator so silent snapping is auditable.
        try:
            from sys import stderr
            stderr.write(
                f"[action_grid] {len(coercion_notes)} slot value(s) snapped to nearest table level:\n"
            )
            for note in coercion_notes[:10]:
                stderr.write(f"  - {note}\n")
            if len(coercion_notes) > 10:
                stderr.write(f"  ... and {len(coercion_notes) - 10} more (see action-config payload)\n")
        except Exception:
            pass

    fixed_specs = tuple(_mapping_to_specs(payload.get("fixed", {}) or {}))
    range_specs = tuple(_mapping_to_specs(payload.get("ranges", {}) or payload.get("range", {}) or {}))
    metadata = _extract_action_config_metadata(payload, path=path, num_layers=num_layers)
    return ActionGridConfig(
        base_action_vec=base_action_vec,
        fixed_specs=fixed_specs,
        range_specs=range_specs,
        metadata=metadata,
    )


def _extract_action_config_metadata(
        payload: Mapping[str, object],
        *,
        path: Path,
        num_layers: int,
        ) -> Dict[str, Any]:
    """Keep non-action metadata alongside the selected candidate.

    Fusion-count fixed-action configs store the semantic option choices under
    ``group``.  The flat ``action_vec`` is still the executable input, but final
    eval reports need the original group metadata to compare declared map
    choices with the realized optimizer/replan result.
    """
    metadata: Dict[str, Any] = {
        "source_path": str(path),
        "num_layers": int(num_layers),
    }
    for key in (
        "schema_version",
        "profile",
        "group",
        "base",
        "action_vec",
        "legacy_action_vec",
        "rescale_optimizer_mode",
        "optimizer_mode",
    ):
        if key in payload:
            metadata[key] = payload[key]
    return metadata


def build_action_candidates(
    *,
    num_layers: int,
    profile: str,
    base_action_vec: Optional[Sequence[int]] = None,
    fixed_specs: Sequence[str] = (),
    range_specs: Sequence[str] = (),
    action_config_path: str = "",
) -> List[ActionCandidate]:
    num_layers = int(num_layers)
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")

    config = None
    if action_config_path:
        config = load_action_grid_config(
            action_config_path,
            num_layers_hint=num_layers,
            profile=profile,
        )

    cfg_fixed = config.fixed_specs if config is not None else ()
    cfg_ranges = config.range_specs if config is not None else ()
    fixed = tuple(cfg_fixed) + tuple(fixed_specs or ())
    ranges = tuple(cfg_ranges) + tuple(range_specs or ())

    if base_action_vec is None and config is not None:
        base_action_vec = config.base_action_vec
    base = _normalize_base_action(base_action_vec, num_layers)

    max_sfs = _load_max_sfs_cached(profile)
    for spec in fixed:
        selector, values = parse_action_spec(spec)
        if len(values) != 1:
            raise ValueError(f"fixed action spec must contain exactly one value: {spec!r}")
        _set_selector_value(base, num_layers, max_sfs, selector, int(values[0]))

    if not ranges:
        return [
            ActionCandidate(
                name="ActionSelected",
                action_vec=base.copy(),
                overrides={},
                metadata=(dict(config.metadata) if config is not None else {}),
            )
        ]

    parsed_ranges = []
    for spec in ranges:
        selector, values = parse_action_spec(spec)
        if not values:
            raise ValueError(f"range action spec has no values: {spec!r}")
        parsed_ranges.append((selector, [int(v) for v in values]))

    candidates: List[ActionCandidate] = []
    for idx, values in enumerate(itertools.product(*[v for _s, v in parsed_ranges]), start=1):
        vec = base.copy()
        overrides: Dict[str, int] = {}
        for (selector, _values), value in zip(parsed_ranges, values):
            _set_selector_value(vec, num_layers, max_sfs, selector, int(value))
            overrides[_canonical_selector_name(selector)] = int(value)
        label = "ActionGrid_" + "_".join(f"{k}{v}" for k, v in overrides.items())
        if len(label) > 96:
            label = f"ActionGrid_{idx:03d}"
        candidates.append(
            ActionCandidate(
                name=label,
                action_vec=vec,
                overrides=overrides,
                metadata=(dict(config.metadata) if config is not None else {}),
            )
        )
    return candidates


def build_random_action_candidates(
    *,
    num_layers: int,
    count: int,
    seed: int,
    base_action_vec: Optional[Sequence[int]] = None,
    fixed_specs: Sequence[str] = (),
    profile: str = "default",
) -> List[ActionCandidate]:
    base_candidates = build_action_candidates(
        num_layers=num_layers,
        profile=profile,
        base_action_vec=base_action_vec,
        fixed_specs=fixed_specs,
        range_specs=(),
    )
    selected = base_candidates[0]
    out = [selected]
    dims = np.asarray(action_dims_for_config(num_layers), dtype=int)
    rng = np.random.default_rng(int(seed))
    for idx in range(max(0, int(count))):
        vec = rng.integers(low=0, high=dims, size=dims.shape[0], dtype=np.int64)
        out.append(
            ActionCandidate(
                name=f"ActionRandom_{idx + 1:03d}",
                action_vec=vec,
                overrides={"random_index": idx + 1},
            )
        )
    return out


@dataclass(frozen=True)
class CostMatchedSamplingDiagnostics:
    """Counters describing one cost-matched-random sampling run.

    All counters are mutually exclusive sums; ``attempts == invalid + cost_mismatch + accepted``.
    """
    target_total_bits: int
    target_total_fusion: int
    target_sum_k: int
    accepted: int
    attempts: int
    invalid: int
    cost_mismatch: int
    avg_k_prefilter_skipped: int
    max_attempts: int
    requested_count: int


def build_cost_matched_random_action_candidates(
    *,
    num_layers: int,
    profile: str,
    selected_action_vec: Sequence[int],
    selected_total_bits: int,
    selected_total_fusion: int,
    selected_sum_k: int,
    bridge,
    max_sfs,
    gelu_degree,
    attn_degree,
    seed: int,
    count: int = 50,
    max_attempts: int = 5000,
    fixed_specs: Sequence[str] = (),
    log_fn=None,
) -> Tuple[List[ActionCandidate], CostMatchedSamplingDiagnostics]:
    """Reject-sample random action vectors whose Rescale_optimizer cost matches
    ``selected_*`` exactly on all three dimensions:
    ``(total_bits_sum, total_fusion_count, sum_truncation_k)``.

    Each attempt — even ones discarded by the cheap pre-filter — counts toward
    ``max_attempts`` per user spec. Invalid (modulus chain) draws also count.
    Accepted draws are returned in attempt order; the caller is responsible
    for prepending the selected action as the comparison anchor.
    """
    dims = np.asarray(action_dims_for_config(num_layers), dtype=int)
    rng = np.random.default_rng(int(seed))
    gelu_arr = np.asarray(gelu_degree, dtype=int)
    attn_arr = np.asarray(attn_degree, dtype=int)
    target_sum_k = int(selected_sum_k)
    target_total_bits = int(selected_total_bits)
    target_total_fusion = int(selected_total_fusion)
    accepted: List[ActionCandidate] = []
    invalid_n = 0
    mismatch_n = 0
    prefilter_n = 0
    attempts = 0

    parsed_fixed_specs: List[Tuple[str, int]] = []
    for spec in fixed_specs or ():
        selector, values = parse_action_spec(spec)
        if len(values) != 1:
            raise ValueError(f"fixed action spec must contain exactly one value: {spec!r}")
        parsed_fixed_specs.append((selector, int(values[0])))

    def _apply_fixed(vec: np.ndarray) -> None:
        if not parsed_fixed_specs:
            return
        for selector, value in parsed_fixed_specs:
            _set_selector_value(vec, num_layers, max_sfs, selector, int(value))

    while len(accepted) < int(count) and attempts < int(max_attempts):
        attempts += 1
        vec = rng.integers(low=0, high=dims, size=dims.shape[0], dtype=np.int64)
        if parsed_fixed_specs:
            _apply_fixed(vec)
        # Cheap pre-filter: sum_k can be computed directly from the action.
        sum_k = sum_truncation_k_in_action(vec, int(num_layers))
        if int(sum_k) != target_sum_k:
            prefilter_n += 1
            continue
        # Decode + optimizer call (this is the expensive bit).
        decoded = action_vector_to_cfgs(
            action_vec=vec,
            max_sfs=max_sfs,
            num_layers=int(num_layers),
            gelu_degree=gelu_arr,
            attn_degree=attn_arr,
        )
        try:
            requests = build_optimizer_requests(profile, decoded.cfgs_dict())
            outputs = bridge.evaluate_blocks(requests)
        except Exception as exc:
            invalid_n += 1
            if log_fn is not None:
                log_fn(f"  [cost-match][attempt {attempts}] optimizer error: {exc}")
            continue
        # Aggregate (import locally to avoid pulling rescale_optimizer_bridge
        # into action_grid's import chain at module load time).
        from rescale_optimizer_bridge import aggregate_optimizer_signals as _agg
        signals = _agg(outputs)
        if bool(signals.any_invalid):
            invalid_n += 1
            continue
        if int(signals.total_bits_sum) != target_total_bits:
            mismatch_n += 1
            continue
        if int(signals.total_fusion_count) != target_total_fusion:
            mismatch_n += 1
            continue
        accepted.append(ActionCandidate(
            name=f"ActionRandomSameCost_{len(accepted) + 1:03d}",
            action_vec=vec,
            overrides={"sampling": "cost_matched_random", "attempt": int(attempts)},
        ))

    diagnostics = CostMatchedSamplingDiagnostics(
        target_total_bits=target_total_bits,
        target_total_fusion=target_total_fusion,
        target_sum_k=target_sum_k,
        accepted=int(len(accepted)),
        attempts=int(attempts),
        invalid=int(invalid_n),
        cost_mismatch=int(mismatch_n),
        avg_k_prefilter_skipped=int(prefilter_n),
        max_attempts=int(max_attempts),
        requested_count=int(count),
    )
    if log_fn is not None:
        log_fn(
            "  [cost-match] sampling done: "
            f"accepted={diagnostics.accepted}/{count} "
            f"attempts={diagnostics.attempts}/{max_attempts} "
            f"invalid={diagnostics.invalid} "
            f"cost_mismatch={diagnostics.cost_mismatch} "
            f"avg_k_prefilter_skipped={diagnostics.avg_k_prefilter_skipped}"
        )
        if diagnostics.accepted < int(count):
            log_fn(
                "  [cost-match][warning] reached max_attempts before filling "
                f"count={count}. Consider raising --cost-match-max-attempts or "
                f"loosening cost match (current targets: total_bits={selected_total_bits}, "
                f"total_fusion={selected_total_fusion}, sum_k={selected_sum_k})."
            )
    return accepted, diagnostics


def parse_action_spec(spec: str) -> Tuple[str, Tuple[int, ...]]:
    text = str(spec or "").strip()
    if not text:
        raise ValueError("empty action spec")
    if "=" not in text:
        raise ValueError(f"action spec must use NAME=VALUE[,VALUE...]: {spec!r}")
    name, raw_values = text.split("=", 1)
    name = name.strip()
    raw_values = raw_values.strip()
    if not name or not raw_values:
        raise ValueError(f"invalid action spec: {spec!r}")
    if raw_values.startswith("["):
        parsed = json.loads(raw_values)
        values = tuple(int(v) for v in parsed)
    else:
        values = tuple(int(v.strip()) for v in raw_values.split(",") if v.strip())
    return name, values


def _normalize_base_action(base_action_vec: Optional[Sequence[int]], num_layers: int) -> np.ndarray:
    expected = len(action_dims_for_config(num_layers))
    if base_action_vec is None:
        return make_all_max_action_vector(num_layers).astype(int)
    if isinstance(base_action_vec, str):
        text = base_action_vec.strip().lower()
        if text in ("", "max", "all-max", "all_max", "blb-baseline", "blb_baseline", "rescale-baseline", "rescale_baseline"):
            return make_all_max_action_vector(num_layers).astype(int)
        if text in ("min", "all-min", "all_min"):
            return make_all_min_action_vector(num_layers).astype(int)
        if text.startswith("["):
            base_action_vec = json.loads(text)
        else:
            base_action_vec = [int(v.strip()) for v in text.split(",") if v.strip()]
    arr = np.asarray(list(base_action_vec), dtype=int).reshape(-1)
    if arr.size != expected:
        raise ValueError(
            f"action vector length {arr.size} != expected {expected} for {num_layers} layers"
        )
    dims = np.asarray(action_dims_for_config(num_layers), dtype=int)
    invalid = np.where((arr < 0) | (arr >= dims))[0]
    if invalid.size:
        first = int(invalid[0])
        raise ValueError(
            f"action vector index {first}={int(arr[first])} outside [0,{int(dims[first])})"
        )
    return arr.copy()


def _parse_base_action_vec(base_raw, num_layers_hint: int) -> Optional[np.ndarray | str]:
    if base_raw in (None, ""):
        return None
    if isinstance(base_raw, str):
        text = base_raw.strip()
        if text.lower() in (
            "max", "all-max", "all_max", "min", "all-min", "all_min",
            "blb-baseline", "blb_baseline", "rescale-baseline", "rescale_baseline",
        ):
            return text
        if text.startswith("["):
            return np.asarray(json.loads(text), dtype=int)
        return np.asarray([int(v.strip()) for v in text.split(",") if v.strip()], dtype=int)
    if isinstance(base_raw, Sequence):
        return np.asarray(list(base_raw), dtype=int)
    raise ValueError("action config base_action must be 'max', 'min', or an integer list")


def _set_selector_value(vec, num_layers, max_sfs, selector: str, value: int) -> None:
    slots = _selector_slots(num_layers, selector)
    if not slots:
        raise ValueError(f"unknown action selector: {selector!r}")
    for slot in slots:
        idx = _value_to_action_index(
            value=value,
            block_idx=slot["block_idx"],
            field_name=slot["field_name"],
            kind=slot["kind"],
            max_sfs=max_sfs,
        )
        vec[int(slot["offset"])] = int(idx)


def _selector_slots(num_layers: int, selector: str) -> List[Dict[str, object]]:
    cache_key = (int(num_layers), str(selector), id(per_layer_field_offsets))
    cached = _SELECTOR_SLOT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    parsed = _parse_selector(selector, num_layers)
    name = parsed["field_name"]
    exact_block = parsed["block_idx"]
    target_layers = parsed["layer_indices"]
    fields = per_layer_field_offsets()
    layer_dim = len(fields)
    slots: List[Dict[str, object]] = []

    if name in ("first_input", "firstinput"):
        raise ValueError(
            "first_input is deprecated and is not selectable; the first HE "
            "config is treated as lossless and no first_input noise is installed"
        )

    for layer_idx in range(int(num_layers)):
        if target_layers is not None and layer_idx not in target_layers:
            continue
        for field_offset, (block_idx, field_name, kind) in enumerate(fields):
            include = False
            if exact_block is not None:
                include = int(block_idx) == int(exact_block) and _selector_field_matches(
                    selector_field=name,
                    field_name=str(field_name),
                    kind=str(kind),
                )
            else:
                include = _selector_field_matches(
                    selector_field=name,
                    field_name=str(field_name),
                    kind=str(kind),
                )
            if include:
                slots.append({
                    "offset": layer_idx * layer_dim + field_offset,
                    "block_idx": int(block_idx),
                    "layer_idx": int(layer_idx),
                    "field_name": str(field_name),
                    "kind": str(kind),
                })
    _SELECTOR_SLOT_CACHE[cache_key] = slots
    return slots


def _sf_choice_lookup(kind: str, max_sf: int, levels: int) -> Tuple[Dict[int, int], Tuple[int, ...]]:
    cache_key = (id(sf_from), str(kind), int(max_sf), int(levels))
    cached = _SF_CHOICE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    choices = tuple(int(sf_from(idx, max_sf, levels)) for idx in range(levels))
    lookup = {choice: idx for idx, choice in enumerate(choices)}
    cached = (lookup, choices)
    _SF_CHOICE_CACHE[cache_key] = cached
    return cached


def _value_to_action_index(*, value: int, block_idx: int, field_name: str, kind: str, max_sfs) -> int:
    if field_name == "first_input":
        raise ValueError(
            "first_input is deprecated and is not selectable; the first HE "
            "config is treated as lossless and no first_input noise is installed"
        )

    if kind == "K":
        try:
            return _K_LEVEL_INDEX[int(value)]
        except KeyError as exc:
            raise ValueError(
                f"truncation={value} is not selectable; expected one of {_SORTED_K_LEVEL_CHOICES}"
            ) from exc

    levels = NUM_LEVELS_PER_DIM_BY_BLOCK_KIND[str(kind)]
    max_sf = max_sfs.get(int(block_idx), str(field_name))
    lookup, choices = _sf_choice_lookup(str(kind), int(max_sf), int(levels))
    idx = lookup.get(int(value))
    if idx is None:
        raise ValueError(
            f"{field_name}={value} is not selectable for block{block_idx}; "
            f"expected one of {choices}"
        )
    return int(idx)


def _canonical_selector_name(selector: str) -> str:
    text = str(selector or "").strip().lower().replace("-", "_")
    aliases = {
        "trunc": "truncation",
        "truncation_k": "truncation",
        "output_k": "truncation",
        "first_input_sf": "first_input",
        "input": "first_input",
    }
    return aliases.get(text, text)


def _parse_selector(selector: str, num_layers: int) -> Dict[str, object]:
    name = _canonical_selector_name(selector)
    parts = [part for part in name.split(".") if part]
    layer_indices = None
    block_idx = None
    field_parts: List[str] = []

    for part in parts:
        parsed_layers = _parse_layer_selector(part, num_layers)
        if parsed_layers is not None:
            if layer_indices is not None:
                raise ValueError(f"selector contains multiple layer filters: {selector!r}")
            layer_indices = parsed_layers
            continue
        parsed_block = _parse_block_selector(part)
        if parsed_block is not None:
            if block_idx is not None:
                raise ValueError(f"selector contains multiple block filters: {selector!r}")
            block_idx = parsed_block
            continue
        field_parts.append(part)

    field_name = ".".join(field_parts) if field_parts else name
    return {
        "field_name": _canonical_field_name(field_name),
        "block_idx": block_idx,
        "layer_indices": layer_indices,
    }


def _parse_layer_selector(part: str, num_layers: int):
    match = re.fullmatch(r"(?:layer|layers|l)(\d+)", str(part))
    if not match:
        return None
    idx = int(match.group(1))
    if idx < 0 or idx >= int(num_layers):
        raise ValueError(f"layer index {idx} outside [0,{int(num_layers)})")
    return (idx,)


def _parse_block_selector(part: str):
    match = re.fullmatch(r"block([1-5])", str(part))
    if not match:
        return None
    return int(match.group(1))


def _canonical_field_name(name: str) -> str:
    text = str(name or "").strip().lower().replace("-", "_")
    aliases = {
        "trunc": "output_truncation_k",
        "truncation": "output_truncation_k",
        "truncation_k": "output_truncation_k",
        "k": "output_truncation_k",
        "output_k": "output_truncation_k",
        "firstinput": "first_input",
        "first_input_sf": "first_input",
        "input": "first_input",
        "wffn1": "wffn1_sf",
        "wffn1_rescale": "wffn1_rescale_sf",
        "wffn2": "wffn2_sf",
        # ``wffn2_rescale`` alias removed 2026-05-14:
        # ``wffn2_rescale_sf`` RL slot was deleted because mrpc baseline skeleton
        # never places a rescale at ctct_ffn2_rescale. Cfg's ``wffn2_result_rescale``
        # is fixed to None. Selectors mentioning ``wffn2_rescale`` now resolve to
        # the literal text (no slot match) and the action-grid expansion errors
        # explicitly — which is the desired behaviour.
    }
    return aliases.get(text, text)


def _selector_field_matches(*, selector_field: str, field_name: str, kind: str) -> bool:
    if selector_field == "output_truncation_k":
        return str(kind) == "K"
    return str(field_name) == str(selector_field)


def _mapping_to_specs(mapping: Mapping) -> List[str]:
    out: List[str] = []
    for key, value in mapping.items():
        if isinstance(value, (list, tuple)):
            raw = ",".join(str(int(v)) for v in value)
        else:
            raw = str(value)
        out.append(f"{key}={raw}")
    return out


def _split_spec_string(text: str) -> List[str]:
    return [chunk.strip() for chunk in str(text).split(";") if chunk.strip()]
