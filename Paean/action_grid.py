from __future__ import annotations

import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from blb_stage2_rl.action_space import (
    K_LEVELS,
    NUM_LEVELS_PER_DIM_BY_BLOCK_KIND,
    action_dims_for_config,
    layer_dims,
    load_max_sfs,
    make_all_max_action_vector,
    make_all_min_action_vector,
    per_layer_field_offsets,
    sf_from,
)


@dataclass(frozen=True)
class ActionCandidate:
    name: str
    action_vec: np.ndarray
    overrides: Dict[str, int]


@dataclass(frozen=True)
class ActionGridConfig:
    base_action_vec: Optional[np.ndarray]
    fixed_specs: Tuple[str, ...]
    range_specs: Tuple[str, ...]


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


def load_action_grid_config(path_value: str) -> ActionGridConfig:
    path = Path(str(path_value or "").strip())
    if not path.is_file():
        raise FileNotFoundError(f"final_eval action config does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, Mapping):
        raise ValueError("--action-config JSON must be an object")

    base_raw = (
        payload.get("action_vec")
        or payload.get("base_action_vec")
        or payload.get("base_action")
    )
    base_action_vec = _parse_base_action_vec(base_raw, int(payload.get("num_layers", 0) or 0))
    fixed_specs = tuple(_mapping_to_specs(payload.get("fixed", {}) or {}))
    range_specs = tuple(_mapping_to_specs(payload.get("ranges", {}) or payload.get("range", {}) or {}))
    return ActionGridConfig(
        base_action_vec=base_action_vec,
        fixed_specs=fixed_specs,
        range_specs=range_specs,
    )


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
        config = load_action_grid_config(action_config_path)

    cfg_fixed = config.fixed_specs if config is not None else ()
    cfg_ranges = config.range_specs if config is not None else ()
    fixed = tuple(cfg_fixed) + tuple(fixed_specs or ())
    ranges = tuple(cfg_ranges) + tuple(range_specs or ())

    if base_action_vec is None and config is not None:
        base_action_vec = config.base_action_vec
    base = _normalize_base_action(base_action_vec, num_layers)

    max_sfs = load_max_sfs(profile)
    for spec in fixed:
        selector, values = parse_action_spec(spec)
        if len(values) != 1:
            raise ValueError(f"fixed action spec must contain exactly one value: {spec!r}")
        _set_selector_value(base, num_layers, max_sfs, selector, int(values[0]))

    if not ranges:
        return [ActionCandidate(name="ActionSelected", action_vec=base.copy(), overrides={})]

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
        candidates.append(ActionCandidate(name=label, action_vec=vec, overrides=overrides))
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
    parsed = _parse_selector(selector, num_layers)
    name = parsed["field_name"]
    exact_block = parsed["block_idx"]
    target_layers = parsed["layer_indices"]
    fields = per_layer_field_offsets()
    layer_dim = len(fields)
    slots: List[Dict[str, object]] = []

    if name in ("first_input", "firstinput"):
        return [{
            "offset": int(num_layers) * layer_dim,
            "block_idx": 0,
            "layer_idx": None,
            "field_name": "first_input",
            "kind": "F",
        }]

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
    return slots


def _value_to_action_index(*, value: int, block_idx: int, field_name: str, kind: str, max_sfs) -> int:
    if field_name == "first_input":
        levels = 5
        max_sf = 30
        for idx in range(levels):
            if int(sf_from(idx, max_sf, levels)) == int(value):
                return idx
        raise ValueError(f"first_input={value} is not selectable; expected one of 22,24,26,28,30")

    if kind == "K":
        if int(value) not in K_LEVELS:
            raise ValueError(f"truncation={value} is not selectable; expected one of {sorted(K_LEVELS)}")
        return list(K_LEVELS).index(int(value))

    levels = NUM_LEVELS_PER_DIM_BY_BLOCK_KIND[str(kind)]
    max_sf = max_sfs.get(int(block_idx), str(field_name))
    choices = [int(sf_from(idx, max_sf, levels)) for idx in range(levels)]
    if int(value) not in choices:
        raise ValueError(
            f"{field_name}={value} is not selectable for block{block_idx}; "
            f"expected one of {choices}"
        )
    return choices.index(int(value))


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
