#!/usr/bin/env python3
"""Generate a human-readable MRPC fusion-count map report and fixed actions.

The script intentionally avoids importing ``blb_stage2_rl`` because local
developer machines used for report generation may not have torch installed.
It reads the map JSON artifacts and parses the action slot table from
``action_space.py`` as data.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, OrderedDict
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cli_parse_utils import parse_json_int_list  # noqa: E402
from json_utils import read_json_file, write_json_file  # noqa: E402
from report_format_utils import html_table  # noqa: E402

ACTION_SPACE_PATH = REPO_ROOT / "blb_stage2_rl" / "action_space.py"
DEFAULT_MAP_DIR = REPO_ROOT / "blb_stage2_rl" / "fusion_maps" / "mrpc"

K_LEVELS = (8, 9, 11, 13, 10, 12)
BASELINE_K_INDEX = K_LEVELS.index(13)
# Keep in lockstep with action_space.LEVELS_* (this script stays torch-free so
# it mirrors the literal): 15-level uniform step-1 grid since 2026-06-11.
LEVELS_BY_KIND = {"F": 15, "W": 15, "M": 15, "S": 15, "R": 15, "K": len(K_LEVELS)}
FIRST_INPUT_LEVELS = 5

DEFAULT_GELU = [1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1]
DEFAULT_SOFTMAX = [6] * 12


def _parse_block_fields() -> Dict[int, List[Tuple[str, str, int]]]:
    tree = ast.parse(ACTION_SPACE_PATH.read_text(encoding="utf-8-sig"))
    out: Dict[int, List[Tuple[str, str, int]]] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if not (name.startswith("_BLOCK") and name.endswith("_FIELDS")):
            continue
        block_txt = name[len("_BLOCK") : -len("_FIELDS")]
        if not block_txt.isdigit():
            continue
        if not isinstance(node.value, ast.Call):
            continue
        fields_node = None
        for kw in node.value.keywords:
            if kw.arg == "fields":
                fields_node = kw.value
                break
        if fields_node is None:
            continue
        fields = ast.literal_eval(fields_node)
        out[int(block_txt)] = [
            (str(fname), str(kind), int(max_sf))
            for fname, kind, max_sf in fields
        ]
    missing = [b for b in (1, 2, 3, 4, 5) if b not in out]
    if missing:
        raise RuntimeError(f"failed to parse action_space block fields: missing {missing}")
    return out


def _load_maps(map_dir: Path) -> OrderedDict[str, dict]:
    graphs: OrderedDict[str, dict] = OrderedDict()
    for path in _iter_map_paths(map_dir):
        payload = read_json_file(path)
        if not isinstance(payload, Mapping) or "graph_key" not in payload or "options" not in payload:
            continue
        graphs[str(payload["graph_key"])] = dict(payload)
    if not graphs:
        raise RuntimeError(f"no fusion-count maps found under {map_dir}")
    return graphs


def _iter_map_paths(map_dir: Path) -> Iterable[Path]:
    try:
        with os.scandir(map_dir) as entries:
            names = sorted(
                entry.name
                for entry in entries
                if entry.is_file() and _looks_like_map_name(entry.name)
            )
    except OSError:
        names = []
    for name in names:
        yield map_dir / name


def _looks_like_map_file(path: Path) -> bool:
    return _looks_like_map_name(path.name)


def _looks_like_map_name(name: str) -> bool:
    if name.startswith("._") or name.startswith("_"):
        return False
    if not name.endswith(".json"):
        return False
    stem = name[:-5]
    return (
        stem == "block4"
        or stem.startswith("block1_")
        or stem.startswith("block2_")
        or stem.startswith("block3_exp_n")
        or stem.startswith("block5_n")
    )


def _block_offsets(fields_by_block: Mapping[int, Sequence[Tuple[str, str, int]]]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    cursor = 0
    for block_idx in (1, 2, 3, 4, 5):
        out[block_idx] = cursor
        cursor += len(fields_by_block[block_idx])
    return out


def _layer_width(fields_by_block: Mapping[int, Sequence[Tuple[str, str, int]]]) -> int:
    return sum(len(fields_by_block[b]) for b in (1, 2, 3, 4, 5))


def _make_all_max_action(fields_by_block: Mapping[int, Sequence[Tuple[str, str, int]]], num_layers: int) -> List[int]:
    dims: List[int] = []
    for _layer in range(int(num_layers)):
        for block_idx in (1, 2, 3, 4, 5):
            dims.extend(LEVELS_BY_KIND[kind] for _fname, kind, _max_sf in fields_by_block[block_idx])
    dims.append(FIRST_INPUT_LEVELS)
    action = [int(d) - 1 for d in dims]
    width = _layer_width(fields_by_block)
    offsets = _block_offsets(fields_by_block)
    for layer_idx in range(int(num_layers)):
        base = layer_idx * width
        for block_idx in (1, 2, 3, 4, 5):
            block_base = base + offsets[block_idx]
            for local_idx, (_fname, kind, _max_sf) in enumerate(fields_by_block[block_idx]):
                if kind == "K":
                    action[block_base + local_idx] = BASELINE_K_INDEX
    return action


def _graph_key(block_idx: int, profile: str, gelu_degree: int, softmax_degree: int) -> str:
    if int(block_idx) == 1:
        return f"block1_{profile}"
    if int(block_idx) == 2:
        return f"block2_{profile}"
    if int(block_idx) == 3:
        return f"block3_exp_n{int(softmax_degree)}"
    if int(block_idx) == 4:
        return "block4"
    if int(block_idx) == 5:
        return f"block5_n{int(gelu_degree)}"
    raise ValueError(f"unknown block_idx={block_idx}")


def _schedule(num_layers: int, profile: str, gelu: Sequence[int], softmax: Sequence[int]) -> List[dict]:
    out: List[dict] = []
    step_idx = 0
    for layer_idx in range(int(num_layers)):
        block_order = (2, 4, 5) if layer_idx == 0 else (1, 2, 4, 5)
        for block_idx in block_order:
            out.append({
                "step_idx": step_idx,
                "layer_idx": layer_idx,
                "block_idx": block_idx,
                "graph_key": _graph_key(block_idx, profile, gelu[layer_idx], softmax[layer_idx]),
            })
            step_idx += 1
    return out


def _choose_option(graph: Mapping[str, Any], target: str | int) -> Tuple[int, int, bool]:
    options = graph.get("options", [])
    if target == "max":
        count = None
        best_option_id = None
        best_total_bits = 0.0
        for option in options:
            option_count = int(option["fusion_count"])
            option_id = int(option.get("option_id", 0))
            total_bits = float(option.get("total_bits", 0.0))
            if (
                count is None
                or option_count > count
                or (
                    option_count == count
                    and (
                        best_option_id is None
                        or option_id < best_option_id
                        or (option_id == best_option_id and total_bits < best_total_bits)
                    )
                )
            ):
                count = option_count
                best_option_id = option_id
                best_total_bits = total_bits
        if count is None or best_option_id is None:
            raise ValueError(f"graph {graph.get('graph_key')} has no options")
        return int(best_option_id), int(count), False

    requested_count = int(target)
    min_count = None
    lower_count = None
    best_option_id = None
    best_total_bits = 0.0
    for option in options:
        option_count = int(option["fusion_count"])
        if min_count is None or option_count < min_count:
            min_count = option_count
        if option_count <= requested_count and (lower_count is None or option_count > lower_count):
            lower_count = option_count
        if option_count != requested_count:
            continue
        option_id = int(option.get("option_id", 0))
        total_bits = float(option.get("total_bits", 0.0))
        if (
            best_option_id is None
            or option_id < best_option_id
            or (option_id == best_option_id and total_bits < best_total_bits)
        ):
            best_option_id = option_id
            best_total_bits = total_bits
    if min_count is None:
        raise ValueError(f"graph {graph.get('graph_key')} has no options")
    if best_option_id is not None:
        return int(best_option_id), requested_count, False

    count = lower_count if lower_count is not None else min_count
    best_total_bits = 0.0
    for option in options:
        if int(option["fusion_count"]) != count:
            continue
        option_id = int(option.get("option_id", 0))
        total_bits = float(option.get("total_bits", 0.0))
        if (
            best_option_id is None
            or option_id < best_option_id
            or (option_id == best_option_id and total_bits < best_total_bits)
        ):
            best_option_id = option_id
            best_total_bits = total_bits
    if best_option_id is None:
        raise ValueError(f"graph {graph.get('graph_key')} has no options")
    clamped = requested_count != count
    return int(best_option_id), int(count), bool(clamped)


def _option_by_id(graph: Mapping[str, Any], option_id: int) -> Mapping[str, Any]:
    for option in graph.get("options", []):
        if int(option.get("option_id")) == int(option_id):
            return option
    raise KeyError(f"graph {graph.get('graph_key')} has no option {option_id}")


def _option_index_by_graph(
    graphs: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Dict[int, Mapping[str, Any]]]:
    out: Dict[str, Dict[int, Mapping[str, Any]]] = {}
    for graph_key, graph in graphs.items():
        index: Dict[int, Mapping[str, Any]] = {}
        for option in graph.get("options", []):
            index.setdefault(int(option.get("option_id")), option)
        out[str(graph_key)] = index
    return out


def _option_by_index(
    graph: Mapping[str, Any],
    option_id: int,
    option_index_by_graph: Mapping[str, Mapping[int, Mapping[str, Any]]],
) -> Mapping[str, Any]:
    graph_key = str(graph.get("graph_key"))
    try:
        return option_index_by_graph[graph_key][int(option_id)]
    except KeyError as exc:
        raise KeyError(f"graph {graph.get('graph_key')} has no option {option_id}") from exc


def _field_kinds_by_block(
    fields_by_block: Mapping[int, Sequence[Tuple[str, str, int]]]
) -> Dict[int, Dict[str, str]]:
    return {
        int(block_idx): {
            str(field): str(kind)
            for field, kind, _max_sf in fields
        }
        for block_idx, fields in fields_by_block.items()
    }


def _adjusted_block_action(
    graph: Mapping[str, Any],
    option: Mapping[str, Any],
    fields_by_block: Mapping[int, Sequence[Tuple[str, str, int]]],
) -> Tuple[int, ...]:
    graph_key = str(graph.get("graph_key"))
    block_idx = int(graph["block_idx"])
    block_action = [int(v) for v in option.get("action_indices", [])]
    k_slot_index = int(graph["k_slot_index"])
    if 0 <= k_slot_index < len(block_action):
        block_action[k_slot_index] = BASELINE_K_INDEX
    expected = len(fields_by_block[block_idx])
    if len(block_action) != expected:
        raise RuntimeError(
            f"{graph_key}: action_indices len {len(block_action)} != block{block_idx} field count {expected}"
        )
    return tuple(block_action)


def _block_actions_by_option(
    graphs: Mapping[str, Mapping[str, Any]],
    fields_by_block: Mapping[int, Sequence[Tuple[str, str, int]]],
) -> Dict[str, Dict[int, Tuple[int, ...]]]:
    out: Dict[str, Dict[int, Tuple[int, ...]]] = {}
    for graph_key, graph in graphs.items():
        out[str(graph_key)] = {
            int(option.get("option_id")): _adjusted_block_action(graph, option, fields_by_block)
            for option in graph.get("options", [])
        }
    return out


def _option_id_for_step(step: Mapping[str, Any], option_by_graph: Mapping[str, int], option_by_step: Mapping[str, int] | None = None) -> int:
    step_key = str(step["step_idx"])
    if option_by_step and step_key in option_by_step:
        return int(option_by_step[step_key])
    return int(option_by_graph[str(step["graph_key"])])


def _splice_group_action(
    *,
    fields_by_block: Mapping[int, Sequence[Tuple[str, str, int]]],
    graphs: Mapping[str, Mapping[str, Any]],
    num_layers: int,
    schedule: Sequence[Mapping[str, Any]],
    option_by_graph: Mapping[str, int],
    option_by_step: Mapping[str, int] | None = None,
    base_action: Sequence[int] | None = None,
    layer_width: int | None = None,
    block_offsets: Mapping[int, int] | None = None,
    option_index_by_graph: Mapping[str, Mapping[int, Mapping[str, Any]]] | None = None,
    block_actions_by_option: Mapping[str, Mapping[int, Sequence[int]]] | None = None,
) -> List[int]:
    action = list(base_action) if base_action is not None else _make_all_max_action(fields_by_block, num_layers)
    width = int(layer_width) if layer_width is not None else _layer_width(fields_by_block)
    offsets = block_offsets if block_offsets is not None else _block_offsets(fields_by_block)
    block_action_cache: Dict[Tuple[str, int], Sequence[int]] = {}
    for step in schedule:
        graph_key = str(step["graph_key"])
        graph = graphs[graph_key]
        block_idx = int(step["block_idx"])
        layer_idx = int(step["layer_idx"])
        option_id = _option_id_for_step(step, option_by_graph, option_by_step)
        if block_actions_by_option is not None:
            block_action = block_actions_by_option[graph_key][option_id]
        else:
            cache_key = (graph_key, option_id)
            block_action = block_action_cache.get(cache_key)
            if block_action is None:
                option = (
                    _option_by_index(graph, option_id, option_index_by_graph)
                    if option_index_by_graph is not None
                    else _option_by_id(graph, option_id)
                )
                block_action = _adjusted_block_action(graph, option, fields_by_block)
                block_action_cache[cache_key] = block_action
        start = layer_idx * width + offsets[block_idx]
        action[start : start + len(block_action)] = block_action
    return action


def _slot_label(block_idx: int, kind: str, field: str) -> str:
    return f"B{int(block_idx)}.{kind}.{field}"


def _short_field_label(field_name: str, kind: str) -> str:
    if str(kind) == "K":
        return ""
    field = str(field_name)
    if field.startswith("square_rescale_sf_"):
        return "sq" + field.rsplit("_", 1)[-1]
    if field.startswith("gelu_power_rescale_sf_"):
        return "gp" + field.rsplit("_", 1)[-1]
    if field.startswith("gelu_coeff_mul_rescale_sf_"):
        return "gc" + field.rsplit("_", 1)[-1]
    if field.endswith("_rescale_sf"):
        return field[: -len("_rescale_sf")] + "_r"
    if field.endswith("_sf"):
        return field[: -len("_sf")]
    return field


def _canonical_slot_label(layer_idx: int, block_idx: int, kind: str, field: str) -> str:
    base = f"L{int(layer_idx)}.B{int(block_idx)}.{str(kind)}"
    short = _short_field_label(field, kind)
    return base if not short else f"{base}.{short}"


def _bound_slot_values(block_idx: int, slots: Mapping[str, Any]) -> Dict[str, Any]:
    """Expand action-space bindings so slot-form configs replay map options.

    Fusion maps store the real optimizer-facing slot values sparsely.  The
    executable action config starts from the legacy per-block action_indices and
    then overlays these values; bound compat slots must be overlaid too, or the
    decoded cfg can drift from the map option that was actually audited.
    """
    out = {str(k): v for k, v in slots.items()} if slots else {}
    if int(block_idx) == 2:
        if "inv_std_fresh_sf" in out:
            out.setdefault("x_centered_fresh_sf", out["inv_std_fresh_sf"])
        if "wk_sf" in out:
            out.setdefault("wq_sf", out["wk_sf"])
        if "kt_mask1_sf" in out:
            out.setdefault("q_mask1_sf", out["kt_mask1_sf"])
        if "kt_mask2_sf" in out:
            out.setdefault("q_mask2_sf", out["kt_mask2_sf"])
    elif int(block_idx) == 4:
        if "softmax_out_mask_sf" in out:
            out.setdefault("v_mask_sf", out["softmax_out_mask_sf"])
    elif int(block_idx) == 5:
        if "x_centered_fresh_sf" in out:
            out.setdefault("inv_std_fresh_sf", out["x_centered_fresh_sf"])
    return out


def _slot_entries_for_option(
    block_idx: int,
    option: Mapping[str, Any],
    field_kinds: Mapping[str, str],
) -> Tuple[Tuple[str, str, Any], ...]:
    slot_values = _bound_slot_values(block_idx, option.get("slots", {}))
    if not slot_values:
        return ()
    entries: List[Tuple[str, str, Any]] = []
    for field_name, value in sorted(slot_values.items()):
        kind = field_kinds.get(str(field_name))
        if kind is None:
            continue
        entries.append((str(field_name), kind, value))
    return tuple(entries)


def _slot_entries_by_option(
    graphs: Mapping[str, Mapping[str, Any]],
    field_kinds_by_block: Mapping[int, Mapping[str, str]],
) -> Dict[str, Dict[int, Tuple[Tuple[str, str, Any], ...]]]:
    out: Dict[str, Dict[int, Tuple[Tuple[str, str, Any], ...]]] = {}
    for graph_key, graph in graphs.items():
        block_idx = int(graph["block_idx"])
        field_kinds = field_kinds_by_block.get(block_idx, {})
        out[str(graph_key)] = {
            int(option.get("option_id")): _slot_entries_for_option(block_idx, option, field_kinds)
            for option in graph.get("options", [])
        }
    return out


def _splice_group_slots(
    *,
    fields_by_block: Mapping[int, Sequence[Tuple[str, str, int]]],
    graphs: Mapping[str, Mapping[str, Any]],
    schedule: Sequence[Mapping[str, Any]],
    option_by_graph: Mapping[str, int],
    option_by_step: Mapping[str, int] | None = None,
    option_index_by_graph: Mapping[str, Mapping[int, Mapping[str, Any]]] | None = None,
    field_kinds_by_block: Mapping[int, Mapping[str, str]] | None = None,
    slot_entries_by_option: Mapping[str, Mapping[int, Sequence[Tuple[str, str, Any]]]] | None = None,
) -> List[dict]:
    entries: List[dict] = []
    field_kind_cache: Dict[int, Mapping[str, str]] = dict(field_kinds_by_block or {})
    slot_entry_cache: Dict[Tuple[str, int], Sequence[Tuple[str, str, Any]]] = {}
    for step in schedule:
        graph_key = str(step["graph_key"])
        graph = graphs[graph_key]
        block_idx = int(step["block_idx"])
        layer_idx = int(step["layer_idx"])
        option_id = _option_id_for_step(step, option_by_graph, option_by_step)
        if slot_entries_by_option is not None:
            slot_entries = slot_entries_by_option[graph_key][option_id]
        else:
            cache_key = (graph_key, option_id)
            slot_entries = slot_entry_cache.get(cache_key)
            if slot_entries is None:
                option = (
                    _option_by_index(graph, option_id, option_index_by_graph)
                    if option_index_by_graph is not None
                    else _option_by_id(graph, option_id)
                )
                field_kinds = field_kind_cache.get(block_idx)
                if field_kinds is None:
                    field_kinds = {
                        str(field): str(kind)
                        for field, kind, _max_sf in fields_by_block[block_idx]
                    }
                    field_kind_cache[block_idx] = field_kinds
                slot_entries = _slot_entries_for_option(block_idx, option, field_kinds)
                slot_entry_cache[cache_key] = slot_entries
        if not slot_entries:
            continue
        for field_name, kind, value in slot_entries:
            if kind == "K":
                entries.append({
                    "label": _canonical_slot_label(layer_idx, block_idx, kind, field_name),
                    "truncation_bits": int(value),
                })
            else:
                entries.append({
                    "label": _canonical_slot_label(layer_idx, block_idx, kind, field_name),
                    "scaling_factor": value,
                })
    return entries


def _group_specs(graphs: Mapping[str, Mapping[str, Any]], schedule: Sequence[Mapping[str, Any]]) -> List[dict]:
    graph_order = list(graphs.keys())
    occurrence_counts = Counter(str(s["graph_key"]) for s in schedule)
    choice_cache: Dict[Tuple[str, str | int], Tuple[int, int, bool]] = {}

    def choose(graph_key: str, target: str | int) -> Tuple[int, int, bool]:
        key = (str(graph_key), target)
        if key not in choice_cache:
            choice_cache[key] = _choose_option(graphs[graph_key], target)
        return choice_cache[key]

    specs: List[dict] = []
    for name, target in (
        ("all_fusion0", 0),
        ("all_fusion1_available", 1),
        ("all_fusionmax", "max"),
    ):
        option_by_graph = {}
        count_by_graph = {}
        clamped_graphs = []
        for graph_key in graph_order:
            opt, count, clamped = choose(graph_key, target)
            option_by_graph[graph_key] = opt
            count_by_graph[graph_key] = count
            if clamped:
                clamped_graphs.append(graph_key)
        specs.append({
            "name": name,
            "family": "global",
            "target": target,
            "option_by_graph": option_by_graph,
            "fusion_count_by_graph": count_by_graph,
            "clamped_graphs": clamped_graphs,
            "occurrence_counts": dict(occurrence_counts),
        })

    for graph_key in graph_order:
        option_by_graph = {}
        count_by_graph = {}
        for candidate in graph_order:
            target = "max" if candidate == graph_key else 0
            opt, count, _clamped = choose(candidate, target)
            option_by_graph[candidate] = opt
            count_by_graph[candidate] = count
        specs.append({
            "name": f"one_hot_{graph_key}",
            "family": "one_hot",
            "target_graph": graph_key,
            "no_op": occurrence_counts.get(graph_key, 0) == 0 or count_by_graph.get(graph_key, 0) == 0,
            "option_by_graph": option_by_graph,
            "fusion_count_by_graph": count_by_graph,
            "occurrence_counts": dict(occurrence_counts),
        })

    option_by_graph = {}
    count_by_graph = {}
    for graph_key in graph_order:
        target = "max" if graph_key == "block2_mrpc" or graph_key.startswith("block5_") else 0
        opt, count, _clamped = choose(graph_key, target)
        option_by_graph[graph_key] = opt
        count_by_graph[graph_key] = count
    specs.append({
        "name": "block2_block5_all_layers_fusionmax",
        "family": "combined",
        "target_graphs": [g for g in graph_order if g == "block2_mrpc" or g.startswith("block5_")],
        "option_by_graph": option_by_graph,
        "fusion_count_by_graph": count_by_graph,
        "occurrence_counts": dict(occurrence_counts),
    })

    block4_graph = graphs.get("block4")
    if block4_graph is not None:
        b4_max_opt, b4_max_count, _clamped = choose("block4", "max")
        for name, layers in (
            ("block4_fusionmax_1_layer", [0]),
            ("block4_fusionmax_2_layers", [0, 6]),
            ("block4_fusionmax_4_layers", [0, 3, 6, 9]),
        ):
            base_options = {}
            base_counts = {}
            for graph_key in graph_order:
                opt, count, _ = choose(graph_key, 0)
                base_options[graph_key] = opt
                base_counts[graph_key] = count
            selected_layers = set(layers)
            option_by_step = {
                str(step["step_idx"]): int(b4_max_opt)
                for step in schedule
                if str(step["graph_key"]) == "block4" and int(step["layer_idx"]) in selected_layers
            }
            specs.append({
                "name": name,
                "family": "partial_block4",
                "target_graph": "block4",
                "selected_layers": [int(v) for v in layers],
                "option_by_graph": base_options,
                "option_by_step": option_by_step,
                "fusion_count_by_graph": {
                    **base_counts,
                    "block4_selected_layers": int(b4_max_count),
                },
                "occurrence_counts": dict(occurrence_counts),
            })
    return specs


def _write_action_configs(
    *,
    output_dir: Path,
    fields_by_block: Mapping[int, Sequence[Tuple[str, str, int]]],
    graphs: Mapping[str, Mapping[str, Any]],
    num_layers: int,
    schedule: Sequence[Mapping[str, Any]],
    group_specs: Sequence[Mapping[str, Any]],
    profile: str,
    gelu: Sequence[int],
    softmax: Sequence[int],
) -> Dict[str, str]:
    action_dir = output_dir / "action_configs"
    action_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    base_action = _make_all_max_action(fields_by_block, num_layers)
    width = _layer_width(fields_by_block)
    offsets = _block_offsets(fields_by_block)
    option_index_by_graph = _option_index_by_graph(graphs)
    field_kinds_by_block = _field_kinds_by_block(fields_by_block)
    block_actions_by_option = _block_actions_by_option(graphs, fields_by_block)
    slot_entries_by_option = _slot_entries_by_option(graphs, field_kinds_by_block)
    for spec in group_specs:
        name = str(spec["name"])
        action = _splice_group_action(
            fields_by_block=fields_by_block,
            graphs=graphs,
            num_layers=num_layers,
            schedule=schedule,
            option_by_graph=spec["option_by_graph"],
            option_by_step=spec.get("option_by_step"),
            base_action=base_action,
            layer_width=width,
            block_offsets=offsets,
            option_index_by_graph=option_index_by_graph,
            block_actions_by_option=block_actions_by_option,
        )
        slots = _splice_group_slots(
            fields_by_block=fields_by_block,
            graphs=graphs,
            schedule=schedule,
            option_by_graph=spec["option_by_graph"],
            option_by_step=spec.get("option_by_step"),
            option_index_by_graph=option_index_by_graph,
            field_kinds_by_block=field_kinds_by_block,
            slot_entries_by_option=slot_entries_by_option,
        )
        payload = {
            "schema_version": "fusion_count_fixed_action_v1",
            "num_layers": int(num_layers),
            "profile": str(profile),
            "gelu_degree": [int(v) for v in gelu],
            "attn_degree": [int(v) for v in softmax],
            "base": action,
            "base_action": "legacy_fusion_count_map_action_indices_with_baseline_k",
            "k_levels": list(K_LEVELS),
            "baseline_k_index": int(BASELINE_K_INDEX),
            "group": dict(spec),
            "slots": slots,
            "legacy_action_vec": action,
            "execution_note": (
                "Use slots + base to execute this config. legacy_action_vec is "
                "kept for audit only because map action_indices can drift from "
                "the current action-space SF tables."
            ),
        }
        path = action_dir / f"{name}.json"
        write_json_file(path, payload)
        paths[name] = str(path)
    return paths


def _graph_occurrences(schedule: Sequence[Mapping[str, Any]]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for step in schedule:
        out.setdefault(str(step["graph_key"]), []).append(int(step["layer_idx"]))
    return {k: sorted(set(v)) for k, v in sorted(out.items())}


def _int_slot_mapping(slots: Any) -> Dict[str, int]:
    if not slots:
        return {}
    items = slots.items() if hasattr(slots, "items") else dict(slots).items()
    return {str(k): int(v) for k, v in items}


def _option_slot_summary(
    graph: Mapping[str, Any],
    fields: Sequence[Tuple[str, str, int]],
    option: Mapping[str, Any],
    base_option: Mapping[str, Any],
    *,
    base_action: Sequence[int] | None = None,
    base_slots: Mapping[str, int] | None = None,
) -> dict:
    action = [int(v) for v in option.get("action_indices", [])]
    if base_action is None:
        base_action = [int(v) for v in base_option.get("action_indices", [])]
    slots = _int_slot_mapping(option.get("slots", {}))
    if base_slots is None:
        base_slots = _int_slot_mapping(base_option.get("slots", {}))
    rows = []
    changed_raw = []
    changed_real = []
    added_real = []
    removed_real = []
    for idx, (field, kind, _max_sf) in enumerate(fields):
        raw = int(action[idx])
        base_raw = int(base_action[idx])
        value = None
        if kind == "K":
            value = K_LEVELS[raw] if 0 <= raw < len(K_LEVELS) else None
            real_status = "truncation_k"
        elif field in slots:
            value = int(slots[field])
            real_status = "real_replan_slot"
        else:
            real_status = "not_in_replan_slots"
        if raw != base_raw:
            changed_raw.append({
                "slot_index": idx,
                "field": field,
                "kind": kind,
                "base_action_index": base_raw,
                "action_index": raw,
            })
        if field in slots and base_slots.get(field) != slots[field]:
            changed_real.append({
                "slot_index": idx,
                "field": field,
                "kind": kind,
                "base_value": base_slots.get(field),
                "value": slots[field],
            })
        if field in slots and field not in base_slots:
            added_real.append(field)
        if field not in slots and field in base_slots:
            removed_real.append(field)
        rows.append({
            "slot_index": idx,
            "label": _slot_label(int(graph["block_idx"]), kind, field),
            "field": field,
            "kind": kind,
            "action_index": raw,
            "base_action_index": base_raw,
            "decoded_value": value,
            "real_status": real_status,
            "changed_raw_vs_fusion0": raw != base_raw,
            "changed_real_vs_fusion0": field in slots and base_slots.get(field) != slots[field],
        })
    return {
        "option_id": int(option["option_id"]),
        "fusion_count": int(option["fusion_count"]),
        "total_bits": float(option.get("total_bits", 0.0)),
        "total_variance": float(option.get("total_variance", 0.0)),
        "changed_raw_slots_vs_fusion0": changed_raw,
        "changed_real_slots_vs_fusion0": changed_real,
        "added_real_slots_vs_fusion0": added_real,
        "removed_real_slots_vs_fusion0": removed_real,
        "slot_rows": rows,
        "real_slots": slots,
    }


def _options_in_id_order(options: Iterable[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    out = options if isinstance(options, list) else list(options)
    previous_id = None
    for option in out:
        option_id = int(option.get("option_id", 0))
        if previous_id is not None and option_id < previous_id:
            return sorted(out, key=lambda o: int(o.get("option_id", 0)))
        previous_id = option_id
    return out


def _base_option_from_ordered_options(
    graph: Mapping[str, Any],
    options: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if options and int(options[0].get("option_id", -1)) == 0:
        return options[0]
    for option in options:
        if int(option.get("option_id", -1)) == 0:
            return option
    raise KeyError(f"graph {graph.get('graph_key')} has no option 0")


def _build_report_payload(
    *,
    graphs: Mapping[str, Mapping[str, Any]],
    fields_by_block: Mapping[int, Sequence[Tuple[str, str, int]]],
    schedule: Sequence[Mapping[str, Any]],
    group_specs: Sequence[Mapping[str, Any]],
    action_config_paths: Mapping[str, str],
    profile: str,
    gelu: Sequence[int],
    softmax: Sequence[int],
) -> dict:
    occurrences = _graph_occurrences(schedule)
    graph_payload = []
    for graph_key, graph in graphs.items():
        block_idx = int(graph["block_idx"])
        fields = fields_by_block[block_idx]
        options = _options_in_id_order(graph.get("options", []))
        base = _base_option_from_ordered_options(graph, options)
        base_action = [int(v) for v in base.get("action_indices", [])]
        base_slots = _int_slot_mapping(base.get("slots", {}))
        available_fusion_counts = set()
        option_summaries = []
        for option in options:
            available_fusion_counts.add(int(option["fusion_count"]))
            option_summaries.append(
                _option_slot_summary(
                    graph,
                    fields,
                    option,
                    base,
                    base_action=base_action,
                    base_slots=base_slots,
                )
            )
        graph_payload.append({
            "graph_key": graph_key,
            "block_idx": block_idx,
            "gelu_degree": graph.get("gelu_degree"),
            "attn_degree": graph.get("attn_degree"),
            "k_slot_index": int(graph["k_slot_index"]),
            "block_num_slots": int(graph["block_num_slots"]),
            "available_fusion_counts": sorted(available_fusion_counts),
            "current_schedule_layers": occurrences.get(graph_key, []),
            "current_schedule_occurrences": int(len(occurrences.get(graph_key, []))),
            "options": option_summaries,
        })
    return {
        "schema_version": "fusion_count_map_report_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "profile": profile,
        "num_layers": len(gelu),
        "stage1_gelu": [int(v) for v in gelu],
        "stage1_softmax": [int(v) for v in softmax],
        "k_levels": list(K_LEVELS),
        "baseline_k_index": BASELINE_K_INDEX,
        "baseline_k_value": K_LEVELS[BASELINE_K_INDEX],
        "schedule_occurrences": occurrences,
        "graphs": graph_payload,
        "eval_group_specs": [dict(g) for g in group_specs],
        "action_config_paths": dict(action_config_paths),
        "interpretation_note": (
            "fusion_count is the Rescale_optimizer/replan measured fusion count for an option. "
            "The map does not store a separate 'slot was fused away' flag; this report therefore "
            "shows the full raw action slots, the real replan slots, and the real slot changes "
            "relative to fusion_count=0."
        ),
    }


def _fmt_layers(layers: Sequence[int]) -> str:
    if not layers:
        return "unused"
    return ", ".join(f"L{int(v)}" for v in layers)


def _render_html(payload: Mapping[str, Any]) -> str:
    graph_rows = []
    for graph in payload["graphs"]:
        graph_rows.append([
            graph["graph_key"],
            f"B{graph['block_idx']}",
            ",".join(str(v) for v in graph["available_fusion_counts"]),
            graph["block_num_slots"],
            graph["k_slot_index"],
            graph["current_schedule_occurrences"],
            _fmt_layers(graph["current_schedule_layers"]),
        ])

    group_rows = []
    for group in payload["eval_group_specs"]:
        name = str(group["name"])
        no_op = "yes" if group.get("no_op") else ""
        counts = ", ".join(f"{k}:{v}" for k, v in group["fusion_count_by_graph"].items())
        opts = ", ".join(f"{k}:opt{v}" for k, v in group["option_by_graph"].items())
        details = ""
        if group.get("selected_layers"):
            details = "layers=" + ",".join(f"L{int(v)}" for v in group["selected_layers"])
        if group.get("target_graphs"):
            details = "graphs=" + ",".join(str(v) for v in group["target_graphs"])
        if group.get("option_by_step"):
            step_txt = ",".join(f"s{k}:opt{v}" for k, v in sorted(group["option_by_step"].items(), key=lambda kv: int(kv[0])))
            details = (details + "; " if details else "") + step_txt
        group_rows.append([name, group.get("family", ""), no_op, details, counts, opts])

    parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>MRPC Fusion Count Map Report</title>",
        "<style>",
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:32px;color:#1f2933;background:#fbfcfd}",
        "h1,h2,h3{color:#111827} .meta{color:#52606d} table{border-collapse:collapse;width:100%;margin:14px 0;background:white}",
        "th,td{border:1px solid #d9e2ec;padding:7px 9px;text-align:left;vertical-align:top;font-size:13px}",
        "th{background:#eef2f7}.pill{display:inline-block;padding:2px 7px;border-radius:6px;background:#e0f2fe;color:#075985;font-size:12px}",
        ".warn{background:#fff7ed;border-left:4px solid #fb923c;padding:10px 12px;margin:12px 0}.changed{color:#9a3412;font-weight:600}",
        "code{background:#eef2f7;padding:1px 4px;border-radius:4px}.section{margin-top:28px}",
        "</style></head><body>",
        "<h1>MRPC Fusion Count Map Report</h1>",
        f"<p class='meta'>Generated: {html.escape(str(payload['generated_at_utc']))}</p>",
        "<div class='warn'>"
        "说明：<code>fusion_count</code> 是 replan/Rescale_optimizer 对该 option 的实测融合次数。"
        "当前 map 不保存“某个槽位被融合删除”的单独标志；因此本报告同时列出 raw action 槽位、"
        "真正进入 replan 的 real slots，以及相对 fusion_count=0 的 real slot 变化。"
        "</div>",
        "<h2>Stage-1 / Schedule Context</h2>",
        html_table(
            ["profile", "GELU", "Softmax", "K levels", "baseline K"],
            [[
                payload["profile"],
                json.dumps(payload["stage1_gelu"]),
                json.dumps(payload["stage1_softmax"]),
                json.dumps(payload["k_levels"]),
                payload["baseline_k_value"],
            ]],
            allow_html_cells=True,
        ),
        "<h2>Block Fusion Count Summary</h2>",
        html_table(
            ["graph/block", "block", "fusion counts", "slot count", "K slot", "occurrences", "layers"],
            graph_rows,
            allow_html_cells=True,
        ),
        "<h2>Server Evaluation Groups Prepared</h2>",
        html_table(
            ["group", "family", "no-op", "details", "fusion count by graph", "option by graph"],
            group_rows,
            allow_html_cells=True,
        ),
    ]

    for graph in payload["graphs"]:
        parts.append(f"<div class='section'><h2>{html.escape(graph['graph_key'])}</h2>")
        parts.append(
            f"<p class='meta'>Block B{graph['block_idx']} | available fusion_count="
            f"{graph['available_fusion_counts']} | current layers={html.escape(_fmt_layers(graph['current_schedule_layers']))}</p>"
        )
        option_rows = []
        for option in graph["options"]:
            changed_real = option["changed_real_slots_vs_fusion0"]
            changed_raw = option["changed_raw_slots_vs_fusion0"]
            real_txt = "<br>".join(
                html.escape(f"{r['field']}: {r['base_value']} -> {r['value']}")
                for r in changed_real
            ) or "none"
            raw_txt = "<br>".join(
                html.escape(f"{r['slot_index']} {r['field']}: idx {r['base_action_index']} -> {r['action_index']}")
                for r in changed_raw
            ) or "none"
            option_rows.append([
                option["option_id"],
                option["fusion_count"],
                f"{option['total_bits']:.0f}",
                f"{option['total_variance']:.6g}",
                f"<span class='changed'>{real_txt}</span>" if changed_real else real_txt,
                f"<span class='changed'>{raw_txt}</span>" if changed_raw else raw_txt,
            ])
        parts.append(html_table(
            ["option", "fusion_count", "total_bits", "total_variance", "real slot changes vs fc0", "raw action changes vs fc0"],
            option_rows,
            allow_html_cells=True,
        ))
        for option in graph["options"]:
            parts.append(f"<h3>Option {option['option_id']} / fusion_count {option['fusion_count']}</h3>")
            slot_rows = []
            for row in option["slot_rows"]:
                slot_rows.append([
                    row["slot_index"],
                    row["label"],
                    row["action_index"],
                    row["decoded_value"] if row["decoded_value"] is not None else "",
                    row["real_status"],
                    "yes" if row["changed_raw_vs_fusion0"] else "",
                    "yes" if row["changed_real_vs_fusion0"] else "",
                ])
            parts.append(html_table(
                ["slot", "true slot label", "action index", "real value/K", "status", "raw changed", "real changed"],
                slot_rows,
                allow_html_cells=True,
            ))
        parts.append("</div>")

    parts.extend(["</body></html>"])
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-dir", default=str(DEFAULT_MAP_DIR))
    parser.add_argument("--output-dir", default="experiments/server_command_runs/fusion_count_map_action_eval_20260610")
    parser.add_argument("--html", default="reports/html_reports/20260610_mrpc_fusion_count_map_slots.html")
    parser.add_argument("--json", default="")
    parser.add_argument("--profile", default="mrpc")
    parser.add_argument("--gelu", default=json.dumps(DEFAULT_GELU))
    parser.add_argument("--softmax", default=json.dumps(DEFAULT_SOFTMAX))
    args = parser.parse_args()

    map_dir = Path(args.map_dir)
    if not map_dir.is_absolute():
        map_dir = REPO_ROOT / map_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    html_path = Path(args.html)
    if not html_path.is_absolute():
        html_path = REPO_ROOT / html_path
    json_path = Path(args.json) if args.json else output_dir / "fusion_count_map_report.json"
    if not json_path.is_absolute():
        json_path = REPO_ROOT / json_path

    gelu = parse_json_int_list(args.gelu, default=DEFAULT_GELU, name="--gelu")
    softmax = parse_json_int_list(args.softmax, default=DEFAULT_SOFTMAX, name="--softmax")
    if len(gelu) != len(softmax):
        raise SystemExit("GELU and Softmax degree lists must have equal length")

    fields_by_block = _parse_block_fields()
    graphs = _load_maps(map_dir)
    schedule = _schedule(len(gelu), args.profile, gelu, softmax)
    missing = sorted({s["graph_key"] for s in schedule if s["graph_key"] not in graphs})
    if missing:
        raise SystemExit(f"fusion map missing graph(s) required by current schedule: {missing}")

    groups = _group_specs(graphs, schedule)
    action_paths = _write_action_configs(
        output_dir=output_dir,
        fields_by_block=fields_by_block,
        graphs=graphs,
        num_layers=len(gelu),
        schedule=schedule,
        group_specs=groups,
        profile=args.profile,
        gelu=gelu,
        softmax=softmax,
    )
    payload = _build_report_payload(
        graphs=graphs,
        fields_by_block=fields_by_block,
        schedule=schedule,
        group_specs=groups,
        action_config_paths=action_paths,
        profile=args.profile,
        gelu=gelu,
        softmax=softmax,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_file(json_path, payload)
    html_path.write_text(_render_html(payload), encoding="utf-8")
    print(json.dumps({
        "html": str(html_path),
        "json": str(json_path),
        "action_config_dir": str(output_dir / "action_configs"),
        "action_configs": action_paths,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
