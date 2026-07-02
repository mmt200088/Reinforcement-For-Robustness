"""Orphan-slot audit for BLB Stage-2 RL.

This script walks the three layers of the action -> optimizer chain and reports
where action slots fail to reach Rescale_optimizer cost (or where graph nodes
are not under RL control). All extraction is static (AST + JSON), so it runs
without torch / numpy.

The chain is:

    RL action slot (e.g. ``wffn2_sf``)
        -- make_block{N}_default_config(**slot_kwargs) -->
    cfg attribute (e.g. ``Block1NoiseConfig.wffn2_encode``)
        -- default_block{N}_cfg_to_delta(cfg) -->
    delta_overrides key = graph node name (e.g. ``ctpt_ffn2``)
        -- replan_with_user_actions on Rescale_optimizer/configs/<profile>/<graph>.json -->
    actual cost contribution

For each (block, graph) pair we emit:

    Section 1: action_slot -> cfg_field -> graph_node table, plus "node in graph?"
    Section 2: action slots dropped at the bridge (bridge never reads that cfg field)
    Section 3: graph nodes never targeted by the bridge (graph-only)
    Section 4: bridge orphans (bridge sends a key not present in the graph)

Usage::

    python scripts/blb_orphan_slot_audit.py
    python scripts/blb_orphan_slot_audit.py --profile mrpc --out reports/blb_opt/orphan_slots/
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
_AST_CACHE: Dict[Path, ast.AST] = {}
_GRAPH_CONFIG_NAMES_CACHE: Dict[Path, Tuple[str, ...]] = {}


def _load_ast(rel_path: str) -> ast.AST:
    path = (REPO_ROOT / rel_path).resolve()
    tree = _AST_CACHE.get(path)
    if tree is None:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        _AST_CACHE[path] = tree
    return tree


# ---------------------------------------------------------------------------
# Layer 1: slot -> cfg field, extracted from function_handler.make_block{N}_default_config
# ---------------------------------------------------------------------------
def _attr_chain(node: ast.AST) -> Optional[str]:
    """``cfg.foo.bar`` → ``"foo.bar"``; ``foo`` → ``"foo"``; otherwise None."""
    parts: List[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        return ".".join(reversed(parts)) if parts else node.id
    return None


def _unwrap_int_call(node: ast.AST) -> Optional[ast.AST]:
    """``int(x)`` → ``x``; ``int(x) if cond else None`` → x's sub-node."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "int":
        if node.args:
            return node.args[0]
    return node


def _noise_point_arg_slot(call: ast.Call) -> Optional[str]:
    """For ``NoisePoint("...", int(<slot>), int(N))``, return ``<slot>``."""
    if not (isinstance(call.func, ast.Name) and call.func.id == "NoisePoint"):
        return None
    if len(call.args) < 2:
        return None
    sf_arg = _unwrap_int_call(call.args[1])
    if isinstance(sf_arg, ast.Name):
        return sf_arg.id
    return None


def load_slot_to_cfg_field(block_idx: int) -> Dict[str, Tuple[str, str]]:
    """Return ``{slot_name: (cfg_field, kind)}`` for one block.

    kind is ``"core"`` (NoisePoint is always constructed) or ``"rescale_optional"``
    (NoisePoint is constructed inside ``if <slot>_rescale_sf is not None``).
    """
    tree = _load_ast("function_handler.py")
    out: Dict[str, Tuple[str, str]] = {}
    fn_name = f"make_block{block_idx}_default_config"
    for fn in ast.walk(tree):
        if not (isinstance(fn, ast.FunctionDef) and fn.name == fn_name):
            continue
        # Case A: cfg = BlockNNoiseConfig(field=NoisePoint(..., int(slot), int(N)), ...)
        for sub in ast.walk(fn):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id.endswith("NoiseConfig"):
                for kw in sub.keywords:
                    if kw.arg is None:
                        continue
                    inner = kw.value
                    if isinstance(inner, ast.Call):
                        slot = _noise_point_arg_slot(inner)
                        if slot:
                            out[slot] = (kw.arg, "core")
        # Case B: cfg.<field> = NoisePoint("...", int(<slot>), int(N)) inside if blocks
        for stmt in fn.body:
            if not isinstance(stmt, ast.If):
                continue
            for inner_stmt in stmt.body:
                if isinstance(inner_stmt, ast.Assign):
                    if (
                        len(inner_stmt.targets) == 1
                        and isinstance(inner_stmt.targets[0], ast.Attribute)
                        and isinstance(inner_stmt.value, ast.Call)
                    ):
                        chain = _attr_chain(inner_stmt.targets[0])
                        slot = _noise_point_arg_slot(inner_stmt.value)
                        if chain and slot:
                            # Strip leading "cfg." if present
                            cfg_field = chain.split(".", 1)[-1]
                            out[slot] = (cfg_field, "rescale_optional")
    return out


# ---------------------------------------------------------------------------
# Layer 2: cfg field -> graph node, extracted from default_block{N}_cfg_to_delta
# ---------------------------------------------------------------------------
def load_cfg_field_to_graph_node(block_idx: int) -> Dict[str, Tuple[str, str]]:
    """Return ``{graph_node: (cfg_field_or_literal, kind)}`` for one block.

    kind ∈ {"cfg_field", "literal_x2", "literal_int", "literal_other"}.
    """
    tree = _load_ast("rescale_optimizer_bridge.py")
    out: Dict[str, Tuple[str, str]] = {}
    fn_name = f"default_block{block_idx}_cfg_to_delta"
    for fn in ast.walk(tree):
        if not (isinstance(fn, ast.FunctionDef) and fn.name == fn_name):
            continue
        for sub in ast.walk(fn):
            if not isinstance(sub, ast.Dict):
                continue
            for k_node, v_node in zip(sub.keys, sub.values):
                if not (isinstance(k_node, ast.Constant) and isinstance(k_node.value, str)):
                    continue
                key = k_node.value
                # Case A: int(cfg.<chain>) — record the chain
                v = _unwrap_int_call(v_node)
                if isinstance(v, ast.Attribute):
                    chain = _attr_chain(v)
                    if chain:
                        # Strip cfg. prefix and ".scaling_factor" suffix
                        if chain.startswith("cfg."):
                            chain = chain[4:]
                        if chain.endswith(".scaling_factor"):
                            chain = chain[: -len(".scaling_factor")]
                        out[key] = (chain, "cfg_field")
                        continue
                # Case B: literal "x2" / int / etc.
                if isinstance(v_node, ast.Constant):
                    if v_node.value == "x2":
                        out[key] = ("x2", "literal_x2")
                    elif isinstance(v_node.value, int):
                        out[key] = (str(v_node.value), "literal_int")
                    else:
                        out[key] = (str(v_node.value), "literal_other")
                    continue
            # Also walk for ``deltas[K] = V`` style assignments (block3/5 use these
            # for the degree-aware square_rescales loop).
        for sub in ast.walk(fn):
            if isinstance(sub, ast.Assign) and len(sub.targets) == 1 and isinstance(sub.targets[0], ast.Subscript):
                tgt = sub.targets[0]
                key_const = tgt.slice if isinstance(tgt.slice, ast.Constant) else None
                if not (key_const and isinstance(key_const.value, str)):
                    # Try f-string key like ctct_square_{k+1} — fall through with placeholder
                    if isinstance(tgt.slice, ast.JoinedStr):
                        # Reconstruct a placeholder pattern
                        pattern = ""
                        for part in tgt.slice.values:
                            if isinstance(part, ast.Constant):
                                pattern += str(part.value)
                            else:
                                pattern += "{i}"
                        out[pattern] = ("(loop-generated)", "loop")
                    continue
                key = key_const.value
                v = _unwrap_int_call(sub.value)
                if isinstance(v, ast.Attribute):
                    chain = _attr_chain(v)
                    if chain:
                        if chain.startswith("cfg."):
                            chain = chain[4:]
                        if chain.endswith(".scaling_factor"):
                            chain = chain[: -len(".scaling_factor")]
                        out[key] = (chain, "cfg_field")
                elif isinstance(sub.value, ast.Constant):
                    if sub.value.value == "x2":
                        out[key] = ("x2", "literal_x2")
                    elif isinstance(sub.value.value, int):
                        out[key] = (str(sub.value.value), "literal_int")
    return out


# ---------------------------------------------------------------------------
# Layer 2b: cfg fields fed into t_new (per graph_key) — DEFAULT_CFG_TO_T_NEW_MAP
# ---------------------------------------------------------------------------
def load_t_new_map() -> Dict[str, List[Tuple[str, Optional[int]]]]:
    """Return ``{graph_key: [(cfg_field, tuple_index_or_None), ...]}`` from
    ``DEFAULT_CFG_TO_T_NEW_MAP``. ``cfg_field`` may carry a list-like cfg attribute
    (e.g. ``square_rescales``) — in that case ``tuple_index`` is set.
    """
    tree = _load_ast("rescale_optimizer_bridge.py")
    out: Dict[str, List[Tuple[str, Optional[int]]]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign):
            target = node.target
            value = node.value
        elif isinstance(node, ast.Assign):
            target = node.targets[0] if len(node.targets) == 1 else None
            value = node.value
        else:
            continue
        if not (isinstance(target, ast.Name) and target.id == "DEFAULT_CFG_TO_T_NEW_MAP"):
            continue
        if not isinstance(value, ast.Dict):
            continue
        for k_node, v_node in zip(value.keys, value.values):
            if not (isinstance(k_node, ast.Constant) and isinstance(k_node.value, str)):
                continue
            graph_key = k_node.value
            entries: List[Tuple[str, Optional[int]]] = []
            if isinstance(v_node, ast.Tuple):
                items = v_node.elts
            elif isinstance(v_node, ast.List):
                items = v_node.elts
            else:
                items = []
            for item in items:
                if not (isinstance(item, ast.Call) and isinstance(item.func, ast.Name) and item.func.id == "_SkelEntry"):
                    continue
                cfg_field: Optional[str] = None
                tuple_index: Optional[int] = None
                if item.args:
                    a0 = item.args[0]
                    if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                        cfg_field = a0.value
                if len(item.args) >= 2:
                    a1 = item.args[1]
                    if isinstance(a1, ast.Constant) and isinstance(a1.value, int):
                        tuple_index = a1.value
                    elif isinstance(a1, ast.UnaryOp) and isinstance(a1.op, ast.USub) and isinstance(a1.operand, ast.Constant):
                        tuple_index = -int(a1.operand.value)
                for kw in item.keywords:
                    if kw.arg == "cfg_field" and isinstance(kw.value, ast.Constant):
                        cfg_field = str(kw.value.value)
                    elif kw.arg == "tuple_index" and isinstance(kw.value, ast.Constant):
                        tuple_index = int(kw.value.value)
                if cfg_field is not None:
                    entries.append((cfg_field, tuple_index))
            out[graph_key] = entries
        # Stop after first match
        if out:
            break
    return out


# Per-block, which cfg-field families correspond to action-slot families that
# feed into t_new (regardless of which graph the slot ends up at). Used to
# coalesce "square_rescales" tuple → "square_rescale_sf_0..3" action slots, etc.
_T_NEW_FAMILY_EXPANSION: Dict[str, List[str]] = {
    "square_rescales": [f"square_rescales[{i}]" for i in range(6)],
    "gelu_coeff_mul_rescales": [f"gelu_coeff_mul_rescales[{i}]" for i in range(4)],
    "gelu_power_rescales": [f"gelu_power_rescales[{i}]" for i in range(3)],
}


def cfg_fields_in_t_new_for_block(
    block_idx: int,
    t_new_map: Dict[str, List[Tuple[str, Optional[int]]]],
) -> Set[str]:
    """Aggregate cfg fields fed into t_new across all graphs for one block."""
    out: Set[str] = set()
    for graph_key, entries in t_new_map.items():
        # Match graph_key to block
        if block_idx == 1 and not graph_key.startswith("block1_"):
            continue
        if block_idx == 2 and not graph_key.startswith("block2_"):
            continue
        if block_idx == 3 and not graph_key.startswith("block3_"):
            continue
        if block_idx == 4 and not graph_key == "block4":
            continue
        if block_idx == 5 and not graph_key.startswith("block5_"):
            continue
        for cfg_field, _idx in entries:
            out.add(cfg_field)
    return out


# ---------------------------------------------------------------------------
# Layer 3: graph node names from Rescale_optimizer graph JSONs
# ---------------------------------------------------------------------------
def load_graph_node_names(graph_path: Path) -> List[Tuple[str, str]]:
    """Return ordered list of (node_name, node_type). Source first, then per-stage cut_point."""
    doc = json.loads(graph_path.read_text(encoding="utf-8"))
    out: List[Tuple[str, str]] = []
    src = doc.get("source", {})
    if isinstance(src, dict) and src.get("name"):
        out.append((str(src["name"]), str(src.get("op_type", "")).upper() or "SOURCE"))
    for stage in doc.get("stages", []) or []:
        cp = stage.get("cut_point", {}) if isinstance(stage, dict) else {}
        if isinstance(cp, dict) and cp.get("name"):
            out.append((str(cp["name"]), str(cp.get("type", "")).upper()))
    return out


def _graph_names_for_block(block_idx: int, profile: str, names: Iterable[str]) -> List[str]:
    if block_idx == 1:
        target = f"block1_{profile}.json"
        return [name for name in names if name == target]
    if block_idx == 2:
        target = f"block2_{profile}.json"
        return [name for name in names if name == target]
    if block_idx == 3:
        return [
            name
            for name in names
            if name.startswith("block3_exp_n") and name.endswith(".json")
        ]
    if block_idx == 4:
        return [name for name in names if name == "block4.json"]
    if block_idx == 5:
        return [
            name
            for name in names
            if name.startswith("block5_n") and name.endswith(".json")
        ]
    return []


def _graph_config_names(configs_dir: Path) -> Tuple[str, ...]:
    key = configs_dir.resolve()
    cached = _GRAPH_CONFIG_NAMES_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        with os.scandir(configs_dir) as entries:
            names = tuple(sorted(
                entry.name
                for entry in entries
                if entry.is_file() and entry.name.endswith(".json")
            ))
    except OSError:
        names = ()
    _GRAPH_CONFIG_NAMES_CACHE[key] = names
    return names


def graphs_for_block(block_idx: int, profile: str, configs_dir: Path) -> List[Path]:
    names = _graph_config_names(configs_dir)
    return [configs_dir / name for name in _graph_names_for_block(block_idx, profile, names)]


# ---------------------------------------------------------------------------
# Compose + render
# ---------------------------------------------------------------------------
# Action slots that feed *tuple-valued* cfg attributes — static knowledge,
# because make_block{N}_default_config receives a Sequence kwarg and the per-slot
# wiring (square_rescale_sf_0..3 → square_rescale_sfs[0..3]) happens at the
# blb_rl_bridge.build_block{N}_cfg_from_action layer. AST extraction of that
# would be fragile; we encode the patterns we know.
_TUPLE_FED_SLOTS_BY_BLOCK: Dict[int, Dict[str, Tuple[str, int]]] = {
    3: {
        f"square_rescale_sf_{k}": ("square_rescales", k)
        for k in range(4)  # max degree=4
    },
    5: {
        # gelu_power_rescales has length degree-1 (max degree=4 → 3 entries)
        **{f"gelu_power_rescale_sf_{k}": ("gelu_power_rescales", k) for k in range(3)},
        # gelu_coeff_mul_rescales has length degree (max=4)
        **{f"gelu_coeff_mul_rescale_sf_{k}": ("gelu_coeff_mul_rescales", k) for k in range(4)},
    },
}


def compose_block(
    block_idx: int,
    t_new_map: Dict[str, List[Tuple[str, Optional[int]]]],
) -> Dict[str, Any]:
    slot_to_cfg = load_slot_to_cfg_field(block_idx)        # {slot: (cfg_field, kind)}
    cfg_to_node = load_cfg_field_to_graph_node(block_idx)  # {node: (cfg_field|literal, kind)}

    # Merge in tuple-fed slot mappings as additional slot rows
    tuple_fed = _TUPLE_FED_SLOTS_BY_BLOCK.get(block_idx, {})
    for slot, (tuple_field, idx) in tuple_fed.items():
        cfg_attr = f"{tuple_field}[{idx}]"
        slot_to_cfg[slot] = (cfg_attr, "rescale_optional")

    # Invert cfg_to_node by cfg_field (for delta_overrides path)
    cfg_field_to_node: Dict[str, str] = {}
    for node, (src, kind) in cfg_to_node.items():
        if kind == "cfg_field":
            cfg_field_to_node[src] = node

    # Aggregate t_new-reached cfg fields across all graphs of this block.
    # A tuple cfg attribute "square_rescales[k]" reaches optimizer if t_new
    # references "square_rescales" with tuple_index=k (or any index, for
    # graphs that route the same tuple field through multiple stages).
    t_new_fields: Set[str] = set()         # bare cfg field names
    t_new_indexed: Set[Tuple[str, int]] = set()  # (cfg_field, tuple_index) pairs
    for graph_key, entries in t_new_map.items():
        # Match graph_key to block_idx
        if block_idx == 1 and not graph_key.startswith("block1_"):
            continue
        if block_idx == 2 and not graph_key.startswith("block2_"):
            continue
        if block_idx == 3 and not graph_key.startswith("block3_"):
            continue
        if block_idx == 4 and graph_key != "block4":
            continue
        if block_idx == 5 and not graph_key.startswith("block5_"):
            continue
        for cfg_field, tuple_index in entries:
            if tuple_index is None:
                t_new_fields.add(cfg_field)
            else:
                t_new_indexed.add((cfg_field, int(tuple_index)))

    def reaches_via_t_new(cfg_attr: str) -> bool:
        if "[" in cfg_attr and cfg_attr.endswith("]"):
            base, idx_str = cfg_attr.split("[", 1)
            try:
                idx = int(idx_str.rstrip("]"))
            except ValueError:
                return False
            # Match exact (base, idx) OR base-only entry which means "the
            # entire tuple feeds t_new (one slot per stage)".
            if (base, idx) in t_new_indexed:
                return True
            # If t_new entry is bare base (no index), interpret as "all tuple
            # entries feed t_new". (This case doesn't currently happen but is
            # safe to handle.)
            return base in t_new_fields
        return cfg_attr in t_new_fields

    rows: List[Dict[str, Any]] = []
    for slot, (cfg_field, slot_kind) in sorted(slot_to_cfg.items()):
        delta_node = cfg_field_to_node.get(cfg_field)
        t_new_hit = reaches_via_t_new(cfg_field)
        rows.append({
            "slot": slot,
            "cfg_field": cfg_field,
            "slot_kind": slot_kind,
            "delta_node": delta_node,        # cfg field → delta_overrides graph node (may be None)
            "reaches_via_delta": delta_node is not None,
            "reaches_via_t_new": bool(t_new_hit),
            "reaches_optimizer": delta_node is not None or bool(t_new_hit),
        })

    # Bridge keys (graph nodes) that the bridge sends — split into cfg-field-driven vs literal
    bridge_keys = sorted(cfg_to_node.keys())

    # Loop-generated keys (block3 ctct_square_{k}, block5 may have similar) — flag separately
    loop_keys = [k for k, (_, kind) in cfg_to_node.items() if kind == "loop"]

    return {
        "slot_rows": rows,
        "cfg_to_node": cfg_to_node,
        "bridge_keys": bridge_keys,
        "loop_keys": loop_keys,
        "t_new_fields": sorted(t_new_fields),
        "t_new_indexed": sorted(f"{f}[{i}]" for f, i in t_new_indexed),
    }


def render_block_section(
    block_idx: int,
    composed: Dict[str, Any],
    graphs: List[Tuple[str, List[Tuple[str, str]]]],
) -> str:
    lines: List[str] = []
    lines.append(f"## Block {block_idx}\n")

    rows = composed["slot_rows"]
    bridge_keys: List[str] = composed["bridge_keys"]
    cfg_to_node: Dict[str, Tuple[str, str]] = composed["cfg_to_node"]
    loop_keys: List[str] = composed["loop_keys"]

    reached = [r for r in rows if r["reaches_optimizer"]]
    dropped = [r for r in rows if not r["reaches_optimizer"]]

    lines.append(
        f"Action slots: **{len(rows)}** total "
        f"(reach optimizer: **{len(reached)}**, dropped: **{len(dropped)}**).  "
        f"Bridge `delta_overrides` keys: **{len(bridge_keys)}** "
        f"({sum(1 for v in cfg_to_node.values() if v[1] == 'cfg_field')} cfg-field-driven, "
        f"{sum(1 for v in cfg_to_node.values() if v[1].startswith('literal'))} literal, "
        f"{len(loop_keys)} loop-generated).  "
        f"`t_new` feeds: **{len(composed['t_new_fields']) + len(composed['t_new_indexed'])}** cfg fields.\n"
    )

    lines.append("### Slot → cfg field → optimizer reach\n")
    lines.append("| slot | cfg field | optional? | delta_overrides node | t_new? | reaches optimizer |")
    lines.append("|------|-----------|-----------|----------------------|--------|-------------------|")
    for r in rows:
        delta_cell = f"`{r['delta_node']}`" if r["delta_node"] else "—"
        opt = "yes" if r["slot_kind"] == "rescale_optional" else "no"
        t_new_cell = "yes" if r["reaches_via_t_new"] else "—"
        reach = "yes" if r["reaches_optimizer"] else "**no**"
        lines.append(
            f"| `{r['slot']}` | `{r['cfg_field']}` | {opt} | {delta_cell} | {t_new_cell} | {reach} |"
        )
    lines.append("")

    if dropped:
        lines.append("### Slots dropped (model noise installs, optimizer cost never sees)\n")
        for r in dropped:
            lines.append(f"- `{r['slot']}` → cfg `{r['cfg_field']}` (not in delta map, not in t_new map)")
        lines.append("")

    if not graphs:
        lines.append("_No graph JSON found for this block._\n")
        return "\n".join(lines)

    for graph_name, graph_nodes in graphs:
        node_names = {n for n, _t in graph_nodes}
        lines.append(f"### Graph `{graph_name}`  ({len(graph_nodes)} cut points)\n")
        lines.append("| # | node | type | bridge driver | RL-driven slot |")
        lines.append("|---|------|------|---------------|----------------|")
        # Build a slot-driver index: graph_node → action slot (via delta path)
        driver_index: Dict[str, str] = {}
        for r in rows:
            if r["delta_node"]:
                driver_index.setdefault(r["delta_node"], r["slot"])
        for i, (name, ntype) in enumerate(graph_nodes):
            bridge_src = cfg_to_node.get(name)
            if bridge_src:
                src_str = f"`cfg.{bridge_src[0]}`" if bridge_src[1] == "cfg_field" else f"literal `{bridge_src[0]}`"
            else:
                # Check loop pattern match
                matched_loop = None
                for lk in loop_keys:
                    prefix = lk.split("{", 1)[0]
                    if name.startswith(prefix):
                        matched_loop = lk
                        break
                src_str = f"loop `{matched_loop}`" if matched_loop else "**(none)**"
            slot = driver_index.get(name, "—")
            slot_str = f"`{slot}`" if slot != "—" else "—"
            lines.append(f"| {i} | `{name}` | {ntype} | {src_str} | {slot_str} |")
        lines.append("")

        # Bridge orphans (bridge sends a key the graph doesn't have)
        bridge_orphans = sorted(
            k for k in bridge_keys
            if k not in node_names and "{" not in k
        )
        if bridge_orphans:
            lines.append(
                f"**Bridge orphans for `{graph_name}`** (bridge sends but graph has no such node):"
            )
            for k in bridge_orphans:
                if cfg_to_node[k][1] == "cfg_field":
                    lines.append(f"  - `{k}` ← `cfg.{cfg_to_node[k][0]}`")
                else:
                    lines.append(f"  - `{k}` ← literal `{cfg_to_node[k][0]}`")
            lines.append("")

        graph_only = sorted(
            n for n in node_names if n not in bridge_keys and not any(
                n.startswith(lk.split("{", 1)[0]) for lk in loop_keys
            )
        )
        if graph_only:
            lines.append(
                f"**Graph-only nodes for `{graph_name}`** (graph has these, no bridge key targets them — "
                "optimizer uses graph defaults; note that `source` nodes are intentionally driven by `t_new[0]`):"
            )
            for n in graph_only:
                lines.append(f"  - `{n}`")
            lines.append("")
    return "\n".join(lines)


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="BLB orphan-slot audit")
    ap.add_argument("--profile", default="mrpc")
    ap.add_argument(
        "--rescale-optimizer-root",
        default=str(REPO_ROOT / "Rescale_optimizer"),
    )
    ap.add_argument(
        "--out",
        default=str(REPO_ROOT / "reports" / "blb_opt" / "orphan_slots"),
    )
    args = ap.parse_args(argv)

    configs_dir = Path(args.rescale_optimizer_root) / "configs" / args.profile
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append(f"# BLB orphan-slot audit — profile `{args.profile}`\n")
    lines.append(
        "Three-layer trace of every RL action slot through to a Rescale_optimizer graph node. "
        "Generated by `scripts/blb_orphan_slot_audit.py`. Static AST + JSON only (no torch needed).\n"
    )
    lines.append("**Layers**:")
    lines.append("- L1 — `make_block{N}_default_config` in `function_handler.py` (slot kwarg → cfg attribute)")
    lines.append("- L2 — `default_block{N}_cfg_to_delta` in `rescale_optimizer_bridge.py` (cfg attribute → delta_overrides key)")
    lines.append(f"- L3 — `{configs_dir}/block*.json` (graph node names)\n")
    lines.append("**Reading the tables**:")
    lines.append("- *reaches optimizer = no* → action slot influences model noise install but is invisible to cost.")
    lines.append("- *bridge driver = (none)* → graph node is NOT under RL control; optimizer uses graph defaults for it.")
    lines.append("- Bridge orphans = bridge code sends a key that this graph doesn't have (real bug or stale name).\n")

    t_new_map = load_t_new_map()

    summary: Dict[str, Any] = {"profile": args.profile, "blocks": {}}
    for block_idx in (1, 2, 3, 4, 5):
        composed = compose_block(block_idx, t_new_map)
        graphs = [
            (gp.stem, load_graph_node_names(gp))
            for gp in graphs_for_block(block_idx, args.profile, configs_dir)
        ]
        lines.append(render_block_section(block_idx, composed, graphs))

        rows = composed["slot_rows"]
        bridge_keys = composed["bridge_keys"]
        block_summary: Dict[str, Any] = {
            "action_slot_count": len(rows),
            "slots_reaching_optimizer": [r["slot"] for r in rows if r["reaches_optimizer"]],
            "slots_dropped": [r["slot"] for r in rows if not r["reaches_optimizer"]],
            "slots_via_delta": [r["slot"] for r in rows if r["reaches_via_delta"]],
            "slots_via_t_new": [r["slot"] for r in rows if r["reaches_via_t_new"]],
            "bridge_keys": bridge_keys,
            "loop_keys": composed["loop_keys"],
            "t_new_fields": composed["t_new_fields"],
            "t_new_indexed": composed["t_new_indexed"],
            "graphs": {},
        }
        for graph_name, graph_nodes in graphs:
            node_names = {n for n, _t in graph_nodes}
            block_summary["graphs"][graph_name] = {
                "node_count": len(graph_nodes),
                "nodes": [n for n, _t in graph_nodes],
                "bridge_orphans": sorted(
                    k for k in bridge_keys
                    if k not in node_names and "{" not in k
                ),
                "graph_only_nodes": sorted(
                    n for n in node_names if n not in bridge_keys and not any(
                        n.startswith(lk.split("{", 1)[0]) for lk in composed["loop_keys"]
                    )
                ),
            }
        summary["blocks"][f"block{block_idx}"] = block_summary

    lines.append("\n## Known caveats\n")
    lines.append(
        "- **Block 5 degree-aware bridge orphans.** "
        "`default_block5_cfg_to_delta` only writes `ctct_gelu_x2` when `cfg.gelu_degree >= 2` "
        "and `ctct_gelu_x4` when `cfg.gelu_degree >= 4`. The AST extractor doesn't see these "
        "guards, so it lists them as orphans for `block5_n1` (no gelu_x*) and `block5_n2` "
        "(no gelu_x4). At runtime the bridge skips them for those graphs."
    )
    lines.append(
        "- **Block 2 Q/K SF tie.** Q and K share `ctpt_wq_wk` by BLB constraint. "
        "The bridge sends `cfg.wq_encode.scaling_factor` only; `wk_sf` action is decoded into "
        "model noise but ignored on the optimizer side. If the optimizer requires Q=K tied, "
        "`tied_group` enforcement should be in the registry."
    )
    lines.append(
        "- **`source` nodes appear as graph-only.** Each graph's `source` node (e.g. "
        "`gelu_out`, `inv_std`, `X`, `rot_softmax`, `x_mean`) is intentionally driven by "
        "`t_new[0]` from the cfg's `*_fresh` field, not by `delta_overrides`. The audit "
        "lists them as graph-only for clarity; they are NOT actually orphan."
    )
    lines.append(
        "- **Loop-generated keys** (`ctct_square_{k+1}` in block 3) come from a runtime loop "
        "over `cfg.degree`. The audit flags them as `loop` instead of mapping to a single "
        "static cfg field."
    )
    md_path = out_dir / f"audit_{args.profile}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    json_path = out_dir / f"audit_{args.profile}.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"wrote {md_path}")
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
