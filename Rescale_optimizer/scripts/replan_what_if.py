"""
replan_what_if.py
=================

CLI for ``rescale_optimizer.replan.replan_with_user_actions``.

It reads:

  1. A graph config JSON (the same kind ``optimize_rescale`` consumes);
  2. A baseline rescale skeleton + baseline ``t`` vector
     (either from a static-skeleton archive produced by
     ``batch_run_configs.py``, or supplied manually via flags);
  3. A **new** ``t`` vector + optional propagation delta overrides,
     either from one unified ``--actions-file`` JSON, or from separate
     flags/files.

It runs scale propagation under the new ``t``, builds the new modulus
chain, and applies the **fusion-tolerant feasibility check** (see
``rescale_optimizer/replan.py``).  It prints a summary and (optionally)
dumps the full ``ReplanResult`` as JSON.

Examples
--------

::

    # 1) Pull baseline from a static-skeletons file produced by
    #    scripts/batch_run_configs.py, replan with new t per stage.
    python scripts/replan_what_if.py \
        --config configs/block1_wnli.json \
        --baseline-from diagnose_certacc_output/static_skeletons.json \
        --t-new 30 35 35

    # 2) Supply baseline manually (skeleton + t).
    python scripts/replan_what_if.py \
        --config configs/block1_wnli.json \
        --skeleton 0 1 3 5 \
        --t-baseline 10 35 30 \
        --t-new 10 30 25

    # 3) Read t_new from a JSON file.
    python scripts/replan_what_if.py \
        --config configs/block1_wnli.json \
        --baseline-from .../static_skeletons.json \
        --t-new-file my_user_actions.json \
        --out replan_block1.json

    # 4) Co-optimize t + propagation_deltas from ONE file.
    python scripts/replan_what_if.py \
        --config configs/block1_wnli.json \
        --baseline-from configs/static_skeletons.json \
        --actions-file configs/replan_actions_block1_example.json \
        --out replan_block1.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rescale_optimizer import (
    DEFAULT_FUSION_POLICY,
    ReplanInputs,
    load_graph_from_json,
    replan_with_user_actions,
    resolve_allowed_fusion_pairs,
)
from rescale_optimizer.feasibility import build_feasibility_dag
from rescale_optimizer.graph import NodeType, propagate_scale
from rescale_optimizer.utils import setup_logging


def _load_baseline_from_archive(
    archive_path: Path,
    config_name: str,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Pull baseline (skeleton, t_baseline, q_bits_baseline) from a
    static-skeletons archive (output of batch_run_configs.py).

    Supports both the new compact schema v2 (``cut_point_sf`` /
    ``modulus_chain.drop_order``) and the legacy verbose schema v1.
    """
    with open(archive_path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    for entry in doc.get("results", []):
        if entry.get("config_name") != config_name:
            continue
        if not entry.get("success", False):
            raise ValueError(
                f"baseline archive entry for '{config_name}' is not successful: "
                f"{entry.get('message')}"
            )
        skel = [int(x) for x in entry.get("skeleton", [])]

        # ---- compact schema v2 ----
        if "cut_point_sf" in entry and "modulus_chain" in entry:
            # In v2 schema, cut points have either {sf} (source / non-rescale)
            # or {sf_pre, sf_post, drop} (rescale).  The baseline t per
            # skeleton stage is:
            #   stage 0 (source)   : sf
            #   stage r >= 1       : sf_post   (post-rescale working scale)
            t_for_idx: Dict[int, int] = {}
            for row in entry["cut_point_sf"]:
                i = int(row["i"])
                if "sf_post" in row:
                    t_for_idx[i] = int(row["sf_post"])
                elif "sf" in row:
                    t_for_idx[i] = int(row["sf"])
            t_base: List[int] = []
            for cp_idx in skel:
                if cp_idx in t_for_idx:
                    t_base.append(t_for_idx[cp_idx])
            mc = entry["modulus_chain"]
            drop_order = list(mc.get("drop_order", []))
            # drop_order = [q_head, q_1, ..., q_R, q_tail]
            q_base = [int(x) for x in drop_order[1:-1]] if len(drop_order) >= 2 else []
            return skel, t_base, q_base

        # ---- legacy schema v1 ----
        t_base = list(entry.get("t_per_stage", []))
        q_base = list(
            entry.get("dp_drop_bits")
            or entry.get("drop_bits_per_stage")
            or []
        )
        return skel, t_base, q_base
    raise ValueError(
        f"config_name '{config_name}' not found in archive {archive_path}"
    )


def _load_t_new(args: argparse.Namespace, expected_len: int) -> List[int]:
    if args.t_new_file:
        with open(args.t_new_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            t = data
        elif isinstance(data, dict) and "t_new" in data:
            t = data["t_new"]
        else:
            raise ValueError(
                f"--t-new-file must contain a JSON list, or {{'t_new': [...]}}. "
                f"got {type(data).__name__}"
            )
    elif args.t_new is not None:
        t = list(args.t_new)
    else:
        raise ValueError("either --t-new or --t-new-file must be provided")
    if len(t) != expected_len:
        raise ValueError(
            f"t_new length must be {expected_len} (R+1 for the baseline skeleton), "
            f"got {len(t)}"
        )
    return [int(x) for x in t]


def _load_actions_file(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("--actions-file must be a JSON object")
    return data


def _parse_delta_token(tok: Any) -> Union[int, str]:
    if tok == "x2":
        return "x2"
    try:
        return int(tok)
    except (TypeError, ValueError):
        raise ValueError(f"delta must be int or 'x2', got {tok!r}")


def _load_delta_overrides(args: argparse.Namespace) -> Dict[str, Union[int, str]]:
    """
    Load optional propagation delta overrides from CLI flags.

    Supports:
      - repeated ``--delta-override <node_name> <delta>``
      - ``--delta-overrides-file`` JSON, either:
          * {"node_name": 20, "ctct_x": "x2", ...}
          * {"propagation_deltas":[{"name":"...", "delta": ...}, ...]}
    """
    out: Dict[str, Union[int, str]] = {}

    if args.delta_overrides_file:
        with open(args.delta_overrides_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "propagation_deltas" in data:
            rows = data["propagation_deltas"]
            if not isinstance(rows, list):
                raise ValueError("'propagation_deltas' must be a list")
            for row in rows:
                if not isinstance(row, dict) or "name" not in row or "delta" not in row:
                    raise ValueError("each propagation_deltas item must have {name, delta}")
                out[str(row["name"])] = _parse_delta_token(row["delta"])
        elif isinstance(data, dict):
            for k, v in data.items():
                out[str(k)] = _parse_delta_token(v)
        else:
            raise ValueError(
                "--delta-overrides-file must be a JSON object, or "
                "{'propagation_deltas': [{'name': ..., 'delta': ...}, ...]}"
            )

    if args.delta_override:
        for name, raw in args.delta_override:
            out[str(name)] = _parse_delta_token(raw)

    return out


def _load_t_new_with_actions(
    args: argparse.Namespace,
    expected_len: int,
    actions_doc: Dict[str, Any],
) -> List[int]:
    # CLI has highest priority.
    if args.t_new is not None or args.t_new_file:
        return _load_t_new(args, expected_len)
    if "t_new" not in actions_doc:
        raise ValueError(
            "t_new is missing. Provide --t-new / --t-new-file, or include "
            "'t_new' in --actions-file."
        )
    t = actions_doc["t_new"]
    if not isinstance(t, list):
        raise ValueError("actions-file field 't_new' must be a list")
    if len(t) != expected_len:
        raise ValueError(
            f"t_new length must be {expected_len} (R+1 for the baseline skeleton), "
            f"got {len(t)}"
        )
    return [int(x) for x in t]


def _load_delta_overrides_with_actions(
    args: argparse.Namespace,
    actions_doc: Dict[str, Any],
) -> Dict[str, Union[int, str]]:
    merged: Dict[str, Union[int, str]] = {}

    if "delta_overrides" in actions_doc:
        raw = actions_doc["delta_overrides"]
        if not isinstance(raw, dict):
            raise ValueError("actions-file field 'delta_overrides' must be a JSON object")
        for k, v in raw.items():
            merged[str(k)] = _parse_delta_token(v)
    if "propagation_deltas" in actions_doc:
        rows = actions_doc["propagation_deltas"]
        if not isinstance(rows, list):
            raise ValueError("actions-file field 'propagation_deltas' must be a list")
        for row in rows:
            if not isinstance(row, dict) or "name" not in row or "delta" not in row:
                raise ValueError("actions-file propagation_deltas must have {name, delta}")
            merged[str(row["name"])] = _parse_delta_token(row["delta"])

    # Explicit CLI/file overrides take precedence.
    merged.update(_load_delta_overrides(args))
    return merged


def _load_allowed_fusion_pairs_with_actions(
    args: argparse.Namespace,
    actions_doc: Dict[str, Any],
    graph_key: str,
) -> Optional[List[Tuple[int, int]]]:
    raw: Any = DEFAULT_FUSION_POLICY

    if "fusion_policy" in actions_doc:
        raw = actions_doc["fusion_policy"]
    if "allowed_fusion_pairs" in actions_doc:
        raw = actions_doc["allowed_fusion_pairs"]
    elif "fusion_pairs" in actions_doc:
        raw = actions_doc["fusion_pairs"]

    if args.fusion_policy != DEFAULT_FUSION_POLICY:
        raw = args.fusion_policy
    if args.allowed_fusion_pair:
        raw = [[int(a), int(b)] for a, b in args.allowed_fusion_pair]

    return resolve_allowed_fusion_pairs(graph_key, raw)


def _fusion_pairs_to_json(allowed_fusion_pairs: Optional[List[Tuple[int, int]]]):
    if allowed_fusion_pairs is None:
        return None
    return [[int(a), int(b)] for a, b in allowed_fusion_pairs]


def _extract_current_propagation_deltas(graph) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for node in graph.nodes:
        if node.node_type == NodeType.CTPT_MUL:
            rows.append({
                "node_id": int(node.node_id),
                "name": node.name,
                "type": "CTPT_MUL",
                "delta": int(node.scale_delta_bits),
            })
        elif node.node_type == NodeType.CTCT_MUL:
            rows.append({
                "node_id": int(node.node_id),
                "name": node.name,
                "type": "CTCT_MUL",
                "delta": ("x2" if node.other_ct_scale_bits is None else int(node.other_ct_scale_bits)),
            })
    return rows


def _build_new_compact_config(
    graph,
    config_name: str,
    result,
) -> Optional[Dict[str, Any]]:
    if not result.valid or result.chain is None:
        return None

    chain = result.chain
    skeleton = list(result.skeleton)
    t_vec = list(result.t_final)
    M = graph.M
    R = chain.R

    rescale_index_at: Dict[int, int] = {skeleton[r]: r for r in range(1, R + 1)}
    cut_point_sf: List[Dict[str, Any]] = []

    for i in range(M + 1):
        cp = graph.cut_points[i]
        tname = cp.node.node_type.name

        if i in rescale_index_at:
            r = rescale_index_at[i]
            sf_pre = int(propagate_scale(t_vec[r - 1], graph.nodes_between(skeleton[r - 1], i)))
            cut_point_sf.append({
                "i": i, "name": cp.node.name, "type": tname,
                "sf_pre": sf_pre,
                "sf_post": int(t_vec[r]),
                "drop": int(chain.q_bits[r - 1]),
            })
            continue

        if i == skeleton[0]:
            cut_point_sf.append({
                "i": i, "name": cp.node.name, "type": tname,
                "sf": int(t_vec[0]),
            })
            continue

        r_prev = max(r for r in range(R + 1) if skeleton[r] <= i)
        sf = int(propagate_scale(t_vec[r_prev], graph.nodes_between(skeleton[r_prev], i)))
        cut_point_sf.append({
            "i": i, "name": cp.node.name, "type": tname,
            "sf": sf,
        })

    drop_order = [int(chain.q_head_bits)] + [int(b) for b in chain.q_bits] + [int(chain.q_tail_bits)]
    seal_order = [int(chain.q_head_bits)] + list(reversed([int(b) for b in chain.q_bits])) + [int(chain.q_tail_bits)]

    # ---- effective rotations (must mirror batch_run_configs._extract_effective_rotations) ----
    # A ROTATION whose stage_anchor k matches a rescale cut point in the
    # skeleton (k ∈ skeleton[1..R]) executes at the post-rescale working
    # scale = t_vec[r] where skeleton[r] == k.
    skel_full = list(skeleton) + ([] if skeleton[-1] == graph.dummy_sink_index else [graph.dummy_sink_index])
    rescale_index_at: Dict[int, int] = {
        skel_full[r]: r for r in range(1, len(skel_full) - 1)
    }
    effective_rotations: List[Dict[str, Any]] = []
    for node in graph.nodes:
        if node.node_type != NodeType.ROTATION:
            continue
        k = int(node.stage_anchor)
        if k not in rescale_index_at:
            continue
        r = rescale_index_at[k]
        effective_rotations.append({
            "node_id": int(node.node_id),
            "name": node.name,
            "after_cut_point": k,
            "sf": int(t_vec[r]),
            "count": int(node.count),
        })

    return {
        "config_name": config_name,
        "success": True,
        "skeleton": [int(x) for x in skeleton],
        "cut_point_sf": cut_point_sf,
        "propagation_deltas": _extract_current_propagation_deltas(graph),
        "modulus_chain": {
            "drop_order": drop_order,
            "seal_order": seal_order,
            "total_bits": int(chain.total_bits),
        },
        "effective_rotations": effective_rotations,
    }


def _is_primitive(x: Any) -> bool:
    return isinstance(x, (str, int, float, bool)) or x is None


def _dumps_compact_json(obj: Any, indent: int = 2, _level: int = 0) -> str:
    """
    Compact-but-readable JSON renderer:
      - keep overall hierarchy multi-line
      - keep small dicts / short arrays on one line
      - render list-of-small-dicts one item per line (item itself one-line)
    """
    pad = " " * (indent * _level)
    pad_in = " " * (indent * (_level + 1))

    if _is_primitive(obj):
        return json.dumps(obj, ensure_ascii=False)

    if isinstance(obj, list):
        if not obj:
            return "[]"
        if all(_is_primitive(v) for v in obj) and len(obj) <= 12:
            return json.dumps(obj, ensure_ascii=False, separators=(", ", ": "))

        # list of small primitive dicts: one-line per item
        if all(isinstance(v, dict) and len(v) <= 10 and all(_is_primitive(x) for x in v.values()) for v in obj):
            lines = ["["]
            for i, item in enumerate(obj):
                suf = "," if i < len(obj) - 1 else ""
                lines.append(f"{pad_in}{json.dumps(item, ensure_ascii=False, separators=(', ', ': '))}{suf}")
            lines.append(f"{pad}]")
            return "\n".join(lines)

        lines = ["["]
        for i, item in enumerate(obj):
            suf = "," if i < len(obj) - 1 else ""
            lines.append(f"{pad_in}{_dumps_compact_json(item, indent, _level + 1)}{suf}")
        lines.append(f"{pad}]")
        return "\n".join(lines)

    if isinstance(obj, dict):
        if not obj:
            return "{}"
        if len(obj) <= 6 and all(_is_primitive(v) for v in obj.values()):
            return json.dumps(obj, ensure_ascii=False, separators=(", ", ": "))

        lines = ["{"]
        last_idx = len(obj) - 1
        for i, (k, v) in enumerate(obj.items()):
            suf = "," if i < last_idx else ""
            vv = _dumps_compact_json(v, indent, _level + 1)
            lines.append(f"{pad_in}{json.dumps(k, ensure_ascii=False)}: {vv}{suf}")
        lines.append(f"{pad}}}")
        return "\n".join(lines)

    # fallback
    return json.dumps(obj, ensure_ascii=False)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Replan with user-supplied scaling factors and fusion-tolerant feasibility.",
    )
    p.add_argument(
        "--config", required=True,
        help="Path to the graph config JSON (same as optimize_rescale).",
    )
    p.add_argument(
        "--actions-file", default=None,
        help=("Unified JSON file containing both t_new and optional delta overrides. "
              "Supports keys: t_new, delta_overrides, propagation_deltas."),
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--baseline-from",
        help=("Path to a static-skeletons archive (from batch_run_configs.py). "
              "Together with --config-name (or auto-derived from --config stem) "
              "we pick the matching baseline skeleton + t."),
    )
    src.add_argument(
        "--skeleton", nargs="+", type=int,
        help="Manual baseline skeleton (cut-point indices). The trailing dummy-sink "
             "is auto-appended if missing.",
    )
    p.add_argument(
        "--config-name",
        default=None,
        help=("Override config_name used to look up baseline in --baseline-from. "
              "Defaults to the stem of --config."),
    )
    p.add_argument(
        "--t-baseline", nargs="*", type=int, default=None,
        help="Baseline t per skeleton stage (length R+1). "
             "Used only for diagnostics / delta_q reporting.",
    )

    t_group = p.add_mutually_exclusive_group(required=False)
    t_group.add_argument(
        "--t-new", nargs="+", type=int,
        help="New t per skeleton stage (length R+1).",
    )
    t_group.add_argument(
        "--t-new-file",
        help="JSON file containing the new t vector "
             "(either a flat list or {'t_new': [...]}).",
    )

    p.add_argument(
        "--out", default=None,
        help="Optional JSON path to write the full ReplanResult.",
    )
    p.add_argument(
        "--log-level", default="WARNING",
        help="Logger level for graph loading (default WARNING).",
    )
    p.add_argument(
        "--delta-override", nargs=2, action="append", default=None,
        metavar=("NODE_NAME", "DELTA"),
        help=("Override one multiplication node propagation delta. "
              "For CTPT: DELTA=int. For CTCT: DELTA=int or 'x2'. "
              "May be repeated."),
    )
    p.add_argument(
        "--delta-overrides-file", default=None,
        help=("JSON file with propagation delta overrides. Supports either "
              "{node_name: delta, ...} or "
              "{'propagation_deltas':[{'name':..., 'delta':...}, ...]}."),
    )
    p.add_argument(
        "--fusion-policy",
        choices=[DEFAULT_FUSION_POLICY, "all", "none"],
        default=DEFAULT_FUSION_POLICY,
        help=("Legal fusion policy override. default uses graph_key-specific policy; "
              "all keeps legacy adjacent fusion; none disables fusion."),
    )
    p.add_argument(
        "--allowed-fusion-pair",
        nargs=2,
        type=int,
        action="append",
        default=None,
        metavar=("STAGE_A", "STAGE_B"),
        help=("Allow one original 1-indexed rescale-stage pair to fuse. "
              "May be repeated and overrides --fusion-policy/actions-file."),
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    setup_logging(level=getattr(logging, args.log_level.upper(), logging.WARNING))

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"config not found: {cfg_path}", file=sys.stderr)
        return 1

    graph, _opt_cfg, _amp_budgets = load_graph_from_json(cfg_path)
    # we need stage_node_lists populated for replan
    build_feasibility_dag(graph)

    # ----- baseline -----------------------------------------------------
    baseline_q_bits: Optional[List[int]] = None
    t_baseline: Optional[List[int]] = None
    if args.baseline_from is not None:
        cfg_name = args.config_name or cfg_path.stem
        skeleton, t_baseline, baseline_q_bits = _load_baseline_from_archive(
            Path(args.baseline_from), cfg_name,
        )
    else:
        skeleton = list(args.skeleton)
        t_baseline = list(args.t_baseline) if args.t_baseline else None

    if not skeleton:
        print("empty skeleton", file=sys.stderr)
        return 1
    if skeleton[-1] != graph.dummy_sink_index:
        skeleton = skeleton + [graph.dummy_sink_index]

    R = len(skeleton) - 2
    expected_len = R + 1

    try:
        actions_doc = _load_actions_file(args.actions_file)
        t_new = _load_t_new_with_actions(args, expected_len, actions_doc)
    except ValueError as e:
        print(f"bad t_new: {e}", file=sys.stderr)
        return 1
    try:
        delta_overrides = _load_delta_overrides_with_actions(args, actions_doc)
    except ValueError as e:
        print(f"bad delta overrides: {e}", file=sys.stderr)
        return 1
    graph_key = args.config_name or cfg_path.stem
    try:
        allowed_fusion_pairs = _load_allowed_fusion_pairs_with_actions(
            args, actions_doc, graph_key
        )
    except ValueError as e:
        print(f"bad allowed fusion pairs: {e}", file=sys.stderr)
        return 1

    inputs = ReplanInputs(
        skeleton=skeleton,
        t_baseline=t_baseline,
        t_new=t_new,
        delta_overrides=delta_overrides or None,
        allowed_fusion_pairs=allowed_fusion_pairs,
    )

    result = replan_with_user_actions(graph, inputs, baseline_q_bits=baseline_q_bits)

    print(result.summary())

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_doc: Dict[str, Any] = {
            "config_path": str(cfg_path),
            "config_name": graph_key,
            "valid": result.valid,
            "fusion_count": result.fusion_count,
            "actions_file": args.actions_file,
            "baseline": {
                "skeleton": skeleton,
                "t_baseline": t_baseline,
                "q_bits_baseline": baseline_q_bits,
            },
            "t_new": t_new,
            "delta_overrides": delta_overrides,
            "allowed_fusion_pairs": _fusion_pairs_to_json(allowed_fusion_pairs),
            "result": {
                "valid": result.valid,
                "message": result.message,
                "fusion_count": result.fusion_count,
                "skeleton": result.skeleton,
                "q_initial": result.q_initial,
                "q_final": result.q_final,
                "t_final": result.t_final,
                "delta_q_vs_baseline": result.delta_q_vs_baseline,
                "applied_delta_overrides": result.applied_delta_overrides,
                "fusions": [
                    {
                        "fused_position": ev.fused_position,
                        "fused_into": ev.fused_into,
                        "small_q": ev.small_q,
                        "neighbour_q_before": ev.neighbour_q_before,
                        "neighbour_q_after": ev.neighbour_q_after,
                    }
                    for ev in result.fusions
                ],
                "chain": (
                    None if result.chain is None else {
                        "q_head_bits": result.chain.q_head_bits,
                        "q_bits": list(result.chain.q_bits),
                        "q_tail_bits": result.chain.q_tail_bits,
                        "total_bits": result.chain.total_bits,
                        "R": result.chain.R,
                    }
                ),
                "invalid_chain": (
                    None if result.invalid_chain is None else {
                        "q_head_bits": result.invalid_chain.q_head_bits,
                        "q_bits": list(result.invalid_chain.q_bits),
                        "q_tail_bits": result.invalid_chain.q_tail_bits,
                    }
                ),
            },
        }
        compact = _build_new_compact_config(graph, graph_key, result)
        if compact is not None:
            compact["fusion_count"] = result.fusion_count
            out_doc["new_compact_config"] = compact
        out_text = _dumps_compact_json(out_doc) + "\n"
        out_path.write_text(out_text, encoding="utf-8")
        print(f"\n[replan] wrote {out_path}")

    return 0 if result.valid else 3


if __name__ == "__main__":
    sys.exit(main())
