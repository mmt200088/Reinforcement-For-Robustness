"""
batch_run_configs.py

Run a set (or all) of configs/*.json and dump every config's "static
rescale skeleton" summary into a single compact JSON archive.

Per-config schema (compact, one-line-per-row arrays of small dicts):

    {
      "config_name": "block1_mrpc",
      "success": true,
      "skeleton": [0, 2, 4, 5],

      "cut_point_sf": [
        {"i": 0, "name": "gelu_out",        "type": "SOURCE",   "sf": 32},
        {"i": 1, "name": "ctpt_ffn2",       "type": "CTPT_MUL", "sf": 52},
        {"i": 2, "name": "ctpt_inv_d_1",    "type": "CTPT_MUL",
         "sf_pre": 72, "sf_post": 34, "drop": 38},
        {"i": 3, "name": "ctct_ext_square", "type": "CTCT_MUL", "sf": 68},
        {"i": 4, "name": "ctpt_inv_d_2",    "type": "CTPT_MUL",
         "sf_pre": 88, "sf_post": 32, "drop": 56}
      ],

      "propagation_deltas": [
        {"name": "ctpt_ffn2",       "type": "CTPT_MUL", "delta": 20},
        {"name": "ctct_ext_square", "type": "CTCT_MUL", "delta": "x2"},
        ...
      ],

      "modulus_chain": {
        "drop_order": [60, 38, 56, 60],
        "seal_order": [60, 56, 38, 60],
        "total_bits": 214
      },

      "effective_rotations": [
        {"node_id": 7, "name": "rot_sum2", "after_cut_point": 4,
         "sf": 32, "count": 3}
      ]
    }

Field meanings:
  * cut_point_sf[i] (per cut point i = 0..M):
      - For source / non-rescale cut points: a single ``sf`` field giving
        the scale after this node's multiplication has executed (= what
        ``propagate_scale`` reaches at this cut point).
      - For rescale cut points (those in the skeleton, except source):
        two fields ``sf_pre`` (scale just before rescale, = propagation
        result) and ``sf_post`` (scale just after rescale, derived from
        the chain prime size: sf_post = sf_pre - chain_q[r-1]). ``drop``
        equals the chain prime consumed at this rescale.

      The "sf_post" view is **chain-consistent** (the actual SEAL
      semantics: scaling factor decreases by exactly ``log2(prime)``
      bits at rescale).  When ``compress_headroom`` introduces slack
      relative to the optimizer's stored ``t`` (because CTCT doubling
      makes propagation non-linear in the start scale), this view
      still gives the operationally correct post-rescale scale.

  * propagation_deltas   -- per-multiplication contribution to the working
                            scale from the OTHER operand:
                              CTPT_MUL             -> scale_delta_bits
                              CTCT_MUL asymmetric  -> other_ct_scale_bits
                              CTCT_MUL symmetric   -> "x2" (doubles)
  * modulus_chain.drop_order = [q_head, q_1, ..., q_R, q_tail]  (rescale order)
  * modulus_chain.seal_order = [q_head, q_R, ..., q_1, q_tail]  (SEAL order)
  * effective_rotations -- rotation nodes that execute IMMEDIATELY after a
                           rescale (i.e. with no non-rescale cut point in
                           between).  A ROTATION with stage_anchor = k is
                           "effective" iff k is a rescale cut point in the
                           skeleton (k in skeleton[1:R+1]); then it
                           operates at the post-rescale working scale
                           (= sf_post of c_k).  Rotations sitting after
                           the source or after a non-rescale CTPT/CTCT
                           cut point are "ineffective" (they run at a
                           higher accumulated scale) and are NOT listed.
                             - node_id        : graph-global node id
                             - name           : rotation node name
                             - after_cut_point: index k of the preceding rescale cut point
                             - sf             : scaling factor at execution (= sf_post of c_k)
                             - count          : repetition count (cost weight)

Failed configs only emit  {"config_name", "success": false, "message"}.

Usage:
    python scripts/batch_run_configs.py
    python scripts/batch_run_configs.py --configs block1_mrpc block4
    python scripts/batch_run_configs.py --out my_skeletons.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rescale_optimizer import (
    OptimizationResult,
    load_graph_from_json,
    optimize_rescale,
)
from rescale_optimizer.graph import NodeType, propagate_scale
from rescale_optimizer.utils import setup_logging

DEFAULT_CONFIGS_DIR = REPO_ROOT / "configs"
DEFAULT_OUT = REPO_ROOT / "configs" / "static_skeletons.json"


def _extract_compact_entry(
    config_name: str,
    result: OptimizationResult,
    graph,
) -> Dict[str, Any]:
    if not result.success or result.chain_result.chain is None:
        return {
            "config_name": config_name,
            "success": False,
            "message": result.message or "optimization failed",
        }

    M = graph.M
    chain = result.chain_result.chain
    skeleton = list(result.skeleton)


    source_sf = int(result.chain_result.t[0]) if result.chain_result.t else 0
    R = chain.R

    sf_at_rescale: List[int] = [source_sf]
    for r in range(1, R + 1):
        s_prev = skeleton[r - 1]
        s_curr = skeleton[r]
        path = graph.nodes_between(s_prev, s_curr)
        sf_pre = propagate_scale(sf_at_rescale[r - 1], path)
        sf_post = sf_pre - int(chain.q_bits[r - 1])
        sf_at_rescale.append(sf_post)

    rescale_index_at: Dict[int, int] = {
        skeleton[r]: r for r in range(1, R + 1)
    }

    cut_point_sf: List[Dict[str, Any]] = []
    for i in range(M + 1):
        cp = graph.cut_points[i]
        type_name = cp.node.node_type.name

        if i == 0:
            cut_point_sf.append({
                "i": i, "name": cp.node.name, "type": type_name,
                "sf": int(sf_at_rescale[0]),
            })
            continue

        if i in rescale_index_at:
            r = rescale_index_at[i]
            s_prev = skeleton[r - 1]
            path = graph.nodes_between(s_prev, i)
            sf_pre = int(propagate_scale(sf_at_rescale[r - 1], path))
            sf_post = int(sf_at_rescale[r])
            cut_point_sf.append({
                "i": i, "name": cp.node.name, "type": type_name,
                "sf_pre": sf_pre, "sf_post": sf_post,
                "drop": int(chain.q_bits[r - 1]),
            })
            continue


        r_prev = max(r for r in range(R + 1) if skeleton[r] <= i)
        s_prev = skeleton[r_prev]
        path = graph.nodes_between(s_prev, i)
        sf = int(propagate_scale(sf_at_rescale[r_prev], path))
        cut_point_sf.append({
            "i": i, "name": cp.node.name, "type": type_name,
            "sf": sf,
        })

    propagation_deltas: List[Dict[str, Any]] = []
    for node in graph.nodes:
        if node.node_type == NodeType.CTPT_MUL:
            propagation_deltas.append({
                "node_id": int(node.node_id),
                "name": node.name,
                "type": "CTPT_MUL",
                "delta": int(node.scale_delta_bits),
            })
        elif node.node_type == NodeType.CTCT_MUL:
            if node.other_ct_scale_bits is not None:
                propagation_deltas.append({
                    "node_id": int(node.node_id),
                    "name": node.name,
                    "type": "CTCT_MUL",
                    "delta": int(node.other_ct_scale_bits),
                })
            else:
                propagation_deltas.append({
                    "node_id": int(node.node_id),
                    "name": node.name,
                    "type": "CTCT_MUL",
                    "delta": "x2",
                })

    drop_order = [int(chain.q_head_bits)] + [int(b) for b in chain.q_bits] + [int(chain.q_tail_bits)]
    seal_order = [int(chain.q_head_bits)] + list(reversed([int(b) for b in chain.q_bits])) + [int(chain.q_tail_bits)]

    effective_rotations = _extract_effective_rotations(
        graph, skeleton, sf_at_rescale,
    )

    return {
        "config_name": config_name,
        "success": True,
        "skeleton": [int(x) for x in skeleton],
        "cut_point_sf": cut_point_sf,
        "propagation_deltas": propagation_deltas,
        "modulus_chain": {
            "drop_order": drop_order,
            "seal_order": seal_order,
            "total_bits": int(chain.total_bits),
        },
        "effective_rotations": effective_rotations,
    }


def _extract_effective_rotations(
    graph,
    skeleton: List[int],
    sf_at_rescale: List[int],
) -> List[Dict[str, Any]]:
    """
    A ROTATION node N with ``stage_anchor = k`` is "effective" iff c_k is
    a rescale cut point in the skeleton (k ∈ skeleton[1..R]).  In that
    case N executes immediately after the rescale at c_k (no non-rescale
    cut point in between, since cut points only sit at stage boundaries
    and the only nodes between c_k and N are scale-preserving rotations
    / pt leaves), so it runs at the **post-rescale** working scale
    ``sf_at_rescale[r]`` where ``skeleton[r] == k``.

    Rotations whose preceding cut point is the SOURCE (k = 0) or a
    non-rescale multiplication cut point are "ineffective" (they
    accumulate a higher scale) and are NOT included.

    ``sf_at_rescale`` is the per-skeleton-stage post-rescale scale array
    of length R+1 (matches ``cr.t`` in the chain-consistent view).
    """
    rescale_index_at: Dict[int, int] = {
        skeleton[r]: r for r in range(1, len(skeleton) - 1)
    }
    out: List[Dict[str, Any]] = []
    for node in graph.nodes:
        if node.node_type != NodeType.ROTATION:
            continue
        k = int(node.stage_anchor)
        if k not in rescale_index_at:
            continue
        r = rescale_index_at[k]
        out.append({
            "node_id": int(node.node_id),
            "name": node.name,
            "after_cut_point": k,
            "sf": int(sf_at_rescale[r]),
            "count": int(node.count),
        })
    return out


def _dumps_oneline(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(", ", ": "))


def _format_entry(e: Dict[str, Any]) -> str:
    """Render one entry with one-line dicts inside arrays."""
    lines: List[str] = []
    if not e.get("success"):
        lines.append("    {")
        lines.append(f'      "config_name": {_dumps_oneline(e["config_name"])},')
        lines.append('      "success": false,')
        lines.append(f'      "message": {_dumps_oneline(e.get("message", ""))}')
        lines.append("    }")
        return "\n".join(lines)

    lines.append("    {")
    lines.append(f'      "config_name": {_dumps_oneline(e["config_name"])},')
    lines.append('      "success": true,')
    lines.append(f'      "skeleton": {_dumps_oneline(e["skeleton"])},')

    lines.append('      "cut_point_sf": [')
    items = e["cut_point_sf"]
    for k, item in enumerate(items):
        suffix = "," if k < len(items) - 1 else ""
        lines.append(f'        {_dumps_oneline(item)}{suffix}')
    lines.append("      ],")

    lines.append('      "propagation_deltas": [')
    items = e["propagation_deltas"]
    for k, item in enumerate(items):
        suffix = "," if k < len(items) - 1 else ""
        lines.append(f'        {_dumps_oneline(item)}{suffix}')
    lines.append("      ],")

    mc = e["modulus_chain"]
    lines.append('      "modulus_chain": {')
    lines.append(f'        "drop_order": {_dumps_oneline(mc["drop_order"])},')
    lines.append(f'        "seal_order": {_dumps_oneline(mc["seal_order"])},')
    lines.append(f'        "total_bits": {int(mc["total_bits"])}')
    lines.append("      },")

    rots = e.get("effective_rotations", [])
    if not rots:
        lines.append('      "effective_rotations": []')
    else:
        lines.append('      "effective_rotations": [')
        for k, item in enumerate(rots):
            suffix = "," if k < len(rots) - 1 else ""
            lines.append(f'        {_dumps_oneline(item)}{suffix}')
        lines.append("      ]")
    lines.append("    }")
    return "\n".join(lines)


def _format_doc(entries: List[Dict[str, Any]],
                n_configs: int,
                n_success: int) -> str:
    return "\n".join(_iter_doc_lines(entries, n_configs, n_success)) + "\n"


def _iter_doc_lines(entries: List[Dict[str, Any]],
                    n_configs: int,
                    n_success: int):
    yield "{"
    yield '  "schema_version": 2,'
    yield '  "generated_by": "scripts/batch_run_configs.py",'
    yield f'  "n_configs": {n_configs},'
    yield f'  "n_success": {n_success},'
    yield '  "results": ['
    for k, e in enumerate(entries):
        body = _format_entry(e)
        suffix = "," if k < len(entries) - 1 else ""
        yield body + suffix
    yield "  ]"
    yield "}"


def _write_doc(f, entries: List[Dict[str, Any]],
               n_configs: int,
               n_success: int) -> None:
    for line in _iter_doc_lines(entries, n_configs, n_success):
        f.write(line)
        f.write("\n")


def _discover_configs(configs_dir: Path,
                      explicit: Optional[List[str]]) -> List[Path]:
    if explicit:
        out: List[Path] = []
        for name in explicit:
            cand = name if name.endswith(".json") else name + ".json"
            p = Path(cand)
            if not p.is_absolute():
                p = configs_dir / p
            if not p.exists():
                logging.warning("config not found: %s", p)
                continue
            out.append(p)
        return out
    names = sorted(
        entry.name
        for entry in os.scandir(configs_dir)
        if entry.is_file()
        and entry.name.endswith(".json")
        and not entry.name.startswith("static_skeletons")
    )
    return [configs_dir / name for name in names]


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Run rescale optimizer on a set of configs and dump compact unified results.",
    )
    p.add_argument("--configs-dir", default=str(DEFAULT_CONFIGS_DIR),
                   help="Directory of config JSONs")
    p.add_argument("--configs", nargs="*", default=None,
                   help="Explicit list of config names (with or without .json).")
    p.add_argument("--out", default=str(DEFAULT_OUT),
                   help="Path to unified JSON output")
    p.add_argument("--log-level", default="WARNING",
                   help="Per-config optimizer log level (default WARNING)")
    args = p.parse_args(argv)

    setup_logging(level=getattr(logging, args.log_level.upper(), logging.WARNING))

    configs_dir = Path(args.configs_dir)
    configs = _discover_configs(configs_dir, args.configs)
    if not configs:
        print(f"No configs found in {configs_dir}.", file=sys.stderr)
        return 1

    print(f"[batch] running {len(configs)} configs")

    entries: List[Dict[str, Any]] = []
    n_ok = 0
    for cfg_path in configs:
        config_name = cfg_path.stem
        t0 = time.time()
        try:
            graph, opt_config, _amp_budgets = load_graph_from_json(cfg_path)
            result = optimize_rescale(graph, opt_config)
            elapsed_ms = (time.time() - t0) * 1000.0
            entry = _extract_compact_entry(config_name, result, graph)
            entries.append(entry)
            n_ok += int(result.success)
            tag = "OK " if result.success else "FAIL"
            if result.success and result.chain_result.chain is not None:
                ch = result.chain_result.chain
                summary = f"R={ch.R} drop={[ch.q_head_bits] + list(ch.q_bits) + [ch.q_tail_bits]}"
            else:
                summary = "<no chain>"
            print(f"[batch] {tag} {config_name:<25} ({elapsed_ms:7.1f} ms)  {summary}")
        except Exception as e:
            elapsed_ms = (time.time() - t0) * 1000.0
            traceback.print_exc()
            print(f"[batch] CRASH {config_name:<25} ({elapsed_ms:7.1f} ms)  {e}")
            entries.append({
                "config_name": config_name,
                "success": False,
                "message": f"crash: {e}",
            })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        _write_doc(f, entries, n_configs=len(configs), n_success=n_ok)

    print(f"\n[batch] wrote {out_path}")
    print(f"[batch] success: {n_ok} / {len(configs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
