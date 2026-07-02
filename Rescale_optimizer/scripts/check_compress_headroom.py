"""
check_compress_headroom.py
==========================

Diagnose whether ``compress_headroom`` (Alg 7) leaves the chain
consistent with the post-compress ``t``.

The current implementation in ``rescale_optimizer/modulus_chain.py``
uses the linear-propagation approximation

    d_r' = q_r - (c_{r-1} - c_r)

to update prime sizes after lowering ``t`` by ``c``.  This is correct
only when every node on the path between two skeleton points is *not*
a symmetric ``CTCT_MUL`` (which propagates ``s -> 2s``).  When the
path contains ``k`` symmetric CTCTs, the actual drop after lowering
``t_{r-1}`` by ``c_{r-1}`` and ``t_r`` by ``c_r`` is

    d_r_natural = propagate_scale(t_{r-1} - c_{r-1}, path) - (t_r - c_r)
                = q_r - (2^k * c_{r-1} - c_r)

which is **smaller** than what the linear formula yields by
``(2^k - 1) * c_{r-1}`` bits.  Result: the chain prime ``q_r'`` is
bigger than necessary by that many bits — pure modulus-chain slack
(wasted total bits, smaller available rescale margin downstream).

This script:

  1. Runs ``optimize_rescale`` on every config in --configs-dir.
  2. After ``optimize_rescale`` produces ``(skeleton, chain, t)``,
     it walks the skeleton forward using ``propagate_scale`` and the
     **chain-consistent** rule

         t_post_chain[r] = propagate_scale(t_post_chain[r-1], path) - q_bits[r-1]

     starting from ``t_post_chain[0] = t[0]`` (the source scale).
  3. Compares to the optimizer's stored ``t[r]``.  If they diverge,
     reports the slack at each rescale stage and identifies the
     symmetric CTCTs in the predecessor path that caused it.

Output table per config:

    config    | r | s_r | path summary                        | t[r] | t_chain[r] | slack
    ----------|---|-----|-------------------------------------|------|------------|------
    block1    | 1 | 2   | CTPT(20), CTPT(20)                  |  34  |     34     |  0
    block1    | 2 | 4   | CTCT(x2), CTPT(20)                  |  34  |     32     |  2

A non-zero slack indicates ``compress_headroom`` was over-conservative
on this stage.  A negative slack would be a real correctness bug
(would mean the chain prime is too small for the natural drop).

Usage::

    python scripts/check_compress_headroom.py
    python scripts/check_compress_headroom.py --configs block1_wnli block4
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rescale_optimizer import (
    load_graph_from_json,
    optimize_rescale,
)
from rescale_optimizer.graph import NodeType, propagate_scale
from rescale_optimizer.utils import setup_logging


def _path_summary(path) -> str:
    """Render a short human-readable string for a node path."""
    parts: List[str] = []
    for n in path:
        if n.node_type == NodeType.CTCT_MUL:
            if n.other_ct_scale_bits is not None:
                parts.append(f"CTCT(+{n.other_ct_scale_bits})")
            else:
                parts.append("CTCT(x2)")
        elif n.node_type == NodeType.CTPT_MUL:
            parts.append(f"CTPT(+{n.scale_delta_bits})")
        # ROTATION / PT_OP / PT contribute 0 to scale; skip in summary
    return ", ".join(parts) if parts else "<no muls>"


def _count_symmetric_ctct(path) -> int:
    return sum(
        1 for n in path
        if n.node_type == NodeType.CTCT_MUL and n.other_ct_scale_bits is None
    )


def diagnose(config_path: Path) -> Optional[Dict[str, Any]]:
    graph, opt_config, _ = load_graph_from_json(config_path)
    result = optimize_rescale(graph, opt_config)
    if not result.success or result.chain_result.chain is None:
        return None

    chain = result.chain_result.chain
    t_optimizer = list(result.chain_result.t)
    skeleton = list(result.skeleton)
    R = chain.R

    # Walk forward chain-consistently.
    t_chain: List[int] = [int(t_optimizer[0])]
    rows: List[Dict[str, Any]] = []
    for r in range(1, R + 1):
        s_prev = skeleton[r - 1]
        s_curr = skeleton[r]
        path = graph.nodes_between(s_prev, s_curr)
        sf_pre = propagate_scale(t_chain[r - 1], path)
        sf_post_chain = sf_pre - int(chain.q_bits[r - 1])
        t_chain.append(sf_post_chain)

        slack = int(t_optimizer[r]) - sf_post_chain  # optimizer claims higher
        rows.append({
            "r": r,
            "s_r": int(s_curr),
            "path_summary": _path_summary(path),
            "n_sym_ctct": _count_symmetric_ctct(path),
            "t_optimizer": int(t_optimizer[r]),
            "t_chain": sf_post_chain,
            "slack_bits": slack,
            "q_bits": int(chain.q_bits[r - 1]),
            "sf_pre": int(sf_pre),
        })

    return {
        "config": config_path.stem,
        "skeleton": skeleton,
        "t_optimizer": [int(x) for x in t_optimizer],
        "t_chain": t_chain,
        "rows": rows,
    }


def _discover_configs(configs_dir: Path, explicit: Optional[List[str]]) -> List[Path]:
    if explicit:
        configs = []
        for n in explicit:
            cand = n if n.endswith(".json") else n + ".json"
            p_ = configs_dir / cand if not Path(cand).is_absolute() else Path(cand)
            if p_.exists():
                configs.append(p_)
        return configs

    names = sorted(
        entry.name
        for entry in os.scandir(configs_dir)
        if entry.is_file()
        and entry.name.endswith(".json")
        and entry.name != "static_skeletons.json"
    )
    return [configs_dir / name for name in names]


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--configs-dir", default=str(REPO_ROOT / "configs"))
    p.add_argument("--configs", nargs="*", default=None,
                   help="Specific configs (with or without .json)")
    args = p.parse_args(argv)

    setup_logging(level=logging.WARNING)

    configs_dir = Path(args.configs_dir)
    configs = _discover_configs(configs_dir, args.configs)

    print()
    print(f"{'config':<22} | {'r':>2} | {'s_r':>3} | {'path summary':<35} "
          f"| {'q':>3} | {'sf_pre':>6} | {'t_opt':>5} | {'t_chain':>7} | {'slack':>5}")
    print("-" * 110)

    total_slack_configs = 0
    total_slack_stages = 0
    total_slack_bits = 0
    suspect_negative = 0

    for cfg in configs:
        d = diagnose(cfg)
        if d is None:
            print(f"{cfg.stem:<22} |  -  |  -  | <opt failed>")
            continue
        any_slack = False
        for row in d["rows"]:
            slack_marker = ""
            if row["slack_bits"] > 0:
                slack_marker = "  <-- slack"
                any_slack = True
                total_slack_stages += 1
                total_slack_bits += row["slack_bits"]
            elif row["slack_bits"] < 0:
                slack_marker = "  <<< NEGATIVE (correctness bug?)"
                suspect_negative += 1
            print(f"{d['config']:<22} | {row['r']:>2} | {row['s_r']:>3} | "
                  f"{row['path_summary']:<35} | {row['q_bits']:>3} | "
                  f"{row['sf_pre']:>6} | {row['t_optimizer']:>5} | "
                  f"{row['t_chain']:>7} | {row['slack_bits']:>5}"
                  f"{slack_marker}")
        if any_slack:
            total_slack_configs += 1

    print("-" * 110)
    print()
    print(f"Configs with positive slack    : {total_slack_configs} / {len(configs)}")
    print(f"Stages with positive slack     : {total_slack_stages}")
    print(f"Total wasted modulus bits      : {total_slack_bits}")
    print(f"Negative-slack stages          : {suspect_negative}  "
          f"(should be 0 for correctness)")
    print()

    if total_slack_bits > 0:
        print("Diagnosis: compress_headroom uses a linear-propagation assumption")
        print("           which over-estimates required prime sizes whenever the")
        print("           path between two skeleton points contains a SYMMETRIC")
        print("           CTCT_MUL (s -> 2s).  See script docstring for fix.")
    if suspect_negative > 0:
        print("WARNING: negative slack found — chain prime smaller than natural drop.")
        print("         This would be a real correctness bug (post-rescale t > stored t).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
