"""Diagnose which BLB (layer, block) configs fail invalid_chain under
Rescale_optimizer for a given action.

The runtime aggregate (``signals.invalid_chain_count == N``) tells you N
blocks failed, but not *which* ones. Final-eval reports drop that detail.
This sidecar walks the per-config optimizer outputs and renders both a
human report (stdout + Markdown) and a machine-readable JSON of every
(layer, block) status — invalid rows include the optimizer's reason
string and the cfg snippet that produced it.

Typical use:

    python scripts/blb_diagnose_invalid_blocks.py \\
        --action-config "Parting Chapter/persistent/rl/bert-base/mrpc/<slug>/stage2_noise/progress/diagnostics/best_action_vec.json" \\
        --output-dir reports/blb_opt/invalid_blocks/<slug>/

Requires torch + the local Rescale_optimizer package (same env as RL
training / Paean final-eval). For a pre-built action JSON the input
schema is the SF/K-first ``best_action_vec.json`` Paean writes (we read
its top-level ``action_vec`` field; ``slots`` is used as a human
sidekick for the output report).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cli_parse_utils import parse_int_list_text  # noqa: E402
from blb_stage2_rl.action_space import (  # noqa: E402
    avg_truncation_k_in_action,
    build_optimizer_requests,
    load_max_sfs,
    parse_config_name,
)
from blb_stage2_rl.action_io import action_vec_to_slots_list  # noqa: E402
from blb_stage2_rl.optimizer_cost import evaluate_action_for_cost  # noqa: E402
from json_utils import read_json_file, write_json_file  # noqa: E402
from rescale_optimizer_bridge import (  # noqa: E402
    InProcessInvoker,
    RescaleOptimizerBridge,
    _strip_layer_suffix,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_action_vec(action_config_path: str, num_layers: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load a Paean-style action JSON, return (action_vec, metadata)."""
    payload = read_json_file(action_config_path)
    if "action_vec" not in payload:
        raise ValueError(
            f"{action_config_path}: missing top-level 'action_vec'. "
            "This script reads the legacy flat int vector; if you only have "
            "the slots-form view, run it through Paean's slots_payload_to_action_vec first."
        )
    arr = np.asarray(payload["action_vec"], dtype=int).reshape(-1)
    meta = {k: v for k, v in payload.items() if k not in {"action_vec", "slots", "decode_rules"}}
    meta["source_path"] = str(action_config_path)
    meta["source_num_layers"] = int(payload.get("num_layers", num_layers))
    return arr, meta


def _stage1_degrees_from_meta(
        meta: Mapping[str, Any],
        num_layers: int,
        *,
        model_type: str = "bert-base",
        ) -> Tuple[List[int], List[int]]:
    """Best-effort recovery of stage-1 (gelu/softmax) per-layer degrees.

    The action JSON itself doesn't carry the stage-1 vector; we read it from
    the canonical ``glue_final_configs_best_ppo.json``. The schema is:

        {
          "bert-base": {
            "mrpc":  {"stage1": {"gelu": [..L..], "softmax": [..L..]}, ...},
            "cola":  {"stage1": {...}, ...},
            ...
          },
          "bert-large": {...},
          "gpt-2":     {...},
        }

    Falls back to all-4 ONLY if absolutely nothing parseable is found — and
    prints a loud warning so an operator can spot the mismatch in the report.
    Previously (2026-05-17 first cut) the path was wrong (``cfg[dataset]``
    instead of ``cfg[model_type][dataset]['stage1']``) and the script silently
    fell back to all-4, producing an invalid-block report that listed every
    failing block as ``block5_n4`` / ``block3_exp_n4`` even when the real
    stage-1 vector had degree 1 / 2 / 5 / 6 for many layers.
    """
    stage1_default = ([4] * num_layers, [4] * num_layers)
    src = meta.get("stage1_config_path") or "glue_final_configs_best_ppo.json"
    path = REPO_ROOT / src
    try:
        cfg = read_json_file(path)
    except Exception as exc:
        print(f"[warn] could not read stage-1 config {path}: {exc}", file=sys.stderr)
        return stage1_default

    dataset = str(meta.get("meta", {}).get("profile", "mrpc"))

    # Preferred shape: cfg[model_type][dataset]["stage1"]["gelu" / "softmax"].
    try:
        stage1 = cfg[str(model_type)][dataset]["stage1"]
        gelu = stage1["gelu"]
        softmax = stage1["softmax"]
        if len(gelu) == num_layers and len(softmax) == num_layers:
            return [int(x) for x in gelu], [int(x) for x in softmax]
    except (KeyError, TypeError):
        pass

    # Older / alternate shapes — keep as fallbacks but warn if they hit.
    entry = cfg.get(dataset, {})
    gelu = entry.get("gelu_degree") or entry.get("manual_stage1_gelu") or []
    softmax = entry.get("softmax_degree") or entry.get("manual_stage1_softmax") or []
    if gelu and softmax and len(gelu) == num_layers and len(softmax) == num_layers:
        print(
            f"[warn] read stage-1 from legacy top-level key '{dataset}' in {path}; "
            "expected schema is {model_type}.{dataset}.stage1.{gelu,softmax}",
            file=sys.stderr,
        )
        return [int(x) for x in gelu], [int(x) for x in softmax]

    print(
        f"[warn] could not locate stage-1 per-layer degrees in {path} for "
        f"model_type={model_type!r} dataset={dataset!r} -- falling back to "
        f"[4]*{num_layers}. Report's graph_keys (block3_exp_n4 / block5_n4) "
        f"WILL be wrong if real stage-1 is non-4.",
        file=sys.stderr,
    )
    return stage1_default


def _invalid_chain_reason(invalid_chain: Optional[Mapping[str, Any]]) -> str:
    """Compact one-line reason string for an invalid_chain dict."""
    if not invalid_chain:
        return "(none)"
    if not isinstance(invalid_chain, Mapping):
        return str(invalid_chain)
    bits = []
    for k in ("reason", "message", "stage", "primes_over_q_max", "primes_under_q_min"):
        if k in invalid_chain and invalid_chain[k] not in (None, "", []):
            bits.append(f"{k}={invalid_chain[k]}")
    if not bits:
        bits = [f"{k}={v}" for k, v in invalid_chain.items() if v not in (None, "", [])]
    return "; ".join(bits) if bits else json.dumps(invalid_chain, ensure_ascii=False)


def _slot_summary_for_layer_block(
        slots: Sequence[Mapping[str, Any]],
        layer: int,
        block: int,
        ) -> List[str]:
    """Return one-line strings for every slot that belongs to (layer, block)."""
    out: List[str] = []
    for s in slots:
        if int(s.get("layer", -1)) != int(layer):
            continue
        if int(s.get("block") or 0) != int(block):
            continue
        kind = str(s.get("kind", ""))
        field = str(s.get("field_name", ""))
        if kind == "K":
            v = s.get("truncation_bits")
            value = f"truncation_bits={v}"
        else:
            v = s.get("scaling_factor")
            value = "scaling_factor=off" if v is None else f"scaling_factor={v}"
        flag = "" if bool(s.get("effective", True)) else " [inactive]"
        out.append(f"{kind}.{field} {value}{flag}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Walk Rescale_optimizer per-block outputs for an action; "
                    "highlight which (layer, block) configs report invalid_chain."
    )
    p.add_argument(
        "--action-config", required=True,
        help="Path to the Paean-style action JSON (must contain 'action_vec').",
    )
    p.add_argument("--profile", default="mrpc")
    p.add_argument("--num-layers", type=int, default=12)
    p.add_argument(
        "--model-type", default="bert-base",
        help="Used to look up stage-1 degrees under cfg[model_type][dataset].",
    )
    p.add_argument(
        "--rescale-optimizer-root", default="Rescale_optimizer",
        help="Local Rescale_optimizer package root (default: ./Rescale_optimizer).",
    )
    p.add_argument(
        "--output-dir", default="",
        help="If set, write report.md + report.json under this dir.",
    )
    p.add_argument(
        "--gelu-degree", default="",
        help="Comma-separated per-layer GELU degree override. "
             "If empty, derive from glue_final_configs_best_ppo.json.",
    )
    p.add_argument(
        "--softmax-degree", default="",
        help="Comma-separated per-layer softmax degree override. "
             "If empty, derive from glue_final_configs_best_ppo.json.",
    )
    args = p.parse_args(argv)

    action, meta = _load_action_vec(args.action_config, args.num_layers)
    print(f"[load] action_vec dim={action.size}, action source={meta.get('source')!r}", flush=True)

    if args.gelu_degree and args.softmax_degree:
        gelu = parse_int_list_text(args.gelu_degree, allow_semicolon=False)
        softmax = parse_int_list_text(args.softmax_degree, allow_semicolon=False)
    else:
        gelu, softmax = _stage1_degrees_from_meta(
            meta, args.num_layers, model_type=args.model_type,
        )
    print(f"[stage1] gelu={gelu}", flush=True)
    print(f"[stage1] softmax={softmax}", flush=True)

    bridge = RescaleOptimizerBridge(
        invoker=InProcessInvoker.from_profile(
            rescale_optimizer_root=args.rescale_optimizer_root,
            profile=args.profile,
        )
    )

    eval_result = evaluate_action_for_cost(
        action,
        profile=args.profile,
        num_layers=args.num_layers,
        max_sfs=load_max_sfs(args.profile),
        rescale_bridge=bridge,
        gelu_degree=gelu,
        attn_degree=softmax,
    )

    # Build a decoded slot view (for the human-side per-block annotations)
    slots_view: List[Mapping[str, Any]] = []
    try:
        slots_view = list(action_vec_to_slots_list(
            action,
            max_sfs=load_max_sfs(args.profile),
            num_layers=args.num_layers,
            gelu_degree=gelu,
            attn_degree=softmax,
            profile=args.profile,
        ))
    except Exception as exc:
        print(f"[warn] action_vec_to_slots_list failed: {exc}", file=sys.stderr)

    rows: List[Dict[str, Any]] = []
    invalid_rows: List[Dict[str, Any]] = []
    for cn in sorted(
        eval_result.outputs.keys(),
        key=lambda x: (
            (lambda parsed: (parsed[2], parsed[0]))(parse_config_name(x))
        ),
    ):
        out = eval_result.outputs[cn]
        graph_key, layer_idx = _strip_layer_suffix(cn)
        block_idx, _profile, _ = parse_config_name(cn)
        invalid = out.invalid_chain is not None
        slot_lines = _slot_summary_for_layer_block(slots_view, int(layer_idx), int(block_idx))
        row = {
            "config_name": cn,
            "graph_key": graph_key,
            "layer": int(layer_idx),
            "block": int(block_idx),
            "valid": (not invalid),
            "fusion_count": int(out.fusion_count),
            "total_bits": int(out.total_bits),
            "invalid_chain_reason": _invalid_chain_reason(out.invalid_chain),
            "slot_summary": slot_lines,
        }
        rows.append(row)
        if invalid:
            invalid_rows.append(row)

    # ---- console summary ----
    n_total = len(rows)
    n_invalid = len(invalid_rows)
    avg_k = float(avg_truncation_k_in_action(action, args.num_layers))
    print()
    print(f"=== Rescale_optimizer evaluation summary ===")
    print(f"  profile={args.profile}  num_layers={args.num_layers}")
    print(f"  configs evaluated: {n_total}")
    print(f"  valid:             {n_total - n_invalid}")
    print(f"  invalid:           {n_invalid}")
    print(f"  any_invalid:       {bool(eval_result.signals.any_invalid)}")
    print(f"  total_bits_sum:    {int(eval_result.signals.total_bits_sum)}")
    print(f"  total_fusion:      {int(eval_result.signals.total_fusion_count)}")
    print(f"  avg_k in action:   {avg_k:.3f}")
    print()
    if n_invalid:
        print(f"=== Invalid blocks ({n_invalid}) ===")
        # Group by graph_key for quick pattern spotting
        by_graph = Counter(r["graph_key"] for r in invalid_rows)
        for gk, n in sorted(by_graph.items(), key=lambda kv: -kv[1]):
            print(f"  · {gk}: {n} layer(s)")
        print()
        for row in invalid_rows:
            tag = f"L{row['layer']:02d}-B{row['block']}"
            print(f"--- {tag} ({row['config_name']}) ---")
            print(f"    graph_key:      {row['graph_key']}")
            print(f"    invalid reason: {row['invalid_chain_reason']}")
            print(f"    cost signals:   total_bits={row['total_bits']}, fusion_count={row['fusion_count']}")
            print(f"    slots ({len(row['slot_summary'])}):")
            for s in row["slot_summary"]:
                print(f"      · {s}")
            print()
    else:
        print("All configs valid — no invalid_chain reported.")
        print()

    # ---- optional file outputs ----
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json_file(
            out_dir / "report.json",
            {
                "action_config": args.action_config,
                "profile": args.profile,
                "num_layers": args.num_layers,
                "summary": {
                    "n_total": n_total,
                    "n_invalid": n_invalid,
                    "any_invalid": bool(eval_result.signals.any_invalid),
                    "total_bits_sum": int(eval_result.signals.total_bits_sum),
                    "total_fusion": int(eval_result.signals.total_fusion_count),
                    "avg_k": float(avg_k),
                },
                "rows": rows,
                "invalid_rows": invalid_rows,
            },
        )
        md = ["# BLB invalid-block diagnosis", ""]
        md.append(f"- action_config: `{args.action_config}`")
        md.append(f"- profile: `{args.profile}` · num_layers: `{args.num_layers}`")
        md.append("")
        md.append("## Summary")
        md.append("")
        md.append(f"- configs evaluated: **{n_total}**")
        md.append(f"- valid: **{n_total - n_invalid}**")
        md.append(f"- invalid: **{n_invalid}**")
        md.append(f"- any_invalid: **{bool(eval_result.signals.any_invalid)}**")
        md.append(f"- total_bits_sum: **{int(eval_result.signals.total_bits_sum)}**")
        md.append(f"- total_fusion_count: **{int(eval_result.signals.total_fusion_count)}**")
        md.append(f"- avg_k in action: **{avg_k:.3f}**")
        md.append("")
        if invalid_rows:
            md.append("## Invalid blocks")
            md.append("")
            md.append("| (L, B) | graph_key | total_bits | fusion | reason |")
            md.append("|:-------|:----------|-----------:|-------:|:-------|")
            for r in invalid_rows:
                tag = f"L{r['layer']:02d}-B{r['block']}"
                md.append(
                    f"| `{tag}` | `{r['graph_key']}` | {r['total_bits']} | "
                    f"{r['fusion_count']} | {r['invalid_chain_reason']} |"
                )
            md.append("")
            md.append("## Slot configs of invalid blocks")
            md.append("")
            for r in invalid_rows:
                tag = f"L{r['layer']:02d}-B{r['block']}"
                md.append(f"### {tag} (`{r['graph_key']}`)")
                md.append("")
                md.append(f"- invalid reason: `{r['invalid_chain_reason']}`")
                md.append("- slots:")
                for s in r["slot_summary"]:
                    md.append(f"  - `{s}`")
                md.append("")
        else:
            md.append("All configs valid — no invalid_chain reported.")
            md.append("")
        (out_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
        print(f"[write] {out_dir / 'report.json'}")
        print(f"[write] {out_dir / 'report.md'}")

    return 0 if not n_invalid else 0  # Non-error exit even on invalids — this is a diagnostic.


if __name__ == "__main__":
    raise SystemExit(main())
