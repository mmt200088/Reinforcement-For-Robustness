#!/usr/bin/env python3
"""Render the paired MRPC fixed-Block2/Block5 fusion comparison."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


CONTROL = "all_fusion0"
TREATMENT = "block2_block5_all_layers_fusionmax"


def _find_group(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    for group in payload.get("group_results") or []:
        if str(group.get("name")) == name:
            return group
    raise ValueError(f"missing required group {name!r}")


def _metric(group: Mapping[str, Any], key: str) -> float:
    return float((group.get("metrics") or {})[key])


def _relevant_steps(group: Mapping[str, Any]) -> list[Dict[str, Any]]:
    return [
        dict(step)
        for step in group.get("step_records") or []
        if int(step.get("block_idx", -1)) in (2, 4, 5)
    ]


def _block_totals(group: Mapping[str, Any]) -> Dict[str, int]:
    raw = group.get("fusion_by_block") or {}
    return {str(block): int(raw.get(str(block), raw.get(block, 0)) or 0) for block in (2, 4, 5)}


def _group_summary(group: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "name": str(group.get("name")),
        "loss": _metric(group, "loss_mean"),
        "loss_std": _metric(group, "loss_std"),
        "accuracy": _metric(group, "metric1_mean"),
        "accuracy_std": _metric(group, "metric1_std"),
        "weighted_f1": _metric(group, "metric2_mean"),
        "weighted_f1_std": _metric(group, "metric2_std"),
        "fusion_total": int(group.get("fusion_total", 0) or 0),
        "fusion_by_block": _block_totals(group),
        "k_distribution": dict(group.get("k_distribution") or {}),
        "step_records": _relevant_steps(group),
    }


def _all_steps_valid(*groups: Mapping[str, Any]) -> bool:
    steps = [step for group in groups for step in group.get("step_records") or []]
    return bool(steps) and all(bool(step.get("valid")) for step in steps)


def _k_is_13(*groups: Mapping[str, Any]) -> bool:
    steps = [step for group in groups for step in group.get("step_records") or []]
    return bool(steps) and all(int(step.get("k_value", -1)) == 13 for step in steps)


def _fusion_pattern(group: Mapping[str, Any], expected: Mapping[int, int]) -> bool:
    steps = _relevant_steps(group)
    if not steps:
        return False
    for step in steps:
        block = int(step["block_idx"])
        fusion = int(step.get("fusion_count_replan", 0) or 0)
        if fusion != int(expected[block]):
            return False
    return True


def _boost_pattern(group: Mapping[str, Any]) -> bool:
    steps = _relevant_steps(group)
    return bool(steps) and all(
        bool(step.get("boosted")) == (int(step["block_idx"]) in (2, 5))
        for step in steps
    )


def _pair(label: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
    control_raw = _find_group(payload, CONTROL)
    treatment_raw = _find_group(payload, TREATMENT)
    control = _group_summary(control_raw)
    treatment = _group_summary(treatment_raw)
    deltas = {
        "loss": treatment["loss"] - control["loss"],
        "accuracy": treatment["accuracy"] - control["accuracy"],
        "weighted_f1": treatment["weighted_f1"] - control["weighted_f1"],
    }
    percent_deltas = {
        key: (100.0 * value / control[key] if control[key] != 0 else None)
        for key, value in deltas.items()
    }
    return {
        "label": label,
        "stage1_gelu": [int(x) for x in payload.get("stage1_gelu") or []],
        "stage1_softmax": [int(x) for x in payload.get("stage1_softmax") or []],
        "repeat": int(payload.get("repeat", 0) or 0),
        "probe_size": int(payload.get("probe_size", 0) or 0),
        "install_path": str(payload.get("install_path", "")),
        "control": control,
        "treatment": treatment,
        "deltas": deltas,
        "percent_deltas": percent_deltas,
        "gates": {
            "all_steps_valid": _all_steps_valid(control_raw, treatment_raw),
            "k_is_13_everywhere": _k_is_13(control_raw, treatment_raw),
            "control_is_all_zero": _fusion_pattern(control_raw, {2: 0, 4: 0, 5: 0}),
            "treatment_is_b2_b5_one_b4_zero": _fusion_pattern(
                treatment_raw, {2: 1, 4: 0, 5: 1},
            ),
            "treatment_boost_matches_b2_b5": _boost_pattern(treatment_raw),
        },
    }


def build_summary(
        *,
        stage1_best: Mapping[str, Any],
        gelu4: Mapping[str, Any],
        source_commit: str,
        ) -> Dict[str, Any]:
    pairs = [
        _pair("Stage-1 best GELU", stage1_best),
        _pair("GELU degree 4", gelu4),
    ]
    return {
        "schema_version": "fixed_b2b5_fusion_comparison_v1",
        "source_commit": str(source_commit),
        "control_definition": "Block2=0, Block4=0, Block5=0; K=13",
        "treatment_definition": "Block2=1, Block4=0, Block5=1; K=13",
        "pairs": pairs,
        "all_gates_pass": all(all(pair["gates"].values()) for pair in pairs),
    }


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    head = "".join(f"<th>{html.escape(str(value))}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(_fmt(value))}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _decision_rows(group: Mapping[str, Any]) -> list[list[Any]]:
    return [
        [
            f"Layer {int(step['layer_idx'])}",
            f"Block {int(step['block_idx'])}",
            str(step.get("graph_key", "")),
            f"map option {int(step.get('map_option_id', step.get('option_id', 0)))}",
            int(step.get("fusion_count_replan", 0) or 0),
            "boosted" if step.get("boosted") else "plain",
            f"K={int(step.get('k_value', -1))}",
            "valid" if step.get("valid") else "INVALID",
        ]
        for step in group["step_records"]
    ]


def render_html(summary: Mapping[str, Any]) -> str:
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Stage-2 fixed Block2/Block5 fusion comparison</title>",
        "<style>body{font:14px/1.45 system-ui,sans-serif;margin:28px;color:#1f2937}"
        "h1,h2,h3{color:#111827}code{background:#f3f4f6;padding:2px 4px}"
        "table{border-collapse:collapse;width:100%;margin:12px 0 26px}"
        "th,td{border:1px solid #d1d5db;padding:6px 8px;text-align:left}"
        "th{background:#f3f4f6}.pass{color:#067647;font-weight:700}"
        ".fail{color:#b42318;font-weight:700}.meta{color:#475467}</style></head><body>",
        "<h1>Stage-2 fixed Block2/Block5 fusion comparison</h1>",
        f"<p class='meta'>Source commit: <code>{html.escape(str(summary['source_commit']))}</code></p>",
        f"<p>Control: {html.escape(str(summary['control_definition']))}<br>"
        f"Treatment: {html.escape(str(summary['treatment_definition']))}</p>",
        f"<p class={'pass' if summary['all_gates_pass'] else 'fail'}>"
        f"Overall action/install gates: {_fmt(bool(summary['all_gates_pass']))}</p>",
    ]

    metric_rows = []
    for pair in summary["pairs"]:
        c = pair["control"]
        t = pair["treatment"]
        d = pair["deltas"]
        metric_rows.append([
            pair["label"], c["loss"], t["loss"], d["loss"],
            c["accuracy"], t["accuracy"], d["accuracy"],
            c["weighted_f1"], t["weighted_f1"], d["weighted_f1"],
        ])
    parts.append("<h2>Metric comparison</h2>")
    parts.append(_table(
        ["GELU profile", "control loss", "treatment loss", "delta loss",
         "control accuracy", "treatment accuracy", "delta accuracy",
         "control weighted F1", "treatment weighted F1", "delta weighted F1"],
        metric_rows,
    ))

    for pair in summary["pairs"]:
        parts.extend([
            f"<h2>{html.escape(str(pair['label']))}</h2>",
            f"<p>GELU: <code>{html.escape(json.dumps(pair['stage1_gelu']))}</code><br>"
            f"Softmax: <code>{html.escape(json.dumps(pair['stage1_softmax']))}</code><br>"
            f"Full validation probe size: {pair['probe_size']}; repeated noise trials: {pair['repeat']}<br>"
            f"Install path: <code>{html.escape(pair['install_path'])}</code></p>",
            "<h3>Protocol gates</h3>",
            _table(["gate", "result"], [[key, value] for key, value in pair["gates"].items()]),
        ])
        for group_key, title in (("control", "Control decisions"), ("treatment", "Treatment decisions")):
            group = pair[group_key]
            parts.extend([
                f"<h3>{title}</h3>",
                f"<p>Fusion total={group['fusion_total']}; by block="
                f"<code>{html.escape(json.dumps(group['fusion_by_block'], sort_keys=True))}</code>; "
                f"K distribution=<code>{html.escape(json.dumps(group['k_distribution'], sort_keys=True))}</code></p>",
                _table(
                    ["layer", "block", "graph", "real map option", "fusion count", "precision", "truncation", "status"],
                    _decision_rows(group),
                ),
            ])
    parts.append("</body></html>")
    return "\n".join(parts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-best-json", required=True)
    parser.add_argument("--gelu4-json", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-html", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stage1_best = json.loads(Path(args.stage1_best_json).read_text(encoding="utf-8"))
    gelu4 = json.loads(Path(args.gelu4_json).read_text(encoding="utf-8"))
    summary = build_summary(
        stage1_best=stage1_best,
        gelu4=gelu4,
        source_commit=args.source_commit,
    )
    output_json = Path(args.output_json)
    output_html = Path(args.output_html)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_html.write_text(render_html(summary), encoding="utf-8")
    print(f"report={output_html} summary={output_json} gates={summary['all_gates_pass']}")
    return 0 if summary["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
