#!/usr/bin/env python3
"""Render the exact model-installed Stage-2 SF/K configuration from runtime audits."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


BLOCKS = ("block1", "block2", "block3", "block4", "block5")
LAYERS = tuple(range(12))
BLOCK3_SLOTS = (
    "x_fresh",
    "inv_2n_encode",
    "x_inv_2n_result_rescale",
    "square_rescales[0]",
    "square_rescales[1]",
    "square_rescales[2]",
    "square_rescales[3]",
    "square_rescales[4]",
    "square_rescales[5]",
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def infer_kind(slot: str, active_value: Optional[Mapping[str, Any]]) -> str:
    if active_value is not None and active_value.get("distribution"):
        return str(active_value["distribution"])
    if "fresh" in slot:
        return "fresh"
    if "encode" in slot:
        return "encoding"
    if "rescale" in slot:
        return "rescale"
    return "noise"


def flattened_noise_slots(cfg: Mapping[str, Any]) -> Dict[str, Optional[Mapping[str, Any]]]:
    out: Dict[str, Optional[Mapping[str, Any]]] = {}
    for name, value in cfg.items():
        if name.startswith("rotation_after_") or name in {
            "degree",
            "gelu_degree",
            "output_truncation_k",
            "output_truncation_mode",
        }:
            continue
        if isinstance(value, Mapping) and "scaling_factor" in value:
            out[str(name)] = value
            continue
        if isinstance(value, list):
            for index, item in enumerate(value):
                key = f"{name}[{index}]"
                out[key] = item if isinstance(item, Mapping) else None
            continue
        if value is None and any(token in name for token in ("fresh", "encode", "rescale")):
            out[str(name)] = None
    return out


def cell_for_point(point: Optional[Mapping[str, Any]], *, layer_installed: bool) -> str:
    if not layer_installed:
        return '<span class="off">not installed</span>'
    if point is None:
        return '<span class="none">None</span>'
    sf = html.escape(str(point.get("scaling_factor")))
    return f'<span class="sf">{sf}</span>'


def installed_k(config: Mapping[str, Any], block: str, layer: int) -> Optional[int]:
    cfg = config.get(block, {}).get(str(layer))
    if not isinstance(cfg, Mapping):
        return None
    value = cfg.get("output_truncation_k")
    return None if value is None else int(value)


def fusion_by_position(fixed_action: Mapping[str, Any]) -> Dict[Tuple[int, int], int]:
    result: Dict[Tuple[int, int], int] = {}
    for row in fixed_action["group"]["choices_by_step"]:
        result[(int(row["layer"]), int(row["block"]))] = int(row["fusion_count"])
    return result


def fmt_float(value: Any, digits: int = 6) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def render(
    *,
    audit_a: Mapping[str, Any],
    audit_b: Mapping[str, Any],
    selected: Mapping[str, Any],
    fixed_action: Mapping[str, Any],
    snapshot: Mapping[str, Any],
) -> str:
    if audit_a["config"] != audit_b["config"]:
        raise ValueError("the two runtime install audits disagree on installed config")
    if audit_a["installed_layers"] != audit_b["installed_layers"]:
        raise ValueError("the two runtime install audits disagree on installed layers")
    for audit in (audit_a, audit_b):
        expected = str(audit["expected_config_sha256"])
        if str(audit["supplied_to_bridge_sha256"]) != expected:
            raise ValueError("bridge-supplied config hash does not match decoded config")
        if str(audit["installed_in_handler_sha256"]) != expected:
            raise ValueError("handler-installed config hash does not match decoded config")

    config = audit_a["config"]
    fusion = fusion_by_position(fixed_action)
    layerwise = {int(row["layer"]): row for row in snapshot["layerwise_configuration"]}
    selected_metrics = snapshot["selected_candidate"]["metrics"]
    baseline_metrics = snapshot["baseline_f4"]["pooled"]
    actual_k_values = [
        installed_k(config, block, layer)
        for block in ("block1", "block2", "block4", "block5")
        for layer in LAYERS
        if installed_k(config, block, layer) is not None
    ]
    actual_removed_k_bits = sum(13 - value for value in actual_k_values)

    k_rows = []
    for layer in LAYERS:
        cells = []
        for block_idx, block in enumerate(BLOCKS, 1):
            actual = installed_k(config, block, layer)
            if actual is None:
                actual_text = '<span class="off">not installed</span>'
            else:
                actual_text = f'<span class="k">{actual}</span>'
            if block == "block3":
                decoded = layerwise[layer].get("block3_k")
                actual_text += f'<small>legacy vector: {html.escape(str(decoded))}</small>'
            cells.append(f"<td>{actual_text}</td>")
        k_rows.append(f"<tr><th>L{layer:02d}</th>{''.join(cells)}</tr>")

    fusion_rows = []
    for layer in LAYERS:
        values = []
        for block_idx in range(1, 6):
            value = fusion.get((layer, block_idx))
            values.append(
                '<td><span class="off">not an installed fusion action</span></td>'
                if value is None
                else f'<td><span class="fusion">{value}</span></td>'
            )
        fusion_rows.append(f"<tr><th>L{layer:02d}</th>{''.join(values)}</tr>")

    slot_sections = []
    for block in BLOCKS:
        layer_cfgs = config.get(block, {})
        slots = set(BLOCK3_SLOTS if block == "block3" else ())
        per_layer: Dict[int, Dict[str, Optional[Mapping[str, Any]]]] = {}
        for layer in LAYERS:
            cfg = layer_cfgs.get(str(layer)) if isinstance(layer_cfgs, Mapping) else None
            flat = flattened_noise_slots(cfg) if isinstance(cfg, Mapping) else {}
            per_layer[layer] = flat
            slots.update(flat)
        body = []
        for slot in sorted(slots):
            active_example = next(
                (per_layer[layer].get(slot) for layer in LAYERS if per_layer[layer].get(slot) is not None),
                None,
            )
            cells = []
            for layer in LAYERS:
                cfg_exists = isinstance(layer_cfgs, Mapping) and str(layer) in layer_cfgs
                cells.append(
                    f"<td>{cell_for_point(per_layer[layer].get(slot), layer_installed=cfg_exists)}</td>"
                )
            body.append(
                "<tr>"
                f"<th><code>{html.escape(slot)}</code></th>"
                f"<td>{html.escape(infer_kind(slot, active_example))}</td>"
                f"{''.join(cells)}"
                "</tr>"
            )
        slot_sections.append(
            f"<section><h2>{block.upper()} actual SF</h2>"
            '<div class="table-wrap"><table class="wide"><thead><tr>'
            '<th>slot</th><th>type</th>'
            + "".join(f"<th>L{layer:02d}</th>" for layer in LAYERS)
            + "</tr></thead><tbody>"
            + "".join(body)
            + "</tbody></table></div></section>"
        )

    metric_rows = []
    for label, key, lower_better in (
        ("Loss", "loss", True),
        ("Accuracy (m1)", "metric1", False),
        ("Weighted F1 (m2)", "metric2", False),
    ):
        metric_rows.append(
            "<tr>"
            f"<th>{label}</th>"
            f"<td>{fmt_float(baseline_metrics.get(key + '_mean'))}</td>"
            f"<td>{fmt_float(baseline_metrics.get(key + '_std'))}</td>"
            f"<td>{fmt_float(selected_metrics.get(key + '_mean'))}</td>"
            f"<td>{fmt_float(selected_metrics.get(key + '_std'))}</td>"
            "</tr>"
        )

    training = snapshot["training"]
    stopped = snapshot["stopped"]
    config_hash = str(audit_a["installed_in_handler_sha256"])
    source_commit = str(snapshot["source_commit"])
    runtime_commit = str(selected["export_runtime_commit"])
    warning = (
        "Block3 is absent from installed_layers in both inference seeds. The exact runtime "
        "bridge intentionally ignores block3_cfgs, so Block3 SF and truncation K were not "
        "applied to the model. Values labeled legacy vector are decoded bookkeeping only. "
        "The RL summary counted 59 K positions and 101 removed bits, while the handler "
        f"actually installed {len(actual_k_values)} K positions and {actual_removed_k_bits} "
        "removed bits."
    )

    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage-2 MRPC actual installed SF and K</title>
<style>
:root{{--ink:#17202a;--muted:#667085;--line:#d0d5dd;--panel:#f7f8fa;--accent:#176b5b;--warn:#9a3412;--warn-bg:#fff7ed;--none:#6b7280}}
*{{box-sizing:border-box}} body{{margin:0;background:#fff;color:var(--ink);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1500px;margin:0 auto;padding:32px 24px 64px}} h1{{font-size:28px;margin:0 0 8px;letter-spacing:0}} h2{{font-size:18px;margin:32px 0 10px;letter-spacing:0}}
p{{max-width:1000px}} code{{font:12px/1.35 ui-monospace,SFMono-Regular,Menlo,monospace}}
.meta{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:8px;margin:18px 0}}
.meta div{{border:1px solid var(--line);padding:10px;background:var(--panel);border-radius:4px}} .meta b{{display:block;font-size:12px;color:var(--muted);margin-bottom:4px}}
.warning{{border-left:4px solid var(--warn);background:var(--warn-bg);padding:12px 14px;margin:18px 0;color:#7c2d12}}
.table-wrap{{overflow:auto;border:1px solid var(--line)}} table{{border-collapse:collapse;width:100%}} th,td{{border-bottom:1px solid var(--line);border-right:1px solid var(--line);padding:7px 8px;text-align:center;white-space:nowrap}} th{{background:#f2f4f7;font-weight:600}} tbody th{{text-align:left}} tr:last-child td,tr:last-child th{{border-bottom:0}} td:last-child,th:last-child{{border-right:0}}
.wide{{min-width:1250px}} .sf,.k,.fusion{{font-weight:700;color:var(--accent)}} .none{{color:var(--none)}} .off{{color:var(--warn);font-size:12px}} small{{display:block;color:var(--muted);font-size:10px}}
.legend{{display:flex;gap:18px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin:8px 0 14px}}
.hash{{word-break:break-all;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}
@media(max-width:700px){{main{{padding:20px 12px 48px}}h1{{font-size:22px}}}}
</style></head><body><main>
<h1>BERT-base MRPC Stage-2 actual installed SF and truncation K</h1>
<p>Source of truth: two independent GLUE inference runs captured the cfg supplied to <code>BLBNoiseRLBridge.apply</code> and the cfg objects installed in the model handler. The three hashes matched in each run, and both seeds produced the same installed config.</p>
<div class="meta">
<div><b>Training source commit</b><span class="hash">{html.escape(source_commit)}</span></div>
<div><b>Submission runtime commit</b><span class="hash">{html.escape(runtime_commit)}</span></div>
<div><b>Checkpoint</b>episode {int(selected['episode'])}, PPO update {int(selected['ppo_update_count'])}</div>
<div><b>Candidate key</b><span class="hash">{html.escape(str(audit_a['candidate_key']))}</span></div>
<div><b>Installed config SHA-256</b><span class="hash">{html.escape(config_hash)}</span></div>
<div><b>Repeated installation seeds</b>{int(audit_a['seed'])}, {int(audit_b['seed'])}</div>
<div><b>Training state</b>converged={html.escape(str(training.get('converged')))}, stopped={html.escape(str(stopped.get('resumable')))}</div>
<div><b>Stage-1 fixed configuration</b>GELU=[4] x 12; Softmax=[6] x 12</div>
<div><b>Actual installed K positions</b>{len(actual_k_values)} positions; {actual_removed_k_bits} bits removed from K=13</div>
<div><b>RL summary K accounting</b>{int(snapshot['resource_summary']['valid_k_slots'])} positions; {int(snapshot['resource_summary']['removed_k_bits'])} bits removed</div>
</div>
<div class="warning"><strong>Runtime finding:</strong> {html.escape(warning)}</div>
<h2>Strict selected candidate metrics</h2>
<div class="table-wrap"><table><thead><tr><th>metric</th><th>baseline mean</th><th>baseline std</th><th>selected mean</th><th>selected std</th></tr></thead><tbody>{''.join(metric_rows)}</tbody></table></div>
<h2>Actual per-block truncation K</h2>
<p>K is read from the cfg objects that were actually installed in the handler. Block3's legacy-vector value is shown only to expose the install mismatch.</p>
<div class="table-wrap"><table><thead><tr><th>layer</th>{''.join(f'<th>B{i}</th>' for i in range(1,6))}</tr></thead><tbody>{''.join(k_rows)}</tbody></table></div>
<h2>Fusion count represented by the installed fixed action</h2>
<div class="table-wrap"><table><thead><tr><th>layer</th>{''.join(f'<th>B{i}</th>' for i in range(1,6))}</tr></thead><tbody>{''.join(fusion_rows)}</tbody></table></div>
<div class="legend"><span><b class="sf">number</b>: installed SF/K/fusion</span><span><b class="none">None</b>: cfg slot exists but has no noise point, including fused-away rescales</span><span><b class="off">not installed</b>: the layer/block cfg was not attached to the model</span></div>
{''.join(slot_sections)}
<h2>Evidence contract</h2>
<ul>
<li>Two runtime configs identical: <code>{canonical_hash(audit_a['config'])}</code></li>
<li><code>expected_config_sha256 == supplied_to_bridge_sha256 == installed_in_handler_sha256</code>: <code>{html.escape(config_hash)}</code></li>
<li>Action config SHA-256: <code>{html.escape(str(audit_a['action_config_sha256']))}</code></li>
<li>Action vector SHA-256: <code>{html.escape(str(audit_a['action_vec_sha256']))}</code></li>
</ul>
</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = args.evidence_dir
    rendered = render(
        audit_a=load_json(evidence / "install_audit_seed_2026071901.json"),
        audit_b=load_json(evidence / "install_audit_seed_2026071902.json"),
        selected=load_json(evidence / "selected_from_checkpoint.json"),
        fixed_action=load_json(evidence / "strict_best_fusion_fixed_action.json"),
        snapshot=load_json(evidence / "graceful_stop_summary.json"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
