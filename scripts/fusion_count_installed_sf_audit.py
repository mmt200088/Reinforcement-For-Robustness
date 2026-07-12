#!/usr/bin/env python3
"""Audit post-replan Stage-2 configs installed by ``bridge.apply``.

This module is deliberately independent from the RL implementation.  It can
wrap one fixed-action evaluation and proves that reported scaling factors came
from the same config objects installed into ``function_handler``.
"""
from __future__ import annotations

from collections import defaultdict
import html
import json
import math
from typing import Any, Iterable, Mapping, Sequence


AUTHORITATIVE_PROVENANCE = "post_replan_bridge_apply"
SCHEMA_VERSION = "mrpc-allfusion1-installed-sf-audit-v1"
_AUDITED_BLOCKS = ("block2", "block4", "block5")


def _plain_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    candidate = getattr(value, "value", None)
    if candidate is not None and isinstance(candidate, (str, bool, int, float)):
        return candidate
    return str(value)


def _noise_row(
    *,
    block: str,
    layer: int,
    point: str,
    value: Any,
    provenance: str,
) -> dict[str, Any]:
    sf = getattr(value, "scaling_factor", None) if value is not None else None
    active = sf is not None
    return {
        "layer": int(layer),
        "block": str(block),
        "point": str(point),
        "type": type(value).__name__ if value is not None else "None",
        "distribution": _plain_scalar(getattr(value, "distribution", None)),
        "N": (
            None
            if getattr(value, "N", None) is None
            else int(getattr(value, "N"))
        ),
        "scaling_factor": None if sf is None else int(sf),
        "truncation_k": None,
        "installation_state": "installed" if active else "not_installed",
        "active": bool(active),
        "provenance": provenance,
    }


def _cfg_rows(*, block: str, layer: int, cfg: Any, provenance: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for attr, value in vars(cfg).items():
        if attr == "output_truncation_mode" or attr.startswith("rotation_after_"):
            continue
        if attr == "output_truncation_k":
            rows.append({
                "layer": int(layer),
                "block": str(block),
                "point": attr,
                "type": "truncation",
                "distribution": None,
                "N": None,
                "scaling_factor": None,
                "truncation_k": None if value is None else int(value),
                "installation_state": "installed" if value is not None else "not_installed",
                "active": value is not None,
                "provenance": provenance,
            })
            continue
        if isinstance(value, tuple):
            for index, item in enumerate(value):
                rows.append(_noise_row(
                    block=block,
                    layer=layer,
                    point=f"{attr}[{index}]",
                    value=item,
                    provenance=provenance,
                ))
            continue
        if value is None or hasattr(value, "scaling_factor"):
            rows.append(_noise_row(
                block=block,
                layer=layer,
                point=attr,
                value=value,
                provenance=provenance,
            ))
    return rows


def serialize_installed_cfgs(
    *,
    block2_cfgs: Mapping[int, Any],
    block4_cfgs: Mapping[int, Any],
    block5_cfgs: Mapping[int, Any],
    provenance: str,
) -> list[dict[str, Any]]:
    """Serialize only final cfg objects observed at the model-install boundary."""
    if provenance != AUTHORITATIVE_PROVENANCE:
        raise ValueError(
            "installed SF audit accepts only the authoritative "
            f"{AUTHORITATIVE_PROVENANCE!r} provenance"
        )
    rows: list[dict[str, Any]] = []
    cfg_sets = {
        "block2": block2_cfgs,
        "block4": block4_cfgs,
        "block5": block5_cfgs,
    }
    for block in _AUDITED_BLOCKS:
        for layer, cfg in sorted((cfg_sets[block] or {}).items()):
            rows.extend(_cfg_rows(
                block=block,
                layer=int(layer),
                cfg=cfg,
                provenance=provenance,
            ))
    return rows


class InstalledConfigCapture:
    """Callable replacement for one bridge instance's bound ``apply`` method."""

    def __init__(self, *, original_apply, handler: Any, expected_layers: Iterable[int]):
        self.original_apply = original_apply
        self.handler = handler
        self.expected_layers = set(int(layer) for layer in expected_layers)
        self._captures: list[dict[str, Any]] = []

    def apply(
        self,
        *,
        block2_cfgs=None,
        block4_cfgs=None,
        block5_cfgs=None,
        **kwargs,
    ) -> None:
        supplied = {
            "block2": dict(block2_cfgs or {}),
            "block4": dict(block4_cfgs or {}),
            "block5": dict(block5_cfgs or {}),
        }
        self.original_apply(
            block2_cfgs=block2_cfgs,
            block4_cfgs=block4_cfgs,
            block5_cfgs=block5_cfgs,
            **kwargs,
        )

        getter = getattr(self.handler, "get_active_blb_noise_layers", None)
        if not callable(getter):
            raise RuntimeError("handler does not expose active BLB noise layers")
        active_raw = getter() or {}
        active = {
            block: sorted(int(layer) for layer in active_raw.get(block, set()))
            for block in _AUDITED_BLOCKS
        }
        expected = sorted(self.expected_layers)
        for block in _AUDITED_BLOCKS:
            if active[block] != expected:
                raise RuntimeError(
                    f"{block} active layers mismatch: {active[block]} != {expected}"
                )

        identity_match = True
        for block in _AUDITED_BLOCKS:
            installed = getattr(self.handler, f"{block}_cfg_per_layer", {})
            for layer in self.expected_layers:
                if installed.get(layer) is not supplied[block].get(layer):
                    identity_match = False
                    break
        if not identity_match:
            raise RuntimeError(
                "handler cfg object identity does not match post-replan bridge.apply args"
            )

        rows = serialize_installed_cfgs(
            block2_cfgs=supplied["block2"],
            block4_cfgs=supplied["block4"],
            block5_cfgs=supplied["block5"],
            provenance=AUTHORITATIVE_PROVENANCE,
        )
        self._captures.append({
            "bridge_apply_seen": True,
            "bridge_apply_call_index": len(self._captures),
            "handler_active_layers": active,
            "expected_active_layers": {
                block: expected for block in _AUDITED_BLOCKS
            },
            "handler_cfg_object_identity_match": True,
            "installed_config_rows": rows,
        })

    def assert_complete(self) -> dict[str, Any]:
        if len(self._captures) != 1:
            raise RuntimeError(
                "installed SF audit requires exactly one candidate bridge.apply call; "
                f"observed {len(self._captures)}"
            )
        return self._captures[0]


def build_validation_row_lookup(
    source_rows: Iterable[Mapping[str, Any]],
) -> tuple[dict[int, int], dict[int, int]]:
    """Map original MRPC ``idx`` to its stable unshuffled row ordinal."""
    lookup: dict[int, int] = {}
    labels: dict[int, int] = {}
    for row_id, row in enumerate(source_rows):
        if "idx" not in row:
            raise ValueError(f"validation source row {row_id} is missing idx")
        source_idx = int(row["idx"])
        if source_idx in lookup:
            raise ValueError(f"duplicate MRPC validation idx: {source_idx}")
        if "label" in row:
            label = row["label"]
        elif "labels" in row:
            label = row["labels"]
        else:
            raise ValueError(f"validation source row {row_id} is missing label")
        lookup[source_idx] = int(row_id)
        labels[int(row_id)] = int(label)
    expected = list(range(len(lookup)))
    if sorted(lookup.values()) != expected:
        raise ValueError("validation row mapping is not contiguous")
    return lookup, labels


def _prediction_argmax(logits: Sequence[Any]) -> int:
    if len(logits) != 2:
        raise ValueError(f"MRPC prediction must contain exactly two logits, got {len(logits)}")
    values = [float(value) for value in logits]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("prediction logits must be finite")
    return 0 if values[0] >= values[1] else 1


def aggregate_prediction_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    row_lookup: Mapping[int, int],
    labels: Mapping[int, int],
    expected_groups: Sequence[str],
    expected_trials: int,
) -> list[dict[str, Any]]:
    """Translate historical rows and remove every obsolete identity field."""
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for source in rows:
        source_idx = int(source["dataset_idx"])
        if source_idx not in row_lookup:
            raise ValueError(f"prediction references unknown MRPC idx {source_idx}")
        row_id = int(row_lookup[source_idx])
        group = str(source["group"])
        if group not in expected_groups:
            raise ValueError(f"unexpected prediction group {group!r}")
        gold = int(source["gold_label"])
        if gold != int(labels[row_id]):
            raise ValueError(f"gold label mismatch for validation row {row_id}")
        logits = [float(value) for value in source["logits"]]
        prediction = int(source["predicted_label"])
        expected_prediction = _prediction_argmax(logits)
        if prediction != expected_prediction:
            raise ValueError(f"prediction/logit mismatch for validation row {row_id}")
        correct = bool(source["correct"])
        if correct != (prediction == gold):
            raise ValueError(f"correctness mismatch for validation row {row_id}")
        grouped[(row_id, group)].append({
            "run_seed": int(source["run_seed"]),
            "trial_index": int(source["trial_index"]),
            "trial_seed": (
                None if source.get("trial_seed") is None else int(source["trial_seed"])
            ),
            "predicted_label": prediction,
            "correct": correct,
            "logits": logits,
        })

    output: list[dict[str, Any]] = []
    for row_id in sorted(int(key) for key in labels):
        group_payload: dict[str, Any] = {}
        for group in expected_groups:
            outcomes = grouped.get((row_id, group), [])
            outcomes.sort(key=lambda item: (
                item["run_seed"], item["trial_index"],
                -1 if item["trial_seed"] is None else item["trial_seed"],
            ))
            if len(outcomes) != int(expected_trials):
                raise ValueError(
                    f"validation row {row_id} group {group!r} has "
                    f"{len(outcomes)} outcomes, expected {expected_trials}"
                )
            trial_keys = {
                (item["run_seed"], item["trial_index"], item["trial_seed"])
                for item in outcomes
            }
            if len(trial_keys) != int(expected_trials):
                raise ValueError(
                    f"validation row {row_id} group {group!r} has duplicate trials"
                )
            group_payload[group] = {
                "correct_count": sum(1 for item in outcomes if item["correct"]),
                "trial_count": int(expected_trials),
                "outcomes": outcomes,
            }
        output.append({
            "validation_row_id": int(row_id),
            "gold_label": int(labels[row_id]),
            "groups": group_payload,
        })
    return output


def validate_allfusion1_result(result: Mapping[str, Any]) -> dict[str, Any]:
    steps = list(result.get("step_records") or [])
    if len(steps) != 47:
        raise ValueError(f"expected 47 Stage-2 steps, got {len(steps)}")
    for step in steps:
        if not bool(step.get("valid")):
            raise ValueError(f"step {step.get('step_idx')} is invalid")
        if int(step.get("k_value", -1)) != 13:
            raise ValueError(f"step {step.get('step_idx')} does not use K=13")
        if not bool(step.get("model_uses_replan_config")):
            raise ValueError(f"step {step.get('step_idx')} did not apply replan config")
        if int(step.get("block_idx", -1)) in (2, 4, 5):
            if int(step.get("fusion_count_replan", -1)) != 1:
                raise ValueError(f"step {step.get('step_idx')} is not fusion-count 1")
            if not bool(step.get("boosted")):
                raise ValueError(f"step {step.get('step_idx')} is not boosted")
    fusion = {str(key): int(value) for key, value in dict(result.get("fusion_by_block") or {}).items()}
    if any(fusion.get(block) != 12 for block in ("2", "4", "5")):
        raise ValueError(f"expected B2/B4/B5 fusion totals 12/12/12, got {fusion}")
    metrics = dict(result.get("metrics") or {})
    required_metrics = ("loss_mean", "metric1_mean", "metric2_mean")
    for name in required_metrics:
        value = float(metrics[name])
        if not math.isfinite(value):
            raise ValueError(f"metric {name} is not finite")
    return {
        "passed": True,
        "valid_step_count": 47,
        "fusion_by_block": {"block2": 12, "block4": 12, "block5": 12},
        "truncation_k": 13,
        "metrics_finite": True,
    }


def annotate_replan_removals(
    installed_rows: Sequence[Mapping[str, Any]],
    result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Annotate inactive cfg fields only when replan explicitly fused them away."""
    fused: set[tuple[str, int, str]] = set()
    for step in result.get("step_records") or []:
        block = f"block{int(step.get('block_idx', -1))}"
        layer = int(step.get("layer_idx", -1))
        application = step.get("replan_application") or {}
        for detail in (application.get("per_config") or {}).values():
            for override in detail.get("overrides") or []:
                if override.get("source") == "rescale_fused_away":
                    fused.add((block, layer, str(override.get("cfg_attr"))))
    output: list[dict[str, Any]] = []
    for source in installed_rows:
        row = dict(source)
        key = (str(row["block"]), int(row["layer"]), str(row["point"]))
        if key in fused:
            if row.get("scaling_factor") is not None or row.get("active"):
                raise ValueError(f"replan fused-away field is still installed: {key}")
            row["installation_state"] = "fused_away"
            row["replan_source"] = "rescale_fused_away"
        output.append(row)
    missing = fused - {
        (str(row["block"]), int(row["layer"]), str(row["point"]))
        for row in output
    }
    if missing:
        raise ValueError(f"replan fused-away fields missing from installed cfg capture: {sorted(missing)}")
    return output


def _format_sf(row: Mapping[str, Any]) -> str:
    if row.get("truncation_k") is not None:
        return f"K={int(row['truncation_k'])}"
    if row.get("scaling_factor") is not None:
        return str(int(row["scaling_factor"]))
    state = str(row.get("installation_state", "not_installed"))
    return "OFF / fused away" if state == "fused_away" else "OFF / not installed"


def _json_script(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return text.replace("</", "<\\/")


def render_audit_html(payload: Mapping[str, Any]) -> str:
    capture = dict(payload.get("capture") or {})
    rows = list(capture.get("installed_config_rows") or [])
    for row in rows:
        if row.get("provenance") != AUTHORITATIVE_PROVENANCE:
            raise ValueError("installed SF report contains non-authoritative provenance")

    sf_rows = []
    for row in sorted(rows, key=lambda item: (
        int(item["layer"]), str(item["block"]), str(item["point"]),
    )):
        sf_rows.append(
            "<tr>"
            f"<td>{int(row['layer'])}</td>"
            f"<td>{html.escape(str(row['block']).upper())}</td>"
            f"<td><code>{html.escape(str(row['point']))}</code></td>"
            f"<td>{html.escape(_format_sf(row))}</td>"
            f"<td>{html.escape(str(row.get('distribution') or '-'))}</td>"
            f"<td>{html.escape(str(row.get('N') if row.get('N') is not None else '-'))}</td>"
            f"<td><code>{AUTHORITATIVE_PROVENANCE}</code></td>"
            "</tr>"
        )

    validation_rows = []
    for row in payload.get("validation_rows") or []:
        row_id = int(row["validation_row_id"])
        cells = []
        details = []
        for group, aggregate in row.get("groups", {}).items():
            cells.append(
                f"<strong>{html.escape(str(group))}</strong>: "
                f"{int(aggregate['correct_count'])}/{int(aggregate['trial_count'])}"
            )
            details.append(
                f"<h4>{html.escape(str(group))}</h4><pre>"
                f"{html.escape(json.dumps(aggregate['outcomes'], ensure_ascii=False, indent=2))}"
                "</pre>"
            )
        validation_rows.append(
            f"<tr id=\"validation-row-{row_id}\">"
            f"<td>Validation row {row_id}</td>"
            f"<td>{int(row['gold_label'])}</td>"
            f"<td>{'<br>'.join(cells) or '-'}</td>"
            f"<td><details><summary>Trial details</summary>{''.join(details)}</details></td>"
            "</tr>"
        )

    protocol = payload.get("protocol") or {}
    verdict = "PASS" if bool((payload.get("gate") or {}).get("passed")) else "FAIL"
    return """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MRPC all-fusion1 actual installed SF audit</title>
<style>
:root{color-scheme:light;--ink:#17212b;--muted:#5b6773;--line:#d9e0e6;--green:#0b6b3a;--blue:#185fa5;--bg:#f6f8fa}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;letter-spacing:0}
header,main{max-width:1500px;margin:auto;padding:24px}header{background:#fff;border-bottom:1px solid var(--line);max-width:none}h1{font-size:28px;margin:0 0 8px}.lead{color:var(--muted);margin:0}
.band{padding:14px 16px;border-left:4px solid var(--green);background:#eef8f2;margin:18px 0}.verdict{font-size:24px;font-weight:800;color:var(--green)}
section{background:#fff;border:1px solid var(--line);border-radius:6px;padding:18px;margin:16px 0;overflow:auto}h2{font-size:20px;margin:0 0 12px}code{font-size:12px}
table{border-collapse:collapse;width:100%}th,td{border:1px solid var(--line);padding:7px 9px;font-size:12px;text-align:left;vertical-align:top}th{background:#eef2f5;position:sticky;top:0}pre{white-space:pre-wrap;max-height:420px;overflow:auto;background:#f7f8fa;padding:10px}details summary{cursor:pointer;color:var(--blue)}
</style></head><body>
<header><h1>MRPC all-fusion1 actual installed SF audit</h1>
<p class="lead">Only post-replan configs observed at the model installation boundary are reported.</p></header><main>
<div class="band"><span class="verdict">""" + verdict + """</span><br>
Handler object identity: """ + html.escape(str(bool(capture.get("handler_cfg_object_identity_match")))) + """<br>
SF provenance: <code>post_replan_bridge_apply</code></div>
<section><h2>Protocol</h2><p>GELU: <code>""" + html.escape(str(protocol.get("gelu"))) + """</code> &nbsp; Softmax: <code>""" + html.escape(str(protocol.get("softmax"))) + """</code> &nbsp; truncation: K=""" + html.escape(str(protocol.get("k"))) + """</p></section>
<section><h2>Final SF actually passed into the model</h2>
<table><thead><tr><th>Layer</th><th>Block</th><th>Slot</th><th>Final SF / K</th><th>Distribution</th><th>N</th><th>Evidence boundary</th></tr></thead><tbody>""" + "".join(sf_rows) + """</tbody></table></section>
<section><h2>MRPC validation rows 0–407</h2>
<table><thead><tr><th>Validation row</th><th>Gold</th><th>Correct count</th><th>Predictions and logits</th></tr></thead><tbody>""" + "".join(validation_rows) + """</tbody></table></section>
<script type="application/json" id="audit-data">""" + _json_script({
        "schema_version": payload.get("schema_version", SCHEMA_VERSION),
        "validation_rows": payload.get("validation_rows") or [],
    }) + """</script></main></body></html>"""


__all__ = [
    "AUTHORITATIVE_PROVENANCE",
    "SCHEMA_VERSION",
    "InstalledConfigCapture",
    "aggregate_prediction_rows",
    "annotate_replan_removals",
    "build_validation_row_lookup",
    "render_audit_html",
    "serialize_installed_cfgs",
    "validate_allfusion1_result",
]
