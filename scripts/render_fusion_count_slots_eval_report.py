#!/usr/bin/env python3
"""Render a detailed HTML report for fixed fusion-count action evals.

The Paean final-eval JSON contains the model-installed BLB config under
``config_details.full_noise_config.entries``.  The action config JSON contains
the slot-form request sent into final-eval.  This report shows both side by
side for each requested group.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from json_utils import read_json_file  # noqa: E402
from report_format_utils import html_table  # noqa: E402

_OPTION_INDEX_CACHE: dict[int, tuple[object, int | None, dict[int, Mapping[str, Any]]]] = {}


def _esc(value: Any) -> str:
    if value is None:
        return ""
    return html.escape(str(value))


def _fmt_num(value: Any, digits: int = 6) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.{int(digits)}f}"
    except Exception:
        return str(value)


class _HtmlPartsWriter:
    def __init__(self, path: Path):
        self._handle = path.open("w", encoding="utf-8")
        self._first = True

    def append(self, value: Any) -> None:
        if not self._first:
            self._handle.write("\n")
        self._handle.write(str(value))
        self._first = False

    def extend(self, values: Iterable[Any]) -> None:
        for value in values:
            self.append(value)

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "_HtmlPartsWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def _scalar_rows(payload: Mapping[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for key in sorted(payload):
        value = payload[key]
        if isinstance(value, (str, int, float, bool)) or value is None:
            rows.append([key, value])
    return rows


def _resolve_action_config(artifact_dir: Path, raw_path: str) -> Path | None:
    raw = Path(str(raw_path))
    candidates = [
        raw,
        artifact_dir / "action_configs" / raw.name,
        artifact_dir / raw.name,
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _load_maps(map_dir: Path | None) -> dict[str, Mapping[str, Any]]:
    if map_dir is None:
        return {}
    out: dict[str, Mapping[str, Any]] = {}
    for path in _iter_map_paths(map_dir):
        payload = read_json_file(path)
        if isinstance(payload, Mapping) and payload.get("graph_key"):
            out[str(payload["graph_key"])] = payload
    return out


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
    return f"block{int(block_idx)}"


def _schedule_by_layer_block(action_config: Mapping[str, Any]) -> dict[tuple[int, int], Mapping[str, Any]]:
    profile = str(action_config.get("profile") or "mrpc")
    gelu = [int(v) for v in action_config.get("gelu_degree") or []]
    softmax = [int(v) for v in action_config.get("attn_degree") or []]
    out: dict[tuple[int, int], Mapping[str, Any]] = {}
    step_idx = 0
    for layer_idx in range(min(len(gelu), len(softmax))):
        block_order = (2, 4, 5) if layer_idx == 0 else (1, 2, 4, 5)
        for block_idx in block_order:
            out[(int(layer_idx), int(block_idx))] = {
                "step_idx": int(step_idx),
                "graph_key": _graph_key(block_idx, profile, gelu[layer_idx], softmax[layer_idx]),
            }
            step_idx += 1
    return out


def _option_index_for_graph(graph: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    options = graph.get("options") or []
    cache_key = id(options)
    try:
        options_len: int | None = len(options)  # type: ignore[arg-type]
    except TypeError:
        options_len = None
    cached = _OPTION_INDEX_CACHE.get(cache_key)
    if cached is not None and cached[0] is options and cached[1] == options_len:
        return cached[2]
    index: dict[int, Mapping[str, Any]] = {}
    for option in options:
        if not isinstance(option, Mapping):
            continue
        try:
            option_id = int(option.get("option_id", -1))
        except Exception:
            continue
        index[option_id] = option
    _OPTION_INDEX_CACHE[cache_key] = (options, options_len, index)
    return index


def _option_by_id(graph: Mapping[str, Any] | None, option_id: Any) -> Mapping[str, Any] | None:
    if graph is None or option_id == "":
        return None
    return _option_index_for_graph(graph).get(int(option_id))


def _source_for_slot(
    *,
    label: str,
    action_config: Mapping[str, Any],
    maps: Mapping[str, Mapping[str, Any]],
    schedule: Mapping[tuple[int, int], Mapping[str, Any]],
) -> Mapping[str, Any]:
    match = re.match(r"^L(?P<layer>\d+)\.B(?P<block>\d+)\.", str(label))
    if not match:
        return {}
    step = schedule.get((int(match.group("layer")), int(match.group("block"))))
    if not step:
        return {}
    group = action_config.get("group") or {}
    graph_key = str(step["graph_key"])
    step_key = str(step["step_idx"])
    option_by_step = group.get("option_by_step") or {}
    option_by_graph = group.get("option_by_graph") or {}
    option_id = option_by_step.get(step_key, option_by_graph.get(graph_key, ""))
    graph = maps.get(graph_key)
    option = _option_by_id(graph, option_id)
    build_meta = graph.get("build_meta", {}) if isinstance(graph, Mapping) else {}
    return {
        "graph_key": graph_key,
        "step_idx": step_key,
        "option_id": option_id,
        "fusion_count": "" if option is None else option.get("fusion_count", ""),
        "boosted": "" if option is None else bool(option.get("boosted", False)),
        "precision_boost_applied": bool(build_meta.get("precision_boost_applied", False)),
    }


def _action_slot_rows(
    action_config: Mapping[str, Any],
    *,
    maps: Mapping[str, Mapping[str, Any]],
) -> list[list[Any]]:
    rows: list[list[Any]] = []
    schedule = _schedule_by_layer_block(action_config)
    for idx, entry in enumerate(action_config.get("slots") or []):
        if not isinstance(entry, Mapping):
            continue
        src = _source_for_slot(
            label=str(entry.get("label", "")),
            action_config=action_config,
            maps=maps,
            schedule=schedule,
        )
        rows.append([
            idx,
            entry.get("label", ""),
            entry.get("scaling_factor", ""),
            entry.get("truncation_bits", ""),
            src.get("graph_key", ""),
            src.get("step_idx", ""),
            src.get("option_id", ""),
            src.get("fusion_count", ""),
            src.get("boosted", ""),
            src.get("precision_boost_applied", ""),
        ])
    return rows


def _boost_audit_rows(result: Mapping[str, Any], maps: Mapping[str, Mapping[str, Any]]) -> list[list[Any]]:
    group = result.get("fusion_group") or {}
    rows: list[list[Any]] = []
    for graph_key, option_id in sorted((group.get("option_by_graph") or {}).items()):
        graph = maps.get(str(graph_key))
        option = _option_by_id(graph, option_id)
        build_meta = graph.get("build_meta", {}) if isinstance(graph, Mapping) else {}
        explicit = option.get("explicit_field_values") if isinstance(option, Mapping) else {}
        slots = option.get("slots") if isinstance(option, Mapping) else {}
        rows.append([
            graph_key,
            "graph",
            "",
            option_id,
            "" if option is None else option.get("fusion_count", ""),
            "" if option is None else bool(option.get("boosted", False)),
            bool(build_meta.get("precision_boost_applied", False)),
            len(explicit or {}),
            len(slots or {}),
            "" if option is None else option.get("boost_description", ""),
        ])
    target_graph = group.get("target_graph") or "block4"
    for step_idx, option_id in sorted((group.get("option_by_step") or {}).items(), key=lambda item: int(item[0])):
        graph = maps.get(str(target_graph))
        option = _option_by_id(graph, option_id)
        build_meta = graph.get("build_meta", {}) if isinstance(graph, Mapping) else {}
        explicit = option.get("explicit_field_values") if isinstance(option, Mapping) else {}
        slots = option.get("slots") if isinstance(option, Mapping) else {}
        rows.append([
            target_graph,
            "step_override",
            step_idx,
            option_id,
            "" if option is None else option.get("fusion_count", ""),
            "" if option is None else bool(option.get("boosted", False)),
            bool(build_meta.get("precision_boost_applied", False)),
            len(explicit or {}),
            len(slots or {}),
            "" if option is None else option.get("boost_description", ""),
        ])
    return rows


def _selected_option_rows(result: Mapping[str, Any], maps: Mapping[str, Mapping[str, Any]]) -> list[list[Any]]:
    group = result.get("fusion_group") or {}
    rows: list[list[Any]] = []
    for graph_key, option_id in sorted((group.get("option_by_graph") or {}).items()):
        graph = maps.get(str(graph_key))
        option = _option_by_id(graph, option_id)
        build_meta = graph.get("build_meta", {}) if isinstance(graph, Mapping) else {}
        rows.append([
            graph_key,
            "graph",
            "",
            option_id,
            "" if option is None else option.get("fusion_count", ""),
            "" if option is None else bool(option.get("boosted", False)),
            bool(build_meta.get("precision_boost_applied", False)),
            "" if option is None else option.get("boost_description", ""),
        ])
    target_graph = group.get("target_graph") or "block4"
    for step_idx, option_id in sorted((group.get("option_by_step") or {}).items(), key=lambda item: int(item[0])):
        graph = maps.get(str(target_graph))
        option = _option_by_id(graph, option_id)
        build_meta = graph.get("build_meta", {}) if isinstance(graph, Mapping) else {}
        rows.append([
            target_graph,
            "step_override",
            step_idx,
            option_id,
            "" if option is None else option.get("fusion_count", ""),
            "" if option is None else bool(option.get("boosted", False)),
            bool(build_meta.get("precision_boost_applied", False)),
            "" if option is None else option.get("boost_description", ""),
        ])
    return rows


def _installed_slot_rows(result: Mapping[str, Any]) -> list[list[Any]]:
    config = result.get("config_details", {}).get("full_noise_config", {})
    entries = config.get("entries") or []
    rows: list[list[Any]] = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            continue
        rows.append([
            idx,
            entry.get("path", ""),
            entry.get("type", ""),
            entry.get("layer", ""),
            entry.get("block", ""),
            entry.get("point", ""),
            entry.get("distribution", ""),
            entry.get("N", ""),
            entry.get("scaling_factor", ""),
            entry.get("truncation_k", ""),
            entry.get("value", ""),
            entry.get("active", ""),
            entry.get("note", ""),
        ])
    return rows


def _active_entry_key(entry: Mapping[str, Any]) -> str:
    return str(entry.get("path") or "")


def _active_entry_value(entry: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        entry.get("type", ""),
        entry.get("distribution", ""),
        entry.get("N", ""),
        entry.get("scaling_factor", ""),
        entry.get("truncation_k", ""),
        entry.get("value", ""),
        entry.get("active", ""),
    )


def _active_installed_entry_map(result: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    config = result.get("config_details", {}).get("full_noise_config", {})
    entries = config.get("entries") or []
    out: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping) or not entry.get("active"):
            continue
        out[_active_entry_key(entry)] = entry
    return out


def _changed_installed_rows(
    result: Mapping[str, Any],
    baseline_entries: Mapping[str, Mapping[str, Any]],
) -> list[list[Any]]:
    rows: list[list[Any]] = []
    current = _active_installed_entry_map(result)
    for path in sorted(set(current) | set(baseline_entries)):
        entry = current.get(path)
        baseline = baseline_entries.get(path)
        if entry is not None and baseline is not None and _active_entry_value(entry) == _active_entry_value(baseline):
            continue
        src = entry if entry is not None else {}
        base = baseline if baseline is not None else {}
        rows.append([
            path,
            "added" if baseline is None else ("removed" if entry is None else "changed"),
            base.get("scaling_factor", ""),
            src.get("scaling_factor", ""),
            base.get("truncation_k", ""),
            src.get("truncation_k", ""),
            base.get("value", ""),
            src.get("value", ""),
            src.get("type", base.get("type", "")),
            src.get("distribution", base.get("distribution", "")),
            src.get("note", base.get("note", "")),
        ])
    return rows


def _repeat_trial_rows(result: Mapping[str, Any]) -> list[list[Any]]:
    repeat = result.get("repeat_evaluation") or {}
    rows: list[list[Any]] = []
    for trial in repeat.get("trials") or []:
        rows.append([
            trial.get("trial", ""),
            _fmt_num(trial.get("loss", "")),
            _fmt_num(trial.get("p", "")),
            _fmt_num(trial.get("s", "")),
            _fmt_num(trial.get("time_ms", ""), 3),
        ])
    stats = repeat.get("stats") or {}
    if stats:
        rows.append([
            "mean",
            _fmt_num(stats.get("loss_mean", "")),
            _fmt_num(stats.get("p_mean", "")),
            _fmt_num(stats.get("s_mean", "")),
            _fmt_num(stats.get("time_mean_ms", ""), 3),
        ])
        rows.append([
            "std",
            _fmt_num(stats.get("loss_std", "")),
            _fmt_num(stats.get("p_std", "")),
            _fmt_num(stats.get("s_std", "")),
            _fmt_num(stats.get("time_std_ms", ""), 3),
        ])
    return rows


def _metric_summary_rows(combined: Mapping[str, Any]) -> list[list[Any]]:
    rows = []
    baseline = combined.get("baseline", {})
    rows.append([
        "baseline_plaintext",
        "baseline",
        "",
        baseline.get("loss", ""),
        baseline.get("loss_std", ""),
        baseline.get("p", ""),
        baseline.get("p_std", ""),
        baseline.get("s", ""),
        baseline.get("s_std", ""),
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
    ])
    for result in combined.get("group_results") or []:
        group = result.get("fusion_group") or {}
        rows.append([
            result.get("name", ""),
            group.get("family", ""),
            "yes" if group.get("no_op") else "",
            result.get("loss", ""),
            result.get("loss_std", ""),
            result.get("p", ""),
            result.get("p_std", ""),
            result.get("s", ""),
        result.get("s_std", ""),
        (result.get("repeat_evaluation") or {}).get("stats", {}).get("n", ""),
        (result.get("repeat_evaluation") or {}).get("stats", {}).get("time_std_ms", ""),
        result.get("loss_delta_vs_baseline", result.get("delta_loss_vs_baseline", "")),
        result.get("p_delta_vs_baseline", result.get("delta_p_vs_baseline", "")),
            result.get("s_delta_vs_baseline", result.get("delta_s_vs_baseline", "")),
            result.get("total_bits_sum", ""),
            result.get("total_fusion_count", ""),
            result.get("time_ms", ""),
        ])
    return rows


def _mapping_rows(combined: Mapping[str, Any]) -> list[list[Any]]:
    rows = []
    for result in combined.get("group_results") or []:
        group = result.get("fusion_group") or {}
        rows.append([
            result.get("name", ""),
            result.get("canonical_run", ""),
            "yes" if result.get("reused_from_canonical") else "",
            result.get("action_hash", "")[:16],
            json.dumps(group.get("fusion_count_by_graph", {}), ensure_ascii=False, sort_keys=True),
            json.dumps(group.get("option_by_graph", {}), ensure_ascii=False, sort_keys=True),
            json.dumps(group.get("option_by_step", {}), ensure_ascii=False, sort_keys=True),
        ])
    return rows


def _schedule_rows(ctx: Mapping[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for graph_key, layers in sorted((ctx.get("schedule_occurrences") or {}).items()):
        layer_list = [int(v) for v in layers]
        rows.append([
            graph_key,
            len(layer_list),
            ", ".join(f"L{v}" for v in layer_list),
        ])
    return rows


def _boost_summary_rows(
    combined: Mapping[str, Any],
    maps: Mapping[str, Mapping[str, Any]],
) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for result in combined.get("group_results") or []:
        group = result.get("fusion_group") or {}
        boosted_fc1: list[str] = []
        unboosted_fc1: list[str] = []
        boosted_steps: list[str] = []
        unboosted_steps: list[str] = []
        for graph_key, option_id in sorted((group.get("option_by_graph") or {}).items()):
            graph = maps.get(str(graph_key))
            option = _option_by_id(graph, option_id)
            if option is None or int(option.get("fusion_count", -1)) != 1:
                continue
            target = boosted_fc1 if bool(option.get("boosted", False)) else unboosted_fc1
            target.append(str(graph_key))
        target_graph = str(group.get("target_graph") or "block4")
        for step_idx, option_id in sorted((group.get("option_by_step") or {}).items(), key=lambda item: int(item[0])):
            graph = maps.get(target_graph)
            option = _option_by_id(graph, option_id)
            if option is None or int(option.get("fusion_count", -1)) != 1:
                continue
            target = boosted_steps if bool(option.get("boosted", False)) else unboosted_steps
            target.append(f"{target_graph}@step{step_idx}")
        if unboosted_fc1 or unboosted_steps:
            verdict = "MIXED: selected fc=1 includes unboosted option(s)"
        elif boosted_fc1 or boosted_steps:
            verdict = "BOOSTED where fc=1 is selected"
        else:
            verdict = "NO boosted fc=1 selected"
        rows.append([
            result.get("name", ""),
            result.get("total_fusion_count", ""),
            "yes" if group.get("no_op") else "",
            ", ".join(boosted_fc1),
            ", ".join(unboosted_fc1),
            ", ".join(boosted_steps),
            ", ".join(unboosted_steps),
            verdict,
        ])
    return rows


def _emit_rendered_html(
    combined: Mapping[str, Any],
    *,
    artifact_dir: Path,
    run_name: str,
    maps: Mapping[str, Mapping[str, Any]],
    parts: Any,
) -> None:
    ctx = combined.get("map_report_context", {})
    protocol = combined.get("evaluation_protocol", {})
    parts.extend([
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Fusion Count Slot Eval Detailed Report</title>",
        "<style>",
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;margin:0;color:#1f2933;background:#fff}",
        "main{max-width:1500px;margin:0 auto;padding:28px 24px 64px}",
        "h1{font-size:28px;margin:0 0 8px}h2{font-size:20px;margin:30px 0 12px}h3{font-size:16px;margin:22px 0 10px}",
        "p{color:#64748b;line-height:1.45}.note{background:#eef6ff;border-left:4px solid #2563eb;padding:10px 12px;margin:12px 0}",
        ".warn{background:#fff7ed;border-left:4px solid #f97316;padding:10px 12px;margin:12px 0}.muted{color:#64748b}.small{font-size:12px}",
        "code{background:#f1f5f9;padding:1px 4px;border-radius:4px}pre{white-space:pre-wrap;margin:0;font-size:12px;line-height:1.35}",
        "table{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0 18px;background:#fff}",
        "th,td{border:1px solid #d8dee8;padding:6px 8px;text-align:left;vertical-align:top}th{background:#f3f6fa;position:sticky;top:0;z-index:1}",
        ".slotwrap{max-height:560px;overflow:auto;border:1px solid #d8dee8;margin:8px 0 18px}.slotwrap table{margin:0}",
        "details{border-top:1px solid #d8dee8;padding:3px 0 10px}summary{cursor:pointer;font-weight:700;padding:10px 0}",
        ".num{text-align:right;font-variant-numeric:tabular-nums}.missing{color:#b42318;font-weight:650}",
        "</style></head><body><main>",
        "<h1>Fusion Count Slot Eval Detailed Report</h1>",
        f"<p>Server run: <code>{_esc(run_name)}</code>. Generated locally: <code>{_esc(datetime.now().isoformat(timespec='seconds'))}</code>.</p>",
        "<div class='note'>Action config slots are the slot-form fusion-count option requests sent into final-eval. "
        "Model-installed slots come from <code>config_details.full_noise_config.entries</code>, i.e. the BLB config installed immediately before model forward. "
        "This report keeps action/noise details in tables instead of nested JSON blocks.</div>",
        "<div class='warn'>For graphs without a reachable fusion_count=1 option, the map generator clamps to the nearest available count and marks those groups as no-op/reused where applicable.</div>",
        "<div class='warn'>Precision-boost audit uses the committed map JSON. <code>boosted=True</code> means the selected fusion option stores explicit field values from the 加大精度 post-process; unboosted fc=1 options are shown explicitly. Step overrides are audited separately from graph-wide options.</div>",
        "<h2>Protocol</h2>",
        html_table(
            [
                "profile", "Stage-1 GELU", "Stage-1 Softmax", "baseline K",
                "split", "stage2 groups", "unique action runs",
                "requested groups", "artifact dir",
            ],
            [[
                ctx.get("profile", ""),
                json.dumps(ctx.get("stage1_gelu", []), ensure_ascii=False),
                json.dumps(ctx.get("stage1_softmax", []), ensure_ascii=False),
                ctx.get("baseline_k_value", ""),
                protocol.get("split", ""),
                protocol.get("stage2_groups", ""),
                protocol.get("unique_action_runs", ""),
                protocol.get("requested_group_count", ""),
                str(artifact_dir),
            ]],
        ),
        "<h2>Schedule Graph Occurrences</h2>",
        "<p class='small muted'>Only graphs listed here are actually used to construct model noise slots for this Stage-1 GELU/Softmax schedule.</p>",
        html_table(["graph", "occurrences", "layers"], _schedule_rows(ctx)),
        "<h2>Metrics Summary</h2>",
        html_table(
            [
                "group", "family", "no-op", "loss", "loss std", "Accuracy",
                "Accuracy std", "F1", "F1 std", "repeat n", "time std ms",
                "loss delta", "Accuracy delta", "F1 delta", "bits", "fusion",
                "time ms",
            ],
            _metric_summary_rows(combined),
        ),
        "<h2>Fusion Option Mapping</h2>",
        html_table(
            ["group", "canonical run", "reused", "action hash", "fusion count by graph", "option by graph", "option by step"],
            _mapping_rows(combined),
        ),
        "<h2>Precision Boost Summary</h2>",
        html_table(
            [
                "group", "realized fusion", "no-op", "boosted fc=1 graph options",
                "unboosted fc=1 graph options", "boosted fc=1 step overrides",
                "unboosted fc=1 step overrides", "verdict",
            ],
            _boost_summary_rows(combined, maps),
        ),
    ])

    baseline = combined.get("baseline", {})
    baseline_result = next(
        (r for r in combined.get("group_results") or [] if str(r.get("name")) == "all_fusion0"),
        {},
    )
    baseline_installed_entries = _active_installed_entry_map(baseline_result)
    parts.extend([
        "<h2>Baseline Scalars</h2>",
        html_table(["metric", "value"], _scalar_rows(baseline)),
    ])

    for result in combined.get("group_results") or []:
        name = str(result.get("name", "unnamed"))
        action_path = _resolve_action_config(artifact_dir, str(result.get("action_config_path", "")))
        action_payload: Mapping[str, Any] = {}
        action_note = ""
        if action_path is not None:
            action_payload = read_json_file(action_path)
            action_note = str(action_path)
        else:
            action_note = f"missing action config: {result.get('action_config_path', '')}"
        action_rows = _action_slot_rows(action_payload, maps=maps)
        installed_rows = _installed_slot_rows(result)
        changed_rows = _changed_installed_rows(result, baseline_installed_entries)
        boost_rows = _boost_audit_rows(result, maps)
        selected_rows = _selected_option_rows(result, maps)
        repeat_rows = _repeat_trial_rows(result)
        parts.append(
            f"<details><summary>{_esc(name)} "
            f"<span class='muted small'>family={_esc((result.get('fusion_group') or {}).get('family', ''))}; "
            f"canonical={_esc(result.get('canonical_run', ''))}; "
            f"slots={len(action_rows)}; installed_entries={len(installed_rows)}</span></summary>"
        )
        parts.extend([
            "<h3>All Scalar Metrics</h3>",
            html_table(["metric", "value"], _scalar_rows(result)),
            "<h3>5-Repeat Trial Metrics</h3>",
            html_table(["trial", "loss", "Accuracy", "F1", "time ms"], repeat_rows),
            "<h3>Selected Fusion Options</h3>",
            html_table(
                [
                    "graph", "source", "step", "option", "fusion_count",
                    "boosted", "map precision_boost_applied", "boost description",
                ],
                selected_rows,
            ),
            "<h3>Fusion Option Precision-Boost Audit</h3>",
            html_table(
                [
                    "graph", "source", "step", "option", "fusion_count", "boosted",
                    "map precision_boost_applied", "explicit_field_values", "slots",
                    "boost description",
                ],
                boost_rows,
            ),
            "<h3>Active Model-Installed Slot Deltas vs all_fusion0</h3>",
            "<div class='slotwrap'>",
            html_table(
                [
                    "path", "change", "baseline scaling", "current scaling",
                    "baseline truncation", "current truncation", "baseline value",
                    "current value", "type", "distribution", "note",
                ],
                changed_rows,
            ),
            "</div>",
            "<h3>Action Config Slots Sent Into Final-Eval</h3>",
            f"<p class='small muted'>{_esc(action_note)}</p>" if action_path is not None else f"<p class='missing'>{_esc(action_note)}</p>",
            "<div class='slotwrap'>",
            html_table(
                [
                    "#", "label", "scaling_factor", "truncation_bits",
                    "source_graph", "step", "source_option", "source_fusion_count",
                    "source_boosted", "map precision_boost_applied",
                ],
                action_rows,
            ),
            "</div>",
            "<h3>Model-Installed Slots Before Forward</h3>",
            "<div class='slotwrap'>",
            html_table(
                [
                    "#", "path", "type", "layer", "block", "point", "distribution",
                    "N", "scaling_factor", "truncation_k", "value", "active", "note",
                ],
                installed_rows,
            ),
            "</div>",
            "</details>",
        ])

    parts.append("</main></body></html>")


def render(
    combined: Mapping[str, Any],
    *,
    artifact_dir: Path,
    run_name: str,
    maps: Mapping[str, Mapping[str, Any]],
) -> str:
    parts: list[str] = []
    _emit_rendered_html(
        combined,
        artifact_dir=artifact_dir,
        run_name=run_name,
        maps=maps,
        parts=parts,
    )
    return "\n".join(parts)


def write_rendered_html(
    output_html: Path,
    combined: Mapping[str, Any],
    *,
    artifact_dir: Path,
    run_name: str,
    maps: Mapping[str, Mapping[str, Any]],
) -> None:
    with _HtmlPartsWriter(output_html) as parts:
        _emit_rendered_html(
            combined,
            artifact_dir=artifact_dir,
            run_name=run_name,
            maps=maps,
            parts=parts,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined-json", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--output-html", required=True)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--map-dir", default="blb_stage2_rl/fusion_maps/mrpc")
    args = parser.parse_args()

    combined_path = Path(args.combined_json)
    artifact_dir = Path(args.artifact_dir)
    output_html = Path(args.output_html)
    map_dir = Path(args.map_dir) if args.map_dir else None
    run_name = args.run_name or artifact_dir.name
    combined = read_json_file(combined_path)
    maps = _load_maps(map_dir)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    write_rendered_html(
        output_html,
        combined,
        artifact_dir=artifact_dir,
        run_name=run_name,
        maps=maps,
    )
    print(output_html)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
