#!/usr/bin/env python3
"""离线再生 BLB Stage-2 RL 的「图片 / 检测报告」产物（torch-free 边车工具）。

用途
----
1. 在一个**已完成（或进行中）的 Stage-2 run** 上，按对齐 Stage-1 的新版式重新生成
   训练曲线 + 熵曲线 + 局部最优检测报告——无需重训、无需 torch、无需服务器，
   本地即可肉眼核对。
2. 回填历史 run（这些产物以前要么是旧版式、要么根本没有）。
3. 作为 ``persistence.write_training_curves`` / ``rl_local_optimum`` 的端到端验证手段。

只读输入：``diagnostics/episodes.jsonl(.gz)``、``diagnostics/ppo_updates.jsonl``、
``blb_stage2_status.json``、``blb_stage2_report.md``（取 baseline 参考线，可缺）。
写出（到 ``--out-dir``，默认就是 progress 目录本身）：
``blb_stage2_training_curve.png/.npz``、``blb_stage2_reward_paper.png/.pdf``、
``blb_stage2_entropy_curve.png``、``blb_stage2_search_log.txt``。

用法
----
    python scripts/blb_regen_stage2_outputs.py "Parting Chapter/stage2/bert base mrpc/progress"
    python scripts/blb_regen_stage2_outputs.py <combo_dir>            # 自动找 progress/
    python scripts/blb_regen_stage2_outputs.py <dir> --out-dir /tmp/preview --metric1-name accuracy
"""
from __future__ import annotations

import argparse
import html
import importlib.util
import math
import os
import re
import sys

# rl_local_optimum 只依赖 numpy（torch-free），直接 import。
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from jsonl_utils import iter_jsonl  # noqa: E402
from json_utils import read_json_file  # noqa: E402
import rl_local_optimum  # noqa: E402


_CONSTRAINT_PROBABILITY_KEYS = (
    "loss_precision_probability",
    "metric1_precision_probability",
    "metric2_precision_probability",
    "loss_stability_probability",
    "metric1_stability_probability",
    "metric2_stability_probability",
)


def _load_persistence_module():
    """加载 blb_stage2_rl/persistence.py，但**绕过** ``blb_stage2_rl/__init__``
    （后者 import runner → torch）。这样无 torch 的机器也能出图。"""
    path = os.path.join(_REPO_ROOT, "blb_stage2_rl", "persistence.py")
    spec = importlib.util.spec_from_file_location("blb_persistence_standalone", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _layerwise_k_levels():
    path = os.path.join(_REPO_ROOT, "blb_stage2_rl", "truncation_levels.py")
    spec = importlib.util.spec_from_file_location("blb_truncation_levels_standalone", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return tuple(mod.validate_exact_k_domain(mod.K_LEVELS))


def _load_layerwise_action_module():
    """Load the canonical H/M/L action codec without importing torch."""
    module_name = "blb_layerwise_action_standalone"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    module_dir = os.path.join(_REPO_ROOT, "blb_stage2_rl")
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    path = os.path.join(module_dir, "layerwise_action.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _layerwise_action_table(action_matrix):
    """Decode current H/M/L or historical policy actions into readable rows."""
    matrix = [list(row) for row in action_matrix]
    if matrix and all(len(row) == 2 for row in matrix):
        descriptions = _load_layerwise_action_module().describe_layerwise_action_matrix(
            matrix,
        )
        rows = []
        for item in descriptions:
            k_by_block = item["truncation_k_by_block"]
            rows.append({
                "layer": int(item["layer_idx"]),
                "block4_fusion": int(item["block4_fusion_count"]),
                "precision_preset": str(item["precision_preset_name"]),
                **{
                    f"k_b{block_idx}": int(k_by_block[f"block{block_idx}"])
                    for block_idx in range(1, 6)
                },
            })
        return rows
    if len(matrix) != 12 or any(len(row) != 6 for row in matrix):
        raise ValueError("layerwise action matrix must be num_layers x 2 or legacy 12x6")
    k_levels = _layerwise_k_levels()
    rows = []
    for layer_idx, raw_row in enumerate(matrix):
        values = [int(value) for value in raw_row]
        if values[0] not in (0, 1):
            raise ValueError(f"layer {layer_idx} Block4 fusion must be 0 or 1")
        for slot_idx in range(1, 6):
            if not 0 <= values[slot_idx] < len(k_levels):
                raise ValueError(
                    f"layer {layer_idx} K slot {slot_idx} index {values[slot_idx]} is invalid"
                )
        rows.append({
            "layer": layer_idx,
            "block4_fusion": values[0],
            "k_b1": k_levels[values[1]],
            "k_b2": k_levels[values[2]],
            "k_b3": k_levels[values[3]],
            "k_b4": k_levels[values[4]],
            "k_b5": k_levels[values[5]],
        })
    return rows


def _html_table(headers, rows):
    out = ["<table><thead><tr>"]
    out.extend(f"<th>{html.escape(str(label))}</th>" for _key, label in headers)
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>")
        for key, _label in headers:
            value = row.get(key)
            out.append(f"<td>{html.escape('-' if value is None else str(value))}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def _line_chart_svg(title, series, *, y_domain=None):
    active = [(label, points) for label, points in series if points]
    if not active:
        return ""
    all_points = [point for _label, points in active for point in points]
    x_min = min(point[0] for point in all_points)
    x_max = max(point[0] for point in all_points)
    if x_min == x_max:
        x_max = x_min + 1.0
    if y_domain is None:
        y_min = min(point[1] for point in all_points)
        y_max = max(point[1] for point in all_points)
        if y_min == y_max:
            pad = max(abs(y_min) * 0.05, 0.01)
            y_min -= pad
            y_max += pad
    else:
        y_min, y_max = (float(value) for value in y_domain)
    width, plot_height = 960.0, 280.0
    left, right, top, bottom = 70.0, 24.0, 20.0, 46.0
    legend_rows = (len(active) + 2) // 3
    height = plot_height + top + bottom + 28.0 * legend_rows
    plot_width = width - left - right

    def sx(value):
        return left + (float(value) - x_min) * plot_width / (x_max - x_min)

    def sy(value):
        return top + (y_max - float(value)) * plot_height / (y_max - y_min)

    colors = ("#1565c0", "#d84315", "#2e7d32", "#6a1b9a", "#00838f", "#c62828")
    parts = [
        f"<section class='curve-section'><h2>{html.escape(title)}</h2>",
        f"<svg class='line-chart' viewBox='0 0 {width:.0f} {height:.0f}' role='img' "
        f"aria-label='{html.escape(title)}'>",
    ]
    for tick in range(5):
        value = y_min + (y_max - y_min) * tick / 4.0
        y = sy(value)
        parts.append(
            f"<line x1='{left:.1f}' y1='{y:.1f}' x2='{width - right:.1f}' y2='{y:.1f}' "
            "stroke='#e0e3e7' stroke-width='1'/>"
            f"<text x='{left - 10:.1f}' y='{y + 4:.1f}' text-anchor='end'>{value:.3f}</text>"
        )
    parts.extend((
        f"<line x1='{left:.1f}' y1='{top + plot_height:.1f}' x2='{width - right:.1f}' "
        f"y2='{top + plot_height:.1f}' stroke='#5f6368'/>",
        f"<text x='{left:.1f}' y='{top + plot_height + 25:.1f}'>{x_min:g}</text>",
        f"<text x='{width - right:.1f}' y='{top + plot_height + 25:.1f}' "
        f"text-anchor='end'>{x_max:g}</text>",
        f"<text x='{width / 2:.1f}' y='{top + plot_height + 25:.1f}' "
        "text-anchor='middle'>Episode</text>",
    ))
    legend_top = top + plot_height + bottom
    for index, (label, points) in enumerate(active):
        color = colors[index % len(colors)]
        coordinates = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in points)
        parts.append(
            f"<polyline points='{coordinates}' fill='none' stroke='{color}' "
            "stroke-width='2.5' stroke-linejoin='round' stroke-linecap='round'/>"
        )
        legend_x = left + (index % 3) * (plot_width / 3.0)
        legend_y = legend_top + (index // 3) * 28.0
        parts.append(
            f"<line x1='{legend_x:.1f}' y1='{legend_y:.1f}' x2='{legend_x + 24:.1f}' "
            f"y2='{legend_y:.1f}' stroke='{color}' stroke-width='3'/>"
            f"<text x='{legend_x + 32:.1f}' y='{legend_y + 4:.1f}'>"
            f"{html.escape(label)}</text>"
        )
    parts.append("</svg></section>")
    return "".join(parts)


def _write_layerwise_html_report(
        out_dir, *, summary, baseline, curve_paths, layerwise_curves=None,
        progress_snapshot=None,
):
    """Write a compact auditable report for the robust layerwise result."""
    os.makedirs(out_dir, exist_ok=True)
    action_matrix = summary.get("best_action_matrix")
    action_rows = _layerwise_action_table(action_matrix) if action_matrix else []
    assessment = summary.get("best_assessment") or {}
    metrics = summary.get("best_metrics") or {}
    probability_rows = [
        {"channel": name, "probability": f"{float(value):.6f}"}
        for name, value in assessment.items()
        if str(name).endswith("_probability")
    ]
    metric_rows = [
        {"metric": name, "best": f"{float(value):.9f}"}
        for name, value in metrics.items()
    ]
    baseline_rows = [
        {"metric": name, "baseline": value}
        for name, value in baseline.items()
        if name in ("loss_mean", "metric1_mean", "metric2_mean", "loss_std", "metric1_std", "metric2_std")
    ]
    baseline_reference = summary.get("baseline_reference") or {}
    baseline_groups = baseline_reference.get("groups") or []

    def mean_std(values):
        numbers = [float(value) for value in values or ()]
        if not numbers:
            return "-"
        mean = sum(numbers) / len(numbers)
        if len(numbers) < 2:
            return f"{mean:.9f} +/- 0.000000000"
        variance = sum((value - mean) ** 2 for value in numbers) / (len(numbers) - 1)
        return f"{mean:.9f} +/- {math.sqrt(variance):.9f}"

    baseline_group_rows = [
        {
            "group": int(group.get("group_index", index)),
            "trials": len(group.get("loss_trials") or ()),
            "loss": mean_std(group.get("loss_trials")),
            "metric1": mean_std(group.get("metric1_trials")),
            "metric2": mean_std(group.get("metric2_trials")),
        }
        for index, group in enumerate(baseline_groups)
        if isinstance(group, dict)
    ]
    promotion = summary.get("best_promotion_evidence") or {}
    promotion_rows = [
        {"field": "status", "value": promotion.get("status", "missing")},
        {"field": "trial_count", "value": promotion.get("trial_count", 0)},
        {
            "field": "seed_count",
            "value": len((promotion.get("trials") or {}).get("seeds") or ()),
        },
    ]
    final_evidence = summary.get("final_evidence") or {}
    final_rows = [
        {"field": "status", "value": final_evidence.get("status", "missing")},
        {
            "field": "required_probability",
            "value": final_evidence.get("required_probability", "-"),
        },
        {
            "field": "required_trial_count",
            "value": final_evidence.get("required_trial_count", "-"),
        },
    ]
    images = []
    for label, path in curve_paths.items():
        if path and str(path).lower().endswith(".png"):
            images.append(
                f"<figure><img src='{html.escape(os.path.basename(path))}' "
                f"alt='{html.escape(str(label))}'><figcaption>{html.escape(str(label))}</figcaption></figure>"
            )
    layerwise_curves = layerwise_curves or {}
    entropy_curves = layerwise_curves.get("entropy") or {}
    entropy_chart = _line_chart_svg(
        "Policy Entropy by Action Type",
        (
            ("Block4 fusion entropy", entropy_curves.get("block4_entropy", [])),
            ("Truncation K entropy", entropy_curves.get("k_entropy", [])),
        ),
    )
    fresh_probability_curves = (
        layerwise_curves.get("fresh_constraint_probabilities") or {}
    )
    fresh_probability_chart = _line_chart_svg(
        "Fresh Five-Trial Reward Constraint Probabilities",
        tuple((key, fresh_probability_curves.get(key, [])) for key in _CONSTRAINT_PROBABILITY_KEYS),
        y_domain=(0.0, 1.0),
    )
    pooled_probability_curves = (
        layerwise_curves.get("pooled_constraint_probabilities") or {}
    )
    pooled_probability_chart = _line_chart_svg(
        "Pooled Ranking and Promotion Constraint Probabilities",
        tuple((key, pooled_probability_curves.get(key, [])) for key in _CONSTRAINT_PROBABILITY_KEYS),
        y_domain=(0.0, 1.0),
    )
    resource_objective = summary.get("best_resource_objective") or {}
    resource_labels = (
        ("compute_saving", "Compute saving"),
        ("communication_saving", "Communication saving"),
        ("robust_floor", "Robust floor"),
        ("secondary_progress", "Secondary progress"),
        ("ppo_resource_score", "PPO resource score"),
        ("compute_shapley_credit", "Compute Shapley credit"),
        ("communication_shapley_credit", "Communication Shapley credit"),
    )
    resource_rows = [
        {
            "resource": label,
            "value": (
                "N/A"
                if resource_objective.get(field_name) is None
                else f"{float(resource_objective[field_name]):.9f}"
            ),
        }
        for field_name, label in resource_labels
    ]
    pareto_rows = []
    for row in summary.get("strict_pareto_frontier") or ():
        if not isinstance(row, dict):
            continue
        pareto_rows.append({
            "candidate": row.get("candidate_key", "-"),
            "compute": f"{float(row.get('compute_saving', 0.0)):.9f}",
            "communication": (
                f"{float(row.get('communication_saving', 0.0)):.9f}"
            ),
            "floor": f"{float(row.get('robust_floor', 0.0)):.9f}",
            "secondary": f"{float(row.get('secondary_progress', 0.0)):.9f}",
        })
    best_cost = summary.get("best_variable_cost")
    cost_text = "N/A" if best_cost is None else f"{float(best_cost):.6f}"
    candidate_status = (
        "" if action_rows else
        "<p><strong>No strict feasible candidate selected.</strong> "
        "Baseline and evidence status remain available below.</p>"
    )
    snapshot = progress_snapshot or {}
    snapshot_rows = [
        {"field": "Status", "value": snapshot.get("status", "-")},
        {
            "field": "Completed episodes",
            "value": (
                f"{int(snapshot['completed_episodes'])} / "
                f"{int(snapshot['planned_episodes'])} "
                f"({float(snapshot['progress_percent']):.2f}%)"
                if snapshot.get("planned_episodes")
                else snapshot.get("completed_episodes", "-")
            ),
        },
        {"field": "PPO updates", "value": snapshot.get("ppo_updates", "-")},
        {
            "field": "Latest PPO-window throughput",
            "value": (
                f"{float(snapshot['latest_window_episodes_per_hour']):.2f} episodes/hour"
                if snapshot.get("latest_window_episodes_per_hour") is not None
                else "-"
            ),
        },
        {
            "field": "Recent two-window throughput",
            "value": (
                f"{float(snapshot['recent_window_episodes_per_hour']):.2f} episodes/hour"
                if snapshot.get("recent_window_episodes_per_hour") is not None
                else "-"
            ),
        },
        {"field": "Block4 entropy", "value": snapshot.get("block4_entropy", "-")},
        {"field": "Precision-preset entropy", "value": snapshot.get("k_entropy", "-")},
        {"field": "Converged", "value": snapshot.get("converged", False)},
    ]
    document = "".join([
        "<!doctype html><html><head><meta charset='utf-8'><title>Stage-2 Layerwise Robust PPO</title>",
        "<style>body{font-family:Arial,sans-serif;margin:28px;color:#202124}h1,h2{letter-spacing:0}",
        "table{border-collapse:collapse;width:100%;margin:12px 0 24px}th,td{border:1px solid #d5d9df;padding:7px;text-align:right}",
        "th:first-child,td:first-child{text-align:left}th{background:#f2f4f7}img{max-width:100%;height:auto}",
        ".curve-section{margin:24px 0}.line-chart{display:block;width:100%;height:auto;overflow:visible}",
        ".line-chart text{font-size:12px;fill:#3c4043}</style></head><body>",
        "<h1>Stage-2 Layerwise Robust PPO</h1>",
        candidate_status,
        "<h2>Training Snapshot</h2>",
        _html_table((("field", "Field"), ("value", "Value")), snapshot_rows),
        "<h2>Best Dual-Resource Objective</h2>",
        _html_table((("resource", "Resource"), ("value", "Value")), resource_rows),
        f"<p><strong>Compatibility PPO score:</strong> {cost_text}</p>",
        "<h2>Strict Resource Pareto Frontier</h2>",
        _html_table((
            ("candidate", "Candidate"), ("compute", "Compute saving"),
            ("communication", "Communication saving"),
            ("floor", "Robust floor"),
            ("secondary", "Secondary progress"),
        ), pareto_rows),
        "<h2>Best Metrics</h2>",
        _html_table((("metric", "Metric"), ("best", "Best")), metric_rows),
        "<h2>Baseline</h2>",
        _html_table((("metric", "Metric"), ("baseline", "Baseline")), baseline_rows),
        "<h2>Baseline Trial Distributions</h2>",
        _html_table((
            ("group", "Group"), ("trials", "Trials"), ("loss", "Loss mean +/- std"),
            ("metric1", "Metric1 mean +/- std"), ("metric2", "Metric2 mean +/- std"),
        ), baseline_group_rows),
        "<h2>Best Pooled Ranking Constraint Probabilities</h2>",
        _html_table((("channel", "Channel"), ("probability", "Probability")), probability_rows),
        "<h2>Promotion Evidence</h2>",
        _html_table((("field", "Field"), ("value", "Value")), promotion_rows),
        "<h2>Final Revalidation Evidence</h2>",
        _html_table((("field", "Field"), ("value", "Value")), final_rows),
        "<h2>Selected Layerwise Configuration</h2>",
        _html_table((
            ("layer", "Layer"), ("block4_fusion", "Block4 Fusion"),
            ("precision_preset", "Precision Preset"),
            ("k_b1", "K B1"), ("k_b2", "K B2"), ("k_b3", "K B3"),
            ("k_b4", "K B4"), ("k_b5", "K B5"),
        ), action_rows),
        entropy_chart,
        fresh_probability_chart,
        pooled_probability_chart,
        "<h2>Other Curves</h2>", "".join(images),
        "</body></html>",
    ])
    path = os.path.join(out_dir, "blb_stage2_layerwise_report.html")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(document)
    return path


def _read_layerwise_summary(progress_dir):
    manifest = read_json_file(
        os.path.join(progress_dir, "layerwise_run_manifest.json"), default={},
    )
    summary = read_json_file(
        os.path.join(progress_dir, "layerwise_summary.json"), default={},
    )
    merged = dict(manifest) if isinstance(manifest, dict) else {}
    if isinstance(summary, dict):
        merged.update(summary)
    latest_update = None
    try:
        for row in iter_jsonl(
                _progress_jsonl_path(progress_dir, "ppo_updates.jsonl"),
                gzip_fallback=True,
        ):
            latest_update = row
    except FileNotFoundError:
        pass
    if latest_update is not None:
        for key in (
                "completed_episodes", "block4_entropy", "k_entropy",
                "stall_update_windows", "selected_action_identity",
                "selected_action_stable_update_windows", "converged",
                "extension_required", "plateau_ready",
                "strict_revalidation_passed", "strict_revalidation_status",
                "termination_reason",
        ):
            if key in latest_update:
                merged[key] = latest_update[key]
        merged["ppo_update_count"] = int(latest_update.get("update", 0) or 0)
        frontier = latest_update.get("strict_pareto_frontier") or []
        if frontier:
            best = frontier[0]
            merged.update({
                "strict_pareto_frontier": frontier,
                "best_action_matrix": best.get("action_matrix"),
                "best_layer_configurations": best.get("layer_configurations"),
                "best_full_vector": best.get("full_vector"),
                "best_assessment": best.get("assessment"),
                "best_metrics": best.get("metrics"),
                "best_resource_objective": {
                    key: best.get(key)
                    for key in (
                        "compute_saving", "communication_saving", "robust_floor",
                        "secondary_progress", "ppo_resource_score",
                        "compute_shapley_credit", "communication_shapley_credit",
                        "fusion_count", "removed_k_bits", "layer_resource_rewards",
                        "slot_resource_rewards",
                    )
                },
                "best_variable_cost": best.get("variable_cost"),
                "best_reward": best.get("reward"),
                "best_promotion_evidence": best.get("promotion_evidence"),
                "best_axis_counterfactuals": best.get("axis_counterfactuals"),
                "final_evidence": merged.get("final_evidence") or {
                    "status": "running_not_final_certified",
                    "required_probability": "pending",
                    "required_trial_count": "pending",
                },
            })
    if not isinstance(merged.get("baseline_reference"), dict):
        baseline_references = merged.get("baseline_references") or {}
        if isinstance(baseline_references, dict) and isinstance(
                baseline_references.get("F1"), dict,
        ):
            merged["baseline_reference"] = baseline_references["F1"]
    return merged


def _read_layerwise_curves(progress_dir):
    """Read optional layerwise diagnostics without requiring torch or plotting libs."""
    entropy = {"block4_entropy": [], "k_entropy": []}
    fresh_probabilities = {key: [] for key in _CONSTRAINT_PROBABILITY_KEYS}
    pooled_probabilities = {key: [] for key in _CONSTRAINT_PROBABILITY_KEYS}
    try:
        for index, row in enumerate(iter_jsonl(
                _progress_jsonl_path(progress_dir, "ppo_updates.jsonl"),
                gzip_fallback=True,
        )):
            x_value = row.get("completed_episodes", index + 1)
            for key in entropy:
                value = row.get(key)
                if value is None:
                    continue
                point = (float(x_value), float(value))
                if all(math.isfinite(component) for component in point):
                    entropy[key].append(point)
    except FileNotFoundError:
        pass
    try:
        for index, row in enumerate(iter_jsonl(
                _progress_jsonl_path(progress_dir, "episodes.jsonl"),
                gzip_fallback=True,
        )):
            x_value = row.get("episode", index + 1)
            for source_key, target in (
                    ("fresh_constraint_probabilities", fresh_probabilities),
                    ("pooled_constraint_probabilities", pooled_probabilities),
            ):
                values = row.get(source_key)
                if not isinstance(values, dict):
                    continue
                for key in _CONSTRAINT_PROBABILITY_KEYS:
                    value = values.get(key)
                    if value is None:
                        continue
                    point = (float(x_value), float(value))
                    if all(math.isfinite(component) for component in point):
                        target[key].append(point)
            if not isinstance(row.get("pooled_constraint_probabilities"), dict):
                values = row.get("constraint_probabilities")
                if isinstance(values, dict):
                    for key in _CONSTRAINT_PROBABILITY_KEYS:
                        value = values.get(key)
                        if value is None:
                            continue
                        point = (float(x_value), float(value))
                        if all(math.isfinite(component) for component in point):
                            pooled_probabilities[key].append(point)
    except FileNotFoundError:
        pass
    return {
        "entropy": entropy,
        "fresh_constraint_probabilities": fresh_probabilities,
        "pooled_constraint_probabilities": pooled_probabilities,
    }


def _read_layerwise_progress_snapshot(progress_dir, summary):
    """Summarize checkpoint progress using completed PPO-window wall time."""
    updates = []
    try:
        for row in iter_jsonl(
                _progress_jsonl_path(progress_dir, "ppo_updates.jsonl"),
                gzip_fallback=True,
        ):
            updates.append(row)
            if len(updates) > 3:
                updates.pop(0)
    except FileNotFoundError:
        updates = []
    latest = updates[-1] if updates else {}
    completed = int(
        latest.get("completed_episodes", summary.get("completed_episodes", 0)) or 0
    )
    planned = int(
        summary.get("planned_episodes", summary.get("total_episodes", 0)) or 0
    )
    window_rates = []
    for previous, current in zip(updates, updates[1:]):
        episode_delta = int(current.get("completed_episodes", 0) or 0) - int(
            previous.get("completed_episodes", 0) or 0
        )
        elapsed = float(current.get("elapsed_sec", 0.0) or 0.0)
        if episode_delta > 0 and elapsed > 0.0:
            window_rates.append((episode_delta, elapsed))

    def combined_rate(windows):
        episodes = sum(item[0] for item in windows)
        seconds = sum(item[1] for item in windows)
        return None if seconds <= 0.0 else episodes / seconds * 3600.0

    return {
        "status": summary.get("status", "running"),
        "completed_episodes": completed,
        "planned_episodes": planned,
        "progress_percent": (100.0 * completed / planned if planned else 0.0),
        "ppo_updates": int(latest.get("update", summary.get("ppo_update_count", 0)) or 0),
        "latest_window_episodes_per_hour": combined_rate(window_rates[-1:]),
        "recent_window_episodes_per_hour": combined_rate(window_rates[-2:]),
        "block4_entropy": latest.get("block4_entropy", summary.get("block4_entropy")),
        "k_entropy": latest.get("k_entropy", summary.get("k_entropy")),
        "converged": bool(latest.get("converged", summary.get("converged", False))),
    }


def _resolve_progress_dir(path: str) -> str:
    """接受 progress 目录、combo 目录，或它们的父目录；返回 progress 目录。"""
    path = os.path.abspath(path)
    cands = [
        path,
        os.path.join(path, "progress"),
    ]
    for c in cands:
        if os.path.isdir(os.path.join(c, "diagnostics")) or os.path.isfile(
            os.path.join(c, "blb_stage2_status.json")
        ):
            return c
    # last resort: a *.../progress under path
    for root, dirs, _files in os.walk(path):
        if os.path.basename(root) == "progress" and os.path.isdir(
            os.path.join(root, "diagnostics")
        ):
            return root
    raise FileNotFoundError(
        f"找不到 Stage-2 progress 目录（需含 diagnostics/ 或 blb_stage2_status.json）：{path}"
    )


def _progress_jsonl_path(progress_dir: str, name: str) -> str:
    return os.path.join(progress_dir, "diagnostics", name)


def _read_episodes(progress_dir: str):
    """返回每回合并列序列 dict。total_reward = per_step_sum + terminal_reward。"""
    series = {k: [] for k in (
        "returns", "losses", "metric1s", "metric2s", "fusion", "k_gain", "priority",
        # ADR-014 debug fields (absent on pre-2026-06-14 runs -> stay empty).
        "fusion_b2", "fusion_b4", "fusion_b5", "worst_signed_margin",
        "acc_barrier_sat", "acc_barrier_vio", "cost_score", "p3_metric_margin",
        "metric1_std",
    )}
    present: set = set()
    series["_present"] = present  # type: ignore[assignment]
    _extra = (
        ("fusion_b2", "fusion_count_b2"),
        ("fusion_b4", "fusion_count_b4"),
        ("fusion_b5", "fusion_count_b5"),
        ("worst_signed_margin", "terminal_worst_signed_margin"),
        ("acc_barrier_sat", "terminal_acc_barrier_sat"),
        ("acc_barrier_vio", "terminal_acc_barrier_vio"),
        ("cost_score", "terminal_cost_score"),
        ("p3_metric_margin", "terminal_p3_metric_margin_reward"),
        ("metric1_std", "terminal_metric1_std"),
    )
    row_count = 0
    try:
        for d in iter_jsonl(
            _progress_jsonl_path(progress_dir, "episodes.jsonl"),
            gzip_fallback=True,
        ):
            total = float(d.get("per_step_sum", 0.0) or 0.0) + float(
                d.get("terminal_reward", 0.0) or 0.0
            )
            series["returns"].append(total)
            series["losses"].append(float(d.get("terminal_loss_mean", 0.0) or 0.0))
            series["metric1s"].append(float(d.get("terminal_metric1_mean", 0.0) or 0.0))
            series["metric2s"].append(float(d.get("terminal_metric2_mean", 0.0) or 0.0))
            series["fusion"].append(float(d.get("fusion_count", 0) or 0))
            series["k_gain"].append(float(d.get("terminal_k_gain", 0.0) or 0.0))
            series["priority"].append(int(d.get("terminal_priority", 0) or 0))
            for key, jkey in _extra:
                if jkey in d:
                    present.add(key)
                    if len(series[key]) < row_count:
                        series[key].extend([0.0] * (row_count - len(series[key])))
                    series[key].append(float(d.get(jkey, 0.0) or 0.0))
                elif key in present:
                    series[key].append(0.0)
            row_count += 1
    except FileNotFoundError:
        return series
    return series


def _read_entropy(progress_dir: str):
    """ppo_updates.jsonl → (entropy_series, completed_episodes)。"""
    ent, eps = [], []
    try:
        for d in iter_jsonl(
            _progress_jsonl_path(progress_dir, "ppo_updates.jsonl"),
            gzip_fallback=True,
        ):
            if d.get("entropy") is None:
                continue
            ent.append(float(d["entropy"]))
            eps.append(float(d.get("completed_episodes", len(ent) * 1) or len(ent)))
    except FileNotFoundError:
        return ent, eps
    return ent, eps


_BASELINE_KEYS = ("loss_mean", "metric1_mean", "metric2_mean", "avg_k")
_BASELINE_PATTERNS = {
    key: re.compile(rf"`{re.escape(key)}`\s*\|\s*([-\d.eE+]+)")
    for key in _BASELINE_KEYS
}
_SUMMARY_AVG_K_PATTERN = re.compile(r"baseline avg_k.*?\*\*([\d.]+)\*\*")


def _parse_baselines(progress_dir: str):
    """从 blb_stage2_report.md §3 baseline 表解析参考线值（缺失则返回部分/空）。"""
    out = {}
    layerwise_summary = _read_layerwise_summary(progress_dir)
    baseline_reference = layerwise_summary.get("baseline_reference")
    if isinstance(baseline_reference, dict):
        pooled = baseline_reference.get("pooled")
        if isinstance(pooled, dict):
            for key in (
                    "loss_mean", "loss_std", "metric1_mean", "metric1_std",
                    "metric2_mean", "metric2_std",
            ):
                if pooled.get(key) is not None:
                    out[key] = float(pooled[key])
            out.setdefault("avg_k", 13.0)
    report = os.path.join(progress_dir, "blb_stage2_report.md")
    if os.path.isfile(report):
        try:
            missing = set(_BASELINE_KEYS)
            with open(report, "r", encoding="utf-8") as f:
                for line in f:
                    for key in tuple(missing):
                        m = _BASELINE_PATTERNS[key].search(line)
                        if m:
                            out.setdefault(key, float(m.group(1)))
                            missing.remove(key)
                    if not missing:
                        break
        except Exception:
            pass
    # diagnostics_summary.md 兜底 avg_k
    if "avg_k" not in out:
        summ = os.path.join(progress_dir, "diagnostics", "diagnostics_summary.md")
        if os.path.isfile(summ):
            try:
                with open(summ, "r", encoding="utf-8") as f:
                    for line in f:
                        m = _SUMMARY_AVG_K_PATTERN.search(line)
                        if m:
                            out["avg_k"] = float(m.group(1))
                            break
            except Exception:
                pass
    return out


def _write_search_health_report(
        out_dir,
        *,
        persistence,
        layerwise_summary,
        episode_returns,
        entropies,
        priority,
        fusion_count,
        worst_signed_margin,
        log_fn,
):
    """Use the accepted layerwise convergence contract instead of legacy plateau logic."""
    if str(layerwise_summary.get("schema_version", "")).startswith(
            "stage2_layerwise_robust_"
    ):
        final_evidence = layerwise_summary.get("final_evidence") or {}
        path = os.path.join(out_dir, persistence.BLB_SEARCH_LOG_TXT)
        lines = [
            "Stage-2 layerwise robust PPO search status",
            f"completed_episodes: {int(layerwise_summary.get('completed_episodes', 0) or 0)}",
            f"converged: {bool(layerwise_summary.get('converged', False))}",
            f"block4_entropy: {layerwise_summary.get('block4_entropy')}",
            f"k_entropy: {layerwise_summary.get('k_entropy')}",
            f"stall_update_windows: {int(layerwise_summary.get('stall_update_windows', 0) or 0)}",
            f"final_evidence_status: {final_evidence.get('status', 'missing')}",
            f"P1: {sum(1 for value in priority if int(value) == 1)}",
            f"P2: {sum(1 for value in priority if int(value) == 2)}",
            f"P3: {sum(1 for value in priority if int(value) == 3)}",
        ]
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        log_fn(f"[regen] layerwise convergence status -> {path}")
        return path
    return rl_local_optimum.write_local_optimum_report(
        os.path.join(out_dir, persistence.BLB_SEARCH_LOG_TXT),
        episode_returns=episode_returns,
        episode_entropies=entropies or None,
        best_score_history=None,
        completed_episodes=len(episode_returns),
        title="BLB Stage-2 RL",
        extra_lines=[
            "",
            "--- priority histogram ---",
            f"  P1(acc): {sum(1 for value in priority if int(value) == 1)}",
            f"  P2(stab): {sum(1 for value in priority if int(value) == 2)}",
            f"  P3(cost): {sum(1 for value in priority if int(value) == 3)}",
        ],
        priority=priority,
        fusion_count=fusion_count,
        worst_signed_margin=worst_signed_margin,
        log_fn=log_fn,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description="Regenerate Stage-1-aligned Stage-2 RL outputs (torch-free).")
    ap.add_argument("progress_dir", help="Stage-2 progress/ dir, or a combo dir containing progress/.")
    ap.add_argument("--out-dir", default=None, help="Where to write artifacts (default: the progress dir itself).")
    ap.add_argument("--metric1-name", default="metric1", help="Label for the metric1 panel (e.g. accuracy).")
    ap.add_argument("--metric2-name", default="metric2", help="Label for the metric2 panel (e.g. f1).")
    ap.add_argument("--ma-window", type=int, default=None, help="Moving-average window (default: auto).")
    args = ap.parse_args(argv)

    progress_dir = _resolve_progress_dir(args.progress_dir)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else progress_dir
    os.makedirs(out_dir, exist_ok=True)
    persistence = _load_persistence_module()

    print(f"[regen] progress dir : {progress_dir}")
    print(f"[regen] output  dir : {out_dir}")

    ep = _read_episodes(progress_dir)
    n = len(ep["returns"])
    if n == 0:
        print("[regen][ERROR] episodes.jsonl 为空或缺失，无法出图。")
        return 2
    ent, ent_eps = _read_entropy(progress_dir)
    baselines = _parse_baselines(progress_dir)
    base_avg_k = float(baselines.get("avg_k", 13.0))
    avg_ks = [base_avg_k - kg for kg in ep["k_gain"]]
    print(f"[regen] episodes={n}  ppo_updates(entropy)={len(ent)}  baselines={baselines}")

    curve_paths = persistence.write_training_curves(
        out_dir,
        episode_returns=ep["returns"],
        episode_losses=ep["losses"],
        episode_metric1s=ep["metric1s"],
        episode_metric2s=ep["metric2s"],
        episode_fusion_counts=ep["fusion"],
        episode_avg_ks=avg_ks,
        baselines={
            "loss": baselines.get("loss_mean"),
            "metric1": baselines.get("metric1_mean"),
            "metric2": baselines.get("metric2_mean"),
            "avg_k": base_avg_k,
        },
        metric1_name=args.metric1_name,
        metric2_name=args.metric2_name,
        entropy_series=ent or None,
        entropy_episodes=ent_eps or None,
        ma_window=args.ma_window,
        log_fn=print,
        render_plots=True,
    )
    for k, v in curve_paths.items():
        if v:
            print(f"[regen]   {k:11s} → {v}  ({os.path.getsize(v)} bytes)")

    # ADR-014 崩溃诊断曲线（仅当 episodes.jsonl 带新调试字段时才有内容）。
    present = ep.get("_present", set())

    def _opt(key):
        return ep[key] if key in present else None

    diag_curve = persistence.write_diagnostic_curves(
        out_dir,
        priority=ep["priority"],
        fusion_count=ep["fusion"],
        fusion_b2=_opt("fusion_b2"),
        fusion_b4=_opt("fusion_b4"),
        fusion_b5=_opt("fusion_b5"),
        worst_signed_margin=_opt("worst_signed_margin"),
        acc_barrier_sat=_opt("acc_barrier_sat"),
        acc_barrier_vio=_opt("acc_barrier_vio"),
        cost_score=_opt("cost_score"),
        p3_metric_margin=_opt("p3_metric_margin"),
        metric1_std=_opt("metric1_std"),
        log_fn=print,
        render_plots=True,
    )
    if diag_curve.get("diagnostics_png"):
        print(f"[regen]   diagnostics → {diag_curve['diagnostics_png']}  "
              f"({os.path.getsize(diag_curve['diagnostics_png'])} bytes)")
    elif not present:
        print("[regen]   diagnostics → skipped (run predates ADR-014 debug fields)")

    layerwise_summary = _read_layerwise_summary(progress_dir)
    report_path = _write_search_health_report(
        out_dir,
        persistence=persistence,
        layerwise_summary=layerwise_summary,
        episode_returns=ep["returns"],
        entropies=ent,
        priority=ep["priority"],
        fusion_count=ep["fusion"],
        worst_signed_margin=_opt("worst_signed_margin"),
        log_fn=print,
    )
    if report_path:
        print(f"[regen]   search_log  → {report_path}")
    if str(layerwise_summary.get("schema_version", "")).startswith(
            "stage2_layerwise_robust_"
    ):
        html_path = _write_layerwise_html_report(
            out_dir,
            summary=layerwise_summary,
            baseline=baselines,
            curve_paths=curve_paths,
            layerwise_curves=_read_layerwise_curves(progress_dir),
            progress_snapshot=_read_layerwise_progress_snapshot(
                progress_dir, layerwise_summary,
            ),
        )
        print(f"[regen]   layerwise_html → {html_path}")
    print("[regen] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
