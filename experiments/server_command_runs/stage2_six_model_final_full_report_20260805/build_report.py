#!/usr/bin/env python3
"""Build the standalone six-model Stage-2 RL final report from Git archives."""
from __future__ import annotations

import csv
import gzip
import html
import io
import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from json_utils import write_json_file
from report_format_utils import format_elapsed, format_float, html_table
from stats_utils import mean_from_total, safe_div_or_none


AUDIT_COMMIT = "2bb175defb4c042773650c26d7b853bdd81e0a59"
AUDIT_PATH = (
    "experiments/server_command_runs/stage2_six_model_backup_audit_20260805/"
    "six_model_backup_audit.json"
)
REPORT_DIR = Path(__file__).resolve().parent
REPORT_JSON = REPORT_DIR / "stage2_six_model_final_full_report_data.json"
REPORT_HTML = REPORT_DIR / "stage2_six_model_final_full_report.html"
MAX_CURVE_POINTS = 480


MODEL_META = {
    ("bert-base", "mrpc"): {
        "label": "BERT-base MRPC",
        "family": "BERT-base",
        "metric1": "Accuracy",
        "metric2": "Weighted F1",
    },
    ("bert-base", "rte"): {
        "label": "BERT-base RTE",
        "family": "BERT-base",
        "metric1": "Accuracy",
        "metric2": "Accuracy",
    },
    ("bert-base", "sst2"): {
        "label": "BERT-base SST-2",
        "family": "BERT-base",
        "metric1": "Accuracy",
        "metric2": "Accuracy",
    },
    ("bert-large", "mrpc"): {
        "label": "BERT-large MRPC",
        "family": "BERT-large",
        "metric1": "Accuracy",
        "metric2": "Weighted F1",
    },
    ("bert-large", "rte"): {
        "label": "BERT-large RTE",
        "family": "BERT-large",
        "metric1": "Accuracy",
        "metric2": "Accuracy",
    },
    ("bert-large", "sst2"): {
        "label": "BERT-large SST-2",
        "family": "BERT-large",
        "metric1": "Accuracy",
        "metric2": "Accuracy",
    },
}


def git_bytes(commit: str, path: str) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{commit}:{path}"],
        stderr=subprocess.DEVNULL,
    )


def git_json(commit: str, path: str, *, default: Any = None) -> Any:
    try:
        return json.loads(git_bytes(commit, path))
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return default


def git_text(commit: str, path: str) -> str:
    return git_bytes(commit, path).decode("utf-8")


def iter_git_gzip_jsonl(commit: str, path: str) -> Iterator[dict[str, Any]]:
    """Stream a gzip JSONL Git blob without materializing the raw artifact."""
    proc = subprocess.Popen(
        ["git", "show", f"{commit}:{path}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdout is not None
    try:
        with gzip.GzipFile(fileobj=proc.stdout, mode="rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    continue
                row = json.loads(raw_line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number}: expected JSON object")
                yield row
    finally:
        proc.stdout.close()
        stderr = b"" if proc.stderr is None else proc.stderr.read()
        if proc.stderr is not None:
            proc.stderr.close()
        return_code = proc.wait()
        if return_code != 0:
            raise RuntimeError(
                f"git show failed for {commit}:{path}: "
                f"{stderr.decode('utf-8', errors='replace')}"
            )


def archive_stream_paths(commit: str, archive_root: str) -> tuple[str, str]:
    text = git_text(commit, f"{archive_root}/stream_map.tsv")
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    if reader.fieldnames and "root" in reader.fieldnames:
        paths: dict[str, str] = {}
        for row in reader:
            if row.get("root") != "structured":
                continue
            relative = row.get("relative_path")
            if relative in {"episodes.jsonl", "ppo_updates.jsonl"}:
                paths[str(relative)] = str(row["archive_path"])
        return (
            f"{archive_root}/{paths['episodes.jsonl']}",
            f"{archive_root}/{paths['ppo_updates.jsonl']}",
        )
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    paths = {str(row["archive_name"]): str(row["archive_name"]) for row in reader}
    return (
        f"{archive_root}/streams/{paths['structured_episodes.jsonl.gz']}",
        f"{archive_root}/streams/{paths['structured_ppo_updates.jsonl.gz']}",
    )


def finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


class RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.total_sq = 0.0
        self.minimum: float | None = None
        self.maximum: float | None = None

    def add(self, value: Any) -> None:
        numeric = finite_float(value)
        if numeric is None:
            return
        self.count += 1
        self.total += numeric
        self.total_sq += numeric * numeric
        self.minimum = numeric if self.minimum is None else min(self.minimum, numeric)
        self.maximum = numeric if self.maximum is None else max(self.maximum, numeric)

    def payload(self) -> dict[str, Any]:
        mean = mean_from_total(self.total, self.count, default=0.0)
        variance = max(
            mean_from_total(self.total_sq, self.count, default=0.0) - mean * mean,
            0.0,
        )
        return {
            "count": self.count,
            "mean": mean,
            "std": math.sqrt(variance),
            "min": self.minimum,
            "max": self.maximum,
        }


EPISODE_FIELDS = (
    "total_reward",
    "best_reward_so_far",
    "terminal_loss_mean",
    "terminal_metric1_mean",
    "terminal_metric2_mean",
    "variable_cost",
    "compute_saving",
    "communication_saving",
)


def summarize_episode_stream(
    rows: Iterable[Mapping[str, Any]],
    expected_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bucket_size = max(1, math.ceil(max(expected_count, 1) / MAX_CURVE_POINTS))
    curve: list[dict[str, Any]] = []
    bucket: dict[str, RunningStats] = {key: RunningStats() for key in EPISODE_FIELDS}
    overall_reward = RunningStats()
    priorities: Counter[int] = Counter()
    invalid_episodes = 0
    row_count = 0
    bucket_count = 0
    last_episode = 0
    last_best_reward: float | None = None

    def flush() -> None:
        nonlocal bucket, bucket_count
        if not bucket_count:
            return
        point: dict[str, Any] = {"episode": last_episode}
        for key, stats in bucket.items():
            point[key] = stats.payload()["mean"] if stats.count else None
        point["best_reward_so_far"] = last_best_reward
        curve.append(point)
        bucket = {key: RunningStats() for key in EPISODE_FIELDS}
        bucket_count = 0

    for row in rows:
        row_count += 1
        last_episode = int(row.get("episode", row_count - 1)) + 1
        priority = int(row.get("terminal_priority", 0) or 0)
        priorities[priority] += 1
        if int(row.get("invalid_steps", 0) or 0) > 0 or str(
            row.get("terminal_materialization_failure_reason", "") or ""
        ):
            invalid_episodes += 1
        overall_reward.add(row.get("total_reward"))
        for key in EPISODE_FIELDS:
            bucket[key].add(row.get(key))
        best_value = finite_float(row.get("best_reward_so_far"))
        if best_value is not None:
            last_best_reward = best_value
        bucket_count += 1
        if bucket_count >= bucket_size:
            flush()
    flush()
    if row_count != expected_count:
        raise ValueError(f"episode row mismatch: expected {expected_count}, got {row_count}")
    return curve, {
        "row_count": row_count,
        "bucket_size": bucket_size,
        "priority_counts": {f"P{key}": priorities[key] for key in sorted(priorities)},
        "invalid_episodes": invalid_episodes,
        "reward": overall_reward.payload(),
    }


PPO_FIELDS = (
    "entropy",
    "block4_entropy",
    "k_entropy",
    "window_mean_return",
    "best_reward_so_far",
    "policy_loss",
    "value_loss",
    "approx_kl",
    "value_explained_variance_post",
    "lr",
)


def summarize_ppo_stream(
    rows: Iterable[Mapping[str, Any]],
    expected_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    curve: list[dict[str, Any]] = []
    last: dict[str, Any] = {}
    for row in rows:
        point: dict[str, Any] = {
            "update": int(row.get("update", len(curve) + 1)),
            "completed_episodes": int(row.get("completed_episodes", 0) or 0),
        }
        for key in PPO_FIELDS:
            point[key] = finite_float(row.get(key))
        curve.append(point)
        last = dict(row)
    if len(curve) != expected_count:
        raise ValueError(f"PPO row mismatch: expected {expected_count}, got {len(curve)}")
    return curve, {key: last.get(key) for key in (
        "completed_episodes",
        "update",
        "elapsed_sec",
        "entropy",
        "block4_entropy",
        "k_entropy",
        "policy_loss",
        "value_loss",
        "approx_kl",
        "value_explained_variance_post",
        "lr",
        "stall_update_windows",
        "selected_action_stable_update_windows",
        "converged",
        "strict_revalidation_status",
    )}


METRIC_KEYS = (
    "loss_mean",
    "loss_std",
    "metric1_mean",
    "metric1_std",
    "metric2_mean",
    "metric2_std",
)


def metric_payload(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    payload = {key: finite_float(value.get(key)) for key in METRIC_KEYS}
    payload["trial_count"] = int(value.get("trial_count", 0) or 0)
    limits = value.get("limits")
    payload["limits"] = dict(limits) if isinstance(limits, Mapping) else None
    return payload


def comparison_payload(
    baseline: Mapping[str, Any] | None,
    candidate: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if baseline is None or candidate is None:
        return None
    limits = baseline.get("limits") or {}
    output: dict[str, Any] = {}
    for metric in ("loss", "metric1", "metric2"):
        mean_key = f"{metric}_mean"
        std_key = f"{metric}_std"
        base_mean = finite_float(baseline.get(mean_key))
        best_mean = finite_float(candidate.get(mean_key))
        base_std = finite_float(baseline.get(std_key))
        best_std = finite_float(candidate.get(std_key))
        mean_delta = None if base_mean is None or best_mean is None else best_mean - base_mean
        relative_delta = (
            None
            if mean_delta is None or base_mean is None or base_mean == 0.0
            else 100.0 * mean_delta / abs(base_mean)
        )
        std_ratio = (
            None
            if base_std is None or best_std is None
            else safe_div_or_none(best_std, base_std)
        )
        mean_limit = finite_float(limits.get(metric))
        std_limit = finite_float(limits.get(f"{metric}_std"))
        mean_pass = None
        if best_mean is not None and mean_limit is not None:
            mean_pass = best_mean <= mean_limit if metric == "loss" else best_mean >= mean_limit
        std_pass = None
        if best_std is not None and std_limit is not None:
            std_pass = best_std <= std_limit
        output[metric] = {
            "baseline_mean": base_mean,
            "baseline_std": base_std,
            "candidate_mean": best_mean,
            "candidate_std": best_std,
            "mean_delta": mean_delta,
            "relative_delta_percent": relative_delta,
            "std_ratio": std_ratio,
            "mean_limit": mean_limit,
            "std_limit": std_limit,
            "precision_point_pass": mean_pass,
            "stability_point_pass": std_pass,
        }
    output["all_point_gates_pass"] = all(
        bool(item[key])
        for item in output.values()
        if isinstance(item, Mapping)
        for key in ("precision_point_pass", "stability_point_pass")
    )
    return output


def load_model(audit_row: Mapping[str, Any]) -> dict[str, Any]:
    model = str(audit_row["model"])
    dataset = str(audit_row["dataset"])
    meta = MODEL_META[(model, dataset)]
    commit = str(audit_row["archive_commit"])
    root = str(audit_row["archive_path"])
    progress = f"{root}/small_files/run/stage2_noise/progress"
    summary = git_json(commit, f"{progress}/layerwise_summary.json", default={}) or {}
    status = git_json(commit, f"{progress}/blb_stage2_status.json", default={}) or {}
    manifest = git_json(commit, f"{progress}/layerwise_run_manifest.json", default={}) or {}
    checkpoint = git_json(commit, f"{root}/checkpoint_summary.json", default={}) or {}
    strict_checkpoint = checkpoint.get("strict_best") or {}

    exact_episode = int(
        summary.get("completed_episodes")
        or checkpoint.get("episode")
        or status.get("completed_episodes")
        or 0
    )
    exact_ppo = int(
        summary.get("ppo_update_count")
        or checkpoint.get("ppo_update_count")
        or status.get("ppo_update_count")
        or 0
    )
    planned = int(
        (summary.get("termination") or {}).get("episode_limit")
        or checkpoint.get("planned_total_episodes")
        or status.get("total_episodes")
        or exact_episode
    )

    baseline_reference = summary.get("baseline_reference") or {}
    f4_reference = (
        baseline_reference.get("authoritative_validation_full")
        or (manifest.get("baseline_references") or {}).get("F4")
        or {}
    )
    f1_reference = (
        baseline_reference
        if baseline_reference
        else ((manifest.get("baseline_references") or {}).get("F1") or {}).get("pooled")
        or {}
    )

    bank_b_best = summary.get("bank_b_best") or {}
    strict_pass = bool(summary.get("strict_revalidation_passed", False))
    if strict_pass:
        evidence_level = "strict_final"
        evidence_label = "F4 严格认证"
        evidence_class = "pass"
        baseline_final = metric_payload(f4_reference.get("final_reference_abc"))
    elif bank_b_best or strict_checkpoint:
        evidence_level = "bank_b_candidate"
        evidence_label = "F4 Bank-B 候选"
        evidence_class = "provisional"
        baseline_final = metric_payload(f4_reference.get("promotion_reference_ab"))
    else:
        evidence_level = "baseline_fallback"
        evidence_label = "无合格候选，回退 baseline"
        evidence_class = "fail"
        baseline_final = metric_payload(
            f4_reference.get("final_reference_abc")
            or f4_reference.get("promotion_reference_ab")
        )

    candidate_metrics = metric_payload(
        summary.get("best_metrics") or strict_checkpoint.get("metrics")
    )
    promotion_evidence = (
        summary.get("best_promotion_evidence")
        or strict_checkpoint.get("promotion_evidence")
        or {}
    )
    if candidate_metrics is not None:
        candidate_metrics["trial_count"] = int(
            promotion_evidence.get("trial_count")
            or candidate_metrics.get("trial_count")
            or (45 if strict_pass else 30)
        )

    resource = summary.get("best_resource_objective") or strict_checkpoint
    action_matrix = summary.get("best_action_matrix") or strict_checkpoint.get("action_matrix")
    layer_configs = (
        summary.get("best_layer_configurations")
        or strict_checkpoint.get("layer_configurations")
    )
    assessment = summary.get("best_assessment") or strict_checkpoint.get("assessment")
    convergence = (
        {
            "converged": summary.get("converged"),
            "plateau_ready": summary.get("plateau_ready"),
            "stall_update_windows": summary.get("stall_update_windows"),
            "selected_action_stable_update_windows": summary.get(
                "selected_action_stable_update_windows"
            ),
            "block4_entropy": summary.get("block4_entropy"),
            "k_entropy": summary.get("k_entropy"),
            "strict_revalidation_status": summary.get("strict_revalidation_status"),
            "termination_reason": summary.get("termination_reason"),
        }
        if summary
        else dict(checkpoint.get("convergence_state") or {})
    )

    fixed_gelu = list(manifest.get("fixed_gelu") or audit_row["installed_gelu"])
    fixed_softmax = list(manifest.get("fixed_softmax") or audit_row["installed_softmax"])
    if fixed_gelu != list(audit_row["expected_gelu"]):
        raise ValueError(f"{meta['label']}: Stage-1 GELU mismatch")
    if fixed_softmax != list(audit_row["expected_softmax"]):
        raise ValueError(f"{meta['label']}: Stage-1 Softmax mismatch")

    episodes_path, ppo_path = archive_stream_paths(commit, root)
    episode_curve, episode_stats = summarize_episode_stream(
        iter_git_gzip_jsonl(commit, episodes_path), exact_episode
    )
    ppo_curve, ppo_last = summarize_ppo_stream(
        iter_git_gzip_jsonl(commit, ppo_path), exact_ppo
    )
    elapsed = finite_float(ppo_last.get("elapsed_sec")) or finite_float(status.get("elapsed_sec"))
    throughput = (
        None if elapsed is None or elapsed <= 0.0 else exact_episode * 3600.0 / elapsed
    )

    if layer_configs:
        block4_count = sum(
            int(row.get("block4_fusion_count", 0) or 0) for row in layer_configs
        )
        preset_counts = Counter(
            str(row.get("precision_preset_name", "unknown")) for row in layer_configs
        )
        actual_fusion_total = 2 * len(layer_configs) + block4_count
    else:
        block4_count = 0
        preset_counts = Counter()
        actual_fusion_total = 0

    constraints = {
        "precision_tolerance": finite_float(
            baseline_reference.get("precision_tolerance")
            or ((manifest.get("algorithm_contract") or {}).get("precision_tolerance"))
            or 0.001
        ),
        "stability_multiplier": finite_float(
            baseline_reference.get("stability_multiplier")
            or ((manifest.get("algorithm_contract") or {}).get("stability_multiplier"))
            or 2.0
        ),
        "axis_precision_tolerances": list(
            summary.get("axis_precision_tolerances")
            or (manifest.get("algorithm_contract") or {}).get("axis_precision_tolerances")
            or [0.0005, 0.0005]
        ),
        "online_probability": 0.5,
        "promotion_probability": 0.8,
        "final_probability": 0.95,
        "training_trials_per_episode": 3,
        "baseline_groups_per_bank": 5,
        "trials_per_baseline_group": 3,
    }

    final_evidence = summary.get("final_evidence") or {
        "status": "bank_b_confirmed_not_final_certified" if strict_checkpoint else "no_candidate",
        "bank_a_trial_count": 15,
        "bank_b_trial_count": 15,
        "bank_c_trial_count": 15,
        "pooled_final_trial_count": 45,
        "note": "Recovered from the exact graceful-stop checkpoint boundary.",
    }

    return {
        "model": model,
        "dataset": dataset,
        "label": meta["label"],
        "family": meta["family"],
        "metric1_name": meta["metric1"],
        "metric2_name": meta["metric2"],
        "profile": manifest.get("profile"),
        "archive": {
            "branch": audit_row["archive_branch"],
            "commit": commit,
            "path": root,
            "restore_status": audit_row["restore_status"],
            "sha256_manifest_file_set_complete": audit_row[
                "sha256_manifest_file_set_complete"
            ],
        },
        "training": {
            "episodes": exact_episode,
            "planned_episodes": planned,
            "ppo_updates": exact_ppo,
            "status": summary.get("status") or "graceful_stop_checkpoint",
            "termination_reason": convergence.get("termination_reason"),
            "elapsed_sec": elapsed,
            "throughput_episodes_per_hour": throughput,
            "policy_network_variant": (
                summary.get("policy_network_variant")
                or checkpoint.get("policy_network_variant")
                or manifest.get("policy_network_variant")
            ),
            "policy_network": summary.get("policy_network") or manifest.get("policy_network"),
            "rl_variant": summary.get("rl_variant") or manifest.get("rl_variant"),
            "algorithm_revision": (
                summary.get("algorithm_revision")
                or checkpoint.get("algorithm_revision")
                or manifest.get("algorithm_revision")
            ),
            "learning_rate_final": finite_float(ppo_last.get("lr")),
            "convergence": convergence,
        },
        "stage1": {
            "gelu": fixed_gelu,
            "softmax": fixed_softmax,
            "exact_match_authoritative_html": True,
            "report_sha256": audit_row["stage1_report_sha256"],
        },
        "constraints": constraints,
        "validation": {
            "split": f4_reference.get("split", "validation_full"),
            "example_count": int(
                f4_reference.get("example_count")
                or ((manifest.get("evidence_tiers") or {}).get("F4") or {}).get(
                    "example_count", 0
                )
                or 0
            ),
            "baseline_reference": baseline_final,
            "candidate_metrics": candidate_metrics,
            "comparison": comparison_payload(baseline_final, candidate_metrics),
            "assessment": assessment,
            "evidence_level": evidence_level,
            "evidence_label": evidence_label,
            "evidence_class": evidence_class,
            "final_evidence": final_evidence,
            "training_probe_baseline": metric_payload(f1_reference),
        },
        "result": {
            "best_reward": finite_float(summary.get("best_reward") or strict_checkpoint.get("reward")),
            "resource": {
                key: resource.get(key)
                for key in (
                    "compute_saving",
                    "communication_saving",
                    "robust_floor",
                    "secondary_progress",
                    "ppo_resource_score",
                    "fusion_count",
                    "removed_k_bits",
                )
            },
            "action_matrix": action_matrix,
            "layer_configurations": layer_configs,
            "block4_fusion_count": block4_count,
            "actual_b2_b4_b5_fusion_total": actual_fusion_total,
            "precision_preset_counts": dict(preset_counts),
            "baseline_fallback": evidence_level == "baseline_fallback",
        },
        "curves": {
            "episodes": episode_curve,
            "ppo": ppo_curve,
        },
        "stream_summary": {
            "episodes": episode_stats,
            "ppo_last": ppo_last,
        },
    }


def fmt_metric(mean: Any, std: Any) -> str:
    if finite_float(mean) is None:
        return "-"
    return f"{float(mean):.6f} ± {float(std or 0.0):.6f}"


def fmt_percent(value: Any, *, signed: bool = False) -> str:
    numeric = finite_float(value)
    if numeric is None:
        return "-"
    return f"{numeric:+.2f}%" if signed else f"{numeric:.2f}%"


def badge(text: str, kind: str) -> str:
    return f'<span class="badge {html.escape(kind)}">{html.escape(text)}</span>'


def td_code(value: Any) -> str:
    return f"<code>{html.escape(str(value))}</code>"


def chart_svg(
    title: str,
    points: Sequence[Mapping[str, Any]],
    x_key: str,
    series: Sequence[tuple[str, str, str]],
    *,
    baselines: Sequence[tuple[str, float | None, str]] = (),
    y_label: str = "",
) -> str:
    width, height = 900, 280
    left, right, top, bottom = 66, 22, 38, 44
    plot_w, plot_h = width - left - right, height - top - bottom
    xs = [finite_float(row.get(x_key)) for row in points]
    xs = [value for value in xs if value is not None]
    values: list[float] = []
    for _, key, _ in series:
        values.extend(
            value
            for value in (finite_float(row.get(key)) for row in points)
            if value is not None
        )
    values.extend(value for _, value, _ in baselines if value is not None)
    if not xs or not values:
        return (
            '<div class="chart-frame"><div class="chart-title">'
            f"{html.escape(title)}</div><div class=\"empty\">无可用曲线数据</div></div>"
        )
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(values), max(values)
    if x_max == x_min:
        x_max = x_min + 1.0
    span = y_max - y_min
    pad = max(span * 0.10, max(abs(y_min), abs(y_max), 1.0) * 0.015)
    y_min -= pad
    y_max += pad
    if y_max == y_min:
        y_max = y_min + 1.0

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    parts = [
        '<div class="chart-frame">',
        f'<div class="chart-title">{html.escape(title)}</div>',
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" class="plot-bg"/>',
    ]
    for idx in range(5):
        ratio = idx / 4.0
        y = top + ratio * plot_h
        value = y_max - ratio * (y_max - y_min)
        parts.append(f'<line x1="{left}" x2="{left + plot_w}" y1="{y:.2f}" y2="{y:.2f}" class="grid"/>')
        parts.append(f'<text x="{left - 9}" y="{y + 4:.2f}" text-anchor="end" class="tick">{value:.4g}</text>')
    for idx in range(5):
        ratio = idx / 4.0
        x = left + ratio * plot_w
        value = x_min + ratio * (x_max - x_min)
        parts.append(f'<text x="{x:.2f}" y="{height - 18}" text-anchor="middle" class="tick">{value:,.0f}</text>')
    for label, value, color in baselines:
        if value is None:
            continue
        y = sy(float(value))
        parts.append(f'<line x1="{left}" x2="{left + plot_w}" y1="{y:.2f}" y2="{y:.2f}" stroke="{color}" class="baseline-line"/>')
    for label, key, color in series:
        coords = []
        for row in points:
            x_value = finite_float(row.get(x_key))
            y_value = finite_float(row.get(key))
            if x_value is None or y_value is None:
                continue
            coords.append(f"{sx(x_value):.2f},{sy(y_value):.2f}")
        if coords:
            parts.append(f'<polyline points="{" ".join(coords)}" fill="none" stroke="{color}" class="series-line"/>')
    parts.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 2}" text-anchor="middle" class="axis-label">Episode</text>')
    if y_label:
        parts.append(f'<text transform="translate(15 {top + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle" class="axis-label">{html.escape(y_label)}</text>')
    parts.append("</svg><div class=\"legend\">")
    for label, _, color in series:
        parts.append(f'<span><i style="background:{color}"></i>{html.escape(label)}</span>')
    for label, value, color in baselines:
        if value is not None:
            parts.append(f'<span><i class="dash" style="border-color:{color}"></i>{html.escape(label)}</span>')
    parts.append("</div></div>")
    return "".join(parts)


def metric_table(model: Mapping[str, Any]) -> str:
    validation = model["validation"]
    baseline = validation["baseline_reference"] or {}
    candidate = validation["candidate_metrics"] or {}
    comparison = validation["comparison"] or {}
    names = {
        "loss": "Loss",
        "metric1": model["metric1_name"],
        "metric2": model["metric2_name"],
    }
    rows = []
    row_classes = []
    for key in ("loss", "metric1", "metric2"):
        item = comparison.get(key) or {}
        if candidate:
            pass_text = "PASS" if (
                item.get("precision_point_pass") and item.get("stability_point_pass")
            ) else "FAIL"
            pass_kind = "pass" if pass_text == "PASS" else "fail"
        else:
            pass_text, pass_kind = "无候选", "muted"
        rows.append([
            names[key],
            fmt_metric(baseline.get(f"{key}_mean"), baseline.get(f"{key}_std")),
            fmt_metric(candidate.get(f"{key}_mean"), candidate.get(f"{key}_std")) if candidate else "-",
            format_float(item.get("mean_delta"), digits=6, none_text="-"),
            fmt_percent(item.get("relative_delta_percent"), signed=True),
            format_float(item.get("std_ratio"), digits=3, none_text="-"),
            badge(pass_text, pass_kind),
        ])
        row_classes.append(pass_kind)
    return html_table(
        ["指标", "Baseline mean ± std", "候选 mean ± std", "Δ mean", "相对变化", "std 倍率", "点门禁"],
        rows,
        allow_html_cells=True,
        row_classes=row_classes,
        table_attrs='class="metric-table"',
    )


def config_table(model: Mapping[str, Any]) -> str:
    configs = model["result"]["layer_configurations"]
    if not configs:
        return (
            '<div class="notice fail-note"><b>最终配置：</b>没有 Bank-B 合格候选，'
            "训练输出回退到 baseline（B2/B4/B5 fusion 均为 0，truncation K 均为 13）。</div>"
        )
    rows = []
    for row in configs:
        k = row.get("truncation_k_by_block") or {}
        preset = str(row.get("precision_preset_name", "unknown"))
        rows.append([
            int(row.get("layer_idx", 0)) + 1,
            badge(str(row.get("block4_fusion_count", 0)), "fusion-on" if row.get("block4_fusion_count") else "muted"),
            badge(preset.upper(), f"preset-{preset}"),
            k.get("block1"), k.get("block2"), k.get("block3"), k.get("block4"), k.get("block5"),
        ])
    return html_table(
        ["层", "Block4 fusion", "精度预设", "B1 K", "B2 K", "B3 K", "B4 K", "B5 K"],
        rows,
        allow_html_cells=True,
        table_attrs='class="config-table"',
    )


def model_section(model: Mapping[str, Any]) -> str:
    training = model["training"]
    validation = model["validation"]
    result = model["result"]
    stream = model["stream_summary"]
    ppo_last = stream["ppo_last"]
    resource = result["resource"]
    probe_baseline = validation.get("training_probe_baseline") or {}
    episode_curve = model["curves"]["episodes"]
    ppo_curve = model["curves"]["ppo"]
    stage1_gelu = ", ".join(str(value) for value in model["stage1"]["gelu"])
    stage1_softmax = ", ".join(str(value) for value in model["stage1"]["softmax"])
    priority = stream["episodes"]["priority_counts"]
    preset_counts = result["precision_preset_counts"]

    charts = [
        chart_svg(
            "Reward 曲线（F1 训练探针，分桶均值）",
            episode_curve,
            "episode",
            [
                ("Episode reward", "total_reward", "#2563eb"),
                ("Best reward", "best_reward_so_far", "#059669"),
            ],
            y_label="Reward",
        ),
        chart_svg(
            "策略熵（PPO update）",
            ppo_curve,
            "completed_episodes",
            [
                ("Block4 fusion entropy", "block4_entropy", "#7c3aed"),
                ("Truncation preset entropy", "k_entropy", "#0f766e"),
            ],
            y_label="Entropy",
        ),
        chart_svg(
            "Loss 曲线（F1 训练探针）",
            episode_curve,
            "episode",
            [("Loss", "terminal_loss_mean", "#dc2626")],
            baselines=[("F1 baseline", finite_float(probe_baseline.get("loss_mean")), "#64748b")],
            y_label="Loss",
        ),
        chart_svg(
            "M1 / M2 曲线（F1 训练探针）",
            episode_curve,
            "episode",
            [
                (model["metric1_name"], "terminal_metric1_mean", "#059669"),
                (model["metric2_name"], "terminal_metric2_mean", "#d97706"),
            ],
            baselines=[
                ("M1 baseline", finite_float(probe_baseline.get("metric1_mean")), "#64748b"),
                ("M2 baseline", finite_float(probe_baseline.get("metric2_mean")), "#94a3b8"),
            ],
            y_label="Metric",
        ),
    ]

    return f"""
<section id="{model['model']}-{model['dataset']}" class="model-section">
  <div class="section-head">
    <div>
      <div class="eyebrow">{html.escape(model['family'])} · {html.escape(model['dataset'].upper())}</div>
      <h2>{html.escape(model['label'])}</h2>
    </div>
    {badge(validation['evidence_label'], validation['evidence_class'])}
  </div>
  <div class="summary-grid">
    <div><span>训练边界</span><strong>{training['episodes']:,} / {training['planned_episodes']:,}</strong><small>{training['ppo_updates']} PPO updates</small></div>
    <div><span>训练吞吐</span><strong>{format_float(training['throughput_episodes_per_hour'], digits=1, none_text='-')} ep/h</strong><small>{format_elapsed(training['elapsed_sec'] or 0)}</small></div>
    <div><span>Best reward</span><strong>{format_float(result['best_reward'], digits=6, none_text='-')}</strong><small>训练/候选选择 reward</small></div>
    <div><span>资源目标</span><strong>{fmt_percent(100.0 * float(resource.get('ppo_resource_score') or 0.0)) if result['layer_configurations'] else '-'}</strong><small>compute / communication = {format_float(resource.get('compute_saving'), digits=3, none_text='-')} / {format_float(resource.get('communication_saving'), digits=3, none_text='-')}</small></div>
    <div><span>最终熵</span><strong>{format_float(ppo_last.get('block4_entropy'), digits=4, none_text='-')} / {format_float(ppo_last.get('k_entropy'), digits=4, none_text='-')}</strong><small>Block4 / truncation preset</small></div>
    <div><span>F4 全验证集</span><strong>{validation['example_count']:,} examples</strong><small>candidate trials: {(validation.get('candidate_metrics') or {}).get('trial_count', 0)}</small></div>
  </div>

  <h3>最终指标与约束</h3>
  <p class="context-line">精度约束 0.1%，稳定性上限为 baseline std 的 200%；表中候选和 baseline 使用相同 F4 validation_full 银行口径。</p>
  <div class="table-wrap">{metric_table(model)}</div>

  <h3>训练过程</h3>
  <div class="chart-grid">{''.join(charts)}</div>
  <div class="run-facts">
    <span>P1/P2/P3：{priority.get('P1', 0):,} / {priority.get('P2', 0):,} / {priority.get('P3', 0):,}</span>
    <span>Invalid episodes：{stream['episodes']['invalid_episodes']:,}</span>
    <span>Reward mean ± std：{format_float(stream['episodes']['reward']['mean'], digits=4)} ± {format_float(stream['episodes']['reward']['std'], digits=4)}</span>
    <span>LR：{format_float(training['learning_rate_final'], digits=6)}</span>
  </div>

  <h3>最终动作配置</h3>
  <div class="action-summary">
    <span>Block4 fusion=1：<b>{result['block4_fusion_count']}</b> / {len(model['stage1']['gelu'])}</span>
    <span>实际 B2+B4+B5 fusion 总数：<b>{result['actual_b2_b4_b5_fusion_total']}</b></span>
    <span>H/M/L：<b>{preset_counts.get('high', 0)} / {preset_counts.get('medium', 0)} / {preset_counts.get('low', 0)}</b></span>
    <span>减少 K bits：<b>{resource.get('removed_k_bits') if resource.get('removed_k_bits') is not None else '-'}</b></span>
  </div>
  <div class="table-wrap">{config_table(model)}</div>

  <details>
    <summary>Stage-1 前置配置与运行证据</summary>
    <dl class="evidence-list">
      <dt>GELU</dt><dd><code>[{html.escape(stage1_gelu)}]</code></dd>
      <dt>Softmax</dt><dd><code>[{html.escape(stage1_softmax)}]</code></dd>
      <dt>Stage-1 核对</dt><dd>{badge('与权威 HTML 精确一致', 'pass')}</dd>
      <dt>策略网络</dt><dd><code>{html.escape(str(training['policy_network_variant']))}</code></dd>
      <dt>终止状态</dt><dd><code>{html.escape(str(training['status']))}</code>；converged={html.escape(str((training['convergence'] or {}).get('converged')))}</dd>
      <dt>证据状态</dt><dd><code>{html.escape(str((validation['final_evidence'] or {}).get('status')))}</code></dd>
      <dt>归档</dt><dd><code>{html.escape(model['archive']['branch'])}@{html.escape(model['archive']['commit'][:12])}</code><br><code>{html.escape(model['archive']['path'])}</code></dd>
    </dl>
  </details>
</section>
"""


def build_html(payload: Mapping[str, Any]) -> str:
    models = payload["models"]
    strict_count = sum(model["validation"]["evidence_level"] == "strict_final" for model in models)
    provisional_count = sum(
        model["validation"]["evidence_level"] == "bank_b_candidate" for model in models
    )
    fallback_count = sum(
        model["validation"]["evidence_level"] == "baseline_fallback" for model in models
    )
    total_episodes = sum(model["training"]["episodes"] for model in models)
    total_updates = sum(model["training"]["ppo_updates"] for model in models)

    overview_rows = []
    overview_classes = []
    for model in models:
        training = model["training"]
        validation = model["validation"]
        result = model["result"]
        resource = result["resource"]
        overview_rows.append([
            f'<a href="#{model["model"]}-{model["dataset"]}">{html.escape(model["label"])}</a>',
            f"{training['episodes']:,}",
            f"{training['ppo_updates']:,}",
            badge(validation["evidence_label"], validation["evidence_class"]),
            format_float(result["best_reward"], digits=4, none_text="-"),
            format_float(resource.get("compute_saving"), digits=3, none_text="-"),
            format_float(resource.get("communication_saving"), digits=3, none_text="-"),
            str(result["block4_fusion_count"]) if result["layer_configurations"] else "-",
            str(resource.get("removed_k_bits")) if resource.get("removed_k_bits") is not None else "-",
            format_float(model["stream_summary"]["ppo_last"].get("block4_entropy"), digits=3, none_text="-"),
            format_float(model["stream_summary"]["ppo_last"].get("k_entropy"), digits=3, none_text="-"),
        ])
        overview_classes.append(validation["evidence_class"])
    overview = html_table(
        ["模型", "Episodes", "PPO", "最终证据", "Best reward", "Compute", "Comm", "B4=1", "Removed K", "B4 H", "K H"],
        overview_rows,
        allow_html_cells=True,
        row_classes=overview_classes,
        table_attrs='class="overview-table"',
    )

    nav = "".join(
        f'<a href="#{model["model"]}-{model["dataset"]}">{html.escape(model["label"])}</a>'
        for model in models
    )
    sections = "".join(model_section(model) for model in models)
    source_rows = [
        [model["label"], td_code(model["archive"]["branch"]), td_code(model["archive"]["commit"]), td_code(model["archive"]["path"]), model["archive"]["restore_status"]]
        for model in models
    ]
    sources = html_table(
        ["模型", "归档分支", "Commit", "归档路径", "恢复校验"],
        source_rows,
        allow_html_cells=True,
        table_attrs='class="source-table"',
    )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stage-2 RL 六模型最终汇总全版报告</title>
<style>
:root {{ color-scheme: light; --ink:#172033; --muted:#667085; --line:#d8dee9; --surface:#ffffff; --band:#f5f7fa; --pass:#087f5b; --warn:#a15c00; --fail:#b42318; }}
* {{ box-sizing:border-box; }}
html {{ scroll-behavior:smooth; }}
body {{ margin:0; background:#eef1f5; color:var(--ink); font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; letter-spacing:0; }}
a {{ color:#175cd3; text-decoration:none; }} a:hover {{ text-decoration:underline; }}
header {{ background:#fff; border-bottom:1px solid var(--line); }}
.masthead {{ max-width:1480px; margin:0 auto; padding:36px 32px 28px; }}
.eyebrow {{ color:#475467; font-size:12px; font-weight:700; text-transform:uppercase; }}
h1 {{ margin:6px 0 8px; font-size:30px; line-height:1.2; }}
.subtitle {{ max-width:960px; margin:0; color:var(--muted); font-size:15px; }}
.nav {{ position:sticky; top:0; z-index:10; display:flex; gap:4px; overflow:auto; padding:8px max(20px,calc((100vw - 1480px)/2 + 32px)); background:#fff; border-bottom:1px solid var(--line); }}
.nav a {{ flex:none; padding:7px 10px; color:#344054; font-size:12px; font-weight:600; border-bottom:2px solid transparent; }}
.nav a:hover {{ border-color:#175cd3; text-decoration:none; }}
main {{ max-width:1480px; margin:0 auto; background:#fff; }}
.executive {{ padding:28px 32px 34px; background:var(--band); border-bottom:1px solid var(--line); }}
.kpis {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:1px; margin:18px 0 24px; background:var(--line); border:1px solid var(--line); }}
.kpis div {{ min-width:0; padding:16px; background:#fff; }}
.kpis span,.summary-grid span {{ display:block; color:var(--muted); font-size:12px; }}
.kpis strong,.summary-grid strong {{ display:block; margin-top:3px; font-size:21px; line-height:1.2; }}
.kpis small,.summary-grid small {{ display:block; margin-top:5px; color:var(--muted); }}
.model-section {{ padding:34px 32px 42px; border-bottom:1px solid var(--line); scroll-margin-top:54px; }}
.model-section:nth-of-type(even) {{ background:#fbfcfd; }}
.section-head {{ display:flex; align-items:flex-start; justify-content:space-between; gap:20px; margin-bottom:20px; }}
h2 {{ margin:3px 0 0; font-size:24px; }} h3 {{ margin:28px 0 8px; font-size:16px; }}
.summary-grid {{ display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:1px; border:1px solid var(--line); background:var(--line); }}
.summary-grid>div {{ min-width:0; padding:13px; background:#fff; }} .summary-grid strong {{ font-size:16px; }}
.badge {{ display:inline-flex; align-items:center; min-height:24px; padding:3px 8px; border:1px solid transparent; border-radius:4px; font-size:12px; font-weight:700; white-space:nowrap; }}
.badge.pass {{ color:#05603f; background:#ecfdf3; border-color:#abefc6; }}
.badge.provisional {{ color:#854a0e; background:#fffaeb; border-color:#fedf89; }}
.badge.fail {{ color:#912018; background:#fef3f2; border-color:#fecdca; }}
.badge.muted {{ color:#475467; background:#f2f4f7; border-color:#d0d5dd; }}
.badge.fusion-on {{ color:#174ea6; background:#eff4ff; border-color:#b2ccff; }}
.badge.preset-high {{ color:#344054; background:#f2f4f7; border-color:#d0d5dd; }}
.badge.preset-medium {{ color:#854a0e; background:#fffaeb; border-color:#fedf89; }}
.badge.preset-low {{ color:#05603f; background:#ecfdf3; border-color:#abefc6; }}
.table-wrap {{ width:100%; overflow:auto; border:1px solid var(--line); background:#fff; }}
table {{ width:100%; border-collapse:collapse; font-size:12px; }}
th {{ position:sticky; top:0; z-index:1; padding:9px 10px; background:#eaecf0; color:#344054; text-align:left; white-space:nowrap; }}
td {{ padding:8px 10px; border-top:1px solid #eaecf0; vertical-align:top; }}
tr:hover td {{ background:#f9fafb; }} code {{ font:11px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace; word-break:break-all; }}
.context-line {{ margin:0 0 12px; color:var(--muted); }}
.chart-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; }}
.chart-frame {{ min-width:0; padding:10px 12px 8px; border:1px solid var(--line); background:#fff; }}
.chart-title {{ margin-bottom:2px; font-size:12px; font-weight:700; }}
.chart-frame svg {{ display:block; width:100%; aspect-ratio:900/280; min-height:190px; }}
.plot-bg {{ fill:#fbfcfd; stroke:#d8dee9; }} .grid {{ stroke:#e4e7ec; stroke-width:1; }}
.tick,.axis-label {{ fill:#667085; font-size:11px; }} .series-line {{ stroke-width:2; vector-effect:non-scaling-stroke; }}
.baseline-line {{ stroke-width:1.3; stroke-dasharray:5 4; vector-effect:non-scaling-stroke; }}
.legend {{ display:flex; flex-wrap:wrap; gap:12px; min-height:20px; color:#475467; font-size:11px; }}
.legend span {{ display:inline-flex; align-items:center; gap:5px; }} .legend i {{ width:16px; height:3px; }} .legend i.dash {{ height:0; border-top:2px dashed; background:none!important; }}
.run-facts,.action-summary {{ display:flex; flex-wrap:wrap; gap:8px 18px; margin-top:10px; padding:10px 12px; background:#f8fafc; border-left:3px solid #98a2b3; color:#475467; font-size:12px; }}
.notice {{ padding:12px 14px; border:1px solid; }} .fail-note {{ color:#912018; background:#fef3f2; border-color:#fecdca; }}
details {{ margin-top:18px; border-top:1px solid var(--line); }} summary {{ cursor:pointer; padding:13px 0 6px; font-weight:700; }}
.evidence-list {{ display:grid; grid-template-columns:150px 1fr; gap:8px 14px; margin:10px 0 0; }} .evidence-list dt {{ color:var(--muted); }} .evidence-list dd {{ min-width:0; margin:0; }}
.appendix {{ padding:34px 32px 44px; background:var(--band); }}
.method {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; margin:14px 0 28px; }}
.method div {{ padding:14px; border-left:3px solid #667085; background:#fff; }} .method b {{ display:block; margin-bottom:4px; }}
.empty {{ padding:70px 20px; color:var(--muted); text-align:center; }}
footer {{ max-width:1480px; margin:0 auto; padding:18px 32px 28px; color:var(--muted); font-size:11px; }}
@media (max-width:1100px) {{ .summary-grid {{ grid-template-columns:repeat(3,minmax(0,1fr)); }} .kpis {{ grid-template-columns:repeat(3,minmax(0,1fr)); }} }}
@media (max-width:760px) {{ .masthead,.executive,.model-section,.appendix {{ padding-left:16px; padding-right:16px; }} h1 {{ font-size:24px; }} .kpis,.summary-grid,.method {{ grid-template-columns:1fr 1fr; }} .chart-grid {{ grid-template-columns:1fr; }} .section-head {{ display:block; }} .section-head>.badge {{ margin-top:10px; }} .evidence-list {{ grid-template-columns:1fr; }} }}
@media print {{ body {{ background:#fff; }} .nav {{ display:none; }} main {{ max-width:none; }} .model-section {{ break-before:page; }} details {{ display:block; }} details>* {{ display:block; }} }}
</style>
</head>
<body>
<header><div class="masthead">
  <div class="eyebrow">Reinforcement-For-Robustness · Stage-2 RL</div>
  <h1>六模型最终汇总全版报告</h1>
  <p class="subtitle">BERT-base / BERT-large × MRPC / RTE / SST-2。最终指标来自 validation_full F4 银行；训练曲线来自各自完整归档的 F1 在线探针流。所有配置与结果均可追溯到 Git 云端归档。</p>
</div></header>
<nav class="nav"><a href="#overview">总览</a>{nav}<a href="#appendix">审计附录</a></nav>
<main>
<section id="overview" class="executive">
  <h2>结果总览</h2>
  <div class="kpis">
    <div><span>模型数</span><strong>6</strong><small>Base / Large × 3 datasets</small></div>
    <div><span>总 Episodes</span><strong>{total_episodes:,}</strong><small>{total_updates:,} PPO updates</small></div>
    <div><span>严格认证</span><strong>{strict_count}</strong><small>通过 Bank A+B+C</small></div>
    <div><span>Bank-B 候选</span><strong>{provisional_count}</strong><small>尚未完成 Bank-C 最终认证</small></div>
    <div><span>Baseline 回退</span><strong>{fallback_count}</strong><small>无 Bank-B 合格候选</small></div>
  </div>
  <div class="table-wrap">{overview}</div>
</section>
{sections}
<section id="appendix" class="appendix">
  <h2>口径与审计附录</h2>
  <div class="method">
    <div><b>训练信号 F1</b>256 个分层 probe 样本，每个 episode 3 次噪声 trial。曲线只描述 PPO 实际看到的训练过程。</div>
    <div><b>候选门禁 F4</b>验证集全集；Bank A/B/C 各 5 组 × 3 trials。严格结果使用 45 trials，Bank-B 候选使用 30 trials。</div>
    <div><b>约束与资源</b>精度 0.1%，稳定性 200%；compute 与 communication 权重 1:1，各分配 0.05% 精度预算。</div>
  </div>
  <h3>证据等级说明</h3>
  <p><b>F4 严格认证</b>表示候选通过完整 A+B+C 点门禁和 compute-only / communication-only 反事实门禁。<b>Bank-B 候选</b>是已通过 A+B 两银行的可用候选，但未完成 Bank-C 最终认证。概率值只用于诊断/确定性排序，硬判定依据六个 point gate。</p>
  <h3>Git 云端来源</h3>
  <div class="table-wrap">{sources}</div>
</section>
</main>
<footer>生成时间：{html.escape(payload['generated_at'])} · 统一备份审计：<code>{AUDIT_COMMIT}</code> · 报告数据：<code>{REPORT_JSON.name}</code></footer>
</body></html>"""


def main() -> None:
    audit = git_json(AUDIT_COMMIT, AUDIT_PATH)
    if not isinstance(audit, Mapping) or audit.get("status") != "PASS":
        raise RuntimeError("six-model backup audit is not PASS")
    audit_models = audit.get("models") or []
    if len(audit_models) != 6:
        raise RuntimeError(f"expected six models, got {len(audit_models)}")
    models = [load_model(row) for row in audit_models]
    payload = {
        "schema_version": "stage2_six_model_final_full_report_v1",
        "generated_at": __import__("datetime").datetime.now().astimezone().isoformat(),
        "audit_commit": AUDIT_COMMIT,
        "audit_status": audit.get("status"),
        "authoritative_stage1_reports": audit.get("authoritative_stage1_reports"),
        "models": models,
    }
    write_json_file(REPORT_JSON, payload)
    REPORT_HTML.write_text(build_html(payload), encoding="utf-8")
    print(f"wrote {REPORT_JSON}")
    print(f"wrote {REPORT_HTML}")


if __name__ == "__main__":
    main()
