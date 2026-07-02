#!/usr/bin/env python3
"""Summarize GPU utilization evidence from RL run artifacts.

The script is intentionally dependency-free. It reads structured episode
diagnostics plus optional nvidia-smi CSV samples and reports whether visible
GPUs were actually exercised by terminal reward probes.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

LOW_UTIL_THRESHOLD_PCT = 10.0
HOT_PATH_TIMING_FIELDS = (
    "jsonl_write_wall_seconds",
    "report_render_wall_seconds",
    "diagnostics_write_wall_seconds",
    "episode_callback_wall_seconds",
)
REPLAN_TIMING_FIELDS = (
    "replan_wall_seconds",
    "per_step_optimizer_wall_seconds",
    "rejection_optimizer_wall_seconds",
    "terminal_cost_eval_wall_seconds",
)


def _device_sort_key(device: str) -> tuple[int, int | str]:
    match = re.fullmatch(r"cuda:(\d+)", device)
    if match:
        return (0, int(match.group(1)))
    return (1, device)


def normalize_device_token(value: object) -> str:
    text = str(value).strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered in {"none", "null", "nil", "-1"}:
        return ""
    if lowered == "cpu":
        return "cpu"
    if lowered.startswith("cuda:"):
        suffix = lowered.split("cuda:", 1)[1].strip()
        return f"cuda:{suffix}" if suffix else ""
    if lowered.isdigit():
        return f"cuda:{lowered}"
    return text


def parse_device_spec(value: str | Sequence[object] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items: Iterable[object] = value.split(",")
    else:
        raw_items = value
    devices = [normalize_device_token(item) for item in raw_items]
    return [device for device in devices if device]


def _find_episodes_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    candidates = [
        candidate / "episodes.jsonl",
        candidate / "diagnostics" / "episodes.jsonl",
        candidate / "progress" / "diagnostics" / "episodes.jsonl",
    ]
    for item in candidates:
        if item.is_file():
            return item
    for dirpath, dirnames, filenames in os.walk(candidate):
        dirnames.sort()
        filenames.sort()
        if "episodes.jsonl" in filenames:
            item = Path(dirpath) / "episodes.jsonl"
            if item.is_file():
                return item
    raise FileNotFoundError(f"could not find episodes.jsonl under {candidate}")


def _iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    episodes_path = _find_episodes_path(path)
    with episodes_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{episodes_path}:{line_no}: invalid JSON") from exc
            if isinstance(row, dict):
                yield row


def _float_value(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text)
    if not match:
        return None
    return float(match.group(0))


def _int_list(value: object) -> list[int]:
    if not isinstance(value, list):
        return []
    out: list[int] = []
    for item in value:
        parsed = _float_value(item)
        if parsed is not None:
            out.append(int(parsed))
    return out


class _RunningStats:
    __slots__ = ("count", "max_value", "min_value", "total")

    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.min_value: float | None = None
        self.max_value: float | None = None

    def add(self, value: float) -> None:
        number = float(value)
        self.count += 1
        self.total += number
        min_value = self.min_value
        max_value = self.max_value
        self.min_value = number if min_value is None or number < min_value else min_value
        self.max_value = number if max_value is None or number > max_value else max_value

    def as_dict(self) -> dict[str, float | int | None]:
        if not self.count:
            return {"count": 0, "mean": None, "min": None, "max": None}
        return {
            "count": self.count,
            "mean": self.total / float(self.count),
            "min": self.min_value,
            "max": self.max_value,
        }


def _per_device_probe_walls(
    row: Mapping[str, Any],
    devices: Sequence[str],
    fallback_wall_seconds: float | None,
) -> dict[str, float]:
    mapped = row.get("terminal_probe_wall_seconds_by_device")
    if isinstance(mapped, Mapping):
        out: dict[str, float] = {}
        for raw_device, raw_value in mapped.items():
            device = normalize_device_token(raw_device)
            value = _float_value(raw_value)
            if device and value is not None:
                out[device] = value
        if out:
            return out

    listed = row.get("terminal_probe_device_wall_seconds")
    if isinstance(listed, list) and len(listed) == len(devices):
        out = {}
        for device, raw_value in zip(devices, listed):  # noqa: B905 - lengths checked above for py39.
            value = _float_value(raw_value)
            if value is not None:
                out[device] = value
        if out:
            return out

    if fallback_wall_seconds is None:
        return {}
    return {device: fallback_wall_seconds for device in devices}


def _normalized_fieldnames(row: Mapping[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in row.items():
        normalized = re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_")
        out[normalized] = value
    return out


def _first_present(row: Mapping[str, str], keys: Sequence[str]) -> str | None:
    for key in keys:
        if key in row:
            return row[key]
    return None


def _first_float(row: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        value = _float_value(row.get(key))
        if value is not None:
            return value
    return None


def _load_nvidia_smi_csv(path: str | Path | None) -> dict[str, dict[str, float | int]]:
    if not path:
        return {}
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"nvidia-smi CSV not found: {csv_path}")
    samples: dict[str, dict[str, float | int]] = collections.defaultdict(
        lambda: {
            "samples": 0,
            "util_sum": 0.0,
            "max_util_pct": 0.0,
            "active_samples": 0,
            "max_memory_mib": 0.0,
        }
    )
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            row = _normalized_fieldnames(raw_row)
            idx = _first_present(row, ["index", "gpu_idx", "gpu_index", "gpu"])
            util = _first_present(
                row,
                [
                    "utilization_gpu",
                    "utilization_gpu_pct",
                    "util_pct",
                    "gpu_util_pct",
                    "gpu_util",
                ],
            )
            mem = _first_present(
                row,
                [
                    "memory_used",
                    "memory_used_mib",
                    "mem_used_mib",
                    "memory_used_mi_b",
                    "memory_used_mib",
                ],
            )
            device = normalize_device_token(idx) if idx is not None else ""
            util_pct = _float_value(util)
            mem_mib = _float_value(mem)
            if device and util_pct is not None:
                stats = samples[device]
                stats["samples"] = int(stats["samples"]) + 1
                stats["util_sum"] = float(stats["util_sum"]) + float(util_pct)
                stats["max_util_pct"] = max(float(stats["max_util_pct"]), float(util_pct))
                if float(util_pct) > 0.0:
                    stats["active_samples"] = int(stats["active_samples"]) + 1
                if mem_mib is not None:
                    stats["max_memory_mib"] = max(float(stats["max_memory_mib"]), float(mem_mib))
    summary: dict[str, dict[str, float | int]] = {}
    for device in sorted(samples, key=_device_sort_key):
        stats = samples[device]
        sample_count = int(stats["samples"])
        summary[device] = {
            "samples": sample_count,
            "mean_util_pct": float(stats["util_sum"]) / float(sample_count),
            "max_util_pct": float(stats["max_util_pct"]),
            "active_sample_rate": float(stats["active_samples"]) / float(sample_count),
            "max_memory_mib": float(stats["max_memory_mib"]),
        }
    return summary


def summarize_rows(
        rows: Iterable[Mapping[str, Any]],
        *,
        gpu_utilization: Mapping[str, Mapping[str, float | int]] | None = None,
        visible_devices: str | Sequence[object] | None = None,
        low_util_threshold_pct: float = LOW_UTIL_THRESHOLD_PCT,
        ) -> dict[str, Any]:
    gpu_utilization = dict(gpu_utilization or {})
    used_devices: set[str] = set()
    device_sets: set[tuple[str, ...]] = set()
    trial_splits: set[tuple[int, ...]] = set()
    trial_counts: collections.Counter[str] = collections.Counter()
    warnings: list[str] = []
    recommendations: list[str] = []
    terminal_probe_wall = _RunningStats()
    policy_rollout_wall = _RunningStats()
    replan_wall = _RunningStats()
    probe_episode_counts: collections.Counter[str] = collections.Counter()
    probe_wall_by_device: dict[str, _RunningStats] = collections.defaultdict(_RunningStats)
    hot_path_timings: dict[str, _RunningStats] = collections.defaultdict(_RunningStats)
    mismatched_trial_rows = 0
    episode_count = 0

    for row in rows:
        episode_count += 1
        devices = parse_device_spec(row.get("terminal_probe_devices") if isinstance(row, Mapping) else None)
        counts = _int_list(row.get("terminal_probe_trial_counts") if isinstance(row, Mapping) else None)
        if devices:
            used_devices.update(devices)
            device_sets.add(tuple(devices))
            for device in devices:
                probe_episode_counts[device] += 1
        if counts:
            trial_splits.add(tuple(counts))
        if devices and counts:
            if len(devices) == len(counts):
                for device, count in zip(devices, counts):  # noqa: B905 - lengths checked above for py39.
                    trial_counts[device] += int(count)
            else:
                mismatched_trial_rows += 1

        probe_s = _float_value(row.get("terminal_probe_wall_seconds"))
        if probe_s is not None:
            terminal_probe_wall.add(probe_s)
        if devices:
            for device, wall_s in _per_device_probe_walls(row, devices, probe_s).items():
                probe_wall_by_device[device].add(wall_s)
        policy_s = _float_value(row.get("policy_rollout_wall_seconds"))
        if policy_s is not None:
            policy_rollout_wall.add(policy_s)
        replan_s = _first_float(row, REPLAN_TIMING_FIELDS)
        if replan_s is not None:
            replan_wall.add(replan_s)
        for field in HOT_PATH_TIMING_FIELDS:
            value = _float_value(row.get(field))
            if value is not None:
                hot_path_timings[field].add(value)

    visible = parse_device_spec(visible_devices)
    if not visible:
        visible = sorted(set(gpu_utilization) | used_devices, key=_device_sort_key)
    idle_visible = sorted(set(visible) - used_devices, key=_device_sort_key)
    sorted_used = sorted(used_devices, key=_device_sort_key)

    if episode_count and not sorted_used:
        warnings.append("No terminal_probe_devices were recorded in episode diagnostics.")
        recommendations.append("Enable terminal reward-probe diagnostics before judging GPU utilization.")
    if idle_visible:
        joined = ", ".join(idle_visible)
        warnings.append(f"visible GPUs were not used by terminal probes: {joined}")
        recommendations.append("Forward all intended devices with --stage2-rl-devices or --blb-v3-reward-devices.")
    if mismatched_trial_rows:
        warnings.append(
            f"{mismatched_trial_rows} episode rows had terminal_probe_devices/trial_counts length mismatches."
        )
    for device, info in gpu_utilization.items():
        max_util = float(info.get("max_util_pct", 0.0) or 0.0)
        if max_util < float(low_util_threshold_pct):
            warnings.append(
                f"{device} max utilization {max_util:.1f}% below {float(low_util_threshold_pct):.1f}%."
            )
            recommendations.append("Check whether reward probes are balanced across visible GPUs.")

    return {
        "episodes": episode_count,
        "visible_devices": sorted(visible, key=_device_sort_key),
        "used_probe_devices": sorted_used,
        "idle_visible_devices": idle_visible,
        "probe_episode_counts_by_device": dict(sorted(probe_episode_counts.items(), key=lambda item: _device_sort_key(item[0]))),
        "probe_trial_counts_by_device": dict(sorted(trial_counts.items(), key=lambda item: _device_sort_key(item[0]))),
        "probe_wall_seconds_by_device": {
            device: values.as_dict()
            for device, values in sorted(probe_wall_by_device.items(), key=lambda item: _device_sort_key(item[0]))
        },
        "probe_device_sets": [list(item) for item in sorted(device_sets, key=lambda item: (_device_sort_key(item[0]) if item else (9, ""), item))],
        "probe_trial_splits": [list(item) for item in sorted(trial_splits)],
        "terminal_probe_wall_seconds": terminal_probe_wall.as_dict(),
        "policy_rollout_wall_seconds": policy_rollout_wall.as_dict(),
        "replan_wall_seconds": replan_wall.as_dict(),
        "hot_path_wall_seconds": {
            field: values.as_dict()
            for field, values in sorted(hot_path_timings.items())
        },
        "gpu_utilization": gpu_utilization,
        "warnings": warnings,
        "recommendations": sorted(set(recommendations)),
    }


def summarize_run(
        episodes: str | Path,
        *,
        nvidia_smi_csv: str | Path | None = None,
        visible_devices: str | Sequence[object] | None = None,
        low_util_threshold_pct: float = LOW_UTIL_THRESHOLD_PCT,
        ) -> dict[str, Any]:
    return summarize_rows(
        _iter_jsonl(episodes),
        gpu_utilization=_load_nvidia_smi_csv(nvidia_smi_csv),
        visible_devices=visible_devices,
        low_util_threshold_pct=low_util_threshold_pct,
    )


def _join_or_none(values: Sequence[str]) -> str:
    return ", ".join(values) if values else "none"


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# GPU Utilization Report",
        "",
        f"Episodes: {summary.get('episodes', 0)}",
        f"Visible devices: {_join_or_none(list(summary.get('visible_devices') or []))}",
        f"Used probe devices: {_join_or_none(list(summary.get('used_probe_devices') or []))}",
        f"Idle visible devices: {_join_or_none(list(summary.get('idle_visible_devices') or []))}",
        "",
        "## Probe Timing",
    ]
    probe_stats = summary.get("terminal_probe_wall_seconds") or {}
    policy_stats = summary.get("policy_rollout_wall_seconds") or {}
    replan_stats = summary.get("replan_wall_seconds") or {}
    lines.append(f"Terminal probe mean seconds: {probe_stats.get('mean')}")
    lines.append(f"Policy rollout mean seconds: {policy_stats.get('mean')}")
    lines.append(f"Replan/optimizer mean seconds: {replan_stats.get('mean')}")
    lines.append("")
    lines.append("## Probe Wall By Device")
    wall_by_device = summary.get("probe_wall_seconds_by_device") or {}
    episodes_by_device = summary.get("probe_episode_counts_by_device") or {}
    if wall_by_device:
        for device, stats in wall_by_device.items():
            lines.append(
                f"- {device}: episodes={episodes_by_device.get(device, 0)}, "
                f"mean_s={stats.get('mean')}, min_s={stats.get('min')}, "
                f"max_s={stats.get('max')}"
            )
    else:
        lines.append("- none")
    lines.append("")
    hot_path = summary.get("hot_path_wall_seconds") or {}
    if hot_path:
        lines.append("## Hot Path Timing")
        for field, stats in hot_path.items():
            lines.append(
                f"- {field}: count={stats.get('count')}, mean_s={stats.get('mean')}, "
                f"max_s={stats.get('max')}"
            )
        lines.append("")
    lines.append("## Trial Balance")
    counts = summary.get("probe_trial_counts_by_device") or {}
    if counts:
        for device, count in counts.items():
            lines.append(f"- {device}: {count}")
    else:
        lines.append("- none")
    gpu_util = summary.get("gpu_utilization") or {}
    if gpu_util:
        lines.append("")
        lines.append("## Nvidia SMI")
        for device, info in gpu_util.items():
            lines.append(
                "- "
                f"{device}: max_util_pct={info.get('max_util_pct')}, "
                f"mean_util_pct={info.get('mean_util_pct')}, "
                f"active_sample_rate={info.get('active_sample_rate')}, "
                f"max_memory_mib={info.get('max_memory_mib')}"
            )
    lines.append("")
    lines.append("## Warnings")
    warnings = summary.get("warnings") or []
    if warnings:
        for item in warnings:
            lines.append(f"- {item}")
    else:
        lines.append("- none")
    recommendations = summary.get("recommendations") or []
    if recommendations:
        lines.append("")
        lines.append("## Recommendations")
        for item in recommendations:
            lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", required=True, help="episodes.jsonl path or run directory")
    parser.add_argument("--nvidia-smi-csv", default=None, help="optional nvidia-smi CSV sample log")
    parser.add_argument("--visible-devices", default="", help="comma-separated visible GPU ids, e.g. 0,1,2,3")
    parser.add_argument("--out-json", default="", help="optional JSON output path")
    parser.add_argument("--out-md", default="", help="optional markdown output path")
    parser.add_argument(
        "--low-util-threshold-pct",
        type=float,
        default=LOW_UTIL_THRESHOLD_PCT,
        help="warn when sampled max GPU utilization is below this percent",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = summarize_run(
        args.episodes,
        nvidia_smi_csv=args.nvidia_smi_csv,
        visible_devices=args.visible_devices,
        low_util_threshold_pct=float(args.low_util_threshold_pct),
    )
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = render_markdown(summary)
    if args.out_md:
        Path(args.out_md).write_text(markdown, encoding="utf-8")
    if not args.out_json and not args.out_md:
        sys.stdout.write(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
