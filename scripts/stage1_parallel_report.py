#!/usr/bin/env python3
"""Summarize Stage-1 parallel rollout diagnostics from training logs.

The parser is dependency-free and torch-free. It consumes the existing
``[stage1-rollout]`` and ``[stage1-rollout-total]`` log lines emitted by
``layer_importance_evaluator.py`` and turns them into compact JSON/Markdown
evidence for 1GPU vs NGPU Stage-1 throughput checks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Mapping, Sequence

ROLLOUT_RE = re.compile(
    r"\[stage1-rollout\]\s+"
    r"window=(?P<window>\d+)\s+"
    r"eps_per_worker=(?P<eps_per_worker>\d+)\s+"
    r"devices=\[(?P<devices>[^\]]*)\]\s+"
    r"counts=\[(?P<counts>[^\]]*)\]\s+"
    r"wall=(?P<wall>[0-9.]+)s\s+"
    r"worker_seconds=\[(?P<worker_seconds>[^\]]*)\]\s+"
    r"speedup=(?P<speedup>[0-9.]+)x"
)
CACHE_RE = re.compile(
    r"\[stage1-rollout\]\s+"
    r"window=(?P<window>\d+)\s+"
    r"eval_cache hits=(?P<hits>\d+)\s+"
    r"misses=(?P<misses>\d+)\s+"
    r"distinct=(?P<distinct>\d+)\s+"
    r"hit_rate=(?P<hit_rate>[0-9.]+)%"
)
TOTAL_RE = re.compile(
    r"\[stage1-rollout-total\]\s+"
    r"window=(?P<window>\d+)\s+"
    r"episodes=(?P<episodes>\d+)\s+"
    r"total=(?P<total>[0-9.]+)s\s+"
    r"collect=(?P<collect>[0-9.]+)s\s+"
    r"replay=(?P<replay>[0-9.]+)s\s+"
    r"detail=(?P<detail>[0-9.]+)s\s+"
    r"ppo_update=(?P<ppo_update>[0-9.]+)s\s+"
    r"other=(?P<other>[0-9.]+)s\s+"
    r"throughput=(?P<throughput>[0-9.]+)ep/h"
)
COMPONENT_KEYS = ("collect", "replay", "detail", "ppo_update", "other")


def _split_csv(text: str) -> list[str]:
    return [item.strip() for item in str(text or "").split(",") if item.strip()]


def _parse_float_list(text: str) -> list[float]:
    out: list[float] = []
    for item in _split_csv(text):
        out.append(float(item))
    return out


def _parse_int_list(text: str) -> list[int]:
    out: list[int] = []
    for item in _split_csv(text):
        out.append(int(item))
    return out


def _safe_div(numer: float, denom: float) -> float | None:
    if denom <= 0.0:
        return None
    return float(numer) / float(denom)


def parse_log_text(text: str) -> dict[str, Any]:
    rollout_windows: list[dict[str, Any]] = []
    totals: list[dict[str, Any]] = []
    cache_rows: list[dict[str, Any]] = []
    worker_episode_counts: dict[str, int] = {}
    devices_seen: set[str] = set()
    warnings: list[str] = []

    for line in str(text or "").splitlines():
        rollout_match = ROLLOUT_RE.search(line)
        if rollout_match:
            devices = _split_csv(rollout_match.group("devices"))
            counts = _parse_int_list(rollout_match.group("counts"))
            worker_seconds = _parse_float_list(rollout_match.group("worker_seconds"))
            for device, count in zip(devices, counts):  # noqa: B905 - diagnostics emit matching lists on py39.
                devices_seen.add(device)
                worker_episode_counts[device] = worker_episode_counts.get(device, 0) + int(count)
            rollout_windows.append(
                {
                    "window": int(rollout_match.group("window")),
                    "eps_per_worker": int(rollout_match.group("eps_per_worker")),
                    "devices": devices,
                    "counts": counts,
                    "wall_seconds": float(rollout_match.group("wall")),
                    "worker_seconds": worker_seconds,
                    "speedup_vs_sequential": float(rollout_match.group("speedup")),
                }
            )
            continue

        cache_match = CACHE_RE.search(line)
        if cache_match:
            cache_rows.append(
                {
                    "window": int(cache_match.group("window")),
                    "hits": int(cache_match.group("hits")),
                    "misses": int(cache_match.group("misses")),
                    "distinct": int(cache_match.group("distinct")),
                    "hit_rate": float(cache_match.group("hit_rate")) / 100.0,
                }
            )
            continue

        total_match = TOTAL_RE.search(line)
        if total_match:
            row = {
                "window": int(total_match.group("window")),
                "episodes": int(total_match.group("episodes")),
                "total_seconds": float(total_match.group("total")),
                "throughput_ep_per_hour": float(total_match.group("throughput")),
            }
            for key in COMPONENT_KEYS:
                row[f"{key}_seconds"] = float(total_match.group(key))
            totals.append(row)

    total_episodes = int(sum(row["episodes"] for row in totals))
    total_wall_seconds = float(sum(row["total_seconds"] for row in totals))
    component_seconds = {
        key: float(sum(row[f"{key}_seconds"] for row in totals))
        for key in COMPONENT_KEYS
    }
    component_share = {
        key: _safe_div(value, total_wall_seconds)
        for key, value in component_seconds.items()
    }
    speedups = [float(row["speedup_vs_sequential"]) for row in rollout_windows]
    throughput = (
        total_episodes * 3600.0 / total_wall_seconds
        if total_episodes and total_wall_seconds > 0.0
        else None
    )
    if not rollout_windows:
        warnings.append("No [stage1-rollout] worker timing lines found.")
    if not totals:
        warnings.append("No [stage1-rollout-total] window timing lines found.")
    if worker_episode_counts:
        counts = list(worker_episode_counts.values())
        if min(counts) and max(counts) / min(counts) > 1.2:
            warnings.append("Worker episode counts are imbalanced across devices.")

    last_cache = cache_rows[-1] if cache_rows else {
        "window": None,
        "hits": 0,
        "misses": 0,
        "distinct": 0,
        "hit_rate": None,
    }

    return {
        "windows": max(len(rollout_windows), len(totals)),
        "total_episodes": total_episodes,
        "total_wall_seconds": total_wall_seconds,
        "throughput_ep_per_hour": throughput,
        "mean_worker_speedup": float(statistics.mean(speedups)) if speedups else None,
        "max_worker_speedup": float(max(speedups)) if speedups else None,
        "device_count": len(devices_seen),
        "devices": sorted(devices_seen),
        "worker_episode_counts_by_device": dict(sorted(worker_episode_counts.items())),
        "component_seconds": component_seconds,
        "component_share": component_share,
        "eval_cache": last_cache,
        "warnings": warnings,
    }


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Stage-1 Parallel Report",
        "",
        f"Windows: {summary.get('windows', 0)}",
        f"Episodes: {summary.get('total_episodes', 0)}",
        f"Total wall seconds: {summary.get('total_wall_seconds')}",
        f"Throughput: {float(summary.get('throughput_ep_per_hour') or 0.0):.3f} ep/h",
        f"Mean worker speedup: {summary.get('mean_worker_speedup')}",
        "",
        "## Worker Balance",
    ]
    counts = summary.get("worker_episode_counts_by_device") or {}
    if counts:
        for device, count in counts.items():
            lines.append(f"- {device}: {count}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Component Wall Seconds")
    components = summary.get("component_seconds") or {}
    shares = summary.get("component_share") or {}
    if components:
        for key, seconds in components.items():
            share = shares.get(key)
            share_text = "n/a" if share is None else f"{float(share):.3f}"
            lines.append(f"- {key}: {seconds} ({share_text})")
    else:
        lines.append("- none")
    cache = summary.get("eval_cache") or {}
    lines.extend(
        [
            "",
            "## Eval Cache",
            f"Hits: {cache.get('hits', 0)}",
            f"Misses: {cache.get('misses', 0)}",
            f"Distinct: {cache.get('distinct', 0)}",
            f"Hit rate: {cache.get('hit_rate')}",
            "",
            "## Warnings",
        ]
    )
    warnings = summary.get("warnings") or []
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", action="append", required=True, help="Stage-1 training log path")
    parser.add_argument("--out-json", default="", help="optional JSON output path")
    parser.add_argument("--out-md", default="", help="optional Markdown output path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    text = "\n".join(Path(path).read_text(encoding="utf-8", errors="replace") for path in args.log)
    summary = parse_log_text(text)
    markdown = render_markdown(summary)
    if args.out_json:
        Path(args.out_json).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.out_md:
        Path(args.out_md).write_text(markdown, encoding="utf-8")
    if not args.out_json and not args.out_md:
        sys.stdout.write(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
