#!/usr/bin/env python3
"""Summarize Stage-1 parallel rollout diagnostics from training logs.

The parser is dependency-free and torch-free. It consumes the existing
``[stage1-rollout]`` and ``[stage1-rollout-total]`` log lines emitted by
``layer_importance_evaluator.py`` and turns them into compact JSON/Markdown
evidence for 1GPU vs NGPU Stage-1 throughput checks.
"""

from __future__ import annotations

import argparse
from collections.abc import Collection
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cli_parse_utils import parse_int_list_text, split_int_tokens  # noqa: E402
from stats_utils import safe_div_or_none  # noqa: E402
from text_utils import iter_text_lines  # noqa: E402

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
    return split_int_tokens(text, allow_semicolon=False)


def _worker_counts_imbalanced(counts: Iterable[int]) -> bool:
    if isinstance(counts, Collection):
        if not counts:
            return False
        min_count = min(counts)
        if not min_count:
            return False
        return max(counts) / min_count > 1.2

    min_count: int | None = None
    max_count: int | None = None
    for raw_count in counts:
        count = int(raw_count)
        if min_count is None or count < min_count:
            min_count = count
        if max_count is None or count > max_count:
            max_count = count
    return bool(min_count and max_count is not None and max_count / min_count > 1.2)


def parse_log_lines(lines: Iterable[str]) -> dict[str, Any]:
    rollout_window_count = 0
    total_window_count = 0
    total_episodes = 0
    total_wall_seconds = 0.0
    component_seconds = {key: 0.0 for key in COMPONENT_KEYS}
    speedup_sum = 0.0
    speedup_count = 0
    max_speedup: float | None = None
    last_cache = {
        "window": None,
        "hits": 0,
        "misses": 0,
        "distinct": 0,
        "hit_rate": None,
    }
    worker_episode_counts: dict[str, int] = {}
    devices_seen: set[str] = set()
    warnings: list[str] = []

    for line in lines:
        if "[stage1-rollout" not in line:
            continue
        rollout_match = ROLLOUT_RE.search(line)
        if rollout_match:
            devices = _split_csv(rollout_match.group("devices"))
            counts = parse_int_list_text(rollout_match.group("counts"), allow_semicolon=False)
            for device, count in zip(devices, counts):  # noqa: B905 - diagnostics emit matching lists on py39.
                devices_seen.add(device)
                worker_episode_counts[device] = worker_episode_counts.get(device, 0) + int(count)
            speedup = float(rollout_match.group("speedup"))
            speedup_sum += speedup
            speedup_count += 1
            max_speedup = speedup if max_speedup is None else max(max_speedup, speedup)
            rollout_window_count += 1
            continue

        cache_match = CACHE_RE.search(line)
        if cache_match:
            last_cache = {
                "window": int(cache_match.group("window")),
                "hits": int(cache_match.group("hits")),
                "misses": int(cache_match.group("misses")),
                "distinct": int(cache_match.group("distinct")),
                "hit_rate": float(cache_match.group("hit_rate")) / 100.0,
            }
            continue

        total_match = TOTAL_RE.search(line)
        if total_match:
            total_window_count += 1
            total_episodes += int(total_match.group("episodes"))
            total_wall_seconds += float(total_match.group("total"))
            for key in COMPONENT_KEYS:
                component_seconds[key] += float(total_match.group(key))

    component_share = {
        key: safe_div_or_none(value, total_wall_seconds)
        for key, value in component_seconds.items()
    }
    throughput = (
        total_episodes * 3600.0 / total_wall_seconds
        if total_episodes and total_wall_seconds > 0.0
        else None
    )
    if not rollout_window_count:
        warnings.append("No [stage1-rollout] worker timing lines found.")
    if not total_window_count:
        warnings.append("No [stage1-rollout-total] window timing lines found.")
    if _worker_counts_imbalanced(worker_episode_counts.values()):
        warnings.append("Worker episode counts are imbalanced across devices.")

    return {
        "windows": max(rollout_window_count, total_window_count),
        "total_episodes": total_episodes,
        "total_wall_seconds": total_wall_seconds,
        "throughput_ep_per_hour": throughput,
        "mean_worker_speedup": speedup_sum / speedup_count if speedup_count else None,
        "max_worker_speedup": max_speedup,
        "device_count": len(devices_seen),
        "devices": sorted(devices_seen),
        "worker_episode_counts_by_device": dict(sorted(worker_episode_counts.items())),
        "component_seconds": component_seconds,
        "component_share": component_share,
        "eval_cache": last_cache,
        "warnings": warnings,
    }


def parse_log_text(text: str) -> dict[str, Any]:
    return parse_log_lines(iter_text_lines(text))


def _iter_log_lines(paths: Sequence[str]) -> Iterable[str]:
    for path in paths:
        with Path(path).open(encoding="utf-8", errors="replace") as handle:
            yield from handle


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
    summary = parse_log_lines(_iter_log_lines(args.log))
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
