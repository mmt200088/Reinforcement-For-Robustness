#!/usr/bin/env python3
"""Render Stage-2 reward-probe GPU scaling benchmark summaries.

The benchmark shell script writes one compact ``runs.jsonl`` file plus optional
per-run episode diagnostics and nvidia-smi samples.  This module keeps the
post-processing dependency-free and streams those JSONL/CSV inputs so report
generation stays cheap even when a benchmark keeps more probe episodes.
"""
from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from csv_field_utils import first_present_by_index, normalized_field_index  # noqa: E402
from json_utils import write_json_file  # noqa: E402
from jsonl_utils import iter_jsonl  # noqa: E402
from numeric_parse_utils import parse_first_float  # noqa: E402
from report_format_utils import format_float  # noqa: E402
from stats_utils import median_sorted  # noqa: E402


def _summarize_episodes(path: Path) -> dict[str, Any]:
    probe_walls: list[float] = []
    wall_total = 0.0
    speedup_total = 0.0
    speedup_count = 0
    devices_seen: set[str] = set()
    counts_seen: set[tuple[int, ...]] = set()
    if not path.exists():
        return {
            "probe_calls": 0,
            "mean_wall": None,
            "median_wall": None,
            "mean_speedup": None,
            "devices_seen": [],
            "trial_splits": [],
        }

    for rec in iter_jsonl(path, errors="raise"):
        wall = parse_first_float(rec.get("terminal_probe_wall_seconds")) or 0.0
        if wall > 0.0:
            probe_walls.append(float(wall))
            wall_total += float(wall)
        speedup = parse_first_float(rec.get("terminal_probe_speedup")) or 0.0
        if speedup > 0.0:
            speedup_total += float(speedup)
            speedup_count += 1
        for dev in rec.get("terminal_probe_devices") or []:
            devices_seen.add(str(dev))
        counts = rec.get("terminal_probe_trial_counts") or []
        if isinstance(counts, list) and counts:
            parsed_counts = []
            for item in counts:
                value = parse_first_float(item)
                if value is not None:
                    parsed_counts.append(int(value))
            if parsed_counts:
                counts_seen.add(tuple(parsed_counts))

    median_wall = None
    if probe_walls:
        probe_walls.sort()
        median_wall = median_sorted(probe_walls)

    return {
        "probe_calls": len(probe_walls),
        "mean_wall": wall_total / float(len(probe_walls)) if probe_walls else None,
        "median_wall": median_wall,
        "mean_speedup": speedup_total / float(speedup_count) if speedup_count else None,
        "devices_seen": sorted(devices_seen),
        "trial_splits": [list(item) for item in sorted(counts_seen)],
    }


def _summarize_gpu_samples(path: Path) -> tuple[dict[str, float], dict[str, float]]:
    gpu_util: dict[str, float] = {}
    gpu_mem: dict[str, float] = {}
    if not path.exists():
        return gpu_util, gpu_mem
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return gpu_util, gpu_mem
        field_index = normalized_field_index(header)
        for raw_row in reader:
            if not raw_row or all(not cell or cell.isspace() for cell in raw_row):
                continue
            idx = first_present_by_index(raw_row, field_index, ["index", "gpu_index", "gpu"])
            mem = first_present_by_index(
                raw_row,
                field_index,
                ["memory_used_mib", "memory_used", "memory_used_mi_b", "mem_used_mib"],
            )
            util = first_present_by_index(
                raw_row,
                field_index,
                ["utilization_gpu_pct", "utilization_gpu", "gpu_util_pct", "gpu_util"],
            )
            if idx is None:
                continue
            key = str(idx).strip()
            mem_value = parse_first_float(mem)
            util_value = parse_first_float(util)
            if mem_value is not None:
                gpu_mem[key] = max(gpu_mem.get(key, 0.0), float(mem_value))
            if util_value is not None:
                gpu_util[key] = max(gpu_util.get(key, 0.0), float(util_value))
    return gpu_util, gpu_mem


def build_summary(root: str | Path) -> dict[str, Any]:
    root_path = Path(root)
    runs_path = root_path / "runs.jsonl"
    rows: list[dict[str, Any]] = []
    for run in iter_jsonl(runs_path, errors="raise"):
        label = str(run["label"])
        episode_summary = _summarize_episodes(root_path / f"{label}_episodes.jsonl")
        gpu_util, gpu_mem = _summarize_gpu_samples(root_path / f"{label}_nvidia_smi.csv")
        rows.append(
            {
                **run,
                **episode_summary,
                "max_gpu_util_pct": gpu_util,
                "max_gpu_mem_mib": gpu_mem,
            }
        )

    completed = [
        row for row in rows
        if int(row.get("rc", 1) or 0) == 0 and row.get("mean_wall") is not None
    ]
    best = min(completed, key=lambda row: float(row["mean_wall"])) if completed else None
    return {"runs": rows, "best": best}


def render_html(summary: Mapping[str, Any]) -> str:
    rows = summary.get("runs") or ()
    best = summary.get("best")
    trs: list[str] = []
    for row in rows:
        trs.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('label', '')))}</td>"
            f"<td>{row.get('batch_size', '')}</td>"
            f"<td>{row.get('gpu_count', '')}</td>"
            f"<td>{html.escape(str(row.get('device_spec', '')))}</td>"
            f"<td>{row.get('rc', '')}</td>"
            f"<td>{row.get('probe_calls', '')}</td>"
            f"<td>{format_float(row.get('mean_wall'), digits=4)}</td>"
            f"<td>{format_float(row.get('median_wall'), digits=4)}</td>"
            f"<td>{format_float(row.get('mean_speedup'), digits=4)}</td>"
            f"<td>{html.escape(str(row.get('devices_seen', [])))}</td>"
            f"<td>{html.escape(str(row.get('trial_splits', [])))}</td>"
            f"<td>{html.escape(str(row.get('max_gpu_util_pct', {})))}</td>"
            f"<td>{html.escape(str(row.get('max_gpu_mem_mib', {})))}</td>"
            "</tr>"
        )

    if isinstance(best, Mapping):
        best_html = (
            f"<p><strong>Best observed:</strong> {html.escape(str(best.get('label', '')))}, "
            f"mean probe wall {float(best.get('mean_wall', 0.0)):.4f}s.</p>"
        )
    else:
        best_html = "<p><strong>Best observed:</strong> none; check failed runs.</p>"

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Stage-2 Reward Probe GPU Scaling</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:32px;line-height:1.45;color:#1f2933}}
table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border:1px solid #d8dee4;padding:6px;vertical-align:top}}th{{background:#f6f8fa;text-align:left}}
code{{background:#f6f8fa;padding:2px 4px;border-radius:4px}}
</style></head><body>
<h1>Stage-2 Reward Probe GPU Scaling Benchmark</h1>
<p>This benchmark runs the real Stage-2 RL reward probe path with <code>K=4</code>
trials over the 256-example validation probe subset. For 4 GPUs, the expected
trial split is one independent trial per GPU.</p>
{best_html}
<table><thead><tr><th>run</th><th>batch</th><th>GPUs</th><th>visible devices</th><th>rc</th><th>probe calls</th><th>mean wall s</th><th>median wall s</th><th>mean speedup</th><th>devices seen</th><th>trial splits</th><th>max GPU util %</th><th>max GPU mem MiB</th></tr></thead>
<tbody>{''.join(trs)}</tbody></table>
</body></html>"""


def write_report(root: str | Path) -> dict[str, Any]:
    root_path = Path(root)
    summary = build_summary(root_path)
    write_json_file(root_path / "benchmark_summary.json", summary, trailing_newline=False)
    best = summary.get("best")
    if isinstance(best, Mapping):
        (root_path / "best_batch_size.txt").write_text(
            f"{best.get('batch_size')}\n",
            encoding="utf-8",
        )
    (root_path / "stage2_reward_probe_scaling_report.html").write_text(
        render_html(summary),
        encoding="utf-8",
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", help="benchmark artifact directory containing runs.jsonl")
    args = parser.parse_args(argv)
    root = Path(args.artifact_dir)
    summary = write_report(root)
    print(f"[bench] wrote {root / 'benchmark_summary.json'}")
    print(f"[bench] wrote {root / 'stage2_reward_probe_scaling_report.html'}")
    best = summary.get("best")
    if isinstance(best, Mapping):
        print(f"[bench] best_batch_size={best.get('batch_size')} best_label={best.get('label')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
