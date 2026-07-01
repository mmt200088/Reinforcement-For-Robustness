#!/usr/bin/env python3
"""Capture a lightweight server resource/source snapshot before heavy runs.

The script is dependency-free and read-only. It records GPU inventory/utilization,
basic CPU/load data, and git source state so server A/B evidence can explain
whether hardware was actually available and which source snapshot was used.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Sequence


def _int_from_text(value: object, default: int = 0) -> int:
    text = str(value or "").strip()
    if not text:
        return int(default)
    token = text.split()[0].strip().strip("%")
    try:
        return int(float(token))
    except ValueError:
        return int(default)


def parse_nvidia_smi_lines(lines: Iterable[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        rows.append(
            {
                "index": _int_from_text(parts[0]),
                "name": parts[1],
                "memory_total_mib": _int_from_text(parts[2]),
                "memory_used_mib": _int_from_text(parts[3]),
                "utilization_gpu_pct": _int_from_text(parts[4]),
            }
        )
    return rows


def parse_nvidia_smi_csv(text: str) -> list[dict[str, Any]]:
    return parse_nvidia_smi_lines(str(text or "").splitlines())


def _run_command(cmd: Sequence[str], *, cwd: Path | None = None) -> str:
    try:
        completed = subprocess.run(
            list(cmd),
            cwd=str(cwd) if cwd else None,
            check=False,
            text=True,
            capture_output=True,
        )
    except Exception:
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def _query_nvidia_smi() -> str:
    return _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )


def _git_summary(root: Path) -> dict[str, Any]:
    commit = _run_command(["git", "rev-parse", "HEAD"], cwd=root)
    branch = _run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root)
    status = _run_command(["git", "status", "--porcelain"], cwd=root)
    dirty = [line for line in status.splitlines() if line.strip()]
    return {
        "commit": commit,
        "branch": branch,
        "dirty_file_count": len(dirty),
        "dirty_examples": dirty[:20],
    }


def _system_summary() -> dict[str, Any]:
    load_avg = []
    if hasattr(os, "getloadavg"):
        try:
            load_avg = [float(value) for value in os.getloadavg()]
        except OSError:
            load_avg = []
    return {
        "cpu_count": os.cpu_count() or 0,
        "load_average": load_avg,
        "python": sys.version.split()[0],
    }


def _gpu_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    gpu_count = len(rows)
    active = [
        row for row in rows
        if int(row.get("memory_used_mib", 0) or 0) > 0
        or int(row.get("utilization_gpu_pct", 0) or 0) > 0
    ]
    return {
        "gpu_count": gpu_count,
        "active_gpu_count": len(active),
        "idle_gpu_count": max(0, gpu_count - len(active)),
        "memory_total_mib": int(sum(int(row.get("memory_total_mib", 0) or 0) for row in rows)),
        "memory_used_mib": int(sum(int(row.get("memory_used_mib", 0) or 0) for row in rows)),
        "max_utilization_gpu_pct": max(
            (int(row.get("utilization_gpu_pct", 0) or 0) for row in rows),
            default=0,
        ),
    }


def collect_snapshot(root: str | Path, *, nvidia_smi_csv: str | Path | None = None) -> dict[str, Any]:
    root_path = Path(root).resolve()
    if nvidia_smi_csv:
        with Path(nvidia_smi_csv).open(encoding="utf-8", errors="replace") as handle:
            gpus = parse_nvidia_smi_lines(handle)
    else:
        smi_text = _query_nvidia_smi()
        gpus = parse_nvidia_smi_csv(smi_text)
    return {
        "schema": "server_resource_snapshot_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root_path),
        "system": _system_summary(),
        "git": _git_summary(root_path),
        "gpus": gpus,
        "gpu_summary": _gpu_summary(gpus),
    }


def render_markdown(snapshot: dict[str, Any]) -> str:
    gpu_summary = snapshot.get("gpu_summary", {})
    git = snapshot.get("git", {})
    system = snapshot.get("system", {})
    lines = [
        "# Server Resource Snapshot",
        "",
        f"Created at: {snapshot.get('created_at', '')}",
        f"Root: `{snapshot.get('root', '')}`",
        "",
        "## Source",
        f"- branch: `{git.get('branch', '')}`",
        f"- commit: `{git.get('commit', '')}`",
        f"- dirty files: {git.get('dirty_file_count', 0)}",
        "",
        "## System",
        f"- CPU count: {system.get('cpu_count', 0)}",
        f"- load average: {system.get('load_average', [])}",
        "",
        "## GPU Summary",
        f"- GPU count: {gpu_summary.get('gpu_count', 0)}",
        f"- active GPUs: {gpu_summary.get('active_gpu_count', 0)}",
        f"- idle GPUs: {gpu_summary.get('idle_gpu_count', 0)}",
        f"- memory used MiB: {gpu_summary.get('memory_used_mib', 0)} / {gpu_summary.get('memory_total_mib', 0)}",
        f"- max GPU utilization pct: {gpu_summary.get('max_utilization_gpu_pct', 0)}",
        "",
        "## GPUs",
    ]
    gpus = snapshot.get("gpus") or []
    if gpus:
        for row in gpus:
            lines.append(
                "- "
                f"cuda:{row.get('index')}: {row.get('name')} "
                f"util={row.get('utilization_gpu_pct')}% "
                f"mem={row.get('memory_used_mib')}/{row.get('memory_total_mib')} MiB"
            )
    else:
        lines.append("- none detected")
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="repository root")
    parser.add_argument("--nvidia-smi-csv", default="", help="offline nvidia-smi query CSV")
    parser.add_argument("--out-json", default="", help="optional JSON output path")
    parser.add_argument("--out-md", default="", help="optional Markdown output path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    snapshot = collect_snapshot(
        args.root,
        nvidia_smi_csv=args.nvidia_smi_csv or None,
    )
    markdown = render_markdown(snapshot)
    if args.out_json:
        Path(args.out_json).write_text(
            json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.out_md:
        Path(args.out_md).write_text(markdown, encoding="utf-8")
    if not args.out_json and not args.out_md:
        sys.stdout.write(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
