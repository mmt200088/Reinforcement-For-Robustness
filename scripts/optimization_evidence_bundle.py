#!/usr/bin/env python3
"""Build a compact optimization evidence bundle for local/server runs.

This script ties together the torch-free optimization evidence tools:

* project-wide flow/artifact inventory
* Stage-1 parallel rollout timing summaries
* Stage-2 GPU utilization summaries
* Stage-2 persistent output verification

It is meant to run after a server smoke/A-B/full run so evidence promotion does
not depend on manually stitching several reports together.
"""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import importlib.util
import io
import json
from pathlib import Path
import sys
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent


def _load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _rel(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _git_commit(root: Path) -> str:
    git_head = root / ".git" / "HEAD"
    if not git_head.is_file():
        return ""
    head = git_head.read_text(encoding="utf-8", errors="replace").strip()
    if head.startswith("ref: "):
        ref = head.split(" ", 1)[1].strip()
        ref_path = root / ".git" / ref
        if ref_path.is_file():
            return ref_path.read_text(encoding="utf-8", errors="replace").strip()
        return ref
    return head


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="repository root")
    parser.add_argument("--out-dir", required=True, help="bundle output directory")
    parser.add_argument("--artifact-root", action="append", default=[], help="artifact root for project audit")
    parser.add_argument("--stage1-log", action="append", default=[], help="Stage-1 training log path")
    parser.add_argument("--stage2-episodes", default="", help="Stage-2 episodes.jsonl or run directory")
    parser.add_argument("--nvidia-smi-csv", default="", help="optional nvidia-smi CSV for Stage-2 GPU report")
    parser.add_argument("--visible-devices", default="", help="visible devices for Stage-2 GPU report")
    parser.add_argument("--stage2-progress-dir", default="", help="Stage-2 progress dir for persistent verifier")
    parser.add_argument("--stage2-run-dir", default="", help="Stage-2 run dir for persistent verifier")
    parser.add_argument("--min-episodes", type=int, default=1)
    parser.add_argument("--min-ppo-updates", type=int, default=1)
    parser.add_argument("--require-png", action="store_true")
    parser.add_argument("--no-require-details", action="store_true")
    return parser


def _run_project_audit(root: Path, out_dir: Path, artifact_roots: Sequence[str]) -> dict[str, Any]:
    mod = _load_script_module("project_optimization_audit", SCRIPT_DIR / "project_optimization_audit.py")
    report = mod.build_project_audit(root, artifact_roots=artifact_roots)
    json_path = out_dir / "project_optimization_audit.json"
    md_path = out_dir / "project_optimization_audit.md"
    _write_json(json_path, report)
    _write_text(md_path, mod.render_markdown(report))
    return {
        "json": json_path.name,
        "markdown": md_path.name,
        "summary": report.get("summary", {}),
        "missing_evidence": report.get("artifact_summary", {}).get("missing_evidence", []),
    }


def _run_stage1_report(logs: Sequence[str], out_dir: Path) -> dict[str, Any] | None:
    if not logs:
        return None
    mod = _load_script_module("stage1_parallel_report", SCRIPT_DIR / "stage1_parallel_report.py")
    text = "\n".join(Path(path).read_text(encoding="utf-8", errors="replace") for path in logs)
    report = mod.parse_log_text(text)
    json_path = out_dir / "stage1_parallel_report.json"
    md_path = out_dir / "stage1_parallel_report.md"
    _write_json(json_path, report)
    _write_text(md_path, mod.render_markdown(report))
    return {
        "json": json_path.name,
        "markdown": md_path.name,
        "logs": [str(path) for path in logs],
        "windows": report.get("windows", 0),
        "throughput_ep_per_hour": report.get("throughput_ep_per_hour"),
        "warnings": report.get("warnings", []),
    }


def _run_stage2_gpu_report(args: argparse.Namespace, out_dir: Path) -> dict[str, Any] | None:
    if not args.stage2_episodes:
        return None
    mod = _load_script_module("gpu_utilization_report", SCRIPT_DIR / "gpu_utilization_report.py")
    report = mod.summarize_run(
        args.stage2_episodes,
        nvidia_smi_csv=args.nvidia_smi_csv or None,
        visible_devices=args.visible_devices,
    )
    json_path = out_dir / "stage2_gpu_utilization_report.json"
    md_path = out_dir / "stage2_gpu_utilization_report.md"
    _write_json(json_path, report)
    _write_text(md_path, mod.render_markdown(report))
    return {
        "json": json_path.name,
        "markdown": md_path.name,
        "episodes": report.get("episodes", 0),
        "used_probe_devices": report.get("used_probe_devices", []),
        "idle_visible_devices": report.get("idle_visible_devices", []),
        "warnings": report.get("warnings", []),
    }


def _run_stage2_verifier(args: argparse.Namespace, out_dir: Path) -> dict[str, Any] | None:
    if not args.stage2_progress_dir and not args.stage2_run_dir:
        return None
    mod = _load_script_module(
        "verify_stage2_persistent_outputs",
        SCRIPT_DIR / "verify_stage2_persistent_outputs.py",
    )
    verify_args = argparse.Namespace(
        run_dir=args.stage2_run_dir,
        progress_dir=args.stage2_progress_dir,
        min_episodes=int(args.min_episodes),
        min_ppo_updates=int(args.min_ppo_updates),
        require_png=bool(args.require_png),
        require_details=not bool(args.no_require_details),
    )
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        rc = int(mod.verify(verify_args))
    output_path = out_dir / "stage2_persistent_verify.txt"
    _write_text(output_path, buffer.getvalue())
    return {
        "returncode": rc,
        "output": output_path.name,
    }


def render_index(manifest: dict[str, Any]) -> str:
    lines = [
        "# Optimization Evidence Bundle",
        "",
        f"Status: {manifest['status']}",
        f"Created at: {manifest['created_at']}",
        f"Root: `{manifest['root']}`",
    ]
    if manifest.get("git_commit"):
        lines.append(f"Git commit: `{manifest['git_commit']}`")
    lines.extend(["", "## Reports"])
    for section in ("project_audit", "stage1_parallel_report", "stage2_gpu_utilization_report"):
        payload = manifest.get(section)
        if not payload:
            continue
        lines.append(f"- {section}: `{payload.get('markdown')}` / `{payload.get('json')}`")
    verifier = manifest.get("stage2_persistent_verify")
    if verifier:
        lines.append(
            f"- stage2_persistent_verify: `{verifier.get('output')}` "
            f"(rc={verifier.get('returncode')})"
        )
    warnings = manifest.get("warnings") or []
    lines.extend(["", "## Warnings"])
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "schema": "optimization_evidence_bundle_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "out_dir": str(out_dir),
        "git_commit": _git_commit(root),
        "status": "ok",
        "warnings": [],
    }

    manifest["project_audit"] = _run_project_audit(root, out_dir, args.artifact_root)
    stage1_report = _run_stage1_report(args.stage1_log, out_dir)
    if stage1_report:
        manifest["stage1_parallel_report"] = stage1_report
        manifest["warnings"].extend(stage1_report.get("warnings", []))
    stage2_gpu_report = _run_stage2_gpu_report(args, out_dir)
    if stage2_gpu_report:
        manifest["stage2_gpu_utilization_report"] = stage2_gpu_report
        manifest["warnings"].extend(stage2_gpu_report.get("warnings", []))
    verifier = _run_stage2_verifier(args, out_dir)
    if verifier:
        manifest["stage2_persistent_verify"] = verifier
        if int(verifier.get("returncode", 0)) != 0:
            manifest["status"] = "failed"
            manifest["warnings"].append("Stage-2 persistent verifier failed.")

    _write_json(out_dir / "manifest.json", manifest)
    _write_text(out_dir / "index.md", render_index(manifest))
    return 0 if manifest["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
