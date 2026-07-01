#!/usr/bin/env python3
"""Build a whole-project runtime optimization audit.

The report is intentionally static and dependency-free. It does not import
training code, torch, transformers, or Rescale_optimizer internals. Its job is
to show whether the full optimization flow has the expected source files and
whether available artifacts contain timing/GPU evidence.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

FLOW_STAGES: list[dict[str, Any]] = [
    {
        "id": "launcher",
        "name": "Launcher, presets, and server bridge",
        "files": [
            "llama_7B_LayerImportance.sh",
            "presets/mrpc-blb-stage2-rl.conf",
            "Paean/run_final_eval.sh",
            "SERVER_COMMAND.md",
            "scripts/server_resource_snapshot.py",
        ],
        "optimization_surfaces": [
            "GPU flag forwarding",
            "server source sync",
            "run manifest",
            "strict resource gates",
        ],
    },
    {
        "id": "stage1",
        "name": "Stage-1 plaintext RL and validation",
        "files": [
            "layer_importance_evaluator.py",
            "stage1_rl/parallel_runner.py",
            "stage1_rl/eval_cache.py",
            "function_handler.py",
            "scripts/stage1_parallel_report.py",
        ],
        "optimization_surfaces": [
            "validation_full forward reuse",
            "multi-GPU rollout collection",
            "deterministic eval cache",
            "hot-path report decoupling",
        ],
    },
    {
        "id": "stage2",
        "name": "Stage-2 BLB RL and reward probes",
        "files": [
            "blb_stage2_rl/parallel_runner.py",
            "blb_stage2_rl/probe_runner.py",
            "blb_stage2_rl/sequential_runner.py",
            "scripts/stage2_ngpu_ab_compare.py",
            "scripts/gpu_utilization_report.py",
        ],
        "optimization_surfaces": [
            "episode-parallel GPU workers",
            "reward-probe device balance",
            "replan/probe timing",
            "JSONL write overhead",
        ],
    },
    {
        "id": "rescale",
        "name": "Rescale optimizer and fusion maps",
        "files": [
            "Rescale_optimizer/rescale_optimizer/replan_interface.py",
            "Rescale_optimizer/rescale_optimizer/replan.py",
            "scripts/blb_build_fusion_count_map.py",
            "blb_stage2_rl/fusion_count_map.py",
        ],
        "optimization_surfaces": [
            "ReplanSession reuse",
            "graph/baseline cache",
            "streaming fusion-map build",
            "CPU worker scheduling",
        ],
    },
    {
        "id": "paean",
        "name": "Paean final evaluation",
        "files": [
            "Paean/run_final_eval.py",
            "Paean/config.py",
            "Paean/action_grid.py",
            "Paean/blb_action_eval.py",
            "final_evaluation_module.py",
        ],
        "optimization_surfaces": [
            "model/tokenizer reuse",
            "action-grid batching",
            "independent-config scheduling",
            "report/render decoupling",
        ],
    },
    {
        "id": "artifacts",
        "name": "Structured data, reports, and sync",
        "files": [
            "rl_data_points.py",
            "scripts/verify_stage2_persistent_outputs.py",
            "scripts/optimization_evidence_bundle.py",
            "tools/paper_figures.py",
            "experiments/index.md",
        ],
        "optimization_surfaces": [
            "complete JSON/JSONL mirrors",
            "compact hot-path writes",
            "post-run report rendering",
            "artifact/source commit linkage",
        ],
    },
]


ARTIFACT_PATTERNS = {
    "episodes_jsonl": "episodes.jsonl",
    "ppo_updates_jsonl": "ppo_updates.jsonl",
    "nvidia_smi_csv": "nvidia*.csv",
    "status_json": "*status*.json",
    "html_reports": "*.html",
    "npz_curves": "*.npz",
}


def _file_entry(root: Path, relative_path: str) -> dict[str, Any]:
    path = root / relative_path
    present = path.exists()
    return {
        "path": relative_path,
        "present": bool(present),
        "kind": "dir" if path.is_dir() else "file",
        "size_bytes": int(path.stat().st_size) if path.is_file() else 0,
    }


def _stage_report(root: Path, stage: Mapping[str, Any]) -> dict[str, Any]:
    files = [_file_entry(root, str(path)) for path in stage.get("files", [])]
    present = sum(1 for item in files if item["present"])
    missing = len(files) - present
    return {
        "id": stage["id"],
        "name": stage["name"],
        "files": files,
        "present_files": present,
        "missing_files": missing,
        "optimization_surfaces": list(stage.get("optimization_surfaces", [])),
    }


def _default_artifact_roots(root: Path) -> list[Path]:
    candidates = [
        root / "Parting Chapter",
        root / "experiments" / "server_command_runs",
        root / "reports",
        root / "rl_training_data_points",
    ]
    return [path for path in candidates if path.exists()]


def _iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return (path for path in root.rglob("*") if path.is_file())


def summarize_artifacts(
        root: Path,
        artifact_roots: Sequence[str | Path] = (),
        *,
        example_limit: int = 5,
        walk_files: Callable[[Path], Iterable[Path]] = _iter_files,
        ) -> dict[str, Any]:
    roots = [Path(path) for path in artifact_roots] if artifact_roots else _default_artifact_roots(root)
    roots = [path if path.is_absolute() else root / path for path in roots]
    counts = {key: 0 for key in ARTIFACT_PATTERNS}
    examples: dict[str, list[str]] = {key: [] for key in ARTIFACT_PATTERNS}

    for artifact_root in roots:
        for path in walk_files(artifact_root):
            if not Path(path).is_file():
                continue
            name = Path(path).name
            for key, pattern in ARTIFACT_PATTERNS.items():
                if not fnmatch.fnmatch(name, pattern):
                    continue
                counts[key] += 1
                if len(examples[key]) < int(example_limit):
                    examples[key].append(str(path))

    missing_evidence: list[str] = []
    if roots and counts["episodes_jsonl"] == 0:
        missing_evidence.append("episodes.jsonl")
    if roots and counts["nvidia_smi_csv"] == 0:
        missing_evidence.append("nvidia-smi CSV")
    if roots and counts["status_json"] == 0:
        missing_evidence.append("status JSON")

    return {
        "roots_scanned": len(roots),
        "roots": [str(path) for path in roots],
        "counts": counts,
        "examples": examples,
        "missing_evidence": missing_evidence,
    }


def build_project_audit(
        root: str | Path,
        artifact_roots: Sequence[str | Path] = (),
        ) -> dict[str, Any]:
    root_path = Path(root).resolve()
    stages = [_stage_report(root_path, stage) for stage in FLOW_STAGES]
    total_files = sum(len(stage["files"]) for stage in stages)
    missing_files = sum(int(stage["missing_files"]) for stage in stages)
    return {
        "root": str(root_path),
        "summary": {
            "total_flow_stages": len(stages),
            "total_expected_files": total_files,
            "present_files": total_files - missing_files,
            "missing_files": missing_files,
        },
        "flow_stages": stages,
        "artifact_summary": summarize_artifacts(root_path, artifact_roots),
        "next_steps": [
            "Run this audit before and after performance work.",
            "Use server_resource_snapshot.py before expensive server runs.",
            "Use stage1_parallel_report.py for Stage-1 rollout/cache timing evidence.",
            "Use gpu_utilization_report.py for run-level GPU evidence.",
            "Use optimization_evidence_bundle.py to package server evidence before promotion.",
            "Use stage2_ngpu_ab_compare.py before promoting Stage-2 GPU defaults.",
            "Keep report rendering off the training hot path when possible.",
        ],
    }


def _format_count(value: object) -> str:
    return str(int(value)) if isinstance(value, int) else str(value)


def render_markdown(report: Mapping[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Project Optimization Audit",
        "",
        f"Root: `{report.get('root', '')}`",
        "",
        "## Summary",
        "",
        f"- Flow stages: {_format_count(summary.get('total_flow_stages', 0))}",
        f"- Expected files: {_format_count(summary.get('total_expected_files', 0))}",
        f"- Present files: {_format_count(summary.get('present_files', 0))}",
        f"- Missing files: {_format_count(summary.get('missing_files', 0))}",
        "",
        "## Flow Stages",
        "",
    ]
    for stage in report.get("flow_stages", []):
        lines.append(f"### {stage['id']}: {stage['name']}")
        lines.append(f"- Present files: {stage['present_files']}")
        lines.append(f"- Missing files: {stage['missing_files']}")
        for item in stage.get("files", []):
            mark = "present" if item.get("present") else "missing"
            lines.append(f"- `{item['path']}`: {mark}")
        surfaces = ", ".join(stage.get("optimization_surfaces", []))
        lines.append(f"- Optimization surfaces: {surfaces}")
        lines.append("")
    artifacts = report.get("artifact_summary", {})
    lines.extend([
        "## Artifact Evidence",
        "",
        f"- Roots scanned: {artifacts.get('roots_scanned', 0)}",
    ])
    counts = artifacts.get("counts", {})
    for key in sorted(counts):
        lines.append(f"- {key}: {counts[key]}")
    missing = artifacts.get("missing_evidence", [])
    lines.append(f"- Missing evidence: {', '.join(missing) if missing else 'none'}")
    lines.append("")
    lines.append("## Next Steps")
    lines.append("")
    for item in report.get("next_steps", []):
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="repository root to audit")
    parser.add_argument(
        "--artifact-root",
        action="append",
        default=[],
        help="artifact root to scan; may be repeated",
    )
    parser.add_argument("--out-json", default="", help="optional JSON output path")
    parser.add_argument("--out-md", default="", help="optional Markdown output path")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_project_audit(args.root, artifact_roots=args.artifact_root)
    if args.out_json:
        Path(args.out_json).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    markdown = render_markdown(report)
    if args.out_md:
        Path(args.out_md).write_text(markdown, encoding="utf-8")
    if not args.out_json and not args.out_md:
        print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
