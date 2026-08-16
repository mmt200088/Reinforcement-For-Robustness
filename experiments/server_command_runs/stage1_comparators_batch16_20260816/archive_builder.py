#!/usr/bin/env python3
import csv
import datetime
import hashlib
import json
from pathlib import Path
import statistics
import sys


root = Path(sys.argv[1])
validations = {
    algorithm: json.loads((root / "logs" / f"{algorithm}_validation.log").read_text())
    for algorithm in ("greedy", "bo_rf", "coinn_ga")
}
rows = []
with (root / "gpu_samples.csv").open() as handle:
    next(handle)
    for row in csv.reader(handle):
        if len(row) >= 6:
            rows.append(
                (
                    float(row[3].strip()),
                    float(row[4].strip()),
                    float(row[5].strip()),
                )
            )

summary = {
    "schema_version": "comparator_stage1_batch16_run_summary_v1",
    "authoritative": True,
    "completed_at": "2026-08-16T21:03:12+08:00",
    "dataset": "mrpc",
    "model": "textattack/bert-base-uncased-MRPC",
    "validation_examples": 408,
    "batch_size": 16,
    "micro_batch_size": 16,
    "stage1_accuracy_tolerance": 0.001,
    "source_commit": "9d833d90760b1bf85fca4c8650e8149f61119ad2",
    "source_tree": "918fb6e4f5e6ea6fa659a30045331f99dc48800e",
    "order": ["greedy", "bo_rf", "coinn_ga"],
    "wall_seconds": {"greedy": 372, "bo_rf": 729, "coinn_ga": 4486},
    "results": validations,
    "gpu": {
        "model": "NVIDIA GeForce RTX 4090",
        "samples": len(rows),
        "utilization_mean_percent": sum(row[0] for row in rows) / len(rows),
        "utilization_median_percent": statistics.median(row[0] for row in rows),
        "utilization_p95_percent": sorted(row[0] for row in rows)[
            int(0.95 * (len(rows) - 1))
        ],
        "memory_max_mib": max(row[1] for row in rows),
        "power_mean_w": sum(row[2] for row in rows) / len(rows),
    },
}
(root / "RUN_SUMMARY.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)

supersedes = {
    "schema_version": "comparator_result_supersession_v1",
    "authoritative_run_root": str(root),
    "reason": (
        "Historical Stage-1 RL alignment requires batch_size=16; prior "
        "comparator formal runs used batch_size=64."
    ),
    "superseded_server_roots_deleted": [
        "/hy-tmp/comparator_stage1_queue_20260812_v3",
        "/hy-tmp/comparator_stage1_bo_rf_10k_20260815",
        "/hy-tmp/comparator_stage1_ga_200_20260814",
    ],
    "superseded_result_branches_retained_as_immutable_history": {
        "codex/result-stage1-greedy-20260812": (
            "afd9efd46c3a9b4d75e74650b3ac7c96b512adfa"
        ),
        "codex/result-stage1-bo-rf-10k-20260815": (
            "ed6e60195267348996b5de0c1e9daf741c008498"
        ),
        "codex/result-stage1-ga-200-20260814": (
            "4edf8f62a24d3f0a10960b001b9b678843a3dd71"
        ),
    },
}
(root / "SUPERSEDES.json").write_text(
    json.dumps(supersedes, indent=2, sort_keys=True) + "\n"
)

readme = f"""# Stage-1 comparator batch16 formal run

This is the authoritative replacement for the prior batch64 Greedy, BO-RF,
and COINN-GA Stage-1 runs.

- Source commit: `{summary["source_commit"]}`
- Source tree: `{summary["source_tree"]}`
- Model/dataset: `textattack/bert-base-uncased-MRPC` / GLUE MRPC validation (408 examples)
- Batch/micro-batch: 16 / 16
- Order: Greedy, BO-RF, COINN-GA
- Greedy: {validations["greedy"]["evaluation_count"]} evaluations, `{validations["greedy"]["termination_reason"]}`
- BO-RF: {validations["bo_rf"]["evaluation_count"]} evaluations, `{validations["bo_rf"]["termination_reason"]}`
- COINN-GA: {validations["coinn_ga"]["evaluation_count"]} evaluations, `{validations["coinn_ga"]["termination_reason"]}` (200 complete generations)

All persistent observations, histories, checkpoints, manifests, summaries,
model logs, launcher logs, GPU samples, queue state, and validation summaries
are included. `SHA256SUMS` verifies every archived file other than itself.
"""
(root / "README.md").write_text(readme)

files = []
for path in sorted(root.rglob("*")):
    if path.is_file() and path.name not in {"FILE_MANIFEST.json", "SHA256SUMS"}:
        data = path.read_bytes()
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
manifest = {
    "schema_version": "comparator_stage1_batch16_file_manifest_v1",
    "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "file_count_excluding_manifest_and_sha256sums": len(files),
    "files": files,
}
(root / "FILE_MANIFEST.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
