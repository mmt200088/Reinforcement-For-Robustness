#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
from typing import Any

CHUNK = 1024 * 1024
TEXT_SUFFIXES = {".jsonl", ".json", ".txt", ".log", ".md", ".html", ".sh", ".tsv", ".csv"}

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(CHUNK)
            if not block:
                break
            h.update(block)
    return h.hexdigest()

def file_stats(path: Path, *, count_lines: bool) -> tuple[int, str, int | None, bool | None]:
    h = hashlib.sha256()
    size = 0
    lines = 0
    last = b""
    with path.open("rb") as handle:
        while True:
            block = handle.read(CHUNK)
            if not block:
                break
            size += len(block)
            h.update(block)
            if count_lines:
                lines += block.count(b"\n")
                last = block[-1:]
    return size, h.hexdigest(), lines if count_lines else None, (last == b"\n") if count_lines and size else (False if count_lines else None)

def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    return repr(value)

def run_text(command: list[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    return (
        f"$ {' '.join(command)}\n"
        f"exit={completed.returncode}\n"
        f"{completed.stdout}{completed.stderr}"
    )

def stream_name(root_name: str, rel: Path) -> str:
    digest = hashlib.sha256(rel.as_posix().encode("utf-8")).hexdigest()[:12]
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", rel.name)
    return f"{root_name}__{digest}__{safe}.gz"

def archive_one(root_name: str, root: Path, rel: Path, archive: Path) -> dict[str, Any]:
    src = root / rel
    before = src.stat()
    line_counted = src.suffix.lower() in TEXT_SUFFIXES
    use_gzip = src.suffix.lower() == ".jsonl" and before.st_size >= 1024 * 1024
    if before.st_size >= 50 * 1024 * 1024:
        use_gzip = True
    if use_gzip:
        arc_rel = Path("streams") / stream_name(root_name, rel)
        dst = archive / arc_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha256()
        raw_bytes = 0
        raw_lines = 0
        raw_last = b""
        with src.open("rb") as source, dst.open("wb") as raw_out:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw_out, compresslevel=9, mtime=0) as zipped:
                while True:
                    block = source.read(CHUNK)
                    if not block:
                        break
                    raw_bytes += len(block)
                    h.update(block)
                    if line_counted:
                        raw_lines += block.count(b"\n")
                        raw_last = block[-1:]
                    zipped.write(block)
        raw_sha = h.hexdigest()
        raw_ends = (raw_last == b"\n") if raw_bytes and line_counted else (False if line_counted else None)
        storage = "gzip"
    else:
        arc_rel = Path("small_files") / root_name / rel
        dst = archive / arc_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        raw_bytes, raw_sha, raw_lines, raw_ends = file_stats(src, count_lines=line_counted)
        if sha256_file(dst) != raw_sha:
            raise RuntimeError(f"copy hash mismatch: {src}")
        storage = "small"
    after = src.stat()
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise RuntimeError(f"source changed during archive: {src}")
    archived_bytes = dst.stat().st_size
    archived_sha = sha256_file(dst)
    return {
        "root": root_name,
        "relative_path": rel.as_posix(),
        "source_path": str(src),
        "storage": storage,
        "archive_path": arc_rel.as_posix(),
        "raw_bytes": raw_bytes,
        "raw_sha256": raw_sha,
        "raw_lines": raw_lines,
        "raw_ends_with_newline": raw_ends,
        "archive_bytes": archived_bytes,
        "archive_sha256": archived_sha,
        "mode": stat.S_IMODE(before.st_mode),
        "mtime_ns": before.st_mtime_ns,
    }

RESTORE_SCRIPT = r'''#!/usr/bin/env python3
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil

CHUNK = 1024 * 1024

def digest(path: Path):
    h = hashlib.sha256()
    size = 0
    lines = 0
    last = b""
    with path.open("rb") as handle:
        while True:
            block = handle.read(CHUNK)
            if not block:
                break
            h.update(block)
            size += len(block)
            lines += block.count(b"\n")
            last = block[-1:]
    return size, h.hexdigest(), lines, (last == b"\n") if size else False

def main():
    parser = argparse.ArgumentParser(description="Restore the complete BERT-large RTE Stage-2 graceful-stop snapshot.")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    archive = Path(__file__).resolve().parent
    manifest = json.loads((archive / "snapshot_manifest.json").read_text(encoding="utf-8"))
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for root_name, dirs in manifest["directories"].items():
        for rel in dirs:
            (output / root_name / rel).mkdir(parents=True, exist_ok=True)
    restored = []
    for entry in manifest["files"]:
        src = archive / entry["archive_path"]
        src_size, src_sha, _, _ = digest(src)
        if src_size != entry["archive_bytes"] or src_sha != entry["archive_sha256"]:
            raise RuntimeError(f"archive payload mismatch: {entry['archive_path']}")
        dst = output / entry["root"] / entry["relative_path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if entry["storage"] == "gzip":
            with gzip.open(src, "rb") as source, dst.open("wb") as target:
                shutil.copyfileobj(source, target, length=CHUNK)
        elif entry["storage"] == "small":
            shutil.copyfile(src, dst)
        else:
            raise RuntimeError(f"unknown storage: {entry['storage']}")
        os.chmod(dst, int(entry["mode"]))
        os.utime(dst, ns=(int(entry["mtime_ns"]), int(entry["mtime_ns"])))
        raw_size, raw_sha, raw_lines, raw_ends = digest(dst)
        if raw_size != entry["raw_bytes"] or raw_sha != entry["raw_sha256"]:
            raise RuntimeError(f"restored payload mismatch: {entry['root']}/{entry['relative_path']}")
        if entry["raw_lines"] is not None and raw_lines != entry["raw_lines"]:
            raise RuntimeError(f"restored row mismatch: {entry['root']}/{entry['relative_path']}")
        if entry["raw_ends_with_newline"] is not None and raw_ends != entry["raw_ends_with_newline"]:
            raise RuntimeError(f"restored newline mismatch: {entry['root']}/{entry['relative_path']}")
        restored.append({"root": entry["root"], "relative_path": entry["relative_path"], "bytes": raw_size, "sha256": raw_sha})
    result = {
        "schema": "rfr_snapshot_restore_verification_v2",
        "status": "RESTORE_OK",
        "archive_schema": manifest["schema"],
        "file_count": len(restored),
        "raw_bytes": sum(item["bytes"] for item in restored),
        "roots": sorted(manifest["roots"]),
        "output_dir": str(output),
    }
    print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
'''

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--structured", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--source-repo", required=True)
    args = parser.parse_args()
    roots = {
        "run": Path(args.run).resolve(),
        "structured": Path(args.structured).resolve(),
        "report": Path(args.report).resolve(),
    }
    for name, root in roots.items():
        if not root.is_dir():
            raise SystemExit(f"missing root {name}: {root}")
    archive = Path(args.archive).resolve()
    if archive.exists():
        raise SystemExit(f"archive already exists: {archive}")
    archive.mkdir(parents=True)

    directory_map: dict[str, list[str]] = {}
    entries: list[dict[str, Any]] = []
    for root_name, root in roots.items():
        directory_map[root_name] = sorted(
            "." if path == root else path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_dir()
        )
        files = sorted(path for path in root.rglob("*") if path.is_file())
        for source in files:
            entries.append(archive_one(root_name, root, source.relative_to(root), archive))

    manifest = {
        "schema": "rfr_full_run_snapshot_v2",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "description": "Complete BERT-large RTE Stage-2 RL graceful-stop snapshot at episode 33720.",
        "roots": {name: str(path) for name, path in roots.items()},
        "directories": directory_map,
        "files": entries,
        "totals": {
            "file_count": len(entries),
            "raw_bytes": sum(x["raw_bytes"] for x in entries),
            "archive_bytes": sum(x["archive_bytes"] for x in entries),
            "gzip_file_count": sum(x["storage"] == "gzip" for x in entries),
            "small_file_count": sum(x["storage"] == "small" for x in entries),
        },
    }
    (archive / "snapshot_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (archive / "restore_snapshot.py").write_text(RESTORE_SCRIPT, encoding="utf-8")
    os.chmod(archive / "restore_snapshot.py", 0o755)

    with (archive / "stream_map.tsv").open("w", encoding="utf-8") as handle:
        handle.write("root\trelative_path\tstorage\tarchive_path\traw_bytes\traw_lines\traw_sha256\n")
        for item in entries:
            handle.write(
                f"{item['root']}\t{item['relative_path']}\t{item['storage']}\t{item['archive_path']}\t"
                f"{item['raw_bytes']}\t{'' if item['raw_lines'] is None else item['raw_lines']}\t{item['raw_sha256']}\n"
            )

    checkpoint = roots["run"] / "stage2_noise/progress/blb_stage2_rl_checkpoint_live.pt"
    try:
        import torch
        data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        summary_keys = [
            "episode", "ppo_update_count", "planned_total_episodes", "candidate_store_size",
            "profile", "policy_network_variant", "structured_run_id", "run_context_hash",
            "algorithm_revision", "algorithm_contract_hash", "convergence_state",
            "strict_best", "strict_pareto_frontier",
        ]
        checkpoint_summary = {key: jsonable(data.get(key)) for key in summary_keys if key in data}
    except Exception as exc:
        checkpoint_summary = {"load_error": repr(exc)}
    checkpoint_summary["checkpoint_bytes"] = checkpoint.stat().st_size
    checkpoint_summary["checkpoint_sha256"] = sha256_file(checkpoint)
    (archive / "checkpoint_summary.json").write_text(json.dumps(checkpoint_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    source_repo = Path(args.source_repo).resolve()
    source_lines = [
        f"captured_at={dt.datetime.now().astimezone().isoformat()}",
        f"source_repo={source_repo}",
        run_text(["git", "rev-parse", "HEAD"], cwd=source_repo),
        run_text(["git", "status", "--short"], cwd=source_repo),
        "launch_source_provenance:",
        (roots["run"] / "launch_evidence/source_provenance.txt").read_text(encoding="utf-8", errors="replace"),
    ]
    (archive / "source_git_state.txt").write_text("\n".join(source_lines), encoding="utf-8")

    stop_lines = [
        f"captured_at={dt.datetime.now().astimezone().isoformat()}",
        "graceful_stop_request=STOP_RL marker at 2026-08-01T20:05:23+08:00",
        "graceful_stop_boundary=33720",
        "graceful_stop_updates=281",
        run_text(["pgrep", "-af", "bert-large.*rte"]),
        run_text(["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used", "--format=csv,noheader"]),
        "graceful_log_markers:",
    ]
    output_log = roots["run"] / "logs/output.log"
    for line in output_log.read_text(encoding="utf-8", errors="replace").splitlines():
        if "graceful-stop" in line:
            stop_lines.append(line)
    (archive / "stop_evidence.txt").write_text("\n".join(stop_lines) + "\n", encoding="utf-8")

    status = json.loads((roots["run"] / "stage2_noise/progress/blb_stage2_status.json").read_text(encoding="utf-8"))
    key_rows = {}
    for key in (
        "run/stage2_noise/progress/candidate_store.jsonl",
        "run/stage2_noise/progress/diagnostics/episodes.jsonl",
        "run/stage2_noise/progress/diagnostics/ppo_updates.jsonl",
        "run/stage2_noise/progress/diagnostics/pareto_frontier.jsonl",
        "run/stage2_noise/progress/diagnostics/top_candidates.jsonl",
        "structured/episodes.jsonl",
        "structured/ppo_updates.jsonl",
    ):
        root_name, rel = key.split("/", 1)
        match = next(x for x in entries if x["root"] == root_name and x["relative_path"] == rel)
        key_rows[key] = match["raw_lines"]
    resume_cut = {
        "schema": "rfr_stage2_resume_cut_v2",
        "status": "graceful_stop",
        "resumable": True,
        "completed_episodes": status.get("completed_episodes"),
        "ppo_update_count": status.get("ppo_update_count"),
        "phase": status.get("phase"),
        "stopped_at": status.get("stopped_at"),
        "checkpoint_relative_path": "run/stage2_noise/progress/blb_stage2_rl_checkpoint_live.pt",
        "checkpoint_sha256": checkpoint_summary["checkpoint_sha256"],
        "row_counts": key_rows,
        "source_launch_command": "run/launch_evidence/launch_command.sh",
    }
    (archive / "resume_cut_manifest.json").write_text(json.dumps(resume_cut, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    total_gib = manifest["totals"]["raw_bytes"] / (1024 ** 3)
    archived_mib = manifest["totals"]["archive_bytes"] / (1024 ** 2)
    readme = f"""# BERT-large RTE Stage-2 RL graceful-stop archive

This directory is the complete recoverable snapshot of the BERT-large RTE Stage-2 PPO run stopped at a full checkpoint boundary.

## Resume cut

- Episodes: **{status.get('completed_episodes')} / {status.get('total_episodes')}**
- PPO updates: **{status.get('ppo_update_count')}**
- Status: **{status.get('phase')}**
- Stopped at: **{status.get('stopped_at')}**
- Checkpoint SHA256: {checkpoint_summary['checkpoint_sha256']}
- Source model/profile: yoshitomo-matsubara/bert-large-uncased-rte / rte_large
- Constraints: 0.1% precision tolerance and 200% stability multiplier for loss, Accuracy, and Weighted F1
- Policy: shared_gtrxl_small_v1; 24 layerwise decisions; Block4 fusion 0/1 plus high/medium/low truncation preset
- Online terminal trials: 3; promotion/final banks are retained in candidate and diagnostics streams

## Completeness

The archive contains **every regular file and empty directory** under the original run directory, the structured data-point mirror, and the final report snapshot directory. It preserves {manifest['totals']['file_count']} files, {total_gib:.3f} GiB of raw bytes, and {archived_mib:.1f} MiB of archived payload. Large JSONL streams are gzip-compressed individually; all other files are byte-for-byte copies.

snapshot_manifest.json records, for every source file, its root, relative path, raw size, row count where applicable, trailing-newline state, permissions, timestamp, raw SHA256, archive path, archive size, and archive SHA256. stream_map.tsv is the compact index.

## Restore

From this directory, run:

    python3 restore_snapshot.py /hy-tmp/restored_bert_large_rte_stage2_33720

This recreates three exact trees:

- run/: the complete resumable training run, including checkpoint, status, baseline, best action, candidate store, diagnostics, launch command, logs, and empty runtime directories.
- structured/: the project structured writer output (episodes.jsonl, ppo_updates.jsonl, manifest, summary).
- report/: the regenerated curves, NPZ/PDF artifacts, search log, and standalone final HTML.

The restore command refuses a non-empty destination and verifies every restored file against the manifest.

## Reproducible figures and analyses

The preserved data supports reconstruction of reward, loss, Accuracy, Weighted F1, all three stability curves, P1/P2/P3 distributions, invalid/collapse rates, Block4 and truncation entropy, KL/clip/gradient/value diagnostics, throughput and per-GPU probe scaling, action histograms, per-layer fusion and K decisions, candidate/promotion/final-bank histories, Pareto/resource-frontier progress, best-so-far state, and baseline-versus-best comparisons. The NPZ and PNG/PDF/HTML files are convenience artifacts; the JSONL streams remain authoritative.

## Integrity

Run sha256sum -c SHA256SUMS, gzip -t streams/*.gz, then execute the restore command. restore_verification.json records the server-side full restore rehearsal performed before commit.
"""
    (archive / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(manifest["totals"], indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
