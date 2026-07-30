#!/usr/bin/env python3
"""Restore and verify the archived Stage-2 training state."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path


STREAM_TARGETS = {
    "streams/progress_candidate_store.jsonl.gz": "progress/candidate_store.jsonl",
    "streams/diagnostics_episodes.jsonl.gz": "progress/diagnostics/episodes.jsonl",
    "streams/diagnostics_ppo_updates.jsonl.gz": "progress/diagnostics/ppo_updates.jsonl",
    "streams/diagnostics_pareto_frontier.jsonl.gz": "progress/diagnostics/pareto_frontier.jsonl",
    "streams/diagnostics_top_candidates.jsonl.gz": "progress/diagnostics/top_candidates.jsonl",
    "streams/structured_episodes.jsonl.gz": "structured/episodes.jsonl",
    "streams/structured_ppo_updates.jsonl.gz": "structured/ppo_updates.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("resume", "full"), default="resume")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def copy_prefix(source: Path, target: Path, limit: int, expected_hash: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    remaining = limit
    last = b""
    with gzip.open(source, "rb") as src, target.open("wb") as dst:
        while remaining:
            chunk = src.read(min(4 << 20, remaining))
            if not chunk:
                raise RuntimeError(f"{source} ended {remaining} bytes early")
            dst.write(chunk)
            digest.update(chunk)
            remaining -= len(chunk)
            last = chunk[-1:]
    if last != b"\n":
        raise RuntimeError(f"{source} restore boundary is not a complete JSONL row")
    if digest.hexdigest() != expected_hash:
        raise RuntimeError(f"{source} raw SHA-256 mismatch")


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    for name in ("progress", "structured"):
        source = root / "small_files" / name
        if source.exists():
            shutil.copytree(source, output / name, dirs_exist_ok=True)

    full_manifest = json.loads((root / "snapshot_manifest.json").read_text())
    resume_manifest = json.loads((root / "resume_cut_manifest.json").read_text())
    metadata = (
        resume_manifest["streams"]
        if args.mode == "resume"
        else full_manifest["streams"]
    )

    for source_rel, target_rel in STREAM_TARGETS.items():
        if source_rel not in metadata:
            if args.mode == "resume":
                continue
            raise RuntimeError(f"missing stream metadata: {source_rel}")
        item = metadata[source_rel]
        byte_key = "raw_bytes" if args.mode == "resume" else "raw_complete_bytes"
        hash_key = "sha256" if args.mode == "resume" else "raw_sha256"
        copy_prefix(
            root / source_rel,
            output / target_rel,
            int(item[byte_key]),
            str(item[hash_key]),
        )

    print(f"RESTORE_OK mode={args.mode} output={output}")


if __name__ == "__main__":
    main()
