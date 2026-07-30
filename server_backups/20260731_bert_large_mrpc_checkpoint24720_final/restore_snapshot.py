#!/usr/bin/env python3
"""Restore and verify the stopped BERT-large MRPC Stage-2 training state."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path


STREAM_TARGETS = {
    "streams/progress_candidate_store.jsonl.gz": (
        "run/stage2_noise/progress/candidate_store.jsonl"
    ),
    "streams/diagnostics_episodes.jsonl.gz": (
        "run/stage2_noise/progress/diagnostics/episodes.jsonl"
    ),
    "streams/diagnostics_ppo_updates.jsonl.gz": (
        "run/stage2_noise/progress/diagnostics/ppo_updates.jsonl"
    ),
    "streams/diagnostics_pareto_frontier.jsonl.gz": (
        "run/stage2_noise/progress/diagnostics/pareto_frontier.jsonl"
    ),
    "streams/diagnostics_top_candidates.jsonl.gz": (
        "run/stage2_noise/progress/diagnostics/top_candidates.jsonl"
    ),
    "streams/structured_episodes.jsonl.gz": "structured/episodes.jsonl",
    "streams/structured_ppo_updates.jsonl.gz": "structured/ppo_updates.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("resume", "full"),
        default="resume",
        help="Both modes are byte-identical because this archive ends at a checkpoint.",
    )
    return parser.parse_args()


def restore_stream(
    source: Path,
    target: Path,
    expected_bytes: int,
    expected_rows: int,
    expected_hash: str,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    raw_bytes = 0
    rows = 0
    last_byte = b""

    with gzip.open(source, "rb") as src, target.open("wb") as dst:
        while chunk := src.read(4 << 20):
            dst.write(chunk)
            digest.update(chunk)
            raw_bytes += len(chunk)
            rows += chunk.count(b"\n")
            last_byte = chunk[-1:]

    if raw_bytes != expected_bytes:
        raise RuntimeError(
            f"{source}: raw size {raw_bytes} != expected {expected_bytes}"
        )
    if rows != expected_rows:
        raise RuntimeError(f"{source}: rows {rows} != expected {expected_rows}")
    if last_byte != b"\n":
        raise RuntimeError(f"{source}: restored JSONL does not end on a row boundary")
    if digest.hexdigest() != expected_hash:
        raise RuntimeError(f"{source}: raw SHA-256 mismatch")


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    shutil.copytree(
        root / "small_files" / "run",
        output / "run",
        dirs_exist_ok=True,
    )
    shutil.copytree(
        root / "small_files" / "structured",
        output / "structured",
        dirs_exist_ok=True,
    )

    manifest = json.loads((root / "snapshot_manifest.json").read_text())
    for source_rel, target_rel in STREAM_TARGETS.items():
        item = manifest["streams"][source_rel]
        restore_stream(
            root / source_rel,
            output / target_rel,
            int(item["raw_bytes"]),
            int(item["rows"]),
            str(item["raw_sha256"]),
        )

    print(
        "RESTORE_OK "
        f"mode={args.mode} episode={manifest['checkpoint']['episode']} "
        f"updates={manifest['checkpoint']['ppo_update_count']} output={output}"
    )


if __name__ == "__main__":
    main()
