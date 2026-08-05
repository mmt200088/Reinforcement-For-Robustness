#!/usr/bin/env python3
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
    value = hashlib.sha256()
    size = 0
    lines = 0
    last = b""
    with path.open("rb") as handle:
        while block := handle.read(CHUNK):
            value.update(block)
            size += len(block)
            lines += block.count(b"\n")
            last = block[-1:]
    return size, value.hexdigest(), lines, (last == b"\n") if size else False


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore the archived Stage-2 training snapshot.")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    archive_root = Path(__file__).resolve().parent
    manifest = json.loads((archive_root / "snapshot_manifest.json").read_text(encoding="utf-8"))
    output_root = Path(args.output_dir).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"refusing non-empty output directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    for root_name, directories in manifest["directories"].items():
        for relative_path in directories:
            (output_root / root_name / relative_path).mkdir(parents=True, exist_ok=True)

    restored = []
    for entry in manifest["files"]:
        archived = archive_root / entry["archive_path"]
        archived_size, archived_sha, _, _ = digest(archived)
        if archived_size != entry["archive_bytes"] or archived_sha != entry["archive_sha256"]:
            raise RuntimeError(f"archive payload mismatch: {entry['archive_path']}")

        destination = output_root / entry["root"] / entry["relative_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if entry["storage"] == "gzip":
            with gzip.open(archived, "rb") as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target, length=CHUNK)
        elif entry["storage"] == "small":
            shutil.copyfile(archived, destination)
        else:
            raise RuntimeError(f"unknown storage: {entry['storage']}")

        os.chmod(destination, int(entry["mode"]))
        os.utime(destination, ns=(int(entry["mtime_ns"]), int(entry["mtime_ns"])))
        raw_size, raw_sha, raw_lines, raw_ends = digest(destination)
        if raw_size != entry["raw_bytes"] or raw_sha != entry["raw_sha256"]:
            raise RuntimeError(f"restored payload mismatch: {entry['root']}/{entry['relative_path']}")
        if entry["raw_lines"] is not None and raw_lines != entry["raw_lines"]:
            raise RuntimeError(f"restored row mismatch: {entry['root']}/{entry['relative_path']}")
        if entry["raw_ends_with_newline"] is not None and raw_ends != entry["raw_ends_with_newline"]:
            raise RuntimeError(f"restored newline mismatch: {entry['root']}/{entry['relative_path']}")
        restored.append(entry)

    print(json.dumps({
        "schema": "rfr_snapshot_restore_verification_v3",
        "status": "RESTORE_OK",
        "file_count": len(restored),
        "raw_bytes": sum(int(item["raw_bytes"]) for item in restored),
        "roots": sorted(manifest["roots"]),
        "output_dir": str(output_root),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
