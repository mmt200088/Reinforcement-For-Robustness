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
    parser = argparse.ArgumentParser(description="Restore the complete BERT-large SST2 Stage-2 graceful-stop snapshot.")
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
