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
from typing import Any


CHUNK = 1024 * 1024
TEXT_SUFFIXES = {
    ".jsonl",
    ".json",
    ".txt",
    ".log",
    ".md",
    ".html",
    ".sh",
    ".tsv",
    ".csv",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def file_stats(path: Path, *, count_lines: bool) -> tuple[int, str, int | None, bool | None]:
    digest = hashlib.sha256()
    size = 0
    lines = 0
    last = b""
    with path.open("rb") as handle:
        while block := handle.read(CHUNK):
            size += len(block)
            digest.update(block)
            if count_lines:
                lines += block.count(b"\n")
                last = block[-1:]
    ends_with_newline = None
    if count_lines:
        ends_with_newline = last == b"\n" if size else False
    return size, digest.hexdigest(), lines if count_lines else None, ends_with_newline


def stream_name(root_name: str, relative_path: Path) -> str:
    identity = f"{root_name}/{relative_path.as_posix()}"
    short_hash = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", relative_path.name)
    return f"{root_name}__{short_hash}__{safe_name}.gz"


def archive_file(
    root_name: str,
    root: Path,
    relative_path: Path,
    archive_root: Path,
) -> dict[str, Any]:
    source = root / relative_path
    before = source.stat()
    count_lines = source.suffix.lower() in TEXT_SUFFIXES
    use_gzip = source.suffix.lower() == ".jsonl" or before.st_size >= 50 * 1024 * 1024

    if use_gzip:
        archive_relative = Path("streams") / stream_name(root_name, relative_path)
        destination = archive_root / archive_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        raw_digest = hashlib.sha256()
        raw_bytes = 0
        raw_lines = 0
        raw_last = b""
        with source.open("rb") as source_handle, destination.open("wb") as raw_output:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw_output,
                compresslevel=9,
                mtime=0,
            ) as compressed:
                while block := source_handle.read(CHUNK):
                    raw_bytes += len(block)
                    raw_digest.update(block)
                    if count_lines:
                        raw_lines += block.count(b"\n")
                        raw_last = block[-1:]
                    compressed.write(block)
        raw_sha256 = raw_digest.hexdigest()
        raw_ends_with_newline = None
        if count_lines:
            raw_ends_with_newline = raw_last == b"\n" if raw_bytes else False
        storage = "gzip"
    else:
        archive_relative = Path("small_files") / root_name / relative_path
        destination = archive_root / archive_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        raw_bytes, raw_sha256, raw_lines, raw_ends_with_newline = file_stats(
            source,
            count_lines=count_lines,
        )
        if sha256_file(destination) != raw_sha256:
            raise RuntimeError(f"copy hash mismatch: {source}")
        storage = "small"

    after = source.stat()
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise RuntimeError(f"source changed during archive: {source}")

    return {
        "root": root_name,
        "relative_path": relative_path.as_posix(),
        "source_path": str(source),
        "storage": storage,
        "archive_path": archive_relative.as_posix(),
        "raw_bytes": raw_bytes,
        "raw_sha256": raw_sha256,
        "raw_lines": raw_lines,
        "raw_ends_with_newline": raw_ends_with_newline,
        "archive_bytes": destination.stat().st_size,
        "archive_sha256": sha256_file(destination),
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
'''


def parse_root(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("root must be NAME=PATH")
    name, path_text = value.split("=", 1)
    if not name or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise argparse.ArgumentTypeError(f"invalid root name: {name!r}")
    return name, Path(path_text).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", type=parse_root, required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--description", required=True)
    args = parser.parse_args()

    roots = dict(args.root)
    if len(roots) != len(args.root):
        raise SystemExit("duplicate root name")
    for name, root in roots.items():
        if not root.is_dir():
            raise SystemExit(f"missing root {name}: {root}")

    archive_root = Path(args.archive).resolve()
    if archive_root.exists():
        raise SystemExit(f"archive already exists: {archive_root}")
    archive_root.mkdir(parents=True)

    directories: dict[str, list[str]] = {}
    entries: list[dict[str, Any]] = []
    for root_name, root in roots.items():
        directories[root_name] = sorted(
            "." if path == root else path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_dir()
        )
        for source in sorted(path for path in root.rglob("*") if path.is_file()):
            entries.append(
                archive_file(root_name, root, source.relative_to(root), archive_root)
            )

    manifest = {
        "schema": "rfr_full_run_snapshot_v3",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "description": args.description,
        "roots": {name: str(path) for name, path in roots.items()},
        "directories": directories,
        "files": entries,
        "totals": {
            "file_count": len(entries),
            "raw_bytes": sum(int(item["raw_bytes"]) for item in entries),
            "archive_bytes": sum(int(item["archive_bytes"]) for item in entries),
            "gzip_file_count": sum(item["storage"] == "gzip" for item in entries),
            "small_file_count": sum(item["storage"] == "small" for item in entries),
        },
    }
    (archive_root / "snapshot_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (archive_root / "restore_snapshot.py").write_text(RESTORE_SCRIPT, encoding="utf-8")
    os.chmod(archive_root / "restore_snapshot.py", 0o755)

    with (archive_root / "stream_map.tsv").open("w", encoding="utf-8") as handle:
        handle.write(
            "root\trelative_path\tstorage\tarchive_path\traw_bytes\traw_lines\traw_sha256\n"
        )
        for entry in entries:
            raw_lines = "" if entry["raw_lines"] is None else entry["raw_lines"]
            handle.write(
                f"{entry['root']}\t{entry['relative_path']}\t{entry['storage']}\t"
                f"{entry['archive_path']}\t{entry['raw_bytes']}\t{raw_lines}\t"
                f"{entry['raw_sha256']}\n"
            )

    print(json.dumps(manifest["totals"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
