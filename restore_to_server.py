#!/usr/bin/env python3
"""Restore and verify the content-addressed RFR data overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile


BUFFER_SIZE = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(BUFFER_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_hash(path: Path) -> str:
    digest = hashlib.sha1()
    size = path.stat().st_size
    digest.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as handle:
        while chunk := handle.read(BUFFER_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def safe_target(root: Path, relative: str) -> Path:
    root = root.resolve()
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"unsafe relative path: {relative!r}")
    target = (root / candidate).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"path escapes target root: {relative!r}")
    return target


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def overlay_specs(archive_dir: Path) -> list[tuple[str, Path, Path]]:
    specs = [
        (
            "overlay",
            archive_dir / "overlay_manifest.jsonl",
            archive_dir / "chunks.json",
        )
    ]
    ignored_manifest = archive_dir / "ignored_overlay_manifest.jsonl"
    ignored_chunks = archive_dir / "ignored_chunks.json"
    if ignored_manifest.is_file() or ignored_chunks.is_file():
        if not ignored_manifest.is_file() or not ignored_chunks.is_file():
            raise FileNotFoundError("incomplete ignored-overlay control files")
        specs.append(("ignored_overlay", ignored_manifest, ignored_chunks))
    return specs


def verify_control_files(archive_dir: Path) -> None:
    sums = archive_dir / "SHA256SUMS"
    for line in sums.read_text(encoding="ascii").splitlines():
        expected, name = line.split("  ", 1)
        path = archive_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"archive control file missing: {name}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"archive checksum mismatch for {name}: {actual} != {expected}"
            )


def verify_base(archive_dir: Path, target: Path) -> None:
    rows = read_jsonl(archive_dir / "base_tracked_manifest.jsonl")
    failures: list[str] = []
    for row in rows:
        path = safe_target(target, row["path"])
        if not path.is_file():
            failures.append(f"missing: {row['path']}")
            continue
        actual = git_blob_hash(path)
        if actual != row["git_blob"]:
            failures.append(
                f"blob mismatch: {row['path']} {actual} != {row['git_blob']}"
            )
    if failures:
        sample = "\n".join(failures[:20])
        raise RuntimeError(
            f"base verification failed for {len(failures)} file(s):\n{sample}"
        )
    print(f"BASE_VERIFY_OK files={len(rows)}")


def verify_overlay(archive_dir: Path, target: Path) -> None:
    failures: list[str] = []
    total_files = 0
    for _, manifest_path, _ in overlay_specs(archive_dir):
        rows = read_jsonl(manifest_path)
        total_files += len(rows)
        for row in rows:
            path = safe_target(target, row["path"])
            if not path.is_file():
                failures.append(f"missing: {row['path']}")
                continue
            actual = sha256_file(path)
            if actual != row["sha256"]:
                failures.append(
                    f"sha256 mismatch: {row['path']} {actual} != {row['sha256']}"
                )
    if failures:
        sample = "\n".join(failures[:20])
        raise RuntimeError(
            f"overlay verification failed for {len(failures)} file(s):\n{sample}"
        )
    print(f"OVERLAY_VERIFY_OK files={total_files}")


def extract_objects(
    archive_dir: Path,
    temporary_dir: Path,
    overlay_name: str,
    chunks_path: Path,
) -> Path:
    chunk_manifest = json.loads(chunks_path.read_text(encoding="utf-8"))
    payload = temporary_dir / f"{overlay_name}_objects.tar.gz"
    with payload.open("wb") as output:
        for part in chunk_manifest["parts"]:
            path = archive_dir / part["name"]
            actual = sha256_file(path)
            if actual != part["sha256"]:
                raise RuntimeError(f"chunk checksum mismatch: {part['name']}")
            with path.open("rb") as source:
                shutil.copyfileobj(source, output, BUFFER_SIZE)
    actual_payload = sha256_file(payload)
    if actual_payload != chunk_manifest["payload_sha256"]:
        raise RuntimeError(
            "combined payload checksum mismatch: "
            f"{actual_payload} != {chunk_manifest['payload_sha256']}"
        )

    object_root = temporary_dir / f"{overlay_name}_payload"
    object_root.mkdir()
    with tarfile.open(payload, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                raise RuntimeError(f"unexpected non-file payload member: {member.name}")
            safe_target(object_root, member.name)
        archive.extractall(object_root)
    return object_root


def restore(archive_dir: Path, target: Path) -> None:
    all_rows: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="rfr-data-restore-") as temporary:
        for overlay_name, manifest_path, chunks_path in overlay_specs(archive_dir):
            rows = read_jsonl(manifest_path)
            all_rows.extend(rows)
            object_root = extract_objects(
                archive_dir,
                Path(temporary),
                overlay_name,
                chunks_path,
            )
            verified_objects: set[str] = set()
            for row in rows:
                digest = row["sha256"]
                source = object_root / row["object_path"]
                if digest not in verified_objects:
                    actual = sha256_file(source)
                    if actual != digest:
                        raise RuntimeError(
                            f"payload object checksum mismatch: {digest} != {actual}"
                        )
                    verified_objects.add(digest)

                destination = safe_target(target, row["path"])
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary_destination = destination.with_name(
                    f".{destination.name}.rfr-restore-tmp"
                )
                shutil.copyfile(source, temporary_destination)
                os.chmod(temporary_destination, int(row["mode"]))
                os.utime(
                    temporary_destination,
                    ns=(int(row["mtime_ns"]), int(row["mtime_ns"])),
                )
                os.replace(temporary_destination, destination)

    verify_overlay(archive_dir, target)
    print(
        "RESTORE_OK "
        f"files={len(all_rows)} "
        f"unique_objects={len({row['sha256'] for row in all_rows})}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument("--target", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--verify-base-only", action="store_true")
    modes.add_argument("--verify-overlay-only", action="store_true")
    args = parser.parse_args()

    archive_dir = args.archive_dir.resolve()
    target = args.target.resolve()
    if not target.is_dir():
        raise NotADirectoryError(target)
    verify_control_files(archive_dir)

    if args.verify_base_only:
        verify_base(archive_dir, target)
    elif args.verify_overlay_only:
        verify_overlay(archive_dir, target)
    else:
        restore(archive_dir, target)


if __name__ == "__main__":
    main()
