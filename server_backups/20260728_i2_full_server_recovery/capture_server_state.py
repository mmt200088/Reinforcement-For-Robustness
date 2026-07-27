#!/usr/bin/env python3
"""Capture a content-addressed inventory of project state under /hy-tmp.

Tracked worktree content is represented by Git commit IDs. Server-only files,
including ignored run logs but excluding disposable interpreter caches, are
listed for an exact tar payload. Hugging Face cache blobs receive a separate
hash inventory because model weights are reproducible inputs rather than run
results and are too large for ordinary Git hosting.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Iterable, Iterator


CACHE_COMPONENTS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}
CACHE_SUFFIXES = {".pyc", ".pyo"}


def run(
    *args: str,
    cwd: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        args,
        cwd=cwd,
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def decode_paths(payload: bytes) -> list[str]:
    return [
        os.fsdecode(value)
        for value in payload.split(b"\0")
        if value
    ]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_path(path: Path) -> tuple[str, int, str]:
    if path.is_symlink():
        payload = os.readlink(path).encode("utf-8", "surrogateescape")
        return sha256_bytes(payload), len(payload), "symlink"
    if path.is_file():
        return sha256_file(path), path.stat().st_size, "file"
    raise ValueError(f"payload path is not a regular file or symlink: {path}")


def is_disposable_cache(relative_path: str) -> bool:
    path = Path(relative_path)
    return (
        any(component in CACHE_COMPONENTS for component in path.parts)
        or path.suffix in CACHE_SUFFIXES
    )


def iter_regular_paths(root: Path) -> Iterator[Path]:
    if root.is_symlink() or root.is_file():
        yield root
        return
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        retained_directories = []
        for name in directory_names:
            path = current_path / name
            if path.is_symlink():
                yield path
            else:
                retained_directories.append(name)
        directory_names[:] = retained_directories
        for name in file_names:
            path = current_path / name
            mode = path.lstat().st_mode
            if stat.S_ISREG(mode) or stat.S_ISLNK(mode):
                yield path


def discover_worktrees(hy_tmp: Path) -> list[Path]:
    return sorted(
        path
        for path in hy_tmp.iterdir()
        if path.is_dir() and (path / ".git").exists()
    )


def git_extra_paths(worktree: Path) -> tuple[set[Path], int, int]:
    untracked = decode_paths(run(
        "git", "ls-files", "-z", "--others", "--exclude-standard",
        cwd=worktree,
    ).stdout)
    ignored = decode_paths(run(
        "git", "ls-files", "-z", "--others", "--ignored", "--exclude-standard",
        cwd=worktree,
    ).stdout)
    ignored_cache_count = sum(is_disposable_cache(path) for path in ignored)
    relevant_ignored = [
        path for path in ignored if not is_disposable_cache(path)
    ]
    paths = {
        worktree / path
        for path in [*untracked, *relevant_ignored]
        if (worktree / path).is_file() or (worktree / path).is_symlink()
    }
    return paths, len(ignored), ignored_cache_count


def tree_byte_size(path: Path) -> int:
    result = run("du", "-sb", str(path), check=False)
    if result.returncode == 0:
        output = result.stdout.decode("utf-8", "replace")
        return int(output.split(None, 1)[0])
    if path.is_symlink():
        return len(os.readlink(path).encode("utf-8", "surrogateescape"))
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in iter_regular_paths(path):
        if item.is_symlink():
            total += len(os.readlink(item).encode("utf-8", "surrogateescape"))
        else:
            total += item.stat().st_size
    return total


def write_tsv(path: Path, header: Iterable[str], rows: Iterable[Iterable[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\t".join(header) + "\n")
        for row in rows:
            values = [str(value) for value in row]
            if any("\t" in value or "\n" in value for value in values):
                raise ValueError(f"TSV value contains a tab or newline: {values!r}")
            handle.write("\t".join(values) + "\n")


def capture_processes(hy_tmp: Path) -> list[dict[str, object]]:
    rows = []
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return rows
    for proc in proc_root.iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            cwd = os.readlink(proc / "cwd")
            raw = (proc / "cmdline").read_bytes().split(b"\0")
            executable = os.path.basename(os.fsdecode(raw[0])) if raw and raw[0] else ""
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if cwd == str(hy_tmp) or cwd.startswith(str(hy_tmp) + os.sep):
            rows.append({
                "pid": int(proc.name),
                "cwd": cwd,
                "executable": executable,
            })
    return sorted(rows, key=lambda row: int(row["pid"]))


def capture_hf_sources(hf_cache: Path) -> dict[str, object]:
    sources = []
    hub = hf_cache / "hub"
    if hub.is_dir():
        for repository in sorted(hub.iterdir()):
            if not repository.is_dir() or repository.name == ".locks":
                continue
            refs = {}
            refs_dir = repository / "refs"
            if refs_dir.is_dir():
                for ref in sorted(refs_dir.rglob("*")):
                    if ref.is_file():
                        refs[str(ref.relative_to(refs_dir))] = (
                            ref.read_text(encoding="utf-8").strip()
                        )
            sources.append({
                "cache_key": repository.name,
                "refs": refs,
            })
    return {
        "schema_version": "rfr_hf_cache_sources_v1",
        "cache_root": str(hf_cache),
        "repositories": sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hy-tmp", type=Path, default=Path("/hy-tmp"))
    parser.add_argument("--staging", type=Path, required=True)
    args = parser.parse_args()

    hy_tmp = args.hy_tmp.resolve()
    staging = args.staging.resolve()
    staging.mkdir(parents=True, exist_ok=False)
    worktrees = discover_worktrees(hy_tmp)
    worktree_set = set(worktrees)
    hf_cache = hy_tmp / "hf_cache"

    payload: dict[Path, str] = {}
    ignored_rows = []
    worktree_rows = []
    ref_rows = []
    for worktree in worktrees:
        head = run("git", "rev-parse", "HEAD", cwd=worktree).stdout.decode().strip()
        branch = run(
            "git", "branch", "--show-current", cwd=worktree,
        ).stdout.decode().strip() or "(detached)"
        status = decode_paths(run(
            "git", "status", "--porcelain=v1", "-z", "--untracked-files=all",
            cwd=worktree,
        ).stdout)
        tracked_tree = run(
            "git", "ls-tree", "-r", "-z", "--full-tree", "HEAD",
            cwd=worktree,
        ).stdout
        extras, ignored_count, ignored_cache_count = git_extra_paths(worktree)
        for path in extras:
            payload[path] = "git_worktree_extra"
        ignored_rows.append((
            str(worktree),
            ignored_count,
            ignored_cache_count,
            ignored_count - ignored_cache_count,
        ))
        worktree_rows.append((
            str(worktree),
            str(worktree.relative_to(hy_tmp.parent)),
            head,
            branch,
            len(status),
            len(decode_paths(run("git", "ls-files", "-z", cwd=worktree).stdout)),
            sha256_bytes(tracked_tree),
            len(extras),
        ))
        refs = run(
            "git", "for-each-ref",
            "--format=%(refname)\t%(objectname)",
            "refs/heads", "refs/tags",
            cwd=worktree,
        ).stdout.decode("utf-8", "replace").splitlines()
        for ref in refs:
            if not ref:
                continue
            ref_name, object_name = ref.split("\t", 1)
            ref_rows.append((str(worktree), ref_name, object_name))

    external_roots = []
    for path in sorted(hy_tmp.iterdir()):
        if path in worktree_set or path == hf_cache or path == staging:
            continue
        external_roots.append(path)
        for item in iter_regular_paths(path):
            payload[item] = "external_artifact"

    manifest_rows = []
    payload_bytes = 0
    for path, category in sorted(payload.items(), key=lambda item: str(item[0])):
        digest, size, kind = hash_path(path)
        archive_path = str(path.relative_to(hy_tmp.parent))
        manifest_rows.append((
            digest,
            size,
            kind,
            str(path),
            archive_path,
            category,
        ))
        payload_bytes += size

    write_tsv(
        staging / "payload_manifest.tsv",
        ("sha256", "bytes", "type", "original_path", "archive_path", "category"),
        manifest_rows,
    )
    with (staging / "payload_paths.null").open("wb") as handle:
        for row in manifest_rows:
            handle.write(os.fsencode(row[4]) + b"\0")
    with (staging / "payload_paths.txt").open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row[4], ensure_ascii=False) + "\n")

    write_tsv(
        staging / "git_worktrees.tsv",
        (
            "original_path",
            "archive_path",
            "head",
            "branch",
            "status_rows",
            "tracked_files",
            "tracked_tree_sha256",
            "payload_extra_files",
        ),
        worktree_rows,
    )
    write_tsv(
        staging / "git_refs.tsv",
        ("worktree", "ref", "object"),
        sorted(set(ref_rows)),
    )
    write_tsv(
        staging / "ignored_cache_summary.tsv",
        (
            "worktree",
            "ignored_files",
            "disposable_cache_files_excluded",
            "relevant_ignored_files_in_payload",
        ),
        ignored_rows,
    )

    top_level_rows = []
    for path in sorted(hy_tmp.iterdir()):
        if path == staging:
            category = "backup_staging_excluded"
        elif path == hf_cache:
            category = "rebuildable_hf_cache_manifest_only"
        elif path in worktree_set:
            category = "git_commit_plus_server_extras"
        else:
            category = "full_payload"
        top_level_rows.append((
            str(path),
            tree_byte_size(path),
            category,
            os.readlink(path) if path.is_symlink() else "",
        ))
    write_tsv(
        staging / "hy_tmp_top_level_inventory.tsv",
        ("path", "bytes", "backup_category", "symlink_target"),
        top_level_rows,
    )

    hf_rows = []
    hf_bytes = 0
    if hf_cache.exists():
        for path in sorted(iter_regular_paths(hf_cache), key=str):
            digest, size, kind = hash_path(path)
            hf_rows.append((
                digest,
                size,
                kind,
                str(path),
                str(path.relative_to(hy_tmp.parent)),
            ))
            hf_bytes += size
    write_tsv(
        staging / "hf_cache_manifest.tsv",
        ("sha256", "bytes", "type", "original_path", "archive_path"),
        hf_rows,
    )
    with (staging / "hf_cache_sources.json").open("w", encoding="utf-8") as handle:
        json.dump(capture_hf_sources(hf_cache), handle, indent=2, sort_keys=True)
        handle.write("\n")

    processes = capture_processes(hy_tmp)
    with (staging / "project_processes.json").open("w", encoding="utf-8") as handle:
        json.dump(processes, handle, indent=2, sort_keys=True)
        handle.write("\n")

    system_lines = [
        run("uname", "-a").stdout.decode("utf-8", "replace").strip(),
        run("git", "--version").stdout.decode("utf-8", "replace").strip(),
        run(sys.executable, "--version").stdout.decode(
            "utf-8", "replace",
        ).strip(),
        run("df", "-h", str(hy_tmp)).stdout.decode("utf-8", "replace").strip(),
    ]
    with (staging / "system_snapshot.txt").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(system_lines) + "\n")
    try:
        nvidia = run(
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,driver_version",
            "--format=csv,noheader",
            check=False,
        ).stdout
    except FileNotFoundError:
        nvidia = b"nvidia-smi unavailable\n"
    (staging / "gpu_inventory.txt").write_bytes(nvidia)

    summary = {
        "schema_version": "rfr_full_server_recovery_v1",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "hy_tmp": str(hy_tmp),
        "payload_file_count": len(manifest_rows),
        "payload_uncompressed_file_bytes": payload_bytes,
        "external_root_count": len(external_roots),
        "git_worktree_count": len(worktrees),
        "hf_cache_file_count": len(hf_rows),
        "hf_cache_file_bytes_excluded_from_git_payload": hf_bytes,
        "project_processes_at_capture": processes,
        "external_roots": [str(path) for path in external_roots],
    }
    with (staging / "capture_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
