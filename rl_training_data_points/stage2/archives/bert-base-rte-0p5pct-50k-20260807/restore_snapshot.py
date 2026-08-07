#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import stat
from pathlib import Path


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def inspect_raw(path: Path, is_text: bool):
    h = hashlib.sha256()
    total = 0
    lines = 0 if is_text else None
    last = b''
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
            total += len(chunk)
            if is_text:
                lines += chunk.count(b'\n')
                last = chunk[-1:]
    return h.hexdigest(), total, lines, (last == b'\n') if is_text and total else None


def main() -> int:
    parser = argparse.ArgumentParser(description='Restore and verify an RFR training snapshot')
    parser.add_argument('--archive', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    archive = args.archive.resolve()
    destination = args.destination.resolve()
    manifest = json.loads((archive / 'snapshot_manifest.json').read_text())
    if destination.exists() and any(destination.iterdir()):
        raise SystemExit(f'destination is not empty: {destination}')
    destination.mkdir(parents=True, exist_ok=True)
    restored = []
    for entry in manifest['entries']:
        root = entry['root']
        rel = Path(entry['relative_path'])
        if rel.is_absolute() or '..' in rel.parts or '/' in root or root in {'', '.', '..'}:
            raise SystemExit(f'unsafe manifest path: {root}/{rel}')
        source = archive / entry['archive_path']
        if not source.is_file():
            raise SystemExit(f'missing archive payload: {source}')
        if source.stat().st_size != entry['archive_bytes'] or sha256_path(source) != entry['archive_sha256']:
            raise SystemExit(f'archive payload mismatch: {source}')
        target = destination / root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(target.name + '.partial')
        if entry['storage'] == 'gzip':
            with gzip.open(source, 'rb') as src, temp.open('wb') as dst:
                shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        elif entry['storage'] == 'copy':
            shutil.copyfile(source, temp)
        else:
            raise SystemExit(f"unknown storage: {entry['storage']}")
        is_text = entry['raw_lines'] is not None
        raw_hash, raw_bytes, raw_lines, raw_newline = inspect_raw(temp, is_text)
        expected = (entry['raw_sha256'], entry['raw_bytes'], entry['raw_lines'], entry['raw_ends_with_newline'])
        actual = (raw_hash, raw_bytes, raw_lines, raw_newline)
        if actual != expected:
            raise SystemExit(f'raw restore mismatch: {root}/{rel}: {actual} != {expected}')
        os.chmod(temp, entry['mode'])
        os.utime(temp, ns=(entry['mtime_ns'], entry['mtime_ns']))
        temp.replace(target)
        restored.append({'root': root, 'relative_path': rel.as_posix(), 'raw_bytes': raw_bytes, 'raw_sha256': raw_hash})
    result = {
        'schema': 'rfr_snapshot_restore_verification_v1',
        'status': 'RESTORE_OK',
        'run_id': manifest['run_id'],
        'restored_file_count': len(restored),
        'restored_total_bytes': sum(x['raw_bytes'] for x in restored),
        'manifest_source_file_count': manifest['source_file_count'],
        'source_commit': manifest['source_commit'],
        'source_tree': manifest['source_tree'],
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
