#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 /path/to/source/checkout /empty/recovery/root" >&2
  exit 2
fi

bundle_dir="$(cd "$(dirname "$0")" && pwd -P)"
source_repo="$(cd "$1" && pwd -P)"
destination="$2"
hy_tmp="$destination/hy-tmp"
canonical_name="Reinforcement-For-Robustness"
canonical_repo="$hy_tmp/$canonical_name"

if [[ ! -d "$source_repo/.git" && ! -f "$source_repo/.git" ]]; then
  echo "source checkout is not a Git worktree: $source_repo" >&2
  exit 2
fi
mkdir -p "$destination"
if find "$destination" -mindepth 1 -print -quit | grep -q .; then
  echo "refusing to restore into a non-empty destination: $destination" >&2
  exit 2
fi
mkdir -p "$hy_tmp"

hash_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

verify_part_hashes() {
  while IFS=$'\t' read -r expected relative_path; do
    [[ -n "$expected" ]] || continue
    actual="$(hash_file "$bundle_dir/$relative_path")"
    if [[ "$actual" != "$expected" ]]; then
      echo "archive part hash mismatch: $relative_path" >&2
      echo "expected $expected" >&2
      echo "actual   $actual" >&2
      exit 1
    fi
  done < "$bundle_dir/PART_SHA256SUMS.tsv"
}

verify_metadata_hashes() {
  while IFS=$'\t' read -r expected relative_path; do
    [[ -n "$expected" ]] || continue
    actual="$(hash_file "$bundle_dir/$relative_path")"
    if [[ "$actual" != "$expected" ]]; then
      echo "metadata hash mismatch: $relative_path" >&2
      echo "expected $expected" >&2
      echo "actual   $actual" >&2
      exit 1
    fi
  done < "$bundle_dir/METADATA_SHA256SUMS.tsv"
}

verify_metadata_hashes
verify_part_hashes

canonical_head="$(
  awk -F $'\t' -v path="hy-tmp/$canonical_name" \
    'NR > 1 && $2 == path {print $3}' "$bundle_dir/git_worktrees.tsv"
)"
if [[ -z "$canonical_head" ]]; then
  echo "canonical server checkout is missing from git_worktrees.tsv" >&2
  exit 1
fi

git clone --no-hardlinks "$source_repo" "$canonical_repo"
git -C "$canonical_repo" checkout --detach "$canonical_head"

while IFS=$'\t' read -r original_path archive_path head branch status_count; do
  [[ "$original_path" == "original_path" ]] && continue
  [[ "$archive_path" == "hy-tmp/$canonical_name" ]] && continue
  target="$destination/$archive_path"
  if ! git -C "$canonical_repo" cat-file -e "$head^{commit}"; then
    echo "source checkout does not contain required commit $head for $original_path" >&2
    echo "fetch all archive branches from origin, then retry" >&2
    exit 1
  fi
  git -C "$canonical_repo" worktree add --detach "$target" "$head"
done < "$bundle_dir/git_worktrees.tsv"

cat "$bundle_dir"/archives/server_payload.tar.gz.part* \
  | gzip -dc \
  | tar -xpf - -C "$destination"

python3 - "$destination" "$bundle_dir/payload_manifest.tsv" <<'PY'
import hashlib
import os
import pathlib
import sys

destination = pathlib.Path(sys.argv[1])
manifest_path = pathlib.Path(sys.argv[2])
checked = 0

with manifest_path.open("r", encoding="utf-8") as handle:
    header = handle.readline().rstrip("\n").split("\t")
    columns = {name: index for index, name in enumerate(header)}
    required = {"sha256", "bytes", "type", "archive_path"}
    if not required.issubset(columns):
        raise SystemExit(f"payload manifest missing columns: {sorted(required - columns)}")
    for line in handle:
        row = line.rstrip("\n").split("\t")
        archive_path = row[columns["archive_path"]]
        path = destination / archive_path
        expected_hash = row[columns["sha256"]]
        expected_size = int(row[columns["bytes"]])
        kind = row[columns["type"]]
        if kind == "symlink":
            if not path.is_symlink():
                raise SystemExit(f"restored symlink missing: {archive_path}")
            payload = os.readlink(path).encode("utf-8", "surrogateescape")
        elif kind == "file":
            if not path.is_file():
                raise SystemExit(f"restored file missing: {archive_path}")
            payload = None
        else:
            raise SystemExit(f"unsupported manifest type {kind!r}: {archive_path}")
        observed_size = len(payload) if payload is not None else path.stat().st_size
        if observed_size != expected_size:
            raise SystemExit(
                f"restored size mismatch: {archive_path}: "
                f"{observed_size} != {expected_size}"
            )
        digest = hashlib.sha256()
        if payload is not None:
            digest.update(payload)
        else:
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
        if digest.hexdigest() != expected_hash:
            raise SystemExit(f"restored SHA-256 mismatch: {archive_path}")
        checked += 1

print(f"restore and payload verification complete: {checked} files")
PY

echo "recovery root: $destination"
echo "Hugging Face cache payloads are intentionally reconstructed separately;"
echo "see hf_cache_manifest.tsv and README.md."
