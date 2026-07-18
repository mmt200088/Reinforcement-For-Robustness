#!/usr/bin/env bash
set -euo pipefail

backup_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "$backup_root" rev-parse --show-toplevel)"

usage() {
  cat <<'EOF'
Usage:
  restore_snapshot.sh verify-all
  restore_snapshot.sh verify SNAPSHOT
  restore_snapshot.sh restore SNAPSHOT DESTINATION
EOF
}

snapshot_dir() {
  local name="$1"
  local dir="$backup_root/$name"
  test -d "$dir" || {
    echo "Unknown snapshot: $name" >&2
    exit 2
  }
  printf '%s\n' "$dir"
}

hash_path() {
  local kind="$1"
  local path="$2"
  if [[ "$kind" == "symlink" ]]; then
    readlink "$path" | tr -d '\n' | sha256sum | awk '{print $1}'
  else
    sha256sum "$path" | awk '{print $1}'
  fi
}

verify_snapshot() {
  local name="$1"
  local dir
  dir="$(snapshot_dir "$name")"
  local base
  base="$(tr -d '\r\n' < "$dir/base_commit.txt")"

  git -C "$repo_root" cat-file -e "$base^{commit}"
  (
    cd "$dir"
    sha256sum -c metadata_files.sha256 >/dev/null
    sha256sum -c archive_parts.sha256 >/dev/null
  )

  local expected_archive actual_archive expected_count archive_count
  expected_archive="$(awk '{print $1}' "$dir/extra_archive.sha256")"
  actual_archive="$(cat "$dir"/extra_files.tar.gz.part-* | sha256sum | awk '{print $1}')"
  [[ "$actual_archive" == "$expected_archive" ]]

  expected_count="$(awk -F '\t' 'NR > 1 {n++} END {print n+0}' "$dir/extra_files.tsv")"
  archive_count="$(cat "$dir"/extra_files.tar.gz.part-* | tar -tzf - | wc -l | tr -d ' ')"
  [[ "$archive_count" == "$expected_count" ]]

  echo "VERIFY_OK snapshot=$name base=$base extra_files=$expected_count archive_sha256=$actual_archive"
}

verify_restored_files() {
  local name="$1"
  local destination="$2"
  local dir
  dir="$(snapshot_dir "$name")"

  while IFS=$'\t' read -r expected bytes kind rel description; do
    [[ "$expected" != "sha256" ]] || continue
    local actual
    actual="$(hash_path "$kind" "$destination/$rel")"
    [[ "$actual" == "$expected" ]] || {
      echo "Hash mismatch in extra file: $rel" >&2
      exit 1
    }
  done < "$dir/extra_files.tsv"

  while IFS=$'\t' read -r expected bytes state rel description; do
    [[ "$expected" != "sha256" ]] || continue
    if [[ "$state" == "deleted" ]]; then
      [[ ! -e "$destination/$rel" ]]
      continue
    fi
    local kind="file"
    [[ "$state" != "symlink" ]] || kind="symlink"
    local actual
    actual="$(hash_path "$kind" "$destination/$rel")"
    [[ "$actual" == "$expected" ]] || {
      echo "Hash mismatch in modified tracked file: $rel" >&2
      exit 1
    }
  done < "$dir/modified_tracked_files.tsv"
}

restore_snapshot() {
  local name="$1"
  local destination="$2"
  local dir
  dir="$(snapshot_dir "$name")"
  [[ ! -e "$destination" ]] || {
    echo "Destination already exists: $destination" >&2
    exit 2
  }

  verify_snapshot "$name"
  local base
  base="$(tr -d '\r\n' < "$dir/base_commit.txt")"
  git -C "$repo_root" worktree add --detach "$destination" "$base"
  git -C "$destination" apply --binary "$dir/tracked_changes.patch"
  cat "$dir"/extra_files.tar.gz.part-* | tar -xzf - -C "$destination"
  verify_restored_files "$name" "$destination"
  echo "RESTORE_OK snapshot=$name destination=$destination"
}

command="${1:-}"
case "$command" in
  verify-all)
    for dir in "$backup_root"/stage2_k_convergence_*; do
      [[ -d "$dir" ]] || continue
      verify_snapshot "$(basename "$dir")"
    done
    ;;
  verify)
    [[ $# -eq 2 ]] || { usage >&2; exit 2; }
    verify_snapshot "$2"
    ;;
  restore)
    [[ $# -eq 3 ]] || { usage >&2; exit 2; }
    restore_snapshot "$2" "$3"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
