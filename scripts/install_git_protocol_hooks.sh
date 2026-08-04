#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
hook_path="${repo_root}/.githooks/pre-push"
guard_path="${repo_root}/scripts/repo_sync_guard.py"
hooks_dir="${repo_root}/.githooks"

branch="$(git symbolic-ref --quiet --short HEAD || true)"
if [[ "${branch}" != "jk_standard_rl" ]]; then
  echo "install hooks from the canonical jk_standard_rl checkout, found: ${branch:-detached}" >&2
  exit 2
fi

if [[ ! -f "${hook_path}" || ! -x "${hook_path}" ]]; then
  echo "git protocol hook is missing or not executable: ${hook_path}" >&2
  exit 2
fi
if [[ ! -f "${guard_path}" ]]; then
  echo "git protocol guard is missing: ${guard_path}" >&2
  exit 2
fi

current="$(git config --local --get core.hooksPath || true)"
if [[ -n "${current}" && "${current}" != "${hooks_dir}" ]]; then
  echo "refusing to overwrite existing core.hooksPath=${current}" >&2
  exit 2
fi

git config --local core.hooksPath "${hooks_dir}"
effective="$(git config --local --get core.hooksPath)"
if [[ "${effective}" != "${hooks_dir}" ]]; then
  echo "failed to configure repository hook path" >&2
  exit 2
fi
printf '%s\n' "${effective}"
