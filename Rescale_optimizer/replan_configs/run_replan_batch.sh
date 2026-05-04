#!/usr/bin/env bash
# Batch-run replan_what_if.py over one or more profile directories.
#
# Layout (relative to repo root):
#   replan_configs/<profile>/replan_actions_<config_stem>.json   — input actions
#   configs/<profile>/<config_stem>.json                         — graph config
#   configs/<profile>/static_skeletons_<profile>.json            — baseline archive
#   replan_configs/<profile>/replan_<config_stem>.json           — output
#
# Usage:
#   cd /path/to/Rescale_optimizer
#
#   # No args -> auto-discover ALL profiles under replan_configs/
#   #            (any subdir that has at least one replan_actions_*.json)
#   bash replan_configs/run_replan_batch.sh
#
#   # Specific profiles
#   bash replan_configs/run_replan_batch.sh wnli
#   bash replan_configs/run_replan_batch.sh wnli mrpc
#   ./replan_configs/run_replan_batch.sh mrpc

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/scripts/replan_what_if.py"

if [[ ! -f "${PY}" ]]; then
  echo "error: ${PY} not found" >&2
  exit 1
fi

# Resolve target profiles
declare -a PROFILES
if [[ $# -gt 0 ]]; then
  PROFILES=("$@")
else
  shopt -s nullglob
  for d in "${ROOT}/replan_configs/"*/; do
    p="$(basename "${d}")"
    if compgen -G "${d}replan_actions_*.json" > /dev/null; then
      PROFILES+=("${p}")
    fi
  done
  shopt -u nullglob
  if [[ ${#PROFILES[@]} -eq 0 ]]; then
    echo "no profiles found under ${ROOT}/replan_configs/" >&2
    exit 1
  fi
fi

echo "[batch] profiles: ${PROFILES[*]}"

total_ok=0
total_fail=0
overall_fail=0

for PROFILE in "${PROFILES[@]}"; do
  BASELINE="${ROOT}/configs/${PROFILE}/static_skeletons_${PROFILE}.json"
  ACTIONS_DIR="${ROOT}/replan_configs/${PROFILE}"
  OUT_DIR="${ACTIONS_DIR}"

  echo
  echo "================================================================"
  echo "[batch] profile=${PROFILE}"
  echo "================================================================"

  if [[ ! -f "${BASELINE}" ]]; then
    echo "error: baseline not found: ${BASELINE}" >&2
    echo "  run: python3 scripts/batch_run_configs.py --configs-dir configs/${PROFILE} --out ${BASELINE#${ROOT}/}" >&2
    overall_fail=1
    continue
  fi

  shopt -s nullglob
  mapfile -t FILES < <(find "${ACTIONS_DIR}" -maxdepth 1 -name 'replan_actions_*.json' -print | sort)
  shopt -u nullglob

  if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "no replan_actions_*.json under ${ACTIONS_DIR}" >&2
    overall_fail=1
    continue
  fi

  ok=0
  fail=0
  for actions in "${FILES[@]}"; do
    base="$(basename "${actions}" .json)"
    stem="${base#replan_actions_}"
    cfg="${ROOT}/configs/${PROFILE}/${stem}.json"
    out="${OUT_DIR}/replan_${stem}.json"

    if [[ ! -f "${cfg}" ]]; then
      echo "[SKIP] ${PROFILE}/${stem} — missing graph config: ${cfg}"
      ((fail++)) || true
      continue
    fi

    echo "[RUN ] ${PROFILE}/${stem}"
    if python3 "${PY}" \
        --config "${cfg}" \
        --baseline-from "${BASELINE}" \
        --actions-file "${actions}" \
        --out "${out}"; then
      fc="$(python3 -c "import json; print(json.load(open('${out}', encoding='utf-8'))['fusion_count'])" 2>/dev/null || echo "?")"
      echo "[ OK ] ${out}  fusion_count=${fc}"
      ((ok++)) || true
    else
      ec=$?
      echo "[FAIL] ${PROFILE}/${stem} (exit ${ec})" >&2
      ((fail++)) || true
    fi
    echo
  done

  echo "[batch] profile=${PROFILE} done: ok=${ok} fail=${fail}"
  total_ok=$((total_ok + ok))
  total_fail=$((total_fail + fail))
  if [[ "${fail}" -gt 0 ]]; then overall_fail=1; fi
done

echo
echo "================================================================"
echo "[batch] all done: total_ok=${total_ok}  total_fail=${total_fail}"
[[ "${overall_fail}" -eq 0 ]]
