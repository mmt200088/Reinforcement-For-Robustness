#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="${1:?source root is required}"
OUTPUT_ROOT="${2:?output root is required}"
MATCH_LIB="/hy-tmp/nvidia_userspace_580.173.02/root/usr/lib/x86_64-linux-gnu"

export LD_LIBRARY_PATH="$MATCH_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="/var/tmp/root-home/miniconda3/envs/llm_ist/bin:$PATH"
export HF_HOME="/var/tmp/root-home/.cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export GLUE_LOCAL_DATASET_DIR="/hy-tmp/rfr_cleanup_glue_local_4a16e810"

mkdir -p "$OUTPUT_ROOT/logs"
cd "$SOURCE_ROOT"

run_case() {
  local label="$1"
  local timeout_seconds="$2"
  local needs_lock="$3"
  shift 3

  local dry_output run_root command_text log_path rc lock_path lock_fd
  log_path="$OUTPUT_ROOT/logs/$label.log"
  if [[ -f "$log_path" ]] && grep -q '^accepted_status=' "$log_path"; then
    printf '%s\n' "$label=reused"
    return 0
  fi
  dry_output="$(bash llama_7B_LayerImportance.sh "$@" --dry-run)"
  run_root="$(printf '%s\n' "$dry_output" | sed -n 's/^Run directory: //p')"
  command_text="$(printf '%s\n' "$dry_output" | sed -n 's/^Command: //p')"
  if [[ -z "$run_root" || -z "$command_text" ]]; then
    printf '%s\n' "unable to parse launcher output for $label" >&2
    return 2
  fi

  {
    printf 'label=%s\n' "$label"
    printf 'run_root=%s\n' "$run_root"
    printf 'source_commit=%s\n' "$(git rev-parse HEAD)"
    printf 'source_tree=%s\n' "$(git rev-parse HEAD^{tree})"
    printf 'command=%s\n' "$command_text"
    printf 'started_at=%s\n' "$(date -Iseconds)"
  } > "$log_path"

  lock_fd=""
  if [[ "$needs_lock" == "true" ]]; then
    lock_path="$(dirname "$run_root")/.$(basename "$run_root").stage2_rl.lock"
    mkdir -p "$(dirname "$lock_path")"
    exec {lock_fd}>>"$lock_path"
    flock -n "$lock_fd"
    export BLB_STAGE2_RUN_LOCK_FD="$lock_fd"
    export BLB_STAGE2_RUN_LOCK_PATH="$lock_path"
  else
    unset BLB_STAGE2_RUN_LOCK_FD BLB_STAGE2_RUN_LOCK_PATH || true
  fi

  set +e
  timeout --signal=INT --kill-after=60 "$timeout_seconds" \
    bash -c "$command_text" 2>&1 | tee -a "$log_path"
  rc=${PIPESTATUS[0]}
  set -e

  printf 'finished_at=%s\n' "$(date -Iseconds)" >> "$log_path"
  printf 'exit_code=%s\n' "$rc" >> "$log_path"
  if [[ -n "$lock_fd" ]]; then
    flock -u "$lock_fd"
    eval "exec ${lock_fd}>&-"
    unset BLB_STAGE2_RUN_LOCK_FD BLB_STAGE2_RUN_LOCK_PATH
  fi
  if [[ "$rc" -ne 0 ]]; then
    if [[ "$label" == stage2-* ]] \
        && grep -q "DegenerateBaselineVariance" "$log_path"; then
      printf 'accepted_status=scientific_fail_closed\n' >> "$log_path"
      printf '%s\n' "$label=scientific_fail_closed"
      return 0
    fi
    printf '%s\n' "$label failed with exit code $rc" >&2
    return "$rc"
  fi
  printf 'accepted_status=passed\n' >> "$log_path"
  printf '%s\n' "$label=pass"
}

profiles=(
  bert-base-mrpc
  bert-base-rte
  bert-base-sst2
  bert-large-mrpc
  bert-large-rte
  bert-large-sst2
)

for profile in "${profiles[@]}"; do
  run_case "stage1-$profile" 1200 false \
    run rl --preset "$profile-stage1-rl" --fresh \
    --persistent-root "$OUTPUT_ROOT/persistent" \
    --stage1-search-episodes 1 \
    --ppo-update-interval 1 \
    --stage1-rl-devices 0 \
    --elastic-gpu-mode off
done

for profile in "${profiles[@]}"; do
  run_case "stage2-$profile" 1800 true \
    run rl --preset "$profile-stage2-rl" --fresh \
    --persistent-root "$OUTPUT_ROOT/persistent" \
    --run-tag cleanup_gpu_gate \
    --stage2-search-episodes 1 \
    --ppo-update-interval 1 \
    --stage2-rollout-size 1 \
    --stage2-save-interval 1 \
    --stage2-eval-interval 1 \
    --blb-v3-reward-devices 0 \
    --elastic-gpu-mode off
done

for algorithm in bo_rf greedy coinn_ga; do
  run_case "comparator-$algorithm" 1800 true \
    run "$algorithm" --fresh --comparator-smoke \
    --persistent-root "$OUTPUT_ROOT/persistent" \
    --run-tag cleanup_gpu_gate
done

python - "$OUTPUT_ROOT" <<'PY'
from pathlib import Path
import hashlib
import json
import sys

root = Path(sys.argv[1])
logs = sorted((root / "logs").glob("*.log"))
rows = []
for path in logs:
    text = path.read_text(encoding="utf-8", errors="replace")
    accepted_status = text.rsplit("accepted_status=", 1)[1].splitlines()[0]
    rows.append({
        "case": path.stem,
        "exit_code": int(text.rsplit("exit_code=", 1)[1].splitlines()[0]),
        "accepted_status": accepted_status,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    })
payload = {
    "schema": "production_cleanup_real_profile_smoke_v1",
    "case_count": len(rows),
    "all_accepted": len(rows) == 15 and all(
        row["accepted_status"] in {"passed", "scientific_fail_closed"}
        for row in rows
    ),
    "cases": rows,
}
(root / "summary.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
if not payload["all_accepted"]:
    raise SystemExit(1)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
