#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/hy-tmp/comparator_stage1_queue_20260812_v3
REPO=/hy-tmp/rfr_three_comparator_opt_task_20260811_v2
PY=/var/tmp/root-home/miniconda3/envs/llm_ist/bin/python
EXPECTED_COMMIT=73ad927a53e411035f7d7756538a0c8838944b22
EXPECTED_TREE=63374369c330499fcb8e670787055126e624df8c
CURRENT_ALGORITHM=""
CURRENT_RUN_DIR=""
CURRENT_PID=""

export HOME=/var/tmp/root-home
export PATH=/var/tmp/root-home/miniconda3/envs/llm_ist/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export HF_HOME=/hy-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export GLUE_LOCAL_DATASET_DIR=/hy-tmp/glue_saved_from_cache_20260805
export CUDA_VISIBLE_DEVICES=0
export BLB_NOISE_INSTALL_LOGS=0
export BLB_STRICT=0
export PYTHONUNBUFFERED=1
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE
unset BLB_STAGE2_PROBE_PROFILE_PATH RFR_GPU_AUDIT_STRICT

write_state() {
  local status="$1"
  local algorithm="$2"
  local detail="$3"
  local run_dir="${4:-}"
  local pid="${5:-}"
  local evaluations="${6:-}"
  "$PY" - "$ROOT/status.json" "$status" "$algorithm" "$detail" \
    "$run_dir" "$pid" "$evaluations" <<'PY'
import datetime
import json
import os
import sys

path, status, algorithm, detail, run_dir, pid, evaluations = sys.argv[1:]
payload = {
    "schema_version": "comparator_stage1_queue_status_v2",
    "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "status": status,
    "current_algorithm": algorithm or None,
    "detail": detail or None,
    "run_dir": run_dir or None,
    "process_pid": int(pid) if pid else None,
    "observed_evaluations": int(evaluations) if evaluations else None,
    "order": ["greedy", "bo_rf", "coinn_ga"],
    "source_commit": "73ad927a53e411035f7d7756538a0c8838944b22",
    "source_tree": "63374369c330499fcb8e670787055126e624df8c",
    "stage2_enabled": False,
    "final_eval_enabled": False,
}
temporary = path + ".tmp"
with open(temporary, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, path)
PY
}

check_source() {
  local head tree remote status
  head="$(git -C "$REPO" rev-parse HEAD)"
  tree="$(git -C "$REPO" rev-parse 'HEAD^{tree}')"
  status="$(git -C "$REPO" status --porcelain --untracked-files=no)"
  remote="$(git -C "$REPO" ls-remote origin refs/heads/jk_standard_rl | awk '{print $1}')"
  [ "$head" = "$EXPECTED_COMMIT" ] || {
    echo "Server checkout commit drift: $head" >&2
    return 71
  }
  [ "$tree" = "$EXPECTED_TREE" ] || {
    echo "Server checkout tree drift: $tree" >&2
    return 72
  }
  [ -z "$status" ] || {
    echo "Server checkout tracked state is dirty" >&2
    printf '%s\n' "$status" >&2
    return 73
  }
  [ "$remote" = "$EXPECTED_COMMIT" ] || {
    echo "Remote canonical advanced before a queue stage: $remote" >&2
    return 74
  }
}

check_gpu_idle() {
  local apps=""
  local attempt
  for attempt in $(seq 1 12); do
    apps="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader,nounits 2>/dev/null || true)"
    if [ -z "$(printf '%s' "$apps" | tr -d '[:space:]')" ]; then
      return 0
    fi
    sleep 5
  done
  echo "GPU has an existing compute process; refusing to overlap:" >&2
  printf '%s\n' "$apps" >&2
  return 75
}

process_is_live() {
  local pid="$1"
  [ -r "/proc/$pid/stat" ] && [ "$(awk '{print $3}' "/proc/$pid/stat")" != Z ]
}

observation_count() {
  local algorithm="$1"
  local run_dir="$2"
  local path="$run_dir/stage1_comparator/$algorithm/observations.jsonl"
  if [ -f "$path" ]; then
    wc -l < "$path" | tr -d '[:space:]'
  else
    printf '0'
  fi
}

validate_stage1_result() {
  local algorithm="$1"
  local run_dir="$2"
  PYTHONPATH="$REPO" "$PY" - "$run_dir" "$algorithm" <<'PY'
import json
import os
from pathlib import Path
import sys

run_dir = Path(sys.argv[1])
algorithm = sys.argv[2]
from stage1_rl.search_runner import load_completed_search_result

output_dir = run_dir / "stage1_comparator" / algorithm
result = load_completed_search_result(output_dir)
manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
statuses = metadata.get("stage_status") or {}
if result.algorithm != algorithm:
    raise SystemExit(
        f"backend mismatch: expected {algorithm}, got {result.algorithm}"
    )
if manifest.get("status") != "complete":
    raise SystemExit(f"Stage-1 manifest is not complete: {manifest!r}")
if manifest.get("stage1_bound_into_stage2") is not False:
    raise SystemExit("Stage-1-only result claims a Stage-2 binding")
if manifest.get("stage2_backend") is not None:
    raise SystemExit("Stage-1-only result names a Stage-2 backend")
if statuses.get("stage1_search") not in {"completed", "completed_infeasible"}:
    raise SystemExit(f"Stage-1 status is not complete: {statuses!r}")
if statuses.get("stage2_search") != "skipped":
    raise SystemExit(f"Stage-2 was not skipped: {statuses!r}")
if statuses.get("final_eval") != "skipped":
    raise SystemExit(f"final-eval was not skipped: {statuses!r}")
if (run_dir / "two_stage_result.json").exists():
    raise SystemExit("unexpected two_stage_result.json in Stage-1-only run")
summary = {
    "schema_version": "comparator_stage1_formal_validation_v1",
    "algorithm": result.algorithm,
    "evaluation_count": result.evaluation_count,
    "unique_evaluation_count": result.unique_evaluation_count,
    "termination_reason": result.termination_reason,
    "best_feasible": result.best.feasible,
    "best_action": list(result.best.action),
    "best_gelu_degrees": list(result.best.gelu_degrees),
    "best_softmax_degrees": list(result.best.softmax_degrees),
    "stage1_bound_into_stage2": False,
    "stage2_backend": None,
    "stage_status": statuses,
    "source_commit": "73ad927a53e411035f7d7756538a0c8838944b22",
    "source_tree": "63374369c330499fcb8e670787055126e624df8c",
}
summary_path = run_dir / "stage1_validation_summary.json"
temporary = summary_path.with_suffix(".json.tmp")
temporary.write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(temporary, summary_path)
print(json.dumps(summary, sort_keys=True))
PY
}

run_one() {
  local algorithm="$1"
  local group run_dir pid launch_log validations evaluations elapsed
  local started_epoch last_status_epoch last_gpu_epoch now

  CURRENT_ALGORITHM="$algorithm"
  CURRENT_RUN_DIR=""
  CURRENT_PID=""
  check_source
  check_gpu_idle
  launch_log="$ROOT/logs/${algorithm}_launcher.log"
  write_state "launching" "$algorithm" "canonical launcher starting"
  printf '%s\t%s\tlaunching\n' "$(date -Iseconds)" "$algorithm" \
    >> "$ROOT/timeline.tsv"

  cd "$REPO"
  bash llama_7B_LayerImportance.sh run "$algorithm" \
    --preset mrpc-blb-stage2-rl \
    --dataset mrpc \
    --model-type bert-base \
    --comparator-stage1-only \
    --persistent-root "$ROOT/persistent" \
    --fresh \
    > "$launch_log" 2>&1

  group="$ROOT/persistent/$algorithm/bert-base/mrpc"
  run_dir="$(cat "$group/LATEST_RUN_DIR")"
  pid="$(cat "$group/LATEST_PID")"
  case "$run_dir" in
    "$group"/*) ;;
    *) echo "Unexpected run directory: $run_dir" >&2; return 76 ;;
  esac
  CURRENT_RUN_DIR="$run_dir"
  CURRENT_PID="$pid"
  printf '%s\t%s\tstarted\t%s\t%s\n' \
    "$(date -Iseconds)" "$algorithm" "$pid" "$run_dir" \
    >> "$ROOT/timeline.tsv"

  started_epoch="$(date +%s)"
  last_status_epoch=0
  last_gpu_epoch=0
  while process_is_live "$pid"; do
    now="$(date +%s)"
    if [ $((now - last_status_epoch)) -ge 60 ]; then
      evaluations="$(observation_count "$algorithm" "$run_dir")"
      elapsed=$((now - started_epoch))
      write_state "running" "$algorithm" \
        "formal Stage-1 search; elapsed_seconds=$elapsed" \
        "$run_dir" "$pid" "$evaluations"
      last_status_epoch="$now"
    fi
    if [ $((now - last_gpu_epoch)) -ge 30 ]; then
      nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used \
        --format=csv,noheader,nounits >> "$ROOT/gpu_samples.csv" 2>/dev/null || true
      last_gpu_epoch="$now"
    fi
    sleep 5
  done

  validations="$ROOT/logs/${algorithm}_validation.log"
  validate_stage1_result "$algorithm" "$run_dir" > "$validations" 2>&1
  evaluations="$(observation_count "$algorithm" "$run_dir")"
  elapsed=$(($(date +%s) - started_epoch))
  printf '%s\t%s\tvalidated\t%s\t%s\n' \
    "$(date -Iseconds)" "$algorithm" "$evaluations" "$elapsed" \
    >> "$ROOT/timeline.tsv"
  write_state "stage_completed" "$algorithm" \
    "formal Stage-1 result validated; elapsed_seconds=$elapsed" \
    "$run_dir" "" "$evaluations"
}

on_exit() {
  local rc=$?
  trap - EXIT
  if [ "$rc" -eq 0 ]; then
    write_state "completed" "" \
      "all three formal Stage-1 searches completed and validated"
  else
    write_state "failed" "$CURRENT_ALGORITHM" \
      "queue exit code $rc" "$CURRENT_RUN_DIR" "$CURRENT_PID"
  fi
  printf '%s\tqueue\texit\t%s\n' "$(date -Iseconds)" "$rc" \
    >> "$ROOT/timeline.tsv"
  exit "$rc"
}
trap on_exit EXIT

exec 9>"$ROOT/queue.lock"
if ! flock -n 9; then
  echo "Another queue process already owns $ROOT/queue.lock" >&2
  exit 77
fi

printf '%s\n' "$$" > "$ROOT/queue.pid"
printf 'timestamp\talgorithm\tevent\tvalue\textra\n' > "$ROOT/timeline.tsv"
printf '%s\n' \
  'timestamp, index, name, utilization.gpu [%], memory.used [MiB]' \
  > "$ROOT/gpu_samples.csv"
write_state "starting" "greedy" "queue initialized"

run_one greedy
run_one bo_rf
run_one coinn_ga
