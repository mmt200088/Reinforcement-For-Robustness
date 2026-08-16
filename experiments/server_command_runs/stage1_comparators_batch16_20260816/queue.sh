#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/hy-tmp/comparator_stage1_batch16_20260816
REPO=/hy-tmp/rfr_three_comparator_opt_task_20260811_v2
PY=/var/tmp/root-home/miniconda3/envs/llm_ist/bin/python
EXPECTED_COMMIT=9d833d90760b1bf85fca4c8650e8149f61119ad2
EXPECTED_TREE=918fb6e4f5e6ea6fa659a30045331f99dc48800e
CURRENT_ALGORITHM=""
CURRENT_RUN_DIR=""
CURRENT_PID=""

export HOME=/var/tmp/root-home
export PATH=/var/tmp/root-home/miniconda3/envs/llm_ist/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH=/hy-tmp/nvidia_userspace_580.173.02/nvidia_driver-linux-x86_64-580.173.02-archive/lib:/usr/lib/x86_64-linux-gnu
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
    "schema_version": "comparator_stage1_batch16_queue_v1",
    "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "status": status,
    "current_algorithm": algorithm or None,
    "detail": detail or None,
    "run_dir": run_dir or None,
    "process_pid": int(pid) if pid else None,
    "observed_evaluations": int(evaluations) if evaluations else None,
    "order": ["greedy", "bo_rf", "coinn_ga"],
    "batch_size": 16,
    "micro_batch_size": 16,
    "source_commit": "9d833d90760b1bf85fca4c8650e8149f61119ad2",
    "source_tree": "918fb6e4f5e6ea6fa659a30045331f99dc48800e",
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
  [ "$head" = "$EXPECTED_COMMIT" ] || return 71
  [ "$tree" = "$EXPECTED_TREE" ] || return 72
  [ -z "$status" ] || return 73
  [ "$remote" = "$EXPECTED_COMMIT" ] || return 74
}

check_gpu_idle() {
  local apps attempt
  for attempt in $(seq 1 12); do
    apps="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader,nounits 2>/dev/null || true)"
    [ -n "$(printf '%s' "$apps" | tr -d '[:space:]')" ] || return 0
    sleep 5
  done
  printf 'GPU busy:\n%s\n' "$apps" >&2
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
  local launch_log="$3"
  grep -q -- '--batch_size 16 --micro_batch_size 16' "$launch_log"
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
    raise SystemExit(f"backend mismatch: {result.algorithm}")
if manifest.get("status") != "complete":
    raise SystemExit("manifest is not complete")
if manifest.get("comparator_smoke") is not False:
    raise SystemExit("formal run is marked as smoke")
if manifest.get("stage1_bound_into_stage2") is not False:
    raise SystemExit("Stage-1-only result claims a Stage-2 binding")
if manifest.get("stage2_backend") is not None:
    raise SystemExit("Stage-1-only result names a Stage-2 backend")
if statuses.get("stage1_search") not in {"completed", "completed_infeasible"}:
    raise SystemExit(f"Stage-1 status is not complete: {statuses!r}")
if statuses.get("stage2_search") != "skipped" or statuses.get("final_eval") != "skipped":
    raise SystemExit(f"later stages were not skipped: {statuses!r}")
if (run_dir / "two_stage_result.json").exists():
    raise SystemExit("unexpected two-stage artifact")
config = result.config
if int(config.seed) != 42:
    raise SystemExit("search seed drift")
if algorithm == "greedy":
    if result.termination_reason != "verified_local_optimum":
        raise SystemExit(f"Greedy termination mismatch: {result.termination_reason}")
    if int(config.greedy_max_starts) != 3:
        raise SystemExit("Greedy did not use three starts")
elif algorithm == "bo_rf":
    if int(config.evaluation_cap) != 10_000:
        raise SystemExit("BO-RF evaluation cap drift")
    if int(config.bo_no_improvement_patience) != 1_000:
        raise SystemExit("BO-RF patience drift")
    if result.evaluation_count > 10_000:
        raise SystemExit("BO-RF exceeded evaluation cap")
    if result.termination_reason not in {"no_improvement_convergence", "evaluation_budget"}:
        raise SystemExit(f"BO-RF termination mismatch: {result.termination_reason}")
elif algorithm == "coinn_ga":
    if int(config.ga_population_size) != 64 or int(config.ga_update_generations) != 200:
        raise SystemExit("GA population/generation contract drift")
    if not config.ga_require_full_generations or config.ga_stop_on_no_improvement:
        raise SystemExit("GA early-stop contract drift")
    if int(config.evaluation_cap) != 11_464:
        raise SystemExit("GA evaluation cap drift")
    if result.evaluation_count != 11_464 or result.unique_evaluation_count != 11_464:
        raise SystemExit("GA did not complete the exact 200-generation budget")
    if result.termination_reason != "completed_generations":
        raise SystemExit(f"GA termination mismatch: {result.termination_reason}")
else:
    raise SystemExit(f"unsupported algorithm: {algorithm}")
if result.best.metadata.get("split") != "validation_full":
    raise SystemExit("best result did not use validation_full")
summary = {
    "schema_version": "comparator_stage1_batch16_validation_v1",
    "algorithm": result.algorithm,
    "batch_size": 16,
    "micro_batch_size": 16,
    "evaluation_count": result.evaluation_count,
    "unique_evaluation_count": result.unique_evaluation_count,
    "termination_reason": result.termination_reason,
    "best_feasible": result.best.feasible,
    "best_action": list(result.best.action),
    "best_gelu_degrees": list(result.best.gelu_degrees),
    "best_softmax_degrees": list(result.best.softmax_degrees),
    "best_loss": result.best.loss,
    "best_metrics": dict(zip(result.best.constraints.metric_names, result.best.metrics)),
    "best_cost": result.best.cost,
    "stage_status": statuses,
    "source_commit": "9d833d90760b1bf85fca4c8650e8149f61119ad2",
    "source_tree": "918fb6e4f5e6ea6fa659a30045331f99dc48800e",
}
path = run_dir / "stage1_validation_summary.json"
temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
print(json.dumps(summary, sort_keys=True))
PY
}

run_one() {
  local algorithm="$1"
  local group run_dir pid launch_log validation_log evaluations elapsed
  local started_epoch last_status_epoch last_gpu_epoch now

  CURRENT_ALGORITHM="$algorithm"
  CURRENT_RUN_DIR=""
  CURRENT_PID=""
  check_source
  check_gpu_idle
  launch_log="$ROOT/logs/${algorithm}_launcher.log"
  write_state launching "$algorithm" "canonical launcher starting"
  printf '%s\t%s\tlaunching\n' "$(date -Iseconds)" "$algorithm" >> "$ROOT/timeline.tsv"

  cd "$REPO"
  bash llama_7B_LayerImportance.sh run "$algorithm" \
    --preset mrpc-blb-stage2-rl \
    --dataset mrpc \
    --model-type bert-base \
    --comparator-stage1-only \
    --persistent-root "$ROOT/persistent" \
    --fresh > "$launch_log" 2>&1

  group="$ROOT/persistent/$algorithm/bert-base/mrpc"
  run_dir="$(cat "$group/LATEST_RUN_DIR")"
  pid="$(cat "$group/LATEST_PID")"
  case "$run_dir" in "$group"/*) ;; *) return 76 ;; esac
  CURRENT_RUN_DIR="$run_dir"
  CURRENT_PID="$pid"
  printf '%s\t%s\tstarted\t%s\t%s\n' "$(date -Iseconds)" "$algorithm" "$pid" "$run_dir" >> "$ROOT/timeline.tsv"

  started_epoch="$(date +%s)"
  last_status_epoch=0
  last_gpu_epoch=0
  while process_is_live "$pid"; do
    now="$(date +%s)"
    if [ $((now - last_status_epoch)) -ge 30 ]; then
      evaluations="$(observation_count "$algorithm" "$run_dir")"
      elapsed=$((now - started_epoch))
      write_state running "$algorithm" "formal Stage-1 search; elapsed_seconds=$elapsed" "$run_dir" "$pid" "$evaluations"
      last_status_epoch="$now"
    fi
    if [ $((now - last_gpu_epoch)) -ge 15 ]; then
      nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,power.draw \
        --format=csv,noheader,nounits >> "$ROOT/gpu_samples.csv" 2>/dev/null || true
      last_gpu_epoch="$now"
    fi
    sleep 5
  done

  validation_log="$ROOT/logs/${algorithm}_validation.log"
  validate_stage1_result "$algorithm" "$run_dir" "$launch_log" > "$validation_log" 2>&1
  evaluations="$(observation_count "$algorithm" "$run_dir")"
  elapsed=$(($(date +%s) - started_epoch))
  printf '%s\t%s\tvalidated\t%s\t%s\n' "$(date -Iseconds)" "$algorithm" "$evaluations" "$elapsed" >> "$ROOT/timeline.tsv"
  write_state stage_completed "$algorithm" "formal Stage-1 result validated; elapsed_seconds=$elapsed" "$run_dir" "" "$evaluations"
}

run_or_reuse() {
  local algorithm="$1"
  local group run_dir launch_log validation_log evaluations
  group="$ROOT/persistent/$algorithm/bert-base/mrpc"
  launch_log="$ROOT/logs/${algorithm}_launcher.log"
  if [ -s "$group/LATEST_RUN_DIR" ]; then
    run_dir="$(cat "$group/LATEST_RUN_DIR")"
    if [ -f "$run_dir/stage1_comparator/$algorithm/COMPLETED" ]; then
      check_source
      validation_log="$ROOT/logs/${algorithm}_validation.log"
      validate_stage1_result "$algorithm" "$run_dir" "$launch_log" > "$validation_log" 2>&1
      evaluations="$(observation_count "$algorithm" "$run_dir")"
      printf '%s\t%s\trevalidated\t%s\t%s\n' "$(date -Iseconds)" "$algorithm" "$evaluations" "$run_dir" >> "$ROOT/timeline.tsv"
      write_state stage_completed "$algorithm" "existing formal Stage-1 result revalidated" "$run_dir" "" "$evaluations"
      return 0
    fi
  fi
  run_one "$algorithm"
}

on_exit() {
  local rc=$?
  trap - EXIT
  if [ "$rc" -eq 0 ]; then
    write_state completed "" "all three formal Stage-1 searches completed and validated"
  else
    write_state failed "$CURRENT_ALGORITHM" "queue exit code $rc" "$CURRENT_RUN_DIR" "$CURRENT_PID"
  fi
  printf '%s\tqueue\texit\t%s\n' "$(date -Iseconds)" "$rc" >> "$ROOT/timeline.tsv"
  exit "$rc"
}
trap on_exit EXIT

mkdir -p "$ROOT/logs"
exec 9>"$ROOT/queue.lock"
flock -n 9 || exit 77
printf '%s\n' "$$" > "$ROOT/queue.pid"
[ -f "$ROOT/timeline.tsv" ] || printf 'timestamp\talgorithm\tevent\tvalue\textra\n' > "$ROOT/timeline.tsv"
[ -f "$ROOT/gpu_samples.csv" ] || printf '%s\n' 'timestamp, index, name, utilization.gpu [%], memory.used [MiB], power.draw [W]' > "$ROOT/gpu_samples.csv"
write_state starting greedy "queue initialized"

run_or_reuse greedy
run_or_reuse bo_rf
run_or_reuse coinn_ga
