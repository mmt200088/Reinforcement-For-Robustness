#!/usr/bin/env bash
set -uo pipefail

# Serial Stage-2 baseline/optimized parity and runtime gate. Case launch failures
# are recorded instead of aborting so every requested optimized case is run.

: "${BASELINE_ROOT:?set BASELINE_ROOT to the clean baseline worktree}"
: "${OPTIMIZED_ROOT:?set OPTIMIZED_ROOT to the clean optimized worktree}"
: "${ARTIFACT_DIR:?set ARTIFACT_DIR}"
BATCH_SIZES="${BATCH_SIZES:-64 128 256}"
EPISODES="${EPISODES:-600}"
REWARD_DEVICES="${REWARD_DEVICES:-0,1,2,3,4}"

GPU_SAMPLE_INTERVAL_SECONDS="${GPU_SAMPLE_INTERVAL_SECONDS:-2}"
GATE_POLL_INTERVAL_SECONDS="${GATE_POLL_INTERVAL_SECONDS:-2}"
STAGE2_GATE_CASE_TIMEOUT_SECONDS="${STAGE2_GATE_CASE_TIMEOUT_SECONDS:-14400}"
STAGE2_GATE_PYTHON="${STAGE2_GATE_PYTHON:-python3}"
STAGE2_GATE_TERMINATION_GRACE_SECONDS="${STAGE2_GATE_TERMINATION_GRACE_SECONDS:-10}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
COMPARATOR="${STAGE2_GATE_COMPARATOR:-${SCRIPT_DIR}/stage2_ngpu_ab_compare.py}"
GPU_REPORTER="${STAGE2_GATE_GPU_REPORTER:-${SCRIPT_DIR}/gpu_utilization_report.py}"
EXPECTED_BASELINE_SHA="${EXPECTED_BASELINE_SHA:-48b03e869934aa8b3aa904a1fe8b611a1e2d618a}"
EXPECTED_OPTIMIZED_SHA="${EXPECTED_OPTIMIZED_SHA:-$(
  git -C "${SCRIPT_DIR}/.." rev-parse HEAD 2>/dev/null
)}"

fatal() {
  printf '[gate][FATAL] %s\n' "$*" >&2
  exit 2
}

canonical_directory() {
  local path="$1"
  [ -d "$path" ] || return 1
  (cd "$path" && pwd -P)
}

path_is_within() {
  local path="$1"
  local scope="$2"
  [ "$path" = "$scope" ] || [ "${path#"$scope"/}" != "$path" ]
}

preflight_clean_root() {
  local label="$1"
  local root="$2"
  local allowed_scope="${3:-}"
  local allowed_relative=""
  local path
  local rejected=""
  if ! git -C "$root" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    printf '[gate][FATAL] %s root is not a Git worktree: %s\n' "$label" "$root" >&2
    return 1
  fi
  if ! git -C "$root" diff --quiet --ignore-submodules -- \
      || ! git -C "$root" diff --cached --quiet --ignore-submodules --; then
    printf '[gate][FATAL] %s root is dirty: tracked modifications in %s\n' \
      "$label" "$root" >&2
    git -C "$root" status --short --untracked-files=no >&2 || true
    return 1
  fi

  if [ -n "$allowed_scope" ] && path_is_within "$allowed_scope" "$root"; then
    if [ "$allowed_scope" = "$root" ]; then
      allowed_relative="."
    else
      allowed_relative="${allowed_scope#"$root"/}"
    fi
  fi
  while IFS= read -r -d '' path; do
    if [ "$allowed_relative" = "." ]; then
      continue
    fi
    case "$path" in
      "$allowed_relative"|"$allowed_relative"/*) ;;
      *) rejected="${rejected}${rejected:+$'\n'}?? ${path}" ;;
    esac
  done < <(git -C "$root" ls-files --others --exclude-standard -z)
  if [ -n "$rejected" ]; then
    printf '[gate][FATAL] %s root is dirty outside the artifact scope: %s\n%s\n' \
      "$label" "$root" "$rejected" >&2
    return 1
  fi
}

BASELINE_ROOT="$(canonical_directory "$BASELINE_ROOT")" \
  || fatal "baseline root does not exist: ${BASELINE_ROOT}"
OPTIMIZED_ROOT="$(canonical_directory "$OPTIMIZED_ROOT")" \
  || fatal "optimized root does not exist: ${OPTIMIZED_ROOT}"
[ "$BASELINE_ROOT" != "$OPTIMIZED_ROOT" ] \
  || fatal "baseline and optimized roots resolve to the same root: ${BASELINE_ROOT}"

mkdir -p "$ARTIFACT_DIR" || fatal "cannot create ARTIFACT_DIR: ${ARTIFACT_DIR}"
ARTIFACT_DIR="$(canonical_directory "$ARTIFACT_DIR")" \
  || fatal "cannot resolve ARTIFACT_DIR: ${ARTIFACT_DIR}"
STAGE2_GATE_ARTIFACT_SCOPE="${STAGE2_GATE_ARTIFACT_SCOPE:-$(dirname "$ARTIFACT_DIR")}"
STAGE2_GATE_ARTIFACT_SCOPE="$(canonical_directory "$STAGE2_GATE_ARTIFACT_SCOPE")" \
  || fatal "artifact scope does not exist: ${STAGE2_GATE_ARTIFACT_SCOPE}"
path_is_within "$ARTIFACT_DIR" "$STAGE2_GATE_ARTIFACT_SCOPE" \
  || fatal "ARTIFACT_DIR must be inside STAGE2_GATE_ARTIFACT_SCOPE"

case "$EXPECTED_BASELINE_SHA" in
  *[!0-9a-fA-F]*|'') fatal "EXPECTED_BASELINE_SHA must be a Git object id" ;;
esac
case "$EXPECTED_OPTIMIZED_SHA" in
  *[!0-9a-fA-F]*|'') fatal "EXPECTED_OPTIMIZED_SHA must be a Git object id" ;;
esac
[ "${#EXPECTED_BASELINE_SHA}" -eq 40 ] \
  || fatal "EXPECTED_BASELINE_SHA must be a full 40-character SHA"
[ "${#EXPECTED_OPTIMIZED_SHA}" -eq 40 ] \
  || fatal "EXPECTED_OPTIMIZED_SHA must be a full 40-character SHA"
actual_baseline_sha="$(git -C "$BASELINE_ROOT" rev-parse HEAD)" \
  || fatal "cannot resolve baseline HEAD"
actual_optimized_sha="$(git -C "$OPTIMIZED_ROOT" rev-parse HEAD)" \
  || fatal "cannot resolve optimized HEAD"
[ "$actual_baseline_sha" = "$EXPECTED_BASELINE_SHA" ] \
  || fatal "baseline HEAD ${actual_baseline_sha} != expected ${EXPECTED_BASELINE_SHA}"
[ "$actual_optimized_sha" = "$EXPECTED_OPTIMIZED_SHA" ] \
  || fatal "optimized HEAD ${actual_optimized_sha} != expected ${EXPECTED_OPTIMIZED_SHA}"

preflight_clean_root baseline "$BASELINE_ROOT" || exit $?
preflight_clean_root optimized "$OPTIMIZED_ROOT" "$STAGE2_GATE_ARTIFACT_SCOPE" || exit $?

case "$EPISODES" in
  ''|*[!0-9]*) fatal "EPISODES must be a positive integer: ${EPISODES}" ;;
  0) fatal "EPISODES must be positive" ;;
esac
case "$STAGE2_GATE_CASE_TIMEOUT_SECONDS" in
  ''|*[!0-9]*|0) fatal "STAGE2_GATE_CASE_TIMEOUT_SECONDS must be positive" ;;
esac
case "$STAGE2_GATE_TERMINATION_GRACE_SECONDS" in
  ''|*[!0-9]*|0) fatal "STAGE2_GATE_TERMINATION_GRACE_SECONDS must be positive" ;;
esac
case "$REWARD_DEVICES" in
  ''|*[!0-9,]*) fatal "REWARD_DEVICES must be a comma-separated index list" ;;
  *,) fatal "REWARD_DEVICES must not end with a comma" ;;
  *,,*) fatal "REWARD_DEVICES must not contain empty indices" ;;
esac
seen_devices=","
old_ifs="$IFS"
IFS=','
for reward_device in $REWARD_DEVICES; do
  case "$seen_devices" in
    *",${reward_device},"*) fatal "duplicate GPU index in REWARD_DEVICES: ${reward_device}" ;;
  esac
  seen_devices="${seen_devices}${reward_device},"
done
IFS="$old_ifs"
command -v nvidia-smi >/dev/null 2>&1 \
  || fatal "nvidia-smi is required for the idle-GPU preflight"
command -v setsid >/dev/null 2>&1 \
  || fatal "setsid is required for owned training process groups"
command -v "$STAGE2_GATE_PYTHON" >/dev/null 2>&1 \
  || fatal "Python interpreter not found: ${STAGE2_GATE_PYTHON}"
[ -f "$COMPARATOR" ] || fatal "Stage-2 comparator not found: ${COMPARATOR}"
[ -f "$GPU_REPORTER" ] || fatal "GPU utilization reporter not found: ${GPU_REPORTER}"

compute_apps_output="$({
  nvidia-smi --id="$REWARD_DEVICES" \
    --query-compute-apps=pid,gpu_uuid,used_gpu_memory \
    --format=csv,noheader,nounits
} 2>&1)"
compute_apps_rc=$?
if [ "$compute_apps_rc" -ne 0 ]; then
  fatal "failed to query requested GPUs ${REWARD_DEVICES}: ${compute_apps_output}"
fi
busy_compute_apps="$(
  printf '%s\n' "$compute_apps_output" \
    | awk 'NF && $0 !~ /No running processes found/ {print}'
)"
if [ -n "$busy_compute_apps" ]; then
  printf '[gate][FATAL] requested GPUs are not idle; compute owners:\n%s\n' \
    "$busy_compute_apps" >&2
  exit 3
fi

mkdir -p "$ARTIFACT_DIR/cases"

printf '%s\n' "$BASELINE_ROOT" > "$ARTIFACT_DIR/baseline_root.txt"
printf '%s\n' "$OPTIMIZED_ROOT" > "$ARTIFACT_DIR/optimized_root.txt"
printf '%s\n' "$actual_baseline_sha" > "$ARTIFACT_DIR/baseline_head.txt"
printf '%s\n' "$actual_optimized_sha" > "$ARTIFACT_DIR/optimized_head.txt"
printf '%s\n' "$EXPECTED_BASELINE_SHA" > "$ARTIFACT_DIR/expected_baseline_head.txt"
printf '%s\n' "$EXPECTED_OPTIMIZED_SHA" > "$ARTIFACT_DIR/expected_optimized_head.txt"

logical_device_spec() {
  local physical_spec="$1"
  local old_ifs="$IFS"
  local device
  local index=0
  local logical=""
  IFS=','
  for device in $physical_spec; do
    case "$device" in
      ''|*[!0-9]*) IFS="$old_ifs"; return 1 ;;
    esac
    if [ -n "$logical" ]; then
      logical="${logical},"
    fi
    logical="${logical}${index}"
    index=$((index + 1))
  done
  IFS="$old_ifs"
  [ "$index" -gt 0 ] || return 1
  printf '%s\n' "$logical"
}

prepare_stage1_record() {
  local source_root="$1"
  local case_dir="$2"
  local source="${STAGE2_GATE_STAGE1_RECORD_SOURCE:-Parting Chapter/stage1/record}"
  local target="$case_dir/stage1/record"
  if [ "${source#/}" = "$source" ]; then
    source="$source_root/$source"
  fi
  if [ -e "$target" ]; then
    return 0
  fi
  if [ ! -e "$source" ]; then
    printf '[gate][warning] Stage-1 record source not found: %s\n' "$source"
    return 0
  fi
  mkdir -p "$(dirname "$target")" || return 1
  ln -s "$(cd "$(dirname "$source")" && pwd -P)/$(basename "$source")" "$target"
}

write_instrumented_launcher() {
  local source_root="$1"
  local case_dir="$2"
  local source="$source_root/llama_7B_LayerImportance.sh"
  local target="$case_dir/instrumented_llama_7B_LayerImportance.sh"
  [ -f "$source" ] || {
    printf '[gate][FATAL] launcher not found: %s\n' "$source" >&2
    return 1
  }
  cp "$source" "$target" || return 1
  cat >> "$target" <<'SH'

if [ -n "${STAGE2_GATE_TRAIN_EXIT_FILE:-}" ]; then
  set +e
  wait "$JOB_PID"
  stage2_gate_train_rc=$?
  printf '%s\n' "$stage2_gate_train_rc" > "$STAGE2_GATE_TRAIN_EXIT_FILE"
  exit "$stage2_gate_train_rc"
fi
SH
  chmod +x "$target" || return 1
  printf '%s\n' "$target"
}

process_group_exists() {
  local pgid="$1"
  kill -0 -- "-${pgid}" 2>/dev/null
}

wait_for_process_group_exit() {
  local pgid="$1"
  local timeout_seconds="$2"
  local started now
  started="$(date +%s)"
  while process_group_exists "$pgid"; do
    now="$(date +%s)"
    [ $((now - started)) -lt "$timeout_seconds" ] || return 1
    sleep 1
  done
}

terminate_process_group() {
  local pgid="$1"
  process_group_exists "$pgid" || return 0
  printf '[gate] sending SIGINT to owned process group %s\n' "$pgid" >&2
  kill -INT -- "-${pgid}" 2>/dev/null || true
  wait_for_process_group_exit "$pgid" "$STAGE2_GATE_TERMINATION_GRACE_SECONDS" \
    && return 0
  printf '[gate] sending SIGTERM to owned process group %s\n' "$pgid" >&2
  kill -TERM -- "-${pgid}" 2>/dev/null || true
  wait_for_process_group_exit "$pgid" "$STAGE2_GATE_TERMINATION_GRACE_SECONDS" \
    && return 0
  printf '[gate] sending SIGKILL to owned process group %s\n' "$pgid" >&2
  kill -KILL -- "-${pgid}" 2>/dev/null || true
  wait_for_process_group_exit "$pgid" "$STAGE2_GATE_TERMINATION_GRACE_SECONDS" \
    || true
}

wait_for_owned_launcher() {
  local launcher_pid="$1"
  local timeout_seconds="$2"
  local started now wait_rc timed_out=0
  started="$(date +%s)"
  while kill -0 "$launcher_pid" 2>/dev/null; do
    now="$(date +%s)"
    if [ $((now - started)) -ge "$timeout_seconds" ]; then
      printf '[gate][FATAL] owned training group %s exceeded %ss\n' \
        "$launcher_pid" "$timeout_seconds" >&2
      terminate_process_group "$launcher_pid"
      timed_out=1
      break
    fi
    sleep "$GATE_POLL_INTERVAL_SECONDS"
  done
  wait "$launcher_pid"
  wait_rc=$?
  if process_group_exists "$launcher_pid"; then
    printf '[gate][warning] reaping residual processes in group %s\n' \
      "$launcher_pid" >&2
    terminate_process_group "$launcher_pid"
  fi
  [ "$timed_out" -eq 0 ] || return 124
  return "$wait_rc"
}

default_case_launcher() {
  local case_name="$1"
  local source_root="$2"
  local probe_batch_size="$3"
  local episodes="$4"
  local physical_devices="$5"
  local case_dir="$6"
  local persistent_root="$case_dir/persistent"
  local training_exit_file="$case_dir/training_exit_code.txt"
  local logical_devices
  local instrumented_launcher
  local launcher_pid
  local launch_rc reported_rc
  local -a probe_batch_args=()

  logical_devices="$(logical_device_spec "$physical_devices")" || return 64
  prepare_stage1_record "$source_root" "$case_dir" || return 65
  mkdir -p "$persistent_root" || return 66
  instrumented_launcher="$(write_instrumented_launcher "$source_root" "$case_dir")" \
    || return 68

  # The 48b03e8 baseline predates these flags and gets its F1/F4 batch 64
  # behavior from the preset/evaluator defaults. Optimized cases set both.
  if [ "$case_name" != "base64" ]; then
    probe_batch_args=(
      --blb-v3-probe-batch-size "$probe_batch_size"
      --blb-v3-validation-probe-batch-size "$probe_batch_size"
    )
  fi

  (
    cd "$source_root" || exit 67
    export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${source_root}:${source_root}/Rescale_optimizer"
    export HF_HOME="${HF_HOME:-/hy-tmp/hf_cache}"
    export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
    export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
    export GLUE_LOCAL_DATASET_DIR="${GLUE_LOCAL_DATASET_DIR:-/hy-tmp/glue_data}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
    export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
    export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
    export BLB_STAGE2_POLICY_DEVICE="${BLB_STAGE2_POLICY_DEVICE:-worker}"
    export BLB_STAGE2_DYNAMIC_ASSIGNMENT="${BLB_STAGE2_DYNAMIC_ASSIGNMENT:-1}"
    export ALLOW_SHORT_RL_BENCHMARK=1
    export CUDA_VISIBLE_DEVICES="$physical_devices"
    export STAGE2_GATE_TRAIN_EXIT_FILE="$training_exit_file"

    exec setsid bash "$instrumented_launcher" run rl \
        --preset mrpc-blb-stage2-rl \
        --persistent-root "$persistent_root" \
        --stage2-fixed-config-source all4 \
        --blb-v3-fusion-count-action 1 \
        --blb-v3-sequential-rl true \
        --blb-v3-substage-mode false \
        --blb-v3-decision-granularity layer \
        --blb-v3-reward-design robust_constrained \
        --blb-v3-sequential-invalid-penalty 1.0 \
        --blb-v3-sequential-cost-shaping-coeff 0.0 \
        --blb-v3-sequential-fusion-shaping-coeff 0.0 \
        --stage2-search-episodes "$episodes" \
        --stage2-search-lr 5e-5 \
        --ppo-update-interval 120 \
        --stage2-rollout-size 120 \
        --stage2-k-trials 5 \
        --blb-v3-online-k-trials 5 \
        --stage2-probe-size 256 \
        --batch-size 64 \
        --stage1-accuracy-tolerance 0.001 \
        --stage2-limit-tolerance 0.001 \
        --stage2-stability-tolerance 1.2 \
        --stage2-stability-multiplier 2.0 \
        --stage2-calibrate-baseline-samples 8 \
        --blb-v3-promotion-validation-trials 25 \
        --blb-v3-final-selection-validation-trials 25 \
        --blb-v3-baseline-groups 5 \
        --blb-v3-baseline-trials-per-group 5 \
        --blb-v3-constraint-bootstrap-samples 4096 \
        --blb-v3-online-constraint-probability 0.50 \
        --blb-v3-promotion-constraint-probability 0.80 \
        --blb-v3-final-constraint-probability 0.95 \
        --blb-v3-reward-devices "$logical_devices" \
        --stage2-workers-per-device 1 \
        --stage2-save-interval 200 \
        --stage2-eval-interval 100 \
        --random-seed 42 \
        --rl-algo ppo \
        --skip-final-eval \
        --fresh \
        "${probe_batch_args[@]}"
  ) > "$case_dir/launch.log" 2>&1 &
  launcher_pid=$!
  printf '%s\n' "$launcher_pid" > "$case_dir/training_process_group.txt"
  printf '%s\n' "$launcher_pid" > "$case_dir/worker_pids.txt"
  printf '[gate] %s owns training process group=%s\n' \
    "$case_name" "$launcher_pid"

  wait_for_owned_launcher "$launcher_pid" "$STAGE2_GATE_CASE_TIMEOUT_SECONDS"
  launch_rc=$?
  reported_rc="$launch_rc"
  if [ -s "$training_exit_file" ]; then
    reported_rc="$(tr -d '[:space:]' < "$training_exit_file")"
    case "$reported_rc" in
      ''|*[!0-9]*)
        printf '[gate][FATAL] %s wrote invalid training exit code: %s\n' \
          "$case_name" "$reported_rc" >&2
        reported_rc=125
        ;;
    esac
  else
    printf '%s\n' "$reported_rc" > "$training_exit_file"
  fi
  return "$reported_rc"
}

sample_gpu_usage() {
  local out_file="$1"
  printf 'timestamp,index,name,memory_used_mib,utilization_gpu_pct\n' > "$out_file"
  while true; do
    nvidia-smi --id="$REWARD_DEVICES" \
      --query-gpu=timestamp,index,name,memory.used,utilization.gpu \
      --format=csv,noheader,nounits >> "$out_file" 2>/dev/null || true
    sleep "$GPU_SAMPLE_INTERVAL_SECONDS"
  done
}

process_tree_pids() {
  local launcher_pid="$1"
  local worker_pid_file="$2"
  local roots="$launcher_pid"
  local pid
  if [ -f "$worker_pid_file" ]; then
    while IFS= read -r pid; do
      case "$pid" in
        ''|*[!0-9]*) ;;
        *) roots="${roots},${pid}" ;;
      esac
    done < "$worker_pid_file"
  fi
  ps -e -o pid= -o ppid= 2>/dev/null | awk -v roots="$roots" '
    BEGIN {
      count = split(roots, values, ",")
      for (i = 1; i <= count; i++) wanted[values[i]] = 1
    }
    { pid[NR] = $1; parent[NR] = $2 }
    END {
      changed = 1
      while (changed) {
        changed = 0
        for (i = 1; i <= NR; i++) {
          if (wanted[parent[i]] && !wanted[pid[i]]) {
            wanted[pid[i]] = 1
            changed = 1
          }
        }
      }
      for (value in wanted) if (value ~ /^[0-9]+$/) print value
    }
  ' | LC_ALL=C sort -n -u
}

sample_worker_inventory() {
  local case_dir="$1"
  local launcher_pid="$2"
  local out_file="$case_dir/worker_thread_inventory.txt"
  local worker_pid_file="$case_dir/worker_pids.txt"
  local timestamp pid task task_count command
  timestamp="$(date '+%Y-%m-%dT%H:%M:%S')"
  printf '# sample=%s launcher_pid=%s worker PID/thread inventory\n' \
    "$timestamp" "$launcher_pid" >> "$out_file"
  process_tree_pids "$launcher_pid" "$worker_pid_file" | while IFS= read -r pid; do
    kill -0 "$pid" 2>/dev/null || continue
    task_count="n/a"
    if [ -d "/proc/$pid/task" ]; then
      task_count=0
      for task in "/proc/$pid/task"/[0-9]*; do
        [ -d "$task" ] || continue
        task_count=$((task_count + 1))
      done
    fi
    command="$(ps -p "$pid" -o comm= 2>/dev/null | tr '\t' ' ' | head -n 1)"
    printf 'timestamp=%s pid=%s thread_count=%s command=%s\n' \
      "$timestamp" "$pid" "$task_count" "$command" >> "$out_file"
  done
}

find_latest_case_file() {
  local case_dir="$1"
  local filename="$2"
  local destination="$case_dir/diagnostics/$filename"
  if [ -f "$destination" ]; then
    printf '%s\n' "$destination"
    return 0
  fi
  find "$case_dir" -type f -name "$filename" ! -path "$destination" \
    -print 2>/dev/null | LC_ALL=C sort | tail -n 1
}

copy_case_evidence() {
  local case_dir="$1"
  local filename source destination
  mkdir -p "$case_dir/diagnostics"
  for filename in \
    episodes.jsonl ppo_updates.jsonl candidate_store.jsonl \
    diagnostics_summary.json; do
    destination="$case_dir/diagnostics/$filename"
    [ -f "$destination" ] && continue
    source="$(find_latest_case_file "$case_dir" "$filename")"
    [ -n "$source" ] || continue
    cp "$source" "$destination" || return 1
  done
}

normalize_candidate_evidence() {
  local input_path="$1"
  local output_path="$2"
  "$STAGE2_GATE_PYTHON" - \
    "$input_path" "$output_path" "${SCRIPT_DIR}/.." <<'PY'
import json
import pathlib
import sys

source = pathlib.Path(sys.argv[1])
destination = pathlib.Path(sys.argv[2])
repo_root = pathlib.Path(sys.argv[3])
sys.path.insert(0, str(repo_root))

from blb_stage2_rl.candidate_store import CandidateStore

rows = CandidateStore(source).iter_active_records()

storage_fields = {
    "created_at",
    "logical_generation",
    "raw_action_indices",
    "effective_action_indices",
    "raw_action_hash",
    "action_hash",
    "action_vector_hash",
    "effective_action_hash",
    "candidate_key_basis",
    "candidate_key",
    "identity_context_hash",
    "legacy_record",
    "rank_key",
}
trial_types = {"candidate_trial_group_v1", "candidate_trial_group_v2"}
promotion_types = {
    "candidate_promotion_status_v1",
    "candidate_promotion_status_v2",
}

def identity_context(row):
    context = row.get("identity_context")
    if isinstance(context, dict):
        return context
    metadata = row.get("trial_group_metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("identity_context"), dict):
        return metadata["identity_context"]
    raise ValueError("candidate evidence row has no logical identity context")

normalized = []
for row in rows:
    record_type = str(row.get("record_type", ""))
    if record_type in {"candidate_identity_context_v1", "candidate_store_recovery_v1"}:
        continue
    payload = {
        str(key): value
        for key, value in row.items()
        if key not in storage_fields and key != "record_type"
    }
    if record_type in trial_types:
        payload["record_type"] = "candidate_trial_group"
        payload["action_indices"] = [int(value) for value in row["action_indices"]]
        payload["identity_context"] = identity_context(row)
        metadata = row.get("trial_group_metadata")
        metadata = dict(metadata) if isinstance(metadata, dict) else {}
        metadata.pop("identity_context", None)
        if str(row.get("fidelity", "")).upper() == "F1":
            metadata.pop("boosted_overrides", None)
        payload["trial_group_metadata"] = metadata
    elif record_type in promotion_types:
        payload["record_type"] = "candidate_promotion_status"
        payload["action_indices"] = [int(value) for value in row["action_indices"]]
        payload["identity_context"] = identity_context(row)
        metadata = row.get("promotion_metadata")
        payload["promotion_metadata"] = (
            dict(metadata) if isinstance(metadata, dict) else {}
        )
    else:
        payload["record_type"] = record_type
    normalized.append(payload)

encoded = [json.dumps(row, sort_keys=True, separators=(",", ":")) for row in normalized]
encoded.sort()
with destination.open("w", encoding="utf-8") as handle:
    for line in encoded:
        handle.write(line + "\n")
PY
}

validate_case_evidence() {
  local case_dir="$1"
  local expected_episodes="$2"
  local missing=""
  local episode_count
  local wall
  local candidate="$case_dir/diagnostics/candidate_store.jsonl"
  local normalized="$case_dir/diagnostics/candidate_store.normalized.jsonl"
  local gpu_json="$case_dir/gpu_utilization.json"
  local gpu_markdown="$case_dir/gpu_utilization.md"
  local filename
  for filename in episodes.jsonl ppo_updates.jsonl candidate_store.jsonl; do
    if [ ! -s "$case_dir/diagnostics/$filename" ]; then
      missing="${missing} ${filename}"
    fi
  done
  if [ ! -s "$case_dir/wall_seconds.txt" ]; then
    missing="${missing} wall_seconds.txt"
  fi
  if [ -n "$missing" ]; then
    printf 'FAIL missing:%s\n' "$missing" > "$case_dir/evidence_status.txt"
    return 1
  fi
  episode_count="$(wc -l < "$case_dir/diagnostics/episodes.jsonl" | tr -d '[:space:]')"
  if [ "$episode_count" != "$expected_episodes" ]; then
    printf 'FAIL episode_count=%s expected=%s\n' \
      "$episode_count" "$expected_episodes" > "$case_dir/evidence_status.txt"
    return 1
  fi
  wall="$(tr -d '[:space:]' < "$case_dir/wall_seconds.txt")"
  if ! awk -v value="$wall" 'BEGIN { exit !(value + 0 > 0) }'; then
    printf 'FAIL invalid wall_seconds=%s\n' "$wall" > "$case_dir/evidence_status.txt"
    return 1
  fi
  if ! normalize_candidate_evidence "$candidate" "$normalized" \
      > "$case_dir/candidate_normalization.log" 2>&1; then
    printf 'FAIL candidate normalization\n' > "$case_dir/evidence_status.txt"
    return 1
  fi
  if ! "$STAGE2_GATE_PYTHON" "$GPU_REPORTER" \
      --episodes "$case_dir/diagnostics/episodes.jsonl" \
      --nvidia-smi-csv "$case_dir/nvidia_smi_samples.csv" \
      --visible-devices "$REWARD_DEVICES" \
      --out-json "$gpu_json" \
      --out-md "$gpu_markdown" \
      --require-all-visible-sampled-active \
      > "$case_dir/gpu_activity_validation.log" 2>&1; then
    printf 'FAIL requested GPU activity\n' > "$case_dir/gpu_activity_status.txt"
    printf 'FAIL requested GPU activity\n' > "$case_dir/evidence_status.txt"
    return 1
  fi
  printf 'PASS\n' > "$case_dir/gpu_activity_status.txt"
  printf 'PASS\n' > "$case_dir/evidence_status.txt"
}

LAST_LAUNCH_RC=0
LAST_EVIDENCE_PASS=0
ACTIVE_CASE_DIR=""
ACTIVE_SAMPLER_PID=""

cleanup_active_case() {
  local pgid=""
  local gate_launcher_pid=""
  if [ -n "$ACTIVE_SAMPLER_PID" ]; then
    kill "$ACTIVE_SAMPLER_PID" 2>/dev/null || true
    wait "$ACTIVE_SAMPLER_PID" 2>/dev/null || true
  fi
  if [ -n "$ACTIVE_CASE_DIR" ] \
      && [ -s "$ACTIVE_CASE_DIR/training_process_group.txt" ]; then
    pgid="$(tr -d '[:space:]' < "$ACTIVE_CASE_DIR/training_process_group.txt")"
    case "$pgid" in
      ''|*[!0-9]*) ;;
      *) terminate_process_group "$pgid" ;;
    esac
  fi
  if [ -n "$ACTIVE_CASE_DIR" ] \
      && [ -s "$ACTIVE_CASE_DIR/gate_launcher_pid.txt" ]; then
    gate_launcher_pid="$(
      tr -d '[:space:]' < "$ACTIVE_CASE_DIR/gate_launcher_pid.txt"
    )"
    case "$gate_launcher_pid" in
      ''|*[!0-9]*) ;;
      *)
        kill -TERM "$gate_launcher_pid" 2>/dev/null || true
        wait "$gate_launcher_pid" 2>/dev/null || true
        ;;
    esac
  fi
}

handle_gate_signal() {
  local signal_name="$1"
  trap - HUP INT TERM
  printf '[gate][FATAL] received %s; cleaning active case\n' \
    "$signal_name" >&2
  cleanup_active_case
  exit 130
}

trap 'handle_gate_signal HUP' HUP
trap 'handle_gate_signal INT' INT
trap 'handle_gate_signal TERM' TERM

run_case() {
  local case_name="$1"
  local source_root="$2"
  local batch_size="$3"
  local case_dir="$4"
  local gpu_samples="$case_dir/nvidia_smi_samples.csv"
  local launcher_log="$case_dir/gate_launcher.log"
  local inventory="$case_dir/worker_thread_inventory.txt"
  local start_s end_s measured_wall
  local sampler_pid launcher_pid launch_rc

  mkdir -p "$case_dir/diagnostics"
  ACTIVE_CASE_DIR="$case_dir"
  : > "$inventory"
  start_s="$(date +%s)"
  sample_gpu_usage "$gpu_samples" &
  sampler_pid=$!
  ACTIVE_SAMPLER_PID="$sampler_pid"

  if [ -n "${STAGE2_GATE_CASE_LAUNCHER:-}" ]; then
    "$STAGE2_GATE_CASE_LAUNCHER" \
      "$case_name" "$source_root" "$batch_size" "$EPISODES" \
      "$REWARD_DEVICES" "$case_dir" > "$launcher_log" 2>&1 &
  else
    default_case_launcher \
      "$case_name" "$source_root" "$batch_size" "$EPISODES" \
      "$REWARD_DEVICES" "$case_dir" > "$launcher_log" 2>&1 &
  fi
  launcher_pid=$!
  printf '%s\n' "$launcher_pid" > "$case_dir/gate_launcher_pid.txt"

  sample_worker_inventory "$case_dir" "$launcher_pid"
  while kill -0 "$launcher_pid" 2>/dev/null; do
    sleep "$GATE_POLL_INTERVAL_SECONDS"
    sample_worker_inventory "$case_dir" "$launcher_pid"
  done
  wait "$launcher_pid"
  launch_rc=$?

  kill "$sampler_pid" 2>/dev/null || true
  wait "$sampler_pid" 2>/dev/null || true
  ACTIVE_SAMPLER_PID=""
  end_s="$(date +%s)"
  measured_wall=$((end_s - start_s))
  if [ ! -s "$case_dir/wall_seconds.txt" ]; then
    printf '%s\n' "$measured_wall" > "$case_dir/wall_seconds.txt"
  fi
  printf '%s\n' "$launch_rc" > "$case_dir/launcher_exit_code.txt"

  copy_case_evidence "$case_dir" \
    > "$case_dir/evidence_copy.log" 2>&1 || true
  if validate_case_evidence "$case_dir" "$EPISODES"; then
    LAST_EVIDENCE_PASS=1
  else
    LAST_EVIDENCE_PASS=0
  fi
  LAST_LAUNCH_RC="$launch_rc"
  ACTIVE_CASE_DIR=""
  printf '[gate] case=%s launch_rc=%s evidence=%s wall=%ss\n' \
    "$case_name" "$launch_rc" "$LAST_EVIDENCE_PASS" \
    "$(cat "$case_dir/wall_seconds.txt" 2>/dev/null || printf 'n/a')"
}

# Default cases are base64,opt64,opt128,opt256. Custom BATCH_SIZES keeps the
# baseline fixed at 64 and creates one optimized case per requested size.
CASE_NAMES=(base64)
CASE_ROOTS=("$BASELINE_ROOT")
CASE_BATCHES=(64)
for batch_size in $BATCH_SIZES; do
  case "$batch_size" in
    ''|*[!0-9]*|0) fatal "invalid optimized probe batch size: ${batch_size}" ;;
  esac
  CASE_NAMES+=("opt${batch_size}")
  CASE_ROOTS+=("$OPTIMIZED_ROOT")
  CASE_BATCHES+=("$batch_size")
done

for case_name in "${CASE_NAMES[@]}"; do
  [ ! -e "$ARTIFACT_DIR/cases/$case_name" ] \
    || fatal "case artifact directory already exists: $ARTIFACT_DIR/cases/$case_name"
done

CASE_LAUNCH_RCS=()
CASE_EVIDENCE_PASSES=()
for case_index in "${!CASE_NAMES[@]}"; do
  case_name="${CASE_NAMES[$case_index]}"
  case_root="${CASE_ROOTS[$case_index]}"
  case_batch="${CASE_BATCHES[$case_index]}"
  case_dir="$ARTIFACT_DIR/cases/$case_name"
  printf '[gate] running %s from %s (batch=%s)\n' \
    "$case_name" "$case_root" "$case_batch"
  run_case "$case_name" "$case_root" "$case_batch" "$case_dir"
  CASE_LAUNCH_RCS+=("$LAST_LAUNCH_RC")
  CASE_EVIDENCE_PASSES+=("$LAST_EVIDENCE_PASS")
done

RESULTS_TSV="$ARTIFACT_DIR/case_results.tsv"
printf 'case\tkind\tsource_root\tbatch\tlaunch_pass\tevidence_pass\tcomparator_pass\tcandidate_pass\tsemantic_pass\twall_seconds\tspeedup\tcomparison_report\n' \
  > "$RESULTS_TSV"

base_dir="$ARTIFACT_DIR/cases/base64"
base_wall="$(tr -d '[:space:]' < "$base_dir/wall_seconds.txt" 2>/dev/null || true)"
base_launch_pass=0
[ "${CASE_LAUNCH_RCS[0]}" -eq 0 ] && base_launch_pass=1
base_evidence_pass="${CASE_EVIDENCE_PASSES[0]}"
base_semantic_pass=0
if [ "$base_launch_pass" -eq 1 ] && [ "$base_evidence_pass" -eq 1 ]; then
  base_semantic_pass=1
fi
printf 'base64\tbaseline\t%s\t64\t%s\t%s\t1\t1\t%s\t%s\t1.0\t%s\n' \
  "$BASELINE_ROOT" "$base_launch_pass" "$base_evidence_pass" \
  "$base_semantic_pass" "$base_wall" "$base_dir/reference.txt" >> "$RESULTS_TSV"

all_optimized_pass=1
winner_case=""
winner_wall=""

for case_index in "${!CASE_NAMES[@]}"; do
  [ "$case_index" -eq 0 ] && continue
  case_name="${CASE_NAMES[$case_index]}"
  case_root="${CASE_ROOTS[$case_index]}"
  case_batch="${CASE_BATCHES[$case_index]}"
  case_dir="$ARTIFACT_DIR/cases/$case_name"
  comparison_report="$case_dir/semantic_comparison.txt"
  candidate_report="$case_dir/candidate_comparison.txt"
  launch_pass=0
  evidence_pass="${CASE_EVIDENCE_PASSES[$case_index]}"
  comparator_pass=0
  candidate_pass=0
  semantic_pass=0
  [ "${CASE_LAUNCH_RCS[$case_index]}" -eq 0 ] && launch_pass=1

  if [ "$base_semantic_pass" -eq 1 ] && [ "$launch_pass" -eq 1 ] \
      && [ "$evidence_pass" -eq 1 ]; then
    "$STAGE2_GATE_PYTHON" "$COMPARATOR" \
      --one "$base_dir/diagnostics/episodes.jsonl" \
      --many "$case_dir/diagnostics/episodes.jsonl" \
      --one-ppo "$base_dir/diagnostics/ppo_updates.jsonl" \
      --many-ppo "$case_dir/diagnostics/ppo_updates.jsonl" \
      --one-wall "$base_dir/wall_seconds.txt" \
      --many-wall "$case_dir/wall_seconds.txt" \
      --one-log "$base_dir/launch.log" \
      --many-log "$case_dir/launch.log" \
      --require-equal \
      --out "$comparison_report" >/dev/null 2>&1
    comparator_rc=$?
    [ "$comparator_rc" -eq 0 ] && comparator_pass=1

    if cmp -s \
        "$base_dir/diagnostics/candidate_store.normalized.jsonl" \
        "$case_dir/diagnostics/candidate_store.normalized.jsonl"; then
      candidate_pass=1
      printf 'PASS: normalized logical candidate evidence matches base64\n' \
        > "$candidate_report"
    else
      printf 'FAIL: normalized logical candidate evidence mismatch vs base64\n' \
        > "$candidate_report"
    fi
  else
    printf 'SKIPPED: launch or evidence prerequisite failed\n' > "$comparison_report"
    printf 'SKIPPED: launch or evidence prerequisite failed\n' > "$candidate_report"
  fi

  if [ "$comparator_pass" -eq 1 ] && [ "$candidate_pass" -eq 1 ]; then
    semantic_pass=1
  else
    all_optimized_pass=0
  fi

  wall="$(tr -d '[:space:]' < "$case_dir/wall_seconds.txt" 2>/dev/null || true)"
  speedup=""
  if [ -n "$base_wall" ] && [ -n "$wall" ]; then
    speedup="$(awk -v baseline="$base_wall" -v optimized="$wall" '
      BEGIN { if (optimized + 0 > 0) printf "%.9f", baseline / optimized }
    ')"
  fi
  if [ "$semantic_pass" -eq 1 ]; then
    if [ -z "$winner_case" ] \
        || awk -v candidate="$wall" -v winner="$winner_wall" \
          'BEGIN { exit !(candidate + 0 < winner + 0) }'; then
      winner_case="$case_name"
      winner_wall="$wall"
    fi
  fi

  printf '%s\toptimized\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$case_name" "$case_root" "$case_batch" "$launch_pass" \
    "$evidence_pass" "$comparator_pass" "$candidate_pass" \
    "$semantic_pass" "$wall" "$speedup" "$comparison_report" \
    >> "$RESULTS_TSV"
done

if ! "$STAGE2_GATE_PYTHON" - \
    "$RESULTS_TSV" "$ARTIFACT_DIR/verdict.json" "$ARTIFACT_DIR/verdict.md" \
    "$winner_case" "$all_optimized_pass" <<'PY'
import csv
import json
import pathlib
import sys

results_path = pathlib.Path(sys.argv[1])
json_path = pathlib.Path(sys.argv[2])
markdown_path = pathlib.Path(sys.argv[3])
winner = sys.argv[4] or None
all_optimized_pass = sys.argv[5] == "1"

with results_path.open(encoding="utf-8", newline="") as handle:
    raw_rows = list(csv.DictReader(handle, delimiter="\t"))

rows = []
for raw in raw_rows:
    row = dict(raw)
    for key in (
        "launch_pass",
        "evidence_pass",
        "comparator_pass",
        "candidate_pass",
        "semantic_pass",
    ):
        row[key] = row[key] == "1"
    row["batch"] = int(row["batch"])
    for key in ("wall_seconds", "speedup"):
        row[key] = float(row[key]) if row[key] else None
    rows.append(row)

verdict = {
    "schema_version": "stage2_runtime_optimization_gate_v1",
    "semantic_parity": "PASS" if all_optimized_pass else "FAIL",
    "fastest_eligible_case": winner,
    "speed_ranking_policy": "lowest wall time among semantic-passing optimized cases",
    "cases": rows,
}
json_path.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8")

lines = [
    "# Stage-2 Runtime Optimization Gate",
    "",
    "| Case | Batch | Launch | Evidence | Semantic parity | Candidate parity | Wall (s) | Speedup |",
    "|---|---:|---|---|---|---|---:|---:|",
]
for row in rows:
    speedup = "n/a" if row["speedup"] is None else f'{row["speedup"]:.3f}x'
    wall = "n/a" if row["wall_seconds"] is None else f'{row["wall_seconds"]:.3f}'
    lines.append(
        f'| {row["case"]} | {row["batch"]} | '
        f'{"PASS" if row["launch_pass"] else "FAIL"} | '
        f'{"PASS" if row["evidence_pass"] else "FAIL"} | '
        f'{"PASS" if row["semantic_pass"] else "FAIL"} | '
        f'{"PASS" if row["candidate_pass"] else "FAIL"} | {wall} | {speedup} |'
    )
lines.extend([
    "",
    f'Overall semantic parity: {"PASS" if all_optimized_pass else "FAIL"}',
    f'Fastest eligible case (winner): {winner or "none"}',
])
markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
then
  fatal "failed to write verdict.json and verdict.md"
fi

printf '[gate] verdict=%s winner=%s\n' \
  "$ARTIFACT_DIR/verdict.json" "${winner_case:-none}"
if [ "$all_optimized_pass" -ne 1 ]; then
  printf '[gate][FAIL] one or more optimized cases failed semantic parity\n' >&2
  exit 1
fi
printf '[gate][PASS] all optimized cases passed semantic parity\n'
