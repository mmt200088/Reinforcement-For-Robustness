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
STAGE2_GATE_MIN_GPU_ACTIVE_SAMPLES="${STAGE2_GATE_MIN_GPU_ACTIVE_SAMPLES:-3}"
STAGE2_GATE_MIN_GPU_ACTIVE_SAMPLE_RATE="${STAGE2_GATE_MIN_GPU_ACTIVE_SAMPLE_RATE:-0.05}"
STAGE2_GATE_MIN_GPU_MAX_UTIL_PCT="${STAGE2_GATE_MIN_GPU_MAX_UTIL_PCT:-10}"
STAGE2_GATE_MIN_PROBE_EPISODE_COVERAGE="${STAGE2_GATE_MIN_PROBE_EPISODE_COVERAGE:-0.95}"
STAGE2_GATE_MIN_PROBE_TRIAL_BALANCE="${STAGE2_GATE_MIN_PROBE_TRIAL_BALANCE:-0.95}"

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
  local rejected_ignored=""
  if ! git -C "$root" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    printf '[gate][FATAL] %s root is not a Git worktree: %s\n' "$label" "$root" >&2
    return 1
  fi
  if ! git -C "$root" diff --quiet --ignore-submodules -- \
      || ! git -C "$root" diff --cached --quiet --ignore-submodules --; then
    printf '[gate][FATAL] %s root is dirty: tracked modifications in %s\n' \
      "$label" "$root" >&2
    git -C "$root" status --porcelain --untracked-files=no >&2 || true
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
  while IFS= read -r -d '' path; do
    case "$path" in
      "$allowed_relative"|"$allowed_relative"/*)
        [ -n "$allowed_relative" ] && continue
        ;;
    esac
    case "$path" in
      __pycache__/*|*/__pycache__/*|*.py[co]|.pytest_cache/*|*/.pytest_cache/*|\
      .mypy_cache/*|*/.mypy_cache/*|.ruff_cache/*|*/.ruff_cache/*)
        ;;
      *)
        rejected_ignored="${rejected_ignored}${rejected_ignored:+$'\n'}!! ${path}"
        ;;
    esac
  done < <(git -C "$root" ls-files --others --ignored --exclude-standard -z)
  if [ -n "$rejected_ignored" ]; then
    printf '[gate][FATAL] %s root has ignored runtime inputs outside the artifact scope: %s\n%s\n' \
      "$label" "$root" "$rejected_ignored" >&2
    return 1
  fi
}

verify_source_state() {
  local label="$1"
  local root="$2"
  local expected_sha="$3"
  local allowed_scope="${4:-}"
  local actual_sha
  actual_sha="$(git -C "$root" rev-parse HEAD 2>/dev/null)" || {
    printf '[gate][FATAL] cannot resolve %s source HEAD: %s\n' \
      "$label" "$root" >&2
    return 1
  }
  if [ "$actual_sha" != "$expected_sha" ]; then
    printf '[gate][FATAL] %s source HEAD %s != expected %s\n' \
      "$label" "$actual_sha" "$expected_sha" >&2
    return 1
  fi
  preflight_clean_root "$label" "$root" "$allowed_scope"
}

verify_all_source_states() {
  verify_source_state baseline "$BASELINE_ROOT" "$EXPECTED_BASELINE_SHA" \
    || return 1
  verify_source_state optimized "$OPTIMIZED_ROOT" "$EXPECTED_OPTIMIZED_SHA" \
    "$STAGE2_GATE_ARTIFACT_SCOPE"
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
[ "$STAGE2_GATE_ARTIFACT_SCOPE" != "$OPTIMIZED_ROOT" ] \
  || fatal "STAGE2_GATE_ARTIFACT_SCOPE cannot be the optimized worktree root"
if path_is_within "$STAGE2_GATE_ARTIFACT_SCOPE" "$OPTIMIZED_ROOT"; then
  CANONICAL_SERVER_RUN_ROOT="$OPTIMIZED_ROOT/experiments/server_command_runs"
  [ -d "$CANONICAL_SERVER_RUN_ROOT" ] \
    || fatal "artifact scope inside optimized source must use experiments/server_command_runs"
  CANONICAL_SERVER_RUN_ROOT="$(canonical_directory "$CANONICAL_SERVER_RUN_ROOT")" \
    || fatal "cannot resolve canonical server run root"
  path_is_within "$STAGE2_GATE_ARTIFACT_SCOPE" "$CANONICAL_SERVER_RUN_ROOT" \
    || fatal "artifact scope inside optimized source must be under experiments/server_command_runs"
fi

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

verify_all_source_states || exit $?

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
case "$STAGE2_GATE_MIN_GPU_ACTIVE_SAMPLES" in
  ''|*[!0-9]*|0) fatal "STAGE2_GATE_MIN_GPU_ACTIVE_SAMPLES must be positive" ;;
esac
if ! awk -v value="$STAGE2_GATE_MIN_GPU_ACTIVE_SAMPLE_RATE" '
    BEGIN {
      valid = value ~ /^[0-9]+([.][0-9]+)?$/ && value + 0 > 0 && value + 0 <= 1
      exit !valid
    }'; then
  fatal "STAGE2_GATE_MIN_GPU_ACTIVE_SAMPLE_RATE must be in (0, 1]"
fi
if ! awk -v value="$STAGE2_GATE_MIN_GPU_MAX_UTIL_PCT" '
    BEGIN {
      valid = value ~ /^[0-9]+([.][0-9]+)?$/ && value + 0 > 0 && value + 0 <= 100
      exit !valid
    }'; then
  fatal "STAGE2_GATE_MIN_GPU_MAX_UTIL_PCT must be in (0, 100]"
fi
for rate_name in \
    STAGE2_GATE_MIN_PROBE_EPISODE_COVERAGE \
    STAGE2_GATE_MIN_PROBE_TRIAL_BALANCE; do
  rate_value="${!rate_name}"
  if ! awk -v value="$rate_value" '
      BEGIN {
        valid = value ~ /^[0-9]+([.][0-9]+)?$/ && value + 0 > 0 && value + 0 <= 1
        exit !valid
      }'; then
    fatal "${rate_name} must be in (0, 1]"
  fi
done
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

verify_requested_gpus_idle() {
  local boundary="$1"
  local compute_apps_output compute_apps_rc busy_compute_apps
  compute_apps_output="$({
    nvidia-smi --id="$REWARD_DEVICES" \
      --query-compute-apps=pid,gpu_uuid,used_gpu_memory \
      --format=csv,noheader,nounits
  } 2>&1)"
  compute_apps_rc=$?
  if [ "$compute_apps_rc" -ne 0 ]; then
    printf '[gate][FATAL] failed to query requested GPUs %s at %s: %s\n' \
      "$REWARD_DEVICES" "$boundary" "$compute_apps_output" >&2
    return 3
  fi
  busy_compute_apps="$(
    printf '%s\n' "$compute_apps_output" \
      | awk 'NF && $0 !~ /No running processes found/ {print}'
  )"
  if [ -n "$busy_compute_apps" ]; then
    printf '[gate][FATAL] requested GPUs are not idle at %s; compute owners:\n%s\n' \
      "$boundary" "$busy_compute_apps" >&2
    return 3
  fi
}

verify_requested_gpus_idle preflight || exit $?

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

process_group_exists() {
  local pgid="$1"
  kill -0 -- "-${pgid}" 2>/dev/null
}

wait_for_process_group_exit() {
  local pgid="$1"
  local timeout_seconds="$2"
  local started_ns now_ns timeout_ns
  started_ns="$(date +%s%N)"
  timeout_ns=$((timeout_seconds * 1000000000))
  while process_group_exists "$pgid"; do
    now_ns="$(date +%s%N)"
    [ $((now_ns - started_ns)) -lt "$timeout_ns" ] || return 1
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

terminate_owned_pid_or_group() {
  local pid="$1"
  local pgid
  case "$pid" in
    ''|*[!0-9]*) return 0 ;;
  esac
  pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d '[:space:]')"
  if [ "$pgid" = "$pid" ]; then
    terminate_process_group "$pgid"
  else
    kill -TERM "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
}

wait_for_owned_launcher() {
  local launcher_pid="$1"
  local timeout_seconds="$2"
  local case_dir="${3:-}"
  local started_ns now_ns timeout_ns wait_rc timed_out=0 residual=0
  started_ns="$(date +%s%N)"
  timeout_ns=$((timeout_seconds * 1000000000))
  while kill -0 "$launcher_pid" 2>/dev/null; do
    now_ns="$(date +%s%N)"
    if [ $((now_ns - started_ns)) -ge "$timeout_ns" ]; then
      printf '[gate][FATAL] owned training group %s exceeded %ss\n' \
        "$launcher_pid" "$timeout_seconds" >&2
      terminate_owned_pid_or_group "$launcher_pid"
      timed_out=1
      break
    fi
    sleep "$GATE_POLL_INTERVAL_SECONDS"
    if [ -n "$case_dir" ]; then
      sample_worker_inventory "$case_dir" "$launcher_pid"
    fi
  done
  wait "$launcher_pid"
  wait_rc=$?
  if process_group_exists "$launcher_pid"; then
    printf '[gate][warning] reaping residual processes in group %s\n' \
      "$launcher_pid" >&2
    terminate_process_group "$launcher_pid"
    residual=1
  fi
  [ "$timed_out" -eq 0 ] || return 124
  if [ "$residual" -eq 1 ] && [ "$wait_rc" -eq 0 ]; then
    return 125
  fi
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
  local source_launcher="$source_root/llama_7B_LayerImportance.sh"
  local logical_devices
  local -a probe_batch_args=()

  logical_devices="$(logical_device_spec "$physical_devices")" || return 64
  prepare_stage1_record "$source_root" "$case_dir" || return 65
  mkdir -p "$persistent_root" || return 66
  [ -f "$source_launcher" ] || {
    printf '[gate][FATAL] launcher not found: %s\n' "$source_launcher" >&2
    return 68
  }

  # The 48b03e8 baseline predates these flags and gets its F1/F4 batch 64
  # behavior from the preset/evaluator defaults. Optimized cases set both.
  if [ "$case_name" != "base64" ]; then
    probe_batch_args=(
      --blb-v3-probe-batch-size "$probe_batch_size"
      --blb-v3-validation-probe-batch-size "$probe_batch_size"
    )
  fi

  cd "$source_root" || return 67
  export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${source_root}:${source_root}/Rescale_optimizer"
  export HF_HOME="${HF_HOME:-/hy-tmp/hf_cache}"
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
  export GLUE_LOCAL_DATASET_DIR="${GLUE_LOCAL_DATASET_DIR:-/hy-tmp/glue_data}"
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1
  export BLB_STAGE2_PROBE_BACKEND=process
  export BLB_STAGE2_PROBE_INTRAOP_THREADS=1
  export BLB_STAGE2_PROBE_INTEROP_THREADS=1
  export BLB_STAGE2_POLICY_DEVICE="${BLB_STAGE2_POLICY_DEVICE:-worker}"
  export BLB_STAGE2_DYNAMIC_ASSIGNMENT="${BLB_STAGE2_DYNAMIC_ASSIGNMENT:-1}"
  export ALLOW_SHORT_RL_BENCHMARK=1
  export CUDA_VISIBLE_DEVICES="$physical_devices"
  export STAGE2_GATE_TRAIN_EXIT_FILE="$training_exit_file"

  exec setsid bash -c '
    launcher_path="$1"
    shift
    source "$launcher_path"
    set +e
    wait "$JOB_PID"
    stage2_gate_train_rc=$?
    printf "%s\n" "$stage2_gate_train_rc" > "$STAGE2_GATE_TRAIN_EXIT_FILE"
    exit "$stage2_gate_train_rc"
  ' "$source_launcher" "$source_launcher" run rl \
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
        "${probe_batch_args[@]}" \
        > "$case_dir/launch.log" 2>&1
}

sample_gpu_usage() {
  local out_file="$1"
  local compute_file="$2"
  local sample_timestamp compute_output compute_row
  printf 'timestamp,index,name,memory_used_mib,utilization_gpu_pct\n' > "$out_file"
  printf 'sample_timestamp,pid,gpu_uuid,used_gpu_memory_mib\n' > "$compute_file"
  while true; do
    nvidia-smi --id="$REWARD_DEVICES" \
      --query-gpu=timestamp,index,name,memory.used,utilization.gpu \
      --format=csv,noheader,nounits >> "$out_file" 2>/dev/null || true
    sample_timestamp="$(date '+%Y-%m-%dT%H:%M:%S')"
    compute_output="$(
      nvidia-smi --id="$REWARD_DEVICES" \
        --query-compute-apps=pid,gpu_uuid,used_gpu_memory \
        --format=csv,noheader,nounits 2>/dev/null || true
    )"
    while IFS= read -r compute_row; do
      [ -n "$compute_row" ] || continue
      case "$compute_row" in
        *"No running processes found"*) continue ;;
      esac
      printf '%s,%s\n' "$sample_timestamp" "$compute_row" >> "$compute_file"
    done <<< "$compute_output"
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
    diagnostics_summary.json probe_pool_topology.json; do
    destination="$case_dir/diagnostics/$filename"
    [ -f "$destination" ] && continue
    source="$(find_latest_case_file "$case_dir" "$filename")"
    [ -n "$source" ] || continue
    cp "$source" "$destination" || return 1
  done
}

collect_case_training_data_points() {
  local source_root="$1"
  local case_dir="$2"
  local relative source destination
  local moved_manifest="$case_dir/rl_training_data_points_files.txt"
  : > "$moved_manifest"
  while IFS= read -r -d '' relative; do
    case "$relative" in
      rl_training_data_points/*) ;;
      *)
        printf '[gate][FATAL] unexpected structured-data path: %s\n' \
          "$relative" >&2
        return 1
        ;;
    esac
    source="$source_root/$relative"
    destination="$case_dir/$relative"
    [ -f "$source" ] || {
      printf '[gate][FATAL] structured-data file disappeared: %s\n' \
        "$source" >&2
      return 1
    }
    mkdir -p "$(dirname "$destination")" || return 1
    mv -- "$source" "$destination" || return 1
    printf '%s\n' "$relative" >> "$moved_manifest"
  done < <(
    git -C "$source_root" ls-files --others --exclude-standard -z -- \
      rl_training_data_points
  )
  if [ -d "$source_root/rl_training_data_points" ]; then
    find "$source_root/rl_training_data_points" -depth -type d -empty \
      -delete 2>/dev/null || return 1
  fi
}

normalize_candidate_evidence() {
  local input_path="$1"
  local output_path="$2"
  "$STAGE2_GATE_PYTHON" - \
    "$input_path" "$output_path" "${SCRIPT_DIR}/.." <<'PY'
import hashlib
import json
import pathlib
import shutil
import sys

source = pathlib.Path(sys.argv[1])
destination = pathlib.Path(sys.argv[2])
repo_root = pathlib.Path(sys.argv[3])
sys.path.insert(0, str(repo_root))

from blb_stage2_rl.candidate_store import (
    CandidateStore,
    action_hash,
    candidate_key,
    sha256_json,
)

validation_copy = destination.with_name(destination.name + ".validation-copy.tmp")


def file_signature(path):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


original_signature = file_signature(source)
try:
    validation_copy.unlink()
except FileNotFoundError:
    pass
shutil.copyfile(source, validation_copy)
if file_signature(validation_copy) != original_signature:
    raise ValueError("candidate evidence changed while creating validation copy")

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

def canonical_identity(row):
    context = identity_context(row)
    action = [int(value) for value in row["action_indices"]]
    raw_action = [
        int(value) for value in row.get("raw_action_indices", action)
    ]
    effective_action = [
        int(value) for value in row.get("effective_action_indices", action)
    ]
    expected_raw_hash = action_hash(raw_action)
    expected_effective_hash = action_hash(effective_action)
    for field in ("raw_action_hash", "action_hash", "action_vector_hash"):
        if str(row.get(field, "")) != expected_raw_hash:
            raise ValueError(f"candidate {field} mismatch")
    if str(row.get("effective_action_hash", "")) != expected_effective_hash:
        raise ValueError("candidate effective_action_hash mismatch")
    expected_basis = "effective_action_hash + identity_context"
    if str(row.get("candidate_key_basis", "")) != expected_basis:
        raise ValueError("candidate_key_basis mismatch")
    expected_context_hash = sha256_json(context)
    if str(row.get("identity_context_hash", "")) != expected_context_hash:
        raise ValueError("candidate identity_context_hash mismatch")
    expected_key = candidate_key(
        action,
        context,
        effective_action_indices=effective_action,
        effective_action_hash_value=expected_effective_hash,
    )
    if str(row.get("candidate_key", "")) != expected_key:
        raise ValueError("candidate_key mismatch")
    return (
        context,
        effective_action,
        expected_effective_hash,
        expected_basis,
        expected_key,
        expected_context_hash,
    )


normalized = []
try:
    rows = CandidateStore(validation_copy).iter_active_records()
    for row in rows:
        record_type = str(row.get("record_type", ""))
        if record_type in {
            "candidate_identity_context_v1",
            "candidate_store_recovery_v1",
        }:
            continue
        payload = {
            str(key): value
            for key, value in row.items()
            if key not in storage_fields and key != "record_type"
        }
        if record_type in trial_types or record_type in promotion_types:
            (
                context,
                effective_action,
                effective_hash,
                key_basis,
                key,
                context_hash,
            ) = canonical_identity(row)
            payload["action_indices"] = effective_action
            payload["effective_action_hash"] = effective_hash
            payload["candidate_key_basis"] = key_basis
            payload["candidate_key"] = key
            payload["identity_context_hash"] = context_hash
            payload["identity_context"] = context
        if record_type in trial_types:
            payload["record_type"] = "candidate_trial_group"
            metadata = row.get("trial_group_metadata")
            metadata = dict(metadata) if isinstance(metadata, dict) else {}
            metadata.pop("identity_context", None)
            if str(row.get("fidelity", "")).upper() == "F1":
                metadata.pop("boosted_overrides", None)
            payload["trial_group_metadata"] = metadata
        elif record_type in promotion_types:
            payload["record_type"] = "candidate_promotion_status"
            metadata = row.get("promotion_metadata")
            payload["promotion_metadata"] = (
                dict(metadata) if isinstance(metadata, dict) else {}
            )
        else:
            payload["record_type"] = record_type
        normalized.append(payload)

    if file_signature(validation_copy) != original_signature:
        raise ValueError("candidate evidence requires tail repair")
    if file_signature(source) != original_signature:
        raise ValueError("candidate evidence changed during validation")
finally:
    try:
        validation_copy.unlink()
    except FileNotFoundError:
        pass

encoded = [json.dumps(row, sort_keys=True, separators=(",", ":")) for row in normalized]
with destination.open("w", encoding="utf-8") as handle:
    for line in encoded:
        handle.write(line + "\n")
PY
}

validate_gpu_evidence() {
  local gpu_json="$1"
  local episodes_jsonl="$2"
  local compute_csv="$3"
  local worker_inventory="$4"
  local topology_json="$5"
  local require_pool_telemetry="$6"
  local logical_devices
  logical_devices="$(logical_device_spec "$REWARD_DEVICES")" || return 1
  "$STAGE2_GATE_PYTHON" - \
    "$gpu_json" "$episodes_jsonl" "$compute_csv" "$worker_inventory" \
    "$topology_json" "$REWARD_DEVICES" "$logical_devices" \
    "$require_pool_telemetry" \
    "$STAGE2_GATE_MIN_GPU_ACTIVE_SAMPLES" \
    "$STAGE2_GATE_MIN_GPU_ACTIVE_SAMPLE_RATE" \
    "$STAGE2_GATE_MIN_GPU_MAX_UTIL_PCT" \
    "$STAGE2_GATE_MIN_PROBE_EPISODE_COVERAGE" \
    "$STAGE2_GATE_MIN_PROBE_TRIAL_BALANCE" <<'PY'
import csv
import json
import math
import pathlib
import re
import sys

summary = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
episodes_path = pathlib.Path(sys.argv[2])
compute_path = pathlib.Path(sys.argv[3])
inventory_path = pathlib.Path(sys.argv[4])
topology_path = pathlib.Path(sys.argv[5])
physical_devices = {f"cuda:{int(value)}" for value in sys.argv[6].split(",")}
logical_devices = {f"cuda:{int(value)}" for value in sys.argv[7].split(",")}
require_pool_telemetry = sys.argv[8] == "1"
minimum_active_samples = int(sys.argv[9])
minimum_active_rate = float(sys.argv[10])
minimum_max_util = float(sys.argv[11])
minimum_episode_coverage = float(sys.argv[12])
minimum_trial_balance = float(sys.argv[13])
gpu_utilization = summary.get("gpu_utilization") or {}
errors = []

for device in sorted(physical_devices):
    info = gpu_utilization.get(device)
    if not isinstance(info, dict):
        errors.append(f"{device}: no nvidia-smi samples")
        continue
    samples = int(info.get("samples", 0) or 0)
    active_rate = float(info.get("active_sample_rate", 0.0) or 0.0)
    max_util = float(info.get("max_util_pct", 0.0) or 0.0)
    if samples < minimum_active_samples:
        errors.append(
            f"{device}: samples={samples} < {minimum_active_samples}"
        )
    if active_rate * samples + 1.0e-9 < minimum_active_samples:
        errors.append(
            f"{device}: active_samples={active_rate * samples:.3f} "
            f"< {minimum_active_samples}"
        )
    if active_rate < minimum_active_rate:
        errors.append(
            f"{device}: active_sample_rate={active_rate:.6f} "
            f"< {minimum_active_rate:.6f}"
        )
    if max_util < minimum_max_util:
        errors.append(
            f"{device}: max_util_pct={max_util:.3f} < {minimum_max_util:.3f}"
        )

used_probe_devices = set(summary.get("used_probe_devices") or ())
if used_probe_devices != logical_devices:
    errors.append(
        "logical probe devices mismatch: "
        f"observed={sorted(used_probe_devices)} expected={sorted(logical_devices)}"
    )
episode_total = int(summary.get("episodes", 0) or 0)
minimum_covered_episodes = int(math.ceil(episode_total * minimum_episode_coverage))
episode_counts = summary.get("probe_episode_counts_by_device") or {}
trial_counts = summary.get("probe_trial_counts_by_device") or {}
for device in sorted(logical_devices):
    episode_count = int(episode_counts.get(device, 0) or 0)
    trial_count = int(trial_counts.get(device, 0) or 0)
    if episode_count < minimum_covered_episodes:
        errors.append(
            f"probe_episode_counts_by_device[{device}]={episode_count} "
            f"< {minimum_covered_episodes}"
        )
    if trial_count < minimum_covered_episodes:
        errors.append(
            f"probe_trial_counts_by_device[{device}]={trial_count} "
            f"< {minimum_covered_episodes}"
        )
logical_trial_counts = [
    int(trial_counts.get(device, 0) or 0) for device in logical_devices
]
if logical_trial_counts and max(logical_trial_counts) > 0:
    observed_balance = min(logical_trial_counts) / float(max(logical_trial_counts))
    if observed_balance < minimum_trial_balance:
        errors.append(
            f"probe trial balance={observed_balance:.6f} "
            f"< {minimum_trial_balance:.6f}"
        )

episode_rows = []
with episodes_path.open(encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            errors.append(f"episode row {line_number} is not an object")
            continue
        episode_rows.append(row)
if require_pool_telemetry:
    expected_process_count = max(0, len(logical_devices) - 1)
    pool_ids = set()
    for row_number, row in enumerate(episode_rows, 1):
        pool_id = str(row.get("pool_id", ""))
        if not pool_id:
            errors.append(f"episode row {row_number} has no pool_id")
        else:
            pool_ids.add(pool_id)
        if str(row.get("batch_set_key", "")) not in {"F1", "F4"}:
            errors.append(f"episode row {row_number} has invalid batch_set_key")
        if int(row.get("batch_count", 0) or 0) <= 0:
            errors.append(f"episode row {row_number} has nonpositive batch_count")
        if int(row.get("process_count", -1)) != expected_process_count:
            errors.append(
                f"episode row {row_number} process_count="
                f"{row.get('process_count')} expected={expected_process_count}"
            )
        if int(row.get("worker_intraop_threads", 0) or 0) != 1:
            errors.append(f"episode row {row_number} worker_intraop_threads != 1")
        if int(row.get("worker_interop_threads", 0) or 0) != 1:
            errors.append(f"episode row {row_number} worker_interop_threads != 1")
    if len(pool_ids) != 1:
        errors.append(f"expected one shared probe pool_id, observed={sorted(pool_ids)}")

inventory_text = inventory_path.read_text(encoding="utf-8", errors="replace")
inventory_pids = {
    int(match.group(1))
    for match in re.finditer(
        r"\bpid=([0-9]+)\b",
        inventory_text,
    )
}
inventory_thread_counts = {}
for match in re.finditer(
        r"\bpid=([0-9]+)\s+thread_count=([0-9]+)\b",
        inventory_text,
):
    pid = int(match.group(1))
    inventory_thread_counts.setdefault(pid, set()).add(int(match.group(2)))
compute_pids = set()
with compute_path.open(newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        raw_pid = str(row.get("pid", "")).strip()
        if not raw_pid:
            continue
        try:
            compute_pids.add(int(raw_pid))
        except ValueError:
            errors.append(f"invalid sampled compute PID: {raw_pid!r}")
if not compute_pids:
    errors.append("no runtime compute PID was sampled")
unowned_compute_pids = sorted(compute_pids - inventory_pids)
if unowned_compute_pids:
    errors.append(f"sampled compute PIDs outside owned process tree: {unowned_compute_pids}")

if require_pool_telemetry:
    topology = json.loads(topology_path.read_text(encoding="utf-8"))
    if topology.get("schema_version") != "probe_pool_topology_v1":
        errors.append("invalid probe-pool topology schema")
    topology_pool_id = str(topology.get("pool_id", ""))
    if not topology_pool_id or pool_ids != {topology_pool_id}:
        errors.append(
            "probe-pool topology pool_id mismatch: "
            f"topology={topology_pool_id!r} episodes={sorted(pool_ids)}"
        )
    if str(topology.get("backend", "")) != "process":
        errors.append("probe-pool topology backend is not process")
    topology_devices = set(topology.get("devices") or ())
    if topology_devices != logical_devices:
        errors.append(
            "probe-pool topology devices mismatch: "
            f"observed={sorted(topology_devices)} "
            f"expected={sorted(logical_devices)}"
        )
    expected_process_count = max(0, len(logical_devices) - 1)
    if int(topology.get("process_count", -1)) != expected_process_count:
        errors.append(
            "probe-pool topology process_count="
            f"{topology.get('process_count')} expected={expected_process_count}"
        )

    raw_worker_pids = topology.get("worker_pids") or []
    try:
        worker_pids = [int(value) for value in raw_worker_pids]
        primary_pid = int(topology.get("primary_pid", 0) or 0)
    except (TypeError, ValueError):
        worker_pids = []
        primary_pid = 0
        errors.append("probe-pool topology contains invalid worker PIDs")
    if (
            len(worker_pids) != expected_process_count
            or len(set(worker_pids)) != expected_process_count
            or any(pid <= 0 for pid in worker_pids)
    ):
        errors.append(
            "probe-pool topology worker PID count/identity mismatch: "
            f"observed={worker_pids} expected_count={expected_process_count}"
        )
    if primary_pid <= 0 or primary_pid in worker_pids:
        errors.append(f"probe-pool topology primary_pid is invalid: {primary_pid}")
    topology_pids = (
        {primary_pid, *worker_pids} if primary_pid > 0 else set(worker_pids)
    )
    missing_inventory_pids = sorted(topology_pids - inventory_pids)
    if missing_inventory_pids:
        errors.append(
            "probe-pool PIDs absent from owned process inventory: "
            f"{missing_inventory_pids}"
        )
    missing_thread_inventory = sorted(
        topology_pids - set(inventory_thread_counts)
    )
    if missing_thread_inventory:
        errors.append(
            "probe-pool PIDs lack numeric thread inventory: "
            f"{missing_thread_inventory}"
        )
    nonpositive_thread_inventory = sorted(
        pid for pid in topology_pids
        if pid in inventory_thread_counts
        and not any(value > 0 for value in inventory_thread_counts[pid])
    )
    if nonpositive_thread_inventory:
        errors.append(
            "probe-pool PIDs have nonpositive thread inventory: "
            f"{nonpositive_thread_inventory}"
        )
    missing_compute_pids = sorted(topology_pids - compute_pids)
    if missing_compute_pids:
        errors.append(
            "probe-pool PIDs were never observed as GPU compute processes: "
            f"{missing_compute_pids}"
        )

    expected_worker_total = len(logical_devices)
    for field_name in ("worker_intraop_threads", "worker_interop_threads"):
        try:
            values = [int(value) for value in topology.get(field_name, [])]
        except (TypeError, ValueError):
            values = []
        if (
                len(values) != expected_worker_total
                or any(value != 1 for value in values)
        ):
            errors.append(
                f"probe-pool {field_name}={values} "
                f"expected={[1] * expected_worker_total}"
            )

    batch_sets = topology.get("batch_sets") or {}
    call_counts = topology.get("call_counts_by_batch_set") or {}
    trial_counts = topology.get("trial_counts_by_batch_set") or {}
    for batch_set_key in ("F1", "F4"):
        batch_info = batch_sets.get(batch_set_key) or {}
        if int(batch_info.get("batch_count", 0) or 0) <= 0:
            errors.append(f"probe-pool {batch_set_key} batch_count is nonpositive")
        if int(call_counts.get(batch_set_key, 0) or 0) <= 0:
            errors.append(f"probe-pool {batch_set_key} call_count is nonpositive")
        if int(trial_counts.get(batch_set_key, 0) or 0) <= 0:
            errors.append(f"probe-pool {batch_set_key} trial_count is nonpositive")

if errors:
    for error in errors:
        print(f"[gate][FATAL] {error}", file=sys.stderr)
    raise SystemExit(2)
PY
}

validate_case_evidence() {
  local case_dir="$1"
  local expected_episodes="$2"
  local require_pool_telemetry="$3"
  local missing=""
  local episode_count
  local wall
  local candidate="$case_dir/diagnostics/candidate_store.jsonl"
  local normalized="$case_dir/diagnostics/candidate_store.normalized.jsonl"
  local gpu_json="$case_dir/gpu_utilization.json"
  local gpu_markdown="$case_dir/gpu_utilization.md"
  local topology="$case_dir/diagnostics/probe_pool_topology.json"
  local filename
  for filename in episodes.jsonl ppo_updates.jsonl candidate_store.jsonl; do
    if [ ! -s "$case_dir/diagnostics/$filename" ]; then
      missing="${missing} ${filename}"
    fi
  done
  if [ ! -s "$case_dir/wall_seconds.txt" ]; then
    missing="${missing} wall_seconds.txt"
  fi
  for filename in nvidia_compute_samples.csv worker_thread_inventory.txt; do
    if [ ! -s "$case_dir/$filename" ]; then
      missing="${missing} ${filename}"
    fi
  done
  if [ "$require_pool_telemetry" = "1" ] && [ ! -s "$topology" ]; then
    missing="${missing} probe_pool_topology.json"
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
  if ! validate_gpu_evidence \
      "$gpu_json" \
      "$case_dir/diagnostics/episodes.jsonl" \
      "$case_dir/nvidia_compute_samples.csv" \
      "$case_dir/worker_thread_inventory.txt" \
      "$topology" \
      "$require_pool_telemetry" \
      >> "$case_dir/gpu_activity_validation.log" 2>&1; then
    printf 'FAIL sustained GPU or probe-device coverage\n' \
      > "$case_dir/gpu_activity_status.txt"
    printf 'FAIL sustained GPU or probe-device coverage\n' \
      > "$case_dir/evidence_status.txt"
    return 1
  fi
  printf 'PASS\n' > "$case_dir/gpu_activity_status.txt"
  printf 'PASS\n' > "$case_dir/evidence_status.txt"
}

LAST_LAUNCH_RC=0
LAST_EVIDENCE_PASS=0
ACTIVE_CASE_DIR=""
ACTIVE_SAMPLER_PID=""
ACTIVE_GATE_LAUNCHER_PID=""

cleanup_active_case() {
  local pgid=""
  local child_pid
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
  if [ -n "$ACTIVE_GATE_LAUNCHER_PID" ]; then
    terminate_owned_pid_or_group "$ACTIVE_GATE_LAUNCHER_PID"
  else
    while IFS= read -r child_pid; do
      [ "$child_pid" = "$ACTIVE_SAMPLER_PID" ] && continue
      terminate_owned_pid_or_group "$child_pid"
    done < <(jobs -pr)
  fi
  ACTIVE_GATE_LAUNCHER_PID=""
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
  local compute_samples="$case_dir/nvidia_compute_samples.csv"
  local launcher_log="$case_dir/gate_launcher.log"
  local inventory="$case_dir/worker_thread_inventory.txt"
  local training_exit_file="$case_dir/training_exit_code.txt"
  local start_s end_s measured_wall
  local sampler_pid launcher_pid launch_rc training_rc require_pool_telemetry=1

  mkdir -p "$case_dir/diagnostics"
  ACTIVE_CASE_DIR="$case_dir"
  verify_requested_gpus_idle "${case_name}:before" || exit $?
  : > "$inventory"
  start_s="$(date +%s)"
  sample_gpu_usage "$gpu_samples" "$compute_samples" &
  sampler_pid=$!
  ACTIVE_SAMPLER_PID="$sampler_pid"

  if [ -n "${STAGE2_GATE_CASE_LAUNCHER:-}" ]; then
    setsid "$STAGE2_GATE_CASE_LAUNCHER" \
      "$case_name" "$source_root" "$batch_size" "$EPISODES" \
      "$REWARD_DEVICES" "$case_dir" > "$launcher_log" 2>&1 &
  else
    default_case_launcher \
      "$case_name" "$source_root" "$batch_size" "$EPISODES" \
      "$REWARD_DEVICES" "$case_dir" > "$launcher_log" 2>&1 &
  fi
  launcher_pid=$!
  ACTIVE_GATE_LAUNCHER_PID="$launcher_pid"
  printf '%s\n' "$launcher_pid" > "$case_dir/gate_launcher_pid.txt"
  printf '%s\n' "$launcher_pid" > "$case_dir/training_process_group.txt"

  sample_worker_inventory "$case_dir" "$launcher_pid"
  wait_for_owned_launcher \
    "$launcher_pid" "$STAGE2_GATE_CASE_TIMEOUT_SECONDS" "$case_dir"
  launch_rc=$?
  ACTIVE_GATE_LAUNCHER_PID=""

  if [ -z "${STAGE2_GATE_CASE_LAUNCHER:-}" ]; then
    if [ -s "$training_exit_file" ]; then
      training_rc="$(tr -d '[:space:]' < "$training_exit_file")"
      case "$training_rc" in
        ''|*[!0-9]*)
          printf '[gate][FATAL] %s wrote invalid training exit code: %s\n' \
            "$case_name" "$training_rc" >&2
          launch_rc=125
          ;;
        *)
          if [ "$launch_rc" -eq 124 ]; then
            launch_rc=124
          elif [ "$training_rc" -ne "$launch_rc" ]; then
            printf '[gate][FATAL] %s launcher/training exit mismatch: %s != %s\n' \
              "$case_name" "$launch_rc" "$training_rc" >&2
            launch_rc=125
          fi
          ;;
      esac
    else
      printf '%s\n' "$launch_rc" > "$training_exit_file"
    fi
  fi

  kill "$sampler_pid" 2>/dev/null || true
  wait "$sampler_pid" 2>/dev/null || true
  ACTIVE_SAMPLER_PID=""
  verify_requested_gpus_idle "${case_name}:after" || exit $?
  collect_case_training_data_points "$source_root" "$case_dir" || exit $?
  end_s="$(date +%s)"
  measured_wall=$((end_s - start_s))
  if [ ! -s "$case_dir/wall_seconds.txt" ]; then
    printf '%s\n' "$measured_wall" > "$case_dir/wall_seconds.txt"
  fi
  printf '%s\n' "$launch_rc" > "$case_dir/launcher_exit_code.txt"

  copy_case_evidence "$case_dir" \
    > "$case_dir/evidence_copy.log" 2>&1 || true
  [ "$case_name" != "base64" ] || require_pool_telemetry=0
  if validate_case_evidence \
      "$case_dir" "$EPISODES" "$require_pool_telemetry"; then
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
  verify_all_source_states || exit $?
  printf '[gate] running %s from %s (batch=%s)\n' \
    "$case_name" "$case_root" "$case_batch"
  run_case "$case_name" "$case_root" "$case_batch" "$case_dir"
  verify_all_source_states || exit $?
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
    PYTHONPATH="${SCRIPT_DIR}/..${PYTHONPATH:+:${PYTHONPATH}}" \
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
