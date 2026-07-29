#!/usr/bin/env bash
set -euo pipefail

# Real Stage-2 layerwise 1-GPU vs N-GPU throughput gate.
#
# This script intentionally compares full end-to-end short runs, not isolated
# policy-forward timings. N-GPU parallelism mirrors production by splitting the
# deterministic K reward trials across --blb-v3-reward-devices; the layerwise
# runner rejects the legacy --stage2-rl-devices episode-parallel path. The
# verdict is strict on episode/PPO equality and uses wrapper wall time for speed.
# Run only when no long training job owns the GPUs.

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}.:Rescale_optimizer"
export HF_HOME="${HF_HOME:-/hy-tmp/hf_cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export GLUE_LOCAL_DATASET_DIR="${GLUE_LOCAL_DATASET_DIR:-/hy-tmp/glue_data}"

# Keep PyTorch CPU helper pools from multiplying across rollout workers. This
# does not change RL sampling/probe seeds; the comparator still verifies output
# equality before accepting any speed result.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${RUN_ID:-stage2_ngpu_speed_ab_${TS}}"
ARTIFACT_DIR="${ARTIFACT_DIR:-experiments/server_command_runs/${RUN_ID}}"
RUN_STAGE2="${RUN_STAGE2:-${ARTIFACT_DIR}/stage2}"
STAGE1_RECORD_SOURCE="${STAGE1_RECORD_SOURCE:-Parting Chapter/stage1/record}"
MODEL_TYPE="${MODEL_TYPE:-bert-base}"
MODEL_DIR_LABEL="${MODEL_TYPE//-/ }"
EPISODES_AB="${EPISODES_AB:-600}"
KTRIALS="${KTRIALS:-3}"
PROBE_SIZE="${PROBE_SIZE:-256}"
ROLLOUT_SIZE="${ROLLOUT_SIZE:-120}"
PPO_UPDATE_INTERVAL="${PPO_UPDATE_INTERVAL:-120}"
BATCH_SIZE="${BATCH_SIZE:-64}"
STAGE2_SEARCH_LR="${STAGE2_SEARCH_LR:-5e-5}"
STAGE1_ACCURACY_TOLERANCE="${STAGE1_ACCURACY_TOLERANCE:-0.001}"
STAGE2_LIMIT_TOLERANCE="${STAGE2_LIMIT_TOLERANCE:-0.001}"
STAGE2_STABILITY_TOLERANCE="${STAGE2_STABILITY_TOLERANCE:-1.2}"
STAGE2_STABILITY_MULTIPLIER="${STAGE2_STABILITY_MULTIPLIER:-2.0}"
ONLINE_KTRIALS="${ONLINE_KTRIALS:-3}"
SAVE_INTERVAL="${SAVE_INTERVAL:-200}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
CALIBRATE_BASELINE_SAMPLES="${CALIBRATE_BASELINE_SAMPLES:-8}"
RANDOM_SEED="${RANDOM_SEED:-42}"
ONE_DEVS="${ONE_DEVS:-0}"
MANY_DEVS="${MANY_DEVS:-0,1,2,3,4}"
ONE_WORKERS_PER_DEVICE="${ONE_WORKERS_PER_DEVICE:-1}"
MANY_WORKERS_PER_DEVICE="${MANY_WORKERS_PER_DEVICE:-1}"
POLICY_DEVICE="${POLICY_DEVICE:-worker}"
DYNAMIC_ASSIGNMENT="${DYNAMIC_ASSIGNMENT:-1}"
MIN_SPEEDUP="${MIN_SPEEDUP:-3.4}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-14400}"
GPU_SAMPLE_INTERVAL_SECONDS="${GPU_SAMPLE_INTERVAL_SECONDS:-2}"
REQUIRE_IDLE_GPUS="${REQUIRE_IDLE_GPUS:-1}"
IDLE_MEM_MIB="${IDLE_MEM_MIB:-2048}"
CANON_STAGE2="${CANON_STAGE2:-Parting Chapter/stage2}"
REUSE_ONE_EPISODES="${REUSE_ONE_EPISODES:-}"
REUSE_ONE_WALL="${REUSE_ONE_WALL:-}"
REUSE_ONE_LOG="${REUSE_ONE_LOG:-}"
REUSE_ONE_PPO="${REUSE_ONE_PPO:-}"
PRINT_EFFECTIVE_COMMANDS="${PRINT_EFFECTIVE_COMMANDS:-0}"

mkdir -p "$ARTIFACT_DIR"
exec > >(tee "${ARTIFACT_DIR}/stage2_ngpu_speed_ab_stdout.log") 2>&1

echo "[ab] artifact_dir=${ARTIFACT_DIR}"
echo "[ab] model_type=${MODEL_TYPE}"
echo "[ab] episodes=${EPISODES_AB} rollout_size=${ROLLOUT_SIZE} ppo_update=${PPO_UPDATE_INTERVAL}"
echo "[ab] one=${ONE_DEVS} wpd=${ONE_WORKERS_PER_DEVICE}; many=${MANY_DEVS} wpd=${MANY_WORKERS_PER_DEVICE}"
echo "[ab] parallelism=layerwise reward-device K-split"
echo "[ab] cpu_threads OMP=${OMP_NUM_THREADS} MKL=${MKL_NUM_THREADS} OPENBLAS=${OPENBLAS_NUM_THREADS}"

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git rev-parse HEAD > "${ARTIFACT_DIR}/HEAD.txt"
  git status --short > "${ARTIFACT_DIR}/git_status_short.txt" || true
fi

logical_device_spec() {
  local visible_devices="$1"
  local -a visible_ids=()
  local token logical=""
  IFS=',' read -r -a visible_ids <<< "$visible_devices"
  if [ "${#visible_ids[@]}" -eq 0 ]; then
    echo "[FATAL] empty CUDA device list" >&2
    return 2
  fi
  for token in "${visible_ids[@]}"; do
    case "$token" in
      ''|*[!0-9]*)
        echo "[FATAL] invalid CUDA device list: ${visible_devices}" >&2
        return 2
        ;;
    esac
  done
  for ((token = 0; token < ${#visible_ids[@]}; token++)); do
    if [ -n "$logical" ]; then
      logical+=","
    fi
    logical+="$token"
  done
  printf '%s\n' "$logical"
}

build_case_command() {
  local label="$1"
  local visible_devices="$2"
  local workers_per_device="$3"
  local persistent_root="${ARTIFACT_DIR}/persistent_${label}"
  local reward_devices
  reward_devices="$(logical_device_spec "$visible_devices")"
  CASE_ENV=(
    env
    "CUDA_VISIBLE_DEVICES=${visible_devices}"
    "BLB_STAGE2_POLICY_DEVICE=${POLICY_DEVICE}"
    "BLB_STAGE2_DYNAMIC_ASSIGNMENT=${DYNAMIC_ASSIGNMENT}"
    timeout "$TIMEOUT_SECONDS"
  )
  CASE_COMMAND=(
    bash llama_7B_LayerImportance.sh run rl
    --preset mrpc-blb-stage2-rl
    --model-type "$MODEL_TYPE"
    --persistent-root "$persistent_root"
    --stage2-fixed-config-source all4
    --blb-v3-fusion-count-action 1
    --blb-v3-sequential-rl true
    --blb-v3-substage-mode false
    --blb-v3-decision-granularity layer
    --blb-v3-reward-design robust_constrained
    --blb-v3-sequential-invalid-penalty 1.0
    --blb-v3-sequential-cost-shaping-coeff 0.0
    --blb-v3-sequential-fusion-shaping-coeff 0.0
    --stage2-search-episodes "$EPISODES_AB"
    --stage2-search-lr "$STAGE2_SEARCH_LR"
    --ppo-update-interval "$PPO_UPDATE_INTERVAL"
    --stage2-rollout-size "$ROLLOUT_SIZE"
    --stage2-k-trials "$KTRIALS"
    --blb-v3-online-k-trials "$ONLINE_KTRIALS"
    --blb-v3-terminal-eval-batch-size 4
    --stage2-probe-size "$PROBE_SIZE"
    --batch-size "$BATCH_SIZE"
    --stage1-accuracy-tolerance "$STAGE1_ACCURACY_TOLERANCE"
    --stage2-limit-tolerance "$STAGE2_LIMIT_TOLERANCE"
    --stage2-stability-tolerance "$STAGE2_STABILITY_TOLERANCE"
    --stage2-stability-multiplier "$STAGE2_STABILITY_MULTIPLIER"
    --stage2-calibrate-baseline-samples "$CALIBRATE_BASELINE_SAMPLES"
    --blb-v3-promotion-validation-trials 15
    --blb-v3-final-selection-validation-trials 15
    --blb-v3-baseline-groups 5
    --blb-v3-baseline-trials-per-group 3
    --blb-v3-constraint-bootstrap-samples 4096
    --blb-v3-online-constraint-probability 0.50
    --blb-v3-promotion-constraint-probability 0.80
    --blb-v3-final-constraint-probability 0.95
    --blb-v3-reward-devices "$reward_devices"
    --stage2-workers-per-device "$workers_per_device"
    --stage2-save-interval "$SAVE_INTERVAL"
    --stage2-eval-interval "$EVAL_INTERVAL"
    --random-seed "$RANDOM_SEED"
    --rl-algo ppo
    --skip-final-eval
    --fresh
  )
}

print_case_command() {
  local label="$1"
  local visible_devices="$2"
  local workers_per_device="$3"
  local arg
  build_case_command "$label" "$visible_devices" "$workers_per_device"
  printf '[ab][effective] %s' "$label"
  for arg in "${CASE_ENV[@]}"; do
    printf ' %q' "$arg"
  done
  for arg in "${CASE_COMMAND[@]}"; do
    printf ' %q' "$arg"
  done
  printf '\n'
}

{
  print_case_command one "$ONE_DEVS" "$ONE_WORKERS_PER_DEVICE"
  print_case_command many "$MANY_DEVS" "$MANY_WORKERS_PER_DEVICE"
} | tee "${ARTIFACT_DIR}/effective_commands.txt"

if [ "$PRINT_EFFECTIVE_COMMANDS" = "1" ]; then
  echo "[ab] effective command preflight complete; GPU inventory was not queried"
  exit 0
fi

nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv \
  | tee "${ARTIFACT_DIR}/gpu_inventory_pre.csv"

if [ "$REQUIRE_IDLE_GPUS" = "1" ]; then
  busy="$(
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | awk -F, -v lim="$IDLE_MEM_MIB" '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($2+0 > lim) print $1 ":" $2 "MiB"}'
  )"
  if [ -n "$busy" ]; then
    echo "[FATAL] GPUs are not idle enough for a clean A/B (limit ${IDLE_MEM_MIB}MiB):"
    printf '%s\n' "$busy"
    echo "[FATAL] Set REQUIRE_IDLE_GPUS=0 only for an intentionally contaminated diagnostic run."
    exit 20
  fi
fi

sample_gpu_usage() {
  local out_file="$1"
  printf 'timestamp,index,name,memory_used_mib,utilization_gpu_pct\n' > "$out_file"
  while true; do
    nvidia-smi \
      --query-gpu=timestamp,index,name,memory.used,utilization.gpu \
      --format=csv,noheader,nounits >> "$out_file" 2>/dev/null || true
    sleep "$GPU_SAMPLE_INTERVAL_SECONDS"
  done
}

wait_for_background_run() {
  local pid="$1"
  local timeout_seconds="$2"
  local elapsed=0
  while kill -0 "$pid" 2>/dev/null; do
    if [ "$elapsed" -ge "$timeout_seconds" ]; then
      echo "[ab] pid=${pid} exceeded ${timeout_seconds}s; sending SIGINT"
      kill -INT "$pid" 2>/dev/null || true
      sleep 10
      kill -TERM "$pid" 2>/dev/null || true
      return 124
    fi
    sleep 10
    elapsed=$((elapsed + 10))
  done
  return 0
}

background_pid_exists() {
  local pid="$1"
  if [ -z "$pid" ]; then
    return 1
  fi
  kill -0 "$pid" 2>/dev/null
}

prepare_stage1_record() {
  local target="${ARTIFACT_DIR}/stage1/record"
  if [ -e "$target" ]; then
    return 0
  fi
  if [ ! -e "$STAGE1_RECORD_SOURCE" ]; then
    echo "[ab][warning] Stage-1 record source not found: ${STAGE1_RECORD_SOURCE}"
    return 0
  fi
  local source_parent source_abs
  source_parent="$(cd "$(dirname "$STAGE1_RECORD_SOURCE")" && pwd -P)"
  source_abs="${source_parent}/$(basename "$STAGE1_RECORD_SOURCE")"
  mkdir -p "$(dirname "$target")"
  ln -s "$source_abs" "$target"
  echo "[ab] linked Stage-1 record: ${target} -> ${source_abs}"
}

find_episodes_jsonl() {
  local persistent_root="$1"
  local candidate=""
  candidate="$(find "$persistent_root" -path '*/diagnostics/episodes.jsonl' -type f 2>/dev/null \
    | sort | tail -1 || true)"
  if [ -n "$candidate" ]; then
    printf '%s\n' "$candidate"
    return 0
  fi
  local run_latest_dir=""
  run_latest_dir="$(cat "${RUN_STAGE2}/LATEST_RUN_DIR" 2>/dev/null || true)"
  if [ -n "$run_latest_dir" ] && [ -f "${run_latest_dir}/diagnostics/episodes.jsonl" ]; then
    printf '%s\n' "${run_latest_dir}/diagnostics/episodes.jsonl"
    return 0
  fi
  candidate="$(find "$RUN_STAGE2" -path '*/diagnostics/episodes.jsonl' -type f 2>/dev/null \
    | sort | tail -1 || true)"
  if [ -n "$candidate" ]; then
    printf '%s\n' "$candidate"
    return 0
  fi
  local latest_run_dir=""
  latest_run_dir="$(cat "${CANON_STAGE2}/LATEST_RUN_DIR" 2>/dev/null || true)"
  if [ -n "$latest_run_dir" ] && [ -f "${latest_run_dir}/diagnostics/episodes.jsonl" ]; then
    printf '%s\n' "${latest_run_dir}/diagnostics/episodes.jsonl"
    return 0
  fi
  candidate="$(find "$CANON_STAGE2" -path '*/diagnostics/episodes.jsonl' -type f 2>/dev/null \
    | sort | tail -1 || true)"
  if [ -n "$candidate" ]; then
    printf '%s\n' "$candidate"
    return 0
  fi
  return 1
}

find_ppo_updates_jsonl() {
  local persistent_root="$1"
  local candidate=""
  candidate="$(find "$persistent_root" -path '*/diagnostics/ppo_updates.jsonl' -type f 2>/dev/null \
    | sort | tail -1 || true)"
  if [ -n "$candidate" ]; then
    printf '%s\n' "$candidate"
    return 0
  fi
  local run_latest_dir=""
  run_latest_dir="$(cat "${RUN_STAGE2}/LATEST_RUN_DIR" 2>/dev/null || true)"
  if [ -n "$run_latest_dir" ] && [ -f "${run_latest_dir}/diagnostics/ppo_updates.jsonl" ]; then
    printf '%s\n' "${run_latest_dir}/diagnostics/ppo_updates.jsonl"
    return 0
  fi
  candidate="$(find "$RUN_STAGE2" -path '*/diagnostics/ppo_updates.jsonl' -type f 2>/dev/null \
    | sort | tail -1 || true)"
  if [ -n "$candidate" ]; then
    printf '%s\n' "$candidate"
    return 0
  fi
  local latest_run_dir=""
  latest_run_dir="$(cat "${CANON_STAGE2}/LATEST_RUN_DIR" 2>/dev/null || true)"
  if [ -n "$latest_run_dir" ] && [ -f "${latest_run_dir}/diagnostics/ppo_updates.jsonl" ]; then
    printf '%s\n' "${latest_run_dir}/diagnostics/ppo_updates.jsonl"
    return 0
  fi
  candidate="$(find "$CANON_STAGE2" -path '*/diagnostics/ppo_updates.jsonl' -type f 2>/dev/null \
    | sort | tail -1 || true)"
  if [ -n "$candidate" ]; then
    printf '%s\n' "$candidate"
    return 0
  fi
  return 1
}

run_case() {
  local label="$1"
  local devices="$2"
  local workers_per_device="$3"
  local persistent_root="${ARTIFACT_DIR}/persistent_${label}"
  local launch_log="${ARTIFACT_DIR}/${label}_launch.log"
  local gpu_sample_file="${ARTIFACT_DIR}/${label}_nvidia_smi.csv"
  local latest_pid_file="${persistent_root}/rl/${MODEL_TYPE}/mrpc/LATEST_PID"
  local wall_file="${ARTIFACT_DIR}/${label}_wall_seconds.txt"
  local episodes_out="${ARTIFACT_DIR}/${label}_episodes.jsonl"
  local ppo_out="${ARTIFACT_DIR}/${label}_ppo_updates.jsonl"

  echo ""
  echo "================================================================================"
  echo "[ab] ${label}: CUDA_VISIBLE_DEVICES=${devices} reward_devices=$(logical_device_spec "$devices")"
  echo "================================================================================"

  mkdir -p "$persistent_root"
  prepare_stage1_record
  sample_gpu_usage "$gpu_sample_file" &
  local sampler_pid=$!
  local start_s end_s launch_rc wait_rc rc job_pid ep_path ppo_path
  start_s="$(date +%s)"
  set +e
  build_case_command "$label" "$devices" "$workers_per_device"
  "${CASE_ENV[@]}" "${CASE_COMMAND[@]}" 2>&1 | tee "$launch_log"
  launch_rc=${PIPESTATUS[0]}
  rc=$launch_rc
  job_pid="$(cat "$latest_pid_file" 2>/dev/null || true)"
  if [ "$launch_rc" -eq 0 ] && ! background_pid_exists "$job_pid"; then
    local fallback_pid=""
    for pid_file in \
      "${RUN_STAGE2}/LATEST_PID" \
      "${RUN_STAGE2}/${MODEL_DIR_LABEL} mrpc/run.pid" \
      "${RUN_STAGE2}/${MODEL_DIR_LABEL} mrpc/rl.pid" \
      "${CANON_STAGE2}/LATEST_PID"; do
      fallback_pid="$(cat "$pid_file" 2>/dev/null || true)"
      if background_pid_exists "$fallback_pid"; then
        break
      fi
    done
    if background_pid_exists "$fallback_pid"; then
      job_pid="$fallback_pid"
      echo "[ab][warning] ${label} used fallback PID=${job_pid}; expected ${latest_pid_file}"
    else
      echo "[FATAL] ${label} could not identify its background PID under ${persistent_root}"
      echo "[FATAL] latest_pid_file=${latest_pid_file} pid=${job_pid:-<missing>} run_stage2=${RUN_STAGE2} fallback=${fallback_pid:-<missing>}"
      rc=125
    fi
  fi
  if [ "$launch_rc" -eq 0 ] && [ "$rc" -eq 0 ] && [ -n "$job_pid" ]; then
    echo "[ab] ${label} launched pid=${job_pid}; waiting for completion"
    wait_for_background_run "$job_pid" "$TIMEOUT_SECONDS"
    wait_rc=$?
    rc=$wait_rc
  fi
  end_s="$(date +%s)"
  kill "$sampler_pid" 2>/dev/null || true
  wait "$sampler_pid" 2>/dev/null || true
  set -e
  printf '%s\n' "$((end_s - start_s))" > "$wall_file"
  printf '{"label":"%s","devices":"%s","workers_per_device":%s,"launch_rc":%s,"rc":%s,"wall_seconds":%s}\n' \
    "$label" "$devices" "$workers_per_device" "$launch_rc" "$rc" "$((end_s - start_s))" \
    >> "${ARTIFACT_DIR}/runs.jsonl"
  if [ "$rc" -ne 0 ]; then
    echo "[FATAL] ${label} failed rc=${rc}; see ${launch_log}"
    exit "$rc"
  fi
  ep_path="$(find_episodes_jsonl "$persistent_root")"
  cp "$ep_path" "$episodes_out"
  local episode_lines
  episode_lines="$(wc -l < "$episodes_out")"
  echo "[ab] ${label} episodes -> ${episodes_out} (${episode_lines} lines), wall=$(cat "$wall_file")s"
  if [ "$episode_lines" -ne "$EPISODES_AB" ]; then
    echo "[FATAL] ${label} wrote ${episode_lines}/${EPISODES_AB} episodes; refusing to compare partial runs"
    exit 127
  fi
  python3 scripts/gpu_utilization_report.py \
    --episodes "$episodes_out" \
    --nvidia-smi-csv "$gpu_sample_file" \
    --visible-devices "$devices" \
    --out-json "${ARTIFACT_DIR}/${label}_gpu_utilization.json" \
    --out-md "${ARTIFACT_DIR}/${label}_gpu_utilization.md" \
    --require-all-visible-sampled-active
  if ppo_path="$(find_ppo_updates_jsonl "$persistent_root")"; then
    cp "$ppo_path" "$ppo_out"
    echo "[ab] ${label} PPO updates -> ${ppo_out} ($(wc -l < "$ppo_out") lines)"
  else
    echo "[ab][warning] ${label} PPO updates not found; equality gate will compare episodes only"
  fi
}

reuse_one_case() {
  if [ -z "$REUSE_ONE_EPISODES" ] || [ -z "$REUSE_ONE_WALL" ]; then
    echo "[FATAL] REUSE_ONE_EPISODES and REUSE_ONE_WALL must be set together"
    exit 2
  fi
  if [ ! -f "$REUSE_ONE_EPISODES" ]; then
    echo "[FATAL] REUSE_ONE_EPISODES not found: $REUSE_ONE_EPISODES"
    exit 2
  fi
  if [ ! -f "$REUSE_ONE_WALL" ]; then
    echo "[FATAL] REUSE_ONE_WALL not found: $REUSE_ONE_WALL"
    exit 2
  fi
  cp "$REUSE_ONE_EPISODES" "${ARTIFACT_DIR}/one_episodes.jsonl"
  cp "$REUSE_ONE_WALL" "${ARTIFACT_DIR}/one_wall_seconds.txt"
  local reuse_one_gpu_samples
  reuse_one_gpu_samples="$(dirname "$REUSE_ONE_EPISODES")/one_nvidia_smi.csv"
  if [ ! -f "$reuse_one_gpu_samples" ]; then
    echo "[FATAL] reused one baseline lacks GPU samples: ${reuse_one_gpu_samples}"
    exit 2
  fi
  cp "$reuse_one_gpu_samples" "${ARTIFACT_DIR}/one_nvidia_smi.csv"
  local episode_lines
  episode_lines="$(wc -l < "${ARTIFACT_DIR}/one_episodes.jsonl")"
  if [ "$episode_lines" -ne "$EPISODES_AB" ]; then
    echo "[FATAL] reused one baseline has ${episode_lines}/${EPISODES_AB} episodes; refusing to compare partial runs"
    exit 127
  fi
  python3 scripts/gpu_utilization_report.py \
    --episodes "${ARTIFACT_DIR}/one_episodes.jsonl" \
    --nvidia-smi-csv "${ARTIFACT_DIR}/one_nvidia_smi.csv" \
    --visible-devices "$ONE_DEVS" \
    --out-json "${ARTIFACT_DIR}/one_gpu_utilization.json" \
    --out-md "${ARTIFACT_DIR}/one_gpu_utilization.md" \
    --require-all-visible-sampled-active
  if [ -n "$REUSE_ONE_LOG" ] && [ -f "$REUSE_ONE_LOG" ]; then
    cp "$REUSE_ONE_LOG" "${ARTIFACT_DIR}/one_launch.log"
  else
    printf '[ab] one launch log reused; source log unavailable\n' > "${ARTIFACT_DIR}/one_launch.log"
  fi
  if [ -n "$REUSE_ONE_PPO" ] && [ -f "$REUSE_ONE_PPO" ]; then
    cp "$REUSE_ONE_PPO" "${ARTIFACT_DIR}/one_ppo_updates.jsonl"
  elif [ -f "$(dirname "$REUSE_ONE_EPISODES")/one_ppo_updates.jsonl" ]; then
    cp "$(dirname "$REUSE_ONE_EPISODES")/one_ppo_updates.jsonl" "${ARTIFACT_DIR}/one_ppo_updates.jsonl"
  elif [ -f "$(dirname "$REUSE_ONE_EPISODES")/ppo_updates.jsonl" ]; then
    cp "$(dirname "$REUSE_ONE_EPISODES")/ppo_updates.jsonl" "${ARTIFACT_DIR}/one_ppo_updates.jsonl"
  else
    echo "[ab][warning] reused one PPO updates not found; equality gate will compare episodes only"
  fi
  printf '{"label":"one","reused":true,"episodes":"%s","wall":"%s"}\n' \
    "$REUSE_ONE_EPISODES" "$REUSE_ONE_WALL" >> "${ARTIFACT_DIR}/runs.jsonl"
  echo "[ab] one baseline reused from ${REUSE_ONE_EPISODES}; wall=$(cat "${ARTIFACT_DIR}/one_wall_seconds.txt")s"
}

if [ -n "$REUSE_ONE_EPISODES" ] || [ -n "$REUSE_ONE_WALL" ]; then
  reuse_one_case
else
  run_case one "$ONE_DEVS" "$ONE_WORKERS_PER_DEVICE"
fi
run_case many "$MANY_DEVS" "$MANY_WORKERS_PER_DEVICE"

echo ""
echo "================================================================================"
echo "[ab] strict equality + speed verdict"
echo "================================================================================"
compare_extra=()
if [ -f "${ARTIFACT_DIR}/one_ppo_updates.jsonl" ] && [ -f "${ARTIFACT_DIR}/many_ppo_updates.jsonl" ]; then
  compare_extra+=(--one-ppo "${ARTIFACT_DIR}/one_ppo_updates.jsonl")
  compare_extra+=(--many-ppo "${ARTIFACT_DIR}/many_ppo_updates.jsonl")
elif [ -f "${ARTIFACT_DIR}/one_ppo_updates.jsonl" ] || [ -f "${ARTIFACT_DIR}/many_ppo_updates.jsonl" ]; then
  echo "[FATAL] PPO update artifact presence differs; refusing to downgrade equality gate"
  ls -l "${ARTIFACT_DIR}/"*"_ppo_updates.jsonl" 2>/dev/null || true
  exit 126
fi
python3 scripts/stage2_ngpu_ab_compare.py \
  --one "${ARTIFACT_DIR}/one_episodes.jsonl" \
  --many "${ARTIFACT_DIR}/many_episodes.jsonl" \
  --one-wall "${ARTIFACT_DIR}/one_wall_seconds.txt" \
  --many-wall "${ARTIFACT_DIR}/many_wall_seconds.txt" \
  "${compare_extra[@]}" \
  --one-log "${ARTIFACT_DIR}/one_launch.log" \
  --many-log "${ARTIFACT_DIR}/many_launch.log" \
  --require-equal \
  --min-speedup "$MIN_SPEEDUP" \
  --require-speedup \
  --out "${ARTIFACT_DIR}/stage2_ngpu_gate_verdict.txt"

echo "[DONE] Stage-2 N-GPU A/B gate passed: ${ARTIFACT_DIR}/stage2_ngpu_gate_verdict.txt"
