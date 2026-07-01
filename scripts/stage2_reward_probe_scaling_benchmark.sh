#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/hy-tmp/hf_cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export GLUE_LOCAL_DATASET_DIR="${GLUE_LOCAL_DATASET_DIR:-/hy-tmp/glue_data}"

RUN_ID="stage2_reward_probe_scaling_$(date +%Y%m%d_%H%M%S)"
ARTIFACT_DIR="${ARTIFACT_DIR:-experiments/server_command_runs/${RUN_ID}}"
BENCH_EPISODES="${BENCH_EPISODES:-4}"
BENCH_BATCH_SIZES="${BENCH_BATCH_SIZES:-128 256 512}"
PROBE_SIZE="${PROBE_SIZE:-256}"
K_TRIALS="${K_TRIALS:-4}"
DEVICE_SPECS="${DEVICE_SPECS:-0;0,1;0,1,2;0,1,2,3}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-2400}"
GPU_SAMPLE_INTERVAL_SECONDS="${GPU_SAMPLE_INTERVAL_SECONDS:-2}"

mkdir -p "$ARTIFACT_DIR"
exec > >(tee "${ARTIFACT_DIR}/benchmark_stdout.log") 2>&1

echo "[bench] Stage-2 reward probe GPU scaling benchmark"
echo "[bench] artifact_dir=${ARTIFACT_DIR}"
echo "[bench] episodes=${BENCH_EPISODES} k_trials=${K_TRIALS} probe_size=${PROBE_SIZE}"
echo "[bench] batch_sizes=${BENCH_BATCH_SIZES}"
echo "[bench] device_specs=${DEVICE_SPECS}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv \
  | tee "${ARTIFACT_DIR}/nvidia_pre_benchmark.csv"

IFS=';' read -r -a DEVICE_SPEC_ARRAY <<< "$DEVICE_SPECS"

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
      echo "[bench] pid=${pid} exceeded ${timeout_seconds}s; sending SIGINT"
      kill -INT "$pid" 2>/dev/null || true
      sleep 10
      kill -TERM "$pid" 2>/dev/null || true
      return 124
    fi
    sleep 5
    elapsed=$((elapsed + 5))
  done
  return 0
}

for batch_size in $BENCH_BATCH_SIZES; do
  for device_spec in "${DEVICE_SPEC_ARRAY[@]}"; do
    device_spec="$(printf '%s' "$device_spec" | xargs)"
    [ -n "$device_spec" ] || continue
    gpu_count=$(python3 - <<PY
spec = "${device_spec}"
print(len([x for x in spec.split(",") if x.strip()]))
PY
)
    reward_arg=()
    if [ "$gpu_count" -ge 2 ]; then
      reward_arg=(--blb-v3-reward-devices "$device_spec")
    fi
    label="bs${batch_size}_g${gpu_count}"
    persistent_root="${ARTIFACT_DIR}/persistent_${label}"
    latest_pid_file="${persistent_root}/rl/bert-base/mrpc/LATEST_PID"
    log_file="${ARTIFACT_DIR}/${label}.log"
    gpu_sample_file="${ARTIFACT_DIR}/${label}_nvidia_smi.csv"
    echo ""
    echo "================================================================================"
    echo "[bench] ${label}: CUDA_VISIBLE_DEVICES=${device_spec}"
    echo "================================================================================"
    rm -rf "$persistent_root"
    set +e
    ALLOW_SHORT_RL_BENCHMARK=1 CUDA_VISIBLE_DEVICES="$device_spec" timeout "$TIMEOUT_SECONDS" \
      bash llama_7B_LayerImportance.sh run rl \
        --preset mrpc-blb-stage2-rl \
        --persistent-root "$persistent_root" \
        --batch-size "$batch_size" \
        --ppo-update-interval "$BENCH_EPISODES" \
        --stage2-search-episodes "$BENCH_EPISODES" \
        --stage2-rollout-size 60 \
        --stage2-k-trials "$K_TRIALS" \
        --stage2-probe-size "$PROBE_SIZE" \
        --stage2-save-interval 1000 \
        --stage2-eval-interval 1000 \
        --skip-final-eval \
        --fresh \
        "${reward_arg[@]}" 2>&1 | tee "$log_file"
    launch_rc=${PIPESTATUS[0]}
    rc=$launch_rc
    if [ "$launch_rc" -eq 0 ] && [ -f "$latest_pid_file" ]; then
      job_pid="$(cat "$latest_pid_file" | tr -d '[:space:]')"
      echo "[bench] ${label} launched pid=${job_pid}; waiting for completion"
      sample_gpu_usage "$gpu_sample_file" &
      sampler_pid=$!
      set +e
      wait_for_background_run "$job_pid" "$TIMEOUT_SECONDS"
      wait_rc=$?
      set -e
      kill "$sampler_pid" 2>/dev/null || true
      wait "$sampler_pid" 2>/dev/null || true
      rc=$wait_rc
    fi
    set -e
    echo "[bench] ${label} launch_rc=${launch_rc} rc=${rc}"
    progress_dir="${persistent_root}/rl/bert-base/mrpc/s1t0.005_s2t0.005_s2st0.005/stage2_noise/progress"
    diag_dir="${progress_dir}/diagnostics"
    [ -f "${diag_dir}/episodes.jsonl" ] && cp "${diag_dir}/episodes.jsonl" "${ARTIFACT_DIR}/${label}_episodes.jsonl" || true
    [ -f "${progress_dir}/../pruning_search_log.txt" ] && cp "${progress_dir}/../pruning_search_log.txt" "${ARTIFACT_DIR}/${label}_pruning_search_log.txt" || true
    printf '{"label":"%s","batch_size":%s,"gpu_count":%s,"device_spec":"%s","launch_rc":%s,"rc":%s}\n' \
      "$label" "$batch_size" "$gpu_count" "$device_spec" "$launch_rc" "$rc" >> "${ARTIFACT_DIR}/runs.jsonl"
  done
done

python3 scripts/stage2_reward_probe_scaling_report.py "$ARTIFACT_DIR"
