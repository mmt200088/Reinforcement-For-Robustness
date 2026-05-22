#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/hy-tmp/hf_cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export GLUE_LOCAL_DATASET_DIR="${GLUE_LOCAL_DATASET_DIR:-/hy-tmp/glue_data}"

PLANNED_EPISODES="${PLANNED_EPISODES:-10000}"
SMOKE_EPISODES="${SMOKE_EPISODES:-1000}"
ANCHOR_EPISODES="${ANCHOR_EPISODES:-60}"
ROLLOUT_SIZE="${ROLLOUT_SIZE:-60}"
K_TRIALS="${K_TRIALS:-4}"
PROBE_SIZE="${PROBE_SIZE:-256}"
BATCH_SIZE="${BATCH_SIZE:-512}"
RL_CUDA_VISIBLE_DEVICES="${RL_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
REWARD_DEVICES="${REWARD_DEVICES:-0,1,2,3}"
NEIGHBOR_RAMP="${NEIGHBOR_RAMP:-1800}"
NEIGHBOR_MAX_MUTATIONS="${NEIGHBOR_MAX_MUTATIONS:-12}"
NEIGHBOR_MAX_RADIUS="${NEIGHBOR_MAX_RADIUS:-1}"
GUARDED_RADIUS2_ENABLED="${GUARDED_RADIUS2_ENABLED:-1}"
GUARDED_RADIUS2_MIN_EPISODE="${GUARDED_RADIUS2_MIN_EPISODE:-1060}"
GUARDED_RADIUS2_STALL_WINDOW="${GUARDED_RADIUS2_STALL_WINDOW:-600}"
GUARDED_RADIUS2_MAX_MUTATIONS="${GUARDED_RADIUS2_MAX_MUTATIONS:-4}"
GUARDED_RADIUS2_EPISODE_FRACTION="${GUARDED_RADIUS2_EPISODE_FRACTION:-0.15}"
GUARDED_RADIUS2_COOLDOWN_EPISODES="${GUARDED_RADIUS2_COOLDOWN_EPISODES:-300}"
ENT_COEF="${ENT_COEF:-0.06}"
ENT_RAMP="${ENT_RAMP:-600}"
WARMSTART_BIAS_GAIN="${WARMSTART_BIAS_GAIN:-1.2}"

RUN_ID="stage2_rl_${PLANNED_EPISODES}_curve_$(date +%Y%m%d_%H%M%S)"
ARTIFACT_DIR="experiments/server_command_runs/${RUN_ID}"
PERSIST_ROOT="Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.005_s2t0.005_s2st0.005"
STAGE2_NOISE="${PERSIST_ROOT}/stage2_noise"
NVS_LOG="${ARTIFACT_DIR}/nvidia_smi_during_rl.csv"
export ARTIFACT_DIR STAGE2_NOISE

mkdir -p "$ARTIFACT_DIR" logs
exec > >(tee "${ARTIFACT_DIR}/server_command_stdout.log") 2>&1

echo "[goal] Fresh GTrXL v2-scale Stage-2 BLB RL run for ${PLANNED_EPISODES} episodes."
echo "[goal] Preserve no-collapse safety while reducing baseline lock-in with non-monotonic cost-boundary exploration."
echo "[goal] Watch KL/LR scale, entropy recovery, empirical offset stats, Pareto progress, and four-GPU reward probe health."

stop_rl_at_dir() {
  local dir="$1"
  local pidfile="$dir/rl.pid"
  [ -f "$pidfile" ] || { echo "[stop-rl] $dir: no rl.pid"; return 0; }
  local pid
  pid="$(cat "$pidfile")"
  if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
    echo "[stop-rl] $dir: pid=$pid already dead"
    return 0
  fi
  echo "[stop-rl] $dir: running pid=$pid, SIGINT ..."
  kill -INT "$pid" 2>/dev/null || true
  for _ in 1 2 3 4 5 6; do sleep 10; kill -0 "$pid" 2>/dev/null || break; done
  if kill -0 "$pid" 2>/dev/null; then
    echo "[stop-rl] $dir: still alive after 60s, SIGTERM ..."
    kill -TERM "$pid" 2>/dev/null || true
    for _ in 1 2 3; do sleep 10; kill -0 "$pid" 2>/dev/null || break; done
  fi
  if kill -0 "$pid" 2>/dev/null; then
    echo "[stop-rl] $dir: still alive after 90s, SIGKILL ..."
    kill -KILL "$pid" 2>/dev/null || true
    sleep 3
  fi
}

monitor_once() {
  local phase="$1"
  local planned="$2"
  python scripts/stage2_first10k_monitor.py \
    --phase "$phase" \
    --artifact-dir "$ARTIFACT_DIR" \
    --stage2-noise "$STAGE2_NOISE" \
    --nvidia-log "$NVS_LOG" \
    --planned "$planned" \
    --anchor "$ANCHOR_EPISODES" \
    --rollout "$ROLLOUT_SIZE" \
    --horizon 59 \
    --k-trials "$K_TRIALS" \
    --probe-size "$PROBE_SIZE" \
    --expected-reward-devices "$REWARD_DEVICES"
}

copy_artifacts() {
  local diag_dir="${STAGE2_NOISE}/progress/diagnostics"
  [ -f "${diag_dir}/episodes.jsonl" ] && cp "${diag_dir}/episodes.jsonl" "${ARTIFACT_DIR}/episodes.jsonl" || true
  [ -f "${diag_dir}/ppo_updates.jsonl" ] && cp "${diag_dir}/ppo_updates.jsonl" "${ARTIFACT_DIR}/ppo_updates.jsonl" || true
  [ -f "${diag_dir}/pareto_frontier.jsonl" ] && cp "${diag_dir}/pareto_frontier.jsonl" "${ARTIFACT_DIR}/pareto_frontier.jsonl" || true
  [ -f "${diag_dir}/pareto_frontier.json" ] && cp "${diag_dir}/pareto_frontier.json" "${ARTIFACT_DIR}/pareto_frontier.json" || true
  [ -f "${diag_dir}/pareto_frontier.html" ] && cp "${diag_dir}/pareto_frontier.html" "${ARTIFACT_DIR}/pareto_frontier.html" || true
  [ -f "${STAGE2_NOISE}/warning.txt" ] && cp "${STAGE2_NOISE}/warning.txt" "${ARTIFACT_DIR}/warning.txt" || true
  [ -f "${STAGE2_NOISE}/pruning_search_log.txt" ] && tail -n 40000 "${STAGE2_NOISE}/pruning_search_log.txt" > "${ARTIFACT_DIR}/pruning_search_log_tail_source.txt" || true
  for path in \
    "${STAGE2_NOISE}/progress/blb_stage2_best_action_full.json" \
    "${STAGE2_NOISE}/progress/blb_stage2_best_action_full.md" \
    "${STAGE2_NOISE}/progress/blb_stage2_baseline_action_full.json" \
    "${STAGE2_NOISE}/progress/blb_stage2_baseline_action_full.md" \
    "${STAGE2_NOISE}/progress/blb_stage2_report.md" \
    "${STAGE2_NOISE}/progress/blb_stage2_status.json" \
    "${STAGE2_NOISE}/progress/blb_stage2_training_curve.png"; do
    [ -f "$path" ] && cp "$path" "$ARTIFACT_DIR/" || true
  done
}

stop_with_signal() {
  local pid="$1"
  local reason="$2"
  echo "$reason" > "${ARTIFACT_DIR}/abort_reason.txt"
  echo "[watchdog] stopping pid=$pid: $reason"
  kill -INT "$pid" 2>/dev/null || true
  for _ in 1 2 3 4 5 6; do sleep 10; kill -0 "$pid" 2>/dev/null || break; done
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
    for _ in 1 2 3; do sleep 10; kill -0 "$pid" 2>/dev/null || break; done
  fi
  if kill -0 "$pid" 2>/dev/null; then
    kill -KILL "$pid" 2>/dev/null || true
    sleep 3
  fi
}

launch_and_watch_rl() {
  local label="$1"
  local planned="$2"
  local log_file="$3"

  echo ""
  echo "================================================================================"
  echo "RL phase: ${label} (${planned} episodes, reward GPUs=${REWARD_DEVICES}, CUDA_VISIBLE_DEVICES=${RL_CUDA_VISIBLE_DEVICES})"
  echo "================================================================================"
  set +e
  CUDA_VISIBLE_DEVICES="$RL_CUDA_VISIBLE_DEVICES" bash llama_7B_LayerImportance.sh run rl \
    --preset mrpc-blb-stage2-rl \
    --batch-size "$BATCH_SIZE" \
    --stage2-search-episodes "$planned" \
    --stage2-rollout-size "$ROLLOUT_SIZE" \
    --stage2-k-trials "$K_TRIALS" \
    --stage2-probe-size "$PROBE_SIZE" \
    --stage2-save-interval 500 \
    --stage2-eval-interval 300 \
    --blb-v3-warmstart-anchor-episodes "$ANCHOR_EPISODES" \
    --blb-v3-warmstart-neighbor-ramp-episodes "$NEIGHBOR_RAMP" \
    --blb-v3-warmstart-neighbor-max-mutations "$NEIGHBOR_MAX_MUTATIONS" \
    --blb-v3-warmstart-neighbor-max-radius "$NEIGHBOR_MAX_RADIUS" \
    --blb-v3-warmstart-bias-gain "$WARMSTART_BIAS_GAIN" \
    --blb-v3-guarded-radius2-enabled "$GUARDED_RADIUS2_ENABLED" \
    --blb-v3-guarded-radius2-min-episode "$GUARDED_RADIUS2_MIN_EPISODE" \
    --blb-v3-guarded-radius2-stall-window "$GUARDED_RADIUS2_STALL_WINDOW" \
    --blb-v3-guarded-radius2-max-mutations "$GUARDED_RADIUS2_MAX_MUTATIONS" \
    --blb-v3-guarded-radius2-episode-fraction "$GUARDED_RADIUS2_EPISODE_FRACTION" \
    --blb-v3-guarded-radius2-cooldown-episodes "$GUARDED_RADIUS2_COOLDOWN_EPISODES" \
    --blb-v3-ent-coef "$ENT_COEF" \
    --blb-v3-ent-coef-ramp-episodes "$ENT_RAMP" \
    --blb-v3-reward-devices "$REWARD_DEVICES" \
    --skip-final-eval \
    --fresh 2>&1 | tee "${ARTIFACT_DIR}/${log_file}"
  local launch_rc=${PIPESTATUS[0]}
  set -e
  echo "[rl:${label}] launcher rc=$launch_rc"
  if [ "$launch_rc" -ne 0 ]; then
    return "$launch_rc"
  fi

  local rl_pid_file="${PERSIST_ROOT}/rl.pid"
  for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
    [ -s "$rl_pid_file" ] && break
    sleep 2
  done
  if [ ! -s "$rl_pid_file" ]; then
    echo "[fail:${label}] launcher returned success but did not write $rl_pid_file"
    return 12
  fi
  local run_pid
  run_pid="$(cat "$rl_pid_file")"
  echo "[rl:${label}] background pid=$run_pid; online watchdog enabled"

  local last_episodes=-1
  local last_progress_ts
  last_progress_ts=$(date +%s)
  while kill -0 "$run_pid" 2>/dev/null; do
    local episodes_done=0
    if [ -f "${STAGE2_NOISE}/progress/diagnostics/episodes.jsonl" ]; then
      episodes_done=$(wc -l < "${STAGE2_NOISE}/progress/diagnostics/episodes.jsonl" | tr -d ' ')
    fi
    copy_artifacts
    set +e
    monitor_once live "$planned"
    local mon_rc=$?
    set -e
    echo "[rl-monitor:${label}] pid=$run_pid alive; episodes_jsonl=$episodes_done; monitor_rc=$mon_rc; $(date -Is)"
    if [ "$mon_rc" -eq 2 ] && [ "$episodes_done" -gt "$ANCHOR_EPISODES" ]; then
      stop_with_signal "$run_pid" "hard failure reported by ${label} live monitor"
      break
    fi
    if [ "$episodes_done" -gt "$last_episodes" ]; then
      last_episodes="$episodes_done"
      last_progress_ts=$(date +%s)
    elif [ "$episodes_done" -lt "$planned" ]; then
      local now_ts
      now_ts=$(date +%s)
      if [ $((now_ts - last_progress_ts)) -gt 1200 ]; then
        stop_with_signal "$run_pid" "${label} episodes stalled for more than 20 minutes"
        break
      fi
    fi
    sleep 60
  done

  echo "[rl:${label}] background pid=$run_pid exited or was stopped"
  copy_artifacts
  set +e
  monitor_once final "$planned"
  local final_rc=$?
  set -e
  echo "[monitor:${label}] final rc=$final_rc"
  if [ -f "${ARTIFACT_DIR}/monitor_summary.json" ]; then
    cp "${ARTIFACT_DIR}/monitor_summary.json" "${ARTIFACT_DIR}/${label}_monitor_summary.json" || true
  fi
  if [ -f "${ARTIFACT_DIR}/server_monitor_report.html" ]; then
    cp "${ARTIFACT_DIR}/server_monitor_report.html" "${ARTIFACT_DIR}/${label}_server_monitor_report.html" || true
  fi
  if [ -f "${ARTIFACT_DIR}/episodes.jsonl" ]; then
    cp "${ARTIFACT_DIR}/episodes.jsonl" "${ARTIFACT_DIR}/${label}_episodes.jsonl" || true
  fi
  if [ -f "${ARTIFACT_DIR}/pareto_frontier.json" ]; then
    cp "${ARTIFACT_DIR}/pareto_frontier.json" "${ARTIFACT_DIR}/${label}_pareto_frontier.json" || true
  fi
  return "$final_rc"
}

stop_rl_at_dir "$PERSIST_ROOT"
stop_rl_at_dir "${PERSIST_ROOT}_rdv2"

echo ""
echo "================================================================================"
echo "Step 1/6: git pull latest local source changes"
echo "================================================================================"
set +e
timeout 180 git pull --ff-only
PULL_RC=$?
set -e
if [ "$PULL_RC" -ne 0 ]; then
  echo "[abort] git pull failed or timed out (rc=$PULL_RC); refusing to run on stale HEAD."
  exit "$PULL_RC"
fi
echo "[git] HEAD = $(git rev-parse --short HEAD)"

cat > "${ARTIFACT_DIR}/run_manifest.json" <<JSON
{
  "run_id": "${RUN_ID}",
  "git_head": "$(git rev-parse --short HEAD)",
  "planned_episodes": ${PLANNED_EPISODES},
  "smoke_episodes": ${SMOKE_EPISODES},
  "anchor_episodes": ${ANCHOR_EPISODES},
  "rollout_size": ${ROLLOUT_SIZE},
  "k_trials": ${K_TRIALS},
  "probe_size": ${PROBE_SIZE},
  "batch_size": ${BATCH_SIZE},
  "neighbor_ramp": ${NEIGHBOR_RAMP},
  "neighbor_max_mutations": ${NEIGHBOR_MAX_MUTATIONS},
  "neighbor_max_radius": ${NEIGHBOR_MAX_RADIUS},
  "guarded_radius2_enabled": ${GUARDED_RADIUS2_ENABLED},
  "guarded_radius2_min_episode": ${GUARDED_RADIUS2_MIN_EPISODE},
  "guarded_radius2_stall_window": ${GUARDED_RADIUS2_STALL_WINDOW},
  "guarded_radius2_max_mutations": ${GUARDED_RADIUS2_MAX_MUTATIONS},
  "guarded_radius2_episode_fraction": ${GUARDED_RADIUS2_EPISODE_FRACTION},
  "guarded_radius2_cooldown_episodes": ${GUARDED_RADIUS2_COOLDOWN_EPISODES},
  "ent_coef": ${ENT_COEF},
  "ent_ramp": ${ENT_RAMP},
  "warmstart_bias_gain": ${WARMSTART_BIAS_GAIN},
  "policy_variant": "blb_v3_sequential_gtrxl_v2scale",
  "baseline_prior_schedule": "1.2 anchor; 1.0->0.45 ep60-600; 0.45->0.15 ep600-2000; 0.15 thereafter",
  "exploration_design": "non-monotonic empirical cost-boundary exploration",
  "cuda_visible_devices": "${RL_CUDA_VISIBLE_DEVICES}",
  "reward_devices": "${REWARD_DEVICES}"
}
JSON

echo ""
echo "================================================================================"
echo "Step 2/6: local/contract tests on server"
echo "================================================================================"
set +e
python -m unittest tests.test_sequential_smoke.WarmstartFixedRegressionTest tests.test_sequential_smoke.EntCoefScheduleRegressionTest -v 2>&1 | tee "${ARTIFACT_DIR}/test_sequential_smoke.log"
TEST1_RC=${PIPESTATUS[0]}
BLB_STRICT=0 python -m unittest discover -s tests -p "test_blb_*.py" -v 2>&1 | tee "${ARTIFACT_DIR}/test_blb_contracts.log"
TEST2_RC=${PIPESTATUS[0]}
python scripts/blb_verify_noise_install.py --mode smoke --profile mrpc --num-layers 12 2>&1 | tee "${ARTIFACT_DIR}/blb_verify_noise_install.log"
TEST3_RC=${PIPESTATUS[0]}
set -e
if [ "$TEST1_RC" -ne 0 ] || [ "$TEST2_RC" -ne 0 ] || [ "$TEST3_RC" -ne 0 ]; then
  echo "[abort] tests failed: sequential=$TEST1_RC contracts=$TEST2_RC install=$TEST3_RC"
  exit 10
fi

echo ""
echo "================================================================================"
echo "Step 3/6: GPU visibility"
echo "================================================================================"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | tee "${ARTIFACT_DIR}/nvidia_pre_rl.csv"
N_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
echo "[nvidia-smi] visible GPUs = $N_GPUS"
if [ "$N_GPUS" -lt 4 ]; then
  echo "[abort] need >= 4 GPUs for this run; saw $N_GPUS"
  exit 11
fi

(
  printf "timestamp,gpu_idx,util_pct,mem_used_mib\n" > "$NVS_LOG"
  while true; do
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
      --format=csv,noheader,nounits >> "$NVS_LOG" 2>/dev/null || true
    sleep 15
  done
) &
NVS_PID=$!
trap "kill $NVS_PID 2>/dev/null || true" EXIT

echo ""
echo "================================================================================"
echo "Step 4/6: fresh ${SMOKE_EPISODES}-episode smoke run before formal RL"
echo "================================================================================"
launch_and_watch_rl "smoke" "$SMOKE_EPISODES" "rl_smoke_four_gpu.log"
SMOKE_RC=$?
if [ "$SMOKE_RC" -ne 0 ]; then
  kill "$NVS_PID" 2>/dev/null || true
  trap - EXIT
  exit "$SMOKE_RC"
fi

echo ""
echo "================================================================================"
echo "Step 5/6: fresh ${PLANNED_EPISODES}-episode formal four-GPU RL run"
echo "================================================================================"
launch_and_watch_rl "formal" "$PLANNED_EPISODES" "rl_${PLANNED_EPISODES}_four_gpu.log"
FORMAL_RC=$?
kill "$NVS_PID" 2>/dev/null || true
trap - EXIT
if [ "$FORMAL_RC" -ne 0 ]; then
  exit "$FORMAL_RC"
fi

echo ""
echo "================================================================================"
echo "Step 5.5/6: final artifact copy"
echo "================================================================================"
copy_artifacts

echo ""
echo "================================================================================"
echo "Step 6/6: best-effort git artifact publish"
echo "================================================================================"
git add -f "$ARTIFACT_DIR" || true
git commit -m "Add GTrXL guarded-radius2 ${PLANNED_EPISODES}-episode RL monitor results" || true
git push || true

exit 0
