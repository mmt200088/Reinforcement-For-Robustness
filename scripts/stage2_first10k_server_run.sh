#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/hy-tmp/hf_cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export GLUE_LOCAL_DATASET_DIR="${GLUE_LOCAL_DATASET_DIR:-/hy-tmp/glue_data}"

PLANNED_EPISODES="${PLANNED_EPISODES:-10000}"
ANCHOR_EPISODES="${ANCHOR_EPISODES:-120}"
ROLLOUT_SIZE="${ROLLOUT_SIZE:-60}"
K_TRIALS="${K_TRIALS:-5}"
PROBE_SIZE="${PROBE_SIZE:-256}"
NEIGHBOR_RAMP="${NEIGHBOR_RAMP:-3000}"
NEIGHBOR_MAX_MUTATIONS="${NEIGHBOR_MAX_MUTATIONS:-16}"
NEIGHBOR_MAX_RADIUS="${NEIGHBOR_MAX_RADIUS:-3}"
ENT_COEF="${ENT_COEF:-0.04}"
ENT_RAMP="${ENT_RAMP:-1200}"

RUN_ID="stage2_rl_first10k_curve_$(date +%Y%m%d_%H%M%S)"
ARTIFACT_DIR="experiments/server_command_runs/${RUN_ID}"
PERSIST_ROOT="Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.005_s2t0.005_s2st0.005"
STAGE2_NOISE="${PERSIST_ROOT}/stage2_noise"
NVS_LOG="${ARTIFACT_DIR}/nvidia_smi_during_rl.csv"
export ARTIFACT_DIR STAGE2_NOISE

mkdir -p "$ARTIFACT_DIR" logs
exec > >(tee "${ARTIFACT_DIR}/server_command_stdout.log") 2>&1

echo "[goal] Optimize first ${PLANNED_EPISODES} BLB Stage-2 RL episodes."
echo "[goal] Preserve no-collapse safety while improving the early reward curve."
echo "[goal] Watch entropy/clip_fraction, safe-neighbor coverage, cost progress, and dual-GPU reward probe health."

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
  python scripts/stage2_first10k_monitor.py \
    --phase "$phase" \
    --artifact-dir "$ARTIFACT_DIR" \
    --stage2-noise "$STAGE2_NOISE" \
    --nvidia-log "$NVS_LOG" \
    --planned "$PLANNED_EPISODES" \
    --anchor "$ANCHOR_EPISODES" \
    --rollout "$ROLLOUT_SIZE" \
    --horizon 59 \
    --k-trials "$K_TRIALS" \
    --probe-size "$PROBE_SIZE"
}

copy_artifacts() {
  local diag_dir="${STAGE2_NOISE}/progress/diagnostics"
  [ -f "${diag_dir}/episodes.jsonl" ] && cp "${diag_dir}/episodes.jsonl" "${ARTIFACT_DIR}/episodes.jsonl" || true
  [ -f "${diag_dir}/ppo_updates.jsonl" ] && cp "${diag_dir}/ppo_updates.jsonl" "${ARTIFACT_DIR}/ppo_updates.jsonl" || true
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
  echo "[warn] git pull failed or timed out (rc=$PULL_RC); continuing with current HEAD."
fi
echo "[git] HEAD = $(git rev-parse --short HEAD)"

cat > "${ARTIFACT_DIR}/run_manifest.json" <<JSON
{
  "run_id": "${RUN_ID}",
  "git_head": "$(git rev-parse --short HEAD)",
  "planned_episodes": ${PLANNED_EPISODES},
  "anchor_episodes": ${ANCHOR_EPISODES},
  "rollout_size": ${ROLLOUT_SIZE},
  "k_trials": ${K_TRIALS},
  "probe_size": ${PROBE_SIZE},
  "neighbor_ramp": ${NEIGHBOR_RAMP},
  "neighbor_max_mutations": ${NEIGHBOR_MAX_MUTATIONS},
  "neighbor_max_radius": ${NEIGHBOR_MAX_RADIUS},
  "ent_coef": ${ENT_COEF},
  "ent_ramp": ${ENT_RAMP},
  "reward_devices": "0,1"
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
set -e
if [ "$TEST1_RC" -ne 0 ] || [ "$TEST2_RC" -ne 0 ]; then
  echo "[abort] tests failed: sequential=$TEST1_RC contracts=$TEST2_RC"
  exit 10
fi

echo ""
echo "================================================================================"
echo "Step 3/6: GPU visibility"
echo "================================================================================"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | tee "${ARTIFACT_DIR}/nvidia_pre_rl.csv"
N_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
echo "[nvidia-smi] visible GPUs = $N_GPUS"
if [ "$N_GPUS" -lt 2 ]; then
  echo "[abort] need >= 2 GPUs for this run; saw $N_GPUS"
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
echo "Step 4/6: fresh ${PLANNED_EPISODES}-episode dual-GPU RL run"
echo "================================================================================"
set +e
CUDA_VISIBLE_DEVICES=0,1 bash llama_7B_LayerImportance.sh run rl \
  --preset mrpc-blb-stage2-rl \
  --stage2-search-episodes "$PLANNED_EPISODES" \
  --stage2-rollout-size "$ROLLOUT_SIZE" \
  --stage2-k-trials "$K_TRIALS" \
  --stage2-probe-size "$PROBE_SIZE" \
  --stage2-save-interval 500 \
  --stage2-eval-interval 300 \
  --blb-v3-warmstart-anchor-episodes "$ANCHOR_EPISODES" \
  --blb-v3-warmstart-neighbor-ramp-episodes "$NEIGHBOR_RAMP" \
  --blb-v3-warmstart-neighbor-max-mutations "$NEIGHBOR_MAX_MUTATIONS" \
  --blb-v3-warmstart-neighbor-max-radius "$NEIGHBOR_MAX_RADIUS" \
  --blb-v3-ent-coef "$ENT_COEF" \
  --blb-v3-ent-coef-ramp-episodes "$ENT_RAMP" \
  --blb-v3-reward-devices 0,1 \
  --skip-final-eval \
  --fresh 2>&1 | tee "${ARTIFACT_DIR}/rl_10000_dual_gpu.log"
LAUNCH_RC=${PIPESTATUS[0]}
set -e
echo "[rl] launcher rc=$LAUNCH_RC"
if [ "$LAUNCH_RC" -ne 0 ]; then
  kill "$NVS_PID" 2>/dev/null || true
  trap - EXIT
  exit "$LAUNCH_RC"
fi

RL_PID_FILE="${PERSIST_ROOT}/rl.pid"
for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  [ -s "$RL_PID_FILE" ] && break
  sleep 2
done
if [ ! -s "$RL_PID_FILE" ]; then
  echo "[fail] launcher returned success but did not write $RL_PID_FILE"
  exit 12
fi
RUN_PID="$(cat "$RL_PID_FILE")"
echo "[rl] background pid=$RUN_PID; online watchdog enabled"

LAST_EPISODES=-1
LAST_PROGRESS_TS=$(date +%s)
while kill -0 "$RUN_PID" 2>/dev/null; do
  EPISODES_DONE=0
  if [ -f "${STAGE2_NOISE}/progress/diagnostics/episodes.jsonl" ]; then
    EPISODES_DONE=$(wc -l < "${STAGE2_NOISE}/progress/diagnostics/episodes.jsonl" | tr -d ' ')
  fi
  copy_artifacts
  set +e
  monitor_once live
  MON_RC=$?
  set -e
  echo "[rl-monitor] pid=$RUN_PID alive; episodes_jsonl=$EPISODES_DONE; monitor_rc=$MON_RC; $(date -Is)"
  if [ "$MON_RC" -eq 2 ] && [ "$EPISODES_DONE" -gt "$ANCHOR_EPISODES" ]; then
    stop_with_signal "$RUN_PID" "hard failure reported by live monitor"
    break
  fi
  if [ "$EPISODES_DONE" -gt "$LAST_EPISODES" ]; then
    LAST_EPISODES="$EPISODES_DONE"
    LAST_PROGRESS_TS=$(date +%s)
  elif [ "$EPISODES_DONE" -lt "$PLANNED_EPISODES" ]; then
    NOW_TS=$(date +%s)
    if [ $((NOW_TS - LAST_PROGRESS_TS)) -gt 1200 ]; then
      stop_with_signal "$RUN_PID" "episodes stalled for more than 20 minutes"
      break
    fi
  fi
  sleep 60
done

echo "[rl] background pid=$RUN_PID exited or was stopped"
kill "$NVS_PID" 2>/dev/null || true
trap - EXIT

echo ""
echo "================================================================================"
echo "Step 5/6: final monitor and artifact copy"
echo "================================================================================"
copy_artifacts
set +e
monitor_once final
MONITOR_RC=$?
set -e
echo "[monitor] final rc=$MONITOR_RC"

echo ""
echo "================================================================================"
echo "Step 6/6: best-effort git artifact publish"
echo "================================================================================"
git add -f "$ARTIFACT_DIR" || true
git commit -m "Add server first-10k RL monitor results" || true
git push || true

exit "$MONITOR_RC"
