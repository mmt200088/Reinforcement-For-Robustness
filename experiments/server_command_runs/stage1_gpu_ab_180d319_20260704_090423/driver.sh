#!/usr/bin/env bash
set -uo pipefail
RUN=/hy-tmp/rfr_stage1_ab_180d319_20260704_090423_min170_wait
PKG=/hy-tmp/rfr_stage1_ab_180d319_20260704_090423.tgz
SRC="$RUN/src"
OUT="$RUN/out"
STATUS="$RUN/status.json"
SOURCE_COMMIT=180d319
EPISODES=170
PPO_INTERVAL=170
mkdir -p "$SRC" "$OUT"
write_status() {
  local state="$1" detail="$2"
  STATUS_PATH="$STATUS" STATE="$state" DETAIL="$detail" RUN_PATH="$RUN" python3 - <<'PY'
import json, os, time
from pathlib import Path
Path(os.environ['STATUS_PATH']).write_text(json.dumps({'state': os.environ['STATE'], 'detail': os.environ['DETAIL'], 'time': time.time(), 'run': os.environ['RUN_PATH']}, indent=2, sort_keys=True)+"\n")
PY
}
write_status extracting start
{
  echo "RUN=$RUN"
  echo "PKG=$PKG"
  echo "SOURCE_COMMIT=$SOURCE_COMMIT"
  echo "EPISODES=$EPISODES"
  echo "PPO_INTERVAL=$PPO_INTERVAL"
  echo "START=$(date -Is)"
  echo "HOST=$(hostname)"
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader || true
} > "$OUT/driver.log" 2>&1

tar --warning=no-unknown-keyword -xzf "$PKG" -C "$SRC" >> "$OUT/driver.log" 2>&1
cd "$SRC" || exit 2
export HF_HOME=/hy-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export GLUE_LOCAL_DATASET_DIR=/hy-tmp/glue_data
export PYTHONUNBUFFERED=1
python3 - <<'PY' >> "$OUT/driver.log" 2>&1
import os, torch
print('PYTHON_OK')
print('TORCH', torch.__version__)
print('CUDA_AVAILABLE', torch.cuda.is_available())
print('CUDA_COUNT', torch.cuda.device_count())
print('GLUE_LOCAL_DATASET_DIR', os.environ.get('GLUE_LOCAL_DATASET_DIR'))
PY
py_rc=$?
echo "$py_rc" > "$OUT/python_env_rc.txt"
if [ "$py_rc" -ne 0 ]; then
  write_status failed python_env
  exit "$py_rc"
fi

find_stage1_log() {
  local label="$1"
  local label_out="$OUT/$label"
  local latest_run_file="$label_out/stage1/LATEST_RUN_DIR"
  local latest_run=""
  if [ -f "$latest_run_file" ]; then
    latest_run=$(cat "$latest_run_file" 2>/dev/null || true)
    if [ -n "$latest_run" ] && [ -f "$latest_run/logs/output.log" ]; then
      printf '%s\n' "$latest_run/logs/output.log"
      return 0
    fi
  fi
  find "$label_out/stage1" -type f \( -name 'stage1_rl.log' -o -name 'pruning_search_log.txt' -o -name 'output.log' -o -name '*stage1*log*.txt' \) 2>/dev/null | sort | head -n 1
}

wait_for_training_pid() {
  local label="$1"
  local label_out="$OUT/$label"
  local pid_file="$label_out/stage1/LATEST_PID"
  local pid=""
  for _ in $(seq 1 60); do
    if [ -s "$pid_file" ]; then
      pid=$(cat "$pid_file" 2>/dev/null || true)
      break
    fi
    sleep 1
  done
  printf '%s\n' "$pid" > "$label_out/training_pid.txt"
  if [ -z "$pid" ]; then
    echo "missing_pid" > "$label_out/training_wait_status.txt"
    return 1
  fi
  echo "pid=$pid" > "$label_out/training_wait_status.txt"
  while kill -0 "$pid" 2>/dev/null; do
    {
      printf 'sample_time=%s\n' "$(date -Is)"
      nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || true
    } >> "$label_out/gpu_samples.csv"
    sleep 15
  done
  echo "exited" >> "$label_out/training_wait_status.txt"
  return 0
}

run_one() {
  local label="$1"
  local cvd="$2"
  local devices="$3"
  local label_out="$OUT/$label"
  mkdir -p "$label_out"
  write_status "running_${label}" "devices_${devices}"
  echo "=== RUN ${label} START $(date -Is) devices=${devices} CUDA_VISIBLE_DEVICES=${cvd}" | tee -a "$OUT/driver.log"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > "$label_out/nvidia_before.csv" 2>&1 || true
  local start end launcher_rc wait_rc log_path completed_marker
  start=$(date +%s)
  CUDA_VISIBLE_DEVICES="$cvd" timeout 7200 bash llama_7B_LayerImportance.sh run rl \
    --mode stage1-only \
    --dataset mrpc \
    --model-type bert-base \
    --rl-algo ppo \
    --stage1-search-episodes "$EPISODES" \
    --ppo-update-interval "$PPO_INTERVAL" \
    --stage1-accuracy-tolerance 0.0 \
    --stage1-search-lr 2e-5 \
    --stage1-rl-devices "$devices" \
    --persistent-root "$label_out/persistent" \
    --fresh \
    > "$label_out/launch.log" 2>&1
  launcher_rc=$?
  echo "$launcher_rc" > "$label_out/launcher_rc.txt"
  wait_rc=1
  if [ "$launcher_rc" -eq 0 ]; then
    wait_for_training_pid "$label"
    wait_rc=$?
  fi
  end=$(date +%s)
  echo "$launcher_rc" > "$label_out/rc.txt"
  echo "$wait_rc" > "$label_out/wait_rc.txt"
  echo "$((end-start))" > "$label_out/walltime_s.txt"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > "$label_out/nvidia_after.csv" 2>&1 || true
  log_path=$(find_stage1_log "$label" || true)
  printf '%s\n' "$log_path" > "$label_out/stage1_log_path.txt"
  completed_marker="false"
  if [ -f "$label_out/stage1/bert base mrpc/COMPLETED" ]; then
    completed_marker="true"
  fi
  printf '%s\n' "$completed_marker" > "$label_out/completed_marker.txt"
  if [ -n "$log_path" ] && [ -f "$log_path" ]; then
    python3 scripts/stage1_parallel_report.py --log "$log_path" --out-json "$label_out/stage1_parallel_summary.json" --out-md "$label_out/stage1_parallel_summary.md" > "$label_out/report_parser.stdout" 2> "$label_out/report_parser.stderr"
    rg -n "stage1-rollout|stage1-rollout-total|rollout_sig|eval_cache|model_forward|report_write|Stage-1 entropy|Episode" "$log_path" > "$label_out/timing_markers.txt" 2>/dev/null || true
  fi
  echo "=== RUN ${label} END $(date -Is) launcher_rc=${launcher_rc} wait_rc=${wait_rc} wall=$((end-start))s completed=${completed_marker} log=${log_path}" | tee -a "$OUT/driver.log"
  return 0
}

run_one g1 0 0
run_one g4 0,1,2,3 0,1,2,3
write_status summarizing compare
OUT_FOR_COMPARE="$OUT" SOURCE_COMMIT="$SOURCE_COMMIT" EPISODES_FOR_COMPARE="$EPISODES" PPO_INTERVAL_FOR_COMPARE="$PPO_INTERVAL" python3 - <<'PY' > "$OUT/comparison.json"
import json, os
from pathlib import Path
out = Path(os.environ['OUT_FOR_COMPARE'])
rows = {}
for label in ('g1','g4'):
    root = out / label
    row = {
        'rc': int((root/'rc.txt').read_text().strip()) if (root/'rc.txt').exists() else None,
        'launcher_rc': int((root/'launcher_rc.txt').read_text().strip()) if (root/'launcher_rc.txt').exists() else None,
        'wait_rc': int((root/'wait_rc.txt').read_text().strip()) if (root/'wait_rc.txt').exists() else None,
        'walltime_s': int((root/'walltime_s.txt').read_text().strip()) if (root/'walltime_s.txt').exists() else None,
        'training_pid': (root/'training_pid.txt').read_text().strip() if (root/'training_pid.txt').exists() else '',
        'training_wait_status': (root/'training_wait_status.txt').read_text().strip().splitlines() if (root/'training_wait_status.txt').exists() else [],
        'completed_marker': (root/'completed_marker.txt').read_text().strip() if (root/'completed_marker.txt').exists() else '',
        'stage1_log_path': (root/'stage1_log_path.txt').read_text().strip() if (root/'stage1_log_path.txt').exists() else '',
    }
    summary_path = root / 'stage1_parallel_summary.json'
    if summary_path.exists():
        row['summary'] = json.loads(summary_path.read_text())
    rows[label] = row

def get(label, key):
    s = rows.get(label, {}).get('summary') or {}
    return s.get(key)
wall_g1 = rows.get('g1', {}).get('walltime_s') or 0
wall_g4 = rows.get('g4', {}).get('walltime_s') or 0
thr_g1 = get('g1', 'throughput_ep_per_hour') or 0
thr_g4 = get('g4', 'throughput_ep_per_hour') or 0
comparison = {
    'source_commit': os.environ['SOURCE_COMMIT'],
    'episodes_per_run': int(os.environ['EPISODES_FOR_COMPARE']),
    'ppo_update_interval': int(os.environ['PPO_INTERVAL_FOR_COMPARE']),
    'rows': rows,
    'wall_clock_speedup_g4_over_g1': (wall_g1 / wall_g4) if wall_g1 and wall_g4 else None,
    'parser_throughput_speedup_g4_over_g1': (thr_g4 / thr_g1) if thr_g1 and thr_g4 else None,
    'all_rc_zero': all((rows.get(label, {}).get('rc') == 0) for label in ('g1','g4')),
    'all_wait_zero': all((rows.get(label, {}).get('wait_rc') == 0) for label in ('g1','g4')),
    'all_completed': all((rows.get(label, {}).get('completed_marker') == 'true') for label in ('g1','g4')),
}
print(json.dumps(comparison, indent=2, sort_keys=True))
PY
summary_rc=$?
echo "$summary_rc" > "$OUT/summary_rc.txt"
write_status complete rc_done
{
  echo "END=$(date -Is)"
  cat "$OUT/comparison.json" || true
} >> "$OUT/driver.log" 2>&1
exit 0
