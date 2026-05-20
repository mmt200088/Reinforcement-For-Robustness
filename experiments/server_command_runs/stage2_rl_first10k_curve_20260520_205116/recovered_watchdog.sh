#!/usr/bin/env bash
set -euo pipefail
WORK='/hy-tmp/Reinforcement-For-Robustness-pareto-r1'
ART='experiments/server_command_runs/stage2_rl_first10k_curve_20260520_205116'
STAGE='Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.005_s2t0.005_s2st0.005/stage2_noise'
NVS="$ART/nvidia_smi_during_rl.csv"
PID=215318
cd "$WORK"
copy_artifacts() {
  local diag="$STAGE/progress/diagnostics"
  [ -f "$diag/episodes.jsonl" ] && cp "$diag/episodes.jsonl" "$ART/episodes.jsonl" || true
  [ -f "$diag/ppo_updates.jsonl" ] && cp "$diag/ppo_updates.jsonl" "$ART/ppo_updates.jsonl" || true
  [ -f "$diag/pareto_frontier.jsonl" ] && cp "$diag/pareto_frontier.jsonl" "$ART/pareto_frontier.jsonl" || true
  [ -f "$diag/pareto_frontier.json" ] && cp "$diag/pareto_frontier.json" "$ART/pareto_frontier.json" || true
  [ -f "$diag/pareto_frontier.html" ] && cp "$diag/pareto_frontier.html" "$ART/pareto_frontier.html" || true
  [ -f "$STAGE/warning.txt" ] && cp "$STAGE/warning.txt" "$ART/warning.txt" || true
  [ -f "$STAGE/pruning_search_log.txt" ] && tail -n 40000 "$STAGE/pruning_search_log.txt" > "$ART/pruning_search_log_tail_source.txt" || true
  for path in \
    "$STAGE/progress/blb_stage2_best_action_full.json" \
    "$STAGE/progress/blb_stage2_best_action_full.md" \
    "$STAGE/progress/blb_stage2_baseline_action_full.json" \
    "$STAGE/progress/blb_stage2_baseline_action_full.md" \
    "$STAGE/progress/blb_stage2_report.md" \
    "$STAGE/progress/blb_stage2_status.json" \
    "$STAGE/progress/blb_stage2_training_curve.png"; do
    [ -f "$path" ] && cp "$path" "$ART/" || true
  done
}
monitor_once() {
  local phase="$1"
  python scripts/stage2_first10k_monitor.py \
    --phase "$phase" \
    --artifact-dir "$ART" \
    --stage2-noise "$STAGE" \
    --nvidia-log "$NVS" \
    --planned 1000 \
    --anchor 60 \
    --rollout 60 \
    --horizon 59 \
    --k-trials 5 \
    --probe-size 256
}
if [ ! -f "$NVS" ]; then printf 'timestamp,gpu_idx,util_pct,mem_used_mib\n' > "$NVS"; fi
(
  while true; do
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader,nounits >> "$NVS" 2>/dev/null || true
    sleep 15
  done
) &
NVS_PID=$!
trap 'kill $NVS_PID 2>/dev/null || true' EXIT
while kill -0 "$PID" 2>/dev/null; do
  copy_artifacts
  set +e
  monitor_once live
  rc=$?
  set -e
  eps=0
  [ -f "$ART/episodes.jsonl" ] && eps=$(wc -l < "$ART/episodes.jsonl" | tr -d ' ')
  echo "[recovered-watchdog] $(date -Is) pid=$PID episodes=$eps monitor_rc=$rc"
  if [ "$rc" -eq 2 ] && [ "$eps" -gt 60 ]; then
    echo "hard failure from monitor; stopping $PID" > "$ART/abort_reason.txt"
    kill -INT "$PID" 2>/dev/null || true
    sleep 60
    kill -TERM "$PID" 2>/dev/null || true
    break
  fi
  sleep 60
done
copy_artifacts
set +e
monitor_once final
final_rc=$?
set -e
echo "[recovered-watchdog] final_rc=$final_rc $(date -Is)"
exit "$final_rc"
