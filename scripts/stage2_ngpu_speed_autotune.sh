#!/usr/bin/env bash
set -euo pipefail

# Two-stage real Stage-2 N-GPU autotune.
#
# Stage 1 runs a short strict-equality sweep across scheduling knobs to find
# promising candidates quickly. Stage 2 reruns the fastest candidates with the
# real A/B episode count. This keeps the equality oracle and wall-clock verdict
# in stage2_ngpu_speed_ab.sh while avoiding a full-length run for every point
# in the grid.

TS="$(date +%Y%m%d_%H%M%S)"
AUTOTUNE_ID="${AUTOTUNE_ID:-stage2_ngpu_speed_autotune_${TS}}"
AUTOTUNE_ROOT="${AUTOTUNE_ROOT:-experiments/server_command_runs/${AUTOTUNE_ID}}"
PILOT_EPISODES="${PILOT_EPISODES:-180}"
FINAL_EPISODES="${FINAL_EPISODES:-600}"
FINAL_TOP_K="${FINAL_TOP_K:-3}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-18000}"

ONE_DEVS="${ONE_DEVS:-0}"
MANY_DEVS="${MANY_DEVS:-0,1,2,3,4}"
ONE_WORKERS_PER_DEVICE="${ONE_WORKERS_PER_DEVICE:-1}"
SWEEP_MANY_WPD_LIST="${SWEEP_MANY_WPD_LIST:-1}"
SWEEP_POLICY_DEVICE_LIST="${SWEEP_POLICY_DEVICE_LIST:-worker}"
SWEEP_DYNAMIC_LIST="${SWEEP_DYNAMIC_LIST:-1}"
REQUIRE_IDLE_GPUS="${REQUIRE_IDLE_GPUS:-1}"
IDLE_MEM_MIB="${IDLE_MEM_MIB:-2048}"

mkdir -p "$AUTOTUNE_ROOT"
exec > >(tee "${AUTOTUNE_ROOT}/stage2_ngpu_speed_autotune_stdout.log") 2>&1

echo "[autotune] root=${AUTOTUNE_ROOT}"
echo "[autotune] pilot_episodes=${PILOT_EPISODES} final_episodes=${FINAL_EPISODES} top_k=${FINAL_TOP_K}"
echo "[autotune] many_wpd_list=${SWEEP_MANY_WPD_LIST}"
echo "[autotune] policy_device_list=${SWEEP_POLICY_DEVICE_LIST}"
echo "[autotune] dynamic_list=${SWEEP_DYNAMIC_LIST}"

extract_field() {
  local verdict="$1"
  local label="$2"
  grep -F "${label}:" "$verdict" \
    | tail -1 \
    | cut -d: -f2- \
    | sed -E 's/^[[:space:]]*//; s/x$//' || true
}

extract_ngpu_timing_total() {
  local verdict="$1"
  local key="$2"
  awk -v target="${key}:" '
    /^NGPU rollout timing log:/ { in_ngpu=1; next }
    /^[A-Za-z0-9]+ rollout timing log:/ { if ($1 != "NGPU") in_ngpu=0 }
    in_ngpu && $1 == target {
      for (i = 1; i <= NF; i++) {
        if ($i ~ /^total_s=/) {
          sub(/^total_s=/, "", $i)
          print $i
          exit
        }
      }
    }
  ' "$verdict" || true
}

pilot_root="${AUTOTUNE_ROOT}/pilot"
echo ""
echo "================================================================================"
echo "[autotune] stage 1: pilot sweep"
echo "================================================================================"
SWEEP_ID="${AUTOTUNE_ID}_pilot" \
SWEEP_ROOT="$pilot_root" \
EPISODES_AB="$PILOT_EPISODES" \
TIMEOUT_SECONDS="$TIMEOUT_SECONDS" \
ONE_DEVS="$ONE_DEVS" \
MANY_DEVS="$MANY_DEVS" \
ONE_WORKERS_PER_DEVICE="$ONE_WORKERS_PER_DEVICE" \
SWEEP_MANY_WPD_LIST="$SWEEP_MANY_WPD_LIST" \
SWEEP_POLICY_DEVICE_LIST="$SWEEP_POLICY_DEVICE_LIST" \
SWEEP_DYNAMIC_LIST="$SWEEP_DYNAMIC_LIST" \
REQUIRE_IDLE_GPUS="$REQUIRE_IDLE_GPUS" \
IDLE_MEM_MIB="$IDLE_MEM_MIB" \
bash scripts/stage2_ngpu_speed_sweep.sh

pilot_summary="${pilot_root}/sweep_summary.tsv"
top_candidates="${AUTOTUNE_ROOT}/pilot_top_candidates.tsv"
awk -F'\t' '
  NR == 1 { next }
  $2 == "ok" && $7 != "" {
    print $7 "\t" $3 "\t" $4 "\t" $5 "\t" $12
  }
' "$pilot_summary" | sort -t $'\t' -k1,1gr | head -n "$FINAL_TOP_K" > "$top_candidates"

if [ ! -s "$top_candidates" ]; then
  echo "[autotune][FATAL] pilot sweep produced no successful candidates"
  exit 21
fi

echo "[autotune] pilot top candidates:"
cat "$top_candidates"

final_root="${AUTOTUNE_ROOT}/final"
mkdir -p "$final_root"
final_summary="${final_root}/final_summary.tsv"
printf 'run_id\tstatus\tpilot_speedup\tmany_wpd\tpolicy_device\tdynamic_assignment\tone_reused\tspeedup\tone_eph\tmany_eph\tngpu_collect_total_s\tngpu_sync_total_s\tartifact_dir\n' > "$final_summary"

declare -A reuse_one_episodes_by_policy=()
declare -A reuse_one_wall_by_policy=()
declare -A reuse_one_log_by_policy=()
declare -A reuse_one_ppo_by_policy=()

rank=0
while IFS=$'\t' read -r pilot_speed many_wpd policy_device dynamic_assignment _pilot_artifact; do
  rank=$((rank + 1))
  run_id="${AUTOTUNE_ID}_final_rank${rank}_wpd${many_wpd}_policy${policy_device}_dyn${dynamic_assignment}"
  artifact_dir="${final_root}/${run_id}"
  reuse_one_episodes="${reuse_one_episodes_by_policy[$policy_device]:-}"
  reuse_one_wall="${reuse_one_wall_by_policy[$policy_device]:-}"
  reuse_one_log="${reuse_one_log_by_policy[$policy_device]:-}"
  reuse_one_ppo="${reuse_one_ppo_by_policy[$policy_device]:-}"
  one_reused="0"
  if [ -n "$reuse_one_episodes" ]; then
    one_reused="1"
  fi

  echo ""
  echo "================================================================================"
  echo "[autotune] stage 2: final candidate rank=${rank} pilot_speed=${pilot_speed} wpd=${many_wpd} policy=${policy_device} dyn=${dynamic_assignment}"
  echo "================================================================================"

  set +e
  RUN_ID="$run_id" \
  ARTIFACT_DIR="$artifact_dir" \
  REUSE_ONE_EPISODES="$reuse_one_episodes" \
  REUSE_ONE_WALL="$reuse_one_wall" \
  REUSE_ONE_LOG="$reuse_one_log" \
  REUSE_ONE_PPO="$reuse_one_ppo" \
  EPISODES_AB="$FINAL_EPISODES" \
  TIMEOUT_SECONDS="$TIMEOUT_SECONDS" \
  ONE_DEVS="$ONE_DEVS" \
  MANY_DEVS="$MANY_DEVS" \
  ONE_WORKERS_PER_DEVICE="$ONE_WORKERS_PER_DEVICE" \
  MANY_WORKERS_PER_DEVICE="$many_wpd" \
  POLICY_DEVICE="$policy_device" \
  DYNAMIC_ASSIGNMENT="$dynamic_assignment" \
  REQUIRE_IDLE_GPUS="$REQUIRE_IDLE_GPUS" \
  IDLE_MEM_MIB="$IDLE_MEM_MIB" \
  MIN_SPEEDUP=0 \
    bash scripts/stage2_ngpu_speed_ab.sh
  rc=$?
  set -e

  verdict="${artifact_dir}/stage2_ngpu_gate_verdict.txt"
  if [ "$rc" -eq 0 ] && [ -f "$verdict" ]; then
    speedup="$(extract_field "$verdict" "speedup")"
    one_eph="$(extract_field "$verdict" "1GPU episodes/hour")"
    many_eph="$(extract_field "$verdict" "NGPU episodes/hour")"
    ngpu_collect="$(extract_ngpu_timing_total "$verdict" "collect_s")"
    ngpu_sync="$(extract_ngpu_timing_total "$verdict" "sync_s")"
    status="ok"
    if [ -z "${reuse_one_episodes_by_policy[$policy_device]:-}" ]; then
      reuse_one_episodes_by_policy[$policy_device]="${artifact_dir}/one_episodes.jsonl"
      reuse_one_wall_by_policy[$policy_device]="${artifact_dir}/one_wall_seconds.txt"
      reuse_one_log_by_policy[$policy_device]="${artifact_dir}/one_launch.log"
      if [ -f "${artifact_dir}/one_ppo_updates.jsonl" ]; then
        reuse_one_ppo_by_policy[$policy_device]="${artifact_dir}/one_ppo_updates.jsonl"
      fi
      echo "[autotune] cached final 1GPU baseline for policy_device=${policy_device}"
    fi
  else
    speedup=""
    one_eph=""
    many_eph=""
    ngpu_collect=""
    ngpu_sync=""
    status="rc${rc}"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$run_id" "$status" "$pilot_speed" "$many_wpd" "$policy_device" "$dynamic_assignment" "$one_reused" \
    "$speedup" "$one_eph" "$many_eph" "$ngpu_collect" "$ngpu_sync" "$artifact_dir" >> "$final_summary"
done < "$top_candidates"

best_tsv="${AUTOTUNE_ROOT}/best_candidate.tsv"
best_env="${AUTOTUNE_ROOT}/best_candidate.env"
awk -F'\t' '
  NR == 1 { header = $0; next }
  $2 == "ok" && $8 != "" {
    speed = $8 + 0.0
    if (!seen || speed > best_speed) {
      seen = 1
      best_speed = speed
      best = $0
    }
  }
  END {
    if (seen) {
      print header
      print best
    }
  }
' "$final_summary" > "$best_tsv"

if [ -s "$best_tsv" ] && [ "$(wc -l < "$best_tsv")" -ge 2 ]; then
  awk -F'\t' 'NR == 2 {
    print "MANY_WORKERS_PER_DEVICE=" $4
    print "POLICY_DEVICE=" $5
    print "DYNAMIC_ASSIGNMENT=" $6
    print "BEST_SPEEDUP=" $8
    print "BEST_ONE_EPH=" $9
    print "BEST_MANY_EPH=" $10
    print "BEST_ARTIFACT_DIR=\"" $13 "\""
  }' "$best_tsv" > "$best_env"
  echo ""
  echo "================================================================================"
  echo "[autotune] best final candidate"
  echo "================================================================================"
  cat "$best_tsv"
  echo "[autotune] best env=${best_env}"
else
  rm -f "$best_tsv" "$best_env"
  echo "[autotune][FATAL] final stage produced no successful candidates"
  exit 22
fi

echo "[DONE] Stage-2 N-GPU autotune complete: ${AUTOTUNE_ROOT}"
