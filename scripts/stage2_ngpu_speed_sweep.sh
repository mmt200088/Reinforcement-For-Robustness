#!/usr/bin/env bash
set -euo pipefail

# Sweep real Stage-2 1GPU-vs-5GPU gates across a small set of rollout
# scheduling knobs. Each candidate delegates to stage2_ngpu_speed_ab.sh, so the
# equality oracle, idle-GPU guard, and wall-clock throughput verdict stay in one
# place.

TS="$(date +%Y%m%d_%H%M%S)"
SWEEP_ID="${SWEEP_ID:-stage2_ngpu_speed_sweep_${TS}}"
SWEEP_ROOT="${SWEEP_ROOT:-experiments/server_command_runs/${SWEEP_ID}}"
EPISODES_AB="${EPISODES_AB:-600}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-14400}"
ONE_DEVS="${ONE_DEVS:-0}"
MANY_DEVS="${MANY_DEVS:-0,1,2,3,4}"
ONE_WORKERS_PER_DEVICE="${ONE_WORKERS_PER_DEVICE:-1}"

# Defaults use the fastest validated profile-off A/B candidate:
# workers_per_device=1, policy on each worker GPU, dynamic assignment enabled.
# Override these lists when intentionally re-running a broader exploration.
SWEEP_MANY_WPD_LIST="${SWEEP_MANY_WPD_LIST:-1}"
SWEEP_POLICY_DEVICE_LIST="${SWEEP_POLICY_DEVICE_LIST:-worker}"
SWEEP_DYNAMIC_LIST="${SWEEP_DYNAMIC_LIST:-1}"
REQUIRE_IDLE_GPUS="${REQUIRE_IDLE_GPUS:-1}"
IDLE_MEM_MIB="${IDLE_MEM_MIB:-2048}"

mkdir -p "$SWEEP_ROOT"
exec > >(tee "${SWEEP_ROOT}/stage2_ngpu_speed_sweep_stdout.log") 2>&1

summary="${SWEEP_ROOT}/sweep_summary.tsv"
printf 'run_id\tstatus\tmany_wpd\tpolicy_device\tdynamic_assignment\tone_reused\tspeedup\tone_eph\tmany_eph\tngpu_collect_total_s\tngpu_sync_total_s\tartifact_dir\n' > "$summary"

echo "[sweep] root=${SWEEP_ROOT}"
echo "[sweep] episodes=${EPISODES_AB} one=${ONE_DEVS} many=${MANY_DEVS}"
echo "[sweep] many_wpd_list=${SWEEP_MANY_WPD_LIST}"
echo "[sweep] policy_device_list=${SWEEP_POLICY_DEVICE_LIST}"
echo "[sweep] dynamic_list=${SWEEP_DYNAMIC_LIST}"

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

declare -A reuse_one_episodes_by_policy=()
declare -A reuse_one_wall_by_policy=()
declare -A reuse_one_log_by_policy=()
declare -A reuse_one_ppo_by_policy=()

for many_wpd in ${SWEEP_MANY_WPD_LIST}; do
  for policy_device in ${SWEEP_POLICY_DEVICE_LIST}; do
    for dynamic_assignment in ${SWEEP_DYNAMIC_LIST}; do
      run_id="${SWEEP_ID}_wpd${many_wpd}_policy${policy_device}_dyn${dynamic_assignment}"
      artifact_dir="${SWEEP_ROOT}/${run_id}"
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
      echo "[sweep] candidate=${run_id}"
      if [ "$one_reused" = "1" ]; then
        echo "[sweep] reusing 1GPU baseline for policy_device=${policy_device}: ${reuse_one_episodes}"
      fi
      echo "================================================================================"
      set +e
      RUN_ID="$run_id" \
      ARTIFACT_DIR="$artifact_dir" \
      REUSE_ONE_EPISODES="$reuse_one_episodes" \
      REUSE_ONE_WALL="$reuse_one_wall" \
      REUSE_ONE_LOG="$reuse_one_log" \
      REUSE_ONE_PPO="$reuse_one_ppo" \
      EPISODES_AB="$EPISODES_AB" \
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
          echo "[sweep] cached 1GPU baseline for policy_device=${policy_device}"
        fi
      else
        speedup=""
        one_eph=""
        many_eph=""
        ngpu_collect=""
        ngpu_sync=""
        status="rc${rc}"
      fi
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$run_id" "$status" "$many_wpd" "$policy_device" "$dynamic_assignment" "$one_reused" \
        "$speedup" "$one_eph" "$many_eph" "$ngpu_collect" "$ngpu_sync" "$artifact_dir" >> "$summary"
      echo "[sweep] candidate=${run_id} status=${status} speedup=${speedup:-n/a}"

      # If the idle guard fires, every remaining candidate would fail the same
      # way. Stop early so we do not spam logs while a long training run owns
      # the GPUs.
      if [ "$rc" -eq 20 ]; then
        echo "[sweep] idle gate failed; stopping sweep early"
        echo "[sweep] summary=${summary}"
        exit 20
      fi
    done
  done
done

echo ""
echo "================================================================================"
echo "[sweep] summary"
echo "================================================================================"
cat "$summary"
best_tsv="${SWEEP_ROOT}/best_candidate.tsv"
best_env="${SWEEP_ROOT}/best_candidate.env"
awk -F'\t' '
  NR == 1 { header = $0; next }
  $2 == "ok" && $7 != "" {
    speed = $7 + 0.0
    if (!seen || speed > best_speed) {
      seen = 1
      best_speed = speed
      best = $0
      wpd = $3
      policy = $4
      dyn = $5
      artifact = $12
    }
  }
  END {
    if (seen) {
      print header
      print best
    }
  }
' "$summary" > "$best_tsv"
if [ -s "$best_tsv" ] && [ "$(wc -l < "$best_tsv")" -ge 2 ]; then
  awk -F'\t' 'NR == 2 {
    print "MANY_WORKERS_PER_DEVICE=" $3
    print "POLICY_DEVICE=" $4
    print "DYNAMIC_ASSIGNMENT=" $5
    print "BEST_SPEEDUP=" $7
    print "BEST_ONE_EPH=" $8
    print "BEST_MANY_EPH=" $9
    print "BEST_ARTIFACT_DIR=\"" $12 "\""
  }' "$best_tsv" > "$best_env"
  echo "[sweep] best candidate:"
  cat "$best_tsv"
  echo "[sweep] best env: ${best_env}"
else
  rm -f "$best_tsv" "$best_env"
  echo "[sweep] no successful candidate found"
fi
echo "[DONE] Stage-2 N-GPU speed sweep complete: ${summary}"
