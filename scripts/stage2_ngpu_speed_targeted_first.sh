#!/usr/bin/env bash
set -euo pipefail

# Targeted-first Stage-2 N-GPU gate.
#
# Run the most likely 5-GPU candidates first with the full strict A/B harness.
# If any candidate reaches TARGET_MIN_SPEEDUP with episode + PPO equality
# passing, stop early. Otherwise fall back to the broader autotune.

TS="$(date +%Y%m%d_%H%M%S)"
TARGET_ID="${TARGET_ID:-stage2_ngpu_speed_targeted_first_${TS}}"
TARGET_ROOT="${TARGET_ROOT:-experiments/server_command_runs/${TARGET_ID}}"
EPISODES_AB="${EPISODES_AB:-600}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-18000}"
TARGET_MIN_SPEEDUP="${TARGET_MIN_SPEEDUP:-3.4}"
TARGET_CANDIDATES="${TARGET_CANDIDATES:-1:worker:1}"
FALLBACK_AUTOTUNE="${FALLBACK_AUTOTUNE:-0}"

ONE_DEVS="${ONE_DEVS:-0}"
MANY_DEVS="${MANY_DEVS:-0,1,2,3,4}"
ONE_WORKERS_PER_DEVICE="${ONE_WORKERS_PER_DEVICE:-1}"
REQUIRE_IDLE_GPUS="${REQUIRE_IDLE_GPUS:-1}"
IDLE_MEM_MIB="${IDLE_MEM_MIB:-2048}"

candidate_fatal() {
  local candidate="$1"
  echo "[targeted][FATAL] malformed TARGET_CANDIDATES entry: ${candidate}" >&2
  echo "[targeted][FATAL] expected many_wpd:policy_device:dynamic_assignment, e.g. 1:worker:1" >&2
  exit 24
}

validate_candidate_entry() {
  local candidate="$1"
  local many_wpd="" policy_device="" dynamic_assignment="" extra=""
  IFS=':' read -r many_wpd policy_device dynamic_assignment extra <<< "$candidate"
  if [ -z "$many_wpd" ] || [ -z "$policy_device" ] || [ -z "$dynamic_assignment" ] || [ -n "$extra" ]; then
    candidate_fatal "$candidate"
  fi
  case "$many_wpd" in
    *[!0-9]*)
      candidate_fatal "$candidate"
      ;;
  esac
  if [ "$many_wpd" -lt 1 ]; then
    candidate_fatal "$candidate"
  fi
  case "$policy_device" in
    worker|cpu)
      ;;
    *)
      candidate_fatal "$candidate"
      ;;
  esac
  case "$dynamic_assignment" in
    0|1)
      ;;
    *)
      candidate_fatal "$candidate"
      ;;
  esac
}

if [ -z "${TARGET_CANDIDATES//[[:space:]]/}" ]; then
  candidate_fatal "<empty>"
fi
for candidate in ${TARGET_CANDIDATES}; do
  validate_candidate_entry "$candidate"
done

mkdir -p "$TARGET_ROOT"
exec > >(tee "${TARGET_ROOT}/stage2_ngpu_speed_targeted_first_stdout.log") 2>&1

summary="${TARGET_ROOT}/targeted_summary.tsv"
printf 'run_id\tstatus\tmany_wpd\tpolicy_device\tdynamic_assignment\tone_reused\tspeedup\tone_eph\tmany_eph\tartifact_dir\n' > "$summary"

echo "[targeted] root=${TARGET_ROOT}"
echo "[targeted] episodes=${EPISODES_AB} target_min_speedup=${TARGET_MIN_SPEEDUP}"
echo "[targeted] candidates=${TARGET_CANDIDATES}"

extract_field() {
  local verdict="$1"
  local label="$2"
  grep -F "${label}:" "$verdict" \
    | tail -1 \
    | cut -d: -f2- \
    | sed -E 's/^[[:space:]]*//; s/x$//' || true
}

speed_meets_target() {
  local speed="$1"
  python3 - "$speed" "$TARGET_MIN_SPEEDUP" <<'PY'
import sys
try:
    speed = float(sys.argv[1])
    target = float(sys.argv[2])
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if speed >= target else 1)
PY
}

declare -A reuse_one_episodes_by_policy=()
declare -A reuse_one_wall_by_policy=()
declare -A reuse_one_log_by_policy=()
declare -A reuse_one_ppo_by_policy=()

best_tsv="${TARGET_ROOT}/best_candidate.tsv"
best_env="${TARGET_ROOT}/best_candidate.env"

for candidate in ${TARGET_CANDIDATES}; do
  IFS=':' read -r many_wpd policy_device dynamic_assignment <<< "$candidate"
  run_id="${TARGET_ID}_wpd${many_wpd}_policy${policy_device}_dyn${dynamic_assignment}"
  artifact_dir="${TARGET_ROOT}/${run_id}"
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
  echo "[targeted] candidate=${run_id}"
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
    status="ok"
    if [ -z "${reuse_one_episodes_by_policy[$policy_device]:-}" ]; then
      reuse_one_episodes_by_policy[$policy_device]="${artifact_dir}/one_episodes.jsonl"
      reuse_one_wall_by_policy[$policy_device]="${artifact_dir}/one_wall_seconds.txt"
      reuse_one_log_by_policy[$policy_device]="${artifact_dir}/one_launch.log"
      if [ -f "${artifact_dir}/one_ppo_updates.jsonl" ]; then
        reuse_one_ppo_by_policy[$policy_device]="${artifact_dir}/one_ppo_updates.jsonl"
      fi
      echo "[targeted] cached 1GPU baseline for policy_device=${policy_device}"
    fi
  else
    speedup=""
    one_eph=""
    many_eph=""
    status="rc${rc}"
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$run_id" "$status" "$many_wpd" "$policy_device" "$dynamic_assignment" "$one_reused" \
    "$speedup" "$one_eph" "$many_eph" "$artifact_dir" >> "$summary"
  echo "[targeted] candidate=${run_id} status=${status} speedup=${speedup:-n/a}"

  if [ "$status" = "ok" ] && speed_meets_target "$speedup"; then
    {
      head -n 1 "$summary"
      tail -n 1 "$summary"
    } > "$best_tsv"
    awk -F'\t' 'NR == 2 {
      print "MANY_WORKERS_PER_DEVICE=" $3
      print "POLICY_DEVICE=" $4
      print "DYNAMIC_ASSIGNMENT=" $5
      print "BEST_SPEEDUP=" $7
      print "BEST_ONE_EPH=" $8
      print "BEST_MANY_EPH=" $9
      print "BEST_ARTIFACT_DIR=\"" $10 "\""
    }' "$best_tsv" > "$best_env"
    echo "[targeted] target met; best candidate:"
    cat "$best_tsv"
    echo "[DONE] Stage-2 targeted-first gate complete: ${TARGET_ROOT}"
    exit 0
  fi
done

echo "[targeted] no targeted candidate reached ${TARGET_MIN_SPEEDUP}x"
if [ "$FALLBACK_AUTOTUNE" != "1" ]; then
  echo "[targeted][FATAL] fallback disabled"
  exit 23
fi

echo ""
echo "================================================================================"
echo "[targeted] fallback: broader autotune"
echo "================================================================================"
AUTOTUNE_ID="${TARGET_ID}_fallback_autotune" \
AUTOTUNE_ROOT="${TARGET_ROOT}/fallback_autotune" \
PILOT_EPISODES="${PILOT_EPISODES:-180}" \
FINAL_EPISODES="${FINAL_EPISODES:-${EPISODES_AB}}" \
FINAL_TOP_K="${FINAL_TOP_K:-3}" \
TIMEOUT_SECONDS="$TIMEOUT_SECONDS" \
ONE_DEVS="$ONE_DEVS" \
MANY_DEVS="$MANY_DEVS" \
ONE_WORKERS_PER_DEVICE="$ONE_WORKERS_PER_DEVICE" \
SWEEP_MANY_WPD_LIST="${SWEEP_MANY_WPD_LIST:-1}" \
SWEEP_POLICY_DEVICE_LIST="${SWEEP_POLICY_DEVICE_LIST:-worker}" \
SWEEP_DYNAMIC_LIST="${SWEEP_DYNAMIC_LIST:-1}" \
REQUIRE_IDLE_GPUS="$REQUIRE_IDLE_GPUS" \
IDLE_MEM_MIB="$IDLE_MEM_MIB" \
bash scripts/stage2_ngpu_speed_autotune.sh

if [ -f "${TARGET_ROOT}/fallback_autotune/best_candidate.tsv" ]; then
  cp "${TARGET_ROOT}/fallback_autotune/best_candidate.tsv" "$best_tsv"
fi
if [ -f "${TARGET_ROOT}/fallback_autotune/best_candidate.env" ]; then
  cp "${TARGET_ROOT}/fallback_autotune/best_candidate.env" "$best_env"
fi
echo "[DONE] Stage-2 targeted-first fallback complete: ${TARGET_ROOT}"
