#!/usr/bin/env bash
# Multi-seed BLB Stage-2 RL training driver.
#
# Runs the same preset N times with N different seeds, each isolated in its
# own persistent dir (via the --run-tag flag). After all runs finish, calls
# tools/aggregate_seeds.py to produce a single Markdown summary with
# mean ± std of training-best and final-eval metrics.
#
# Usage:
#   bash tools/run_multi_seed.sh <preset> <seeds_comma> [run_name] [extra_args...]
#
# Examples:
#   # Run 5 seeds with the default preset
#   bash tools/run_multi_seed.sh mrpc-blb-stage2-rl 1,2,3,4,5 myrun
#
#   # Run 3 seeds with extra launcher arg
#   bash tools/run_multi_seed.sh mrpc-blb-stage2-rl 11,22,33 trial2 \
#       --stage2-search-episodes 3000
#
# Notes:
#   - Each seed gets its own persistent dir via --run-tag <RUN>_s<SEED>.
#     Auto-resume works per-seed: re-running the same command (without
#     --fresh) picks up from each seed's ckpt.
#   - Pass --fresh ONCE the first time to confirm overwrite (per seed).
#   - The output of each seed lives at:
#       Parting Chapter/persistent/rl/bert-base/<dataset>/<slug>__<RUN>_s<SEED>/
#     The aggregator (tools/aggregate_seeds.py) walks these and writes:
#       experiments/multi_seed/<RUN>/seed_summary.md

set -euo pipefail

if [ $# -lt 3 ]; then
  cat <<EOF
Usage: bash tools/run_multi_seed.sh <preset> <seeds_comma> <run_name> [extra_launcher_args...]
Example:
  bash tools/run_multi_seed.sh mrpc-blb-stage2-rl 1,2,3,4,5 myrun --fresh
EOF
  exit 1
fi

PRESET="$1"
SEEDS_CSV="$2"
RUN_NAME="$3"
shift 3
EXTRA_ARGS=("$@")

# Sanitize run name: only [a-zA-Z0-9_-]
RUN_NAME_SAFE="$(printf '%s' "$RUN_NAME" | tr -c 'A-Za-z0-9_-' '_' )"

# Parse seeds
IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"
if [ ${#SEEDS[@]} -lt 2 ]; then
  echo "[warn] multi-seed framework with only ${#SEEDS[@]} seed; consider passing >= 3 for significance." >&2
fi

# Output dir for the aggregated summary (separate from the per-seed
# persistent dirs).
SUMMARY_DIR="experiments/multi_seed/${RUN_NAME_SAFE}"
mkdir -p "$SUMMARY_DIR"
SEED_LIST_FILE="$SUMMARY_DIR/seed_list.txt"
: > "$SEED_LIST_FILE"

echo "================================================================"
echo "  BLB Stage-2 RL · multi-seed sweep"
echo "  preset     = $PRESET"
echo "  seeds      = ${SEEDS[*]}"
echo "  run_name   = $RUN_NAME_SAFE"
echo "  extra args = ${EXTRA_ARGS[*]:-(none)}"
echo "  summary dir= $SUMMARY_DIR"
echo "================================================================"

# Run each seed sequentially.
for SEED in "${SEEDS[@]}"; do
  SEED="${SEED// /}"   # strip whitespace
  RUN_TAG="${RUN_NAME_SAFE}_s${SEED}"
  echo
  echo "----------------------------------------------------------------"
  echo "  Seed ${SEED} → run-tag ${RUN_TAG}"
  echo "----------------------------------------------------------------"
  echo "$SEED  $RUN_TAG" >> "$SEED_LIST_FILE"

  # The launcher's --blb-v3-seed overrides BLBStage2TrainConfig.seed
  # (default 42). The --run-tag flag appends to the persistent-dir slug
  # so each seed has its own auto-resumable subtree.
  set +e
  bash llama_7B_LayerImportance.sh run rl \
      --preset "$PRESET" \
      --blb-v3-seed "$SEED" \
      --run-tag "$RUN_TAG" \
      "${EXTRA_ARGS[@]}"
  RC=$?
  set -e

  if [ $RC -ne 0 ]; then
    echo "[warn] seed=${SEED} run exited with code ${RC}. Continuing with next seed." >&2
    echo "${SEED}  ${RUN_TAG}  EXIT=${RC}" >> "$SUMMARY_DIR/_failures.txt"
  fi
done

echo
echo "================================================================"
echo "  All seeds done. Aggregating ..."
echo "================================================================"
python3 tools/aggregate_seeds.py \
    --run-name "$RUN_NAME_SAFE" \
    --seed-list "$SEED_LIST_FILE" \
    --output-dir "$SUMMARY_DIR" \
    || echo "[warn] aggregation failed; per-seed run dirs are still intact." >&2

echo
echo "Summary written to: $SUMMARY_DIR/seed_summary.md"
