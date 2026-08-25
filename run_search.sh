#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

usage() {
  cat <<'EOF'
Usage:
  bash run_search.sh run rl --preset NAME [options]
  bash run_search.sh run bo_rf [options]
  bash run_search.sh run greedy [options]
  bash run_search.sh run coinn_ga [options]
  bash run_search.sh eval --preset NAME [options]
  bash run_search.sh --list-presets

Supported profiles:
  --dataset mrpc|rte|sst2
  --model-type bert-base|bert-large

Run control:
  --mode stage1-only|stage2-only
  --fresh
  --persistent-root PATH
  --run-tag NAME
  --logfile FILE
  --batch-size N
  --stage2-inference-batch-size N
  --dry-run

Stage 1 PPO:
  --stage1-search-episodes N
  --stage1-entropy-stop-threshold FLOAT
  --stage1-search-lr FLOAT
  --ppo-update-interval N
  --stage1-accuracy-tolerance FLOAT
  --stage1-rl-devices auto|DEVICE_LIST

Stage 2 layerwise PPO:
  --stage2-search-episodes N
  --stage2-search-lr FLOAT
  --stage2-rollout-size N
  --stage2-save-interval N
  --stage2-eval-interval N
  --stage2-fixed-config-source all4|stage1_result|json|manual
  --stage2-fixed-config PATH
  --stage2-manual-gelu JSON_ARRAY
  --stage2-manual-softmax JSON_ARRAY
  --stage2-limit-tolerance FLOAT
  --stage2-stability-multiplier FLOAT
  --stage2-k-trials N
  --stage2-probe-size N
  --blb-v3-reward-devices auto|DEVICE_LIST
  --elastic-gpu-mode auto|off
  --elastic-gpu-recovery-interval SEC
  --elastic-gpu-max-restarts N

Comparators:
  --comparator-stage1-only
  --comparator-smoke

Final evaluation is a separate command:
  bash run_search.sh eval --preset NAME
EOF
}

fail() {
  printf 'Error: %s\n' "$1" >&2
  exit 1
}

require_value() {
  [ "$#" -ge 2 ] || fail "option $1 requires a value"
}

is_nonnegative_integer() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

is_positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_nonnegative_number() {
  [[ "$1" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$ ]]
}

is_positive_number() {
  is_nonnegative_number "$1" && awk -v value="$1" 'BEGIN { exit !((value + 0) > 0) }'
}

list_presets() {
  local file
  for file in "$ROOT_DIR"/configs/presets/*.conf; do
    [ -f "$file" ] || continue
    basename "$file" .conf
  done | sort
}

load_preset() {
  local name="$1"
  local file="$ROOT_DIR/configs/presets/${name}.conf"
  local line
  [ -f "$file" ] || fail "preset not found: $name"
  while IFS= read -r line; do
    line="$(printf '%s' "$line" | sed 's/#.*//' | xargs)"
    [ -n "$line" ] || continue
    read -r -a fields <<< "$line"
    PRESET_ARGS+=("${fields[@]}")
  done < "$file"
}

normalize_algorithm() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    rl|ppo) printf 'rl\n' ;;
    bo|bo-rf|bo_rf|bayesian) printf 'bo_rf\n' ;;
    greedy|greedy-search|greedy_search) printf 'greedy\n' ;;
    coinn|coinn-ga|coinn_ga) printf 'coinn_ga\n' ;;
    *) fail "run supports rl, bo_rf, greedy, and coinn_ga" ;;
  esac
}

lowercase() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

reject_comparator_overrides() {
  local arg
  for arg in "$@"; do
    case "$arg" in
      --algorithm|--algorithm=*|--search-algorithm|--search-algorithm=*|\
      --mode|--mode=*|--batch-size|--batch-size=*|\
      --stage2-inference-batch-size|--stage2-inference-batch-size=*|\
      --random-seed|--random-seed=*|--blb-v3-seed|--blb-v3-seed=*|\
      --stage1-accuracy-tolerance|--stage1-accuracy-tolerance=*|\
      --stage2-limit-tolerance|--stage2-limit-tolerance=*|\
      --stage2-stability-multiplier|--stage2-stability-multiplier=*|\
      --stage2-k-trials|--stage2-k-trials=*|\
      --stage2-probe-size|--stage2-probe-size=*|\
      --stage2-fixed-config-source|--stage2-fixed-config-source=*|\
      --blb-v3-search-backend|--blb-v3-search-backend=*|\
      --blb-v3-search-evaluation-budget|--blb-v3-search-evaluation-budget=*|\
      --blb-v3-search-initial-design-size|--blb-v3-search-initial-design-size=*|\
      --blb-v3-search-candidate-pool-size|--blb-v3-search-candidate-pool-size=*|\
      --blb-v3-search-population-size|--blb-v3-search-population-size=*|\
      --blb-v3-search-patience-generations|--blb-v3-search-patience-generations=*|\
      --blb-v3-search-rf-n-estimators|--blb-v3-search-rf-n-estimators=*|\
      --blb-v3-search-rf-min-samples-leaf|--blb-v3-search-rf-min-samples-leaf=*)
        fail "comparator contracts do not allow overriding $arg"
        ;;
    esac
  done
}

if [ "${1:-}" = "eval" ]; then
  shift
  exec python3 -m rfr.cli.evaluate "$@"
fi

if [ "${1:-}" = "--list-presets" ]; then
  list_presets
  exit 0
fi

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

[ "${1:-}" = "run" ] || { usage >&2; exit 1; }
[ "$#" -ge 2 ] || fail "run requires an algorithm"
ALGORITHM="$(normalize_algorithm "$2")"
shift 2

RAW_CLI_ARGS=("$@")
if [ "$ALGORITHM" != "rl" ]; then
  reject_comparator_overrides "${RAW_CLI_ARGS[@]}"
fi

PRESET_ARGS=()
CLI_ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --preset)
      require_value "$@"
      load_preset "$2"
      shift 2
      ;;
    *)
      CLI_ARGS+=("$1")
      shift
      ;;
  esac
done
set -- "${PRESET_ARGS[@]}" "${CLI_ARGS[@]}"

DATASET="mrpc"
MODEL_TYPE="bert-base"
MODE=""
LOGFILE="training.log"
BATCH_SIZE="128"
STAGE2_INFERENCE_BATCH_SIZE="64"
PERSISTENT_ROOT="Parting Chapter/persistent"
RUN_TAG=""
RESUME_FROM=""
STAGE1_RUN_ID=""
FRESH="false"
DRY_RUN="false"

STAGE1_EPISODES="0"
STAGE1_ENTROPY_STOP_THRESHOLD="0.1"
STAGE1_LR="2e-5"
PPO_UPDATE_INTERVAL="120"
STAGE1_ACCURACY_TOLERANCE="0.001"
STAGE1_RL_DEVICES=""

STAGE2_EPISODES="150000"
STAGE2_LR="5e-5"
STAGE2_ROLLOUT_SIZE="120"
STAGE2_SAVE_INTERVAL="200"
STAGE2_EVAL_INTERVAL="100"
STAGE2_FIXED_CONFIG_SOURCE="all4"
STAGE2_FIXED_CONFIG=""
STAGE2_MANUAL_GELU=""
STAGE2_MANUAL_SOFTMAX=""
STAGE2_LIMIT_TOLERANCE="0.001"
STAGE2_STABILITY_TOLERANCE="1.2"
STAGE2_STABILITY_MULTIPLIER="2.0"
STAGE2_COMMUNICATION_IMPORTANCE_RATIO="1.0"
STAGE2_K_TRIALS="3"
STAGE2_PROBE_SIZE="256"
REWARD_DEVICES=""
ELASTIC_GPU_MODE="auto"
ELASTIC_GPU_RECOVERY_INTERVAL="60"
ELASTIC_GPU_MAX_RESTARTS="8"

RANDOM_SEED="42"
FINAL_EVAL_PRESET="default"
SKIP_FINAL_EVAL="false"
COMPARATOR_SMOKE="false"
COMPARATOR_STAGE1_ONLY="false"

BASELINE_GROUPS="5"
BASELINE_TRIALS_PER_GROUP="3"
CALIBRATE_BASELINE_SAMPLES="8"
CONSTRAINT_BOOTSTRAP_SAMPLES="4096"
ONLINE_CONSTRAINT_PROBABILITY="0.50"
PROMOTION_CONSTRAINT_PROBABILITY="0.80"
FINAL_CONSTRAINT_PROBABILITY="0.95"
PROMOTION_VALIDATION_TRIALS="15"
FINAL_SELECTION_TOP_N="20"
FINAL_SELECTION_VALIDATION_TRIALS="15"
MIN_CONVERGENCE_EPISODES="90000"
CONVERGENCE_PATIENCE_UPDATES="100"
ONLINE_K_TRIALS="3"
TERMINAL_EVAL_BATCH_SIZE="4"

SEARCH_BACKEND="ppo"
SEARCH_EVALUATION_BUDGET="0"
SEARCH_INITIAL_DESIGN_SIZE="64"
SEARCH_CANDIDATE_POOL_SIZE="2048"
SEARCH_POPULATION_SIZE="64"
SEARCH_PATIENCE="100"
SEARCH_MUTATION_MAX_COORDINATES="3"
SEARCH_RF_N_ESTIMATORS="128"
SEARCH_RF_MIN_SAMPLES_LEAF="2"
SEARCH_FULL_VALIDATION="true"

while [ "$#" -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --dataset) require_value "$@"; DATASET="$2"; shift 2 ;;
    --model-type) require_value "$@"; MODEL_TYPE="$2"; shift 2 ;;
    --mode) require_value "$@"; MODE="$2"; shift 2 ;;
    --logfile) require_value "$@"; LOGFILE="$2"; shift 2 ;;
    --batch-size) require_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
    --stage2-inference-batch-size) require_value "$@"; STAGE2_INFERENCE_BATCH_SIZE="$2"; shift 2 ;;
    --persistent-root) require_value "$@"; PERSISTENT_ROOT="$2"; shift 2 ;;
    --run-tag) require_value "$@"; RUN_TAG="$2"; shift 2 ;;
    --resume-from) require_value "$@"; RESUME_FROM="$2"; shift 2 ;;
    --stage1-run-id) require_value "$@"; STAGE1_RUN_ID="$2"; shift 2 ;;
    --fresh) FRESH="true"; shift ;;
    --fresh-start) FRESH="true"; shift ;;
    --dry-run) DRY_RUN="true"; shift ;;
    --stage1-search-episodes) require_value "$@"; STAGE1_EPISODES="$2"; shift 2 ;;
    --stage1-entropy-stop-threshold) require_value "$@"; STAGE1_ENTROPY_STOP_THRESHOLD="$2"; shift 2 ;;
    --stage1-search-lr) require_value "$@"; STAGE1_LR="$2"; shift 2 ;;
    --ppo-update-interval) require_value "$@"; PPO_UPDATE_INTERVAL="$2"; shift 2 ;;
    --stage1-accuracy-tolerance) require_value "$@"; STAGE1_ACCURACY_TOLERANCE="$2"; shift 2 ;;
    --stage1-rl-devices) require_value "$@"; STAGE1_RL_DEVICES="$2"; shift 2 ;;
    --stage2-search-episodes) require_value "$@"; STAGE2_EPISODES="$2"; shift 2 ;;
    --stage2-search-lr) require_value "$@"; STAGE2_LR="$2"; shift 2 ;;
    --stage2-rollout-size) require_value "$@"; STAGE2_ROLLOUT_SIZE="$2"; shift 2 ;;
    --stage2-save-interval) require_value "$@"; STAGE2_SAVE_INTERVAL="$2"; shift 2 ;;
    --stage2-eval-interval) require_value "$@"; STAGE2_EVAL_INTERVAL="$2"; shift 2 ;;
    --stage2-fixed-config-source) require_value "$@"; STAGE2_FIXED_CONFIG_SOURCE="$2"; shift 2 ;;
    --stage2-fixed-config) require_value "$@"; STAGE2_FIXED_CONFIG="$2"; shift 2 ;;
    --stage2-manual-gelu) require_value "$@"; STAGE2_MANUAL_GELU="$2"; shift 2 ;;
    --stage2-manual-softmax) require_value "$@"; STAGE2_MANUAL_SOFTMAX="$2"; shift 2 ;;
    --stage2-limit-tolerance) require_value "$@"; STAGE2_LIMIT_TOLERANCE="$2"; shift 2 ;;
    --stage2-stability-tolerance) require_value "$@"; STAGE2_STABILITY_TOLERANCE="$2"; shift 2 ;;
    --stage2-stability-multiplier) require_value "$@"; STAGE2_STABILITY_MULTIPLIER="$2"; shift 2 ;;
    --stage2-communication-importance-ratio) require_value "$@"; STAGE2_COMMUNICATION_IMPORTANCE_RATIO="$2"; shift 2 ;;
    --stage2-k-trials) require_value "$@"; STAGE2_K_TRIALS="$2"; shift 2 ;;
    --stage2-probe-size) require_value "$@"; STAGE2_PROBE_SIZE="$2"; shift 2 ;;
    --blb-v3-reward-devices) require_value "$@"; REWARD_DEVICES="$2"; shift 2 ;;
    --elastic-gpu-mode) require_value "$@"; ELASTIC_GPU_MODE="$2"; shift 2 ;;
    --elastic-gpu-recovery-interval) require_value "$@"; ELASTIC_GPU_RECOVERY_INTERVAL="$2"; shift 2 ;;
    --elastic-gpu-max-restarts) require_value "$@"; ELASTIC_GPU_MAX_RESTARTS="$2"; shift 2 ;;
    --random-seed) require_value "$@"; RANDOM_SEED="$2"; shift 2 ;;
    --final-eval-preset) require_value "$@"; FINAL_EVAL_PRESET="$2"; shift 2 ;;
    --skip-final-eval) SKIP_FINAL_EVAL="true"; shift ;;
    --comparator-smoke) COMPARATOR_SMOKE="true"; shift ;;
    --comparator-stage1-only) COMPARATOR_STAGE1_ONLY="true"; shift ;;
    --search-algorithm|--algorithm)
      require_value "$@"
      [ "$(normalize_algorithm "$2")" = "$ALGORITHM" ] || fail "preset algorithm does not match run subcommand"
      shift 2
      ;;
    --blb-v3-baseline-groups) require_value "$@"; BASELINE_GROUPS="$2"; shift 2 ;;
    --blb-v3-baseline-trials-per-group) require_value "$@"; BASELINE_TRIALS_PER_GROUP="$2"; shift 2 ;;
    --stage2-calibrate-baseline-samples|--blb-v3-calibrate-baseline-samples) require_value "$@"; CALIBRATE_BASELINE_SAMPLES="$2"; shift 2 ;;
    --blb-v3-constraint-bootstrap-samples) require_value "$@"; CONSTRAINT_BOOTSTRAP_SAMPLES="$2"; shift 2 ;;
    --blb-v3-online-constraint-probability) require_value "$@"; ONLINE_CONSTRAINT_PROBABILITY="$2"; shift 2 ;;
    --blb-v3-promotion-constraint-probability) require_value "$@"; PROMOTION_CONSTRAINT_PROBABILITY="$2"; shift 2 ;;
    --blb-v3-final-constraint-probability) require_value "$@"; FINAL_CONSTRAINT_PROBABILITY="$2"; shift 2 ;;
    --blb-v3-promotion-validation-trials) require_value "$@"; PROMOTION_VALIDATION_TRIALS="$2"; shift 2 ;;
    --blb-v3-final-selection-top-n) require_value "$@"; FINAL_SELECTION_TOP_N="$2"; shift 2 ;;
    --blb-v3-final-selection-validation-trials) require_value "$@"; FINAL_SELECTION_VALIDATION_TRIALS="$2"; shift 2 ;;
    --blb-v3-min-convergence-episodes) require_value "$@"; MIN_CONVERGENCE_EPISODES="$2"; shift 2 ;;
    --blb-v3-convergence-patience-updates) require_value "$@"; CONVERGENCE_PATIENCE_UPDATES="$2"; shift 2 ;;
    --blb-v3-online-k-trials) require_value "$@"; ONLINE_K_TRIALS="$2"; shift 2 ;;
    --blb-v3-terminal-eval-batch-size) require_value "$@"; TERMINAL_EVAL_BATCH_SIZE="$2"; shift 2 ;;
    *) fail "unsupported option: $1" ;;
  esac
done

DATASET="$(lowercase "$DATASET")"
MODEL_TYPE="$(lowercase "$MODEL_TYPE")"
MODE="$(lowercase "$MODE")"
STAGE2_FIXED_CONFIG_SOURCE="$(lowercase "$STAGE2_FIXED_CONFIG_SOURCE")"
ELASTIC_GPU_MODE="$(lowercase "$ELASTIC_GPU_MODE")"

case "$DATASET" in mrpc|rte|sst2) ;; *) fail "unsupported dataset: $DATASET" ;; esac
case "$MODEL_TYPE" in bert-base|bert-large) ;; *) fail "unsupported model type: $MODEL_TYPE" ;; esac
case "$STAGE2_FIXED_CONFIG_SOURCE" in all4|stage1_result|json|manual) ;; *) fail "unsupported Stage-2 fixed-config source" ;; esac
case "$ELASTIC_GPU_MODE" in auto|off) ;; *) fail "elastic GPU mode must be auto or off" ;; esac

for value in "$BATCH_SIZE" "$STAGE2_INFERENCE_BATCH_SIZE" "$PPO_UPDATE_INTERVAL" "$STAGE2_ROLLOUT_SIZE" "$STAGE2_SAVE_INTERVAL" "$STAGE2_EVAL_INTERVAL" "$STAGE2_K_TRIALS" "$STAGE2_PROBE_SIZE" "$ELASTIC_GPU_MAX_RESTARTS"; do
  is_positive_integer "$value" || fail "expected a positive integer, got: $value"
done
for value in "$STAGE1_EPISODES" "$STAGE2_EPISODES" "$ELASTIC_GPU_RECOVERY_INTERVAL"; do
  is_nonnegative_integer "$value" || fail "expected a nonnegative integer, got: $value"
done
for value in "$STAGE1_LR" "$STAGE2_LR" "$STAGE2_STABILITY_MULTIPLIER"; do
  is_positive_number "$value" || fail "expected a positive number, got: $value"
done
for value in "$STAGE1_ACCURACY_TOLERANCE" "$STAGE2_LIMIT_TOLERANCE" "$STAGE2_STABILITY_TOLERANCE"; do
  is_nonnegative_number "$value" || fail "expected a nonnegative number, got: $value"
done

case "$MODEL_TYPE:$DATASET" in
  bert-base:mrpc) BASE_MODEL="textattack/bert-base-uncased-MRPC" ;;
  bert-base:rte) BASE_MODEL="textattack/bert-base-uncased-RTE" ;;
  bert-base:sst2) BASE_MODEL="textattack/bert-base-uncased-SST-2" ;;
  bert-large:mrpc) BASE_MODEL="yoshitomo-matsubara/bert-large-uncased-mrpc" ;;
  bert-large:rte) BASE_MODEL="yoshitomo-matsubara/bert-large-uncased-rte" ;;
  bert-large:sst2) BASE_MODEL="yoshitomo-matsubara/bert-large-uncased-sst2" ;;
esac

SKIP_STAGE1="false"
SKIP_STAGE2="false"
DECOUPLED_LAYOUT="false"

if [ "$ALGORITHM" = "rl" ]; then
  case "$MODE" in
    stage1-only)
      SKIP_STAGE2="true"
      SKIP_FINAL_EVAL="true"
      DECOUPLED_LAYOUT="true"
      STAGE2_EPISODES="0"
      ;;
    stage2-only)
      SKIP_STAGE1="true"
      SKIP_FINAL_EVAL="true"
      STAGE1_EPISODES="51000"
      ;;
    *) fail "run rl requires --mode stage1-only or stage2-only" ;;
  esac
else
  [ "$MODEL_TYPE:$DATASET" = "bert-base:mrpc" ] || fail "formal comparators support only bert-base MRPC"
  MODE="train"
  BATCH_SIZE="16"
  STAGE2_INFERENCE_BATCH_SIZE="64"
  RANDOM_SEED="42"
  STAGE1_ACCURACY_TOLERANCE="0.001"
  STAGE2_LIMIT_TOLERANCE="0.001"
  STAGE2_STABILITY_TOLERANCE="1.2"
  STAGE2_STABILITY_MULTIPLIER="2.0"
  STAGE2_K_TRIALS="3"
  STAGE2_FIXED_CONFIG_SOURCE="stage1_result"
  FINAL_SELECTION_TOP_N="5"
  SEARCH_MUTATION_MAX_COORDINATES="4"
  case "$ALGORITHM" in
    bo_rf)
      SEARCH_BACKEND="bo_rf"
      SEARCH_EVALUATION_BUDGET="50000"
      SEARCH_PATIENCE="2000"
      ;;
    greedy)
      SEARCH_BACKEND="greedy"
      SEARCH_EVALUATION_BUDGET="2176782336"
      ;;
    coinn_ga)
      SEARCH_BACKEND="coinn_ga"
      SEARCH_EVALUATION_BUDGET="11464"
      SEARCH_PATIENCE="5"
      ;;
  esac
  if [ "$COMPARATOR_STAGE1_ONLY" = "true" ]; then
    SKIP_STAGE2="true"
    SKIP_FINAL_EVAL="true"
    STAGE2_EPISODES="0"
    if [ "$ALGORITHM" = "bo_rf" ]; then
      SEARCH_EVALUATION_BUDGET="10000"
      SEARCH_PATIENCE="1000"
    fi
  fi
  if [ "$COMPARATOR_SMOKE" = "true" ]; then
    SEARCH_EVALUATION_BUDGET="1"
    SEARCH_FULL_VALIDATION="false"
    SKIP_FINAL_EVAL="true"
  fi
fi

if [ "$STAGE2_FIXED_CONFIG_SOURCE" = "json" ]; then
  [ -f "$STAGE2_FIXED_CONFIG" ] || fail "Stage-2 fixed-config file not found: $STAGE2_FIXED_CONFIG"
fi
if [ "$STAGE2_FIXED_CONFIG_SOURCE" = "manual" ]; then
  [ -n "$STAGE2_MANUAL_GELU" ] && [ -n "$STAGE2_MANUAL_SOFTMAX" ] || fail "manual Stage-2 source requires GELU and Softmax vectors"
fi

if [ "$ELASTIC_GPU_MODE" = "auto" ] && [ "$ALGORITHM" = "rl" ]; then
  if [ "$MODE" = "stage1-only" ] && [ -z "$STAGE1_RL_DEVICES" ]; then
    STAGE1_RL_DEVICES="auto"
  fi
  if [ "$MODE" = "stage2-only" ] && [ -z "$REWARD_DEVICES" ]; then
    REWARD_DEVICES="auto"
  fi
fi

STABILITY_KEY="stage2_stability_multiplier"
STABILITY_VALUE="$STAGE2_STABILITY_MULTIPLIER"
CONSTRAINT_SLUG="s1t${STAGE1_ACCURACY_TOLERANCE}_s2t${STAGE2_LIMIT_TOLERANCE}_s2st${STABILITY_VALUE}"
if [ -n "$RUN_TAG" ]; then
  SAFE_TAG="$(printf '%s' "$RUN_TAG" | tr -c 'A-Za-z0-9_-' '_')"
  CONSTRAINT_SLUG="${CONSTRAINT_SLUG}__${SAFE_TAG}"
fi

if [ "$ALGORITHM" = "rl" ] && [ "$MODE" = "stage1-only" ]; then
  COMBO="${MODEL_TYPE//-/ } ${DATASET}"
  RUN_ROOT="$(dirname "$PERSISTENT_ROOT")/stage1/$COMBO"
else
  RUN_ROOT="$PERSISTENT_ROOT/$ALGORITHM/$MODEL_TYPE/$DATASET/$CONSTRAINT_SLUG"
fi

if [ "$MODE" != "stage1-only" ]; then
  command -v flock >/dev/null 2>&1 || fail "flock is required for Stage-2 persistent locking"
  mkdir -p "$(dirname "$RUN_ROOT")"
  LOCK_PATH="$(dirname "$RUN_ROOT")/.$(basename "$RUN_ROOT").stage2_rl.lock"
  exec 9>>"$LOCK_PATH"
  flock -n 9 || fail "persistent directory is already active: $RUN_ROOT"
  printf 'pid=%s\n' "$$" 1>&9
  export BLB_STAGE2_RUN_LOCK_FD=9 BLB_STAGE2_RUN_LOCK_PATH="$LOCK_PATH"
fi

if [ "$FRESH" = "true" ] && [ -d "$RUN_ROOT" ]; then
  rm -rf "$RUN_ROOT"
fi
if [ -f "$RUN_ROOT/COMPLETED" ] && [ "$FRESH" != "true" ]; then
  fail "run is already complete; pass --fresh to start again: $RUN_ROOT"
fi
if [ -z "$RESUME_FROM" ] && [ -f "$RUN_ROOT/metadata.json" ]; then
  RESUME_FROM="$RUN_ROOT"
fi
mkdir -p "$RUN_ROOT/logs"

if [ ! -f "$RUN_ROOT/metadata.json" ]; then
  python3 - "$RUN_ROOT/metadata.json" "$ALGORITHM" "$MODEL_TYPE" "$DATASET" "$STAGE1_ACCURACY_TOLERANCE" "$STAGE2_LIMIT_TOLERANCE" "$STABILITY_KEY" "$STABILITY_VALUE" "$BATCH_SIZE" "$STAGE2_INFERENCE_BATCH_SIZE" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path, algorithm, model, dataset, stage1_tol, stage2_tol, stability_key, stability_value, stage1_batch, stage2_batch = sys.argv[1:]
payload = {
    "algorithm": algorithm,
    "model_type": model,
    "dataset": dataset,
    "stage1_accuracy_tolerance": float(stage1_tol),
    "stage2_limit_tolerance": float(stage2_tol),
    stability_key: float(stability_value),
    "policy_network_variant": "shared_gtrxl_small_v1",
    "stage1_inference_batch_size": int(stage1_batch),
    "stage2_inference_batch_size": int(stage2_batch),
    "created_at": datetime.now(timezone.utc).isoformat(),
}
Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
fi

CMD=(
  python3 -m rfr.cli.run
  --base_model "$BASE_MODEL"
  --data_path "$DATASET"
  --output_dir "$RUN_ROOT"
  --glue_train_probe_fixture_path "$ROOT_DIR/fixtures/reproducibility/glue_train_probe_v1.json"
  --batch_size "$BATCH_SIZE"
  --stage1_rl_episodes "$STAGE1_EPISODES"
  --stage2_rl_episodes "$STAGE2_EPISODES"
  --stage1_rl_episodes_specified "$([ "$SKIP_STAGE1" = "false" ] && printf true || printf false)"
  --stage2_rl_episodes_specified "$([ "$SKIP_STAGE2" = "false" ] && printf true || printf false)"
  --stage1_entropy_stop_threshold "$STAGE1_ENTROPY_STOP_THRESHOLD"
  --ppo_update_interval "$PPO_UPDATE_INTERVAL"
  --skip_stage1_rl "$SKIP_STAGE1"
  --skip_noise_rl "$SKIP_STAGE2"
  --skip_final_eval "$SKIP_FINAL_EVAL"
  --resume_run_dir "$RESUME_FROM"
  --decoupled_layout "$DECOUPLED_LAYOUT"
  --stage1_run_id "$STAGE1_RUN_ID"
  --stage1_rl_lr "$STAGE1_LR"
  --stage2_rl_lr "$STAGE2_LR"
  --stage1_accuracy_tolerance "$STAGE1_ACCURACY_TOLERANCE"
  --stage2_limit_tolerance "$STAGE2_LIMIT_TOLERANCE"
  --stage2_stability_tolerance "$STAGE2_STABILITY_TOLERANCE"
  --stage2_stability_multiplier "$STAGE2_STABILITY_MULTIPLIER"
  --stage2_communication_importance_ratio "$STAGE2_COMMUNICATION_IMPORTANCE_RATIO"
  --stage2_k_trials "$STAGE2_K_TRIALS"
  --stage2_probe_size "$STAGE2_PROBE_SIZE"
  --stage2_inference_batch_size "$STAGE2_INFERENCE_BATCH_SIZE"
  --stage2_fixed_config_source "$STAGE2_FIXED_CONFIG_SOURCE"
  --stage2_fixed_config_path "$STAGE2_FIXED_CONFIG"
  --stage2_manual_gelu "$STAGE2_MANUAL_GELU"
  --stage2_manual_softmax "$STAGE2_MANUAL_SOFTMAX"
  --final_eval_random_seed "$RANDOM_SEED"
  --final_eval_preset "$FINAL_EVAL_PRESET"
  --blb_v3_rollout_size "$STAGE2_ROLLOUT_SIZE"
  --blb_v3_save_interval "$STAGE2_SAVE_INTERVAL"
  --blb_v3_eval_interval "$STAGE2_EVAL_INTERVAL"
  --blb_v3_calibrate_baseline_samples "$CALIBRATE_BASELINE_SAMPLES"
  --blb_v3_inproc_rescale_optimizer_root configs/preparation/rescale
  --blb_v3_seed "$RANDOM_SEED"
  --blb_v3_online_k_trials "$ONLINE_K_TRIALS"
  --blb_v3_terminal_eval_batch_size "$TERMINAL_EVAL_BATCH_SIZE"
  --blb_v3_promotion_validation_trials "$PROMOTION_VALIDATION_TRIALS"
  --blb_v3_final_selection_top_n "$FINAL_SELECTION_TOP_N"
  --blb_v3_final_selection_validation_trials "$FINAL_SELECTION_VALIDATION_TRIALS"
  --blb_v3_baseline_groups "$BASELINE_GROUPS"
  --blb_v3_baseline_trials_per_group "$BASELINE_TRIALS_PER_GROUP"
  --blb_v3_constraint_bootstrap_samples "$CONSTRAINT_BOOTSTRAP_SAMPLES"
  --blb_v3_online_constraint_probability "$ONLINE_CONSTRAINT_PROBABILITY"
  --blb_v3_promotion_constraint_probability "$PROMOTION_CONSTRAINT_PROBABILITY"
  --blb_v3_final_constraint_probability "$FINAL_CONSTRAINT_PROBABILITY"
  --blb_v3_min_convergence_episodes "$MIN_CONVERGENCE_EPISODES"
  --blb_v3_convergence_patience_updates "$CONVERGENCE_PATIENCE_UPDATES"
  --blb_v3_search_backend "$SEARCH_BACKEND"
  --blb_v3_search_evaluation_budget "$SEARCH_EVALUATION_BUDGET"
  --blb_v3_search_initial_design_size "$SEARCH_INITIAL_DESIGN_SIZE"
  --blb_v3_search_candidate_pool_size "$SEARCH_CANDIDATE_POOL_SIZE"
  --blb_v3_search_population_size "$SEARCH_POPULATION_SIZE"
  --blb_v3_search_patience_generations "$SEARCH_PATIENCE"
  --blb_v3_search_mutation_max_coordinates "$SEARCH_MUTATION_MAX_COORDINATES"
  --blb_v3_search_rf_n_estimators "$SEARCH_RF_N_ESTIMATORS"
  --blb_v3_search_rf_min_samples_leaf "$SEARCH_RF_MIN_SAMPLES_LEAF"
  --blb_v3_search_full_validation "$SEARCH_FULL_VALIDATION"
  --comparator_smoke "$COMPARATOR_SMOKE"
  --comparator_stage1_only "$COMPARATOR_STAGE1_ONLY"
)

if [ -n "$REWARD_DEVICES" ]; then
  CMD+=(--blb_v3_reward_devices "$REWARD_DEVICES")
fi
if [ -n "$STAGE1_RL_DEVICES" ]; then
  CMD+=(--stage1_rl_devices "$STAGE1_RL_DEVICES")
fi
if [ "$ALGORITHM" != "rl" ]; then
  CMD+=(--mrpc_reproducibility_fixture_path "$ROOT_DIR/fixtures/reproducibility/mrpc_validation_v1.json")
fi

LAUNCH_CMD=("${CMD[@]}")
if [ "$ALGORITHM" = "rl" ] && [ "$ELASTIC_GPU_MODE" = "auto" ]; then
  LAUNCH_CMD=(
    python3 -m rfr.search.runtime.supervisor
    --run-dir "$RUN_ROOT"
    --recovery-interval "$ELASTIC_GPU_RECOVERY_INTERVAL"
    --max-restarts "$ELASTIC_GPU_MAX_RESTARTS"
    --
    "${CMD[@]}"
  )
fi

printf -v COMMAND_TEXT '%q ' "${LAUNCH_CMD[@]}"
printf 'Run directory: %s\n' "$RUN_ROOT"
printf 'Command: %s\n' "$COMMAND_TEXT"

if [ "$DRY_RUN" = "true" ]; then
  exit 0
fi

LOG_PATH="$RUN_ROOT/logs/$(basename "$LOGFILE")"
nohup "${LAUNCH_CMD[@]}" >"$LOG_PATH" 2>&1 &
PID=$!
printf '%s\n' "$PID" > "$RUN_ROOT/run.pid"
printf '%s\n' "$RUN_ROOT" > "$(dirname "$RUN_ROOT")/LATEST_RUN_DIR"
printf '%s\n' "$PID" > "$(dirname "$RUN_ROOT")/LATEST_PID"

printf 'Started PID %s\n' "$PID"
printf 'Log: %s\n' "$LOG_PATH"
printf 'Graceful stop: kill -INT %s\n' "$PID"
