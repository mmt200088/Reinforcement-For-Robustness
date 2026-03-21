#!/bin/bash
#
# Transformer Approximation Error Non-Accumulation Experiment Runner
#
# Runs all experiment blocks sequentially in background (nohup). Each block
# saves results to its own subdirectory under experiment_results/. Log and PID are
# written to experiment_results/run.log and experiment_results/pid.txt.
#
# Usage:
#   bash run_all_experiments.sh              # Run all experiments in background
#   bash run_all_experiments.sh --quick      # Quick test (sst2, mrpc) in background
#   bash run_all_experiments.sh --foreground # Run in foreground (no nohup)
#
# Check status:  ps aux | grep -E "experiment_|run_all"
# Stop process:  kill -9 $(cat experiment_results/pid.txt)
# View log:      tail -f experiment_results/run.log
#

set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
source /var/tmp/root-home/miniconda3/etc/profile.d/conda.sh
conda activate llm_ist

DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_LENGTH="${MAX_LENGTH:-128}"
N_BOOTSTRAP="${N_BOOTSTRAP:-100}"
RESULTS_DIR="${RESULTS_DIR:-experiment_results}"

if [ "$1" = "--quick" ]; then
    TASKS="--tasks sst2 mrpc"
    N_BOOTSTRAP=30
    QUICK_MODE=1
    shift
else
    TASKS=""
    QUICK_MODE=0
fi

FOREGROUND=0
if [ "$1" = "--foreground" ]; then
    FOREGROUND=1
fi

mkdir -p "$RESULTS_DIR"

run_experiments() {
    echo "============================================================"
    echo "  Experiment Configuration"
    echo "============================================================"
    echo "  Device:       $DEVICE"
    echo "  Batch Size:   $BATCH_SIZE"
    echo "  Max Length:   $MAX_LENGTH"
    echo "  N Bootstrap:  $N_BOOTSTRAP"
    echo "  Results Dir:  $RESULTS_DIR"
    [ "$QUICK_MODE" = "1" ] && echo "  Mode:         QUICK (sst2, mrpc only)"
    echo "============================================================"

    echo ""
    echo "============================================================"
    echo "  [1/5] Supplementary Test 1: Single-Layer Degradation"
    echo "============================================================"
    python -u experiment_single_layer_degradation.py \
        $TASKS \
        --device "$DEVICE" \
        --batch_size "$BATCH_SIZE" \
        --max_length "$MAX_LENGTH" \
        --output_dir "$RESULTS_DIR/single_layer"

    echo ""
    echo "============================================================"
    echo "  [2/5] Supplementary Test 2: Stepwise Degradation"
    echo "============================================================"
    python -u experiment_stepwise_degradation.py \
        $TASKS \
        --device "$DEVICE" \
        --batch_size "$BATCH_SIZE" \
        --max_length "$MAX_LENGTH" \
        --n_trials 5 \
        --ppo_config glue_configs_best_ppo.json \
        --output_dir "$RESULTS_DIR/stepwise"

    echo ""
    echo "============================================================"
    echo "  [3/5] Block 1: Non-Monotonicity Statistical Test"
    echo "============================================================"
    python -u experiment_block1_monotonicity.py \
        $TASKS \
        --device "$DEVICE" \
        --batch_size "$BATCH_SIZE" \
        --max_length "$MAX_LENGTH" \
        --n_pairs 30 \
        --n_bootstrap "$N_BOOTSTRAP" \
        --output_dir "$RESULTS_DIR/block1"

    echo ""
    echo "============================================================"
    echo "  [4/5] Block 2: ANOVA Interaction Effect"
    echo "============================================================"
    python -u experiment_block2_anova.py \
        $TASKS \
        --device "$DEVICE" \
        --batch_size "$BATCH_SIZE" \
        --max_length "$MAX_LENGTH" \
        --n_bootstrap "$N_BOOTSTRAP" \
        --output_dir "$RESULTS_DIR/block2"

    echo ""
    echo "============================================================"
    echo "  [5/5] Block 3: Cross-Task Robustness Analysis"
    echo "============================================================"
    python -u experiment_block3_cross_task.py \
        $TASKS \
        --device "$DEVICE" \
        --batch_size "$BATCH_SIZE" \
        --max_length "$MAX_LENGTH" \
        --output_dir "$RESULTS_DIR/block3"

    echo ""
    echo "============================================================"
    echo "  ALL EXPERIMENTS COMPLETED"
    echo "  Results saved to: $RESULTS_DIR/"
    echo "============================================================"
    echo "  $RESULTS_DIR/"
    echo "  ├── single_layer/     # Test 1: Single-layer degradation"
    echo "  ├── stepwise/         # Test 2: Stepwise degradation curves"
    echo "  ├── block1/           # Non-monotonicity statistical test"
    echo "  ├── block2/           # ANOVA interaction effects"
    echo "  └── block3/           # Cross-task Spearman analysis"
}

if [ "$FOREGROUND" = "1" ]; then
    run_experiments
else
    export DEVICE BATCH_SIZE MAX_LENGTH N_BOOTSTRAP RESULTS_DIR TASKS QUICK_MODE
    nohup bash -c "
        source /var/tmp/root-home/miniconda3/etc/profile.d/conda.sh
        conda activate llm_ist
        cd $(pwd)
        $(declare -f run_experiments)
        run_experiments
    " > "$RESULTS_DIR/run.log" 2>&1 &
    echo $! > "$RESULTS_DIR/pid.txt"
    disown
    echo "Experiments started in background."
    echo "  PID:  $(cat "$RESULTS_DIR/pid.txt")"
    echo "  Log:  $RESULTS_DIR/run.log"
    echo "  Check: tail -f $RESULTS_DIR/run.log"
    echo "  Stop: kill -9 \$(cat $RESULTS_DIR/pid.txt)"
fi
