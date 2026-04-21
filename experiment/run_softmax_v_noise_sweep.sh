#!/usr/bin/env bash
# Background launcher for the softmax/V fresh-noise scaling sweep.
#
# Examples:
#   bash experiment/run_softmax_v_noise_sweep.sh
#   bash experiment/run_softmax_v_noise_sweep.sh --tasks mrpc
#   bash experiment/run_softmax_v_noise_sweep.sh --foreground --tasks mrpc --scaling_factors 10 48 --max_eval_samples 32
#
# Background outputs:
#   experiment/outputs/noise/softmax_v_sweep/run.log
#   experiment/outputs/noise/softmax_v_sweep/pid.txt

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [ -f /var/tmp/root-home/miniconda3/etc/profile.d/conda.sh ]; then
    # Match the existing project launchers when that environment is available.
    # If activation fails, keep the current shell environment.
    source /var/tmp/root-home/miniconda3/etc/profile.d/conda.sh
    conda activate llm_ist || true
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"

FOREGROUND=0
OUTPUT_DIR=""
PASS_ARGS=()

while (($#)); do
    case "$1" in
        --foreground)
            FOREGROUND=1
            shift
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            PASS_ARGS+=("$1" "$2")
            shift 2
            ;;
        *)
            PASS_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="experiment/outputs/noise/softmax_v_sweep"
    PASS_ARGS+=("--output_dir" "$OUTPUT_DIR")
fi

case "$OUTPUT_DIR" in
    /*|[A-Za-z]:*)
        OUTPUT_DIR_PATH="$OUTPUT_DIR"
        ;;
    *)
        OUTPUT_DIR_PATH="$REPO_ROOT/$OUTPUT_DIR"
        ;;
esac

mkdir -p "$OUTPUT_DIR_PATH"

run_experiment() {
    echo "============================================================"
    echo "  Softmax/V Fresh-Noise Sweep"
    echo "============================================================"
    echo "  Output Dir:   $OUTPUT_DIR"
    echo "  Device:       ${CUDA_VISIBLE_DEVICES:-0}"
    echo "  Python Args:  ${PASS_ARGS[*]}"
    echo "  Default eval: validation_full, repeat_n=5"
    echo "============================================================"
    cd "$REPO_ROOT"
    python -u -m experiment.scripts.noise.softmax_v_noise_sweep "${PASS_ARGS[@]}"
}

if [ "$FOREGROUND" -eq 1 ]; then
    run_experiment
else
    nohup bash "$SCRIPT_PATH" --foreground "${PASS_ARGS[@]}" > "$OUTPUT_DIR_PATH/run.log" 2>&1 &
    echo $! > "$OUTPUT_DIR_PATH/pid.txt"
    disown || true
    echo "Experiments started in background."
    echo "  PID:  $(cat "$OUTPUT_DIR_PATH/pid.txt")"
    echo "  Log:  $OUTPUT_DIR_PATH/run.log"
    echo "  Check: tail -f $OUTPUT_DIR_PATH/run.log"
    echo "  Stop:  kill -9 \$(cat $OUTPUT_DIR_PATH/pid.txt)"
fi
