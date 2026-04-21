#!/usr/bin/env bash
set -euo pipefail

usage() {
cat <<'EOF'
Usage:
  bash experiment/run_softmax_v_noise_sweep.sh [launcher options] [python options]

Launcher options:
  --foreground              Run in the foreground instead of nohup background mode.
  --output_dir DIR          Output directory. Default:
                            experiment/outputs/noise/softmax_v_sweep
  --logfile FILE            Background log filename. Default: run.log
  -h, --help                Show this help.

Common python options passed through to the experiment:
  --tasks mrpc
  --tasks mnli sst2 mrpc stsb qnli cola rte wnli
  --device cuda
  --batch_size 16
  --eval_split validation_full
  --repeat_n 5
  --scaling_factors 10 12 14 ... 48
                            Optional override. Formal runs should omit this
                            flag so the script scans the full MAP:
                            10,12,14,...,48.
  --max_eval_samples 128

Examples:
  bash experiment/run_softmax_v_noise_sweep.sh --tasks mrpc
  bash experiment/run_softmax_v_noise_sweep.sh --foreground --tasks mrpc --scaling_factors 10 12 --max_eval_samples 32

Background files:
  <output_dir>/run.log
  <output_dir>/pid.txt
  <output_dir>/run.pid
  <output_dir>/LATEST_RUN_DIR
  <output_dir>/LATEST_PID
EOF
}

err() {
  echo "Error: $1" >&2
  exit 1
}

needv() {
  [ "$#" -ge 2 ] || err "Option $1 requires a value."
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

FOREGROUND=0
OUTPUT_DIR="experiment/outputs/noise/softmax_v_sweep"
LOGFILE="run.log"
PASS_ARGS=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --foreground)
      FOREGROUND=1
      shift
      ;;
    --output_dir)
      needv "$@"
      OUTPUT_DIR="$2"
      PASS_ARGS+=("$1" "$2")
      shift 2
      ;;
    --logfile)
      needv "$@"
      LOGFILE="$2"
      shift 2
      ;;
    *)
      PASS_ARGS+=("$1")
      shift
      ;;
  esac
done

case "$OUTPUT_DIR" in
  /*)
    RUN_ROOT="$OUTPUT_DIR"
    ;;
  *)
    RUN_ROOT="$REPO_ROOT/$OUTPUT_DIR"
    ;;
esac

mkdir -p "$RUN_ROOT"

LOGFILE_PATH="$RUN_ROOT/$LOGFILE"
PID_PATH="$RUN_ROOT/pid.txt"
RUN_PID_PATH="$RUN_ROOT/run.pid"
LATEST_RUN_DIR_PATH="$RUN_ROOT/LATEST_RUN_DIR"
LATEST_PID_PATH="$RUN_ROOT/LATEST_PID"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES=0
fi

CMD=(python -u -m experiment.scripts.noise.softmax_v_noise_sweep "${PASS_ARGS[@]}")
printf -v CMD_STR '%q ' "${CMD[@]}"

echo "Launch configuration:"
echo "  Experiment: softmax/V fresh-noise sweep"
echo "  Repo root:  $REPO_ROOT"
echo "  Output dir: $RUN_ROOT"
echo "  Log file:   $LOGFILE_PATH"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "  Default eval: validation_full, repeat_n=5"
echo "  Command:    $CMD_STR"

cd "$REPO_ROOT"

if [ "$FOREGROUND" -eq 1 ]; then
  "${CMD[@]}"
  exit $?
fi

nohup "${CMD[@]}" > "$LOGFILE_PATH" 2>&1 &
JOB_PID=$!

echo "$JOB_PID" > "$PID_PATH"
echo "$JOB_PID" > "$RUN_PID_PATH"
echo "$RUN_ROOT" > "$LATEST_RUN_DIR_PATH"
echo "$JOB_PID" > "$LATEST_PID_PATH"

echo
echo "Started in background."
echo "  PID: $JOB_PID"
echo "  Log: tail -f $LOGFILE_PATH"
echo "  Stop: kill -INT $JOB_PID"
echo "  Force stop: kill -9 \$(cat $PID_PATH)"
