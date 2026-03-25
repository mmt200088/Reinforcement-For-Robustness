#!/usr/bin/env bash
# ============================================================================
# 噪声 Scaling Factor 扫描实验后台启动脚本
# ============================================================================
# 一、脚本用途
# 本脚本用于以和现有 run_gelu_analysis.sh / run_all_experiments.sh 类似的方式，
# 后台启动 experiment_noise_scaling_sweep.py。
#
# 默认行为：
# 1. 使用 nohup 后台运行
# 2. 自动创建输出目录
# 3. 将主日志写入：
#    <output_dir>/run.log
# 4. 将后台进程 PID 写入：
#    <output_dir>/pid.txt
#
# 二、默认输出目录
# 如果你没有显式传 --output_dir，则默认使用：
# experiment_results/noise_scaling_sweep
#
# 三、支持的典型用法
# 1. 后台运行全部数据集正式实验：
#    bash run_noise_scaling_sweep.sh
# 2. 后台只跑单个数据集：
#    bash run_noise_scaling_sweep.sh --tasks sst2
# 3. 前台运行（便于调试）：
#    bash run_noise_scaling_sweep.sh --foreground --tasks sst2 --repeat_n 2 --max_eval_samples 32
# 4. 指定输出目录：
#    bash run_noise_scaling_sweep.sh --output_dir experiment_results/noise_scaling_sweep_v2
# 5. 指定重复次数：
#    bash run_noise_scaling_sweep.sh --repeat_n 100
#
# 四、查看日志与停止任务
# 1. 查看日志：
#    tail -f experiment_results/noise_scaling_sweep/run.log
# 2. 查看 PID：
#    cat experiment_results/noise_scaling_sweep/pid.txt
# 3. 停止任务：
#    kill -9 $(cat experiment_results/noise_scaling_sweep/pid.txt)
#
# 五、说明
# 1. 本脚本会自动透传除 --foreground 之外的所有参数给 Python 实验脚本。
# 2. 若显式传入 --output_dir，则 run.log / pid.txt 也会写入该目录。
# 3. 推荐正式实验保留默认 repeat_n=50 或更高。
# ============================================================================

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
source /var/tmp/root-home/miniconda3/etc/profile.d/conda.sh
conda activate llm_ist

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
    OUTPUT_DIR="experiment_results/noise_scaling_sweep"
    PASS_ARGS+=("--output_dir" "$OUTPUT_DIR")
fi

mkdir -p "$OUTPUT_DIR"

run_experiment() {
    echo "============================================================"
    echo "  Noise Scaling Sweep Configuration"
    echo "============================================================"
    echo "  Output Dir:   $OUTPUT_DIR"
    echo "  Device:       ${CUDA_VISIBLE_DEVICES:-0}"
    echo "  Python Args:  ${PASS_ARGS[*]}"
    echo "============================================================"
    python -u experiment_noise_scaling_sweep.py "${PASS_ARGS[@]}"
}

if [ "$FOREGROUND" -eq 1 ]; then
    cd "$SCRIPT_DIR"
    run_experiment
else
    nohup bash "$SCRIPT_PATH" --foreground "${PASS_ARGS[@]}" > "$OUTPUT_DIR/run.log" 2>&1 &
    echo $! > "$OUTPUT_DIR/pid.txt"
    disown
    echo "Experiments started in background."
    echo "  PID:  $(cat "$OUTPUT_DIR/pid.txt")"
    echo "  Log:  $OUTPUT_DIR/run.log"
    echo "  Check: tail -f $OUTPUT_DIR/run.log"
    echo "  Stop:  kill -9 \$(cat $OUTPUT_DIR/pid.txt)"
fi
