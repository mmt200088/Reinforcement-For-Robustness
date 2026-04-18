#!/usr/bin/env bash
# ============================================================================
# 噪声 Scaling Factor 扫描实验后台启动脚本
# ============================================================================
# 
# 运行时可选项（重要：本脚本会“透传”大多数参数给 Python 脚本）
# ----------------------------------------------------------------------------
# 一、本脚本自身支持的参数
# - --foreground
#   - 含义：前台运行（不使用 nohup，不写 pid.txt），日志直接输出到当前终端。
#   - 适用：调试/小规模试跑，想立刻看到报错与进度。
#
# - --output_dir <DIR>
#   - 含义：指定输出目录；不传时默认 `experiment_results/noise_scaling_sweep`。
#   - 影响：
#     - 后台模式下写入：`<output_dir>/run.log` 与 `<output_dir>/pid.txt`
#     - Python 脚本也会把图与 JSON 输出到该目录。
#
# 二、透传给 `experiment_noise_scaling_sweep.py` 的参数（与 Python 脚本一致）
# - --tasks <task1> <task2> ...
#   - 含义：选择要跑的 GLUE 任务子集；不传则默认全跑：
#     mnli sst2 mrpc stsb qnli cola rte wnli
#
# - --device <cuda|cpu|...>
#   - 含义：选择推理设备；默认 `cuda`。若机器无 CUDA 会自动回退到 CPU（会非常慢）。
#   - 备注：本脚本还通过环境变量 `CUDA_VISIBLE_DEVICES` 控制可见 GPU（默认 0）。
#
# - --batch_size <INT>
#   - 含义：评估 batch size；默认 16。增大可提速但更易 OOM。
#
# - --max_length <INT>
#   - 含义：tokenizer 最大长度；默认 128。
#
# - --eval_split <SPLIT_NAME>
#   - 含义：评估使用的数据切分；默认 `validation_full`。
#
# - --repeat_n <INT>
#   - 含义：每个扫描点重复评估次数；默认 50（非常耗时）。
#   - 建议：调试可用 `--repeat_n 2` 或 `--repeat_n 5`。
#
# - --max_eval_samples <INT>
#   - 含义：限制每次评估最多用多少样本；默认 0 表示不限制（全量评估，极慢）。
#   - 建议：调试可用 `--max_eval_samples 32/128` 快速出图/出 JSON。
#
# - --seed <INT>
#   - 含义：随机种子；默认 42。
#
# - --noise_base_source <json|manual>
#   - 含义：“当前噪声配置”（x 与 6 个 W 的基准）来源；默认 `json`。
#
# - --noise_base_config <PATH>
#   - 含义：当 `--noise_base_source json` 时读取的噪声配置文件；
#     默认 `glue_noise_configs_best_ppo.json`。
#
# - --manual_noise_config <JSON_STRING>
#   - 含义：当 `--noise_base_source manual` 时使用的手动噪声配置（JSON 字符串）。
#
# - --approx_base_config <PATH>
#   - 含义：固定的 GELU / Softmax 近似配置来源；默认 `glue_configs_best_ppo.json`。
#
# 三、后台模式输出文件
# - `<output_dir>/run.log`：nohup 的 stdout/stderr（主要进度在这里）
# - `<output_dir>/pid.txt`：后台进程 PID，用于停止任务
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
    OUTPUT_DIR="experiment/outputs/noise/scaling_sweep"
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
    python -u -m experiment.scripts.noise.noise_scaling_sweep "${PASS_ARGS[@]}"
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
