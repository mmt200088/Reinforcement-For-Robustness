#!/usr/bin/env bash
set -euo pipefail

# ======================================================================
# 用途说明
# ======================================================================
# 这个脚本用 nohup 在后台启动 generate_glue_submission.py，
# 在 GLUE 官方测试集上运行推理（含/不含 GELU+Softmax 多项式近似、
# 含/不含噪声注入），并把所有任务（CoLA, SST-2, MRPC, STS-B, MNLI-m,
# MNLI-mm, QNLI, RTE, WNLI, QQP, AX）的预测结果保存为可直接打包提交
# 到 https://gluebenchmark.com/ 的 TSV 文件 + submission.zip。
#
# 与原始 python 脚本的差异：
#   * 后台运行（nohup &），日志写到 RUN 目录下 logs/<logfile>
#   * 所有产物统一放在 glue_submission/<run_id>/ 下
#   * 自动生成时间戳目录，便于多次实验对比
#
# ======================================================================
# 命令格式
# ======================================================================
#   bash run_glue_submission.sh <mode> <run_name> [logfile] [extra args...]
#
# ----------------------------------------------------------------------
# 必填位置参数
# ----------------------------------------------------------------------
#   $1  mode       运行模式，四选一：
#                    baseline    纯基线，原始 GELU+exp，无近似、无噪声
#                    approx      仅近似（默认 config = glue_configs_best_ppo.json）
#                    noise       仅噪声（默认 noise_config = glue_noise_configs_best_ppo.json）
#                    full        近似 + 噪声（两阶段优化结果一起用）
#
#   $2  run_name   本次运行的子目录名，例如 "ppo_v1"
#                  最终输出会落在 glue_submission/<run_name>_<timestamp>/
#
# ----------------------------------------------------------------------
# 可选位置参数
# ----------------------------------------------------------------------
#   $3  logfile    nohup 日志文件名（默认 output.log）。
#                  实际路径会是 glue_submission/<run_name>_<ts>/logs/<logfile>
#
# ----------------------------------------------------------------------
# 可选命名参数（透传给 python 脚本）
# ----------------------------------------------------------------------
#   --config PATH         覆盖 approx/full 模式默认的 GELU/Softmax 配置文件
#   --noise_config PATH   覆盖 noise/full 模式默认的噪声配置文件
#   --tasks T1 T2 ...     仅运行指定 GLUE 任务（默认全部 11 个）
#                         可选值: cola sst2 mrpc stsb mnli qnli rte wnli qqp ax
#   --device DEV          推理设备（默认 cuda）
#   --max_length N        最大序列长度（默认 128）
#   --batch_size N        推理 batch size（默认 16）
#
# ======================================================================
# 示例
# ======================================================================
#   # 1) 基线，全部任务
#   bash run_glue_submission.sh baseline base_run
#
#   # 2) 仅近似，使用默认 ppo 配置
#   bash run_glue_submission.sh approx approx_ppo
#
#   # 3) 仅噪声，自定义噪声配置
#   bash run_glue_submission.sh noise noise_max output.log \
#        --noise_config glue_noise_configs_best_ppo.json
#
#   # 4) 近似+噪声完整组合
#   bash run_glue_submission.sh full full_ppo output.log
#
#   # 5) 只跑 mrpc + qnli
#   bash run_glue_submission.sh full quick_test output.log --tasks mrpc qnli
#
# ======================================================================
# 输出
# ======================================================================
#   glue_submission/<run_name>_<timestamp>/
#       logs/<logfile>          后台日志
#       CoLA.tsv SST-2.tsv ... AX.tsv QQP.tsv
#       submission.zip          可直接上传到 GLUE benchmark 网站
#
# ======================================================================

if [ "$#" -lt 2 ]; then
    echo "Usage: bash run_glue_submission.sh <mode> <run_name> [logfile] [extra args...]"
    echo "  mode: baseline | approx | noise | full"
    exit 1
fi

MODE="$1"
RUN_NAME="$2"
shift 2

LOGFILE="output.log"
if [ "$#" -ge 1 ] && [[ "$1" != --* ]]; then
    LOGFILE="$1"
    shift
fi

EXTRA_ARGS=("$@")

DEFAULT_APPROX_CFG="glue_configs_best_ppo.json"
DEFAULT_NOISE_CFG="glue_noise_configs_best_ppo.json"

MODE_ARGS=()
case "$MODE" in
    baseline)
        MODE_ARGS=(--no_approx)
        ;;
    approx)
        MODE_ARGS=(--config "$DEFAULT_APPROX_CFG")
        ;;
    noise)
        MODE_ARGS=(--no_approx --noise_config "$DEFAULT_NOISE_CFG")
        ;;
    full)
        MODE_ARGS=(--config "$DEFAULT_APPROX_CFG" --noise_config "$DEFAULT_NOISE_CFG")
        ;;
    *)
        echo "[Error] Unknown mode: $MODE (expected baseline|approx|noise|full)"
        exit 1
        ;;
esac

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${RUN_NAME}_${RUN_TS}"
RUN_ROOT="glue_submission/${RUN_ID}"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/${LOGFILE}"

CMD=(
    python generate_glue_submission.py
    "${MODE_ARGS[@]}"
    --output_dir "${RUN_ID}"
    "${EXTRA_ARGS[@]}"
)

echo "Mode:        $MODE"
echo "Run root:    $RUN_ROOT"
echo "Log file:    $LOG_PATH"
echo "Command:     ${CMD[*]}"

nohup "${CMD[@]}" > "$LOG_PATH" 2>&1 &
JOB_PID=$!
echo "Background PID: $JOB_PID"
echo "Tail logs with:  tail -f $LOG_PATH"
