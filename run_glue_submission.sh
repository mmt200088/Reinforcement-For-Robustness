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
#   --model_type TYPE     预训练骨干切换，二选一：
#                           bert-base   （默认）使用 textattack/bert-base-uncased-*
#                                       覆盖所有 11 个 GLUE 任务
#                           bert-large  使用 yoshitomo-matsubara/bert-large-uncased-*
#                                       仅支持 cola / sst2 / mrpc / stsb / qnli / rte
#                                       其余任务（mnli / wnli / ax / qqp）会自动跳过
#                                       或填充为占位结果
#                         脚本会把该参数透传给 generate_glue_submission.py，并且
#                         根据选择从同一 JSON 文件中读取对应层数的 bert-base / bert-large
#                         子段（新版 JSON schema 已按变体分段，详见 README）。
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
#        --noise_config glue_noise_configs_best_ppo.json \
#        --config glue_configs_best_ppo.json
#
#   # 4) 近似+噪声完整组合
#   bash run_glue_submission.sh full full_ppo output.log
#
#   # 5) 只跑 mrpc + qnli
#   bash run_glue_submission.sh full quick_test output.log --tasks mrpc qnli
#
#   # 6) bert-large 的近似+噪声完整提交（自动跳过 mnli/wnli/ax/qqp）
#   bash run_glue_submission.sh full large_ppo output.log --model_type bert-large
#
#   # 7) bert-large 仅跑 mrpc 的近似提交
#   bash run_glue_submission.sh approx large_mrpc output.log \
#        --model_type bert-large --tasks mrpc
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
    --output_dir "${RUN_ROOT}"
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
