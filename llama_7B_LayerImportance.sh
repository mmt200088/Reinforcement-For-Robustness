#!/usr/bin/env bash

# ======================================================================
# 用途说明
# ======================================================================
# 这个脚本是项目的主启动脚本，用来在服务器上后台启动 rl_tune.py，
# 进行基于强化学习的层重要性搜索（第一阶段），以及可选的噪声 RL
# 优化（第二阶段），最后执行 FINAL EVALUATION。
#
# ======================================================================
# 命令格式
# ======================================================================
#   bash llama_7B_LayerImportance.sh <lora_r> <lora_alpha> <logfile> <rl_lr> <degree> [可选参数...]
#
# ======================================================================
# 必填位置参数（共 5 个）
# ======================================================================
#   $1  lora_r        LoRA rank，当前实验固定写 32
#   $2  lora_alpha    LoRA alpha，当前实验固定写 64
#   $3  logfile_path  nohup 日志输出路径（训练/评估日志均写入此文件）
#   $4  rl_lr         PPO 学习率控制：
#                       - 若 < 1 则直接用作 PPO LR（如 3e-5）
#                       - 旧版整数值如 20 / 40 会被解读为 20e-6 / 40e-6
#                       - 第一阶段和第二阶段共用同一个 PPO LR
#   $5  degree        旧版调试参数，已废弃，固定传 2
#
# ======================================================================
# 全部可选命名参数（跟在 5 个位置参数后面）
# ======================================================================
#
# ---- 第一阶段：最终评估配置来源 ----
#
#   --final-eval-source search|json|manual
#       最终评估使用的 GELU/Softmax 配置来源：
#         search  （默认）使用本次 RL 搜索得到的最优配置
#         json    从 JSON 文件读取历史保存的配置
#         manual  使用手动指定的每层配置
#
#   --final-eval-config PATH
#       当 --final-eval-source json 时使用。
#       指定 JSON 配置文件路径，默认为 glue_configs_best_ppo.json。
#       程序会根据当前数据集名称（如 mrpc）自动读取对应条目。
#
#   --manual-gelu "[1,1,1,4,...]"
#       当 --final-eval-source manual 时使用。
#       手动指定每层 GELU degree（JSON 数组格式）。
#       必须与 --manual-softmax 一起使用。
#
#   --manual-softmax "[2,3,4,6,...]"
#       当 --final-eval-source manual 时使用。
#       手动指定每层 Softmax degree（JSON 数组格式）。
#       必须与 --manual-gelu 一起使用。
#
# ---- 随机对照实验 ----
#
#   --random-seed N
#       随机实验的种子值，默认 42。
#
#   --perm-trials N
#       Permutation 随机对照实验次数（在最优配置上做排列），默认 10。
#
#   --cost-trials N
#       精确 cost-matched 随机对照实验次数，默认 10。
#
#   --budget-trials N
#       同总预算随机对照实验次数，默认 10。
#
# ---- 第二阶段：噪声 RL（Noise RL）----
#
#   --skip-noise-rl
#       跳过第二阶段噪声 RL，只运行第一阶段 GELU/Softmax 搜索。
#       默认情况下，第二阶段会在第一阶段配置确定后自动运行。
#       第二阶段保持第一阶段选定的 GELU/Softmax 不变，
#       用 PPO 学习每层的 7 个噪声 scaling factor：
#         x       输入噪声    动作空间 {20, 22, 24, 26, 28, 30}
#         wq      Query 权重噪声
#         wk      Key 权重噪声
#         wv      Value 权重噪声     动作空间均为
#         wo      Attn输出权重噪声   {10, 12, 14, 16, 18, 20, 22}
#         wffn1   FFN第一层权重噪声
#         wffn2   FFN第二层权重噪声
#       第二阶段产出的文件：
#         noise_ppo_step_info.txt        每步动作/概率日志
#         noise_ppo_training_curve.png   训练曲线图
#         noise_ppo_entropy_curve.png    策略熵曲线图
#       第二阶段的逻辑位于独立模块 noise_rl_module.py 中。
#
# ---- 帮助 ----
#
#   -h, --help
#       显示用法帮助信息并退出。
#
# ======================================================================
# 使用示例
# ======================================================================
#
# 1. 默认完整流程（第一阶段 RL 搜索 + 第二阶段噪声 RL）
#    bash llama_7B_LayerImportance.sh 32 64 output.log 20 2
#
# 2. 只跑第一阶段，跳过第二阶段噪声 RL
#    bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
#      --skip-noise-rl
#
# 3. 从 JSON 文件加载配置做最终评估（跳过第一阶段搜索，仍运行第二阶段）
#    bash llama_7B_LayerImportance.sh 32 64 output_json.log 20 2 \
#      --final-eval-source json \
#      --final-eval-config glue_configs_best_ppo.json
#
# 4. 手动指定每层配置做最终评估
#    bash llama_7B_LayerImportance.sh 32 64 output_manual.log 20 2 \
#      --final-eval-source manual \
#      --manual-gelu "[1,1,1,4,1,1,1,1,1,1,1,1]" \
#      --manual-softmax "[2,3,4,6,4,4,5,4,4,5,5,2]"
#
# 5. 提高随机对照实验次数
#    bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
#      --perm-trials 30 --cost-trials 30 --budget-trials 30
#
# 6. 从 JSON 加载配置 + 跳过噪声 RL
#    bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
#      --final-eval-source json \
#      --final-eval-config glue_configs_best_ppo.json \
#      --skip-noise-rl
#
# ======================================================================
# 项目相关说明
# ======================================================================
# - 当前默认数据集：base_model=textattack/bert-base-uncased-MRPC, data_path=mrpc
# - 脚本使用 nohup 后台运行，日志查看：tail -f <logfile_path>
# - 停止任务：ps aux | grep rl_tune.py，然后 kill -9 <PID>
# - 服务器部署时需同步的 Python 文件：
#     final_evaluation_module.py    第一阶段最终评估模块
#     noise_rl_module.py            第二阶段噪声 RL 模块
# ======================================================================

usage() {
    echo "Usage:"
    echo "  bash llama_7B_LayerImportance.sh [lora_r] [lora_alpha] [logfile_path] [rl_lr] [degree] [options]"
    echo
    echo "Required positional arguments:"
    echo "  lora_r                LoRA rank, keep 32 in current experiments."
    echo "  lora_alpha            LoRA alpha, keep 64 in current experiments."
    echo "  logfile_path          Output log path for nohup."
    echo "  rl_lr                 PPO LR control. Use a direct LR if < 1."
    echo "                        Legacy values like 20/40 map to 20e-6/40e-6."
    echo "  degree                Legacy debug argument, keep 2."
    echo
    echo "Optional final-evaluation arguments:"
    echo "  --final-eval-source search|json|manual"
    echo "  --final-eval-config PATH"
    echo "  --manual-gelu \"[1,1,...]\""
    echo "  --manual-softmax \"[2,2,...]\""
    echo "  --random-seed N"
    echo "  --perm-trials N"
    echo "  --cost-trials N"
    echo "  --budget-trials N"
    echo
    echo "Second-stage noise RL:"
    echo "  --skip-noise-rl          Skip stage-2 noise RL entirely."
    echo "  By default, stage-2 runs after the GELU/Softmax config is selected."
    echo "  The fixed config source still follows --final-eval-source."
    echo "  x action space: {20,22,24,26,28,30}"
    echo "  Weight action space (wq/wk/wv/wo/wffn1/wffn2): {10,12,14,16,18,20,22}"
    echo "  Writes noise_ppo_step_info.txt, noise_ppo_training_curve.png,"
    echo "  and noise_ppo_entropy_curve.png."
    echo
    echo "Examples:"
    echo "  bash llama_7B_LayerImportance.sh 32 64 output.log 20 2"
    echo "  bash llama_7B_LayerImportance.sh 32 64 output_json.log 20 2 --final-eval-source json --final-eval-config glue_configs_best_ppo.json"
    echo "  bash llama_7B_LayerImportance.sh 32 64 output_manual.log 20 2 --final-eval-source manual --manual-gelu \"[1,1,1,4,1,1,1,1,1,1,1,1]\" --manual-softmax \"[2,3,4,6,4,4,5,4,4,5,5,2]\""
    echo "  bash llama_7B_LayerImportance.sh 32 64 output_stage2.log 20 2 --perm-trials 30 --cost-trials 30"
    echo "  bash llama_7B_LayerImportance.sh 32 64 output_no_noise.log 20 2 --skip-noise-rl"
}

require_option_value() {
    if [ "$#" -lt 2 ]; then
        echo "Option $1 requires a value."
        exit 1
    fi
}

if [ "$#" -lt 5 ]; then
    usage
    exit 1
fi

LORA_R="$1"
LORA_ALPHA="$2"
LOGFILE_PATH="$3"
RL_LR="$4"
DEGREE="$5"
shift 5

FINAL_EVAL_SOURCE="search"
FINAL_EVAL_CONFIG_PATH="glue_configs_best_ppo.json"
MANUAL_FINAL_GELU=""
MANUAL_FINAL_SOFTMAX=""
FINAL_EVAL_RANDOM_SEED="42"
FINAL_EVAL_PERMUTATION_TRIALS="10"
FINAL_EVAL_COST_EQUIVALENT_TRIALS="10"
FINAL_EVAL_BUDGET_EQUIVALENT_TRIALS="10"
SKIP_NOISE_RL="false"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --final-eval-source)
            require_option_value "$@"
            FINAL_EVAL_SOURCE="$2"
            shift 2
            ;;
        --final-eval-config|--final-eval-config-path)
            require_option_value "$@"
            FINAL_EVAL_CONFIG_PATH="$2"
            shift 2
            ;;
        --manual-gelu)
            require_option_value "$@"
            MANUAL_FINAL_GELU="$2"
            shift 2
            ;;
        --manual-softmax)
            require_option_value "$@"
            MANUAL_FINAL_SOFTMAX="$2"
            shift 2
            ;;
        --random-seed)
            require_option_value "$@"
            FINAL_EVAL_RANDOM_SEED="$2"
            shift 2
            ;;
        --perm-trials|--permutation-trials)
            require_option_value "$@"
            FINAL_EVAL_PERMUTATION_TRIALS="$2"
            shift 2
            ;;
        --cost-trials|--cost-equivalent-trials)
            require_option_value "$@"
            FINAL_EVAL_COST_EQUIVALENT_TRIALS="$2"
            shift 2
            ;;
        --budget-trials|--budget-equivalent-trials)
            require_option_value "$@"
            FINAL_EVAL_BUDGET_EQUIVALENT_TRIALS="$2"
            shift 2
            ;;
        --skip-noise-rl)
            SKIP_NOISE_RL="true"
            shift 1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

case "$FINAL_EVAL_SOURCE" in
    search|json|manual)
        ;;
    *)
        echo "Invalid --final-eval-source: $FINAL_EVAL_SOURCE"
        echo "Expected one of: search, json, manual"
        exit 1
        ;;
esac

if [ "$FINAL_EVAL_SOURCE" = "manual" ]; then
    if [ -z "$MANUAL_FINAL_GELU" ] || [ -z "$MANUAL_FINAL_SOFTMAX" ]; then
        echo "Manual final evaluation requires both --manual-gelu and --manual-softmax."
        exit 1
    fi
fi

export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0

CMD=(
    python rl_tune.py
    --base_model "textattack/bert-base-uncased-MRPC"
    --data_path "mrpc"
    --output_dir "$LOGFILE_PATH"
    --batch_size 16
    --micro_batch_size 16
    --num_epochs 1
    --learning_rate 2e-4
    --cutoff_len 256
    --val_set_size 120
    --eval_step 80
    --adapter_name lora
    --target_modules "[\"q_proj\", \"k_proj\", \"v_proj\", \"up_proj\", \"down_proj\"]"
    --lora_r "$LORA_R"
    --lora_alpha "$LORA_ALPHA"
    --rl_lr "$RL_LR"
    --degree "$DEGREE"
    --use_ist
    --final_eval_config_source "$FINAL_EVAL_SOURCE"
    --final_eval_config_path "$FINAL_EVAL_CONFIG_PATH"
    --manual_final_gelu "$MANUAL_FINAL_GELU"
    --manual_final_softmax "$MANUAL_FINAL_SOFTMAX"
    --final_eval_random_seed "$FINAL_EVAL_RANDOM_SEED"
    --final_eval_permutation_trials "$FINAL_EVAL_PERMUTATION_TRIALS"
    --final_eval_cost_equivalent_trials "$FINAL_EVAL_COST_EQUIVALENT_TRIALS"
    --final_eval_budget_equivalent_trials "$FINAL_EVAL_BUDGET_EQUIVALENT_TRIALS"
    --skip_noise_rl "$SKIP_NOISE_RL"
)

echo "Launching RL tune job with final evaluation source: $FINAL_EVAL_SOURCE"
if [ "$SKIP_NOISE_RL" = "true" ]; then
    echo "Stage-2 noise RL: SKIPPED (--skip-noise-rl)"
else
    echo "Stage-2 noise RL will run automatically after the fixed GELU/Softmax config is selected."
fi
echo "Log file: $LOGFILE_PATH"

nohup "${CMD[@]}" > "$LOGFILE_PATH" 2>&1 &

# different data for base model
# --base_model "textattack/bert-base-uncased-WNLI"
# --base_model "textattack/bert-base-uncased-RTE"
# --base_model "textattack/bert-base-uncased-CoLA"
# --base_model "textattack/bert-base-uncased-QNLI"
# --base_model "textattack/bert-base-uncased-MRPC"
# --base_model "textattack/bert-base-uncased-SST-2"
# --base_model "textattack/bert-base-uncased-STS-B"

# different data for data_path
# --data_path "wnli"
# --data_path "rte"
# --data_path "cola"
# --data_path "qnli"
# --data_path "mrpc"
# --data_path "sst2"
# --data_path "stsb"

# ----------------------------------------------------------------------
# Original llama_7B_LayerImportance.sh content before the compatibility
# extension. Kept here as a commented reference for server-side usage.
# ----------------------------------------------------------------------
# export NCCL_DEBUG=INFO
# export CUDA_VISIBLE_DEVICES=0
# nohup  python rl_tune.py \
#     --base_model 'textattack/bert-base-uncased-MRPC' \
#     --data_path 'mrpc' \
#     --output_dir $3 \
#     --batch_size 16  --micro_batch_size 16 --num_epochs 1 \
#     --learning_rate 2e-4 --cutoff_len 256 --val_set_size 120 \
#     --eval_step 80  --adapter_name lora \
#     --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
#     --lora_r $1 --lora_alpha $2 --rl_lr $4 --degree $5 --use_ist > $3 2>&1 &
#
# # different data for base model
# # --base_model 'textattack/bert-base-uncased-WNLI' \
# # --base_model 'textattack/bert-base-uncased-RTE' \
# # --base_model 'textattack/bert-base-uncased-CoLA' \
# # --base_model 'textattack/bert-base-uncased-QNLI' \
# # --base_model 'textattack/bert-base-uncased-MRPC' \
# # --base_model 'textattack/bert-base-uncased-SST-2' \
# # --base_model 'textattack/bert-base-uncased-STS-B' \
#
# #different data for data_path
# # --data_path 'wnli' \
# # --data_path 'rte' \
# # --data_path 'cola' \
# # --data_path 'qnli' \
# # --data_path 'mrpc' \
# # --data_path 'sst2' \
# # --data_path 'stsb' \
#
# # CUDA_VISIBLE_DEVICES=$4 python commonsense_evaluate.py \
# #     --model LLaMA-7B \
# #     --adapter LoRA \
# #     --dataset boolq \
# #     --base_model 'yahma/llama-7b-hf' \
# #     --batch_size 1 \
# #     --lora_weights $3|tee -a $3/boolq.txt
#
# # CUDA_VISIBLE_DEVICES=$4 python commonsense_evaluate.py \
# #     --model LLaMA-7B \
# #     --adapter LoRA \
# #     --dataset piqa \
# #     --base_model 'yahma/llama-7b-hf' \
# #     --batch_size 1 \
# #     --lora_weights $3|tee -a $3/piqa.txt
#
# # CUDA_VISIBLE_DEVICES=$4 python commonsense_evaluate.py \
# #     --model LLaMA-7B \
# #     --adapter LoRA \
# #     --dataset social_i_qa \
# #     --base_model 'yahma/llama-7b-hf' \
# #     --batch_size 1 \
# #     --lora_weights $3|tee -a $3/social_i_qa.txt
#
# # CUDA_VISIBLE_DEVICES=$4 python commonsense_evaluate.py \
# #     --model LLaMA-7B \
# #     --adapter LoRA \
# #     --dataset hellaswag \
# #     --base_model 'yahma/llama-7b-hf' \
# #     --batch_size 1 \
# #     --lora_weights $3|tee -a $3/hellaswag.txt
#
# # CUDA_VISIBLE_DEVICES=$4 python commonsense_evaluate.py \
# #     --model LLaMA-7B \
# #     --adapter LoRA \
# #     --dataset winogrande \
# #     --base_model 'yahma/llama-7b-hf' \
# #     --batch_size 1 \
# #     --lora_weights $3|tee -a $3/winogrande.txt
#
# # CUDA_VISIBLE_DEVICES=$4 python commonsense_evaluate.py \
# #     --model LLaMA-7B \
# #     --adapter LoRA \
# #     --dataset ARC-Challenge \
# #     --base_model 'yahma/llama-7b-hf' \
# #     --batch_size 1 \
# #     --lora_weights $3|tee -a $3/ARC-Challenge.txt
#
# # CUDA_VISIBLE_DEVICES=$4 python commonsense_evaluate.py \
# #     --model LLaMA-7B \
# #     --adapter LoRA \
# #     --dataset ARC-Easy \
# #     --base_model 'yahma/llama-7b-hf' \
# #     --batch_size 1 \
# #     --lora_weights $3|tee -a $3/ARC-Easy.txt
#
# # CUDA_VISIBLE_DEVICES=$4 python commonsense_evaluate.py \
# #     --model LLaMA-7B \
# #     --adapter LoRA \
# #     --dataset openbookqa \
# #     --base_model 'yahma/llama-7b-hf' \
# #     --batch_size 1 \
# #     --lora_weights $3|tee -a $3/openbookqa.txt
