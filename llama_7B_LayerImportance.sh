#!/usr/bin/env bash

# ======================================================================
# 用途说明
# ----------------------------------------------------------------------
# 这个脚本是项目当前的主启动脚本，用来在服务器上后台启动 `rl_tune.py`，
# 进行基于强化学习的层重要性搜索，并在最后执行 FINAL EVALUATION。
#
# 这个脚本现在同时支持两种使用方式：
# 1. 兼容旧版的 5 个位置参数调用方式
# 2. 在旧版命令后面继续追加可选命名参数，控制最终测试阶段的配置来源
#
# ----------------------------------------------------------------------
# 一、最常用的旧版运行方式
# ----------------------------------------------------------------------
# 命令格式：
#   bash llama_7B_LayerImportance.sh [lora_r] [lora_alpha] [logfile_path] [rl_lr] [degree]
#
# 参数含义：
#   $1 lora_r
#      LoRA 的 rank。当前实验里通常固定写 32。
#
#   $2 lora_alpha
#      LoRA 的 alpha。当前实验里通常固定写 64。
#
#   $3 logfile_path
#      nohup 的日志输出文件路径。训练日志和最终评估日志都会写到这里。
#
#   $4 rl_lr
#      强化学习更新时的学习率。你目前常用 20、30、40 这一类设置。
#
#   $5 degree
#      旧版调试参数，现在基本废弃，按原项目习惯继续传 2 即可。
#
# 典型命令：
#   bash llama_7B_LayerImportance.sh 32 64 output.log 20 2
#
# 兼容性说明：
#   上面这条旧命令仍然完全可用，不需要修改。
#
# ----------------------------------------------------------------------
# 二、脚本当前默认行为
# ----------------------------------------------------------------------
# 如果你只传前 5 个参数，那么脚本会：
# 1. 正常运行 RL 搜索
# 2. 最终评估阶段默认使用本次 RL 搜索得到的配置
# 3. 随机对照实验会一起运行
#
# 也就是说，默认等价于：
#   --final-eval-source search
#
# ----------------------------------------------------------------------
# 三、可选的 FINAL EVALUATION 扩展参数
# ----------------------------------------------------------------------
# 你现在可以在前 5 个位置参数之后，继续追加下面这些命名参数：
#
#   --final-eval-source search|json|manual
#      指定最终评估配置的来源：
#      - search：使用当前这次 RL/greedy 搜索得到的配置
#      - json：从 JSON 文件读取当前数据集对应的配置
#      - manual：手动输入每层 GELU / Softmax 配置
#
#   --final-eval-config PATH
#      当 --final-eval-source json 时使用，指定 JSON 配置文件路径。
#      例如：glue_configs_best_ppo.json
#
#   --manual-gelu "[1,1,1,...]"
#      当 --final-eval-source manual 时使用，手动指定每层 GELU degree。
#
#   --manual-softmax "[2,2,2,...]"
#      当 --final-eval-source manual 时使用，手动指定每层 Softmax degree。
#
#   --random-seed N
#      随机实验的随机种子。
#
#   --perm-trials N
#      permutation 随机实验次数。
#
#   --cost-trials N
#      精确 cost-matched 随机实验次数。
#
#   --budget-trials N
#      同总预算随机实验次数。
#
# ----------------------------------------------------------------------
# 四、三种最终评估模式怎么选
# ----------------------------------------------------------------------
# 1. search 模式
#    这是默认模式，适合“先跑 RL，再测试 RL 学到的配置”。
#
# 2. json 模式
#    适合“跳过当前 RL 搜索，直接使用历史保存的配置文件做最终测试”。
#    程序会根据当前任务的数据集名字自动读取 JSON 中对应条目。
#    比如当前 data_path 是 mrpc，就会读取 JSON 里的 "mrpc" 配置。
#
# 3. manual 模式
#    适合“跳过 RL，直接测试你手动指定的一组层级配置”。
#    这种模式下必须同时提供：
#    - --manual-gelu
#    - --manual-softmax
#
# ----------------------------------------------------------------------
# 五、使用示例
# ----------------------------------------------------------------------
# 1. 默认方式：跑 RL，并用 RL 学到的配置做最终评估
#    bash llama_7B_LayerImportance.sh 32 64 output.log 20 2
#
# 2. 使用 JSON 文件中的配置做最终评估
#    bash llama_7B_LayerImportance.sh 32 64 output_json.log 20 2 \
#      --final-eval-source json \
#      --final-eval-config glue_configs_best_ppo.json
#
# 3. 使用手动输入的配置做最终评估
#    bash llama_7B_LayerImportance.sh 32 64 output_manual.log 20 2 \
#      --final-eval-source manual \
#      --manual-gelu "[1,1,1,4,1,1,1,1,1,1,1,1]" \
#      --manual-softmax "[2,3,4,6,4,4,5,4,4,5,5,2]"
#
# 4. 提高随机对照实验次数
#    bash llama_7B_LayerImportance.sh 32 64 output_more_random.log 20 2 \
#      --perm-trials 30 \
#      --cost-trials 30 \
#      --budget-trials 30
#
# ----------------------------------------------------------------------
# 六、与当前项目设置相关的说明
# ----------------------------------------------------------------------
# 1. 这个脚本目前仍然保持原来的数据集设置，没有在这里改动任务数据集。
#    当前默认写的是：
#      --base_model "textattack/bert-base-uncased-MRPC"
#      --data_path "mrpc"
#
# 2. 这个脚本只负责把参数传给 `rl_tune.py`，不会改动你原来在项目里的
#    训练/验证/测试划分逻辑。
#
# 3. 如果你在服务器上运行，记得把新增的 Python 文件一起同步过去：
#      final_evaluation_module.py
#    否则 FINAL EVALUATION 的新逻辑无法导入。
#
# 4. 这个脚本仍然是后台运行方式：
#      nohup ... > logfile 2>&1 &
#    所以提交命令后，日志需要查看你传入的 logfile_path。
#
# ----------------------------------------------------------------------
# 七、如何停止任务
# ----------------------------------------------------------------------
# 可以先查看进程：
#   ps aux | grep rl_tune.py
# 然后再手动 kill 对应进程。
# ======================================================================

usage() {
    echo "Usage:"
    echo "  bash llama_7B_LayerImportance.sh [lora_r] [lora_alpha] [logfile_path] [rl_lr] [degree] [options]"
    echo
    echo "Required positional arguments:"
    echo "  lora_r                LoRA rank, keep 32 in current experiments."
    echo "  lora_alpha            LoRA alpha, keep 64 in current experiments."
    echo "  logfile_path          Output log path for nohup."
    echo "  rl_lr                 Reinforcement learning rate."
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
    echo "Examples:"
    echo "  bash llama_7B_LayerImportance.sh 32 64 output.log 20 2"
    echo "  bash llama_7B_LayerImportance.sh 32 64 output_json.log 20 2 --final-eval-source json --final-eval-config glue_configs_best_ppo.json"
    echo "  bash llama_7B_LayerImportance.sh 32 64 output_manual.log 20 2 --final-eval-source manual --manual-gelu \"[1,1,1,4,1,1,1,1,1,1,1,1]\" --manual-softmax \"[2,3,4,6,4,4,5,4,4,5,5,2]\""
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
)

echo "Launching RL tune job with final evaluation source: $FINAL_EVAL_SOURCE"
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
