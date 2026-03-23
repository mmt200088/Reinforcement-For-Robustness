#!/usr/bin/env bash
set -euo pipefail

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
# ---- 第一阶段：GELU/Softmax 强化学习与最终评估 ----
#
#   第一阶段 PPO/贪心搜索 与 第一阶段最终评估（Phase 3+4）在执行开关上可独立控制，
#   但为了避免流程混用，若执行了第一阶段 RL/贪心搜索，则后续配置来源只能是 search。
#   可支持的安全组合如下：
#     - 都运行（默认）
#     - 只跑 RL/搜索不做第一阶段最终评估（--skip-stage1-final-eval）
#     - 只做最终评估不跑 RL（--skip-stage1-rl + --final-eval-source json|manual）
#     - 都跳过第一阶段最终评估且跳过搜索（--skip-stage1-rl --skip-stage1-final-eval
#       且 --final-eval-source 为 json 或 manual，用于只跑第二阶段等）
#
#   --skip-stage1-rl
#       跳过整个第一阶段搜索准备与搜索流程：
#         Phase 1   baseline 建立
#         Phase 1.5 GELU 输入分布分析
#         Phase 2   PPO 搜索
#         Phase 2.5 贪心搜索
#       不会 自动跳过第一阶段最终评估；要跳过请另加 --skip-stage1-final-eval。
#       若 --final-eval-source 为 search 则非法（无搜索结果），须配合 json 或 manual。
#
# ---- 第一阶段：最终评估配置来源 ----
#
#   --final-eval-source search|json|manual
#       最终评估/第二阶段入口选用的 GELU/Softmax 配置来源：
#         search  （默认）使用本次第一阶段搜索得到的最优配置（执行第一阶段 RL 时只能选它）
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
# ---- 第二阶段：噪声 RL 训练（Noise RL Training）----
#
#   --skip-noise-rl
#       跳过第二阶段噪声 RL 训练。
#       注意：跳过 RL 训练 不会 跳过噪声最终评估。
#       可以跳过 RL 训练但仍用 json/manual 配置运行噪声最终评估。
#       默认情况下，第二阶段 RL 训练会在第一阶段配置确定后自动运行。
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
#       第二阶段 RL 训练逻辑位于 noise_rl_module.py 中。
#
# ---- 第二阶段：噪声最终评估 ----
#
#   噪声 RL 训练和噪声最终评估在执行开关上可独立控制，
#   但为了避免流程混用，若执行了噪声 RL 且保留噪声最终评估，
#   则噪声最终评估配置来源只能是 search。
#   可支持的安全组合如下：
#     - 都运行（默认）
#     - 只跑 RL 训练不做最终评估（--skip-noise-final-eval）
#     - 只做最终评估不跑 RL 训练（--skip-noise-rl + --noise-eval-source json/manual）
#     - 都跳过（--skip-noise-rl --skip-noise-final-eval）
#
#   --skip-noise-final-eval
#       跳过第二阶段噪声最终评估。
#       即使运行了噪声 RL 训练，也不执行 PHASE 5.5 的评估对比。
#
#   --noise-eval-source search|json|manual
#       噪声最终评估使用的噪声配置来源：
#         search  （默认）使用本次噪声 RL 搜索得到的最优配置
#         json    从 JSON 文件读取历史保存的噪声配置
#         manual  使用手动指定的噪声配置
#       当 --skip-noise-rl 时，search 不可用（无 RL 结果），
#       需要指定 json 或 manual。
#
#   --noise-eval-config PATH
#       当 --noise-eval-source json 时使用。
#       指定噪声配置 JSON 文件路径，默认为 glue_noise_configs_best_ppo.json。
#       程序会根据当前数据集名称（如 mrpc）自动读取对应条目。
#       JSON 文件格式示例：
#         {
#           "mrpc": {
#             "x": [20,22,24,26,28,30,20,22,24,26,28,30],
#             "wq": [10,12,14,16,18,20,22,10,12,14,16,18]
#             ...（共 7 个噪声类型：x, wq, wk, wv, wo, wffn1, wffn2）
#           }
#         }
#
#   --manual-noise-config '{"x": [...], "wq": [...], ...}'
#       当 --noise-eval-source manual 时使用。
#       手动指定 7 个噪声类型的 scaling factor 数组（JSON 对象格式）。
#       支持短名称（x, wq, wk, wv, wo, wffn1, wffn2）和
#       全名称（input_noise_scaling_factors 等）。
#
#   --noise-eval-repeat N
#       对噪声最终评估的选定配置执行 N 次重复评估，输出 N 次结果
#       及均值/标准差统计。默认为 1（不重复）。
#
# ---- 跳过第一阶段最终评估 ----
#
#   --skip-stage1-final-eval
#       跳过第一阶段的最终评估（Phase 3 + Phase 4），不跑随机对照与灵敏度分析。
#       仍根据 --final-eval-source 解析 GELU/Softmax 后进入第二阶段。
#       若同时未使用 --skip-stage1-rl，则 search 表示使用本次 RL/贪心结果。
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
# 1. 默认完整流程（第一阶段 RL + 最终评估 + 第二阶段噪声 RL + 噪声最终评估）
#    bash llama_7B_LayerImportance.sh 32 64 output.log 20 2
#
# 2. 只跑第一阶段，跳过第二阶段噪声 RL 训练和噪声最终评估
#    bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
#      --skip-noise-rl --skip-noise-final-eval
#
# 3. 从 JSON 读 GELU/Softmax、不跑第一阶段 RL，仍做第一阶段最终评估（与旧版「仅 JSON」等价）
#    bash llama_7B_LayerImportance.sh 32 64 output_json.log 20 2 \
#      --skip-stage1-rl \
#      --final-eval-source json \
#      --final-eval-config glue_configs_best_ppo.json
#
# 4. 手动指定每层配置；若不跑 RL 需同时加 --skip-stage1-rl
#    bash llama_7B_LayerImportance.sh 32 64 output_manual.log 20 2 \
#      --skip-stage1-rl \
#      --final-eval-source manual \
#      --manual-gelu "[1,1,1,4,1,1,1,1,1,1,1,1]" \
#      --manual-softmax "[2,3,4,6,4,4,5,4,4,5,5,2]"
#
# 5. 提高随机对照实验次数
#    bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
#      --perm-trials 30 --cost-trials 30 --budget-trials 30
#
# 6. 跳过噪声 RL 训练，但仍用 JSON 配置运行噪声最终评估
#    bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
#      --skip-noise-rl \
#      --noise-eval-source json \
#      --noise-eval-config glue_noise_configs_best_ppo.json
#
# 7. 跳过第一阶段最终评估，直接用 JSON 配置进入第二阶段噪声 RL
#    bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
#      --skip-stage1-rl \
#      --final-eval-source json \
#      --final-eval-config glue_configs_best_ppo.json \
#      --skip-stage1-final-eval
#
# 8. 跳过 RL 训练 + 手动指定噪声配置，只做噪声最终评估
#    bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
#      --skip-noise-rl \
#      --noise-eval-source manual \
#      --manual-noise-config '{"x":[20,22,24,26,28,30,20,22,24,26,28,30],"wq":[10,12,14,16,18,20,22,10,12,14,16,18],"wk":[10,12,14,16,18,20,22,10,12,14,16,18],"wv":[10,12,14,16,18,20,22,10,12,14,16,18],"wo":[10,12,14,16,18,20,22,10,12,14,16,18],"wffn1":[10,12,14,16,18,20,22,10,12,14,16,18],"wffn2":[10,12,14,16,18,20,22,10,12,14,16,18]}'
#
# 9. 第二阶段噪声配置重复评估 5 次
#    bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
#      --noise-eval-repeat 5
#
# 10. 运行噪声 RL 训练但跳过噪声最终评估（只要 RL 结果，不做评估对比）
#     bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
#       --skip-noise-final-eval
#
# 10b. 运行第一阶段 RL 但跳过第一阶段最终评估（只要搜索曲线，不进 Phase 3/4）
#     bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
#       --skip-stage1-final-eval
#
# 11. 完全跳过两个阶段的搜索/训练，直接手动指定所有配置做最终评估
#     bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
#       --skip-stage1-rl \
#       --final-eval-source manual \
#       --manual-gelu "[1,1,1,1,1,4,1,1,1,1,1,1]" \
#       --manual-softmax "[2,2,5,5,5,2,5,2,5,5,6,2]" \
#       --skip-stage1-final-eval \
#       --skip-noise-rl \
#       --noise-eval-source manual \
#       --manual-noise-config '{"x":[20,22,24,26,28,30,20,22,24,26,28,30],"wq":[10,12,14,16,18,20,22,10,12,14,16,18],"wk":[10,12,14,16,18,20,22,10,12,14,16,18],"wv":[10,12,14,16,18,20,22,10,12,14,16,18],"wo":[10,12,14,16,18,20,22,10,12,14,16,18],"wffn1":[10,12,14,16,18,20,22,10,12,14,16,18],"wffn2":[10,12,14,16,18,20,22,10,12,14,16,18]}'
#
# ======================================================================
# 项目相关说明
# ======================================================================
# - 当前默认数据集：base_model=textattack/bert-base-uncased-MRPC, data_path=mrpc
# - 脚本使用 nohup 后台运行，日志查看：tail -f <logfile_path>
# - 停止任务：ps aux | grep rl_tune.py，然后 kill -9 <PID>
# - 服务器部署时需同步的 Python 文件：
#     final_evaluation_module.py         第一阶段最终评估模块
#     noise_rl_module.py                 第二阶段噪声 RL 模块
#     noise_final_evaluation_module.py   第二阶段噪声最终评估模块
# ======================================================================

usage() {
    echo "用法："
    echo "  bash llama_7B_LayerImportance.sh [lora_r] [lora_alpha] [logfile_path] [rl_lr] [degree] [options]"
    echo
    echo "必填位置参数："
    echo "  lora_r                LoRA rank，当前实验固定传 32。"
    echo "  lora_alpha            LoRA alpha，当前实验固定传 64。"
    echo "  logfile_path          nohup 输出日志路径。"
    echo "  rl_lr                 PPO 学习率控制。若 < 1 则直接作为 LR；"
    echo "                        旧值 20/40 会解释为 20e-6/40e-6。"
    echo "  degree                历史调试参数，固定传 2。"
    echo
    echo "第一阶段：GELU/Softmax 搜索与评估："
    echo "  --skip-stage1-rl          跳过整个第一阶段搜索准备与搜索流程。"
    echo "                            包括 Phase 1 baseline、Phase 1.5 GELU 输入分布分析、"
    echo "                            Phase 2 PPO、Phase 2.5 贪心搜索。"
    echo "  --final-eval-source search|json|manual"
    echo "                            第一阶段配置来源。"
    echo "                            安全约束：若执行第一阶段 RL，则只能为 search；"
    echo "                            若使用 json/manual，则必须加 --skip-stage1-rl。"
    echo "  --final-eval-config PATH  当 source=json 时使用。"
    echo "  --manual-gelu \"[1,1,...]\""
    echo "  --manual-softmax \"[2,2,...]\""
    echo "                            当 source=manual 时必须同时提供。"
    echo "  --random-seed N"
    echo "  --perm-trials N"
    echo "  --cost-trials N"
    echo "  --budget-trials N"
    echo
    echo "第二阶段：噪声 RL 与最终评估："
    echo "  --skip-noise-rl          跳过第二阶段噪声 RL 训练。"
    echo "  --skip-noise-final-eval  跳过第二阶段噪声最终评估。"
    echo "  --noise-eval-source search|json|manual"
    echo "                            噪声最终评估配置来源。"
    echo "                            安全约束：若执行噪声 RL 且保留噪声最终评估，"
    echo "                            则只能为 search；若使用 json/manual，则必须加"
    echo "                            --skip-noise-rl。"
    echo "  --noise-eval-config PATH 当 source=json 时使用。"
    echo "  --manual-noise-config '{\"x\":[...],\"wq\":[...],...}'"
    echo "                            当 source=manual 时必须提供。"
    echo "  --noise-eval-repeat N    噪声最终评估重复次数，默认 1。"
    echo
    echo "其他开关："
    echo "  --skip-stage1-final-eval 跳过第一阶段最终评估（Phase 3+4），"
    echo "                            仍根据第一阶段配置来源解析 GELU/Softmax 后进入第二阶段。"
    echo
    echo "示例："
    echo "  bash llama_7B_LayerImportance.sh 32 64 output.log 20 2"
    echo "  bash llama_7B_LayerImportance.sh 32 64 output_json.log 20 2 --skip-stage1-rl --final-eval-source json --final-eval-config glue_configs_best_ppo.json"
    echo "  bash llama_7B_LayerImportance.sh 32 64 output_manual.log 20 2 --skip-stage1-rl --final-eval-source manual --manual-gelu \"[1,1,1,4,1,1,1,1,1,1,1,1]\" --manual-softmax \"[2,3,4,6,4,4,5,4,4,5,5,2]\""
    echo "  bash llama_7B_LayerImportance.sh 32 64 output_noise.log 20 2 --skip-noise-rl --noise-eval-source json --noise-eval-config glue_noise_configs_best_ppo.json"
    echo "  bash llama_7B_LayerImportance.sh 32 64 output_manual_all.log 20 2 --skip-stage1-rl --final-eval-source manual --manual-gelu \"[1,1,...]\" --manual-softmax \"[2,2,...]\" --skip-stage1-final-eval --skip-noise-rl --noise-eval-source manual --manual-noise-config '{\"x\":[...],...}'"
}

require_option_value() {
    if [ "$#" -lt 2 ]; then
        echo "错误: 选项 $1 缺少取值。" >&2
        exit 1
    fi
}

error_exit() {
    echo "错误: $1" >&2
    exit 1
}

is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

if [ "$#" -eq 1 ] && { [ "$1" = "-h" ] || [ "$1" = "--help" ]; }; then
    usage
    exit 0
fi

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
FINAL_EVAL_CONFIG_PATH_SPECIFIED="false"
MANUAL_FINAL_GELU_SPECIFIED="false"
MANUAL_FINAL_SOFTMAX_SPECIFIED="false"
FINAL_EVAL_RANDOM_SEED="42"
FINAL_EVAL_PERMUTATION_TRIALS="10"
FINAL_EVAL_COST_EQUIVALENT_TRIALS="10"
FINAL_EVAL_BUDGET_EQUIVALENT_TRIALS="10"
SKIP_NOISE_RL="false"
NOISE_EVAL_SOURCE="search"
NOISE_EVAL_CONFIG_PATH="glue_noise_configs_best_ppo.json"
MANUAL_NOISE_CONFIG=""
NOISE_EVAL_CONFIG_PATH_SPECIFIED="false"
MANUAL_NOISE_CONFIG_SPECIFIED="false"
NOISE_EVAL_REPEAT_N="1"
SKIP_STAGE1_RL="false"
SKIP_STAGE1_FINAL_EVAL="false"
SKIP_NOISE_FINAL_EVAL="false"

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
            FINAL_EVAL_CONFIG_PATH_SPECIFIED="true"
            shift 2
            ;;
        --manual-gelu)
            require_option_value "$@"
            MANUAL_FINAL_GELU="$2"
            MANUAL_FINAL_GELU_SPECIFIED="true"
            shift 2
            ;;
        --manual-softmax)
            require_option_value "$@"
            MANUAL_FINAL_SOFTMAX="$2"
            MANUAL_FINAL_SOFTMAX_SPECIFIED="true"
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
        --noise-eval-source)
            require_option_value "$@"
            NOISE_EVAL_SOURCE="$2"
            shift 2
            ;;
        --noise-eval-config|--noise-eval-config-path)
            require_option_value "$@"
            NOISE_EVAL_CONFIG_PATH="$2"
            NOISE_EVAL_CONFIG_PATH_SPECIFIED="true"
            shift 2
            ;;
        --manual-noise-config)
            require_option_value "$@"
            MANUAL_NOISE_CONFIG="$2"
            MANUAL_NOISE_CONFIG_SPECIFIED="true"
            shift 2
            ;;
        --noise-eval-repeat)
            require_option_value "$@"
            NOISE_EVAL_REPEAT_N="$2"
            shift 2
            ;;
        --skip-stage1-rl)
            SKIP_STAGE1_RL="true"
            shift 1
            ;;
        --skip-stage1-final-eval)
            SKIP_STAGE1_FINAL_EVAL="true"
            shift 1
            ;;
        --skip-noise-final-eval)
            SKIP_NOISE_FINAL_EVAL="true"
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

case "$NOISE_EVAL_SOURCE" in
    search|json|manual)
        ;;
    *)
        echo "Invalid --noise-eval-source: $NOISE_EVAL_SOURCE"
        echo "Expected one of: search, json, manual"
        exit 1
        ;;
esac

if ! is_positive_integer "$NOISE_EVAL_REPEAT_N"; then
    error_exit "--noise-eval-repeat 必须是正整数，当前值为 '$NOISE_EVAL_REPEAT_N'。"
fi

if [ "$NOISE_EVAL_SOURCE" = "manual" ]; then
    if [ -z "$MANUAL_NOISE_CONFIG" ]; then
        error_exit "当 --noise-eval-source=manual 时，必须提供 --manual-noise-config。"
    fi
elif [ "$MANUAL_NOISE_CONFIG_SPECIFIED" = "true" ]; then
    error_exit "只有在 --noise-eval-source=manual 时才能提供 --manual-noise-config。"
fi

if [ "$FINAL_EVAL_SOURCE" = "manual" ]; then
    if [ -z "$MANUAL_FINAL_GELU" ] || [ -z "$MANUAL_FINAL_SOFTMAX" ]; then
        error_exit "当 --final-eval-source=manual 时，必须同时提供 --manual-gelu 和 --manual-softmax。"
    fi
elif [ "$MANUAL_FINAL_GELU_SPECIFIED" = "true" ] || [ "$MANUAL_FINAL_SOFTMAX_SPECIFIED" = "true" ]; then
    error_exit "只有在 --final-eval-source=manual 时才能提供 --manual-gelu / --manual-softmax。"
fi

if [ "$SKIP_STAGE1_RL" = "true" ] && [ "$FINAL_EVAL_SOURCE" = "search" ]; then
    error_exit "--skip-stage1-rl 与 --final-eval-source search 不能同时使用：跳过第一阶段搜索后没有搜索结果可供评估。请改用 json/manual，或去掉 --skip-stage1-rl。"
fi

if [ "$SKIP_STAGE1_RL" = "false" ] && [ "$FINAL_EVAL_SOURCE" != "search" ]; then
    error_exit "检测到第一阶段 RL/贪心搜索将执行，但 --final-eval-source=$FINAL_EVAL_SOURCE。为避免“前面跑 RL、后面却用手动/JSON 配置评估”的流程混用，执行第一阶段 RL 时只能使用 search。若要使用 json/manual，请添加 --skip-stage1-rl。"
fi

if [ "$FINAL_EVAL_SOURCE" = "json" ]; then
    [ -f "$FINAL_EVAL_CONFIG_PATH" ] || error_exit "第一阶段 JSON 配置文件不存在：$FINAL_EVAL_CONFIG_PATH"
elif [ "$FINAL_EVAL_CONFIG_PATH_SPECIFIED" = "true" ]; then
    error_exit "只有在 --final-eval-source=json 时才能提供 --final-eval-config。"
fi

if [ "$SKIP_NOISE_FINAL_EVAL" = "false" ] && [ "$SKIP_NOISE_RL" = "true" ] && [ "$NOISE_EVAL_SOURCE" = "search" ]; then
    error_exit "--skip-noise-rl 与 --noise-eval-source search 不能同时用于噪声最终评估：跳过噪声 RL 后没有搜索结果可供评估。请改用 json/manual，或去掉 --skip-noise-rl。"
fi

if [ "$SKIP_NOISE_FINAL_EVAL" = "false" ] && [ "$SKIP_NOISE_RL" = "false" ] && [ "$NOISE_EVAL_SOURCE" != "search" ]; then
    error_exit "检测到第二阶段噪声 RL 将执行，且噪声最终评估未跳过，但 --noise-eval-source=$NOISE_EVAL_SOURCE。为避免“前面跑噪声 RL、后面却用手动/JSON 配置评估”的流程混用，此时只能使用 search。若要使用 json/manual，请添加 --skip-noise-rl。"
fi

if [ "$SKIP_NOISE_FINAL_EVAL" = "false" ] && [ "$NOISE_EVAL_SOURCE" = "json" ]; then
    [ -f "$NOISE_EVAL_CONFIG_PATH" ] || error_exit "噪声 JSON 配置文件不存在：$NOISE_EVAL_CONFIG_PATH"
elif [ "$NOISE_EVAL_CONFIG_PATH_SPECIFIED" = "true" ] && [ "$SKIP_NOISE_FINAL_EVAL" = "false" ]; then
    error_exit "只有在 --noise-eval-source=json 时才能提供 --noise-eval-config。"
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
    --noise_eval_config_source "$NOISE_EVAL_SOURCE"
    --noise_eval_config_path "$NOISE_EVAL_CONFIG_PATH"
    --manual_noise_config "$MANUAL_NOISE_CONFIG"
    --noise_eval_repeat_n "$NOISE_EVAL_REPEAT_N"
    --skip_stage1_rl "$SKIP_STAGE1_RL"
    --skip_stage1_final_eval "$SKIP_STAGE1_FINAL_EVAL"
    --skip_noise_final_eval "$SKIP_NOISE_FINAL_EVAL"
)

echo "Launching RL tune job with final evaluation source: $FINAL_EVAL_SOURCE"
if [ "$SKIP_STAGE1_RL" = "true" ]; then
    echo "Stage-1 RL/greedy search: SKIPPED (--skip-stage1-rl)"
else
    echo "Stage-1 RL/greedy search: will run (unless SEARCH_MODE in code disables it)."
fi
if [ "$SKIP_STAGE1_FINAL_EVAL" = "true" ]; then
    echo "Stage-1 final evaluation: SKIPPED (--skip-stage1-final-eval)"
fi
if [ "$SKIP_NOISE_RL" = "true" ]; then
    echo "Stage-2 noise RL training: SKIPPED (--skip-noise-rl)"
else
    echo "Stage-2 noise RL training will run after the fixed GELU/Softmax config is selected."
fi
if [ "$SKIP_NOISE_FINAL_EVAL" = "true" ]; then
    echo "Stage-2 noise final evaluation: SKIPPED (--skip-noise-final-eval)"
else
    echo "Stage-2 noise final evaluation: noise_eval_source=$NOISE_EVAL_SOURCE, repeat=$NOISE_EVAL_REPEAT_N"
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
