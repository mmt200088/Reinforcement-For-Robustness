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
#   $3  logfile_path  nohup 日志文件名提示；实际日志会写入自动生成的 run 目录
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
# ---- 数据集选择 ----
#
#   --model DATASET
#       选择数据集及对应的预训练模型，自动设置 --base_model 和 --data_path。
#       同时 rl_tune.py 会自动适配 tokenizer 输入列和 num_labels。
#       支持的值：mrpc, sst2, stsb, cola, qnli, rte, wnli
#       默认值：mrpc
#       参数大小写不敏感（如 --model MRPC 与 --model mrpc 等价）。
#       映射关系：
#         mrpc → textattack/bert-base-uncased-MRPC
#         sst2 → textattack/bert-base-uncased-SST-2
#         stsb → textattack/bert-base-uncased-STS-B
#         cola → textattack/bert-base-uncased-CoLA
#         qnli → textattack/bert-base-uncased-QNLI
#         rte  → textattack/bert-base-uncased-RTE
#         wnli → textattack/bert-base-uncased-WNLI
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
# 1. 默认完整流程（使用 mrpc 数据集）
#    bash llama_7B_LayerImportance.sh 32 64 output.log 20 2
#
# 1b. 切换到 stsb 数据集运行完整流程
#    bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 --model stsb
#
# 1c. 切换到 sst2 数据集
#    bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 --model sst2
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
# - 当前默认数据集：mrpc（可通过 --model 参数切换）
# - 脚本使用 nohup 后台运行；实际日志路径会在启动时打印出来
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
    echo "  logfile_path          nohup 日志文件名提示。"
    echo "                        实际日志路径会自动解析到"
    echo "                        rl_results/layer_importance_runs/<dataset>/<timestamp>_pid<PID>/logs/"
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
    echo "数据集选择："
    echo "  --model DATASET          选择数据集及对应的预训练模型。"
    echo "                            支持: mrpc, sst2, stsb, cola, qnli, rte, wnli"
    echo "                            默认: mrpc"
    echo "                            大小写不敏感: MRPC/stsb/QNLI 均可。"
    echo "                            映射: mrpc->...-MRPC, sst2->...-SST-2, stsb->...-STS-B,"
    echo "                                  cola->...-CoLA, qnli->...-QNLI, rte->...-RTE, wnli->...-WNLI"
    echo
    echo "其他开关："
    echo "  --skip-stage1-final-eval 跳过第一阶段最终评估（Phase 3+4），"
    echo "                            仍根据第一阶段配置来源解析 GELU/Softmax 后进入第二阶段。"
    echo
    echo "断点续训："
    echo "  --resume-from PATH       从之前的 run 目录恢复 RL 训练。"
    echo "                            PATH 为之前的 run 目录路径，例如："
    echo "                            rl_results/layer_importance_runs/mrpc/20260404_151155_pid711833"
    echo "                            程序会自动在该目录下查找 checkpoint 文件。"
    echo "                            续训时 --stage1/2-rl-episodes 表示总轮数（非追加轮数）。"
    echo
    echo "示例："
    echo "  bash llama_7B_LayerImportance.sh 32 64 output.log 20 2"
    echo "  bash llama_7B_LayerImportance.sh 32 64 output_json.log 20 2 --skip-stage1-rl --final-eval-source json --final-eval-config glue_configs_best_ppo.json"
    echo "  bash llama_7B_LayerImportance.sh 32 64 output_manual.log 20 2 --skip-stage1-rl --final-eval-source manual --manual-gelu \"[1,1,1,4,1,1,1,1,1,1,1,1]\" --manual-softmax \"[2,3,4,6,4,4,5,4,4,5,5,2]\""
    echo "  bash llama_7B_LayerImportance.sh 32 64 output_noise.log 20 2 --skip-noise-rl --noise-eval-source json --noise-eval-config glue_noise_configs_best_ppo.json"
    echo "  bash llama_7B_LayerImportance.sh 32 64 output_manual_all.log 20 2 --skip-stage1-rl --final-eval-source manual --manual-gelu \"[1,1,...]\" --manual-softmax \"[2,2,...]\" --skip-stage1-final-eval --skip-noise-rl --noise-eval-source manual --manual-noise-config '{\"x\":[...],...}'"
    echo "  bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 --skip-stage1-rl --final-eval-source json --final-eval-config glue_configs_best_ppo.json --skip-stage1-final-eval --noise-eval-repeat 200 --model mrpc"
    echo "  bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 --skip-stage1-rl --final-eval-source json --final-eval-config glue_configs_best_ppo.json --skip-stage1-final-eval --noise-eval-repeat 200 --model stsb"
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

infer_config_algorithm_family() {
    local lowered
    lowered="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
    if [[ "$lowered" == *genetic* ]] || [[ "$lowered" == *_ga.* ]] || [[ "$lowered" == *-ga.* ]] || [[ "$lowered" == ga_* ]]; then
        echo "ga"
    elif [[ "$lowered" == *ppo* ]]; then
        echo "rl"
    else
        echo "unknown"
    fi
}

usage() {
    cat <<'EOF'
用法:
  bash llama_7B_LayerImportance.sh [lora_r] [lora_alpha] [logfile_path] [rl_lr] [degree] [options]

必填位置参数:
  lora_r                LoRA rank，当前实验固定传 32
  lora_alpha            LoRA alpha，当前实验固定传 64
  logfile_path          nohup 日志文件名提示；实际日志位于当前 run 目录下的 logs/
  rl_lr                 兼容历史命令而保留。RL 模式下作为 PPO 学习率；GA 模式下保留但不参与遗传搜索决策
  degree                历史调试参数，固定传 2

搜索算法选择:
  --search-algorithm rl|ga
                        rl: 现有强化学习 / PPO 流程，入口为 rl_tune.py
                        ga: 新增遗传算法流程，入口为 rl_tune_genetic.py
                        默认值: rl

第一阶段: GELU / Softmax 搜索与最终评估
  --skip-stage1-search
                        推荐的算法无关写法。跳过整个第一阶段搜索流程
  --skip-stage1-rl
                        RL 兼容别名。仅在 --search-algorithm=rl 时允许使用
  --stage1-search-episodes N
                        推荐的算法无关写法。设置第一阶段搜索预算
  --stage1-rl-episodes N
                        RL 兼容别名。仅在 --search-algorithm=rl 时允许使用
  --final-eval-source search|json|manual
                        第一阶段最终评估配置来源
                        若执行第一阶段搜索，则只能为 search
                        若使用 json/manual，则必须跳过第一阶段搜索
  --final-eval-config PATH
                        仅在 --final-eval-source=json 时使用
                        rl 默认: glue_configs_best_ppo.json
                        ga 默认: glue_configs_best_genetic.json
  --manual-gelu "[1,1,...]"
  --manual-softmax "[2,2,...]"
                        仅在 --final-eval-source=manual 时同时提供
  --skip-stage1-final-eval
                        跳过第一阶段最终评估，但仍会解析进入第二阶段所需的固定配置

第二阶段: 噪声搜索与最终评估
  --skip-noise-search
                        推荐的算法无关写法。跳过第二阶段噪声搜索
  --skip-noise-rl
                        RL 兼容别名。仅在 --search-algorithm=rl 时允许使用
  --stage2-search-episodes N
                        推荐的算法无关写法。设置第二阶段搜索预算
  --stage2-rl-episodes N
                        RL 兼容别名。仅在 --search-algorithm=rl 时允许使用
  --skip-noise-final-eval
                        跳过第二阶段噪声最终评估
  --noise-eval-source search|json|manual
                        第二阶段噪声最终评估配置来源
                        若执行第二阶段搜索且保留最终评估，则只能为 search
                        若使用 json/manual，则必须跳过第二阶段搜索
  --noise-eval-config PATH
                        仅在 --noise-eval-source=json 时使用
                        rl 默认: glue_noise_configs_best_ppo.json
                        ga 默认: glue_noise_configs_best_genetic.json
  --manual-noise-config '{"x":[...],"wq":[...],...}'
                        仅在 --noise-eval-source=manual 时提供
  --noise-eval-repeat N
                        噪声最终评估重复次数，默认 1

一致性检查:
  1. 选择 --search-algorithm=ga 后，不允许再使用 RL 命名的兼容别名
     例如 --skip-stage1-rl、--skip-noise-rl、--stage1-rl-episodes、--stage2-rl-episodes
  2. 选择 --search-algorithm=ga 且使用 json 配置时，不允许继续使用 PPO/RL 家族配置文件
     例如 glue_configs_best_ppo.json、glue_noise_configs_best_ppo.json
  3. 选择 --search-algorithm=rl 且使用 json 配置时，不允许使用 genetic/ga 家族配置文件
     例如 glue_configs_best_genetic.json、glue_noise_configs_best_genetic.json
  4. 自定义 JSON 文件名若不包含 ppo / genetic / _ga 等算法标识，则按自定义路径处理，不额外拦截

其他:
  --model DATASET          支持: mrpc, sst2, stsb, cola, qnli, rte, wnli
  --model-type TYPE        支持: bert-base, bert-large, gpt-2
  --resume-from PATH       从历史 run 目录恢复未跳过阶段的搜索进度

示例:
  bash llama_7B_LayerImportance.sh 32 64 output.log 20 2
  bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 --search-algorithm ga
  bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 --search-algorithm rl --skip-stage1-rl --final-eval-source json --final-eval-config glue_configs_best_ppo.json
  bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 --search-algorithm ga --skip-stage1-search --final-eval-source json --final-eval-config glue_configs_best_genetic.json
  bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 --search-algorithm ga --skip-noise-search --noise-eval-source manual --manual-noise-config '{"x":[...],"wq":[...],...}'
EOF
}

usage() {
    cat <<'EOF'
Usage:
  bash llama_7B_LayerImportance.sh [lora_r] [lora_alpha] [logfile_path] [rl_lr] [degree] [options]

Required positional args:
  lora_r
  lora_alpha
  logfile_path
  rl_lr
  degree

Common options:
  --batch_size N
      Sets both --batch_size and --micro_batch_size to N.
      Also propagates the same batch size into rl_tune.py evaluation
      and layer_importance_evaluator.py dataloaders.
      Default: 16

  --stage1-rl-episodes N
      Sets the Stage-1 RL episode count. Default: 51000
      Must be >= 170 when Stage-1 RL is enabled.

  --stage2-rl-episodes N
      Sets the Stage-2 noise RL episode count. Default: 40000
      Must be >= 170 when Stage-2 RL is enabled.

  --model DATASET
      Supported: mrpc, sst2, stsb, cola, qnli, rte, wnli

  --model-type {bert-base|bert-large|gpt-2}
      Select the underlying pretrained backbone. Default: bert-base.
      bert-base  uses textattack/bert-base-uncased-* (12 layers).
      bert-large uses yoshitomo-matsubara/bert-large-uncased-* (24 layers).
      gpt-2      uses openai-community/gpt2 (12 layers) for all tasks.
                 Note: GPT-2 的 SequenceClassification head 在 HuggingFace 上
                 没有覆盖 GLUE 全任务的公开微调权重, 此处所有任务共用同一个
                 基座权重, 分类 head 由 rl_tune.py 的训练流水线自行初始化并
                 微调. 另外, GPT-2 路径当前不支持 softmax 近似 (Stage 1 的
                 softmax 部分自动跳过), 只启用 GELU 近似与 Stage 2 噪声注入.
      Note: bert-large currently supports the following GLUE tasks only:
        mrpc, cola, stsb, rte, sst2, qnli
      Picking an unsupported task with --model-type bert-large will exit
      with a clear "current not supported" error.

  --skip-stage1-rl
  --skip-stage1-final-eval
  --skip-noise-rl
  --skip-noise-final-eval
  --final-eval-source search|json|manual
  --final-eval-config PATH
  --noise-eval-source search|json|manual
  --noise-eval-config PATH
  --noise-eval-repeat N

  --resume-from PATH
      Resume RL training from a previous run directory.
      PATH should be a previous run directory, e.g.:
        rl_results/layer_importance_runs/mrpc/20260404_151155_pid711833
      The system will look for checkpoint files in:
        <PATH>/stage1/stage1_rl_checkpoint.pt  (Stage-1)
        <PATH>/stage2_noise/progress/noise_rl_checkpoint.pt  (Stage-2)

  -h, --help

Examples:
  bash llama_7B_LayerImportance.sh 32 64 output.log 20 2
  bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 --batch_size 8 --model mrpc
  bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 --stage1-rl-episodes 1020 --stage2-rl-episodes 3400 --model mrpc
  bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 --skip-stage1-rl --final-eval-source json --final-eval-config glue_configs_best_ppo.json --skip-stage1-final-eval --noise-eval-repeat 200 --model mrpc
EOF
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
RAW_LOGFILE_PATH="$3"
RL_LR="$4"
DEGREE="$5"
shift 5

SEARCH_ALGORITHM="rl"
FINAL_EVAL_SOURCE="search"
FINAL_EVAL_CONFIG_PATH=""
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
NOISE_EVAL_CONFIG_PATH=""
MANUAL_NOISE_CONFIG=""
NOISE_EVAL_CONFIG_PATH_SPECIFIED="false"
MANUAL_NOISE_CONFIG_SPECIFIED="false"
NOISE_EVAL_REPEAT_N="1"
SKIP_STAGE1_RL="false"
SKIP_STAGE1_FINAL_EVAL="false"
SKIP_NOISE_FINAL_EVAL="false"
DATASET="mrpc"
MODEL_TYPE="bert-base"
BATCH_SIZE="16"
STAGE1_RL_EPISODES="51000"
STAGE2_RL_EPISODES="40000"
STAGE1_RL_EPISODES_SPECIFIED="false"
STAGE2_RL_EPISODES_SPECIFIED="false"
SEARCH_ALGORITHM_SPECIFIED="false"
LEGACY_STAGE1_SKIP_ALIAS_USED="false"
LEGACY_STAGE2_SKIP_ALIAS_USED="false"
LEGACY_STAGE1_EPISODES_ALIAS_USED="false"
LEGACY_STAGE2_EPISODES_ALIAS_USED="false"
MIN_RL_EPISODES="170"
RESUME_RUN_DIR=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --search-algorithm|--search_algorithm|--algorithm)
            require_option_value "$@"
            SEARCH_ALGORITHM="$2"
            SEARCH_ALGORITHM_SPECIFIED="true"
            shift 2
            ;;
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
        --skip-noise-search|--skip-stage2-search)
            SKIP_NOISE_RL="true"
            shift 1
            ;;
        --skip-noise-rl)
            SKIP_NOISE_RL="true"
            LEGACY_STAGE2_SKIP_ALIAS_USED="true"
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
        --batch_size|--batch-size)
            require_option_value "$@"
            BATCH_SIZE="$2"
            shift 2
            ;;
        --stage1-search-episodes)
            require_option_value "$@"
            STAGE1_RL_EPISODES="$2"
            STAGE1_RL_EPISODES_SPECIFIED="true"
            shift 2
            ;;
        --stage1-rl-episodes)
            require_option_value "$@"
            STAGE1_RL_EPISODES="$2"
            STAGE1_RL_EPISODES_SPECIFIED="true"
            LEGACY_STAGE1_EPISODES_ALIAS_USED="true"
            shift 2
            ;;
        --stage2-search-episodes|--noise-search-episodes)
            require_option_value "$@"
            STAGE2_RL_EPISODES="$2"
            STAGE2_RL_EPISODES_SPECIFIED="true"
            shift 2
            ;;
        --stage2-rl-episodes)
            require_option_value "$@"
            STAGE2_RL_EPISODES="$2"
            STAGE2_RL_EPISODES_SPECIFIED="true"
            LEGACY_STAGE2_EPISODES_ALIAS_USED="true"
            shift 2
            ;;
        --skip-stage1-search)
            SKIP_STAGE1_RL="true"
            shift 1
            ;;
        --skip-stage1-rl)
            SKIP_STAGE1_RL="true"
            LEGACY_STAGE1_SKIP_ALIAS_USED="true"
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
        --model)
            require_option_value "$@"
            DATASET="$2"
            shift 2
            ;;
        --model-type|--model_type)
            require_option_value "$@"
            MODEL_TYPE="$2"
            shift 2
            ;;
        --resume-from)
            require_option_value "$@"
            RESUME_RUN_DIR="$2"
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

DATASET="$(printf '%s' "$DATASET" | tr '[:upper:]' '[:lower:]')"

SEARCH_ALGORITHM="$(printf '%s' "$SEARCH_ALGORITHM" | tr '[:upper:]' '[:lower:]')"
case "$SEARCH_ALGORITHM" in
    rl|ppo)
        SEARCH_ALGORITHM="rl"
        ;;
    ga|genetic)
        SEARCH_ALGORITHM="ga"
        ;;
    *)
        error_exit "不支持的 --search-algorithm: $SEARCH_ALGORITHM。支持的选项: rl, ga"
        ;;
esac

if [ "$FINAL_EVAL_CONFIG_PATH_SPECIFIED" = "false" ]; then
    if [ "$SEARCH_ALGORITHM" = "ga" ]; then
        FINAL_EVAL_CONFIG_PATH="glue_configs_best_genetic.json"
    else
        FINAL_EVAL_CONFIG_PATH="glue_configs_best_ppo.json"
    fi
fi

if [ "$NOISE_EVAL_CONFIG_PATH_SPECIFIED" = "false" ]; then
    if [ "$SEARCH_ALGORITHM" = "ga" ]; then
        NOISE_EVAL_CONFIG_PATH="glue_noise_configs_best_genetic.json"
    else
        NOISE_EVAL_CONFIG_PATH="glue_noise_configs_best_ppo.json"
    fi
fi

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

if [ "$SEARCH_ALGORITHM" = "ga" ] && {
    [ "$LEGACY_STAGE1_SKIP_ALIAS_USED" = "true" ] ||
    [ "$LEGACY_STAGE2_SKIP_ALIAS_USED" = "true" ] ||
    [ "$LEGACY_STAGE1_EPISODES_ALIAS_USED" = "true" ] ||
    [ "$LEGACY_STAGE2_EPISODES_ALIAS_USED" = "true" ]
}; then
    error_exit "已选择 --search-algorithm=ga，但仍使用了 RL 命名的兼容选项。请改用算法无关写法：--skip-stage1-search / --skip-noise-search / --stage1-search-episodes / --stage2-search-episodes。"
fi

if ! is_positive_integer "$NOISE_EVAL_REPEAT_N"; then
    error_exit "--noise-eval-repeat 必须是正整数，当前值为 '$NOISE_EVAL_REPEAT_N'。"
fi

if ! is_positive_integer "$BATCH_SIZE"; then
    error_exit "--batch_size must be a positive integer, got '$BATCH_SIZE'."
fi

if ! is_positive_integer "$STAGE1_RL_EPISODES"; then
    error_exit "--stage1-rl-episodes must be a positive integer, got '$STAGE1_RL_EPISODES'."
fi

if ! is_positive_integer "$STAGE2_RL_EPISODES"; then
    error_exit "--stage2-rl-episodes must be a positive integer, got '$STAGE2_RL_EPISODES'."
fi

if [ "$SKIP_STAGE1_RL" = "true" ] && [ "$STAGE1_RL_EPISODES_SPECIFIED" = "true" ]; then
    error_exit "--stage1-search-episodes / --stage1-rl-episodes cannot be used together with --skip-stage1-search / --skip-stage1-rl."
fi

if [ "$SKIP_NOISE_RL" = "true" ] && [ "$STAGE2_RL_EPISODES_SPECIFIED" = "true" ]; then
    error_exit "--stage2-search-episodes / --stage2-rl-episodes cannot be used together with --skip-noise-search / --skip-noise-rl."
fi

if [ "$SEARCH_ALGORITHM" = "rl" ] && [ "$SKIP_STAGE1_RL" = "false" ] && [ "$STAGE1_RL_EPISODES" -lt "$MIN_RL_EPISODES" ]; then
    error_exit "--stage1-search-episodes / --stage1-rl-episodes must be >= $MIN_RL_EPISODES in RL mode so Stage-1 PPO can update at least once."
fi

if [ "$SEARCH_ALGORITHM" = "rl" ] && [ "$SKIP_NOISE_RL" = "false" ] && [ "$STAGE2_RL_EPISODES" -lt "$MIN_RL_EPISODES" ]; then
    error_exit "--stage2-search-episodes / --stage2-rl-episodes must be >= $MIN_RL_EPISODES in RL mode so Stage-2 PPO can update at least once."
fi

if [ -n "$RESUME_RUN_DIR" ] && [ ! -d "$RESUME_RUN_DIR" ]; then
    error_exit "--resume-from 指定的目录不存在: $RESUME_RUN_DIR"
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

if [ "$FINAL_EVAL_SOURCE" = "json" ]; then
    FINAL_CONFIG_FAMILY="$(infer_config_algorithm_family "$FINAL_EVAL_CONFIG_PATH")"
    if [ "$SEARCH_ALGORITHM" = "ga" ] && [ "$FINAL_CONFIG_FAMILY" = "rl" ]; then
        error_exit "已选择 --search-algorithm=ga，但第一阶段 JSON 配置仍指向 RL/PPO 家族文件：$FINAL_EVAL_CONFIG_PATH。请改用 genetic/ga 家族配置，例如 glue_configs_best_genetic.json。"
    fi
    if [ "$SEARCH_ALGORITHM" = "rl" ] && [ "$FINAL_CONFIG_FAMILY" = "ga" ]; then
        error_exit "已选择 --search-algorithm=rl，但第一阶段 JSON 配置仍指向遗传算法家族文件：$FINAL_EVAL_CONFIG_PATH。请改用 RL/PPO 家族配置，例如 glue_configs_best_ppo.json。"
    fi
fi

if [ "$SKIP_STAGE1_RL" = "true" ] && [ "$FINAL_EVAL_SOURCE" = "search" ]; then
    error_exit "--skip-stage1-search / --skip-stage1-rl 与 --final-eval-source search 不能同时使用：跳过第一阶段搜索后没有搜索结果可供评估。请改用 json/manual，或去掉跳过选项。"
fi

if [ "$SKIP_STAGE1_RL" = "false" ] && [ "$FINAL_EVAL_SOURCE" != "search" ]; then
    error_exit "检测到第一阶段搜索算法将执行（--search-algorithm=$SEARCH_ALGORITHM），但 --final-eval-source=$FINAL_EVAL_SOURCE。为避免流程混用，执行第一阶段搜索时只能使用 search。若要使用 json/manual，请添加 --skip-stage1-search。"
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

if [ "$SKIP_NOISE_FINAL_EVAL" = "false" ] && [ "$NOISE_EVAL_SOURCE" = "json" ]; then
    NOISE_CONFIG_FAMILY="$(infer_config_algorithm_family "$NOISE_EVAL_CONFIG_PATH")"
    if [ "$SEARCH_ALGORITHM" = "ga" ] && [ "$NOISE_CONFIG_FAMILY" = "rl" ]; then
        error_exit "已选择 --search-algorithm=ga，但第二阶段噪声 JSON 配置仍指向 RL/PPO 家族文件：$NOISE_EVAL_CONFIG_PATH。请改用 genetic/ga 家族配置，例如 glue_noise_configs_best_genetic.json。"
    fi
    if [ "$SEARCH_ALGORITHM" = "rl" ] && [ "$NOISE_CONFIG_FAMILY" = "ga" ]; then
        error_exit "已选择 --search-algorithm=rl，但第二阶段噪声 JSON 配置仍指向遗传算法家族文件：$NOISE_EVAL_CONFIG_PATH。请改用 RL/PPO 家族配置，例如 glue_noise_configs_best_ppo.json。"
    fi
fi

if [ "$SKIP_NOISE_FINAL_EVAL" = "false" ] && [ "$SKIP_NOISE_RL" = "true" ] && [ "$NOISE_EVAL_SOURCE" = "search" ]; then
    error_exit "--skip-noise-search / --skip-noise-rl 与 --noise-eval-source search 不能同时用于噪声最终评估：跳过第二阶段搜索后没有搜索结果可供评估。请改用 json/manual，或去掉跳过选项。"
fi

if [ "$SKIP_NOISE_FINAL_EVAL" = "false" ] && [ "$SKIP_NOISE_RL" = "false" ] && [ "$NOISE_EVAL_SOURCE" != "search" ]; then
    error_exit "检测到第二阶段搜索算法将执行（--search-algorithm=$SEARCH_ALGORITHM），且噪声最终评估未跳过，但 --noise-eval-source=$NOISE_EVAL_SOURCE。为避免流程混用，此时只能使用 search。若要使用 json/manual，请添加 --skip-noise-search。"
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

MODEL_TYPE="$(printf '%s' "$MODEL_TYPE" | tr '[:upper:]' '[:lower:]')"
case "$MODEL_TYPE" in
    bert-base|bert_base|bertbase)   MODEL_TYPE="bert-base"  ;;
    bert-large|bert_large|bertlarge) MODEL_TYPE="bert-large" ;;
    gpt-2|gpt2|gpt_2)                MODEL_TYPE="gpt-2"      ;;
    *) error_exit "不支持的 --model-type: $MODEL_TYPE。支持的选项: bert-base, bert-large, gpt-2" ;;
esac

# 数据集 → DATA_PATH
case "$DATASET" in
    wnli|rte|cola|qnli|mrpc|sst2|stsb) DATA_PATH="$DATASET" ;;
    *) error_exit "不支持的数据集: $DATASET。支持的选项: wnli, rte, cola, qnli, mrpc, sst2, stsb" ;;
esac

# (model_type, dataset) → 预训练 checkpoint
# bert-base 使用 textattack 系列，与历史实验保持一致；
# bert-large 使用 yoshitomo-matsubara/bert-large-uncased-* 系列。
# 若该 checkpoint 在 HuggingFace 上不存在，需要在下表中显式标记不支持，
# 并在运行时给出明确的“当前不支持”提示，避免下载阶段才报 404。
if [ "$MODEL_TYPE" = "bert-base" ]; then
    case "$DATASET" in
        wnli)  BASE_MODEL="textattack/bert-base-uncased-WNLI"  ;;
        rte)   BASE_MODEL="textattack/bert-base-uncased-RTE"   ;;
        cola)  BASE_MODEL="textattack/bert-base-uncased-CoLA"  ;;
        qnli)  BASE_MODEL="textattack/bert-base-uncased-QNLI"  ;;
        mrpc)  BASE_MODEL="textattack/bert-base-uncased-MRPC"  ;;
        sst2)  BASE_MODEL="textattack/bert-base-uncased-SST-2" ;;
        stsb)  BASE_MODEL="textattack/bert-base-uncased-STS-B" ;;
    esac
elif [ "$MODEL_TYPE" = "bert-large" ]; then
    # bert-large：yoshitomo-matsubara 系列覆盖大部分 GLUE 任务，但并非全部。
    # 暂未确认存在 fine-tuned bert-large checkpoint 的任务在此显式拒绝，
    # 提示用户先暂时跳过这些任务。
    case "$DATASET" in
        mrpc)  BASE_MODEL="yoshitomo-matsubara/bert-large-uncased-mrpc" ;;
        cola)  BASE_MODEL="yoshitomo-matsubara/bert-large-uncased-cola" ;;
        stsb)  BASE_MODEL="yoshitomo-matsubara/bert-large-uncased-stsb" ;;
        rte)   BASE_MODEL="yoshitomo-matsubara/bert-large-uncased-rte"  ;;
        sst2)  BASE_MODEL="yoshitomo-matsubara/bert-large-uncased-sst2" ;;
        qnli)  BASE_MODEL="yoshitomo-matsubara/bert-large-uncased-qnli" ;;
        wnli)
            error_exit "bert-large 当前不支持数据集: $DATASET。已支持的 bert-large 任务: mrpc, cola, stsb, rte, sst2, qnli。请暂时跳过该任务，或改用 --model-type bert-base。"
            ;;
    esac
else
    # gpt-2：使用 PavanNeerudu/gpt2-finetuned-<task> 系列 (每个 GLUE 任务都有
    # 已经在 GLUE 训练集上微调好的 GPT2ForSequenceClassification 权重, 分类
    # head 和 backbone 均已收敛). RL 训练过程中这些权重保持固定 (参见
    # rl_tune.py 里对 model.parameters() 的 requires_grad_(False) 冻结),
    # 只有 PPO policy/value 网络在更新.
    case "$DATASET" in
        cola)  BASE_MODEL="PavanNeerudu/gpt2-finetuned-cola" ;;
        sst2)  BASE_MODEL="PavanNeerudu/gpt2-finetuned-sst2" ;;
        mrpc)  BASE_MODEL="PavanNeerudu/gpt2-finetuned-mrpc" ;;
        stsb)  BASE_MODEL="PavanNeerudu/gpt2-finetuned-stsb" ;;
        qnli)  BASE_MODEL="PavanNeerudu/gpt2-finetuned-qnli" ;;
        rte)   BASE_MODEL="PavanNeerudu/gpt2-finetuned-rte"  ;;
        wnli)  BASE_MODEL="PavanNeerudu/gpt2-finetuned-wnli" ;;
    esac
fi

export NCCL_DEBUG=INFO

LOGFILE_BASENAME="$(basename "$RAW_LOGFILE_PATH")"
if [ -z "$LOGFILE_BASENAME" ] || [ "$LOGFILE_BASENAME" = "." ] || [ "$LOGFILE_BASENAME" = "/" ] || [ "$LOGFILE_BASENAME" = "\\" ]; then
    LOGFILE_BASENAME="output.log"
fi
RUN_TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
RUN_ID="${RUN_TIMESTAMP}_pid$$"
RUN_ROOT="rl_results/layer_importance_runs/${DATASET}/${RUN_ID}"
LOGFILE_PATH="${RUN_ROOT}/logs/${LOGFILE_BASENAME}"
mkdir -p "${RUN_ROOT}/logs"

if [ "$SEARCH_ALGORITHM" = "ga" ]; then
    SEARCH_ENTRYPOINT="rl_tune_genetic.py"
else
    SEARCH_ENTRYPOINT="rl_tune.py"
fi

CMD=(
    python "$SEARCH_ENTRYPOINT"
    --base_model "$BASE_MODEL"
    --data_path "$DATA_PATH"
    --output_dir "$RUN_ROOT"
    --batch_size "$BATCH_SIZE"
    --micro_batch_size "$BATCH_SIZE"
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
    --stage1_rl_episodes "$STAGE1_RL_EPISODES"
    --stage2_rl_episodes "$STAGE2_RL_EPISODES"
    --stage1_rl_episodes_specified "$STAGE1_RL_EPISODES_SPECIFIED"
    --stage2_rl_episodes_specified "$STAGE2_RL_EPISODES_SPECIFIED"
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
    --resume_run_dir "$RESUME_RUN_DIR"
)

echo "Search algorithm: $SEARCH_ALGORITHM (entrypoint=$SEARCH_ENTRYPOINT)"
echo "Model type: $MODEL_TYPE"
echo "Dataset: $DATASET (base_model=$BASE_MODEL, data_path=$DATA_PATH)"
echo "Launching search job with final evaluation source: $FINAL_EVAL_SOURCE"
if [ "$SKIP_STAGE1_RL" = "true" ]; then
    echo "Stage-1 search: SKIPPED"
else
    echo "Stage-1 search: will run with algorithm=$SEARCH_ALGORITHM"
fi
if [ "$SKIP_STAGE1_FINAL_EVAL" = "true" ]; then
    echo "Stage-1 final evaluation: SKIPPED (--skip-stage1-final-eval)"
fi
if [ "$SKIP_NOISE_RL" = "true" ]; then
    echo "Stage-2 noise search: SKIPPED"
else
    echo "Stage-2 noise search will run after the fixed GELU/Softmax config is selected."
fi
if [ "$SKIP_NOISE_FINAL_EVAL" = "true" ]; then
    echo "Stage-2 noise final evaluation: SKIPPED (--skip-noise-final-eval)"
else
    echo "Stage-2 noise final evaluation: noise_eval_source=$NOISE_EVAL_SOURCE, repeat=$NOISE_EVAL_REPEAT_N"
fi
echo "Requested log filename: $RAW_LOGFILE_PATH"
echo "Resolved run root: $RUN_ROOT"
echo "Resolved nohup log: $LOGFILE_PATH"
echo "Batch size: $BATCH_SIZE (syncs --batch_size and --micro_batch_size)"
echo "Stage-1 search budget: $STAGE1_RL_EPISODES"
echo "Stage-2 search budget: $STAGE2_RL_EPISODES"
if [ -n "$RESUME_RUN_DIR" ]; then
    echo "Resume from: $RESUME_RUN_DIR"
fi
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
else
    echo "CUDA_VISIBLE_DEVICES: <unset>"
fi

nohup "${CMD[@]}" > "$LOGFILE_PATH" 2>&1 &
JOB_PID=$!
echo "Background PID: $JOB_PID"

# ---------------------------------------------------------------
# 记录 PID / run 目录，便于后续优雅停止（Graceful Stop）
# ---------------------------------------------------------------
RL_PID_FILE="${RUN_ROOT}/rl.pid"
echo "$JOB_PID" > "$RL_PID_FILE"
# 同时维护一个数据集级 LATEST 指针，方便“停上一个运行”场景
LATEST_POINTER_DIR="rl_results/layer_importance_runs/${DATASET}"
echo "$RUN_ROOT" > "${LATEST_POINTER_DIR}/LATEST_RUN_DIR"
echo "$JOB_PID" > "${LATEST_POINTER_DIR}/LATEST_PID"

echo ""
echo "========================================================================"
echo "  优雅停止 (Graceful Stop) — 任选其一即可安全中断训练并保存 checkpoint"
echo "========================================================================"
echo "  方式 A (推荐，最简单)：直接发送 SIGINT 到当前进程"
echo "      kill -INT $JOB_PID"
echo ""
echo "  方式 B：通过数据集级 LATEST 指针（无需记住 PID / run 目录）"
echo "      kill -INT \$(cat ${LATEST_POINTER_DIR}/LATEST_PID)"
echo ""
echo "  方式 C：创建停止标志文件（不依赖信号，适合脚本化批量停止）"
echo "      touch ${RUN_ROOT}/stage1/STOP_RL              # 停 Stage-1 RL"
echo "      touch ${RUN_ROOT}/stage2_noise/progress/STOP_RL  # 停 Stage-2 噪声 RL"
echo ""
echo "  停止后续训：下次启动时加上 --resume-from ${RUN_ROOT}"
echo "  ⚠ 切勿使用 kill -9（SIGKILL），它会绕过 checkpoint 保存导致续训断层！"
echo "========================================================================"

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
# CUDA selection is now caller-controlled via the environment.
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
