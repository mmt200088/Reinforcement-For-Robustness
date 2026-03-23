This is a Repository for Transformer robustness evaluation using Reinforcement Learning.

Please Ignore the LLM-Adapters, EzPC, and importance-aware-sparse-tuning-IST-paper in root directory. Sorry, but the code is DIRTY now!

## How to Start

### Allocate enough memory for Docker container
    mount -o remount,size=64G /dev/shm
### Activate python enviroment first
    conda activate llm_ist
### Go into the .sh located directory (/root/ppml/MoE-Privacy)
    cd /root/ppml/MoE-Privacy 
    cd /var/tmp/root-home/Reinforcement-For-Robustness
### Execute the running scripts 
    bash llama_7B_LayerImportance.sh [lora_r] [lora_alpha] [logfile_path] [rl_lr] [degree]

lora_r: parameter for lora, ignore (we just use the Lora Framework to inference...), just set is to 32.  
lora_alpha: parameter for lora, ignore, just set is to 64.  
logfile_path: the log file output path, you can change it when the learning rate varies.  
rl_lr: PPO learning-rate control. If `rl_lr < 1`, it is used directly as the PPO optimizer LR. For backward compatibility, legacy values like `20` / `40` are interpreted as `20e-6` / `40e-6`.  
degree: parameter for early debug, now deprecated. Just set it to 2.  

example: `bash llama_7B_LayerImportance.sh 32 64 output.log 20 2`

#### 全部可选命名参数

旧版 5 参数命令完全兼容。可在 5 个位置参数之后追加以下可选命名参数：

**第一阶段：最终评估配置来源**

| 参数 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `--final-eval-source search\|json\|manual` | 最终评估使用的 GELU/Softmax 配置来源：`search` 使用 RL 搜索结果；`json` 从 JSON 文件读取；`manual` 手动指定 | `search` |
| `--final-eval-config PATH` | `json` 模式下指定的 JSON 配置文件路径。程序根据当前数据集名（如 `mrpc`）自动读取对应条目 | `glue_configs_best_ppo.json` |
| `--manual-gelu "[1,1,1,4,...]"` | `manual` 模式下手动指定每层 GELU degree（JSON 数组），必须与 `--manual-softmax` 一起使用 | — |
| `--manual-softmax "[2,3,4,6,...]"` | `manual` 模式下手动指定每层 Softmax degree（JSON 数组），必须与 `--manual-gelu` 一起使用 | — |

**随机对照实验**

| 参数 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `--random-seed N` | 随机实验种子 | `42` |
| `--perm-trials N` | Permutation 随机对照实验次数 | `10` |
| `--cost-trials N` | 精确 cost-matched 随机对照实验次数 | `10` |
| `--budget-trials N` | 同总预算随机对照实验次数 | `10` |

**第二阶段：噪声 RL**

| 参数 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `--skip-noise-rl` | 跳过第二阶段噪声 RL，只运行第一阶段。默认自动运行第二阶段 | 不跳过 |

第二阶段保持第一阶段选定的 GELU/Softmax 不变，用 PPO 学习每层 7 个噪声 scaling factor：

| 噪声对象 | 模型路径 | 动作空间 |
| :--- | :--- | :--- |
| `x`（输入噪声） | 层输入 hidden_states | `{20, 22, 24, 26, 28, 30}` |
| `wq`（Query 权重噪声） | attention.self.query | `{10, 12, 14, 16, 18, 20, 22}` |
| `wk`（Key 权重噪声） | attention.self.key | `{10, 12, 14, 16, 18, 20, 22}` |
| `wv`（Value 权重噪声） | attention.self.value | `{10, 12, 14, 16, 18, 20, 22}` |
| `wo`（Attn 输出权重噪声） | attention.output.dense | `{10, 12, 14, 16, 18, 20, 22}` |
| `wffn1`（FFN 第一层权重噪声） | intermediate.dense | `{10, 12, 14, 16, 18, 20, 22}` |
| `wffn2`（FFN 第二层权重噪声） | output.dense | `{10, 12, 14, 16, 18, 20, 22}` |

第二阶段的逻辑位于独立模块 `noise_rl_module.py` 中（与 `final_evaluation_module.py` 架构一致）。

第二阶段产出文件：
- `noise_ppo_step_info.txt` — 每步动作/概率日志
- `noise_ppo_training_curve.png` — 训练曲线图
- `noise_ppo_entropy_curve.png` — 策略熵曲线图
- 主日志中搜索 `PHASE 5: SECOND-STAGE NOISE RL` 和 `Best Noise Configuration Found`

**第二阶段：噪声最终评估配置来源**

| 参数 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `--noise-eval-source search\|json\|manual` | 噪声最终评估使用的配置来源：`search` 使用噪声 RL 搜索结果；`json` 从 JSON 文件读取；`manual` 手动指定 | `search` |
| `--noise-eval-config PATH` | `json` 模式下指定的噪声配置 JSON 文件路径。程序根据当前数据集名自动读取对应条目 | `glue_noise_configs_best_ppo.json` |
| `--manual-noise-config '{"x":[...],...}'` | `manual` 模式下手动指定 7 种噪声 scaling factor 数组（JSON 对象格式），支持短名称 `x, wq, wk, wv, wo, wffn1, wffn2` | — |
| `--noise-eval-repeat N` | 对选定配置执行 N 次重复评估，输出 N 次结果及均值/标准差统计 | `1` |

噪声配置 JSON 文件格式：
```json
{
  "mrpc": {
    "x": [20, 22, 24, 26, 28, 30, 20, 22, 24, 26, 28, 30],
    "wq": [10, 12, 14, 16, 18, 20, 22, 10, 12, 14, 16, 18],
    "wk": [10, 12, 14, ...],
    "wv": [10, 12, 14, ...],
    "wo": [10, 12, 14, ...],
    "wffn1": [10, 12, 14, ...],
    "wffn2": [10, 12, 14, ...]
  }
}
```

噪声最终评估的逻辑位于独立模块 `noise_final_evaluation_module.py` 中，功能与第一阶段 `final_evaluation_module.py` 一致，并新增 N 次重复评估。

噪声最终评估产出文件（位于 `experiment_results/noise_final_evaluation/` 目录）：
- `noise_final_eval_results_<dataset>.json` — 结果 JSON
- `noise_final_eval_comparison_<dataset>.png` — 对比图

**跳过第一阶段最终评估**

| 参数 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `--skip-stage1-final-eval` | 跳过第一阶段的最终评估（Phase 3 + Phase 4），直接使用 `--final-eval-source` 指定来源的 GELU/Softmax 配置进入第二阶段噪声 RL | 不跳过 |

该选项适用于只关注第二阶段噪声 RL、不需要重复运行第一阶段评估的场景。配合 `--final-eval-source` 可指定 GELU/Softmax 配置来源（search / json / manual）。

**帮助**

| 参数 | 说明 |
| :--- | :--- |
| `-h`, `--help` | 显示用法帮助信息并退出 |

#### 使用示例

默认完整流程（第一阶段 + 第二阶段噪声 RL）：
`bash llama_7B_LayerImportance.sh 32 64 output.log 20 2`

只跑第一阶段，跳过噪声 RL：
`bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 --skip-noise-rl`

从 JSON 加载配置（仍运行第二阶段）：
`bash llama_7B_LayerImportance.sh 32 64 output_json.log 20 2 --final-eval-source json --final-eval-config glue_configs_best_ppo.json`

手动指定每层配置：
`bash llama_7B_LayerImportance.sh 32 64 output_manual.log 20 2 --final-eval-source manual --manual-gelu "[1,1,1,4,1,1,1,1,1,1,1,1]" --manual-softmax "[2,3,4,6,4,4,5,4,4,5,5,2]"`

提高随机对照次数：
`bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 --perm-trials 30 --cost-trials 30 --budget-trials 30`

从 JSON 加载 + 跳过噪声 RL：
`bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 --final-eval-source json --final-eval-config glue_configs_best_ppo.json --skip-noise-rl`

跳过第一阶段最终评估，用 JSON 配置直接进入第二阶段噪声 RL：
`bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 --final-eval-source json --final-eval-config glue_configs_best_ppo.json --skip-stage1-final-eval`

手动指定噪声配置做第二阶段最终评估：
`bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 --noise-eval-source manual --manual-noise-config '{"x":[20,22,24,26,28,30,20,22,24,26,28,30],"wq":[10,12,14,16,18,20,22,10,12,14,16,18],"wk":[10,12,14,16,18,20,22,10,12,14,16,18],"wv":[10,12,14,16,18,20,22,10,12,14,16,18],"wo":[10,12,14,16,18,20,22,10,12,14,16,18],"wffn1":[10,12,14,16,18,20,22,10,12,14,16,18],"wffn2":[10,12,14,16,18,20,22,10,12,14,16,18]}' --noise-eval-repeat 100`

第二阶段噪声评估重复 5 次：
`bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 --noise-eval-repeat 5`

手动指定第一、第二阶段的配置进行第二阶段的最终评估
`bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
  --final-eval-source manual \
  --manual-gelu "[1,1,1,1,1,4,1,1,1,1,1,1]" \
  --manual-softmax "[2,2,5,5,5,2,5,2,5,5,6,2]" \
  --skip-stage1-final-eval \
  --skip-noise-rl \
  --noise-eval-source manual \
  --manual-noise-config '{"x":[20,22,24,26,28,30,20,22,24,26,28,30],"wq":[10,12,14,16,18,20,22,10,12,14,16,18],"wk":[10,12,14,16,18,20,22,10,12,14,16,18],"wv":[10,12,14,16,18,20,22,10,12,14,16,18],"wo":[10,12,14,16,18,20,22,10,12,14,16,18],"wffn1":[10,12,14,16,18,20,22,10,12,14,16,18],"wffn2":[10,12,14,16,18,20,22,10,12,14,16,18]}'`

从 JSON 加载噪声配置：
`bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 --noise-eval-source json --noise-eval-config glue_noise_configs_best_ppo.json`

#### Note: Though we call the script "llama_7B_LayerImportance.sh", we just evaluate the Bert-base model for different tasks now, please check out the .sh for more detials!

### The Result file 
The result outputs to file importance_scores_.....txt in /root/ppml/MoE-Privacy. You can modified the name in variable self.log_path in layer_importance_evaluator.py

### Stop the process
Cause running the sh now is using nohup, so we run it in backend.  
When you want interrupt it, run
`ps aux | grep rl_tune.py`
to check the process (rl_tune.py is the starting point of our evaluate, because we use the LLM-Adapter framework).  
And then kill the first process:
`kill -9 [process_id_of_rl_tune.py]` 

### data support
"stsb", "mnli", "sst2", "cola", "qnli", "rte", "wnli", "mrpc"

### 各数据集描述
| 数据集 | 任务类型 | 训练集 (Train) | 验证集 (Dev) | 测试集 (Test) | 评价指标 (Metrics) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MNLI** | 自然语言推理 | 392,702 | 9,815/9,832 | 9,796/9,847 | Matched/Mismatched Acc. |
| **QQP** | 句子对等判定 | 363,846 | 40,430 | 390,965 | F1 / Accuracy |
| **QNLI** | 问答蕴含 | 104,743 | 5,463 | 5,463 | Accuracy |
| **SST-2** | 情感分析 | 67,349 | 872 | 1,821 | Accuracy |
| **CoLA** | 语法可接受性 | 8,551 | 1,043 | 1,063 | Matthews Corr. |
| **STS-B** | 语义相似度 | 5,749 | 1,500 | 1,379 | Pearson/Spearman Corr. |
| **MRPC** | 句子对等判定 | 3,668 | 408 | 1,725 | F1 / Accuracy |
| **RTE** | 文本蕴含 | 2,490 | 277 | 3,000 | Accuracy |
| **WNLI** | 指代消解蕴含 | 635 | 71 | 146 | Accuracy |

### 实验运行
完整实验（所有 8 个数据集）
conda activate llm_ist
bash run_all_experiments.sh

快速测试（仅 sst2, mrpc）
bash run_all_experiments.sh --quick

单独运行某个版块
python experiment_single_layer_degradation.py --tasks sst2 mrpc --device cuda
python experiment_block1_monotonicity.py --tasks sst2 --n_bootstrap 100 --device cuda
