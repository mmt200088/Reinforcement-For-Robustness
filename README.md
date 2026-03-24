This is a Repository for Transformer robustness evaluation using Reinforcement Learning.

Please Ignore the LLM-Adapters, EzPC, and importance-aware-sparse-tuning-IST-paper in root directory. Sorry, but the code is DIRTY now!

## 使用说明

### 运行前准备

```bash
mount -o remount,size=64G /dev/shm
conda activate llm_ist
cd /var/tmp/root-home/Reinforcement-For-Robustness
```

### 基础命令

```bash
bash llama_7B_LayerImportance.sh [lora_r] [lora_alpha] [logfile_path] [rl_lr] [degree]
```

位置参数说明：


| 参数             | 说明                                                                |
| -------------- | ----------------------------------------------------------------- |
| `lora_r`       | LoRA rank，当前固定传 `32`                                              |
| `lora_alpha`   | LoRA alpha，当前固定传 `64`                                             |
| `logfile_path` | nohup 输出日志路径                                                      |
| `rl_lr`        | PPO 学习率控制。若 `< 1` 则直接作为学习率；旧值如 `20` / `40` 会解释为 `20e-6` / `40e-6` |
| `degree`       | 历史调试参数，固定传 `2`                                                    |


基础示例：
`bash llama_7B_LayerImportance.sh 32 64 output.log 20 2`

### 命名参数与安全约束

旧版 5 参数命令依然兼容，但新增了更严格的流程校验，避免“前面跑 RL，后面却拿手动/JSON 配置做评估”的混用。

**第一阶段：GELU/Softmax RL 与最终评估**

第一阶段的“是否执行 RL”与“是否执行最终评估”仍可独立控制，但配置来源现在有安全约束：

- 若执行第一阶段 RL，则 `--final-eval-source` 只能为 `search`
- 若使用 `json` 或 `manual`，则必须显式添加 `--skip-stage1-rl`
- 若跳过第一阶段 RL，则不能再使用 `search`
- `--skip-stage1-rl` 会一并跳过 Phase 1 baseline 建立、Phase 1.5 GELU 输入分布分析、Phase 2 PPO 和 Phase 2.5 贪心搜索


| 参数                                       | 说明                                                                                  | 默认值                          |
| ---------------------------------------- | ----------------------------------------------------------------------------------- | ---------------------------- |
| `--skip-stage1-rl`                       | 跳过整个第一阶段搜索准备与搜索流程：Phase 1 baseline、Phase 1.5 GELU 输入分布分析、Phase 2 PPO、Phase 2.5 贪心搜索 | 不跳过                          |
| `--skip-stage1-final-eval`               | 跳过第一阶段最终评估（Phase 3 + Phase 4），但仍会先解析第一阶段配置，再进入第二阶段                                  | 不跳过                          |
| `--final-eval-source search|json|manual` | 第一阶段配置来源。执行第一阶段 RL 时只能为 `search`；使用 `json/manual` 时必须加 `--skip-stage1-rl`           | `search`                     |
| `--final-eval-config PATH`               | `json` 模式下的配置文件路径                                                                   | `glue_configs_best_ppo.json` |
| `--manual-gelu "[1,1,...]"`              | `manual` 模式下的每层 GELU degree，必须与 `--manual-softmax` 同时提供                             | —                            |
| `--manual-softmax "[2,2,...]"`           | `manual` 模式下的每层 Softmax degree，必须与 `--manual-gelu` 同时提供                             | —                            |
| `--random-seed N`                        | 随机实验种子                                                                              | `42`                         |
| `--perm-trials N`                        | Permutation 随机对照实验次数                                                                | `10`                         |
| `--cost-trials N`                        | 精确 cost-matched 随机对照实验次数                                                            | `10`                         |
| `--budget-trials N`                      | 同总预算随机对照实验次数                                                                        | `10`                         |


**第二阶段：噪声 RL 与噪声最终评估**

第二阶段同样增加了安全约束：

- 若执行噪声 RL，且没有跳过噪声最终评估，则 `--noise-eval-source` 只能为 `search`
- 若使用 `json` 或 `manual` 做噪声最终评估，则必须显式添加 `--skip-noise-rl`
- 若跳过噪声 RL，则不能在噪声最终评估中再使用 `search`


| 参数                                        | 说明                                                                              | 默认值                                |
| ----------------------------------------- | ------------------------------------------------------------------------------- | ---------------------------------- |
| `--skip-noise-rl`                         | 跳过第二阶段噪声 RL 训练                                                                  | 不跳过                                |
| `--skip-noise-final-eval`                 | 跳过第二阶段噪声最终评估                                                                    | 不跳过                                |
| `--noise-eval-source search|json|manual`  | 噪声最终评估配置来源。执行噪声 RL 且保留最终评估时只能为 `search`；使用 `json/manual` 时必须加 `--skip-noise-rl` | `search`                           |
| `--noise-eval-config PATH`                | `json` 模式下的噪声配置文件路径                                                             | `glue_noise_configs_best_ppo.json` |
| `--manual-noise-config '{"x":[...],...}'` | `manual` 模式下的噪声配置，需包含 7 类噪声数组                                                   | —                                  |
| `--noise-eval-repeat N`                   | 噪声最终评估重复次数，必须为正整数                                                               | `1`                                |


第二阶段 RL 训练保持第一阶段选定的 GELU/Softmax 不变，用 PPO 学习每层 7 个噪声 scaling factor：


| 噪声对象                 | 模型路径                   | 动作空间                           |
| -------------------- | ---------------------- | ------------------------------ |
| `x`（输入噪声）            | 层输入 hidden_states      | `{20, 22, 24, 26, 28, 30}`     |
| `wq`（Query 权重噪声）     | attention.self.query   | `{10, 12, 14, 16, 18, 20, 22}` |
| `wk`（Key 权重噪声）       | attention.self.key     | `{10, 12, 14, 16, 18, 20, 22}` |
| `wv`（Value 权重噪声）     | attention.self.value   | `{10, 12, 14, 16, 18, 20, 22}` |
| `wo`（Attn 输出权重噪声）    | attention.output.dense | `{10, 12, 14, 16, 18, 20, 22}` |
| `wffn1`（FFN 第一层权重噪声） | intermediate.dense     | `{10, 12, 14, 16, 18, 20, 22}` |
| `wffn2`（FFN 第二层权重噪声） | output.dense           | `{10, 12, 14, 16, 18, 20, 22}` |


第二阶段 RL 训练逻辑位于 `noise_rl_module.py`，噪声最终评估逻辑位于 `noise_final_evaluation_module.py`。

第二阶段产出文件：

- `noise_ppo_step_info.txt` — 每步动作/概率日志
- `noise_ppo_training_curve.png` — 训练曲线图
- `noise_ppo_entropy_curve.png` — 策略熵曲线图
- 主日志中搜索 `PHASE 5: SECOND-STAGE NOISE RL` 和 `Best Noise Configuration Found`

**噪声最终评估配置来源**（仅在未 `--skip-noise-final-eval` 时生效）


| 参数                                        | 说明                                                                                                    | 默认值                                |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------- | ---------------------------------- |
| `--noise-eval-source search/json/manual`  | 噪声最终评估使用的配置来源：`search` 使用本次噪声 RL 搜索结果；`json` 从 JSON 文件读取；`manual` 手动指定。若执行噪声 RL 且保留最终评估，则只能为 `search` | `search`                           |
| `--noise-eval-config PATH`                | `json` 模式下指定的噪声配置 JSON 文件路径。程序根据当前数据集名自动读取对应条目                                                        | `glue_noise_configs_best_ppo.json` |
| `--manual-noise-config '{"x":[...],...}'` | `manual` 模式下手动指定 7 种噪声 scaling factor 数组（JSON 对象格式），支持短名称 `x, wq, wk, wv, wo, wffn1, wffn2`           | —                                  |
| `--noise-eval-repeat N`                   | 对选定配置执行 N 次重复评估，输出 N 次结果及均值/标准差统计                                                                     | `1`                                |


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

### 使用示例

默认完整流程（第一阶段 RL + 最终评估 + 第二阶段噪声 RL + 噪声最终评估）：
`bash llama_7B_LayerImportance.sh 32 64 output.log 20 2`

只运行第一阶段，跳过第二阶段：
`bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 --skip-noise-rl --skip-noise-final-eval`

不跑第一阶段 RL，直接从 JSON 读取第一阶段配置：
`bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 --skip-stage1-rl --final-eval-source json --final-eval-config glue_configs_best_ppo.json`

不跑第一阶段 RL，手动指定每层 GELU/Softmax：
`bash llama_7B_LayerImportance.sh 32 64 output_manual.log 20 2 --skip-stage1-rl --final-eval-source manual --manual-gelu "[1,1,1,4,1,1,1,1,1,1,1,1]" --manual-softmax "[2,3,4,6,4,4,5,4,4,5,5,2]"`

跳过噪声 RL，直接从 JSON 做噪声最终评估：
`bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 --skip-noise-rl --noise-eval-source json --noise-eval-config glue_noise_configs_best_ppo.json`

跳过噪声 RL，手动指定噪声配置并重复评估 100 次：
`bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 --skip-noise-rl --noise-eval-source manual --manual-noise-config '{"x":[20,22,24,26,28,30,20,22,24,26,28,30],"wq":[10,12,14,16,18,20,22,10,12,14,16,18],"wk":[10,12,14,16,18,20,22,10,12,14,16,18],"wv":[10,12,14,16,18,20,22,10,12,14,16,18],"wo":[10,12,14,16,18,20,22,10,12,14,16,18],"wffn1":[10,12,14,16,18,20,22,10,12,14,16,18],"wffn2":[10,12,14,16,18,20,22,10,12,14,16,18]}' --noise-eval-repeat 100`

只进行第二阶段rl
`bash llama_7B_LayerImportance.sh 32 64 output.log 50 2 --skip-stage1-rl --final-eval-source json --final-eval-config glue_configs_best_ppo.json --skip-stage1-final-eval --noise-eval-repeat 200`

完全跳过两个阶段的搜索/训练，手动指定所有配置只做后续评估：

```bash
bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
  --skip-stage1-rl \
  --final-eval-source manual \
  --manual-gelu "[1,1,1,1,1,4,1,1,1,1,1,1]" \
  --manual-softmax "[2,2,5,5,5,2,5,2,5,5,6,2]" \
  --skip-stage1-final-eval \
  --skip-noise-rl \
  --noise-eval-source manual \
  --manual-noise-config '{"x":[20,22,24,26,28,30,20,22,24,26,28,30],"wq":[10,12,14,16,18,20,22,10,12,14,16,18],"wk":[10,12,14,16,18,20,22,10,12,14,16,18],"wv":[10,12,14,16,18,20,22,10,12,14,16,18],"wo":[10,12,14,16,18,20,22,10,12,14,16,18],"wffn1":[10,12,14,16,18,20,22,10,12,14,16,18],"wffn2":[10,12,14,16,18,20,22,10,12,14,16,18]}'
```

帮助：
`bash llama_7B_LayerImportance.sh --help`

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


| 数据集       | 任务类型   | 训练集 (Train) | 验证集 (Dev)   | 测试集 (Test)  | 评价指标 (Metrics)          |
| --------- | ------ | ----------- | ----------- | ----------- | ----------------------- |
| **MNLI**  | 自然语言推理 | 392,702     | 9,815/9,832 | 9,796/9,847 | Matched/Mismatched Acc. |
| **QQP**   | 句子对等判定 | 363,846     | 40,430      | 390,965     | F1 / Accuracy           |
| **QNLI**  | 问答蕴含   | 104,743     | 5,463       | 5,463       | Accuracy                |
| **SST-2** | 情感分析   | 67,349      | 872         | 1,821       | Accuracy                |
| **CoLA**  | 语法可接受性 | 8,551       | 1,043       | 1,063       | Matthews Corr.          |
| **STS-B** | 语义相似度  | 5,749       | 1,500       | 1,379       | Pearson/Spearman Corr.  |
| **MRPC**  | 句子对等判定 | 3,668       | 408         | 1,725       | F1 / Accuracy           |
| **RTE**   | 文本蕴含   | 2,490       | 277         | 3,000       | Accuracy                |
| **WNLI**  | 指代消解蕴含 | 635         | 71          | 146         | Accuracy                |


### 实验运行

完整实验（所有 8 个数据集）
conda activate llm_ist
bash run_all_experiments.sh

快速测试（仅 sst2, mrpc）
bash run_all_experiments.sh --quick

单独运行某个版块
python experiment_single_layer_degradation.py --tasks sst2 mrpc --device cuda
python experiment_block1_monotonicity.py --tasks sst2 --n_bootstrap 100 --device cuda