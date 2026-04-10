## 命令行参数总表

```bash
bash llama_7B_LayerImportance.sh [lora_r] [lora_alpha] [logfile_path] [rl_lr] [degree] [options]
```

### 位置参数

| 参数 | 说明 |
| --- | --- |
| `lora_r` | LoRA rank，当前固定传 `32` |
| `lora_alpha` | LoRA alpha，当前固定传 `64` |
| `logfile_path` | nohup 日志文件名提示；真实日志自动写入 run 目录下 `logs/` |
| `rl_lr` | PPO 学习率控制。若 `< 1` 则直接作为学习率；旧值 `20`/`40` 解释为 `20e-6`/`40e-6` |
| `degree` | 历史调试参数，固定传 `2` |

### 可选参数一览

| 参数 | 适用模式 | 默认值 | 说明 |
| --- | --- | --- | --- |
| **算法选择** | | | |
| `--search-algorithm` | 全局 | `rl` | 搜索算法：`rl`（PPO）/ `ga`（遗传）/ `general-rl`（通用策略） |
| **模型与数据集** | | | |
| `--model DATASET` | 全局 | `mrpc` | 数据集：mrpc, sst2, stsb, cola, qnli, rte, wnli |
| `--model-type TYPE` | 全局 | `bert-base` | 骨干模型：bert-base, bert-large, gpt-2 |
| `--batch_size N` | 全局 | `16` | 推理批次大小（同步设 batch_size 和 micro_batch_size） |
| **rl / ga 共享参数** | | | |
| `--stage1-search-episodes N` | rl, ga | `51000` | Stage-1 搜索回合数 |
| `--stage2-search-episodes N` | rl, ga | `40000` | Stage-2 噪声搜索回合数 |
| `--skip-stage1-search` | rl, ga | — | 跳过 Stage-1 搜索（GA 推荐写法） |
| `--skip-noise-search` | rl, ga | — | 跳过 Stage-2 噪声搜索（GA 推荐写法） |
| `--skip-stage1-final-eval` | rl, ga | — | 跳过 Stage-1 最终评估 |
| `--skip-noise-final-eval` | rl, ga | — | 跳过 Stage-2 噪声最终评估 |
| `--final-eval-source` | rl, ga | `search` | Stage-1 评估配置来源：search / json / manual |
| `--final-eval-config PATH` | rl, ga | 自动 | Stage-1 JSON 配置文件路径 |
| `--noise-eval-source` | rl, ga | `search` | Stage-2 评估配置来源：search / json / manual |
| `--noise-eval-config PATH` | rl, ga | 自动 | Stage-2 噪声 JSON 配置文件路径 |
| `--noise-eval-repeat N` | rl, ga | `1` | 噪声最终评估重复次数 |
| `--manual-gelu` | rl, ga | — | Stage-1 手动 GELU 配置 |
| `--manual-softmax` | rl, ga | — | Stage-1 手动 Softmax 配置 |
| `--manual-noise-config` | rl, ga | — | Stage-2 手动噪声配置（JSON 字符串） |
| `--random-seed` | rl, ga | `42` | 最终评估随机种子 |
| `--perm-trials N` | rl, ga | `10` | 置换试验次数 |
| `--cost-trials N` | rl, ga | `10` | 等价成本试验次数 |
| `--budget-trials N` | rl, ga | `10` | 等价预算试验次数 |
| `--resume-from PATH` | rl, ga, general-rl | — | 从之前的 run 目录恢复训练（rl/ga/general-rl 均支持） |
| **rl 兼容别名**（GA 模式下禁用） | | | |
| `--stage1-rl-episodes N` | rl | `51000` | 等价于 --stage1-search-episodes |
| `--stage2-rl-episodes N` | rl | `40000` | 等价于 --stage2-search-episodes |
| `--skip-stage1-rl` | rl | — | 等价于 --skip-stage1-search |
| `--skip-noise-rl` | rl | — | 等价于 --skip-noise-search |
| **general-rl 专用参数**（仅 general-rl 模式可用） | | | |
| `--general-rl-mode` | general-rl | `infer` | 运行模式：`train`（多任务训练）/ `infer`（离线推断） |
| `--general-rl-tasks T1,T2,...` | general-rl (train) | 同 --model | 逗号分隔的训练任务列表 |
| `--general-rl-rounds N` | general-rl (train) | `50` | Round-robin 训练轮数 |
| `--general-rl-episodes-per-round N` | general-rl (train) | `170` | 每轮每任务的回合数 |
| `--general-rl-lr FLOAT` | general-rl (train) | `3e-5` | 通用策略训练学习率 |
| `--general-rl-num-rollouts N` | general-rl (infer) | `500` | 离线 rollout 次数 |
| `--general-rl-greedy` | general-rl (infer) | — | 使用贪心（argmax）rollout |
| `--general-stage1-policy PATH` | general-rl (infer) | — | 已训练的通用 Stage-1 策略文件路径（**必需**） |
| `--general-stage2-policy PATH` | general-rl (infer) | — | 已训练的通用 Stage-2 噪声策略文件路径（可选） |
| `--general-rl-skip-stage2` | general-rl | — | 跳过 Stage-2 训练/推断 |
| `--general-rl-stage1-config-json PATH` | general-rl (train) | — | Stage-2 训练时各任务的 Stage-1 配置 JSON |

### 模式互斥规则

- 选择 `--search-algorithm=general-rl` 后，**不能**使用 rl/ga 专用参数（如 `--stage1-rl-episodes`、`--skip-stage1-rl`、`--final-eval-config`、`--manual-gelu` 等），否则脚本报错。
- 选择 `--search-algorithm=rl` 或 `ga` 后，**不能**使用 general-rl 专用参数（如 `--general-stage1-policy`、`--general-rl-tasks` 等），否则脚本报错。
- 选择 `--search-algorithm=ga` 后，**不能**使用 RL 兼容别名（如 `--skip-stage1-rl`、`--stage1-rl-episodes`），必须使用算法无关写法。

---

## 搜索算法可选项说明（强化学习 / 遗传算法 / 通用RL）

统一通过主脚本选择搜索算法：

```bash
bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 --search-algorithm rl
bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 --search-algorithm ga
bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 --search-algorithm general-rl --general-rl-mode train --general-rl-tasks mrpc,cola,rte,stsb
bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 --search-algorithm general-rl --general-rl-mode infer --general-stage1-policy general_stage1_policy.pt --model mrpc
```

### 1. 选项定义

| 选项 | 可选值 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--search-algorithm` | `rl` / `ga` / `general-rl` | `rl` | 统一控制搜索使用哪种算法 |

对应关系如下：

| 选项值 | 含义 | 实际入口 |
| --- | --- | --- |
| `rl` | Per-task 两阶段 PPO 搜索 | `rl_tune.py` |
| `ga` | COINN 风格两阶段遗传算法搜索 | `rl_tune_genetic.py` |
| `general-rl` | 多任务通用策略训练 / 离线推断 | `rl_tune_general.py` |

不写 `--search-algorithm` 时，默认走 RL。

### 2. 与现有流程的关系

- `rl` 模式下，保持原有项目逻辑：Stage-1 搜索 `GELU/Softmax`，Stage-2 搜索 7 类噪声 scaling factor。
- `ga` 模式下，模型、数据集、评估约束、最终评估模块、结果目录结构都与原流程保持一致，只把”搜索算法本体”替换为遗传算法。
- `general-rl` 模式下，使用多任务 round-robin 训练通用策略（含 Critic），部署时只需冻结策略做离线 rollout，无需为每个新任务重新训练数万回合。
- `rl` 和 `ga` 共用相同的阶段跳过逻辑、最终评估逻辑、JSON/manual 配置读取逻辑和结果输出逻辑，因此对比更公平。`general-rl` 使用独立的参数体系。

### 3. 推荐写法与兼容写法

为了避免“选了 GA，但后面的参数仍然沿用 RL 命名”的混用，主脚本新增了算法无关写法，并保留了 RL 兼容别名：

| 推荐写法 | 旧兼容别名 | 说明 |
| --- | --- | --- |
| `--skip-stage1-search` | `--skip-stage1-rl` | 跳过第一阶段搜索 |
| `--skip-noise-search` | `--skip-noise-rl` | 跳过第二阶段噪声搜索 |
| `--stage1-search-episodes N` | `--stage1-rl-episodes N` | 设置第一阶段搜索预算 |
| `--stage2-search-episodes N` | `--stage2-rl-episodes N` | 设置第二阶段搜索预算 |

使用规则：

- 选择 `--search-algorithm=rl` 时，推荐写法和旧兼容别名都可以使用。
- 选择 `--search-algorithm=ga` 时，必须使用推荐写法；如果还使用 `--skip-stage1-rl`、`--skip-noise-rl`、`--stage1-rl-episodes`、`--stage2-rl-episodes`，脚本会直接报错。

### 4. JSON 配置文件的算法家族约束

当第一阶段或第二阶段最终评估使用 `json` 配置来源时，脚本会检查配置文件名是否和算法家族一致。

默认配置文件如下：

| 算法 | 第一阶段 JSON 默认值 | 第二阶段 JSON 默认值 |
| --- | --- | --- |
| `rl` | `glue_configs_best_ppo.json` | `glue_noise_configs_best_ppo.json` |
| `ga` | `glue_configs_best_genetic.json` | `glue_noise_configs_best_genetic.json` |

一致性校验规则如下：

- 如果选择 `--search-algorithm=ga`，但第一阶段 JSON 仍使用 `glue_configs_best_ppo.json` 这类 PPO/RL 家族文件，脚本会报错。
- 如果选择 `--search-algorithm=ga`，但第二阶段 JSON 仍使用 `glue_noise_configs_best_ppo.json` 这类 PPO/RL 家族文件，脚本会报错。
- 如果选择 `--search-algorithm=rl`，但 JSON 文件名明显属于 genetic/ga 家族，脚本也会报错。
- 如果你使用的是自定义文件名，只要文件名里不带明显的 `ppo` / `genetic` / `_ga` 等家族标识，脚本会按“自定义配置文件”处理，不会额外阻止。

### 5. 与已有阶段约束的组合规则

这部分规则没有变，但现在会带上算法选择一起检查：

- 若第一阶段搜索会执行，则 `--final-eval-source` 只能为 `search`。
- 若使用 `--final-eval-source json|manual`，则必须跳过第一阶段搜索。
- 若第二阶段搜索会执行，且没有跳过噪声最终评估，则 `--noise-eval-source` 只能为 `search`。
- 若使用 `--noise-eval-source json|manual`，则必须跳过第二阶段搜索。

也就是说，算法选择只是决定”搜索器是 RL 还是 GA”，但不会放宽已有的流程一致性约束。`general-rl` 模式下这些 rl/ga 阶段约束不适用。

### 6. 推荐命令示例

#### 6.1 默认强化学习流程

```bash
bash llama_7B_LayerImportance.sh 32 64 output.log 20 2
```

等价于：

```bash
bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 --search-algorithm rl
```

#### 6.2 完整遗传算法流程

```bash
bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 --search-algorithm ga
```

#### 6.3 遗传算法模式下跳过第一阶段搜索，直接读取第一阶段 JSON 配置

```bash
bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
  --search-algorithm ga \
  --skip-stage1-search \
  --final-eval-source json \
  --final-eval-config glue_configs_best_genetic.json
```

#### 6.4 遗传算法模式下跳过第二阶段搜索，直接读取噪声 JSON 配置

```bash
bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
  --search-algorithm ga \
  --skip-noise-search \
  --noise-eval-source json \
  --noise-eval-config glue_noise_configs_best_genetic.json
```

#### 6.5 强化学习模式下沿用旧参数名

```bash
bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
  --search-algorithm rl \
  --skip-stage1-rl \
  --final-eval-source json \
  --final-eval-config glue_configs_best_ppo.json
```

#### 6.6 通用 RL — 多任务训练（Phase A: 一次性训练通用策略）

```bash
bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
  --search-algorithm general-rl \
  --general-rl-mode train \
  --general-rl-tasks mrpc,cola,rte,stsb \
  --general-rl-rounds 50 \
  --general-rl-episodes-per-round 170 \
  --general-rl-lr 3e-5 \
  --model mrpc
```

训练完成后会在 run 目录下生成 `general_stage1_policy.pt` 和 `general_stage2_noise_policy.pt`。

#### 6.7 通用 RL — 离线推断（Phase B: 快速部署到新任务）

```bash
bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
  --search-algorithm general-rl \
  --general-rl-mode infer \
  --general-stage1-policy general_stage1_policy.pt \
  --general-stage2-policy general_stage2_noise_policy.pt \
  --general-rl-num-rollouts 500 \
  --model qnli
```

只做 Stage-1 推断（跳过 Stage-2）：

```bash
bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
  --search-algorithm general-rl \
  --general-rl-mode infer \
  --general-stage1-policy general_stage1_policy.pt \
  --general-rl-skip-stage2 \
  --model mrpc
```

贪心 rollout（确定性推断，rollout 次数自动置 1）：

```bash
bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
  --search-algorithm general-rl \
  --general-rl-mode infer \
  --general-stage1-policy general_stage1_policy.pt \
  --general-rl-greedy \
  --model mrpc
```

### 7. 常见错误示例

下面这些组合现在会被脚本直接拦下：

```bash
# 错误：GA 模式却继续使用 RL 家族 JSON
bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
  --search-algorithm ga \
  --skip-stage1-search \
  --final-eval-source json \
  --final-eval-config glue_configs_best_ppo.json

# 错误：GA 模式却继续使用 RL 命名兼容别名
bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
  --search-algorithm ga \
  --skip-stage1-rl

# 错误：RL 模式却读取 genetic 家族 JSON
bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
  --search-algorithm rl \
  --skip-noise-search \
  --noise-eval-source json \
  --noise-eval-config glue_noise_configs_best_genetic.json

# 错误：general-rl 模式下使用了 rl/ga 专用参数
bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
  --search-algorithm general-rl \
  --general-rl-mode infer \
  --general-stage1-policy general_stage1_policy.pt \
  --stage1-rl-episodes 51000    # 不允许！

# 错误：rl 模式下使用了 general-rl 专用参数
bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
  --search-algorithm rl \
  --general-stage1-policy general_stage1_policy.pt    # 不允许！
```

### 8. 迁移建议

- 老命令如果不加 `--search-algorithm`，默认仍走 RL，不会影响已有实验。
- 新做 GA 对比实验时，建议显式加 `--search-algorithm ga`，并统一使用 `--skip-stage1-search` / `--skip-noise-search` / `--stage1-search-episodes` / `--stage2-search-episodes` 这组算法无关参数。
- 如果准备长期保留 GA 的 JSON 配置，建议按默认命名方式保存为 `glue_configs_best_genetic.json` 和 `glue_noise_configs_best_genetic.json`，这样脚本能自动做家族一致性检查。
- 使用通用 RL 时，建议先在少量任务上 `train` 训练策略，然后用 `infer` 部署到新任务。通用策略做 500 次 rollout 的耗时约为 per-task RL 的 1/100。

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
| `logfile_path` | nohup 日志文件名提示；真实日志会自动写入当前 run 目录下的 `logs/` 子目录                    |
| `rl_lr`        | PPO 学习率控制。若 `< 1` 则直接作为学习率；旧值如 `20` / `40` 会解释为 `20e-6` / `40e-6` |
| `degree`       | 历史调试参数，固定传 `2`                                                    |


基础示例：
`bash llama_7B_LayerImportance.sh 32 64 output.log 20 2`

### 并行安全运行（Concurrent-safe run layout）

命令格式不变，但 `logfile_path` 现在只用于提示日志文件名（实际写入位置由 run 目录决定）。
每次启动都会自动创建一个唯一的 run 目录：

```text
experiment_results/layer_importance_runs/<dataset>/<YYYYmmdd_HHMMSS>_pid<PID>/
```

启动器会把各类输出写入不同子目录：

- nohup 日志：`.../logs/<basename(logfile_path)>`
- 第一阶段（stage-1）搜索日志 / step 信息 / 曲线：`.../stage1/`
- 第一阶段（stage-1）最终评估输出：`.../stage1_final_eval/`
- 第二阶段（stage-2）噪声 RL 输出与进度快照：`.../stage2_noise/`
- 第二阶段（stage-2）噪声最终评估输出：`.../stage2_noise_final_eval/`

因此，下面这些命令可以同时并行运行，即使它们都使用相同的 `output.log`：

```bash
bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 --skip-stage1-rl --final-eval-source json --final-eval-config glue_configs_best_ppo.json --skip-stage1-final-eval --noise-eval-repeat 200 --model mrpc

bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 --skip-stage1-rl --final-eval-source json --final-eval-config glue_configs_best_ppo.json --skip-stage1-final-eval --noise-eval-repeat 200 --model stsb
```

脚本本身也不再强制设置 `CUDA_VISIBLE_DEVICES=0`。如果你想让并行运行时分别绑定不同 GPU，请在脚本外部设置：

```bash
CUDA_VISIBLE_DEVICES=0 bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 --model mrpc
CUDA_VISIBLE_DEVICES=1 bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 --model stsb
```

#### 如何在命令行并行跑多数据集

实现并行的关键是：每个进程都会自动落到独立的 run 目录（包含 `<dataset>/<YYYYmmdd_HHMMSS>_pid<PID>/`），所以你可以在并行任务里重复使用同一个 `logfile_path`（例如都传 `output.log`），不会互相覆盖产出。

并行常用做法：

1. 最推荐：分别在不同终端窗口/会话里启动不同 `--model`（每条命令就是一个独立实验进程）。
2. 需要在同一终端里同时跑：把每条命令放到后台执行（给命令后面加 `&`），例如 `bash ... &`。

示例（并行跑 MRPC + STS-B；与上面“命令并行可运行”的示例一致）：

```bash
bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
  --skip-stage1-rl --final-eval-source json --final-eval-config glue_configs_best_ppo.json \
  --skip-stage1-final-eval --noise-eval-repeat 200 --model mrpc &

bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
  --skip-stage1-rl --final-eval-source json --final-eval-config glue_configs_best_ppo.json \
  --skip-stage1-final-eval --noise-eval-repeat 200 --model stsb &
```

如果你有多张 GPU，建议再给每条命令绑定不同 GPU（避免显存互抢）：

```bash
CUDA_VISIBLE_DEVICES=0 bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 --model mrpc &
CUDA_VISIBLE_DEVICES=1 bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 --model stsb &
```

#### 并行相关可选参数怎么用（对应上面的并行示例）

下表只聚焦并行时最常用、也出现在上面示例里的参数；更完整的各阶段参数与安全约束见后文各表格。


| 参数                         | 作用                                 | 并行时该怎么配                                                                |
| -------------------------- | ---------------------------------- | ---------------------------------------------------------------------- |
| `logfile_path`             | `nohup` 日志名提示（会取 basename 作为日志文件名） | 多个并行进程可传相同文件名（产出仍在各自 run 目录下）                                          |
| `--model`                  | 选择数据集（并自动匹配对应 `base_model`）        | 并行时让不同进程分别用不同 `--model` 值                                              |
| `--skip-stage1-rl`         | 跳过第一阶段 RL 搜索/训练                    | 并行加速的常用开关：先有/后用已有配置或搜索结果时可加                                            |
| `--final-eval-source json` | 第一阶段最终评估配置来源为 JSON                 | 当 `--final-eval-source` 取 `json`（或 `manual`）时，需要显式加 `--skip-stage1-rl` |
| `--final-eval-config PATH` | 第一阶段最终评估用的 JSON 配置文件路径             | 一般并行时保持一致，避免同时改动多个配置来源                                                 |
| `--skip-stage1-final-eval` | 跳过第一阶段最终评估                         | 只关心后续阶段（例如噪声阶段）时可加                                                     |
| `--noise-eval-repeat N`    | 噪声最终评估重复次数                         | 并行时想要统计更稳可调大；想缩短总耗时可调小                                                 |
| `--skip-noise-rl`          | 跳过第二阶段噪声 RL 训练                     | 只想跑噪声最终评估时加；当 `--noise-eval-source` 用 `json/manual` 时也需要显式加            |
| `--skip-noise-final-eval`  | 跳过第二阶段噪声最终评估                       | 只关心噪声 RL 训练过程/中间产物时加                                                   |
| `--noise-eval-source`      | 噪声最终评估配置来源（`search/json/manual`）   | 并行时常用 `json`：配合 `--noise-eval-config` 直接读配置                            |
| `--noise-eval-config PATH` | `json` 模式下的噪声配置文件                  | 例如默认的 `glue_noise_configs_best_ppo.json`                               |
| `--manual-noise-config`    | `manual` 模式下的噪声配置（JSON 字符串）        | 配置很少且不想改文件时用                                                           |


### --model 数据集+模型切换

可以通过 `--model` 一次性切换数据集和对应 `base_model`，不需要再手动改
`llama_7B_LayerImportance.sh` 里的 `--base_model` / `--data_path`，也不需要再手动改 `rl_tune.py`。

支持值（大小写不敏感）：

- `mrpc`
- `stsb`
- `sst2`
- `wnli`
- `rte`
- `cola`
- `qnli`

映射关系：


| `--model` 值 | 自动设置 `--base_model`                  |
| ----------- | ------------------------------------ |
| `mrpc`      | `textattack/bert-base-uncased-MRPC`  |
| `stsb`      | `textattack/bert-base-uncased-STS-B` |
| `sst2`      | `textattack/bert-base-uncased-SST-2` |
| `wnli`      | `textattack/bert-base-uncased-WNLI`  |
| `rte`       | `textattack/bert-base-uncased-RTE`   |
| `cola`      | `textattack/bert-base-uncased-CoLA`  |
| `qnli`      | `textattack/bert-base-uncased-QNLI`  |


同时自动设置 `--data_path` 为同名任务（如 `--model qnli` -> `--data_path qnli`）。

示例：

```bash
# 默认 mrpc（不写 --model 也可以）
bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 --model mrpc

# 切换到 STS-B（回归任务）
bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 --model stsb

# 切换到 QNLI（问句-句子对）
bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 --model qnli
```

说明：`rl_tune.py` 已改为按 `data_path` 自动选择输入列与 `num_labels`，例如
`stsb -> num_labels=1`，`qnli -> question+sentence`，`sst2/cola -> sentence`，`mrpc/rte/wnli/stsb -> sentence1+sentence2`。

### --model-type 预训练骨干切换（bert-base / bert-large / gpt-2）

通过 `--model-type` 可以在不修改任何 Python 代码的前提下，把整条流程
（第一阶段 GELU/Softmax 搜索、第二阶段噪声 RL、最终评估）从 12 层的
bert-base 切换到 24 层的 bert-large，或切换到 12 层的 gpt-2。
`total_layers` 由 `layer_importance_evaluator.py` 在加载模型后从
`model.bert.encoder.layer` / `model.transformer.h` 等路径自动检测，
下游 PPO 状态向量、动作序列长度、GTrXL 位置嵌入、噪声 RL 等都会按层数
自适应，无需额外参数。

支持值（大小写不敏感）：

- `bert-base`（默认）
- `bert-large`
- `gpt-2`（别名：`gpt2`, `gpt_2`）

映射关系：


| `--model-type` 值 | 预训练 checkpoint 系列                          | 层数  |
| ---------------- | ------------------------------------------ | --- |
| `bert-base`      | `textattack/bert-base-uncased-*`           | 12  |
| `bert-large`     | `yoshitomo-matsubara/bert-large-uncased-*` | 24  |
| `gpt-2`          | `PavanNeerudu/gpt2-finetuned-<task>`（每任务独立微调） | 12  |


`--model-type` 与 `--model` 组合后会按 `(model-type, dataset)` 解析最终
`--base_model`。`bert-base` 兼容此前所有 7 个 GLUE 任务；`bert-large`
当前仅支持以下任务（其余任务暂时跳过，运行时会以
“bert-large 当前不支持数据集: …” 错误退出）：

- `mrpc`
- `cola`
- `stsb`
- `rte`
- `sst2`
- `qnli`

不支持的组合（例如 `--model-type bert-large --model wnli`）会在脚本
启动阶段立即报错并提示当前支持列表，避免到 HuggingFace 下载阶段才
失败。如果未来需要新增 bert-large checkpoint，可在
`llama_7B_LayerImportance.sh` 的 `MODEL_TYPE=bert-large` 分支里
扩展 `case "$DATASET"` 列表。

示例：

```bash
# 在 mrpc 上用 bert-large 跑完整两阶段流程（搜索 + 评估）
bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 --model mrpc --model-type bert-large

# 在 cola 上用 bert-large 跳过第一阶段 RL，仅做最终评估
bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
  --model cola --model-type bert-large \
  --skip-stage1-rl --final-eval-source json --final-eval-config glue_configs_best_ppo.json

# 不写 --model-type 时等价于历史行为（bert-base）
bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 --model mrpc
```

注意事项：

1. bert-large 第一阶段每个 episode 需要在所有 24 层上各做一步决策，
  单次 PPO update 的 token 数也按 `total_layers` 自动翻倍，因此显存
   占用、单 episode 耗时大约是 bert-base 的 2 倍，建议在 24GB 及以上
   显存上运行，必要时通过 `--batch_size` 适当调小。
2. 第二阶段噪声 RL 的状态/动作序列同样按 24 层展开，`noise_rl_module_v2.py`
  已读取 `evaluator.total_layers` 自适应，无需额外配置。
3. 第一阶段最终评估、噪声最终评估、随机对照实验都会按 `total_layers`
  自动扩展数组长度，原有的 `glue_configs_best_ppo.json` /
   `glue_noise_configs_best_ppo.json` 等历史配置文件如果是按 12 层
   保存的，会被 `final_evaluation_module.py` 自动按"最后一个值填充
   或截断"补齐到 24 层并打印 `[Info]` 提示；为了 bert-large 的
   实验复现，建议为 bert-large 单独维护一份按 24 层书写的配置文件。

#### gpt-2 分支使用说明

`gpt-2` 分支把整条 RL / 噪声评估流水线迁移到 HuggingFace 的
GPT-2 骨干（12 层 transformer，768 hidden size）。该分支在模块路径、
QKV 融合、激活函数替换等层面与 BERT 做了专门适配，
`rl_tune.py` / `layer_importance_evaluator.py` / `noise_rl_module_v2.py`
/ `final_evaluation_module.py` / `generate_glue_submission.py` 无需手动
切换，按 `--model-type gpt-2` 一处开关即可。

基座与 checkpoint 来源：

- 采用 **PavanNeerudu 的 `gpt2-finetuned-<task>` 系列**
  (<https://huggingface.co/PavanNeerudu>)，每个 GLUE 任务都有一份已经
  在对应训练集上微调好的 `GPT2ForSequenceClassification` 权重
  （cola / sst2 / mrpc / stsb / qnli / rte / wnli / mnli 均覆盖；stsb
  自动使用回归 head `num_labels=1`，mnli 使用 `num_labels=3`，其余为
  `num_labels=2`）。`llama_7B_LayerImportance.sh` 的 `gpt-2` 分支会按
  `(dataset)` 直接解析到对应 checkpoint，RL 无需从头微调 head，直接
  在一个"已经具备任务能力的骨干"上优化近似/噪声 schedule。
- **Backbone 在 RL 全过程冻结**。`rl_tune.py` 在 `from_pretrained`
  之后会立即执行：
  ```python
  for p in model.parameters():
      p.requires_grad_(False)
  model.eval()
  ```
  PPO 的 `.backward()` 只作用在 policy_net / value_net 上（详见
  `layer_importance_evaluator.py` 内所有 `loss.backward()` 调用），
  所有对 HF 模型的 `forward` 都包在 `torch.no_grad()` 内。因此无论
  近似函数替换、噪声注入还是 GELU 分布分析 hook，都不会把梯度写回
  预训练权重，也不会触发 dropout/LayerNorm 的 train-mode 行为 ——
  这避免了"RL 白训"（即 RL 以为自己在优化噪声 schedule，实际却在
  偷偷微调 backbone 导致最终评估奖励被稀释）。
- Tokenizer 在 `rl_tune.py` 中已统一执行
  `tokenizer.pad_token = tokenizer.eos_token`，并在加载模型时传入
  `pad_token_id=tokenizer.pad_token_id`，满足
  `GPT2ForSequenceClassification` 要求的"末 token pooling + 必须有
  pad token"约束。

功能兼容范围：


| 阶段 / 功能                           | bert-base | bert-large | gpt-2                   |
| --------------------------------- | --------- | ---------- | ----------------------- |
| Stage 1 GELU 多项式近似                | ✅         | ✅          | ✅                       |
| Stage 1 Softmax 指数近似              | ✅         | ✅          | ❌ (自动跳过)                |
| Stage 2 x / Wo / Wffn1 / Wffn2 噪声 | ✅         | ✅          | ✅                       |
| Stage 2 Wq / Wk / Wv 噪声           | ✅         | ✅          | ✅（通过融合 c_attn 的按槽位加噪实现） |
| 最终评估 (`final_evaluation_module`)  | ✅         | ✅          | ✅                       |
| GLUE 提交文件生成                       | ✅         | ✅          | ✅（直接复用 PavanNeerudu 微调权重） |


**为什么 GPT-2 不支持 Softmax 近似？** BERT 的 `BertSelfAttention` 模块
能够被整体替换为 `BertSelfAttentionWithAproximation`，从而在 forward
里用指数近似替换 softmax；而 HuggingFace 的 `GPT2Attention` 将 Q/K/V
融合到单个 Conv1D (`c_attn`)，并把因果 mask + scale + softmax + c_proj
绑死在同一个 forward 里，没有提供同等的可分离入口。本 repo 当前选择
在 GPT-2 上**自动跳过 Stage 1 的 softmax 近似**（`replace_layer_softmax`
会在 GPT-2 上打印警告并直接返回），Stage 1 仍可启用 GELU 近似，
Stage 2 七种噪声全部可用。如果后续需要完整复现 BERT 的 Stage 1 行为，
可以在 `function_handler.py` 里新增 `GPT2AttentionWithApproximation`
包装类对齐 HF 的 `GPT2Attention.forward` 逻辑。

**Q/K/V 噪声在 GPT-2 上的实现细节**：`ReversibleLayerHandler` 会在首次
为某一层调用 `replace_layer_{query,key,value}_noise` 时，包装该层的
`attn.c_attn.forward`。被包装后的 forward 在原有 `W @ x + b` 输出的
基础上，额外计算 `We @ x` 并只写入 `[0, d)` / `[d, 2d)` / `[2d, 3d)`
这三个槽位中被激活的那些，从而让 q/k/v 三路噪声相互独立。当
`restore_layer_{query,key,value}_noise` 把所有槽位都清空后，c_attn
的原始 forward 会被恢复。

`glue_configs_best_ppo.json` 与 `glue_noise_configs_best_ppo.json` 中
都已新增 `"gpt-2"` 顶层段（12 层占位配置，GELU 全部 4、噪声全部保守
值），用于在未跑完 RL 之前也能走通最终评估与 GLUE 提交生成流程。跑完
RL 后请把 PPO 输出的最优配置覆写到这两个文件的 `"gpt-2"` 段。

示例：

```bash
# 在 sst2 上用 gpt-2 跑完整两阶段 RL + 最终评估
bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 --model sst2 --model-type gpt-2

# 在 mrpc 上跳过第一阶段 RL，直接用 JSON 中的 gpt-2 段做最终评估
bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
  --model mrpc --model-type gpt-2 \
  --skip-stage1-rl --final-eval-source json --final-eval-config glue_configs_best_ppo.json

# 用 gpt-2 基座生成 GLUE 官网提交文件（前提：已在训练阶段完成 head 微调）
python generate_glue_submission.py \
  --config glue_configs_best_ppo.json \
  --noise_config glue_noise_configs_best_ppo.json \
  --model_type gpt-2 \
  --output_dir gpt2_run
```

### 命名参数与安全约束

旧版 5 参数命令依然兼容，但新增了更严格的流程校验，避免“前面跑 RL，后面却拿手动/JSON 配置做评估”的混用。

**第一阶段：GELU/Softmax RL 与最终评估**

第一阶段的“是否执行 RL”与“是否执行最终评估”仍可独立控制，但配置来源现在有安全约束：

- 若执行第一阶段 RL，则 `--final-eval-source` 只能为 `search`
- 若使用 `json` 或 `manual`，则必须显式添加 `--skip-stage1-rl`
- 若跳过第一阶段 RL，则不能再使用 `search`
- `--skip-stage1-rl` 会一并跳过 Phase 1 baseline 建立、Phase 1.5 GELU 输入分布分析、Phase 2 PPO 和 Phase 2.5 贪心搜索


| 参数                             | 说明                                                                                  | 默认值                          |
| ------------------------------ | ----------------------------------------------------------------------------------- | ---------------------------- |
| `--skip-stage1-rl`             | 跳过整个第一阶段搜索准备与搜索流程：Phase 1 baseline、Phase 1.5 GELU 输入分布分析、Phase 2 PPO、Phase 2.5 贪心搜索 | 不跳过                          |
| `--skip-stage1-final-eval`     | 跳过第一阶段最终评估（Phase 3 + Phase 4），但仍会先解析第一阶段配置，再进入第二阶段                                  | 不跳过                          |
| `--final-eval-source` | 第一阶段最终评估配置来源：可选 `search`（本次搜索结果）、`json`（`--final-eval-config`）、`manual`（`--manual-gelu` / `--manual-softmax`） | `search`                      |
| `--final-eval-config PATH`     | `json` 模式下的配置文件路径                                                                   | `glue_configs_best_ppo.json` |
| `--manual-gelu "[1,1,...]"`    | `manual` 模式下的每层 GELU degree，必须与 `--manual-softmax` 同时提供                             | —                            |
| `--manual-softmax "[2,2,...]"` | `manual` 模式下的每层 Softmax degree，必须与 `--manual-gelu` 同时提供                             | —                            |
| `--random-seed N`              | 随机实验种子                                                                              | `42`                         |
| `--perm-trials N`              | Permutation 随机对照实验次数                                                                | `10`                         |
| `--cost-trials N`              | 精确 cost-matched 随机对照实验次数                                                            | `10`                         |
| `--budget-trials N`            | 同总预算随机对照实验次数                                                                        | `10`                         |


**第二阶段：噪声 RL 与噪声最终评估**

第二阶段同样增加了安全约束：

- 若执行噪声 RL，且没有跳过噪声最终评估，则 `--noise-eval-source` 只能为 `search`
- 若使用 `json` 或 `manual` 做噪声最终评估，则必须显式添加 `--skip-noise-rl`
- 若跳过噪声 RL，则不能在噪声最终评估中再使用 `search`


| 参数                                        | 说明                            | 默认值                                |
| ----------------------------------------- | ----------------------------- | ---------------------------------- |
| `--skip-noise-rl`                         | 跳过第二阶段噪声 RL 训练                | 不跳过                                |
| `--skip-noise-final-eval`                 | 跳过第二阶段噪声最终评估                  | 不跳过                                |
| `--noise-eval-source` | 第二阶段噪声最终评估配置来源：可选 `search`、`json`（`--noise-eval-config`）、`manual`（`--manual-noise-config`） | `search`                            |
| `--noise-eval-config PATH`                | `json` 模式下的噪声配置文件路径           | `glue_noise_configs_best_ppo.json` |
| `--manual-noise-config '{"x":[...],...}'` | `manual` 模式下的噪声配置，需包含 7 类噪声数组 | —                                  |
| `--noise-eval-repeat N`                   | 噪声最终评估重复次数，必须为正整数             | `1`                                |


第二阶段 RL 训练保持第一阶段选定的 GELU/Softmax 不变，用 PPO 学习每层 7 个噪声 scaling factor：


| 噪声对象                 | 模型路径                   | 动作空间                       |
| -------------------- | ---------------------- | -------------------------- |
| `x`（输入噪声）            | 层输入 hidden_states      | `{20, 22, 24, 26, 28, 30}` |
| `wq`（Query 权重噪声）     | attention.self.query   | `{14, 16, 18, 20, 22}`     |
| `wk`（Key 权重噪声）       | attention.self.key     | `{14, 16, 18, 20, 22}`     |
| `wv`（Value 权重噪声）     | attention.self.value   | `{14, 16, 18, 20, 22}`     |
| `wo`（Attn 输出权重噪声）    | attention.output.dense | `{14, 16, 18, 20, 22}`     |
| `wffn1`（FFN 第一层权重噪声） | intermediate.dense     | `{16, 18, 20, 22, 24}`     |
| `wffn2`（FFN 第二层权重噪声） | output.dense           | `{14, 16, 18, 20, 22}`     |


第二阶段 RL 训练逻辑位于 `noise_rl_module.py`，噪声最终评估逻辑位于 `noise_final_evaluation_module.py`。

第二阶段产出文件：

- `noise_ppo_step_info.txt` — 每步动作/概率日志
- `noise_ppo_training_curve.png` — 训练曲线图
- `noise_ppo_entropy_curve.png` — 策略熵曲线图
- 主日志中搜索 `PHASE 5: SECOND-STAGE NOISE RL` 和 `Best Noise Configuration Found`

**噪声最终评估配置来源**（仅在未 `--skip-noise-final-eval` 时生效）


| 参数                                                | 说明                                                                                                    | 默认值                                |
| ------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | ---------------------------------- |
| `--noise-eval-source search/json/manual`          | 噪声最终评估使用的配置来源：`search` 使用本次噪声 RL 搜索结果；`json` 从 JSON 文件读取；`manual` 手动指定。若执行噪声 RL 且保留最终评估，则只能为 `search` | `search`                           |
| `--noise-eval-config PATH`                        | `json` 模式下指定的噪声配置 JSON 文件路径。程序根据当前数据集名自动读取对应条目                                                        | `glue_noise_configs_best_ppo.json` |
| `--manual-noise-config '{"x":[...],...}'`         | `manual` 模式下手动指定 7 种噪声 scaling factor 数组（JSON 对象格式），支持短名称 `x, wq, wk, wv, wo, wffn1, wffn2`           | —                                  |
| `--noise-eval-repeat N`                           | 对选定配置执行 N 次重复评估，输出 N 次结果及均值/标准差统计                                                                     | `1`                                |
| 环境变量 `NOISE_RANDOM_MODE=x_only/x_w/x_w_nonlinear` | 噪声随机消融实验模式，控制 `Full Random` 对照组中随机化的范围（详见下一节）                                                         | `x_w`                              |


#### 噪声随机消融实验模式（`NOISE_RANDOM_MODE`）

噪声最终评估会针对选定配置生成一组 `Full Random` 随机对照实验，用于检验所选配置相对随机基线的优势。该随机对照支持三种消融模式，便于分别考察不同噪声因子的影响：


| 模式              | 含义                                                                                             |
| --------------- | ---------------------------------------------------------------------------------------------- |
| `x_only`        | **只随机 X**：每次只对输入噪声 `x` 重新采样，6 类权重噪声 `wq/wk/wv/wo/wffn1/wffn2` 与非线性层（GELU/Softmax）保持为选定配置的值     |
| `x_w`           | **随机 X + 所有 W**（默认，即原始行为）：对 `x` 与全部 6 类权重噪声独立随机采样，非线性层固定                                       |
| `x_w_nonlinear` | **随机 X + 所有 W + 非线性层**：在 `x_w` 基础上，再对每层的 GELU 阶数（`{0,1,2,4}`）和 Softmax 阶数（`{2,3,4,5,6}`）独立随机采样 |


**手动选择方式**：通过环境变量 `NOISE_RANDOM_MODE` 在执行命令前指定，例如：

```bash
# 1) 仅随机 X
NOISE_RANDOM_MODE=x_only bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
  --skip-stage1-rl --final-eval-source json --final-eval-config glue_configs_best_ppo.json \
  --skip-stage1-final-eval --skip-noise-rl --noise-eval-source json \
  --noise-eval-config glue_noise_configs_best_ppo.json --noise-eval-repeat 200 --model mrpc

# 2) 随机 X + 所有 W （默认，可省略环境变量）
NOISE_RANDOM_MODE=x_w bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
  --skip-stage1-rl --final-eval-source json --final-eval-config glue_configs_best_ppo.json \
  --skip-stage1-final-eval --skip-noise-rl --noise-eval-source json \
  --noise-eval-config glue_noise_configs_best_ppo.json --noise-eval-repeat 200 --model mrpc

# 3) 随机 X + 所有 W + 非线性层（GELU/Softmax）
NOISE_RANDOM_MODE=x_w_nonlinear bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
  --skip-stage1-rl --final-eval-source json --final-eval-config glue_configs_best_ppo.json \
  --skip-stage1-final-eval --skip-noise-rl --noise-eval-source json \
  --noise-eval-config glue_noise_configs_best_ppo.json --noise-eval-repeat 200 --model mrpc
```

> Windows PowerShell 下设置环境变量请用 `$env:NOISE_RANDOM_MODE="x_only"; bash ...`；CMD 下用 `set NOISE_RANDOM_MODE=x_only && bash ...`。
>
> 选择的模式会在主日志开头以 `NOISE_RANDOM_MODE=...` 的形式打印，便于结果归档对照。

噪声配置 JSON 文件格式：

```json
{
  "mrpc": {
    "x": [20, 22, 24, 26, 28, 30, 20, 22, 24, 26, 28, 30],
    "wq": [14, 16, 18, 20, 22, 14, 16, 18, 20, 22, 14, 16],
    "wk": [14, 16, 18, ...],
    "wv": [14, 16, 18, ...],
    "wo": [14, 16, 18, ...],
    "wffn1": [16, 18, 20, ...],
    "wffn2": [14, 16, 18, ...]
  }
}
```

噪声最终评估的逻辑位于独立模块 `noise_final_evaluation_module.py` 中，功能与第一阶段 `final_evaluation_module.py` 一致，并新增 N 次重复评估。

噪声最终评估产出文件（位于当前 run 目录下的 `stage2_noise_final_eval/` 目录）：

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
`bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 --skip-noise-rl --noise-eval-source manual --manual-noise-config '{"x":[20,22,24,26,28,30,20,22,24,26,28,30],"wq":[14,16,18,20,22,14,16,18,20,22,14,16],"wk":[14,16,18,20,22,14,16,18,20,22,14,16],"wv":[14,16,18,20,22,14,16,18,20,22,14,16],"wo":[14,16,18,20,22,14,16,18,20,22,14,16],"wffn1":[16,18,20,22,24,16,18,20,22,24,16,18],"wffn2":[14,16,18,20,22,14,16,18,20,22,14,16]}' --noise-eval-repeat 100`

只进行第二阶段rl  
`CUDA_VISIBLE_DEVICES=0 bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 --skip-stage1-rl --final-eval-source json --final-eval-config glue_configs_best_ppo.json --skip-stage1-final-eval --noise-eval-repeat 200 --model mrpc --stage2-rl-episodes [轮数] --batch_size [batch size 大小]`

（实例）

`CUDA_VISIBLE_DEVICES=0 bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 --skip-stage1-rl --final-eval-source json --final-eval-config glue_configs_best_ppo.json --skip-stage1-final-eval --noise-eval-repeat 200 --model mrpc --stage2-rl-episodes 15000 --batch_size 128`

只进行第二阶段最终评估

`CUDA_VISIBLE_DEVICES=0 NOISE_RANDOM_MODE=x_w_nonlinear bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 --skip-stage1-rl --final-eval-source json --final-eval-config glue_configs_best_ppo.json --skip-stage1-final-eval --skip-noise-rl --noise-eval-source json --noise-eval-config glue_noise_configs_best_ppo.json --noise-eval-repeat 200 --model mrpc --batch_size 128`

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
  --manual-noise-config '{"x":[20,22,24,26,28,30,20,22,24,26,28,30],"wq":[14,16,18,20,22,14,16,18,20,22,14,16],"wk":[14,16,18,20,22,14,16,18,20,22,14,16],"wv":[14,16,18,20,22,14,16,18,20,22,14,16],"wo":[14,16,18,20,22,14,16,18,20,22,14,16],"wffn1":[16,18,20,22,24,16,18,20,22,24,16,18],"wffn2":[14,16,18,20,22,14,16,18,20,22,14,16]}'
```

帮助：  
`bash llama_7B_LayerImportance.sh --help`

使用 JSON 配置生成 GLUE 官网提交的 TSV（以及可选的 `submission.zip`），输出目录为 **`glue_submission/<output_dir>/`**；`--output_dir` 默认为 `run`，即默认写到 `glue_submission/run`。默认骨干为 **`bert-base`**；若使用仓库里为 GPT-2 准备的配置段，需加 **`--model_type gpt-2`**，且须先在训练流程中完成分类头微调。若无 GPU 且需跑 CPU，需加 **`--allow_cpu`**（会很慢）。

```bash
python generate_glue_submission.py \
  --config glue_configs_best_ppo.json \
  --noise_config glue_noise_configs_best_ppo.json \
  --output_dir my_glue_run
```

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

### `--batch_size` 可选项

可以通过命令行额外传入 `--batch_size N` 来覆盖当前脚本默认的批大小设置。


| 参数               | 说明                                                                                                                                                                | 默认值  |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- |
| `--batch_size N` | 统一设置 `llama_7B_LayerImportance.sh` 启动后强化学习与评估阶段使用的批大小。脚本会同步把 `--batch_size` 和 `--micro_batch_size` 都设为 `N`，并继续传递给 `rl_tune.py` 和 `layer_importance_evaluator.py`。 | `16` |


使用说明：

- `N` 必须是正整数，例如 `4`、`8`、`16`、`32`。
- 这个参数会影响 `Trainer` 的评估批大小，以及 `layer_importance_evaluator.py` 内部各个 dataloader 的 batch size。
- 数值调大后通常吞吐会更高，但显存占用也会更高；如果出现 OOM，建议先降到 `8` 或 `4`。
- 当前脚本为了保持原有行为一致，会把 `micro_batch_size` 一并设置成和 `batch_size` 相同的值。

示例：

```bash
# 使用 batch size = 8 运行 MRPC
bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 --batch_size 8 --model mrpc

# 使用 batch size = 4，只运行第二阶段 noise RL
CUDA_VISIBLE_DEVICES=0 bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
  --batch_size 4 \
  --skip-stage1-rl \
  --final-eval-source json \
  --final-eval-config glue_configs_best_ppo.json \
  --skip-stage1-final-eval \
  --noise-eval-repeat 200 \
  --model mrpc
```

### `--stage1-rl-episodes` / `--stage2-rl-episodes` 可选项

可以通过命令行额外传入第一阶段和第二阶段强化学习轮数，分别控制 Stage-1 GELU/Softmax RL 和 Stage-2 noise RL 的 episode 数。


| 参数                       | 说明                                                                   | 默认值     |
| ------------------------ | -------------------------------------------------------------------- | ------- |
| `--stage1-rl-episodes N` | 设置第一阶段强化学习轮数，对应 `layer_importance_evaluator.py` 中的 Stage-1 PPO 搜索轮数。 | `51000` |
| `--stage2-rl-episodes N` | 设置第二阶段强化学习轮数，对应 `noise_rl_module.py` 中的 Stage-2 noise PPO 搜索轮数。      | `40000` |


使用说明：

- `N` 必须是正整数。
- 当对应阶段没有被跳过时，`N` 必须大于等于 `170`。
这是因为当前 `PPO_UPDATE_INTERVAL=170`，如果轮数小于 `170`，PPO 将无法完成一次真正的策略更新。
- 如果使用了 `--skip-stage1-rl`，就不能再同时显式传入 `--stage1-rl-episodes`。
- 如果使用了 `--skip-noise-rl`，就不能再同时显式传入 `--stage2-rl-episodes`。
- 这两个参数只控制强化学习搜索轮数，不影响最终评估重复次数；最终评估重复次数仍然由 `--noise-eval-repeat` 等参数控制。

示例：

```bash
# 同时自定义第一阶段和第二阶段 RL 轮数
bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
  --stage1-rl-episodes 1020 \
  --stage2-rl-episodes 3400 \
  --model mrpc

# 跳过第一阶段，只运行第二阶段 noise RL，并把第二阶段轮数改成 680
CUDA_VISIBLE_DEVICES=0 bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
  --skip-stage1-rl \
  --final-eval-source json \
  --final-eval-config glue_configs_best_ppo.json \
  --skip-stage1-final-eval \
  --stage2-rl-episodes 680 \
  --noise-eval-repeat 200 \
  --model mrpc
```

### `--resume-from` 断点续训可选项

当一次搜索/训练完成后，如果发现轮数不够，可以通过 `--resume-from` 指定之前的 run 目录，在之前训练的基础上继续训练更多轮次。效果等价于一次性训练更多轮（例如先训 30000 轮，再续训 10000 轮，等价于一次性训练 40000 轮）。**此功能适用于 rl、ga、general-rl 三种搜索算法。**

训练过程中会自动在 run 目录下保存 checkpoint 文件（每次 PPO 更新窗口 / 遗传代际 / round 结束时保存）：

| 搜索算法 | Stage-1 checkpoint 路径 | Stage-2 checkpoint 路径 |
| --- | --- | --- |
| `rl` | `<run_dir>/stage1/stage1_rl_checkpoint.pt` | `<run_dir>/stage2_noise/progress/noise_rl_checkpoint.pt` |
| `ga` | `<run_dir>/stage1/ga_stage1_checkpoint.pt` | `<run_dir>/stage2_noise/progress/ga_stage2_noise_checkpoint.pt` |
| `general-rl` | `<run_dir>/stage1/general_stage1_train_checkpoint.pt` | `<run_dir>/stage2_noise/general_stage2_train_checkpoint.pt` |


| 参数                   | 说明                                                                                                                                          | 默认值 |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | --- |
| `--resume-from PATH` | 指定之前的 run 目录路径，从该目录的 checkpoint 恢复训练。程序会根据 `--search-algorithm` 自动在对应路径查找 checkpoint 文件。如果 checkpoint 不存在，则从头开始训练。 | 空   |


使用说明：

- `PATH` 必须是一个已存在的 run 目录（例如 `rl_results/layer_importance_runs/mrpc/20260404_151155_pid711833`）。
- 续训时指定的轮数表示**总轮数**（而非追加轮数）。例如之前训了 30000 轮，想再加 10000 轮，则设置 `--stage2-search-episodes 40000`。
- 如果指定的总轮数小于等于 checkpoint 中已完成的轮数，则该阶段不会追加训练。
- `--resume-from` 可以与各模式的跳过选项组合使用：只有未被跳过的阶段才会尝试加载对应的 checkpoint。
- 续训产出的新日志和文件会写入新生成的 run 目录（不会覆盖原目录），但模型状态和训练统计会从旧 checkpoint 恢复。
- checkpoint 会在每次进度快照时自动保存，因此即使训练中途被中断，也可以从最近的 checkpoint 恢复。

示例：

```bash
# ---- RL 模式续训 ----

# 第一次训练：Stage-2 训练 15000 轮
CUDA_VISIBLE_DEVICES=0 bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
  --skip-stage1-rl \
  --final-eval-source json \
  --final-eval-config glue_configs_best_ppo.json \
  --skip-stage1-final-eval \
  --noise-eval-repeat 200 \
  --model mrpc \
  --stage2-rl-episodes 15000

# 发现轮数不够，续训到 30000 轮
CUDA_VISIBLE_DEVICES=0 bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
  --skip-stage1-rl \
  --final-eval-source json \
  --final-eval-config glue_configs_best_ppo.json \
  --skip-stage1-final-eval \
  --noise-eval-repeat 200 \
  --model mrpc \
  --stage2-rl-episodes 30000 \
  --resume-from rl_results/layer_importance_runs/mrpc/20260404_151155_pid711833

# Stage-1 续训示例：先训 10000 轮，再续训到 20000 轮
bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
  --model mrpc \
  --stage1-rl-episodes 10000 \
  --skip-noise-rl --skip-noise-final-eval

bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
  --model mrpc \
  --stage1-rl-episodes 20000 \
  --skip-noise-rl --skip-noise-final-eval \
  --resume-from rl_results/layer_importance_runs/mrpc/<之前的run目录>

# ---- GA 模式续训 ----

# 第一次 GA 搜索
bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
  --search-algorithm ga \
  --model mrpc

# GA 续训（从上次中断处继续）
bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
  --search-algorithm ga \
  --model mrpc \
  --resume-from rl_results/layer_importance_runs/mrpc/<之前的ga_run目录>

# ---- General-RL 模式续训 ----

# 第一次通用 RL 训练
bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
  --search-algorithm general-rl \
  --general-rl-mode train \
  --general-rl-tasks mrpc,cola,rte,stsb \
  --general-rl-rounds 50 \
  --model mrpc

# 从上次训练中断处继续
bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
  --search-algorithm general-rl \
  --general-rl-mode train \
  --general-rl-tasks mrpc,cola,rte,stsb \
  --general-rl-rounds 100 \
  --model mrpc \
  --resume-from rl_results/layer_importance_runs/mrpc/<之前的general-rl_run目录>
```

### 优雅停止与断点续训（Graceful Stop / Resume）

由于 `llama_7B_LayerImportance.sh` 内部以 `nohup ... &` 的方式把实际的 Python
训练进程放到后台运行，因此**请不要再使用 `kill -9 <PID>`（SIGKILL）**，否则会
绕过 checkpoint 保存逻辑、导致下次续训时训练曲线出现明显断层。正确的做法是
**发送 SIGINT 触发优雅停止**，程序会在下一次安全边界（PPO 更新边界 / 遗传代际边界 / round 边界）
保存 checkpoint，然后安全退出；之后用 `--resume-from` 指向同一个 run 目录即可
**严丝合缝**地继续训练。**此功能适用于 rl、ga、general-rl 三种搜索算法。**

#### 启动脚本会提示什么

每次执行 `bash llama_7B_LayerImportance.sh ...` 启动一个新任务后，脚本会在终端
打印类似下面的块（同一份内容也会写入 `logs/*.log`），复制粘贴即可停止训练：

```
========================================================================
  优雅停止 (Graceful Stop) — 任选其一即可安全中断训练并保存 checkpoint
========================================================================
  方式 A (推荐，最简单)：直接发送 SIGINT 到当前进程
      kill -INT 712345

  方式 B：通过数据集级 LATEST 指针（无需记住 PID / run 目录）
      kill -INT $(cat rl_results/layer_importance_runs/mrpc/LATEST_PID)

  方式 C：创建停止标志文件（不依赖信号，适合脚本化批量停止）
    ── rl 或 ga 模式：
      touch rl_results/.../20260408_.../stage1/STOP_RL              # 停 Stage-1
      touch rl_results/.../20260408_.../stage2_noise/progress/STOP_RL  # 停 Stage-2
    ── general-rl 模式：
      touch rl_results/.../20260408_.../STOP_RL                     # 停当前 Stage

  停止后续训：下次启动时加上 --resume-from rl_results/.../20260408_...
  ⚠ 切勿使用 kill -9（SIGKILL），它会绕过 checkpoint 保存导致续训断层！
========================================================================
```

脚本同时在以下位置持久化了 PID / run 目录信息：

- `<run_dir>/rl.pid`：当前任务的 Python 进程 PID
- `rl_results/layer_importance_runs/<dataset>/LATEST_PID`：该数据集最近一次启动的 PID
- `rl_results/layer_importance_runs/<dataset>/LATEST_RUN_DIR`：该数据集最近一次启动的 run 目录

#### 完整示例：启动 → 优雅停止 → 续训

以你给出的命令为例：

```bash
# 1) 启动训练（脚本内部会 nohup 后台运行并打印 PID / 停止命令）
CUDA_VISIBLE_DEVICES=0 bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
    --skip-stage1-rl \
    --final-eval-source json \
    --final-eval-config glue_configs_best_ppo.json \
    --skip-stage1-final-eval \
    --noise-eval-repeat 200 \
    --model mrpc \
    --stage2-rl-episodes 15000 \
    --batch_size 128
# -> 终端会打印 Background PID: 712345 以及 Resolved run root: rl_results/.../20260408_.../
```

```bash
# 2) 想停的时候，任选一条（同一条命令可多次执行，首次触发后即会进入保存流程）：

# 方式 A：直接用启动时打印的 PID
kill -INT 712345

# 方式 B：用 LATEST 指针（最省事，不必记 PID）
kill -INT $(cat rl_results/layer_importance_runs/mrpc/LATEST_PID)

# 方式 C：创建停止标志文件（Stage-2 训练时）
touch "$(cat rl_results/layer_importance_runs/mrpc/LATEST_RUN_DIR)/stage2_noise/progress/STOP_RL"
```

收到信号后，你会在日志里看到：

（这里ctrl+c等价于命令行再用一次kill -INT pid）

```
  [优雅停止] 收到中断信号 (Ctrl+C)，将在当前回合结束后保存 checkpoint 并退出；再按一次 Ctrl+C 立即强退。
  [优雅停止] 已检测到停止请求，正在于 PPO 边界保存 Stage-2 checkpoint (episode=..., update#...) ...
  [优雅停止] checkpoint 已写入 → rl_results/.../20260408_.../stage2_noise/progress/noise_rl_checkpoint.pt
  下次启动请使用 --resume-from 指向本 run 目录以严丝合缝续训。
```

```bash
# 3) 续训：只需把原命令加上 --resume-from 指向上次的 run 目录即可
#    （建议同时把 --stage2-rl-episodes 调到你想要的“总轮数”）
CUDA_VISIBLE_DEVICES=0 bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
    --skip-stage1-rl \
    --final-eval-source json \
    --final-eval-config glue_configs_best_ppo.json \
    --skip-stage1-final-eval \
    --noise-eval-repeat 200 \
    --model mrpc \
    --stage2-rl-episodes 30000 \
    --batch_size 128 \
    --resume-from "$(cat rl_results/layer_importance_runs/mrpc/LATEST_RUN_DIR)"
```

续训会加载上次保存的 GTrXL 网络权重、优化器状态、PPO 更新计数、`reward_history`
/ `reward_mean` / `reward_std`、`return_normalizer`（值函数归一化统计量）、
incumbent / best_config / window_best、以及所有 episode 级统计列表，因此
training-curve 曲线与 PPO 训练曲线在续训前后能**严丝合缝地接上**。

#### 完整示例：GA 模式下的启动 → 优雅停止 → 续训

```bash
# 1) 启动 GA 搜索
bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
    --search-algorithm ga \
    --model mrpc
# -> 脚本打印 Background PID: 812345 及停止命令

# 2) 优雅停止（方式 A / B / C 均可）
kill -INT 812345

# 或创建停止标志文件：
# touch rl_results/.../stage1/STOP_RL        # 停 GA Stage-1
# touch rl_results/.../stage2_noise/progress/STOP_RL  # 停 GA Stage-2

# 3) 续训：加上 --resume-from 指向上次的 run 目录
bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
    --search-algorithm ga \
    --model mrpc \
    --resume-from "$(cat rl_results/layer_importance_runs/mrpc/LATEST_RUN_DIR)"
```

#### 完整示例：General-RL 模式下的启动 → 优雅停止 → 续训

```bash
# 1) 启动通用 RL 多任务训练
bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
    --search-algorithm general-rl \
    --general-rl-mode train \
    --general-rl-tasks mrpc,cola,rte,stsb \
    --general-rl-rounds 50 \
    --model mrpc
# -> 脚本打印 Background PID: 912345 及停止命令

# 2) 优雅停止
kill -INT 912345

# 或创建停止标志文件（general-rl 的 STOP_RL 位于 run 根目录）：
# touch rl_results/.../STOP_RL

# 3) 续训：增大 rounds 并指定 --resume-from
bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
    --search-algorithm general-rl \
    --general-rl-mode train \
    --general-rl-tasks mrpc,cola,rte,stsb \
    --general-rl-rounds 100 \
    --model mrpc \
    --resume-from "$(cat rl_results/layer_importance_runs/mrpc/LATEST_RUN_DIR)"
```

#### 严丝合缝续训的关键机制

为了让“停→续”与“一次性训练到底”等效，代码在不同算法中强制在安全边界保存 checkpoint：

- **Stage-1**：每个 PPO 更新窗口结束（即 `(episode+1) % PPO_UPDATE_INTERVAL == 0`）
后会写一次 checkpoint；优雅停止检查紧跟其后，因此保存时 buffer 刚刚清空、
`gtrxl_ppo_update_count` 和 `episode+1` 恰好对齐，resume 时直接从下一个 PPO 窗口
开始，不会出现“窗口长度不足”的短窗现象。
- **Stage-2**：同样在 PPO 更新 + buffer clear + `noise_ppo_update_count` 递增之后
才检查停止请求并强制保存，保证 `completed_episodes` 正好是 `PPO_UPDATE_INTERVAL`
的整数倍。原本的周期性 checkpoint（每 `NOISE_STAGE_CHECKPOINT_PPO_INTERVAL`
次 PPO 更新一次）继续保留，作为“就算没人触发优雅停止、机器异常断电也能续上”的兜底。

由于停止点必然落在 PPO 边界上，resume 后的第一次 PPO 更新仍然会收集整整一个
`PPO_UPDATE_INTERVAL` 的新 rollout 再做更新，策略/价值网络的梯度步数与回合计数
与“一次性训练到底”完全一致。

**ga 模式（遗传算法）：** 在每一代（generation）的遗传操作完成后保存 checkpoint。续训时从下一代直接开始，种群状态、历史统计、缓存均完整恢复。

**general-rl 模式（通用RL）：** 在每一轮 round-robin 结束后保存 checkpoint。续训时从下一个 round 开始，通用策略/critic 权重、优化器状态、round 计数均完整恢复。

由于停止点必然落在安全边界上（PPO 更新边界 / 代际边界 / round 边界），resume 后的训练与“一次性训练到底”在统计上完全一致。

#### 注意事项

- **必须用 `kill -INT`（SIGINT）或停止标志文件，禁用 `kill -9`（SIGKILL）。** SIGKILL
不会被 Python 捕获，无法触发 checkpoint 保存，续训会丢失最近一段窗口的训练成果。
- 停止请求在“下一次安全边界”生效。rl 模式最多延迟 PPO_UPDATE_INTERVAL 个
episode，ga 模式在当代结束后生效，general-rl 模式在当前 round 结束后生效。
日志里出现 [优雅停止] checkpoint 已写入 就说明安全退出完成。
- 若再按一次 `Ctrl+C`（或再次 `kill -INT`），程序会抛出 `KeyboardInterrupt`
立刻强退——此时已保存的 checkpoint 仍然有效，只是最新窗口内未保存的 rollout 会丢失。
- 若停止时恰好处于 RL 训练之外（例如 Stage-1/2 的最终评估阶段），停止标志不会生效；
请等 RL 训练进入下一个 PPO 窗口时再观察日志。
- 续训时 `--stage1-rl-episodes` / `--stage2-rl-episodes` 指的是**总轮数**（不是追加
轮数）。若总轮数小于等于 checkpoint 中已完成的轮数，该阶段不会再追加训练。
- 成功停止后，脚本会自动删除已消费的 `STOP_RL` 文件，避免下次启动被误触发。
- Windows 下同样可用：`Ctrl+C` 会走 SIGINT 分支；停止标志文件用资源管理器或
`type NUL > STOP_RL` 手动创建即可。

### 可迁移 Policy / Critic（Portable Policy & Critic Transfer）

第一阶段（Stage-1，层重要度 PPO）和第二阶段（Stage-2，噪声 PPO）训练结束后，
除了常规的"续训用" checkpoint 之外，还会额外写出一份**便携 policy 文件**，
仅包含网络权重 + 架构超参 + 元数据，与训练状态彻底解耦，可作为 base policy
迁移到**不同的准确度约束**或**不同的数据集**上。

#### 1. 与"续训"的区别

| 机制 | 文件 | 内容 | 作用 |
| --- | --- | --- | --- |
| 续训 (resume) | `stage1_rl_checkpoint.pt` / `noise_rl_checkpoint.pt` | net + optimizer + episode counter + best 配置 + 训练统计 | 在**同一个** run 中接着训练 |
| 迁移 (transfer) | `stage1_policy.pt` / `stage2_noise_policy.pt` | 仅 `net_state_dict` + `arch` + `metadata` | 把权重作为 base policy 用于**新 run / 新任务** |

迁移启动时是一个全新训练：episode 从 0 开始、optimizer 全新、best 配置清空，
**只有 policy 与 critic 网络的权重**继承自预训练 artifact。两套机制互不干扰，
你既可以 `--resume-run-dir` 续训，也可以从另一个任务的 portable policy 启动一个全新 run。

#### 2. 自动保存路径

只要保持默认开关（见下文 §5），训练正常结束时会自动写出：

```text
<run_dir>/stage1/stage1_policy.pt              # 第一阶段便携 policy
<run_dir>/stage2_noise/stage2_noise_policy.pt  # 第二阶段便携 noise policy
```

文件格式（`torch.save` 的 dict）：

```python
{
    "version": 1,
    "kind": "stage1_gtrxl_policy" | "stage2_noise_gtrxl_policy",
    "net_state_dict": <state_dict>,           # actor + critic 一并包含
    "arch": {
        "num_layers": 12, "d_model": ..., "n_heads": ...,
        "n_gtrxl_layers": ..., "d_ff": ..., "dropout": ...,
    },
    "metadata": {
        "trained_episodes": ...,
        "best_reward": ...,                    # 仅 stage1
        "best_final_selection_score": ...,     # 仅 stage2
        "best_cost": ...,
        "error_threshold": ...,                # 仅 stage1：来源任务的准确度约束
        "correlation_drop_ratio": ...,         # 仅 stage1：来源任务的指标下降容忍
    },
}
```

`metadata` 里写入了"来源任务的约束"，方便你在迁移时核对来源。

#### 3. 启用迁移：作为 base policy 加载

迁移不通过命令行参数，而是通过文件顶部的 `RL_OPT_FLAGS` /
`NOISE_RL_OPT_FLAGS` 字典控制，便于做消融实验。**改完之后正常启动训练命令即可。**

**第一阶段（Stage-1）迁移**——编辑 `layer_importance_evaluator.py` 顶部：

```python
RL_OPT_FLAGS = {
    ...
    # 把这一项从 None 改成你要迁移的 stage1_policy.pt 的绝对路径
    "stage1_pretrained_policy_path": "/abs/path/to/old_run/stage1/stage1_policy.pt",
    ...
}
```

**第二阶段（Stage-2）迁移**——编辑 `noise_rl_module_v2.py` 顶部：

```python
NOISE_RL_OPT_FLAGS = {
    ...
    "pretrained_policy_path": "/abs/path/to/old_run/stage2_noise/stage2_noise_policy.pt",
    ...
}
```

设好之后正常 `bash llama_7B_LayerImportance.sh ...` 启动即可。新 run 启动日志中
会出现一行：

```
[迁移] 已加载预训练 policy/critic ← /abs/path/.../stage1_policy.pt (missing=0, unexpected=0)
```

`missing` / `unexpected` 表示 strict=False 加载时不匹配的层数，理想情况下都为 0。

#### 4. 支持的迁移场景

| 迁移场景 | 兼容性 | 备注 |
| --- | --- | --- |
| **同数据集 + 不同准确度约束**（例如 1.5% → 1.0%） | ✅ 100% 命中 | 架构完全一致，权重全量迁移；新约束下 reward 函数变了，policy 在已学到的"层敏感度先验"基础上继续微调即可。 |
| **同架构 + 不同数据集**（例如 BERT-base 12 层 MRPC → STSB） | ✅ 100% 命中 | `total_layers` 不变 ⇒ layer-index embedding 与所有 GTrXL 层 shape 一致，可全部继承。 |
| **不同模型规模**（BERT-large 24 层 → BERT-base 12 层） | ⚠️ 部分迁移 | `total_layers` 不同 ⇒ layer embedding / 位置相关层 shape 不匹配，被 `strict=False` 跳过；GTrXL 主干、actor / critic head 仍可继承。日志会报 `missing=X, unexpected=Y`。 |
| **跨阶段迁移**（Stage-1 ↔ Stage-2） | ❌ 不支持 | 两个阶段动作空间维度完全不同（GELU/Softmax vs 7 类噪声 scaling factor），actor head 不兼容；必须分别迁移。 |

> ⚠️ 加载失败 / shape 不匹配的层会以**新随机初始化**继续训练，而不是报错中断；
> 因此即使是部分迁移也是安全的，但你应当读一下日志里的 missing / unexpected 数量，
> 确认迁移的"覆盖率"符合预期。

#### 5. 一键关闭 / 一键回滚

所有迁移功能都由 flag 控制，关闭后等价于优化前的旧行为：

```python
# layer_importance_evaluator.py
RL_OPT_FLAGS["stage1_save_portable_policy"]   = False  # 不再写出 stage1_policy.pt
RL_OPT_FLAGS["stage1_pretrained_policy_path"] = None   # 不加载预训练 policy

# noise_rl_module_v2.py
NOISE_RL_OPT_FLAGS["save_portable_policy"]    = False  # 不再写出 stage2_noise_policy.pt
NOISE_RL_OPT_FLAGS["pretrained_policy_path"]  = None   # 不加载预训练 noise policy
```

#### 6. 推荐工作流

1. **第一次训练**（来源任务）：保持默认开关跑完一轮 Stage-1 / Stage-2，
   会在 `<run_dir>/stage1/stage1_policy.pt` 与
   `<run_dir>/stage2_noise/stage2_noise_policy.pt` 拿到便携 artifact。
2. **复制 / 备份**：把这两个文件挪到一个长期保存的目录（例如
   `experiment_results/portable_policies/mrpc_strict15/`），便于后续引用。
3. **新任务迁移训练**：把对应路径填进 `RL_OPT_FLAGS` /
   `NOISE_RL_OPT_FLAGS`，正常启动训练命令；新 run 会从迁移权重开始，
   通常能显著缩短到达高 reward 平台的回合数。
4. **核对日志**：训练开头务必检查 `[迁移] ... missing=? unexpected=?` 的输出，
   确认权重加载覆盖率符合预期。
5. **配合局部最优检测**：训练结束后查看 `<run_dir>/stage{1,2}/pruning_search_log.txt`，
   若报告 `LIKELY LOCAL-OPTIMUM`，可考虑换一个更"远"的 base policy 重新迁移。

### 通用策略 / 通用 Critic（General Policy & General Critic）

`general_policy_module.py` 提供了**跨任务通用策略（General Policy）**和**通用 Critic（General Critic）**的训练与离线部署能力，是对现有 per-task online RL 的补充。

#### 核心思想

1. **多任务轮训（Multi-Task Round-Robin Training）**：在多个数据集 / 约束设定上轮流采集 rollout，共同更新同一个 policy + critic 网络，使 policy 学到"哪些层对精度更敏感"的通用先验，critic 学到跨任务的状态价值模式。
2. **离线推断（Offline Inference）**：加载训练好的 general policy，在新任务上**只做前向推理（不训练）**，通过 best-of-K rollout 找到该任务的最优配置。
3. **通用 Critic 快速评分**：Policy 和 Critic 共享 GTrXL 骨干网络。通用 Critic 可以用 V(s) 对候选配置快速评分（不做模型评测），适合预筛选大量候选。
4. **与现有 online RL 完全并存**：现有的 per-task online RL（`layer_importance_evaluator.py` 的 Stage-1 PPO、`noise_rl_module_v2.py` 的 Stage-2 噪声 PPO）完全不受影响。

#### 网络架构

通用策略网络在原有 `GTrXLStrategyNetwork` 基础上添加了**任务上下文嵌入（task_context_proj）**分支：

- 输入：5 维任务上下文向量 `[baseline_loss, baseline_m1, baseline_m2, error_threshold, correlation_drop_ratio]`
- 投影：`Linear → LayerNorm → SiLU → Linear → (d_model)`
- 注入方式：作为可加偏置注入每个 token，不改变基础 GTrXL 架构
- **零初始化**：最后一层线性层零初始化，加载 per-task 权重时 `task_context_proj` 输出全零，等价于无任务上下文，保证前向兼容

Stage-1 和 Stage-2 分别对应两个网络类：

| 网络类 | 基类 | 用途 |
| --- | --- | --- |
| `GeneralStage1PolicyNetwork` | `GTrXLStrategyNetwork` | Stage-1 GELU/Softmax 通用策略 + Critic |
| `GeneralStage2NoisePolicyNetwork` | `_NoiseGTrXLStrategyNetwork` | Stage-2 噪声 scaling factor 通用策略 + Critic |

#### 使用流程

##### Phase A：多任务训练（一次性，生成通用策略文件）

**Stage-1 通用策略训练**

```python
from general_policy_module import (
    prepare_stage1_task, multi_task_train_stage1,
)

# 为每个任务准备 task config
tasks = {}
for name in ["mrpc", "stsb", "cola", "rte"]:
    ev = create_evaluator(name)  # 你自己的 evaluator 创建逻辑
    tasks[name] = prepare_stage1_task(ev)

# 多任务 round-robin 训练
result = multi_task_train_stage1(
    tasks,
    output_path="general_stage1_policy.pt",
    total_rounds=50,                    # round-robin 轮数
    episodes_per_task_per_round=170,    # 每轮每任务的 episode 数
    lr=3e-5,                            # 学习率
    device="cuda",
)
# 产出：general_stage1_policy.pt
```

**Stage-2 通用噪声策略训练**

```python
from general_policy_module import (
    prepare_stage2_task, multi_task_train_stage2,
)

tasks = {}
for name in ["mrpc", "stsb", "cola", "rte"]:
    ev = create_evaluator(name)
    fixed_gelu, fixed_softmax = get_stage1_config(name)  # 该任务的 Stage-1 确定配置
    tasks[name] = prepare_stage2_task(ev, fixed_gelu, fixed_softmax)

result = multi_task_train_stage2(
    tasks,
    output_path="general_stage2_noise_policy.pt",
    total_rounds=50,
    episodes_per_task_per_round=170,
    lr=3e-5,
    device="cuda",
)
# 产出：general_stage2_noise_policy.pt
```

`multi_task_train_stage1` / `multi_task_train_stage2` 的关键参数：

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `tasks` | dict，键为任务名，值由 `prepare_stage1_task` / `prepare_stage2_task` 产出 | — |
| `output_path` | 保存通用策略文件的路径 | — |
| `total_rounds` | round-robin 总轮数 | `50` |
| `episodes_per_task_per_round` | 每轮每任务采集的 episode 数 | `170` |
| `lr` | Adam 优化器学习率 | `3e-5` |
| `device` | PyTorch 设备 | `"cuda"` |

##### Phase B：离线部署（在新 / 旧任务上做推断，不训练）

**Stage-1 离线 rollout 找最优配置**

```python
from general_policy_module import offline_find_best_config_stage1

ev_new = create_evaluator("qqp")
result = offline_find_best_config_stage1(
    ev_new,
    general_policy_path="general_stage1_policy.pt",
    num_rollouts=500,    # rollout 次数（greedy=True 时自动置 1）
    greedy=False,        # 是否使用 argmax 而非采样
    device="cuda",
)
print(result["best_config"])     # {"gelu": [...], "softmax": [...], "cost": ...}
print(result["best_reward"])
```

**Stage-2 离线 rollout 找最优噪声配置**

```python
from general_policy_module import offline_find_best_config_stage2

result = offline_find_best_config_stage2(
    ev_new,
    general_policy_path="general_stage2_noise_policy.pt",
    fixed_gelu=fixed_gelu,
    fixed_softmax=fixed_softmax,
    noise_env=noise_env,             # 已创建的 _NoiseOptEnv 实例
    num_rollouts=500,
    greedy=False,
    device="cuda",
)
print(result["best_noise_config"])
```

离线推断关键参数：

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `general_policy_path` | 通用策略文件路径 | — |
| `num_rollouts` | rollout 次数 | `500` |
| `greedy` | 是否贪心（argmax），`True` 时只做 1 次 rollout | `False` |
| `device` | PyTorch 设备 | `"cuda"` |

##### Phase C：作为 base policy 微调（利用已有的 online RL）

通用策略文件与 per-task 便携 policy 文件格式兼容，因此也可以直接作为 base policy 加载到现有 online RL 流程中微调：

```python
# 编辑 layer_importance_evaluator.py 顶部
RL_OPT_FLAGS = {
    ...
    "stage1_pretrained_policy_path": "general_stage1_policy.pt",
    ...
}

# 编辑 noise_rl_module_v2.py 顶部
NOISE_RL_OPT_FLAGS = {
    ...
    "pretrained_policy_path": "general_stage2_noise_policy.pt",
    ...
}
```

然后正常运行 `bash llama_7B_LayerImportance.sh ...` 即可。新 run 会从通用策略的权重开始微调，通常能显著缩短到达高 reward 平台的回合数。

> **注意**：加载时使用 `strict=False`，`task_context_proj` 层在 per-task 网络中会被跳过（`unexpected`），不影响推理与后续训练。

#### 通用 Critic 快速评分

通用 Critic 可以在不做真实模型评测的前提下，仅用 V(s) 对一组候选配置快速评分、排序：

```python
from general_policy_module import critic_quick_rank_stage1

candidates = [
    {"gelu": [1,1,1,4,1,1,1,1,1,1,1,1], "softmax": [2,3,4,6,4,4,5,4,4,5,5,2]},
    {"gelu": [4,4,4,4,4,4,4,4,4,4,4,4], "softmax": [6,6,6,6,6,6,6,6,6,6,6,6]},
    # ... 更多候选
]

# 返回 [(score, config), ...] 按 score 降序
ranked = critic_quick_rank_stage1(
    evaluator,
    general_policy_path="general_stage1_policy.pt",
    candidate_configs=candidates,
    device="cuda",
)
for score, cfg in ranked[:5]:
    print(f"  score={score:.4f}  gelu={cfg['gelu']}  softmax={cfg['softmax']}")
```

适用场景：

- 预筛选大量候选配置（例如 1000+ 个），再对 top-K 做真实评测
- 纯 GPU 推理，速度比真实评测快两个数量级
- 精度取决于 Critic 训练质量，建议与真实评测配合使用

#### 保存文件格式

通用策略文件（`torch.save` 的 dict）：

```python
{
    "version": 1,
    "kind": "general_stage1_gtrxl_policy" | "general_stage2_noise_gtrxl_policy",
    "net_state_dict": <state_dict>,               # actor + critic 权重
    "arch": {
        "num_layers": 12, "d_model": ..., "n_heads": ...,
        "n_gtrxl_layers": ..., "d_ff": ..., "dropout": ...,
        "task_context_dim": 5,
    },
    "metadata": {
        "task_names": ["mrpc", "stsb", "cola", "rte"],
        "total_rounds": 50,
        "episodes_per_task_per_round": 170,
        "total_episodes": 34000,
        "ppo_updates": 200,
        "best_rewards": {"mrpc": 1.23, "stsb": 0.98, ...},
    },
}
```

#### 与 per-task 便携 policy 的兼容性

| 场景 | 兼容性 | 备注 |
| --- | --- | --- |
| 通用策略 → per-task online RL（作为 base policy） | ✅ | `task_context_proj` 层被 `strict=False` 跳过，不影响运行 |
| per-task 便携 policy → 通用策略离线推断 | ✅ | 缺失的 `task_context_proj` 保持零初始化 |
| 不同 `total_layers`（如 12 层 → 24 层） | ⚠️ 部分迁移 | layer embedding 不匹配的层会被跳过 |
| Stage-1 ↔ Stage-2 | ❌ | 动作空间不同，必须分别训练 |

#### 推荐工作流

1. **多任务训练**：选 4-7 个 GLUE 任务，用 `multi_task_train_stage1` / `multi_task_train_stage2` 分别训练通用 Stage-1 和 Stage-2 策略。
2. **离线快速部署**：新任务到来时，用 `offline_find_best_config_stage1` / `offline_find_best_config_stage2` 做 500 次 rollout 快速找到最优配置，无需数万回合 online RL。
3. **精细化微调**：如果离线结果不够理想，可将通用策略作为 base policy 加载到 online RL 中继续微调，通常只需原来 1/5 ~ 1/3 的回合数即可收敛。
4. **Critic 预筛选**：如果有大量候选配置需要评估，先用 `critic_quick_rank_stage1` 快速排序，再对 top-K 做真实评测。
