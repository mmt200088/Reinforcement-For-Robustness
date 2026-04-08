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

| `--model-type` 值 | 预训练 checkpoint 系列                                | 层数 |
| ---------------- | --------------------------------------------------- | ---- |
| `bert-base`      | `textattack/bert-base-uncased-*`                    | 12   |
| `bert-large`     | `yoshitomo-matsubara/bert-large-uncased-*`          | 24   |
| `gpt-2`          | `openai-community/gpt2`（所有任务共用同一个基座）      | 12   |

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
`openai-community/gpt2` 骨干（12 层 transformer，768 hidden size）。
该分支在模块路径、QKV 融合、激活函数替换等层面与 BERT 做了专门适配，
`rl_tune.py` / `layer_importance_evaluator.py` / `noise_rl_module_v2.py`
/ `final_evaluation_module.py` / `generate_glue_submission.py` 无需手动
切换，按 `--model-type gpt-2` 一处开关即可。

基座与 checkpoint 来源：

- GPT-2 在 HuggingFace 上没有覆盖全部 GLUE 任务的权威微调系列（不像
  `textattack/bert-base-uncased-*`），因此此处**所有任务共用同一个
  预训练基座 `openai-community/gpt2`**；`AutoModelForSequenceClassification`
  会给每个任务随机初始化一个分类 head，并由 `rl_tune.py` 的训练循环
  自行完成 head 微调。首次在某个 GLUE 任务上使用前，建议让脚本自然走
  完 fine-tune 阶段（不要带 `--skip-stage1-rl`/`--skip-noise-rl`），否则
  评估得到的是"随机分类 head"的结果。
- Tokenizer 在 `rl_tune.py` 中已统一执行
  `tokenizer.pad_token = tokenizer.eos_token`，并在加载模型时传入
  `pad_token_id=tokenizer.pad_token_id`，满足 GPT-2
  `GPT2ForSequenceClassification` 要求的"末 token pooling + 必须有
  pad token"约束。

功能兼容范围：

| 阶段 / 功能                         | bert-base | bert-large | gpt-2 |
| ----------------------------------- | :-------: | :--------: | :---: |
| Stage 1 GELU 多项式近似             | ✅        | ✅         | ✅    |
| Stage 1 Softmax 指数近似            | ✅        | ✅         | ❌ (自动跳过) |
| Stage 2 x / Wo / Wffn1 / Wffn2 噪声 | ✅        | ✅         | ✅    |
| Stage 2 Wq / Wk / Wv 噪声           | ✅        | ✅         | ✅（通过融合 c_attn 的按槽位加噪实现） |
| 最终评估 (`final_evaluation_module`) | ✅        | ✅         | ✅    |
| GLUE 提交文件生成                   | ✅        | ✅         | ✅（分类 head 需先微调） |

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
| `--final-eval-source search    | json                                                                                | manual`                      |
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
| `--noise-eval-source search               | json                          | manual`                            |
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

`CUDA_VISIBLE_DEVICES=0 bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 --skip-stage1-rl --final-eval-source json --final-eval-config glue_configs_best_ppo.json --skip-stage1-final-eval --skip-noise-rl --noise-eval-source json --noise-eval-config glue_noise_configs_best_ppo.json --noise-eval-repeat 200 --model mrpc --batch_size 128`

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

使用json文件生成glue官网提交测试文件
`python generate_glue_submission.py --config glue_configs_best_ppo.json --noise_config glue_noise_configs_best_ppo.json`

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

当一次强化学习训练完成后，如果发现轮数不够，可以通过 `--resume-from` 指定之前的 run 目录，在之前训练的基础上继续训练更多轮次。效果等价于一次性训练更多轮（例如先训 30000 轮，再续训 10000 轮，等价于一次性训练 40000 轮）。

训练过程中会自动在 run 目录下保存 checkpoint 文件（每次 PPO 更新窗口结束时保存）：

- Stage-1 checkpoint: `<run_dir>/stage1/stage1_rl_checkpoint.pt`
- Stage-2 checkpoint: `<run_dir>/stage2_noise/progress/noise_rl_checkpoint.pt`


| 参数                    | 说明                                                                                                                                         | 默认值 |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | --- |
| `--resume-from PATH`  | 指定之前的 run 目录路径，从该目录的 checkpoint 恢复训练。程序会自动在 `<PATH>/stage1/` 和 `<PATH>/stage2_noise/progress/` 下查找 checkpoint 文件。如果 checkpoint 不存在，则从头开始训练。 | 空   |


使用说明：

- `PATH` 必须是一个已存在的 run 目录（例如 `rl_results/layer_importance_runs/mrpc/20260404_151155_pid711833`）。
- 续训时，`--stage1-rl-episodes` / `--stage2-rl-episodes` 表示的是**总轮数**（而非追加轮数）。例如之前训了 30000 轮，想再加 10000 轮，则设置 `--stage2-rl-episodes 40000`。
- 如果指定的总轮数小于等于 checkpoint 中已完成的轮数，则该阶段不会追加训练。
- `--resume-from` 可以与 `--skip-stage1-rl`、`--skip-noise-rl` 等跳过选项组合使用：只有未被跳过的阶段才会尝试加载对应的 checkpoint。
- 续训产出的新日志和文件会写入新生成的 run 目录（不会覆盖原目录），但模型状态和训练统计会从旧 checkpoint 恢复。
- checkpoint 会在每次进度快照时自动保存，因此即使训练中途被中断，也可以从最近的 checkpoint 恢复。

示例：

```bash
# 第一次训练：Stage-2 训练 15000 轮
CUDA_VISIBLE_DEVICES=0 bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
  --skip-stage1-rl \
  --final-eval-source json \
  --final-eval-config glue_configs_best_ppo.json \
  --skip-stage1-final-eval \
  --noise-eval-repeat 200 \
  --model mrpc \
  --stage2-rl-episodes 15000

# 发现轮数不够，续训到 30000 轮（在之前 15000 轮的基础上再训 15000 轮）
# 这里的 PATH 填第一次训练生成的 run 目录
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

# 续训 Stage-1 到 20000 轮
bash llama_7B_LayerImportance.sh 32 64 output.log 20 2 \
  --model mrpc \
  --stage1-rl-episodes 20000 \
  --skip-noise-rl --skip-noise-final-eval \
  --resume-from rl_results/layer_importance_runs/mrpc/<之前的run目录>
```

