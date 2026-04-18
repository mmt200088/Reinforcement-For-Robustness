## 快速开始（预设系统）

最简单的使用方式是通过预设（preset）启动：

```bash
# 列出所有可用预设
bash llama_7B_LayerImportance.sh --list-presets

# 首次运行（必须加 --fresh-start）
bash llama_7B_LayerImportance.sh --preset mrpc-rl-default --fresh-start

# 续训练（自动从 checkpoint 恢复，无需额外参数）
bash llama_7B_LayerImportance.sh --preset mrpc-rl-default

# 预设 + 自定义覆盖（命令行参数优先于预设）
bash llama_7B_LayerImportance.sh --preset mrpc-rl-default --stage2-search-episodes 60000
```

预设文件位于 `presets/` 目录下，格式为每行一个命令行参数（支持 `#` 注释）。可自行添加新预设。

## 命令行参数总表

```bash
bash llama_7B_LayerImportance.sh [可选参数]
```

### 说明

- 现在**不再支持位置参数**，统一改为可选参数。
- `--model` 已废弃，请改用 `--dataset`。
- `lora_r`、`lora_alpha`、`degree` 已从命令行入口移除，因为当前流程不会实际读取它们。
- 本节是当前命令行入口的**最新说明**。如果下文个别历史实验片段与本节不一致，以本节为准。

### 可选参数总表

| 参数 | 适用模式 | 默认值 | 说明 |
| --- | --- | --- | --- |
| **全局参数** | | | |
| `--dataset DATASET` | 全局 | `mrpc` | 数据集名称：`mrpc`、`sst2`、`stsb`、`cola`、`qnli`、`rte`、`wnli` |
| `--search-algorithm ALG` | 全局 | `rl` | 搜索算法：`rl` / `ga` / `general-rl` / `rl-and-ga-compare` |
| `--logfile FILE` | 全局 | `output.log` | launcher 的 nohup 日志文件名；真实运行目录下也会自动生成阶段日志 |
| `--model-type TYPE` | 全局 | `bert-base` | 骨干模型类型：`bert-base` / `bert-large` / `gpt-2` |
| `--batch-size N` | 全局 | `16` | 统一设置 `batch_size` 与 `micro_batch_size` |
| `--resume-from PATH` | 兼容保留 | — | 兼容保留的内部恢复参数；当前 launcher 的正式续训流程已改为按持久化目录自动恢复，普通命令行不建议手动传 |
| **准确度约束参数** | | | |
| `--stage1-accuracy-tolerance FLOAT` | `rl`、`ga`、`rl-and-ga-compare` | `0.005` | Stage-1 指标约束百分比。0.005 表示允许 loss 上浮 0.5%、指标下降 0.5% |
| `--stage2-limit-tolerance FLOAT` | `rl`、`ga`、`rl-and-ga-compare` | `0.05` | Stage-2 指标约束百分比（以 baseline 为基准，与 `--stage1-accuracy-tolerance` 同构）。0.05 表示允许 loss 上浮 5%、metric1/metric2 下降 5% |
| `--stage2-stability-tolerance FLOAT` | `rl`、`ga`、`rl-and-ga-compare` | `0.05` | Stage-2 稳定性约束百分比（以 baseline 探针的纯噪声采样 std 为基准）。0.05 表示允许 std 上浮 5% |
| `--stage2-k-trials INT` | `rl`、`ga`、`rl-and-ga-compare` | `5` | Stage-2 稳定性评测噪声试验次数 K。每次评测在同一份固定分层探针上跑 K 个独立噪声种子，std 反映纯噪声采样方差（不进入持久化目录 slug，仅作为采样预算） |
| `--stage2-probe-size INT` | `rl`、`ga`、`rl-and-ga-compare` | `256` | Stage-2 稳定性评测探针子集大小。用分层采样从验证集中抽取一份固定子集，K 次 trial 共用同一份数据；默认 K×probe = 5×256 = 1280 次前向 |
| **持久化与续训练** | | | |
| `--fresh-start` | `rl`、`ga`、`general-rl` 训练 | — | 清空当前参数组合对应的整个持久化目录并从头开始；首次运行某参数组合时**必须指定**，否则报错 |
| `--fresh-stage1` | `rl`、`ga` | — | 仅清空已有持久化目录中的 `stage1/` 与 `stage1_final_eval/`，保留 Stage-2；仅适用于已有持久化目录的续训场景 |
| `--fresh-stage2` | `rl`、`ga` | — | 仅清空已有持久化目录中的 `stage2_noise/` 与 `stage2_noise_final_eval/`，保留 Stage-1；仅适用于已有持久化目录的续训场景 |
| **普通 RL / GA 搜索预算** | | | |
| `--stage1-search-episodes N` | `rl` | `51000` | Stage-1 搜索回合数，仅用于普通 RL |
| `--stage2-search-episodes N` | `rl` | `40000` | Stage-2 噪声搜索回合数，仅用于普通 RL |
| `--ppo-update-interval N` | `rl`、`general-rl` | `120` | PPO 更新间隔（每多少个 episode 触发一次策略更新）；同时决定 `details/` 下每个 txt 的回合数（= `3 × N`，默认 `360`）。`general-rl` 训练模式下还等同于"每轮每任务的 episode 数"。必须 ≥ 该值才能完成至少一次 PPO 更新 |
| `--stage1-search-generations N` | `ga` | 按模型自动推导 | Stage-1 GA 搜索迭代代数；仅在未跳过 Stage-1 搜索时生效 |
| `--stage2-search-generations N` | `ga` | 按模型自动推导 | Stage-2 GA 噪声搜索迭代代数；仅在未跳过 Stage-2 搜索时生效 |
| `--skip-stage1-search` | `rl`、`ga` | — | 跳过 Stage-1 搜索 |
| `--skip-noise-search` | `rl`、`ga` | — | 跳过 Stage-2 搜索 |
| `--skip-stage1-final-eval` | `rl`、`ga` | — | 跳过 Stage-1 最终评估 |
| `--skip-noise-final-eval` | `rl`、`ga` | — | 跳过 Stage-2 最终评估 |
| `--final-eval-source search/json/manual` | `rl`、`ga` | `search` | Stage-1 最终评估配置来源 |
| `--final-eval-config PATH` | `rl`、`ga` | 自动 | `json` 模式下的 Stage-1 最终评估配置文件路径 |
| `--manual-gelu JSON_ARRAY` | `rl`、`ga` | — | `manual` 模式下的 Stage-1 GELU 配置 |
| `--manual-softmax JSON_ARRAY` | `rl`、`ga` | — | `manual` 模式下的 Stage-1 Softmax 配置 |
| `--stage2-fixed-config-source stage1_result/json/manual` | `rl`、`ga` | 兼容继承 Stage-1 参数 | Stage-2 固定 GELU/Softmax 的来源；`stage1_result` 表示直接使用 Stage-1 搜索结果，不再借用 Stage-1 final eval 参数表达 |
| `--stage2-fixed-config PATH` | `rl`、`ga` | 自动 | `json` 模式下的 Stage-2 固定 GELU/Softmax 配置文件路径 |
| `--stage2-manual-gelu JSON_ARRAY` | `rl`、`ga` | — | `manual` 模式下的 Stage-2 固定 GELU 配置 |
| `--stage2-manual-softmax JSON_ARRAY` | `rl`、`ga` | — | `manual` 模式下的 Stage-2 固定 Softmax 配置 |
| `--noise-eval-source search/json/manual` | `rl`、`ga` | `search` | Stage-2 噪声最终评估配置来源 |
| `--noise-eval-config PATH` | `rl`、`ga` | 自动 | `json` 模式下的 Stage-2 噪声配置文件路径 |
| `--manual-noise-config JSON_OBJECT` | `rl`、`ga` | — | `manual` 模式下的 7 类噪声配置 |
| `--noise-eval-repeat N` | `rl`、`ga` | `1` | Stage-2 最终评估重复次数 |
| `--stage2-compare-repeats N` | `rl-and-ga-compare` | `1` | `rl-and-ga-compare` 唯一正式的 Stage-2 多次对比次数入口；会让 RL/GA 两侧都重复评估，并在报告中汇总均值、标准差、方差、最大值、最小值 |
| `--random-seed N` | `rl`、`ga`、`rl-and-ga-compare` | `42` | 随机种子 |
| `--perm-trials N` | `rl`、`ga`、`rl-and-ga-compare` | `10` | 随机置换对照次数 |
| `--cost-trials N` | `rl`、`ga`、`rl-and-ga-compare` | `10` | 等价成本对照次数 |
| `--budget-trials N` | `rl`、`ga`、`rl-and-ga-compare` | `10` | 等价预算对照次数 |
| **普通 RL 专用** | | | |
| `--stage1-search-lr FLOAT` | `rl` | `1e-4` | 普通 RL 的 Stage-1 学习率 |
| `--stage2-search-lr FLOAT` | `rl` | `1e-4` | 普通 RL 的 Stage-2 学习率 |
| **对比实验专用** | | | |
| `--compare-config-mode direct/persistent` | `rl-and-ga-compare` | `direct` | 对比配置来源：`direct`=显式指定 4 个 JSON；`persistent`=按数据集/模型/约束从持久化目录自动寻找 |
| `--compare-persistent-root PATH` | `rl-and-ga-compare` | `rl_results/persistent` | `persistent` 模式下的持久化目录根路径；启动前会检查推导出的 RL / GA 目标目录是否存在 |
| `--rl-compare-stage1-json PATH` | `rl-and-ga-compare` | — | `direct` 模式下 RL 的 Stage-1 JSON；可传配置模板，也可传最终评估结果 JSON |
| `--rl-compare-stage2-json PATH` | `rl-and-ga-compare` | — | `direct` 模式下 RL 的 Stage-2 JSON；可传配置模板，也可传最终评估结果 JSON |
| `--ga-compare-stage1-json PATH` | `rl-and-ga-compare` | — | `direct` 模式下 GA 的 Stage-1 JSON；可传配置模板，也可传最终评估结果 JSON |
| `--ga-compare-stage2-json PATH` | `rl-and-ga-compare` | — | `direct` 模式下 GA 的 Stage-2 JSON；可传配置模板，也可传最终评估结果 JSON |
| `--rl-compare-stage1-accuracy-tolerance FLOAT` | `rl-and-ga-compare` | 继承 `--stage1-accuracy-tolerance` 或 `0.005` | `persistent` 模式下 RL 侧的 Stage-1 约束，用于定位 RL 持久化目录 |
| `--rl-compare-stage2-limit-tolerance FLOAT` | `rl-and-ga-compare` | 继承 `--stage2-limit-tolerance` 或 `0.05` | `persistent` 模式下 RL 侧的 Stage-2 指标约束，用于定位 RL 持久化目录 |
| `--rl-compare-stage2-stability-tolerance FLOAT` | `rl-and-ga-compare` | 继承 `--stage2-stability-tolerance` 或 `0.05` | `persistent` 模式下 RL 侧的 Stage-2 稳定性约束，用于定位 RL 持久化目录 |
| `--ga-compare-stage1-accuracy-tolerance FLOAT` | `rl-and-ga-compare` | 继承 `--stage1-accuracy-tolerance` 或 `0.005` | `persistent` 模式下 GA 侧的 Stage-1 约束，用于定位 GA 持久化目录 |
| `--ga-compare-stage2-limit-tolerance FLOAT` | `rl-and-ga-compare` | 继承 `--stage2-limit-tolerance` 或 `0.05` | `persistent` 模式下 GA 侧的 Stage-2 指标约束，用于定位 GA 持久化目录 |
| `--ga-compare-stage2-stability-tolerance FLOAT` | `rl-and-ga-compare` | 继承 `--stage2-stability-tolerance` 或 `0.05` | `persistent` 模式下 GA 侧的 Stage-2 稳定性约束，用于定位 GA 持久化目录 |
| **通用 RL 专用** | | | |
| `--general-rl-mode train/search` | `general-rl` | `search` | 通用 RL 的运行模式；`search` 为正式名称，`infer` 仍保留为兼容别名 |
| `--general-rl-tasks T1,T2,...` | `general-rl` 训练 | 同 `--dataset` | 逗号分隔的训练任务列表 |
| `--general-rl-rounds N` | `general-rl` 训练 | `50` | Round-robin 训练轮数 |
| `--general-rl-lr FLOAT` | `general-rl` 训练 | `3e-5` | 通用策略训练学习率 |
| `--general-rl-num-rollouts N` | `general-rl` 搜索 | `500` | 离线 rollout 次数 |
| `--general-rl-greedy` | `general-rl` 搜索 | — | 使用贪心 rollout |
| `--general-stage1-policy PATH` | `general-rl` 搜索 | — | Stage-1 通用策略文件；可显式指定，或由 `--general-policy-dir` 自动推导 |
| `--general-stage2-policy PATH` | `general-rl` 搜索 | — | Stage-2 通用噪声策略文件，可选；也可由 `--general-policy-dir` 自动推导 |
| `--general-policy-dir PATH` | `general-rl` 搜索 | — | 指向一个已训练好的通用 RL 持久化目录；launcher 会自动寻找 `general_stage1_policy.pt`，若存在也会自动带上 `general_stage2_noise_policy.pt` |
| `--general-rl-skip-stage2` | `general-rl` | — | 跳过 Stage-2 训练或搜索 |
| `--general-rl-stage1-config-json PATH` | `general-rl` 训练 | — | Stage-2 训练时各任务的 Stage-1 配置 |
| `--general-rl-accuracy-tolerances T1,T2,...` | `general-rl` | — | 逗号分隔的准确度容忍比例列表（如 `0.005,0.01,0.02`）；训练时每轮随机采样一个 tolerance 让策略泛化到不同准确度要求，搜索时取第一个值作为目标 tolerance |
| `--general-rl-accuracy-tolerance-range MIN,MAX` | `general-rl` 训练 | — | 连续准确度容忍区间；训练时在 `[MIN, MAX]` 内采样 tolerance 让策略泛化，要求 `0 < MIN < MAX < 1` |

### 搜索算法与实际入口

| `--search-algorithm` | 含义 | 实际入口 |
| --- | --- | --- |
| `rl` | Per-task 两阶段普通 RL 搜索 | `rl_tune.py` |
| `ga` | COINN 风格两阶段遗传算法搜索 | `rl_tune_genetic.py` |
| `general-rl` | 多任务通用策略训练 / 离线搜索 | `rl_tune_general.py` |
| `rl-and-ga-compare` | 不再重新训练 RL/GA；而是直接读取显式 JSON 或持久化目录中的已有结果，并生成 Stage-1 / Stage-2 对比报告 | `rl_ga_compare_runner.py` |

### 模式互斥规则

- 选择 `--search-algorithm=general-rl` 后，不能再混用普通 RL / GA 的阶段搜索、Stage-2 固定 GELU/Softmax 参数或最终评估参数；`search` 模式必须提供 `--general-stage1-policy` 或 `--general-policy-dir`。`infer` 仅作为 `search` 的兼容别名保留。
- 选择 `--search-algorithm=rl` 或 `ga` 后，不能再混用 `--general-rl-*` 参数；也**不能**再传 `--resume-from`（已改用持久化目录自动续训练）。
- 选择 `--search-algorithm=rl` 后，不能传 `--stage1-search-generations` / `--stage2-search-generations`。
- 选择 `--search-algorithm=ga` 后，不能传 `--stage1-search-lr` / `--stage2-search-lr`，也不能再传 `--stage1-search-episodes` / `--stage2-search-episodes`。
- 选择 `--search-algorithm=ga` 后，如果已经传了 `--skip-stage1-search` 或 `--skip-noise-search`，就不能再显式传对应阶段的 `--stage1-search-generations` / `--stage2-search-generations`；被跳过阶段的默认预算会自动忽略，不需要手动补 0。
- 选择 `--search-algorithm=rl-and-ga-compare` 后：
  - 仍然**不允许**使用全局的 `--skip-stage1-search`、`--skip-noise-search`、`--final-eval-source`、`--stage2-fixed-config-*`、`--noise-eval-source`、`--final-eval-config`、`--noise-eval-config`、`--manual-*`、`--resume-from`。
  - 也不再支持旧的 compare 专用参数：`--rl/ga-skip-*`、`--rl/ga-*-eval-source`、`--rl/ga-*-eval-config`。
  - `direct` 模式必须同时提供 4 个 JSON：RL/GA 各自的 Stage-1 与 Stage-2。
  - `persistent` 模式必须提供 `--compare-persistent-root`；RL 与 GA 可以使用不同约束参数，但模型类型与数据集必须一致。
  - `persistent` 模式下，如未显式提供 `--rl/ga-compare-*` 约束参数，会分别继承全局的 `--stage1-accuracy-tolerance`、`--stage2-limit-tolerance`、`--stage2-stability-tolerance`。
  - `persistent` 模式启动前会先推导出 RL / GA 各自的目标持久化目录；如果目录或其中的 `metadata.json` 不存在，会直接报错并打印具体路径。
  - `rl-and-ga-compare` 模式下，Stage-2 多次对比只看 `--stage2-compare-repeats`。
  - `--noise-eval-repeat` 不再作为 compare 模式的正式参数；仅保留兼容别名行为。若旧命令只传了它而没传 `--stage2-compare-repeats`，系统会临时按 `--stage2-compare-repeats` 处理并给出警告。
  - 对比实验始终保留 Stage-1 / Stage-2 最终评估，因此依然不支持 `--skip-stage1-final-eval` 与 `--skip-noise-final-eval`。

### JSON 配置文件默认值

| 算法 | Stage-1 JSON 默认值 | Stage-2 JSON 默认值 |
| --- | --- | --- |
| `rl` | `glue_configs_best_ppo.json` | `glue_noise_configs_best_ppo.json` |
| `ga` | `glue_configs_best_genetic.json` | `glue_noise_configs_best_genetic.json` |

当 `--final-eval-source=json` 或 `--noise-eval-source=json` 时，脚本会检查文件名是否与所选算法家族匹配，避免把 PPO 配置错用到 GA，或把 GA 配置错用到 PPO。当前仓库已内置 PPO 与 GA 两套默认 JSON；其中 `glue_configs_best_genetic.json` / `glue_noise_configs_best_genetic.json` 目前按 PPO 默认配置生成，便于在 `ga` 或 `rl-and-ga-compare` 模式下直接跳过搜索。如果你后续得到 GA 自己的更优配置，可以直接覆盖这两个 genetic JSON，或者显式通过 `--final-eval-config` / `--noise-eval-config` 指定。
`--stage2-fixed-config-source=json` 时，`--stage2-fixed-config` 默认也会沿用这一套 Stage-1 JSON 默认值；如果你不显式传 `--stage2-fixed-config-*`，脚本会继续兼容旧命令，把 Stage-1 final eval 的来源自动映射为 Stage-2 固定 `GELU/Softmax` 的来源。
`rl-and-ga-compare` 的 `direct` 模式同样沿用这套家族一致性检查；如果显式传入的是最终评估结果 JSON，则还会额外校验其中记录的数据集与模型层数，避免把不同模型或不同数据集的结果拿来直接对比。

### 推荐命令示例

#### 1. 默认普通 RL

```bash
bash llama_7B_LayerImportance.sh --dataset mrpc
```

#### 2. 普通 RL，分别指定 Stage-1 / Stage-2 学习率

```bash
bash llama_7B_LayerImportance.sh \
  --dataset mrpc \
  --search-algorithm rl \
  --stage1-search-lr 3e-5 \
  --stage2-search-lr 1e-5
```

#### 3. 完整 GA

```bash
bash llama_7B_LayerImportance.sh \
  --dataset mrpc \
  --search-algorithm ga \
  --fresh-start
```

#### 3b. 自定义准确度约束

通过 `--stage1-accuracy-tolerance`、`--stage2-limit-tolerance`、`--stage2-stability-tolerance` 调整搜索约束：

```bash
# Stage-1 指标容忍度放宽到 1%，Stage-2 约束放宽到 10% 波动
bash llama_7B_LayerImportance.sh \
  --dataset mrpc \
  --fresh-start \
  --stage1-accuracy-tolerance 0.01 \
  --stage2-limit-tolerance 0.1 \
  --stage2-stability-tolerance 0.1
```

参数含义：

- `--stage1-accuracy-tolerance 0.01`：Stage-1 允许 loss 上浮 1%、主指标下降 1%（默认 0.005 = 0.5%）。
- `--stage2-limit-tolerance 0.1`：Stage-2 指标约束以 baseline 为基准，允许 loss 上浮 10%、metric1/metric2 下降 10%（默认 0.05 = 5%）。
- `--stage2-stability-tolerance 0.1`：Stage-2 稳定性约束以 baseline 探针 std 为基准，允许 std 上浮 10%（默认 0.05 = 5%）。

三个参数值越小，搜索越保守（更贴近 baseline）；越大，允许波动越多，搜索空间越大。

#### 3c. 持久化目录与自动续训练（rl / ga）

`rl` 和 `ga` 模式采用**持久化目录**方案：相同的 `(算法, 模型, 数据集, 约束参数)` 组合映射到唯一确定性目录，后续相同参数运行自动从 checkpoint 续训练，无需手动指定 `--resume-from`。

```bash
# 首次运行：必须加 --fresh-start
bash llama_7B_LayerImportance.sh \
  --dataset mrpc \
  --fresh-start \
  --stage2-search-episodes 15000

# 发现轮数不够，直接加大预算再跑（不加 --fresh-start → 自动续训练）
bash llama_7B_LayerImportance.sh \
  --dataset mrpc \
  --stage2-search-episodes 30000

# 换一组约束参数 → 新的持久化目录，需要再次 --fresh-start
bash llama_7B_LayerImportance.sh \
  --dataset mrpc \
  --fresh-start \
  --stage1-accuracy-tolerance 0.01 \
  --stage2-search-episodes 15000
```

持久化目录路径格式：

```text
rl_results/persistent/<algorithm>/<model_type>/<dataset>/s1t<T>_s2t<L>_s2st<S>/
```

示例：

```text
rl_results/persistent/rl/bert-base/mrpc/s1t0.005_s2t0.05_s2st0.05/
rl_results/persistent/ga/bert-large/stsb/s1t0.01_s2t0.1_s2st0.1/
```

目录下的 `metadata.json` 记录算法、模型、数据集、约束参数值、创建时间和运行次数。

补充说明：

- `ga` 现在和 `rl` 一样支持单独跳过 Stage-1 或 Stage-2 搜索。
- 如果某一阶段被跳过，该阶段的默认 GA 代数只会保留在内部兼容逻辑里，不会再被当成“你显式指定了预算”。
- 但如果你已经传了 `--skip-stage1-search`，就不要再同时传 `--stage1-search-generations`；`--skip-noise-search` 与 `--stage2-search-generations` 同理。

示例：只跑 GA 的 Stage-2，Stage-1 结构配置从 JSON 读取。

```bash
bash llama_7B_LayerImportance.sh \
  --dataset mrpc \
  --search-algorithm ga \
  --fresh-start \
  --skip-stage1-search \
  --skip-stage1-final-eval \
  --stage2-search-generations 2500 \
  --final-eval-source json \
  --final-eval-config glue_configs_best_genetic.json \
  --stage2-fixed-config-source json \
  --stage2-fixed-config glue_configs_best_genetic.json
```

#### 3d. 独立指定 Stage-2 固定 GELU / Softmax 来源

现在 Stage-2 固定的 `GELU/Softmax` 不再依赖 `--final-eval-source` 的语义来间接表达，而是用单独的一组参数控制：

- `--stage2-fixed-config-source stage1_result`：直接使用 Stage-1 搜索结果
- `--stage2-fixed-config-source json`：从 JSON 读取固定的 Stage-1 配置
- `--stage2-fixed-config-source manual`：手动传入 `--stage2-manual-gelu` 与 `--stage2-manual-softmax`

兼容行为：

- 如果你**没有显式传** `--stage2-fixed-config-*`，脚本会按旧行为自动兼容：
  - `--final-eval-source=search` 会映射为 `--stage2-fixed-config-source=stage1_result`
  - `--final-eval-source=json/manual` 会把对应配置继承给 Stage-2 固定配置
- 如果你只传了 `--stage2-fixed-config`，launcher 会自动把 `--stage2-fixed-config-source` 视为 `json`；如果你只传了 `--stage2-manual-gelu` / `--stage2-manual-softmax`，launcher 会自动把来源视为 `manual`。

安全规则：

- 只要 Stage-2 搜索或 Stage-2 最终评估会执行，就必须能解析出 Stage-2 固定的 `GELU/Softmax`。
- 如果使用 `--stage2-fixed-config-source=stage1_result`，且本次同时传了 `--skip-stage1-search`，那么当前持久化目录中必须已经存在历史 Stage-1 搜索结果；否则 launcher 会直接报错，并要求改用 `json` / `manual`，或先运行一次 Stage-1。
- 如果使用 `json`，launcher 会在启动前检查 JSON 文件是否存在，并校验它与当前算法家族一致。
- 如果使用 `manual`，必须同时提供 `--stage2-manual-gelu` 和 `--stage2-manual-softmax`。

示例 1：Stage-2 固定配置来自 Stage-1 搜索结果。

```bash
bash llama_7B_LayerImportance.sh \
  --dataset mrpc \
  --search-algorithm rl \
  --stage2-fixed-config-source stage1_result
```

示例 2：跳过 Stage-1 搜索，直接用 JSON 指定 Stage-2 固定配置。

```bash
bash llama_7B_LayerImportance.sh \
  --dataset mrpc \
  --search-algorithm rl \
  --skip-stage1-search \
  --skip-stage1-final-eval \
  --stage2-fixed-config-source json \
  --stage2-fixed-config glue_configs_best_ppo.json
```

#### 4. RL 与 GA 对比实验（直接读取已有结果）

`rl-and-ga-compare` 已经不再重新启动一份 RL 和一份 GA 完整训练；现在只负责把**已有 JSON** 或 **持久化目录中的已有结果**整理成统一的对比报告。

方式 1：`direct`，显式指定 4 个 JSON。

```bash
bash llama_7B_LayerImportance.sh \
  --dataset mrpc \
  --search-algorithm rl-and-ga-compare \
  --compare-config-mode direct \
  --stage2-compare-repeats 20 \
  --rl-compare-stage1-json glue_configs_best_ppo.json \
  --rl-compare-stage2-json glue_noise_configs_best_ppo.json \
  --ga-compare-stage1-json glue_configs_best_genetic.json \
  --ga-compare-stage2-json glue_noise_configs_best_genetic.json
```

方式 2：`persistent`，只给数据集、模型和约束，脚本自动去持久化目录定位对应的 RL / GA 结果。

启动前，launcher 会先根据 `(algorithm, model_type, dataset, 约束参数)` 推导出 RL / GA 两侧的目标持久化目录，并检查：

- 目录是否存在
- 目录下的 `metadata.json` 是否存在

任一检查失败都会直接报错并打印具体路径，不会等到 compare runner 启动后再失败。

```bash
# RL 与 GA 使用同一组约束（继承全局约束参数）
bash llama_7B_LayerImportance.sh \
  --dataset mrpc \
  --model-type bert-base \
  --search-algorithm rl-and-ga-compare \
  --compare-config-mode persistent \
  --compare-persistent-root rl_results/persistent \
  --stage1-accuracy-tolerance 0.005 \
  --stage2-limit-tolerance 0.05 \
  --stage2-stability-tolerance 0.05 \
  --stage2-compare-repeats 20
```

```bash
# RL 与 GA 使用不同约束，但模型和数据集必须一致
bash llama_7B_LayerImportance.sh \
  --dataset mrpc \
  --model-type bert-base \
  --search-algorithm rl-and-ga-compare \
  --compare-config-mode persistent \
  --compare-persistent-root rl_results/persistent \
  --rl-compare-stage1-accuracy-tolerance 0.005 \
  --rl-compare-stage2-limit-tolerance 0.05 \
  --rl-compare-stage2-stability-tolerance 0.05 \
  --ga-compare-stage1-accuracy-tolerance 0.01 \
  --ga-compare-stage2-limit-tolerance 0.1 \
  --ga-compare-stage2-stability-tolerance 0.1 \
  --stage2-compare-repeats 20
```

安全检查：

- `direct` 模式下，4 个 JSON 都会做算法家族检查；如果是最终结果 JSON，还会检查其中记录的数据集和模型层数。
- `persistent` 模式下，RL / GA 两侧可以使用不同约束参数，但脚本会强制要求两侧的 `dataset` 与 `model_type` 一致，否则直接报错。
- `persistent` 模式下，如果推导出的 RL / GA 持久化目录不存在，或目录中缺少 `metadata.json`，launcher 会立即报错并打印缺失路径，提示你先运行对应实验或检查约束参数。
- `Stage-2` 对比不是“只比较噪声配置”。最终评估会固定各自 Stage-1 的 GELU/Softmax 配置，再叠加各自 Stage-2 的噪声配置后比较。

该模式会在当前 compare run 根目录下生成：

- `reports/stage1_compare_summary_<dataset>.json`：Stage-1 结构化对比摘要，适合脚本读取
- `reports/stage1_compare_report_<dataset>.md`：Stage-1 对比文本报告
- `reports/stage1_compare_plot_<dataset>.png`：Stage-1 对比图，展示 RL/GA 在 Loss、主指标、次指标/Time、Cost 上的对比
- `reports/stage2_compare_summary_<dataset>.json`：Stage-2 结构化对比摘要，适合脚本读取
- `reports/stage2_compare_report_<dataset>.md`：Stage-2 对比文本报告；会明确写出 RL/GA 各自固定的 Stage-1 配置、选中的 Stage-2 噪声配置，以及多次评估后的均值、标准差、方差、最大值、最小值
- `reports/stage2_compare_plot_<dataset>.png`：Stage-2 对比图；会用均值±标准差柱状图展示 Loss、主指标、次指标/Time，并标注最小值/最大值，另附噪声 Cost 对比
- `meta/compare_metadata.json`：对比实验元信息，包括 compare 模式、RL/GA 输入来源、持久化目录或显式 JSON 路径、告警信息等
- `meta/compare_status.json`：运行中状态快照
- `meta/compare_final_status.json`：最终状态与报告路径汇总

补充说明：

- `direct` 模式下，如果传入的是配置模板而不是最终结果 JSON，compare runner 会先补做该侧最终评估，再进入对比。
- `reports/stage2_compare_summary_<dataset>.json` 中会保留 `fixed_stage1_config`、`selected.noise_config`、`repeat_evaluation.stats` 等字段，便于后续核对这次对比到底用了什么组合配置。

#### 5. 通用 RL 训练 — 数据集泛化

训练一个跨多个数据集泛化的通用策略。`general-rl train` 现在也使用持久化目录：
`rl_results/persistent/general-rl/<model_type>/<taskset_id>/<accuracy_slug>/`。
首次运行必须加 `--fresh-start`；后续相同 `taskset_id + accuracy_slug` 直接再次运行即可自动续训。

```bash
bash llama_7B_LayerImportance.sh \
  --dataset mrpc \
  --search-algorithm general-rl \
  --general-rl-mode train \
  --general-rl-tasks mrpc,cola,rte,stsb \
  --general-rl-rounds 50 \
  --ppo-update-interval 120 \
  --general-rl-lr 3e-5 \
  --fresh-start
```

#### 5b. 通用 RL 训练 — 准确度容忍泛化

训练一个适配不同准确度要求的通用策略。例如 `0.005,0.01,0.02` 表示策略需要同时适应 0.5%、1%、2% 三种 loss/指标波动容忍度。
此时持久化目录中的 `accuracy_slug` 会变成 `discrete_0.50pct_1.00pct_2.00pct` 这一类形式：

```bash
bash llama_7B_LayerImportance.sh \
  --dataset mrpc \
  --search-algorithm general-rl \
  --general-rl-mode train \
  --general-rl-tasks mrpc \
  --general-rl-accuracy-tolerances 0.005,0.01,0.02 \
  --general-rl-rounds 50 \
  --ppo-update-interval 120 \
  --general-rl-lr 3e-5 \
  --fresh-start
```

#### 5c. 通用 RL 训练 — 数据集 + 准确度联合泛化

同时在多个数据集和多个准确度要求上训练：

```bash
bash llama_7B_LayerImportance.sh \
  --dataset mrpc \
  --search-algorithm general-rl \
  --general-rl-mode train \
  --general-rl-tasks mrpc,cola,rte,stsb \
  --general-rl-accuracy-tolerances 0.005,0.01,0.02 \
  --general-rl-rounds 50 \
  --ppo-update-interval 120 \
  --general-rl-lr 3e-5 \
  --fresh-start
```

#### 5d. 通用 RL 训练 — 连续准确度范围泛化

如果你希望策略覆盖一个连续的准确度容忍区间，而不是一组离散值，可以使用
`--general-rl-accuracy-tolerance-range MIN,MAX`：

```bash
bash llama_7B_LayerImportance.sh \
  --dataset mrpc \
  --search-algorithm general-rl \
  --general-rl-mode train \
  --general-rl-tasks mrpc,cola,rte,stsb \
  --general-rl-accuracy-tolerance-range 0.005,0.02 \
  --general-rl-rounds 50 \
  --ppo-update-interval 120 \
  --general-rl-lr 3e-5 \
  --fresh-start
```

不同泛化模式的持久化目录示意：

| 泛化模式 | 条件 | 输出目录前缀 |
| --- | --- | --- |
| 数据集泛化 | `--general-rl-tasks` 含多个任务 | `rl_results/persistent/general-rl/<model_type>/<taskset_id>/default/` |
| 准确度离散泛化 | `--general-rl-accuracy-tolerances` 含多个值 | `rl_results/persistent/general-rl/<model_type>/<taskset_id>/discrete_.../` |
| 准确度范围泛化 | `--general-rl-accuracy-tolerance-range MIN,MAX` | `rl_results/persistent/general-rl/<model_type>/<taskset_id>/range_<lo>pct_<hi>pct/` |
| 联合泛化 | 多任务 + 多容忍 | `rl_results/persistent/general-rl/<model_type>/<taskset_id>/(discrete_... 或 range_...)/` |
| 单任务单容忍 | 均为单值 | `rl_results/persistent/general-rl/<model_type>/<dataset>/default/` |

> **安全检查**：
> - `--general-rl-accuracy-tolerances` 中的每个值必须是 (0, 1) 区间的正数（如 0.01 表示 1%）。
> - 训练模式下如果既没有提供多个任务也没有提供多个容忍值，脚本会发出警告。
> - `--general-rl-accuracy-tolerances` 不能在非 `general-rl` 模式下使用。

#### 6. 通用 RL 离线搜索

`general-rl` 现在正式使用 `--general-rl-mode search`；`infer` 仍然可以继续使用，但仅作为兼容别名。
如果你已经有一个训练好的通用 RL 持久化目录，也可以直接通过 `--general-policy-dir` 自动推导 Stage-1 / Stage-2 策略文件。

```bash
bash llama_7B_LayerImportance.sh \
  --search-algorithm general-rl \
  --general-rl-mode search \
  --general-stage1-policy general_stage1_policy.pt \
  --general-stage2-policy general_stage2_noise_policy.pt \
  --general-rl-num-rollouts 500 \
  --dataset qnli
```

指定搜索时的准确度容忍目标（使用准确度泛化策略时）：

```bash
bash llama_7B_LayerImportance.sh \
  --search-algorithm general-rl \
  --general-rl-mode search \
  --general-stage1-policy general_stage1_policy.pt \
  --general-rl-accuracy-tolerances 0.01 \
  --general-rl-num-rollouts 500 \
  --dataset mrpc
```

只做 Stage-1 搜索（跳过 Stage-2）：

```bash
bash llama_7B_LayerImportance.sh --logfile output.log \
  --search-algorithm general-rl \
  --general-rl-mode search \
  --general-stage1-policy general_stage1_policy.pt \
  --general-rl-skip-stage2 \
  --dataset mrpc
```

通过持久化目录自动推导策略文件：

```bash
bash llama_7B_LayerImportance.sh --logfile output.log \
  --search-algorithm general-rl \
  --general-rl-mode search \
  --general-policy-dir rl_results/persistent/general-rl/bert-base/cola_mrpc_rte_stsb/default \
  --dataset mrpc
```

贪心 rollout（确定性搜索，rollout 次数自动置 1）：

```bash
bash llama_7B_LayerImportance.sh --logfile output.log \
  --search-algorithm general-rl \
  --general-rl-mode search \
  --general-stage1-policy general_stage1_policy.pt \
  --general-rl-greedy \
  --dataset mrpc
```

### 7. 常见错误示例

下面这些组合现在会被脚本直接拦下：

```bash
# 错误：GA 模式却继续使用 RL 家族 JSON
bash llama_7B_LayerImportance.sh --logfile output.log \
  --dataset mrpc \
  --search-algorithm ga \
  --skip-stage1-search \
  --final-eval-source json \
  --final-eval-config glue_configs_best_ppo.json

# 错误：GA 模式却继续使用 RL 专用学习率参数
bash llama_7B_LayerImportance.sh \
  --dataset mrpc \
  --search-algorithm ga \
  --stage1-search-lr 1e-4

# 错误：RL 模式却读取 genetic 家族 JSON
bash llama_7B_LayerImportance.sh \
  --dataset mrpc \
  --search-algorithm rl \
  --skip-noise-search \
  --noise-eval-source json \
  --noise-eval-config glue_noise_configs_best_genetic.json

# 错误：general-rl 模式下使用了普通 RL / GA 阶段参数
bash llama_7B_LayerImportance.sh \
  --dataset mrpc \
  --search-algorithm general-rl \
  --general-rl-mode search \
  --general-stage1-policy general_stage1_policy.pt \
  --stage1-search-episodes 51000

# 错误：rl 模式下使用了 general-rl 专用参数
bash llama_7B_LayerImportance.sh \
  --dataset mrpc \
  --search-algorithm rl \
  --general-stage1-policy general_stage1_policy.pt
```

### 8. 迁移建议

- 老命令如果不加 `--search-algorithm`，默认仍走 RL，不会影响已有实验。
- 新做 GA 实验时，建议显式加 `--search-algorithm ga`，并统一使用 `--stage1-search-generations` / `--stage2-search-generations` 这组 GA 预算参数。
- 如果准备长期保留 GA 的 JSON 配置，建议按默认命名方式保存为 `glue_configs_best_genetic.json` 和 `glue_noise_configs_best_genetic.json`，这样脚本能自动做家族一致性检查。
- 使用通用 RL 时，建议先在少量任务上 `train` 训练策略，然后用 `search` 部署到新任务。`infer` 仍可用作兼容别名。通用策略做 500 次 rollout 的耗时约为 per-task RL 的 1/100。

This is a Repository for Transformer robustness evaluation using Reinforcement Learning.

Please Ignore the LLM-Adapters, EzPC, and importance-aware-sparse-tuning-IST-paper in root directory. Sorry, but the code is DIRTY now!

## 使用说明

> 下面这一大节里仍保留了部分历史实验记录与说明文字。若命令写法与上面的“命令行参数总表”冲突，请统一按新 CLI 规则理解：
>
> - `--model` → `--dataset`
> - `--batch_size` → `--batch-size`
> - `--skip-stage1-rl` → `--skip-stage1-search`
> - `--skip-noise-rl` → `--skip-noise-search`
> - `--stage1-rl-episodes` → `--stage1-search-episodes`（普通 RL）
> - `--stage2-rl-episodes` → `--stage2-search-episodes`（普通 RL）
> - GA 模式不再使用 `episode` 作为预算单位，改为 `--stage1-search-generations` / `--stage2-search-generations`
> - 删除旧的 5 个位置参数写法，统一改用全可选参数

### 运行前准备

```bash
mount -o remount,size=64G /dev/shm
conda activate llm_ist
cd /var/tmp/root-home/Reinforcement-For-Robustness
```

### 基础命令

```bash
bash llama_7B_LayerImportance.sh [可选参数]
```

当前脚本入口已经统一改成**全可选参数**，不再接受旧版 5 个位置参数。
已经移除的旧入口参数有：

- `lora_r`
- `lora_alpha`
- `degree`

基础示例：

```bash
bash llama_7B_LayerImportance.sh --dataset mrpc
```

### 并行安全运行（Concurrent-safe run layout）

命令格式已经改成全可选参数；`--logfile` 现在只用于提示日志文件名，实际日志写入位置由 run 目录决定。
现在会先按**模式**分层，再进入具体数据集/任务集合目录。每次启动都会自动创建唯一的 `run_id=<YYYYmmdd_HHMMSS>_pid<PID>`。

#### 1. 各模式的根目录

```text
# rl / ga 持久化目录（确定性路径，相同参数自动复用）
rl_results/persistent/<algorithm>/<model_type>/<dataset>/s1t<T>_s2t<L>_s2st<S>/

# general-rl train 持久化目录（确定性路径，相同 taskset + accuracy_slug 自动复用）
rl_results/persistent/general-rl/<model_type>/<taskset_id>/<accuracy_slug>/

# general-rl search / compare 仍使用时间戳目录
rl_results/runs/general_rl/search/<dataset>/<run_id>/
rl_results/runs/compare/rl_vs_ga/<dataset>/<run_id>/
```

说明：

- **rl / ga 持久化目录**：`<algorithm>` 为 `rl` 或 `ga`；`<model_type>` 为 `bert-base`、`bert-large`、`gpt-2`；`s1t<T>_s2t<L>_s2st<S>` 由三个约束参数拼成确定性标识（例如 `s1t0.005_s2t0.05_s2st0.05`）。首次运行需加 `--fresh-start`，后续相同参数自动续训练。
- **general-rl train 持久化目录**：`<taskset_id>` 由 `--general-rl-tasks` 规范化得到；`<accuracy_slug>` 由 `--general-rl-accuracy-tolerances` 或 `--general-rl-accuracy-tolerance-range` 决定，例如 `default`、`discrete_0.50pct_1.00pct_2.00pct`、`range_0.50pct_2.00pct`。首次运行同样需加 `--fresh-start`，后续相同参数自动续训练。
- `dataset` 是当前单任务数据集，例如 `mrpc`、`stsb`。
- `taskset_id` 是训练任务集合的规范化标识，例如 `mrpc,cola,rte,stsb` 会落为 `mrpc_cola_rte_stsb`。
- `accuracy_slug` 是准确度泛化标识：
  - `default`：未显式提供泛化容忍集合
  - `discrete_*`：使用 `--general-rl-accuracy-tolerances`
  - `range_*`：使用 `--general-rl-accuracy-tolerance-range`
- `compare` 模式单独放在 `compare/rl_vs_ga/` 下，不再与普通 `rl` / `ga` 共用同一个根目录层。

#### 2. 各模式内部目录

普通 `rl` / `ga` / `general-rl search` 的核心目录：

- nohup 启动日志与错误摘要：`<run_dir>/logs/`
- 第一阶段搜索：`<run_dir>/stage1/`
- 第一阶段最终评估：`<run_dir>/stage1_final_eval/`
- 第二阶段搜索：`<run_dir>/stage2_noise/`
- 第二阶段最终评估：`<run_dir>/stage2_noise_final_eval/`

`general-rl train` 额外会写出：

- 通用策略文件：`<run_dir>/general_stage1_policy.pt`
- 通用噪声策略文件：`<run_dir>/general_stage2_noise_policy.pt`
- 训练 checkpoint：`<run_dir>/general_stage1_train_checkpoint.pt`、`<run_dir>/general_stage2_train_checkpoint.pt`
- 各任务 evaluator 目录：`<run_dir>/task_<task_name>/`

`rl-and-ga-compare` 额外会写出：

- `reports/`：Stage-1 / Stage-2 的 `compare_summary_*.json`、`compare_report_*.md`、`compare_plot_*.png`
- `meta/`：`compare_metadata.json`、`compare_status.json`、`compare_final_status.json`、`compare.pid` 等元信息
- `children/rl/`、`children/ga/`：仅在 `direct` 模式且输入为“结果 JSON 物化”或需要补做最终评估时使用；`persistent` 模式通常直接复用 RL / GA 自己的持久化目录

#### 3. 示例

普通 RL 的持久化目录（默认约束参数）：

```text
rl_results/persistent/rl/bert-base/mrpc/s1t0.005_s2t0.05_s2st0.05/
```

GA 的持久化目录（自定义约束参数）：

```text
rl_results/persistent/ga/bert-base/mrpc/s1t0.01_s2t0.1_s2st0.1/
```

通用 RL 多任务训练的一个持久化目录（数据集泛化模式）：

```text
rl_results/persistent/general-rl/bert-base/mrpc_cola_rte_stsb/default/
```

通用 RL 准确度泛化训练的一个持久化目录：

```text
rl_results/persistent/general-rl/bert-base/mrpc/discrete_0.50pct_1.00pct_2.00pct/
```

通用 RL 联合泛化训练的一个持久化目录：

```text
rl_results/persistent/general-rl/bert-base/mrpc_cola_rte_stsb/discrete_0.50pct_1.00pct_2.00pct/
```

RL 与 GA 对比实验的一个 run：

```text
rl_results/runs/compare/rl_vs_ga/mrpc/<run_id>/
```

因此，下面这些命令可以同时并行运行，即使它们都使用相同的 `output.log`：

```bash
bash llama_7B_LayerImportance.sh --logfile output.log \
  --dataset mrpc \
  --skip-stage1-search \
  --final-eval-source json \
  --final-eval-config glue_configs_best_ppo.json \
  --skip-stage1-final-eval \
  --noise-eval-repeat 200

bash llama_7B_LayerImportance.sh --logfile output.log \
  --dataset stsb \
  --skip-stage1-search \
  --final-eval-source json \
  --final-eval-config glue_configs_best_ppo.json \
  --skip-stage1-final-eval \
  --noise-eval-repeat 200
```

脚本本身也不再强制设置 `CUDA_VISIBLE_DEVICES=0`。如果你想让并行运行时分别绑定不同 GPU，请在脚本外部设置：

```bash
CUDA_VISIBLE_DEVICES=0 bash llama_7B_LayerImportance.sh --logfile output.log --dataset mrpc
CUDA_VISIBLE_DEVICES=1 bash llama_7B_LayerImportance.sh --logfile output.log --dataset stsb
```

#### 如何在命令行并行跑多数据集

实现并行的关键是：每个进程都会自动落到独立的 run 目录（按模式分层后再进入 `<dataset_or_taskset>/<YYYYmmdd_HHMMSS>_pid<PID>/`），所以你可以在并行任务里重复使用同一个 `logfile_path`（例如都传 `output.log`），不会互相覆盖产出。

并行常用做法：

1. 最推荐：分别在不同终端窗口/会话里启动不同 `--dataset`（每条命令就是一个独立实验进程）。
2. 需要在同一终端里同时跑：把每条命令放到后台执行（给命令后面加 `&`），例如 `bash ... &`。

示例（并行跑 MRPC + STS-B；与上面“命令并行可运行”的示例一致）：

```bash
bash llama_7B_LayerImportance.sh --logfile output.log \
  --dataset mrpc \
  --skip-stage1-search \
  --final-eval-source json \
  --final-eval-config glue_configs_best_ppo.json \
  --skip-stage1-final-eval \
  --noise-eval-repeat 200 &

bash llama_7B_LayerImportance.sh --logfile output.log \
  --dataset stsb \
  --skip-stage1-search \
  --final-eval-source json \
  --final-eval-config glue_configs_best_ppo.json \
  --skip-stage1-final-eval \
  --noise-eval-repeat 200 &
```

如果你有多张 GPU，建议再给每条命令绑定不同 GPU（避免显存互抢）：

```bash
CUDA_VISIBLE_DEVICES=0 bash llama_7B_LayerImportance.sh --logfile output.log --dataset mrpc &
CUDA_VISIBLE_DEVICES=1 bash llama_7B_LayerImportance.sh --logfile output.log --dataset stsb &
```

#### 并行相关可选参数怎么用（对应上面的并行示例）

下表只聚焦并行时最常用、也出现在上面示例里的参数；更完整的各阶段参数与安全约束见后文各表格。


| 参数                         | 作用                                 | 并行时该怎么配                                                                |
| -------------------------- | ---------------------------------- | ---------------------------------------------------------------------- |
| `--logfile FILE`           | `nohup` 日志名提示（会取 basename 作为日志文件名） | 多个并行进程可传相同文件名（产出仍在各自 run 目录下）                                          |
| `--dataset`                | 选择数据集（并自动匹配对应 `base_model`）        | 并行时让不同进程分别用不同 `--dataset` 值                                            |
| `--skip-stage1-search`     | 跳过第一阶段搜索                            | 并行加速的常用开关：先有/后用已有配置或搜索结果时可加                                            |
| `--final-eval-source json` | 第一阶段最终评估配置来源为 JSON                 | 当 `--final-eval-source` 取 `json`（或 `manual`）时，需要显式加 `--skip-stage1-search` |
| `--final-eval-config PATH` | 第一阶段最终评估用的 JSON 配置文件路径             | 一般并行时保持一致，避免同时改动多个配置来源                                                 |
| `--skip-stage1-final-eval` | 跳过第一阶段最终评估                         | 只关心后续阶段（例如噪声阶段）时可加                                                     |
| `--noise-eval-repeat N`    | 噪声最终评估重复次数                         | 并行时想要统计更稳可调大；想缩短总耗时可调小                                                 |
| `--skip-noise-search`      | 跳过第二阶段噪声搜索                         | 只想跑噪声最终评估时加；当 `--noise-eval-source` 用 `json/manual` 时也需要显式加            |
| `--skip-noise-final-eval`  | 跳过第二阶段噪声最终评估                       | 只关心噪声 RL 训练过程/中间产物时加                                                   |
| `--noise-eval-source`      | 噪声最终评估配置来源（`search/json/manual`）   | 并行时常用 `json`：配合 `--noise-eval-config` 直接读配置                            |
| `--noise-eval-config PATH` | `json` 模式下的噪声配置文件                  | 例如默认的 `glue_noise_configs_best_ppo.json`                               |
| `--manual-noise-config`    | `manual` 模式下的噪声配置（JSON 字符串）        | 配置很少且不想改文件时用                                                           |


### --dataset 数据集+模型切换

可以通过 `--dataset` 一次性切换数据集和对应 `base_model`，不需要再手动改
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


| `--dataset` 值 | 自动设置 `--base_model`                  |
| ----------- | ------------------------------------ |
| `mrpc`      | `textattack/bert-base-uncased-MRPC`  |
| `stsb`      | `textattack/bert-base-uncased-STS-B` |
| `sst2`      | `textattack/bert-base-uncased-SST-2` |
| `wnli`      | `textattack/bert-base-uncased-WNLI`  |
| `rte`       | `textattack/bert-base-uncased-RTE`   |
| `cola`      | `textattack/bert-base-uncased-CoLA`  |
| `qnli`      | `textattack/bert-base-uncased-QNLI`  |


同时自动设置 `--data_path` 为同名任务（如 `--dataset qnli` -> `--data_path qnli`）。

示例：

```bash
# 默认 mrpc（不写 --dataset 也可以）
bash llama_7B_LayerImportance.sh --dataset mrpc

# 切换到 STS-B（回归任务）
bash llama_7B_LayerImportance.sh --dataset stsb

# 切换到 QNLI（问句-句子对）
bash llama_7B_LayerImportance.sh --dataset qnli
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


`--model-type` 与 `--dataset` 组合后会按 `(model-type, dataset)` 解析最终
`--base_model`。`bert-base` 兼容此前所有 7 个 GLUE 任务；`bert-large`
当前仅支持以下任务（其余任务暂时跳过，运行时会以
“bert-large 当前不支持数据集: …” 错误退出）：

- `mrpc`
- `cola`
- `stsb`
- `rte`
- `sst2`
- `qnli`

不支持的组合（例如 `--model-type bert-large --dataset wnli`）会在脚本
启动阶段立即报错并提示当前支持列表，避免到 HuggingFace 下载阶段才
失败。如果未来需要新增 bert-large checkpoint，可在
`llama_7B_LayerImportance.sh` 的 `MODEL_TYPE=bert-large` 分支里
扩展 `case "$DATASET"` 列表。

示例：

```bash
# 在 mrpc 上用 bert-large 跑完整两阶段流程（搜索 + 评估）
bash llama_7B_LayerImportance.sh --dataset mrpc --model-type bert-large

# 在 cola 上用 bert-large 跳过第一阶段 RL，仅做最终评估
bash llama_7B_LayerImportance.sh \
  --dataset cola --model-type bert-large \
  --skip-stage1-search --final-eval-source json --final-eval-config glue_configs_best_ppo.json

# 不写 --model-type 时等价于历史行为（bert-base）
bash llama_7B_LayerImportance.sh --dataset mrpc
```

注意事项：

1. bert-large 第一阶段每个 episode 需要在所有 24 层上各做一步决策，
  单次 PPO update 的 token 数也按 `total_layers` 自动翻倍，因此显存
   占用、单 episode 耗时大约是 bert-base 的 2 倍，建议在 24GB 及以上
   显存上运行，必要时通过 `--batch-size` 适当调小。
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


**GPT-2 的 Softmax 近似实现**：BERT 的 `BertSelfAttention` 模块
能够被整体替换为 `BertSelfAttentionWithAproximation`，从而在 forward
里用指数近似替换 softmax。GPT-2 的 `GPT2Attention` 将 Q/K/V 融合到
单个 Conv1D (`c_attn`)，并把因果 mask + scale + softmax 绑在同一个
`eager_attention_forward` 函数里。本 repo 通过 monkey-patch 该函数，
在 `replace_layer_softmax` 被调用时动态替换为使用近似 softmax 的版本
（`_make_gpt2_approx_attn_forward`），使 GPT-2 的 Stage 1 同时支持
GELU 和 Softmax 两种近似，Stage 2 七种噪声全部可用。恢复时
`restore_layer_softmax` 会还原原始 forward。

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
bash llama_7B_LayerImportance.sh --dataset sst2 --model-type gpt-2

# 在 mrpc 上跳过第一阶段 RL，直接用 JSON 中的 gpt-2 段做最终评估
bash llama_7B_LayerImportance.sh \
  --dataset mrpc --model-type gpt-2 \
  --skip-stage1-search --final-eval-source json --final-eval-config glue_configs_best_ppo.json

# 用 gpt-2 基座生成 GLUE 官网提交文件（前提：已在训练阶段完成 head 微调）
python generate_glue_submission.py \
  --config glue_configs_best_ppo.json \
  --noise_config glue_noise_configs_best_ppo.json \
  --model-type gpt-2 \
  --output_dir gpt2_run
```

### 命名参数与安全约束

当前脚本已经统一改成全可选参数，并增加了更严格的流程校验，避免“前面跑搜索，后面却拿手动/JSON 配置做评估”的混用。

**第一阶段：GELU/Softmax RL 与最终评估**

第一阶段的“是否执行 RL”与“是否执行最终评估”仍可独立控制，但配置来源现在有安全约束：

- 若执行第一阶段搜索，则 `--final-eval-source` 只能为 `search`
- 若使用 `json` 或 `manual`，则必须显式添加 `--skip-stage1-search`
- 若跳过第一阶段搜索，则不能再使用 `search`
- `--skip-stage1-search` 会一并跳过 Phase 1 baseline 建立、Phase 1.5 GELU 输入分布分析、Phase 2 PPO 和 Phase 2.5 贪心搜索


| 参数                             | 说明                                                                                  | 默认值                          |
| ------------------------------ | ----------------------------------------------------------------------------------- | ---------------------------- |
| `--skip-stage1-search`         | 跳过整个第一阶段搜索准备与搜索流程：Phase 1 baseline、Phase 1.5 GELU 输入分布分析、Phase 2 PPO、Phase 2.5 贪心搜索 | 不跳过                          |
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

- 若执行噪声搜索，且没有跳过噪声最终评估，则 `--noise-eval-source` 只能为 `search`
- 若使用 `json` 或 `manual` 做噪声最终评估，则必须显式添加 `--skip-noise-search`
- 若跳过噪声搜索，则不能在噪声最终评估中再使用 `search`


| 参数                                        | 说明                            | 默认值                                |
| ----------------------------------------- | ----------------------------- | ---------------------------------- |
| `--skip-noise-search`                     | 跳过第二阶段噪声搜索                  | 不跳过                                |
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
NOISE_RANDOM_MODE=x_only bash llama_7B_LayerImportance.sh --logfile output.log \
  --dataset mrpc \
  --skip-stage1-search --final-eval-source json --final-eval-config glue_configs_best_ppo.json \
  --skip-stage1-final-eval --skip-noise-search --noise-eval-source json \
  --noise-eval-config glue_noise_configs_best_ppo.json --noise-eval-repeat 200

# 2) 随机 X + 所有 W （默认，可省略环境变量）
NOISE_RANDOM_MODE=x_w bash llama_7B_LayerImportance.sh --logfile output.log \
  --dataset mrpc \
  --skip-stage1-search --final-eval-source json --final-eval-config glue_configs_best_ppo.json \
  --skip-stage1-final-eval --skip-noise-search --noise-eval-source json \
  --noise-eval-config glue_noise_configs_best_ppo.json --noise-eval-repeat 200

# 3) 随机 X + 所有 W + 非线性层（GELU/Softmax）
NOISE_RANDOM_MODE=x_w_nonlinear bash llama_7B_LayerImportance.sh --logfile output.log \
  --dataset mrpc \
  --skip-stage1-search --final-eval-source json --final-eval-config glue_configs_best_ppo.json \
  --skip-stage1-final-eval --skip-noise-search --noise-eval-source json \
  --noise-eval-config glue_noise_configs_best_ppo.json --noise-eval-repeat 200
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
`bash llama_7B_LayerImportance.sh --dataset mrpc`

只运行第一阶段，跳过第二阶段：
`bash llama_7B_LayerImportance.sh --dataset mrpc --skip-noise-search --skip-noise-final-eval`

不跑第一阶段 RL，直接从 JSON 读取第一阶段配置：
`bash llama_7B_LayerImportance.sh --dataset mrpc --skip-stage1-search --final-eval-source json --final-eval-config glue_configs_best_ppo.json`

不跑第一阶段 RL，手动指定每层 GELU/Softmax：
`bash llama_7B_LayerImportance.sh --logfile output_manual.log --dataset mrpc --skip-stage1-search --final-eval-source manual --manual-gelu "[1,1,1,4,1,1,1,1,1,1,1,1]" --manual-softmax "[2,3,4,6,4,4,5,4,4,5,5,2]"`

跳过噪声 RL，直接从 JSON 做噪声最终评估：  
`bash llama_7B_LayerImportance.sh --dataset mrpc --skip-noise-search --noise-eval-source json --noise-eval-config glue_noise_configs_best_ppo.json`

跳过噪声 RL，手动指定噪声配置并重复评估 100 次：
`bash llama_7B_LayerImportance.sh --dataset mrpc --skip-noise-search --noise-eval-source manual --manual-noise-config '{"x":[20,22,24,26,28,30,20,22,24,26,28,30],"wq":[14,16,18,20,22,14,16,18,20,22,14,16],"wk":[14,16,18,20,22,14,16,18,20,22,14,16],"wv":[14,16,18,20,22,14,16,18,20,22,14,16],"wo":[14,16,18,20,22,14,16,18,20,22,14,16],"wffn1":[16,18,20,22,24,16,18,20,22,24,16,18],"wffn2":[14,16,18,20,22,14,16,18,20,22,14,16]}' --noise-eval-repeat 100`

只进行第二阶段rl  
`CUDA_VISIBLE_DEVICES=0 bash llama_7B_LayerImportance.sh --logfile output.log --dataset mrpc --skip-stage1-search --final-eval-source json --final-eval-config glue_configs_best_ppo.json --skip-stage1-final-eval --noise-eval-repeat 200 --stage2-search-episodes [轮数] --batch-size [batch size 大小]`

（实例）

`CUDA_VISIBLE_DEVICES=0 bash llama_7B_LayerImportance.sh --logfile output.log --dataset mrpc --skip-stage1-search --final-eval-source json --final-eval-config glue_configs_best_ppo.json --skip-stage1-final-eval --noise-eval-repeat 200 --stage2-search-episodes 15000 --batch-size 128`

只进行第二阶段最终评估

`CUDA_VISIBLE_DEVICES=0 NOISE_RANDOM_MODE=x_w_nonlinear bash llama_7B_LayerImportance.sh --logfile output.log --dataset mrpc --skip-stage1-search --final-eval-source json --final-eval-config glue_configs_best_ppo.json --skip-stage1-final-eval --skip-noise-search --noise-eval-source json --noise-eval-config glue_noise_configs_best_ppo.json --noise-eval-repeat 200 --batch-size 128`

完全跳过两个阶段的搜索/训练，手动指定所有配置只做后续评估：

```bash
bash llama_7B_LayerImportance.sh --logfile output.log --dataset mrpc \
  --skip-stage1-search \
  --final-eval-source manual \
  --manual-gelu "[1,1,1,1,1,4,1,1,1,1,1,1]" \
  --manual-softmax "[2,2,5,5,5,2,5,2,5,5,6,2]" \
  --skip-stage1-final-eval \
  --skip-noise-search \
  --noise-eval-source manual \
  --manual-noise-config '{"x":[20,22,24,26,28,30,20,22,24,26,28,30],"wq":[14,16,18,20,22,14,16,18,20,22,14,16],"wk":[14,16,18,20,22,14,16,18,20,22,14,16],"wv":[14,16,18,20,22,14,16,18,20,22,14,16],"wo":[14,16,18,20,22,14,16,18,20,22,14,16],"wffn1":[16,18,20,22,24,16,18,20,22,24,16,18],"wffn2":[14,16,18,20,22,14,16,18,20,22,14,16]}'
```

帮助：  
`bash llama_7B_LayerImportance.sh --help`

使用 JSON 配置生成 GLUE 官网提交的 TSV（以及可选的 `submission.zip`），输出目录为 **`glue_submission/<output_dir>/`**；`--output_dir` 默认为 `run`，即默认写到 `glue_submission/run`。默认骨干为 **`bert-base`**；若使用仓库里为 GPT-2 准备的配置段，需加 **`--model-type gpt-2`**，且须先在训练流程中完成分类头微调。若无 GPU 且需跑 CPU，需加 **`--allow_cpu`**（会很慢）。

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
`ps aux | grep rl_ga_compare_runner.py`
`ps aux | grep rl_tune_genetic.py`
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

### `--batch-size` 可选项

可以通过命令行额外传入 `--batch-size N` 来覆盖当前脚本默认的批大小设置。


| 参数               | 说明                                                                                                                                                                | 默认值  |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- |
| `--batch-size N` | 统一设置 `llama_7B_LayerImportance.sh` 启动后强化学习与评估阶段使用的批大小。脚本会同步把 `--batch_size` 和 `--micro_batch_size` 都设为 `N`，并继续传递给 `rl_tune.py` 和 `layer_importance_evaluator.py`。 | `16` |


使用说明：

- `N` 必须是正整数，例如 `4`、`8`、`16`、`32`。
- 这个参数会影响 `Trainer` 的评估批大小，以及 `layer_importance_evaluator.py` 内部各个 dataloader 的 batch size。
- 数值调大后通常吞吐会更高，但显存占用也会更高；如果出现 OOM，建议先降到 `8` 或 `4`。
- 当前脚本为了保持原有行为一致，会把 `micro_batch_size` 一并设置成和 `batch_size` 相同的值。

示例：

```bash
# 使用 batch size = 8 运行 MRPC
bash llama_7B_LayerImportance.sh --logfile output.log --batch-size 8 --dataset mrpc

# 使用 batch size = 4，只运行第二阶段 noise RL
CUDA_VISIBLE_DEVICES=0 bash llama_7B_LayerImportance.sh --logfile output.log \
  --batch-size 4 \
  --dataset mrpc \
  --skip-stage1-search \
  --final-eval-source json \
  --final-eval-config glue_configs_best_ppo.json \
  --skip-stage1-final-eval \
  --noise-eval-repeat 200
```

### 搜索预算参数可选项

当前命令行把普通 RL 和 GA 的搜索预算单位拆开了：

- 普通 RL 使用 `episode`，对应 `--stage1-search-episodes` / `--stage2-search-episodes`。
- GA 使用 `generation`，对应 `--stage1-search-generations` / `--stage2-search-generations`。

| 参数 | 适用算法 | 说明 | 默认值 |
| --- | --- | --- | --- |
| `--stage1-search-episodes N` | `rl` | 设置普通 RL 的 Stage-1 PPO 搜索回合数。 | `51000` |
| `--stage2-search-episodes N` | `rl` | 设置普通 RL 的 Stage-2 noise PPO 搜索回合数。 | `40000` |
| `--stage1-search-generations N` | `ga` | 设置 GA 的 Stage-1 搜索迭代代数。 | 按模型自动推导 |
| `--stage2-search-generations N` | `ga` | 设置 GA 的 Stage-2 噪声搜索迭代代数。 | 按模型自动推导 |

使用说明：

- 所有 `N` 都必须是正整数。
- 普通 RL 在对应阶段未跳过时，`--stage1-search-episodes` / `--stage2-search-episodes` 必须大于等于 `--ppo-update-interval`（默认 `120`）。这是因为 PPO 每 `PPO_UPDATE_INTERVAL` 个 episode 才更新一次，回合数小于该值时 PPO 无法完成一次真正的策略更新。
- GA 不再使用 `episode` 作为搜索预算单位；请改用 `--stage1-search-generations` / `--stage2-search-generations`。
- `ga` 模式下如果不显式传代数，脚本会按模型层数自动推导默认值，以对齐旧版本的默认搜索预算。
- 在单独的 `rl` / `ga` 模式下，如果使用了 `--skip-stage1-search`，就不要再把该阶段预算当作“本次要执行的搜索预算”来理解。
- 在单独的 `rl` / `ga` 模式下，如果使用了 `--skip-noise-search`，就不要再把该阶段预算当作“本次要执行的搜索预算”来理解。
- 这些参数只控制搜索预算，不影响最终评估重复次数。
- 单独的 `rl` / `ga` 模式仍使用 `--noise-eval-repeat` 控制 Stage-2 最终评估重复次数。
- `rl-and-ga-compare` 不再重跑搜索，因此也不再消费这些预算参数；compare 模式只使用 `--stage2-compare-repeats` 控制 Stage-2 多次对比次数。

示例：

```bash
# 同时自定义第一阶段和第二阶段 RL 回合数
bash llama_7B_LayerImportance.sh --logfile output.log \
  --dataset mrpc \
  --stage1-search-episodes 1020 \
  --stage2-search-episodes 3400

# 自定义 GA 的第一阶段和第二阶段迭代代数
bash llama_7B_LayerImportance.sh --logfile output.log \
  --search-algorithm ga \
  --dataset mrpc \
  --stage1-search-generations 120 \
  --stage2-search-generations 90

# 跳过第一阶段，只运行第二阶段 noise RL，并把第二阶段回合数改成 680
CUDA_VISIBLE_DEVICES=0 bash llama_7B_LayerImportance.sh --logfile output.log \
  --dataset mrpc \
  --skip-stage1-search \
  --final-eval-source json \
  --final-eval-config glue_configs_best_ppo.json \
  --skip-stage1-final-eval \
  --stage2-search-episodes 680 \
  --noise-eval-repeat 200
```

### 持久化目录与自动续训练（`--fresh-start` / `--fresh-stage1` / `--fresh-stage2`）

`rl` 和 `ga` 模式使用**持久化目录**方案：相同的 `(算法, 模型, 数据集, 约束参数)` 组合映射到一个唯一的确定性目录。后续相同参数运行时自动检测 checkpoint 并续训练，无需手动指定 `--resume-from`。

#### 持久化目录路径格式

```text
rl_results/persistent/<algorithm>/<model_type>/<dataset>/s1t<T>_s2t<L>_s2st<S>/
```

其中 `s1t<T>` = `--stage1-accuracy-tolerance`，`s2t<L>` = `--stage2-limit-tolerance`，`s2st<S>` = `--stage2-stability-tolerance`。

示例：

```text
rl_results/persistent/rl/bert-base/mrpc/s1t0.005_s2t0.05_s2st0.05/
rl_results/persistent/ga/bert-large/stsb/s1t0.01_s2t0.1_s2st0.1/
```

目录下会自动创建 `metadata.json`，记录算法、模型、数据集、约束参数值、创建时间和运行次数。

#### 运行状态与 fresh 行为

| 状态 | 条件 | 行为 |
| --- | --- | --- |
| **首次运行** | 目录不存在 + 未跳过所有搜索 | 必须指定 `--fresh-start`，否则报错 |
| **自动续训练** | 目录存在 + 有 `metadata.json` | 自动从 checkpoint 续训练（不加 `--fresh-start`） |
| **从头训练** | 指定 `--fresh-start` + 目录已存在 | 清除旧目录并重新开始 |
| **仅重置 Stage-1** | 目录存在 + 有 `metadata.json` + `--fresh-stage1` | 删除 `stage1/` 与 `stage1_final_eval/`，保留 Stage-2 |
| **仅重置 Stage-2** | 目录存在 + 有 `metadata.json` + `--fresh-stage2` | 删除 `stage2_noise/` 与 `stage2_noise_final_eval/`，保留 Stage-1 |
| **Eval-only** | 目录不存在 + 所有搜索都跳过 | 自动创建目录，无需 `--fresh-start` |

补充说明：

- `--fresh-stage1` 和 `--fresh-stage2` 只适用于**已有持久化目录**的续训场景；如果这是第一次运行该参数组合，仍然必须使用 `--fresh-start`。
- `--fresh-start` 会清空整个持久化目录；`--fresh-stage1` / `--fresh-stage2` 只会清空对应阶段及其最终评估目录。
- 如果你只想重做某一个阶段，优先使用分阶段 fresh，而不是直接 `--fresh-start`。

#### 使用示例

```bash
# 首次运行：必须加 --fresh-start
bash llama_7B_LayerImportance.sh \
  --dataset mrpc \
  --fresh-start \
  --stage2-search-episodes 15000

# 发现轮数不够，直接加大预算再跑（不加 --fresh-start → 自动续训练）
bash llama_7B_LayerImportance.sh \
  --dataset mrpc \
  --stage2-search-episodes 30000

# 仅重做 Stage-2，保留已有 Stage-1 结果
bash llama_7B_LayerImportance.sh \
  --dataset mrpc \
  --fresh-stage2 \
  --skip-stage1-search \
  --skip-stage1-final-eval \
  --stage2-search-episodes 30000

# 仅重做 Stage-1，保留 Stage-2 目录；通常建议同时跳过 Stage-2
bash llama_7B_LayerImportance.sh \
  --dataset mrpc \
  --fresh-stage1 \
  --skip-noise-search \
  --skip-noise-final-eval \
  --stage1-search-episodes 60000

# 换一组约束参数 → 新的持久化目录，需要再次 --fresh-start
bash llama_7B_LayerImportance.sh \
  --dataset mrpc \
  --fresh-start \
  --stage1-accuracy-tolerance 0.01

# Eval-only：跳过所有搜索，无需 --fresh-start
bash llama_7B_LayerImportance.sh \
  --dataset mrpc \
  --skip-stage1-search \
  --final-eval-source json \
  --skip-stage1-final-eval \
  --skip-noise-search \
  --skip-noise-final-eval
```

#### 安全注意事项

1. **`--fresh-start` 会删除整个持久化目录**（`rm -rf`）。如果该目录中已有 Stage-1 搜索结果，`--fresh-start` 会一并删除，即使你同时使用了 `--skip-stage1-search`。

   - 如果你只想重做 Stage-2 而保留 Stage-1 结果，**不要使用 `--fresh-start`**，直接运行即可。自动续训练会保留 Stage-1 结果，Stage-2 从 checkpoint 继续。
   - 如果你误用了 `--fresh-start` + `--skip-stage1-search`，脚本会打印警告并等待 5 秒，给你取消的机会。

2. **`--fresh-stage1` / `--fresh-stage2` 是分阶段清理，不会删除整个持久化目录**。

   - `--fresh-stage1` 会删除 `stage1/` 与 `stage1_final_eval/`，并把 metadata 中的 Stage-1 状态重置为 `not_started`。
   - `--fresh-stage2` 会删除 `stage2_noise/` 与 `stage2_noise_final_eval/`，并把 metadata 中的 Stage-2 状态重置为 `not_started`。
   - 如果目录不存在，脚本不会把它当作“首次运行的分阶段 fresh”；这时仍然需要显式使用 `--fresh-start`。

3. **Stage-1 配置一致性校验**：Stage-2 checkpoint 会记录训练时使用的 Stage-1 GELU/Softmax 配置。如果续训练时 Stage-1 配置发生了变化（例如上次跳过 Stage-1 用 JSON 配置，这次做了实际 Stage-1 搜索得到不同结果），系统会打印 `⚠⚠ [Stage-1 配置不一致警告]`，提醒你 Stage-2 policy 可能需要重新训练。

4. **并发安全**：同一参数组合的持久化目录在同一时刻只能由一个进程使用。如需并行运行，请确保使用不同的约束参数或不同的数据集（它们会落到不同的持久化目录）。

5. **`general-rl` 的搜索运行目录和 `rl-and-ga-compare` 的报告目录仍使用时间戳目录**：`general-rl` 训练已改用持久化目录自动续训；`search` 模式仍使用时间戳目录。`rl-and-ga-compare` 的 `persistent` 模式会去读取 `rl` / `ga` 的持久化目录作为输入来源。

### 自动续训与 checkpoint

> **注意**：当前 launcher 的正式续训方式已经统一为“按确定性目录自动恢复”。
>
> - `rl` / `ga`：按 `(algorithm, model_type, dataset, 约束参数)` 使用持久化目录自动续训。
> - `general-rl train`：按 `(model_type, taskset_id, accuracy_slug)` 使用持久化目录自动续训。
> - `general-rl search`：只产生时间戳结果目录，不属于训练续训流程。
> - `rl-and-ga-compare`：只读取已有结果，不属于训练续训流程。
>
> `--resume-from` 目前仅作为兼容保留参数存在；普通 launcher 用法不再建议手动传它。

训练过程中会自动保存 checkpoint：

| 搜索算法 | Stage-1 checkpoint 路径 | Stage-2 checkpoint 路径 |
| --- | --- | --- |
| `rl` | `<persistent_dir>/stage1/stage1_rl_checkpoint.pt` | `<persistent_dir>/stage2_noise/progress/noise_rl_checkpoint.pt` |
| `ga` | `<persistent_dir>/stage1/ga_stage1_checkpoint.pt` | `<persistent_dir>/stage2_noise/progress/ga_stage2_checkpoint.pt` |
| `general-rl train` | `<persistent_dir>/general_stage1_train_checkpoint.pt` | `<persistent_dir>/general_stage2_train_checkpoint.pt` |

使用说明：

- **rl / ga 模式**：相同参数组合直接再次运行即可自动续训，无需 `--resume-from`。
- **general-rl train**：相同的 `--general-rl-tasks` 与准确度配置再次运行即可自动续训；首次运行必须加 `--fresh-start`。
- 续训时指定的是**总搜索预算**而不是追加量。
- 如果指定的总轮数小于等于 checkpoint 中已完成的轮数，则该阶段不会追加训练。
- checkpoint 会在每次进度快照时自动保存，因此即使训练中途被中断，也可以从最近的 checkpoint 恢复。

示例：

```bash
# ---- RL 模式续训（自动）----

# 首次运行
bash llama_7B_LayerImportance.sh --logfile output.log \
  --dataset mrpc \
  --fresh-start \
  --stage2-search-episodes 15000

# 发现轮数不够，直接加大预算再跑（相同参数 → 自动续训练）
bash llama_7B_LayerImportance.sh --logfile output.log \
  --dataset mrpc \
  --stage2-search-episodes 30000

# ---- GA 模式续训（自动）----

# 首次运行
bash llama_7B_LayerImportance.sh --logfile output.log \
  --search-algorithm ga \
  --dataset mrpc \
  --fresh-start \
  --stage1-search-generations 120 \
  --stage2-search-generations 90

# 发现代数不够，直接加大预算再跑
bash llama_7B_LayerImportance.sh --logfile output.log \
  --search-algorithm ga \
  --dataset mrpc \
  --stage1-search-generations 180 \
  --stage2-search-generations 140

# ---- General-RL train 模式续训（自动）----

# 第一次通用 RL 训练（必须加 --fresh-start）
bash llama_7B_LayerImportance.sh --logfile output.log \
  --search-algorithm general-rl \
  --general-rl-mode train \
  --general-rl-tasks mrpc,cola,rte,stsb \
  --general-rl-rounds 50 \
  --dataset mrpc \
  --fresh-start

# 从上次训练中断处继续（相同 taskset + accuracy_slug，直接再次运行）
bash llama_7B_LayerImportance.sh --logfile output.log \
  --search-algorithm general-rl \
  --general-rl-mode train \
  --general-rl-tasks mrpc,cola,rte,stsb \
  --general-rl-rounds 100 \
  --dataset mrpc
```

### 优雅停止与断点续训（Graceful Stop / Resume）

由于 `llama_7B_LayerImportance.sh` 内部以 `nohup ... &` 的方式把实际的 Python
训练进程放到后台运行，因此**请不要再使用 `kill -9 <PID>`（SIGKILL）**，否则会
绕过 checkpoint 保存逻辑、导致下次续训时训练曲线出现明显断层。正确的做法是
**发送 SIGINT 触发优雅停止**，程序会在下一次安全边界（PPO 更新边界 / 遗传代际边界 / round 边界）
保存 checkpoint，然后安全退出。**此功能适用于 rl、ga、general-rl 三种搜索算法。**

续训方式因模式而异：
- **rl / ga**：持久化目录方案，checkpoint 保存在确定性路径中。下次用相同参数运行即可自动续训练，无需 `--resume-from`。
- **general-rl train**：持久化目录方案。下次用相同的 `taskset_id + accuracy_slug` 直接再次运行即可自动续训练。
- **general-rl search**：是离线搜索结果生成流程，不是训练续训流程。

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
      kill -INT $(cat rl_results/runs/rl/mrpc/LATEST_PID)

  方式 C：创建停止标志文件（不依赖信号，适合脚本化批量停止）
    ── rl 或 ga 模式：
      touch rl_results/.../20260408_.../stage1/STOP_RL              # 停 Stage-1
      touch rl_results/.../20260408_.../stage2_noise/progress/STOP_RL  # 停 Stage-2
    ── general-rl 模式：
      touch rl_results/.../20260408_.../STOP_RL                     # 停当前 Stage

  停止后续训：
    rl / ga：相同参数直接再跑即可自动续训练
    general-rl train：相同 taskset + accuracy_slug 直接再跑即可自动续训练
  ⚠ 切勿使用 kill -9（SIGKILL），它会绕过 checkpoint 保存导致续训断层！
========================================================================
```

脚本同时在以下位置持久化了 PID / run 目录信息：

- `<run_dir>/run.pid`：当前任务的 Python 进程 PID（非 compare 模式）
- `rl_results/runs/rl/<dataset>/LATEST_PID`：普通 RL 最近一次启动的 PID
- `rl_results/runs/rl/<dataset>/LATEST_RUN_DIR`：普通 RL 最近一次启动的 run 目录
- `rl_results/runs/ga/<dataset>/LATEST_PID`：GA 最近一次启动的 PID
- `rl_results/runs/ga/<dataset>/LATEST_RUN_DIR`：GA 最近一次启动的 run 目录
- `rl_results/persistent/general-rl/<model_type>/<taskset_id>/LATEST_PID`：通用 RL 训练最近一次启动的 PID
- `rl_results/persistent/general-rl/<model_type>/<taskset_id>/LATEST_RUN_DIR`：通用 RL 训练最近一次启动的持久化目录
- `rl_results/runs/general_rl/search/<dataset>/LATEST_PID`：通用 RL 搜索最近一次启动的 PID
- `rl_results/runs/general_rl/search/<dataset>/LATEST_RUN_DIR`：通用 RL 搜索最近一次启动的 run 目录
- `rl_results/runs/compare/rl_vs_ga/<dataset>/LATEST_PID`：对比实验最近一次启动的 compare runner PID
- `rl_results/runs/compare/rl_vs_ga/<dataset>/LATEST_RUN_DIR`：对比实验最近一次启动的 run 目录

补充说明：

- 由于目录现在按模式分层，**不要再跨模式复用同一组 LATEST 指针**。例如要停止 GA，就看 `rl_results/runs/ga/<dataset>/LATEST_PID`，不要看 `rl_results/runs/rl/<dataset>/LATEST_PID`。
- `general-rl train` 的 `taskset_id` 由 `--general-rl-tasks` 规范化得到。例如 `mrpc,cola,rte,stsb` 会落为 `mrpc_cola_rte_stsb`，并写到 `rl_results/persistent/general-rl/<model_type>/mrpc_cola_rte_stsb/<accuracy_slug>/` 下。
- `compare` 模式除了 `LATEST_RUN_DIR` / `LATEST_PID` 之外，还会额外兼容写入 `LATEST_COMPARE_RUN_DIR` / `LATEST_COMPARE_PID`。
- 当前 compare run 目录下会写出 `meta/compare.pid`，对应 compare runner 本身；不再额外启动 RL / GA 子进程。

#### 完整示例：RL 模式下的启动 → 优雅停止 → 续训（持久化目录）

```bash
# 1) 首次启动训练（必须加 --fresh-start）
CUDA_VISIBLE_DEVICES=0 bash llama_7B_LayerImportance.sh --logfile output.log \
    --dataset mrpc \
    --fresh-start \
    --skip-stage1-search \
    --final-eval-source json \
    --final-eval-config glue_configs_best_ppo.json \
    --skip-stage1-final-eval \
    --noise-eval-repeat 200 \
    --stage2-search-episodes 15000 \
    --batch-size 128
# -> 终端会打印 Background PID: 712345
# -> 持久化目录：rl_results/persistent/rl/bert-base/mrpc/s1t0.005_s2t0.05_s2st0.05/
```

```bash
# 2) 想停的时候，任选一条（同一条命令可多次执行，首次触发后即会进入保存流程）：

# 方式 A：直接用启动时打印的 PID
kill -INT 712345

# 方式 C：创建停止标志文件（Stage-2 训练时）
touch rl_results/persistent/rl/bert-base/mrpc/s1t0.005_s2t0.05_s2st0.05/stage2_noise/progress/STOP_RL
```

收到信号后，你会在日志里看到：

```
  [优雅停止] 收到中断信号 (Ctrl+C)，将在当前回合结束后保存 checkpoint 并退出；再按一次 Ctrl+C 立即强退。
  [优雅停止] checkpoint 已写入 → rl_results/persistent/.../stage2_noise/progress/noise_rl_checkpoint.pt
  下次用相同参数直接运行即可自动续训练。
```

```bash
# 3) 续训：相同参数直接运行，不加 --fresh-start → 自动检测 checkpoint 续训练
#    （建议同时把 --stage2-search-episodes 调到你想要的”总轮数”）
CUDA_VISIBLE_DEVICES=0 bash llama_7B_LayerImportance.sh --logfile output.log \
    --dataset mrpc \
    --skip-stage1-search \
    --final-eval-source json \
    --final-eval-config glue_configs_best_ppo.json \
    --skip-stage1-final-eval \
    --noise-eval-repeat 200 \
    --stage2-search-episodes 30000 \
    --batch-size 128
```

续训会加载上次保存的 GTrXL 网络权重、优化器状态、PPO 更新计数、`reward_history`
/ `reward_mean` / `reward_std`、`return_normalizer`（值函数归一化统计量）、
incumbent / best_config / window_best、以及所有 episode 级统计列表，因此
training-curve 曲线与 PPO 训练曲线在续训前后能**严丝合缝地接上**。

#### 完整示例：GA 模式下的启动 → 优雅停止 → 续训（持久化目录）

```bash
# 1) 首次启动 GA 搜索（必须加 --fresh-start）
bash llama_7B_LayerImportance.sh --logfile output.log \
    --search-algorithm ga \
    --dataset mrpc \
    --fresh-start \
    --stage1-search-generations 120 \
    --stage2-search-generations 90
# -> 脚本打印 Background PID: 812345 及停止命令
# -> 持久化目录：rl_results/persistent/ga/bert-base/mrpc/s1t0.005_s2t0.05_s2st0.05/

# 2) 优雅停止（方式 A / C 均可）
kill -INT 812345

# 或创建停止标志文件：
# touch rl_results/persistent/ga/bert-base/mrpc/s1t0.005_s2t0.05_s2st0.05/stage1/STOP_RL
# touch rl_results/persistent/ga/bert-base/mrpc/s1t0.005_s2t0.05_s2st0.05/stage2_noise/progress/STOP_RL

# 3) 续训：相同参数直接运行，不加 --fresh-start
bash llama_7B_LayerImportance.sh --logfile output.log \
    --search-algorithm ga \
    --dataset mrpc \
    --stage1-search-generations 180 \
    --stage2-search-generations 140
```

#### 完整示例：General-RL 模式下的启动 → 优雅停止 → 续训

```bash
# 1) 启动通用 RL 多任务训练
bash llama_7B_LayerImportance.sh --logfile output.log \
    --search-algorithm general-rl \
    --general-rl-mode train \
    --general-rl-tasks mrpc,cola,rte,stsb \
    --general-rl-rounds 50 \
    --dataset mrpc
# -> 脚本打印 Background PID: 912345 及停止命令

# 2) 优雅停止
kill -INT 912345

# 或创建停止标志文件（general-rl 的 STOP_RL 位于 run 根目录）：
# touch rl_results/.../STOP_RL

# 3) 续训：增大 rounds，直接再次运行即可自动续训
bash llama_7B_LayerImportance.sh --logfile output.log \
    --search-algorithm general-rl \
    --general-rl-mode train \
    --general-rl-tasks mrpc,cola,rte,stsb \
    --general-rl-rounds 100 \
    --dataset mrpc
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
- 续训时普通 RL 的 `--stage1-search-episodes` / `--stage2-search-episodes` 指的是**总回合数**，GA 的 `--stage1-search-generations` / `--stage2-search-generations` 指的是**总代数**；都不是追加量。若总预算小于等于 checkpoint 中已完成的值，该阶段不会再追加训练。
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

此外，搜索完成后还会自动汇总到 `best_policy/` 目录：

```text
<run_dir>/best_policy/
├── stage1_policy.pt           # 第一阶段最佳 policy 副本
├── stage2_noise_policy.pt     # 第二阶段最佳 noise policy 副本
└── constraint_metadata.json   # 约束参数元信息（tolerance、dataset、algorithm）
```

`constraint_metadata.json` 记录了训练时使用的 `stage1_accuracy_tolerance`、`stage2_limit_tolerance`、`stage2_stability_tolerance`、`dataset`、`search_algorithm` 等信息，便于下游模块（如通用 RL）识别 policy 的训练条件。

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
2. **离线搜索（Offline Search）**：加载训练好的 general policy，在新任务上**只做前向推理（不训练）**，通过 best-of-K rollout 找到该任务的最优配置。
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
    episodes_per_task_per_round=120,    # 每轮每任务的 episode 数
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
    episodes_per_task_per_round=120,
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
| `episodes_per_task_per_round` | 每轮每任务采集的 episode 数（需等于 `PPO_UPDATE_INTERVAL`，确保 PPO buffer 不跨任务混合） | `120` |
| `lr` | Adam 优化器学习率 | `3e-5` |
| `device` | PyTorch 设备 | `"cuda"` |

> 使用 CLI (`llama_7B_LayerImportance.sh`) 启动通用 RL 时，`--ppo-update-interval N` 会同时设置 `PPO_UPDATE_INTERVAL` 与 `episodes_per_task_per_round`，无需单独配置。

##### Phase B：离线部署（在新 / 旧任务上做搜索，不训练）

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

离线搜索关键参数：

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
        "episodes_per_task_per_round": 120,
        "total_episodes": 24000,
        "ppo_updates": 200,
        "best_rewards": {"mrpc": 1.23, "stsb": 0.98, ...},
    },
}
```

#### 与 per-task 便携 policy 的兼容性

| 场景 | 兼容性 | 备注 |
| --- | --- | --- |
| 通用策略 → per-task online RL（作为 base policy） | ✅ | `task_context_proj` 层被 `strict=False` 跳过，不影响运行 |
| per-task 便携 policy → 通用策略离线搜索 | ✅ | 缺失的 `task_context_proj` 保持零初始化 |
| 不同 `total_layers`（如 12 层 → 24 层） | ⚠️ 部分迁移 | layer embedding 不匹配的层会被跳过 |
| Stage-1 ↔ Stage-2 | ❌ | 动作空间不同，必须分别训练 |

#### 推荐工作流

1. **多任务训练**：选 4-7 个 GLUE 任务，用 `multi_task_train_stage1` / `multi_task_train_stage2` 分别训练通用 Stage-1 和 Stage-2 策略。
2. **离线快速部署**：新任务到来时，用 `offline_find_best_config_stage1` / `offline_find_best_config_stage2` 做 500 次 rollout 快速找到最优配置，无需数万回合 online RL。
3. **精细化微调**：如果离线结果不够理想，可将通用策略作为 base policy 加载到 online RL 中继续微调，通常只需原来 1/5 ~ 1/3 的回合数即可收敛。
4. **Critic 预筛选**：如果有大量候选配置需要评估，先用 `critic_quick_rank_stage1` 快速排序，再对 top-K 做真实评测。


所有 v3 更新汇总
改动位置速览
全部优化通过 noise_rl_module_v2.py:211 的 NOISE_RL_OPT_FLAGS 字典控制，默认全部开启，每项可单独关闭回退到 v2 行为。

各优化项说明
v3-A 熵退火调度（Cosine + Plateau）
问题： 旧 schedule 让熵长期停在 ~0.02，policy 近乎 uniform，7 个动作头无法真正收敛。

改动： 前 8% 回合维持高探索（0.02），之后 cosine 衰减至 0.0008，允许策略真正 commit。

代码： 293-315 _resolve_stage2_entropy_coef，主循环 2551 注入。

控制键：


"use_v3_entropy_schedule": True,   # False → 退回 v2 schedule
"v3_entropy_start": 0.02,
"v3_entropy_end": 0.0008,
"v3_entropy_floor": 0.0006,
"v3_entropy_plateau_ratio": 0.08,
v3-B 逐头（Per-head）熵地板
问题： 全局熵均值合格时，某些头（如 wffn1）可能已完全坍缩，但被其他头的高熵"平均掉"。

改动： 对每个动作头单独检测是否低于 log(action_dim) × 22%，只对真正坍缩的头施加补偿。

代码： 辅助函数 318-336；evaluate_actions 新增 return_per_head_entropy 参数 1097；PPO 更新中调用。

控制键：


"use_per_head_entropy_recovery": True,   # False → 退回全局均值检测
"v3_per_head_entropy_floor_frac": 0.22,
"v3_per_head_recovery_multiplier": 6.0,
v3-C 势能塑形（Potential-based Reward Shaping）
问题： 仅终止步有奖励（稀疏），前 11 步全为 0，策略梯度信号极弱。

改动： 每步叠加 shaping_weight × (φ(s') - φ(s))，φ 定义为成本超支和噪声债务的负值。数学上保证最优策略不变（Ng et al. 1999 势能塑形定理）。终止步补偿 -φ(s_prev) 使总回报不偏移。

代码： 辅助函数 338-353；env.reset() 1249；env.step() 非终止 1463 与终止 1482 分支。

控制键：


"use_potential_shaping": True,   # False → 完全禁用（纯稀疏奖励）
"shaping_weight": 0.06,
v3-D PPO 超参优化 + KL 自适应学习率
问题： eps_clip=0.12 + K=6 步长太保守；LR 恒定，KL 超标时仍强行更新。

改动：

eps_clip 0.12 → 0.2，K_epochs 6 → 4，mini_batch 8 → 12（更大更少批次）
LR × 2.5 倍（避免 LR 过低导致有效更新太慢）
KL 自适应：上次 update 均值 KL > 2× target → LR × 0.5；< 0.5× target → LR × 1.5，比率限制在 [0.25, 2.5]
代码： 1650-1690。

控制键：


"use_v3_ppo_hparams": True,   # False → 退回 v2 所有 PPO 超参（clip/K/mini_batch/LR 全不变）
"v3_eps_clip": 0.2,
"v3_k_epochs": 4,
"v3_mini_batch_episodes": 12,
"v3_lr_multiplier": 2.5,
"v3_adaptive_lr_kl": True,
"v3_kl_adaptive_target": 0.015,
"v3_kl_adaptive_min_ratio": 0.25,
"v3_kl_adaptive_max_ratio": 2.5,
v3-E Challenger 确认优化（三合一）
问题三点：

trigger margin=0 → 任何比 incumbent 高一点的候选都触发昂贵 confirm（10 段评测）
训练 MC std 已超标时 confirm 必败，但仍白白评测
confirm fail 写 -50 惩罚 → 主导整 batch advantage，其他 episode 梯度被压成接近 0
改动：

trigger margin 提高到 0.010（需显著提升才触发）
训练期 last_mc_eval std > cap × 1.5 时直接跳过 confirm，日志标记 [v3 precheck]
confirm fail penalty 从 -50 裁剪到 -5
代码： 2732-2875。

控制键：


"v3_confirm_trigger_margin": 0.010,   # 只有 use_v3_ppo_hparams=True 时生效
"v3_confirm_precheck_std": True,
"v3_confirm_penalty_clip_min": -5.0,  # 只有 use_v3_ppo_hparams=True 时生效
v3-F 策略冷启动偏置（Warmstart Bias）
问题： 初始化时每个动作等概率（~20%），policy 从噪声最高处开始探索，触发大量约束违反 → 负奖励风暴。

改动： 对每个动作头的 bias[-1]（= 最大 SF = 最低风险动作）加 1.2，使初始概率约 47% vs 其他 13%，从 baseline 附近出发。恢复 checkpoint 时 load_state_dict 覆盖该偏置，不影响续训。

代码： 1017-1026。

控制键：


"v3_warmstart_baseline_bias": True,
"v3_warmstart_bias_gain": 1.2,
v3-G 稳健 Advantage 归一化
问题： confirm fail 的 -50 惩罚使 adv.std() 暴增，把其余 episode 的 advantage 全压成接近 0，policy 梯度几乎消失。

改动： z-score 之前先按 中位数 ± 6×MAD 裁剪离群点，再做标准化；只影响统计稳定性，不改变方向。

代码： 1715-1730。

控制键：


"v3_robust_advantage_norm": True,
"v3_adv_outlier_clip": 6.0,   # 中位数 ± 6×MAD
消融/回滚方式
要回退哪项	操作
全部 v3	把所有 "use_v3_*": True 改为 False
仅 PPO 超参	"use_v3_ppo_hparams": False
仅熵调度	"use_v3_entropy_schedule": False
仅 shaping	"use_potential_shaping": False
仅 per-head 熵	"use_per_head_entropy_recovery": False
仅 robust adv norm	"v3_robust_advantage_norm": False
仅 warmstart	"v3_warmstart_baseline_bias": False
仅 confirm 优化	"v3_confirm_precheck_std": False + "use_v3_ppo_hparams": False（margin/penalty 随之失效）