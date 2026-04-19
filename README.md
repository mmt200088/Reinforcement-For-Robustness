## ⚠️ 最终评估 CLI 统一说明（请先读此节）

Stage-1（GELU/Softmax 多项式次数）与 Stage-2（噪声 scaling factor）原先是两套独立的
"最终评估"流程，参数各自分家。现在已合并为**单一的统一最终评估**，请用新 flag：

| 旧 flag / 文件 | 新 flag / 文件 | 说明 |
|---|---|---|
| `--skip-stage1-final-eval` + `--skip-noise-final-eval` | `--skip-final-eval` | 一次跳过两个阶段合并的最终评估 |
| `--manual-gelu` | `--manual-stage1-gelu` | 仅改名，语义不变 |
| `--manual-softmax` | `--manual-stage1-softmax` | 仅改名，语义不变 |
| `--manual-noise-config` | `--manual-stage2-noise` | 仅改名，语义不变 |
| `--noise-eval-source` | `--final-eval-source` | 统一成一个 source，对 Stage-1 / Stage-2 同时生效 |
| `--noise-eval-config` | `--final-eval-config` | 同一份合并 JSON 同时覆盖两个阶段 |
| `--noise-eval-repeat` | `--final-eval-repeat` | 重复次数对合并流程整体生效 |
| `glue_configs_best_*.json` + `glue_noise_configs_best_*.json` | `glue_final_configs_best_*.json` | 合并为一份；`{变体: {任务: {stage1: {...}, stage2: {...}}}}` |
| `noise_final_evaluation_module.py` | `final_evaluation_module.py`（`UnifiedFinalEvaluationModule`） | 旧模块已吸收进统一评估器 |

本 README 下文已批量替换为新名称。若你在旧 shell / notebook 里还看到旧 flag，请按照
上表对照替换；直接沿用旧名称会报 `unknown argument`。

---

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
| `--skip-final-eval` | `rl`、`ga` | — | 一次跳过 Stage-1 + Stage-2 合并的最终评估（取代旧的两个分开 flag） |
| `--final-eval-source search/json/manual` | `rl`、`ga` | `search` | 统一最终评估的配置来源，同时覆盖 Stage-1 与 Stage-2 |
| `--final-eval-config PATH` | `rl`、`ga` | 自动 | `json` 模式下的合并 JSON（同时包含 stage1/stage2 两块） |
| `--manual-stage1-gelu JSON_ARRAY` | `rl`、`ga` | — | `manual` 模式下的 Stage-1 GELU 配置 |
| `--manual-stage1-softmax JSON_ARRAY` | `rl`、`ga` | — | `manual` 模式下的 Stage-1 Softmax 配置 |
| `--manual-stage2-noise JSON_OBJECT` | `rl`、`ga` | — | `manual` 模式下的 7 类 Stage-2 噪声配置；`manual` 要求三个 manual-stage* 参数同时提供 |
| `--final-eval-repeat N` | `rl`、`ga` | `1` | 统一最终评估的重复次数 |
| `--stage2-fixed-config-source stage1_result/json/manual` | `rl`、`ga` | 兼容继承 Stage-1 参数 | Stage-2 RL/GA 训练中"固定 GELU/Softmax"的来源；与最终评估的 Stage-1 配置是两套不同参数 |
| `--stage2-fixed-config PATH` | `rl`、`ga` | 自动 | `json` 模式下的 Stage-2 固定 GELU/Softmax 配置文件路径 |
| `--stage2-manual-gelu JSON_ARRAY` | `rl`、`ga` | — | `manual` 模式下的 Stage-2 固定 GELU 配置 |
| `--stage2-manual-softmax JSON_ARRAY` | `rl`、`ga` | — | `manual` 模式下的 Stage-2 固定 Softmax 配置 |
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


### 安全性约束补充（精简版）

> 命令行参数仅以上面的“可选参数总表”为主维护入口。本节只保留高风险约束，避免重复说明。

1. 以脚本校验为准：launcher 运行时校验优先于文档叙述。
2. 首次运行保护：`rl`、`ga`、`general-rl(train)` 的新参数组合必须显式传 `--fresh-start`。
3. 续训安全：持久化流程下不要手工传 `--resume-from`。
4. 跳阶段一致性：跳过某阶段时，不要再显式设置该阶段预算参数。
5. Stage-2 固定配置约束：`stage1_result/json/manual` 来源必须与前置条件匹配。
6. 对比模式隔离：`rl-and-ga-compare` 不与普通 RL/GA 搜索参数混用。
7. 对比输入完整性：`direct` 必须提供 4 个 JSON；`persistent` 必须存在目标目录与 `metadata.json`。
8. 评估来源一致性：`--final-eval-source` 必须与 `--final-eval-config` / `manual-*` 配套使用。
9. 并发写入安全：同一个持久化目录同一时刻仅允许一个训练进程写入。
10. 终止安全：优先优雅停止（SIGINT/停止标志），避免 checkpoint 状态不完整。

## 高级模块说明（非 CLI）

下面内容描述可迁移策略与通用策略能力，不展开命令行细节。

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
