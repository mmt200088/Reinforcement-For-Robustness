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

同时，final eval 已从训练 CLI 中拆成独立模块：

- 独立入口：`bash Paean/run_final_eval.sh ...`
- 独立 preset：`Paean/presets/*.conf`
- 独立输出：默认写入 `Paean/outputs/{dataset}/{algorithm}/{run}/final_eval/`
- 兼容入口：`bash llama_7B_LayerImportance.sh eval ...` 会直接转交给 `Paean/run_final_eval.sh`
- 主动调用默认只评估你指定的配置；BLB action final eval 可用 `--range truncation=8,9,11,13` 这类参数展开笛卡尔积配置网格，也支持 `block3.truncation=8,9`、`layer7.block3.truncation=11`、`layer2.block5.wffn1_sf=18` 这类细粒度 selector。
- 随机配置对照必须显式传 `--random`；`--random` 模式只能基于一个固定配置生成随机对照，不能和 `--range` 同时使用。
- 训练结束后的被动 final eval：训练进程会用 `--final-eval-preset` 指向的 `Paean/presets/*.conf` 触发一次评估；这次评估的 repeat、seed、随机对照数量、输出根目录等不再由训练命令行参数控制。BLB 训练会评估训练找到的最佳 action 配置，并按 final eval preset 的随机对照数量生成对应 random 配置。

---

## 快速开始

推荐优先使用子命令。旧版纯 flag 入口仍然兼容，但不再作为主阅读路径。

```bash
# 列出可用预设
bash llama_7B_LayerImportance.sh --list-presets

# 首次 BLB Stage-2 RL 运行：必须显式确认 fresh
bash llama_7B_LayerImportance.sh run rl --preset mrpc-blb-stage2-rl --fresh

# 续训练：同一参数组合会自动从持久化目录恢复
bash llama_7B_LayerImportance.sh run rl --preset mrpc-blb-stage2-rl

# 只跑最终评估（独立 Paean final eval 模块）
bash Paean/run_final_eval.sh --preset mrpc-final-eval-only

# 只测指定 BLB action 网格：truncation 4 个值 × wffn1 2 个值，共 8 组
bash Paean/run_final_eval.sh --preset mrpc-blb-action-range

# 细粒度 BLB action 网格：只 sweep Block3 的 truncation，并固定第 2 层 Block5 的 wffn1 scaling factor
bash Paean/run_final_eval.sh --preset mrpc-blb-baseline-fixed \
  --range block3.truncation=8,9,10,11,12,13 \
  --action-fixed layer2.block5.wffn1_sf=18

# 只测 BLB baseline 非 truncation 配置 + truncation 8/9/10/11/12/13 六档
bash Paean/run_final_eval.sh --preset mrpc-blb-baseline-truncation-sweep

# 主动 final eval 若需要随机对照，必须显式启用 --random
bash Paean/run_final_eval.sh --preset mrpc-final-eval-only --random --budget 10

# 也可以继续使用兼容入口，它会转交给 Paean/run_final_eval.sh
bash llama_7B_LayerImportance.sh eval --preset mrpc-final-eval-only

# 只跑 BLB Stage-2 最大动作 / 最大噪声配置的最终评估
bash Paean/run_final_eval.sh --preset mrpc-blb-max-final-eval

# 只重跑 Stage-2，Stage-1 固定配置自动从 --config 里取
bash llama_7B_LayerImportance.sh run ga --dataset mrpc --mode stage2-only \
  --generations 1,800 --config glue_final_configs_best_genetic.json

# RL vs GA 对比：默认从持久化目录查找结果
bash llama_7B_LayerImportance.sh compare --dataset mrpc
```

训练预设文件位于 `presets/`，final eval 预设文件位于 `Paean/presets/`。
格式都是每行一个参数，支持 `#` 注释。命令行参数排在预设之后，优先级更高。
BLB Stage-2 RL 推荐先用 `presets/mrpc-blb-stage2-rl.conf`；这份 preset 不内置 `--fresh`，
因此首次运行在命令行加 `--fresh`，后续续训练直接复用同一 preset。

BLB Stage-2 RL 的完整运行流程见 [`docs/BLB_stage2_rl_FULL_FLOW.md`](docs/BLB_stage2_rl_FULL_FLOW.md)，
简明参数说明见 [`docs/BLB_stage2_rl_README.md`](docs/BLB_stage2_rl_README.md)。

## 命令行参数总表

### 常用参数

普通用户默认只需要下面 8 个参数以内：

| 参数 | 适用子命令 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--preset NAME` | 全部 | — | 读取 `presets/NAME.conf`；命令行后续参数会覆盖预设 |
| `--dataset DATASET` | 全部 | `mrpc` | `mrpc`、`sst2`、`stsb`、`cola`、`qnli`、`rte`、`wnli` |
| `--algorithm rl/ga/greedy` | `eval`、旧版入口 | `rl` | `run` 子命令直接用 `run rl` / `run ga` / `run greedy` |
| `--fresh` | `run`、`general train` | — | 等价于 `--fresh-start`，首次训练某个参数组合时必须传 |
| `--mode train/eval/stage2-only/stage1-only/search-only` | `run`、`eval` | `train` | 高层动作，替代常见 skip 参数组合 |
| `--final-eval-preset NAME` | `run rl`、`run ga`、`run greedy` | `default` | 训练结束后被动调用 `Paean/presets/NAME.conf`；控制被动 final eval 的 repeat/seed/随机对照/输出根目录 |
| `--budget N` | `run`、独立 `eval`、`compare` | `10`；独立 `eval` 默认不启用随机对照 | 统一设置 Perm / Equiv / Budget 随机对照数量；独立 final eval 只有传 `--random` 后才会执行随机对照，并同步 Stage1Budget / Stage2Budget |
| `--eval-repeat N` | `run`、`eval`、`compare` | `50` 或 compare 的 `1` | 普通流程映射到 `--final-eval-repeat`；compare 映射到 `--stage2-compare-repeats` |
| `--batch-size N` | 全部 | `16` | 同时传给 `batch_size` 和 `micro_batch_size` |

搜索预算另有两个直观快捷参数：

| 参数 | 适用子命令 | 说明 |
| --- | --- | --- |
| `--episodes S1,S2` | `run rl` | 设置 Stage-1 / Stage-2 RL episode 数；传单个 `N` 表示两阶段相同 |
| `--generations S1,S2` | `run ga`、`run greedy` | 设置 Stage-1 / Stage-2 GA/Greedy 代数；传单个 `N` 表示两阶段相同 |

### 子命令

| 子命令 | 作用 | 常用示例 |
| --- | --- | --- |
| `run rl` | 单任务 Stage-1 + Stage-2 RL 搜索 | `bash llama_7B_LayerImportance.sh run rl --dataset mrpc --episodes 51000,80000 --fresh` |
| `run ga` | 单任务 GA 搜索 | `bash llama_7B_LayerImportance.sh run ga --dataset mrpc --generations 200,800 --fresh` |
| `run greedy` | 单任务 Greedy 搜索 | `bash llama_7B_LayerImportance.sh run greedy --dataset mrpc --generations 200,200 --fresh` |
| `eval` | 转交给独立 final eval 模块 | `bash llama_7B_LayerImportance.sh eval --preset mrpc-final-eval-only` |
| `compare` | 对比已有 RL / GA 结果 | `bash llama_7B_LayerImportance.sh compare --dataset mrpc` |
| `general train` | 训练跨任务通用策略 | `bash llama_7B_LayerImportance.sh general train --general-rl-tasks mrpc,cola,rte,stsb --fresh` |
| `general search` | 用通用策略做离线 rollout 搜索 | `bash llama_7B_LayerImportance.sh general search --dataset mrpc --general-policy-dir <dir>` |

### `--mode`

`--mode` 是对旧的 `--skip-stage1-search`、`--skip-noise-search`、`--skip-final-eval`、`--final-eval-only` 的安全封装。
新的 `eval` 子命令已经改为独立 final eval 入口；旧版纯 flag 形式 `--mode eval` 仍作为兼容路径保留。

| 模式 | 等价动作 | 典型用途 |
| --- | --- | --- |
| `train` | 不自动跳阶段 | 正常两阶段搜索 + 最终评估 |
| `eval` | 跳过 Stage-1 / Stage-2 搜索，并启用 final-eval-only | 只评估已有 JSON 或已有搜索结果 |
| `stage2-only` | 跳过 Stage-1 搜索，运行 Stage-2 搜索 | 固定已有 GELU/Softmax，只重搜噪声 |
| `stage1-only` | 运行 Stage-1，跳过 Stage-2 | 只更新 GELU/Softmax；最终评估需要 `--config` 提供 Stage-2 回退 |
| `search-only` | 跳过最终评估 | 只产出搜索结果，之后再单独 `eval` |

高级用户仍可直接使用旧的 skip 参数组合；脚本会检查冲突，避免 `--mode eval` 与 `--skip-final-eval` 这种无执行项组合。

### 高级参数附录

| 参数 | 适用子命令 | 默认值 | 说明 |
| --- | --- | --- | --- |
| **全局/兼容** | | | |
| `--search-algorithm ALG` | 旧版入口 | `rl` | 兼容旧用法；新用法建议子命令 |
| `--logfile FILE` | 全部 | `output.log` | launcher 后台日志文件名；实际路径在当前 run 的 `logs/` 目录下 |
| `--model-type TYPE` | 全部 | `bert-base` | `bert-base` / `bert-large` / `gpt-2` |
| `--resume-from PATH` | `eval` | — | 指向已有 run 目录，让 final-eval-only 从其中读取搜索结果；训练模式禁用 |
| **约束** | | | |
| `--stage1-accuracy-tolerance FLOAT` | `run`、`compare` | `0.005` | Stage-1 指标容忍比例 |
| `--stage2-limit-tolerance FLOAT` | `run`、`compare` | `0.05` | Stage-2 指标容忍比例 |
| `--stage2-stability-tolerance FLOAT` | `run`、`compare` | `0.05` | Stage-2 稳定性容忍比例 |
| `--stage2-k-trials INT` | `run`、`compare` | `5` | Stage-2 稳定性评测噪声 trial 数 |
| `--stage2-probe-size INT` | `run`、`compare` | `256` | Stage-2 稳定性评测探针子集大小 |
| **持久化/重置** | | | |
| `--fresh-start` | `run`、`general train` | — | `--fresh` 的完整写法 |
| `--fresh-stage1` | `run` | — | 仅清空 Stage-1 产物 |
| `--fresh-stage2` | `run` | — | 仅清空 Stage-2 产物 |
| **RL 搜索** | | | |
| `--stage1-search-episodes N` | `run rl` | `51000` | Stage-1 RL episode 数；推荐用 `--episodes` |
| `--stage2-search-episodes N` | `run rl` | `40000` | Stage-2 RL episode 数；推荐用 `--episodes` |
| `--ppo-update-interval N` | `run rl`、`general train` | `120` | PPO 更新间隔；BLB Stage-2 RL 下会作为默认 rollout size，general train 下也是每轮每任务 episode 数 |
| `--stage1-search-lr FLOAT` | `run rl` | `1e-4` | Stage-1 RL 学习率 |
| `--stage2-search-lr FLOAT` | `run rl` | `1e-4` | Stage-2 RL 学习率 |
| **Stage-2 RL variant** | | | |
| `--stage2-rl-variant blb_v3/legacy_v2` | `run rl` | `blb_v3` | 选择 Stage-2 RL 实现：`blb_v3`（默认，加强版 BLB Stage-2 RL，覆盖 Block 1-5 + first-input fresh 全部噪声候选点；详见 `docs/BLB_stage2_rl_FULL_FLOW.md`）；`legacy_v2`（旧版 `noise_rl_module_v2`，仅优化 `*_scaling_factors`） |
| `--stage2-rescale-invoker heuristic/subprocess/stub` / `--blb-v3-rescale-invoker-kind heuristic/subprocess/stub` | `run rl` | `heuristic` | BLB v3 的 Rescale_optimizer 调用方式；缺省 `heuristic` 使用内置启发式估计（不依赖外部 Rescale_optimizer 子项目） |
| `--stage2-rescale-root PATH` / `--blb-v3-subprocess-optimizer-root PATH` | `run rl` | — | invoker=`subprocess` 时 Rescale_optimizer 子项目根目录；当前还需内部补齐 configs 与 baseline archive，否则会 fallback 到 `heuristic` |
| `--stage2-rescale-cli-module MODULE` / `--blb-v3-subprocess-cli-module MODULE` | `run rl` | `rescale_optimizer.replan` | invoker=`subprocess` 时调用的 CLI module |
| `--stage2-rollout-size N` / `--blb-v3-rollout-size N` | `run rl` | 跟随 `--ppo-update-interval` | BLB v3 PPO rollout 大小（多少 episode 触发一次 PPO update） |
| `--stage2-save-interval N` / `--blb-v3-save-interval N` | `run rl` | `200` | BLB v3 live checkpoint 保存间隔 |
| `--stage2-eval-interval N` / `--blb-v3-eval-interval N` | `run rl` | `100` | BLB v3 训练日志评估间隔 |
| `--stage2-calibrate-baseline-samples N` / `--blb-v3-calibrate-baseline-samples N` | `run rl` | `8` | BLB v3 reward 权重校准样本数 |
| **GA / Greedy 搜索** | | | |
| `--stage1-search-generations N` | `run ga`、`run greedy` | 自动 | Stage-1 代数；推荐用 `--generations` |
| `--stage2-search-generations N` | `run ga`、`run greedy` | 自动 | Stage-2 代数；推荐用 `--generations` |
| **高级阶段控制** | | | |
| `--skip-stage1-search` | `run` | — | 高级兼容入口；一般用 `--mode stage2-only` |
| `--skip-noise-search` | `run` | — | 高级兼容入口；一般用 `--mode stage1-only` |
| `--skip-final-eval` | `run` | — | 高级兼容入口；一般用 `--mode search-only` |
| `--final-eval-only` | `run` / `eval` | — | 高级兼容入口；一般用 `--mode eval` 或 `eval` 子命令 |
| `--final-eval-preset NAME` | `run` | `default` | 普通训练完成后，被动调用 `Paean/presets/NAME.conf`；训练命令行中的 `--random-seed`、`--budget`、`--final-eval-repeat` 不控制这次被动评估 |
| **最终评估** | | | |
| `--final-eval-source search/json/manual/max` | `run`、兼容 `--mode eval` | `search` | 兼容参数；普通训练完成后的被动 final eval 会强制使用训练刚找到的 search 配置，评估细节由 `--final-eval-preset` 指向的 preset 控制 |
| `--source search/json/manual/max` | `run`、`eval` | 同上 | `--final-eval-source` 的短写 |
| `--final-eval-config PATH` | `run`、`eval` | 按算法自动 | 合并 JSON；短写是 `--config` |
| `--config PATH` | `run`、`eval` | — | 同时用于 final-eval；`stage2-only` 下也自动作为 Stage-2 固定配置来源 |
| `--manual-stage1-gelu JSON_ARRAY` | `run`、`eval` | — | `manual` 来源下的 GELU 配置 |
| `--manual-stage1-softmax JSON_ARRAY` | `run`、`eval` | — | `manual` 来源下的 Softmax 配置 |
| `--manual-stage2-noise JSON_OBJECT` | `run`、`eval` | — | `manual` 来源下的 Stage-2 噪声配置 |
| `--final-eval-repeat N` | `run`、`eval` | `50` | 正式重复评估次数；推荐用 `--eval-repeat` |
| `--random-seed N` | `run`、`eval`、`compare` | `42` | 随机种子 |
| `--random` / `--enable-random` | 独立 `eval` | 关闭 | 主动 final eval 默认不做随机对照；启用后才使用下面的 trial 数量 |
| `--action-config PATH` | 独立 `eval` | — | BLB action final eval 配置 JSON，可指定固定 action、`fixed` 和 `ranges` |
| `--range NAME=V1,V2,...` / `--action-range ...` | 独立 `eval` | — | BLB action 网格展开参数，可重复传；多个 range 做笛卡尔积。`NAME` 支持全局、per-block、per-layer 精确 selector |
| `--action-fixed NAME=V` | 独立 `eval` | — | BLB action 固定覆写，可重复传。`NAME` 支持全局、per-block、per-layer 精确 selector |
| `--perm-trials N` | `run`、独立 `eval`、`compare` | 训练/compare 为 `10`；独立 `eval` 为 `0` | permutation 对照数量；独立 `eval` 需配合 `--random` |
| `--cost-trials N` | `run`、独立 `eval`、`compare` | 训练/compare 为 `10`；独立 `eval` 为 `0` | 等价成本对照数量；独立 `eval` 需配合 `--random` |
| `--budget-trials N` | `run`、独立 `eval`、`compare` | 训练/compare 为 `10`；独立 `eval` 为 `0` | 等价预算对照数量；独立 `eval` 需配合 `--random` |
| `--stage1-budget-trials N` | 独立 `eval` | `0` | Stage1Budget 数量；传 `--random` 且未指定任何 trial 时自动回到 `10` |
| `--stage2-budget-trials N` | 独立 `eval` | `0` | Stage2Budget 数量；传 `--random` 且未指定任何 trial 时自动回到 `10` |
| **Stage-2 固定 Stage-1 配置** | | | |
| `--stage2-fixed-config-source stage1_result/json/manual` | `run` | 自动 | 一般不需要手动传；`stage2-only --config` 会自动用 `json` |
| `--stage2-fixed-config PATH` | `run` | 自动 | JSON 来源时的固定 GELU/Softmax 文件 |
| `--stage2-manual-gelu JSON_ARRAY` | `run` | — | Stage-2 固定配置 manual GELU |
| `--stage2-manual-softmax JSON_ARRAY` | `run` | — | Stage-2 固定配置 manual Softmax |
| **Compare** | | | |
| `--compare-config-mode persistent/direct` | `compare` | `persistent` | 默认从 `rl_results/persistent` 自动定位 RL/GA；`direct` 需要四个 JSON |
| `--compare-persistent-root PATH` | `compare` | `rl_results/persistent` | persistent 模式的根目录 |
| `--stage2-compare-repeats N` | `compare` | `1` | compare 的重复评估次数；推荐用 `--eval-repeat` |
| `--rl-compare-stage1-json PATH` | `compare --compare-config-mode direct` | — | direct 模式 RL Stage-1 JSON |
| `--rl-compare-stage2-json PATH` | `compare --compare-config-mode direct` | — | direct 模式 RL Stage-2 JSON |
| `--ga-compare-stage1-json PATH` | `compare --compare-config-mode direct` | — | direct 模式 GA Stage-1 JSON |
| `--ga-compare-stage2-json PATH` | `compare --compare-config-mode direct` | — | direct 模式 GA Stage-2 JSON |
| `--rl-compare-stage1-accuracy-tolerance FLOAT` | `compare persistent` | 继承全局约束 | RL 侧目录定位约束 |
| `--rl-compare-stage2-limit-tolerance FLOAT` | `compare persistent` | 继承全局约束 | RL 侧目录定位约束 |
| `--rl-compare-stage2-stability-tolerance FLOAT` | `compare persistent` | 继承全局约束 | RL 侧目录定位约束 |
| `--ga-compare-stage1-accuracy-tolerance FLOAT` | `compare persistent` | 继承全局约束 | GA 侧目录定位约束 |
| `--ga-compare-stage2-limit-tolerance FLOAT` | `compare persistent` | 继承全局约束 | GA 侧目录定位约束 |
| `--ga-compare-stage2-stability-tolerance FLOAT` | `compare persistent` | 继承全局约束 | GA 侧目录定位约束 |
| **General-RL** | | | |
| `--general-rl-tasks T1,T2,...` | `general train` | 同 `--dataset` | 训练任务列表 |
| `--general-rl-rounds N` | `general train` | `50` | round-robin 轮数 |
| `--general-rl-lr FLOAT` | `general train` | `3e-5` | 通用策略学习率 |
| `--general-rl-stage1-config-json PATH` | `general train` | — | Stage-2 训练时各任务的 Stage-1 配置 |
| `--general-rl-accuracy-tolerances T1,T2,...` | `general train/search` | — | 离散准确度容忍列表；search 取第一个 |
| `--general-rl-accuracy-tolerance-range MIN,MAX` | `general train` | — | 连续准确度容忍范围 |
| `--general-policy-dir PATH` | `general search` | — | 自动推导 `general_stage1_policy.pt` 与可选 Stage-2 policy |
| `--general-stage1-policy PATH` | `general search` | — | 显式指定 Stage-1 通用策略 |
| `--general-stage2-policy PATH` | `general search` | — | 显式指定 Stage-2 通用噪声策略 |
| `--general-rl-num-rollouts N` | `general search` | `500` | 离线 rollout 次数 |
| `--general-rl-greedy` | `general search` | — | 使用贪心 rollout |
| `--general-rl-skip-stage2` | `general train/search` | — | 跳过 Stage-2 |

### 独立 Paean final eval 参数

独立 final eval 入口支持 `bash Paean/run_final_eval.sh ...`，也支持兼容入口
`bash llama_7B_LayerImportance.sh eval ...`。它拥有自己的 preset 目录 `Paean/presets/`，不读取训练 preset。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--preset NAME` | — | 读取 `Paean/presets/NAME.conf` |
| `--list-presets` | — | 列出 final_eval 可用 preset |
| `--dataset DATASET` | `mrpc` | `mrpc`、`sst2`、`stsb`、`cola`、`qnli`、`rte`、`wnli` |
| `--algorithm rl/ga/greedy` | `rl` | 选择用 RL、GA 或 Greedy 家族的结果/JSON |
| `--model-type bert-base/bert-large/gpt-2` | `bert-base` | 模型类型 |
| `--batch-size N` | `16` | final eval 批大小 |
| `--source search/json/manual/max/blb-max` | `json` | 配置来源；`search` 需要 `--resume-from` 指向已有训练目录 |
| `--config PATH` / `--final-eval-config PATH` | 按算法自动 | 合并 JSON 配置文件 |
| `--resume-from PATH` | — | `--source search` 时读取已有训练目录中的搜索结果 |
| `--output-root PATH` | `Paean/outputs` | 独立 final eval 输出根目录 |
| `--run-name NAME` | 时间戳或训练目录名 | 输出目录名 |
| `--repeat N` / `--eval-repeat N` | `50` | 被选配置和随机对照组重复评估次数 |
| `--logfile FILE` | `final_eval.log` | 后台运行时写入输出目录下的 `logs/FILE`；终端只打印 PID 和 tail 命令 |
| `--foreground` | 关闭 | 调试用：不后台化，直接在当前终端运行 |
| `--random-seed N` | `42` | final eval 随机种子 |
| `--random` / `--enable-random` | 关闭 | 启用随机配置对照；不传时只评估选中的固定配置或 range 网格 |
| `--perm-trials N` | `0` | Perm 对照组数量；只有 `--random` 模式才允许非零 |
| `--cost-trials N` | `0` | Equiv 对照组数量；只有 `--random` 模式才允许非零 |
| `--budget-trials N` | `0` | Budget 对照组数量；只有 `--random` 模式才允许非零 |
| `--budget N` | — | 同时设置 Perm / Equiv / Budget / Stage1Budget / Stage2Budget 数量；独立 final eval 需配合 `--random` |
| `--stage1-budget-trials N` | `0` | Stage1Budget 对照组数量；`--random` 且未设置任何 trial 时自动取 `10` |
| `--stage2-budget-trials N` | `0` | Stage2Budget 对照组数量；`--random` 且未设置任何 trial 时自动取 `10` |
| `--stage1-accuracy-tolerance FLOAT` | `0.005` | Stage-1 约束比例 |
| `--stage2-limit-tolerance FLOAT` | `0.05` | Stage-2 指标约束比例 |
| `--stage2-stability-tolerance FLOAT` | `0.05` | Stage-2 稳定性约束比例 |
| `--stage2-k-trials N` | `5` | Stage-2 稳定性噪声 trial 数 |
| `--stage2-probe-size N` | `256` | Stage-2 稳定性探针子集大小 |
| `--stage2-rl-variant blb_v3/legacy_v2` | `blb_v3` | RL final eval 下用于兼容 BLB/legacy 行为 |
| `--action-config PATH` | — | BLB action final eval 的 JSON 配置；可包含 `action_vec` / `base_action_vec`、`fixed`、`ranges` |
| `--range NAME=V1,V2,...` / `--action-range ...` | — | BLB action range，可重复传；多个 range 做笛卡尔积，如 `truncation=8,9,11,13` 与 `wffn1=18,20` 会测试 8 组；也可写 `block3.truncation=8,9` 或 `layer7.block5.wffn1_sf=18,20` |
| `--action-fixed NAME=V` / `--fixed-action NAME=V` | — | 固定某个 BLB action 维度，可重复传；支持 `blockN.FIELD` 和 `layerI.blockN.FIELD` |
| `--rescale-invoker-kind heuristic/in_process/subprocess/stub` | `heuristic` | BLB action final eval 调用 Rescale_optimizer 的方式；正式实验推荐 `in_process` |
| `--rescale-optimizer-root PATH` | — | `in_process` / `subprocess` 模式下的 `Rescale_optimizer` 根目录 |
| `--require-rescale-optimizer` | 关闭 | 要求使用真实 Rescale_optimizer；初始化失败时直接报错，不 fallback 到 heuristic |
| `--manual-stage1-gelu JSON_ARRAY` | — | `--source manual` 时的 Stage-1 GELU |
| `--manual-stage1-softmax JSON_ARRAY` | — | `--source manual` 时的 Stage-1 Softmax |
| `--manual-stage2-noise JSON_OBJECT` | — | `--source manual` 时的 Stage-2 噪声配置 |
| `--dry-run` | — | 只打印将执行的底层 Python 命令，不加载模型 |

BLB action selector 支持三种粒度：

- 全局字段：`truncation`、`wffn1`、`wffn1_rescale`、`wffn2`、`first_input`，会作用到所有匹配位置。
- Per-block：`block1.truncation`、`block3.output_truncation_k`、`block5.wffn1_sf`，只作用到指定 block 的所有层。
- Per-layer + per-block：`layer0.block1.truncation`、`layer7.block3.truncation`、`layer2.block5.wffn1_sf`，只作用到一个精确 action 点。

字段名使用 `blb_stage2_rl/action_space.py` 中各 block 的 action 字段名；`truncation` / `k` 是 `output_truncation_k` 的别名。主动调用不传 `--random` 时只测试这些选中配置；传 `--random` 时必须使用一个固定配置，不能再传 `--range`。BLB action final eval 的 Markdown/JSON 结果会同时写出完整 `full_noise_config`，逐层逐 block 列出每个 scaling factor 点和 truncation 点的实际值。


### 数据集补充说明（精简）

1. 当前 launcher 支持的数据集：`mrpc`、`sst2`、`stsb`、`cola`、`qnli`、`rte`、`wnli`。
2. 任务类型：
`stsb` 是回归任务（`num_labels=1`）；其余为二分类任务（`num_labels=2`）。
3. 输入字段约定：
`sst2/cola` 使用 `sentence`；`qnli` 使用 `question + sentence`；`mrpc/stsb/rte/wnli` 使用 `sentence1 + sentence2`。
4. `--model-type` 与数据集兼容性：
`bert-base` 支持全部 7 个任务；`bert-large` 当前不支持 `wnli`；`gpt-2` 支持全部 7 个任务。
5. 若数据集与模型类型不兼容，launcher 会在启动前直接报错并中止，不会进入训练阶段。

### 安全性约束补充（精简版）

> 本节只保留高风险约束；完整参数见上面的常用参数与高级参数附录。

1. 以脚本校验为准：launcher 运行时校验优先于文档叙述。
2. 首次运行保护：`run rl`、`run ga`、`run greedy`、`general train` 的新参数组合必须显式传 `--fresh` 或 `--fresh-start`。
3. 续训安全：训练流程按持久化目录自动续训；`--resume-from` 只建议在 `eval` / `--mode eval` 下使用。
4. 跳阶段一致性：跳过某阶段时，不要再显式设置该阶段预算参数。
5. Stage-2 固定配置约束：`stage2-only --config` 会自动推断为 JSON 来源；手动使用 `stage1_result/json/manual` 时仍必须与前置条件匹配。
6. 对比模式隔离：`compare` 不与普通 RL/GA/Greedy 搜索参数混用。
7. 对比输入完整性：默认 `persistent` 必须存在目标目录与 `metadata.json`；高级 `direct` 必须提供 4 个 JSON。
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

# 查看最近一次后台任务
cat rl_results/LATEST_PID
cat rl_results/LATEST_RUN_DIR

# 推荐优雅停止
kill -INT "$(cat rl_results/LATEST_PID)"
