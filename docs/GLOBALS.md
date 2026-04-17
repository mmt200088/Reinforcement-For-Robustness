# 全局部件清单 / Global Parts Registry

> 这个文档的目的：**你动其中任何一项，就要知道会影响到谁。**
>
> 所谓"全局部件"指的是——**一处修改会跨多个文件 / 跨多次运行产生影响的东西**：
> - 写死的路径 / 文件名 / 目录名
> - 跨模块共享的算法超参 / 维度常量
> - 跨模块共享的结构（dict key 顺序、action 空间）
> - 一次性的硬编码逻辑（sys.path.append、魔法数字）
> - 已产生的结果目录里固化的命名约定
>
> **集中化入口**：[config/paths.py](../config/paths.py) 和 [config/constants.py](../config/constants.py)
> 已经把"应该集中"的内容聚合进去了，但 Phase-1 阶段**只是文档性质的同步**——
> 各模块里仍然保留自己的定义。Phase-3 会把各模块改成从 `config/` 导入。

---

## 目录

- [A. 写死的路径 / 目录名 / 文件名](#a-写死的路径--目录名--文件名)
- [B. 跨模块共享的算法常量](#b-跨模块共享的算法常量)
- [C. 跨模块的数据结构约定](#c-跨模块的数据结构约定)
- [D. Opt-flag 字典（消融开关）](#d-opt-flag-字典消融开关)
- [E. 循环依赖 / 相互 import](#e-循环依赖--相互-import)
- [F. sys.path.append 魔法 & 其他隐藏耦合](#f-syspathappend-魔法--其他隐藏耦合)
- [G. 已产生的结果里固化的命名](#g-已产生的结果里固化的命名)
- [H. Phase-3 迁移路线图](#h-phase-3-迁移路线图)

---

## A. 写死的路径 / 目录名 / 文件名

> 所有"真相来源"列出的位置都已经在 [config/paths.py](../config/paths.py) 里
> 重新声明了一遍。**改动**时必须同时更新所有"同步点"，否则会出现路径不一致 bug。

### A.1 结果根目录

| 变量 | 值 | 真相来源 | 还出现在 |
|---|---|---|---|
| `RL_RESULTS_ROOT` | `rl_results` | `layer_importance_evaluator.py:160-162`, `noise_rl_module_v2.py:191`, `final_evaluation_module.py:36`, `rl_tune_general.py:17` 注释 | `rl_ga_compare_runner.py:2379` (`--compare-persistent-root rl_results/persistent`) |
| `EXPERIMENT_RESULTS_ROOT` | `experiment_results` | `experiment_noise_scaling_sweep.py:279` | `experiment_block*.py`（隐含） |
| `GLUE_SUBMISSION_ROOT` | `glue_submission` | `generate_glue_submission.py:955-964` | README |
| `GLUE_BASELINE_ROOT` / `GLUE_APPROX_ROOT` | `glue_baseline` / `glue_approx` | `generate_glue_submission.py:82-85` | README |

### A.2 rl_results 下的二级目录名

| 变量 | 值 | 真相来源（单一） | 备注 |
|---|---|---|---|
| `PERSISTENT_SUBDIR` | `persistent` | 多处字符串拼接 | 没有集中定义，Phase-3 必须统一 |
| `RUNS_SUBDIR` | `runs` | 多处 | 同上 |
| `COMPARE_RL_VS_GA_SUBDIR` | `compare/rl_vs_ga` | `rl_ga_compare_runner.py` | 同上 |
| `NOISE_RL_PROGRESS_SUBDIR` | `rl_results/noise_rl_progress` | `noise_rl_module_v2.py:191` | 被 `rl_tune_genetic.py` 复用 |
| `STAGE1_FINAL_EVAL_SUBDIR` | `rl_results/final_evaluation` | `layer_importance_evaluator.py:160` | `final_evaluation_module.py:36` 重复声明（必须同步） |
| `NOISE_FINAL_EVAL_SUBDIR` | `rl_results/noise_final_evaluation` | `layer_importance_evaluator.py:162` | `noise_final_evaluation_module.py:83` 重复声明（必须同步） |

### A.3 标签文件

| 文件名 | 用途 | 生产者 | 消费者 |
|---|---|---|---|
| `LATEST_PID` | 记录最近一次训练进程 PID | `layer_importance_evaluator`（update_persistent_metadata_stage） | `rl_ga_compare_runner`（process guard） |
| `LATEST_RUN_DIR` | 最近一次运行目录名 | 同上 | 同上 + `status_board.py` |
| `LATEST_COMPARE_PID` | 对比运行 PID | `rl_ga_compare_runner` | 同上 |
| `LATEST_COMPARE_RUN_DIR` | 对比运行目录 | 同上 | `status_board.py` |

### A.4 checkpoint / 持久化文件名

| 变量 | 值 | 重复定义的位置 |
|---|---|---|
| `STAGE1_RL_CHECKPOINT_FILENAME` | `stage1_rl_checkpoint.pt` | `noise_rl_module_v2.py:358` + `rl_ga_compare_runner.py:24` ⚠️ |
| `NOISE_STAGE_CHECKPOINT_FILENAME` | `noise_rl_checkpoint.pt` | `noise_rl_module_v2.py:357` + `rl_ga_compare_runner.py:25` ⚠️ |
| `GA_STAGE1_CHECKPOINT_FILENAME` | `ga_stage1_checkpoint.pt` | `genetic_search_module.py:113` + `rl_ga_compare_runner.py:26` ⚠️ |
| `GA_STAGE2_CHECKPOINT_FILENAME` | `ga_stage2_checkpoint.pt` | `genetic_search_module.py:114` + `rl_ga_compare_runner.py:27` ⚠️ |
| `GENERAL_STAGE1_TRAIN_CHECKPOINT` | `general_stage1_train_checkpoint.pt` | `general_policy_module.py:103` |
| `GENERAL_STAGE2_TRAIN_CHECKPOINT` | `general_stage2_train_checkpoint.pt` | `general_policy_module.py:104` |

⚠️ 的行表示**同一个字符串在两个文件里写死**——其中任何一处改了另一处没改，
对比运行就会找不到 checkpoint 而静默失败。

### A.5 JSON 配置文件

这些 JSON 在项目根，多个 CLI 默认读取：

| 文件 | 读取方 | 写入方 |
|---|---|---|
| `glue_configs_best_ppo.json` | `generate_glue_submission.py`, `final_evaluation_module.py` | RL Stage-1 训练完成后写 |
| `glue_configs_best_genetic.json` | 同上 | GA Stage-1 完成后写 |
| `glue_noise_configs_best_ppo.json` | `generate_glue_submission.py`, `noise_final_evaluation_module.py` | RL Stage-2 完成后写 |
| `glue_noise_configs_best_genetic.json` | 同上 | GA Stage-2 完成后写 |
| `glue_configs.json` | `generate_glue_submission.py`（未 `--config` 时兜底） | 手维护 |

---

## B. 跨模块共享的算法常量

### B.1 action 空间映射（Stage-1）

```
GELU_MAP       = {0: 4, 1: 2, 2: 1, 3: 0}
GELU_COST      = {4: 3.0, 2: 2.5, 1: 1.0, 0: -1.0}
SOFTMAX_MAP    = {0: 6, 1: 5, 2: 4, 3: 3, 4: 2}
SOFTMAX_COST   = {6: 3.0, 5: 2.5, 4: 2.0, 3: 1.5, 2: 1.0}
```

**真相来源**：`layer_importance_evaluator.py:32-35`
**被依赖**：PPO 网络输出层 size、环境动作解码、配置 JSON 反查、GA 染色体编码
**影响**：改了这里 → 所有**旧 checkpoint 的动作分布 head 作废**

### B.2 action 空间映射（Stage-2 噪声）

```
INPUT_NOISE_ALLOWED_SCALING_FACTORS   = (22, 24, 26, 28, 30)
WEIGHT_NOISE_ALLOWED_SCALING_FACTORS  = (14, 16, 18, 20, 22)
WFFN1_NOISE_ALLOWED_SCALING_FACTORS   = (16, 18, 20, 22, 24)
NOISE_KEYS                            = ('x','wq','wk','wv','wo','wffn1','wffn2')
NOISE_STAGE_NUM_ACTIONS               = 7
```

**真相来源**：`function_handler.py:106-111`、`generate_glue_submission.py:151`、
`layer_importance_evaluator.py:127`
**被依赖**：几乎所有核心模块都导入
**影响**：动 `NOISE_KEYS` 的顺序 → GLUE 提交生成、metric 打印、报告对齐全部错位

### B.3 Stage-2 PPO 超参

```
NOISE_STAGE_PPO_MAX_EPISODES   = 40000
NOISE_STAGE_PPO_EPS_CLIP       = 0.12
NOISE_STAGE_PPO_K_EPOCHS       = 6
NOISE_STAGE_PPO_GAMMA          = 1.0
NOISE_STAGE_VALUE_CLIP_RANGE   = 1.0
```

**真相来源**：`noise_rl_module_v2.py:125-138`
**影响**：所有 Stage-2 训练曲线直接会变；不影响 Stage-1。

### B.4 GTrXL 网络结构

```
NOISE_STAGE_GTRXL_D_MODEL   = 256
NOISE_STAGE_GTRXL_N_HEADS   = 8
NOISE_STAGE_GTRXL_N_LAYERS  = 4
NOISE_STAGE_GTRXL_D_FF      = 512
NOISE_STAGE_GTRXL_DROPOUT   = 0.1
```

**真相来源**：`noise_rl_module_v2.py:125-129`（Stage-1 的同名常量另藏在
`layer_importance_evaluator.py`）
**影响**：动了 → **所有旧 checkpoint 直接加载失败**

### B.5 GA 超参

```
SCORE_EXP_BASE                  = 4.0
MAX_PENALTY_EXP_ARG             = 700.0
STAGE1_DEFAULT_POPULATION       = 32
STAGE2_DEFAULT_POPULATION       = 32
STAGNATION_TOLERANCE_BASE       = 10
STAGE1_GA_CONSTRAINT_RATIO      = 0.005
GA_STAGE1_ALLOWED_GELU_DEGREES  = (4, 2, 1)
```

**真相来源**：`genetic_search_module.py:59-66`
**影响**：GA 搜索轨迹会变；不影响 RL 与 compare。

### B.6 通用策略常量

```
TASK_CONTEXT_DIM                      = 5   # task embedding 维度
GENERAL_POLICY_VERSION                = 1
GENERAL_STAGE1_ALLOWED_GELU_DEGREES   = (4, 2, 1)
```

**真相来源**：`general_policy_module.py:109-111`
**影响**：Phase-A 训练的 `.pt` 和 Phase-B 离线推理**必须维度一致**，动这里要重训。

### B.7 数据切分 / 种子

```
RL_DATASET_SPLIT_SEED = 42
```

**真相来源**：`layer_importance_evaluator.py:723`
**影响**：所有 RL 训练 / 验证集切分都锁在这个 seed；改后旧 checkpoint 的 val
曲线不可比。

---

## C. 跨模块的数据结构约定

### C.1 `stage_status` 字典（持久化 metadata.json）

```json
{
  "stage1_search": "not_started | running | completed | skipped",
  "stage1_final_eval": "...",
  "stage2_search": "...",
  "stage2_final_eval": "..."
}
```

**生产者**：`layer_importance_evaluator.update_persistent_metadata_stage`、
`genetic_search_module`（GA 分支）、`noise_rl_module_v2`（Stage-2 分支）
**消费者**：`rl_ga_compare_runner`（判断是否跳过）、`status_board.py`

### C.2 `compare_final_status.json`

见 [ARCHITECTURE.md §4.2](ARCHITECTURE.md#42-一次性对比-rl_resultsrunscompererl_vs_ga)。
关键字段：`rl.state`, `ga.state`, `stage{1,2}_final_eval_ready`,
`stage{1,2}_report_path`。

**生产者**：`rl_ga_compare_runner`
**消费者**：`status_board.py`

### C.3 `NOISE_KEYS` 顺序（已在 B.2 提过）

这是一个**事实上的契约**：所有"噪声 scaling vector"在序列化成 JSON / 打印时
都按这个顺序。目前没有 runtime 校验。

---

## D. Opt-flag 字典（消融开关）

这些字典不是简单的常量，而是一批 **True/False 开关**，改一个会让算法的一个
环节切回旧逻辑。**最适合记住位置，不适合随手改**。

| 字典 | 位置 | 作用 |
|---|---|---|
| `RL_OPT_FLAGS` | `layer_importance_evaluator.py:485` | Stage-1 PPO 的各类优化开关，含 `stage1_pretrained_policy_path`（用于加载通用策略） |
| `NOISE_RL_OPT_FLAGS` | `noise_rl_module_v2.py:211` | Stage-2 PPO 的各类优化开关，含 `pretrained_policy_path`、熵下界覆盖、KL early-stop 等 |

---

## E. 循环依赖 / 相互 import

### ⚠️ `layer_importance_evaluator` ↔ `noise_rl_module_v2`

- `layer_importance_evaluator.py:25`：
  `from noise_rl_module_v2 import _log_rounded_box, _progress_bar, _fmt_elapsed, NOISE_RL_PROGRESS_BOX_PPO_INTERVAL`
- `noise_rl_module_v2.py:1984, 3072, 3246`：
  `from layer_importance_evaluator import LayerImportanceEvaluator, update_persistent_metadata_stage, detect_rl_local_optimum`

目前能跑是因为 `noise_rl_module_v2` 的反向 import 都**在函数体内**（延迟 import），
避免了 top-level 循环。**别把这些延迟 import 提到文件顶部**，否则 import 链直接死锁。

### `layer_importance_evaluator` / `noise_rl_module_v2` / `genetic_search_module`
都依赖 `function_handler`、`final_evaluation_module`、`noise_final_evaluation_module`。
这三个下层模块**没有反向 import 上层**，是干净的。

---

## F. sys.path.append 魔法 & 其他隐藏耦合

### F.1 PEFT 本地路径

```python
sys.path.append(os.path.join(os.getcwd(), "./importance-aware-sparse-tuning-IST-paper/peft/src/"))
```

- 出现在 `rl_tune.py:17`、`rl_tune_general.py:41`、`rl_tune_genetic.py:18`、`generate_glue_submission.py`
- **依赖 cwd**：必须从项目根运行这些 CLI，否则 path 不对
- 对应的子目录 `importance-aware-sparse-tuning-IST-paper/peft/` 是一份 PEFT 源码拷贝
- **Phase-3 目标**：用 `pip install -e ./importance-aware-sparse-tuning-IST-paper/peft` 或
  环境变量 `PYTHONPATH` 替代

### F.2 `TARGET_MODULES_LITERAL`

```python
TARGET_MODULES_LITERAL = '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]'
```
- `rl_ga_compare_runner.py:23`
- 以**字符串**形式传给子进程 CLI（LLaMA 训练）
- 这是 LLaMA 的投影名，改成 BERT 要换成对应名字；目前没有 per-model dispatch

### F.3 shell 脚本里的写死路径

`llama_7B_LayerImportance.sh`（81KB）和 `run_*.sh` 里大量写死了：
- 根目录下的 Python 入口文件名
- `rl_results/persistent/...` 路径
- JSON 配置文件名

搬动任何一个 Python 文件都必须同步改这些脚本。

---

## G. 已产生的结果里固化的命名

**已存在于磁盘上的目录名** = 也是一种"全局部件"：改了命名约定，旧 run 识别不了。

目前已固化的命名：
- `rl_results/persistent/rl/bert-base/mrpc/s1t0.005_s2q0.05_s2sq0.05/` — slug 拼接规则
- `rl_results/persistent/ga/bert-base/mrpc/s1t0.005_s2q0.05_s2sq0.05/` — 对应 GA
- `rl_results/runs/compare/rl_vs_ga/mrpc/comp_1/`、`comp_2/`、
  `20260413_213848_pid1143993/` — 自增编号 OR 时间戳+PID
- `rl_results/第二阶段强化学习测试阶段/` — **中文目录名**（历史遗留，未来建议迁到英文）

---

## H. Phase-3 迁移路线图

**目标**：让所有 A/B 项有**唯一**定义，其他模块只导入不硬编码。

| 批次 | 动作 | 风险 | 验证 |
|---|---|---|---|
| 3.1 | 把 `STAGE1_RL_CHECKPOINT_FILENAME` / `NOISE_STAGE_CHECKPOINT_FILENAME` / `GA_STAGE*_CHECKPOINT_FILENAME` 从重复定义改为在 `noise_rl_module_v2` / `genetic_search_module` 里定义，`rl_ga_compare_runner` 只 import | 低 | 跑一次 compare 全流程 |
| 3.2 | 把 `NOISE_STAGE_PROGRESS_DIR` / `STAGE1_FINAL_EVAL_SUBDIR` / `NOISE_FINAL_EVAL_SUBDIR` 改为 `from config.paths import ...` | 低 | `tests/test_output_layout_regression.py` |
| 3.3 | 把 `PEFT` 路径改用 `PYTHONPATH` / `setup.py`，去掉所有 `sys.path.append` | 中（可能影响 CI） | 全部 CLI 冒烟跑一次 |
| 3.4 | 把 `NUM_LAYERS=12` / `NOISE_KEYS` / 各动作映射改为 `from config.constants import ...` | 中（很多文件要改） | pytest 全跑 |
| 3.5 | （可选）把 `RL_OPT_FLAGS` / `NOISE_RL_OPT_FLAGS` 搬到 `config/flags.py`，加 pytest 覆盖 | 高 | 需要单独设计 |

每一批次完成后都**更新本文档的"真相来源"栏**，保证它不和代码脱节。
