# 项目架构速览

> 本文档是**读代码之前的入门向导**。目标：任何人（或未来的你）读完后应该知道
> - 每个文件大致在干什么
> - 哪些是 CLI 入口、哪些是核心库、哪些是一次性脚本
> - 跨文件的依赖方向
> - 结果/输出写到哪里

配套文档：
- [GLOBALS.md](GLOBALS.md) — 全局常量 / 写死路径 / 跨文件耦合点清单
- [REORG_PLAN.md](REORG_PLAN.md) — 下一步（Phase 2）的目录重构计划
- [STATUS.md](STATUS.md) — 由 `tools/status_board.py` 生成的 RL/GA/General-RL/Compare 任务进度总板

---

## 1. 目录地图（当前 / Phase 1 后）

```
Local_program/
├── config/                ★ 新：集中化配置入口（仅常量定义，无 IO）
│   ├── __init__.py
│   ├── paths.py           # 所有写死的相对路径 / 目录名 / 文件名
│   └── constants.py       # 所有全局算法 / 模型超参
│
├── docs/                  ★ 新：所有文档 / 说明 / 静态资料
│   ├── ARCHITECTURE.md    # 本文件
│   ├── GLOBALS.md         # 全局部件清单
│   ├── REORG_PLAN.md      # Phase-2 目录重构方案
│   ├── STATUS.md          # 自动生成的进度总板（tools/status_board.py）
│   └── assets/            # 论文、结构图、Excel 等参考资料
│
├── tools/                 ★ 新：辅助工具（非训练核心逻辑）
│   └── status_board.py    # 扫描 rl_results/ 生成进度总板
│
├── presets/               # bash 脚本使用的 .conf 预设
├── tests/                 # pytest 测试（import 路径基于项目根）
├── rl_results/            # 所有 RL/GA/General-RL/Compare 结果的根
│   ├── persistent/
│   │   ├── rl/{model}/{task}/{slug}/        # Stage-1/Stage-2 RL 续训
│   │   ├── ga/{model}/{task}/{slug}/        # Stage-1/Stage-2 GA 续搜
│   │   └── general-rl/{model}/{taskset}/... # 跨任务通用策略
│   ├── runs/
│   │   └── compare/rl_vs_ga/{task}/comp_*/  # 一次性 RL-vs-GA 对比
│   └── noise_rl_progress/                   # Stage-2 训练过程日志（临时）
│
├── experiment_results/    # 一次性实验脚本（experiment_block*, noise_scaling_sweep）
├── glue_submission/       # generate_glue_submission.py 的输出
├── glue_approx/, glue_baseline/
│
├── rl_tune.py              # CLI: 单任务 Stage-1/Stage-2 RL
├── rl_tune_general.py      # CLI: 通用 policy 跨任务训练 + 离线搜索
├── rl_tune_genetic.py      # CLI: GA 搜索
├── rl_ga_compare_runner.py # CLI: 并行跑 RL 与 GA 并比对
├── generate_glue_submission.py # CLI: 生成 GLUE 提交
├── runtime_error_reporter.py   # 小工具：fire 入口 + 错误摘要
│
├── layer_importance_evaluator.py  # 【核心】Stage-1 env/PPO/GTrXL/评测（6771 行）
├── noise_rl_module_v2.py          # 【核心】Stage-2 噪声 RL（3365 行）
├── genetic_search_module.py       # 【核心】两阶段 GA（2045 行）
├── general_policy_module.py       # 【核心】通用策略训练 / 离线推理（2279 行）
├── final_evaluation_module.py     # Stage-1 最终评估（996 行）
├── noise_final_evaluation_module.py # Stage-2 最终评估（1968 行）
├── function_handler.py            # GELU/Softmax/噪声注入的算子（1204 行）
│
├── experiment_*.py         # 一次性实验脚本
├── analyze_*.py            # 数据分布分析脚本
├── run_*.sh                # 实验驱动 shell
└── llama_7B_LayerImportance.sh  # 主训练入口（RL + GA + compare 一条龙）
```

★ 号目录是 Phase 1 引入的新目录，其他都是既有结构。

---

## 2. 分层概念

```
┌─────────────────────────────────────────────────────────────────┐
│  CLI 入口层                                                      │
│   rl_tune.py    rl_tune_general.py    rl_tune_genetic.py         │
│   rl_ga_compare_runner.py    generate_glue_submission.py         │
└────────────────┬────────────────────────────────┬────────────────┘
                 │                                │
                 ▼                                ▼
┌────────────────────────────────┐   ┌──────────────────────────────┐
│  搜索算法层                     │   │  评估层                       │
│   layer_importance_evaluator   │   │   final_evaluation_module    │
│   noise_rl_module_v2           │◀──│   noise_final_evaluation_… │
│   genetic_search_module        │   └──────────────────────────────┘
│   general_policy_module        │
└────────────────┬───────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  算子 / 环境层                    │
│   function_handler              │   (GELU/Softmax 近似 + 噪声注入)
└─────────────────────────────────┘
```

依赖方向：上层可以 `import` 下层；下层不允许反向 `import` 上层。
但实际上 ``layer_importance_evaluator`` 和 ``noise_rl_module_v2`` **相互引用**，
见 [GLOBALS.md §循环依赖](GLOBALS.md)。

---

## 3. CLI 入口 vs 核心模块

| CLI 入口 | 核心模块 | 主要目的 |
|---|---|---|
| `rl_tune.py` | `layer_importance_evaluator`, `function_handler` | 单任务 Stage-1（GELU/Softmax 优化） |
| `rl_tune.py` (stage2) | + `noise_rl_module_v2` | 单任务 Stage-2（噪声优化） |
| `rl_tune_general.py` | `general_policy_module` + 上面全部 | 多任务通用策略训练 / 离线搜索 |
| `rl_tune_genetic.py` | `genetic_search_module` | GA 搜索（两阶段） |
| `rl_ga_compare_runner.py` | 调用上面三者 + 比对报告 | 并行 RL / GA 并生成对比 |
| `generate_glue_submission.py` | `function_handler` | 按给定配置生成 GLUE 提交 |

---

## 4. 结果目录命名规则

### 4.1 持久化（可续）训练：`rl_results/persistent/`

```
rl_results/persistent/{algorithm}/{model_type}/{dataset}/{accuracy_slug}/
  metadata.json           # algorithm/model/dataset + stage_status 字典
  stage1/pruning_search_log.txt
  stage2_noise/{details,progress}/
  rl.pid / ga.pid / run.pid
```

- `algorithm` ∈ `{rl, ga, general-rl}`（值来自 `config.paths.PERSISTENT_*_BRANCH`）
- `accuracy_slug` 由精度容忍度 + Stage-2 波动百分比组合而成，如 `s1t0.005_s2t0.05_s2st0.05`
- 同一 slug → 同一目录 → 自动续训练
- 每个 dataset 目录下还有 `LATEST_PID` / `LATEST_RUN_DIR` 两个标签文件，指向最近一次运行

### 4.2 一次性对比：`rl_results/runs/compare/rl_vs_ga/`

```
rl_results/runs/compare/rl_vs_ga/{dataset}/
  LATEST_COMPARE_PID
  LATEST_COMPARE_RUN_DIR
  comp_1/, comp_2/, ..., 20260413_213848_pid1143993/
    children/rl/, children/ga/        # 分别是 RL 和 GA 子 run 的产物
    meta/compare_final_status.json    # 完成后状态（status_board 用这个）
    meta/compare_metadata.json        # 数据集/模型/base_model
    meta/compare_status.json          # 运行中状态
    meta/compare.pid
    reports/stage1_compare_report_{dataset}.md
    reports/stage2_compare_report_{dataset}.md
    reports/stage{1,2}_compare_plot_{dataset}.png
    reports/stage{1,2}_compare_summary_{dataset}.json
```

`status_board.py` 把这些 meta/reports 聚合成总板。

---

## 5. 一份模块功能速查

### `layer_importance_evaluator.py`（6771 行，最大）
- **Stage-1 RL 全家桶**：GTrXL 策略网络、PPO 环境、经验回放、训练循环、评测、
  持久化 metadata 更新、各种 opt-flags。
- 含有全局常数：`GELU_MAP/COST`、`SOFTMAX_MAP/COST`、噪声 action 映射、
  `RL_OPT_FLAGS` 字典（消融开关）、默认日志/曲线文件名。
- **对外 API**：`LayerImportanceEvaluator`、`update_persistent_metadata_stage`、
  `detect_rl_local_optimum`、`RunningMeanStd`。

### `noise_rl_module_v2.py`（3365 行）
- **Stage-2 噪声 RL**：NoiseGTrXL 网络、`_NoiseOptEnv`、`_NoiseRecurrentRolloutBuffer`、
  `run_noise_rl_stage2`。
- 含有巨量 Stage-2 超参（见 config/constants.py 里的 `NOISE_STAGE_*`）。
- 提供优雅停止：在进度目录创建 `STOP_RL` 文件即可。
- 与 `layer_importance_evaluator` **双向引用**：后者引用其进度 box 工具，前者
  引用后者的 checkpoint 文件名常量。

### `genetic_search_module.py`（2045 行）
- GA 全家桶：Stage-1 / Stage-2 GA、适应度、选择、交叉、变异、持久化 checkpoint。
- 从 `function_handler` 和 `noise_rl_module_v2` 导入噪声常量与权重；从
  `noise_final_evaluation_module` 导入评估；从 `final_evaluation_module` 导入 Stage-1 评估。

### `general_policy_module.py`（2279 行）
- 多任务轮训 / 离线推理通用策略。
- 复用 `layer_importance_evaluator` 的 GTrXL 网络和环境；复用
  `noise_rl_module_v2` 的 Stage-2 网络 / 环境 / 缓冲。
- **输出**：一个 `.pt` 文件，可作为 `RL_OPT_FLAGS["stage1_pretrained_policy_path"]`
  或 `NOISE_RL_OPT_FLAGS["pretrained_policy_path"]` 供原 CLI 使用。

### `final_evaluation_module.py` / `noise_final_evaluation_module.py`
- 对搜索到的最优配置做"最终评估"（permutation / cost-equivalent / budget-equivalent）。
- 写 JSON + PNG 到各自默认目录。

### `function_handler.py`
- `ReversibleLayerHandler`：在线替换模型里的 GELU / Softmax、注入各类噪声。
- 噪声 variance 表 `INPUT_NOISE_VARIANCE_TABLE`、各类 scaling factor 的允许集合 / 默认值。
- GELU/SiLU 的多项式逼近系数 `GELU_COEEF`, `SiLU_COEEF`、指数 Taylor bound `Exp_bound`。
- 所有依赖 RL / GA 的模块最终都会 import 这里。

### `runtime_error_reporter.py`
- 给 `fire` CLI 套一层壳，把未捕获异常写成结构化的 `error_summary.json`。
- 被所有 `rl_tune*.py` 入口 wraps。

---

## 6. 为什么现在要这样组织

目前的**问题**：
- 12 个核心模块全在根目录，扁平 import（`from xxx import ...`），搬动任何一个都要
  改动几十处。
- 大量路径 / 超参直接写死在 3-4 个模块里，要改一个"噪声进度目录"得搜 10 个位置。
- `tests/` 里的导入依赖项目根，不能简单移动。
- README.md 120KB，里面大量引用根目录下的脚本名。

所以 Phase 1 选择的路径是：**不动核心 .py**，先
- 建 `config/` 聚合路径 / 常量（新代码可用，旧代码先不改）
- 建 `docs/` 放所有人类可读文档
- 建 `tools/status_board.py` 做进度总板
- 写好 Phase-2 的挪移计划 [REORG_PLAN.md](REORG_PLAN.md)，等你确认后再动核心模块

---

## 7. 快速上手一些常见任务

- **查看当前各个任务跑到哪了**：`python tools/status_board.py`
- **改一个全局路径**：先去 [GLOBALS.md](GLOBALS.md) 搜一下，看看几个文件里都有它；Phase 1 还得一处一处改，Phase 3 会变成只改 `config/paths.py`
- **新加一个 GLUE 任务**：搜 `TASK_REGISTRY`，至少 4 个模块要同步加
- **停掉正在跑的 Stage-2 RL**：在对应 `rl_results/noise_rl_progress/.../` 或持久化目录里 `touch STOP_RL`
