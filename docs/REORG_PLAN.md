# Phase-2 目录重构方案（待你确认）

> Phase 1（已完成）没有搬动任何 .py，只加了 `config/` / `docs/` / `tools/` 三个目录。
> Phase 2（本文档）才真正把核心模块挪进子目录，让根目录看起来干净。

---

## 为什么要再等你点头

挪 12 个核心模块意味着：

- **~60 处跨模块 import 要改**（`from noise_rl_module_v2 import ...` → `from rlga.stage2.rl import ...`）
- **tests/ 里 ~12 个文件要改 import**
- **run_*.sh / llama_7B_LayerImportance.sh 要改**所有 `python xxx.py` 调用路径
- **generate_glue_submission.py` 等**里写死了模块名用于错误提示的地方
- **README.md（120KB）** 里所有脚本名引用要 find-and-replace
- **持久化 checkpoint 里**可能 pickle 了 `from noise_rl_module_v2 import _NoiseGTrXLStrategyNetwork` 的完整路径（需要实验确认）

**关键风险**：Phase-2 过程中**不要跑新的训练**，避免一半 checkpoint 按旧路径、
一半按新路径。

---

## 建议的最终目录树

```
Local_program/
├── config/                           # 已在 Phase 1 引入
│   ├── paths.py
│   ├── constants.py
│   └── flags.py                     # 新：RL_OPT_FLAGS / NOISE_RL_OPT_FLAGS
│
├── docs/                             # 已在 Phase 1 引入
│
├── tools/                            # 已在 Phase 1 引入
│   └── status_board.py
│
├── rlga/                            ★ 新：核心算法包
│   ├── __init__.py
│   ├── ops/
│   │   ├── function_handler.py      # 从根目录挪进来
│   │   └── runtime_error_reporter.py
│   ├── stage1/
│   │   └── rl.py                    # 现 layer_importance_evaluator.py（拆分或全搬）
│   ├── stage2/
│   │   └── rl.py                    # 现 noise_rl_module_v2.py
│   ├── evaluation/
│   │   └── unified.py               # 现 final_evaluation_module.py（合并 Stage-1 + Stage-2）
│   ├── ga/
│   │   └── search.py                # 现 genetic_search_module.py
│   └── general/
│       └── policy.py                # 现 general_policy_module.py
│
├── cli/                             ★ 新：所有 fire 入口
│   ├── rl_tune.py
│   ├── rl_tune_general.py
│   ├── rl_tune_genetic.py
│   ├── rl_ga_compare_runner.py
│   └── generate_glue_submission.py
│
├── experiments/                     ★ 新：一次性实验脚本
│   ├── experiment_core.py
│   ├── experiment_block1_monotonicity.py
│   ├── experiment_block2_anova.py
│   ├── experiment_block3_cross_task.py
│   ├── experiment_noise_scaling_sweep.py
│   ├── experiment_single_layer_degradation.py
│   ├── experiment_stepwise_degradation.py
│   └── analyze/
│       ├── analyze_all_distribution.py
│       ├── analyze_all_distribution_new.py
│       └── analyze_gelu_distribution.py
│
├── scripts/                         ★ 新：shell 驱动脚本
│   ├── run_all_experiments.sh
│   ├── run_gelu_analysis.sh
│   ├── run_glue_submission.sh
│   ├── run_noise_scaling_sweep.sh
│   └── llama_7B_LayerImportance.sh
│
├── presets/                         # 保持不动
├── tests/                           # 保持，只改 import
│
├── rl_results/                      # 结果目录不动
├── experiment_results/              # 同上
├── glue_submission/  glue_approx/  glue_baseline/
│
├── third_party/                     ★ 新：外部依赖源码
│   └── IST-peft/                    # 原 importance-aware-sparse-tuning-IST-paper/
│
├── README.md
├── pyproject.toml / setup.py        ★ 新：把 rlga/ 变成可安装的包
└── .gitignore
```

---

## 落地顺序（每批都要独立验证）

### 批次 2.1 — 安全搬运（不动 import）
1. 挪纯资料：`Bert-structure.txt`、`coinn_paper.txt`、`*.png`、`*.xlsx` → `docs/assets/`
2. 挪 shell 脚本 → `scripts/`，**立刻**全局搜索 `run_.*\.sh` 引用并更新
3. 挪 `.tmp_*` 目录到 `/tmp/` 或直接删掉（看看里面是否还在用）

**验证**：`git grep` 确认没有旧路径引用。

### 批次 2.2 — 建立 `rlga/` 包（旧路径保留，加 re-export）
1. 新建 `rlga/__init__.py` 和子包目录，把 .py **拷贝**（不是移动）进去
2. 在每个**根目录原文件**里留一个 re-export shim：

   ```python
   # 原 noise_rl_module_v2.py（根目录，兼容用）
   from rlga.stage2.rl import *  # noqa
   from rlga.stage2.rl import (  # 显式 re-export 给循环 import 用
       _NoiseGTrXLStrategyNetwork, _NoiseOptEnv, ...
   )
   ```

3. 这样既有代码 + 测试 + 脚本继续能跑；新代码 `from rlga.stage2.rl import ...`

**验证**：pytest 全部通过 + 手跑一次 `rl_tune.py` 冒烟测试。

### 批次 2.3 — 改 import
1. 扫一轮 `from (layer_importance_evaluator|noise_rl_module_v2|...)`，改成 `from rlga....`
2. **对比运行前后行为**：checkpoint 能继续加载、日志格式不变

### 批次 2.4 — 把 CLI 挪进 `cli/`
1. 移动 `rl_tune.py` 等 → `cli/`
2. 更新 `scripts/*.sh` 里的 `python rl_tune.py` → `python -m cli.rl_tune`
3. 更新 README

### 批次 2.5 — 删除根目录 shim
旧的 re-export 文件删掉；git history 里还能看到。

### 批次 2.6 — 把项目变成可安装包
1. 加 `pyproject.toml`：entry points 里挂上 CLI
2. `pip install -e .` 后 `rlga-tune --data_path mrpc ...` 即可
3. 去掉所有 `sys.path.append` PEFT 路径的写死

---

## 预估工作量

- 批次 2.1：30 分钟
- 批次 2.2：2 小时（shim 要仔细，尤其循环 import 的符号要全）
- 批次 2.3：1 小时（主要是 grep + sed 批改）
- 批次 2.4：1 小时
- 批次 2.5：30 分钟
- 批次 2.6：2 小时（entry points + 去 PEFT 写死）

**总计 6-7 小时**，分 2-3 次做比较稳。

---

## 何时启动

建议：
- **当前有 RL / GA 在跑**：不启动，等空窗期
- **没任务在跑**：确认一次"这一周不会新开训练"后开工
- **跑到一半**：在进度目录 touch `STOP_RL` 优雅停掉，再开工

---

## 需要你决定的三件事

1. **是否接受 `rlga/` 这个包名**？（我可以改成 `core`、`src`、`fhelearn` 等）
2. **核心模块是否拆分**？例如 `layer_importance_evaluator.py`（6771 行）是
   直接挪进 `rlga/stage1/rl.py`，还是借机拆成几个小文件？
3. **PEFT 的处理方案**：保留 `third_party/IST-peft/` + PYTHONPATH，还是彻底
   `pip install -e` 化？

这三个确认后，我可以按上面的批次顺序开工。
