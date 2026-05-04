# 加强版 BLB Stage 2 强化学习 · 使用说明

> 设计 spec 见 `docs/BLB_stage2_rl_spec.md`。本文档面向**使用者**，介绍如何启用、
> 切换、调参以及与旧版 stage 2 RL 共存的方式。

## 1. 一句话摘要

新版（`stage2_rl_variant=blb_v3`，**默认**）覆盖 BLB 5 个 block + first-input fresh
的全部噪声候选点，按"精度 → 稳定性 → CKKS+MPC 总开销"三层优先级训练 PPO；
旧版（`stage2_rl_variant=legacy_v2`）继续使用 `noise_rl_module_v2.NoiseRLModuleV2`，
只优化 `*_scaling_factors`。两条线**并行存在、互不干扰**。

---

## 2. 默认行为

无需任何额外参数即可启用新版：

```bash
bash llama_7B_LayerImportance.sh run rl --dataset mrpc --episodes 200
```

会走 `BLBStage2RLRunner`（新版），训练完后产物：

```
<run_dir>/stage2_noise/progress/
├── blb_stage2_rl_checkpoint_live.pt        # 周期 checkpoint（每 N episode）
├── blb_stage2_rl_checkpoint_final.pt       # 训练结束 final checkpoint
└── blb_stage2_best_cfg.pkl                 # best policy 对应的 BLB cfg + action vec
```

返回的 `noise_stage_result` dict 与旧版兼容，下游 `UnifiedFinalEvaluationModule`
继续按基线（全 max `*_scaling_factors`）跑 final-eval；BLB 真正的最优配置保存在
`blb_v3_best_action_vec` 字段与上面那份 `.pkl` 文件中。

---

## 3. 切回旧版

只在 Python 入口（`rl_tune.py` / `rl_tune_genetic.py`）追加一个参数：

```bash
python rl_tune.py \
    --base_model bert-base-uncased --data_path mrpc \
    --output_dir runs/legacy_v2_demo \
    --stage2_rl_variant legacy_v2 \
    --use_ist
```

或在脚本里直接覆盖：

```python
from layer_importance_evaluator import LayerImportanceEvaluator

evaluator = LayerImportanceEvaluator(
    model=model, train_data=..., test_data=..., data_collator=...,
    stage2_rl_variant="legacy_v2",   # 切回旧版
)
```

合法值（不区分大小写）：
- `'blb_v3'` / `'blb'` / `'v3'`：新版（默认）
- `'legacy_v2'` / `'legacy'` / `'v2'`：旧版

---

## 4. 关键参数（新版独有）

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `stage2_rl_variant` | `blb_v3` | 选择 stage 2 RL 实现 |
| `blb_v3_rescale_invoker_kind` | `heuristic` | `Rescale_optimizer` 调用方式：`heuristic` 内置启发式（不依赖外部子项目）；`subprocess` 真调外部 CLI；`stub` 用预设 JSON |
| `blb_v3_subprocess_optimizer_root` | `None` | invoker=`subprocess` 时外部子项目根目录 |
| `blb_v3_subprocess_cli_module` | `rescale_optimizer.replan` | invoker=`subprocess` 时 CLI 模块 |
| `blb_v3_rollout_size` | `32` | PPO rollout：多少 episode 触发一次 PPO update |
| `blb_v3_eval_interval` | `100` | 每多少 episode 跑一次 deterministic eval（仅日志） |
| `blb_v3_save_interval` | `200` | 每多少 episode 保存一次 live checkpoint |
| `blb_v3_calibrate_baseline_samples` | `8` | 校准 reward 权重时跑多少 random action |

新版仍使用 `LayerImportanceEvaluator` 已有的字段：
- `stage2_rl_episodes`（总 episodes）
- `stage2_rl_lr`（PPO 学习率）
- `stage2_k_trials`（每步 K 次噪声 trial 评估稳定性）
- `stage2_probe_size`（探针子集大小）
- `stage2_limit_tolerance` / `stage2_stability_tolerance`（限制比例 → 自动转阈值）

---

## 5. 与 Rescale_optimizer 子项目的关系

`Rescale_optimizer/` 目录在本仓库默认是空的（用户独立维护）。新版 BLB stage 2 RL
**不强依赖**外部子项目：

- `blb_v3_rescale_invoker_kind=heuristic`（默认）：用内置 `HeuristicStubInvoker`
  按 cfg 字段直接估算 `total_bits` / `fusion_count` / `invalid_chain`。reward 仍单调可指导，
  但绝对量级与真实 CKKS 模数链开销略有偏差，适合快速试跑 / 算法验证。

- `blb_v3_rescale_invoker_kind=subprocess`：当用户拉好 `Rescale_optimizer` 子项目并
  按 [`docs/BLB_stage2_rl_spec.md` §5.2.2](BLB_stage2_rl_spec.md) 准备 configs 后切换。
  cost 信号将基于真实模数链优化结果。

切换调用：

```python
evaluator = LayerImportanceEvaluator(
    ...,
    stage2_rl_variant="blb_v3",
    blb_v3_rescale_invoker_kind="subprocess",
    blb_v3_subprocess_optimizer_root="Rescale_optimizer",
    blb_v3_subprocess_cli_module="rescale_optimizer.replan",
)
```

注：`subprocess` 模式还需要在 evaluator 上额外挂一份
`{config_name: config_path}` mapping（见 `BLBStage2TrainConfig.subprocess_configs`）。

---

## 6. max-SF 表

每个 (block, node) 的"最高挡位 SF"由
`blb_stage2_rl/max_sfs/<profile>.json` 决定。`profile` 默认与 `evaluator.dataset_key`
保持一致（如 `mrpc`、`stsb`）。`<profile>.json` 不存在时自动 fallback 到
`blb_stage2_rl/max_sfs/default.json`，最后再 fallback 到 `_BLOCK_SPECS` 里的
`default_max_sf`（22 / 30 等保守值）。

如果你已经跑过 `Rescale_optimizer` 的 `static_skeletons_<profile>.json`，按 spec §4.4
的脚本可一次性导出对应 profile 的 max-SF JSON；按下面的 schema 放进
`blb_stage2_rl/max_sfs/<profile>.json` 即可：

```json
{
  "block1": {"ctpt_ffn2": 30, ...},
  "block2": {...},
  "block3": {...},
  "block4": {...},
  "block5": {...}
}
```

每个 block 的 node 名见 `blb_stage2_rl/action_space.py:_BLOCK_NODE_NAME_BY_FIELD`。

---

## 7. 验证

### 单元测试

```bash
python -m unittest tests.test_blb_stage2_rl
```

应该看到 15 个测试全部通过（含 1 个端到端 RunnerEndToEndTests，跑 5 个 episode）。

### 回归测试

```bash
python -m unittest discover tests
```

应该看到 83 个测试通过（4 skipped）。任何旧版 unit test 失败都意味着新版改动
影响了旧版逻辑，请 `git diff` 检查。

### 训练 smoke

```bash
python rl_tune.py \
    --base_model bert-base-uncased --data_path mrpc \
    --output_dir runs/blb_v3_smoke \
    --stage1_rl_episodes 200 --stage2_rl_episodes 50 \
    --use_ist --skip_final_eval
```

预期日志中能看到：

```
================================================================================
阶段 5 · 加强版 BLB Stage 2 强化学习（BLB Stage 2 RL · v3）
================================================================================
  * 固定 GELU/Softmax 来源：rl_search    标签：...
  * Profile = 'mrpc'    Total episodes = 50    PPO update interval = 32
  * Baseline cost: total_bits_sum=..., total_fusion_count=..., avg_k=13.00
  * Reward weights: w_bits=..., w_fusion=1, w_k=1
  * 硬约束阈值: acc_threshold=..., stab_threshold=...
训练开始（PPO 单步 episode）...
  [BLB-v3] ep=32/50    return mean=... best=...
训练完成：best_reward=...
  * Final policy 已保存到：.../blb_stage2_rl_checkpoint_final.pt
  * Best BLB cfg 已保存到：.../blb_stage2_best_cfg.pkl
```
