# BLB Stage-2 Sequential RL · 诊断汇总（diagnostics @ episode=60000）

_更新时间: 2026-06-11 17:19:26_  ·  累计用时: **13h40m43s**

**Run meta**：
- `profile` = `mrpc`
- `fixed_label` = `Stage-1 config (stage1_record:bert base mrpc 1 20260610; softmax fixed deg6)`
- `fixed_source` = `stage1_record:bert base mrpc 1 20260610`
- `rl_variant` = `blb_v3_sequential_gtrxl_v2scale_fusioncount_v1`
- `total_episodes_planned` = `60000`
- `rollout_size` = `60`
- `save_interval` = `200`
- `ppo_lr` = `0.0002`
- `ppo_clip_range` = `0.2`
- `ppo_ent_coef` = `0.02`
- `ppo_value_coef` = `0.5`
- `invalid_penalty` = `1.0`
- `cost_shaping_coeff` = `0.05`
- `fusion_shaping_coeff` = `0.0`
- `early_terminate_on_invalid` = `False`
- `static_invalid_level_mask_enabled` = `True`
- `acc_threshold` = `0.85828125`
- `stab_threshold` = `0.01`
- `static_skeletons_archive` = `/hy-tmp/fusion_count_newenum_513a1ff_20260611_005952/src/Rescale_optimizer/configs/mrpc/static_skeletons_mrpc.json`
- `fast_reward_mode_enabled` = `False`
- `online_num_trials_per_step` = `1`
- `terminal_eval_batch_size` = `4`
- `promotion_validation_trials` = `4`
- `promotion_margin_window` = `0.25`

## 1. 训练进度（training progress）

- 已完成回合数: **60000**
- 最近 50 回合 mean return: **+39.2086** (min=+39.0762, max=+39.3201)
- 最近 50 回合 mean terminal reward: **+41.6637**
- 最近 50 回合 mean invalid 子步数: **0.00** / 47
- 训练期 best reward: **+39.4116**
- 训练期 worst reward: **-3.2660**
- PPO 更新次数: **1000**
- baseline avg_k (per-block 加权): **13.000**

## 2. 训练期 Top-20 candidates

**说明**：按 hard-priority + unbounded P3 cost rank 排序。P1/P2 不吃 cost；P3 内部先看无上限 cost rank，再看 fusion/K/bits 与 reward。每条候选的完整 SF / K 配置见 `top_candidates.jsonl` 的 `slots` 字段。

| Rank | Episode | P | cost_rank | total_reward | terminal | per_step_sum | valid | invalid | total_bits | fusion |
|-----:|--------:|--:|----------:|-------------:|---------:|-------------:|------:|--------:|-----------:|-------:|
| 1 | 29481 | 3 | +1930.0000 | +39.3565 | +41.8116 | -2.4551 | 47 | 0 | 11285 | 0 |
| 2 | 29459 | 3 | +1910.0000 | +39.3420 | +41.7971 | -2.4551 | 47 | 0 | 11285 | 0 |
| 3 | 29453 | 3 | +1910.0000 | +39.2505 | +41.7056 | -2.4551 | 47 | 0 | 11285 | 0 |
| 4 | 29595 | 3 | +1910.0000 | +39.2200 | +41.6751 | -2.4551 | 47 | 0 | 11285 | 0 |
| 5 | 29490 | 3 | +1910.0000 | +39.0676 | +41.5227 | -2.4551 | 47 | 0 | 11285 | 0 |
| 6 | 29450 | 3 | +1900.0000 | +39.3957 | +41.8508 | -2.4551 | 47 | 0 | 11285 | 0 |
| 7 | 29494 | 3 | +1900.0000 | +39.2737 | +41.7288 | -2.4551 | 47 | 0 | 11285 | 0 |
| 8 | 29559 | 3 | +1900.0000 | +39.2127 | +41.6678 | -2.4551 | 47 | 0 | 11285 | 0 |
| 9 | 29645 | 3 | +1890.0000 | +39.2969 | +41.7520 | -2.4551 | 47 | 0 | 11285 | 0 |
| 10 | 29463 | 3 | +1890.0000 | +39.2664 | +41.7216 | -2.4551 | 47 | 0 | 11285 | 0 |
| 11 | 29817 | 3 | +1890.0000 | +39.2360 | +41.6911 | -2.4551 | 47 | 0 | 11285 | 0 |
| 12 | 29518 | 3 | +1890.0000 | +39.2360 | +41.6911 | -2.4551 | 47 | 0 | 11285 | 0 |
| 13 | 29513 | 3 | +1890.0000 | +39.2360 | +41.6911 | -2.4551 | 47 | 0 | 11285 | 0 |
| 14 | 29795 | 3 | +1890.0000 | +39.2360 | +41.6911 | -2.4551 | 47 | 0 | 11285 | 0 |
| 15 | 29746 | 3 | +1890.0000 | +39.2055 | +41.6606 | -2.4551 | 47 | 0 | 11285 | 0 |
| 16 | 30346 | 3 | +1880.0000 | +39.4116 | +41.8667 | -2.4551 | 47 | 0 | 11285 | 0 |
| 17 | 30371 | 3 | +1880.0000 | +39.4116 | +41.8667 | -2.4551 | 47 | 0 | 11285 | 0 |
| 18 | 30533 | 3 | +1880.0000 | +39.4116 | +41.8667 | -2.4551 | 47 | 0 | 11285 | 0 |
| 19 | 31311 | 3 | +1880.0000 | +39.4116 | +41.8667 | -2.4551 | 47 | 0 | 11285 | 0 |
| 20 | 31162 | 3 | +1880.0000 | +39.4116 | +41.8667 | -2.4551 | 47 | 0 | 11285 | 0 |

### 2.1 Best vs baseline 槽位 diff（哪些槽变了，变了多少）

_共 105 个槽与 baseline 不同_（60 SF + 45 K）

**Truncation K diffs**：

| Slot | Baseline K | Best K | Δ |
|:-----|----------:|------:|--:|
| `L0.B2.K` | 13 | 8 | -5 |
| `L0.B5.K` | 13 | 12 | -1 |
| `L1.B1.K` | 13 | 12 | -1 |
| `L1.B2.K` | 13 | 12 | -1 |
| `L1.B4.K` | 13 | 8 | -5 |
| `L1.B5.K` | 13 | 8 | -5 |
| `L10.B1.K` | 13 | 9 | -4 |
| `L10.B2.K` | 13 | 9 | -4 |
| `L10.B4.K` | 13 | 9 | -4 |
| `L10.B5.K` | 13 | 9 | -4 |
| `L11.B1.K` | 13 | 9 | -4 |
| `L11.B2.K` | 13 | 9 | -4 |
| `L11.B4.K` | 13 | 9 | -4 |
| `L11.B5.K` | 13 | 9 | -4 |
| `L2.B1.K` | 13 | 12 | -1 |
| `L2.B2.K` | 13 | 12 | -1 |
| `L2.B4.K` | 13 | 9 | -4 |
| `L2.B5.K` | 13 | 9 | -4 |
| `L3.B1.K` | 13 | 9 | -4 |
| `L3.B2.K` | 13 | 9 | -4 |
| `L3.B4.K` | 13 | 9 | -4 |
| `L3.B5.K` | 13 | 9 | -4 |
| `L4.B1.K` | 13 | 9 | -4 |
| `L4.B2.K` | 13 | 9 | -4 |
| `L4.B4.K` | 13 | 9 | -4 |
| `L4.B5.K` | 13 | 9 | -4 |
| `L5.B1.K` | 13 | 9 | -4 |
| `L5.B2.K` | 13 | 9 | -4 |
| `L5.B4.K` | 13 | 9 | -4 |
| `L5.B5.K` | 13 | 9 | -4 |
| `L6.B1.K` | 13 | 9 | -4 |
| `L6.B2.K` | 13 | 9 | -4 |
| `L6.B4.K` | 13 | 9 | -4 |
| `L6.B5.K` | 13 | 9 | -4 |
| `L7.B1.K` | 13 | 9 | -4 |
| `L7.B2.K` | 13 | 9 | -4 |
| `L7.B5.K` | 13 | 9 | -4 |
| `L8.B1.K` | 13 | 9 | -4 |
| `L8.B2.K` | 13 | 9 | -4 |
| `L8.B4.K` | 13 | 9 | -4 |
| `L8.B5.K` | 13 | 9 | -4 |
| `L9.B1.K` | 13 | 9 | -4 |
| `L9.B2.K` | 13 | 9 | -4 |
| `L9.B4.K` | 13 | 9 | -4 |
| `L9.B5.K` | 13 | 9 | -4 |

**Scaling-factor diffs**（前 20 条按 |Δ| 降序）：

| Slot | Kind | Baseline SF | Best SF | Δ |
|:-----|:----:|------------:|--------:|--:|
| `L0.B3.R.sq0` | R | off | 31 | off→on |
| `L0.B3.R.sq1` | R | off | 31 | off→on |
| `L0.B3.R.sq2` | R | off | 31 | off→on |
| `L0.B3.R.sq3` | R | off | 31 | off→on |
| `L0.B3.R.x_inv_2n_r` | R | off | 22 | off→on |
| `L1.B3.R.sq0` | R | off | 31 | off→on |
| `L1.B3.R.sq1` | R | off | 31 | off→on |
| `L1.B3.R.sq2` | R | off | 31 | off→on |
| `L1.B3.R.sq3` | R | off | 31 | off→on |
| `L1.B3.R.x_inv_2n_r` | R | off | 22 | off→on |
| `L2.B3.R.sq0` | R | off | 31 | off→on |
| `L2.B3.R.sq1` | R | off | 31 | off→on |
| `L2.B3.R.sq2` | R | off | 31 | off→on |
| `L2.B3.R.sq3` | R | off | 31 | off→on |
| `L2.B3.R.x_inv_2n_r` | R | off | 22 | off→on |
| `L3.B3.R.sq0` | R | off | 31 | off→on |
| `L3.B3.R.sq1` | R | off | 31 | off→on |
| `L3.B3.R.sq2` | R | off | 31 | off→on |
| `L3.B3.R.sq3` | R | off | 31 | off→on |
| `L3.B3.R.x_inv_2n_r` | R | off | 22 | off→on |

完整 diff 在 `best_action_vec.json` 的 `diff_vs_baseline` 字段。

## 3. First-invalid 频次（哪些 (layer, block) 最先翻车）

_暂无 invalid 记录（所有 episode 完整通过）。_

## 4. PPO 学习动态（最近 10 次更新）

| Update | Eps | policy_loss | value_loss | entropy | clip_frac | ent_coef | approx_kl | lr_scale | entropy_recovery | win_mean_ret | win_mean_inv |
|------:|----:|------------:|-----------:|--------:|----------:|---------:|----------:|---------:|-----------------:|-------------:|-------------:|
| 991 | 59460 | +0.0075 | +0.0000 | +0.0000 | 0.000 | 0.02000 | 0.00000 | 1.250 | 0.25537 | +39.2348 | 0.00 |
| 992 | 59520 | +0.0027 | +0.0000 | +0.0000 | 0.000 | 0.02000 | 0.00000 | 1.250 | 0.25577 | +39.2226 | 0.00 |
| 993 | 59580 | +0.0010 | +0.0000 | +0.0000 | 0.000 | 0.02000 | 0.00000 | 1.250 | 0.25577 | +39.2322 | 0.00 |
| 994 | 59640 | -0.0082 | +0.0000 | +0.0000 | 0.000 | 0.02000 | 0.00000 | 1.250 | 0.25537 | +39.2419 | 0.00 |
| 995 | 59700 | -0.0060 | +0.0000 | +0.0000 | 0.000 | 0.02000 | 0.00000 | 1.250 | 0.25537 | +39.2292 | 0.00 |
| 996 | 59760 | -0.0028 | +0.0000 | +0.0000 | 0.000 | 0.02000 | 0.00000 | 1.250 | 0.25497 | +39.2353 | 0.00 |
| 997 | 59820 | -0.0036 | +0.0000 | +0.0000 | 0.000 | 0.02000 | 0.00000 | 1.250 | 0.25567 | +39.2256 | 0.00 |
| 998 | 59880 | +0.0163 | +0.0000 | +0.0000 | 0.000 | 0.02000 | 0.00000 | 1.250 | 0.25537 | +39.2246 | 0.00 |
| 999 | 59940 | +0.0040 | +0.0000 | +0.0000 | 0.000 | 0.02000 | 0.00000 | 1.250 | 0.25557 | +39.2261 | 0.00 |
| 1000 | 60000 | +0.0066 | +0.0000 | +0.0000 | 0.000 | 0.02000 | 0.00000 | 1.250 | 0.25517 | +39.2129 | 0.00 |

_Entropy 趋势：+2.0705 → +0.0000（下降（policy 在收敛））_

## 5. 动作分布概览（哪些 slot 已经在按 baseline 取最大档）

- **已收敛 slot**（top-1 占比 > 85%）：**284** / 877
- **未收敛 slot**：**47** / 877

已收敛 slot 示例（前 8 个）：
  - slot[008] → action_index=3 （占比 100.0%）
  - slot[018] → action_index=0 （占比 100.0%）
  - slot[019] → action_index=0 （占比 100.0%）
  - slot[023] → action_index=0 （占比 100.0%）
  - slot[024] → action_index=0 （占比 100.0%）
  - slot[025] → action_index=0 （占比 100.0%）
  - slot[026] → action_index=0 （占比 100.0%）
  - slot[029] → action_index=0 （占比 100.0%）

最分散 slot 示例（前 8 个）：
  - slot[129] entropy=1.208 (uniform≈1.792)
  - slot[145] entropy=1.192 (uniform≈1.792)
  - slot[072] entropy=1.173 (uniform≈1.792)
  - slot[154] entropy=1.167 (uniform≈1.792)
  - slot[104] entropy=1.163 (uniform≈1.792)
  - slot[056] entropy=1.157 (uniform≈1.792)
  - slot[177] entropy=1.154 (uniform≈1.792)
  - slot[081] entropy=1.150 (uniform≈1.792)

## 6. 自动诊断（auto-flags）

- ⚠ **熵过低**：最近 3 次 PPO 更新平均 entropy=0.000 (< 0.1)。policy 已经几乎确定性输出，可能过早收敛 — 增大 ent_coef 或 clip_range。

## 7. 原始数据文件（machine-readable）

| 文件 | 内容 |
|------|------|
| `episodes.jsonl` | 完整 per-episode 记录（append-only） |
| `ppo_updates.jsonl` | 完整 per-PPO-update 记录（append-only） |
| `top_candidates.jsonl` | Top-20 训练期 best：含每条候选的完整 `slots` 列表（人类可读） |
| `pareto_frontier.jsonl` | 训练期非支配候选（质量 / 稳定性 / cost 多目标） |
| `pareto_frontier.json` | Pareto frontier 元数据 + 完整候选列表 |
| `pareto_frontier.html` | 可直接用浏览器打开的 Pareto frontier 表格 |
| `first_invalid_counts.json` | (L, B) → 首次 invalid 计数 |
| `action_histogram.npz` | (num_slots, max_levels) 频次矩阵 |
| `baseline_action_vec.json` | static_skeletons baseline 的完整 `slots` 视图（参照系） |
| `best_action_vec.json` | **训练期最优**：`slots` 列表（按 SF/K 选）+ `action_vec` 兜底字段。**可直接喂给 `Paean/run_final_eval.sh --action-config`** |

**重跑 final eval 的最简命令**（无需等训练结束）：

```bash
bash Paean/run_final_eval.sh \
    --preset mrpc-final-eval-only \
    --action-config Parting Chapter/stage2/bert base mrpc/progress/diagnostics/best_action_vec.json
```

**手动调几个槽位**：直接复制 `best_action_vec.json`，改里面 `slots` 列表中对应槽位的 `scaling_factor` 或 `truncation_bits`，存成新文件后 `--action-config` 指过去即可。支持简写 `{"base":"max", "overrides":[{"label":"L05.B3.K", "truncation_bits":10}]}`。