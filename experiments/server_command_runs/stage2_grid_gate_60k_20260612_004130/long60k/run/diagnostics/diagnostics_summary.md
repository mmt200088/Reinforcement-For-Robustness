# BLB Stage-2 Sequential RL · 诊断汇总（diagnostics @ episode=60000）

_更新时间: 2026-06-12 14:53:06_  ·  累计用时: **13h50m40s**

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
- `stab_threshold` = `0.028082183667194838`
- `static_skeletons_archive` = `/hy-tmp/server_command_stage2_adr011_a45f651_20260611_225936/src/Rescale_optimizer/configs/mrpc/static_skeletons_mrpc.json`
- `fast_reward_mode_enabled` = `False`
- `online_num_trials_per_step` = `1`
- `terminal_eval_batch_size` = `4`
- `promotion_validation_trials` = `4`
- `promotion_margin_window` = `0.25`

## 1. 训练进度（training progress）

- 已完成回合数: **60000**
- 最近 50 回合 mean return: **+39.2943** (min=+39.1485, max=+39.4839)
- 最近 50 回合 mean terminal reward: **+41.7494**
- 最近 50 回合 mean invalid 子步数: **0.00** / 47
- 训练期 best reward: **+39.5560**
- 训练期 worst reward: **-7.2343**
- PPO 更新次数: **1000**
- baseline avg_k (per-block 加权): **13.000**

## 2. 训练期 Top-20 candidates

**说明**：按 hard-priority + unbounded P3 cost rank 排序。P1/P2 不吃 cost；P3 内部先看无上限 cost rank，再看 fusion/K/bits 与 reward。每条候选的完整 SF / K 配置见 `top_candidates.jsonl` 的 `slots` 字段。

| Rank | Episode | P | cost_rank | total_reward | terminal | per_step_sum | valid | invalid | total_bits | fusion |
|-----:|--------:|--:|----------:|-------------:|---------:|-------------:|------:|--------:|-----------:|-------:|
| 1 | 39998 | 3 | +2350.0000 | +39.5449 | +42.0000 | -2.4551 | 47 | 0 | 11285 | 0 |
| 2 | 52351 | 3 | +2350.0000 | +39.5449 | +42.0000 | -2.4551 | 47 | 0 | 11285 | 0 |
| 3 | 38713 | 3 | +2350.0000 | +39.5449 | +42.0000 | -2.4551 | 47 | 0 | 11285 | 0 |
| 4 | 45019 | 3 | +2350.0000 | +39.5449 | +42.0000 | -2.4551 | 47 | 0 | 11285 | 0 |
| 5 | 42504 | 3 | +2350.0000 | +39.5449 | +42.0000 | -2.4551 | 47 | 0 | 11285 | 0 |
| 6 | 42593 | 3 | +2350.0000 | +39.5449 | +42.0000 | -2.4551 | 47 | 0 | 11285 | 0 |
| 7 | 52545 | 3 | +2350.0000 | +39.5449 | +42.0000 | -2.4551 | 47 | 0 | 11285 | 0 |
| 8 | 55267 | 3 | +2350.0000 | +39.5449 | +42.0000 | -2.4551 | 47 | 0 | 11285 | 0 |
| 9 | 59125 | 3 | +2350.0000 | +39.5449 | +42.0000 | -2.4551 | 47 | 0 | 11285 | 0 |
| 10 | 39930 | 3 | +2350.0000 | +39.5449 | +42.0000 | -2.4551 | 47 | 0 | 11285 | 0 |
| 11 | 49356 | 3 | +2350.0000 | +39.5449 | +42.0000 | -2.4551 | 47 | 0 | 11285 | 0 |
| 12 | 54297 | 3 | +2350.0000 | +39.5449 | +42.0000 | -2.4551 | 47 | 0 | 11285 | 0 |
| 13 | 59209 | 3 | +2350.0000 | +39.5449 | +42.0000 | -2.4551 | 47 | 0 | 11285 | 0 |
| 14 | 45296 | 3 | +2350.0000 | +39.5449 | +42.0000 | -2.4551 | 47 | 0 | 11285 | 0 |
| 15 | 47606 | 3 | +2350.0000 | +39.5449 | +42.0000 | -2.4551 | 47 | 0 | 11285 | 0 |
| 16 | 46717 | 3 | +2350.0000 | +39.5449 | +42.0000 | -2.4551 | 47 | 0 | 11285 | 0 |
| 17 | 39700 | 3 | +2350.0000 | +39.5144 | +41.9695 | -2.4551 | 47 | 0 | 11285 | 0 |
| 18 | 39740 | 3 | +2350.0000 | +39.5144 | +41.9695 | -2.4551 | 47 | 0 | 11285 | 0 |
| 19 | 42346 | 3 | +2350.0000 | +39.5144 | +41.9695 | -2.4551 | 47 | 0 | 11285 | 0 |
| 20 | 40143 | 3 | +2350.0000 | +39.5144 | +41.9695 | -2.4551 | 47 | 0 | 11285 | 0 |

### 2.1 Best vs baseline 槽位 diff（哪些槽变了，变了多少）

_共 107 个槽与 baseline 不同_（60 SF + 47 K）

**Truncation K diffs**：

| Slot | Baseline K | Best K | Δ |
|:-----|----------:|------:|--:|
| `L0.B2.K` | 13 | 8 | -5 |
| `L0.B4.K` | 13 | 8 | -5 |
| `L0.B5.K` | 13 | 8 | -5 |
| `L1.B1.K` | 13 | 8 | -5 |
| `L1.B2.K` | 13 | 8 | -5 |
| `L1.B4.K` | 13 | 8 | -5 |
| `L1.B5.K` | 13 | 8 | -5 |
| `L10.B1.K` | 13 | 8 | -5 |
| `L10.B2.K` | 13 | 8 | -5 |
| `L10.B4.K` | 13 | 8 | -5 |
| `L10.B5.K` | 13 | 8 | -5 |
| `L11.B1.K` | 13 | 8 | -5 |
| `L11.B2.K` | 13 | 8 | -5 |
| `L11.B4.K` | 13 | 8 | -5 |
| `L11.B5.K` | 13 | 8 | -5 |
| `L2.B1.K` | 13 | 8 | -5 |
| `L2.B2.K` | 13 | 8 | -5 |
| `L2.B4.K` | 13 | 8 | -5 |
| `L2.B5.K` | 13 | 8 | -5 |
| `L3.B1.K` | 13 | 8 | -5 |
| `L3.B2.K` | 13 | 8 | -5 |
| `L3.B4.K` | 13 | 8 | -5 |
| `L3.B5.K` | 13 | 8 | -5 |
| `L4.B1.K` | 13 | 8 | -5 |
| `L4.B2.K` | 13 | 8 | -5 |
| `L4.B4.K` | 13 | 8 | -5 |
| `L4.B5.K` | 13 | 8 | -5 |
| `L5.B1.K` | 13 | 8 | -5 |
| `L5.B2.K` | 13 | 8 | -5 |
| `L5.B4.K` | 13 | 8 | -5 |
| `L5.B5.K` | 13 | 8 | -5 |
| `L6.B1.K` | 13 | 8 | -5 |
| `L6.B2.K` | 13 | 8 | -5 |
| `L6.B4.K` | 13 | 8 | -5 |
| `L6.B5.K` | 13 | 8 | -5 |
| `L7.B1.K` | 13 | 8 | -5 |
| `L7.B2.K` | 13 | 8 | -5 |
| `L7.B4.K` | 13 | 8 | -5 |
| `L7.B5.K` | 13 | 8 | -5 |
| `L8.B1.K` | 13 | 8 | -5 |
| `L8.B2.K` | 13 | 8 | -5 |
| `L8.B4.K` | 13 | 8 | -5 |
| `L8.B5.K` | 13 | 8 | -5 |
| `L9.B1.K` | 13 | 8 | -5 |
| `L9.B2.K` | 13 | 8 | -5 |
| `L9.B4.K` | 13 | 8 | -5 |
| `L9.B5.K` | 13 | 8 | -5 |

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
| 991 | 59460 | +0.0003 | +0.0000 | +0.0000 | 0.000 | 0.02000 | 0.00000 | 0.750 | 0.25537 | +39.3157 | 0.00 |
| 992 | 59520 | +0.0020 | +0.0000 | +0.0000 | 0.011 | 0.02000 | -0.01836 | 0.900 | 0.25577 | +39.3240 | 0.00 |
| 993 | 59580 | +0.0114 | +0.0000 | +0.0000 | 0.000 | 0.02000 | 0.00000 | 0.900 | 0.25577 | +39.3239 | 0.00 |
| 994 | 59640 | -0.0137 | +0.0000 | +0.0000 | 0.000 | 0.02000 | 0.00000 | 1.080 | 0.25537 | +39.3274 | 0.00 |
| 995 | 59700 | -0.0172 | +0.0001 | +0.0000 | 0.017 | 0.02000 | 0.01963 | 1.250 | 0.25537 | +39.3052 | 0.00 |
| 996 | 59760 | +0.0025 | +0.0000 | +0.0000 | 0.000 | 0.02000 | 0.00000 | 1.250 | 0.25497 | +39.3269 | 0.00 |
| 997 | 59820 | -0.0059 | +0.0000 | +0.0000 | 0.000 | 0.02000 | 0.00000 | 1.250 | 0.25567 | +39.3101 | 0.00 |
| 998 | 59880 | +0.0056 | +0.0000 | +0.0000 | 0.000 | 0.02000 | 0.00000 | 1.250 | 0.25537 | +39.3025 | 0.00 |
| 999 | 59940 | -0.0119 | +0.0365 | +0.0000 | 0.016 | 0.02000 | 0.02594 | 1.250 | 0.25557 | +38.5432 | 0.00 |
| 1000 | 60000 | -0.0055 | +0.0000 | +0.0000 | 0.000 | 0.02000 | 0.00000 | 1.250 | 0.25517 | +39.3035 | 0.00 |

_Entropy 趋势：+2.1709 → +0.0000（下降（policy 在收敛））_

## 5. 动作分布概览（哪些 slot 已经在按 baseline 取最大档）

- **已收敛 slot**（top-1 占比 > 85%）：**332** / 877
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
  - slot[056] entropy=1.407 (uniform≈1.792)
  - slot[081] entropy=1.393 (uniform≈1.792)
  - slot[177] entropy=1.392 (uniform≈1.792)
  - slot[031] entropy=1.391 (uniform≈1.792)
  - slot[072] entropy=1.390 (uniform≈1.792)
  - slot[129] entropy=1.386 (uniform≈1.792)
  - slot[104] entropy=1.382 (uniform≈1.792)
  - slot[145] entropy=1.375 (uniform≈1.792)

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