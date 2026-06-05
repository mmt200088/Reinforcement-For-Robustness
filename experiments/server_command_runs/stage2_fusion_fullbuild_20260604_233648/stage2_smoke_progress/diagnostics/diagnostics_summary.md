# BLB Stage-2 Sequential RL · 诊断汇总（diagnostics @ episode=600）

_更新时间: 2026-06-05 15:52:06_  ·  累计用时: **27m46s**

**Run meta**：
- `profile` = `mrpc`
- `fixed_label` = `Stage-1 config (stage1_record:bert base mrpc 1 20260604; softmax fixed deg6)`
- `fixed_source` = `stage1_record:bert base mrpc 1 20260604`
- `rl_variant` = `blb_v3_sequential_gtrxl_v2scale_fusioncount_v1`
- `total_episodes_planned` = `600`
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
- `acc_threshold` = `0.854375`
- `stab_threshold` = `0.01`
- `static_skeletons_archive` = `/hy-tmp/server_command_stage2_fusion_fullbuild_5ed03df_20260604_233625/src/Rescale_optimizer/configs/mrpc/static_skeletons_mrpc.json`
- `fast_reward_mode_enabled` = `False`
- `online_num_trials_per_step` = `1`
- `terminal_eval_batch_size` = `4`
- `promotion_validation_trials` = `4`
- `promotion_margin_window` = `0.25`

## 1. 训练进度（training progress）

- 已完成回合数: **600**
- 最近 50 回合 mean return: **-7.2957** (min=-7.3869, max=-7.1870)
- 最近 50 回合 mean terminal reward: **-5.0000**
- 最近 50 回合 mean invalid 子步数: **0.00** / 47
- 训练期 best reward: **+39.7372**
- 训练期 worst reward: **-7.4208**
- PPO 更新次数: **10**
- baseline avg_k (per-block 加权): **13.000**

## 2. 训练期 Top-20 candidates

**说明**：按 hard-priority + unbounded P3 cost rank 排序。P1/P2 不吃 cost；P3 内部先看无上限 cost rank，再看 fusion/K/bits 与 reward。每条候选的完整 SF / K 配置见 `top_candidates.jsonl` 的 `slots` 字段。

| Rank | Episode | P | cost_rank | total_reward | terminal | per_step_sum | valid | invalid | total_bits | fusion |
|-----:|--------:|--:|----------:|-------------:|---------:|-------------:|------:|--------:|-----------:|-------:|
| 1 | 251 | 3 | +2480.0000 | +39.7372 | +42.0726 | -2.3354 | 47 | 0 | 10861 | 14 |
| 2 | 165 | 3 | +2430.0000 | +39.5306 | +41.8455 | -2.3149 | 47 | 0 | 10799 | 17 |
| 3 | 487 | 3 | +2380.0000 | +39.5089 | +41.8625 | -2.3536 | 47 | 0 | 10913 | 12 |
| 4 | 186 | 3 | +2200.0000 | +39.3169 | +41.6503 | -2.3334 | 47 | 0 | 10891 | 13 |
| 5 | 305 | 3 | +2160.0000 | +39.3431 | +41.7225 | -2.3794 | 47 | 0 | 10969 | 10 |
| 6 | 504 | 3 | +2150.0000 | +39.1979 | +41.5696 | -2.3717 | 47 | 0 | 10965 | 10 |
| 7 | 158 | 3 | +2000.0000 | +39.2287 | +41.5718 | -2.3432 | 47 | 0 | 10899 | 13 |
| 8 | 330 | 3 | +1880.0000 | +39.0979 | +41.4954 | -2.3976 | 47 | 0 | 11021 | 8 |
| 9 | 461 | 3 | +1570.0000 | +39.0440 | +41.4934 | -2.4495 | 47 | 0 | 11133 | 4 |
| 10 | 40 | 3 | +0.0000 | +38.0065 | +40.5000 | -2.4935 | 47 | 0 | 11241 | 0 |
| 11 | 49 | 3 | +0.0000 | +38.0065 | +40.5000 | -2.4935 | 47 | 0 | 11241 | 0 |
| 12 | 52 | 3 | +0.0000 | +38.0065 | +40.5000 | -2.4935 | 47 | 0 | 11241 | 0 |
| 13 | 56 | 3 | +0.0000 | +38.0065 | +40.5000 | -2.4935 | 47 | 0 | 11241 | 0 |
| 14 | 79 | 3 | +0.0000 | +38.0065 | +40.5000 | -2.4935 | 47 | 0 | 11241 | 0 |
| 15 | 64 | 3 | +0.0000 | +38.0065 | +40.5000 | -2.4935 | 47 | 0 | 11241 | 0 |
| 16 | 54 | 3 | +0.0000 | +38.0065 | +40.5000 | -2.4935 | 47 | 0 | 11241 | 0 |
| 17 | 59 | 3 | +0.0000 | +38.0065 | +40.5000 | -2.4935 | 47 | 0 | 11241 | 0 |
| 18 | 77 | 3 | +0.0000 | +38.0065 | +40.5000 | -2.4935 | 47 | 0 | 11241 | 0 |
| 19 | 74 | 3 | +0.0000 | +38.0065 | +40.5000 | -2.4935 | 47 | 0 | 11241 | 0 |
| 20 | 71 | 3 | +0.0000 | +38.0065 | +40.5000 | -2.4935 | 47 | 0 | 11241 | 0 |

### 2.1 Best vs baseline 槽位 diff（哪些槽变了，变了多少）

_共 147 个槽与 baseline 不同_（116 SF + 31 K）

**Truncation K diffs**：

| Slot | Baseline K | Best K | Δ |
|:-----|----------:|------:|--:|
| `L0.B2.K` | 13 | 11 | -2 |
| `L0.B4.K` | 13 | 10 | -3 |
| `L0.B5.K` | 13 | 8 | -5 |
| `L1.B1.K` | 13 | 11 | -2 |
| `L1.B2.K` | 13 | 12 | -1 |
| `L1.B4.K` | 13 | 9 | -4 |
| `L1.B5.K` | 13 | 11 | -2 |
| `L10.B1.K` | 13 | 9 | -4 |
| `L11.B1.K` | 13 | 11 | -2 |
| `L11.B2.K` | 13 | 9 | -4 |
| `L11.B5.K` | 13 | 12 | -1 |
| `L2.B1.K` | 13 | 9 | -4 |
| `L2.B5.K` | 13 | 11 | -2 |
| `L3.B1.K` | 13 | 9 | -4 |
| `L3.B4.K` | 13 | 12 | -1 |
| `L4.B2.K` | 13 | 9 | -4 |
| `L4.B4.K` | 13 | 11 | -2 |
| `L4.B5.K` | 13 | 9 | -4 |
| `L5.B1.K` | 13 | 10 | -3 |
| `L5.B2.K` | 13 | 8 | -5 |
| `L5.B4.K` | 13 | 10 | -3 |
| `L5.B5.K` | 13 | 12 | -1 |
| `L6.B4.K` | 13 | 9 | -4 |
| `L6.B5.K` | 13 | 10 | -3 |
| `L7.B4.K` | 13 | 9 | -4 |
| `L7.B5.K` | 13 | 11 | -2 |
| `L8.B1.K` | 13 | 12 | -1 |
| `L8.B5.K` | 13 | 11 | -2 |
| `L9.B1.K` | 13 | 8 | -5 |
| `L9.B2.K` | 13 | 8 | -5 |
| `L9.B5.K` | 13 | 10 | -3 |

**Scaling-factor diffs**（前 20 条按 |Δ| 降序）：

| Slot | Kind | Baseline SF | Best SF | Δ |
|:-----|:----:|------------:|--------:|--:|
| `L0.B4.R.ln_mean_r` | R | 31 | 21 | -10 |
| `L1.B4.R.ln_mean_r` | R | 31 | 21 | -10 |
| `L2.B4.R.ln_mean_r` | R | 31 | 21 | -10 |
| `L6.B4.R.ln_mean_r` | R | 31 | 21 | -10 |
| `L10.B4.R.ln_mean_r` | R | 31 | 21 | -10 |
| `L0.B2.F.inv_std_fresh` | F | 28 | 20 | -8 |
| `L0.B4.R.softmax_v_matmul_r` | R | 31 | 23 | -8 |
| `L1.B4.R.softmax_v_matmul_r` | R | 31 | 23 | -8 |
| `L2.B4.R.softmax_v_matmul_r` | R | 31 | 23 | -8 |
| `L3.B2.F.inv_std_fresh` | F | 28 | 20 | -8 |
| `L4.B5.W.wffn1` | W | 22 | 14 | -8 |
| `L6.B4.R.softmax_v_matmul_r` | R | 31 | 23 | -8 |
| `L6.B5.W.wffn1` | W | 22 | 14 | -8 |
| `L7.B2.F.inv_std_fresh` | F | 28 | 20 | -8 |
| `L8.B2.F.inv_std_fresh` | F | 28 | 20 | -8 |
| `L9.B2.F.inv_std_fresh` | F | 28 | 20 | -8 |
| `L9.B5.W.wffn1` | W | 22 | 14 | -8 |
| `L10.B4.R.softmax_v_matmul_r` | R | 31 | 23 | -8 |
| `L11.B5.W.wffn1` | W | 22 | 14 | -8 |
| `L0.B2.W.wk` | W | 22 | 16 | -6 |

完整 diff 在 `best_action_vec.json` 的 `diff_vs_baseline` 字段。

## 3. First-invalid 频次（哪些 (layer, block) 最先翻车）

_暂无 invalid 记录（所有 episode 完整通过）。_

## 4. PPO 学习动态（最近 10 次更新）

| Update | Eps | policy_loss | value_loss | entropy | clip_frac | ent_coef | approx_kl | lr_scale | entropy_recovery | win_mean_ret | win_mean_inv |
|------:|----:|------------:|-----------:|--------:|----------:|---------:|----------:|---------:|-----------------:|-------------:|-------------:|
| 1 | 60 | -0.1054 | +0.3365 | +2.2057 | 0.550 | 0.00000 | 0.07481 | 1.000 | 0.00000 | +37.9580 | 0.00 |
| 2 | 120 | -0.0595 | +0.2989 | +2.1195 | 0.520 | 0.00133 | -0.05446 | 0.500 | 0.00000 | +7.7788 | 0.00 |
| 3 | 180 | +0.0026 | +0.0447 | +2.2236 | 0.060 | 0.00333 | 0.00318 | 0.500 | 0.00000 | -5.7547 | 0.00 |
| 4 | 240 | -0.0134 | +0.0230 | +2.2734 | 0.077 | 0.00533 | 0.00871 | 0.600 | 0.00000 | -6.5329 | 0.00 |
| 5 | 300 | -0.0032 | +0.0215 | +2.3091 | 0.065 | 0.00733 | 0.00390 | 0.720 | 0.00000 | -6.5130 | 0.00 |
| 6 | 360 | -0.0002 | +0.0416 | +2.3145 | 0.126 | 0.00933 | 0.01023 | 0.864 | 0.00000 | -5.7589 | 0.00 |
| 7 | 420 | -0.0049 | +0.0002 | +2.3293 | 0.069 | 0.01133 | 0.00991 | 0.864 | 0.00000 | -7.3175 | 0.00 |
| 8 | 480 | +0.0071 | +0.0228 | +2.3381 | 0.177 | 0.01333 | 0.00821 | 1.037 | 0.00000 | -6.5261 | 0.00 |
| 9 | 540 | -0.0091 | +0.0472 | +2.3784 | 0.207 | 0.01533 | 0.01519 | 1.244 | 0.00000 | -5.7480 | 0.00 |
| 10 | 600 | -0.0144 | +0.0001 | +2.4255 | 0.183 | 0.01733 | 0.02252 | 1.244 | 0.00000 | -7.2968 | 0.00 |

_Entropy 趋势：+2.2057 → +2.4255（上升（policy 在分散））_

## 5. 动作分布概览（哪些 slot 已经在按 baseline 取最大档）

- **已收敛 slot**（top-1 占比 > 85%）：**345** / 877
- **未收敛 slot**：**47** / 877

已收敛 slot 示例（前 8 个）：
  - slot[008] → action_index=3 （占比 100.0%）
  - slot[009] → action_index=5 （占比 100.0%）
  - slot[018] → action_index=0 （占比 100.0%）
  - slot[019] → action_index=0 （占比 100.0%）
  - slot[023] → action_index=0 （占比 100.0%）
  - slot[024] → action_index=0 （占比 100.0%）
  - slot[025] → action_index=0 （占比 100.0%）
  - slot[026] → action_index=0 （占比 100.0%）

最分散 slot 示例（前 8 个）：
  - slot[154] entropy=1.701 (uniform≈1.792)
  - slot[104] entropy=1.701 (uniform≈1.792)
  - slot[291] entropy=1.693 (uniform≈1.792)
  - slot[072] entropy=1.693 (uniform≈1.792)
  - slot[510] entropy=1.688 (uniform≈1.792)
  - slot[129] entropy=1.687 (uniform≈1.792)
  - slot[081] entropy=1.683 (uniform≈1.792)
  - slot[437] entropy=1.680 (uniform≈1.792)

## 6. 自动诊断（auto-flags）

- ⚠ **学习退化**：最近 20 回合平均回报 -7.2944 低于前 20 回合 +37.9549（Δ=-45.2493）。建议：降低 lr / 增加 ent_coef / 检查 invalid_penalty 是否过强。

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