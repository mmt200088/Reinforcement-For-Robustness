# BLB Stage-2 Sequential RL · 诊断汇总（diagnostics @ episode=5000）

_更新时间: 2026-05-16 18:10:36_  ·  累计用时: **13m55s**

**Run meta**：
- `profile` = `mrpc`
- `fixed_label` = `Stage-1 config (json)`
- `fixed_source` = `json`
- `rl_variant` = `blb_v3_sequential`
- `total_episodes_planned` = `6000`
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
- `acc_threshold` = `0.0`
- `stab_threshold` = `0.001`
- `static_skeletons_archive` = `/var/tmp/root-home/Reinforcement-For-Robustness/Rescale_optimizer/configs/mrpc/static_skeletons_mrpc.json`

## 1. 训练进度（training progress）

- 已完成回合数: **5000**
- 最近 50 回合 mean return: **-35.4958** (min=-39.3136, max=-33.5526)
- 最近 50 回合 mean terminal reward: **-30.0000**
- 最近 50 回合 mean invalid 子步数: **3.00** / 59
- 训练期 best reward: **+28.6937**
- 训练期 worst reward: **-50.6912**
- PPO 更新次数: **83**
- baseline avg_k (per-block 加权): **11.780**

## 2. 训练期 Top-20 candidates

**说明**：按 total_reward 排序。每条候选的完整 SF / K 配置（按槽位标签）见 `top_candidates.jsonl` 的 `slots` 字段；下面摘要表只列指标。

| Rank | Episode | total_reward | terminal | per_step_sum | valid | invalid | total_bits | fusion |
|-----:|--------:|-------------:|---------:|-------------:|------:|--------:|-----------:|-------:|
| 1 | 3612 | +28.6937 | +31.2339 | -2.5402 | 59 | 0 | 13663 | 7 |
| 2 | 3759 | +27.7598 | +30.3605 | -2.6006 | 59 | 0 | 13797 | 4 |
| 3 | 3642 | +27.4506 | +30.0192 | -2.5687 | 59 | 0 | 13733 | 6 |
| 4 | 3564 | +27.0463 | +29.6192 | -2.5729 | 59 | 0 | 13745 | 6 |
| 5 | 3730 | +26.8486 | +29.4328 | -2.5842 | 59 | 0 | 13745 | 6 |
| 6 | 3540 | +26.2919 | +28.8712 | -2.5792 | 59 | 0 | 13741 | 7 |
| 7 | 3752 | +25.3981 | +27.9684 | -2.5703 | 59 | 0 | 13733 | 8 |
| 8 | 4104 | +25.2665 | +27.8723 | -2.6058 | 59 | 0 | 13833 | 5 |
| 9 | 4073 | +25.0214 | +27.6271 | -2.6058 | 59 | 0 | 13789 | 7 |
| 10 | 3762 | +24.8757 | +27.4655 | -2.5898 | 59 | 0 | 13773 | 7 |
| 11 | 3930 | +24.7875 | +27.3571 | -2.5695 | 59 | 0 | 13731 | 9 |
| 12 | 3457 | +24.4883 | +27.0927 | -2.6044 | 59 | 0 | 13803 | 7 |
| 13 | 3044 | +24.3658 | +26.9627 | -2.5969 | 59 | 0 | 13813 | 7 |
| 14 | 3379 | +24.2745 | +26.8531 | -2.5786 | 59 | 0 | 13739 | 9 |
| 15 | 3095 | +24.0364 | +26.6621 | -2.6258 | 59 | 0 | 13881 | 5 |
| 16 | 2486 | +23.6989 | +26.3232 | -2.6242 | 59 | 0 | 13881 | 5 |
| 17 | 3983 | +23.4197 | +26.0260 | -2.6063 | 59 | 0 | 13805 | 8 |
| 18 | 2665 | +23.3526 | +25.9446 | -2.5921 | 59 | 0 | 13781 | 9 |
| 19 | 4593 | +23.2432 | +25.8610 | -2.6179 | 59 | 0 | 13873 | 6 |
| 20 | 2480 | +23.1852 | +25.7989 | -2.6137 | 59 | 0 | 13853 | 7 |

### 2.1 Best vs baseline 槽位 diff（哪些槽变了，变了多少）

_共 439 个槽与 baseline 不同_（390 SF + 49 K）

**Truncation K diffs**：

| Slot | Baseline K | Best K | Δ |
|:-----|----------:|------:|--:|
| `L0.B1.K` | 13 | 8 | -5 |
| `L0.B2.K` | 10 | 12 | +2 |
| `L0.B3.K` | 13 | 10 | -3 |
| `L0.B4.K` | 10 | 9 | -1 |
| `L0.B5.K` | 13 | 12 | -1 |
| `L1.B1.K` | 13 | 12 | -1 |
| `L1.B2.K` | 10 | 12 | +2 |
| `L1.B3.K` | 13 | 9 | -4 |
| `L1.B4.K` | 10 | 8 | -2 |
| `L1.B5.K` | 13 | 10 | -3 |
| `L10.B2.K` | 10 | 11 | +1 |
| `L10.B3.K` | 13 | 10 | -3 |
| `L10.B4.K` | 10 | 13 | +3 |
| `L10.B5.K` | 13 | 10 | -3 |
| `L11.B1.K` | 13 | 12 | -1 |
| `L11.B3.K` | 13 | 9 | -4 |
| `L11.B4.K` | 10 | 13 | +3 |
| `L2.B1.K` | 13 | 12 | -1 |
| `L2.B2.K` | 10 | 9 | -1 |
| `L2.B3.K` | 13 | 8 | -5 |
| `L2.B4.K` | 10 | 8 | -2 |
| `L3.B1.K` | 13 | 10 | -3 |
| `L3.B2.K` | 10 | 8 | -2 |
| `L3.B5.K` | 13 | 9 | -4 |
| `L4.B1.K` | 13 | 12 | -1 |
| `L4.B2.K` | 10 | 8 | -2 |
| `L4.B3.K` | 13 | 12 | -1 |
| `L4.B4.K` | 10 | 11 | +1 |
| `L4.B5.K` | 13 | 9 | -4 |
| `L5.B2.K` | 10 | 11 | +1 |
| `L5.B5.K` | 13 | 12 | -1 |
| `L6.B2.K` | 10 | 9 | -1 |
| `L6.B3.K` | 13 | 10 | -3 |
| `L6.B4.K` | 10 | 13 | +3 |
| `L6.B5.K` | 13 | 11 | -2 |
| `L7.B1.K` | 13 | 11 | -2 |
| `L7.B2.K` | 10 | 11 | +1 |
| `L7.B3.K` | 13 | 9 | -4 |
| `L7.B4.K` | 10 | 9 | -1 |
| `L8.B1.K` | 13 | 10 | -3 |
| `L8.B2.K` | 10 | 13 | +3 |
| `L8.B3.K` | 13 | 9 | -4 |
| `L8.B4.K` | 10 | 13 | +3 |
| `L8.B5.K` | 13 | 8 | -5 |
| `L9.B1.K` | 13 | 12 | -1 |
| `L9.B2.K` | 10 | 8 | -2 |
| `L9.B3.K` | 13 | 10 | -3 |
| `L9.B4.K` | 10 | 13 | +3 |
| `L9.B5.K` | 13 | 10 | -3 |

**Scaling-factor diffs**（前 20 条按 |Δ| 降序）：

| Slot | Kind | Baseline SF | Best SF | Δ |
|:-----|:----:|------------:|--------:|--:|
| `L0.B1.F.gelu_out` | F | 30 | 22 | -8 |
| `L0.B1.W.wffn2` | W | 20 | 12 | -8 |
| `L0.B2.F.x_centered_fresh` | F | 30 | 22 | -8 |
| `L1.B5.F.inv_std_fresh` | F | 30 | 22 | -8 |
| `L2.B2.F.x_centered_fresh` | F | 30 | 22 | -8 |
| `L2.B4.W.wo` | W | 22 | 14 | -8 |
| `L2.B5.F.inv_std_fresh` | F | 30 | 22 | -8 |
| `L2.B5.F.x_centered_fresh` | F | 30 | 22 | -8 |
| `L3.B2.W.wk` | W | 22 | 14 | -8 |
| `L3.B4.F.softmax_out_fresh` | F | 35 | 27 | -8 |
| `L4.B3.F.x_fresh` | F | 28 | 20 | -8 |
| `L4.B5.W.wffn1` | W | 22 | 14 | -8 |
| `L5.B1.F.gelu_out` | F | 30 | 22 | -8 |
| `L5.B2.F.inv_std_fresh` | F | 31 | 23 | -8 |
| `L5.B2.W.wk` | W | 22 | 14 | -8 |
| `L5.B4.F.softmax_out_fresh` | F | 35 | 27 | -8 |
| `L5.B4.W.wo` | W | 22 | 14 | -8 |
| `L5.B5.W.wffn1` | W | 22 | 14 | -8 |
| `L6.B2.F.inv_std_fresh` | F | 31 | 23 | -8 |
| `L6.B2.F.x_centered_fresh` | F | 30 | 22 | -8 |

完整 diff 在 `best_action_vec.json` 的 `diff_vs_baseline` 字段。

## 3. First-invalid 频次（哪些 (layer, block) 最先翻车）

_包含至少一个 invalid 步的 episode 数：**4935** (98.7% 的总回合)_

| Rank | (L, B) | 频次 | 占 invalid 比 |
|-----:|:------:|----:|-------------:|
| 1 | L01-B1 | 862 | 17.5% |
| 2 | L02-B1 | 532 | 10.8% |
| 3 | L00-B3 | 438 | 8.9% |
| 4 | L02-B3 | 408 | 8.3% |
| 5 | L05-B5 | 325 | 6.6% |
| 6 | L01-B3 | 301 | 6.1% |
| 7 | L03-B1 | 275 | 5.6% |
| 8 | L00-B5 | 233 | 4.7% |
| 9 | L03-B3 | 229 | 4.6% |
| 10 | L04-B1 | 175 | 3.5% |
| 11 | L04-B3 | 154 | 3.1% |
| 12 | L01-B5 | 143 | 2.9% |
| 13 | L05-B3 | 115 | 2.3% |
| 14 | L05-B1 | 87 | 1.8% |
| 15 | L02-B5 | 79 | 1.6% |

## 4. PPO 学习动态（最近 10 次更新）

| Update | Eps | policy_loss | value_loss | entropy | clip_frac | win_mean_ret | win_mean_inv |
|------:|----:|------------:|-----------:|--------:|----------:|-------------:|-------------:|
| 74 | 4440 | +0.0159 | +4.1814 | +12.6235 | 0.508 | -33.2332 | 3.47 |
| 75 | 4500 | +0.0200 | +3.2321 | +12.6341 | 0.507 | -34.5964 | 3.90 |
| 76 | 4560 | +0.0215 | +2.6756 | +12.6781 | 0.505 | -35.3554 | 3.73 |
| 77 | 4620 | +0.0162 | +3.1990 | +12.6717 | 0.453 | -35.3161 | 3.78 |
| 78 | 4680 | +0.0208 | +2.8049 | +12.6496 | 0.499 | -34.8819 | 3.25 |
| 79 | 4740 | +0.0348 | +0.5096 | +12.5257 | 0.549 | -36.3932 | 3.95 |
| 80 | 4800 | +0.0237 | +0.8405 | +12.6236 | 0.544 | -34.7028 | 3.10 |
| 81 | 4860 | +0.0147 | +2.8620 | +12.6210 | 0.486 | -34.0249 | 3.33 |
| 82 | 4920 | +0.0404 | +0.4235 | +12.5807 | 0.536 | -35.7055 | 3.22 |
| 83 | 4980 | +0.0136 | +4.4221 | +12.5360 | 0.446 | -33.8126 | 3.08 |

_Entropy 趋势：+12.7594 → +12.5360（下降（policy 在收敛））_

## 5. 动作分布概览（哪些 slot 已经在按 baseline 取最大档）

- **已收敛 slot**（top-1 占比 > 85%）：**7** / 577
- **未收敛 slot**：**570** / 577

已收敛 slot 示例（前 8 个）：
  - slot[000] → action_index=0 （占比 100.0%）
  - slot[001] → action_index=0 （占比 100.0%）
  - slot[002] → action_index=0 （占比 100.0%）
  - slot[003] → action_index=0 （占比 100.0%）
  - slot[004] → action_index=0 （占比 100.0%）
  - slot[005] → action_index=0 （占比 100.0%）
  - slot[006] → action_index=0 （占比 100.0%）

最分散 slot 示例（前 8 个）：
  - slot[438] entropy=1.769 (uniform≈1.792)
  - slot[390] entropy=1.766 (uniform≈1.792)
  - slot[095] entropy=1.762 (uniform≈1.792)
  - slot[047] entropy=1.761 (uniform≈1.792)
  - slot[486] entropy=1.757 (uniform≈1.792)
  - slot[265] entropy=1.757 (uniform≈1.792)
  - slot[294] entropy=1.755 (uniform≈1.792)
  - slot[246] entropy=1.753 (uniform≈1.792)

## 6. 自动诊断（auto-flags）

- ⚠ **clip_fraction 偏高**：最近 3 次 PPO clip_frac=0.49（>0.40）。lr 可能过大，建议降低 lr 一档。

## 7. 原始数据文件（machine-readable）

| 文件 | 内容 |
|------|------|
| `episodes.jsonl` | 完整 per-episode 记录（append-only） |
| `ppo_updates.jsonl` | 完整 per-PPO-update 记录（append-only） |
| `top_candidates.jsonl` | Top-20 训练期 best：含每条候选的完整 `slots` 列表（人类可读） |
| `first_invalid_counts.json` | (L, B) → 首次 invalid 计数 |
| `action_histogram.npz` | (num_slots, max_levels) 频次矩阵 |
| `baseline_action_vec.json` | static_skeletons baseline 的完整 `slots` 视图（参照系） |
| `best_action_vec.json` | **训练期最优**：`slots` 列表（按 SF/K 选）+ `action_vec` 兜底字段。**可直接喂给 `Paean/run_final_eval.sh --action-config`** |

**重跑 final eval 的最简命令**（无需等训练结束）：

```bash
bash Paean/run_final_eval.sh \
    --preset mrpc-final-eval-only \
    --action-config Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.005_s2t0.005_s2st0.005__blb_stage2_20260516_174426/stage2_noise/progress/diagnostics/best_action_vec.json
```

**手动调几个槽位**：直接复制 `best_action_vec.json`，改里面 `slots` 列表中对应槽位的 `scaling_factor` 或 `truncation_bits`，存成新文件后 `--action-config` 指过去即可。支持简写 `{"base":"max", "overrides":[{"label":"L05.B3.K", "truncation_bits":10}]}`。