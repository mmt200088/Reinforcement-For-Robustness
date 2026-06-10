# BLB Stage-2 Sequential RL · 诊断汇总（diagnostics @ episode=6000）

_更新时间: 2026-06-10 05:33:58_  ·  累计用时: **1h51m12s**

**Run meta**：
- `profile` = `mrpc`
- `fixed_label` = `Stage-1 config (stage1_record:bert base mrpc 1 20260610; softmax fixed deg6)`
- `fixed_source` = `stage1_record:bert base mrpc 1 20260610`
- `rl_variant` = `blb_v3_sequential_gtrxl_v2scale_fusioncount_v1`
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
- `static_invalid_level_mask_enabled` = `True`
- `acc_threshold` = `0.85984375`
- `stab_threshold` = `0.01`
- `static_skeletons_archive` = `/hy-tmp/server_command_ee69ce8_5gpu_20260610/src/Rescale_optimizer/configs/mrpc/static_skeletons_mrpc.json`
- `fast_reward_mode_enabled` = `False`
- `online_num_trials_per_step` = `1`
- `terminal_eval_batch_size` = `4`
- `promotion_validation_trials` = `4`
- `promotion_margin_window` = `0.25`

## 1. 训练进度（training progress）

- 已完成回合数: **6000**
- 最近 50 回合 mean return: **+38.9012** (min=+38.7049, max=+39.0894)
- 最近 50 回合 mean terminal reward: **+41.3563**
- 最近 50 回合 mean invalid 子步数: **0.00** / 47
- 训练期 best reward: **+40.1552**
- 训练期 worst reward: **-3.2110**
- PPO 更新次数: **100**
- baseline avg_k (per-block 加权): **13.000**

## 2. 训练期 Top-20 candidates

**说明**：按 hard-priority + unbounded P3 cost rank 排序。P1/P2 不吃 cost；P3 内部先看无上限 cost rank，再看 fusion/K/bits 与 reward。每条候选的完整 SF / K 配置见 `top_candidates.jsonl` 的 `slots` 字段。

| Rank | Episode | P | cost_rank | total_reward | terminal | per_step_sum | valid | invalid | total_bits | fusion |
|-----:|--------:|--:|----------:|-------------:|---------:|-------------:|------:|--------:|-----------:|-------:|
| 1 | 963 | 3 | +3160.0000 | +40.1301 | +42.4153 | -2.2852 | 47 | 0 | 10837 | 15 |
| 2 | 1441 | 3 | +2980.0000 | +40.0342 | +42.3192 | -2.2850 | 47 | 0 | 10836 | 14 |
| 3 | 1560 | 3 | +2980.0000 | +40.0375 | +42.3192 | -2.2816 | 47 | 0 | 10843 | 14 |
| 4 | 1256 | 3 | +2950.0000 | +39.9391 | +42.1932 | -2.2541 | 47 | 0 | 10782 | 15 |
| 5 | 1209 | 3 | +2930.0000 | +40.0398 | +42.2828 | -2.2430 | 47 | 0 | 10792 | 17 |
| 6 | 855 | 3 | +2860.0000 | +40.1552 | +42.4403 | -2.2851 | 47 | 0 | 10855 | 14 |
| 7 | 1079 | 3 | +2840.0000 | +39.7992 | +42.0785 | -2.2793 | 47 | 0 | 10822 | 15 |
| 8 | 1382 | 3 | +2830.0000 | +40.1133 | +42.3837 | -2.2705 | 47 | 0 | 10834 | 16 |
| 9 | 1033 | 3 | +2810.0000 | +39.9365 | +42.1609 | -2.2244 | 47 | 0 | 10701 | 18 |
| 10 | 1251 | 3 | +2810.0000 | +40.0841 | +42.3692 | -2.2851 | 47 | 0 | 10855 | 14 |
| 11 | 1453 | 3 | +2790.0000 | +39.9968 | +42.2852 | -2.2884 | 47 | 0 | 10848 | 14 |
| 12 | 1552 | 3 | +2790.0000 | +39.8998 | +42.1810 | -2.2812 | 47 | 0 | 10841 | 12 |
| 13 | 741 | 3 | +2780.0000 | +40.1391 | +42.4516 | -2.3125 | 47 | 0 | 10915 | 12 |
| 14 | 1337 | 3 | +2770.0000 | +40.0056 | +42.2360 | -2.2304 | 47 | 0 | 10772 | 17 |
| 15 | 1675 | 3 | +2750.0000 | +40.0299 | +42.2909 | -2.2609 | 47 | 0 | 10806 | 15 |
| 16 | 1493 | 3 | +2750.0000 | +40.0314 | +42.3256 | -2.2942 | 47 | 0 | 10881 | 13 |
| 17 | 1430 | 3 | +2740.0000 | +39.9748 | +42.2141 | -2.2393 | 47 | 0 | 10779 | 16 |
| 18 | 1629 | 3 | +2740.0000 | +40.0220 | +42.3530 | -2.3310 | 47 | 0 | 10987 | 11 |
| 19 | 1452 | 3 | +2730.0000 | +39.9477 | +42.1722 | -2.2245 | 47 | 0 | 10720 | 18 |
| 20 | 774 | 3 | +2730.0000 | +40.0496 | +42.3805 | -2.3308 | 47 | 0 | 10931 | 12 |

### 2.1 Best vs baseline 槽位 diff（哪些槽变了，变了多少）

_共 151 个槽与 baseline 不同_（112 SF + 39 K）

**Truncation K diffs**：

| Slot | Baseline K | Best K | Δ |
|:-----|----------:|------:|--:|
| `L0.B2.K` | 13 | 12 | -1 |
| `L0.B4.K` | 13 | 8 | -5 |
| `L0.B5.K` | 13 | 10 | -3 |
| `L1.B1.K` | 13 | 10 | -3 |
| `L1.B2.K` | 13 | 11 | -2 |
| `L1.B4.K` | 13 | 9 | -4 |
| `L1.B5.K` | 13 | 9 | -4 |
| `L10.B1.K` | 13 | 11 | -2 |
| `L10.B2.K` | 13 | 8 | -5 |
| `L10.B5.K` | 13 | 8 | -5 |
| `L11.B1.K` | 13 | 10 | -3 |
| `L11.B2.K` | 13 | 11 | -2 |
| `L11.B4.K` | 13 | 9 | -4 |
| `L11.B5.K` | 13 | 8 | -5 |
| `L2.B1.K` | 13 | 10 | -3 |
| `L2.B2.K` | 13 | 9 | -4 |
| `L2.B4.K` | 13 | 9 | -4 |
| `L2.B5.K` | 13 | 12 | -1 |
| `L3.B1.K` | 13 | 11 | -2 |
| `L3.B2.K` | 13 | 9 | -4 |
| `L3.B4.K` | 13 | 12 | -1 |
| `L3.B5.K` | 13 | 11 | -2 |
| `L4.B1.K` | 13 | 9 | -4 |
| `L4.B2.K` | 13 | 9 | -4 |
| `L4.B4.K` | 13 | 8 | -5 |
| `L4.B5.K` | 13 | 10 | -3 |
| `L5.B1.K` | 13 | 8 | -5 |
| `L5.B2.K` | 13 | 8 | -5 |
| `L5.B4.K` | 13 | 9 | -4 |
| `L5.B5.K` | 13 | 10 | -3 |
| `L6.B1.K` | 13 | 10 | -3 |
| `L6.B2.K` | 13 | 10 | -3 |
| `L6.B4.K` | 13 | 9 | -4 |
| `L6.B5.K` | 13 | 11 | -2 |
| `L7.B1.K` | 13 | 12 | -1 |
| `L7.B2.K` | 13 | 12 | -1 |
| `L7.B4.K` | 13 | 11 | -2 |
| `L8.B2.K` | 13 | 8 | -5 |
| `L9.B4.K` | 13 | 8 | -5 |

**Scaling-factor diffs**（前 20 条按 |Δ| 降序）：

| Slot | Kind | Baseline SF | Best SF | Δ |
|:-----|:----:|------------:|--------:|--:|
| `L10.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L11.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L10.B4.W.wo` | W | 22 | 11 | -11 |
| `L11.B4.W.wo` | W | 22 | 11 | -11 |
| `L0.B5.F.x_centered_fresh` | F | 31 | 21 | -10 |
| `L7.B5.F.x_centered_fresh` | F | 31 | 21 | -10 |
| `L0.B2.F.inv_std_fresh` | F | 28 | 20 | -8 |
| `L1.B2.F.inv_std_fresh` | F | 28 | 20 | -8 |
| `L2.B2.F.inv_std_fresh` | F | 28 | 20 | -8 |
| `L3.B2.F.inv_std_fresh` | F | 28 | 20 | -8 |
| `L4.B2.F.inv_std_fresh` | F | 28 | 20 | -8 |
| `L5.B2.F.inv_std_fresh` | F | 28 | 20 | -8 |
| `L6.B2.F.inv_std_fresh` | F | 28 | 20 | -8 |
| `L9.B2.F.inv_std_fresh` | F | 28 | 20 | -8 |
| `L10.B2.F.inv_std_fresh` | F | 28 | 20 | -8 |
| `L10.B4.F.v_fresh` | F | 25 | 17 | -8 |
| `L10.B4.S.ln_mean_inv_d` | S | 20 | 12 | -8 |
| `L11.B2.F.inv_std_fresh` | F | 28 | 20 | -8 |
| `L11.B4.F.v_fresh` | F | 25 | 17 | -8 |
| `L11.B4.S.ln_mean_inv_d` | S | 20 | 12 | -8 |

完整 diff 在 `best_action_vec.json` 的 `diff_vs_baseline` 字段。

## 3. First-invalid 频次（哪些 (layer, block) 最先翻车）

_暂无 invalid 记录（所有 episode 完整通过）。_

## 4. PPO 学习动态（最近 10 次更新）

| Update | Eps | policy_loss | value_loss | entropy | clip_frac | ent_coef | approx_kl | lr_scale | entropy_recovery | win_mean_ret | win_mean_inv |
|------:|----:|------------:|-----------:|--------:|----------:|---------:|----------:|---------:|-----------------:|-------------:|-------------:|
| 91 | 5460 | -0.0062 | +0.0007 | +1.6332 | 0.101 | 0.02000 | 0.00826 | 1.250 | 0.05847 | +38.8860 | 0.00 |
| 92 | 5520 | -0.0110 | +0.0014 | +1.6632 | 0.164 | 0.02000 | 0.02015 | 1.250 | 0.05847 | +38.9297 | 0.00 |
| 93 | 5580 | -0.0067 | +0.0015 | +1.6764 | 0.115 | 0.02000 | 0.01161 | 1.250 | 0.05847 | +38.8946 | 0.00 |
| 94 | 5640 | -0.0188 | +0.0009 | +1.6487 | 0.130 | 0.02000 | 0.01115 | 1.250 | 0.05837 | +38.8585 | 0.00 |
| 95 | 5700 | -0.0182 | +0.0009 | +1.6556 | 0.111 | 0.02000 | 0.01141 | 1.250 | 0.05857 | +38.8902 | 0.00 |
| 96 | 5760 | -0.0098 | +0.0008 | +1.6386 | 0.119 | 0.02000 | 0.00892 | 1.250 | 0.05847 | +38.8869 | 0.00 |
| 97 | 5820 | -0.0113 | +0.0008 | +1.6567 | 0.114 | 0.02000 | 0.00708 | 1.250 | 0.05847 | +38.8838 | 0.00 |
| 98 | 5880 | -0.0061 | +0.0006 | +1.6669 | 0.090 | 0.02000 | 0.00716 | 1.250 | 0.05847 | +38.9118 | 0.00 |
| 99 | 5940 | -0.0138 | +0.0015 | +1.6645 | 0.141 | 0.02000 | 0.01215 | 1.250 | 0.05827 | +38.9009 | 0.00 |
| 100 | 6000 | -0.0075 | +0.0007 | +1.6631 | 0.082 | 0.02000 | 0.00911 | 1.250 | 0.05837 | +38.9018 | 0.00 |

_Entropy 趋势：+2.0684 → +1.6631（下降（policy 在收敛））_

## 5. 动作分布概览（哪些 slot 已经在按 baseline 取最大档）

- **已收敛 slot**（top-1 占比 > 85%）：**368** / 877
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
  - slot[446] entropy=1.785 (uniform≈1.792)
  - slot[250] entropy=1.785 (uniform≈1.792)
  - slot[437] entropy=1.783 (uniform≈1.792)
  - slot[494] entropy=1.782 (uniform≈1.792)
  - slot[364] entropy=1.781 (uniform≈1.792)
  - slot[154] entropy=1.781 (uniform≈1.792)
  - slot[031] entropy=1.780 (uniform≈1.792)
  - slot[348] entropy=1.780 (uniform≈1.792)

## 6. 自动诊断（auto-flags）

- ✓ 暂无异常。

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