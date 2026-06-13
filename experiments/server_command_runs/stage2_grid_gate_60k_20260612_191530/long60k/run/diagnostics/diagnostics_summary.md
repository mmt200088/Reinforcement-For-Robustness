# BLB Stage-2 Sequential RL · 诊断汇总（diagnostics @ episode=60000）

_更新时间: 2026-06-13 12:53:08_  ·  累计用时: **17h00m43s**

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
- `static_skeletons_archive` = `/hy-tmp/server_command_stage2_adr012_c887e3b_20260612_183904/src/Rescale_optimizer/configs/mrpc/static_skeletons_mrpc.json`
- `fast_reward_mode_enabled` = `False`
- `online_num_trials_per_step` = `1`
- `terminal_eval_batch_size` = `4`
- `promotion_validation_trials` = `4`
- `promotion_margin_window` = `0.25`

## 1. 训练进度（training progress）

- 已完成回合数: **60000**
- 最近 50 回合 mean return: **-6.9538** (min=-7.0034, max=-6.9415)
- 最近 50 回合 mean terminal reward: **-5.0000**
- 最近 50 回合 mean invalid 子步数: **0.00** / 47
- 训练期 best reward: **+40.8054**
- 训练期 worst reward: **-7.1967**
- PPO 更新次数: **1000**
- baseline avg_k (per-block 加权): **13.000**

## 2. 训练期 Top-20 candidates

**说明**：按 hard-priority + unbounded P3 cost rank 排序。P1/P2 不吃 cost；P3 内部先看无上限 cost rank，再看 fusion/K/bits 与 reward。每条候选的完整 SF / K 配置见 `top_candidates.jsonl` 的 `slots` 字段。

| Rank | Episode | P | cost_rank | total_reward | terminal | per_step_sum | valid | invalid | total_bits | fusion |
|-----:|--------:|--:|----------:|-------------:|---------:|-------------:|------:|--------:|-----------:|-------:|
| 1 | 28689 | 3 | +3760.0000 | +40.6863 | +42.7867 | -2.1004 | 47 | 0 | 10412 | 25 |
| 2 | 28094 | 3 | +3700.0000 | +40.7877 | +42.8732 | -2.0855 | 47 | 0 | 10371 | 26 |
| 3 | 22480 | 3 | +3690.0000 | +40.7135 | +42.8469 | -2.1333 | 47 | 0 | 10469 | 24 |
| 4 | 21680 | 3 | +3680.0000 | +40.6145 | +42.7833 | -2.1688 | 47 | 0 | 10535 | 22 |
| 5 | 27111 | 3 | +3670.0000 | +40.7203 | +42.8207 | -2.1004 | 47 | 0 | 10412 | 25 |
| 6 | 28217 | 3 | +3650.0000 | +40.5791 | +42.6979 | -2.1188 | 47 | 0 | 10465 | 24 |
| 7 | 26653 | 3 | +3630.0000 | +40.6036 | +42.7128 | -2.1092 | 47 | 0 | 10437 | 24 |
| 8 | 27880 | 3 | +3590.0000 | +40.7446 | +42.8411 | -2.0965 | 47 | 0 | 10439 | 26 |
| 9 | 20880 | 3 | +3590.0000 | +40.8054 | +42.9874 | -2.1820 | 47 | 0 | 10575 | 22 |
| 10 | 19280 | 3 | +3580.0000 | +40.5202 | +42.6890 | -2.1688 | 47 | 0 | 10535 | 22 |
| 11 | 27149 | 3 | +3550.0000 | +40.5528 | +42.6894 | -2.1366 | 47 | 0 | 10498 | 23 |
| 12 | 26058 | 3 | +3540.0000 | +40.5185 | +42.6596 | -2.1412 | 47 | 0 | 10491 | 22 |
| 13 | 27920 | 3 | +3510.0000 | +40.5601 | +42.6738 | -2.1138 | 47 | 0 | 10430 | 23 |
| 14 | 27298 | 3 | +3500.0000 | +40.5291 | +42.6479 | -2.1188 | 47 | 0 | 10465 | 24 |
| 15 | 26305 | 3 | +3490.0000 | +40.4840 | +42.6206 | -2.1366 | 47 | 0 | 10498 | 23 |
| 16 | 22280 | 3 | +3470.0000 | +40.6440 | +42.7221 | -2.0781 | 47 | 0 | 10386 | 27 |
| 17 | 28340 | 3 | +3470.0000 | +40.5083 | +42.6078 | -2.0995 | 47 | 0 | 10409 | 24 |
| 18 | 28023 | 3 | +3460.0000 | +40.6325 | +42.7691 | -2.1366 | 47 | 0 | 10498 | 23 |
| 19 | 28253 | 3 | +3420.0000 | +40.4326 | +42.5602 | -2.1276 | 47 | 0 | 10490 | 23 |
| 20 | 26697 | 3 | +3420.0000 | +40.4461 | +42.5730 | -2.1269 | 47 | 0 | 10470 | 23 |

### 2.1 Best vs baseline 槽位 diff（哪些槽变了，变了多少）

_共 202 个槽与 baseline 不同_（166 SF + 36 K）

**Truncation K diffs**：

| Slot | Baseline K | Best K | Δ |
|:-----|----------:|------:|--:|
| `L0.B2.K` | 13 | 8 | -5 |
| `L0.B4.K` | 13 | 12 | -1 |
| `L0.B5.K` | 13 | 8 | -5 |
| `L1.B1.K` | 13 | 9 | -4 |
| `L1.B2.K` | 13 | 11 | -2 |
| `L1.B4.K` | 13 | 9 | -4 |
| `L1.B5.K` | 13 | 9 | -4 |
| `L10.B2.K` | 13 | 9 | -4 |
| `L10.B4.K` | 13 | 12 | -1 |
| `L11.B1.K` | 13 | 11 | -2 |
| `L11.B4.K` | 13 | 12 | -1 |
| `L11.B5.K` | 13 | 8 | -5 |
| `L2.B1.K` | 13 | 10 | -3 |
| `L2.B2.K` | 13 | 12 | -1 |
| `L2.B4.K` | 13 | 9 | -4 |
| `L3.B4.K` | 13 | 11 | -2 |
| `L3.B5.K` | 13 | 12 | -1 |
| `L4.B2.K` | 13 | 11 | -2 |
| `L4.B4.K` | 13 | 12 | -1 |
| `L4.B5.K` | 13 | 9 | -4 |
| `L5.B1.K` | 13 | 9 | -4 |
| `L5.B4.K` | 13 | 10 | -3 |
| `L5.B5.K` | 13 | 10 | -3 |
| `L6.B1.K` | 13 | 11 | -2 |
| `L6.B4.K` | 13 | 8 | -5 |
| `L6.B5.K` | 13 | 10 | -3 |
| `L7.B1.K` | 13 | 12 | -1 |
| `L7.B2.K` | 13 | 9 | -4 |
| `L7.B4.K` | 13 | 9 | -4 |
| `L7.B5.K` | 13 | 12 | -1 |
| `L8.B1.K` | 13 | 11 | -2 |
| `L8.B4.K` | 13 | 10 | -3 |
| `L8.B5.K` | 13 | 9 | -4 |
| `L9.B1.K` | 13 | 11 | -2 |
| `L9.B2.K` | 13 | 9 | -4 |
| `L9.B4.K` | 13 | 9 | -4 |

**Scaling-factor diffs**（前 20 条按 |Δ| 降序）：

| Slot | Kind | Baseline SF | Best SF | Δ |
|:-----|:----:|------------:|--------:|--:|
| `L3.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L4.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L5.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L6.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L7.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L8.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L9.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L10.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L11.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L3.B4.W.wo` | W | 22 | 11 | -11 |
| `L4.B4.W.wo` | W | 22 | 11 | -11 |
| `L5.B4.W.wo` | W | 22 | 11 | -11 |
| `L6.B4.W.wo` | W | 22 | 11 | -11 |
| `L7.B4.W.wo` | W | 22 | 11 | -11 |
| `L8.B4.W.wo` | W | 22 | 11 | -11 |
| `L9.B4.W.wo` | W | 22 | 11 | -11 |
| `L10.B4.W.wo` | W | 22 | 11 | -11 |
| `L11.B4.W.wo` | W | 22 | 11 | -11 |
| `L3.B4.S.ln_mean_inv_d` | S | 20 | 11 | -9 |
| `L3.B5.F.x_centered_fresh` | F | 31 | 22 | -9 |

完整 diff 在 `best_action_vec.json` 的 `diff_vs_baseline` 字段。

## 3. First-invalid 频次（哪些 (layer, block) 最先翻车）

_暂无 invalid 记录（所有 episode 完整通过）。_

## 4. PPO 学习动态（最近 10 次更新）

| Update | Eps | policy_loss | value_loss | entropy | clip_frac | ent_coef | approx_kl | lr_scale | entropy_recovery | win_mean_ret | win_mean_inv |
|------:|----:|------------:|-----------:|--------:|----------:|---------:|----------:|---------:|-----------------:|-------------:|-------------:|
| 991 | 59460 | -0.0012 | +0.0000 | +1.7994 | 0.063 | 0.02000 | 0.00623 | 1.250 | 0.01360 | -6.9524 | 0.00 |
| 992 | 59520 | -0.0010 | +0.0000 | +1.8149 | 0.059 | 0.02000 | 0.00502 | 1.250 | 0.01369 | -6.9558 | 0.00 |
| 993 | 59580 | -0.0050 | +0.0000 | +1.7998 | 0.060 | 0.02000 | 0.00763 | 1.250 | 0.01369 | -6.9512 | 0.00 |
| 994 | 59640 | -0.0066 | +0.0000 | +1.7915 | 0.050 | 0.02000 | 0.00585 | 1.250 | 0.01360 | -6.9561 | 0.00 |
| 995 | 59700 | -0.0078 | +0.0000 | +1.7858 | 0.072 | 0.02000 | 0.00688 | 1.250 | 0.01360 | -6.9536 | 0.00 |
| 996 | 59760 | -0.0196 | +0.0000 | +1.7874 | 0.066 | 0.02000 | 0.00775 | 1.250 | 0.01351 | -6.9528 | 0.00 |
| 997 | 59820 | -0.0033 | +0.0000 | +1.8089 | 0.060 | 0.02000 | 0.00708 | 1.250 | 0.01367 | -6.9531 | 0.00 |
| 998 | 59880 | +0.0044 | +0.0000 | +1.8083 | 0.039 | 0.02000 | 0.00563 | 1.250 | 0.01360 | -6.9533 | 0.00 |
| 999 | 59940 | -0.0048 | +0.0000 | +1.8028 | 0.082 | 0.02000 | 0.00724 | 1.250 | 0.01365 | -6.9521 | 0.00 |
| 1000 | 60000 | -0.0070 | +0.0000 | +1.8028 | 0.049 | 0.02000 | 0.00393 | 1.250 | 0.01355 | -6.9538 | 0.00 |

_Entropy 趋势：+2.1773 → +1.8028（下降（policy 在收敛））_

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
  - slot[615] entropy=1.705 (uniform≈1.792)
  - slot[811] entropy=1.701 (uniform≈1.792)
  - slot[665] entropy=1.697 (uniform≈1.792)
  - slot[656] entropy=1.696 (uniform≈1.792)
  - slot[129] entropy=1.693 (uniform≈1.792)
  - slot[396] entropy=1.691 (uniform≈1.792)
  - slot[177] entropy=1.690 (uniform≈1.792)
  - slot[056] entropy=1.690 (uniform≈1.792)

## 6. 自动诊断（auto-flags）

- ⚠ **学习退化**：最近 20 回合平均回报 -6.9510 低于前 20 回合 +37.8772（Δ=-44.8282）。建议：降低 lr / 增加 ent_coef / 检查 invalid_penalty 是否过强。

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