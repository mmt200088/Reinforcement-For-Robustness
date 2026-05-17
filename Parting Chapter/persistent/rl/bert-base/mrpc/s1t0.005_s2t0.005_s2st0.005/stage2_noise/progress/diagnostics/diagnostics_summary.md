# BLB Stage-2 Sequential RL · 诊断汇总（diagnostics @ episode=1000）

_更新时间: 2026-05-18 00:46:11_  ·  累计用时: **1h15m34s**

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
- `acc_threshold` = `0.86765625`
- `stab_threshold` = `0.05`
- `static_skeletons_archive` = `/var/tmp/root-home/Reinforcement-For-Robustness/Rescale_optimizer/configs/mrpc/static_skeletons_mrpc.json`

## 1. 训练进度（training progress）

- 已完成回合数: **1000**
- 最近 50 回合 mean return: **-164.6182** (min=-212.7627, max=-156.6662)
- 最近 50 回合 mean terminal reward: **-162.1094**
- 最近 50 回合 mean invalid 子步数: **0.00** / 59
- 训练期 best reward: **-156.6662**
- 训练期 worst reward: **-213.0960**
- PPO 更新次数: **16**
- baseline avg_k (per-block 加权): **11.780**

## 2. 训练期 Top-20 candidates

**说明**：按 total_reward 排序。每条候选的完整 SF / K 配置（按槽位标签）见 `top_candidates.jsonl` 的 `slots` 字段；下面摘要表只列指标。

| Rank | Episode | total_reward | terminal | per_step_sum | valid | invalid | total_bits | fusion |
|-----:|--------:|-------------:|---------:|-------------:|------:|--------:|-----------:|-------:|
| 1 | 960 | -156.6662 | -154.1562 | -2.5099 | 59 | 0 | 13615 | 13 |
| 2 | 789 | -157.2875 | -154.7812 | -2.5062 | 59 | 0 | 13607 | 13 |
| 3 | 819 | -157.2942 | -154.7812 | -2.5129 | 59 | 0 | 13625 | 13 |
| 4 | 847 | -157.4328 | -154.9375 | -2.4953 | 59 | 0 | 13569 | 13 |
| 5 | 727 | -157.7511 | -155.2500 | -2.5011 | 59 | 0 | 13591 | 13 |
| 6 | 743 | -158.0828 | -155.5625 | -2.5203 | 59 | 0 | 13649 | 13 |
| 7 | 706 | -158.2388 | -155.7188 | -2.5200 | 59 | 0 | 13659 | 13 |
| 8 | 967 | -158.3815 | -155.8750 | -2.5065 | 59 | 0 | 13607 | 12 |
| 9 | 605 | -158.5669 | -156.0312 | -2.5357 | 59 | 0 | 13701 | 12 |
| 10 | 331 | -158.5698 | -156.0312 | -2.5386 | 59 | 0 | 13663 | 7 |
| 11 | 993 | -158.6943 | -156.1875 | -2.5068 | 59 | 0 | 13607 | 13 |
| 12 | 535 | -158.7270 | -156.1875 | -2.5395 | 59 | 0 | 13713 | 12 |
| 13 | 807 | -158.8466 | -156.3438 | -2.5029 | 59 | 0 | 13597 | 14 |
| 14 | 585 | -158.8707 | -156.3438 | -2.5269 | 59 | 0 | 13673 | 13 |
| 15 | 656 | -159.0237 | -156.5000 | -2.5237 | 59 | 0 | 13661 | 13 |
| 16 | 597 | -159.0303 | -156.5000 | -2.5303 | 59 | 0 | 13683 | 12 |
| 17 | 510 | -159.0367 | -156.5000 | -2.5367 | 59 | 0 | 13705 | 10 |
| 18 | 745 | -159.1673 | -156.6562 | -2.5111 | 59 | 0 | 13621 | 13 |
| 19 | 664 | -159.1814 | -156.6562 | -2.5251 | 59 | 0 | 13665 | 12 |
| 20 | 657 | -159.1863 | -156.6562 | -2.5301 | 59 | 0 | 13683 | 12 |

### 2.1 Best vs baseline 槽位 diff（哪些槽变了，变了多少）

_共 408 个槽与 baseline 不同_（383 SF + 25 K）

**Truncation K diffs**：

| Slot | Baseline K | Best K | Δ |
|:-----|----------:|------:|--:|
| `L0.B1.K` | 13 | 8 | -5 |
| `L0.B3.K` | 13 | 12 | -1 |
| `L0.B5.K` | 13 | 11 | -2 |
| `L1.B3.K` | 13 | 12 | -1 |
| `L1.B5.K` | 13 | 11 | -2 |
| `L10.B3.K` | 13 | 12 | -1 |
| `L10.B5.K` | 13 | 11 | -2 |
| `L11.B3.K` | 13 | 12 | -1 |
| `L11.B5.K` | 13 | 11 | -2 |
| `L2.B3.K` | 13 | 12 | -1 |
| `L2.B5.K` | 13 | 11 | -2 |
| `L3.B3.K` | 13 | 12 | -1 |
| `L3.B5.K` | 13 | 11 | -2 |
| `L4.B3.K` | 13 | 12 | -1 |
| `L4.B5.K` | 13 | 11 | -2 |
| `L5.B3.K` | 13 | 12 | -1 |
| `L5.B5.K` | 13 | 11 | -2 |
| `L6.B3.K` | 13 | 12 | -1 |
| `L6.B5.K` | 13 | 11 | -2 |
| `L7.B3.K` | 13 | 12 | -1 |
| `L7.B5.K` | 13 | 11 | -2 |
| `L8.B3.K` | 13 | 12 | -1 |
| `L8.B5.K` | 13 | 11 | -2 |
| `L9.B3.K` | 13 | 12 | -1 |
| `L9.B5.K` | 13 | 11 | -2 |

**Scaling-factor diffs**（前 20 条按 |Δ| 降序）：

| Slot | Kind | Baseline SF | Best SF | Δ |
|:-----|:----:|------------:|--------:|--:|
| `L0.B1.F.gelu_out` | F | 30 | 22 | -8 |
| `L0.B1.W.wffn2` | W | 20 | 12 | -8 |
| `L0.B2.F.inv_std_fresh` | F | 31 | 23 | -8 |
| `L0.B2.F.x_centered_fresh` | F | 30 | 22 | -8 |
| `L0.B2.W.wk` | W | 22 | 14 | -8 |
| `L0.B4.F.softmax_out_fresh` | F | 35 | 27 | -8 |
| `L0.B4.F.v_fresh` | F | 30 | 22 | -8 |
| `L0.B4.W.wo` | W | 22 | 14 | -8 |
| `L0.B5.F.inv_std_fresh` | F | 30 | 22 | -8 |
| `L0.B5.F.x_centered_fresh` | F | 30 | 22 | -8 |
| `L0.B5.W.wffn1` | W | 22 | 14 | -8 |
| `L1.B2.F.x_centered_fresh` | F | 30 | 22 | -8 |
| `L1.B2.W.wk` | W | 22 | 14 | -8 |
| `L1.B4.F.v_fresh` | F | 30 | 22 | -8 |
| `L1.B4.W.wo` | W | 22 | 14 | -8 |
| `L1.B5.F.x_centered_fresh` | F | 30 | 22 | -8 |
| `L1.B5.W.wffn1` | W | 22 | 14 | -8 |
| `L2.B2.F.x_centered_fresh` | F | 30 | 22 | -8 |
| `L2.B2.W.wk` | W | 22 | 14 | -8 |
| `L2.B3.F.x_fresh` | F | 28 | 20 | -8 |

完整 diff 在 `best_action_vec.json` 的 `diff_vs_baseline` 字段。

## 3. First-invalid 频次（哪些 (layer, block) 最先翻车）

_暂无 invalid 记录（所有 episode 完整通过）。_

## 4. PPO 学习动态（最近 10 次更新）

| Update | Eps | policy_loss | value_loss | entropy | clip_frac | win_mean_ret | win_mean_inv |
|------:|----:|------------:|-----------:|--------:|----------:|-------------:|-------------:|
| 7 | 420 | +0.1519 | +8.2120 | +1.8214 | 0.853 | -212.8414 | 0.00 |
| 8 | 480 | +0.1339 | +70.6266 | +1.7582 | 0.765 | -180.4759 | 0.00 |
| 9 | 540 | +0.0999 | +78.6730 | +1.8114 | 0.652 | -179.1829 | 0.00 |
| 10 | 600 | +0.0876 | +45.2604 | +1.7337 | 0.598 | -172.3007 | 0.00 |
| 11 | 660 | +0.0940 | +8.3816 | +1.8124 | 0.705 | -164.2722 | 0.00 |
| 12 | 720 | +0.0857 | +12.0126 | +1.9828 | 0.734 | -165.5291 | 0.00 |
| 13 | 780 | +0.0555 | +6.4368 | +2.0612 | 0.593 | -164.6038 | 0.00 |
| 14 | 840 | +0.0792 | +1.5927 | +1.8801 | 0.556 | -163.6125 | 0.00 |
| 15 | 900 | +0.0655 | +1.0580 | +1.7721 | 0.566 | -163.8374 | 0.00 |
| 16 | 960 | +0.0720 | +1.0946 | +1.8659 | 0.557 | -163.6641 | 0.00 |

_Entropy 趋势：+8.6811 → +1.8659（下降（policy 在收敛））_

## 5. 动作分布概览（哪些 slot 已经在按 baseline 取最大档）

- **已收敛 slot**（top-1 占比 > 85%）：**111** / 577
- **未收敛 slot**：**466** / 577

已收敛 slot 示例（前 8 个）：
  - slot[000] → action_index=0 （占比 100.0%）
  - slot[001] → action_index=0 （占比 100.0%）
  - slot[002] → action_index=0 （占比 100.0%）
  - slot[003] → action_index=0 （占比 100.0%）
  - slot[004] → action_index=0 （占比 100.0%）
  - slot[005] → action_index=0 （占比 100.0%）
  - slot[006] → action_index=0 （占比 100.0%）
  - slot[018] → action_index=4 （占比 97.8%）

最分散 slot 示例（前 8 个）：
  - slot[427] entropy=1.252 (uniform≈1.792)
  - slot[552] entropy=1.248 (uniform≈1.792)
  - slot[379] entropy=1.237 (uniform≈1.792)
  - slot[504] entropy=1.233 (uniform≈1.792)
  - slot[523] entropy=1.228 (uniform≈1.792)
  - slot[571] entropy=1.226 (uniform≈1.792)
  - slot[187] entropy=1.226 (uniform≈1.792)
  - slot[331] entropy=1.220 (uniform≈1.792)

## 6. 自动诊断（auto-flags）

- ⚠ **clip_fraction 偏高**：最近 3 次 PPO clip_frac=0.56（>0.40）。lr 可能过大，建议降低 lr 一档。

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
    --action-config Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.005_s2t0.005_s2st0.005/stage2_noise/progress/diagnostics/best_action_vec.json
```

**手动调几个槽位**：直接复制 `best_action_vec.json`，改里面 `slots` 列表中对应槽位的 `scaling_factor` 或 `truncation_bits`，存成新文件后 `--action-config` 指过去即可。支持简写 `{"base":"max", "overrides":[{"label":"L05.B3.K", "truncation_bits":10}]}`。