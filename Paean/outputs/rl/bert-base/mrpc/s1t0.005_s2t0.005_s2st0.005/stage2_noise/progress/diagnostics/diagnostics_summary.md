# BLB Stage-2 Sequential RL · 诊断汇总（diagnostics @ episode=5000）

_更新时间: 2026-05-17 00:34:53_  ·  累计用时: **6h08m45s**

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
- 最近 50 回合 mean return: **-153.5722** (min=-156.4351, max=-152.5863)
- 最近 50 回合 mean terminal reward: **-150.0000**
- 最近 50 回合 mean invalid 子步数: **1.00** / 59
- 训练期 best reward: **-152.5760**
- 训练期 worst reward: **-169.5768**
- PPO 更新次数: **83**
- baseline avg_k (per-block 加权): **11.780**

## 2. 训练期 Top-20 candidates

**说明**：按 total_reward 排序。每条候选的完整 SF / K 配置（按槽位标签）见 `top_candidates.jsonl` 的 `slots` 字段；下面摘要表只列指标。

| Rank | Episode | total_reward | terminal | per_step_sum | valid | invalid | total_bits | fusion |
|-----:|--------:|-------------:|---------:|-------------:|------:|--------:|-----------:|-------:|
| 1 | 4905 | -152.5760 | -150.0000 | -2.5760 | 59 | 0 | 13725 | 13 |
| 2 | 4852 | -152.5794 | -150.0000 | -2.5794 | 59 | 0 | 13745 | 10 |
| 3 | 4737 | -152.5819 | -150.0000 | -2.5819 | 59 | 0 | 13749 | 8 |
| 4 | 4848 | -152.5825 | -150.0000 | -2.5825 | 59 | 0 | 13745 | 12 |
| 5 | 4753 | -152.5846 | -150.0000 | -2.5846 | 59 | 0 | 13763 | 11 |
| 6 | 4797 | -152.5854 | -150.0000 | -2.5854 | 59 | 0 | 13743 | 12 |
| 7 | 4966 | -152.5863 | -150.0000 | -2.5863 | 59 | 0 | 13759 | 12 |
| 8 | 4810 | -152.5864 | -150.0000 | -2.5864 | 59 | 0 | 13747 | 10 |
| 9 | 4879 | -152.5878 | -150.0000 | -2.5878 | 59 | 0 | 13765 | 12 |
| 10 | 4789 | -152.5878 | -150.0000 | -2.5878 | 59 | 0 | 13749 | 11 |
| 11 | 4763 | -152.5890 | -150.0000 | -2.5890 | 59 | 0 | 13763 | 14 |
| 12 | 4767 | -152.5900 | -150.0000 | -2.5900 | 59 | 0 | 13757 | 12 |
| 13 | 4690 | -152.5902 | -150.0000 | -2.5902 | 59 | 0 | 13769 | 10 |
| 14 | 2715 | -152.5904 | -150.0000 | -2.5904 | 59 | 0 | 13747 | 13 |
| 15 | 4842 | -152.5919 | -150.0000 | -2.5919 | 59 | 0 | 13763 | 13 |
| 16 | 4846 | -152.5924 | -150.0000 | -2.5924 | 59 | 0 | 13765 | 14 |
| 17 | 4830 | -152.5932 | -150.0000 | -2.5932 | 59 | 0 | 13779 | 12 |
| 18 | 4886 | -152.5934 | -150.0000 | -2.5934 | 59 | 0 | 13761 | 12 |
| 19 | 4794 | -152.5936 | -150.0000 | -2.5936 | 59 | 0 | 13777 | 10 |
| 20 | 4687 | -152.5938 | -150.0000 | -2.5938 | 59 | 0 | 13765 | 11 |

### 2.1 Best vs baseline 槽位 diff（哪些槽变了，变了多少）

_共 427 个槽与 baseline 不同_（373 SF + 54 K）

**Truncation K diffs**：

| Slot | Baseline K | Best K | Δ |
|:-----|----------:|------:|--:|
| `L0.B1.K` | 13 | 8 | -5 |
| `L0.B2.K` | 10 | 11 | +1 |
| `L0.B3.K` | 13 | 8 | -5 |
| `L0.B4.K` | 10 | 11 | +1 |
| `L0.B5.K` | 13 | 9 | -4 |
| `L1.B1.K` | 13 | 12 | -1 |
| `L1.B2.K` | 10 | 11 | +1 |
| `L1.B4.K` | 10 | 11 | +1 |
| `L1.B5.K` | 13 | 10 | -3 |
| `L10.B1.K` | 13 | 8 | -5 |
| `L10.B2.K` | 10 | 11 | +1 |
| `L10.B3.K` | 13 | 8 | -5 |
| `L10.B4.K` | 10 | 11 | +1 |
| `L10.B5.K` | 13 | 10 | -3 |
| `L11.B1.K` | 13 | 8 | -5 |
| `L11.B2.K` | 10 | 11 | +1 |
| `L11.B3.K` | 13 | 8 | -5 |
| `L11.B4.K` | 10 | 11 | +1 |
| `L11.B5.K` | 13 | 10 | -3 |
| `L2.B1.K` | 13 | 12 | -1 |
| `L2.B2.K` | 10 | 11 | +1 |
| `L2.B3.K` | 13 | 12 | -1 |
| `L2.B4.K` | 10 | 11 | +1 |
| `L2.B5.K` | 13 | 9 | -4 |
| `L3.B2.K` | 10 | 11 | +1 |
| `L3.B4.K` | 10 | 11 | +1 |
| `L3.B5.K` | 13 | 10 | -3 |
| `L4.B2.K` | 10 | 11 | +1 |
| `L4.B3.K` | 13 | 12 | -1 |
| `L4.B4.K` | 10 | 11 | +1 |
| `L4.B5.K` | 13 | 9 | -4 |
| `L5.B1.K` | 13 | 11 | -2 |
| `L5.B2.K` | 10 | 11 | +1 |
| `L5.B4.K` | 10 | 11 | +1 |
| `L5.B5.K` | 13 | 10 | -3 |
| `L6.B2.K` | 10 | 11 | +1 |
| `L6.B3.K` | 13 | 12 | -1 |
| `L6.B4.K` | 10 | 11 | +1 |
| `L6.B5.K` | 13 | 10 | -3 |
| `L7.B1.K` | 13 | 8 | -5 |
| `L7.B2.K` | 10 | 11 | +1 |
| `L7.B3.K` | 13 | 8 | -5 |
| `L7.B4.K` | 10 | 11 | +1 |
| `L7.B5.K` | 13 | 10 | -3 |
| `L8.B1.K` | 13 | 8 | -5 |
| `L8.B2.K` | 10 | 11 | +1 |
| `L8.B3.K` | 13 | 8 | -5 |
| `L8.B4.K` | 10 | 11 | +1 |
| `L8.B5.K` | 13 | 10 | -3 |
| `L9.B1.K` | 13 | 8 | -5 |
| `L9.B2.K` | 10 | 11 | +1 |
| `L9.B3.K` | 13 | 8 | -5 |
| `L9.B4.K` | 10 | 11 | +1 |
| `L9.B5.K` | 13 | 10 | -3 |

**Scaling-factor diffs**（前 20 条按 |Δ| 降序）：

| Slot | Kind | Baseline SF | Best SF | Δ |
|:-----|:----:|------------:|--------:|--:|
| `L0.B1.F.gelu_out` | F | 30 | 22 | -8 |
| `L0.B1.W.wffn2` | W | 20 | 12 | -8 |
| `L0.B2.W.wk` | W | 22 | 14 | -8 |
| `L1.B2.W.wk` | W | 22 | 14 | -8 |
| `L1.B5.W.wffn1` | W | 22 | 14 | -8 |
| `L2.B5.W.wffn1` | W | 22 | 14 | -8 |
| `L3.B2.W.wk` | W | 22 | 14 | -8 |
| `L4.B5.W.wffn1` | W | 22 | 14 | -8 |
| `L5.B2.W.wk` | W | 22 | 14 | -8 |
| `L5.B5.W.wffn1` | W | 22 | 14 | -8 |
| `L6.B5.W.wffn1` | W | 22 | 14 | -8 |
| `L7.B2.W.wk` | W | 22 | 14 | -8 |
| `L8.B5.W.wffn1` | W | 22 | 14 | -8 |
| `L9.B2.W.wk` | W | 22 | 14 | -8 |
| `L0.first_input.F` | F | 30 | 22 | -8 |
| `L0.B3.F.x_fresh` | F | 27 | 21 | -6 |
| `L0.B4.F.softmax_out_fresh` | F | 35 | 29 | -6 |
| `L0.B4.W.wo` | W | 22 | 16 | -6 |
| `L1.B2.F.inv_std_fresh` | F | 31 | 25 | -6 |
| `L1.B2.W.wv` | W | 22 | 16 | -6 |

完整 diff 在 `best_action_vec.json` 的 `diff_vs_baseline` 字段。

## 3. First-invalid 频次（哪些 (layer, block) 最先翻车）

_包含至少一个 invalid 步的 episode 数：**4366** (87.3% 的总回合)_

| Rank | (L, B) | 频次 | 占 invalid 比 |
|-----:|:------:|----:|-------------:|
| 1 | L01-B1 | 896 | 20.5% |
| 2 | L05-B5 | 833 | 19.1% |
| 3 | L00-B3 | 461 | 10.6% |
| 4 | L02-B1 | 381 | 8.7% |
| 5 | L02-B3 | 285 | 6.5% |
| 6 | L03-B1 | 220 | 5.0% |
| 7 | L01-B3 | 218 | 5.0% |
| 8 | L04-B3 | 189 | 4.3% |
| 9 | L03-B3 | 178 | 4.1% |
| 10 | L04-B1 | 156 | 3.6% |
| 11 | L05-B1 | 91 | 2.1% |
| 12 | L05-B3 | 72 | 1.6% |
| 13 | L06-B3 | 66 | 1.5% |
| 14 | L08-B3 | 56 | 1.3% |
| 15 | L09-B3 | 56 | 1.3% |

## 4. PPO 学习动态（最近 10 次更新）

| Update | Eps | policy_loss | value_loss | entropy | clip_frac | win_mean_ret | win_mean_inv |
|------:|----:|------------:|-----------:|--------:|----------:|-------------:|-------------:|
| 74 | 4440 | +0.0803 | +0.1245 | +5.1626 | 0.602 | -153.6352 | 1.07 |
| 75 | 4500 | +0.0742 | +0.0783 | +5.2625 | 0.623 | -153.5442 | 0.97 |
| 76 | 4560 | +0.0478 | +0.1442 | +5.4023 | 0.657 | -153.6173 | 1.03 |
| 77 | 4620 | +0.1626 | +0.1323 | +5.4243 | 0.711 | -153.3934 | 0.80 |
| 78 | 4680 | +0.0437 | +0.2137 | +5.3507 | 0.574 | -154.3229 | 1.82 |
| 79 | 4740 | +0.0661 | +0.1123 | +5.3941 | 0.610 | -154.2545 | 1.75 |
| 80 | 4800 | +0.0719 | +0.1005 | +5.4910 | 0.648 | -153.9810 | 1.47 |
| 81 | 4860 | +0.0653 | +0.1108 | +5.4576 | 0.598 | -154.2098 | 1.72 |
| 82 | 4920 | +0.0739 | +0.1045 | +5.8226 | 0.676 | -153.9946 | 1.47 |
| 83 | 4980 | +0.0686 | +0.1050 | +5.8614 | 0.652 | -153.6637 | 1.10 |

_Entropy 趋势：+10.9192 → +5.8614（下降（policy 在收敛））_

## 5. 动作分布概览（哪些 slot 已经在按 baseline 取最大档）

- **已收敛 slot**（top-1 占比 > 85%）：**132** / 577
- **未收敛 slot**：**445** / 577

已收敛 slot 示例（前 8 个）：
  - slot[000] → action_index=0 （占比 100.0%）
  - slot[001] → action_index=0 （占比 100.0%）
  - slot[002] → action_index=0 （占比 100.0%）
  - slot[003] → action_index=0 （占比 100.0%）
  - slot[004] → action_index=0 （占比 100.0%）
  - slot[005] → action_index=0 （占比 100.0%）
  - slot[006] → action_index=0 （占比 100.0%）
  - slot[013] → action_index=0 （占比 94.4%）

最分散 slot 示例（前 8 个）：
  - slot[329] entropy=1.407 (uniform≈1.792)
  - slot[346] entropy=1.401 (uniform≈1.792)
  - slot[377] entropy=1.393 (uniform≈1.792)
  - slot[394] entropy=1.374 (uniform≈1.792)
  - slot[202] entropy=1.373 (uniform≈1.792)
  - slot[185] entropy=1.370 (uniform≈1.792)
  - slot[298] entropy=1.370 (uniform≈1.792)
  - slot[250] entropy=1.368 (uniform≈1.792)

## 6. 自动诊断（auto-flags）

- ⚠ **clip_fraction 偏高**：最近 3 次 PPO clip_frac=0.64（>0.40）。lr 可能过大，建议降低 lr 一档。

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
    --action-config Paean/outputs/rl/bert-base/mrpc/s1t0.005_s2t0.005_s2st0.005/stage2_noise/progress/diagnostics/best_action_vec.json
```

**手动调几个槽位**：直接复制 `best_action_vec.json`，改里面 `slots` 列表中对应槽位的 `scaling_factor` 或 `truncation_bits`，存成新文件后 `--action-config` 指过去即可。支持简写 `{"base":"max", "overrides":[{"label":"L05.B3.K", "truncation_bits":10}]}`。