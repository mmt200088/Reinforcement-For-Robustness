# BLB Stage-2 Sequential RL · 诊断汇总（diagnostics @ episode=6000）

_更新时间: 2026-05-17 09:02:38_  ·  累计用时: **7h25m59s**

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

- 已完成回合数: **6000**
- 最近 50 回合 mean return: **-153.4909** (min=-155.5026, max=-152.5912)
- 最近 50 回合 mean terminal reward: **-150.0000**
- 最近 50 回合 mean invalid 子步数: **0.92** / 59
- 训练期 best reward: **-152.5556**
- 训练期 worst reward: **-169.5768**
- PPO 更新次数: **100**
- baseline avg_k (per-block 加权): **11.780**

## 2. 训练期 Top-20 candidates

**说明**：按 total_reward 排序。每条候选的完整 SF / K 配置（按槽位标签）见 `top_candidates.jsonl` 的 `slots` 字段；下面摘要表只列指标。

| Rank | Episode | total_reward | terminal | per_step_sum | valid | invalid | total_bits | fusion |
|-----:|--------:|-------------:|---------:|-------------:|------:|--------:|-----------:|-------:|
| 1 | 5648 | -152.5556 | -150.0000 | -2.5556 | 59 | 0 | 13685 | 10 |
| 2 | 5699 | -152.5574 | -150.0000 | -2.5574 | 59 | 0 | 13715 | 11 |
| 3 | 5686 | -152.5598 | -150.0000 | -2.5598 | 59 | 0 | 13697 | 13 |
| 4 | 5718 | -152.5620 | -150.0000 | -2.5620 | 59 | 0 | 13731 | 12 |
| 5 | 5666 | -152.5671 | -150.0000 | -2.5671 | 59 | 0 | 13731 | 14 |
| 6 | 5672 | -152.5697 | -150.0000 | -2.5697 | 59 | 0 | 13741 | 12 |
| 7 | 5656 | -152.5715 | -150.0000 | -2.5715 | 59 | 0 | 13705 | 12 |
| 8 | 5644 | -152.5738 | -150.0000 | -2.5738 | 59 | 0 | 13747 | 10 |
| 9 | 5924 | -152.5739 | -150.0000 | -2.5739 | 59 | 0 | 13751 | 9 |
| 10 | 5934 | -152.5751 | -150.0000 | -2.5751 | 59 | 0 | 13761 | 11 |
| 11 | 5892 | -152.5760 | -150.0000 | -2.5760 | 59 | 0 | 13753 | 11 |
| 12 | 4905 | -152.5760 | -150.0000 | -2.5760 | 59 | 0 | 13725 | 13 |
| 13 | 5650 | -152.5762 | -150.0000 | -2.5762 | 59 | 0 | 13737 | 12 |
| 14 | 5538 | -152.5763 | -150.0000 | -2.5763 | 59 | 0 | 13765 | 12 |
| 15 | 5757 | -152.5765 | -150.0000 | -2.5765 | 59 | 0 | 13747 | 10 |
| 16 | 4852 | -152.5794 | -150.0000 | -2.5794 | 59 | 0 | 13745 | 10 |
| 17 | 5407 | -152.5803 | -150.0000 | -2.5803 | 59 | 0 | 13745 | 13 |
| 18 | 5940 | -152.5810 | -150.0000 | -2.5810 | 59 | 0 | 13771 | 10 |
| 19 | 4737 | -152.5819 | -150.0000 | -2.5819 | 59 | 0 | 13749 | 8 |
| 20 | 5646 | -152.5820 | -150.0000 | -2.5820 | 59 | 0 | 13751 | 12 |

### 2.1 Best vs baseline 槽位 diff（哪些槽变了，变了多少）

_共 451 个槽与 baseline 不同_（393 SF + 58 K）

**Truncation K diffs**：

| Slot | Baseline K | Best K | Δ |
|:-----|----------:|------:|--:|
| `L0.B1.K` | 13 | 8 | -5 |
| `L0.B2.K` | 10 | 8 | -2 |
| `L0.B3.K` | 13 | 12 | -1 |
| `L0.B4.K` | 10 | 8 | -2 |
| `L0.B5.K` | 13 | 9 | -4 |
| `L1.B1.K` | 13 | 12 | -1 |
| `L1.B2.K` | 10 | 8 | -2 |
| `L1.B3.K` | 13 | 12 | -1 |
| `L1.B4.K` | 10 | 8 | -2 |
| `L1.B5.K` | 13 | 9 | -4 |
| `L10.B2.K` | 10 | 8 | -2 |
| `L10.B3.K` | 13 | 12 | -1 |
| `L10.B4.K` | 10 | 8 | -2 |
| `L10.B5.K` | 13 | 12 | -1 |
| `L11.B1.K` | 13 | 12 | -1 |
| `L11.B2.K` | 10 | 8 | -2 |
| `L11.B3.K` | 13 | 12 | -1 |
| `L11.B4.K` | 10 | 8 | -2 |
| `L11.B5.K` | 13 | 12 | -1 |
| `L2.B1.K` | 13 | 12 | -1 |
| `L2.B2.K` | 10 | 8 | -2 |
| `L2.B3.K` | 13 | 12 | -1 |
| `L2.B4.K` | 10 | 8 | -2 |
| `L2.B5.K` | 13 | 12 | -1 |
| `L3.B1.K` | 13 | 12 | -1 |
| `L3.B2.K` | 10 | 8 | -2 |
| `L3.B3.K` | 13 | 12 | -1 |
| `L3.B4.K` | 10 | 8 | -2 |
| `L3.B5.K` | 13 | 8 | -5 |
| `L4.B1.K` | 13 | 12 | -1 |
| `L4.B2.K` | 10 | 8 | -2 |
| `L4.B3.K` | 13 | 12 | -1 |
| `L4.B4.K` | 10 | 8 | -2 |
| `L4.B5.K` | 13 | 12 | -1 |
| `L5.B1.K` | 13 | 12 | -1 |
| `L5.B2.K` | 10 | 8 | -2 |
| `L5.B3.K` | 13 | 12 | -1 |
| `L5.B4.K` | 10 | 8 | -2 |
| `L5.B5.K` | 13 | 12 | -1 |
| `L6.B1.K` | 13 | 8 | -5 |
| `L6.B2.K` | 10 | 8 | -2 |
| `L6.B3.K` | 13 | 12 | -1 |
| `L6.B4.K` | 10 | 8 | -2 |
| `L6.B5.K` | 13 | 12 | -1 |
| `L7.B1.K` | 13 | 12 | -1 |
| `L7.B2.K` | 10 | 8 | -2 |
| `L7.B3.K` | 13 | 12 | -1 |
| `L7.B4.K` | 10 | 8 | -2 |
| `L7.B5.K` | 13 | 12 | -1 |
| `L8.B2.K` | 10 | 8 | -2 |
| `L8.B3.K` | 13 | 12 | -1 |
| `L8.B4.K` | 10 | 8 | -2 |
| `L8.B5.K` | 13 | 12 | -1 |
| `L9.B1.K` | 13 | 8 | -5 |
| `L9.B2.K` | 10 | 8 | -2 |
| `L9.B3.K` | 13 | 12 | -1 |
| `L9.B4.K` | 10 | 8 | -2 |
| `L9.B5.K` | 13 | 12 | -1 |

**Scaling-factor diffs**（前 20 条按 |Δ| 降序）：

| Slot | Kind | Baseline SF | Best SF | Δ |
|:-----|:----:|------------:|--------:|--:|
| `L0.B1.F.gelu_out` | F | 30 | 22 | -8 |
| `L0.B1.W.wffn2` | W | 20 | 12 | -8 |
| `L0.B2.W.wk` | W | 22 | 14 | -8 |
| `L1.B2.F.x_centered_fresh` | F | 30 | 22 | -8 |
| `L2.B2.W.wk` | W | 22 | 14 | -8 |
| `L2.B2.W.wv` | W | 22 | 14 | -8 |
| `L2.B4.F.v_fresh` | F | 30 | 22 | -8 |
| `L2.B5.W.wffn1` | W | 22 | 14 | -8 |
| `L3.B2.W.wk` | W | 22 | 14 | -8 |
| `L3.B5.W.wffn1` | W | 22 | 14 | -8 |
| `L4.B2.W.wk` | W | 22 | 14 | -8 |
| `L5.B2.W.wk` | W | 22 | 14 | -8 |
| `L8.B5.W.wffn1` | W | 22 | 14 | -8 |
| `L9.B2.W.wk` | W | 22 | 14 | -8 |
| `L9.B5.W.wffn1` | W | 22 | 14 | -8 |
| `L10.B2.W.wk` | W | 22 | 14 | -8 |
| `L11.B2.W.wk` | W | 22 | 14 | -8 |
| `L0.B2.W.wv` | W | 22 | 16 | -6 |
| `L0.B3.F.x_fresh` | F | 27 | 21 | -6 |
| `L0.B4.F.softmax_out_fresh` | F | 35 | 29 | -6 |

完整 diff 在 `best_action_vec.json` 的 `diff_vs_baseline` 字段。

## 3. First-invalid 频次（哪些 (layer, block) 最先翻车）

_包含至少一个 invalid 步的 episode 数：**4982** (83.0% 的总回合)_

| Rank | (L, B) | 频次 | 占 invalid 比 |
|-----:|:------:|----:|-------------:|
| 1 | L01-B1 | 973 | 19.5% |
| 2 | L05-B5 | 955 | 19.2% |
| 3 | L00-B3 | 467 | 9.4% |
| 4 | L02-B1 | 441 | 8.9% |
| 5 | L02-B3 | 315 | 6.3% |
| 6 | L03-B1 | 270 | 5.4% |
| 7 | L01-B3 | 228 | 4.6% |
| 8 | L04-B3 | 213 | 4.3% |
| 9 | L03-B3 | 205 | 4.1% |
| 10 | L04-B1 | 195 | 3.9% |
| 11 | L05-B1 | 103 | 2.1% |
| 12 | L06-B3 | 77 | 1.5% |
| 13 | L05-B3 | 74 | 1.5% |
| 14 | L08-B3 | 60 | 1.2% |
| 15 | L09-B3 | 58 | 1.2% |

## 4. PPO 学习动态（最近 10 次更新）

| Update | Eps | policy_loss | value_loss | entropy | clip_frac | win_mean_ret | win_mean_inv |
|------:|----:|------------:|-----------:|--------:|----------:|-------------:|-------------:|
| 91 | 5460 | +0.1412 | +0.4290 | +6.0813 | 0.718 | -153.4335 | 0.85 |
| 92 | 5520 | +0.0946 | +0.1345 | +6.0792 | 0.659 | -153.6654 | 1.12 |
| 93 | 5580 | +0.0891 | +0.1343 | +6.1473 | 0.663 | -153.9961 | 1.47 |
| 94 | 5640 | +0.0934 | +0.1531 | +6.0249 | 0.667 | -154.0482 | 1.53 |
| 95 | 5700 | +0.0966 | +0.0950 | +5.8087 | 0.672 | -153.7242 | 1.20 |
| 96 | 5760 | +0.0815 | +0.1240 | +6.1923 | 0.665 | -153.6995 | 1.15 |
| 97 | 5820 | +0.1153 | +0.1078 | +6.4511 | 0.710 | -153.6823 | 1.12 |
| 98 | 5880 | +0.0966 | +0.1007 | +6.4150 | 0.688 | -153.6738 | 1.12 |
| 99 | 5940 | +0.1309 | +0.1166 | +6.3217 | 0.742 | -153.3223 | 0.75 |
| 100 | 6000 | +0.0849 | +0.1163 | +6.4490 | 0.673 | -153.5035 | 0.93 |

_Entropy 趋势：+10.9192 → +6.4490（下降（policy 在收敛））_

## 5. 动作分布概览（哪些 slot 已经在按 baseline 取最大档）

- **已收敛 slot**（top-1 占比 > 85%）：**109** / 577
- **未收敛 slot**：**468** / 577

已收敛 slot 示例（前 8 个）：
  - slot[000] → action_index=0 （占比 100.0%）
  - slot[001] → action_index=0 （占比 100.0%）
  - slot[002] → action_index=0 （占比 100.0%）
  - slot[003] → action_index=0 （占比 100.0%）
  - slot[004] → action_index=0 （占比 100.0%）
  - slot[005] → action_index=0 （占比 100.0%）
  - slot[006] → action_index=0 （占比 100.0%）
  - slot[013] → action_index=0 （占比 91.9%）

最分散 slot 示例（前 8 个）：
  - slot[329] entropy=1.472 (uniform≈1.792)
  - slot[346] entropy=1.471 (uniform≈1.792)
  - slot[377] entropy=1.461 (uniform≈1.792)
  - slot[394] entropy=1.449 (uniform≈1.792)
  - slot[185] entropy=1.441 (uniform≈1.792)
  - slot[202] entropy=1.441 (uniform≈1.792)
  - slot[298] entropy=1.440 (uniform≈1.792)
  - slot[233] entropy=1.436 (uniform≈1.792)

## 6. 自动诊断（auto-flags）

- ⚠ **clip_fraction 偏高**：最近 3 次 PPO clip_frac=0.70（>0.40）。lr 可能过大，建议降低 lr 一档。

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