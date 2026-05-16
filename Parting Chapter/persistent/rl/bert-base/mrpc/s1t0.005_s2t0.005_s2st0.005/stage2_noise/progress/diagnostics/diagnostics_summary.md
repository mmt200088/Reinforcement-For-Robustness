# BLB Stage-2 Sequential RL · 诊断汇总（diagnostics @ episode=200）

_更新时间: 2026-05-17 01:51:46_  ·  累计用时: **15m07s**

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

- 已完成回合数: **200**
- 最近 50 回合 mean return: **-162.9470** (min=-167.7110, max=-158.3008)
- 最近 50 回合 mean terminal reward: **-150.0000**
- 最近 50 回合 mean invalid 子步数: **10.94** / 59
- 训练期 best reward: **-157.3687**
- 训练期 worst reward: **-168.8253**
- PPO 更新次数: **3**
- baseline avg_k (per-block 加权): **11.780**

## 2. 训练期 Top-20 candidates

**说明**：按 total_reward 排序。每条候选的完整 SF / K 配置（按槽位标签）见 `top_candidates.jsonl` 的 `slots` 字段；下面摘要表只列指标。

| Rank | Episode | total_reward | terminal | per_step_sum | valid | invalid | total_bits | fusion |
|-----:|--------:|-------------:|---------:|-------------:|------:|--------:|-----------:|-------:|
| 1 | 45 | -157.3687 | -150.0000 | -7.3687 | 54 | 5 | 13841 | 9 |
| 2 | 44 | -158.2877 | -150.0000 | -8.2877 | 53 | 6 | 13867 | 9 |
| 3 | 181 | -158.3008 | -150.0000 | -8.3008 | 53 | 6 | 13877 | 1 |
| 4 | 186 | -159.1851 | -150.0000 | -9.1851 | 52 | 7 | 13799 | 3 |
| 5 | 187 | -159.1898 | -150.0000 | -9.1898 | 52 | 7 | 13825 | 4 |
| 6 | 191 | -159.2177 | -150.0000 | -9.2177 | 52 | 7 | 13823 | 3 |
| 7 | 190 | -159.2182 | -150.0000 | -9.2182 | 52 | 7 | 13837 | 3 |
| 8 | 199 | -159.2230 | -150.0000 | -9.2230 | 52 | 7 | 13839 | 2 |
| 9 | 36 | -159.2246 | -150.0000 | -9.2246 | 52 | 7 | 13859 | 10 |
| 10 | 183 | -159.2662 | -150.0000 | -9.2662 | 52 | 7 | 13899 | 4 |
| 11 | 57 | -160.0895 | -150.0000 | -10.0895 | 51 | 8 | 13761 | 10 |
| 12 | 198 | -160.1144 | -150.0000 | -10.1144 | 51 | 8 | 13717 | 2 |
| 13 | 182 | -160.1424 | -150.0000 | -10.1424 | 51 | 8 | 13819 | 5 |
| 14 | 94 | -160.1615 | -150.0000 | -10.1615 | 51 | 8 | 13883 | 11 |
| 15 | 193 | -160.1686 | -150.0000 | -10.1686 | 51 | 8 | 13853 | 1 |
| 16 | 195 | -160.1854 | -150.0000 | -10.1854 | 51 | 8 | 13847 | 1 |
| 17 | 25 | -160.1879 | -150.0000 | -10.1879 | 51 | 8 | 13937 | 7 |
| 18 | 18 | -160.2123 | -150.0000 | -10.2123 | 51 | 8 | 13887 | 9 |
| 19 | 81 | -160.2481 | -150.0000 | -10.2481 | 51 | 8 | 13963 | 10 |
| 20 | 3 | -160.2895 | -150.0000 | -10.2895 | 51 | 8 | 13963 | 6 |

### 2.1 Best vs baseline 槽位 diff（哪些槽变了，变了多少）

_共 428 个槽与 baseline 不同_（374 SF + 54 K）

**Truncation K diffs**：

| Slot | Baseline K | Best K | Δ |
|:-----|----------:|------:|--:|
| `L0.B1.K` | 13 | 8 | -5 |
| `L0.B2.K` | 10 | 12 | +2 |
| `L0.B4.K` | 10 | 12 | +2 |
| `L0.B5.K` | 13 | 10 | -3 |
| `L1.B1.K` | 13 | 12 | -1 |
| `L1.B2.K` | 10 | 12 | +2 |
| `L1.B3.K` | 13 | 11 | -2 |
| `L1.B4.K` | 10 | 12 | +2 |
| `L1.B5.K` | 13 | 12 | -1 |
| `L10.B1.K` | 13 | 12 | -1 |
| `L10.B2.K` | 10 | 12 | +2 |
| `L10.B3.K` | 13 | 12 | -1 |
| `L10.B4.K` | 10 | 13 | +3 |
| `L11.B1.K` | 13 | 12 | -1 |
| `L11.B2.K` | 10 | 12 | +2 |
| `L11.B3.K` | 13 | 9 | -4 |
| `L11.B4.K` | 10 | 12 | +2 |
| `L11.B5.K` | 13 | 10 | -3 |
| `L2.B1.K` | 13 | 9 | -4 |
| `L2.B2.K` | 10 | 13 | +3 |
| `L2.B3.K` | 13 | 12 | -1 |
| `L2.B4.K` | 10 | 9 | -1 |
| `L2.B5.K` | 13 | 10 | -3 |
| `L3.B1.K` | 13 | 9 | -4 |
| `L3.B2.K` | 10 | 12 | +2 |
| `L3.B3.K` | 13 | 12 | -1 |
| `L3.B4.K` | 10 | 11 | +1 |
| `L3.B5.K` | 13 | 8 | -5 |
| `L4.B2.K` | 10 | 12 | +2 |
| `L4.B3.K` | 13 | 9 | -4 |
| `L4.B4.K` | 10 | 11 | +1 |
| `L4.B5.K` | 13 | 9 | -4 |
| `L5.B1.K` | 13 | 9 | -4 |
| `L5.B2.K` | 10 | 11 | +1 |
| `L5.B3.K` | 13 | 11 | -2 |
| `L5.B5.K` | 13 | 12 | -1 |
| `L6.B1.K` | 13 | 11 | -2 |
| `L6.B2.K` | 10 | 13 | +3 |
| `L6.B3.K` | 13 | 12 | -1 |
| `L6.B4.K` | 10 | 12 | +2 |
| `L6.B5.K` | 13 | 11 | -2 |
| `L7.B1.K` | 13 | 9 | -4 |
| `L7.B3.K` | 13 | 9 | -4 |
| `L7.B4.K` | 10 | 13 | +3 |
| `L8.B1.K` | 13 | 12 | -1 |
| `L8.B2.K` | 10 | 12 | +2 |
| `L8.B3.K` | 13 | 12 | -1 |
| `L8.B4.K` | 10 | 12 | +2 |
| `L8.B5.K` | 13 | 11 | -2 |
| `L9.B1.K` | 13 | 12 | -1 |
| `L9.B2.K` | 10 | 12 | +2 |
| `L9.B3.K` | 13 | 12 | -1 |
| `L9.B4.K` | 10 | 11 | +1 |
| `L9.B5.K` | 13 | 11 | -2 |

**Scaling-factor diffs**（前 20 条按 |Δ| 降序）：

| Slot | Kind | Baseline SF | Best SF | Δ |
|:-----|:----:|------------:|--------:|--:|
| `L0.B1.F.gelu_out` | F | 30 | 22 | -8 |
| `L0.B1.W.wffn2` | W | 20 | 12 | -8 |
| `L0.B2.F.inv_std_fresh` | F | 31 | 23 | -8 |
| `L0.B2.W.wk` | W | 22 | 14 | -8 |
| `L0.B2.W.wv` | W | 22 | 14 | -8 |
| `L1.B2.F.x_centered_fresh` | F | 30 | 22 | -8 |
| `L2.B3.F.x_fresh` | F | 28 | 20 | -8 |
| `L3.B2.W.wv` | W | 22 | 14 | -8 |
| `L3.B5.W.wffn1` | W | 22 | 14 | -8 |
| `L4.B1.W.wffn2` | W | 20 | 12 | -8 |
| `L4.B2.F.x_centered_fresh` | F | 30 | 22 | -8 |
| `L5.B1.F.gelu_out` | F | 30 | 22 | -8 |
| `L5.B5.F.inv_std_fresh` | F | 30 | 22 | -8 |
| `L5.B5.F.x_centered_fresh` | F | 31 | 23 | -8 |
| `L6.B2.F.x_centered_fresh` | F | 30 | 22 | -8 |
| `L6.B3.F.x_fresh` | F | 28 | 20 | -8 |
| `L7.B2.W.wk` | W | 22 | 14 | -8 |
| `L7.B4.F.v_fresh` | F | 30 | 22 | -8 |
| `L7.B4.W.wo` | W | 22 | 14 | -8 |
| `L10.B2.F.inv_std_fresh` | F | 31 | 23 | -8 |

完整 diff 在 `best_action_vec.json` 的 `diff_vs_baseline` 字段。

## 3. First-invalid 频次（哪些 (layer, block) 最先翻车）

_包含至少一个 invalid 步的 episode 数：**200** (100.0% 的总回合)_

| Rank | (L, B) | 频次 | 占 invalid 比 |
|-----:|:------:|----:|-------------:|
| 1 | L01-B1 | 107 | 53.5% |
| 2 | L00-B3 | 31 | 15.5% |
| 3 | L02-B1 | 28 | 14.0% |
| 4 | L01-B3 | 10 | 5.0% |
| 5 | L03-B1 | 6 | 3.0% |
| 6 | L00-B5 | 5 | 2.5% |
| 7 | L00-B4 | 5 | 2.5% |
| 8 | L02-B3 | 2 | 1.0% |
| 9 | L04-B3 | 2 | 1.0% |
| 10 | L04-B1 | 2 | 1.0% |
| 11 | L01-B5 | 1 | 0.5% |
| 12 | L05-B5 | 1 | 0.5% |

## 4. PPO 学习动态（最近 10 次更新）

| Update | Eps | policy_loss | value_loss | entropy | clip_frac | win_mean_ret | win_mean_inv |
|------:|----:|------------:|-----------:|--------:|----------:|-------------:|-------------:|
| 1 | 60 | +0.1544 | +1198.5562 | +10.9192 | 0.802 | -163.5032 | 11.50 |
| 2 | 120 | +0.4250 | +123.5313 | +6.1935 | 0.962 | -163.9842 | 11.97 |
| 3 | 180 | +0.3259 | +47.3268 | +5.4699 | 0.913 | -164.7077 | 12.80 |

_Entropy 趋势：+10.9192 → +5.4699（下降（policy 在收敛））_

## 5. 动作分布概览（哪些 slot 已经在按 baseline 取最大档）

- **已收敛 slot**（top-1 占比 > 85%）：**7** / 577
- **未收敛 slot**：**570** / 577

## 6. 自动诊断（auto-flags）

- ⚠ **first-invalid 集中**：54% 的 invalid 都首先发生在 L01-B1。建议：查 stage1_degree[1] / max_sfs 表对应 block 1 的项。

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