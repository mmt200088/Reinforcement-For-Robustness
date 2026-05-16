# BLB Stage-2 Sequential RL · 诊断汇总（diagnostics @ episode=4600）

_更新时间: 2026-05-17 00:05:16_  ·  累计用时: **5h39m08s**

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

- 已完成回合数: **4600**
- 最近 50 回合 mean return: **-153.4521** (min=-156.4473, max=-152.6088)
- 最近 50 回合 mean terminal reward: **-150.0000**
- 最近 50 回合 mean invalid 子步数: **0.86** / 59
- 训练期 best reward: **-152.5904**
- 训练期 worst reward: **-169.5768**
- PPO 更新次数: **76**
- baseline avg_k (per-block 加权): **11.780**

## 2. 训练期 Top-20 candidates

**说明**：按 total_reward 排序。每条候选的完整 SF / K 配置（按槽位标签）见 `top_candidates.jsonl` 的 `slots` 字段；下面摘要表只列指标。

| Rank | Episode | total_reward | terminal | per_step_sum | valid | invalid | total_bits | fusion |
|-----:|--------:|-------------:|---------:|-------------:|------:|--------:|-----------:|-------:|
| 1 | 2715 | -152.5904 | -150.0000 | -2.5904 | 59 | 0 | 13747 | 13 |
| 2 | 4487 | -152.5980 | -150.0000 | -2.5980 | 59 | 0 | 13795 | 13 |
| 3 | 1877 | -152.5983 | -150.0000 | -2.5983 | 59 | 0 | 13765 | 9 |
| 4 | 4371 | -152.5985 | -150.0000 | -2.5985 | 59 | 0 | 13791 | 13 |
| 5 | 4435 | -152.5999 | -150.0000 | -2.5999 | 59 | 0 | 13799 | 14 |
| 6 | 4477 | -152.6009 | -150.0000 | -2.6009 | 59 | 0 | 13795 | 14 |
| 7 | 4421 | -152.6043 | -150.0000 | -2.6043 | 59 | 0 | 13813 | 13 |
| 8 | 2938 | -152.6075 | -150.0000 | -2.6075 | 59 | 0 | 13803 | 10 |
| 9 | 4436 | -152.6084 | -150.0000 | -2.6084 | 59 | 0 | 13807 | 12 |
| 10 | 4565 | -152.6088 | -150.0000 | -2.6088 | 59 | 0 | 13815 | 14 |
| 11 | 4335 | -152.6097 | -150.0000 | -2.6097 | 59 | 0 | 13807 | 13 |
| 12 | 2054 | -152.6097 | -150.0000 | -2.6097 | 59 | 0 | 13789 | 9 |
| 13 | 1915 | -152.6100 | -150.0000 | -2.6100 | 59 | 0 | 13793 | 9 |
| 14 | 4464 | -152.6102 | -150.0000 | -2.6102 | 59 | 0 | 13817 | 12 |
| 15 | 2831 | -152.6103 | -150.0000 | -2.6103 | 59 | 0 | 13799 | 11 |
| 16 | 4417 | -152.6103 | -150.0000 | -2.6103 | 59 | 0 | 13819 | 14 |
| 17 | 2246 | -152.6111 | -150.0000 | -2.6111 | 59 | 0 | 13815 | 9 |
| 18 | 1863 | -152.6113 | -150.0000 | -2.6113 | 59 | 0 | 13793 | 8 |
| 19 | 2926 | -152.6120 | -150.0000 | -2.6120 | 59 | 0 | 13819 | 11 |
| 20 | 2458 | -152.6121 | -150.0000 | -2.6121 | 59 | 0 | 13809 | 9 |

### 2.1 Best vs baseline 槽位 diff（哪些槽变了，变了多少）

_共 427 个槽与 baseline 不同_（367 SF + 60 K）

**Truncation K diffs**：

| Slot | Baseline K | Best K | Δ |
|:-----|----------:|------:|--:|
| `L0.B1.K` | 13 | 8 | -5 |
| `L0.B2.K` | 10 | 11 | +1 |
| `L0.B3.K` | 13 | 8 | -5 |
| `L0.B4.K` | 10 | 11 | +1 |
| `L0.B5.K` | 13 | 10 | -3 |
| `L1.B1.K` | 13 | 8 | -5 |
| `L1.B2.K` | 10 | 11 | +1 |
| `L1.B3.K` | 13 | 8 | -5 |
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
| `L2.B1.K` | 13 | 8 | -5 |
| `L2.B2.K` | 10 | 11 | +1 |
| `L2.B3.K` | 13 | 8 | -5 |
| `L2.B4.K` | 10 | 11 | +1 |
| `L2.B5.K` | 13 | 10 | -3 |
| `L3.B1.K` | 13 | 8 | -5 |
| `L3.B2.K` | 10 | 11 | +1 |
| `L3.B3.K` | 13 | 8 | -5 |
| `L3.B4.K` | 10 | 11 | +1 |
| `L3.B5.K` | 13 | 10 | -3 |
| `L4.B1.K` | 13 | 8 | -5 |
| `L4.B2.K` | 10 | 11 | +1 |
| `L4.B3.K` | 13 | 8 | -5 |
| `L4.B4.K` | 10 | 11 | +1 |
| `L4.B5.K` | 13 | 10 | -3 |
| `L5.B1.K` | 13 | 8 | -5 |
| `L5.B2.K` | 10 | 11 | +1 |
| `L5.B3.K` | 13 | 8 | -5 |
| `L5.B4.K` | 10 | 11 | +1 |
| `L5.B5.K` | 13 | 10 | -3 |
| `L6.B1.K` | 13 | 8 | -5 |
| `L6.B2.K` | 10 | 11 | +1 |
| `L6.B3.K` | 13 | 8 | -5 |
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
| `L0.B2.W.wv` | W | 22 | 14 | -8 |
| `L0.B5.W.wffn1` | W | 22 | 14 | -8 |
| `L1.B5.W.wffn1` | W | 22 | 14 | -8 |
| `L2.B2.F.inv_std_fresh` | F | 31 | 23 | -8 |
| `L2.B2.W.wv` | W | 22 | 14 | -8 |
| `L3.B2.W.wv` | W | 22 | 14 | -8 |
| `L4.B2.W.wk` | W | 22 | 14 | -8 |
| `L4.B5.W.wffn1` | W | 22 | 14 | -8 |
| `L5.B5.W.wffn1` | W | 22 | 14 | -8 |
| `L7.B2.W.wv` | W | 22 | 14 | -8 |
| `L7.B5.W.wffn1` | W | 22 | 14 | -8 |
| `L8.B2.W.wv` | W | 22 | 14 | -8 |
| `L10.B2.W.wv` | W | 22 | 14 | -8 |
| `L0.B2.F.inv_std_fresh` | F | 31 | 25 | -6 |
| `L0.B3.F.x_fresh` | F | 27 | 21 | -6 |
| `L0.B4.F.softmax_out_fresh` | F | 35 | 29 | -6 |
| `L0.B5.F.inv_std_fresh` | F | 30 | 24 | -6 |
| `L1.B2.F.inv_std_fresh` | F | 31 | 25 | -6 |

完整 diff 在 `best_action_vec.json` 的 `diff_vs_baseline` 字段。

## 3. First-invalid 频次（哪些 (layer, block) 最先翻车）

_包含至少一个 invalid 步的 episode 数：**4043** (87.9% 的总回合)_

| Rank | (L, B) | 频次 | 占 invalid 比 |
|-----:|:------:|----:|-------------:|
| 1 | L01-B1 | 840 | 20.8% |
| 2 | L05-B5 | 787 | 19.5% |
| 3 | L00-B3 | 457 | 11.3% |
| 4 | L02-B1 | 326 | 8.1% |
| 5 | L02-B3 | 275 | 6.8% |
| 6 | L01-B3 | 211 | 5.2% |
| 7 | L03-B1 | 181 | 4.5% |
| 8 | L04-B3 | 180 | 4.5% |
| 9 | L03-B3 | 172 | 4.3% |
| 10 | L04-B1 | 129 | 3.2% |
| 11 | L05-B1 | 80 | 2.0% |
| 12 | L05-B3 | 67 | 1.7% |
| 13 | L06-B3 | 65 | 1.6% |
| 14 | L08-B3 | 56 | 1.4% |
| 15 | L09-B3 | 56 | 1.4% |

## 4. PPO 学习动态（最近 10 次更新）

| Update | Eps | policy_loss | value_loss | entropy | clip_frac | win_mean_ret | win_mean_inv |
|------:|----:|------------:|-----------:|--------:|----------:|-------------:|-------------:|
| 67 | 4020 | +0.0733 | +0.1214 | +5.4205 | 0.627 | -153.9247 | 1.33 |
| 68 | 4080 | +0.0925 | +0.1052 | +5.3589 | 0.652 | -154.3427 | 1.77 |
| 69 | 4140 | +0.0709 | +0.1429 | +5.8067 | 0.615 | -154.5297 | 1.98 |
| 70 | 4200 | +0.1037 | +0.1241 | +5.4703 | 0.671 | -153.9965 | 1.43 |
| 71 | 4260 | +0.0779 | +0.1152 | +5.3165 | 0.643 | -154.1041 | 1.55 |
| 72 | 4320 | +0.0791 | +0.1341 | +5.2184 | 0.623 | -153.6319 | 1.05 |
| 73 | 4380 | +0.0505 | +0.1510 | +5.0996 | 0.614 | -153.5708 | 1.00 |
| 74 | 4440 | +0.0803 | +0.1245 | +5.1626 | 0.602 | -153.6352 | 1.07 |
| 75 | 4500 | +0.0742 | +0.0783 | +5.2625 | 0.623 | -153.5442 | 0.97 |
| 76 | 4560 | +0.0478 | +0.1442 | +5.4023 | 0.657 | -153.6173 | 1.03 |

_Entropy 趋势：+10.9192 → +5.4023（下降（policy 在收敛））_

## 5. 动作分布概览（哪些 slot 已经在按 baseline 取最大档）

- **已收敛 slot**（top-1 占比 > 85%）：**138** / 577
- **未收敛 slot**：**439** / 577

已收敛 slot 示例（前 8 个）：
  - slot[000] → action_index=0 （占比 100.0%）
  - slot[001] → action_index=0 （占比 100.0%）
  - slot[002] → action_index=0 （占比 100.0%）
  - slot[003] → action_index=0 （占比 100.0%）
  - slot[004] → action_index=0 （占比 100.0%）
  - slot[005] → action_index=0 （占比 100.0%）
  - slot[006] → action_index=0 （占比 100.0%）
  - slot[013] → action_index=0 （占比 95.3%）

最分散 slot 示例（前 8 个）：
  - slot[329] entropy=1.367 (uniform≈1.792)
  - slot[346] entropy=1.357 (uniform≈1.792)
  - slot[377] entropy=1.345 (uniform≈1.792)
  - slot[250] entropy=1.334 (uniform≈1.792)
  - slot[202] entropy=1.332 (uniform≈1.792)
  - slot[185] entropy=1.328 (uniform≈1.792)
  - slot[298] entropy=1.324 (uniform≈1.792)
  - slot[394] entropy=1.323 (uniform≈1.792)

## 6. 自动诊断（auto-flags）

- ⚠ **clip_fraction 偏高**：最近 3 次 PPO clip_frac=0.63（>0.40）。lr 可能过大，建议降低 lr 一档。

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