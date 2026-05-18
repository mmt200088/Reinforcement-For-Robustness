# BLB Stage-2 Sequential RL · 诊断汇总（diagnostics @ episode=6000）

_更新时间: 2026-05-18 07:20:45_  ·  累计用时: **7h50m08s**

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

- 已完成回合数: **6000**
- 最近 50 回合 mean return: **-125.5488** (min=-129.9109, max=-121.0003)
- 最近 50 回合 mean terminal reward: **-122.7687**
- 最近 50 回合 mean invalid 子步数: **0.00** / 59
- 训练期 best reward: **-117.9821**
- 训练期 worst reward: **-213.0960**
- PPO 更新次数: **100**
- baseline avg_k (per-block 加权): **11.780**

## 2. 训练期 Top-20 candidates

**说明**：按 total_reward 排序。每条候选的完整 SF / K 配置（按槽位标签）见 `top_candidates.jsonl` 的 `slots` 字段；下面摘要表只列指标。

| Rank | Episode | total_reward | terminal | per_step_sum | valid | invalid | total_bits | fusion |
|-----:|--------:|-------------:|---------:|-------------:|------:|--------:|-----------:|-------:|
| 1 | 5666 | -117.9821 | -115.2500 | -2.7321 | 59 | 0 | 14061 | 5 |
| 2 | 5678 | -118.0023 | -115.2500 | -2.7523 | 59 | 0 | 14101 | 4 |
| 3 | 1783 | -119.0195 | -116.5000 | -2.5195 | 59 | 0 | 13653 | 13 |
| 4 | 5689 | -119.8608 | -117.1250 | -2.7358 | 59 | 0 | 14071 | 4 |
| 5 | 5859 | -119.8808 | -117.1250 | -2.7558 | 59 | 0 | 14113 | 4 |
| 6 | 3721 | -119.9347 | -117.4375 | -2.4972 | 59 | 0 | 13625 | 13 |
| 7 | 3239 | -119.9548 | -117.4375 | -2.5173 | 59 | 0 | 13663 | 13 |
| 8 | 2024 | -119.9566 | -117.4375 | -2.5191 | 59 | 0 | 13655 | 13 |
| 9 | 5824 | -120.0663 | -117.2812 | -2.7851 | 59 | 0 | 14185 | 2 |
| 10 | 4105 | -120.0940 | -117.5938 | -2.5003 | 59 | 0 | 13621 | 15 |
| 11 | 2511 | -120.1190 | -117.5938 | -2.5252 | 59 | 0 | 13677 | 13 |
| 12 | 2934 | -120.1232 | -117.5938 | -2.5295 | 59 | 0 | 13699 | 13 |
| 13 | 4597 | -120.1842 | -117.5938 | -2.5905 | 59 | 0 | 13827 | 12 |
| 14 | 4590 | -120.1862 | -117.5938 | -2.5925 | 59 | 0 | 13833 | 12 |
| 15 | 3299 | -120.2828 | -117.7500 | -2.5328 | 59 | 0 | 13703 | 13 |
| 16 | 2855 | -120.2886 | -117.7500 | -2.5386 | 59 | 0 | 13721 | 13 |
| 17 | 5185 | -120.3231 | -117.5938 | -2.7294 | 59 | 0 | 14077 | 5 |
| 18 | 5673 | -120.3333 | -117.5938 | -2.7396 | 59 | 0 | 14083 | 4 |
| 19 | 5294 | -120.3407 | -117.5938 | -2.7469 | 59 | 0 | 14111 | 4 |
| 20 | 5652 | -120.3490 | -117.5938 | -2.7552 | 59 | 0 | 14111 | 4 |

### 2.1 Best vs baseline 槽位 diff（哪些槽变了，变了多少）

_共 244 个槽与 baseline 不同_（238 SF + 6 K）

**Truncation K diffs**：

| Slot | Baseline K | Best K | Δ |
|:-----|----------:|------:|--:|
| `L0.B1.K` | 13 | 8 | -5 |
| `L0.B3.K` | 13 | 12 | -1 |
| `L1.B3.K` | 13 | 12 | -1 |
| `L10.B3.K` | 13 | 12 | -1 |
| `L2.B3.K` | 13 | 12 | -1 |
| `L5.B5.K` | 13 | 12 | -1 |

**Scaling-factor diffs**（前 20 条按 |Δ| 降序）：

| Slot | Kind | Baseline SF | Best SF | Δ |
|:-----|:----:|------------:|--------:|--:|
| `L0.B1.F.gelu_out` | F | 30 | 22 | -8 |
| `L0.B1.W.wffn2` | W | 20 | 12 | -8 |
| `L0.B2.F.inv_std_fresh` | F | 31 | 23 | -8 |
| `L0.B2.F.x_centered_fresh` | F | 30 | 22 | -8 |
| `L0.B2.W.wk` | W | 22 | 14 | -8 |
| `L0.B3.F.x_fresh` | F | 27 | 19 | -8 |
| `L0.B4.F.softmax_out_fresh` | F | 35 | 27 | -8 |
| `L0.B4.F.v_fresh` | F | 30 | 22 | -8 |
| `L1.B2.F.inv_std_fresh` | F | 31 | 23 | -8 |
| `L1.B2.F.x_centered_fresh` | F | 30 | 22 | -8 |
| `L1.B2.W.wk` | W | 22 | 14 | -8 |
| `L1.B3.F.x_fresh` | F | 27 | 19 | -8 |
| `L1.B4.F.softmax_out_fresh` | F | 35 | 27 | -8 |
| `L1.B4.F.v_fresh` | F | 30 | 22 | -8 |
| `L2.B2.F.inv_std_fresh` | F | 31 | 23 | -8 |
| `L2.B2.F.x_centered_fresh` | F | 30 | 22 | -8 |
| `L2.B2.W.wk` | W | 22 | 14 | -8 |
| `L2.B3.F.x_fresh` | F | 28 | 20 | -8 |
| `L2.B4.F.softmax_out_fresh` | F | 35 | 27 | -8 |
| `L2.B4.F.v_fresh` | F | 30 | 22 | -8 |

完整 diff 在 `best_action_vec.json` 的 `diff_vs_baseline` 字段。

## 3. First-invalid 频次（哪些 (layer, block) 最先翻车）

_暂无 invalid 记录（所有 episode 完整通过）。_

## 4. PPO 学习动态（最近 10 次更新）

| Update | Eps | policy_loss | value_loss | entropy | clip_frac | win_mean_ret | win_mean_inv |
|------:|----:|------------:|-----------:|--------:|----------:|-------------:|-------------:|
| 91 | 5460 | +4054554.1993 | +0.8340 | +0.9839 | 0.743 | -125.0205 | 0.00 |
| 92 | 5520 | +653919.1633 | +0.6331 | +0.7898 | 0.704 | -125.5584 | 0.00 |
| 93 | 5580 | +0.1622 | +0.5615 | +0.5389 | 0.689 | -125.5588 | 0.00 |
| 94 | 5640 | +96.0227 | +0.6122 | +0.7021 | 0.759 | -125.5380 | 0.00 |
| 95 | 5700 | +526678.4430 | +4.0790 | +0.6497 | 0.702 | -128.6743 | 0.00 |
| 96 | 5760 | +0.1951 | +0.7029 | +0.4736 | 0.707 | -125.5959 | 0.00 |
| 97 | 5820 | +172014387.4526 | +1.0449 | +0.5160 | 0.695 | -125.5056 | 0.00 |
| 98 | 5880 | +9367916.2280 | +1.6344 | +0.6923 | 0.808 | -125.7542 | 0.00 |
| 99 | 5940 | +36379876206.1071 | +1.1099 | +0.8140 | 0.777 | -125.2785 | 0.00 |
| 100 | 6000 | +0.2765 | +0.6556 | +0.8652 | 0.922 | -125.5354 | 0.00 |

_Entropy 趋势：+8.6811 → +0.8652（下降（policy 在收敛））_

## 5. 动作分布概览（哪些 slot 已经在按 baseline 取最大档）

- **已收敛 slot**（top-1 占比 > 85%）：**295** / 577
- **未收敛 slot**：**282** / 577

已收敛 slot 示例（前 8 个）：
  - slot[000] → action_index=0 （占比 100.0%）
  - slot[001] → action_index=0 （占比 100.0%）
  - slot[002] → action_index=0 （占比 100.0%）
  - slot[003] → action_index=0 （占比 100.0%）
  - slot[004] → action_index=0 （占比 100.0%）
  - slot[005] → action_index=0 （占比 100.0%）
  - slot[006] → action_index=0 （占比 100.0%）
  - slot[008] → action_index=0 （占比 95.2%）

最分散 slot 示例（前 8 个）：
  - slot[091] entropy=1.234 (uniform≈1.792)
  - slot[187] entropy=1.233 (uniform≈1.792)
  - slot[331] entropy=1.230 (uniform≈1.792)
  - slot[235] entropy=1.225 (uniform≈1.792)
  - slot[139] entropy=1.223 (uniform≈1.792)
  - slot[043] entropy=1.213 (uniform≈1.792)
  - slot[360] entropy=1.210 (uniform≈1.792)
  - slot[264] entropy=1.203 (uniform≈1.792)

## 6. 自动诊断（auto-flags）

- ⚠ **clip_fraction 偏高**：最近 3 次 PPO clip_frac=0.84（>0.40）。lr 可能过大，建议降低 lr 一档。

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