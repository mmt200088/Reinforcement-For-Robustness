# BLB Stage-2 Sequential RL · 诊断汇总（diagnostics @ episode=1000）

_更新时间: 2026-05-18 18:56:49_  ·  累计用时: **1h15m30s**

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
- `acc_threshold` = `0.8653125`
- `stab_threshold` = `0.01`
- `static_skeletons_archive` = `/var/tmp/root-home/Reinforcement-For-Robustness/Rescale_optimizer/configs/mrpc/static_skeletons_mrpc.json`

## 1. 训练进度（training progress）

- 已完成回合数: **1000**
- 最近 50 回合 mean return: **-7.6993** (min=-7.7362, max=-7.6216)
- 最近 50 回合 mean terminal reward: **-5.0000**
- 最近 50 回合 mean invalid 子步数: **0.00** / 59
- 训练期 best reward: **-7.6216**
- 训练期 worst reward: **-7.8460**
- PPO 更新次数: **16**
- baseline avg_k (per-block 加权): **11.780**

## 2. 训练期 Top-20 candidates

**说明**：按 total_reward 排序。每条候选的完整 SF / K 配置（按槽位标签）见 `top_candidates.jsonl` 的 `slots` 字段；下面摘要表只列指标。

| Rank | Episode | total_reward | terminal | per_step_sum | valid | invalid | total_bits | fusion |
|-----:|--------:|-------------:|---------:|-------------:|------:|--------:|-----------:|-------:|
| 1 | 986 | -7.6216 | -5.0000 | -2.6216 | 59 | 0 | 13909 | 6 |
| 2 | 996 | -7.6480 | -5.0000 | -2.6480 | 59 | 0 | 13945 | 11 |
| 3 | 660 | -7.6528 | -5.0000 | -2.6528 | 59 | 0 | 13999 | 8 |
| 4 | 869 | -7.6534 | -5.0000 | -2.6534 | 59 | 0 | 13981 | 8 |
| 5 | 879 | -7.6591 | -5.0000 | -2.6591 | 59 | 0 | 14019 | 8 |
| 6 | 891 | -7.6600 | -5.0000 | -2.6600 | 59 | 0 | 13989 | 8 |
| 7 | 989 | -7.6644 | -5.0000 | -2.6644 | 59 | 0 | 13999 | 6 |
| 8 | 968 | -7.6667 | -5.0000 | -2.6667 | 59 | 0 | 14005 | 8 |
| 9 | 872 | -7.6671 | -5.0000 | -2.6671 | 59 | 0 | 14043 | 7 |
| 10 | 804 | -7.6678 | -5.0000 | -2.6678 | 59 | 0 | 14019 | 10 |
| 11 | 894 | -7.6688 | -5.0000 | -2.6688 | 59 | 0 | 14015 | 7 |
| 12 | 950 | -7.6708 | -5.0000 | -2.6708 | 59 | 0 | 13989 | 11 |
| 13 | 841 | -7.6708 | -5.0000 | -2.6708 | 59 | 0 | 14031 | 8 |
| 14 | 831 | -7.6714 | -5.0000 | -2.6714 | 59 | 0 | 13991 | 9 |
| 15 | 922 | -7.6719 | -5.0000 | -2.6719 | 59 | 0 | 14037 | 8 |
| 16 | 905 | -7.6722 | -5.0000 | -2.6722 | 59 | 0 | 14029 | 11 |
| 17 | 909 | -7.6722 | -5.0000 | -2.6722 | 59 | 0 | 13991 | 10 |
| 18 | 744 | -7.6724 | -5.0000 | -2.6724 | 59 | 0 | 14033 | 6 |
| 19 | 933 | -7.6724 | -5.0000 | -2.6724 | 59 | 0 | 14017 | 8 |
| 20 | 924 | -7.6740 | -5.0000 | -2.6740 | 59 | 0 | 14041 | 8 |

### 2.1 Best vs baseline 槽位 diff（哪些槽变了，变了多少）

_共 392 个槽与 baseline 不同_（346 SF + 46 K）

**Truncation K diffs**：

| Slot | Baseline K | Best K | Δ |
|:-----|----------:|------:|--:|
| `L0.B1.K` | 13 | 8 | -5 |
| `L0.B4.K` | 10 | 8 | -2 |
| `L0.B5.K` | 13 | 10 | -3 |
| `L1.B1.K` | 13 | 11 | -2 |
| `L1.B2.K` | 10 | 11 | +1 |
| `L1.B3.K` | 13 | 10 | -3 |
| `L1.B4.K` | 10 | 9 | -1 |
| `L1.B5.K` | 13 | 10 | -3 |
| `L10.B1.K` | 13 | 10 | -3 |
| `L10.B2.K` | 10 | 8 | -2 |
| `L10.B3.K` | 13 | 10 | -3 |
| `L10.B5.K` | 13 | 10 | -3 |
| `L11.B1.K` | 13 | 10 | -3 |
| `L11.B2.K` | 10 | 9 | -1 |
| `L11.B3.K` | 13 | 10 | -3 |
| `L11.B4.K` | 10 | 11 | +1 |
| `L11.B5.K` | 13 | 10 | -3 |
| `L2.B1.K` | 13 | 12 | -1 |
| `L2.B3.K` | 13 | 8 | -5 |
| `L2.B5.K` | 13 | 10 | -3 |
| `L3.B1.K` | 13 | 10 | -3 |
| `L3.B3.K` | 13 | 10 | -3 |
| `L3.B5.K` | 13 | 8 | -5 |
| `L4.B1.K` | 13 | 10 | -3 |
| `L4.B2.K` | 10 | 9 | -1 |
| `L4.B3.K` | 13 | 10 | -3 |
| `L4.B4.K` | 10 | 11 | +1 |
| `L4.B5.K` | 13 | 12 | -1 |
| `L5.B1.K` | 13 | 10 | -3 |
| `L5.B2.K` | 10 | 11 | +1 |
| `L5.B3.K` | 13 | 10 | -3 |
| `L5.B5.K` | 13 | 10 | -3 |
| `L6.B1.K` | 13 | 10 | -3 |
| `L6.B3.K` | 13 | 9 | -4 |
| `L6.B5.K` | 13 | 10 | -3 |
| `L7.B1.K` | 13 | 10 | -3 |
| `L7.B3.K` | 13 | 10 | -3 |
| `L7.B5.K` | 13 | 10 | -3 |
| `L8.B1.K` | 13 | 10 | -3 |
| `L8.B2.K` | 10 | 9 | -1 |
| `L8.B3.K` | 13 | 10 | -3 |
| `L8.B5.K` | 13 | 10 | -3 |
| `L9.B1.K` | 13 | 10 | -3 |
| `L9.B3.K` | 13 | 10 | -3 |
| `L9.B4.K` | 10 | 11 | +1 |
| `L9.B5.K` | 13 | 8 | -5 |

**Scaling-factor diffs**（前 20 条按 |Δ| 降序）：

| Slot | Kind | Baseline SF | Best SF | Δ |
|:-----|:----:|------------:|--------:|--:|
| `L0.B1.F.gelu_out` | F | 30 | 22 | -8 |
| `L0.B1.W.wffn2` | W | 20 | 12 | -8 |
| `L0.B2.W.wk` | W | 22 | 14 | -8 |
| `L0.B2.W.wv` | W | 22 | 14 | -8 |
| `L0.B3.F.x_fresh` | F | 27 | 19 | -8 |
| `L0.B5.F.x_centered_fresh` | F | 30 | 22 | -8 |
| `L1.B2.W.wv` | W | 22 | 14 | -8 |
| `L1.B5.F.x_centered_fresh` | F | 30 | 22 | -8 |
| `L2.B4.F.v_fresh` | F | 30 | 22 | -8 |
| `L3.B2.W.wk` | W | 22 | 14 | -8 |
| `L3.B3.F.x_fresh` | F | 28 | 20 | -8 |
| `L4.B2.F.inv_std_fresh` | F | 31 | 23 | -8 |
| `L4.B2.W.wv` | W | 22 | 14 | -8 |
| `L4.B5.F.x_centered_fresh` | F | 30 | 22 | -8 |
| `L5.B2.W.wv` | W | 22 | 14 | -8 |
| `L6.B2.F.x_centered_fresh` | F | 30 | 22 | -8 |
| `L6.B2.W.wk` | W | 22 | 14 | -8 |
| `L6.B4.F.softmax_out_fresh` | F | 35 | 27 | -8 |
| `L6.B5.F.x_centered_fresh` | F | 30 | 22 | -8 |
| `L7.B2.F.inv_std_fresh` | F | 31 | 23 | -8 |

完整 diff 在 `best_action_vec.json` 的 `diff_vs_baseline` 字段。

## 3. First-invalid 频次（哪些 (layer, block) 最先翻车）

_暂无 invalid 记录（所有 episode 完整通过）。_

## 4. PPO 学习动态（最近 10 次更新）

| Update | Eps | policy_loss | value_loss | entropy | clip_frac | win_mean_ret | win_mean_inv |
|------:|----:|------------:|-----------:|--------:|----------:|-------------:|-------------:|
| 7 | 420 | -0.0136 | +0.0046 | +10.5471 | 0.299 | -7.7758 | 0.00 |
| 8 | 480 | -0.0154 | +0.0045 | +10.7698 | 0.286 | -7.7649 | 0.00 |
| 9 | 540 | -0.0176 | +0.0043 | +10.9973 | 0.311 | -7.7529 | 0.00 |
| 10 | 600 | -0.0157 | +0.0046 | +11.1393 | 0.314 | -7.7474 | 0.00 |
| 11 | 660 | -0.0170 | +0.0050 | +11.3354 | 0.301 | -7.7354 | 0.00 |
| 12 | 720 | -0.0193 | +0.0070 | +11.5457 | 0.296 | -7.7205 | 0.00 |
| 13 | 780 | -0.0190 | +0.0054 | +11.6591 | 0.260 | -7.7214 | 0.00 |
| 14 | 840 | -0.0194 | +0.0055 | +11.7691 | 0.284 | -7.7149 | 0.00 |
| 15 | 900 | -0.0226 | +0.0068 | +11.8594 | 0.268 | -7.7059 | 0.00 |
| 16 | 960 | -0.0169 | +0.0071 | +11.9112 | 0.256 | -7.7039 | 0.00 |

_Entropy 趋势：+9.5950 → +11.9112（上升（policy 在分散））_

## 5. 动作分布概览（哪些 slot 已经在按 baseline 取最大档）

- **已收敛 slot**（top-1 占比 > 85%）：**13** / 577
- **未收敛 slot**：**564** / 577

已收敛 slot 示例（前 8 个）：
  - slot[000] → action_index=0 （占比 100.0%）
  - slot[001] → action_index=0 （占比 100.0%）
  - slot[002] → action_index=0 （占比 100.0%）
  - slot[003] → action_index=0 （占比 100.0%）
  - slot[004] → action_index=0 （占比 100.0%）
  - slot[005] → action_index=0 （占比 100.0%）
  - slot[006] → action_index=0 （占比 100.0%）
  - slot[417] → action_index=4 （占比 85.1%）

最分散 slot 示例（前 8 个）：
  - slot[214] entropy=1.386 (uniform≈1.792)
  - slot[502] entropy=1.385 (uniform≈1.792)
  - slot[360] entropy=1.385 (uniform≈1.792)
  - slot[485] entropy=1.384 (uniform≈1.792)
  - slot[091] entropy=1.384 (uniform≈1.792)
  - slot[118] entropy=1.384 (uniform≈1.792)
  - slot[045] entropy=1.384 (uniform≈1.792)
  - slot[515] entropy=1.384 (uniform≈1.792)

## 6. 自动诊断（auto-flags）

- ✓ 暂无异常。

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
    --action-config Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.005_s2t0.005_s2st0.005_rdv2/stage2_noise/progress/diagnostics/best_action_vec.json
```

**手动调几个槽位**：直接复制 `best_action_vec.json`，改里面 `slots` 列表中对应槽位的 `scaling_factor` 或 `truncation_bits`，存成新文件后 `--action-config` 指过去即可。支持简写 `{"base":"max", "overrides":[{"label":"L05.B3.K", "truncation_bits":10}]}`。