# BLB Stage-2 Sequential RL · 诊断汇总（diagnostics @ episode=200）

_更新时间: 2026-05-18 17:56:24_  ·  累计用时: **15m05s**

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

- 已完成回合数: **200**
- 最近 50 回合 mean return: **-7.8009** (min=-7.8382, max=-7.7578)
- 最近 50 回合 mean terminal reward: **-5.0000**
- 最近 50 回合 mean invalid 子步数: **0.00** / 59
- 训练期 best reward: **-7.7578**
- 训练期 worst reward: **-7.8460**
- PPO 更新次数: **3**
- baseline avg_k (per-block 加权): **11.780**

## 2. 训练期 Top-20 candidates

**说明**：按 total_reward 排序。每条候选的完整 SF / K 配置（按槽位标签）见 `top_candidates.jsonl` 的 `slots` 字段；下面摘要表只列指标。

| Rank | Episode | total_reward | terminal | per_step_sum | valid | invalid | total_bits | fusion |
|-----:|--------:|-------------:|---------:|-------------:|------:|--------:|-----------:|-------:|
| 1 | 185 | -7.7578 | -5.0000 | -2.7578 | 59 | 0 | 14211 | 10 |
| 2 | 60 | -7.7589 | -5.0000 | -2.7589 | 59 | 0 | 14225 | 11 |
| 3 | 8 | -7.7606 | -5.0000 | -2.7606 | 59 | 0 | 14245 | 7 |
| 4 | 86 | -7.7652 | -5.0000 | -2.7652 | 59 | 0 | 14247 | 8 |
| 5 | 40 | -7.7703 | -5.0000 | -2.7703 | 59 | 0 | 14267 | 8 |
| 6 | 73 | -7.7716 | -5.0000 | -2.7716 | 59 | 0 | 14269 | 12 |
| 7 | 50 | -7.7723 | -5.0000 | -2.7723 | 59 | 0 | 14261 | 6 |
| 8 | 64 | -7.7732 | -5.0000 | -2.7732 | 59 | 0 | 14255 | 10 |
| 9 | 113 | -7.7735 | -5.0000 | -2.7735 | 59 | 0 | 14273 | 8 |
| 10 | 116 | -7.7758 | -5.0000 | -2.7758 | 59 | 0 | 14293 | 7 |
| 11 | 45 | -7.7759 | -5.0000 | -2.7759 | 59 | 0 | 14283 | 8 |
| 12 | 143 | -7.7759 | -5.0000 | -2.7759 | 59 | 0 | 14273 | 7 |
| 13 | 178 | -7.7778 | -5.0000 | -2.7778 | 59 | 0 | 14283 | 10 |
| 14 | 122 | -7.7780 | -5.0000 | -2.7780 | 59 | 0 | 14275 | 9 |
| 15 | 195 | -7.7789 | -5.0000 | -2.7789 | 59 | 0 | 14285 | 10 |
| 16 | 14 | -7.7794 | -5.0000 | -2.7794 | 59 | 0 | 14257 | 11 |
| 17 | 110 | -7.7807 | -5.0000 | -2.7807 | 59 | 0 | 14273 | 5 |
| 18 | 26 | -7.7811 | -5.0000 | -2.7811 | 59 | 0 | 14301 | 8 |
| 19 | 169 | -7.7818 | -5.0000 | -2.7818 | 59 | 0 | 14275 | 8 |
| 20 | 177 | -7.7826 | -5.0000 | -2.7826 | 59 | 0 | 14273 | 10 |

### 2.1 Best vs baseline 槽位 diff（哪些槽变了，变了多少）

_共 343 个槽与 baseline 不同_（305 SF + 38 K）

**Truncation K diffs**：

| Slot | Baseline K | Best K | Δ |
|:-----|----------:|------:|--:|
| `L0.B1.K` | 13 | 8 | -5 |
| `L0.B3.K` | 13 | 10 | -3 |
| `L0.B5.K` | 13 | 10 | -3 |
| `L1.B1.K` | 13 | 10 | -3 |
| `L1.B3.K` | 13 | 10 | -3 |
| `L1.B5.K` | 13 | 10 | -3 |
| `L10.B1.K` | 13 | 10 | -3 |
| `L10.B3.K` | 13 | 8 | -5 |
| `L10.B5.K` | 13 | 10 | -3 |
| `L11.B1.K` | 13 | 10 | -3 |
| `L11.B3.K` | 13 | 10 | -3 |
| `L11.B5.K` | 13 | 10 | -3 |
| `L2.B1.K` | 13 | 10 | -3 |
| `L2.B3.K` | 13 | 10 | -3 |
| `L2.B5.K` | 13 | 10 | -3 |
| `L3.B1.K` | 13 | 12 | -1 |
| `L3.B4.K` | 10 | 12 | +2 |
| `L3.B5.K` | 13 | 10 | -3 |
| `L4.B1.K` | 13 | 10 | -3 |
| `L4.B3.K` | 13 | 10 | -3 |
| `L4.B5.K` | 13 | 10 | -3 |
| `L5.B1.K` | 13 | 10 | -3 |
| `L5.B3.K` | 13 | 10 | -3 |
| `L5.B4.K` | 10 | 13 | +3 |
| `L6.B1.K` | 13 | 10 | -3 |
| `L6.B3.K` | 13 | 10 | -3 |
| `L6.B5.K` | 13 | 10 | -3 |
| `L7.B1.K` | 13 | 10 | -3 |
| `L7.B2.K` | 10 | 11 | +1 |
| `L7.B3.K` | 13 | 10 | -3 |
| `L7.B5.K` | 13 | 10 | -3 |
| `L8.B1.K` | 13 | 9 | -4 |
| `L8.B3.K` | 13 | 10 | -3 |
| `L8.B5.K` | 13 | 10 | -3 |
| `L9.B1.K` | 13 | 10 | -3 |
| `L9.B2.K` | 10 | 9 | -1 |
| `L9.B3.K` | 13 | 10 | -3 |
| `L9.B5.K` | 13 | 10 | -3 |

**Scaling-factor diffs**（前 20 条按 |Δ| 降序）：

| Slot | Kind | Baseline SF | Best SF | Δ |
|:-----|:----:|------------:|--------:|--:|
| `L0.B1.F.gelu_out` | F | 30 | 22 | -8 |
| `L0.B1.W.wffn2` | W | 20 | 12 | -8 |
| `L0.B4.W.wo` | W | 22 | 14 | -8 |
| `L1.B4.F.softmax_out_fresh` | F | 35 | 27 | -8 |
| `L2.B2.W.wv` | W | 22 | 14 | -8 |
| `L6.B2.W.wv` | W | 22 | 14 | -8 |
| `L6.B3.F.x_fresh` | F | 28 | 20 | -8 |
| `L7.B2.W.wv` | W | 22 | 14 | -8 |
| `L7.B4.F.v_fresh` | F | 30 | 22 | -8 |
| `L7.B5.F.x_centered_fresh` | F | 30 | 22 | -8 |
| `L11.B2.F.inv_std_fresh` | F | 31 | 23 | -8 |
| `L2.B4.F.softmax_out_fresh` | F | 35 | 29 | -6 |
| `L4.B2.W.wk` | W | 22 | 16 | -6 |
| `L4.B5.F.x_centered_fresh` | F | 30 | 24 | -6 |
| `L5.B5.F.x_centered_fresh` | F | 31 | 25 | -6 |
| `L8.B2.W.wv` | W | 22 | 16 | -6 |
| `L10.B2.W.wk` | W | 22 | 16 | -6 |
| `L11.B2.F.x_centered_fresh` | F | 30 | 24 | -6 |
| `L0.B1.S.mean_inv_d` | S | 20 | 16 | -4 |
| `L0.B1.S.var_inv_d` | S | 20 | 16 | -4 |

完整 diff 在 `best_action_vec.json` 的 `diff_vs_baseline` 字段。

## 3. First-invalid 频次（哪些 (layer, block) 最先翻车）

_暂无 invalid 记录（所有 episode 完整通过）。_

## 4. PPO 学习动态（最近 10 次更新）

| Update | Eps | policy_loss | value_loss | entropy | clip_frac | win_mean_ret | win_mean_inv |
|------:|----:|------------:|-----------:|--------:|----------:|-------------:|-------------:|
| 1 | 60 | +0.0123 | +0.3628 | +9.5950 | 0.285 | -7.8046 | 0.00 |
| 2 | 120 | +0.0077 | +0.0884 | +9.6553 | 0.240 | -7.8022 | 0.00 |
| 3 | 180 | -0.0053 | +0.0299 | +9.8693 | 0.279 | -7.8038 | 0.00 |

_Entropy 趋势：+9.5950 → +9.8693（上升（policy 在分散））_

## 5. 动作分布概览（哪些 slot 已经在按 baseline 取最大档）

- **已收敛 slot**（top-1 占比 > 85%）：**168** / 577
- **未收敛 slot**：**409** / 577

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