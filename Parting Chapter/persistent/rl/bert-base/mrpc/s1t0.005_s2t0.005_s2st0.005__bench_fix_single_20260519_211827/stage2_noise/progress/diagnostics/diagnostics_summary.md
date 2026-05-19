# BLB Stage-2 Sequential RL · 诊断汇总（diagnostics @ episode=200）

_更新时间: 2026-05-19 21:28:19_  ·  累计用时: **9m24s**

**Run meta**：
- `profile` = `mrpc`
- `fixed_label` = `Stage-1 config (json)`
- `fixed_source` = `json`
- `rl_variant` = `blb_v3_sequential`
- `total_episodes_planned` = `200`
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
- `stab_threshold` = `0.01`
- `static_skeletons_archive` = `/hy-tmp/Reinforcement-For-Robustness/Rescale_optimizer/configs/mrpc/static_skeletons_mrpc.json`

## 1. 训练进度（training progress）

- 已完成回合数: **200**
- 最近 50 回合 mean return: **-7.8690** (min=-7.9037, max=-7.8210)
- 最近 50 回合 mean terminal reward: **-5.0000**
- 最近 50 回合 mean invalid 子步数: **0.00** / 59
- 训练期 best reward: **+37.9960**
- 训练期 worst reward: **-7.9037**
- PPO 更新次数: **3**
- baseline avg_k (per-block 加权): **11.780**

## 2. 训练期 Top-20 candidates

**说明**：按 total_reward 排序。每条候选的完整 SF / K 配置（按槽位标签）见 `top_candidates.jsonl` 的 `slots` 字段；下面摘要表只列指标。

| Rank | Episode | total_reward | terminal | per_step_sum | valid | invalid | total_bits | fusion |
|-----:|--------:|-------------:|---------:|-------------:|------:|--------:|-----------:|-------:|
| 1 | 106 | +37.9960 | +40.9687 | -2.9728 | 59 | 0 | 14779 | 0 |
| 2 | 87 | +37.9960 | +40.9687 | -2.9728 | 59 | 0 | 14779 | 0 |
| 3 | 1 | +37.8397 | +40.8125 | -2.9728 | 59 | 0 | 14779 | 0 |
| 4 | 97 | +37.8397 | +40.8125 | -2.9728 | 59 | 0 | 14779 | 0 |
| 5 | 16 | +37.8397 | +40.8125 | -2.9728 | 59 | 0 | 14779 | 0 |
| 6 | 33 | +37.8397 | +40.8125 | -2.9728 | 59 | 0 | 14779 | 0 |
| 7 | 44 | +37.7616 | +40.7344 | -2.9728 | 59 | 0 | 14779 | 0 |
| 8 | 70 | +37.7616 | +40.7344 | -2.9728 | 59 | 0 | 14779 | 0 |
| 9 | 23 | +37.7616 | +40.7344 | -2.9728 | 59 | 0 | 14779 | 0 |
| 10 | 101 | +37.7616 | +40.7344 | -2.9728 | 59 | 0 | 14779 | 0 |
| 11 | 9 | +37.7616 | +40.7344 | -2.9728 | 59 | 0 | 14779 | 0 |
| 12 | 10 | +37.6835 | +40.6562 | -2.9728 | 59 | 0 | 14779 | 0 |
| 13 | 12 | +37.6835 | +40.6562 | -2.9728 | 59 | 0 | 14779 | 0 |
| 14 | 84 | +37.6835 | +40.6562 | -2.9728 | 59 | 0 | 14779 | 0 |
| 15 | 118 | +37.6835 | +40.6562 | -2.9728 | 59 | 0 | 14779 | 0 |
| 16 | 90 | +37.6835 | +40.6562 | -2.9728 | 59 | 0 | 14779 | 0 |
| 17 | 28 | +37.6053 | +40.5781 | -2.9728 | 59 | 0 | 14779 | 0 |
| 18 | 95 | +37.6053 | +40.5781 | -2.9728 | 59 | 0 | 14779 | 0 |
| 19 | 46 | +37.6053 | +40.5781 | -2.9728 | 59 | 0 | 14779 | 0 |
| 20 | 58 | +37.6053 | +40.5781 | -2.9728 | 59 | 0 | 14779 | 0 |

### 2.1 Best vs baseline 槽位 diff（哪些槽变了，变了多少）

_共 5 个槽与 baseline 不同_（4 SF + 1 K）

**Truncation K diffs**：

| Slot | Baseline K | Best K | Δ |
|:-----|----------:|------:|--:|
| `L0.B1.K` | 13 | 8 | -5 |

**Scaling-factor diffs**（前 20 条按 |Δ| 降序）：

| Slot | Kind | Baseline SF | Best SF | Δ |
|:-----|:----:|------------:|--------:|--:|
| `L0.B1.F.gelu_out` | F | 30 | 22 | -8 |
| `L0.B1.W.wffn2` | W | 20 | 12 | -8 |
| `L0.B1.S.mean_inv_d` | S | 20 | 16 | -4 |
| `L0.B1.S.var_inv_d` | S | 20 | 16 | -4 |

完整 diff 在 `best_action_vec.json` 的 `diff_vs_baseline` 字段。

## 3. First-invalid 频次（哪些 (layer, block) 最先翻车）

_暂无 invalid 记录（所有 episode 完整通过）。_

## 4. PPO 学习动态（最近 10 次更新）

| Update | Eps | policy_loss | value_loss | entropy | clip_frac | ent_coef | win_mean_ret | win_mean_inv |
|------:|----:|------------:|-----------:|--------:|----------:|---------:|-------------:|-------------:|
| 1 | 60 | -0.1406 | +1.6890 | +4.6814 | 0.806 | 0.00000 | +36.0988 | 0.00 |
| 2 | 120 | -0.1197 | +0.8393 | +4.1582 | 0.793 | 0.00000 | +36.7447 | 0.00 |
| 3 | 180 | -0.0029 | +0.8582 | +4.5379 | 0.505 | 0.00500 | -7.8791 | 0.00 |

_Entropy 趋势：+4.6814 → +4.5379（下降（policy 在收敛））_

## 5. 动作分布概览（哪些 slot 已经在按 baseline 取最大档）

- **已收敛 slot**（top-1 占比 > 85%）：**393** / 577
- **未收敛 slot**：**184** / 577

## 6. 自动诊断（auto-flags）

- ⚠ **学习退化**：最近 20 回合平均回报 -7.8516 低于前 20 回合 +37.5038（Δ=-45.3554）。建议：降低 lr / 增加 ent_coef / 检查 invalid_penalty 是否过强。

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
    --action-config Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.005_s2t0.005_s2st0.005__bench_fix_single_20260519_211827/stage2_noise/progress/diagnostics/best_action_vec.json
```

**手动调几个槽位**：直接复制 `best_action_vec.json`，改里面 `slots` 列表中对应槽位的 `scaling_factor` 或 `truncation_bits`，存成新文件后 `--action-config` 指过去即可。支持简写 `{"base":"max", "overrides":[{"label":"L05.B3.K", "truncation_bits":10}]}`。