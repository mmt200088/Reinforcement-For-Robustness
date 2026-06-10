# BLB Stage-2 Sequential RL · 诊断汇总（diagnostics @ episode=6000）

_更新时间: 2026-06-10 03:41:09_  ·  累计用时: **1h50m36s**

**Run meta**：
- `profile` = `mrpc`
- `fixed_label` = `Stage-1 config (stage1_record:bert base mrpc 1 20260610; softmax fixed deg6)`
- `fixed_source` = `stage1_record:bert base mrpc 1 20260610`
- `rl_variant` = `blb_v3_sequential_gtrxl_v2scale_fusioncount_v1`
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
- `static_invalid_level_mask_enabled` = `True`
- `acc_threshold` = `0.860625`
- `stab_threshold` = `0.01`
- `static_skeletons_archive` = `/hy-tmp/server_command_ee69ce8_5gpu_20260610/src/Rescale_optimizer/configs/mrpc/static_skeletons_mrpc.json`
- `fast_reward_mode_enabled` = `False`
- `online_num_trials_per_step` = `1`
- `terminal_eval_batch_size` = `4`
- `promotion_validation_trials` = `4`
- `promotion_margin_window` = `0.25`

## 1. 训练进度（training progress）

- 已完成回合数: **6000**
- 最近 50 回合 mean return: **+38.4129** (min=-2.3716, max=+40.3949)
- 最近 50 回合 mean terminal reward: **+40.6810**
- 最近 50 回合 mean invalid 子步数: **0.00** / 47
- 训练期 best reward: **+40.6158**
- 训练期 worst reward: **-3.4887**
- PPO 更新次数: **100**
- baseline avg_k (per-block 加权): **13.000**

## 2. 训练期 Top-20 candidates

**说明**：按 hard-priority + unbounded P3 cost rank 排序。P1/P2 不吃 cost；P3 内部先看无上限 cost rank，再看 fusion/K/bits 与 reward。每条候选的完整 SF / K 配置见 `top_candidates.jsonl` 的 `slots` 字段。

| Rank | Episode | P | cost_rank | total_reward | terminal | per_step_sum | valid | invalid | total_bits | fusion |
|-----:|--------:|--:|----------:|-------------:|---------:|-------------:|------:|--------:|-----------:|-------:|
| 1 | 5686 | 3 | +3380.0000 | +40.4811 | +42.7333 | -2.2522 | 47 | 0 | 10800 | 17 |
| 2 | 5311 | 3 | +3380.0000 | +40.4065 | +42.6587 | -2.2522 | 47 | 0 | 10800 | 17 |
| 3 | 5564 | 3 | +3380.0000 | +40.3692 | +42.6214 | -2.2522 | 47 | 0 | 10800 | 17 |
| 4 | 5315 | 3 | +3360.0000 | +40.6158 | +42.8680 | -2.2522 | 47 | 0 | 10800 | 17 |
| 5 | 5645 | 3 | +3360.0000 | +40.3173 | +42.5695 | -2.2522 | 47 | 0 | 10800 | 17 |
| 6 | 5588 | 3 | +3350.0000 | +40.4966 | +42.7488 | -2.2522 | 47 | 0 | 10800 | 17 |
| 7 | 5258 | 3 | +3340.0000 | +40.4147 | +42.6669 | -2.2522 | 47 | 0 | 10800 | 17 |
| 8 | 5496 | 3 | +3320.0000 | +40.4002 | +42.6524 | -2.2522 | 47 | 0 | 10800 | 17 |
| 9 | 5870 | 3 | +3320.0000 | +40.4937 | +42.7643 | -2.2706 | 47 | 0 | 10853 | 16 |
| 10 | 5838 | 3 | +3310.0000 | +40.3556 | +42.6078 | -2.2522 | 47 | 0 | 10800 | 17 |
| 11 | 5982 | 3 | +3310.0000 | +40.3092 | +42.5705 | -2.2613 | 47 | 0 | 10826 | 16 |
| 12 | 5676 | 3 | +3300.0000 | +40.3483 | +42.6005 | -2.2522 | 47 | 0 | 10800 | 17 |
| 13 | 5683 | 3 | +3290.0000 | +40.3784 | +42.6306 | -2.2522 | 47 | 0 | 10800 | 17 |
| 14 | 5899 | 3 | +3290.0000 | +40.4719 | +42.7425 | -2.2706 | 47 | 0 | 10853 | 16 |
| 15 | 5635 | 3 | +3280.0000 | +40.3711 | +42.6233 | -2.2522 | 47 | 0 | 10800 | 17 |
| 16 | 5293 | 3 | +3280.0000 | +40.3338 | +42.5860 | -2.2522 | 47 | 0 | 10800 | 17 |
| 17 | 5841 | 3 | +3280.0000 | +40.1845 | +42.4367 | -2.2522 | 47 | 0 | 10800 | 17 |
| 18 | 5314 | 3 | +3270.0000 | +40.2892 | +42.5414 | -2.2522 | 47 | 0 | 10800 | 17 |
| 19 | 5601 | 3 | +3270.0000 | +40.2892 | +42.5414 | -2.2522 | 47 | 0 | 10800 | 17 |
| 20 | 5154 | 3 | +3270.0000 | +40.3582 | +42.6160 | -2.2578 | 47 | 0 | 10814 | 16 |

### 2.1 Best vs baseline 槽位 diff（哪些槽变了，变了多少）

_共 163 个槽与 baseline 不同_（117 SF + 46 K）

**Truncation K diffs**：

| Slot | Baseline K | Best K | Δ |
|:-----|----------:|------:|--:|
| `L0.B2.K` | 13 | 8 | -5 |
| `L0.B4.K` | 13 | 10 | -3 |
| `L1.B1.K` | 13 | 12 | -1 |
| `L1.B2.K` | 13 | 10 | -3 |
| `L1.B4.K` | 13 | 9 | -4 |
| `L1.B5.K` | 13 | 10 | -3 |
| `L10.B1.K` | 13 | 8 | -5 |
| `L10.B2.K` | 13 | 10 | -3 |
| `L10.B4.K` | 13 | 10 | -3 |
| `L10.B5.K` | 13 | 10 | -3 |
| `L11.B1.K` | 13 | 8 | -5 |
| `L11.B2.K` | 13 | 10 | -3 |
| `L11.B4.K` | 13 | 9 | -4 |
| `L11.B5.K` | 13 | 9 | -4 |
| `L2.B1.K` | 13 | 9 | -4 |
| `L2.B2.K` | 13 | 9 | -4 |
| `L2.B4.K` | 13 | 9 | -4 |
| `L2.B5.K` | 13 | 10 | -3 |
| `L3.B1.K` | 13 | 11 | -2 |
| `L3.B2.K` | 13 | 9 | -4 |
| `L3.B4.K` | 13 | 10 | -3 |
| `L3.B5.K` | 13 | 10 | -3 |
| `L4.B1.K` | 13 | 9 | -4 |
| `L4.B2.K` | 13 | 12 | -1 |
| `L4.B4.K` | 13 | 11 | -2 |
| `L4.B5.K` | 13 | 8 | -5 |
| `L5.B1.K` | 13 | 8 | -5 |
| `L5.B2.K` | 13 | 10 | -3 |
| `L5.B4.K` | 13 | 10 | -3 |
| `L5.B5.K` | 13 | 9 | -4 |
| `L6.B1.K` | 13 | 9 | -4 |
| `L6.B2.K` | 13 | 11 | -2 |
| `L6.B4.K` | 13 | 8 | -5 |
| `L6.B5.K` | 13 | 9 | -4 |
| `L7.B1.K` | 13 | 10 | -3 |
| `L7.B2.K` | 13 | 9 | -4 |
| `L7.B4.K` | 13 | 9 | -4 |
| `L7.B5.K` | 13 | 9 | -4 |
| `L8.B1.K` | 13 | 8 | -5 |
| `L8.B2.K` | 13 | 11 | -2 |
| `L8.B4.K` | 13 | 9 | -4 |
| `L8.B5.K` | 13 | 8 | -5 |
| `L9.B1.K` | 13 | 9 | -4 |
| `L9.B2.K` | 13 | 11 | -2 |
| `L9.B4.K` | 13 | 8 | -5 |
| `L9.B5.K` | 13 | 11 | -2 |

**Scaling-factor diffs**（前 20 条按 |Δ| 降序）：

| Slot | Kind | Baseline SF | Best SF | Δ |
|:-----|:----:|------------:|--------:|--:|
| `L10.B4.F.softmax_out_fresh` | F | 35 | 21 | -14 |
| `L10.B4.W.wo` | W | 22 | 11 | -11 |
| `L0.B5.F.x_centered_fresh` | F | 31 | 21 | -10 |
| `L3.B5.F.x_centered_fresh` | F | 31 | 21 | -10 |
| `L5.B5.F.x_centered_fresh` | F | 31 | 21 | -10 |
| `L6.B5.F.x_centered_fresh` | F | 31 | 21 | -10 |
| `L9.B5.F.x_centered_fresh` | F | 31 | 21 | -10 |
| `L0.B2.F.inv_std_fresh` | F | 28 | 20 | -8 |
| `L1.B2.F.inv_std_fresh` | F | 28 | 20 | -8 |
| `L2.B2.F.inv_std_fresh` | F | 28 | 20 | -8 |
| `L4.B2.F.inv_std_fresh` | F | 28 | 20 | -8 |
| `L6.B2.F.inv_std_fresh` | F | 28 | 20 | -8 |
| `L8.B2.F.inv_std_fresh` | F | 28 | 20 | -8 |
| `L9.B2.F.inv_std_fresh` | F | 28 | 20 | -8 |
| `L10.B2.F.inv_std_fresh` | F | 28 | 20 | -8 |
| `L10.B4.F.v_fresh` | F | 25 | 17 | -8 |
| `L10.B4.S.ln_mean_inv_d` | S | 20 | 12 | -8 |
| `L11.B2.F.inv_std_fresh` | F | 28 | 20 | -8 |
| `L0.B2.W.wk` | W | 22 | 16 | -6 |
| `L0.B5.W.wffn1` | W | 22 | 16 | -6 |

完整 diff 在 `best_action_vec.json` 的 `diff_vs_baseline` 字段。

## 3. First-invalid 频次（哪些 (layer, block) 最先翻车）

_暂无 invalid 记录（所有 episode 完整通过）。_

## 4. PPO 学习动态（最近 10 次更新）

| Update | Eps | policy_loss | value_loss | entropy | clip_frac | ent_coef | approx_kl | lr_scale | entropy_recovery | win_mean_ret | win_mean_inv |
|------:|----:|------------:|-----------:|--------:|----------:|---------:|----------:|---------:|-----------------:|-------------:|-------------:|
| 91 | 5460 | -0.0090 | +0.0544 | +1.7120 | 0.214 | 0.02000 | 0.03464 | 1.161 | 0.05194 | +38.4085 | 0.00 |
| 92 | 5520 | -0.0196 | +0.0932 | +1.6971 | 0.238 | 0.02000 | 0.02217 | 0.580 | 0.05323 | +36.9667 | 0.00 |
| 93 | 5580 | -0.0125 | +0.0534 | +1.6817 | 0.169 | 0.02000 | 0.00670 | 0.580 | 0.05342 | +38.3985 | 0.00 |
| 94 | 5640 | -0.0207 | +0.0434 | +1.6627 | 0.189 | 0.02000 | 0.02570 | 0.697 | 0.05477 | +38.7784 | 0.00 |
| 95 | 5700 | -0.0186 | +0.0672 | +1.6561 | 0.184 | 0.02000 | 0.00990 | 0.697 | 0.05511 | +38.0269 | 0.00 |
| 96 | 5760 | -0.0117 | +0.0004 | +1.6538 | 0.148 | 0.02000 | 0.00978 | 0.836 | 0.05513 | +40.1288 | 0.00 |
| 97 | 5820 | -0.0167 | +0.0003 | +1.6577 | 0.151 | 0.02000 | 0.01661 | 1.003 | 0.05504 | +40.1953 | 0.00 |
| 98 | 5880 | -0.0153 | +0.0654 | +1.6769 | 0.239 | 0.02000 | 0.01984 | 1.003 | 0.05509 | +38.0571 | 0.00 |
| 99 | 5940 | -0.0266 | +0.0520 | +1.6723 | 0.209 | 0.02000 | 0.01133 | 1.003 | 0.05477 | +38.3732 | 0.00 |
| 100 | 6000 | -0.0083 | +0.0527 | +1.6691 | 0.170 | 0.02000 | 0.01013 | 1.003 | 0.05486 | +38.6961 | 0.00 |

_Entropy 趋势：+2.0683 → +1.6691（下降（policy 在收敛））_

## 5. 动作分布概览（哪些 slot 已经在按 baseline 取最大档）

- **已收敛 slot**（top-1 占比 > 85%）：**368** / 877
- **未收敛 slot**：**47** / 877

已收敛 slot 示例（前 8 个）：
  - slot[008] → action_index=3 （占比 100.0%）
  - slot[009] → action_index=5 （占比 100.0%）
  - slot[018] → action_index=0 （占比 100.0%）
  - slot[019] → action_index=0 （占比 100.0%）
  - slot[023] → action_index=0 （占比 100.0%）
  - slot[024] → action_index=0 （占比 100.0%）
  - slot[025] → action_index=0 （占比 100.0%）
  - slot[026] → action_index=0 （占比 100.0%）

最分散 slot 示例（前 8 个）：
  - slot[323] entropy=1.703 (uniform≈1.792)
  - slot[072] entropy=1.691 (uniform≈1.792)
  - slot[081] entropy=1.689 (uniform≈1.792)
  - slot[291] entropy=1.688 (uniform≈1.792)
  - slot[437] entropy=1.682 (uniform≈1.792)
  - slot[177] entropy=1.678 (uniform≈1.792)
  - slot[615] entropy=1.677 (uniform≈1.792)
  - slot[104] entropy=1.666 (uniform≈1.792)

## 6. 自动诊断（auto-flags）

- ✓ 暂无异常。

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