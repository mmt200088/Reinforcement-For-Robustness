# BLB Stage 2 RL 训练报告（最终版）

- 运行名（run_basename）: `bert base mrpc`
- Profile（数据集）: `mrpc`
- 生成时间: 2026-06-11T17:19:26.994271
- 训练时长: 49242.2 秒（约 820.7 分钟）
- Episode 进度: 60000 / 60000
- 模数链 invoker: `in_process_real`

## 1. Reward 概览

- 最优 reward (best): **+39.356497**
- 全程 episode reward 均值: +38.8473
- 全程 episode reward 最大值: +39.4116
- 全程 episode reward 最小值: -3.2660
- 全程 episode reward 标准差: 1.0652

## 2. 最优 reward 拆解

| 字段 | 值 |
|------|------|
| `terminal_priority` | 3 |
| `terminal_reward` | 41.811606052247924 |
| `terminal_cost_score` | 1.4030694668820678 |
| `terminal_cost_rank_score` | 1930.0 |
| `terminal_cost_rank_fusion` | 0.0 |
| `terminal_cost_rank_truncation` | 0.0 |
| `terminal_cost_rank_bits` | 0.0 |
| `terminal_p3_metric_margin_reward` | 0.4085365853658546 |
| `terminal_cost_fusion_bonus` | 0.0 |
| `terminal_cost_truncation_bonus` | 0.0 |
| `terminal_cost_bits_tiebreaker` | 0.0 |
| `terminal_cost_truncation_step_gain` | 0.0 |
| `terminal_fusion_gain` | 0.0 |
| `terminal_k_gain` | 3.2711864406779654 |
| `terminal_bits_gain` | 0.0 |
| `terminal_metric1_mean` | 0.86875 |
| `terminal_metric2_mean` | 0.86875 |
| `terminal_stab_violation` | 0.0 |

## 3. Baseline（全 max action）对照

| 字段 | 值 |
|------|------|
| `total_bits_sum` | 11285 |
| `total_fusion_count` | 0 |
| `avg_k` | 13.0 |
| `loss_mean` | 0.367220401763916 |
| `metric1_mean` | 0.87109375 |
| `metric2_mean` | 0.87109375 |

## 4. Reward 权重

| 字段 | 值 |
|------|------|
| `design` | budgeted_adaptive_scalar_p3_cost |
| `cost_weight` | 1.0 |
| `lambda_stab` | 1.0 |
| `invalid_penalty` | 5.0 |
| `reward_clip_min` | -5.0 |
| `reward_clip_max` | 5.0 |
| `tier_metric_bonus` | 20.0 |
| `tier_stability_bonus` | 20.0 |
| `baseline_metric1` | 0.87109375 |
| `baseline_metric2` | 0.87109375 |
| `cost_reward_mode` | adaptive_scalar |
| `p3_metric_margin_budget` | 0.5 |
| `p3_cost_budget` | 4.5 |
| `cost_fusion_step_bonus` | 0.35 |
| `cost_k_step_bonus` | 0.35 |
| `cost_k_step_size` | 0.08333333333333333 |
| `cost_bits_linear_scale` | 0.1 |
| `cost_bits_tiebreaker_clip` | 0.25 |
| `cost_score_clip_min` | -0.5 |
| `cost_score_clip_max` | 4.5 |

## 5. 最优 action：选了什么 SF / K（人类视图）

完整的逐槽位明细在 `blb_stage2_best_action_full.md` （人类阅读）和 `blb_stage2_best_action_full.json` （可直接喂给 `Paean/run_final_eval.sh --action-config`）。下面只列出与 baseline 不同的槽位。

### 5.1 Best action · 按层 / block 选择概览

| 层 | block | 槽位选择 |
|---|---|---|
| L00 | B1 | F.gelu_out=off, W.wffn2=off, S.mean_inv_d=off, S.var_inv_d=off, R.mean_r=off, R.var_r=off, R.wffn2_r=off, R.square_r=off, K=**None** |
| L00 | B2 | F.inv_std_fresh=28, F.x_centered_fresh=28, M.gamma=20, W.wk=22, W.wv=22, M.kt_mask1=15, M.kt_mask2=15, M.qkt_merge_mask=15, R.gamma_r=28, R.kt_mask2_r=off, R.qkt_merge_mask_r=off, W.wq=22, M.q_mask1=15, M.q_mask2=15, R.normalize_r=off, R.wk_r=off, R.wq_r=off, R.wv_r=off, R.kt_mask1_r=28, R.q_mask1_r=28, R.q_mask2_r=off, R.qkt_matmul_r=28, K=**8** |
| L00 | B3 | F.x_fresh=28, S.inv_2n=15, R.sq0=31, R.sq1=31, R.sq2=31, R.sq3=31, R.x_inv_2n_r=22, K=**13** |
| L00 | B4 | F.softmax_out_fresh=35, F.v_fresh=25, M.softmax_out_mask=14, M.v_mask=14, M.softmax_v_mask=14, S.ln_mean_inv_d=20, S.ln_var_inv_d=20, W.wo=22, R.softmax_v_matmul_r=31, R.ln_mean_r=31, R.ln_var_r=off, R.softmax_out_mask_r=off, R.v_mask_r=off, R.softmax_v_mask_r=off, R.wo_r=off, R.ln_square_r=31, K=**13** |
| L00 | B5 | F.inv_std_fresh=31, F.x_centered_fresh=31, M.gamma=20, W.wffn1=22, M.gelu_coeff=20, R.normalize_r=28, R.gamma_r=off, R.wffn1_r=off, R.gp0=off, R.gp1=off, R.gp2=off, R.gc0=31, R.gc1=off, R.gc2=off, R.gc3=off, K=**12** |
| L01 | B1 | F.gelu_out=30, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=20, R.mean_r=30, R.var_r=27, R.wffn2_r=off, R.square_r=off, K=**12** |
| L01 | B2 | F.inv_std_fresh=28, F.x_centered_fresh=28, M.gamma=20, W.wk=22, W.wv=22, M.kt_mask1=15, M.kt_mask2=15, M.qkt_merge_mask=15, R.gamma_r=28, R.kt_mask2_r=off, R.qkt_merge_mask_r=off, W.wq=22, M.q_mask1=15, M.q_mask2=15, R.normalize_r=off, R.wk_r=off, R.wq_r=off, R.wv_r=off, R.kt_mask1_r=28, R.q_mask1_r=28, R.q_mask2_r=off, R.qkt_matmul_r=28, K=**12** |
| L01 | B3 | F.x_fresh=28, S.inv_2n=15, R.sq0=31, R.sq1=31, R.sq2=31, R.sq3=31, R.x_inv_2n_r=22, K=**13** |
| L01 | B4 | F.softmax_out_fresh=35, F.v_fresh=25, M.softmax_out_mask=14, M.v_mask=14, M.softmax_v_mask=14, S.ln_mean_inv_d=20, S.ln_var_inv_d=20, W.wo=22, R.softmax_v_matmul_r=31, R.ln_mean_r=31, R.ln_var_r=off, R.softmax_out_mask_r=off, R.v_mask_r=off, R.softmax_v_mask_r=off, R.wo_r=off, R.ln_square_r=31, K=**8** |
| L01 | B5 | F.inv_std_fresh=31, F.x_centered_fresh=31, M.gamma=20, W.wffn1=22, M.gelu_coeff=20, R.normalize_r=31, R.gamma_r=off, R.wffn1_r=31, R.gp0=off, R.gp1=off, R.gp2=off, R.gc0=31, R.gc1=off, R.gc2=off, R.gc3=off, K=**8** |
| L02 | B1 | F.gelu_out=30, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=20, R.mean_r=30, R.var_r=27, R.wffn2_r=off, R.square_r=off, K=**12** |
| L02 | B2 | F.inv_std_fresh=28, F.x_centered_fresh=28, M.gamma=20, W.wk=22, W.wv=22, M.kt_mask1=15, M.kt_mask2=15, M.qkt_merge_mask=15, R.gamma_r=28, R.kt_mask2_r=off, R.qkt_merge_mask_r=off, W.wq=22, M.q_mask1=15, M.q_mask2=15, R.normalize_r=off, R.wk_r=off, R.wq_r=off, R.wv_r=off, R.kt_mask1_r=28, R.q_mask1_r=28, R.q_mask2_r=off, R.qkt_matmul_r=28, K=**12** |
| L02 | B3 | F.x_fresh=28, S.inv_2n=15, R.sq0=31, R.sq1=31, R.sq2=31, R.sq3=31, R.x_inv_2n_r=22, K=**13** |
| L02 | B4 | F.softmax_out_fresh=35, F.v_fresh=25, M.softmax_out_mask=14, M.v_mask=14, M.softmax_v_mask=14, S.ln_mean_inv_d=20, S.ln_var_inv_d=20, W.wo=22, R.softmax_v_matmul_r=31, R.ln_mean_r=31, R.ln_var_r=off, R.softmax_out_mask_r=off, R.v_mask_r=off, R.softmax_v_mask_r=off, R.wo_r=off, R.ln_square_r=31, K=**9** |
| L02 | B5 | F.inv_std_fresh=31, F.x_centered_fresh=31, M.gamma=20, W.wffn1=22, M.gelu_coeff=20, R.normalize_r=28, R.gamma_r=off, R.wffn1_r=off, R.gp0=off, R.gp1=off, R.gp2=off, R.gc0=31, R.gc1=off, R.gc2=off, R.gc3=off, K=**9** |
| L03 | B1 | F.gelu_out=30, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=20, R.mean_r=30, R.var_r=27, R.wffn2_r=off, R.square_r=off, K=**9** |
| L03 | B2 | F.inv_std_fresh=28, F.x_centered_fresh=28, M.gamma=20, W.wk=22, W.wv=22, M.kt_mask1=15, M.kt_mask2=15, M.qkt_merge_mask=15, R.gamma_r=28, R.kt_mask2_r=off, R.qkt_merge_mask_r=off, W.wq=22, M.q_mask1=15, M.q_mask2=15, R.normalize_r=off, R.wk_r=off, R.wq_r=off, R.wv_r=off, R.kt_mask1_r=28, R.q_mask1_r=28, R.q_mask2_r=off, R.qkt_matmul_r=28, K=**9** |
| L03 | B3 | F.x_fresh=28, S.inv_2n=15, R.sq0=31, R.sq1=31, R.sq2=31, R.sq3=31, R.x_inv_2n_r=22, K=**13** |
| L03 | B4 | F.softmax_out_fresh=35, F.v_fresh=25, M.softmax_out_mask=14, M.v_mask=14, M.softmax_v_mask=14, S.ln_mean_inv_d=20, S.ln_var_inv_d=20, W.wo=22, R.softmax_v_matmul_r=31, R.ln_mean_r=31, R.ln_var_r=off, R.softmax_out_mask_r=off, R.v_mask_r=off, R.softmax_v_mask_r=off, R.wo_r=off, R.ln_square_r=31, K=**9** |
| L03 | B5 | F.inv_std_fresh=31, F.x_centered_fresh=31, M.gamma=20, W.wffn1=22, M.gelu_coeff=20, R.normalize_r=28, R.gamma_r=off, R.wffn1_r=off, R.gp0=off, R.gp1=off, R.gp2=off, R.gc0=31, R.gc1=off, R.gc2=off, R.gc3=off, K=**9** |
| L04 | B1 | F.gelu_out=30, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=20, R.mean_r=30, R.var_r=27, R.wffn2_r=off, R.square_r=off, K=**9** |
| L04 | B2 | F.inv_std_fresh=28, F.x_centered_fresh=28, M.gamma=20, W.wk=22, W.wv=22, M.kt_mask1=15, M.kt_mask2=15, M.qkt_merge_mask=15, R.gamma_r=28, R.kt_mask2_r=off, R.qkt_merge_mask_r=off, W.wq=22, M.q_mask1=15, M.q_mask2=15, R.normalize_r=off, R.wk_r=off, R.wq_r=off, R.wv_r=off, R.kt_mask1_r=28, R.q_mask1_r=28, R.q_mask2_r=off, R.qkt_matmul_r=28, K=**9** |
| L04 | B3 | F.x_fresh=28, S.inv_2n=15, R.sq0=31, R.sq1=31, R.sq2=31, R.sq3=31, R.x_inv_2n_r=22, K=**13** |
| L04 | B4 | F.softmax_out_fresh=35, F.v_fresh=25, M.softmax_out_mask=14, M.v_mask=14, M.softmax_v_mask=14, S.ln_mean_inv_d=20, S.ln_var_inv_d=20, W.wo=22, R.softmax_v_matmul_r=31, R.ln_mean_r=31, R.ln_var_r=off, R.softmax_out_mask_r=off, R.v_mask_r=off, R.softmax_v_mask_r=off, R.wo_r=off, R.ln_square_r=31, K=**9** |
| L04 | B5 | F.inv_std_fresh=31, F.x_centered_fresh=31, M.gamma=20, W.wffn1=22, M.gelu_coeff=20, R.normalize_r=28, R.gamma_r=off, R.wffn1_r=off, R.gp0=off, R.gp1=off, R.gp2=off, R.gc0=31, R.gc1=off, R.gc2=off, R.gc3=off, K=**9** |
| L05 | B1 | F.gelu_out=30, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=20, R.mean_r=30, R.var_r=27, R.wffn2_r=off, R.square_r=off, K=**9** |
| L05 | B2 | F.inv_std_fresh=28, F.x_centered_fresh=28, M.gamma=20, W.wk=22, W.wv=22, M.kt_mask1=15, M.kt_mask2=15, M.qkt_merge_mask=15, R.gamma_r=28, R.kt_mask2_r=off, R.qkt_merge_mask_r=off, W.wq=22, M.q_mask1=15, M.q_mask2=15, R.normalize_r=off, R.wk_r=off, R.wq_r=off, R.wv_r=off, R.kt_mask1_r=28, R.q_mask1_r=28, R.q_mask2_r=off, R.qkt_matmul_r=28, K=**9** |
| L05 | B3 | F.x_fresh=28, S.inv_2n=15, R.sq0=31, R.sq1=31, R.sq2=31, R.sq3=31, R.x_inv_2n_r=22, K=**13** |
| L05 | B4 | F.softmax_out_fresh=35, F.v_fresh=25, M.softmax_out_mask=14, M.v_mask=14, M.softmax_v_mask=14, S.ln_mean_inv_d=20, S.ln_var_inv_d=20, W.wo=22, R.softmax_v_matmul_r=31, R.ln_mean_r=31, R.ln_var_r=off, R.softmax_out_mask_r=off, R.v_mask_r=off, R.softmax_v_mask_r=off, R.wo_r=off, R.ln_square_r=31, K=**9** |
| L05 | B5 | F.inv_std_fresh=31, F.x_centered_fresh=31, M.gamma=20, W.wffn1=22, M.gelu_coeff=20, R.normalize_r=28, R.gamma_r=off, R.wffn1_r=off, R.gp0=off, R.gp1=off, R.gp2=off, R.gc0=31, R.gc1=off, R.gc2=off, R.gc3=off, K=**9** |
| L06 | B1 | F.gelu_out=30, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=20, R.mean_r=30, R.var_r=27, R.wffn2_r=off, R.square_r=off, K=**9** |
| L06 | B2 | F.inv_std_fresh=28, F.x_centered_fresh=28, M.gamma=20, W.wk=22, W.wv=22, M.kt_mask1=15, M.kt_mask2=15, M.qkt_merge_mask=15, R.gamma_r=28, R.kt_mask2_r=off, R.qkt_merge_mask_r=off, W.wq=22, M.q_mask1=15, M.q_mask2=15, R.normalize_r=off, R.wk_r=off, R.wq_r=off, R.wv_r=off, R.kt_mask1_r=28, R.q_mask1_r=28, R.q_mask2_r=off, R.qkt_matmul_r=28, K=**9** |
| L06 | B3 | F.x_fresh=28, S.inv_2n=15, R.sq0=31, R.sq1=31, R.sq2=31, R.sq3=31, R.x_inv_2n_r=22, K=**13** |
| L06 | B4 | F.softmax_out_fresh=35, F.v_fresh=25, M.softmax_out_mask=14, M.v_mask=14, M.softmax_v_mask=14, S.ln_mean_inv_d=20, S.ln_var_inv_d=20, W.wo=22, R.softmax_v_matmul_r=31, R.ln_mean_r=31, R.ln_var_r=off, R.softmax_out_mask_r=off, R.v_mask_r=off, R.softmax_v_mask_r=off, R.wo_r=off, R.ln_square_r=31, K=**9** |
| L06 | B5 | F.inv_std_fresh=31, F.x_centered_fresh=31, M.gamma=20, W.wffn1=22, M.gelu_coeff=20, R.normalize_r=28, R.gamma_r=off, R.wffn1_r=off, R.gp0=off, R.gp1=off, R.gp2=off, R.gc0=31, R.gc1=off, R.gc2=off, R.gc3=off, K=**9** |
| L07 | B1 | F.gelu_out=30, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=20, R.mean_r=30, R.var_r=27, R.wffn2_r=off, R.square_r=off, K=**9** |
| L07 | B2 | F.inv_std_fresh=28, F.x_centered_fresh=28, M.gamma=20, W.wk=22, W.wv=22, M.kt_mask1=15, M.kt_mask2=15, M.qkt_merge_mask=15, R.gamma_r=28, R.kt_mask2_r=off, R.qkt_merge_mask_r=off, W.wq=22, M.q_mask1=15, M.q_mask2=15, R.normalize_r=off, R.wk_r=off, R.wq_r=off, R.wv_r=off, R.kt_mask1_r=28, R.q_mask1_r=28, R.q_mask2_r=off, R.qkt_matmul_r=28, K=**9** |
| L07 | B3 | F.x_fresh=28, S.inv_2n=15, R.sq0=31, R.sq1=31, R.sq2=31, R.sq3=31, R.x_inv_2n_r=22, K=**13** |
| L07 | B4 | F.softmax_out_fresh=35, F.v_fresh=25, M.softmax_out_mask=14, M.v_mask=14, M.softmax_v_mask=14, S.ln_mean_inv_d=20, S.ln_var_inv_d=20, W.wo=22, R.softmax_v_matmul_r=31, R.ln_mean_r=31, R.ln_var_r=off, R.softmax_out_mask_r=off, R.v_mask_r=off, R.softmax_v_mask_r=off, R.wo_r=off, R.ln_square_r=31, K=**13** |
| L07 | B5 | F.inv_std_fresh=31, F.x_centered_fresh=31, M.gamma=20, W.wffn1=22, M.gelu_coeff=20, R.normalize_r=28, R.gamma_r=off, R.wffn1_r=off, R.gp0=off, R.gp1=off, R.gp2=off, R.gc0=31, R.gc1=off, R.gc2=off, R.gc3=off, K=**9** |
| L08 | B1 | F.gelu_out=30, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=20, R.mean_r=30, R.var_r=27, R.wffn2_r=off, R.square_r=off, K=**9** |
| L08 | B2 | F.inv_std_fresh=28, F.x_centered_fresh=28, M.gamma=20, W.wk=22, W.wv=22, M.kt_mask1=15, M.kt_mask2=15, M.qkt_merge_mask=15, R.gamma_r=28, R.kt_mask2_r=off, R.qkt_merge_mask_r=off, W.wq=22, M.q_mask1=15, M.q_mask2=15, R.normalize_r=off, R.wk_r=off, R.wq_r=off, R.wv_r=off, R.kt_mask1_r=28, R.q_mask1_r=28, R.q_mask2_r=off, R.qkt_matmul_r=28, K=**9** |
| L08 | B3 | F.x_fresh=28, S.inv_2n=15, R.sq0=31, R.sq1=31, R.sq2=31, R.sq3=31, R.x_inv_2n_r=22, K=**13** |
| L08 | B4 | F.softmax_out_fresh=35, F.v_fresh=25, M.softmax_out_mask=14, M.v_mask=14, M.softmax_v_mask=14, S.ln_mean_inv_d=20, S.ln_var_inv_d=20, W.wo=22, R.softmax_v_matmul_r=31, R.ln_mean_r=31, R.ln_var_r=off, R.softmax_out_mask_r=off, R.v_mask_r=off, R.softmax_v_mask_r=off, R.wo_r=off, R.ln_square_r=31, K=**9** |
| L08 | B5 | F.inv_std_fresh=31, F.x_centered_fresh=31, M.gamma=20, W.wffn1=22, M.gelu_coeff=20, R.normalize_r=31, R.gamma_r=off, R.wffn1_r=31, R.gp0=off, R.gp1=off, R.gp2=off, R.gc0=31, R.gc1=off, R.gc2=off, R.gc3=off, K=**9** |
| L09 | B1 | F.gelu_out=30, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=20, R.mean_r=30, R.var_r=27, R.wffn2_r=off, R.square_r=off, K=**9** |
| L09 | B2 | F.inv_std_fresh=28, F.x_centered_fresh=28, M.gamma=20, W.wk=22, W.wv=22, M.kt_mask1=15, M.kt_mask2=15, M.qkt_merge_mask=15, R.gamma_r=28, R.kt_mask2_r=off, R.qkt_merge_mask_r=off, W.wq=22, M.q_mask1=15, M.q_mask2=15, R.normalize_r=off, R.wk_r=off, R.wq_r=off, R.wv_r=off, R.kt_mask1_r=28, R.q_mask1_r=28, R.q_mask2_r=off, R.qkt_matmul_r=28, K=**9** |
| L09 | B3 | F.x_fresh=28, S.inv_2n=15, R.sq0=31, R.sq1=31, R.sq2=31, R.sq3=31, R.x_inv_2n_r=22, K=**13** |
| L09 | B4 | F.softmax_out_fresh=35, F.v_fresh=25, M.softmax_out_mask=14, M.v_mask=14, M.softmax_v_mask=14, S.ln_mean_inv_d=20, S.ln_var_inv_d=20, W.wo=22, R.softmax_v_matmul_r=31, R.ln_mean_r=31, R.ln_var_r=off, R.softmax_out_mask_r=off, R.v_mask_r=off, R.softmax_v_mask_r=off, R.wo_r=off, R.ln_square_r=31, K=**9** |
| L09 | B5 | F.inv_std_fresh=31, F.x_centered_fresh=31, M.gamma=20, W.wffn1=22, M.gelu_coeff=20, R.normalize_r=28, R.gamma_r=off, R.wffn1_r=off, R.gp0=off, R.gp1=off, R.gp2=off, R.gc0=31, R.gc1=off, R.gc2=off, R.gc3=off, K=**9** |
| L10 | B1 | F.gelu_out=30, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=20, R.mean_r=30, R.var_r=27, R.wffn2_r=off, R.square_r=off, K=**9** |
| L10 | B2 | F.inv_std_fresh=28, F.x_centered_fresh=28, M.gamma=20, W.wk=22, W.wv=22, M.kt_mask1=15, M.kt_mask2=15, M.qkt_merge_mask=15, R.gamma_r=28, R.kt_mask2_r=off, R.qkt_merge_mask_r=off, W.wq=22, M.q_mask1=15, M.q_mask2=15, R.normalize_r=off, R.wk_r=off, R.wq_r=off, R.wv_r=off, R.kt_mask1_r=28, R.q_mask1_r=28, R.q_mask2_r=off, R.qkt_matmul_r=28, K=**9** |
| L10 | B3 | F.x_fresh=28, S.inv_2n=15, R.sq0=31, R.sq1=31, R.sq2=31, R.sq3=31, R.x_inv_2n_r=22, K=**13** |
| L10 | B4 | F.softmax_out_fresh=35, F.v_fresh=25, M.softmax_out_mask=14, M.v_mask=14, M.softmax_v_mask=14, S.ln_mean_inv_d=20, S.ln_var_inv_d=20, W.wo=22, R.softmax_v_matmul_r=31, R.ln_mean_r=31, R.ln_var_r=off, R.softmax_out_mask_r=off, R.v_mask_r=off, R.softmax_v_mask_r=off, R.wo_r=off, R.ln_square_r=31, K=**9** |
| L10 | B5 | F.inv_std_fresh=31, F.x_centered_fresh=31, M.gamma=20, W.wffn1=22, M.gelu_coeff=20, R.normalize_r=28, R.gamma_r=off, R.wffn1_r=off, R.gp0=off, R.gp1=off, R.gp2=off, R.gc0=31, R.gc1=off, R.gc2=off, R.gc3=off, K=**9** |
| L11 | B1 | F.gelu_out=30, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=20, R.mean_r=30, R.var_r=27, R.wffn2_r=off, R.square_r=off, K=**9** |
| L11 | B2 | F.inv_std_fresh=28, F.x_centered_fresh=28, M.gamma=20, W.wk=22, W.wv=22, M.kt_mask1=15, M.kt_mask2=15, M.qkt_merge_mask=15, R.gamma_r=28, R.kt_mask2_r=off, R.qkt_merge_mask_r=off, W.wq=22, M.q_mask1=15, M.q_mask2=15, R.normalize_r=off, R.wk_r=off, R.wq_r=off, R.wv_r=off, R.kt_mask1_r=28, R.q_mask1_r=28, R.q_mask2_r=off, R.qkt_matmul_r=28, K=**9** |
| L11 | B3 | F.x_fresh=28, S.inv_2n=15, R.sq0=31, R.sq1=31, R.sq2=31, R.sq3=31, R.x_inv_2n_r=22, K=**13** |
| L11 | B4 | F.softmax_out_fresh=35, F.v_fresh=25, M.softmax_out_mask=14, M.v_mask=14, M.softmax_v_mask=14, S.ln_mean_inv_d=20, S.ln_var_inv_d=20, W.wo=22, R.softmax_v_matmul_r=31, R.ln_mean_r=31, R.ln_var_r=off, R.softmax_out_mask_r=off, R.v_mask_r=off, R.softmax_v_mask_r=off, R.wo_r=off, R.ln_square_r=31, K=**9** |
| L11 | B5 | F.inv_std_fresh=31, F.x_centered_fresh=31, M.gamma=20, W.wffn1=22, M.gelu_coeff=20, R.normalize_r=28, R.gamma_r=off, R.wffn1_r=off, R.gp0=off, R.gp1=off, R.gp2=off, R.gc0=31, R.gc1=off, R.gc2=off, R.gc3=off, K=**9** |
| (legacy) first_input | – | `scaling_factor=30` |

### 5.2 Best vs baseline · 哪些槽位变了

_共 105 个槽位发生变化（60 个 SF + 45 个 K bits）_

**截断 K bits 变化**：

| 槽位 | baseline K | best K | Δ |
|---|---:|---:|---:|
| `L0.B2.K` | 13 | 8 | -5 |
| `L0.B5.K` | 13 | 12 | -1 |
| `L1.B1.K` | 13 | 12 | -1 |
| `L1.B2.K` | 13 | 12 | -1 |
| `L1.B4.K` | 13 | 8 | -5 |
| `L1.B5.K` | 13 | 8 | -5 |
| `L10.B1.K` | 13 | 9 | -4 |
| `L10.B2.K` | 13 | 9 | -4 |
| `L10.B4.K` | 13 | 9 | -4 |
| `L10.B5.K` | 13 | 9 | -4 |
| `L11.B1.K` | 13 | 9 | -4 |
| `L11.B2.K` | 13 | 9 | -4 |
| `L11.B4.K` | 13 | 9 | -4 |
| `L11.B5.K` | 13 | 9 | -4 |
| `L2.B1.K` | 13 | 12 | -1 |
| `L2.B2.K` | 13 | 12 | -1 |
| `L2.B4.K` | 13 | 9 | -4 |
| `L2.B5.K` | 13 | 9 | -4 |
| `L3.B1.K` | 13 | 9 | -4 |
| `L3.B2.K` | 13 | 9 | -4 |
| `L3.B4.K` | 13 | 9 | -4 |
| `L3.B5.K` | 13 | 9 | -4 |
| `L4.B1.K` | 13 | 9 | -4 |
| `L4.B2.K` | 13 | 9 | -4 |
| `L4.B4.K` | 13 | 9 | -4 |
| `L4.B5.K` | 13 | 9 | -4 |
| `L5.B1.K` | 13 | 9 | -4 |
| `L5.B2.K` | 13 | 9 | -4 |
| `L5.B4.K` | 13 | 9 | -4 |
| `L5.B5.K` | 13 | 9 | -4 |
| `L6.B1.K` | 13 | 9 | -4 |
| `L6.B2.K` | 13 | 9 | -4 |
| `L6.B4.K` | 13 | 9 | -4 |
| `L6.B5.K` | 13 | 9 | -4 |
| `L7.B1.K` | 13 | 9 | -4 |
| `L7.B2.K` | 13 | 9 | -4 |
| `L7.B5.K` | 13 | 9 | -4 |
| `L8.B1.K` | 13 | 9 | -4 |
| `L8.B2.K` | 13 | 9 | -4 |
| `L8.B4.K` | 13 | 9 | -4 |
| `L8.B5.K` | 13 | 9 | -4 |
| `L9.B1.K` | 13 | 9 | -4 |
| `L9.B2.K` | 13 | 9 | -4 |
| `L9.B4.K` | 13 | 9 | -4 |
| `L9.B5.K` | 13 | 9 | -4 |

**Scaling factor 变化**（前 25 条按 |Δ| 降序）：

| 槽位 | kind | baseline SF | best SF | Δ |
|---|:---:|---:|---:|---:|
| `L0.B3.R.sq0` | `R` | off | 31 | off→on |
| `L0.B3.R.sq1` | `R` | off | 31 | off→on |
| `L0.B3.R.sq2` | `R` | off | 31 | off→on |
| `L0.B3.R.sq3` | `R` | off | 31 | off→on |
| `L0.B3.R.x_inv_2n_r` | `R` | off | 22 | off→on |
| `L1.B3.R.sq0` | `R` | off | 31 | off→on |
| `L1.B3.R.sq1` | `R` | off | 31 | off→on |
| `L1.B3.R.sq2` | `R` | off | 31 | off→on |
| `L1.B3.R.sq3` | `R` | off | 31 | off→on |
| `L1.B3.R.x_inv_2n_r` | `R` | off | 22 | off→on |
| `L2.B3.R.sq0` | `R` | off | 31 | off→on |
| `L2.B3.R.sq1` | `R` | off | 31 | off→on |
| `L2.B3.R.sq2` | `R` | off | 31 | off→on |
| `L2.B3.R.sq3` | `R` | off | 31 | off→on |
| `L2.B3.R.x_inv_2n_r` | `R` | off | 22 | off→on |
| `L3.B3.R.sq0` | `R` | off | 31 | off→on |
| `L3.B3.R.sq1` | `R` | off | 31 | off→on |
| `L3.B3.R.sq2` | `R` | off | 31 | off→on |
| `L3.B3.R.sq3` | `R` | off | 31 | off→on |
| `L3.B3.R.x_inv_2n_r` | `R` | off | 22 | off→on |
| `L4.B3.R.sq0` | `R` | off | 31 | off→on |
| `L4.B3.R.sq1` | `R` | off | 31 | off→on |
| `L4.B3.R.sq2` | `R` | off | 31 | off→on |
| `L4.B3.R.sq3` | `R` | off | 31 | off→on |
| `L4.B3.R.x_inv_2n_r` | `R` | off | 22 | off→on |

<details>
<summary>调试用：原始 action_vec（整数索引）</summary>

- 长度: 877

```
14, 14, 14, 14, 14, 14, 14, 14, 3, 14, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 14, 14, 14, 0, 0, 0, 0, 14, 14, 0, 14, 0, 14, 14, 14, 14, 14, 14, 14, 3, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 0, 0, 0, 14, 3, 14, 14, 14, 14, 14, 14, 0, 0, 0, 0, 0, 14, 0, 0, 0, 5, 14, 14, 14, 14, 14, 14, 0, 0, 5, 14, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 14, 14, 14, 0, 0, 0, 0, 14, 14, 0, 14, 5, 14, 14, 14, 14, 14, 14, 14, 3, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 0, 0, 0, 14, 0, 14, 14, 14, 14, 14, 14, 0, 14, 0, 0, 0, 14, 0, 0, 0, 0, 14, 14, 14, 14, 14, 14, 0, 0, 5, 14, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 14, 14, 14, 0, 0, 0, 0, 14, 14, 0, 14, 5, 14, 14, 14, 14, 14, 14, 14, 3, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 0, 0, 0, 14, 1, 14, 14, 14, 14, 14, 14, 0, 0, 0, 0, 0, 14, 0, 0, 0, 1, 14, 14, 14, 14, 14, 14, 0, 0, 1, 14, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 14, 14, 14, 0, 0, 0, 0, 14, 14, 0, 14, 1, 14, 14, 14, 14, 14, 14, 14, 3, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 0, 0, 0, 14, 1, 14, 14, 14, 14, 14, 14, 0, 0, 0, 0, 0, 14, 0, 0, 0, 1, 14, 14, 14, 14, 14, 14, 0, 0, 1, 14, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 14, 14, 14, 0, 0, 0, 0, 14, 14, 0, 14, 1, 14, 14, 14, 14, 14, 14, 14, 3, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 0, 0, 0, 14, 1, 14, 14, 14, 14, 14, 14, 0, 0, 0, 0, 0, 14, 0, 0, 0, 1, 14, 14, 14, 14, 14, 14, 0, 0, 1, 14, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 14, 14, 14, 0, 0, 0, 0, 14, 14, 0, 14, 1, 14, 14, 14, 14, 14, 14, 14, 3, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 0, 0, 0, 14, 1, 14, 14, 14, 14, 14, 14, 0, 0, 0, 0, 0, 14, 0, 0, 0, 1, 14, 14, 14, 14, 14, 14, 0, 0, 1, 14, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 14, 14, 14, 0, 0, 0, 0, 14, 14, 0, 14, 1, 14, 14, 14, 14, 14, 14, 14, 3, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 0, 0, 0, 14, 1, 14, 14, 14, 14, 14, 14, 0, 0, 0, 0, 0, 14, 0, 0, 0, 1, 14, 14, 14, 14, 14, 14, 0, 0, 1, 14, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 14, 14, 14, 0, 0, 0, 0, 14, 14, 0, 14, 1, 14, 14, 14, 14, 14, 14, 14, 3, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 0, 0, 0, 14, 3, 14, 14, 14, 14, 14, 14, 0, 0, 0, 0, 0, 14, 0, 0, 0, 1, 14, 14, 14, 14, 14, 14, 0, 0, 1, 14, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 14, 14, 14, 0, 0, 0, 0, 14, 14, 0, 14, 1, 14, 14, 14, 14, 14, 14, 14, 3, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 0, 0, 0, 14, 1, 14, 14, 14, 14, 14, 14, 0, 14, 0, 0, 0, 14, 0, 0, 0, 1, 14, 14, 14, 14, 14, 14, 0, 0, 1, 14, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 14, 14, 14, 0, 0, 0, 0, 14, 14, 0, 14, 1, 14, 14, 14, 14, 14, 14, 14, 3, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 0, 0, 0, 14, 1, 14, 14, 14, 14, 14, 14, 0, 0, 0, 0, 0, 14, 0, 0, 0, 1, 14, 14, 14, 14, 14, 14, 0, 0, 1, 14, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 14, 14, 14, 0, 0, 0, 0, 14, 14, 0, 14, 1, 14, 14, 14, 14, 14, 14, 14, 3, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 0, 0, 0, 14, 1, 14, 14, 14, 14, 14, 14, 0, 0, 0, 0, 0, 14, 0, 0, 0, 1, 14, 14, 14, 14, 14, 14, 0, 0, 1, 14, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 14, 14, 14, 0, 0, 0, 0, 14, 14, 0, 14, 1, 14, 14, 14, 14, 14, 14, 14, 3, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 0, 0, 0, 14, 1, 14, 14, 14, 14, 14, 14, 0, 0, 0, 0, 0, 14, 0, 0, 0, 1, 4
```

</details>

---

> 持久化目录：`Parting Chapter/<run>/stage2_noise/progress/`。live checkpoint / final checkpoint / best_cfg.pkl / 状态板 / 训练曲线（PNG + NPZ）/ 本报告 都在该目录下。