# BLB Stage 2 RL 训练报告（最终版）

- 运行名（run_basename）: `s1t0.001_s2t0.001_s2st3.0__all4_gate_final_20260711_130812`
- Profile（数据集）: `mrpc`
- 生成时间: 2026-07-11T13:09:59.983185
- 训练时长: 8.0 秒（约 0.1 分钟）
- Episode 进度: 20 / 20
- 模数链 invoker: `in_process_real`

## 1. Reward 概览

- 最优 reward (best): **+0.000000**
- 全程 episode reward 均值: -0.3561
- 全程 episode reward 最大值: +0.3996
- 全程 episode reward 最小值: -1.0660
- 全程 episode reward 标准差: 0.4446

## 3. Baseline（全 max action）对照

| 字段 | 值 |
|------|------|
| `total_bits_sum` | 11967 |
| `total_fusion_count` | 0 |
| `avg_k` | 13.0 |
| `loss_mean` | 0.30284643173217773 |
| `metric1_mean` | 0.921875 |
| `metric2_mean` | 0.919995544995545 |

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
| `baseline_metric1` | 0.921875 |
| `baseline_metric2` | 0.919995544995545 |
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
| L00 | B2 | F.inv_std_fresh=28, F.x_centered_fresh=28, M.gamma=20, W.wk=22, W.wv=22, M.kt_mask1=15, M.kt_mask2=15, M.qkt_merge_mask=15, R.gamma_r=28, R.kt_mask2_r=off, R.qkt_merge_mask_r=off, W.wq=22, M.q_mask1=15, M.q_mask2=15, R.normalize_r=off, R.wk_r=off, R.wq_r=off, R.wv_r=off, R.kt_mask1_r=28, R.q_mask1_r=28, R.q_mask2_r=off, R.qkt_matmul_r=28, K=**13** |
| L00 | B3 | F.x_fresh=28, S.inv_2n=15, R.sq0=off, R.sq1=off, R.sq2=off, R.sq3=off, R.x_inv_2n_r=off, K=**13** |
| L00 | B4 | F.softmax_out_fresh=35, F.v_fresh=25, M.softmax_out_mask=14, M.v_mask=14, M.softmax_v_mask=14, S.ln_mean_inv_d=20, S.ln_var_inv_d=20, W.wo=22, R.softmax_v_matmul_r=31, R.ln_mean_r=31, R.ln_var_r=off, R.softmax_out_mask_r=off, R.v_mask_r=off, R.softmax_v_mask_r=off, R.wo_r=off, R.ln_square_r=31, K=**13** |
| L00 | B5 | F.inv_std_fresh=31, F.x_centered_fresh=31, M.gamma=20, W.wffn1=22, M.gelu_coeff=20, R.normalize_r=31, R.gamma_r=off, R.wffn1_r=31, R.gp0=31, R.gp1=off, R.gp2=off, R.gc0=31, R.gc1=off, R.gc2=off, R.gc3=off, K=**13** |
| L01 | B1 | F.gelu_out=30, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=20, R.mean_r=30, R.var_r=27, R.wffn2_r=off, R.square_r=off, K=**13** |
| L01 | B2 | F.inv_std_fresh=28, F.x_centered_fresh=28, M.gamma=20, W.wk=22, W.wv=22, M.kt_mask1=15, M.kt_mask2=15, M.qkt_merge_mask=15, R.gamma_r=28, R.kt_mask2_r=off, R.qkt_merge_mask_r=off, W.wq=22, M.q_mask1=15, M.q_mask2=15, R.normalize_r=off, R.wk_r=off, R.wq_r=off, R.wv_r=off, R.kt_mask1_r=28, R.q_mask1_r=28, R.q_mask2_r=off, R.qkt_matmul_r=28, K=**13** |
| L01 | B3 | F.x_fresh=28, S.inv_2n=15, R.sq0=off, R.sq1=off, R.sq2=off, R.sq3=off, R.x_inv_2n_r=off, K=**13** |
| L01 | B4 | F.softmax_out_fresh=35, F.v_fresh=25, M.softmax_out_mask=14, M.v_mask=14, M.softmax_v_mask=14, S.ln_mean_inv_d=20, S.ln_var_inv_d=20, W.wo=22, R.softmax_v_matmul_r=31, R.ln_mean_r=31, R.ln_var_r=off, R.softmax_out_mask_r=off, R.v_mask_r=off, R.softmax_v_mask_r=off, R.wo_r=off, R.ln_square_r=31, K=**13** |
| L01 | B5 | F.inv_std_fresh=31, F.x_centered_fresh=31, M.gamma=20, W.wffn1=22, M.gelu_coeff=20, R.normalize_r=31, R.gamma_r=off, R.wffn1_r=31, R.gp0=31, R.gp1=off, R.gp2=off, R.gc0=31, R.gc1=off, R.gc2=off, R.gc3=off, K=**13** |
| L02 | B1 | F.gelu_out=30, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=20, R.mean_r=30, R.var_r=27, R.wffn2_r=off, R.square_r=off, K=**13** |
| L02 | B2 | F.inv_std_fresh=28, F.x_centered_fresh=28, M.gamma=20, W.wk=22, W.wv=22, M.kt_mask1=15, M.kt_mask2=15, M.qkt_merge_mask=15, R.gamma_r=28, R.kt_mask2_r=off, R.qkt_merge_mask_r=off, W.wq=22, M.q_mask1=15, M.q_mask2=15, R.normalize_r=off, R.wk_r=off, R.wq_r=off, R.wv_r=off, R.kt_mask1_r=28, R.q_mask1_r=28, R.q_mask2_r=off, R.qkt_matmul_r=28, K=**13** |
| L02 | B3 | F.x_fresh=28, S.inv_2n=15, R.sq0=off, R.sq1=off, R.sq2=off, R.sq3=off, R.x_inv_2n_r=off, K=**13** |
| L02 | B4 | F.softmax_out_fresh=35, F.v_fresh=25, M.softmax_out_mask=14, M.v_mask=14, M.softmax_v_mask=14, S.ln_mean_inv_d=20, S.ln_var_inv_d=20, W.wo=22, R.softmax_v_matmul_r=31, R.ln_mean_r=31, R.ln_var_r=off, R.softmax_out_mask_r=off, R.v_mask_r=off, R.softmax_v_mask_r=off, R.wo_r=off, R.ln_square_r=31, K=**13** |
| L02 | B5 | F.inv_std_fresh=31, F.x_centered_fresh=31, M.gamma=20, W.wffn1=22, M.gelu_coeff=20, R.normalize_r=31, R.gamma_r=off, R.wffn1_r=31, R.gp0=31, R.gp1=off, R.gp2=off, R.gc0=31, R.gc1=off, R.gc2=off, R.gc3=off, K=**13** |
| L03 | B1 | F.gelu_out=30, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=20, R.mean_r=30, R.var_r=27, R.wffn2_r=off, R.square_r=off, K=**13** |
| L03 | B2 | F.inv_std_fresh=28, F.x_centered_fresh=28, M.gamma=20, W.wk=22, W.wv=22, M.kt_mask1=15, M.kt_mask2=15, M.qkt_merge_mask=15, R.gamma_r=28, R.kt_mask2_r=off, R.qkt_merge_mask_r=off, W.wq=22, M.q_mask1=15, M.q_mask2=15, R.normalize_r=off, R.wk_r=off, R.wq_r=off, R.wv_r=off, R.kt_mask1_r=28, R.q_mask1_r=28, R.q_mask2_r=off, R.qkt_matmul_r=28, K=**13** |
| L03 | B3 | F.x_fresh=28, S.inv_2n=15, R.sq0=off, R.sq1=off, R.sq2=off, R.sq3=off, R.x_inv_2n_r=off, K=**13** |
| L03 | B4 | F.softmax_out_fresh=35, F.v_fresh=25, M.softmax_out_mask=14, M.v_mask=14, M.softmax_v_mask=14, S.ln_mean_inv_d=20, S.ln_var_inv_d=20, W.wo=22, R.softmax_v_matmul_r=31, R.ln_mean_r=31, R.ln_var_r=off, R.softmax_out_mask_r=off, R.v_mask_r=off, R.softmax_v_mask_r=off, R.wo_r=off, R.ln_square_r=31, K=**13** |
| L03 | B5 | F.inv_std_fresh=31, F.x_centered_fresh=31, M.gamma=20, W.wffn1=22, M.gelu_coeff=20, R.normalize_r=31, R.gamma_r=off, R.wffn1_r=31, R.gp0=31, R.gp1=off, R.gp2=off, R.gc0=31, R.gc1=off, R.gc2=off, R.gc3=off, K=**13** |
| L04 | B1 | F.gelu_out=30, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=20, R.mean_r=30, R.var_r=27, R.wffn2_r=off, R.square_r=off, K=**13** |
| L04 | B2 | F.inv_std_fresh=28, F.x_centered_fresh=28, M.gamma=20, W.wk=22, W.wv=22, M.kt_mask1=15, M.kt_mask2=15, M.qkt_merge_mask=15, R.gamma_r=28, R.kt_mask2_r=off, R.qkt_merge_mask_r=off, W.wq=22, M.q_mask1=15, M.q_mask2=15, R.normalize_r=off, R.wk_r=off, R.wq_r=off, R.wv_r=off, R.kt_mask1_r=28, R.q_mask1_r=28, R.q_mask2_r=off, R.qkt_matmul_r=28, K=**13** |
| L04 | B3 | F.x_fresh=28, S.inv_2n=15, R.sq0=off, R.sq1=off, R.sq2=off, R.sq3=off, R.x_inv_2n_r=off, K=**13** |
| L04 | B4 | F.softmax_out_fresh=35, F.v_fresh=25, M.softmax_out_mask=14, M.v_mask=14, M.softmax_v_mask=14, S.ln_mean_inv_d=20, S.ln_var_inv_d=20, W.wo=22, R.softmax_v_matmul_r=31, R.ln_mean_r=31, R.ln_var_r=off, R.softmax_out_mask_r=off, R.v_mask_r=off, R.softmax_v_mask_r=off, R.wo_r=off, R.ln_square_r=31, K=**13** |
| L04 | B5 | F.inv_std_fresh=31, F.x_centered_fresh=31, M.gamma=20, W.wffn1=22, M.gelu_coeff=20, R.normalize_r=31, R.gamma_r=off, R.wffn1_r=31, R.gp0=31, R.gp1=off, R.gp2=off, R.gc0=31, R.gc1=off, R.gc2=off, R.gc3=off, K=**13** |
| L05 | B1 | F.gelu_out=30, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=20, R.mean_r=30, R.var_r=27, R.wffn2_r=off, R.square_r=off, K=**13** |
| L05 | B2 | F.inv_std_fresh=28, F.x_centered_fresh=28, M.gamma=20, W.wk=22, W.wv=22, M.kt_mask1=15, M.kt_mask2=15, M.qkt_merge_mask=15, R.gamma_r=28, R.kt_mask2_r=off, R.qkt_merge_mask_r=off, W.wq=22, M.q_mask1=15, M.q_mask2=15, R.normalize_r=off, R.wk_r=off, R.wq_r=off, R.wv_r=off, R.kt_mask1_r=28, R.q_mask1_r=28, R.q_mask2_r=off, R.qkt_matmul_r=28, K=**13** |
| L05 | B3 | F.x_fresh=28, S.inv_2n=15, R.sq0=off, R.sq1=off, R.sq2=off, R.sq3=off, R.x_inv_2n_r=off, K=**13** |
| L05 | B4 | F.softmax_out_fresh=35, F.v_fresh=25, M.softmax_out_mask=14, M.v_mask=14, M.softmax_v_mask=14, S.ln_mean_inv_d=20, S.ln_var_inv_d=20, W.wo=22, R.softmax_v_matmul_r=31, R.ln_mean_r=31, R.ln_var_r=off, R.softmax_out_mask_r=off, R.v_mask_r=off, R.softmax_v_mask_r=off, R.wo_r=off, R.ln_square_r=31, K=**13** |
| L05 | B5 | F.inv_std_fresh=31, F.x_centered_fresh=31, M.gamma=20, W.wffn1=22, M.gelu_coeff=20, R.normalize_r=31, R.gamma_r=off, R.wffn1_r=31, R.gp0=31, R.gp1=off, R.gp2=off, R.gc0=31, R.gc1=off, R.gc2=off, R.gc3=off, K=**13** |
| L06 | B1 | F.gelu_out=30, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=20, R.mean_r=30, R.var_r=27, R.wffn2_r=off, R.square_r=off, K=**13** |
| L06 | B2 | F.inv_std_fresh=28, F.x_centered_fresh=28, M.gamma=20, W.wk=22, W.wv=22, M.kt_mask1=15, M.kt_mask2=15, M.qkt_merge_mask=15, R.gamma_r=28, R.kt_mask2_r=off, R.qkt_merge_mask_r=off, W.wq=22, M.q_mask1=15, M.q_mask2=15, R.normalize_r=off, R.wk_r=off, R.wq_r=off, R.wv_r=off, R.kt_mask1_r=28, R.q_mask1_r=28, R.q_mask2_r=off, R.qkt_matmul_r=28, K=**13** |
| L06 | B3 | F.x_fresh=28, S.inv_2n=15, R.sq0=off, R.sq1=off, R.sq2=off, R.sq3=off, R.x_inv_2n_r=off, K=**13** |
| L06 | B4 | F.softmax_out_fresh=35, F.v_fresh=25, M.softmax_out_mask=14, M.v_mask=14, M.softmax_v_mask=14, S.ln_mean_inv_d=20, S.ln_var_inv_d=20, W.wo=22, R.softmax_v_matmul_r=31, R.ln_mean_r=31, R.ln_var_r=off, R.softmax_out_mask_r=off, R.v_mask_r=off, R.softmax_v_mask_r=off, R.wo_r=off, R.ln_square_r=31, K=**13** |
| L06 | B5 | F.inv_std_fresh=31, F.x_centered_fresh=31, M.gamma=20, W.wffn1=22, M.gelu_coeff=20, R.normalize_r=31, R.gamma_r=off, R.wffn1_r=31, R.gp0=31, R.gp1=off, R.gp2=off, R.gc0=31, R.gc1=off, R.gc2=off, R.gc3=off, K=**13** |
| L07 | B1 | F.gelu_out=30, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=20, R.mean_r=30, R.var_r=27, R.wffn2_r=off, R.square_r=off, K=**13** |
| L07 | B2 | F.inv_std_fresh=28, F.x_centered_fresh=28, M.gamma=20, W.wk=22, W.wv=22, M.kt_mask1=15, M.kt_mask2=15, M.qkt_merge_mask=15, R.gamma_r=28, R.kt_mask2_r=off, R.qkt_merge_mask_r=off, W.wq=22, M.q_mask1=15, M.q_mask2=15, R.normalize_r=off, R.wk_r=off, R.wq_r=off, R.wv_r=off, R.kt_mask1_r=28, R.q_mask1_r=28, R.q_mask2_r=off, R.qkt_matmul_r=28, K=**13** |
| L07 | B3 | F.x_fresh=28, S.inv_2n=15, R.sq0=off, R.sq1=off, R.sq2=off, R.sq3=off, R.x_inv_2n_r=off, K=**13** |
| L07 | B4 | F.softmax_out_fresh=35, F.v_fresh=25, M.softmax_out_mask=14, M.v_mask=14, M.softmax_v_mask=14, S.ln_mean_inv_d=20, S.ln_var_inv_d=20, W.wo=22, R.softmax_v_matmul_r=31, R.ln_mean_r=31, R.ln_var_r=off, R.softmax_out_mask_r=off, R.v_mask_r=off, R.softmax_v_mask_r=off, R.wo_r=off, R.ln_square_r=31, K=**13** |
| L07 | B5 | F.inv_std_fresh=31, F.x_centered_fresh=31, M.gamma=20, W.wffn1=22, M.gelu_coeff=20, R.normalize_r=31, R.gamma_r=off, R.wffn1_r=31, R.gp0=31, R.gp1=off, R.gp2=off, R.gc0=31, R.gc1=off, R.gc2=off, R.gc3=off, K=**13** |
| L08 | B1 | F.gelu_out=30, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=20, R.mean_r=30, R.var_r=27, R.wffn2_r=off, R.square_r=off, K=**13** |
| L08 | B2 | F.inv_std_fresh=28, F.x_centered_fresh=28, M.gamma=20, W.wk=22, W.wv=22, M.kt_mask1=15, M.kt_mask2=15, M.qkt_merge_mask=15, R.gamma_r=28, R.kt_mask2_r=off, R.qkt_merge_mask_r=off, W.wq=22, M.q_mask1=15, M.q_mask2=15, R.normalize_r=off, R.wk_r=off, R.wq_r=off, R.wv_r=off, R.kt_mask1_r=28, R.q_mask1_r=28, R.q_mask2_r=off, R.qkt_matmul_r=28, K=**13** |
| L08 | B3 | F.x_fresh=28, S.inv_2n=15, R.sq0=off, R.sq1=off, R.sq2=off, R.sq3=off, R.x_inv_2n_r=off, K=**13** |
| L08 | B4 | F.softmax_out_fresh=35, F.v_fresh=25, M.softmax_out_mask=14, M.v_mask=14, M.softmax_v_mask=14, S.ln_mean_inv_d=20, S.ln_var_inv_d=20, W.wo=22, R.softmax_v_matmul_r=31, R.ln_mean_r=31, R.ln_var_r=off, R.softmax_out_mask_r=off, R.v_mask_r=off, R.softmax_v_mask_r=off, R.wo_r=off, R.ln_square_r=31, K=**13** |
| L08 | B5 | F.inv_std_fresh=31, F.x_centered_fresh=31, M.gamma=20, W.wffn1=22, M.gelu_coeff=20, R.normalize_r=31, R.gamma_r=off, R.wffn1_r=31, R.gp0=31, R.gp1=off, R.gp2=off, R.gc0=31, R.gc1=off, R.gc2=off, R.gc3=off, K=**13** |
| L09 | B1 | F.gelu_out=30, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=20, R.mean_r=30, R.var_r=27, R.wffn2_r=off, R.square_r=off, K=**13** |
| L09 | B2 | F.inv_std_fresh=28, F.x_centered_fresh=28, M.gamma=20, W.wk=22, W.wv=22, M.kt_mask1=15, M.kt_mask2=15, M.qkt_merge_mask=15, R.gamma_r=28, R.kt_mask2_r=off, R.qkt_merge_mask_r=off, W.wq=22, M.q_mask1=15, M.q_mask2=15, R.normalize_r=off, R.wk_r=off, R.wq_r=off, R.wv_r=off, R.kt_mask1_r=28, R.q_mask1_r=28, R.q_mask2_r=off, R.qkt_matmul_r=28, K=**13** |
| L09 | B3 | F.x_fresh=28, S.inv_2n=15, R.sq0=off, R.sq1=off, R.sq2=off, R.sq3=off, R.x_inv_2n_r=off, K=**13** |
| L09 | B4 | F.softmax_out_fresh=35, F.v_fresh=25, M.softmax_out_mask=14, M.v_mask=14, M.softmax_v_mask=14, S.ln_mean_inv_d=20, S.ln_var_inv_d=20, W.wo=22, R.softmax_v_matmul_r=31, R.ln_mean_r=31, R.ln_var_r=off, R.softmax_out_mask_r=off, R.v_mask_r=off, R.softmax_v_mask_r=off, R.wo_r=off, R.ln_square_r=31, K=**13** |
| L09 | B5 | F.inv_std_fresh=31, F.x_centered_fresh=31, M.gamma=20, W.wffn1=22, M.gelu_coeff=20, R.normalize_r=31, R.gamma_r=off, R.wffn1_r=31, R.gp0=31, R.gp1=off, R.gp2=off, R.gc0=31, R.gc1=off, R.gc2=off, R.gc3=off, K=**13** |
| L10 | B1 | F.gelu_out=30, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=20, R.mean_r=30, R.var_r=27, R.wffn2_r=off, R.square_r=off, K=**13** |
| L10 | B2 | F.inv_std_fresh=28, F.x_centered_fresh=28, M.gamma=20, W.wk=22, W.wv=22, M.kt_mask1=15, M.kt_mask2=15, M.qkt_merge_mask=15, R.gamma_r=28, R.kt_mask2_r=off, R.qkt_merge_mask_r=off, W.wq=22, M.q_mask1=15, M.q_mask2=15, R.normalize_r=off, R.wk_r=off, R.wq_r=off, R.wv_r=off, R.kt_mask1_r=28, R.q_mask1_r=28, R.q_mask2_r=off, R.qkt_matmul_r=28, K=**13** |
| L10 | B3 | F.x_fresh=28, S.inv_2n=15, R.sq0=off, R.sq1=off, R.sq2=off, R.sq3=off, R.x_inv_2n_r=off, K=**13** |
| L10 | B4 | F.softmax_out_fresh=35, F.v_fresh=25, M.softmax_out_mask=14, M.v_mask=14, M.softmax_v_mask=14, S.ln_mean_inv_d=20, S.ln_var_inv_d=20, W.wo=22, R.softmax_v_matmul_r=31, R.ln_mean_r=31, R.ln_var_r=off, R.softmax_out_mask_r=off, R.v_mask_r=off, R.softmax_v_mask_r=off, R.wo_r=off, R.ln_square_r=31, K=**13** |
| L10 | B5 | F.inv_std_fresh=31, F.x_centered_fresh=31, M.gamma=20, W.wffn1=22, M.gelu_coeff=20, R.normalize_r=31, R.gamma_r=off, R.wffn1_r=31, R.gp0=31, R.gp1=off, R.gp2=off, R.gc0=31, R.gc1=off, R.gc2=off, R.gc3=off, K=**13** |
| L11 | B1 | F.gelu_out=30, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=20, R.mean_r=30, R.var_r=27, R.wffn2_r=off, R.square_r=off, K=**13** |
| L11 | B2 | F.inv_std_fresh=28, F.x_centered_fresh=28, M.gamma=20, W.wk=22, W.wv=22, M.kt_mask1=15, M.kt_mask2=15, M.qkt_merge_mask=15, R.gamma_r=28, R.kt_mask2_r=off, R.qkt_merge_mask_r=off, W.wq=22, M.q_mask1=15, M.q_mask2=15, R.normalize_r=off, R.wk_r=off, R.wq_r=off, R.wv_r=off, R.kt_mask1_r=28, R.q_mask1_r=28, R.q_mask2_r=off, R.qkt_matmul_r=28, K=**13** |
| L11 | B3 | F.x_fresh=28, S.inv_2n=15, R.sq0=off, R.sq1=off, R.sq2=off, R.sq3=off, R.x_inv_2n_r=off, K=**13** |
| L11 | B4 | F.softmax_out_fresh=35, F.v_fresh=25, M.softmax_out_mask=14, M.v_mask=14, M.softmax_v_mask=14, S.ln_mean_inv_d=20, S.ln_var_inv_d=20, W.wo=22, R.softmax_v_matmul_r=31, R.ln_mean_r=31, R.ln_var_r=off, R.softmax_out_mask_r=off, R.v_mask_r=off, R.softmax_v_mask_r=off, R.wo_r=off, R.ln_square_r=31, K=**13** |
| L11 | B5 | F.inv_std_fresh=31, F.x_centered_fresh=31, M.gamma=20, W.wffn1=22, M.gelu_coeff=20, R.normalize_r=31, R.gamma_r=off, R.wffn1_r=31, R.gp0=31, R.gp1=off, R.gp2=off, R.gc0=31, R.gc1=off, R.gc2=off, R.gc3=off, K=**13** |
| (legacy) first_input | – | `scaling_factor=30` |

_（baseline diff 不可用 —— 如果想看 best vs baseline 的具体改动，请打开 `diagnostics/best_action_vec.json` 的 `diff_vs_baseline` 字段。）_

<details>
<summary>调试用：原始 action_vec（整数索引）</summary>

- 长度: 877

```
14, 14, 14, 14, 0, 0, 0, 0, 3, 14, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 14, 14, 14, 0, 0, 0, 0, 14, 14, 0, 14, 3, 14, 14, 0, 0, 0, 0, 0, 3, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 0, 0, 0, 14, 3, 14, 14, 14, 14, 14, 14, 0, 14, 14, 0, 0, 14, 0, 0, 0, 3, 14, 14, 14, 14, 14, 14, 0, 0, 3, 14, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 14, 14, 14, 0, 0, 0, 0, 14, 14, 0, 14, 3, 14, 14, 0, 0, 0, 0, 0, 3, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 0, 0, 0, 14, 3, 14, 14, 14, 14, 14, 14, 0, 14, 14, 0, 0, 14, 0, 0, 0, 3, 14, 14, 14, 14, 14, 14, 0, 0, 3, 14, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 14, 14, 14, 0, 0, 0, 0, 14, 14, 0, 14, 3, 14, 14, 0, 0, 0, 0, 0, 3, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 0, 0, 0, 14, 3, 14, 14, 14, 14, 14, 14, 0, 14, 14, 0, 0, 14, 0, 0, 0, 3, 14, 14, 14, 14, 14, 14, 0, 0, 3, 14, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 14, 14, 14, 0, 0, 0, 0, 14, 14, 0, 14, 3, 14, 14, 0, 0, 0, 0, 0, 3, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 0, 0, 0, 14, 3, 14, 14, 14, 14, 14, 14, 0, 14, 14, 0, 0, 14, 0, 0, 0, 3, 14, 14, 14, 14, 14, 14, 0, 0, 3, 14, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 14, 14, 14, 0, 0, 0, 0, 14, 14, 0, 14, 3, 14, 14, 0, 0, 0, 0, 0, 3, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 0, 0, 0, 14, 3, 14, 14, 14, 14, 14, 14, 0, 14, 14, 0, 0, 14, 0, 0, 0, 3, 14, 14, 14, 14, 14, 14, 0, 0, 3, 14, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 14, 14, 14, 0, 0, 0, 0, 14, 14, 0, 14, 3, 14, 14, 0, 0, 0, 0, 0, 3, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 0, 0, 0, 14, 3, 14, 14, 14, 14, 14, 14, 0, 14, 14, 0, 0, 14, 0, 0, 0, 3, 14, 14, 14, 14, 14, 14, 0, 0, 3, 14, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 14, 14, 14, 0, 0, 0, 0, 14, 14, 0, 14, 3, 14, 14, 0, 0, 0, 0, 0, 3, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 0, 0, 0, 14, 3, 14, 14, 14, 14, 14, 14, 0, 14, 14, 0, 0, 14, 0, 0, 0, 3, 14, 14, 14, 14, 14, 14, 0, 0, 3, 14, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 14, 14, 14, 0, 0, 0, 0, 14, 14, 0, 14, 3, 14, 14, 0, 0, 0, 0, 0, 3, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 0, 0, 0, 14, 3, 14, 14, 14, 14, 14, 14, 0, 14, 14, 0, 0, 14, 0, 0, 0, 3, 14, 14, 14, 14, 14, 14, 0, 0, 3, 14, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 14, 14, 14, 0, 0, 0, 0, 14, 14, 0, 14, 3, 14, 14, 0, 0, 0, 0, 0, 3, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 0, 0, 0, 14, 3, 14, 14, 14, 14, 14, 14, 0, 14, 14, 0, 0, 14, 0, 0, 0, 3, 14, 14, 14, 14, 14, 14, 0, 0, 3, 14, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 14, 14, 14, 0, 0, 0, 0, 14, 14, 0, 14, 3, 14, 14, 0, 0, 0, 0, 0, 3, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 0, 0, 0, 14, 3, 14, 14, 14, 14, 14, 14, 0, 14, 14, 0, 0, 14, 0, 0, 0, 3, 14, 14, 14, 14, 14, 14, 0, 0, 3, 14, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 14, 14, 14, 0, 0, 0, 0, 14, 14, 0, 14, 3, 14, 14, 0, 0, 0, 0, 0, 3, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 0, 0, 0, 14, 3, 14, 14, 14, 14, 14, 14, 0, 14, 14, 0, 0, 14, 0, 0, 0, 3, 14, 14, 14, 14, 14, 14, 0, 0, 3, 14, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 14, 14, 14, 0, 0, 0, 0, 14, 14, 0, 14, 3, 14, 14, 0, 0, 0, 0, 0, 3, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 0, 0, 0, 0, 0, 14, 3, 14, 14, 14, 14, 14, 14, 0, 14, 14, 0, 0, 14, 0, 0, 0, 3, 4
```

</details>

---

> 持久化目录：`Parting Chapter/<run>/stage2_noise/progress/`。live checkpoint / final checkpoint / best_cfg.pkl / 状态板 / 训练曲线（PNG + NPZ）/ 本报告 都在该目录下。