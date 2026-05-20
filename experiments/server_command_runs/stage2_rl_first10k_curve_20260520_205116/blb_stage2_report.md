# BLB Stage 2 RL 训练报告（最终版）

- 运行名（run_basename）: `s1t0.005_s2t0.005_s2st0.005`
- Profile（数据集）: `mrpc`
- 生成时间: 2026-05-20T21:23:09.718900
- 训练时长: 1836.5 秒（约 30.6 分钟）
- Episode 进度: 1000 / 1000
- 模数链 invoker: `in_process_real`

## 1. Reward 概览

- 最优 reward (best): **+38.170875**
- 全程 episode reward 均值: +36.9547
- 全程 episode reward 最大值: +38.1709
- 全程 episode reward 最小值: -3.1759
- 全程 episode reward 标准差: 4.5977

## 3. Baseline（全 max action）对照

| 字段 | 值 |
|------|------|
| `total_bits_sum` | 14779 |
| `total_fusion_count` | 0 |
| `avg_k` | 13.0 |
| `loss_mean` | 0.34138768911361694 |
| `metric1_mean` | 0.875 |
| `metric2_mean` | 0.875 |

## 4. Reward 权重

| 字段 | 值 |
|------|------|
| `design` | v2-style rdv2 |
| `cost_weight` | 1.0 |
| `lambda_stab` | 1.0 |
| `invalid_penalty` | 5.0 |
| `reward_clip_min` | -5.0 |
| `reward_clip_max` | 5.0 |
| `tier_metric_bonus` | 20.0 |
| `tier_stability_bonus` | 20.0 |
| `baseline_metric1` | 0.875 |

## 5. 最优 action：选了什么 SF / K（人类视图）

完整的逐槽位明细在 `blb_stage2_best_action_full.md` （人类阅读）和 `blb_stage2_best_action_full.json` （可直接喂给 `Paean/run_final_eval.sh --action-config`）。下面只列出与 baseline 不同的槽位。

### 5.1 Best action · 按层 / block 选择概览

| 层 | block | 槽位选择 |
|---|---|---|
| L00 | B1 | F.gelu_out=off, W.wffn2=off, S.mean_inv_d=off, S.var_inv_d=off, R.mean_r=off, R.var_r=off, R.wffn2_r=off, R.square_r=off, K=**None** |
| L00 | B2 | F.inv_std_fresh=31, F.x_centered_fresh=30, M.gamma=20, W.wk=22, W.wv=22, M.kt_mask1=15, M.kt_mask2=15, M.qkt_merge_mask=15, R.gamma_r=31, R.kt_mask2_r=31, R.qkt_merge_mask_r=28, W.wq=22, M.q_mask1=22, M.q_mask2=22, R.normalize_r=off, R.wk_r=off, R.wq_r=off, R.wv_r=off, R.kt_mask1_r=off, R.q_mask1_r=off, R.q_mask2_r=off, R.qkt_matmul_r=off, K=**13** |
| L00 | B3 | F.x_fresh=27, S.inv_2n=16, R.sq0=34, R.sq1=34, R.sq2=off, R.sq3=off, R.x_inv_2n_r=off, K=**13** |
| L00 | B4 | F.softmax_out_fresh=35, F.v_fresh=30, M.softmax_out_mask=14, M.v_mask=14, M.softmax_v_mask=14, S.ln_mean_inv_d=20, S.ln_var_inv_d=20, W.wo=22, R.softmax_v_matmul_r=31, R.ln_mean_r=31, R.ln_var_r=28, R.softmax_out_mask_r=off, R.v_mask_r=off, R.softmax_v_mask_r=off, R.wo_r=off, R.ln_square_r=off, K=**13** |
| L00 | B5 | F.inv_std_fresh=30, F.x_centered_fresh=30, M.gamma=20, W.wffn1=22, M.gelu_coeff=20, R.normalize_r=off, R.gamma_r=30, R.wffn1_r=off, R.gp0=off, R.gp1=off, R.gp2=off, R.gc0=off, R.gc1=off, R.gc2=off, R.gc3=off, K=**13** |
| L01 | B1 | F.gelu_out=30, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=20, R.mean_r=34, R.var_r=34, R.wffn2_r=off, R.square_r=off, K=**13** |
| L01 | B2 | F.inv_std_fresh=31, F.x_centered_fresh=30, M.gamma=20, W.wk=22, W.wv=22, M.kt_mask1=15, M.kt_mask2=15, M.qkt_merge_mask=13, R.gamma_r=31, R.kt_mask2_r=31, R.qkt_merge_mask_r=28, W.wq=22, M.q_mask1=22, M.q_mask2=22, R.normalize_r=off, R.wk_r=off, R.wq_r=off, R.wv_r=off, R.kt_mask1_r=off, R.q_mask1_r=off, R.q_mask2_r=off, R.qkt_matmul_r=off, K=**13** |
| L01 | B3 | F.x_fresh=27, S.inv_2n=16, R.sq0=34, R.sq1=34, R.sq2=off, R.sq3=off, R.x_inv_2n_r=off, K=**13** |
| L01 | B4 | F.softmax_out_fresh=35, F.v_fresh=30, M.softmax_out_mask=14, M.v_mask=14, M.softmax_v_mask=14, S.ln_mean_inv_d=20, S.ln_var_inv_d=20, W.wo=22, R.softmax_v_matmul_r=31, R.ln_mean_r=31, R.ln_var_r=28, R.softmax_out_mask_r=off, R.v_mask_r=off, R.softmax_v_mask_r=off, R.wo_r=off, R.ln_square_r=off, K=**13** |
| L01 | B5 | F.inv_std_fresh=30, F.x_centered_fresh=30, M.gamma=20, W.wffn1=22, M.gelu_coeff=20, R.normalize_r=off, R.gamma_r=30, R.wffn1_r=off, R.gp0=off, R.gp1=off, R.gp2=off, R.gc0=off, R.gc1=off, R.gc2=off, R.gc3=off, K=**13** |
| L02 | B1 | F.gelu_out=30, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=20, R.mean_r=34, R.var_r=34, R.wffn2_r=off, R.square_r=off, K=**13** |
| L02 | B2 | F.inv_std_fresh=31, F.x_centered_fresh=30, M.gamma=20, W.wk=22, W.wv=22, M.kt_mask1=15, M.kt_mask2=15, M.qkt_merge_mask=15, R.gamma_r=31, R.kt_mask2_r=31, R.qkt_merge_mask_r=28, W.wq=22, M.q_mask1=22, M.q_mask2=22, R.normalize_r=off, R.wk_r=off, R.wq_r=off, R.wv_r=off, R.kt_mask1_r=off, R.q_mask1_r=off, R.q_mask2_r=off, R.qkt_matmul_r=off, K=**13** |
| L02 | B3 | F.x_fresh=28, S.inv_2n=15, R.sq0=31, R.sq1=31, R.sq2=31, R.sq3=31, R.x_inv_2n_r=off, K=**13** |
| L02 | B4 | F.softmax_out_fresh=35, F.v_fresh=30, M.softmax_out_mask=14, M.v_mask=14, M.softmax_v_mask=14, S.ln_mean_inv_d=20, S.ln_var_inv_d=20, W.wo=22, R.softmax_v_matmul_r=31, R.ln_mean_r=31, R.ln_var_r=28, R.softmax_out_mask_r=off, R.v_mask_r=off, R.softmax_v_mask_r=off, R.wo_r=off, R.ln_square_r=off, K=**13** |
| L02 | B5 | F.inv_std_fresh=30, F.x_centered_fresh=30, M.gamma=20, W.wffn1=22, M.gelu_coeff=20, R.normalize_r=off, R.gamma_r=30, R.wffn1_r=off, R.gp0=off, R.gp1=off, R.gp2=off, R.gc0=off, R.gc1=off, R.gc2=off, R.gc3=off, K=**13** |
| L03 | B1 | F.gelu_out=30, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=20, R.mean_r=34, R.var_r=34, R.wffn2_r=off, R.square_r=off, K=**13** |
| L03 | B2 | F.inv_std_fresh=31, F.x_centered_fresh=30, M.gamma=20, W.wk=22, W.wv=22, M.kt_mask1=15, M.kt_mask2=15, M.qkt_merge_mask=15, R.gamma_r=31, R.kt_mask2_r=31, R.qkt_merge_mask_r=28, W.wq=22, M.q_mask1=22, M.q_mask2=22, R.normalize_r=off, R.wk_r=off, R.wq_r=off, R.wv_r=off, R.kt_mask1_r=off, R.q_mask1_r=off, R.q_mask2_r=off, R.qkt_matmul_r=off, K=**12** |
| L03 | B3 | F.x_fresh=28, S.inv_2n=15, R.sq0=29, R.sq1=31, R.sq2=31, R.sq3=31, R.x_inv_2n_r=off, K=**13** |
| L03 | B4 | F.softmax_out_fresh=35, F.v_fresh=30, M.softmax_out_mask=14, M.v_mask=14, M.softmax_v_mask=14, S.ln_mean_inv_d=20, S.ln_var_inv_d=20, W.wo=22, R.softmax_v_matmul_r=31, R.ln_mean_r=31, R.ln_var_r=28, R.softmax_out_mask_r=off, R.v_mask_r=off, R.softmax_v_mask_r=off, R.wo_r=off, R.ln_square_r=off, K=**13** |
| L03 | B5 | F.inv_std_fresh=30, F.x_centered_fresh=30, M.gamma=20, W.wffn1=22, M.gelu_coeff=20, R.normalize_r=off, R.gamma_r=30, R.wffn1_r=off, R.gp0=off, R.gp1=off, R.gp2=off, R.gc0=off, R.gc1=off, R.gc2=off, R.gc3=off, K=**13** |
| L04 | B1 | F.gelu_out=30, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=20, R.mean_r=34, R.var_r=34, R.wffn2_r=off, R.square_r=off, K=**13** |
| L04 | B2 | F.inv_std_fresh=31, F.x_centered_fresh=30, M.gamma=20, W.wk=22, W.wv=22, M.kt_mask1=15, M.kt_mask2=15, M.qkt_merge_mask=15, R.gamma_r=31, R.kt_mask2_r=31, R.qkt_merge_mask_r=28, W.wq=22, M.q_mask1=22, M.q_mask2=22, R.normalize_r=off, R.wk_r=off, R.wq_r=off, R.wv_r=off, R.kt_mask1_r=off, R.q_mask1_r=off, R.q_mask2_r=off, R.qkt_matmul_r=off, K=**13** |
| L04 | B3 | F.x_fresh=28, S.inv_2n=15, R.sq0=31, R.sq1=31, R.sq2=29, R.sq3=31, R.x_inv_2n_r=off, K=**13** |
| L04 | B4 | F.softmax_out_fresh=35, F.v_fresh=30, M.softmax_out_mask=14, M.v_mask=14, M.softmax_v_mask=14, S.ln_mean_inv_d=20, S.ln_var_inv_d=20, W.wo=22, R.softmax_v_matmul_r=31, R.ln_mean_r=31, R.ln_var_r=28, R.softmax_out_mask_r=off, R.v_mask_r=off, R.softmax_v_mask_r=off, R.wo_r=off, R.ln_square_r=off, K=**13** |
| L04 | B5 | F.inv_std_fresh=30, F.x_centered_fresh=30, M.gamma=20, W.wffn1=22, M.gelu_coeff=20, R.normalize_r=off, R.gamma_r=30, R.wffn1_r=off, R.gp0=off, R.gp1=off, R.gp2=off, R.gc0=off, R.gc1=off, R.gc2=off, R.gc3=off, K=**13** |
| L05 | B1 | F.gelu_out=30, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=20, R.mean_r=34, R.var_r=34, R.wffn2_r=off, R.square_r=off, K=**13** |
| L05 | B2 | F.inv_std_fresh=31, F.x_centered_fresh=30, M.gamma=20, W.wk=22, W.wv=22, M.kt_mask1=15, M.kt_mask2=15, M.qkt_merge_mask=15, R.gamma_r=31, R.kt_mask2_r=31, R.qkt_merge_mask_r=28, W.wq=22, M.q_mask1=22, M.q_mask2=22, R.normalize_r=off, R.wk_r=off, R.wq_r=off, R.wv_r=off, R.kt_mask1_r=off, R.q_mask1_r=off, R.q_mask2_r=off, R.qkt_matmul_r=off, K=**13** |
| L05 | B3 | F.x_fresh=27, S.inv_2n=16, R.sq0=34, R.sq1=34, R.sq2=off, R.sq3=off, R.x_inv_2n_r=off, K=**13** |
| L05 | B4 | F.softmax_out_fresh=35, F.v_fresh=30, M.softmax_out_mask=14, M.v_mask=14, M.softmax_v_mask=14, S.ln_mean_inv_d=20, S.ln_var_inv_d=18, W.wo=22, R.softmax_v_matmul_r=31, R.ln_mean_r=31, R.ln_var_r=28, R.softmax_out_mask_r=off, R.v_mask_r=off, R.softmax_v_mask_r=off, R.wo_r=off, R.ln_square_r=off, K=**13** |
| L05 | B5 | F.inv_std_fresh=30, F.x_centered_fresh=31, M.gamma=20, W.wffn1=22, M.gelu_coeff=20, R.normalize_r=31, R.gamma_r=off, R.wffn1_r=31, R.gp0=31, R.gp1=off, R.gp2=off, R.gc0=off, R.gc1=off, R.gc2=off, R.gc3=off, K=**13** |
| L06 | B1 | F.gelu_out=30, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=20, R.mean_r=34, R.var_r=34, R.wffn2_r=off, R.square_r=off, K=**13** |
| L06 | B2 | F.inv_std_fresh=31, F.x_centered_fresh=30, M.gamma=20, W.wk=22, W.wv=22, M.kt_mask1=15, M.kt_mask2=15, M.qkt_merge_mask=15, R.gamma_r=31, R.kt_mask2_r=31, R.qkt_merge_mask_r=28, W.wq=22, M.q_mask1=22, M.q_mask2=22, R.normalize_r=off, R.wk_r=off, R.wq_r=off, R.wv_r=off, R.kt_mask1_r=off, R.q_mask1_r=off, R.q_mask2_r=off, R.qkt_matmul_r=off, K=**13** |
| L06 | B3 | F.x_fresh=28, S.inv_2n=15, R.sq0=31, R.sq1=31, R.sq2=31, R.sq3=31, R.x_inv_2n_r=off, K=**13** |
| L06 | B4 | F.softmax_out_fresh=35, F.v_fresh=30, M.softmax_out_mask=14, M.v_mask=14, M.softmax_v_mask=14, S.ln_mean_inv_d=20, S.ln_var_inv_d=20, W.wo=22, R.softmax_v_matmul_r=31, R.ln_mean_r=31, R.ln_var_r=28, R.softmax_out_mask_r=off, R.v_mask_r=off, R.softmax_v_mask_r=off, R.wo_r=off, R.ln_square_r=off, K=**13** |
| L06 | B5 | F.inv_std_fresh=30, F.x_centered_fresh=30, M.gamma=20, W.wffn1=22, M.gelu_coeff=20, R.normalize_r=off, R.gamma_r=28, R.wffn1_r=off, R.gp0=off, R.gp1=off, R.gp2=off, R.gc0=off, R.gc1=off, R.gc2=off, R.gc3=off, K=**13** |
| L07 | B1 | F.gelu_out=30, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=20, R.mean_r=34, R.var_r=34, R.wffn2_r=off, R.square_r=off, K=**13** |
| L07 | B2 | F.inv_std_fresh=31, F.x_centered_fresh=30, M.gamma=18, W.wk=22, W.wv=22, M.kt_mask1=15, M.kt_mask2=15, M.qkt_merge_mask=15, R.gamma_r=31, R.kt_mask2_r=31, R.qkt_merge_mask_r=28, W.wq=22, M.q_mask1=22, M.q_mask2=22, R.normalize_r=off, R.wk_r=off, R.wq_r=off, R.wv_r=off, R.kt_mask1_r=off, R.q_mask1_r=off, R.q_mask2_r=off, R.qkt_matmul_r=off, K=**13** |
| L07 | B3 | F.x_fresh=27, S.inv_2n=16, R.sq0=34, R.sq1=34, R.sq2=off, R.sq3=off, R.x_inv_2n_r=off, K=**13** |
| L07 | B4 | F.softmax_out_fresh=35, F.v_fresh=30, M.softmax_out_mask=14, M.v_mask=14, M.softmax_v_mask=14, S.ln_mean_inv_d=20, S.ln_var_inv_d=20, W.wo=22, R.softmax_v_matmul_r=31, R.ln_mean_r=31, R.ln_var_r=28, R.softmax_out_mask_r=off, R.v_mask_r=off, R.softmax_v_mask_r=off, R.wo_r=off, R.ln_square_r=off, K=**13** |
| L07 | B5 | F.inv_std_fresh=30, F.x_centered_fresh=30, M.gamma=20, W.wffn1=22, M.gelu_coeff=20, R.normalize_r=off, R.gamma_r=30, R.wffn1_r=off, R.gp0=off, R.gp1=off, R.gp2=off, R.gc0=off, R.gc1=off, R.gc2=off, R.gc3=off, K=**13** |
| L08 | B1 | F.gelu_out=30, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=20, R.mean_r=34, R.var_r=34, R.wffn2_r=off, R.square_r=off, K=**13** |
| L08 | B2 | F.inv_std_fresh=31, F.x_centered_fresh=30, M.gamma=20, W.wk=22, W.wv=22, M.kt_mask1=15, M.kt_mask2=15, M.qkt_merge_mask=15, R.gamma_r=31, R.kt_mask2_r=31, R.qkt_merge_mask_r=28, W.wq=22, M.q_mask1=22, M.q_mask2=22, R.normalize_r=off, R.wk_r=off, R.wq_r=off, R.wv_r=off, R.kt_mask1_r=off, R.q_mask1_r=off, R.q_mask2_r=off, R.qkt_matmul_r=off, K=**13** |
| L08 | B3 | F.x_fresh=28, S.inv_2n=15, R.sq0=31, R.sq1=31, R.sq2=31, R.sq3=31, R.x_inv_2n_r=off, K=**13** |
| L08 | B4 | F.softmax_out_fresh=35, F.v_fresh=30, M.softmax_out_mask=14, M.v_mask=14, M.softmax_v_mask=14, S.ln_mean_inv_d=20, S.ln_var_inv_d=20, W.wo=22, R.softmax_v_matmul_r=31, R.ln_mean_r=31, R.ln_var_r=28, R.softmax_out_mask_r=off, R.v_mask_r=off, R.softmax_v_mask_r=off, R.wo_r=off, R.ln_square_r=off, K=**13** |
| L08 | B5 | F.inv_std_fresh=30, F.x_centered_fresh=30, M.gamma=20, W.wffn1=22, M.gelu_coeff=20, R.normalize_r=off, R.gamma_r=30, R.wffn1_r=off, R.gp0=off, R.gp1=off, R.gp2=off, R.gc0=off, R.gc1=off, R.gc2=off, R.gc3=off, K=**13** |
| L09 | B1 | F.gelu_out=30, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=20, R.mean_r=34, R.var_r=34, R.wffn2_r=off, R.square_r=off, K=**13** |
| L09 | B2 | F.inv_std_fresh=31, F.x_centered_fresh=30, M.gamma=20, W.wk=22, W.wv=22, M.kt_mask1=15, M.kt_mask2=15, M.qkt_merge_mask=15, R.gamma_r=31, R.kt_mask2_r=31, R.qkt_merge_mask_r=28, W.wq=22, M.q_mask1=22, M.q_mask2=22, R.normalize_r=off, R.wk_r=off, R.wq_r=off, R.wv_r=off, R.kt_mask1_r=off, R.q_mask1_r=off, R.q_mask2_r=off, R.qkt_matmul_r=off, K=**13** |
| L09 | B3 | F.x_fresh=28, S.inv_2n=15, R.sq0=31, R.sq1=31, R.sq2=31, R.sq3=31, R.x_inv_2n_r=off, K=**13** |
| L09 | B4 | F.softmax_out_fresh=35, F.v_fresh=30, M.softmax_out_mask=14, M.v_mask=14, M.softmax_v_mask=14, S.ln_mean_inv_d=20, S.ln_var_inv_d=20, W.wo=22, R.softmax_v_matmul_r=31, R.ln_mean_r=31, R.ln_var_r=28, R.softmax_out_mask_r=off, R.v_mask_r=off, R.softmax_v_mask_r=off, R.wo_r=off, R.ln_square_r=off, K=**13** |
| L09 | B5 | F.inv_std_fresh=30, F.x_centered_fresh=30, M.gamma=20, W.wffn1=22, M.gelu_coeff=20, R.normalize_r=off, R.gamma_r=30, R.wffn1_r=off, R.gp0=off, R.gp1=off, R.gp2=off, R.gc0=off, R.gc1=off, R.gc2=off, R.gc3=off, K=**13** |
| L10 | B1 | F.gelu_out=30, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=20, R.mean_r=34, R.var_r=34, R.wffn2_r=off, R.square_r=off, K=**13** |
| L10 | B2 | F.inv_std_fresh=31, F.x_centered_fresh=30, M.gamma=20, W.wk=22, W.wv=22, M.kt_mask1=15, M.kt_mask2=15, M.qkt_merge_mask=15, R.gamma_r=31, R.kt_mask2_r=31, R.qkt_merge_mask_r=28, W.wq=22, M.q_mask1=22, M.q_mask2=22, R.normalize_r=off, R.wk_r=off, R.wq_r=off, R.wv_r=off, R.kt_mask1_r=off, R.q_mask1_r=off, R.q_mask2_r=off, R.qkt_matmul_r=off, K=**13** |
| L10 | B3 | F.x_fresh=28, S.inv_2n=15, R.sq0=31, R.sq1=31, R.sq2=31, R.sq3=31, R.x_inv_2n_r=off, K=**13** |
| L10 | B4 | F.softmax_out_fresh=35, F.v_fresh=30, M.softmax_out_mask=14, M.v_mask=14, M.softmax_v_mask=14, S.ln_mean_inv_d=20, S.ln_var_inv_d=20, W.wo=22, R.softmax_v_matmul_r=31, R.ln_mean_r=31, R.ln_var_r=28, R.softmax_out_mask_r=off, R.v_mask_r=off, R.softmax_v_mask_r=off, R.wo_r=off, R.ln_square_r=off, K=**13** |
| L10 | B5 | F.inv_std_fresh=30, F.x_centered_fresh=30, M.gamma=20, W.wffn1=22, M.gelu_coeff=20, R.normalize_r=off, R.gamma_r=30, R.wffn1_r=off, R.gp0=off, R.gp1=off, R.gp2=off, R.gc0=off, R.gc1=off, R.gc2=off, R.gc3=off, K=**13** |
| L11 | B1 | F.gelu_out=30, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=20, R.mean_r=34, R.var_r=34, R.wffn2_r=off, R.square_r=off, K=**13** |
| L11 | B2 | F.inv_std_fresh=31, F.x_centered_fresh=30, M.gamma=20, W.wk=22, W.wv=22, M.kt_mask1=15, M.kt_mask2=15, M.qkt_merge_mask=15, R.gamma_r=31, R.kt_mask2_r=31, R.qkt_merge_mask_r=28, W.wq=22, M.q_mask1=22, M.q_mask2=22, R.normalize_r=off, R.wk_r=off, R.wq_r=off, R.wv_r=off, R.kt_mask1_r=off, R.q_mask1_r=off, R.q_mask2_r=off, R.qkt_matmul_r=off, K=**13** |
| L11 | B3 | F.x_fresh=27, S.inv_2n=16, R.sq0=34, R.sq1=34, R.sq2=off, R.sq3=off, R.x_inv_2n_r=off, K=**13** |
| L11 | B4 | F.softmax_out_fresh=35, F.v_fresh=30, M.softmax_out_mask=14, M.v_mask=14, M.softmax_v_mask=14, S.ln_mean_inv_d=20, S.ln_var_inv_d=20, W.wo=22, R.softmax_v_matmul_r=31, R.ln_mean_r=31, R.ln_var_r=28, R.softmax_out_mask_r=off, R.v_mask_r=off, R.softmax_v_mask_r=off, R.wo_r=off, R.ln_square_r=off, K=**13** |
| L11 | B5 | F.inv_std_fresh=30, F.x_centered_fresh=30, M.gamma=20, W.wffn1=22, M.gelu_coeff=20, R.normalize_r=off, R.gamma_r=30, R.wffn1_r=off, R.gp0=off, R.gp1=off, R.gp2=off, R.gc0=off, R.gc1=off, R.gc2=off, R.gc3=off, K=**13** |
| (legacy) first_input | – | `scaling_factor=30` |

### 5.2 Best vs baseline · 哪些槽位变了

_共 7 个槽位发生变化（6 个 SF + 1 个 K bits）_

**截断 K bits 变化**：

| 槽位 | baseline K | best K | Δ |
|---|---:|---:|---:|
| `L3.B2.K` | 13 | 12 | -1 |

**Scaling factor 变化**（前 25 条按 |Δ| 降序）：

| 槽位 | kind | baseline SF | best SF | Δ |
|---|:---:|---:|---:|---:|
| `L1.B2.M.qkt_merge_mask` | `M` | 15 | 13 | -2 |
| `L3.B3.R.sq0` | `R` | 31 | 29 | -2 |
| `L4.B3.R.sq2` | `R` | 31 | 29 | -2 |
| `L5.B4.S.ln_var_inv_d` | `S` | 20 | 18 | -2 |
| `L6.B5.R.gamma_r` | `R` | 30 | 28 | -2 |
| `L7.B2.M.gamma` | `M` | 20 | 18 | -2 |

<details>
<summary>调试用：原始 action_vec（整数索引）</summary>

- 长度: 877

```
0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 2, 4, 4, 2, 2, 2, 3, 3, 3, 4, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 2, 3, 3, 0, 0, 0, 3, 4, 4, 2, 2, 2, 2, 2, 4, 3, 3, 3, 0, 0, 0, 0, 0, 3, 4, 4, 2, 4, 2, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 4, 2, 2, 3, 3, 0, 0, 3, 4, 4, 2, 4, 4, 2, 2, 1, 3, 3, 3, 4, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 2, 3, 3, 0, 0, 0, 3, 4, 4, 2, 2, 2, 2, 2, 4, 3, 3, 3, 0, 0, 0, 0, 0, 3, 4, 4, 2, 4, 2, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 4, 2, 2, 3, 3, 0, 0, 3, 4, 4, 2, 4, 4, 2, 2, 2, 3, 3, 3, 4, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 2, 3, 3, 3, 3, 0, 3, 4, 4, 2, 2, 2, 2, 2, 4, 3, 3, 3, 0, 0, 0, 0, 0, 3, 4, 4, 2, 4, 2, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 4, 2, 2, 3, 3, 0, 0, 3, 4, 4, 2, 4, 4, 2, 2, 2, 3, 3, 3, 4, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 5, 4, 2, 2, 3, 3, 3, 0, 3, 4, 4, 2, 2, 2, 2, 2, 4, 3, 3, 3, 0, 0, 0, 0, 0, 3, 4, 4, 2, 4, 2, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 4, 2, 2, 3, 3, 0, 0, 3, 4, 4, 2, 4, 4, 2, 2, 2, 3, 3, 3, 4, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 2, 3, 3, 2, 3, 0, 3, 4, 4, 2, 2, 2, 2, 2, 4, 3, 3, 3, 0, 0, 0, 0, 0, 3, 4, 4, 2, 4, 2, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 4, 2, 2, 3, 3, 0, 0, 3, 4, 4, 2, 4, 4, 2, 2, 2, 3, 3, 3, 4, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 2, 3, 3, 0, 0, 0, 3, 4, 4, 2, 2, 2, 2, 1, 4, 3, 3, 3, 0, 0, 0, 0, 0, 3, 4, 4, 2, 4, 2, 3, 0, 3, 3, 0, 0, 0, 0, 0, 0, 3, 4, 4, 2, 2, 3, 3, 0, 0, 3, 4, 4, 2, 4, 4, 2, 2, 2, 3, 3, 3, 4, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 2, 3, 3, 3, 3, 0, 3, 4, 4, 2, 2, 2, 2, 2, 4, 3, 3, 3, 0, 0, 0, 0, 0, 3, 4, 4, 2, 4, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 4, 2, 2, 3, 3, 0, 0, 3, 4, 4, 1, 4, 4, 2, 2, 2, 3, 3, 3, 4, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 2, 3, 3, 0, 0, 0, 3, 4, 4, 2, 2, 2, 2, 2, 4, 3, 3, 3, 0, 0, 0, 0, 0, 3, 4, 4, 2, 4, 2, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 4, 2, 2, 3, 3, 0, 0, 3, 4, 4, 2, 4, 4, 2, 2, 2, 3, 3, 3, 4, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 2, 3, 3, 3, 3, 0, 3, 4, 4, 2, 2, 2, 2, 2, 4, 3, 3, 3, 0, 0, 0, 0, 0, 3, 4, 4, 2, 4, 2, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 4, 2, 2, 3, 3, 0, 0, 3, 4, 4, 2, 4, 4, 2, 2, 2, 3, 3, 3, 4, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 2, 3, 3, 3, 3, 0, 3, 4, 4, 2, 2, 2, 2, 2, 4, 3, 3, 3, 0, 0, 0, 0, 0, 3, 4, 4, 2, 4, 2, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 4, 2, 2, 3, 3, 0, 0, 3, 4, 4, 2, 4, 4, 2, 2, 2, 3, 3, 3, 4, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 2, 3, 3, 3, 3, 0, 3, 4, 4, 2, 2, 2, 2, 2, 4, 3, 3, 3, 0, 0, 0, 0, 0, 3, 4, 4, 2, 4, 2, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 4, 2, 2, 3, 3, 0, 0, 3, 4, 4, 2, 4, 4, 2, 2, 2, 3, 3, 3, 4, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 2, 3, 3, 0, 0, 0, 3, 4, 4, 2, 2, 2, 2, 2, 4, 3, 3, 3, 0, 0, 0, 0, 0, 3, 4, 4, 2, 4, 2, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4
```

</details>

---

> 持久化目录：`Parting Chapter/<run>/stage2_noise/progress/`。live checkpoint / final checkpoint / best_cfg.pkl / 状态板 / 训练曲线（PNG + NPZ）/ 本报告 都在该目录下。