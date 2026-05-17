# BLB Stage 2 RL 训练报告（最终版）

- 运行名（run_basename）: `s1t0.005_s2t0.005_s2st0.005`
- Profile（数据集）: `mrpc`
- 生成时间: 2026-05-17T09:02:38.473662
- 训练时长: 26759.3 秒（约 446.0 分钟）
- Episode 进度: 6000 / 6000
- 模数链 invoker: `in_process_real`

## 1. Reward 概览

- 最优 reward (best): **-152.555596**
- 全程 episode reward 均值: -155.8874
- 全程 episode reward 最大值: -152.5556
- 全程 episode reward 最小值: -169.5768
- 全程 episode reward 标准差: 3.6864

## 3. Baseline（全 max action）对照

| 字段 | 值 |
|------|------|
| `total_bits_sum` | 14779 |
| `total_fusion_count` | 0 |
| `avg_k` | 11.779661016949152 |
| `loss_mean` | 0.3413879871368408 |
| `metric1_mean` | 0.875 |
| `metric2_mean` | 0.875 |

## 4. Reward 权重

| 字段 | 值 |
|------|------|
| `w_bits` | 0.03333333333333333 |
| `w_fusion` | 1.0 |
| `w_k` | 1.0 |

## 5. 最优 action：选了什么 SF / K（人类视图）

完整的逐槽位明细在 `blb_stage2_best_action_full.md` （人类阅读）和 `blb_stage2_best_action_full.json` （可直接喂给 `Paean/run_final_eval.sh --action-config`）。下面只列出与 baseline 不同的槽位。

### 5.1 Best action · 按层 / block 选择概览

| 层 | block | 槽位选择 |
|---|---|---|
| L00 | B1 | F.gelu_out=22, W.wffn2=12, S.mean_inv_d=16, S.var_inv_d=16, R.mean_r=off, R.var_r=off, K=**8** |
| L00 | B2 | F.inv_std_fresh=27, F.x_centered_fresh=28, M.gamma=20, W.wk=14, W.wv=16, M.kt_mask1=11, M.kt_mask2=11, M.qkt_merge_mask=13, R.gamma_r=off, R.kt_mask2_r=27, R.qkt_merge_mask_r=off, K=**8** |
| L00 | B3 | F.x_fresh=21, S.inv_2n=12, R.sq0=32, R.sq1=off, R.sq2=off, R.sq3=off, K=**12** |
| L00 | B4 | F.softmax_out_fresh=29, F.v_fresh=28, M.softmax_out_mask=14, M.v_mask=14, M.softmax_v_mask=14, S.ln_mean_inv_d=16, S.ln_var_inv_d=16, W.wo=18, R.softmax_v_matmul_r=27, R.ln_mean_r=27, R.ln_var_r=off, K=**8** |
| L00 | B5 | F.inv_std_fresh=24, F.x_centered_fresh=28, M.gamma=20, W.wffn1=18, M.gelu_coeff=20, R.normalize_r=off, R.gamma_r=off, R.wffn1_r=27, R.gp0=31, K=**9** |
| L01 | B1 | F.gelu_out=26, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=16, R.mean_r=off, R.var_r=34, K=**12** |
| L01 | B2 | F.inv_std_fresh=25, F.x_centered_fresh=22, M.gamma=16, W.wk=18, W.wv=18, M.kt_mask1=11, M.kt_mask2=11, M.qkt_merge_mask=13, R.gamma_r=off, R.kt_mask2_r=27, R.qkt_merge_mask_r=off, K=**8** |
| L01 | B3 | F.x_fresh=21, S.inv_2n=12, R.sq0=32, R.sq1=off, R.sq2=29, R.sq3=off, K=**12** |
| L01 | B4 | F.softmax_out_fresh=29, F.v_fresh=28, M.softmax_out_mask=10, M.v_mask=10, M.softmax_v_mask=14, S.ln_mean_inv_d=16, S.ln_var_inv_d=16, W.wo=16, R.softmax_v_matmul_r=31, R.ln_mean_r=27, R.ln_var_r=off, K=**8** |
| L01 | B5 | F.inv_std_fresh=26, F.x_centered_fresh=28, M.gamma=20, W.wffn1=22, M.gelu_coeff=20, R.normalize_r=off, R.gamma_r=30, R.wffn1_r=31, R.gp0=31, K=**9** |
| L02 | B1 | F.gelu_out=26, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=16, R.mean_r=32, R.var_r=34, K=**12** |
| L02 | B2 | F.inv_std_fresh=25, F.x_centered_fresh=28, M.gamma=16, W.wk=14, W.wv=14, M.kt_mask1=11, M.kt_mask2=13, M.qkt_merge_mask=13, R.gamma_r=29, R.kt_mask2_r=27, R.qkt_merge_mask_r=off, K=**8** |
| L02 | B3 | F.x_fresh=22, S.inv_2n=15, R.sq0=29, R.sq1=off, R.sq2=29, R.sq3=off, K=**12** |
| L02 | B4 | F.softmax_out_fresh=35, F.v_fresh=22, M.softmax_out_mask=10, M.v_mask=12, M.softmax_v_mask=14, S.ln_mean_inv_d=16, S.ln_var_inv_d=16, W.wo=16, R.softmax_v_matmul_r=off, R.ln_mean_r=27, R.ln_var_r=off, K=**8** |
| L02 | B5 | F.inv_std_fresh=28, F.x_centered_fresh=28, M.gamma=18, W.wffn1=14, M.gelu_coeff=16, R.normalize_r=off, R.gamma_r=30, R.wffn1_r=29, R.gp0=off, K=**12** |
| L03 | B1 | F.gelu_out=28, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=18, R.mean_r=32, R.var_r=off, K=**12** |
| L03 | B2 | F.inv_std_fresh=29, F.x_centered_fresh=28, M.gamma=20, W.wk=14, W.wv=18, M.kt_mask1=13, M.kt_mask2=11, M.qkt_merge_mask=15, R.gamma_r=off, R.kt_mask2_r=27, R.qkt_merge_mask_r=off, K=**8** |
| L03 | B3 | F.x_fresh=22, S.inv_2n=13, R.sq0=29, R.sq1=27, R.sq2=29, R.sq3=off, K=**12** |
| L03 | B4 | F.softmax_out_fresh=29, F.v_fresh=28, M.softmax_out_mask=14, M.v_mask=14, M.softmax_v_mask=14, S.ln_mean_inv_d=16, S.ln_var_inv_d=20, W.wo=18, R.softmax_v_matmul_r=off, R.ln_mean_r=27, R.ln_var_r=off, K=**8** |
| L03 | B5 | F.inv_std_fresh=24, F.x_centered_fresh=28, M.gamma=20, W.wffn1=14, M.gelu_coeff=20, R.normalize_r=off, R.gamma_r=30, R.wffn1_r=29, R.gp0=27, K=**8** |
| L04 | B1 | F.gelu_out=24, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=16, R.mean_r=32, R.var_r=off, K=**12** |
| L04 | B2 | F.inv_std_fresh=25, F.x_centered_fresh=28, M.gamma=16, W.wk=14, W.wv=18, M.kt_mask1=11, M.kt_mask2=11, M.qkt_merge_mask=13, R.gamma_r=27, R.kt_mask2_r=27, R.qkt_merge_mask_r=off, K=**8** |
| L04 | B3 | F.x_fresh=22, S.inv_2n=11, R.sq0=29, R.sq1=27, R.sq2=29, R.sq3=off, K=**12** |
| L04 | B4 | F.softmax_out_fresh=29, F.v_fresh=30, M.softmax_out_mask=14, M.v_mask=10, M.softmax_v_mask=14, S.ln_mean_inv_d=16, S.ln_var_inv_d=16, W.wo=18, R.softmax_v_matmul_r=27, R.ln_mean_r=off, R.ln_var_r=off, K=**8** |
| L04 | B5 | F.inv_std_fresh=24, F.x_centered_fresh=28, M.gamma=16, W.wffn1=16, M.gelu_coeff=20, R.normalize_r=off, R.gamma_r=30, R.wffn1_r=27, R.gp0=27, K=**12** |
| L05 | B1 | F.gelu_out=28, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=16, R.mean_r=32, R.var_r=off, K=**12** |
| L05 | B2 | F.inv_std_fresh=25, F.x_centered_fresh=30, M.gamma=20, W.wk=14, W.wv=18, M.kt_mask1=11, M.kt_mask2=13, M.qkt_merge_mask=15, R.gamma_r=27, R.kt_mask2_r=27, R.qkt_merge_mask_r=off, K=**8** |
| L05 | B3 | F.x_fresh=21, S.inv_2n=16, R.sq0=32, R.sq1=off, R.sq2=off, R.sq3=off, K=**12** |
| L05 | B4 | F.softmax_out_fresh=29, F.v_fresh=30, M.softmax_out_mask=14, M.v_mask=10, M.softmax_v_mask=14, S.ln_mean_inv_d=16, S.ln_var_inv_d=16, W.wo=20, R.softmax_v_matmul_r=off, R.ln_mean_r=27, R.ln_var_r=off, K=**8** |
| L05 | B5 | F.inv_std_fresh=24, F.x_centered_fresh=31, M.gamma=20, W.wffn1=18, M.gelu_coeff=20, R.normalize_r=off, R.gamma_r=off, R.wffn1_r=off, R.gp0=31, K=**12** |
| L06 | B1 | F.gelu_out=24, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=20, R.mean_r=32, R.var_r=off, K=**8** |
| L06 | B2 | F.inv_std_fresh=25, F.x_centered_fresh=28, M.gamma=20, W.wk=18, W.wv=18, M.kt_mask1=11, M.kt_mask2=11, M.qkt_merge_mask=13, R.gamma_r=off, R.kt_mask2_r=29, R.qkt_merge_mask_r=off, K=**8** |
| L06 | B3 | F.x_fresh=22, S.inv_2n=15, R.sq0=29, R.sq1=off, R.sq2=29, R.sq3=off, K=**12** |
| L06 | B4 | F.softmax_out_fresh=29, F.v_fresh=28, M.softmax_out_mask=10, M.v_mask=14, M.softmax_v_mask=14, S.ln_mean_inv_d=16, S.ln_var_inv_d=16, W.wo=18, R.softmax_v_matmul_r=off, R.ln_mean_r=29, R.ln_var_r=off, K=**8** |
| L06 | B5 | F.inv_std_fresh=24, F.x_centered_fresh=24, M.gamma=20, W.wffn1=16, M.gelu_coeff=16, R.normalize_r=off, R.gamma_r=off, R.wffn1_r=31, R.gp0=31, K=**12** |
| L07 | B1 | F.gelu_out=24, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=16, R.mean_r=32, R.var_r=off, K=**12** |
| L07 | B2 | F.inv_std_fresh=25, F.x_centered_fresh=30, M.gamma=20, W.wk=18, W.wv=18, M.kt_mask1=11, M.kt_mask2=11, M.qkt_merge_mask=13, R.gamma_r=27, R.kt_mask2_r=29, R.qkt_merge_mask_r=off, K=**8** |
| L07 | B3 | F.x_fresh=21, S.inv_2n=16, R.sq0=32, R.sq1=off, R.sq2=29, R.sq3=off, K=**12** |
| L07 | B4 | F.softmax_out_fresh=29, F.v_fresh=28, M.softmax_out_mask=14, M.v_mask=10, M.softmax_v_mask=10, S.ln_mean_inv_d=16, S.ln_var_inv_d=16, W.wo=16, R.softmax_v_matmul_r=27, R.ln_mean_r=off, R.ln_var_r=off, K=**8** |
| L07 | B5 | F.inv_std_fresh=24, F.x_centered_fresh=28, M.gamma=16, W.wffn1=18, M.gelu_coeff=20, R.normalize_r=off, R.gamma_r=off, R.wffn1_r=27, R.gp0=off, K=**12** |
| L08 | B1 | F.gelu_out=24, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=20, R.mean_r=32, R.var_r=off, K=**13** |
| L08 | B2 | F.inv_std_fresh=25, F.x_centered_fresh=28, M.gamma=20, W.wk=18, W.wv=18, M.kt_mask1=11, M.kt_mask2=11, M.qkt_merge_mask=13, R.gamma_r=31, R.kt_mask2_r=off, R.qkt_merge_mask_r=off, K=**8** |
| L08 | B3 | F.x_fresh=22, S.inv_2n=15, R.sq0=29, R.sq1=off, R.sq2=29, R.sq3=off, K=**12** |
| L08 | B4 | F.softmax_out_fresh=29, F.v_fresh=28, M.softmax_out_mask=10, M.v_mask=12, M.softmax_v_mask=10, S.ln_mean_inv_d=16, S.ln_var_inv_d=16, W.wo=16, R.softmax_v_matmul_r=29, R.ln_mean_r=29, R.ln_var_r=off, K=**8** |
| L08 | B5 | F.inv_std_fresh=28, F.x_centered_fresh=28, M.gamma=20, W.wffn1=14, M.gelu_coeff=20, R.normalize_r=off, R.gamma_r=off, R.wffn1_r=27, R.gp0=31, K=**12** |
| L09 | B1 | F.gelu_out=24, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=16, R.mean_r=32, R.var_r=off, K=**8** |
| L09 | B2 | F.inv_std_fresh=25, F.x_centered_fresh=28, M.gamma=20, W.wk=14, W.wv=18, M.kt_mask1=11, M.kt_mask2=11, M.qkt_merge_mask=13, R.gamma_r=31, R.kt_mask2_r=29, R.qkt_merge_mask_r=off, K=**8** |
| L09 | B3 | F.x_fresh=22, S.inv_2n=15, R.sq0=off, R.sq1=27, R.sq2=29, R.sq3=off, K=**12** |
| L09 | B4 | F.softmax_out_fresh=29, F.v_fresh=28, M.softmax_out_mask=14, M.v_mask=10, M.softmax_v_mask=10, S.ln_mean_inv_d=16, S.ln_var_inv_d=16, W.wo=16, R.softmax_v_matmul_r=27, R.ln_mean_r=29, R.ln_var_r=off, K=**8** |
| L09 | B5 | F.inv_std_fresh=28, F.x_centered_fresh=28, M.gamma=20, W.wffn1=14, M.gelu_coeff=20, R.normalize_r=off, R.gamma_r=off, R.wffn1_r=27, R.gp0=31, K=**12** |
| L10 | B1 | F.gelu_out=24, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=16, R.mean_r=32, R.var_r=off, K=**13** |
| L10 | B2 | F.inv_std_fresh=25, F.x_centered_fresh=28, M.gamma=20, W.wk=14, W.wv=18, M.kt_mask1=11, M.kt_mask2=11, M.qkt_merge_mask=13, R.gamma_r=31, R.kt_mask2_r=29, R.qkt_merge_mask_r=off, K=**8** |
| L10 | B3 | F.x_fresh=22, S.inv_2n=15, R.sq0=off, R.sq1=off, R.sq2=29, R.sq3=off, K=**12** |
| L10 | B4 | F.softmax_out_fresh=33, F.v_fresh=28, M.softmax_out_mask=14, M.v_mask=10, M.softmax_v_mask=14, S.ln_mean_inv_d=16, S.ln_var_inv_d=16, W.wo=16, R.softmax_v_matmul_r=off, R.ln_mean_r=29, R.ln_var_r=off, K=**8** |
| L10 | B5 | F.inv_std_fresh=26, F.x_centered_fresh=28, M.gamma=16, W.wffn1=18, M.gelu_coeff=20, R.normalize_r=off, R.gamma_r=off, R.wffn1_r=27, R.gp0=27, K=**12** |
| L11 | B1 | F.gelu_out=24, W.wffn2=18, S.mean_inv_d=20, S.var_inv_d=16, R.mean_r=32, R.var_r=off, K=**12** |
| L11 | B2 | F.inv_std_fresh=27, F.x_centered_fresh=28, M.gamma=16, W.wk=14, W.wv=18, M.kt_mask1=11, M.kt_mask2=11, M.qkt_merge_mask=13, R.gamma_r=off, R.kt_mask2_r=29, R.qkt_merge_mask_r=off, K=**8** |
| L11 | B3 | F.x_fresh=25, S.inv_2n=16, R.sq0=32, R.sq1=off, R.sq2=29, R.sq3=off, K=**12** |
| L11 | B4 | F.softmax_out_fresh=29, F.v_fresh=28, M.softmax_out_mask=14, M.v_mask=10, M.softmax_v_mask=14, S.ln_mean_inv_d=16, S.ln_var_inv_d=16, W.wo=16, R.softmax_v_matmul_r=31, R.ln_mean_r=29, R.ln_var_r=off, K=**8** |
| L11 | B5 | F.inv_std_fresh=26, F.x_centered_fresh=28, M.gamma=20, W.wffn1=22, M.gelu_coeff=20, R.normalize_r=off, R.gamma_r=28, R.wffn1_r=27, R.gp0=27, K=**12** |
| (legacy) first_input | – | `scaling_factor=30` |

### 5.2 Best vs baseline · 哪些槽位变了

_共 451 个槽位发生变化（393 个 SF + 58 个 K bits）_

**截断 K bits 变化**：

| 槽位 | baseline K | best K | Δ |
|---|---:|---:|---:|
| `L0.B1.K` | 13 | 8 | -5 |
| `L0.B2.K` | 10 | 8 | -2 |
| `L0.B3.K` | 13 | 12 | -1 |
| `L0.B4.K` | 10 | 8 | -2 |
| `L0.B5.K` | 13 | 9 | -4 |
| `L1.B1.K` | 13 | 12 | -1 |
| `L1.B2.K` | 10 | 8 | -2 |
| `L1.B3.K` | 13 | 12 | -1 |
| `L1.B4.K` | 10 | 8 | -2 |
| `L1.B5.K` | 13 | 9 | -4 |
| `L10.B2.K` | 10 | 8 | -2 |
| `L10.B3.K` | 13 | 12 | -1 |
| `L10.B4.K` | 10 | 8 | -2 |
| `L10.B5.K` | 13 | 12 | -1 |
| `L11.B1.K` | 13 | 12 | -1 |
| `L11.B2.K` | 10 | 8 | -2 |
| `L11.B3.K` | 13 | 12 | -1 |
| `L11.B4.K` | 10 | 8 | -2 |
| `L11.B5.K` | 13 | 12 | -1 |
| `L2.B1.K` | 13 | 12 | -1 |
| `L2.B2.K` | 10 | 8 | -2 |
| `L2.B3.K` | 13 | 12 | -1 |
| `L2.B4.K` | 10 | 8 | -2 |
| `L2.B5.K` | 13 | 12 | -1 |
| `L3.B1.K` | 13 | 12 | -1 |
| `L3.B2.K` | 10 | 8 | -2 |
| `L3.B3.K` | 13 | 12 | -1 |
| `L3.B4.K` | 10 | 8 | -2 |
| `L3.B5.K` | 13 | 8 | -5 |
| `L4.B1.K` | 13 | 12 | -1 |
| `L4.B2.K` | 10 | 8 | -2 |
| `L4.B3.K` | 13 | 12 | -1 |
| `L4.B4.K` | 10 | 8 | -2 |
| `L4.B5.K` | 13 | 12 | -1 |
| `L5.B1.K` | 13 | 12 | -1 |
| `L5.B2.K` | 10 | 8 | -2 |
| `L5.B3.K` | 13 | 12 | -1 |
| `L5.B4.K` | 10 | 8 | -2 |
| `L5.B5.K` | 13 | 12 | -1 |
| `L6.B1.K` | 13 | 8 | -5 |
| `L6.B2.K` | 10 | 8 | -2 |
| `L6.B3.K` | 13 | 12 | -1 |
| `L6.B4.K` | 10 | 8 | -2 |
| `L6.B5.K` | 13 | 12 | -1 |
| `L7.B1.K` | 13 | 12 | -1 |
| `L7.B2.K` | 10 | 8 | -2 |
| `L7.B3.K` | 13 | 12 | -1 |
| `L7.B4.K` | 10 | 8 | -2 |
| `L7.B5.K` | 13 | 12 | -1 |
| `L8.B2.K` | 10 | 8 | -2 |
| `L8.B3.K` | 13 | 12 | -1 |
| `L8.B4.K` | 10 | 8 | -2 |
| `L8.B5.K` | 13 | 12 | -1 |
| `L9.B1.K` | 13 | 8 | -5 |
| `L9.B2.K` | 10 | 8 | -2 |
| `L9.B3.K` | 13 | 12 | -1 |
| `L9.B4.K` | 10 | 8 | -2 |
| `L9.B5.K` | 13 | 12 | -1 |

**Scaling factor 变化**（前 25 条按 |Δ| 降序）：

| 槽位 | kind | baseline SF | best SF | Δ |
|---|:---:|---:|---:|---:|
| `L0.B1.F.gelu_out` | `F` | 30 | 22 | -8 |
| `L0.B1.W.wffn2` | `W` | 20 | 12 | -8 |
| `L0.B2.W.wk` | `W` | 22 | 14 | -8 |
| `L1.B2.F.x_centered_fresh` | `F` | 30 | 22 | -8 |
| `L2.B2.W.wk` | `W` | 22 | 14 | -8 |
| `L2.B2.W.wv` | `W` | 22 | 14 | -8 |
| `L2.B4.F.v_fresh` | `F` | 30 | 22 | -8 |
| `L2.B5.W.wffn1` | `W` | 22 | 14 | -8 |
| `L3.B2.W.wk` | `W` | 22 | 14 | -8 |
| `L3.B5.W.wffn1` | `W` | 22 | 14 | -8 |
| `L4.B2.W.wk` | `W` | 22 | 14 | -8 |
| `L5.B2.W.wk` | `W` | 22 | 14 | -8 |
| `L8.B5.W.wffn1` | `W` | 22 | 14 | -8 |
| `L9.B2.W.wk` | `W` | 22 | 14 | -8 |
| `L9.B5.W.wffn1` | `W` | 22 | 14 | -8 |
| `L10.B2.W.wk` | `W` | 22 | 14 | -8 |
| `L11.B2.W.wk` | `W` | 22 | 14 | -8 |
| `L0.B2.W.wv` | `W` | 22 | 16 | -6 |
| `L0.B3.F.x_fresh` | `F` | 27 | 21 | -6 |
| `L0.B4.F.softmax_out_fresh` | `F` | 35 | 29 | -6 |
| `L0.B5.F.inv_std_fresh` | `F` | 30 | 24 | -6 |
| `L1.B2.F.inv_std_fresh` | `F` | 31 | 25 | -6 |
| `L1.B3.F.x_fresh` | `F` | 27 | 21 | -6 |
| `L1.B4.F.softmax_out_fresh` | `F` | 35 | 29 | -6 |
| `L1.B4.W.wo` | `W` | 22 | 16 | -6 |

<details>
<summary>调试用：原始 action_vec（整数索引）</summary>

- 长度: 577

```
0, 0, 0, 0, 0, 0, 0, 2, 3, 2, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 2, 0, 0, 0, 5, 1, 3, 2, 2, 2, 0, 0, 2, 1, 1, 0, 0, 1, 3, 2, 2, 2, 0, 0, 1, 3, 1, 2, 4, 2, 0, 0, 3, 5, 1, 0, 0, 2, 2, 0, 0, 1, 0, 1, 0, 0, 1, 0, 2, 0, 2, 0, 5, 1, 3, 0, 0, 2, 0, 0, 1, 3, 1, 0, 0, 2, 3, 2, 4, 2, 0, 3, 3, 3, 1, 2, 4, 2, 0, 2, 3, 5, 1, 3, 0, 0, 0, 0, 1, 1, 2, 1, 0, 0, 1, 2, 2, 0, 2, 0, 5, 4, 0, 0, 1, 2, 0, 0, 1, 0, 1, 0, 0, 3, 3, 1, 0, 0, 0, 3, 2, 0, 5, 3, 4, 2, 1, 2, 0, 5, 3, 3, 2, 0, 2, 1, 0, 2, 0, 1, 0, 0, 1, 1, 2, 1, 2, 0, 5, 1, 3, 2, 2, 2, 0, 2, 2, 0, 1, 0, 0, 1, 3, 2, 0, 2, 0, 3, 2, 1, 0, 1, 4, 2, 0, 2, 0, 5, 1, 3, 0, 0, 2, 0, 0, 1, 1, 1, 0, 0, 1, 0, 2, 1, 2, 0, 5, 1, 4, 2, 0, 2, 0, 0, 2, 1, 0, 0, 0, 1, 3, 0, 1, 2, 0, 3, 1, 1, 5, 3, 4, 2, 0, 2, 0, 5, 1, 4, 2, 0, 2, 0, 1, 2, 1, 1, 0, 0, 1, 2, 2, 0, 0, 0, 5, 1, 4, 2, 0, 2, 0, 0, 3, 0, 1, 0, 0, 1, 4, 2, 2, 2, 0, 0, 0, 3, 5, 1, 4, 2, 2, 2, 0, 0, 1, 3, 2, 2, 2, 0, 0, 1, 0, 2, 0, 0, 1, 2, 2, 0, 2, 0, 5, 1, 3, 0, 2, 2, 0, 0, 2, 0, 2, 0, 0, 1, 1, 2, 1, 0, 0, 0, 3, 3, 5, 1, 4, 2, 0, 2, 0, 5, 1, 4, 2, 2, 2, 0, 0, 1, 1, 2, 0, 0, 1, 2, 2, 0, 2, 0, 5, 1, 3, 2, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 3, 0, 2, 2, 0, 0, 1, 0, 5, 1, 4, 2, 2, 2, 0, 3, 1, 3, 2, 2, 2, 0, 0, 1, 3, 0, 0, 0, 1, 2, 2, 0, 2, 0, 5, 1, 3, 0, 1, 0, 0, 0, 1, 2, 2, 0, 0, 3, 3, 2, 0, 2, 0, 0, 1, 3, 5, 1, 4, 2, 0, 2, 0, 0, 1, 3, 2, 0, 2, 0, 0, 1, 3, 2, 0, 0, 1, 2, 0, 1, 2, 0, 5, 1, 3, 2, 0, 0, 0, 0, 1, 1, 2, 0, 0, 3, 3, 2, 0, 2, 0, 0, 1, 3, 5, 1, 4, 2, 0, 2, 0, 3, 1, 3, 2, 0, 2, 0, 0, 1, 3, 2, 0, 0, 1, 2, 0, 0, 2, 0, 5, 3, 3, 2, 0, 2, 0, 0, 1, 0, 2, 0, 0, 2, 3, 0, 2, 2, 0, 0, 1, 1, 5, 1, 3, 2, 0, 2, 0, 5, 2, 3, 0, 0, 2, 0, 0, 1, 0, 2, 0, 0, 3, 2, 2, 0, 2, 0, 5, 1, 3, 2, 0, 2, 0, 0, 1, 3, 2, 0, 0, 2, 3, 2, 4, 2, 0, 2, 1, 1, 5, 4
```

</details>

---

> 持久化目录：`Parting Chapter/<run>/stage2_noise/progress/`。live checkpoint / final checkpoint / best_cfg.pkl / 状态板 / 训练曲线（PNG + NPZ）/ 本报告 都在该目录下。