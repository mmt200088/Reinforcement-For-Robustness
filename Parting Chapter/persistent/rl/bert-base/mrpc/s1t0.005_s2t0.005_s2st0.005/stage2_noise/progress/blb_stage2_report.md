# BLB Stage 2 RL 训练报告（最终版）

- 运行名（run_basename）: `s1t0.005_s2t0.005_s2st0.005`
- Profile（数据集）: `mrpc`
- 生成时间: 2026-05-18T07:20:45.302113
- 训练时长: 28208.0 秒（约 470.1 分钟）
- Episode 进度: 6000 / 6000
- 模数链 invoker: `in_process_real`

## 1. Reward 概览

- 最优 reward (best): **-117.982140**
- 全程 episode reward 均值: -140.5932
- 全程 episode reward 最大值: -117.9821
- 全程 episode reward 最小值: -213.0960
- 全程 episode reward 标准差: 25.6179

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
| L00 | B2 | F.inv_std_fresh=23, F.x_centered_fresh=22, M.gamma=18, W.wk=14, W.wv=22, M.kt_mask1=13, M.kt_mask2=13, M.qkt_merge_mask=11, R.gamma_r=off, R.kt_mask2_r=31, R.qkt_merge_mask_r=28, K=**10** |
| L00 | B3 | F.x_fresh=19, S.inv_2n=12, R.sq0=30, R.sq1=off, R.sq2=31, R.sq3=27, K=**12** |
| L00 | B4 | F.softmax_out_fresh=27, F.v_fresh=22, M.softmax_out_mask=12, M.v_mask=10, M.softmax_v_mask=12, S.ln_mean_inv_d=18, S.ln_var_inv_d=18, W.wo=22, R.softmax_v_matmul_r=off, R.ln_mean_r=31, R.ln_var_r=28, K=**10** |
| L00 | B5 | F.inv_std_fresh=30, F.x_centered_fresh=30, M.gamma=20, W.wffn1=22, M.gelu_coeff=20, R.normalize_r=off, R.gamma_r=30, R.wffn1_r=off, R.gp0=off, K=**13** |
| L01 | B1 | F.gelu_out=30, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=20, R.mean_r=34, R.var_r=34, K=**13** |
| L01 | B2 | F.inv_std_fresh=23, F.x_centered_fresh=22, M.gamma=18, W.wk=14, W.wv=22, M.kt_mask1=13, M.kt_mask2=13, M.qkt_merge_mask=11, R.gamma_r=off, R.kt_mask2_r=31, R.qkt_merge_mask_r=28, K=**10** |
| L01 | B3 | F.x_fresh=19, S.inv_2n=12, R.sq0=30, R.sq1=off, R.sq2=31, R.sq3=27, K=**12** |
| L01 | B4 | F.softmax_out_fresh=27, F.v_fresh=22, M.softmax_out_mask=12, M.v_mask=10, M.softmax_v_mask=12, S.ln_mean_inv_d=18, S.ln_var_inv_d=18, W.wo=22, R.softmax_v_matmul_r=off, R.ln_mean_r=31, R.ln_var_r=24, K=**10** |
| L01 | B5 | F.inv_std_fresh=30, F.x_centered_fresh=30, M.gamma=20, W.wffn1=22, M.gelu_coeff=20, R.normalize_r=off, R.gamma_r=30, R.wffn1_r=off, R.gp0=off, K=**13** |
| L02 | B1 | F.gelu_out=30, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=20, R.mean_r=34, R.var_r=34, K=**13** |
| L02 | B2 | F.inv_std_fresh=23, F.x_centered_fresh=22, M.gamma=18, W.wk=14, W.wv=22, M.kt_mask1=13, M.kt_mask2=13, M.qkt_merge_mask=11, R.gamma_r=off, R.kt_mask2_r=31, R.qkt_merge_mask_r=28, K=**10** |
| L02 | B3 | F.x_fresh=20, S.inv_2n=11, R.sq0=31, R.sq1=off, R.sq2=27, R.sq3=27, K=**12** |
| L02 | B4 | F.softmax_out_fresh=27, F.v_fresh=22, M.softmax_out_mask=12, M.v_mask=10, M.softmax_v_mask=10, S.ln_mean_inv_d=18, S.ln_var_inv_d=18, W.wo=22, R.softmax_v_matmul_r=off, R.ln_mean_r=31, R.ln_var_r=28, K=**10** |
| L02 | B5 | F.inv_std_fresh=30, F.x_centered_fresh=30, M.gamma=20, W.wffn1=22, M.gelu_coeff=20, R.normalize_r=off, R.gamma_r=30, R.wffn1_r=off, R.gp0=off, K=**13** |
| L03 | B1 | F.gelu_out=30, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=20, R.mean_r=34, R.var_r=34, K=**13** |
| L03 | B2 | F.inv_std_fresh=23, F.x_centered_fresh=22, M.gamma=18, W.wk=14, W.wv=22, M.kt_mask1=13, M.kt_mask2=13, M.qkt_merge_mask=11, R.gamma_r=off, R.kt_mask2_r=31, R.qkt_merge_mask_r=28, K=**10** |
| L03 | B3 | F.x_fresh=28, S.inv_2n=15, R.sq0=31, R.sq1=31, R.sq2=31, R.sq3=31, K=**13** |
| L03 | B4 | F.softmax_out_fresh=27, F.v_fresh=22, M.softmax_out_mask=12, M.v_mask=10, M.softmax_v_mask=12, S.ln_mean_inv_d=18, S.ln_var_inv_d=18, W.wo=22, R.softmax_v_matmul_r=off, R.ln_mean_r=31, R.ln_var_r=28, K=**10** |
| L03 | B5 | F.inv_std_fresh=30, F.x_centered_fresh=30, M.gamma=20, W.wffn1=22, M.gelu_coeff=20, R.normalize_r=off, R.gamma_r=30, R.wffn1_r=off, R.gp0=off, K=**13** |
| L04 | B1 | F.gelu_out=30, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=20, R.mean_r=34, R.var_r=34, K=**13** |
| L04 | B2 | F.inv_std_fresh=23, F.x_centered_fresh=22, M.gamma=18, W.wk=14, W.wv=22, M.kt_mask1=13, M.kt_mask2=13, M.qkt_merge_mask=11, R.gamma_r=off, R.kt_mask2_r=31, R.qkt_merge_mask_r=28, K=**10** |
| L04 | B3 | F.x_fresh=28, S.inv_2n=15, R.sq0=31, R.sq1=31, R.sq2=31, R.sq3=31, K=**13** |
| L04 | B4 | F.softmax_out_fresh=27, F.v_fresh=22, M.softmax_out_mask=12, M.v_mask=10, M.softmax_v_mask=10, S.ln_mean_inv_d=18, S.ln_var_inv_d=18, W.wo=22, R.softmax_v_matmul_r=off, R.ln_mean_r=31, R.ln_var_r=24, K=**10** |
| L04 | B5 | F.inv_std_fresh=30, F.x_centered_fresh=30, M.gamma=20, W.wffn1=22, M.gelu_coeff=20, R.normalize_r=off, R.gamma_r=30, R.wffn1_r=off, R.gp0=off, K=**13** |
| L05 | B1 | F.gelu_out=30, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=20, R.mean_r=34, R.var_r=34, K=**13** |
| L05 | B2 | F.inv_std_fresh=23, F.x_centered_fresh=22, M.gamma=18, W.wk=14, W.wv=22, M.kt_mask1=13, M.kt_mask2=13, M.qkt_merge_mask=11, R.gamma_r=off, R.kt_mask2_r=31, R.qkt_merge_mask_r=24, K=**10** |
| L05 | B3 | F.x_fresh=27, S.inv_2n=16, R.sq0=34, R.sq1=34, R.sq2=off, R.sq3=off, K=**13** |
| L05 | B4 | F.softmax_out_fresh=27, F.v_fresh=22, M.softmax_out_mask=12, M.v_mask=10, M.softmax_v_mask=10, S.ln_mean_inv_d=18, S.ln_var_inv_d=18, W.wo=22, R.softmax_v_matmul_r=off, R.ln_mean_r=31, R.ln_var_r=28, K=**10** |
| L05 | B5 | F.inv_std_fresh=22, F.x_centered_fresh=23, M.gamma=18, W.wffn1=14, M.gelu_coeff=16, R.normalize_r=27, R.gamma_r=18, R.wffn1_r=31, R.gp0=off, K=**12** |
| L06 | B1 | F.gelu_out=30, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=20, R.mean_r=34, R.var_r=34, K=**13** |
| L06 | B2 | F.inv_std_fresh=23, F.x_centered_fresh=22, M.gamma=18, W.wk=14, W.wv=22, M.kt_mask1=13, M.kt_mask2=13, M.qkt_merge_mask=11, R.gamma_r=off, R.kt_mask2_r=31, R.qkt_merge_mask_r=24, K=**10** |
| L06 | B3 | F.x_fresh=28, S.inv_2n=15, R.sq0=31, R.sq1=31, R.sq2=31, R.sq3=31, K=**13** |
| L06 | B4 | F.softmax_out_fresh=27, F.v_fresh=22, M.softmax_out_mask=12, M.v_mask=10, M.softmax_v_mask=10, S.ln_mean_inv_d=18, S.ln_var_inv_d=18, W.wo=22, R.softmax_v_matmul_r=off, R.ln_mean_r=31, R.ln_var_r=28, K=**10** |
| L06 | B5 | F.inv_std_fresh=30, F.x_centered_fresh=30, M.gamma=20, W.wffn1=22, M.gelu_coeff=20, R.normalize_r=off, R.gamma_r=30, R.wffn1_r=off, R.gp0=off, K=**13** |
| L07 | B1 | F.gelu_out=30, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=20, R.mean_r=34, R.var_r=34, K=**13** |
| L07 | B2 | F.inv_std_fresh=23, F.x_centered_fresh=22, M.gamma=20, W.wk=14, W.wv=22, M.kt_mask1=13, M.kt_mask2=13, M.qkt_merge_mask=11, R.gamma_r=off, R.kt_mask2_r=31, R.qkt_merge_mask_r=24, K=**10** |
| L07 | B3 | F.x_fresh=27, S.inv_2n=16, R.sq0=34, R.sq1=34, R.sq2=off, R.sq3=off, K=**13** |
| L07 | B4 | F.softmax_out_fresh=27, F.v_fresh=22, M.softmax_out_mask=12, M.v_mask=10, M.softmax_v_mask=10, S.ln_mean_inv_d=18, S.ln_var_inv_d=18, W.wo=22, R.softmax_v_matmul_r=off, R.ln_mean_r=31, R.ln_var_r=28, K=**10** |
| L07 | B5 | F.inv_std_fresh=30, F.x_centered_fresh=30, M.gamma=20, W.wffn1=22, M.gelu_coeff=20, R.normalize_r=off, R.gamma_r=30, R.wffn1_r=off, R.gp0=off, K=**13** |
| L08 | B1 | F.gelu_out=30, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=20, R.mean_r=34, R.var_r=34, K=**13** |
| L08 | B2 | F.inv_std_fresh=23, F.x_centered_fresh=22, M.gamma=18, W.wk=14, W.wv=22, M.kt_mask1=13, M.kt_mask2=13, M.qkt_merge_mask=11, R.gamma_r=off, R.kt_mask2_r=29, R.qkt_merge_mask_r=28, K=**10** |
| L08 | B3 | F.x_fresh=28, S.inv_2n=15, R.sq0=31, R.sq1=31, R.sq2=31, R.sq3=31, K=**13** |
| L08 | B4 | F.softmax_out_fresh=27, F.v_fresh=22, M.softmax_out_mask=12, M.v_mask=10, M.softmax_v_mask=10, S.ln_mean_inv_d=18, S.ln_var_inv_d=18, W.wo=22, R.softmax_v_matmul_r=off, R.ln_mean_r=31, R.ln_var_r=24, K=**10** |
| L08 | B5 | F.inv_std_fresh=30, F.x_centered_fresh=30, M.gamma=20, W.wffn1=22, M.gelu_coeff=20, R.normalize_r=off, R.gamma_r=30, R.wffn1_r=off, R.gp0=off, K=**13** |
| L09 | B1 | F.gelu_out=30, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=20, R.mean_r=34, R.var_r=34, K=**13** |
| L09 | B2 | F.inv_std_fresh=23, F.x_centered_fresh=22, M.gamma=18, W.wk=14, W.wv=22, M.kt_mask1=13, M.kt_mask2=13, M.qkt_merge_mask=11, R.gamma_r=off, R.kt_mask2_r=31, R.qkt_merge_mask_r=28, K=**10** |
| L09 | B3 | F.x_fresh=28, S.inv_2n=15, R.sq0=31, R.sq1=31, R.sq2=31, R.sq3=31, K=**13** |
| L09 | B4 | F.softmax_out_fresh=27, F.v_fresh=22, M.softmax_out_mask=12, M.v_mask=10, M.softmax_v_mask=10, S.ln_mean_inv_d=18, S.ln_var_inv_d=18, W.wo=22, R.softmax_v_matmul_r=off, R.ln_mean_r=31, R.ln_var_r=24, K=**10** |
| L09 | B5 | F.inv_std_fresh=30, F.x_centered_fresh=30, M.gamma=20, W.wffn1=22, M.gelu_coeff=20, R.normalize_r=off, R.gamma_r=30, R.wffn1_r=off, R.gp0=off, K=**13** |
| L10 | B1 | F.gelu_out=30, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=20, R.mean_r=34, R.var_r=34, K=**13** |
| L10 | B2 | F.inv_std_fresh=23, F.x_centered_fresh=22, M.gamma=18, W.wk=14, W.wv=22, M.kt_mask1=13, M.kt_mask2=13, M.qkt_merge_mask=11, R.gamma_r=off, R.kt_mask2_r=31, R.qkt_merge_mask_r=24, K=**10** |
| L10 | B3 | F.x_fresh=20, S.inv_2n=11, R.sq0=31, R.sq1=off, R.sq2=31, R.sq3=27, K=**12** |
| L10 | B4 | F.softmax_out_fresh=27, F.v_fresh=22, M.softmax_out_mask=12, M.v_mask=10, M.softmax_v_mask=10, S.ln_mean_inv_d=18, S.ln_var_inv_d=18, W.wo=22, R.softmax_v_matmul_r=off, R.ln_mean_r=29, R.ln_var_r=24, K=**10** |
| L10 | B5 | F.inv_std_fresh=30, F.x_centered_fresh=30, M.gamma=20, W.wffn1=22, M.gelu_coeff=20, R.normalize_r=off, R.gamma_r=30, R.wffn1_r=off, R.gp0=off, K=**13** |
| L11 | B1 | F.gelu_out=30, W.wffn2=20, S.mean_inv_d=20, S.var_inv_d=20, R.mean_r=34, R.var_r=34, K=**13** |
| L11 | B2 | F.inv_std_fresh=23, F.x_centered_fresh=22, M.gamma=18, W.wk=14, W.wv=22, M.kt_mask1=13, M.kt_mask2=13, M.qkt_merge_mask=11, R.gamma_r=off, R.kt_mask2_r=31, R.qkt_merge_mask_r=24, K=**10** |
| L11 | B3 | F.x_fresh=27, S.inv_2n=16, R.sq0=34, R.sq1=34, R.sq2=off, R.sq3=off, K=**13** |
| L11 | B4 | F.softmax_out_fresh=27, F.v_fresh=22, M.softmax_out_mask=12, M.v_mask=10, M.softmax_v_mask=10, S.ln_mean_inv_d=18, S.ln_var_inv_d=18, W.wo=22, R.softmax_v_matmul_r=off, R.ln_mean_r=31, R.ln_var_r=24, K=**10** |
| L11 | B5 | F.inv_std_fresh=30, F.x_centered_fresh=30, M.gamma=20, W.wffn1=22, M.gelu_coeff=20, R.normalize_r=off, R.gamma_r=30, R.wffn1_r=off, R.gp0=off, K=**13** |
| (legacy) first_input | – | `scaling_factor=22` |

### 5.2 Best vs baseline · 哪些槽位变了

_共 244 个槽位发生变化（238 个 SF + 6 个 K bits）_

**截断 K bits 变化**：

| 槽位 | baseline K | best K | Δ |
|---|---:|---:|---:|
| `L0.B1.K` | 13 | 8 | -5 |
| `L0.B3.K` | 13 | 12 | -1 |
| `L1.B3.K` | 13 | 12 | -1 |
| `L10.B3.K` | 13 | 12 | -1 |
| `L2.B3.K` | 13 | 12 | -1 |
| `L5.B5.K` | 13 | 12 | -1 |

**Scaling factor 变化**（前 25 条按 |Δ| 降序）：

| 槽位 | kind | baseline SF | best SF | Δ |
|---|:---:|---:|---:|---:|
| `L0.B1.F.gelu_out` | `F` | 30 | 22 | -8 |
| `L0.B1.W.wffn2` | `W` | 20 | 12 | -8 |
| `L0.B2.F.inv_std_fresh` | `F` | 31 | 23 | -8 |
| `L0.B2.F.x_centered_fresh` | `F` | 30 | 22 | -8 |
| `L0.B2.W.wk` | `W` | 22 | 14 | -8 |
| `L0.B3.F.x_fresh` | `F` | 27 | 19 | -8 |
| `L0.B4.F.softmax_out_fresh` | `F` | 35 | 27 | -8 |
| `L0.B4.F.v_fresh` | `F` | 30 | 22 | -8 |
| `L1.B2.F.inv_std_fresh` | `F` | 31 | 23 | -8 |
| `L1.B2.F.x_centered_fresh` | `F` | 30 | 22 | -8 |
| `L1.B2.W.wk` | `W` | 22 | 14 | -8 |
| `L1.B3.F.x_fresh` | `F` | 27 | 19 | -8 |
| `L1.B4.F.softmax_out_fresh` | `F` | 35 | 27 | -8 |
| `L1.B4.F.v_fresh` | `F` | 30 | 22 | -8 |
| `L2.B2.F.inv_std_fresh` | `F` | 31 | 23 | -8 |
| `L2.B2.F.x_centered_fresh` | `F` | 30 | 22 | -8 |
| `L2.B2.W.wk` | `W` | 22 | 14 | -8 |
| `L2.B3.F.x_fresh` | `F` | 28 | 20 | -8 |
| `L2.B4.F.softmax_out_fresh` | `F` | 35 | 27 | -8 |
| `L2.B4.F.v_fresh` | `F` | 30 | 22 | -8 |
| `L3.B2.F.inv_std_fresh` | `F` | 31 | 23 | -8 |
| `L3.B2.F.x_centered_fresh` | `F` | 30 | 22 | -8 |
| `L3.B2.W.wk` | `W` | 22 | 14 | -8 |
| `L3.B4.F.softmax_out_fresh` | `F` | 35 | 27 | -8 |
| `L3.B4.F.v_fresh` | `F` | 30 | 22 | -8 |

<details>
<summary>调试用：原始 action_vec（整数索引）</summary>

- 长度: 577

```
0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 1, 1, 0, 0, 3, 3, 4, 0, 0, 1, 0, 3, 1, 5, 0, 0, 1, 0, 1, 1, 1, 4, 0, 3, 3, 4, 4, 4, 2, 4, 2, 0, 3, 0, 0, 3, 4, 4, 2, 2, 3, 3, 3, 0, 0, 1, 0, 4, 1, 1, 0, 0, 3, 3, 4, 0, 0, 1, 0, 3, 1, 5, 0, 0, 1, 0, 1, 1, 1, 4, 0, 3, 1, 4, 4, 4, 2, 4, 2, 0, 3, 0, 0, 3, 4, 4, 2, 2, 3, 3, 3, 0, 0, 1, 0, 4, 1, 1, 0, 0, 3, 3, 4, 0, 0, 3, 0, 1, 1, 5, 0, 0, 1, 0, 0, 1, 1, 4, 0, 3, 3, 4, 4, 4, 2, 4, 2, 0, 3, 0, 0, 3, 4, 4, 2, 2, 3, 3, 3, 0, 0, 1, 0, 4, 1, 1, 0, 0, 3, 3, 4, 4, 2, 3, 3, 3, 3, 3, 0, 0, 1, 0, 1, 1, 1, 4, 0, 3, 3, 4, 4, 4, 2, 4, 2, 0, 3, 0, 0, 3, 4, 4, 2, 2, 3, 3, 3, 0, 0, 1, 0, 4, 1, 1, 0, 0, 3, 3, 4, 4, 2, 3, 3, 3, 3, 3, 0, 0, 1, 0, 0, 1, 1, 4, 0, 3, 1, 4, 4, 4, 2, 4, 2, 0, 3, 0, 0, 3, 4, 4, 2, 2, 3, 3, 3, 0, 0, 1, 0, 4, 1, 1, 0, 0, 3, 1, 4, 4, 2, 3, 3, 0, 0, 3, 0, 0, 1, 0, 0, 1, 1, 4, 0, 3, 3, 4, 0, 0, 1, 0, 0, 1, 1, 3, 0, 5, 4, 4, 2, 2, 3, 3, 3, 0, 0, 1, 0, 4, 1, 1, 0, 0, 3, 1, 4, 4, 2, 3, 3, 3, 3, 3, 0, 0, 1, 0, 0, 1, 1, 4, 0, 3, 3, 4, 4, 4, 2, 4, 2, 0, 3, 0, 0, 3, 4, 4, 2, 2, 3, 3, 3, 0, 0, 2, 0, 4, 1, 1, 0, 0, 3, 1, 4, 4, 2, 3, 3, 0, 0, 3, 0, 0, 1, 0, 0, 1, 1, 4, 0, 3, 3, 4, 4, 4, 2, 4, 2, 0, 3, 0, 0, 3, 4, 4, 2, 2, 3, 3, 3, 0, 0, 1, 0, 4, 1, 1, 0, 0, 2, 3, 4, 4, 2, 3, 3, 3, 3, 3, 0, 0, 1, 0, 0, 1, 1, 4, 0, 3, 1, 4, 4, 4, 2, 4, 2, 0, 3, 0, 0, 3, 4, 4, 2, 2, 3, 3, 3, 0, 0, 1, 0, 4, 1, 1, 0, 0, 3, 3, 4, 4, 2, 3, 3, 3, 3, 3, 0, 0, 1, 0, 0, 1, 1, 4, 0, 3, 1, 4, 4, 4, 2, 4, 2, 0, 3, 0, 0, 3, 4, 4, 2, 2, 3, 3, 3, 0, 0, 1, 0, 4, 1, 1, 0, 0, 3, 1, 4, 0, 0, 3, 0, 3, 1, 5, 0, 0, 1, 0, 0, 1, 1, 4, 0, 2, 1, 4, 4, 4, 2, 4, 2, 0, 3, 0, 0, 3, 4, 4, 2, 2, 3, 3, 3, 0, 0, 1, 0, 4, 1, 1, 0, 0, 3, 1, 4, 4, 2, 3, 3, 0, 0, 3, 0, 0, 1, 0, 0, 1, 1, 4, 0, 3, 1, 4, 4, 4, 2, 4, 2, 0, 3, 0, 0, 3, 0
```

</details>

---

> 持久化目录：`Parting Chapter/<run>/stage2_noise/progress/`。live checkpoint / final checkpoint / best_cfg.pkl / 状态板 / 训练曲线（PNG + NPZ）/ 本报告 都在该目录下。