# BLB Stage 2 action description: `best`

- profile: `mrpc`
- num_layers: `12`
- action_length: `577`
- records: `577`
- scaling factor slots: `517`
- truncation slots: `60`
- ineffective decoded slots: `29`

**Slot label format**: `L{layer}.B{block}.{kind}[.{short_field}]` 
(kind: F=fresh, W=weight encode, M=mask, S=scalar, R=rescale, K=trunc).

Each row's primary value is the decoded **`scaling_factor`** (for F/W/M/S/R kinds) or **`truncation_bits`** (for K). `action_idx` is the policy-side index that produced that value and is included only for sanity-checking — humans should read the SF / truncation_bits columns.

## 1. Per-layer / per-block 选择概览

| 层 | block | 槽位 → 选择值 |
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

## 2. 全槽位明细（按 global_index）

| idx | slot | location | operation | dist | **value** | kind | action_idx | effective | N | max_sf | level_values | note |
|---:|---|---|---|:---:|---:|:---:|---:|:---:|---:|---:|---|---|
| 0 | `L0.B1.F.gelu_out` | `layer0.block1.gelu_out_sf` | `ctpt_gelu_out` | `fresh` | _off_ | `F` | 0 | False | 8192 | 30 | `22,24,26,28,30` | layer 0 has no upstream FFN2; the first HE config is treated as lossless so block1 noise is not installed (aligned with Rescale_optimizer) |
| 1 | `L0.B1.W.wffn2` | `layer0.block1.wffn2_sf` | `ctpt_ffn2` | `encoding` | _off_ | `W` | 0 | False | 8192 | 20 | `12,14,16,18,20` | layer 0 has no upstream FFN2; the first HE config is treated as lossless so block1 noise is not installed (aligned with Rescale_optimizer) |
| 2 | `L0.B1.S.mean_inv_d` | `layer0.block1.mean_inv_d_sf` | `ctpt_inv_d_1` | `scalar` | _off_ | `S` | 0 | False | 8192 | 20 | `16,18,20` | layer 0 has no upstream FFN2; the first HE config is treated as lossless so block1 noise is not installed (aligned with Rescale_optimizer) |
| 3 | `L0.B1.S.var_inv_d` | `layer0.block1.var_inv_d_sf` | `ctpt_inv_d_2` | `scalar` | _off_ | `S` | 0 | False | 8192 | 20 | `16,18,20` | layer 0 has no upstream FFN2; the first HE config is treated as lossless so block1 noise is not installed (aligned with Rescale_optimizer) |
| 4 | `L0.B1.R.mean_r` | `layer0.block1.mean_rescale_sf` | `ctct_mean_rescale` | `rescale` | _off_ | `R` | 0 | False | 8192 | 34 | `None,30,32,34` | layer 0 has no upstream FFN2; the first HE config is treated as lossless so block1 noise is not installed (aligned with Rescale_optimizer) |
| 5 | `L0.B1.R.var_r` | `layer0.block1.var_rescale_sf` | `ctct_var_rescale` | `rescale` | _off_ | `R` | 0 | False | 8192 | 34 | `None,30,32,34` | layer 0 has no upstream FFN2; the first HE config is treated as lossless so block1 noise is not installed (aligned with Rescale_optimizer) |
| 6 | `L0.B1.K` | `layer0.block1.output_truncation_k` | `block1_output_truncation` | `truncation` | _off_ | `K` | 0 | False | 8192 |  | `8,9,11,13,10,12` | layer 0 has no upstream FFN2; the first HE config is treated as lossless so block1 noise is not installed (aligned with Rescale_optimizer) |
| 7 | `L0.B2.F.inv_std_fresh` | `layer0.block2.inv_std_fresh_sf` | `ctpt_inv_std` | `fresh` | **23** | `F` | 0 | True | 16384 | 31 | `23,25,27,29,31` |  |
| 8 | `L0.B2.F.x_centered_fresh` | `layer0.block2.x_centered_fresh_sf` | `ctpt_x_centered` | `fresh` | **22** | `F` | 0 | True | 16384 | 30 | `22,24,26,28,30` |  |
| 9 | `L0.B2.M.gamma` | `layer0.block2.gamma_sf` | `ctpt_gamma` | `encoding` | **18** | `M` | 1 | True | 16384 | 20 | `16,18,20` |  |
| 10 | `L0.B2.W.wk` | `layer0.block2.wk_sf` | `ctpt_wq_wk` | `encoding` | **14** | `W` | 0 | True | 16384 | 22 | `14,16,18,20,22` |  |
| 11 | `L0.B2.W.wv` | `layer0.block2.wv_sf` | `ctpt_wv` | `encoding` | **22** | `W` | 4 | True | 16384 | 22 | `14,16,18,20,22` |  |
| 12 | `L0.B2.M.kt_mask1` | `layer0.block2.kt_mask1_sf` | `ctpt_rotKT_mask1` | `encoding` | **13** | `M` | 1 | True | 16384 | 15 | `11,13,15` |  |
| 13 | `L0.B2.M.kt_mask2` | `layer0.block2.kt_mask2_sf` | `ctpt_rotKT_mask2` | `encoding` | **13** | `M` | 1 | True | 16384 | 15 | `11,13,15` |  |
| 14 | `L0.B2.M.qkt_merge_mask` | `layer0.block2.qkt_merge_mask_sf` | `ctpt_mask` | `encoding` | **11** | `M` | 0 | True | 16384 | 15 | `11,13,15` |  |
| 15 | `L0.B2.R.gamma_r` | `layer0.block2.gamma_rescale_sf` | `ctct_gamma_rescale` | `rescale` |  | `R` | 0 | True | 16384 | 31 | `None,27,29,31` |  |
| 16 | `L0.B2.R.kt_mask2_r` | `layer0.block2.kt_mask2_rescale_sf` | `ctct_kt_mask2_rescale` | `rescale` | **31** | `R` | 3 | True | 16384 | 31 | `None,27,29,31` |  |
| 17 | `L0.B2.R.qkt_merge_mask_r` | `layer0.block2.qkt_merge_mask_rescale_sf` | `ctct_qkt_merge_mask_rescale` | `rescale` | **28** | `R` | 3 | True | 16384 | 28 | `None,24,26,28` |  |
| 18 | `L0.B2.K` | `layer0.block2.output_truncation_k` | `block2_output_truncation` | `truncation` | **10** | `K` | 4 | True | 16384 |  | `8,9,11,13,10,12` |  |
| 19 | `L0.B3.F.x_fresh` | `layer0.block3.x_fresh_sf` | `ctpt_softmax_x` | `fresh` | **19** | `F` | 0 | True | 8192 | 27 | `19,21,23,25,27` |  |
| 20 | `L0.B3.S.inv_2n` | `layer0.block3.inv_2n_sf` | `ctpt_softmax_inv_2n` | `scalar` | **12** | `S` | 0 | True | 8192 | 16 | `12,14,16` |  |
| 21 | `L0.B3.R.sq0` | `layer0.block3.square_rescale_sf_0` | `ctct_softmax_pow_s1` | `rescale` | **30** | `R` | 1 | True | 8192 | 34 | `None,30,32,34` |  |
| 22 | `L0.B3.R.sq1` | `layer0.block3.square_rescale_sf_1` | `ctct_softmax_pow_s2` | `rescale` |  | `R` | 0 | True | 8192 | 34 | `None,30,32,34` |  |
| 23 | `L0.B3.R.sq2` | `layer0.block3.square_rescale_sf_2` | `ctct_softmax_pow_s3` | `rescale` | _off_ | `R` | 3 | False | 8192 | 31 | `None,27,29,31` | softmax degree 2 does not use this square-rescale slot |
| 24 | `L0.B3.R.sq3` | `layer0.block3.square_rescale_sf_3` | `ctct_softmax_pow_s4` | `rescale` | _off_ | `R` | 1 | False | 8192 | 31 | `None,27,29,31` | softmax degree 2 does not use this square-rescale slot |
| 25 | `L0.B3.K` | `layer0.block3.output_truncation_k` | `block3_output_truncation` | `truncation` | **12** | `K` | 5 | True | 8192 |  | `8,9,11,13,10,12` |  |
| 26 | `L0.B4.F.softmax_out_fresh` | `layer0.block4.softmax_out_fresh_sf` | `ctpt_softmax_out` | `fresh` | **27** | `F` | 0 | True | 16384 | 35 | `27,29,31,33,35` |  |
| 27 | `L0.B4.F.v_fresh` | `layer0.block4.v_fresh_sf` | `ctpt_v` | `fresh` | **22** | `F` | 0 | True | 16384 | 30 | `22,24,26,28,30` |  |
| 28 | `L0.B4.M.softmax_out_mask` | `layer0.block4.softmax_out_mask_sf` | `ctpt_softmax_out_mask` | `encoding` | **12** | `M` | 1 | True | 16384 | 14 | `10,12,14` |  |
| 29 | `L0.B4.M.v_mask` | `layer0.block4.v_mask_sf` | `ctpt_v_mask` | `encoding` | **10** | `M` | 0 | True | 16384 | 14 | `10,12,14` |  |
| 30 | `L0.B4.M.softmax_v_mask` | `layer0.block4.softmax_v_mask_sf` | `ctpt_softmax_v_mask` | `encoding` | **12** | `M` | 1 | True | 16384 | 14 | `10,12,14` |  |
| 31 | `L0.B4.S.ln_mean_inv_d` | `layer0.block4.ln_mean_inv_d_sf` | `ctpt_inv_d_attn_mean` | `scalar` | **18** | `S` | 1 | True | 16384 | 20 | `16,18,20` |  |
| 32 | `L0.B4.S.ln_var_inv_d` | `layer0.block4.ln_var_inv_d_sf` | `ctpt_inv_d_attn_var` | `scalar` | **18** | `S` | 1 | True | 16384 | 20 | `16,18,20` |  |
| 33 | `L0.B4.W.wo` | `layer0.block4.wo_sf` | `ctpt_wo` | `encoding` | **22** | `W` | 4 | True | 16384 | 22 | `14,16,18,20,22` |  |
| 34 | `L0.B4.R.softmax_v_matmul_r` | `layer0.block4.softmax_v_matmul_rescale_sf` | `ctct_softmax_v_matmul_rescale` | `rescale` |  | `R` | 0 | True | 16384 | 31 | `None,27,29,31` |  |
| 35 | `L0.B4.R.ln_mean_r` | `layer0.block4.ln_mean_rescale_sf` | `ctct_attn_mean_rescale` | `rescale` | **31** | `R` | 3 | True | 16384 | 31 | `None,27,29,31` |  |
| 36 | `L0.B4.R.ln_var_r` | `layer0.block4.ln_var_rescale_sf` | `ctct_attn_var_rescale` | `rescale` | **28** | `R` | 3 | True | 16384 | 28 | `None,24,26,28` |  |
| 37 | `L0.B4.K` | `layer0.block4.output_truncation_k` | `block4_output_truncation` | `truncation` | **10** | `K` | 4 | True | 16384 |  | `8,9,11,13,10,12` |  |
| 38 | `L0.B5.F.inv_std_fresh` | `layer0.block5.inv_std_fresh_sf` | `ctpt_attn_inv_std` | `fresh` | **30** | `F` | 4 | True | 8192 | 30 | `22,24,26,28,30` |  |
| 39 | `L0.B5.F.x_centered_fresh` | `layer0.block5.x_centered_fresh_sf` | `ctpt_attn_x_centered` | `fresh` | **30** | `F` | 4 | True | 8192 | 30 | `22,24,26,28,30` |  |
| 40 | `L0.B5.M.gamma` | `layer0.block5.gamma_sf` | `ctpt_gamma_attn` | `encoding` | **20** | `M` | 2 | True | 8192 | 20 | `16,18,20` |  |
| 41 | `L0.B5.W.wffn1` | `layer0.block5.wffn1_sf` | `ctpt_wffn1` | `encoding` | **22** | `W` | 4 | True | 8192 | 22 | `14,16,18,20,22` |  |
| 42 | `L0.B5.M.gelu_coeff` | `layer0.block5.gelu_coeff_sf` | `ctpt_gelu_coeff` | `encoding` | **20** | `M` | 2 | True | 8192 | 20 | `16,18,20` |  |
| 43 | `L0.B5.R.normalize_r` | `layer0.block5.normalize_rescale_sf` | `ctct_attn_normalize_rescale` | `rescale` |  | `R` | 0 | True | 8192 | 31 | `None,27,29,31` |  |
| 44 | `L0.B5.R.gamma_r` | `layer0.block5.gamma_rescale_sf` | `ctct_gamma_attn_rescale` | `rescale` | **30** | `R` | 3 | True | 8192 | 30 | `None,26,28,30` |  |
| 45 | `L0.B5.R.wffn1_r` | `layer0.block5.wffn1_rescale_sf` | `ctct_wffn1_rescale` | `rescale` |  | `R` | 0 | True | 8192 | 31 | `None,27,29,31` |  |
| 46 | `L0.B5.R.gp0` | `layer0.block5.gelu_power_rescale_sf_0` | `ctct_gelu_x2` | `rescale` | _off_ | `R` | 0 | False | 8192 | 31 | `None,27,29,31` | GELU degree 1 does not use this power-rescale slot |
| 47 | `L0.B5.K` | `layer0.block5.output_truncation_k` | `block5_output_truncation` | `truncation` | **13** | `K` | 3 | True | 8192 |  | `8,9,11,13,10,12` |  |
| 48 | `L1.B1.F.gelu_out` | `layer1.block1.gelu_out_sf` | `ctpt_gelu_out` | `fresh` | **30** | `F` | 4 | True | 8192 | 30 | `22,24,26,28,30` |  |
| 49 | `L1.B1.W.wffn2` | `layer1.block1.wffn2_sf` | `ctpt_ffn2` | `encoding` | **20** | `W` | 4 | True | 8192 | 20 | `12,14,16,18,20` |  |
| 50 | `L1.B1.S.mean_inv_d` | `layer1.block1.mean_inv_d_sf` | `ctpt_inv_d_1` | `scalar` | **20** | `S` | 2 | True | 8192 | 20 | `16,18,20` |  |
| 51 | `L1.B1.S.var_inv_d` | `layer1.block1.var_inv_d_sf` | `ctpt_inv_d_2` | `scalar` | **20** | `S` | 2 | True | 8192 | 20 | `16,18,20` |  |
| 52 | `L1.B1.R.mean_r` | `layer1.block1.mean_rescale_sf` | `ctct_mean_rescale` | `rescale` | **34** | `R` | 3 | True | 8192 | 34 | `None,30,32,34` |  |
| 53 | `L1.B1.R.var_r` | `layer1.block1.var_rescale_sf` | `ctct_var_rescale` | `rescale` | **34** | `R` | 3 | True | 8192 | 34 | `None,30,32,34` |  |
| 54 | `L1.B1.K` | `layer1.block1.output_truncation_k` | `block1_output_truncation` | `truncation` | **13** | `K` | 3 | True | 8192 |  | `8,9,11,13,10,12` |  |
| 55 | `L1.B2.F.inv_std_fresh` | `layer1.block2.inv_std_fresh_sf` | `ctpt_inv_std` | `fresh` | **23** | `F` | 0 | True | 16384 | 31 | `23,25,27,29,31` |  |
| 56 | `L1.B2.F.x_centered_fresh` | `layer1.block2.x_centered_fresh_sf` | `ctpt_x_centered` | `fresh` | **22** | `F` | 0 | True | 16384 | 30 | `22,24,26,28,30` |  |
| 57 | `L1.B2.M.gamma` | `layer1.block2.gamma_sf` | `ctpt_gamma` | `encoding` | **18** | `M` | 1 | True | 16384 | 20 | `16,18,20` |  |
| 58 | `L1.B2.W.wk` | `layer1.block2.wk_sf` | `ctpt_wq_wk` | `encoding` | **14** | `W` | 0 | True | 16384 | 22 | `14,16,18,20,22` |  |
| 59 | `L1.B2.W.wv` | `layer1.block2.wv_sf` | `ctpt_wv` | `encoding` | **22** | `W` | 4 | True | 16384 | 22 | `14,16,18,20,22` |  |
| 60 | `L1.B2.M.kt_mask1` | `layer1.block2.kt_mask1_sf` | `ctpt_rotKT_mask1` | `encoding` | **13** | `M` | 1 | True | 16384 | 15 | `11,13,15` |  |
| 61 | `L1.B2.M.kt_mask2` | `layer1.block2.kt_mask2_sf` | `ctpt_rotKT_mask2` | `encoding` | **13** | `M` | 1 | True | 16384 | 15 | `11,13,15` |  |
| 62 | `L1.B2.M.qkt_merge_mask` | `layer1.block2.qkt_merge_mask_sf` | `ctpt_mask` | `encoding` | **11** | `M` | 0 | True | 16384 | 15 | `11,13,15` |  |
| 63 | `L1.B2.R.gamma_r` | `layer1.block2.gamma_rescale_sf` | `ctct_gamma_rescale` | `rescale` |  | `R` | 0 | True | 16384 | 31 | `None,27,29,31` |  |
| 64 | `L1.B2.R.kt_mask2_r` | `layer1.block2.kt_mask2_rescale_sf` | `ctct_kt_mask2_rescale` | `rescale` | **31** | `R` | 3 | True | 16384 | 31 | `None,27,29,31` |  |
| 65 | `L1.B2.R.qkt_merge_mask_r` | `layer1.block2.qkt_merge_mask_rescale_sf` | `ctct_qkt_merge_mask_rescale` | `rescale` | **28** | `R` | 3 | True | 16384 | 28 | `None,24,26,28` |  |
| 66 | `L1.B2.K` | `layer1.block2.output_truncation_k` | `block2_output_truncation` | `truncation` | **10** | `K` | 4 | True | 16384 |  | `8,9,11,13,10,12` |  |
| 67 | `L1.B3.F.x_fresh` | `layer1.block3.x_fresh_sf` | `ctpt_softmax_x` | `fresh` | **19** | `F` | 0 | True | 8192 | 27 | `19,21,23,25,27` |  |
| 68 | `L1.B3.S.inv_2n` | `layer1.block3.inv_2n_sf` | `ctpt_softmax_inv_2n` | `scalar` | **12** | `S` | 0 | True | 8192 | 16 | `12,14,16` |  |
| 69 | `L1.B3.R.sq0` | `layer1.block3.square_rescale_sf_0` | `ctct_softmax_pow_s1` | `rescale` | **30** | `R` | 1 | True | 8192 | 34 | `None,30,32,34` |  |
| 70 | `L1.B3.R.sq1` | `layer1.block3.square_rescale_sf_1` | `ctct_softmax_pow_s2` | `rescale` |  | `R` | 0 | True | 8192 | 34 | `None,30,32,34` |  |
| 71 | `L1.B3.R.sq2` | `layer1.block3.square_rescale_sf_2` | `ctct_softmax_pow_s3` | `rescale` | _off_ | `R` | 3 | False | 8192 | 31 | `None,27,29,31` | softmax degree 2 does not use this square-rescale slot |
| 72 | `L1.B3.R.sq3` | `layer1.block3.square_rescale_sf_3` | `ctct_softmax_pow_s4` | `rescale` | _off_ | `R` | 1 | False | 8192 | 31 | `None,27,29,31` | softmax degree 2 does not use this square-rescale slot |
| 73 | `L1.B3.K` | `layer1.block3.output_truncation_k` | `block3_output_truncation` | `truncation` | **12** | `K` | 5 | True | 8192 |  | `8,9,11,13,10,12` |  |
| 74 | `L1.B4.F.softmax_out_fresh` | `layer1.block4.softmax_out_fresh_sf` | `ctpt_softmax_out` | `fresh` | **27** | `F` | 0 | True | 16384 | 35 | `27,29,31,33,35` |  |
| 75 | `L1.B4.F.v_fresh` | `layer1.block4.v_fresh_sf` | `ctpt_v` | `fresh` | **22** | `F` | 0 | True | 16384 | 30 | `22,24,26,28,30` |  |
| 76 | `L1.B4.M.softmax_out_mask` | `layer1.block4.softmax_out_mask_sf` | `ctpt_softmax_out_mask` | `encoding` | **12** | `M` | 1 | True | 16384 | 14 | `10,12,14` |  |
| 77 | `L1.B4.M.v_mask` | `layer1.block4.v_mask_sf` | `ctpt_v_mask` | `encoding` | **10** | `M` | 0 | True | 16384 | 14 | `10,12,14` |  |
| 78 | `L1.B4.M.softmax_v_mask` | `layer1.block4.softmax_v_mask_sf` | `ctpt_softmax_v_mask` | `encoding` | **12** | `M` | 1 | True | 16384 | 14 | `10,12,14` |  |
| 79 | `L1.B4.S.ln_mean_inv_d` | `layer1.block4.ln_mean_inv_d_sf` | `ctpt_inv_d_attn_mean` | `scalar` | **18** | `S` | 1 | True | 16384 | 20 | `16,18,20` |  |
| 80 | `L1.B4.S.ln_var_inv_d` | `layer1.block4.ln_var_inv_d_sf` | `ctpt_inv_d_attn_var` | `scalar` | **18** | `S` | 1 | True | 16384 | 20 | `16,18,20` |  |
| 81 | `L1.B4.W.wo` | `layer1.block4.wo_sf` | `ctpt_wo` | `encoding` | **22** | `W` | 4 | True | 16384 | 22 | `14,16,18,20,22` |  |
| 82 | `L1.B4.R.softmax_v_matmul_r` | `layer1.block4.softmax_v_matmul_rescale_sf` | `ctct_softmax_v_matmul_rescale` | `rescale` |  | `R` | 0 | True | 16384 | 31 | `None,27,29,31` |  |
| 83 | `L1.B4.R.ln_mean_r` | `layer1.block4.ln_mean_rescale_sf` | `ctct_attn_mean_rescale` | `rescale` | **31** | `R` | 3 | True | 16384 | 31 | `None,27,29,31` |  |
| 84 | `L1.B4.R.ln_var_r` | `layer1.block4.ln_var_rescale_sf` | `ctct_attn_var_rescale` | `rescale` | **24** | `R` | 1 | True | 16384 | 28 | `None,24,26,28` |  |
| 85 | `L1.B4.K` | `layer1.block4.output_truncation_k` | `block4_output_truncation` | `truncation` | **10** | `K` | 4 | True | 16384 |  | `8,9,11,13,10,12` |  |
| 86 | `L1.B5.F.inv_std_fresh` | `layer1.block5.inv_std_fresh_sf` | `ctpt_attn_inv_std` | `fresh` | **30** | `F` | 4 | True | 8192 | 30 | `22,24,26,28,30` |  |
| 87 | `L1.B5.F.x_centered_fresh` | `layer1.block5.x_centered_fresh_sf` | `ctpt_attn_x_centered` | `fresh` | **30** | `F` | 4 | True | 8192 | 30 | `22,24,26,28,30` |  |
| 88 | `L1.B5.M.gamma` | `layer1.block5.gamma_sf` | `ctpt_gamma_attn` | `encoding` | **20** | `M` | 2 | True | 8192 | 20 | `16,18,20` |  |
| 89 | `L1.B5.W.wffn1` | `layer1.block5.wffn1_sf` | `ctpt_wffn1` | `encoding` | **22** | `W` | 4 | True | 8192 | 22 | `14,16,18,20,22` |  |
| 90 | `L1.B5.M.gelu_coeff` | `layer1.block5.gelu_coeff_sf` | `ctpt_gelu_coeff` | `encoding` | **20** | `M` | 2 | True | 8192 | 20 | `16,18,20` |  |
| 91 | `L1.B5.R.normalize_r` | `layer1.block5.normalize_rescale_sf` | `ctct_attn_normalize_rescale` | `rescale` |  | `R` | 0 | True | 8192 | 31 | `None,27,29,31` |  |
| 92 | `L1.B5.R.gamma_r` | `layer1.block5.gamma_rescale_sf` | `ctct_gamma_attn_rescale` | `rescale` | **30** | `R` | 3 | True | 8192 | 30 | `None,26,28,30` |  |
| 93 | `L1.B5.R.wffn1_r` | `layer1.block5.wffn1_rescale_sf` | `ctct_wffn1_rescale` | `rescale` |  | `R` | 0 | True | 8192 | 31 | `None,27,29,31` |  |
| 94 | `L1.B5.R.gp0` | `layer1.block5.gelu_power_rescale_sf_0` | `ctct_gelu_x2` | `rescale` | _off_ | `R` | 0 | False | 8192 | 31 | `None,27,29,31` | GELU degree 1 does not use this power-rescale slot |
| 95 | `L1.B5.K` | `layer1.block5.output_truncation_k` | `block5_output_truncation` | `truncation` | **13** | `K` | 3 | True | 8192 |  | `8,9,11,13,10,12` |  |
| 96 | `L2.B1.F.gelu_out` | `layer2.block1.gelu_out_sf` | `ctpt_gelu_out` | `fresh` | **30** | `F` | 4 | True | 8192 | 30 | `22,24,26,28,30` |  |
| 97 | `L2.B1.W.wffn2` | `layer2.block1.wffn2_sf` | `ctpt_ffn2` | `encoding` | **20** | `W` | 4 | True | 8192 | 20 | `12,14,16,18,20` |  |
| 98 | `L2.B1.S.mean_inv_d` | `layer2.block1.mean_inv_d_sf` | `ctpt_inv_d_1` | `scalar` | **20** | `S` | 2 | True | 8192 | 20 | `16,18,20` |  |
| 99 | `L2.B1.S.var_inv_d` | `layer2.block1.var_inv_d_sf` | `ctpt_inv_d_2` | `scalar` | **20** | `S` | 2 | True | 8192 | 20 | `16,18,20` |  |
| 100 | `L2.B1.R.mean_r` | `layer2.block1.mean_rescale_sf` | `ctct_mean_rescale` | `rescale` | **34** | `R` | 3 | True | 8192 | 34 | `None,30,32,34` |  |
| 101 | `L2.B1.R.var_r` | `layer2.block1.var_rescale_sf` | `ctct_var_rescale` | `rescale` | **34** | `R` | 3 | True | 8192 | 34 | `None,30,32,34` |  |
| 102 | `L2.B1.K` | `layer2.block1.output_truncation_k` | `block1_output_truncation` | `truncation` | **13** | `K` | 3 | True | 8192 |  | `8,9,11,13,10,12` |  |
| 103 | `L2.B2.F.inv_std_fresh` | `layer2.block2.inv_std_fresh_sf` | `ctpt_inv_std` | `fresh` | **23** | `F` | 0 | True | 16384 | 31 | `23,25,27,29,31` |  |
| 104 | `L2.B2.F.x_centered_fresh` | `layer2.block2.x_centered_fresh_sf` | `ctpt_x_centered` | `fresh` | **22** | `F` | 0 | True | 16384 | 30 | `22,24,26,28,30` |  |
| 105 | `L2.B2.M.gamma` | `layer2.block2.gamma_sf` | `ctpt_gamma` | `encoding` | **18** | `M` | 1 | True | 16384 | 20 | `16,18,20` |  |
| 106 | `L2.B2.W.wk` | `layer2.block2.wk_sf` | `ctpt_wq_wk` | `encoding` | **14** | `W` | 0 | True | 16384 | 22 | `14,16,18,20,22` |  |
| 107 | `L2.B2.W.wv` | `layer2.block2.wv_sf` | `ctpt_wv` | `encoding` | **22** | `W` | 4 | True | 16384 | 22 | `14,16,18,20,22` |  |
| 108 | `L2.B2.M.kt_mask1` | `layer2.block2.kt_mask1_sf` | `ctpt_rotKT_mask1` | `encoding` | **13** | `M` | 1 | True | 16384 | 15 | `11,13,15` |  |
| 109 | `L2.B2.M.kt_mask2` | `layer2.block2.kt_mask2_sf` | `ctpt_rotKT_mask2` | `encoding` | **13** | `M` | 1 | True | 16384 | 15 | `11,13,15` |  |
| 110 | `L2.B2.M.qkt_merge_mask` | `layer2.block2.qkt_merge_mask_sf` | `ctpt_mask` | `encoding` | **11** | `M` | 0 | True | 16384 | 15 | `11,13,15` |  |
| 111 | `L2.B2.R.gamma_r` | `layer2.block2.gamma_rescale_sf` | `ctct_gamma_rescale` | `rescale` |  | `R` | 0 | True | 16384 | 31 | `None,27,29,31` |  |
| 112 | `L2.B2.R.kt_mask2_r` | `layer2.block2.kt_mask2_rescale_sf` | `ctct_kt_mask2_rescale` | `rescale` | **31** | `R` | 3 | True | 16384 | 31 | `None,27,29,31` |  |
| 113 | `L2.B2.R.qkt_merge_mask_r` | `layer2.block2.qkt_merge_mask_rescale_sf` | `ctct_qkt_merge_mask_rescale` | `rescale` | **28** | `R` | 3 | True | 16384 | 28 | `None,24,26,28` |  |
| 114 | `L2.B2.K` | `layer2.block2.output_truncation_k` | `block2_output_truncation` | `truncation` | **10** | `K` | 4 | True | 16384 |  | `8,9,11,13,10,12` |  |
| 115 | `L2.B3.F.x_fresh` | `layer2.block3.x_fresh_sf` | `ctpt_softmax_x` | `fresh` | **20** | `F` | 0 | True | 16384 | 28 | `20,22,24,26,28` |  |
| 116 | `L2.B3.S.inv_2n` | `layer2.block3.inv_2n_sf` | `ctpt_softmax_inv_2n` | `scalar` | **11** | `S` | 0 | True | 16384 | 15 | `11,13,15` |  |
| 117 | `L2.B3.R.sq0` | `layer2.block3.square_rescale_sf_0` | `ctct_softmax_pow_s1` | `rescale` | **31** | `R` | 3 | True | 16384 | 31 | `None,27,29,31` |  |
| 118 | `L2.B3.R.sq1` | `layer2.block3.square_rescale_sf_1` | `ctct_softmax_pow_s2` | `rescale` |  | `R` | 0 | True | 16384 | 31 | `None,27,29,31` |  |
| 119 | `L2.B3.R.sq2` | `layer2.block3.square_rescale_sf_2` | `ctct_softmax_pow_s3` | `rescale` | **27** | `R` | 1 | True | 16384 | 31 | `None,27,29,31` |  |
| 120 | `L2.B3.R.sq3` | `layer2.block3.square_rescale_sf_3` | `ctct_softmax_pow_s4` | `rescale` | **27** | `R` | 1 | True | 16384 | 31 | `None,27,29,31` |  |
| 121 | `L2.B3.K` | `layer2.block3.output_truncation_k` | `block3_output_truncation` | `truncation` | **12** | `K` | 5 | True | 16384 |  | `8,9,11,13,10,12` |  |
| 122 | `L2.B4.F.softmax_out_fresh` | `layer2.block4.softmax_out_fresh_sf` | `ctpt_softmax_out` | `fresh` | **27** | `F` | 0 | True | 16384 | 35 | `27,29,31,33,35` |  |
| 123 | `L2.B4.F.v_fresh` | `layer2.block4.v_fresh_sf` | `ctpt_v` | `fresh` | **22** | `F` | 0 | True | 16384 | 30 | `22,24,26,28,30` |  |
| 124 | `L2.B4.M.softmax_out_mask` | `layer2.block4.softmax_out_mask_sf` | `ctpt_softmax_out_mask` | `encoding` | **12** | `M` | 1 | True | 16384 | 14 | `10,12,14` |  |
| 125 | `L2.B4.M.v_mask` | `layer2.block4.v_mask_sf` | `ctpt_v_mask` | `encoding` | **10** | `M` | 0 | True | 16384 | 14 | `10,12,14` |  |
| 126 | `L2.B4.M.softmax_v_mask` | `layer2.block4.softmax_v_mask_sf` | `ctpt_softmax_v_mask` | `encoding` | **10** | `M` | 0 | True | 16384 | 14 | `10,12,14` |  |
| 127 | `L2.B4.S.ln_mean_inv_d` | `layer2.block4.ln_mean_inv_d_sf` | `ctpt_inv_d_attn_mean` | `scalar` | **18** | `S` | 1 | True | 16384 | 20 | `16,18,20` |  |
| 128 | `L2.B4.S.ln_var_inv_d` | `layer2.block4.ln_var_inv_d_sf` | `ctpt_inv_d_attn_var` | `scalar` | **18** | `S` | 1 | True | 16384 | 20 | `16,18,20` |  |
| 129 | `L2.B4.W.wo` | `layer2.block4.wo_sf` | `ctpt_wo` | `encoding` | **22** | `W` | 4 | True | 16384 | 22 | `14,16,18,20,22` |  |
| 130 | `L2.B4.R.softmax_v_matmul_r` | `layer2.block4.softmax_v_matmul_rescale_sf` | `ctct_softmax_v_matmul_rescale` | `rescale` |  | `R` | 0 | True | 16384 | 31 | `None,27,29,31` |  |
| 131 | `L2.B4.R.ln_mean_r` | `layer2.block4.ln_mean_rescale_sf` | `ctct_attn_mean_rescale` | `rescale` | **31** | `R` | 3 | True | 16384 | 31 | `None,27,29,31` |  |
| 132 | `L2.B4.R.ln_var_r` | `layer2.block4.ln_var_rescale_sf` | `ctct_attn_var_rescale` | `rescale` | **28** | `R` | 3 | True | 16384 | 28 | `None,24,26,28` |  |
| 133 | `L2.B4.K` | `layer2.block4.output_truncation_k` | `block4_output_truncation` | `truncation` | **10** | `K` | 4 | True | 16384 |  | `8,9,11,13,10,12` |  |
| 134 | `L2.B5.F.inv_std_fresh` | `layer2.block5.inv_std_fresh_sf` | `ctpt_attn_inv_std` | `fresh` | **30** | `F` | 4 | True | 8192 | 30 | `22,24,26,28,30` |  |
| 135 | `L2.B5.F.x_centered_fresh` | `layer2.block5.x_centered_fresh_sf` | `ctpt_attn_x_centered` | `fresh` | **30** | `F` | 4 | True | 8192 | 30 | `22,24,26,28,30` |  |
| 136 | `L2.B5.M.gamma` | `layer2.block5.gamma_sf` | `ctpt_gamma_attn` | `encoding` | **20** | `M` | 2 | True | 8192 | 20 | `16,18,20` |  |
| 137 | `L2.B5.W.wffn1` | `layer2.block5.wffn1_sf` | `ctpt_wffn1` | `encoding` | **22** | `W` | 4 | True | 8192 | 22 | `14,16,18,20,22` |  |
| 138 | `L2.B5.M.gelu_coeff` | `layer2.block5.gelu_coeff_sf` | `ctpt_gelu_coeff` | `encoding` | **20** | `M` | 2 | True | 8192 | 20 | `16,18,20` |  |
| 139 | `L2.B5.R.normalize_r` | `layer2.block5.normalize_rescale_sf` | `ctct_attn_normalize_rescale` | `rescale` |  | `R` | 0 | True | 8192 | 31 | `None,27,29,31` |  |
| 140 | `L2.B5.R.gamma_r` | `layer2.block5.gamma_rescale_sf` | `ctct_gamma_attn_rescale` | `rescale` | **30** | `R` | 3 | True | 8192 | 30 | `None,26,28,30` |  |
| 141 | `L2.B5.R.wffn1_r` | `layer2.block5.wffn1_rescale_sf` | `ctct_wffn1_rescale` | `rescale` |  | `R` | 0 | True | 8192 | 31 | `None,27,29,31` |  |
| 142 | `L2.B5.R.gp0` | `layer2.block5.gelu_power_rescale_sf_0` | `ctct_gelu_x2` | `rescale` | _off_ | `R` | 0 | False | 8192 | 31 | `None,27,29,31` | GELU degree 1 does not use this power-rescale slot |
| 143 | `L2.B5.K` | `layer2.block5.output_truncation_k` | `block5_output_truncation` | `truncation` | **13** | `K` | 3 | True | 8192 |  | `8,9,11,13,10,12` |  |
| 144 | `L3.B1.F.gelu_out` | `layer3.block1.gelu_out_sf` | `ctpt_gelu_out` | `fresh` | **30** | `F` | 4 | True | 8192 | 30 | `22,24,26,28,30` |  |
| 145 | `L3.B1.W.wffn2` | `layer3.block1.wffn2_sf` | `ctpt_ffn2` | `encoding` | **20** | `W` | 4 | True | 8192 | 20 | `12,14,16,18,20` |  |
| 146 | `L3.B1.S.mean_inv_d` | `layer3.block1.mean_inv_d_sf` | `ctpt_inv_d_1` | `scalar` | **20** | `S` | 2 | True | 8192 | 20 | `16,18,20` |  |
| 147 | `L3.B1.S.var_inv_d` | `layer3.block1.var_inv_d_sf` | `ctpt_inv_d_2` | `scalar` | **20** | `S` | 2 | True | 8192 | 20 | `16,18,20` |  |
| 148 | `L3.B1.R.mean_r` | `layer3.block1.mean_rescale_sf` | `ctct_mean_rescale` | `rescale` | **34** | `R` | 3 | True | 8192 | 34 | `None,30,32,34` |  |
| 149 | `L3.B1.R.var_r` | `layer3.block1.var_rescale_sf` | `ctct_var_rescale` | `rescale` | **34** | `R` | 3 | True | 8192 | 34 | `None,30,32,34` |  |
| 150 | `L3.B1.K` | `layer3.block1.output_truncation_k` | `block1_output_truncation` | `truncation` | **13** | `K` | 3 | True | 8192 |  | `8,9,11,13,10,12` |  |
| 151 | `L3.B2.F.inv_std_fresh` | `layer3.block2.inv_std_fresh_sf` | `ctpt_inv_std` | `fresh` | **23** | `F` | 0 | True | 16384 | 31 | `23,25,27,29,31` |  |
| 152 | `L3.B2.F.x_centered_fresh` | `layer3.block2.x_centered_fresh_sf` | `ctpt_x_centered` | `fresh` | **22** | `F` | 0 | True | 16384 | 30 | `22,24,26,28,30` |  |
| 153 | `L3.B2.M.gamma` | `layer3.block2.gamma_sf` | `ctpt_gamma` | `encoding` | **18** | `M` | 1 | True | 16384 | 20 | `16,18,20` |  |
| 154 | `L3.B2.W.wk` | `layer3.block2.wk_sf` | `ctpt_wq_wk` | `encoding` | **14** | `W` | 0 | True | 16384 | 22 | `14,16,18,20,22` |  |
| 155 | `L3.B2.W.wv` | `layer3.block2.wv_sf` | `ctpt_wv` | `encoding` | **22** | `W` | 4 | True | 16384 | 22 | `14,16,18,20,22` |  |
| 156 | `L3.B2.M.kt_mask1` | `layer3.block2.kt_mask1_sf` | `ctpt_rotKT_mask1` | `encoding` | **13** | `M` | 1 | True | 16384 | 15 | `11,13,15` |  |
| 157 | `L3.B2.M.kt_mask2` | `layer3.block2.kt_mask2_sf` | `ctpt_rotKT_mask2` | `encoding` | **13** | `M` | 1 | True | 16384 | 15 | `11,13,15` |  |
| 158 | `L3.B2.M.qkt_merge_mask` | `layer3.block2.qkt_merge_mask_sf` | `ctpt_mask` | `encoding` | **11** | `M` | 0 | True | 16384 | 15 | `11,13,15` |  |
| 159 | `L3.B2.R.gamma_r` | `layer3.block2.gamma_rescale_sf` | `ctct_gamma_rescale` | `rescale` |  | `R` | 0 | True | 16384 | 31 | `None,27,29,31` |  |
| 160 | `L3.B2.R.kt_mask2_r` | `layer3.block2.kt_mask2_rescale_sf` | `ctct_kt_mask2_rescale` | `rescale` | **31** | `R` | 3 | True | 16384 | 31 | `None,27,29,31` |  |
| 161 | `L3.B2.R.qkt_merge_mask_r` | `layer3.block2.qkt_merge_mask_rescale_sf` | `ctct_qkt_merge_mask_rescale` | `rescale` | **28** | `R` | 3 | True | 16384 | 28 | `None,24,26,28` |  |
| 162 | `L3.B2.K` | `layer3.block2.output_truncation_k` | `block2_output_truncation` | `truncation` | **10** | `K` | 4 | True | 16384 |  | `8,9,11,13,10,12` |  |
| 163 | `L3.B3.F.x_fresh` | `layer3.block3.x_fresh_sf` | `ctpt_softmax_x` | `fresh` | **28** | `F` | 4 | True | 16384 | 28 | `20,22,24,26,28` |  |
| 164 | `L3.B3.S.inv_2n` | `layer3.block3.inv_2n_sf` | `ctpt_softmax_inv_2n` | `scalar` | **15** | `S` | 2 | True | 16384 | 15 | `11,13,15` |  |
| 165 | `L3.B3.R.sq0` | `layer3.block3.square_rescale_sf_0` | `ctct_softmax_pow_s1` | `rescale` | **31** | `R` | 3 | True | 16384 | 31 | `None,27,29,31` |  |
| 166 | `L3.B3.R.sq1` | `layer3.block3.square_rescale_sf_1` | `ctct_softmax_pow_s2` | `rescale` | **31** | `R` | 3 | True | 16384 | 31 | `None,27,29,31` |  |
| 167 | `L3.B3.R.sq2` | `layer3.block3.square_rescale_sf_2` | `ctct_softmax_pow_s3` | `rescale` | **31** | `R` | 3 | True | 16384 | 31 | `None,27,29,31` |  |
| 168 | `L3.B3.R.sq3` | `layer3.block3.square_rescale_sf_3` | `ctct_softmax_pow_s4` | `rescale` | **31** | `R` | 3 | True | 16384 | 31 | `None,27,29,31` |  |
| 169 | `L3.B3.K` | `layer3.block3.output_truncation_k` | `block3_output_truncation` | `truncation` | **13** | `K` | 3 | True | 16384 |  | `8,9,11,13,10,12` |  |
| 170 | `L3.B4.F.softmax_out_fresh` | `layer3.block4.softmax_out_fresh_sf` | `ctpt_softmax_out` | `fresh` | **27** | `F` | 0 | True | 16384 | 35 | `27,29,31,33,35` |  |
| 171 | `L3.B4.F.v_fresh` | `layer3.block4.v_fresh_sf` | `ctpt_v` | `fresh` | **22** | `F` | 0 | True | 16384 | 30 | `22,24,26,28,30` |  |
| 172 | `L3.B4.M.softmax_out_mask` | `layer3.block4.softmax_out_mask_sf` | `ctpt_softmax_out_mask` | `encoding` | **12** | `M` | 1 | True | 16384 | 14 | `10,12,14` |  |
| 173 | `L3.B4.M.v_mask` | `layer3.block4.v_mask_sf` | `ctpt_v_mask` | `encoding` | **10** | `M` | 0 | True | 16384 | 14 | `10,12,14` |  |
| 174 | `L3.B4.M.softmax_v_mask` | `layer3.block4.softmax_v_mask_sf` | `ctpt_softmax_v_mask` | `encoding` | **12** | `M` | 1 | True | 16384 | 14 | `10,12,14` |  |
| 175 | `L3.B4.S.ln_mean_inv_d` | `layer3.block4.ln_mean_inv_d_sf` | `ctpt_inv_d_attn_mean` | `scalar` | **18** | `S` | 1 | True | 16384 | 20 | `16,18,20` |  |
| 176 | `L3.B4.S.ln_var_inv_d` | `layer3.block4.ln_var_inv_d_sf` | `ctpt_inv_d_attn_var` | `scalar` | **18** | `S` | 1 | True | 16384 | 20 | `16,18,20` |  |
| 177 | `L3.B4.W.wo` | `layer3.block4.wo_sf` | `ctpt_wo` | `encoding` | **22** | `W` | 4 | True | 16384 | 22 | `14,16,18,20,22` |  |
| 178 | `L3.B4.R.softmax_v_matmul_r` | `layer3.block4.softmax_v_matmul_rescale_sf` | `ctct_softmax_v_matmul_rescale` | `rescale` |  | `R` | 0 | True | 16384 | 31 | `None,27,29,31` |  |
| 179 | `L3.B4.R.ln_mean_r` | `layer3.block4.ln_mean_rescale_sf` | `ctct_attn_mean_rescale` | `rescale` | **31** | `R` | 3 | True | 16384 | 31 | `None,27,29,31` |  |
| 180 | `L3.B4.R.ln_var_r` | `layer3.block4.ln_var_rescale_sf` | `ctct_attn_var_rescale` | `rescale` | **28** | `R` | 3 | True | 16384 | 28 | `None,24,26,28` |  |
| 181 | `L3.B4.K` | `layer3.block4.output_truncation_k` | `block4_output_truncation` | `truncation` | **10** | `K` | 4 | True | 16384 |  | `8,9,11,13,10,12` |  |
| 182 | `L3.B5.F.inv_std_fresh` | `layer3.block5.inv_std_fresh_sf` | `ctpt_attn_inv_std` | `fresh` | **30** | `F` | 4 | True | 8192 | 30 | `22,24,26,28,30` |  |
| 183 | `L3.B5.F.x_centered_fresh` | `layer3.block5.x_centered_fresh_sf` | `ctpt_attn_x_centered` | `fresh` | **30** | `F` | 4 | True | 8192 | 30 | `22,24,26,28,30` |  |
| 184 | `L3.B5.M.gamma` | `layer3.block5.gamma_sf` | `ctpt_gamma_attn` | `encoding` | **20** | `M` | 2 | True | 8192 | 20 | `16,18,20` |  |
| 185 | `L3.B5.W.wffn1` | `layer3.block5.wffn1_sf` | `ctpt_wffn1` | `encoding` | **22** | `W` | 4 | True | 8192 | 22 | `14,16,18,20,22` |  |
| 186 | `L3.B5.M.gelu_coeff` | `layer3.block5.gelu_coeff_sf` | `ctpt_gelu_coeff` | `encoding` | **20** | `M` | 2 | True | 8192 | 20 | `16,18,20` |  |
| 187 | `L3.B5.R.normalize_r` | `layer3.block5.normalize_rescale_sf` | `ctct_attn_normalize_rescale` | `rescale` |  | `R` | 0 | True | 8192 | 31 | `None,27,29,31` |  |
| 188 | `L3.B5.R.gamma_r` | `layer3.block5.gamma_rescale_sf` | `ctct_gamma_attn_rescale` | `rescale` | **30** | `R` | 3 | True | 8192 | 30 | `None,26,28,30` |  |
| 189 | `L3.B5.R.wffn1_r` | `layer3.block5.wffn1_rescale_sf` | `ctct_wffn1_rescale` | `rescale` |  | `R` | 0 | True | 8192 | 31 | `None,27,29,31` |  |
| 190 | `L3.B5.R.gp0` | `layer3.block5.gelu_power_rescale_sf_0` | `ctct_gelu_x2` | `rescale` | _off_ | `R` | 0 | False | 8192 | 31 | `None,27,29,31` | GELU degree 1 does not use this power-rescale slot |
| 191 | `L3.B5.K` | `layer3.block5.output_truncation_k` | `block5_output_truncation` | `truncation` | **13** | `K` | 3 | True | 8192 |  | `8,9,11,13,10,12` |  |
| 192 | `L4.B1.F.gelu_out` | `layer4.block1.gelu_out_sf` | `ctpt_gelu_out` | `fresh` | **30** | `F` | 4 | True | 8192 | 30 | `22,24,26,28,30` |  |
| 193 | `L4.B1.W.wffn2` | `layer4.block1.wffn2_sf` | `ctpt_ffn2` | `encoding` | **20** | `W` | 4 | True | 8192 | 20 | `12,14,16,18,20` |  |
| 194 | `L4.B1.S.mean_inv_d` | `layer4.block1.mean_inv_d_sf` | `ctpt_inv_d_1` | `scalar` | **20** | `S` | 2 | True | 8192 | 20 | `16,18,20` |  |
| 195 | `L4.B1.S.var_inv_d` | `layer4.block1.var_inv_d_sf` | `ctpt_inv_d_2` | `scalar` | **20** | `S` | 2 | True | 8192 | 20 | `16,18,20` |  |
| 196 | `L4.B1.R.mean_r` | `layer4.block1.mean_rescale_sf` | `ctct_mean_rescale` | `rescale` | **34** | `R` | 3 | True | 8192 | 34 | `None,30,32,34` |  |
| 197 | `L4.B1.R.var_r` | `layer4.block1.var_rescale_sf` | `ctct_var_rescale` | `rescale` | **34** | `R` | 3 | True | 8192 | 34 | `None,30,32,34` |  |
| 198 | `L4.B1.K` | `layer4.block1.output_truncation_k` | `block1_output_truncation` | `truncation` | **13** | `K` | 3 | True | 8192 |  | `8,9,11,13,10,12` |  |
| 199 | `L4.B2.F.inv_std_fresh` | `layer4.block2.inv_std_fresh_sf` | `ctpt_inv_std` | `fresh` | **23** | `F` | 0 | True | 16384 | 31 | `23,25,27,29,31` |  |
| 200 | `L4.B2.F.x_centered_fresh` | `layer4.block2.x_centered_fresh_sf` | `ctpt_x_centered` | `fresh` | **22** | `F` | 0 | True | 16384 | 30 | `22,24,26,28,30` |  |
| 201 | `L4.B2.M.gamma` | `layer4.block2.gamma_sf` | `ctpt_gamma` | `encoding` | **18** | `M` | 1 | True | 16384 | 20 | `16,18,20` |  |
| 202 | `L4.B2.W.wk` | `layer4.block2.wk_sf` | `ctpt_wq_wk` | `encoding` | **14** | `W` | 0 | True | 16384 | 22 | `14,16,18,20,22` |  |
| 203 | `L4.B2.W.wv` | `layer4.block2.wv_sf` | `ctpt_wv` | `encoding` | **22** | `W` | 4 | True | 16384 | 22 | `14,16,18,20,22` |  |
| 204 | `L4.B2.M.kt_mask1` | `layer4.block2.kt_mask1_sf` | `ctpt_rotKT_mask1` | `encoding` | **13** | `M` | 1 | True | 16384 | 15 | `11,13,15` |  |
| 205 | `L4.B2.M.kt_mask2` | `layer4.block2.kt_mask2_sf` | `ctpt_rotKT_mask2` | `encoding` | **13** | `M` | 1 | True | 16384 | 15 | `11,13,15` |  |
| 206 | `L4.B2.M.qkt_merge_mask` | `layer4.block2.qkt_merge_mask_sf` | `ctpt_mask` | `encoding` | **11** | `M` | 0 | True | 16384 | 15 | `11,13,15` |  |
| 207 | `L4.B2.R.gamma_r` | `layer4.block2.gamma_rescale_sf` | `ctct_gamma_rescale` | `rescale` |  | `R` | 0 | True | 16384 | 31 | `None,27,29,31` |  |
| 208 | `L4.B2.R.kt_mask2_r` | `layer4.block2.kt_mask2_rescale_sf` | `ctct_kt_mask2_rescale` | `rescale` | **31** | `R` | 3 | True | 16384 | 31 | `None,27,29,31` |  |
| 209 | `L4.B2.R.qkt_merge_mask_r` | `layer4.block2.qkt_merge_mask_rescale_sf` | `ctct_qkt_merge_mask_rescale` | `rescale` | **28** | `R` | 3 | True | 16384 | 28 | `None,24,26,28` |  |
| 210 | `L4.B2.K` | `layer4.block2.output_truncation_k` | `block2_output_truncation` | `truncation` | **10** | `K` | 4 | True | 16384 |  | `8,9,11,13,10,12` |  |
| 211 | `L4.B3.F.x_fresh` | `layer4.block3.x_fresh_sf` | `ctpt_softmax_x` | `fresh` | **28** | `F` | 4 | True | 16384 | 28 | `20,22,24,26,28` |  |
| 212 | `L4.B3.S.inv_2n` | `layer4.block3.inv_2n_sf` | `ctpt_softmax_inv_2n` | `scalar` | **15** | `S` | 2 | True | 16384 | 15 | `11,13,15` |  |
| 213 | `L4.B3.R.sq0` | `layer4.block3.square_rescale_sf_0` | `ctct_softmax_pow_s1` | `rescale` | **31** | `R` | 3 | True | 16384 | 31 | `None,27,29,31` |  |
| 214 | `L4.B3.R.sq1` | `layer4.block3.square_rescale_sf_1` | `ctct_softmax_pow_s2` | `rescale` | **31** | `R` | 3 | True | 16384 | 31 | `None,27,29,31` |  |
| 215 | `L4.B3.R.sq2` | `layer4.block3.square_rescale_sf_2` | `ctct_softmax_pow_s3` | `rescale` | **31** | `R` | 3 | True | 16384 | 31 | `None,27,29,31` |  |
| 216 | `L4.B3.R.sq3` | `layer4.block3.square_rescale_sf_3` | `ctct_softmax_pow_s4` | `rescale` | **31** | `R` | 3 | True | 16384 | 31 | `None,27,29,31` |  |
| 217 | `L4.B3.K` | `layer4.block3.output_truncation_k` | `block3_output_truncation` | `truncation` | **13** | `K` | 3 | True | 16384 |  | `8,9,11,13,10,12` |  |
| 218 | `L4.B4.F.softmax_out_fresh` | `layer4.block4.softmax_out_fresh_sf` | `ctpt_softmax_out` | `fresh` | **27** | `F` | 0 | True | 16384 | 35 | `27,29,31,33,35` |  |
| 219 | `L4.B4.F.v_fresh` | `layer4.block4.v_fresh_sf` | `ctpt_v` | `fresh` | **22** | `F` | 0 | True | 16384 | 30 | `22,24,26,28,30` |  |
| 220 | `L4.B4.M.softmax_out_mask` | `layer4.block4.softmax_out_mask_sf` | `ctpt_softmax_out_mask` | `encoding` | **12** | `M` | 1 | True | 16384 | 14 | `10,12,14` |  |
| 221 | `L4.B4.M.v_mask` | `layer4.block4.v_mask_sf` | `ctpt_v_mask` | `encoding` | **10** | `M` | 0 | True | 16384 | 14 | `10,12,14` |  |
| 222 | `L4.B4.M.softmax_v_mask` | `layer4.block4.softmax_v_mask_sf` | `ctpt_softmax_v_mask` | `encoding` | **10** | `M` | 0 | True | 16384 | 14 | `10,12,14` |  |
| 223 | `L4.B4.S.ln_mean_inv_d` | `layer4.block4.ln_mean_inv_d_sf` | `ctpt_inv_d_attn_mean` | `scalar` | **18** | `S` | 1 | True | 16384 | 20 | `16,18,20` |  |
| 224 | `L4.B4.S.ln_var_inv_d` | `layer4.block4.ln_var_inv_d_sf` | `ctpt_inv_d_attn_var` | `scalar` | **18** | `S` | 1 | True | 16384 | 20 | `16,18,20` |  |
| 225 | `L4.B4.W.wo` | `layer4.block4.wo_sf` | `ctpt_wo` | `encoding` | **22** | `W` | 4 | True | 16384 | 22 | `14,16,18,20,22` |  |
| 226 | `L4.B4.R.softmax_v_matmul_r` | `layer4.block4.softmax_v_matmul_rescale_sf` | `ctct_softmax_v_matmul_rescale` | `rescale` |  | `R` | 0 | True | 16384 | 31 | `None,27,29,31` |  |
| 227 | `L4.B4.R.ln_mean_r` | `layer4.block4.ln_mean_rescale_sf` | `ctct_attn_mean_rescale` | `rescale` | **31** | `R` | 3 | True | 16384 | 31 | `None,27,29,31` |  |
| 228 | `L4.B4.R.ln_var_r` | `layer4.block4.ln_var_rescale_sf` | `ctct_attn_var_rescale` | `rescale` | **24** | `R` | 1 | True | 16384 | 28 | `None,24,26,28` |  |
| 229 | `L4.B4.K` | `layer4.block4.output_truncation_k` | `block4_output_truncation` | `truncation` | **10** | `K` | 4 | True | 16384 |  | `8,9,11,13,10,12` |  |
| 230 | `L4.B5.F.inv_std_fresh` | `layer4.block5.inv_std_fresh_sf` | `ctpt_attn_inv_std` | `fresh` | **30** | `F` | 4 | True | 8192 | 30 | `22,24,26,28,30` |  |
| 231 | `L4.B5.F.x_centered_fresh` | `layer4.block5.x_centered_fresh_sf` | `ctpt_attn_x_centered` | `fresh` | **30** | `F` | 4 | True | 8192 | 30 | `22,24,26,28,30` |  |
| 232 | `L4.B5.M.gamma` | `layer4.block5.gamma_sf` | `ctpt_gamma_attn` | `encoding` | **20** | `M` | 2 | True | 8192 | 20 | `16,18,20` |  |
| 233 | `L4.B5.W.wffn1` | `layer4.block5.wffn1_sf` | `ctpt_wffn1` | `encoding` | **22** | `W` | 4 | True | 8192 | 22 | `14,16,18,20,22` |  |
| 234 | `L4.B5.M.gelu_coeff` | `layer4.block5.gelu_coeff_sf` | `ctpt_gelu_coeff` | `encoding` | **20** | `M` | 2 | True | 8192 | 20 | `16,18,20` |  |
| 235 | `L4.B5.R.normalize_r` | `layer4.block5.normalize_rescale_sf` | `ctct_attn_normalize_rescale` | `rescale` |  | `R` | 0 | True | 8192 | 31 | `None,27,29,31` |  |
| 236 | `L4.B5.R.gamma_r` | `layer4.block5.gamma_rescale_sf` | `ctct_gamma_attn_rescale` | `rescale` | **30** | `R` | 3 | True | 8192 | 30 | `None,26,28,30` |  |
| 237 | `L4.B5.R.wffn1_r` | `layer4.block5.wffn1_rescale_sf` | `ctct_wffn1_rescale` | `rescale` |  | `R` | 0 | True | 8192 | 31 | `None,27,29,31` |  |
| 238 | `L4.B5.R.gp0` | `layer4.block5.gelu_power_rescale_sf_0` | `ctct_gelu_x2` | `rescale` | _off_ | `R` | 0 | False | 8192 | 31 | `None,27,29,31` | GELU degree 1 does not use this power-rescale slot |
| 239 | `L4.B5.K` | `layer4.block5.output_truncation_k` | `block5_output_truncation` | `truncation` | **13** | `K` | 3 | True | 8192 |  | `8,9,11,13,10,12` |  |
| 240 | `L5.B1.F.gelu_out` | `layer5.block1.gelu_out_sf` | `ctpt_gelu_out` | `fresh` | **30** | `F` | 4 | True | 8192 | 30 | `22,24,26,28,30` |  |
| 241 | `L5.B1.W.wffn2` | `layer5.block1.wffn2_sf` | `ctpt_ffn2` | `encoding` | **20** | `W` | 4 | True | 8192 | 20 | `12,14,16,18,20` |  |
| 242 | `L5.B1.S.mean_inv_d` | `layer5.block1.mean_inv_d_sf` | `ctpt_inv_d_1` | `scalar` | **20** | `S` | 2 | True | 8192 | 20 | `16,18,20` |  |
| 243 | `L5.B1.S.var_inv_d` | `layer5.block1.var_inv_d_sf` | `ctpt_inv_d_2` | `scalar` | **20** | `S` | 2 | True | 8192 | 20 | `16,18,20` |  |
| 244 | `L5.B1.R.mean_r` | `layer5.block1.mean_rescale_sf` | `ctct_mean_rescale` | `rescale` | **34** | `R` | 3 | True | 8192 | 34 | `None,30,32,34` |  |
| 245 | `L5.B1.R.var_r` | `layer5.block1.var_rescale_sf` | `ctct_var_rescale` | `rescale` | **34** | `R` | 3 | True | 8192 | 34 | `None,30,32,34` |  |
| 246 | `L5.B1.K` | `layer5.block1.output_truncation_k` | `block1_output_truncation` | `truncation` | **13** | `K` | 3 | True | 8192 |  | `8,9,11,13,10,12` |  |
| 247 | `L5.B2.F.inv_std_fresh` | `layer5.block2.inv_std_fresh_sf` | `ctpt_inv_std` | `fresh` | **23** | `F` | 0 | True | 16384 | 31 | `23,25,27,29,31` |  |
| 248 | `L5.B2.F.x_centered_fresh` | `layer5.block2.x_centered_fresh_sf` | `ctpt_x_centered` | `fresh` | **22** | `F` | 0 | True | 16384 | 30 | `22,24,26,28,30` |  |
| 249 | `L5.B2.M.gamma` | `layer5.block2.gamma_sf` | `ctpt_gamma` | `encoding` | **18** | `M` | 1 | True | 16384 | 20 | `16,18,20` |  |
| 250 | `L5.B2.W.wk` | `layer5.block2.wk_sf` | `ctpt_wq_wk` | `encoding` | **14** | `W` | 0 | True | 16384 | 22 | `14,16,18,20,22` |  |
| 251 | `L5.B2.W.wv` | `layer5.block2.wv_sf` | `ctpt_wv` | `encoding` | **22** | `W` | 4 | True | 16384 | 22 | `14,16,18,20,22` |  |
| 252 | `L5.B2.M.kt_mask1` | `layer5.block2.kt_mask1_sf` | `ctpt_rotKT_mask1` | `encoding` | **13** | `M` | 1 | True | 16384 | 15 | `11,13,15` |  |
| 253 | `L5.B2.M.kt_mask2` | `layer5.block2.kt_mask2_sf` | `ctpt_rotKT_mask2` | `encoding` | **13** | `M` | 1 | True | 16384 | 15 | `11,13,15` |  |
| 254 | `L5.B2.M.qkt_merge_mask` | `layer5.block2.qkt_merge_mask_sf` | `ctpt_mask` | `encoding` | **11** | `M` | 0 | True | 16384 | 15 | `11,13,15` |  |
| 255 | `L5.B2.R.gamma_r` | `layer5.block2.gamma_rescale_sf` | `ctct_gamma_rescale` | `rescale` |  | `R` | 0 | True | 16384 | 31 | `None,27,29,31` |  |
| 256 | `L5.B2.R.kt_mask2_r` | `layer5.block2.kt_mask2_rescale_sf` | `ctct_kt_mask2_rescale` | `rescale` | **31** | `R` | 3 | True | 16384 | 31 | `None,27,29,31` |  |
| 257 | `L5.B2.R.qkt_merge_mask_r` | `layer5.block2.qkt_merge_mask_rescale_sf` | `ctct_qkt_merge_mask_rescale` | `rescale` | **24** | `R` | 1 | True | 16384 | 28 | `None,24,26,28` |  |
| 258 | `L5.B2.K` | `layer5.block2.output_truncation_k` | `block2_output_truncation` | `truncation` | **10** | `K` | 4 | True | 16384 |  | `8,9,11,13,10,12` |  |
| 259 | `L5.B3.F.x_fresh` | `layer5.block3.x_fresh_sf` | `ctpt_softmax_x` | `fresh` | **27** | `F` | 4 | True | 8192 | 27 | `19,21,23,25,27` |  |
| 260 | `L5.B3.S.inv_2n` | `layer5.block3.inv_2n_sf` | `ctpt_softmax_inv_2n` | `scalar` | **16** | `S` | 2 | True | 8192 | 16 | `12,14,16` |  |
| 261 | `L5.B3.R.sq0` | `layer5.block3.square_rescale_sf_0` | `ctct_softmax_pow_s1` | `rescale` | **34** | `R` | 3 | True | 8192 | 34 | `None,30,32,34` |  |
| 262 | `L5.B3.R.sq1` | `layer5.block3.square_rescale_sf_1` | `ctct_softmax_pow_s2` | `rescale` | **34** | `R` | 3 | True | 8192 | 34 | `None,30,32,34` |  |
| 263 | `L5.B3.R.sq2` | `layer5.block3.square_rescale_sf_2` | `ctct_softmax_pow_s3` | `rescale` | _off_ | `R` | 0 | False | 8192 | 31 | `None,27,29,31` | softmax degree 2 does not use this square-rescale slot |
| 264 | `L5.B3.R.sq3` | `layer5.block3.square_rescale_sf_3` | `ctct_softmax_pow_s4` | `rescale` | _off_ | `R` | 0 | False | 8192 | 31 | `None,27,29,31` | softmax degree 2 does not use this square-rescale slot |
| 265 | `L5.B3.K` | `layer5.block3.output_truncation_k` | `block3_output_truncation` | `truncation` | **13** | `K` | 3 | True | 8192 |  | `8,9,11,13,10,12` |  |
| 266 | `L5.B4.F.softmax_out_fresh` | `layer5.block4.softmax_out_fresh_sf` | `ctpt_softmax_out` | `fresh` | **27** | `F` | 0 | True | 16384 | 35 | `27,29,31,33,35` |  |
| 267 | `L5.B4.F.v_fresh` | `layer5.block4.v_fresh_sf` | `ctpt_v` | `fresh` | **22** | `F` | 0 | True | 16384 | 30 | `22,24,26,28,30` |  |
| 268 | `L5.B4.M.softmax_out_mask` | `layer5.block4.softmax_out_mask_sf` | `ctpt_softmax_out_mask` | `encoding` | **12** | `M` | 1 | True | 16384 | 14 | `10,12,14` |  |
| 269 | `L5.B4.M.v_mask` | `layer5.block4.v_mask_sf` | `ctpt_v_mask` | `encoding` | **10** | `M` | 0 | True | 16384 | 14 | `10,12,14` |  |
| 270 | `L5.B4.M.softmax_v_mask` | `layer5.block4.softmax_v_mask_sf` | `ctpt_softmax_v_mask` | `encoding` | **10** | `M` | 0 | True | 16384 | 14 | `10,12,14` |  |
| 271 | `L5.B4.S.ln_mean_inv_d` | `layer5.block4.ln_mean_inv_d_sf` | `ctpt_inv_d_attn_mean` | `scalar` | **18** | `S` | 1 | True | 16384 | 20 | `16,18,20` |  |
| 272 | `L5.B4.S.ln_var_inv_d` | `layer5.block4.ln_var_inv_d_sf` | `ctpt_inv_d_attn_var` | `scalar` | **18** | `S` | 1 | True | 16384 | 20 | `16,18,20` |  |
| 273 | `L5.B4.W.wo` | `layer5.block4.wo_sf` | `ctpt_wo` | `encoding` | **22** | `W` | 4 | True | 16384 | 22 | `14,16,18,20,22` |  |
| 274 | `L5.B4.R.softmax_v_matmul_r` | `layer5.block4.softmax_v_matmul_rescale_sf` | `ctct_softmax_v_matmul_rescale` | `rescale` |  | `R` | 0 | True | 16384 | 31 | `None,27,29,31` |  |
| 275 | `L5.B4.R.ln_mean_r` | `layer5.block4.ln_mean_rescale_sf` | `ctct_attn_mean_rescale` | `rescale` | **31** | `R` | 3 | True | 16384 | 31 | `None,27,29,31` |  |
| 276 | `L5.B4.R.ln_var_r` | `layer5.block4.ln_var_rescale_sf` | `ctct_attn_var_rescale` | `rescale` | **28** | `R` | 3 | True | 16384 | 28 | `None,24,26,28` |  |
| 277 | `L5.B4.K` | `layer5.block4.output_truncation_k` | `block4_output_truncation` | `truncation` | **10** | `K` | 4 | True | 16384 |  | `8,9,11,13,10,12` |  |
| 278 | `L5.B5.F.inv_std_fresh` | `layer5.block5.inv_std_fresh_sf` | `ctpt_attn_inv_std` | `fresh` | **22** | `F` | 0 | True | 16384 | 30 | `22,24,26,28,30` |  |
| 279 | `L5.B5.F.x_centered_fresh` | `layer5.block5.x_centered_fresh_sf` | `ctpt_attn_x_centered` | `fresh` | **23** | `F` | 0 | True | 16384 | 31 | `23,25,27,29,31` |  |
| 280 | `L5.B5.M.gamma` | `layer5.block5.gamma_sf` | `ctpt_gamma_attn` | `encoding` | **18** | `M` | 1 | True | 16384 | 20 | `16,18,20` |  |
| 281 | `L5.B5.W.wffn1` | `layer5.block5.wffn1_sf` | `ctpt_wffn1` | `encoding` | **14** | `W` | 0 | True | 16384 | 22 | `14,16,18,20,22` |  |
| 282 | `L5.B5.M.gelu_coeff` | `layer5.block5.gelu_coeff_sf` | `ctpt_gelu_coeff` | `encoding` | **16** | `M` | 0 | True | 16384 | 20 | `16,18,20` |  |
| 283 | `L5.B5.R.normalize_r` | `layer5.block5.normalize_rescale_sf` | `ctct_attn_normalize_rescale` | `rescale` | **27** | `R` | 1 | True | 16384 | 31 | `None,27,29,31` |  |
| 284 | `L5.B5.R.gamma_r` | `layer5.block5.gamma_rescale_sf` | `ctct_gamma_attn_rescale` | `rescale` | **18** | `R` | 1 | True | 16384 | 22 | `None,18,20,22` |  |
| 285 | `L5.B5.R.wffn1_r` | `layer5.block5.wffn1_rescale_sf` | `ctct_wffn1_rescale` | `rescale` | **31** | `R` | 3 | True | 16384 | 31 | `None,27,29,31` |  |
| 286 | `L5.B5.R.gp0` | `layer5.block5.gelu_power_rescale_sf_0` | `ctct_gelu_x2` | `rescale` |  | `R` | 0 | True | 16384 | 31 | `None,27,29,31` |  |
| 287 | `L5.B5.K` | `layer5.block5.output_truncation_k` | `block5_output_truncation` | `truncation` | **12** | `K` | 5 | True | 16384 |  | `8,9,11,13,10,12` |  |
| 288 | `L6.B1.F.gelu_out` | `layer6.block1.gelu_out_sf` | `ctpt_gelu_out` | `fresh` | **30** | `F` | 4 | True | 8192 | 30 | `22,24,26,28,30` |  |
| 289 | `L6.B1.W.wffn2` | `layer6.block1.wffn2_sf` | `ctpt_ffn2` | `encoding` | **20** | `W` | 4 | True | 8192 | 20 | `12,14,16,18,20` |  |
| 290 | `L6.B1.S.mean_inv_d` | `layer6.block1.mean_inv_d_sf` | `ctpt_inv_d_1` | `scalar` | **20** | `S` | 2 | True | 8192 | 20 | `16,18,20` |  |
| 291 | `L6.B1.S.var_inv_d` | `layer6.block1.var_inv_d_sf` | `ctpt_inv_d_2` | `scalar` | **20** | `S` | 2 | True | 8192 | 20 | `16,18,20` |  |
| 292 | `L6.B1.R.mean_r` | `layer6.block1.mean_rescale_sf` | `ctct_mean_rescale` | `rescale` | **34** | `R` | 3 | True | 8192 | 34 | `None,30,32,34` |  |
| 293 | `L6.B1.R.var_r` | `layer6.block1.var_rescale_sf` | `ctct_var_rescale` | `rescale` | **34** | `R` | 3 | True | 8192 | 34 | `None,30,32,34` |  |
| 294 | `L6.B1.K` | `layer6.block1.output_truncation_k` | `block1_output_truncation` | `truncation` | **13** | `K` | 3 | True | 8192 |  | `8,9,11,13,10,12` |  |
| 295 | `L6.B2.F.inv_std_fresh` | `layer6.block2.inv_std_fresh_sf` | `ctpt_inv_std` | `fresh` | **23** | `F` | 0 | True | 16384 | 31 | `23,25,27,29,31` |  |
| 296 | `L6.B2.F.x_centered_fresh` | `layer6.block2.x_centered_fresh_sf` | `ctpt_x_centered` | `fresh` | **22** | `F` | 0 | True | 16384 | 30 | `22,24,26,28,30` |  |
| 297 | `L6.B2.M.gamma` | `layer6.block2.gamma_sf` | `ctpt_gamma` | `encoding` | **18** | `M` | 1 | True | 16384 | 20 | `16,18,20` |  |
| 298 | `L6.B2.W.wk` | `layer6.block2.wk_sf` | `ctpt_wq_wk` | `encoding` | **14** | `W` | 0 | True | 16384 | 22 | `14,16,18,20,22` |  |
| 299 | `L6.B2.W.wv` | `layer6.block2.wv_sf` | `ctpt_wv` | `encoding` | **22** | `W` | 4 | True | 16384 | 22 | `14,16,18,20,22` |  |
| 300 | `L6.B2.M.kt_mask1` | `layer6.block2.kt_mask1_sf` | `ctpt_rotKT_mask1` | `encoding` | **13** | `M` | 1 | True | 16384 | 15 | `11,13,15` |  |
| 301 | `L6.B2.M.kt_mask2` | `layer6.block2.kt_mask2_sf` | `ctpt_rotKT_mask2` | `encoding` | **13** | `M` | 1 | True | 16384 | 15 | `11,13,15` |  |
| 302 | `L6.B2.M.qkt_merge_mask` | `layer6.block2.qkt_merge_mask_sf` | `ctpt_mask` | `encoding` | **11** | `M` | 0 | True | 16384 | 15 | `11,13,15` |  |
| 303 | `L6.B2.R.gamma_r` | `layer6.block2.gamma_rescale_sf` | `ctct_gamma_rescale` | `rescale` |  | `R` | 0 | True | 16384 | 31 | `None,27,29,31` |  |
| 304 | `L6.B2.R.kt_mask2_r` | `layer6.block2.kt_mask2_rescale_sf` | `ctct_kt_mask2_rescale` | `rescale` | **31** | `R` | 3 | True | 16384 | 31 | `None,27,29,31` |  |
| 305 | `L6.B2.R.qkt_merge_mask_r` | `layer6.block2.qkt_merge_mask_rescale_sf` | `ctct_qkt_merge_mask_rescale` | `rescale` | **24** | `R` | 1 | True | 16384 | 28 | `None,24,26,28` |  |
| 306 | `L6.B2.K` | `layer6.block2.output_truncation_k` | `block2_output_truncation` | `truncation` | **10** | `K` | 4 | True | 16384 |  | `8,9,11,13,10,12` |  |
| 307 | `L6.B3.F.x_fresh` | `layer6.block3.x_fresh_sf` | `ctpt_softmax_x` | `fresh` | **28** | `F` | 4 | True | 16384 | 28 | `20,22,24,26,28` |  |
| 308 | `L6.B3.S.inv_2n` | `layer6.block3.inv_2n_sf` | `ctpt_softmax_inv_2n` | `scalar` | **15** | `S` | 2 | True | 16384 | 15 | `11,13,15` |  |
| 309 | `L6.B3.R.sq0` | `layer6.block3.square_rescale_sf_0` | `ctct_softmax_pow_s1` | `rescale` | **31** | `R` | 3 | True | 16384 | 31 | `None,27,29,31` |  |
| 310 | `L6.B3.R.sq1` | `layer6.block3.square_rescale_sf_1` | `ctct_softmax_pow_s2` | `rescale` | **31** | `R` | 3 | True | 16384 | 31 | `None,27,29,31` |  |
| 311 | `L6.B3.R.sq2` | `layer6.block3.square_rescale_sf_2` | `ctct_softmax_pow_s3` | `rescale` | **31** | `R` | 3 | True | 16384 | 31 | `None,27,29,31` |  |
| 312 | `L6.B3.R.sq3` | `layer6.block3.square_rescale_sf_3` | `ctct_softmax_pow_s4` | `rescale` | **31** | `R` | 3 | True | 16384 | 31 | `None,27,29,31` |  |
| 313 | `L6.B3.K` | `layer6.block3.output_truncation_k` | `block3_output_truncation` | `truncation` | **13** | `K` | 3 | True | 16384 |  | `8,9,11,13,10,12` |  |
| 314 | `L6.B4.F.softmax_out_fresh` | `layer6.block4.softmax_out_fresh_sf` | `ctpt_softmax_out` | `fresh` | **27** | `F` | 0 | True | 16384 | 35 | `27,29,31,33,35` |  |
| 315 | `L6.B4.F.v_fresh` | `layer6.block4.v_fresh_sf` | `ctpt_v` | `fresh` | **22** | `F` | 0 | True | 16384 | 30 | `22,24,26,28,30` |  |
| 316 | `L6.B4.M.softmax_out_mask` | `layer6.block4.softmax_out_mask_sf` | `ctpt_softmax_out_mask` | `encoding` | **12** | `M` | 1 | True | 16384 | 14 | `10,12,14` |  |
| 317 | `L6.B4.M.v_mask` | `layer6.block4.v_mask_sf` | `ctpt_v_mask` | `encoding` | **10** | `M` | 0 | True | 16384 | 14 | `10,12,14` |  |
| 318 | `L6.B4.M.softmax_v_mask` | `layer6.block4.softmax_v_mask_sf` | `ctpt_softmax_v_mask` | `encoding` | **10** | `M` | 0 | True | 16384 | 14 | `10,12,14` |  |
| 319 | `L6.B4.S.ln_mean_inv_d` | `layer6.block4.ln_mean_inv_d_sf` | `ctpt_inv_d_attn_mean` | `scalar` | **18** | `S` | 1 | True | 16384 | 20 | `16,18,20` |  |
| 320 | `L6.B4.S.ln_var_inv_d` | `layer6.block4.ln_var_inv_d_sf` | `ctpt_inv_d_attn_var` | `scalar` | **18** | `S` | 1 | True | 16384 | 20 | `16,18,20` |  |
| 321 | `L6.B4.W.wo` | `layer6.block4.wo_sf` | `ctpt_wo` | `encoding` | **22** | `W` | 4 | True | 16384 | 22 | `14,16,18,20,22` |  |
| 322 | `L6.B4.R.softmax_v_matmul_r` | `layer6.block4.softmax_v_matmul_rescale_sf` | `ctct_softmax_v_matmul_rescale` | `rescale` |  | `R` | 0 | True | 16384 | 31 | `None,27,29,31` |  |
| 323 | `L6.B4.R.ln_mean_r` | `layer6.block4.ln_mean_rescale_sf` | `ctct_attn_mean_rescale` | `rescale` | **31** | `R` | 3 | True | 16384 | 31 | `None,27,29,31` |  |
| 324 | `L6.B4.R.ln_var_r` | `layer6.block4.ln_var_rescale_sf` | `ctct_attn_var_rescale` | `rescale` | **28** | `R` | 3 | True | 16384 | 28 | `None,24,26,28` |  |
| 325 | `L6.B4.K` | `layer6.block4.output_truncation_k` | `block4_output_truncation` | `truncation` | **10** | `K` | 4 | True | 16384 |  | `8,9,11,13,10,12` |  |
| 326 | `L6.B5.F.inv_std_fresh` | `layer6.block5.inv_std_fresh_sf` | `ctpt_attn_inv_std` | `fresh` | **30** | `F` | 4 | True | 8192 | 30 | `22,24,26,28,30` |  |
| 327 | `L6.B5.F.x_centered_fresh` | `layer6.block5.x_centered_fresh_sf` | `ctpt_attn_x_centered` | `fresh` | **30** | `F` | 4 | True | 8192 | 30 | `22,24,26,28,30` |  |
| 328 | `L6.B5.M.gamma` | `layer6.block5.gamma_sf` | `ctpt_gamma_attn` | `encoding` | **20** | `M` | 2 | True | 8192 | 20 | `16,18,20` |  |
| 329 | `L6.B5.W.wffn1` | `layer6.block5.wffn1_sf` | `ctpt_wffn1` | `encoding` | **22** | `W` | 4 | True | 8192 | 22 | `14,16,18,20,22` |  |
| 330 | `L6.B5.M.gelu_coeff` | `layer6.block5.gelu_coeff_sf` | `ctpt_gelu_coeff` | `encoding` | **20** | `M` | 2 | True | 8192 | 20 | `16,18,20` |  |
| 331 | `L6.B5.R.normalize_r` | `layer6.block5.normalize_rescale_sf` | `ctct_attn_normalize_rescale` | `rescale` |  | `R` | 0 | True | 8192 | 31 | `None,27,29,31` |  |
| 332 | `L6.B5.R.gamma_r` | `layer6.block5.gamma_rescale_sf` | `ctct_gamma_attn_rescale` | `rescale` | **30** | `R` | 3 | True | 8192 | 30 | `None,26,28,30` |  |
| 333 | `L6.B5.R.wffn1_r` | `layer6.block5.wffn1_rescale_sf` | `ctct_wffn1_rescale` | `rescale` |  | `R` | 0 | True | 8192 | 31 | `None,27,29,31` |  |
| 334 | `L6.B5.R.gp0` | `layer6.block5.gelu_power_rescale_sf_0` | `ctct_gelu_x2` | `rescale` | _off_ | `R` | 0 | False | 8192 | 31 | `None,27,29,31` | GELU degree 1 does not use this power-rescale slot |
| 335 | `L6.B5.K` | `layer6.block5.output_truncation_k` | `block5_output_truncation` | `truncation` | **13** | `K` | 3 | True | 8192 |  | `8,9,11,13,10,12` |  |
| 336 | `L7.B1.F.gelu_out` | `layer7.block1.gelu_out_sf` | `ctpt_gelu_out` | `fresh` | **30** | `F` | 4 | True | 8192 | 30 | `22,24,26,28,30` |  |
| 337 | `L7.B1.W.wffn2` | `layer7.block1.wffn2_sf` | `ctpt_ffn2` | `encoding` | **20** | `W` | 4 | True | 8192 | 20 | `12,14,16,18,20` |  |
| 338 | `L7.B1.S.mean_inv_d` | `layer7.block1.mean_inv_d_sf` | `ctpt_inv_d_1` | `scalar` | **20** | `S` | 2 | True | 8192 | 20 | `16,18,20` |  |
| 339 | `L7.B1.S.var_inv_d` | `layer7.block1.var_inv_d_sf` | `ctpt_inv_d_2` | `scalar` | **20** | `S` | 2 | True | 8192 | 20 | `16,18,20` |  |
| 340 | `L7.B1.R.mean_r` | `layer7.block1.mean_rescale_sf` | `ctct_mean_rescale` | `rescale` | **34** | `R` | 3 | True | 8192 | 34 | `None,30,32,34` |  |
| 341 | `L7.B1.R.var_r` | `layer7.block1.var_rescale_sf` | `ctct_var_rescale` | `rescale` | **34** | `R` | 3 | True | 8192 | 34 | `None,30,32,34` |  |
| 342 | `L7.B1.K` | `layer7.block1.output_truncation_k` | `block1_output_truncation` | `truncation` | **13** | `K` | 3 | True | 8192 |  | `8,9,11,13,10,12` |  |
| 343 | `L7.B2.F.inv_std_fresh` | `layer7.block2.inv_std_fresh_sf` | `ctpt_inv_std` | `fresh` | **23** | `F` | 0 | True | 16384 | 31 | `23,25,27,29,31` |  |
| 344 | `L7.B2.F.x_centered_fresh` | `layer7.block2.x_centered_fresh_sf` | `ctpt_x_centered` | `fresh` | **22** | `F` | 0 | True | 16384 | 30 | `22,24,26,28,30` |  |
| 345 | `L7.B2.M.gamma` | `layer7.block2.gamma_sf` | `ctpt_gamma` | `encoding` | **20** | `M` | 2 | True | 16384 | 20 | `16,18,20` |  |
| 346 | `L7.B2.W.wk` | `layer7.block2.wk_sf` | `ctpt_wq_wk` | `encoding` | **14** | `W` | 0 | True | 16384 | 22 | `14,16,18,20,22` |  |
| 347 | `L7.B2.W.wv` | `layer7.block2.wv_sf` | `ctpt_wv` | `encoding` | **22** | `W` | 4 | True | 16384 | 22 | `14,16,18,20,22` |  |
| 348 | `L7.B2.M.kt_mask1` | `layer7.block2.kt_mask1_sf` | `ctpt_rotKT_mask1` | `encoding` | **13** | `M` | 1 | True | 16384 | 15 | `11,13,15` |  |
| 349 | `L7.B2.M.kt_mask2` | `layer7.block2.kt_mask2_sf` | `ctpt_rotKT_mask2` | `encoding` | **13** | `M` | 1 | True | 16384 | 15 | `11,13,15` |  |
| 350 | `L7.B2.M.qkt_merge_mask` | `layer7.block2.qkt_merge_mask_sf` | `ctpt_mask` | `encoding` | **11** | `M` | 0 | True | 16384 | 15 | `11,13,15` |  |
| 351 | `L7.B2.R.gamma_r` | `layer7.block2.gamma_rescale_sf` | `ctct_gamma_rescale` | `rescale` |  | `R` | 0 | True | 16384 | 31 | `None,27,29,31` |  |
| 352 | `L7.B2.R.kt_mask2_r` | `layer7.block2.kt_mask2_rescale_sf` | `ctct_kt_mask2_rescale` | `rescale` | **31** | `R` | 3 | True | 16384 | 31 | `None,27,29,31` |  |
| 353 | `L7.B2.R.qkt_merge_mask_r` | `layer7.block2.qkt_merge_mask_rescale_sf` | `ctct_qkt_merge_mask_rescale` | `rescale` | **24** | `R` | 1 | True | 16384 | 28 | `None,24,26,28` |  |
| 354 | `L7.B2.K` | `layer7.block2.output_truncation_k` | `block2_output_truncation` | `truncation` | **10** | `K` | 4 | True | 16384 |  | `8,9,11,13,10,12` |  |
| 355 | `L7.B3.F.x_fresh` | `layer7.block3.x_fresh_sf` | `ctpt_softmax_x` | `fresh` | **27** | `F` | 4 | True | 8192 | 27 | `19,21,23,25,27` |  |
| 356 | `L7.B3.S.inv_2n` | `layer7.block3.inv_2n_sf` | `ctpt_softmax_inv_2n` | `scalar` | **16** | `S` | 2 | True | 8192 | 16 | `12,14,16` |  |
| 357 | `L7.B3.R.sq0` | `layer7.block3.square_rescale_sf_0` | `ctct_softmax_pow_s1` | `rescale` | **34** | `R` | 3 | True | 8192 | 34 | `None,30,32,34` |  |
| 358 | `L7.B3.R.sq1` | `layer7.block3.square_rescale_sf_1` | `ctct_softmax_pow_s2` | `rescale` | **34** | `R` | 3 | True | 8192 | 34 | `None,30,32,34` |  |
| 359 | `L7.B3.R.sq2` | `layer7.block3.square_rescale_sf_2` | `ctct_softmax_pow_s3` | `rescale` | _off_ | `R` | 0 | False | 8192 | 31 | `None,27,29,31` | softmax degree 2 does not use this square-rescale slot |
| 360 | `L7.B3.R.sq3` | `layer7.block3.square_rescale_sf_3` | `ctct_softmax_pow_s4` | `rescale` | _off_ | `R` | 0 | False | 8192 | 31 | `None,27,29,31` | softmax degree 2 does not use this square-rescale slot |
| 361 | `L7.B3.K` | `layer7.block3.output_truncation_k` | `block3_output_truncation` | `truncation` | **13** | `K` | 3 | True | 8192 |  | `8,9,11,13,10,12` |  |
| 362 | `L7.B4.F.softmax_out_fresh` | `layer7.block4.softmax_out_fresh_sf` | `ctpt_softmax_out` | `fresh` | **27** | `F` | 0 | True | 16384 | 35 | `27,29,31,33,35` |  |
| 363 | `L7.B4.F.v_fresh` | `layer7.block4.v_fresh_sf` | `ctpt_v` | `fresh` | **22** | `F` | 0 | True | 16384 | 30 | `22,24,26,28,30` |  |
| 364 | `L7.B4.M.softmax_out_mask` | `layer7.block4.softmax_out_mask_sf` | `ctpt_softmax_out_mask` | `encoding` | **12** | `M` | 1 | True | 16384 | 14 | `10,12,14` |  |
| 365 | `L7.B4.M.v_mask` | `layer7.block4.v_mask_sf` | `ctpt_v_mask` | `encoding` | **10** | `M` | 0 | True | 16384 | 14 | `10,12,14` |  |
| 366 | `L7.B4.M.softmax_v_mask` | `layer7.block4.softmax_v_mask_sf` | `ctpt_softmax_v_mask` | `encoding` | **10** | `M` | 0 | True | 16384 | 14 | `10,12,14` |  |
| 367 | `L7.B4.S.ln_mean_inv_d` | `layer7.block4.ln_mean_inv_d_sf` | `ctpt_inv_d_attn_mean` | `scalar` | **18** | `S` | 1 | True | 16384 | 20 | `16,18,20` |  |
| 368 | `L7.B4.S.ln_var_inv_d` | `layer7.block4.ln_var_inv_d_sf` | `ctpt_inv_d_attn_var` | `scalar` | **18** | `S` | 1 | True | 16384 | 20 | `16,18,20` |  |
| 369 | `L7.B4.W.wo` | `layer7.block4.wo_sf` | `ctpt_wo` | `encoding` | **22** | `W` | 4 | True | 16384 | 22 | `14,16,18,20,22` |  |
| 370 | `L7.B4.R.softmax_v_matmul_r` | `layer7.block4.softmax_v_matmul_rescale_sf` | `ctct_softmax_v_matmul_rescale` | `rescale` |  | `R` | 0 | True | 16384 | 31 | `None,27,29,31` |  |
| 371 | `L7.B4.R.ln_mean_r` | `layer7.block4.ln_mean_rescale_sf` | `ctct_attn_mean_rescale` | `rescale` | **31** | `R` | 3 | True | 16384 | 31 | `None,27,29,31` |  |
| 372 | `L7.B4.R.ln_var_r` | `layer7.block4.ln_var_rescale_sf` | `ctct_attn_var_rescale` | `rescale` | **28** | `R` | 3 | True | 16384 | 28 | `None,24,26,28` |  |
| 373 | `L7.B4.K` | `layer7.block4.output_truncation_k` | `block4_output_truncation` | `truncation` | **10** | `K` | 4 | True | 16384 |  | `8,9,11,13,10,12` |  |
| 374 | `L7.B5.F.inv_std_fresh` | `layer7.block5.inv_std_fresh_sf` | `ctpt_attn_inv_std` | `fresh` | **30** | `F` | 4 | True | 8192 | 30 | `22,24,26,28,30` |  |
| 375 | `L7.B5.F.x_centered_fresh` | `layer7.block5.x_centered_fresh_sf` | `ctpt_attn_x_centered` | `fresh` | **30** | `F` | 4 | True | 8192 | 30 | `22,24,26,28,30` |  |
| 376 | `L7.B5.M.gamma` | `layer7.block5.gamma_sf` | `ctpt_gamma_attn` | `encoding` | **20** | `M` | 2 | True | 8192 | 20 | `16,18,20` |  |
| 377 | `L7.B5.W.wffn1` | `layer7.block5.wffn1_sf` | `ctpt_wffn1` | `encoding` | **22** | `W` | 4 | True | 8192 | 22 | `14,16,18,20,22` |  |
| 378 | `L7.B5.M.gelu_coeff` | `layer7.block5.gelu_coeff_sf` | `ctpt_gelu_coeff` | `encoding` | **20** | `M` | 2 | True | 8192 | 20 | `16,18,20` |  |
| 379 | `L7.B5.R.normalize_r` | `layer7.block5.normalize_rescale_sf` | `ctct_attn_normalize_rescale` | `rescale` |  | `R` | 0 | True | 8192 | 31 | `None,27,29,31` |  |
| 380 | `L7.B5.R.gamma_r` | `layer7.block5.gamma_rescale_sf` | `ctct_gamma_attn_rescale` | `rescale` | **30** | `R` | 3 | True | 8192 | 30 | `None,26,28,30` |  |
| 381 | `L7.B5.R.wffn1_r` | `layer7.block5.wffn1_rescale_sf` | `ctct_wffn1_rescale` | `rescale` |  | `R` | 0 | True | 8192 | 31 | `None,27,29,31` |  |
| 382 | `L7.B5.R.gp0` | `layer7.block5.gelu_power_rescale_sf_0` | `ctct_gelu_x2` | `rescale` | _off_ | `R` | 0 | False | 8192 | 31 | `None,27,29,31` | GELU degree 1 does not use this power-rescale slot |
| 383 | `L7.B5.K` | `layer7.block5.output_truncation_k` | `block5_output_truncation` | `truncation` | **13** | `K` | 3 | True | 8192 |  | `8,9,11,13,10,12` |  |
| 384 | `L8.B1.F.gelu_out` | `layer8.block1.gelu_out_sf` | `ctpt_gelu_out` | `fresh` | **30** | `F` | 4 | True | 8192 | 30 | `22,24,26,28,30` |  |
| 385 | `L8.B1.W.wffn2` | `layer8.block1.wffn2_sf` | `ctpt_ffn2` | `encoding` | **20** | `W` | 4 | True | 8192 | 20 | `12,14,16,18,20` |  |
| 386 | `L8.B1.S.mean_inv_d` | `layer8.block1.mean_inv_d_sf` | `ctpt_inv_d_1` | `scalar` | **20** | `S` | 2 | True | 8192 | 20 | `16,18,20` |  |
| 387 | `L8.B1.S.var_inv_d` | `layer8.block1.var_inv_d_sf` | `ctpt_inv_d_2` | `scalar` | **20** | `S` | 2 | True | 8192 | 20 | `16,18,20` |  |
| 388 | `L8.B1.R.mean_r` | `layer8.block1.mean_rescale_sf` | `ctct_mean_rescale` | `rescale` | **34** | `R` | 3 | True | 8192 | 34 | `None,30,32,34` |  |
| 389 | `L8.B1.R.var_r` | `layer8.block1.var_rescale_sf` | `ctct_var_rescale` | `rescale` | **34** | `R` | 3 | True | 8192 | 34 | `None,30,32,34` |  |
| 390 | `L8.B1.K` | `layer8.block1.output_truncation_k` | `block1_output_truncation` | `truncation` | **13** | `K` | 3 | True | 8192 |  | `8,9,11,13,10,12` |  |
| 391 | `L8.B2.F.inv_std_fresh` | `layer8.block2.inv_std_fresh_sf` | `ctpt_inv_std` | `fresh` | **23** | `F` | 0 | True | 16384 | 31 | `23,25,27,29,31` |  |
| 392 | `L8.B2.F.x_centered_fresh` | `layer8.block2.x_centered_fresh_sf` | `ctpt_x_centered` | `fresh` | **22** | `F` | 0 | True | 16384 | 30 | `22,24,26,28,30` |  |
| 393 | `L8.B2.M.gamma` | `layer8.block2.gamma_sf` | `ctpt_gamma` | `encoding` | **18** | `M` | 1 | True | 16384 | 20 | `16,18,20` |  |
| 394 | `L8.B2.W.wk` | `layer8.block2.wk_sf` | `ctpt_wq_wk` | `encoding` | **14** | `W` | 0 | True | 16384 | 22 | `14,16,18,20,22` |  |
| 395 | `L8.B2.W.wv` | `layer8.block2.wv_sf` | `ctpt_wv` | `encoding` | **22** | `W` | 4 | True | 16384 | 22 | `14,16,18,20,22` |  |
| 396 | `L8.B2.M.kt_mask1` | `layer8.block2.kt_mask1_sf` | `ctpt_rotKT_mask1` | `encoding` | **13** | `M` | 1 | True | 16384 | 15 | `11,13,15` |  |
| 397 | `L8.B2.M.kt_mask2` | `layer8.block2.kt_mask2_sf` | `ctpt_rotKT_mask2` | `encoding` | **13** | `M` | 1 | True | 16384 | 15 | `11,13,15` |  |
| 398 | `L8.B2.M.qkt_merge_mask` | `layer8.block2.qkt_merge_mask_sf` | `ctpt_mask` | `encoding` | **11** | `M` | 0 | True | 16384 | 15 | `11,13,15` |  |
| 399 | `L8.B2.R.gamma_r` | `layer8.block2.gamma_rescale_sf` | `ctct_gamma_rescale` | `rescale` |  | `R` | 0 | True | 16384 | 31 | `None,27,29,31` |  |
| 400 | `L8.B2.R.kt_mask2_r` | `layer8.block2.kt_mask2_rescale_sf` | `ctct_kt_mask2_rescale` | `rescale` | **29** | `R` | 2 | True | 16384 | 31 | `None,27,29,31` |  |
| 401 | `L8.B2.R.qkt_merge_mask_r` | `layer8.block2.qkt_merge_mask_rescale_sf` | `ctct_qkt_merge_mask_rescale` | `rescale` | **28** | `R` | 3 | True | 16384 | 28 | `None,24,26,28` |  |
| 402 | `L8.B2.K` | `layer8.block2.output_truncation_k` | `block2_output_truncation` | `truncation` | **10** | `K` | 4 | True | 16384 |  | `8,9,11,13,10,12` |  |
| 403 | `L8.B3.F.x_fresh` | `layer8.block3.x_fresh_sf` | `ctpt_softmax_x` | `fresh` | **28** | `F` | 4 | True | 16384 | 28 | `20,22,24,26,28` |  |
| 404 | `L8.B3.S.inv_2n` | `layer8.block3.inv_2n_sf` | `ctpt_softmax_inv_2n` | `scalar` | **15** | `S` | 2 | True | 16384 | 15 | `11,13,15` |  |
| 405 | `L8.B3.R.sq0` | `layer8.block3.square_rescale_sf_0` | `ctct_softmax_pow_s1` | `rescale` | **31** | `R` | 3 | True | 16384 | 31 | `None,27,29,31` |  |
| 406 | `L8.B3.R.sq1` | `layer8.block3.square_rescale_sf_1` | `ctct_softmax_pow_s2` | `rescale` | **31** | `R` | 3 | True | 16384 | 31 | `None,27,29,31` |  |
| 407 | `L8.B3.R.sq2` | `layer8.block3.square_rescale_sf_2` | `ctct_softmax_pow_s3` | `rescale` | **31** | `R` | 3 | True | 16384 | 31 | `None,27,29,31` |  |
| 408 | `L8.B3.R.sq3` | `layer8.block3.square_rescale_sf_3` | `ctct_softmax_pow_s4` | `rescale` | **31** | `R` | 3 | True | 16384 | 31 | `None,27,29,31` |  |
| 409 | `L8.B3.K` | `layer8.block3.output_truncation_k` | `block3_output_truncation` | `truncation` | **13** | `K` | 3 | True | 16384 |  | `8,9,11,13,10,12` |  |
| 410 | `L8.B4.F.softmax_out_fresh` | `layer8.block4.softmax_out_fresh_sf` | `ctpt_softmax_out` | `fresh` | **27** | `F` | 0 | True | 16384 | 35 | `27,29,31,33,35` |  |
| 411 | `L8.B4.F.v_fresh` | `layer8.block4.v_fresh_sf` | `ctpt_v` | `fresh` | **22** | `F` | 0 | True | 16384 | 30 | `22,24,26,28,30` |  |
| 412 | `L8.B4.M.softmax_out_mask` | `layer8.block4.softmax_out_mask_sf` | `ctpt_softmax_out_mask` | `encoding` | **12** | `M` | 1 | True | 16384 | 14 | `10,12,14` |  |
| 413 | `L8.B4.M.v_mask` | `layer8.block4.v_mask_sf` | `ctpt_v_mask` | `encoding` | **10** | `M` | 0 | True | 16384 | 14 | `10,12,14` |  |
| 414 | `L8.B4.M.softmax_v_mask` | `layer8.block4.softmax_v_mask_sf` | `ctpt_softmax_v_mask` | `encoding` | **10** | `M` | 0 | True | 16384 | 14 | `10,12,14` |  |
| 415 | `L8.B4.S.ln_mean_inv_d` | `layer8.block4.ln_mean_inv_d_sf` | `ctpt_inv_d_attn_mean` | `scalar` | **18** | `S` | 1 | True | 16384 | 20 | `16,18,20` |  |
| 416 | `L8.B4.S.ln_var_inv_d` | `layer8.block4.ln_var_inv_d_sf` | `ctpt_inv_d_attn_var` | `scalar` | **18** | `S` | 1 | True | 16384 | 20 | `16,18,20` |  |
| 417 | `L8.B4.W.wo` | `layer8.block4.wo_sf` | `ctpt_wo` | `encoding` | **22** | `W` | 4 | True | 16384 | 22 | `14,16,18,20,22` |  |
| 418 | `L8.B4.R.softmax_v_matmul_r` | `layer8.block4.softmax_v_matmul_rescale_sf` | `ctct_softmax_v_matmul_rescale` | `rescale` |  | `R` | 0 | True | 16384 | 31 | `None,27,29,31` |  |
| 419 | `L8.B4.R.ln_mean_r` | `layer8.block4.ln_mean_rescale_sf` | `ctct_attn_mean_rescale` | `rescale` | **31** | `R` | 3 | True | 16384 | 31 | `None,27,29,31` |  |
| 420 | `L8.B4.R.ln_var_r` | `layer8.block4.ln_var_rescale_sf` | `ctct_attn_var_rescale` | `rescale` | **24** | `R` | 1 | True | 16384 | 28 | `None,24,26,28` |  |
| 421 | `L8.B4.K` | `layer8.block4.output_truncation_k` | `block4_output_truncation` | `truncation` | **10** | `K` | 4 | True | 16384 |  | `8,9,11,13,10,12` |  |
| 422 | `L8.B5.F.inv_std_fresh` | `layer8.block5.inv_std_fresh_sf` | `ctpt_attn_inv_std` | `fresh` | **30** | `F` | 4 | True | 8192 | 30 | `22,24,26,28,30` |  |
| 423 | `L8.B5.F.x_centered_fresh` | `layer8.block5.x_centered_fresh_sf` | `ctpt_attn_x_centered` | `fresh` | **30** | `F` | 4 | True | 8192 | 30 | `22,24,26,28,30` |  |
| 424 | `L8.B5.M.gamma` | `layer8.block5.gamma_sf` | `ctpt_gamma_attn` | `encoding` | **20** | `M` | 2 | True | 8192 | 20 | `16,18,20` |  |
| 425 | `L8.B5.W.wffn1` | `layer8.block5.wffn1_sf` | `ctpt_wffn1` | `encoding` | **22** | `W` | 4 | True | 8192 | 22 | `14,16,18,20,22` |  |
| 426 | `L8.B5.M.gelu_coeff` | `layer8.block5.gelu_coeff_sf` | `ctpt_gelu_coeff` | `encoding` | **20** | `M` | 2 | True | 8192 | 20 | `16,18,20` |  |
| 427 | `L8.B5.R.normalize_r` | `layer8.block5.normalize_rescale_sf` | `ctct_attn_normalize_rescale` | `rescale` |  | `R` | 0 | True | 8192 | 31 | `None,27,29,31` |  |
| 428 | `L8.B5.R.gamma_r` | `layer8.block5.gamma_rescale_sf` | `ctct_gamma_attn_rescale` | `rescale` | **30** | `R` | 3 | True | 8192 | 30 | `None,26,28,30` |  |
| 429 | `L8.B5.R.wffn1_r` | `layer8.block5.wffn1_rescale_sf` | `ctct_wffn1_rescale` | `rescale` |  | `R` | 0 | True | 8192 | 31 | `None,27,29,31` |  |
| 430 | `L8.B5.R.gp0` | `layer8.block5.gelu_power_rescale_sf_0` | `ctct_gelu_x2` | `rescale` | _off_ | `R` | 0 | False | 8192 | 31 | `None,27,29,31` | GELU degree 1 does not use this power-rescale slot |
| 431 | `L8.B5.K` | `layer8.block5.output_truncation_k` | `block5_output_truncation` | `truncation` | **13** | `K` | 3 | True | 8192 |  | `8,9,11,13,10,12` |  |
| 432 | `L9.B1.F.gelu_out` | `layer9.block1.gelu_out_sf` | `ctpt_gelu_out` | `fresh` | **30** | `F` | 4 | True | 8192 | 30 | `22,24,26,28,30` |  |
| 433 | `L9.B1.W.wffn2` | `layer9.block1.wffn2_sf` | `ctpt_ffn2` | `encoding` | **20** | `W` | 4 | True | 8192 | 20 | `12,14,16,18,20` |  |
| 434 | `L9.B1.S.mean_inv_d` | `layer9.block1.mean_inv_d_sf` | `ctpt_inv_d_1` | `scalar` | **20** | `S` | 2 | True | 8192 | 20 | `16,18,20` |  |
| 435 | `L9.B1.S.var_inv_d` | `layer9.block1.var_inv_d_sf` | `ctpt_inv_d_2` | `scalar` | **20** | `S` | 2 | True | 8192 | 20 | `16,18,20` |  |
| 436 | `L9.B1.R.mean_r` | `layer9.block1.mean_rescale_sf` | `ctct_mean_rescale` | `rescale` | **34** | `R` | 3 | True | 8192 | 34 | `None,30,32,34` |  |
| 437 | `L9.B1.R.var_r` | `layer9.block1.var_rescale_sf` | `ctct_var_rescale` | `rescale` | **34** | `R` | 3 | True | 8192 | 34 | `None,30,32,34` |  |
| 438 | `L9.B1.K` | `layer9.block1.output_truncation_k` | `block1_output_truncation` | `truncation` | **13** | `K` | 3 | True | 8192 |  | `8,9,11,13,10,12` |  |
| 439 | `L9.B2.F.inv_std_fresh` | `layer9.block2.inv_std_fresh_sf` | `ctpt_inv_std` | `fresh` | **23** | `F` | 0 | True | 16384 | 31 | `23,25,27,29,31` |  |
| 440 | `L9.B2.F.x_centered_fresh` | `layer9.block2.x_centered_fresh_sf` | `ctpt_x_centered` | `fresh` | **22** | `F` | 0 | True | 16384 | 30 | `22,24,26,28,30` |  |
| 441 | `L9.B2.M.gamma` | `layer9.block2.gamma_sf` | `ctpt_gamma` | `encoding` | **18** | `M` | 1 | True | 16384 | 20 | `16,18,20` |  |
| 442 | `L9.B2.W.wk` | `layer9.block2.wk_sf` | `ctpt_wq_wk` | `encoding` | **14** | `W` | 0 | True | 16384 | 22 | `14,16,18,20,22` |  |
| 443 | `L9.B2.W.wv` | `layer9.block2.wv_sf` | `ctpt_wv` | `encoding` | **22** | `W` | 4 | True | 16384 | 22 | `14,16,18,20,22` |  |
| 444 | `L9.B2.M.kt_mask1` | `layer9.block2.kt_mask1_sf` | `ctpt_rotKT_mask1` | `encoding` | **13** | `M` | 1 | True | 16384 | 15 | `11,13,15` |  |
| 445 | `L9.B2.M.kt_mask2` | `layer9.block2.kt_mask2_sf` | `ctpt_rotKT_mask2` | `encoding` | **13** | `M` | 1 | True | 16384 | 15 | `11,13,15` |  |
| 446 | `L9.B2.M.qkt_merge_mask` | `layer9.block2.qkt_merge_mask_sf` | `ctpt_mask` | `encoding` | **11** | `M` | 0 | True | 16384 | 15 | `11,13,15` |  |
| 447 | `L9.B2.R.gamma_r` | `layer9.block2.gamma_rescale_sf` | `ctct_gamma_rescale` | `rescale` |  | `R` | 0 | True | 16384 | 31 | `None,27,29,31` |  |
| 448 | `L9.B2.R.kt_mask2_r` | `layer9.block2.kt_mask2_rescale_sf` | `ctct_kt_mask2_rescale` | `rescale` | **31** | `R` | 3 | True | 16384 | 31 | `None,27,29,31` |  |
| 449 | `L9.B2.R.qkt_merge_mask_r` | `layer9.block2.qkt_merge_mask_rescale_sf` | `ctct_qkt_merge_mask_rescale` | `rescale` | **28** | `R` | 3 | True | 16384 | 28 | `None,24,26,28` |  |
| 450 | `L9.B2.K` | `layer9.block2.output_truncation_k` | `block2_output_truncation` | `truncation` | **10** | `K` | 4 | True | 16384 |  | `8,9,11,13,10,12` |  |
| 451 | `L9.B3.F.x_fresh` | `layer9.block3.x_fresh_sf` | `ctpt_softmax_x` | `fresh` | **28** | `F` | 4 | True | 16384 | 28 | `20,22,24,26,28` |  |
| 452 | `L9.B3.S.inv_2n` | `layer9.block3.inv_2n_sf` | `ctpt_softmax_inv_2n` | `scalar` | **15** | `S` | 2 | True | 16384 | 15 | `11,13,15` |  |
| 453 | `L9.B3.R.sq0` | `layer9.block3.square_rescale_sf_0` | `ctct_softmax_pow_s1` | `rescale` | **31** | `R` | 3 | True | 16384 | 31 | `None,27,29,31` |  |
| 454 | `L9.B3.R.sq1` | `layer9.block3.square_rescale_sf_1` | `ctct_softmax_pow_s2` | `rescale` | **31** | `R` | 3 | True | 16384 | 31 | `None,27,29,31` |  |
| 455 | `L9.B3.R.sq2` | `layer9.block3.square_rescale_sf_2` | `ctct_softmax_pow_s3` | `rescale` | **31** | `R` | 3 | True | 16384 | 31 | `None,27,29,31` |  |
| 456 | `L9.B3.R.sq3` | `layer9.block3.square_rescale_sf_3` | `ctct_softmax_pow_s4` | `rescale` | **31** | `R` | 3 | True | 16384 | 31 | `None,27,29,31` |  |
| 457 | `L9.B3.K` | `layer9.block3.output_truncation_k` | `block3_output_truncation` | `truncation` | **13** | `K` | 3 | True | 16384 |  | `8,9,11,13,10,12` |  |
| 458 | `L9.B4.F.softmax_out_fresh` | `layer9.block4.softmax_out_fresh_sf` | `ctpt_softmax_out` | `fresh` | **27** | `F` | 0 | True | 16384 | 35 | `27,29,31,33,35` |  |
| 459 | `L9.B4.F.v_fresh` | `layer9.block4.v_fresh_sf` | `ctpt_v` | `fresh` | **22** | `F` | 0 | True | 16384 | 30 | `22,24,26,28,30` |  |
| 460 | `L9.B4.M.softmax_out_mask` | `layer9.block4.softmax_out_mask_sf` | `ctpt_softmax_out_mask` | `encoding` | **12** | `M` | 1 | True | 16384 | 14 | `10,12,14` |  |
| 461 | `L9.B4.M.v_mask` | `layer9.block4.v_mask_sf` | `ctpt_v_mask` | `encoding` | **10** | `M` | 0 | True | 16384 | 14 | `10,12,14` |  |
| 462 | `L9.B4.M.softmax_v_mask` | `layer9.block4.softmax_v_mask_sf` | `ctpt_softmax_v_mask` | `encoding` | **10** | `M` | 0 | True | 16384 | 14 | `10,12,14` |  |
| 463 | `L9.B4.S.ln_mean_inv_d` | `layer9.block4.ln_mean_inv_d_sf` | `ctpt_inv_d_attn_mean` | `scalar` | **18** | `S` | 1 | True | 16384 | 20 | `16,18,20` |  |
| 464 | `L9.B4.S.ln_var_inv_d` | `layer9.block4.ln_var_inv_d_sf` | `ctpt_inv_d_attn_var` | `scalar` | **18** | `S` | 1 | True | 16384 | 20 | `16,18,20` |  |
| 465 | `L9.B4.W.wo` | `layer9.block4.wo_sf` | `ctpt_wo` | `encoding` | **22** | `W` | 4 | True | 16384 | 22 | `14,16,18,20,22` |  |
| 466 | `L9.B4.R.softmax_v_matmul_r` | `layer9.block4.softmax_v_matmul_rescale_sf` | `ctct_softmax_v_matmul_rescale` | `rescale` |  | `R` | 0 | True | 16384 | 31 | `None,27,29,31` |  |
| 467 | `L9.B4.R.ln_mean_r` | `layer9.block4.ln_mean_rescale_sf` | `ctct_attn_mean_rescale` | `rescale` | **31** | `R` | 3 | True | 16384 | 31 | `None,27,29,31` |  |
| 468 | `L9.B4.R.ln_var_r` | `layer9.block4.ln_var_rescale_sf` | `ctct_attn_var_rescale` | `rescale` | **24** | `R` | 1 | True | 16384 | 28 | `None,24,26,28` |  |
| 469 | `L9.B4.K` | `layer9.block4.output_truncation_k` | `block4_output_truncation` | `truncation` | **10** | `K` | 4 | True | 16384 |  | `8,9,11,13,10,12` |  |
| 470 | `L9.B5.F.inv_std_fresh` | `layer9.block5.inv_std_fresh_sf` | `ctpt_attn_inv_std` | `fresh` | **30** | `F` | 4 | True | 8192 | 30 | `22,24,26,28,30` |  |
| 471 | `L9.B5.F.x_centered_fresh` | `layer9.block5.x_centered_fresh_sf` | `ctpt_attn_x_centered` | `fresh` | **30** | `F` | 4 | True | 8192 | 30 | `22,24,26,28,30` |  |
| 472 | `L9.B5.M.gamma` | `layer9.block5.gamma_sf` | `ctpt_gamma_attn` | `encoding` | **20** | `M` | 2 | True | 8192 | 20 | `16,18,20` |  |
| 473 | `L9.B5.W.wffn1` | `layer9.block5.wffn1_sf` | `ctpt_wffn1` | `encoding` | **22** | `W` | 4 | True | 8192 | 22 | `14,16,18,20,22` |  |
| 474 | `L9.B5.M.gelu_coeff` | `layer9.block5.gelu_coeff_sf` | `ctpt_gelu_coeff` | `encoding` | **20** | `M` | 2 | True | 8192 | 20 | `16,18,20` |  |
| 475 | `L9.B5.R.normalize_r` | `layer9.block5.normalize_rescale_sf` | `ctct_attn_normalize_rescale` | `rescale` |  | `R` | 0 | True | 8192 | 31 | `None,27,29,31` |  |
| 476 | `L9.B5.R.gamma_r` | `layer9.block5.gamma_rescale_sf` | `ctct_gamma_attn_rescale` | `rescale` | **30** | `R` | 3 | True | 8192 | 30 | `None,26,28,30` |  |
| 477 | `L9.B5.R.wffn1_r` | `layer9.block5.wffn1_rescale_sf` | `ctct_wffn1_rescale` | `rescale` |  | `R` | 0 | True | 8192 | 31 | `None,27,29,31` |  |
| 478 | `L9.B5.R.gp0` | `layer9.block5.gelu_power_rescale_sf_0` | `ctct_gelu_x2` | `rescale` | _off_ | `R` | 0 | False | 8192 | 31 | `None,27,29,31` | GELU degree 1 does not use this power-rescale slot |
| 479 | `L9.B5.K` | `layer9.block5.output_truncation_k` | `block5_output_truncation` | `truncation` | **13** | `K` | 3 | True | 8192 |  | `8,9,11,13,10,12` |  |
| 480 | `L10.B1.F.gelu_out` | `layer10.block1.gelu_out_sf` | `ctpt_gelu_out` | `fresh` | **30** | `F` | 4 | True | 8192 | 30 | `22,24,26,28,30` |  |
| 481 | `L10.B1.W.wffn2` | `layer10.block1.wffn2_sf` | `ctpt_ffn2` | `encoding` | **20** | `W` | 4 | True | 8192 | 20 | `12,14,16,18,20` |  |
| 482 | `L10.B1.S.mean_inv_d` | `layer10.block1.mean_inv_d_sf` | `ctpt_inv_d_1` | `scalar` | **20** | `S` | 2 | True | 8192 | 20 | `16,18,20` |  |
| 483 | `L10.B1.S.var_inv_d` | `layer10.block1.var_inv_d_sf` | `ctpt_inv_d_2` | `scalar` | **20** | `S` | 2 | True | 8192 | 20 | `16,18,20` |  |
| 484 | `L10.B1.R.mean_r` | `layer10.block1.mean_rescale_sf` | `ctct_mean_rescale` | `rescale` | **34** | `R` | 3 | True | 8192 | 34 | `None,30,32,34` |  |
| 485 | `L10.B1.R.var_r` | `layer10.block1.var_rescale_sf` | `ctct_var_rescale` | `rescale` | **34** | `R` | 3 | True | 8192 | 34 | `None,30,32,34` |  |
| 486 | `L10.B1.K` | `layer10.block1.output_truncation_k` | `block1_output_truncation` | `truncation` | **13** | `K` | 3 | True | 8192 |  | `8,9,11,13,10,12` |  |
| 487 | `L10.B2.F.inv_std_fresh` | `layer10.block2.inv_std_fresh_sf` | `ctpt_inv_std` | `fresh` | **23** | `F` | 0 | True | 16384 | 31 | `23,25,27,29,31` |  |
| 488 | `L10.B2.F.x_centered_fresh` | `layer10.block2.x_centered_fresh_sf` | `ctpt_x_centered` | `fresh` | **22** | `F` | 0 | True | 16384 | 30 | `22,24,26,28,30` |  |
| 489 | `L10.B2.M.gamma` | `layer10.block2.gamma_sf` | `ctpt_gamma` | `encoding` | **18** | `M` | 1 | True | 16384 | 20 | `16,18,20` |  |
| 490 | `L10.B2.W.wk` | `layer10.block2.wk_sf` | `ctpt_wq_wk` | `encoding` | **14** | `W` | 0 | True | 16384 | 22 | `14,16,18,20,22` |  |
| 491 | `L10.B2.W.wv` | `layer10.block2.wv_sf` | `ctpt_wv` | `encoding` | **22** | `W` | 4 | True | 16384 | 22 | `14,16,18,20,22` |  |
| 492 | `L10.B2.M.kt_mask1` | `layer10.block2.kt_mask1_sf` | `ctpt_rotKT_mask1` | `encoding` | **13** | `M` | 1 | True | 16384 | 15 | `11,13,15` |  |
| 493 | `L10.B2.M.kt_mask2` | `layer10.block2.kt_mask2_sf` | `ctpt_rotKT_mask2` | `encoding` | **13** | `M` | 1 | True | 16384 | 15 | `11,13,15` |  |
| 494 | `L10.B2.M.qkt_merge_mask` | `layer10.block2.qkt_merge_mask_sf` | `ctpt_mask` | `encoding` | **11** | `M` | 0 | True | 16384 | 15 | `11,13,15` |  |
| 495 | `L10.B2.R.gamma_r` | `layer10.block2.gamma_rescale_sf` | `ctct_gamma_rescale` | `rescale` |  | `R` | 0 | True | 16384 | 31 | `None,27,29,31` |  |
| 496 | `L10.B2.R.kt_mask2_r` | `layer10.block2.kt_mask2_rescale_sf` | `ctct_kt_mask2_rescale` | `rescale` | **31** | `R` | 3 | True | 16384 | 31 | `None,27,29,31` |  |
| 497 | `L10.B2.R.qkt_merge_mask_r` | `layer10.block2.qkt_merge_mask_rescale_sf` | `ctct_qkt_merge_mask_rescale` | `rescale` | **24** | `R` | 1 | True | 16384 | 28 | `None,24,26,28` |  |
| 498 | `L10.B2.K` | `layer10.block2.output_truncation_k` | `block2_output_truncation` | `truncation` | **10** | `K` | 4 | True | 16384 |  | `8,9,11,13,10,12` |  |
| 499 | `L10.B3.F.x_fresh` | `layer10.block3.x_fresh_sf` | `ctpt_softmax_x` | `fresh` | **20** | `F` | 0 | True | 16384 | 28 | `20,22,24,26,28` |  |
| 500 | `L10.B3.S.inv_2n` | `layer10.block3.inv_2n_sf` | `ctpt_softmax_inv_2n` | `scalar` | **11** | `S` | 0 | True | 16384 | 15 | `11,13,15` |  |
| 501 | `L10.B3.R.sq0` | `layer10.block3.square_rescale_sf_0` | `ctct_softmax_pow_s1` | `rescale` | **31** | `R` | 3 | True | 16384 | 31 | `None,27,29,31` |  |
| 502 | `L10.B3.R.sq1` | `layer10.block3.square_rescale_sf_1` | `ctct_softmax_pow_s2` | `rescale` |  | `R` | 0 | True | 16384 | 31 | `None,27,29,31` |  |
| 503 | `L10.B3.R.sq2` | `layer10.block3.square_rescale_sf_2` | `ctct_softmax_pow_s3` | `rescale` | **31** | `R` | 3 | True | 16384 | 31 | `None,27,29,31` |  |
| 504 | `L10.B3.R.sq3` | `layer10.block3.square_rescale_sf_3` | `ctct_softmax_pow_s4` | `rescale` | **27** | `R` | 1 | True | 16384 | 31 | `None,27,29,31` |  |
| 505 | `L10.B3.K` | `layer10.block3.output_truncation_k` | `block3_output_truncation` | `truncation` | **12** | `K` | 5 | True | 16384 |  | `8,9,11,13,10,12` |  |
| 506 | `L10.B4.F.softmax_out_fresh` | `layer10.block4.softmax_out_fresh_sf` | `ctpt_softmax_out` | `fresh` | **27** | `F` | 0 | True | 16384 | 35 | `27,29,31,33,35` |  |
| 507 | `L10.B4.F.v_fresh` | `layer10.block4.v_fresh_sf` | `ctpt_v` | `fresh` | **22** | `F` | 0 | True | 16384 | 30 | `22,24,26,28,30` |  |
| 508 | `L10.B4.M.softmax_out_mask` | `layer10.block4.softmax_out_mask_sf` | `ctpt_softmax_out_mask` | `encoding` | **12** | `M` | 1 | True | 16384 | 14 | `10,12,14` |  |
| 509 | `L10.B4.M.v_mask` | `layer10.block4.v_mask_sf` | `ctpt_v_mask` | `encoding` | **10** | `M` | 0 | True | 16384 | 14 | `10,12,14` |  |
| 510 | `L10.B4.M.softmax_v_mask` | `layer10.block4.softmax_v_mask_sf` | `ctpt_softmax_v_mask` | `encoding` | **10** | `M` | 0 | True | 16384 | 14 | `10,12,14` |  |
| 511 | `L10.B4.S.ln_mean_inv_d` | `layer10.block4.ln_mean_inv_d_sf` | `ctpt_inv_d_attn_mean` | `scalar` | **18** | `S` | 1 | True | 16384 | 20 | `16,18,20` |  |
| 512 | `L10.B4.S.ln_var_inv_d` | `layer10.block4.ln_var_inv_d_sf` | `ctpt_inv_d_attn_var` | `scalar` | **18** | `S` | 1 | True | 16384 | 20 | `16,18,20` |  |
| 513 | `L10.B4.W.wo` | `layer10.block4.wo_sf` | `ctpt_wo` | `encoding` | **22** | `W` | 4 | True | 16384 | 22 | `14,16,18,20,22` |  |
| 514 | `L10.B4.R.softmax_v_matmul_r` | `layer10.block4.softmax_v_matmul_rescale_sf` | `ctct_softmax_v_matmul_rescale` | `rescale` |  | `R` | 0 | True | 16384 | 31 | `None,27,29,31` |  |
| 515 | `L10.B4.R.ln_mean_r` | `layer10.block4.ln_mean_rescale_sf` | `ctct_attn_mean_rescale` | `rescale` | **29** | `R` | 2 | True | 16384 | 31 | `None,27,29,31` |  |
| 516 | `L10.B4.R.ln_var_r` | `layer10.block4.ln_var_rescale_sf` | `ctct_attn_var_rescale` | `rescale` | **24** | `R` | 1 | True | 16384 | 28 | `None,24,26,28` |  |
| 517 | `L10.B4.K` | `layer10.block4.output_truncation_k` | `block4_output_truncation` | `truncation` | **10** | `K` | 4 | True | 16384 |  | `8,9,11,13,10,12` |  |
| 518 | `L10.B5.F.inv_std_fresh` | `layer10.block5.inv_std_fresh_sf` | `ctpt_attn_inv_std` | `fresh` | **30** | `F` | 4 | True | 8192 | 30 | `22,24,26,28,30` |  |
| 519 | `L10.B5.F.x_centered_fresh` | `layer10.block5.x_centered_fresh_sf` | `ctpt_attn_x_centered` | `fresh` | **30** | `F` | 4 | True | 8192 | 30 | `22,24,26,28,30` |  |
| 520 | `L10.B5.M.gamma` | `layer10.block5.gamma_sf` | `ctpt_gamma_attn` | `encoding` | **20** | `M` | 2 | True | 8192 | 20 | `16,18,20` |  |
| 521 | `L10.B5.W.wffn1` | `layer10.block5.wffn1_sf` | `ctpt_wffn1` | `encoding` | **22** | `W` | 4 | True | 8192 | 22 | `14,16,18,20,22` |  |
| 522 | `L10.B5.M.gelu_coeff` | `layer10.block5.gelu_coeff_sf` | `ctpt_gelu_coeff` | `encoding` | **20** | `M` | 2 | True | 8192 | 20 | `16,18,20` |  |
| 523 | `L10.B5.R.normalize_r` | `layer10.block5.normalize_rescale_sf` | `ctct_attn_normalize_rescale` | `rescale` |  | `R` | 0 | True | 8192 | 31 | `None,27,29,31` |  |
| 524 | `L10.B5.R.gamma_r` | `layer10.block5.gamma_rescale_sf` | `ctct_gamma_attn_rescale` | `rescale` | **30** | `R` | 3 | True | 8192 | 30 | `None,26,28,30` |  |
| 525 | `L10.B5.R.wffn1_r` | `layer10.block5.wffn1_rescale_sf` | `ctct_wffn1_rescale` | `rescale` |  | `R` | 0 | True | 8192 | 31 | `None,27,29,31` |  |
| 526 | `L10.B5.R.gp0` | `layer10.block5.gelu_power_rescale_sf_0` | `ctct_gelu_x2` | `rescale` | _off_ | `R` | 0 | False | 8192 | 31 | `None,27,29,31` | GELU degree 1 does not use this power-rescale slot |
| 527 | `L10.B5.K` | `layer10.block5.output_truncation_k` | `block5_output_truncation` | `truncation` | **13** | `K` | 3 | True | 8192 |  | `8,9,11,13,10,12` |  |
| 528 | `L11.B1.F.gelu_out` | `layer11.block1.gelu_out_sf` | `ctpt_gelu_out` | `fresh` | **30** | `F` | 4 | True | 8192 | 30 | `22,24,26,28,30` |  |
| 529 | `L11.B1.W.wffn2` | `layer11.block1.wffn2_sf` | `ctpt_ffn2` | `encoding` | **20** | `W` | 4 | True | 8192 | 20 | `12,14,16,18,20` |  |
| 530 | `L11.B1.S.mean_inv_d` | `layer11.block1.mean_inv_d_sf` | `ctpt_inv_d_1` | `scalar` | **20** | `S` | 2 | True | 8192 | 20 | `16,18,20` |  |
| 531 | `L11.B1.S.var_inv_d` | `layer11.block1.var_inv_d_sf` | `ctpt_inv_d_2` | `scalar` | **20** | `S` | 2 | True | 8192 | 20 | `16,18,20` |  |
| 532 | `L11.B1.R.mean_r` | `layer11.block1.mean_rescale_sf` | `ctct_mean_rescale` | `rescale` | **34** | `R` | 3 | True | 8192 | 34 | `None,30,32,34` |  |
| 533 | `L11.B1.R.var_r` | `layer11.block1.var_rescale_sf` | `ctct_var_rescale` | `rescale` | **34** | `R` | 3 | True | 8192 | 34 | `None,30,32,34` |  |
| 534 | `L11.B1.K` | `layer11.block1.output_truncation_k` | `block1_output_truncation` | `truncation` | **13** | `K` | 3 | True | 8192 |  | `8,9,11,13,10,12` |  |
| 535 | `L11.B2.F.inv_std_fresh` | `layer11.block2.inv_std_fresh_sf` | `ctpt_inv_std` | `fresh` | **23** | `F` | 0 | True | 16384 | 31 | `23,25,27,29,31` |  |
| 536 | `L11.B2.F.x_centered_fresh` | `layer11.block2.x_centered_fresh_sf` | `ctpt_x_centered` | `fresh` | **22** | `F` | 0 | True | 16384 | 30 | `22,24,26,28,30` |  |
| 537 | `L11.B2.M.gamma` | `layer11.block2.gamma_sf` | `ctpt_gamma` | `encoding` | **18** | `M` | 1 | True | 16384 | 20 | `16,18,20` |  |
| 538 | `L11.B2.W.wk` | `layer11.block2.wk_sf` | `ctpt_wq_wk` | `encoding` | **14** | `W` | 0 | True | 16384 | 22 | `14,16,18,20,22` |  |
| 539 | `L11.B2.W.wv` | `layer11.block2.wv_sf` | `ctpt_wv` | `encoding` | **22** | `W` | 4 | True | 16384 | 22 | `14,16,18,20,22` |  |
| 540 | `L11.B2.M.kt_mask1` | `layer11.block2.kt_mask1_sf` | `ctpt_rotKT_mask1` | `encoding` | **13** | `M` | 1 | True | 16384 | 15 | `11,13,15` |  |
| 541 | `L11.B2.M.kt_mask2` | `layer11.block2.kt_mask2_sf` | `ctpt_rotKT_mask2` | `encoding` | **13** | `M` | 1 | True | 16384 | 15 | `11,13,15` |  |
| 542 | `L11.B2.M.qkt_merge_mask` | `layer11.block2.qkt_merge_mask_sf` | `ctpt_mask` | `encoding` | **11** | `M` | 0 | True | 16384 | 15 | `11,13,15` |  |
| 543 | `L11.B2.R.gamma_r` | `layer11.block2.gamma_rescale_sf` | `ctct_gamma_rescale` | `rescale` |  | `R` | 0 | True | 16384 | 31 | `None,27,29,31` |  |
| 544 | `L11.B2.R.kt_mask2_r` | `layer11.block2.kt_mask2_rescale_sf` | `ctct_kt_mask2_rescale` | `rescale` | **31** | `R` | 3 | True | 16384 | 31 | `None,27,29,31` |  |
| 545 | `L11.B2.R.qkt_merge_mask_r` | `layer11.block2.qkt_merge_mask_rescale_sf` | `ctct_qkt_merge_mask_rescale` | `rescale` | **24** | `R` | 1 | True | 16384 | 28 | `None,24,26,28` |  |
| 546 | `L11.B2.K` | `layer11.block2.output_truncation_k` | `block2_output_truncation` | `truncation` | **10** | `K` | 4 | True | 16384 |  | `8,9,11,13,10,12` |  |
| 547 | `L11.B3.F.x_fresh` | `layer11.block3.x_fresh_sf` | `ctpt_softmax_x` | `fresh` | **27** | `F` | 4 | True | 8192 | 27 | `19,21,23,25,27` |  |
| 548 | `L11.B3.S.inv_2n` | `layer11.block3.inv_2n_sf` | `ctpt_softmax_inv_2n` | `scalar` | **16** | `S` | 2 | True | 8192 | 16 | `12,14,16` |  |
| 549 | `L11.B3.R.sq0` | `layer11.block3.square_rescale_sf_0` | `ctct_softmax_pow_s1` | `rescale` | **34** | `R` | 3 | True | 8192 | 34 | `None,30,32,34` |  |
| 550 | `L11.B3.R.sq1` | `layer11.block3.square_rescale_sf_1` | `ctct_softmax_pow_s2` | `rescale` | **34** | `R` | 3 | True | 8192 | 34 | `None,30,32,34` |  |
| 551 | `L11.B3.R.sq2` | `layer11.block3.square_rescale_sf_2` | `ctct_softmax_pow_s3` | `rescale` | _off_ | `R` | 0 | False | 8192 | 31 | `None,27,29,31` | softmax degree 2 does not use this square-rescale slot |
| 552 | `L11.B3.R.sq3` | `layer11.block3.square_rescale_sf_3` | `ctct_softmax_pow_s4` | `rescale` | _off_ | `R` | 0 | False | 8192 | 31 | `None,27,29,31` | softmax degree 2 does not use this square-rescale slot |
| 553 | `L11.B3.K` | `layer11.block3.output_truncation_k` | `block3_output_truncation` | `truncation` | **13** | `K` | 3 | True | 8192 |  | `8,9,11,13,10,12` |  |
| 554 | `L11.B4.F.softmax_out_fresh` | `layer11.block4.softmax_out_fresh_sf` | `ctpt_softmax_out` | `fresh` | **27** | `F` | 0 | True | 16384 | 35 | `27,29,31,33,35` |  |
| 555 | `L11.B4.F.v_fresh` | `layer11.block4.v_fresh_sf` | `ctpt_v` | `fresh` | **22** | `F` | 0 | True | 16384 | 30 | `22,24,26,28,30` |  |
| 556 | `L11.B4.M.softmax_out_mask` | `layer11.block4.softmax_out_mask_sf` | `ctpt_softmax_out_mask` | `encoding` | **12** | `M` | 1 | True | 16384 | 14 | `10,12,14` |  |
| 557 | `L11.B4.M.v_mask` | `layer11.block4.v_mask_sf` | `ctpt_v_mask` | `encoding` | **10** | `M` | 0 | True | 16384 | 14 | `10,12,14` |  |
| 558 | `L11.B4.M.softmax_v_mask` | `layer11.block4.softmax_v_mask_sf` | `ctpt_softmax_v_mask` | `encoding` | **10** | `M` | 0 | True | 16384 | 14 | `10,12,14` |  |
| 559 | `L11.B4.S.ln_mean_inv_d` | `layer11.block4.ln_mean_inv_d_sf` | `ctpt_inv_d_attn_mean` | `scalar` | **18** | `S` | 1 | True | 16384 | 20 | `16,18,20` |  |
| 560 | `L11.B4.S.ln_var_inv_d` | `layer11.block4.ln_var_inv_d_sf` | `ctpt_inv_d_attn_var` | `scalar` | **18** | `S` | 1 | True | 16384 | 20 | `16,18,20` |  |
| 561 | `L11.B4.W.wo` | `layer11.block4.wo_sf` | `ctpt_wo` | `encoding` | **22** | `W` | 4 | True | 16384 | 22 | `14,16,18,20,22` |  |
| 562 | `L11.B4.R.softmax_v_matmul_r` | `layer11.block4.softmax_v_matmul_rescale_sf` | `ctct_softmax_v_matmul_rescale` | `rescale` |  | `R` | 0 | True | 16384 | 31 | `None,27,29,31` |  |
| 563 | `L11.B4.R.ln_mean_r` | `layer11.block4.ln_mean_rescale_sf` | `ctct_attn_mean_rescale` | `rescale` | **31** | `R` | 3 | True | 16384 | 31 | `None,27,29,31` |  |
| 564 | `L11.B4.R.ln_var_r` | `layer11.block4.ln_var_rescale_sf` | `ctct_attn_var_rescale` | `rescale` | **24** | `R` | 1 | True | 16384 | 28 | `None,24,26,28` |  |
| 565 | `L11.B4.K` | `layer11.block4.output_truncation_k` | `block4_output_truncation` | `truncation` | **10** | `K` | 4 | True | 16384 |  | `8,9,11,13,10,12` |  |
| 566 | `L11.B5.F.inv_std_fresh` | `layer11.block5.inv_std_fresh_sf` | `ctpt_attn_inv_std` | `fresh` | **30** | `F` | 4 | True | 8192 | 30 | `22,24,26,28,30` |  |
| 567 | `L11.B5.F.x_centered_fresh` | `layer11.block5.x_centered_fresh_sf` | `ctpt_attn_x_centered` | `fresh` | **30** | `F` | 4 | True | 8192 | 30 | `22,24,26,28,30` |  |
| 568 | `L11.B5.M.gamma` | `layer11.block5.gamma_sf` | `ctpt_gamma_attn` | `encoding` | **20** | `M` | 2 | True | 8192 | 20 | `16,18,20` |  |
| 569 | `L11.B5.W.wffn1` | `layer11.block5.wffn1_sf` | `ctpt_wffn1` | `encoding` | **22** | `W` | 4 | True | 8192 | 22 | `14,16,18,20,22` |  |
| 570 | `L11.B5.M.gelu_coeff` | `layer11.block5.gelu_coeff_sf` | `ctpt_gelu_coeff` | `encoding` | **20** | `M` | 2 | True | 8192 | 20 | `16,18,20` |  |
| 571 | `L11.B5.R.normalize_r` | `layer11.block5.normalize_rescale_sf` | `ctct_attn_normalize_rescale` | `rescale` |  | `R` | 0 | True | 8192 | 31 | `None,27,29,31` |  |
| 572 | `L11.B5.R.gamma_r` | `layer11.block5.gamma_rescale_sf` | `ctct_gamma_attn_rescale` | `rescale` | **30** | `R` | 3 | True | 8192 | 30 | `None,26,28,30` |  |
| 573 | `L11.B5.R.wffn1_r` | `layer11.block5.wffn1_rescale_sf` | `ctct_wffn1_rescale` | `rescale` |  | `R` | 0 | True | 8192 | 31 | `None,27,29,31` |  |
| 574 | `L11.B5.R.gp0` | `layer11.block5.gelu_power_rescale_sf_0` | `ctct_gelu_x2` | `rescale` | _off_ | `R` | 0 | False | 8192 | 31 | `None,27,29,31` | GELU degree 1 does not use this power-rescale slot |
| 575 | `L11.B5.K` | `layer11.block5.output_truncation_k` | `block5_output_truncation` | `truncation` | **13** | `K` | 3 | True | 8192 |  | `8,9,11,13,10,12` |  |
| 576 | `L0.first_input.F` | `first_input.layer0` | `first_input_fresh` | `fresh` | _off_ | `F` | 0 | False | 8192 | 30 | `22,24,26,28,30` | first_input fresh noise deprecated; the first HE config is treated as lossless. Slot kept for action-vector backward compatibility. |

> JSON 配对文件：`blb_stage2_best_action_full.json`。可以直接喂给 `Paean/run_final_eval.sh --action-config`。