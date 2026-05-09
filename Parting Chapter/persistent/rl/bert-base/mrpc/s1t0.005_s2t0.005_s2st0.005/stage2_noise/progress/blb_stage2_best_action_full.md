# BLB Stage 2 action description: best

- profile: `mrpc`
- num_layers: `12`
- action_length: `877`
- records: `877`
- scaling factor slots: `817`
- truncation slots: `60`
- ineffective decoded slots: `77`

Slot label format: `L{layer}.B{block}.{kind}[.{short_field}]` (kind: F=fresh, W=weight encode, M=mask, S=scalar, R=rescale, K=trunc).

| idx | slot | location | operation | dist | action_idx | value_type | value | effective | N | max_sf | note |
|---:|---|---|---|---|---:|---|---:|---|---:|---:|---|
| 0 | `L0.B1.F.gelu_out` | `layer0.block1.gelu_out_sf` | `ctpt_gelu_out` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 1 | `L0.B1.W.wffn2` | `layer0.block1.wffn2_sf` | `ctpt_ffn2` | `encoding` | 4 | `scaling_factor` | 20 | True | 8192 | 20 |  |
| 2 | `L0.B1.S.mean_inv_d` | `layer0.block1.mean_inv_d_sf` | `ctpt_inv_d_1` | `scalar` | 2 | `scaling_factor` | 20 | True | 8192 | 20 |  |
| 3 | `L0.B1.S.var_inv_d` | `layer0.block1.var_inv_d_sf` | `ctpt_inv_d_2` | `scalar` | 2 | `scaling_factor` | 20 | True | 8192 | 20 |  |
| 4 | `L0.B1.R.wffn2_r` | `layer0.block1.wffn2_rescale_sf` | `ctct_ffn2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 5 | `L0.B1.R.mean_r` | `layer0.block1.mean_rescale_sf` | `ctct_mean_rescale` | `rescale` | 3 | `scaling_factor` | 34 | True | 8192 | 34 |  |
| 6 | `L0.B1.R.square_r` | `layer0.block1.square_rescale_sf` | `ctct_ext_square` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 7 | `L0.B1.R.var_r` | `layer0.block1.var_rescale_sf` | `ctct_var_rescale` | `rescale` | 3 | `scaling_factor` | 34 | True | 8192 | 34 |  |
| 8 | `L0.B1.K` | `layer0.block1.output_truncation_k` | `block1_output_truncation` | `truncation` | 3 | `truncation_k` |  | False | 8192 |  | layer0.block1 has no input-side truncation point; decoded cfg uses None |
| 9 | `L0.B2.F.inv_std_fresh` | `layer0.block2.inv_std_fresh_sf` | `ctpt_inv_std` | `fresh` | 4 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 10 | `L0.B2.F.x_centered_fresh` | `layer0.block2.x_centered_fresh_sf` | `ctpt_x_centered` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 11 | `L0.B2.M.gamma` | `layer0.block2.gamma_sf` | `ctpt_gamma` | `encoding` | 1 | `scaling_factor` | 18 | True | 16384 | 20 |  |
| 12 | `L0.B2.W.wq` | `layer0.block2.wq_sf` | `ctpt_wq` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 13 | `L0.B2.W.wk` | `layer0.block2.wk_sf` | `ctpt_wk` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 14 | `L0.B2.W.wv` | `layer0.block2.wv_sf` | `ctpt_wv` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 15 | `L0.B2.M.kt_mask1` | `layer0.block2.kt_mask1_sf` | `ctpt_kt_mask1` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 16 | `L0.B2.M.kt_mask2` | `layer0.block2.kt_mask2_sf` | `ctpt_kt_mask2` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 17 | `L0.B2.M.q_mask1` | `layer0.block2.q_mask1_sf` | `ctpt_q_mask1` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 18 | `L0.B2.M.q_mask2` | `layer0.block2.q_mask2_sf` | `ctpt_q_mask2` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 19 | `L0.B2.M.qkt_merge_mask` | `layer0.block2.qkt_merge_mask_sf` | `ctpt_qkt_merge_mask` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 20 | `L0.B2.R.normalize_r` | `layer0.block2.normalize_rescale_sf` | `ctct_normalize_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 21 | `L0.B2.R.gamma_r` | `layer0.block2.gamma_rescale_sf` | `ctct_gamma_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 22 | `L0.B2.R.wk_r` | `layer0.block2.wk_rescale_sf` | `ctct_wk_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 23 | `L0.B2.R.wq_r` | `layer0.block2.wq_rescale_sf` | `ctct_wq_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 24 | `L0.B2.R.wv_r` | `layer0.block2.wv_rescale_sf` | `ctct_wv_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 25 | `L0.B2.R.kt_mask1_r` | `layer0.block2.kt_mask1_rescale_sf` | `ctct_kt_mask1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 26 | `L0.B2.R.kt_mask2_r` | `layer0.block2.kt_mask2_rescale_sf` | `ctct_kt_mask2_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 27 | `L0.B2.R.q_mask1_r` | `layer0.block2.q_mask1_rescale_sf` | `ctct_q_mask1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 28 | `L0.B2.R.q_mask2_r` | `layer0.block2.q_mask2_rescale_sf` | `ctct_q_mask2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 29 | `L0.B2.R.qkt_matmul_r` | `layer0.block2.qkt_matmul_rescale_sf` | `ctct_qkt_matmul_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 30 | `L0.B2.R.qkt_merge_mask_r` | `layer0.block2.qkt_merge_mask_rescale_sf` | `ctct_qkt_merge_mask_rescale` | `rescale` | 3 | `scaling_factor` | 28 | True | 16384 | 28 |  |
| 31 | `L0.B2.K` | `layer0.block2.output_truncation_k` | `block2_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 32 | `L0.B3.F.x_fresh` | `layer0.block3.x_fresh_sf` | `ctpt_softmax_x` | `fresh` | 4 | `scaling_factor` | 28 | True | 8192 | 28 |  |
| 33 | `L0.B3.S.inv_2n` | `layer0.block3.inv_2n_sf` | `ctpt_softmax_inv_2n` | `scalar` | 2 | `scaling_factor` | 15 | True | 8192 | 15 |  |
| 34 | `L0.B3.R.x_inv_2n_r` | `layer0.block3.x_inv_2n_rescale_sf` | `ctct_softmax_x_inv_2n_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 35 | `L0.B3.R.sq0` | `layer0.block3.square_rescale_sf_0` | `ctct_softmax_pow_s1` | `rescale` | 3 | `scaling_factor` | 31 | True | 8192 | 31 |  |
| 36 | `L0.B3.R.sq1` | `layer0.block3.square_rescale_sf_1` | `ctct_softmax_pow_s2` | `rescale` | 3 | `scaling_factor` | 31 | True | 8192 | 31 |  |
| 37 | `L0.B3.R.sq2` | `layer0.block3.square_rescale_sf_2` | `ctct_softmax_pow_s3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 31 | softmax degree 2 does not use this square-rescale slot |
| 38 | `L0.B3.R.sq3` | `layer0.block3.square_rescale_sf_3` | `ctct_softmax_pow_s4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 31 | softmax degree 2 does not use this square-rescale slot |
| 39 | `L0.B3.K` | `layer0.block3.output_truncation_k` | `block3_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 40 | `L0.B4.F.softmax_out_fresh` | `layer0.block4.softmax_out_fresh_sf` | `ctpt_softmax_out` | `fresh` | 4 | `scaling_factor` | 35 | True | 16384 | 35 |  |
| 41 | `L0.B4.F.v_fresh` | `layer0.block4.v_fresh_sf` | `ctpt_v` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 42 | `L0.B4.M.softmax_out_mask` | `layer0.block4.softmax_out_mask_sf` | `ctpt_softmax_out_mask` | `encoding` | 1 | `scaling_factor` | 12 | True | 16384 | 14 |  |
| 43 | `L0.B4.M.v_mask` | `layer0.block4.v_mask_sf` | `ctpt_v_mask` | `encoding` | 1 | `scaling_factor` | 20 | True | 16384 | 22 |  |
| 44 | `L0.B4.M.softmax_v_mask` | `layer0.block4.softmax_v_mask_sf` | `ctpt_softmax_v_mask` | `encoding` | 1 | `scaling_factor` | 12 | True | 16384 | 14 |  |
| 45 | `L0.B4.S.ln_mean_inv_d` | `layer0.block4.ln_mean_inv_d_sf` | `ctpt_inv_d_attn_mean` | `scalar` | 2 | `scaling_factor` | 20 | True | 16384 | 20 |  |
| 46 | `L0.B4.S.ln_var_inv_d` | `layer0.block4.ln_var_inv_d_sf` | `ctpt_inv_d_attn_var` | `scalar` | 2 | `scaling_factor` | 20 | True | 16384 | 20 |  |
| 47 | `L0.B4.W.wo` | `layer0.block4.wo_sf` | `ctpt_wo` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 48 | `L0.B4.R.softmax_out_mask_r` | `layer0.block4.softmax_out_mask_rescale_sf` | `ctct_softmax_out_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 49 | `L0.B4.R.v_mask_r` | `layer0.block4.v_mask_rescale_sf` | `ctct_v_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 50 | `L0.B4.R.softmax_v_matmul_r` | `layer0.block4.softmax_v_matmul_rescale_sf` | `ctct_softmax_v_matmul_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 51 | `L0.B4.R.softmax_v_mask_r` | `layer0.block4.softmax_v_mask_rescale_sf` | `ctct_softmax_v_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 52 | `L0.B4.R.wo_r` | `layer0.block4.wo_rescale_sf` | `ctct_wo_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 53 | `L0.B4.R.ln_mean_r` | `layer0.block4.ln_mean_rescale_sf` | `ctct_attn_mean_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 54 | `L0.B4.R.ln_square_r` | `layer0.block4.ln_square_rescale_sf` | `ctct_attn_square_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 55 | `L0.B4.R.ln_var_r` | `layer0.block4.ln_var_rescale_sf` | `ctct_attn_var_rescale` | `rescale` | 3 | `scaling_factor` | 28 | True | 16384 | 28 |  |
| 56 | `L0.B4.K` | `layer0.block4.output_truncation_k` | `block4_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 57 | `L0.B5.F.inv_std_fresh` | `layer0.block5.inv_std_fresh_sf` | `ctpt_attn_inv_std` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 58 | `L0.B5.F.x_centered_fresh` | `layer0.block5.x_centered_fresh_sf` | `ctpt_attn_x_centered` | `fresh` | 4 | `scaling_factor` | 31 | True | 8192 | 31 |  |
| 59 | `L0.B5.M.gamma` | `layer0.block5.gamma_sf` | `ctpt_gamma_attn` | `encoding` | 1 | `scaling_factor` | 18 | True | 8192 | 20 |  |
| 60 | `L0.B5.W.wffn1` | `layer0.block5.wffn1_sf` | `ctpt_wffn1` | `encoding` | 4 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 61 | `L0.B5.M.gelu_coeff` | `layer0.block5.gelu_coeff_sf` | `ctpt_gelu_coeff` | `encoding` | 1 | `scaling_factor` | 18 | True | 8192 | 20 |  |
| 62 | `L0.B5.R.normalize_r` | `layer0.block5.normalize_rescale_sf` | `ctct_attn_normalize_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 8192 | 31 |  |
| 63 | `L0.B5.R.gamma_r` | `layer0.block5.gamma_rescale_sf` | `ctct_gamma_attn_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 64 | `L0.B5.R.wffn1_r` | `layer0.block5.wffn1_rescale_sf` | `ctct_wffn1_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 8192 | 31 |  |
| 65 | `L0.B5.R.gp0` | `layer0.block5.gelu_power_rescale_sf_0` | `ctct_gelu_x2` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 31 | GELU degree 1 does not use this power-rescale slot |
| 66 | `L0.B5.R.gp1` | `layer0.block5.gelu_power_rescale_sf_1` | `ctct_gelu_x3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 67 | `L0.B5.R.gp2` | `layer0.block5.gelu_power_rescale_sf_2` | `ctct_gelu_x4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 68 | `L0.B5.R.gc0` | `layer0.block5.gelu_coeff_mul_rescale_sf_0` | `ctct_gelu_b_x` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 69 | `L0.B5.R.gc1` | `layer0.block5.gelu_coeff_mul_rescale_sf_1` | `ctct_gelu_c_x2` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 70 | `L0.B5.R.gc2` | `layer0.block5.gelu_coeff_mul_rescale_sf_2` | `ctct_gelu_d_x3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 71 | `L0.B5.R.gc3` | `layer0.block5.gelu_coeff_mul_rescale_sf_3` | `ctct_gelu_e_x4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 31 | GELU degree 1 does not use this coefficient-rescale slot |
| 72 | `L0.B5.K` | `layer0.block5.output_truncation_k` | `block5_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 73 | `L1.B1.F.gelu_out` | `layer1.block1.gelu_out_sf` | `ctpt_gelu_out` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 74 | `L1.B1.W.wffn2` | `layer1.block1.wffn2_sf` | `ctpt_ffn2` | `encoding` | 4 | `scaling_factor` | 20 | True | 8192 | 20 |  |
| 75 | `L1.B1.S.mean_inv_d` | `layer1.block1.mean_inv_d_sf` | `ctpt_inv_d_1` | `scalar` | 2 | `scaling_factor` | 20 | True | 8192 | 20 |  |
| 76 | `L1.B1.S.var_inv_d` | `layer1.block1.var_inv_d_sf` | `ctpt_inv_d_2` | `scalar` | 2 | `scaling_factor` | 20 | True | 8192 | 20 |  |
| 77 | `L1.B1.R.wffn2_r` | `layer1.block1.wffn2_rescale_sf` | `ctct_ffn2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 78 | `L1.B1.R.mean_r` | `layer1.block1.mean_rescale_sf` | `ctct_mean_rescale` | `rescale` | 3 | `scaling_factor` | 34 | True | 8192 | 34 |  |
| 79 | `L1.B1.R.square_r` | `layer1.block1.square_rescale_sf` | `ctct_ext_square` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 80 | `L1.B1.R.var_r` | `layer1.block1.var_rescale_sf` | `ctct_var_rescale` | `rescale` | 3 | `scaling_factor` | 34 | True | 8192 | 34 |  |
| 81 | `L1.B1.K` | `layer1.block1.output_truncation_k` | `block1_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 82 | `L1.B2.F.inv_std_fresh` | `layer1.block2.inv_std_fresh_sf` | `ctpt_inv_std` | `fresh` | 4 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 83 | `L1.B2.F.x_centered_fresh` | `layer1.block2.x_centered_fresh_sf` | `ctpt_x_centered` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 84 | `L1.B2.M.gamma` | `layer1.block2.gamma_sf` | `ctpt_gamma` | `encoding` | 1 | `scaling_factor` | 18 | True | 16384 | 20 |  |
| 85 | `L1.B2.W.wq` | `layer1.block2.wq_sf` | `ctpt_wq` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 86 | `L1.B2.W.wk` | `layer1.block2.wk_sf` | `ctpt_wk` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 87 | `L1.B2.W.wv` | `layer1.block2.wv_sf` | `ctpt_wv` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 88 | `L1.B2.M.kt_mask1` | `layer1.block2.kt_mask1_sf` | `ctpt_kt_mask1` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 89 | `L1.B2.M.kt_mask2` | `layer1.block2.kt_mask2_sf` | `ctpt_kt_mask2` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 90 | `L1.B2.M.q_mask1` | `layer1.block2.q_mask1_sf` | `ctpt_q_mask1` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 91 | `L1.B2.M.q_mask2` | `layer1.block2.q_mask2_sf` | `ctpt_q_mask2` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 92 | `L1.B2.M.qkt_merge_mask` | `layer1.block2.qkt_merge_mask_sf` | `ctpt_qkt_merge_mask` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 93 | `L1.B2.R.normalize_r` | `layer1.block2.normalize_rescale_sf` | `ctct_normalize_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 94 | `L1.B2.R.gamma_r` | `layer1.block2.gamma_rescale_sf` | `ctct_gamma_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 95 | `L1.B2.R.wk_r` | `layer1.block2.wk_rescale_sf` | `ctct_wk_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 96 | `L1.B2.R.wq_r` | `layer1.block2.wq_rescale_sf` | `ctct_wq_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 97 | `L1.B2.R.wv_r` | `layer1.block2.wv_rescale_sf` | `ctct_wv_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 98 | `L1.B2.R.kt_mask1_r` | `layer1.block2.kt_mask1_rescale_sf` | `ctct_kt_mask1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 99 | `L1.B2.R.kt_mask2_r` | `layer1.block2.kt_mask2_rescale_sf` | `ctct_kt_mask2_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 100 | `L1.B2.R.q_mask1_r` | `layer1.block2.q_mask1_rescale_sf` | `ctct_q_mask1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 101 | `L1.B2.R.q_mask2_r` | `layer1.block2.q_mask2_rescale_sf` | `ctct_q_mask2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 102 | `L1.B2.R.qkt_matmul_r` | `layer1.block2.qkt_matmul_rescale_sf` | `ctct_qkt_matmul_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 103 | `L1.B2.R.qkt_merge_mask_r` | `layer1.block2.qkt_merge_mask_rescale_sf` | `ctct_qkt_merge_mask_rescale` | `rescale` | 3 | `scaling_factor` | 28 | True | 16384 | 28 |  |
| 104 | `L1.B2.K` | `layer1.block2.output_truncation_k` | `block2_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 105 | `L1.B3.F.x_fresh` | `layer1.block3.x_fresh_sf` | `ctpt_softmax_x` | `fresh` | 4 | `scaling_factor` | 28 | True | 8192 | 28 |  |
| 106 | `L1.B3.S.inv_2n` | `layer1.block3.inv_2n_sf` | `ctpt_softmax_inv_2n` | `scalar` | 2 | `scaling_factor` | 15 | True | 8192 | 15 |  |
| 107 | `L1.B3.R.x_inv_2n_r` | `layer1.block3.x_inv_2n_rescale_sf` | `ctct_softmax_x_inv_2n_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 108 | `L1.B3.R.sq0` | `layer1.block3.square_rescale_sf_0` | `ctct_softmax_pow_s1` | `rescale` | 3 | `scaling_factor` | 31 | True | 8192 | 31 |  |
| 109 | `L1.B3.R.sq1` | `layer1.block3.square_rescale_sf_1` | `ctct_softmax_pow_s2` | `rescale` | 3 | `scaling_factor` | 31 | True | 8192 | 31 |  |
| 110 | `L1.B3.R.sq2` | `layer1.block3.square_rescale_sf_2` | `ctct_softmax_pow_s3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 31 | softmax degree 2 does not use this square-rescale slot |
| 111 | `L1.B3.R.sq3` | `layer1.block3.square_rescale_sf_3` | `ctct_softmax_pow_s4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 31 | softmax degree 2 does not use this square-rescale slot |
| 112 | `L1.B3.K` | `layer1.block3.output_truncation_k` | `block3_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 113 | `L1.B4.F.softmax_out_fresh` | `layer1.block4.softmax_out_fresh_sf` | `ctpt_softmax_out` | `fresh` | 4 | `scaling_factor` | 35 | True | 16384 | 35 |  |
| 114 | `L1.B4.F.v_fresh` | `layer1.block4.v_fresh_sf` | `ctpt_v` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 115 | `L1.B4.M.softmax_out_mask` | `layer1.block4.softmax_out_mask_sf` | `ctpt_softmax_out_mask` | `encoding` | 1 | `scaling_factor` | 12 | True | 16384 | 14 |  |
| 116 | `L1.B4.M.v_mask` | `layer1.block4.v_mask_sf` | `ctpt_v_mask` | `encoding` | 1 | `scaling_factor` | 20 | True | 16384 | 22 |  |
| 117 | `L1.B4.M.softmax_v_mask` | `layer1.block4.softmax_v_mask_sf` | `ctpt_softmax_v_mask` | `encoding` | 1 | `scaling_factor` | 12 | True | 16384 | 14 |  |
| 118 | `L1.B4.S.ln_mean_inv_d` | `layer1.block4.ln_mean_inv_d_sf` | `ctpt_inv_d_attn_mean` | `scalar` | 2 | `scaling_factor` | 20 | True | 16384 | 20 |  |
| 119 | `L1.B4.S.ln_var_inv_d` | `layer1.block4.ln_var_inv_d_sf` | `ctpt_inv_d_attn_var` | `scalar` | 2 | `scaling_factor` | 20 | True | 16384 | 20 |  |
| 120 | `L1.B4.W.wo` | `layer1.block4.wo_sf` | `ctpt_wo` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 121 | `L1.B4.R.softmax_out_mask_r` | `layer1.block4.softmax_out_mask_rescale_sf` | `ctct_softmax_out_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 122 | `L1.B4.R.v_mask_r` | `layer1.block4.v_mask_rescale_sf` | `ctct_v_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 123 | `L1.B4.R.softmax_v_matmul_r` | `layer1.block4.softmax_v_matmul_rescale_sf` | `ctct_softmax_v_matmul_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 124 | `L1.B4.R.softmax_v_mask_r` | `layer1.block4.softmax_v_mask_rescale_sf` | `ctct_softmax_v_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 125 | `L1.B4.R.wo_r` | `layer1.block4.wo_rescale_sf` | `ctct_wo_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 126 | `L1.B4.R.ln_mean_r` | `layer1.block4.ln_mean_rescale_sf` | `ctct_attn_mean_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 127 | `L1.B4.R.ln_square_r` | `layer1.block4.ln_square_rescale_sf` | `ctct_attn_square_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 128 | `L1.B4.R.ln_var_r` | `layer1.block4.ln_var_rescale_sf` | `ctct_attn_var_rescale` | `rescale` | 3 | `scaling_factor` | 28 | True | 16384 | 28 |  |
| 129 | `L1.B4.K` | `layer1.block4.output_truncation_k` | `block4_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 130 | `L1.B5.F.inv_std_fresh` | `layer1.block5.inv_std_fresh_sf` | `ctpt_attn_inv_std` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 131 | `L1.B5.F.x_centered_fresh` | `layer1.block5.x_centered_fresh_sf` | `ctpt_attn_x_centered` | `fresh` | 4 | `scaling_factor` | 31 | True | 8192 | 31 |  |
| 132 | `L1.B5.M.gamma` | `layer1.block5.gamma_sf` | `ctpt_gamma_attn` | `encoding` | 1 | `scaling_factor` | 18 | True | 8192 | 20 |  |
| 133 | `L1.B5.W.wffn1` | `layer1.block5.wffn1_sf` | `ctpt_wffn1` | `encoding` | 4 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 134 | `L1.B5.M.gelu_coeff` | `layer1.block5.gelu_coeff_sf` | `ctpt_gelu_coeff` | `encoding` | 1 | `scaling_factor` | 18 | True | 8192 | 20 |  |
| 135 | `L1.B5.R.normalize_r` | `layer1.block5.normalize_rescale_sf` | `ctct_attn_normalize_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 8192 | 31 |  |
| 136 | `L1.B5.R.gamma_r` | `layer1.block5.gamma_rescale_sf` | `ctct_gamma_attn_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 137 | `L1.B5.R.wffn1_r` | `layer1.block5.wffn1_rescale_sf` | `ctct_wffn1_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 8192 | 31 |  |
| 138 | `L1.B5.R.gp0` | `layer1.block5.gelu_power_rescale_sf_0` | `ctct_gelu_x2` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 31 | GELU degree 1 does not use this power-rescale slot |
| 139 | `L1.B5.R.gp1` | `layer1.block5.gelu_power_rescale_sf_1` | `ctct_gelu_x3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 140 | `L1.B5.R.gp2` | `layer1.block5.gelu_power_rescale_sf_2` | `ctct_gelu_x4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 141 | `L1.B5.R.gc0` | `layer1.block5.gelu_coeff_mul_rescale_sf_0` | `ctct_gelu_b_x` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 142 | `L1.B5.R.gc1` | `layer1.block5.gelu_coeff_mul_rescale_sf_1` | `ctct_gelu_c_x2` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 143 | `L1.B5.R.gc2` | `layer1.block5.gelu_coeff_mul_rescale_sf_2` | `ctct_gelu_d_x3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 144 | `L1.B5.R.gc3` | `layer1.block5.gelu_coeff_mul_rescale_sf_3` | `ctct_gelu_e_x4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 31 | GELU degree 1 does not use this coefficient-rescale slot |
| 145 | `L1.B5.K` | `layer1.block5.output_truncation_k` | `block5_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 146 | `L2.B1.F.gelu_out` | `layer2.block1.gelu_out_sf` | `ctpt_gelu_out` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 147 | `L2.B1.W.wffn2` | `layer2.block1.wffn2_sf` | `ctpt_ffn2` | `encoding` | 4 | `scaling_factor` | 20 | True | 8192 | 20 |  |
| 148 | `L2.B1.S.mean_inv_d` | `layer2.block1.mean_inv_d_sf` | `ctpt_inv_d_1` | `scalar` | 2 | `scaling_factor` | 20 | True | 8192 | 20 |  |
| 149 | `L2.B1.S.var_inv_d` | `layer2.block1.var_inv_d_sf` | `ctpt_inv_d_2` | `scalar` | 2 | `scaling_factor` | 20 | True | 8192 | 20 |  |
| 150 | `L2.B1.R.wffn2_r` | `layer2.block1.wffn2_rescale_sf` | `ctct_ffn2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 151 | `L2.B1.R.mean_r` | `layer2.block1.mean_rescale_sf` | `ctct_mean_rescale` | `rescale` | 3 | `scaling_factor` | 34 | True | 8192 | 34 |  |
| 152 | `L2.B1.R.square_r` | `layer2.block1.square_rescale_sf` | `ctct_ext_square` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 153 | `L2.B1.R.var_r` | `layer2.block1.var_rescale_sf` | `ctct_var_rescale` | `rescale` | 3 | `scaling_factor` | 34 | True | 8192 | 34 |  |
| 154 | `L2.B1.K` | `layer2.block1.output_truncation_k` | `block1_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 155 | `L2.B2.F.inv_std_fresh` | `layer2.block2.inv_std_fresh_sf` | `ctpt_inv_std` | `fresh` | 4 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 156 | `L2.B2.F.x_centered_fresh` | `layer2.block2.x_centered_fresh_sf` | `ctpt_x_centered` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 157 | `L2.B2.M.gamma` | `layer2.block2.gamma_sf` | `ctpt_gamma` | `encoding` | 1 | `scaling_factor` | 18 | True | 16384 | 20 |  |
| 158 | `L2.B2.W.wq` | `layer2.block2.wq_sf` | `ctpt_wq` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 159 | `L2.B2.W.wk` | `layer2.block2.wk_sf` | `ctpt_wk` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 160 | `L2.B2.W.wv` | `layer2.block2.wv_sf` | `ctpt_wv` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 161 | `L2.B2.M.kt_mask1` | `layer2.block2.kt_mask1_sf` | `ctpt_kt_mask1` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 162 | `L2.B2.M.kt_mask2` | `layer2.block2.kt_mask2_sf` | `ctpt_kt_mask2` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 163 | `L2.B2.M.q_mask1` | `layer2.block2.q_mask1_sf` | `ctpt_q_mask1` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 164 | `L2.B2.M.q_mask2` | `layer2.block2.q_mask2_sf` | `ctpt_q_mask2` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 165 | `L2.B2.M.qkt_merge_mask` | `layer2.block2.qkt_merge_mask_sf` | `ctpt_qkt_merge_mask` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 166 | `L2.B2.R.normalize_r` | `layer2.block2.normalize_rescale_sf` | `ctct_normalize_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 167 | `L2.B2.R.gamma_r` | `layer2.block2.gamma_rescale_sf` | `ctct_gamma_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 168 | `L2.B2.R.wk_r` | `layer2.block2.wk_rescale_sf` | `ctct_wk_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 169 | `L2.B2.R.wq_r` | `layer2.block2.wq_rescale_sf` | `ctct_wq_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 170 | `L2.B2.R.wv_r` | `layer2.block2.wv_rescale_sf` | `ctct_wv_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 171 | `L2.B2.R.kt_mask1_r` | `layer2.block2.kt_mask1_rescale_sf` | `ctct_kt_mask1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 172 | `L2.B2.R.kt_mask2_r` | `layer2.block2.kt_mask2_rescale_sf` | `ctct_kt_mask2_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 173 | `L2.B2.R.q_mask1_r` | `layer2.block2.q_mask1_rescale_sf` | `ctct_q_mask1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 174 | `L2.B2.R.q_mask2_r` | `layer2.block2.q_mask2_rescale_sf` | `ctct_q_mask2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 175 | `L2.B2.R.qkt_matmul_r` | `layer2.block2.qkt_matmul_rescale_sf` | `ctct_qkt_matmul_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 176 | `L2.B2.R.qkt_merge_mask_r` | `layer2.block2.qkt_merge_mask_rescale_sf` | `ctct_qkt_merge_mask_rescale` | `rescale` | 3 | `scaling_factor` | 28 | True | 16384 | 28 |  |
| 177 | `L2.B2.K` | `layer2.block2.output_truncation_k` | `block2_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 178 | `L2.B3.F.x_fresh` | `layer2.block3.x_fresh_sf` | `ctpt_softmax_x` | `fresh` | 4 | `scaling_factor` | 28 | True | 16384 | 28 |  |
| 179 | `L2.B3.S.inv_2n` | `layer2.block3.inv_2n_sf` | `ctpt_softmax_inv_2n` | `scalar` | 2 | `scaling_factor` | 15 | True | 16384 | 15 |  |
| 180 | `L2.B3.R.x_inv_2n_r` | `layer2.block3.x_inv_2n_rescale_sf` | `ctct_softmax_x_inv_2n_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 181 | `L2.B3.R.sq0` | `layer2.block3.square_rescale_sf_0` | `ctct_softmax_pow_s1` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 182 | `L2.B3.R.sq1` | `layer2.block3.square_rescale_sf_1` | `ctct_softmax_pow_s2` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 183 | `L2.B3.R.sq2` | `layer2.block3.square_rescale_sf_2` | `ctct_softmax_pow_s3` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 184 | `L2.B3.R.sq3` | `layer2.block3.square_rescale_sf_3` | `ctct_softmax_pow_s4` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 185 | `L2.B3.K` | `layer2.block3.output_truncation_k` | `block3_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 186 | `L2.B4.F.softmax_out_fresh` | `layer2.block4.softmax_out_fresh_sf` | `ctpt_softmax_out` | `fresh` | 4 | `scaling_factor` | 35 | True | 16384 | 35 |  |
| 187 | `L2.B4.F.v_fresh` | `layer2.block4.v_fresh_sf` | `ctpt_v` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 188 | `L2.B4.M.softmax_out_mask` | `layer2.block4.softmax_out_mask_sf` | `ctpt_softmax_out_mask` | `encoding` | 1 | `scaling_factor` | 12 | True | 16384 | 14 |  |
| 189 | `L2.B4.M.v_mask` | `layer2.block4.v_mask_sf` | `ctpt_v_mask` | `encoding` | 1 | `scaling_factor` | 20 | True | 16384 | 22 |  |
| 190 | `L2.B4.M.softmax_v_mask` | `layer2.block4.softmax_v_mask_sf` | `ctpt_softmax_v_mask` | `encoding` | 1 | `scaling_factor` | 12 | True | 16384 | 14 |  |
| 191 | `L2.B4.S.ln_mean_inv_d` | `layer2.block4.ln_mean_inv_d_sf` | `ctpt_inv_d_attn_mean` | `scalar` | 2 | `scaling_factor` | 20 | True | 16384 | 20 |  |
| 192 | `L2.B4.S.ln_var_inv_d` | `layer2.block4.ln_var_inv_d_sf` | `ctpt_inv_d_attn_var` | `scalar` | 2 | `scaling_factor` | 20 | True | 16384 | 20 |  |
| 193 | `L2.B4.W.wo` | `layer2.block4.wo_sf` | `ctpt_wo` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 194 | `L2.B4.R.softmax_out_mask_r` | `layer2.block4.softmax_out_mask_rescale_sf` | `ctct_softmax_out_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 195 | `L2.B4.R.v_mask_r` | `layer2.block4.v_mask_rescale_sf` | `ctct_v_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 196 | `L2.B4.R.softmax_v_matmul_r` | `layer2.block4.softmax_v_matmul_rescale_sf` | `ctct_softmax_v_matmul_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 197 | `L2.B4.R.softmax_v_mask_r` | `layer2.block4.softmax_v_mask_rescale_sf` | `ctct_softmax_v_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 198 | `L2.B4.R.wo_r` | `layer2.block4.wo_rescale_sf` | `ctct_wo_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 199 | `L2.B4.R.ln_mean_r` | `layer2.block4.ln_mean_rescale_sf` | `ctct_attn_mean_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 200 | `L2.B4.R.ln_square_r` | `layer2.block4.ln_square_rescale_sf` | `ctct_attn_square_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 201 | `L2.B4.R.ln_var_r` | `layer2.block4.ln_var_rescale_sf` | `ctct_attn_var_rescale` | `rescale` | 3 | `scaling_factor` | 28 | True | 16384 | 28 |  |
| 202 | `L2.B4.K` | `layer2.block4.output_truncation_k` | `block4_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 203 | `L2.B5.F.inv_std_fresh` | `layer2.block5.inv_std_fresh_sf` | `ctpt_attn_inv_std` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 204 | `L2.B5.F.x_centered_fresh` | `layer2.block5.x_centered_fresh_sf` | `ctpt_attn_x_centered` | `fresh` | 4 | `scaling_factor` | 31 | True | 8192 | 31 |  |
| 205 | `L2.B5.M.gamma` | `layer2.block5.gamma_sf` | `ctpt_gamma_attn` | `encoding` | 1 | `scaling_factor` | 18 | True | 8192 | 20 |  |
| 206 | `L2.B5.W.wffn1` | `layer2.block5.wffn1_sf` | `ctpt_wffn1` | `encoding` | 4 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 207 | `L2.B5.M.gelu_coeff` | `layer2.block5.gelu_coeff_sf` | `ctpt_gelu_coeff` | `encoding` | 1 | `scaling_factor` | 18 | True | 8192 | 20 |  |
| 208 | `L2.B5.R.normalize_r` | `layer2.block5.normalize_rescale_sf` | `ctct_attn_normalize_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 8192 | 31 |  |
| 209 | `L2.B5.R.gamma_r` | `layer2.block5.gamma_rescale_sf` | `ctct_gamma_attn_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 210 | `L2.B5.R.wffn1_r` | `layer2.block5.wffn1_rescale_sf` | `ctct_wffn1_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 8192 | 31 |  |
| 211 | `L2.B5.R.gp0` | `layer2.block5.gelu_power_rescale_sf_0` | `ctct_gelu_x2` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 31 | GELU degree 1 does not use this power-rescale slot |
| 212 | `L2.B5.R.gp1` | `layer2.block5.gelu_power_rescale_sf_1` | `ctct_gelu_x3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 213 | `L2.B5.R.gp2` | `layer2.block5.gelu_power_rescale_sf_2` | `ctct_gelu_x4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 214 | `L2.B5.R.gc0` | `layer2.block5.gelu_coeff_mul_rescale_sf_0` | `ctct_gelu_b_x` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 215 | `L2.B5.R.gc1` | `layer2.block5.gelu_coeff_mul_rescale_sf_1` | `ctct_gelu_c_x2` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 216 | `L2.B5.R.gc2` | `layer2.block5.gelu_coeff_mul_rescale_sf_2` | `ctct_gelu_d_x3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 217 | `L2.B5.R.gc3` | `layer2.block5.gelu_coeff_mul_rescale_sf_3` | `ctct_gelu_e_x4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 31 | GELU degree 1 does not use this coefficient-rescale slot |
| 218 | `L2.B5.K` | `layer2.block5.output_truncation_k` | `block5_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 219 | `L3.B1.F.gelu_out` | `layer3.block1.gelu_out_sf` | `ctpt_gelu_out` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 220 | `L3.B1.W.wffn2` | `layer3.block1.wffn2_sf` | `ctpt_ffn2` | `encoding` | 4 | `scaling_factor` | 20 | True | 8192 | 20 |  |
| 221 | `L3.B1.S.mean_inv_d` | `layer3.block1.mean_inv_d_sf` | `ctpt_inv_d_1` | `scalar` | 2 | `scaling_factor` | 20 | True | 8192 | 20 |  |
| 222 | `L3.B1.S.var_inv_d` | `layer3.block1.var_inv_d_sf` | `ctpt_inv_d_2` | `scalar` | 2 | `scaling_factor` | 20 | True | 8192 | 20 |  |
| 223 | `L3.B1.R.wffn2_r` | `layer3.block1.wffn2_rescale_sf` | `ctct_ffn2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 224 | `L3.B1.R.mean_r` | `layer3.block1.mean_rescale_sf` | `ctct_mean_rescale` | `rescale` | 3 | `scaling_factor` | 34 | True | 8192 | 34 |  |
| 225 | `L3.B1.R.square_r` | `layer3.block1.square_rescale_sf` | `ctct_ext_square` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 226 | `L3.B1.R.var_r` | `layer3.block1.var_rescale_sf` | `ctct_var_rescale` | `rescale` | 3 | `scaling_factor` | 34 | True | 8192 | 34 |  |
| 227 | `L3.B1.K` | `layer3.block1.output_truncation_k` | `block1_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 228 | `L3.B2.F.inv_std_fresh` | `layer3.block2.inv_std_fresh_sf` | `ctpt_inv_std` | `fresh` | 4 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 229 | `L3.B2.F.x_centered_fresh` | `layer3.block2.x_centered_fresh_sf` | `ctpt_x_centered` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 230 | `L3.B2.M.gamma` | `layer3.block2.gamma_sf` | `ctpt_gamma` | `encoding` | 1 | `scaling_factor` | 18 | True | 16384 | 20 |  |
| 231 | `L3.B2.W.wq` | `layer3.block2.wq_sf` | `ctpt_wq` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 232 | `L3.B2.W.wk` | `layer3.block2.wk_sf` | `ctpt_wk` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 233 | `L3.B2.W.wv` | `layer3.block2.wv_sf` | `ctpt_wv` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 234 | `L3.B2.M.kt_mask1` | `layer3.block2.kt_mask1_sf` | `ctpt_kt_mask1` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 235 | `L3.B2.M.kt_mask2` | `layer3.block2.kt_mask2_sf` | `ctpt_kt_mask2` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 236 | `L3.B2.M.q_mask1` | `layer3.block2.q_mask1_sf` | `ctpt_q_mask1` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 237 | `L3.B2.M.q_mask2` | `layer3.block2.q_mask2_sf` | `ctpt_q_mask2` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 238 | `L3.B2.M.qkt_merge_mask` | `layer3.block2.qkt_merge_mask_sf` | `ctpt_qkt_merge_mask` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 239 | `L3.B2.R.normalize_r` | `layer3.block2.normalize_rescale_sf` | `ctct_normalize_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 240 | `L3.B2.R.gamma_r` | `layer3.block2.gamma_rescale_sf` | `ctct_gamma_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 241 | `L3.B2.R.wk_r` | `layer3.block2.wk_rescale_sf` | `ctct_wk_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 242 | `L3.B2.R.wq_r` | `layer3.block2.wq_rescale_sf` | `ctct_wq_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 243 | `L3.B2.R.wv_r` | `layer3.block2.wv_rescale_sf` | `ctct_wv_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 244 | `L3.B2.R.kt_mask1_r` | `layer3.block2.kt_mask1_rescale_sf` | `ctct_kt_mask1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 245 | `L3.B2.R.kt_mask2_r` | `layer3.block2.kt_mask2_rescale_sf` | `ctct_kt_mask2_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 246 | `L3.B2.R.q_mask1_r` | `layer3.block2.q_mask1_rescale_sf` | `ctct_q_mask1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 247 | `L3.B2.R.q_mask2_r` | `layer3.block2.q_mask2_rescale_sf` | `ctct_q_mask2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 248 | `L3.B2.R.qkt_matmul_r` | `layer3.block2.qkt_matmul_rescale_sf` | `ctct_qkt_matmul_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 249 | `L3.B2.R.qkt_merge_mask_r` | `layer3.block2.qkt_merge_mask_rescale_sf` | `ctct_qkt_merge_mask_rescale` | `rescale` | 3 | `scaling_factor` | 28 | True | 16384 | 28 |  |
| 250 | `L3.B2.K` | `layer3.block2.output_truncation_k` | `block2_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 251 | `L3.B3.F.x_fresh` | `layer3.block3.x_fresh_sf` | `ctpt_softmax_x` | `fresh` | 4 | `scaling_factor` | 28 | True | 16384 | 28 |  |
| 252 | `L3.B3.S.inv_2n` | `layer3.block3.inv_2n_sf` | `ctpt_softmax_inv_2n` | `scalar` | 2 | `scaling_factor` | 15 | True | 16384 | 15 |  |
| 253 | `L3.B3.R.x_inv_2n_r` | `layer3.block3.x_inv_2n_rescale_sf` | `ctct_softmax_x_inv_2n_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 254 | `L3.B3.R.sq0` | `layer3.block3.square_rescale_sf_0` | `ctct_softmax_pow_s1` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 255 | `L3.B3.R.sq1` | `layer3.block3.square_rescale_sf_1` | `ctct_softmax_pow_s2` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 256 | `L3.B3.R.sq2` | `layer3.block3.square_rescale_sf_2` | `ctct_softmax_pow_s3` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 257 | `L3.B3.R.sq3` | `layer3.block3.square_rescale_sf_3` | `ctct_softmax_pow_s4` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 258 | `L3.B3.K` | `layer3.block3.output_truncation_k` | `block3_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 259 | `L3.B4.F.softmax_out_fresh` | `layer3.block4.softmax_out_fresh_sf` | `ctpt_softmax_out` | `fresh` | 4 | `scaling_factor` | 35 | True | 16384 | 35 |  |
| 260 | `L3.B4.F.v_fresh` | `layer3.block4.v_fresh_sf` | `ctpt_v` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 261 | `L3.B4.M.softmax_out_mask` | `layer3.block4.softmax_out_mask_sf` | `ctpt_softmax_out_mask` | `encoding` | 1 | `scaling_factor` | 12 | True | 16384 | 14 |  |
| 262 | `L3.B4.M.v_mask` | `layer3.block4.v_mask_sf` | `ctpt_v_mask` | `encoding` | 1 | `scaling_factor` | 20 | True | 16384 | 22 |  |
| 263 | `L3.B4.M.softmax_v_mask` | `layer3.block4.softmax_v_mask_sf` | `ctpt_softmax_v_mask` | `encoding` | 1 | `scaling_factor` | 12 | True | 16384 | 14 |  |
| 264 | `L3.B4.S.ln_mean_inv_d` | `layer3.block4.ln_mean_inv_d_sf` | `ctpt_inv_d_attn_mean` | `scalar` | 2 | `scaling_factor` | 20 | True | 16384 | 20 |  |
| 265 | `L3.B4.S.ln_var_inv_d` | `layer3.block4.ln_var_inv_d_sf` | `ctpt_inv_d_attn_var` | `scalar` | 2 | `scaling_factor` | 20 | True | 16384 | 20 |  |
| 266 | `L3.B4.W.wo` | `layer3.block4.wo_sf` | `ctpt_wo` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 267 | `L3.B4.R.softmax_out_mask_r` | `layer3.block4.softmax_out_mask_rescale_sf` | `ctct_softmax_out_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 268 | `L3.B4.R.v_mask_r` | `layer3.block4.v_mask_rescale_sf` | `ctct_v_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 269 | `L3.B4.R.softmax_v_matmul_r` | `layer3.block4.softmax_v_matmul_rescale_sf` | `ctct_softmax_v_matmul_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 270 | `L3.B4.R.softmax_v_mask_r` | `layer3.block4.softmax_v_mask_rescale_sf` | `ctct_softmax_v_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 271 | `L3.B4.R.wo_r` | `layer3.block4.wo_rescale_sf` | `ctct_wo_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 272 | `L3.B4.R.ln_mean_r` | `layer3.block4.ln_mean_rescale_sf` | `ctct_attn_mean_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 273 | `L3.B4.R.ln_square_r` | `layer3.block4.ln_square_rescale_sf` | `ctct_attn_square_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 274 | `L3.B4.R.ln_var_r` | `layer3.block4.ln_var_rescale_sf` | `ctct_attn_var_rescale` | `rescale` | 3 | `scaling_factor` | 28 | True | 16384 | 28 |  |
| 275 | `L3.B4.K` | `layer3.block4.output_truncation_k` | `block4_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 276 | `L3.B5.F.inv_std_fresh` | `layer3.block5.inv_std_fresh_sf` | `ctpt_attn_inv_std` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 277 | `L3.B5.F.x_centered_fresh` | `layer3.block5.x_centered_fresh_sf` | `ctpt_attn_x_centered` | `fresh` | 4 | `scaling_factor` | 31 | True | 8192 | 31 |  |
| 278 | `L3.B5.M.gamma` | `layer3.block5.gamma_sf` | `ctpt_gamma_attn` | `encoding` | 1 | `scaling_factor` | 18 | True | 8192 | 20 |  |
| 279 | `L3.B5.W.wffn1` | `layer3.block5.wffn1_sf` | `ctpt_wffn1` | `encoding` | 4 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 280 | `L3.B5.M.gelu_coeff` | `layer3.block5.gelu_coeff_sf` | `ctpt_gelu_coeff` | `encoding` | 1 | `scaling_factor` | 18 | True | 8192 | 20 |  |
| 281 | `L3.B5.R.normalize_r` | `layer3.block5.normalize_rescale_sf` | `ctct_attn_normalize_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 8192 | 31 |  |
| 282 | `L3.B5.R.gamma_r` | `layer3.block5.gamma_rescale_sf` | `ctct_gamma_attn_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 283 | `L3.B5.R.wffn1_r` | `layer3.block5.wffn1_rescale_sf` | `ctct_wffn1_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 8192 | 31 |  |
| 284 | `L3.B5.R.gp0` | `layer3.block5.gelu_power_rescale_sf_0` | `ctct_gelu_x2` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 31 | GELU degree 1 does not use this power-rescale slot |
| 285 | `L3.B5.R.gp1` | `layer3.block5.gelu_power_rescale_sf_1` | `ctct_gelu_x3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 286 | `L3.B5.R.gp2` | `layer3.block5.gelu_power_rescale_sf_2` | `ctct_gelu_x4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 287 | `L3.B5.R.gc0` | `layer3.block5.gelu_coeff_mul_rescale_sf_0` | `ctct_gelu_b_x` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 288 | `L3.B5.R.gc1` | `layer3.block5.gelu_coeff_mul_rescale_sf_1` | `ctct_gelu_c_x2` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 289 | `L3.B5.R.gc2` | `layer3.block5.gelu_coeff_mul_rescale_sf_2` | `ctct_gelu_d_x3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 290 | `L3.B5.R.gc3` | `layer3.block5.gelu_coeff_mul_rescale_sf_3` | `ctct_gelu_e_x4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 31 | GELU degree 1 does not use this coefficient-rescale slot |
| 291 | `L3.B5.K` | `layer3.block5.output_truncation_k` | `block5_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 292 | `L4.B1.F.gelu_out` | `layer4.block1.gelu_out_sf` | `ctpt_gelu_out` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 293 | `L4.B1.W.wffn2` | `layer4.block1.wffn2_sf` | `ctpt_ffn2` | `encoding` | 4 | `scaling_factor` | 20 | True | 8192 | 20 |  |
| 294 | `L4.B1.S.mean_inv_d` | `layer4.block1.mean_inv_d_sf` | `ctpt_inv_d_1` | `scalar` | 2 | `scaling_factor` | 20 | True | 8192 | 20 |  |
| 295 | `L4.B1.S.var_inv_d` | `layer4.block1.var_inv_d_sf` | `ctpt_inv_d_2` | `scalar` | 2 | `scaling_factor` | 20 | True | 8192 | 20 |  |
| 296 | `L4.B1.R.wffn2_r` | `layer4.block1.wffn2_rescale_sf` | `ctct_ffn2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 297 | `L4.B1.R.mean_r` | `layer4.block1.mean_rescale_sf` | `ctct_mean_rescale` | `rescale` | 3 | `scaling_factor` | 34 | True | 8192 | 34 |  |
| 298 | `L4.B1.R.square_r` | `layer4.block1.square_rescale_sf` | `ctct_ext_square` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 299 | `L4.B1.R.var_r` | `layer4.block1.var_rescale_sf` | `ctct_var_rescale` | `rescale` | 3 | `scaling_factor` | 34 | True | 8192 | 34 |  |
| 300 | `L4.B1.K` | `layer4.block1.output_truncation_k` | `block1_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 301 | `L4.B2.F.inv_std_fresh` | `layer4.block2.inv_std_fresh_sf` | `ctpt_inv_std` | `fresh` | 4 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 302 | `L4.B2.F.x_centered_fresh` | `layer4.block2.x_centered_fresh_sf` | `ctpt_x_centered` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 303 | `L4.B2.M.gamma` | `layer4.block2.gamma_sf` | `ctpt_gamma` | `encoding` | 1 | `scaling_factor` | 18 | True | 16384 | 20 |  |
| 304 | `L4.B2.W.wq` | `layer4.block2.wq_sf` | `ctpt_wq` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 305 | `L4.B2.W.wk` | `layer4.block2.wk_sf` | `ctpt_wk` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 306 | `L4.B2.W.wv` | `layer4.block2.wv_sf` | `ctpt_wv` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 307 | `L4.B2.M.kt_mask1` | `layer4.block2.kt_mask1_sf` | `ctpt_kt_mask1` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 308 | `L4.B2.M.kt_mask2` | `layer4.block2.kt_mask2_sf` | `ctpt_kt_mask2` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 309 | `L4.B2.M.q_mask1` | `layer4.block2.q_mask1_sf` | `ctpt_q_mask1` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 310 | `L4.B2.M.q_mask2` | `layer4.block2.q_mask2_sf` | `ctpt_q_mask2` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 311 | `L4.B2.M.qkt_merge_mask` | `layer4.block2.qkt_merge_mask_sf` | `ctpt_qkt_merge_mask` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 312 | `L4.B2.R.normalize_r` | `layer4.block2.normalize_rescale_sf` | `ctct_normalize_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 313 | `L4.B2.R.gamma_r` | `layer4.block2.gamma_rescale_sf` | `ctct_gamma_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 314 | `L4.B2.R.wk_r` | `layer4.block2.wk_rescale_sf` | `ctct_wk_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 315 | `L4.B2.R.wq_r` | `layer4.block2.wq_rescale_sf` | `ctct_wq_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 316 | `L4.B2.R.wv_r` | `layer4.block2.wv_rescale_sf` | `ctct_wv_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 317 | `L4.B2.R.kt_mask1_r` | `layer4.block2.kt_mask1_rescale_sf` | `ctct_kt_mask1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 318 | `L4.B2.R.kt_mask2_r` | `layer4.block2.kt_mask2_rescale_sf` | `ctct_kt_mask2_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 319 | `L4.B2.R.q_mask1_r` | `layer4.block2.q_mask1_rescale_sf` | `ctct_q_mask1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 320 | `L4.B2.R.q_mask2_r` | `layer4.block2.q_mask2_rescale_sf` | `ctct_q_mask2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 321 | `L4.B2.R.qkt_matmul_r` | `layer4.block2.qkt_matmul_rescale_sf` | `ctct_qkt_matmul_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 322 | `L4.B2.R.qkt_merge_mask_r` | `layer4.block2.qkt_merge_mask_rescale_sf` | `ctct_qkt_merge_mask_rescale` | `rescale` | 3 | `scaling_factor` | 28 | True | 16384 | 28 |  |
| 323 | `L4.B2.K` | `layer4.block2.output_truncation_k` | `block2_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 324 | `L4.B3.F.x_fresh` | `layer4.block3.x_fresh_sf` | `ctpt_softmax_x` | `fresh` | 4 | `scaling_factor` | 28 | True | 16384 | 28 |  |
| 325 | `L4.B3.S.inv_2n` | `layer4.block3.inv_2n_sf` | `ctpt_softmax_inv_2n` | `scalar` | 2 | `scaling_factor` | 15 | True | 16384 | 15 |  |
| 326 | `L4.B3.R.x_inv_2n_r` | `layer4.block3.x_inv_2n_rescale_sf` | `ctct_softmax_x_inv_2n_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 327 | `L4.B3.R.sq0` | `layer4.block3.square_rescale_sf_0` | `ctct_softmax_pow_s1` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 328 | `L4.B3.R.sq1` | `layer4.block3.square_rescale_sf_1` | `ctct_softmax_pow_s2` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 329 | `L4.B3.R.sq2` | `layer4.block3.square_rescale_sf_2` | `ctct_softmax_pow_s3` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 330 | `L4.B3.R.sq3` | `layer4.block3.square_rescale_sf_3` | `ctct_softmax_pow_s4` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 331 | `L4.B3.K` | `layer4.block3.output_truncation_k` | `block3_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 332 | `L4.B4.F.softmax_out_fresh` | `layer4.block4.softmax_out_fresh_sf` | `ctpt_softmax_out` | `fresh` | 4 | `scaling_factor` | 35 | True | 16384 | 35 |  |
| 333 | `L4.B4.F.v_fresh` | `layer4.block4.v_fresh_sf` | `ctpt_v` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 334 | `L4.B4.M.softmax_out_mask` | `layer4.block4.softmax_out_mask_sf` | `ctpt_softmax_out_mask` | `encoding` | 1 | `scaling_factor` | 12 | True | 16384 | 14 |  |
| 335 | `L4.B4.M.v_mask` | `layer4.block4.v_mask_sf` | `ctpt_v_mask` | `encoding` | 1 | `scaling_factor` | 20 | True | 16384 | 22 |  |
| 336 | `L4.B4.M.softmax_v_mask` | `layer4.block4.softmax_v_mask_sf` | `ctpt_softmax_v_mask` | `encoding` | 1 | `scaling_factor` | 12 | True | 16384 | 14 |  |
| 337 | `L4.B4.S.ln_mean_inv_d` | `layer4.block4.ln_mean_inv_d_sf` | `ctpt_inv_d_attn_mean` | `scalar` | 2 | `scaling_factor` | 20 | True | 16384 | 20 |  |
| 338 | `L4.B4.S.ln_var_inv_d` | `layer4.block4.ln_var_inv_d_sf` | `ctpt_inv_d_attn_var` | `scalar` | 2 | `scaling_factor` | 20 | True | 16384 | 20 |  |
| 339 | `L4.B4.W.wo` | `layer4.block4.wo_sf` | `ctpt_wo` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 340 | `L4.B4.R.softmax_out_mask_r` | `layer4.block4.softmax_out_mask_rescale_sf` | `ctct_softmax_out_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 341 | `L4.B4.R.v_mask_r` | `layer4.block4.v_mask_rescale_sf` | `ctct_v_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 342 | `L4.B4.R.softmax_v_matmul_r` | `layer4.block4.softmax_v_matmul_rescale_sf` | `ctct_softmax_v_matmul_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 343 | `L4.B4.R.softmax_v_mask_r` | `layer4.block4.softmax_v_mask_rescale_sf` | `ctct_softmax_v_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 344 | `L4.B4.R.wo_r` | `layer4.block4.wo_rescale_sf` | `ctct_wo_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 345 | `L4.B4.R.ln_mean_r` | `layer4.block4.ln_mean_rescale_sf` | `ctct_attn_mean_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 346 | `L4.B4.R.ln_square_r` | `layer4.block4.ln_square_rescale_sf` | `ctct_attn_square_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 347 | `L4.B4.R.ln_var_r` | `layer4.block4.ln_var_rescale_sf` | `ctct_attn_var_rescale` | `rescale` | 3 | `scaling_factor` | 28 | True | 16384 | 28 |  |
| 348 | `L4.B4.K` | `layer4.block4.output_truncation_k` | `block4_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 349 | `L4.B5.F.inv_std_fresh` | `layer4.block5.inv_std_fresh_sf` | `ctpt_attn_inv_std` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 350 | `L4.B5.F.x_centered_fresh` | `layer4.block5.x_centered_fresh_sf` | `ctpt_attn_x_centered` | `fresh` | 4 | `scaling_factor` | 31 | True | 8192 | 31 |  |
| 351 | `L4.B5.M.gamma` | `layer4.block5.gamma_sf` | `ctpt_gamma_attn` | `encoding` | 1 | `scaling_factor` | 18 | True | 8192 | 20 |  |
| 352 | `L4.B5.W.wffn1` | `layer4.block5.wffn1_sf` | `ctpt_wffn1` | `encoding` | 4 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 353 | `L4.B5.M.gelu_coeff` | `layer4.block5.gelu_coeff_sf` | `ctpt_gelu_coeff` | `encoding` | 1 | `scaling_factor` | 18 | True | 8192 | 20 |  |
| 354 | `L4.B5.R.normalize_r` | `layer4.block5.normalize_rescale_sf` | `ctct_attn_normalize_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 8192 | 31 |  |
| 355 | `L4.B5.R.gamma_r` | `layer4.block5.gamma_rescale_sf` | `ctct_gamma_attn_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 356 | `L4.B5.R.wffn1_r` | `layer4.block5.wffn1_rescale_sf` | `ctct_wffn1_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 8192 | 31 |  |
| 357 | `L4.B5.R.gp0` | `layer4.block5.gelu_power_rescale_sf_0` | `ctct_gelu_x2` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 31 | GELU degree 1 does not use this power-rescale slot |
| 358 | `L4.B5.R.gp1` | `layer4.block5.gelu_power_rescale_sf_1` | `ctct_gelu_x3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 359 | `L4.B5.R.gp2` | `layer4.block5.gelu_power_rescale_sf_2` | `ctct_gelu_x4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 360 | `L4.B5.R.gc0` | `layer4.block5.gelu_coeff_mul_rescale_sf_0` | `ctct_gelu_b_x` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 361 | `L4.B5.R.gc1` | `layer4.block5.gelu_coeff_mul_rescale_sf_1` | `ctct_gelu_c_x2` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 362 | `L4.B5.R.gc2` | `layer4.block5.gelu_coeff_mul_rescale_sf_2` | `ctct_gelu_d_x3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 363 | `L4.B5.R.gc3` | `layer4.block5.gelu_coeff_mul_rescale_sf_3` | `ctct_gelu_e_x4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 31 | GELU degree 1 does not use this coefficient-rescale slot |
| 364 | `L4.B5.K` | `layer4.block5.output_truncation_k` | `block5_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 365 | `L5.B1.F.gelu_out` | `layer5.block1.gelu_out_sf` | `ctpt_gelu_out` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 366 | `L5.B1.W.wffn2` | `layer5.block1.wffn2_sf` | `ctpt_ffn2` | `encoding` | 4 | `scaling_factor` | 20 | True | 8192 | 20 |  |
| 367 | `L5.B1.S.mean_inv_d` | `layer5.block1.mean_inv_d_sf` | `ctpt_inv_d_1` | `scalar` | 2 | `scaling_factor` | 20 | True | 8192 | 20 |  |
| 368 | `L5.B1.S.var_inv_d` | `layer5.block1.var_inv_d_sf` | `ctpt_inv_d_2` | `scalar` | 2 | `scaling_factor` | 20 | True | 8192 | 20 |  |
| 369 | `L5.B1.R.wffn2_r` | `layer5.block1.wffn2_rescale_sf` | `ctct_ffn2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 370 | `L5.B1.R.mean_r` | `layer5.block1.mean_rescale_sf` | `ctct_mean_rescale` | `rescale` | 3 | `scaling_factor` | 34 | True | 8192 | 34 |  |
| 371 | `L5.B1.R.square_r` | `layer5.block1.square_rescale_sf` | `ctct_ext_square` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 372 | `L5.B1.R.var_r` | `layer5.block1.var_rescale_sf` | `ctct_var_rescale` | `rescale` | 3 | `scaling_factor` | 34 | True | 8192 | 34 |  |
| 373 | `L5.B1.K` | `layer5.block1.output_truncation_k` | `block1_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 374 | `L5.B2.F.inv_std_fresh` | `layer5.block2.inv_std_fresh_sf` | `ctpt_inv_std` | `fresh` | 4 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 375 | `L5.B2.F.x_centered_fresh` | `layer5.block2.x_centered_fresh_sf` | `ctpt_x_centered` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 376 | `L5.B2.M.gamma` | `layer5.block2.gamma_sf` | `ctpt_gamma` | `encoding` | 1 | `scaling_factor` | 18 | True | 16384 | 20 |  |
| 377 | `L5.B2.W.wq` | `layer5.block2.wq_sf` | `ctpt_wq` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 378 | `L5.B2.W.wk` | `layer5.block2.wk_sf` | `ctpt_wk` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 379 | `L5.B2.W.wv` | `layer5.block2.wv_sf` | `ctpt_wv` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 380 | `L5.B2.M.kt_mask1` | `layer5.block2.kt_mask1_sf` | `ctpt_kt_mask1` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 381 | `L5.B2.M.kt_mask2` | `layer5.block2.kt_mask2_sf` | `ctpt_kt_mask2` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 382 | `L5.B2.M.q_mask1` | `layer5.block2.q_mask1_sf` | `ctpt_q_mask1` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 383 | `L5.B2.M.q_mask2` | `layer5.block2.q_mask2_sf` | `ctpt_q_mask2` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 384 | `L5.B2.M.qkt_merge_mask` | `layer5.block2.qkt_merge_mask_sf` | `ctpt_qkt_merge_mask` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 385 | `L5.B2.R.normalize_r` | `layer5.block2.normalize_rescale_sf` | `ctct_normalize_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 386 | `L5.B2.R.gamma_r` | `layer5.block2.gamma_rescale_sf` | `ctct_gamma_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 387 | `L5.B2.R.wk_r` | `layer5.block2.wk_rescale_sf` | `ctct_wk_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 388 | `L5.B2.R.wq_r` | `layer5.block2.wq_rescale_sf` | `ctct_wq_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 389 | `L5.B2.R.wv_r` | `layer5.block2.wv_rescale_sf` | `ctct_wv_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 390 | `L5.B2.R.kt_mask1_r` | `layer5.block2.kt_mask1_rescale_sf` | `ctct_kt_mask1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 391 | `L5.B2.R.kt_mask2_r` | `layer5.block2.kt_mask2_rescale_sf` | `ctct_kt_mask2_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 392 | `L5.B2.R.q_mask1_r` | `layer5.block2.q_mask1_rescale_sf` | `ctct_q_mask1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 393 | `L5.B2.R.q_mask2_r` | `layer5.block2.q_mask2_rescale_sf` | `ctct_q_mask2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 394 | `L5.B2.R.qkt_matmul_r` | `layer5.block2.qkt_matmul_rescale_sf` | `ctct_qkt_matmul_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 395 | `L5.B2.R.qkt_merge_mask_r` | `layer5.block2.qkt_merge_mask_rescale_sf` | `ctct_qkt_merge_mask_rescale` | `rescale` | 3 | `scaling_factor` | 28 | True | 16384 | 28 |  |
| 396 | `L5.B2.K` | `layer5.block2.output_truncation_k` | `block2_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 397 | `L5.B3.F.x_fresh` | `layer5.block3.x_fresh_sf` | `ctpt_softmax_x` | `fresh` | 4 | `scaling_factor` | 28 | True | 8192 | 28 |  |
| 398 | `L5.B3.S.inv_2n` | `layer5.block3.inv_2n_sf` | `ctpt_softmax_inv_2n` | `scalar` | 2 | `scaling_factor` | 15 | True | 8192 | 15 |  |
| 399 | `L5.B3.R.x_inv_2n_r` | `layer5.block3.x_inv_2n_rescale_sf` | `ctct_softmax_x_inv_2n_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 400 | `L5.B3.R.sq0` | `layer5.block3.square_rescale_sf_0` | `ctct_softmax_pow_s1` | `rescale` | 3 | `scaling_factor` | 31 | True | 8192 | 31 |  |
| 401 | `L5.B3.R.sq1` | `layer5.block3.square_rescale_sf_1` | `ctct_softmax_pow_s2` | `rescale` | 3 | `scaling_factor` | 31 | True | 8192 | 31 |  |
| 402 | `L5.B3.R.sq2` | `layer5.block3.square_rescale_sf_2` | `ctct_softmax_pow_s3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 31 | softmax degree 2 does not use this square-rescale slot |
| 403 | `L5.B3.R.sq3` | `layer5.block3.square_rescale_sf_3` | `ctct_softmax_pow_s4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 31 | softmax degree 2 does not use this square-rescale slot |
| 404 | `L5.B3.K` | `layer5.block3.output_truncation_k` | `block3_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 405 | `L5.B4.F.softmax_out_fresh` | `layer5.block4.softmax_out_fresh_sf` | `ctpt_softmax_out` | `fresh` | 4 | `scaling_factor` | 35 | True | 16384 | 35 |  |
| 406 | `L5.B4.F.v_fresh` | `layer5.block4.v_fresh_sf` | `ctpt_v` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 407 | `L5.B4.M.softmax_out_mask` | `layer5.block4.softmax_out_mask_sf` | `ctpt_softmax_out_mask` | `encoding` | 1 | `scaling_factor` | 12 | True | 16384 | 14 |  |
| 408 | `L5.B4.M.v_mask` | `layer5.block4.v_mask_sf` | `ctpt_v_mask` | `encoding` | 1 | `scaling_factor` | 20 | True | 16384 | 22 |  |
| 409 | `L5.B4.M.softmax_v_mask` | `layer5.block4.softmax_v_mask_sf` | `ctpt_softmax_v_mask` | `encoding` | 1 | `scaling_factor` | 12 | True | 16384 | 14 |  |
| 410 | `L5.B4.S.ln_mean_inv_d` | `layer5.block4.ln_mean_inv_d_sf` | `ctpt_inv_d_attn_mean` | `scalar` | 2 | `scaling_factor` | 20 | True | 16384 | 20 |  |
| 411 | `L5.B4.S.ln_var_inv_d` | `layer5.block4.ln_var_inv_d_sf` | `ctpt_inv_d_attn_var` | `scalar` | 2 | `scaling_factor` | 20 | True | 16384 | 20 |  |
| 412 | `L5.B4.W.wo` | `layer5.block4.wo_sf` | `ctpt_wo` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 413 | `L5.B4.R.softmax_out_mask_r` | `layer5.block4.softmax_out_mask_rescale_sf` | `ctct_softmax_out_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 414 | `L5.B4.R.v_mask_r` | `layer5.block4.v_mask_rescale_sf` | `ctct_v_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 415 | `L5.B4.R.softmax_v_matmul_r` | `layer5.block4.softmax_v_matmul_rescale_sf` | `ctct_softmax_v_matmul_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 416 | `L5.B4.R.softmax_v_mask_r` | `layer5.block4.softmax_v_mask_rescale_sf` | `ctct_softmax_v_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 417 | `L5.B4.R.wo_r` | `layer5.block4.wo_rescale_sf` | `ctct_wo_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 418 | `L5.B4.R.ln_mean_r` | `layer5.block4.ln_mean_rescale_sf` | `ctct_attn_mean_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 419 | `L5.B4.R.ln_square_r` | `layer5.block4.ln_square_rescale_sf` | `ctct_attn_square_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 420 | `L5.B4.R.ln_var_r` | `layer5.block4.ln_var_rescale_sf` | `ctct_attn_var_rescale` | `rescale` | 3 | `scaling_factor` | 28 | True | 16384 | 28 |  |
| 421 | `L5.B4.K` | `layer5.block4.output_truncation_k` | `block4_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 422 | `L5.B5.F.inv_std_fresh` | `layer5.block5.inv_std_fresh_sf` | `ctpt_attn_inv_std` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 423 | `L5.B5.F.x_centered_fresh` | `layer5.block5.x_centered_fresh_sf` | `ctpt_attn_x_centered` | `fresh` | 4 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 424 | `L5.B5.M.gamma` | `layer5.block5.gamma_sf` | `ctpt_gamma_attn` | `encoding` | 1 | `scaling_factor` | 18 | True | 16384 | 20 |  |
| 425 | `L5.B5.W.wffn1` | `layer5.block5.wffn1_sf` | `ctpt_wffn1` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 426 | `L5.B5.M.gelu_coeff` | `layer5.block5.gelu_coeff_sf` | `ctpt_gelu_coeff` | `encoding` | 1 | `scaling_factor` | 18 | True | 16384 | 20 |  |
| 427 | `L5.B5.R.normalize_r` | `layer5.block5.normalize_rescale_sf` | `ctct_attn_normalize_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 428 | `L5.B5.R.gamma_r` | `layer5.block5.gamma_rescale_sf` | `ctct_gamma_attn_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 429 | `L5.B5.R.wffn1_r` | `layer5.block5.wffn1_rescale_sf` | `ctct_wffn1_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 430 | `L5.B5.R.gp0` | `layer5.block5.gelu_power_rescale_sf_0` | `ctct_gelu_x2` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 431 | `L5.B5.R.gp1` | `layer5.block5.gelu_power_rescale_sf_1` | `ctct_gelu_x3` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 432 | `L5.B5.R.gp2` | `layer5.block5.gelu_power_rescale_sf_2` | `ctct_gelu_x4` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 433 | `L5.B5.R.gc0` | `layer5.block5.gelu_coeff_mul_rescale_sf_0` | `ctct_gelu_b_x` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 434 | `L5.B5.R.gc1` | `layer5.block5.gelu_coeff_mul_rescale_sf_1` | `ctct_gelu_c_x2` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 435 | `L5.B5.R.gc2` | `layer5.block5.gelu_coeff_mul_rescale_sf_2` | `ctct_gelu_d_x3` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 436 | `L5.B5.R.gc3` | `layer5.block5.gelu_coeff_mul_rescale_sf_3` | `ctct_gelu_e_x4` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 437 | `L5.B5.K` | `layer5.block5.output_truncation_k` | `block5_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 438 | `L6.B1.F.gelu_out` | `layer6.block1.gelu_out_sf` | `ctpt_gelu_out` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 439 | `L6.B1.W.wffn2` | `layer6.block1.wffn2_sf` | `ctpt_ffn2` | `encoding` | 4 | `scaling_factor` | 20 | True | 8192 | 20 |  |
| 440 | `L6.B1.S.mean_inv_d` | `layer6.block1.mean_inv_d_sf` | `ctpt_inv_d_1` | `scalar` | 2 | `scaling_factor` | 20 | True | 8192 | 20 |  |
| 441 | `L6.B1.S.var_inv_d` | `layer6.block1.var_inv_d_sf` | `ctpt_inv_d_2` | `scalar` | 2 | `scaling_factor` | 20 | True | 8192 | 20 |  |
| 442 | `L6.B1.R.wffn2_r` | `layer6.block1.wffn2_rescale_sf` | `ctct_ffn2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 443 | `L6.B1.R.mean_r` | `layer6.block1.mean_rescale_sf` | `ctct_mean_rescale` | `rescale` | 3 | `scaling_factor` | 34 | True | 8192 | 34 |  |
| 444 | `L6.B1.R.square_r` | `layer6.block1.square_rescale_sf` | `ctct_ext_square` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 445 | `L6.B1.R.var_r` | `layer6.block1.var_rescale_sf` | `ctct_var_rescale` | `rescale` | 3 | `scaling_factor` | 34 | True | 8192 | 34 |  |
| 446 | `L6.B1.K` | `layer6.block1.output_truncation_k` | `block1_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 447 | `L6.B2.F.inv_std_fresh` | `layer6.block2.inv_std_fresh_sf` | `ctpt_inv_std` | `fresh` | 4 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 448 | `L6.B2.F.x_centered_fresh` | `layer6.block2.x_centered_fresh_sf` | `ctpt_x_centered` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 449 | `L6.B2.M.gamma` | `layer6.block2.gamma_sf` | `ctpt_gamma` | `encoding` | 1 | `scaling_factor` | 18 | True | 16384 | 20 |  |
| 450 | `L6.B2.W.wq` | `layer6.block2.wq_sf` | `ctpt_wq` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 451 | `L6.B2.W.wk` | `layer6.block2.wk_sf` | `ctpt_wk` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 452 | `L6.B2.W.wv` | `layer6.block2.wv_sf` | `ctpt_wv` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 453 | `L6.B2.M.kt_mask1` | `layer6.block2.kt_mask1_sf` | `ctpt_kt_mask1` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 454 | `L6.B2.M.kt_mask2` | `layer6.block2.kt_mask2_sf` | `ctpt_kt_mask2` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 455 | `L6.B2.M.q_mask1` | `layer6.block2.q_mask1_sf` | `ctpt_q_mask1` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 456 | `L6.B2.M.q_mask2` | `layer6.block2.q_mask2_sf` | `ctpt_q_mask2` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 457 | `L6.B2.M.qkt_merge_mask` | `layer6.block2.qkt_merge_mask_sf` | `ctpt_qkt_merge_mask` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 458 | `L6.B2.R.normalize_r` | `layer6.block2.normalize_rescale_sf` | `ctct_normalize_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 459 | `L6.B2.R.gamma_r` | `layer6.block2.gamma_rescale_sf` | `ctct_gamma_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 460 | `L6.B2.R.wk_r` | `layer6.block2.wk_rescale_sf` | `ctct_wk_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 461 | `L6.B2.R.wq_r` | `layer6.block2.wq_rescale_sf` | `ctct_wq_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 462 | `L6.B2.R.wv_r` | `layer6.block2.wv_rescale_sf` | `ctct_wv_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 463 | `L6.B2.R.kt_mask1_r` | `layer6.block2.kt_mask1_rescale_sf` | `ctct_kt_mask1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 464 | `L6.B2.R.kt_mask2_r` | `layer6.block2.kt_mask2_rescale_sf` | `ctct_kt_mask2_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 465 | `L6.B2.R.q_mask1_r` | `layer6.block2.q_mask1_rescale_sf` | `ctct_q_mask1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 466 | `L6.B2.R.q_mask2_r` | `layer6.block2.q_mask2_rescale_sf` | `ctct_q_mask2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 467 | `L6.B2.R.qkt_matmul_r` | `layer6.block2.qkt_matmul_rescale_sf` | `ctct_qkt_matmul_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 468 | `L6.B2.R.qkt_merge_mask_r` | `layer6.block2.qkt_merge_mask_rescale_sf` | `ctct_qkt_merge_mask_rescale` | `rescale` | 3 | `scaling_factor` | 28 | True | 16384 | 28 |  |
| 469 | `L6.B2.K` | `layer6.block2.output_truncation_k` | `block2_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 470 | `L6.B3.F.x_fresh` | `layer6.block3.x_fresh_sf` | `ctpt_softmax_x` | `fresh` | 4 | `scaling_factor` | 28 | True | 16384 | 28 |  |
| 471 | `L6.B3.S.inv_2n` | `layer6.block3.inv_2n_sf` | `ctpt_softmax_inv_2n` | `scalar` | 2 | `scaling_factor` | 15 | True | 16384 | 15 |  |
| 472 | `L6.B3.R.x_inv_2n_r` | `layer6.block3.x_inv_2n_rescale_sf` | `ctct_softmax_x_inv_2n_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 473 | `L6.B3.R.sq0` | `layer6.block3.square_rescale_sf_0` | `ctct_softmax_pow_s1` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 474 | `L6.B3.R.sq1` | `layer6.block3.square_rescale_sf_1` | `ctct_softmax_pow_s2` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 475 | `L6.B3.R.sq2` | `layer6.block3.square_rescale_sf_2` | `ctct_softmax_pow_s3` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 476 | `L6.B3.R.sq3` | `layer6.block3.square_rescale_sf_3` | `ctct_softmax_pow_s4` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 477 | `L6.B3.K` | `layer6.block3.output_truncation_k` | `block3_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 478 | `L6.B4.F.softmax_out_fresh` | `layer6.block4.softmax_out_fresh_sf` | `ctpt_softmax_out` | `fresh` | 4 | `scaling_factor` | 35 | True | 16384 | 35 |  |
| 479 | `L6.B4.F.v_fresh` | `layer6.block4.v_fresh_sf` | `ctpt_v` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 480 | `L6.B4.M.softmax_out_mask` | `layer6.block4.softmax_out_mask_sf` | `ctpt_softmax_out_mask` | `encoding` | 1 | `scaling_factor` | 12 | True | 16384 | 14 |  |
| 481 | `L6.B4.M.v_mask` | `layer6.block4.v_mask_sf` | `ctpt_v_mask` | `encoding` | 1 | `scaling_factor` | 20 | True | 16384 | 22 |  |
| 482 | `L6.B4.M.softmax_v_mask` | `layer6.block4.softmax_v_mask_sf` | `ctpt_softmax_v_mask` | `encoding` | 1 | `scaling_factor` | 12 | True | 16384 | 14 |  |
| 483 | `L6.B4.S.ln_mean_inv_d` | `layer6.block4.ln_mean_inv_d_sf` | `ctpt_inv_d_attn_mean` | `scalar` | 2 | `scaling_factor` | 20 | True | 16384 | 20 |  |
| 484 | `L6.B4.S.ln_var_inv_d` | `layer6.block4.ln_var_inv_d_sf` | `ctpt_inv_d_attn_var` | `scalar` | 2 | `scaling_factor` | 20 | True | 16384 | 20 |  |
| 485 | `L6.B4.W.wo` | `layer6.block4.wo_sf` | `ctpt_wo` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 486 | `L6.B4.R.softmax_out_mask_r` | `layer6.block4.softmax_out_mask_rescale_sf` | `ctct_softmax_out_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 487 | `L6.B4.R.v_mask_r` | `layer6.block4.v_mask_rescale_sf` | `ctct_v_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 488 | `L6.B4.R.softmax_v_matmul_r` | `layer6.block4.softmax_v_matmul_rescale_sf` | `ctct_softmax_v_matmul_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 489 | `L6.B4.R.softmax_v_mask_r` | `layer6.block4.softmax_v_mask_rescale_sf` | `ctct_softmax_v_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 490 | `L6.B4.R.wo_r` | `layer6.block4.wo_rescale_sf` | `ctct_wo_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 491 | `L6.B4.R.ln_mean_r` | `layer6.block4.ln_mean_rescale_sf` | `ctct_attn_mean_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 492 | `L6.B4.R.ln_square_r` | `layer6.block4.ln_square_rescale_sf` | `ctct_attn_square_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 493 | `L6.B4.R.ln_var_r` | `layer6.block4.ln_var_rescale_sf` | `ctct_attn_var_rescale` | `rescale` | 3 | `scaling_factor` | 28 | True | 16384 | 28 |  |
| 494 | `L6.B4.K` | `layer6.block4.output_truncation_k` | `block4_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 495 | `L6.B5.F.inv_std_fresh` | `layer6.block5.inv_std_fresh_sf` | `ctpt_attn_inv_std` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 496 | `L6.B5.F.x_centered_fresh` | `layer6.block5.x_centered_fresh_sf` | `ctpt_attn_x_centered` | `fresh` | 4 | `scaling_factor` | 31 | True | 8192 | 31 |  |
| 497 | `L6.B5.M.gamma` | `layer6.block5.gamma_sf` | `ctpt_gamma_attn` | `encoding` | 1 | `scaling_factor` | 18 | True | 8192 | 20 |  |
| 498 | `L6.B5.W.wffn1` | `layer6.block5.wffn1_sf` | `ctpt_wffn1` | `encoding` | 4 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 499 | `L6.B5.M.gelu_coeff` | `layer6.block5.gelu_coeff_sf` | `ctpt_gelu_coeff` | `encoding` | 1 | `scaling_factor` | 18 | True | 8192 | 20 |  |
| 500 | `L6.B5.R.normalize_r` | `layer6.block5.normalize_rescale_sf` | `ctct_attn_normalize_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 8192 | 31 |  |
| 501 | `L6.B5.R.gamma_r` | `layer6.block5.gamma_rescale_sf` | `ctct_gamma_attn_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 502 | `L6.B5.R.wffn1_r` | `layer6.block5.wffn1_rescale_sf` | `ctct_wffn1_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 8192 | 31 |  |
| 503 | `L6.B5.R.gp0` | `layer6.block5.gelu_power_rescale_sf_0` | `ctct_gelu_x2` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 31 | GELU degree 1 does not use this power-rescale slot |
| 504 | `L6.B5.R.gp1` | `layer6.block5.gelu_power_rescale_sf_1` | `ctct_gelu_x3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 505 | `L6.B5.R.gp2` | `layer6.block5.gelu_power_rescale_sf_2` | `ctct_gelu_x4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 506 | `L6.B5.R.gc0` | `layer6.block5.gelu_coeff_mul_rescale_sf_0` | `ctct_gelu_b_x` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 507 | `L6.B5.R.gc1` | `layer6.block5.gelu_coeff_mul_rescale_sf_1` | `ctct_gelu_c_x2` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 508 | `L6.B5.R.gc2` | `layer6.block5.gelu_coeff_mul_rescale_sf_2` | `ctct_gelu_d_x3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 509 | `L6.B5.R.gc3` | `layer6.block5.gelu_coeff_mul_rescale_sf_3` | `ctct_gelu_e_x4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 31 | GELU degree 1 does not use this coefficient-rescale slot |
| 510 | `L6.B5.K` | `layer6.block5.output_truncation_k` | `block5_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 511 | `L7.B1.F.gelu_out` | `layer7.block1.gelu_out_sf` | `ctpt_gelu_out` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 512 | `L7.B1.W.wffn2` | `layer7.block1.wffn2_sf` | `ctpt_ffn2` | `encoding` | 4 | `scaling_factor` | 20 | True | 8192 | 20 |  |
| 513 | `L7.B1.S.mean_inv_d` | `layer7.block1.mean_inv_d_sf` | `ctpt_inv_d_1` | `scalar` | 2 | `scaling_factor` | 20 | True | 8192 | 20 |  |
| 514 | `L7.B1.S.var_inv_d` | `layer7.block1.var_inv_d_sf` | `ctpt_inv_d_2` | `scalar` | 2 | `scaling_factor` | 20 | True | 8192 | 20 |  |
| 515 | `L7.B1.R.wffn2_r` | `layer7.block1.wffn2_rescale_sf` | `ctct_ffn2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 516 | `L7.B1.R.mean_r` | `layer7.block1.mean_rescale_sf` | `ctct_mean_rescale` | `rescale` | 3 | `scaling_factor` | 34 | True | 8192 | 34 |  |
| 517 | `L7.B1.R.square_r` | `layer7.block1.square_rescale_sf` | `ctct_ext_square` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 518 | `L7.B1.R.var_r` | `layer7.block1.var_rescale_sf` | `ctct_var_rescale` | `rescale` | 3 | `scaling_factor` | 34 | True | 8192 | 34 |  |
| 519 | `L7.B1.K` | `layer7.block1.output_truncation_k` | `block1_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 520 | `L7.B2.F.inv_std_fresh` | `layer7.block2.inv_std_fresh_sf` | `ctpt_inv_std` | `fresh` | 4 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 521 | `L7.B2.F.x_centered_fresh` | `layer7.block2.x_centered_fresh_sf` | `ctpt_x_centered` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 522 | `L7.B2.M.gamma` | `layer7.block2.gamma_sf` | `ctpt_gamma` | `encoding` | 1 | `scaling_factor` | 18 | True | 16384 | 20 |  |
| 523 | `L7.B2.W.wq` | `layer7.block2.wq_sf` | `ctpt_wq` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 524 | `L7.B2.W.wk` | `layer7.block2.wk_sf` | `ctpt_wk` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 525 | `L7.B2.W.wv` | `layer7.block2.wv_sf` | `ctpt_wv` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 526 | `L7.B2.M.kt_mask1` | `layer7.block2.kt_mask1_sf` | `ctpt_kt_mask1` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 527 | `L7.B2.M.kt_mask2` | `layer7.block2.kt_mask2_sf` | `ctpt_kt_mask2` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 528 | `L7.B2.M.q_mask1` | `layer7.block2.q_mask1_sf` | `ctpt_q_mask1` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 529 | `L7.B2.M.q_mask2` | `layer7.block2.q_mask2_sf` | `ctpt_q_mask2` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 530 | `L7.B2.M.qkt_merge_mask` | `layer7.block2.qkt_merge_mask_sf` | `ctpt_qkt_merge_mask` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 531 | `L7.B2.R.normalize_r` | `layer7.block2.normalize_rescale_sf` | `ctct_normalize_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 532 | `L7.B2.R.gamma_r` | `layer7.block2.gamma_rescale_sf` | `ctct_gamma_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 533 | `L7.B2.R.wk_r` | `layer7.block2.wk_rescale_sf` | `ctct_wk_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 534 | `L7.B2.R.wq_r` | `layer7.block2.wq_rescale_sf` | `ctct_wq_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 535 | `L7.B2.R.wv_r` | `layer7.block2.wv_rescale_sf` | `ctct_wv_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 536 | `L7.B2.R.kt_mask1_r` | `layer7.block2.kt_mask1_rescale_sf` | `ctct_kt_mask1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 537 | `L7.B2.R.kt_mask2_r` | `layer7.block2.kt_mask2_rescale_sf` | `ctct_kt_mask2_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 538 | `L7.B2.R.q_mask1_r` | `layer7.block2.q_mask1_rescale_sf` | `ctct_q_mask1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 539 | `L7.B2.R.q_mask2_r` | `layer7.block2.q_mask2_rescale_sf` | `ctct_q_mask2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 540 | `L7.B2.R.qkt_matmul_r` | `layer7.block2.qkt_matmul_rescale_sf` | `ctct_qkt_matmul_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 541 | `L7.B2.R.qkt_merge_mask_r` | `layer7.block2.qkt_merge_mask_rescale_sf` | `ctct_qkt_merge_mask_rescale` | `rescale` | 3 | `scaling_factor` | 28 | True | 16384 | 28 |  |
| 542 | `L7.B2.K` | `layer7.block2.output_truncation_k` | `block2_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 543 | `L7.B3.F.x_fresh` | `layer7.block3.x_fresh_sf` | `ctpt_softmax_x` | `fresh` | 4 | `scaling_factor` | 28 | True | 8192 | 28 |  |
| 544 | `L7.B3.S.inv_2n` | `layer7.block3.inv_2n_sf` | `ctpt_softmax_inv_2n` | `scalar` | 2 | `scaling_factor` | 15 | True | 8192 | 15 |  |
| 545 | `L7.B3.R.x_inv_2n_r` | `layer7.block3.x_inv_2n_rescale_sf` | `ctct_softmax_x_inv_2n_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 546 | `L7.B3.R.sq0` | `layer7.block3.square_rescale_sf_0` | `ctct_softmax_pow_s1` | `rescale` | 3 | `scaling_factor` | 31 | True | 8192 | 31 |  |
| 547 | `L7.B3.R.sq1` | `layer7.block3.square_rescale_sf_1` | `ctct_softmax_pow_s2` | `rescale` | 3 | `scaling_factor` | 31 | True | 8192 | 31 |  |
| 548 | `L7.B3.R.sq2` | `layer7.block3.square_rescale_sf_2` | `ctct_softmax_pow_s3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 31 | softmax degree 2 does not use this square-rescale slot |
| 549 | `L7.B3.R.sq3` | `layer7.block3.square_rescale_sf_3` | `ctct_softmax_pow_s4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 31 | softmax degree 2 does not use this square-rescale slot |
| 550 | `L7.B3.K` | `layer7.block3.output_truncation_k` | `block3_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 551 | `L7.B4.F.softmax_out_fresh` | `layer7.block4.softmax_out_fresh_sf` | `ctpt_softmax_out` | `fresh` | 4 | `scaling_factor` | 35 | True | 16384 | 35 |  |
| 552 | `L7.B4.F.v_fresh` | `layer7.block4.v_fresh_sf` | `ctpt_v` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 553 | `L7.B4.M.softmax_out_mask` | `layer7.block4.softmax_out_mask_sf` | `ctpt_softmax_out_mask` | `encoding` | 1 | `scaling_factor` | 12 | True | 16384 | 14 |  |
| 554 | `L7.B4.M.v_mask` | `layer7.block4.v_mask_sf` | `ctpt_v_mask` | `encoding` | 1 | `scaling_factor` | 20 | True | 16384 | 22 |  |
| 555 | `L7.B4.M.softmax_v_mask` | `layer7.block4.softmax_v_mask_sf` | `ctpt_softmax_v_mask` | `encoding` | 1 | `scaling_factor` | 12 | True | 16384 | 14 |  |
| 556 | `L7.B4.S.ln_mean_inv_d` | `layer7.block4.ln_mean_inv_d_sf` | `ctpt_inv_d_attn_mean` | `scalar` | 2 | `scaling_factor` | 20 | True | 16384 | 20 |  |
| 557 | `L7.B4.S.ln_var_inv_d` | `layer7.block4.ln_var_inv_d_sf` | `ctpt_inv_d_attn_var` | `scalar` | 2 | `scaling_factor` | 20 | True | 16384 | 20 |  |
| 558 | `L7.B4.W.wo` | `layer7.block4.wo_sf` | `ctpt_wo` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 559 | `L7.B4.R.softmax_out_mask_r` | `layer7.block4.softmax_out_mask_rescale_sf` | `ctct_softmax_out_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 560 | `L7.B4.R.v_mask_r` | `layer7.block4.v_mask_rescale_sf` | `ctct_v_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 561 | `L7.B4.R.softmax_v_matmul_r` | `layer7.block4.softmax_v_matmul_rescale_sf` | `ctct_softmax_v_matmul_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 562 | `L7.B4.R.softmax_v_mask_r` | `layer7.block4.softmax_v_mask_rescale_sf` | `ctct_softmax_v_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 563 | `L7.B4.R.wo_r` | `layer7.block4.wo_rescale_sf` | `ctct_wo_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 564 | `L7.B4.R.ln_mean_r` | `layer7.block4.ln_mean_rescale_sf` | `ctct_attn_mean_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 565 | `L7.B4.R.ln_square_r` | `layer7.block4.ln_square_rescale_sf` | `ctct_attn_square_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 566 | `L7.B4.R.ln_var_r` | `layer7.block4.ln_var_rescale_sf` | `ctct_attn_var_rescale` | `rescale` | 3 | `scaling_factor` | 28 | True | 16384 | 28 |  |
| 567 | `L7.B4.K` | `layer7.block4.output_truncation_k` | `block4_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 568 | `L7.B5.F.inv_std_fresh` | `layer7.block5.inv_std_fresh_sf` | `ctpt_attn_inv_std` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 569 | `L7.B5.F.x_centered_fresh` | `layer7.block5.x_centered_fresh_sf` | `ctpt_attn_x_centered` | `fresh` | 4 | `scaling_factor` | 31 | True | 8192 | 31 |  |
| 570 | `L7.B5.M.gamma` | `layer7.block5.gamma_sf` | `ctpt_gamma_attn` | `encoding` | 1 | `scaling_factor` | 18 | True | 8192 | 20 |  |
| 571 | `L7.B5.W.wffn1` | `layer7.block5.wffn1_sf` | `ctpt_wffn1` | `encoding` | 4 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 572 | `L7.B5.M.gelu_coeff` | `layer7.block5.gelu_coeff_sf` | `ctpt_gelu_coeff` | `encoding` | 1 | `scaling_factor` | 18 | True | 8192 | 20 |  |
| 573 | `L7.B5.R.normalize_r` | `layer7.block5.normalize_rescale_sf` | `ctct_attn_normalize_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 8192 | 31 |  |
| 574 | `L7.B5.R.gamma_r` | `layer7.block5.gamma_rescale_sf` | `ctct_gamma_attn_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 575 | `L7.B5.R.wffn1_r` | `layer7.block5.wffn1_rescale_sf` | `ctct_wffn1_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 8192 | 31 |  |
| 576 | `L7.B5.R.gp0` | `layer7.block5.gelu_power_rescale_sf_0` | `ctct_gelu_x2` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 31 | GELU degree 1 does not use this power-rescale slot |
| 577 | `L7.B5.R.gp1` | `layer7.block5.gelu_power_rescale_sf_1` | `ctct_gelu_x3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 578 | `L7.B5.R.gp2` | `layer7.block5.gelu_power_rescale_sf_2` | `ctct_gelu_x4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 579 | `L7.B5.R.gc0` | `layer7.block5.gelu_coeff_mul_rescale_sf_0` | `ctct_gelu_b_x` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 580 | `L7.B5.R.gc1` | `layer7.block5.gelu_coeff_mul_rescale_sf_1` | `ctct_gelu_c_x2` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 581 | `L7.B5.R.gc2` | `layer7.block5.gelu_coeff_mul_rescale_sf_2` | `ctct_gelu_d_x3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 582 | `L7.B5.R.gc3` | `layer7.block5.gelu_coeff_mul_rescale_sf_3` | `ctct_gelu_e_x4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 31 | GELU degree 1 does not use this coefficient-rescale slot |
| 583 | `L7.B5.K` | `layer7.block5.output_truncation_k` | `block5_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 584 | `L8.B1.F.gelu_out` | `layer8.block1.gelu_out_sf` | `ctpt_gelu_out` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 585 | `L8.B1.W.wffn2` | `layer8.block1.wffn2_sf` | `ctpt_ffn2` | `encoding` | 4 | `scaling_factor` | 20 | True | 8192 | 20 |  |
| 586 | `L8.B1.S.mean_inv_d` | `layer8.block1.mean_inv_d_sf` | `ctpt_inv_d_1` | `scalar` | 2 | `scaling_factor` | 20 | True | 8192 | 20 |  |
| 587 | `L8.B1.S.var_inv_d` | `layer8.block1.var_inv_d_sf` | `ctpt_inv_d_2` | `scalar` | 2 | `scaling_factor` | 20 | True | 8192 | 20 |  |
| 588 | `L8.B1.R.wffn2_r` | `layer8.block1.wffn2_rescale_sf` | `ctct_ffn2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 589 | `L8.B1.R.mean_r` | `layer8.block1.mean_rescale_sf` | `ctct_mean_rescale` | `rescale` | 3 | `scaling_factor` | 34 | True | 8192 | 34 |  |
| 590 | `L8.B1.R.square_r` | `layer8.block1.square_rescale_sf` | `ctct_ext_square` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 591 | `L8.B1.R.var_r` | `layer8.block1.var_rescale_sf` | `ctct_var_rescale` | `rescale` | 3 | `scaling_factor` | 34 | True | 8192 | 34 |  |
| 592 | `L8.B1.K` | `layer8.block1.output_truncation_k` | `block1_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 593 | `L8.B2.F.inv_std_fresh` | `layer8.block2.inv_std_fresh_sf` | `ctpt_inv_std` | `fresh` | 4 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 594 | `L8.B2.F.x_centered_fresh` | `layer8.block2.x_centered_fresh_sf` | `ctpt_x_centered` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 595 | `L8.B2.M.gamma` | `layer8.block2.gamma_sf` | `ctpt_gamma` | `encoding` | 1 | `scaling_factor` | 18 | True | 16384 | 20 |  |
| 596 | `L8.B2.W.wq` | `layer8.block2.wq_sf` | `ctpt_wq` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 597 | `L8.B2.W.wk` | `layer8.block2.wk_sf` | `ctpt_wk` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 598 | `L8.B2.W.wv` | `layer8.block2.wv_sf` | `ctpt_wv` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 599 | `L8.B2.M.kt_mask1` | `layer8.block2.kt_mask1_sf` | `ctpt_kt_mask1` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 600 | `L8.B2.M.kt_mask2` | `layer8.block2.kt_mask2_sf` | `ctpt_kt_mask2` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 601 | `L8.B2.M.q_mask1` | `layer8.block2.q_mask1_sf` | `ctpt_q_mask1` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 602 | `L8.B2.M.q_mask2` | `layer8.block2.q_mask2_sf` | `ctpt_q_mask2` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 603 | `L8.B2.M.qkt_merge_mask` | `layer8.block2.qkt_merge_mask_sf` | `ctpt_qkt_merge_mask` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 604 | `L8.B2.R.normalize_r` | `layer8.block2.normalize_rescale_sf` | `ctct_normalize_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 605 | `L8.B2.R.gamma_r` | `layer8.block2.gamma_rescale_sf` | `ctct_gamma_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 606 | `L8.B2.R.wk_r` | `layer8.block2.wk_rescale_sf` | `ctct_wk_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 607 | `L8.B2.R.wq_r` | `layer8.block2.wq_rescale_sf` | `ctct_wq_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 608 | `L8.B2.R.wv_r` | `layer8.block2.wv_rescale_sf` | `ctct_wv_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 609 | `L8.B2.R.kt_mask1_r` | `layer8.block2.kt_mask1_rescale_sf` | `ctct_kt_mask1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 610 | `L8.B2.R.kt_mask2_r` | `layer8.block2.kt_mask2_rescale_sf` | `ctct_kt_mask2_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 611 | `L8.B2.R.q_mask1_r` | `layer8.block2.q_mask1_rescale_sf` | `ctct_q_mask1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 612 | `L8.B2.R.q_mask2_r` | `layer8.block2.q_mask2_rescale_sf` | `ctct_q_mask2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 613 | `L8.B2.R.qkt_matmul_r` | `layer8.block2.qkt_matmul_rescale_sf` | `ctct_qkt_matmul_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 614 | `L8.B2.R.qkt_merge_mask_r` | `layer8.block2.qkt_merge_mask_rescale_sf` | `ctct_qkt_merge_mask_rescale` | `rescale` | 3 | `scaling_factor` | 28 | True | 16384 | 28 |  |
| 615 | `L8.B2.K` | `layer8.block2.output_truncation_k` | `block2_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 616 | `L8.B3.F.x_fresh` | `layer8.block3.x_fresh_sf` | `ctpt_softmax_x` | `fresh` | 4 | `scaling_factor` | 28 | True | 16384 | 28 |  |
| 617 | `L8.B3.S.inv_2n` | `layer8.block3.inv_2n_sf` | `ctpt_softmax_inv_2n` | `scalar` | 2 | `scaling_factor` | 15 | True | 16384 | 15 |  |
| 618 | `L8.B3.R.x_inv_2n_r` | `layer8.block3.x_inv_2n_rescale_sf` | `ctct_softmax_x_inv_2n_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 619 | `L8.B3.R.sq0` | `layer8.block3.square_rescale_sf_0` | `ctct_softmax_pow_s1` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 620 | `L8.B3.R.sq1` | `layer8.block3.square_rescale_sf_1` | `ctct_softmax_pow_s2` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 621 | `L8.B3.R.sq2` | `layer8.block3.square_rescale_sf_2` | `ctct_softmax_pow_s3` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 622 | `L8.B3.R.sq3` | `layer8.block3.square_rescale_sf_3` | `ctct_softmax_pow_s4` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 623 | `L8.B3.K` | `layer8.block3.output_truncation_k` | `block3_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 624 | `L8.B4.F.softmax_out_fresh` | `layer8.block4.softmax_out_fresh_sf` | `ctpt_softmax_out` | `fresh` | 4 | `scaling_factor` | 35 | True | 16384 | 35 |  |
| 625 | `L8.B4.F.v_fresh` | `layer8.block4.v_fresh_sf` | `ctpt_v` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 626 | `L8.B4.M.softmax_out_mask` | `layer8.block4.softmax_out_mask_sf` | `ctpt_softmax_out_mask` | `encoding` | 1 | `scaling_factor` | 12 | True | 16384 | 14 |  |
| 627 | `L8.B4.M.v_mask` | `layer8.block4.v_mask_sf` | `ctpt_v_mask` | `encoding` | 1 | `scaling_factor` | 20 | True | 16384 | 22 |  |
| 628 | `L8.B4.M.softmax_v_mask` | `layer8.block4.softmax_v_mask_sf` | `ctpt_softmax_v_mask` | `encoding` | 1 | `scaling_factor` | 12 | True | 16384 | 14 |  |
| 629 | `L8.B4.S.ln_mean_inv_d` | `layer8.block4.ln_mean_inv_d_sf` | `ctpt_inv_d_attn_mean` | `scalar` | 2 | `scaling_factor` | 20 | True | 16384 | 20 |  |
| 630 | `L8.B4.S.ln_var_inv_d` | `layer8.block4.ln_var_inv_d_sf` | `ctpt_inv_d_attn_var` | `scalar` | 2 | `scaling_factor` | 20 | True | 16384 | 20 |  |
| 631 | `L8.B4.W.wo` | `layer8.block4.wo_sf` | `ctpt_wo` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 632 | `L8.B4.R.softmax_out_mask_r` | `layer8.block4.softmax_out_mask_rescale_sf` | `ctct_softmax_out_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 633 | `L8.B4.R.v_mask_r` | `layer8.block4.v_mask_rescale_sf` | `ctct_v_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 634 | `L8.B4.R.softmax_v_matmul_r` | `layer8.block4.softmax_v_matmul_rescale_sf` | `ctct_softmax_v_matmul_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 635 | `L8.B4.R.softmax_v_mask_r` | `layer8.block4.softmax_v_mask_rescale_sf` | `ctct_softmax_v_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 636 | `L8.B4.R.wo_r` | `layer8.block4.wo_rescale_sf` | `ctct_wo_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 637 | `L8.B4.R.ln_mean_r` | `layer8.block4.ln_mean_rescale_sf` | `ctct_attn_mean_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 638 | `L8.B4.R.ln_square_r` | `layer8.block4.ln_square_rescale_sf` | `ctct_attn_square_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 639 | `L8.B4.R.ln_var_r` | `layer8.block4.ln_var_rescale_sf` | `ctct_attn_var_rescale` | `rescale` | 3 | `scaling_factor` | 28 | True | 16384 | 28 |  |
| 640 | `L8.B4.K` | `layer8.block4.output_truncation_k` | `block4_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 641 | `L8.B5.F.inv_std_fresh` | `layer8.block5.inv_std_fresh_sf` | `ctpt_attn_inv_std` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 642 | `L8.B5.F.x_centered_fresh` | `layer8.block5.x_centered_fresh_sf` | `ctpt_attn_x_centered` | `fresh` | 4 | `scaling_factor` | 31 | True | 8192 | 31 |  |
| 643 | `L8.B5.M.gamma` | `layer8.block5.gamma_sf` | `ctpt_gamma_attn` | `encoding` | 1 | `scaling_factor` | 18 | True | 8192 | 20 |  |
| 644 | `L8.B5.W.wffn1` | `layer8.block5.wffn1_sf` | `ctpt_wffn1` | `encoding` | 4 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 645 | `L8.B5.M.gelu_coeff` | `layer8.block5.gelu_coeff_sf` | `ctpt_gelu_coeff` | `encoding` | 1 | `scaling_factor` | 18 | True | 8192 | 20 |  |
| 646 | `L8.B5.R.normalize_r` | `layer8.block5.normalize_rescale_sf` | `ctct_attn_normalize_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 8192 | 31 |  |
| 647 | `L8.B5.R.gamma_r` | `layer8.block5.gamma_rescale_sf` | `ctct_gamma_attn_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 648 | `L8.B5.R.wffn1_r` | `layer8.block5.wffn1_rescale_sf` | `ctct_wffn1_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 8192 | 31 |  |
| 649 | `L8.B5.R.gp0` | `layer8.block5.gelu_power_rescale_sf_0` | `ctct_gelu_x2` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 31 | GELU degree 1 does not use this power-rescale slot |
| 650 | `L8.B5.R.gp1` | `layer8.block5.gelu_power_rescale_sf_1` | `ctct_gelu_x3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 651 | `L8.B5.R.gp2` | `layer8.block5.gelu_power_rescale_sf_2` | `ctct_gelu_x4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 652 | `L8.B5.R.gc0` | `layer8.block5.gelu_coeff_mul_rescale_sf_0` | `ctct_gelu_b_x` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 653 | `L8.B5.R.gc1` | `layer8.block5.gelu_coeff_mul_rescale_sf_1` | `ctct_gelu_c_x2` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 654 | `L8.B5.R.gc2` | `layer8.block5.gelu_coeff_mul_rescale_sf_2` | `ctct_gelu_d_x3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 655 | `L8.B5.R.gc3` | `layer8.block5.gelu_coeff_mul_rescale_sf_3` | `ctct_gelu_e_x4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 31 | GELU degree 1 does not use this coefficient-rescale slot |
| 656 | `L8.B5.K` | `layer8.block5.output_truncation_k` | `block5_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 657 | `L9.B1.F.gelu_out` | `layer9.block1.gelu_out_sf` | `ctpt_gelu_out` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 658 | `L9.B1.W.wffn2` | `layer9.block1.wffn2_sf` | `ctpt_ffn2` | `encoding` | 4 | `scaling_factor` | 20 | True | 8192 | 20 |  |
| 659 | `L9.B1.S.mean_inv_d` | `layer9.block1.mean_inv_d_sf` | `ctpt_inv_d_1` | `scalar` | 2 | `scaling_factor` | 20 | True | 8192 | 20 |  |
| 660 | `L9.B1.S.var_inv_d` | `layer9.block1.var_inv_d_sf` | `ctpt_inv_d_2` | `scalar` | 2 | `scaling_factor` | 20 | True | 8192 | 20 |  |
| 661 | `L9.B1.R.wffn2_r` | `layer9.block1.wffn2_rescale_sf` | `ctct_ffn2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 662 | `L9.B1.R.mean_r` | `layer9.block1.mean_rescale_sf` | `ctct_mean_rescale` | `rescale` | 3 | `scaling_factor` | 34 | True | 8192 | 34 |  |
| 663 | `L9.B1.R.square_r` | `layer9.block1.square_rescale_sf` | `ctct_ext_square` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 664 | `L9.B1.R.var_r` | `layer9.block1.var_rescale_sf` | `ctct_var_rescale` | `rescale` | 3 | `scaling_factor` | 34 | True | 8192 | 34 |  |
| 665 | `L9.B1.K` | `layer9.block1.output_truncation_k` | `block1_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 666 | `L9.B2.F.inv_std_fresh` | `layer9.block2.inv_std_fresh_sf` | `ctpt_inv_std` | `fresh` | 4 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 667 | `L9.B2.F.x_centered_fresh` | `layer9.block2.x_centered_fresh_sf` | `ctpt_x_centered` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 668 | `L9.B2.M.gamma` | `layer9.block2.gamma_sf` | `ctpt_gamma` | `encoding` | 1 | `scaling_factor` | 18 | True | 16384 | 20 |  |
| 669 | `L9.B2.W.wq` | `layer9.block2.wq_sf` | `ctpt_wq` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 670 | `L9.B2.W.wk` | `layer9.block2.wk_sf` | `ctpt_wk` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 671 | `L9.B2.W.wv` | `layer9.block2.wv_sf` | `ctpt_wv` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 672 | `L9.B2.M.kt_mask1` | `layer9.block2.kt_mask1_sf` | `ctpt_kt_mask1` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 673 | `L9.B2.M.kt_mask2` | `layer9.block2.kt_mask2_sf` | `ctpt_kt_mask2` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 674 | `L9.B2.M.q_mask1` | `layer9.block2.q_mask1_sf` | `ctpt_q_mask1` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 675 | `L9.B2.M.q_mask2` | `layer9.block2.q_mask2_sf` | `ctpt_q_mask2` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 676 | `L9.B2.M.qkt_merge_mask` | `layer9.block2.qkt_merge_mask_sf` | `ctpt_qkt_merge_mask` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 677 | `L9.B2.R.normalize_r` | `layer9.block2.normalize_rescale_sf` | `ctct_normalize_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 678 | `L9.B2.R.gamma_r` | `layer9.block2.gamma_rescale_sf` | `ctct_gamma_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 679 | `L9.B2.R.wk_r` | `layer9.block2.wk_rescale_sf` | `ctct_wk_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 680 | `L9.B2.R.wq_r` | `layer9.block2.wq_rescale_sf` | `ctct_wq_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 681 | `L9.B2.R.wv_r` | `layer9.block2.wv_rescale_sf` | `ctct_wv_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 682 | `L9.B2.R.kt_mask1_r` | `layer9.block2.kt_mask1_rescale_sf` | `ctct_kt_mask1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 683 | `L9.B2.R.kt_mask2_r` | `layer9.block2.kt_mask2_rescale_sf` | `ctct_kt_mask2_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 684 | `L9.B2.R.q_mask1_r` | `layer9.block2.q_mask1_rescale_sf` | `ctct_q_mask1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 685 | `L9.B2.R.q_mask2_r` | `layer9.block2.q_mask2_rescale_sf` | `ctct_q_mask2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 686 | `L9.B2.R.qkt_matmul_r` | `layer9.block2.qkt_matmul_rescale_sf` | `ctct_qkt_matmul_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 687 | `L9.B2.R.qkt_merge_mask_r` | `layer9.block2.qkt_merge_mask_rescale_sf` | `ctct_qkt_merge_mask_rescale` | `rescale` | 3 | `scaling_factor` | 28 | True | 16384 | 28 |  |
| 688 | `L9.B2.K` | `layer9.block2.output_truncation_k` | `block2_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 689 | `L9.B3.F.x_fresh` | `layer9.block3.x_fresh_sf` | `ctpt_softmax_x` | `fresh` | 4 | `scaling_factor` | 28 | True | 16384 | 28 |  |
| 690 | `L9.B3.S.inv_2n` | `layer9.block3.inv_2n_sf` | `ctpt_softmax_inv_2n` | `scalar` | 2 | `scaling_factor` | 15 | True | 16384 | 15 |  |
| 691 | `L9.B3.R.x_inv_2n_r` | `layer9.block3.x_inv_2n_rescale_sf` | `ctct_softmax_x_inv_2n_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 692 | `L9.B3.R.sq0` | `layer9.block3.square_rescale_sf_0` | `ctct_softmax_pow_s1` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 693 | `L9.B3.R.sq1` | `layer9.block3.square_rescale_sf_1` | `ctct_softmax_pow_s2` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 694 | `L9.B3.R.sq2` | `layer9.block3.square_rescale_sf_2` | `ctct_softmax_pow_s3` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 695 | `L9.B3.R.sq3` | `layer9.block3.square_rescale_sf_3` | `ctct_softmax_pow_s4` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 696 | `L9.B3.K` | `layer9.block3.output_truncation_k` | `block3_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 697 | `L9.B4.F.softmax_out_fresh` | `layer9.block4.softmax_out_fresh_sf` | `ctpt_softmax_out` | `fresh` | 4 | `scaling_factor` | 35 | True | 16384 | 35 |  |
| 698 | `L9.B4.F.v_fresh` | `layer9.block4.v_fresh_sf` | `ctpt_v` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 699 | `L9.B4.M.softmax_out_mask` | `layer9.block4.softmax_out_mask_sf` | `ctpt_softmax_out_mask` | `encoding` | 1 | `scaling_factor` | 12 | True | 16384 | 14 |  |
| 700 | `L9.B4.M.v_mask` | `layer9.block4.v_mask_sf` | `ctpt_v_mask` | `encoding` | 1 | `scaling_factor` | 20 | True | 16384 | 22 |  |
| 701 | `L9.B4.M.softmax_v_mask` | `layer9.block4.softmax_v_mask_sf` | `ctpt_softmax_v_mask` | `encoding` | 1 | `scaling_factor` | 12 | True | 16384 | 14 |  |
| 702 | `L9.B4.S.ln_mean_inv_d` | `layer9.block4.ln_mean_inv_d_sf` | `ctpt_inv_d_attn_mean` | `scalar` | 2 | `scaling_factor` | 20 | True | 16384 | 20 |  |
| 703 | `L9.B4.S.ln_var_inv_d` | `layer9.block4.ln_var_inv_d_sf` | `ctpt_inv_d_attn_var` | `scalar` | 2 | `scaling_factor` | 20 | True | 16384 | 20 |  |
| 704 | `L9.B4.W.wo` | `layer9.block4.wo_sf` | `ctpt_wo` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 705 | `L9.B4.R.softmax_out_mask_r` | `layer9.block4.softmax_out_mask_rescale_sf` | `ctct_softmax_out_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 706 | `L9.B4.R.v_mask_r` | `layer9.block4.v_mask_rescale_sf` | `ctct_v_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 707 | `L9.B4.R.softmax_v_matmul_r` | `layer9.block4.softmax_v_matmul_rescale_sf` | `ctct_softmax_v_matmul_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 708 | `L9.B4.R.softmax_v_mask_r` | `layer9.block4.softmax_v_mask_rescale_sf` | `ctct_softmax_v_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 709 | `L9.B4.R.wo_r` | `layer9.block4.wo_rescale_sf` | `ctct_wo_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 710 | `L9.B4.R.ln_mean_r` | `layer9.block4.ln_mean_rescale_sf` | `ctct_attn_mean_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 711 | `L9.B4.R.ln_square_r` | `layer9.block4.ln_square_rescale_sf` | `ctct_attn_square_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 712 | `L9.B4.R.ln_var_r` | `layer9.block4.ln_var_rescale_sf` | `ctct_attn_var_rescale` | `rescale` | 3 | `scaling_factor` | 28 | True | 16384 | 28 |  |
| 713 | `L9.B4.K` | `layer9.block4.output_truncation_k` | `block4_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 714 | `L9.B5.F.inv_std_fresh` | `layer9.block5.inv_std_fresh_sf` | `ctpt_attn_inv_std` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 715 | `L9.B5.F.x_centered_fresh` | `layer9.block5.x_centered_fresh_sf` | `ctpt_attn_x_centered` | `fresh` | 4 | `scaling_factor` | 31 | True | 8192 | 31 |  |
| 716 | `L9.B5.M.gamma` | `layer9.block5.gamma_sf` | `ctpt_gamma_attn` | `encoding` | 1 | `scaling_factor` | 18 | True | 8192 | 20 |  |
| 717 | `L9.B5.W.wffn1` | `layer9.block5.wffn1_sf` | `ctpt_wffn1` | `encoding` | 4 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 718 | `L9.B5.M.gelu_coeff` | `layer9.block5.gelu_coeff_sf` | `ctpt_gelu_coeff` | `encoding` | 1 | `scaling_factor` | 18 | True | 8192 | 20 |  |
| 719 | `L9.B5.R.normalize_r` | `layer9.block5.normalize_rescale_sf` | `ctct_attn_normalize_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 8192 | 31 |  |
| 720 | `L9.B5.R.gamma_r` | `layer9.block5.gamma_rescale_sf` | `ctct_gamma_attn_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 721 | `L9.B5.R.wffn1_r` | `layer9.block5.wffn1_rescale_sf` | `ctct_wffn1_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 8192 | 31 |  |
| 722 | `L9.B5.R.gp0` | `layer9.block5.gelu_power_rescale_sf_0` | `ctct_gelu_x2` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 31 | GELU degree 1 does not use this power-rescale slot |
| 723 | `L9.B5.R.gp1` | `layer9.block5.gelu_power_rescale_sf_1` | `ctct_gelu_x3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 724 | `L9.B5.R.gp2` | `layer9.block5.gelu_power_rescale_sf_2` | `ctct_gelu_x4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 725 | `L9.B5.R.gc0` | `layer9.block5.gelu_coeff_mul_rescale_sf_0` | `ctct_gelu_b_x` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 726 | `L9.B5.R.gc1` | `layer9.block5.gelu_coeff_mul_rescale_sf_1` | `ctct_gelu_c_x2` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 727 | `L9.B5.R.gc2` | `layer9.block5.gelu_coeff_mul_rescale_sf_2` | `ctct_gelu_d_x3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 728 | `L9.B5.R.gc3` | `layer9.block5.gelu_coeff_mul_rescale_sf_3` | `ctct_gelu_e_x4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 31 | GELU degree 1 does not use this coefficient-rescale slot |
| 729 | `L9.B5.K` | `layer9.block5.output_truncation_k` | `block5_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 730 | `L10.B1.F.gelu_out` | `layer10.block1.gelu_out_sf` | `ctpt_gelu_out` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 731 | `L10.B1.W.wffn2` | `layer10.block1.wffn2_sf` | `ctpt_ffn2` | `encoding` | 4 | `scaling_factor` | 20 | True | 8192 | 20 |  |
| 732 | `L10.B1.S.mean_inv_d` | `layer10.block1.mean_inv_d_sf` | `ctpt_inv_d_1` | `scalar` | 2 | `scaling_factor` | 20 | True | 8192 | 20 |  |
| 733 | `L10.B1.S.var_inv_d` | `layer10.block1.var_inv_d_sf` | `ctpt_inv_d_2` | `scalar` | 2 | `scaling_factor` | 20 | True | 8192 | 20 |  |
| 734 | `L10.B1.R.wffn2_r` | `layer10.block1.wffn2_rescale_sf` | `ctct_ffn2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 735 | `L10.B1.R.mean_r` | `layer10.block1.mean_rescale_sf` | `ctct_mean_rescale` | `rescale` | 3 | `scaling_factor` | 34 | True | 8192 | 34 |  |
| 736 | `L10.B1.R.square_r` | `layer10.block1.square_rescale_sf` | `ctct_ext_square` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 737 | `L10.B1.R.var_r` | `layer10.block1.var_rescale_sf` | `ctct_var_rescale` | `rescale` | 3 | `scaling_factor` | 34 | True | 8192 | 34 |  |
| 738 | `L10.B1.K` | `layer10.block1.output_truncation_k` | `block1_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 739 | `L10.B2.F.inv_std_fresh` | `layer10.block2.inv_std_fresh_sf` | `ctpt_inv_std` | `fresh` | 4 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 740 | `L10.B2.F.x_centered_fresh` | `layer10.block2.x_centered_fresh_sf` | `ctpt_x_centered` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 741 | `L10.B2.M.gamma` | `layer10.block2.gamma_sf` | `ctpt_gamma` | `encoding` | 1 | `scaling_factor` | 18 | True | 16384 | 20 |  |
| 742 | `L10.B2.W.wq` | `layer10.block2.wq_sf` | `ctpt_wq` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 743 | `L10.B2.W.wk` | `layer10.block2.wk_sf` | `ctpt_wk` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 744 | `L10.B2.W.wv` | `layer10.block2.wv_sf` | `ctpt_wv` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 745 | `L10.B2.M.kt_mask1` | `layer10.block2.kt_mask1_sf` | `ctpt_kt_mask1` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 746 | `L10.B2.M.kt_mask2` | `layer10.block2.kt_mask2_sf` | `ctpt_kt_mask2` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 747 | `L10.B2.M.q_mask1` | `layer10.block2.q_mask1_sf` | `ctpt_q_mask1` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 748 | `L10.B2.M.q_mask2` | `layer10.block2.q_mask2_sf` | `ctpt_q_mask2` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 749 | `L10.B2.M.qkt_merge_mask` | `layer10.block2.qkt_merge_mask_sf` | `ctpt_qkt_merge_mask` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 750 | `L10.B2.R.normalize_r` | `layer10.block2.normalize_rescale_sf` | `ctct_normalize_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 751 | `L10.B2.R.gamma_r` | `layer10.block2.gamma_rescale_sf` | `ctct_gamma_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 752 | `L10.B2.R.wk_r` | `layer10.block2.wk_rescale_sf` | `ctct_wk_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 753 | `L10.B2.R.wq_r` | `layer10.block2.wq_rescale_sf` | `ctct_wq_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 754 | `L10.B2.R.wv_r` | `layer10.block2.wv_rescale_sf` | `ctct_wv_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 755 | `L10.B2.R.kt_mask1_r` | `layer10.block2.kt_mask1_rescale_sf` | `ctct_kt_mask1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 756 | `L10.B2.R.kt_mask2_r` | `layer10.block2.kt_mask2_rescale_sf` | `ctct_kt_mask2_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 757 | `L10.B2.R.q_mask1_r` | `layer10.block2.q_mask1_rescale_sf` | `ctct_q_mask1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 758 | `L10.B2.R.q_mask2_r` | `layer10.block2.q_mask2_rescale_sf` | `ctct_q_mask2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 759 | `L10.B2.R.qkt_matmul_r` | `layer10.block2.qkt_matmul_rescale_sf` | `ctct_qkt_matmul_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 760 | `L10.B2.R.qkt_merge_mask_r` | `layer10.block2.qkt_merge_mask_rescale_sf` | `ctct_qkt_merge_mask_rescale` | `rescale` | 3 | `scaling_factor` | 28 | True | 16384 | 28 |  |
| 761 | `L10.B2.K` | `layer10.block2.output_truncation_k` | `block2_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 762 | `L10.B3.F.x_fresh` | `layer10.block3.x_fresh_sf` | `ctpt_softmax_x` | `fresh` | 4 | `scaling_factor` | 28 | True | 16384 | 28 |  |
| 763 | `L10.B3.S.inv_2n` | `layer10.block3.inv_2n_sf` | `ctpt_softmax_inv_2n` | `scalar` | 2 | `scaling_factor` | 15 | True | 16384 | 15 |  |
| 764 | `L10.B3.R.x_inv_2n_r` | `layer10.block3.x_inv_2n_rescale_sf` | `ctct_softmax_x_inv_2n_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 765 | `L10.B3.R.sq0` | `layer10.block3.square_rescale_sf_0` | `ctct_softmax_pow_s1` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 766 | `L10.B3.R.sq1` | `layer10.block3.square_rescale_sf_1` | `ctct_softmax_pow_s2` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 767 | `L10.B3.R.sq2` | `layer10.block3.square_rescale_sf_2` | `ctct_softmax_pow_s3` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 768 | `L10.B3.R.sq3` | `layer10.block3.square_rescale_sf_3` | `ctct_softmax_pow_s4` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 769 | `L10.B3.K` | `layer10.block3.output_truncation_k` | `block3_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 770 | `L10.B4.F.softmax_out_fresh` | `layer10.block4.softmax_out_fresh_sf` | `ctpt_softmax_out` | `fresh` | 4 | `scaling_factor` | 35 | True | 16384 | 35 |  |
| 771 | `L10.B4.F.v_fresh` | `layer10.block4.v_fresh_sf` | `ctpt_v` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 772 | `L10.B4.M.softmax_out_mask` | `layer10.block4.softmax_out_mask_sf` | `ctpt_softmax_out_mask` | `encoding` | 1 | `scaling_factor` | 12 | True | 16384 | 14 |  |
| 773 | `L10.B4.M.v_mask` | `layer10.block4.v_mask_sf` | `ctpt_v_mask` | `encoding` | 1 | `scaling_factor` | 20 | True | 16384 | 22 |  |
| 774 | `L10.B4.M.softmax_v_mask` | `layer10.block4.softmax_v_mask_sf` | `ctpt_softmax_v_mask` | `encoding` | 1 | `scaling_factor` | 12 | True | 16384 | 14 |  |
| 775 | `L10.B4.S.ln_mean_inv_d` | `layer10.block4.ln_mean_inv_d_sf` | `ctpt_inv_d_attn_mean` | `scalar` | 2 | `scaling_factor` | 20 | True | 16384 | 20 |  |
| 776 | `L10.B4.S.ln_var_inv_d` | `layer10.block4.ln_var_inv_d_sf` | `ctpt_inv_d_attn_var` | `scalar` | 2 | `scaling_factor` | 20 | True | 16384 | 20 |  |
| 777 | `L10.B4.W.wo` | `layer10.block4.wo_sf` | `ctpt_wo` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 778 | `L10.B4.R.softmax_out_mask_r` | `layer10.block4.softmax_out_mask_rescale_sf` | `ctct_softmax_out_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 779 | `L10.B4.R.v_mask_r` | `layer10.block4.v_mask_rescale_sf` | `ctct_v_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 780 | `L10.B4.R.softmax_v_matmul_r` | `layer10.block4.softmax_v_matmul_rescale_sf` | `ctct_softmax_v_matmul_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 781 | `L10.B4.R.softmax_v_mask_r` | `layer10.block4.softmax_v_mask_rescale_sf` | `ctct_softmax_v_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 782 | `L10.B4.R.wo_r` | `layer10.block4.wo_rescale_sf` | `ctct_wo_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 783 | `L10.B4.R.ln_mean_r` | `layer10.block4.ln_mean_rescale_sf` | `ctct_attn_mean_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 784 | `L10.B4.R.ln_square_r` | `layer10.block4.ln_square_rescale_sf` | `ctct_attn_square_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 785 | `L10.B4.R.ln_var_r` | `layer10.block4.ln_var_rescale_sf` | `ctct_attn_var_rescale` | `rescale` | 3 | `scaling_factor` | 28 | True | 16384 | 28 |  |
| 786 | `L10.B4.K` | `layer10.block4.output_truncation_k` | `block4_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 787 | `L10.B5.F.inv_std_fresh` | `layer10.block5.inv_std_fresh_sf` | `ctpt_attn_inv_std` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 788 | `L10.B5.F.x_centered_fresh` | `layer10.block5.x_centered_fresh_sf` | `ctpt_attn_x_centered` | `fresh` | 4 | `scaling_factor` | 31 | True | 8192 | 31 |  |
| 789 | `L10.B5.M.gamma` | `layer10.block5.gamma_sf` | `ctpt_gamma_attn` | `encoding` | 1 | `scaling_factor` | 18 | True | 8192 | 20 |  |
| 790 | `L10.B5.W.wffn1` | `layer10.block5.wffn1_sf` | `ctpt_wffn1` | `encoding` | 4 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 791 | `L10.B5.M.gelu_coeff` | `layer10.block5.gelu_coeff_sf` | `ctpt_gelu_coeff` | `encoding` | 1 | `scaling_factor` | 18 | True | 8192 | 20 |  |
| 792 | `L10.B5.R.normalize_r` | `layer10.block5.normalize_rescale_sf` | `ctct_attn_normalize_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 8192 | 31 |  |
| 793 | `L10.B5.R.gamma_r` | `layer10.block5.gamma_rescale_sf` | `ctct_gamma_attn_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 794 | `L10.B5.R.wffn1_r` | `layer10.block5.wffn1_rescale_sf` | `ctct_wffn1_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 8192 | 31 |  |
| 795 | `L10.B5.R.gp0` | `layer10.block5.gelu_power_rescale_sf_0` | `ctct_gelu_x2` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 31 | GELU degree 1 does not use this power-rescale slot |
| 796 | `L10.B5.R.gp1` | `layer10.block5.gelu_power_rescale_sf_1` | `ctct_gelu_x3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 797 | `L10.B5.R.gp2` | `layer10.block5.gelu_power_rescale_sf_2` | `ctct_gelu_x4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 798 | `L10.B5.R.gc0` | `layer10.block5.gelu_coeff_mul_rescale_sf_0` | `ctct_gelu_b_x` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 799 | `L10.B5.R.gc1` | `layer10.block5.gelu_coeff_mul_rescale_sf_1` | `ctct_gelu_c_x2` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 800 | `L10.B5.R.gc2` | `layer10.block5.gelu_coeff_mul_rescale_sf_2` | `ctct_gelu_d_x3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 801 | `L10.B5.R.gc3` | `layer10.block5.gelu_coeff_mul_rescale_sf_3` | `ctct_gelu_e_x4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 31 | GELU degree 1 does not use this coefficient-rescale slot |
| 802 | `L10.B5.K` | `layer10.block5.output_truncation_k` | `block5_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 803 | `L11.B1.F.gelu_out` | `layer11.block1.gelu_out_sf` | `ctpt_gelu_out` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 804 | `L11.B1.W.wffn2` | `layer11.block1.wffn2_sf` | `ctpt_ffn2` | `encoding` | 4 | `scaling_factor` | 20 | True | 8192 | 20 |  |
| 805 | `L11.B1.S.mean_inv_d` | `layer11.block1.mean_inv_d_sf` | `ctpt_inv_d_1` | `scalar` | 2 | `scaling_factor` | 20 | True | 8192 | 20 |  |
| 806 | `L11.B1.S.var_inv_d` | `layer11.block1.var_inv_d_sf` | `ctpt_inv_d_2` | `scalar` | 2 | `scaling_factor` | 20 | True | 8192 | 20 |  |
| 807 | `L11.B1.R.wffn2_r` | `layer11.block1.wffn2_rescale_sf` | `ctct_ffn2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 808 | `L11.B1.R.mean_r` | `layer11.block1.mean_rescale_sf` | `ctct_mean_rescale` | `rescale` | 3 | `scaling_factor` | 34 | True | 8192 | 34 |  |
| 809 | `L11.B1.R.square_r` | `layer11.block1.square_rescale_sf` | `ctct_ext_square` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 810 | `L11.B1.R.var_r` | `layer11.block1.var_rescale_sf` | `ctct_var_rescale` | `rescale` | 3 | `scaling_factor` | 34 | True | 8192 | 34 |  |
| 811 | `L11.B1.K` | `layer11.block1.output_truncation_k` | `block1_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 812 | `L11.B2.F.inv_std_fresh` | `layer11.block2.inv_std_fresh_sf` | `ctpt_inv_std` | `fresh` | 4 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 813 | `L11.B2.F.x_centered_fresh` | `layer11.block2.x_centered_fresh_sf` | `ctpt_x_centered` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 814 | `L11.B2.M.gamma` | `layer11.block2.gamma_sf` | `ctpt_gamma` | `encoding` | 1 | `scaling_factor` | 18 | True | 16384 | 20 |  |
| 815 | `L11.B2.W.wq` | `layer11.block2.wq_sf` | `ctpt_wq` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 816 | `L11.B2.W.wk` | `layer11.block2.wk_sf` | `ctpt_wk` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 817 | `L11.B2.W.wv` | `layer11.block2.wv_sf` | `ctpt_wv` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 818 | `L11.B2.M.kt_mask1` | `layer11.block2.kt_mask1_sf` | `ctpt_kt_mask1` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 819 | `L11.B2.M.kt_mask2` | `layer11.block2.kt_mask2_sf` | `ctpt_kt_mask2` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 820 | `L11.B2.M.q_mask1` | `layer11.block2.q_mask1_sf` | `ctpt_q_mask1` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 821 | `L11.B2.M.q_mask2` | `layer11.block2.q_mask2_sf` | `ctpt_q_mask2` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 822 | `L11.B2.M.qkt_merge_mask` | `layer11.block2.qkt_merge_mask_sf` | `ctpt_qkt_merge_mask` | `encoding` | 1 | `scaling_factor` | 13 | True | 16384 | 15 |  |
| 823 | `L11.B2.R.normalize_r` | `layer11.block2.normalize_rescale_sf` | `ctct_normalize_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 824 | `L11.B2.R.gamma_r` | `layer11.block2.gamma_rescale_sf` | `ctct_gamma_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 825 | `L11.B2.R.wk_r` | `layer11.block2.wk_rescale_sf` | `ctct_wk_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 826 | `L11.B2.R.wq_r` | `layer11.block2.wq_rescale_sf` | `ctct_wq_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 827 | `L11.B2.R.wv_r` | `layer11.block2.wv_rescale_sf` | `ctct_wv_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 828 | `L11.B2.R.kt_mask1_r` | `layer11.block2.kt_mask1_rescale_sf` | `ctct_kt_mask1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 829 | `L11.B2.R.kt_mask2_r` | `layer11.block2.kt_mask2_rescale_sf` | `ctct_kt_mask2_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 830 | `L11.B2.R.q_mask1_r` | `layer11.block2.q_mask1_rescale_sf` | `ctct_q_mask1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 831 | `L11.B2.R.q_mask2_r` | `layer11.block2.q_mask2_rescale_sf` | `ctct_q_mask2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 832 | `L11.B2.R.qkt_matmul_r` | `layer11.block2.qkt_matmul_rescale_sf` | `ctct_qkt_matmul_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 833 | `L11.B2.R.qkt_merge_mask_r` | `layer11.block2.qkt_merge_mask_rescale_sf` | `ctct_qkt_merge_mask_rescale` | `rescale` | 3 | `scaling_factor` | 28 | True | 16384 | 28 |  |
| 834 | `L11.B2.K` | `layer11.block2.output_truncation_k` | `block2_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 835 | `L11.B3.F.x_fresh` | `layer11.block3.x_fresh_sf` | `ctpt_softmax_x` | `fresh` | 4 | `scaling_factor` | 28 | True | 8192 | 28 |  |
| 836 | `L11.B3.S.inv_2n` | `layer11.block3.inv_2n_sf` | `ctpt_softmax_inv_2n` | `scalar` | 2 | `scaling_factor` | 15 | True | 8192 | 15 |  |
| 837 | `L11.B3.R.x_inv_2n_r` | `layer11.block3.x_inv_2n_rescale_sf` | `ctct_softmax_x_inv_2n_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 838 | `L11.B3.R.sq0` | `layer11.block3.square_rescale_sf_0` | `ctct_softmax_pow_s1` | `rescale` | 3 | `scaling_factor` | 31 | True | 8192 | 31 |  |
| 839 | `L11.B3.R.sq1` | `layer11.block3.square_rescale_sf_1` | `ctct_softmax_pow_s2` | `rescale` | 3 | `scaling_factor` | 31 | True | 8192 | 31 |  |
| 840 | `L11.B3.R.sq2` | `layer11.block3.square_rescale_sf_2` | `ctct_softmax_pow_s3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 31 | softmax degree 2 does not use this square-rescale slot |
| 841 | `L11.B3.R.sq3` | `layer11.block3.square_rescale_sf_3` | `ctct_softmax_pow_s4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 31 | softmax degree 2 does not use this square-rescale slot |
| 842 | `L11.B3.K` | `layer11.block3.output_truncation_k` | `block3_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 843 | `L11.B4.F.softmax_out_fresh` | `layer11.block4.softmax_out_fresh_sf` | `ctpt_softmax_out` | `fresh` | 4 | `scaling_factor` | 35 | True | 16384 | 35 |  |
| 844 | `L11.B4.F.v_fresh` | `layer11.block4.v_fresh_sf` | `ctpt_v` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 845 | `L11.B4.M.softmax_out_mask` | `layer11.block4.softmax_out_mask_sf` | `ctpt_softmax_out_mask` | `encoding` | 1 | `scaling_factor` | 12 | True | 16384 | 14 |  |
| 846 | `L11.B4.M.v_mask` | `layer11.block4.v_mask_sf` | `ctpt_v_mask` | `encoding` | 1 | `scaling_factor` | 20 | True | 16384 | 22 |  |
| 847 | `L11.B4.M.softmax_v_mask` | `layer11.block4.softmax_v_mask_sf` | `ctpt_softmax_v_mask` | `encoding` | 1 | `scaling_factor` | 12 | True | 16384 | 14 |  |
| 848 | `L11.B4.S.ln_mean_inv_d` | `layer11.block4.ln_mean_inv_d_sf` | `ctpt_inv_d_attn_mean` | `scalar` | 2 | `scaling_factor` | 20 | True | 16384 | 20 |  |
| 849 | `L11.B4.S.ln_var_inv_d` | `layer11.block4.ln_var_inv_d_sf` | `ctpt_inv_d_attn_var` | `scalar` | 2 | `scaling_factor` | 20 | True | 16384 | 20 |  |
| 850 | `L11.B4.W.wo` | `layer11.block4.wo_sf` | `ctpt_wo` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 851 | `L11.B4.R.softmax_out_mask_r` | `layer11.block4.softmax_out_mask_rescale_sf` | `ctct_softmax_out_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 852 | `L11.B4.R.v_mask_r` | `layer11.block4.v_mask_rescale_sf` | `ctct_v_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 853 | `L11.B4.R.softmax_v_matmul_r` | `layer11.block4.softmax_v_matmul_rescale_sf` | `ctct_softmax_v_matmul_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 854 | `L11.B4.R.softmax_v_mask_r` | `layer11.block4.softmax_v_mask_rescale_sf` | `ctct_softmax_v_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 855 | `L11.B4.R.wo_r` | `layer11.block4.wo_rescale_sf` | `ctct_wo_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 856 | `L11.B4.R.ln_mean_r` | `layer11.block4.ln_mean_rescale_sf` | `ctct_attn_mean_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 16384 | 31 |  |
| 857 | `L11.B4.R.ln_square_r` | `layer11.block4.ln_square_rescale_sf` | `ctct_attn_square_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 858 | `L11.B4.R.ln_var_r` | `layer11.block4.ln_var_rescale_sf` | `ctct_attn_var_rescale` | `rescale` | 3 | `scaling_factor` | 28 | True | 16384 | 28 |  |
| 859 | `L11.B4.K` | `layer11.block4.output_truncation_k` | `block4_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 860 | `L11.B5.F.inv_std_fresh` | `layer11.block5.inv_std_fresh_sf` | `ctpt_attn_inv_std` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 861 | `L11.B5.F.x_centered_fresh` | `layer11.block5.x_centered_fresh_sf` | `ctpt_attn_x_centered` | `fresh` | 4 | `scaling_factor` | 31 | True | 8192 | 31 |  |
| 862 | `L11.B5.M.gamma` | `layer11.block5.gamma_sf` | `ctpt_gamma_attn` | `encoding` | 1 | `scaling_factor` | 18 | True | 8192 | 20 |  |
| 863 | `L11.B5.W.wffn1` | `layer11.block5.wffn1_sf` | `ctpt_wffn1` | `encoding` | 4 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 864 | `L11.B5.M.gelu_coeff` | `layer11.block5.gelu_coeff_sf` | `ctpt_gelu_coeff` | `encoding` | 1 | `scaling_factor` | 18 | True | 8192 | 20 |  |
| 865 | `L11.B5.R.normalize_r` | `layer11.block5.normalize_rescale_sf` | `ctct_attn_normalize_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 8192 | 31 |  |
| 866 | `L11.B5.R.gamma_r` | `layer11.block5.gamma_rescale_sf` | `ctct_gamma_attn_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 867 | `L11.B5.R.wffn1_r` | `layer11.block5.wffn1_rescale_sf` | `ctct_wffn1_rescale` | `rescale` | 3 | `scaling_factor` | 31 | True | 8192 | 31 |  |
| 868 | `L11.B5.R.gp0` | `layer11.block5.gelu_power_rescale_sf_0` | `ctct_gelu_x2` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 31 | GELU degree 1 does not use this power-rescale slot |
| 869 | `L11.B5.R.gp1` | `layer11.block5.gelu_power_rescale_sf_1` | `ctct_gelu_x3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 870 | `L11.B5.R.gp2` | `layer11.block5.gelu_power_rescale_sf_2` | `ctct_gelu_x4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 871 | `L11.B5.R.gc0` | `layer11.block5.gelu_coeff_mul_rescale_sf_0` | `ctct_gelu_b_x` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 872 | `L11.B5.R.gc1` | `layer11.block5.gelu_coeff_mul_rescale_sf_1` | `ctct_gelu_c_x2` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 873 | `L11.B5.R.gc2` | `layer11.block5.gelu_coeff_mul_rescale_sf_2` | `ctct_gelu_d_x3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 874 | `L11.B5.R.gc3` | `layer11.block5.gelu_coeff_mul_rescale_sf_3` | `ctct_gelu_e_x4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 31 | GELU degree 1 does not use this coefficient-rescale slot |
| 875 | `L11.B5.K` | `layer11.block5.output_truncation_k` | `block5_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 876 | `L0.first_input.F` | `first_input.layer0` | `first_input_fresh` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |