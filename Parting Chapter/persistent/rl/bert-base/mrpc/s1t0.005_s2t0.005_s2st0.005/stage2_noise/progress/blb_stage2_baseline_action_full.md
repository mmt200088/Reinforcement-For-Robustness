# BLB Stage 2 action description: baseline

- profile: `mrpc`
- num_layers: `12`
- action_length: `877`
- records: `877`
- scaling factor slots: `817`
- truncation slots: `60`
- ineffective decoded slots: `77`

| idx | location | operation | kind | action_idx | value_type | value | effective | N | max_sf | note |
|---:|---|---|---|---:|---|---:|---|---:|---:|---|
| 0 | `layer0.block1.gelu_out_sf` | `ctpt_gelu_out` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 1 | `layer0.block1.wffn2_sf` | `ctpt_ffn2` | `encoding` | 4 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 2 | `layer0.block1.mean_inv_d_sf` | `ctpt_inv_d_1` | `scalar` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 3 | `layer0.block1.var_inv_d_sf` | `ctpt_inv_d_2` | `scalar` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 4 | `layer0.block1.wffn2_rescale_sf` | `ctct_ffn2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 5 | `layer0.block1.mean_rescale_sf` | `ctct_mean_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 6 | `layer0.block1.square_rescale_sf` | `ctct_ext_square` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 7 | `layer0.block1.var_rescale_sf` | `ctct_var_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 8 | `layer0.block1.output_truncation_k` | `block1_output_truncation` | `truncation` | 3 | `truncation_k` |  | False | 8192 |  | layer0.block1 has no input-side truncation point; decoded cfg uses None |
| 9 | `layer0.block2.inv_std_fresh_sf` | `ctpt_inv_std` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 10 | `layer0.block2.x_centered_fresh_sf` | `ctpt_x_centered` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 11 | `layer0.block2.gamma_sf` | `ctpt_gamma` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 12 | `layer0.block2.wq_sf` | `ctpt_wq` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 13 | `layer0.block2.wk_sf` | `ctpt_wk` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 14 | `layer0.block2.wv_sf` | `ctpt_wv` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 15 | `layer0.block2.kt_mask1_sf` | `ctpt_kt_mask1` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 16 | `layer0.block2.kt_mask2_sf` | `ctpt_kt_mask2` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 17 | `layer0.block2.q_mask1_sf` | `ctpt_q_mask1` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 18 | `layer0.block2.q_mask2_sf` | `ctpt_q_mask2` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 19 | `layer0.block2.qkt_merge_mask_sf` | `ctpt_qkt_merge_mask` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 20 | `layer0.block2.normalize_rescale_sf` | `ctct_normalize_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 21 | `layer0.block2.gamma_rescale_sf` | `ctct_gamma_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 22 | `layer0.block2.wk_rescale_sf` | `ctct_wk_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 23 | `layer0.block2.wq_rescale_sf` | `ctct_wq_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 24 | `layer0.block2.wv_rescale_sf` | `ctct_wv_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 25 | `layer0.block2.kt_mask1_rescale_sf` | `ctct_kt_mask1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 26 | `layer0.block2.kt_mask2_rescale_sf` | `ctct_kt_mask2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 27 | `layer0.block2.q_mask1_rescale_sf` | `ctct_q_mask1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 28 | `layer0.block2.q_mask2_rescale_sf` | `ctct_q_mask2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 29 | `layer0.block2.qkt_matmul_rescale_sf` | `ctct_qkt_matmul_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 30 | `layer0.block2.qkt_merge_mask_rescale_sf` | `ctct_qkt_merge_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 31 | `layer0.block2.output_truncation_k` | `block2_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 32 | `layer0.block3.x_fresh_sf` | `ctpt_softmax_x` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 33 | `layer0.block3.inv_2n_sf` | `ctpt_softmax_inv_2n` | `scalar` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 34 | `layer0.block3.x_inv_2n_rescale_sf` | `ctct_softmax_x_inv_2n_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 35 | `layer0.block3.square_rescale_sf_0` | `ctct_softmax_pow_s1` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 36 | `layer0.block3.square_rescale_sf_1` | `ctct_softmax_pow_s2` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 37 | `layer0.block3.square_rescale_sf_2` | `ctct_softmax_pow_s3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | softmax degree 2 does not use this square-rescale slot |
| 38 | `layer0.block3.square_rescale_sf_3` | `ctct_softmax_pow_s4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | softmax degree 2 does not use this square-rescale slot |
| 39 | `layer0.block3.output_truncation_k` | `block3_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 40 | `layer0.block4.softmax_out_fresh_sf` | `ctpt_softmax_out` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 41 | `layer0.block4.v_fresh_sf` | `ctpt_v` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 42 | `layer0.block4.softmax_out_mask_sf` | `ctpt_softmax_out_mask` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 43 | `layer0.block4.v_mask_sf` | `ctpt_v_mask` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 44 | `layer0.block4.softmax_v_mask_sf` | `ctpt_softmax_v_mask` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 45 | `layer0.block4.ln_mean_inv_d_sf` | `ctpt_inv_d_attn_mean` | `scalar` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 46 | `layer0.block4.ln_var_inv_d_sf` | `ctpt_inv_d_attn_var` | `scalar` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 47 | `layer0.block4.wo_sf` | `ctpt_wo` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 48 | `layer0.block4.softmax_out_mask_rescale_sf` | `ctct_softmax_out_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 49 | `layer0.block4.v_mask_rescale_sf` | `ctct_v_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 50 | `layer0.block4.softmax_v_matmul_rescale_sf` | `ctct_softmax_v_matmul_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 51 | `layer0.block4.softmax_v_mask_rescale_sf` | `ctct_softmax_v_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 52 | `layer0.block4.wo_rescale_sf` | `ctct_wo_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 53 | `layer0.block4.ln_mean_rescale_sf` | `ctct_attn_mean_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 54 | `layer0.block4.ln_square_rescale_sf` | `ctct_attn_square_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 55 | `layer0.block4.ln_var_rescale_sf` | `ctct_attn_var_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 56 | `layer0.block4.output_truncation_k` | `block4_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 57 | `layer0.block5.inv_std_fresh_sf` | `ctpt_attn_inv_std` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 58 | `layer0.block5.x_centered_fresh_sf` | `ctpt_attn_x_centered` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 59 | `layer0.block5.gamma_sf` | `ctpt_gamma_attn` | `encoding` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 60 | `layer0.block5.wffn1_sf` | `ctpt_wffn1` | `encoding` | 4 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 61 | `layer0.block5.gelu_coeff_sf` | `ctpt_gelu_coeff` | `encoding` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 62 | `layer0.block5.normalize_rescale_sf` | `ctct_attn_normalize_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 63 | `layer0.block5.gamma_rescale_sf` | `ctct_gamma_attn_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 64 | `layer0.block5.wffn1_rescale_sf` | `ctct_wffn1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 65 | `layer0.block5.gelu_power_rescale_sf_0` | `ctct_gelu_x2` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 66 | `layer0.block5.gelu_power_rescale_sf_1` | `ctct_gelu_x3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 67 | `layer0.block5.gelu_power_rescale_sf_2` | `ctct_gelu_x4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 68 | `layer0.block5.gelu_coeff_mul_rescale_sf_0` | `ctct_gelu_b_x` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 69 | `layer0.block5.gelu_coeff_mul_rescale_sf_1` | `ctct_gelu_c_x2` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 70 | `layer0.block5.gelu_coeff_mul_rescale_sf_2` | `ctct_gelu_d_x3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 71 | `layer0.block5.gelu_coeff_mul_rescale_sf_3` | `ctct_gelu_e_x4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 72 | `layer0.block5.output_truncation_k` | `block5_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 73 | `layer1.block1.gelu_out_sf` | `ctpt_gelu_out` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 74 | `layer1.block1.wffn2_sf` | `ctpt_ffn2` | `encoding` | 4 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 75 | `layer1.block1.mean_inv_d_sf` | `ctpt_inv_d_1` | `scalar` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 76 | `layer1.block1.var_inv_d_sf` | `ctpt_inv_d_2` | `scalar` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 77 | `layer1.block1.wffn2_rescale_sf` | `ctct_ffn2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 78 | `layer1.block1.mean_rescale_sf` | `ctct_mean_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 79 | `layer1.block1.square_rescale_sf` | `ctct_ext_square` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 80 | `layer1.block1.var_rescale_sf` | `ctct_var_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 81 | `layer1.block1.output_truncation_k` | `block1_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 82 | `layer1.block2.inv_std_fresh_sf` | `ctpt_inv_std` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 83 | `layer1.block2.x_centered_fresh_sf` | `ctpt_x_centered` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 84 | `layer1.block2.gamma_sf` | `ctpt_gamma` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 85 | `layer1.block2.wq_sf` | `ctpt_wq` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 86 | `layer1.block2.wk_sf` | `ctpt_wk` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 87 | `layer1.block2.wv_sf` | `ctpt_wv` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 88 | `layer1.block2.kt_mask1_sf` | `ctpt_kt_mask1` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 89 | `layer1.block2.kt_mask2_sf` | `ctpt_kt_mask2` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 90 | `layer1.block2.q_mask1_sf` | `ctpt_q_mask1` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 91 | `layer1.block2.q_mask2_sf` | `ctpt_q_mask2` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 92 | `layer1.block2.qkt_merge_mask_sf` | `ctpt_qkt_merge_mask` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 93 | `layer1.block2.normalize_rescale_sf` | `ctct_normalize_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 94 | `layer1.block2.gamma_rescale_sf` | `ctct_gamma_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 95 | `layer1.block2.wk_rescale_sf` | `ctct_wk_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 96 | `layer1.block2.wq_rescale_sf` | `ctct_wq_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 97 | `layer1.block2.wv_rescale_sf` | `ctct_wv_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 98 | `layer1.block2.kt_mask1_rescale_sf` | `ctct_kt_mask1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 99 | `layer1.block2.kt_mask2_rescale_sf` | `ctct_kt_mask2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 100 | `layer1.block2.q_mask1_rescale_sf` | `ctct_q_mask1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 101 | `layer1.block2.q_mask2_rescale_sf` | `ctct_q_mask2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 102 | `layer1.block2.qkt_matmul_rescale_sf` | `ctct_qkt_matmul_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 103 | `layer1.block2.qkt_merge_mask_rescale_sf` | `ctct_qkt_merge_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 104 | `layer1.block2.output_truncation_k` | `block2_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 105 | `layer1.block3.x_fresh_sf` | `ctpt_softmax_x` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 106 | `layer1.block3.inv_2n_sf` | `ctpt_softmax_inv_2n` | `scalar` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 107 | `layer1.block3.x_inv_2n_rescale_sf` | `ctct_softmax_x_inv_2n_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 108 | `layer1.block3.square_rescale_sf_0` | `ctct_softmax_pow_s1` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 109 | `layer1.block3.square_rescale_sf_1` | `ctct_softmax_pow_s2` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 110 | `layer1.block3.square_rescale_sf_2` | `ctct_softmax_pow_s3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | softmax degree 2 does not use this square-rescale slot |
| 111 | `layer1.block3.square_rescale_sf_3` | `ctct_softmax_pow_s4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | softmax degree 2 does not use this square-rescale slot |
| 112 | `layer1.block3.output_truncation_k` | `block3_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 113 | `layer1.block4.softmax_out_fresh_sf` | `ctpt_softmax_out` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 114 | `layer1.block4.v_fresh_sf` | `ctpt_v` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 115 | `layer1.block4.softmax_out_mask_sf` | `ctpt_softmax_out_mask` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 116 | `layer1.block4.v_mask_sf` | `ctpt_v_mask` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 117 | `layer1.block4.softmax_v_mask_sf` | `ctpt_softmax_v_mask` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 118 | `layer1.block4.ln_mean_inv_d_sf` | `ctpt_inv_d_attn_mean` | `scalar` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 119 | `layer1.block4.ln_var_inv_d_sf` | `ctpt_inv_d_attn_var` | `scalar` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 120 | `layer1.block4.wo_sf` | `ctpt_wo` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 121 | `layer1.block4.softmax_out_mask_rescale_sf` | `ctct_softmax_out_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 122 | `layer1.block4.v_mask_rescale_sf` | `ctct_v_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 123 | `layer1.block4.softmax_v_matmul_rescale_sf` | `ctct_softmax_v_matmul_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 124 | `layer1.block4.softmax_v_mask_rescale_sf` | `ctct_softmax_v_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 125 | `layer1.block4.wo_rescale_sf` | `ctct_wo_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 126 | `layer1.block4.ln_mean_rescale_sf` | `ctct_attn_mean_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 127 | `layer1.block4.ln_square_rescale_sf` | `ctct_attn_square_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 128 | `layer1.block4.ln_var_rescale_sf` | `ctct_attn_var_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 129 | `layer1.block4.output_truncation_k` | `block4_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 130 | `layer1.block5.inv_std_fresh_sf` | `ctpt_attn_inv_std` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 131 | `layer1.block5.x_centered_fresh_sf` | `ctpt_attn_x_centered` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 132 | `layer1.block5.gamma_sf` | `ctpt_gamma_attn` | `encoding` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 133 | `layer1.block5.wffn1_sf` | `ctpt_wffn1` | `encoding` | 4 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 134 | `layer1.block5.gelu_coeff_sf` | `ctpt_gelu_coeff` | `encoding` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 135 | `layer1.block5.normalize_rescale_sf` | `ctct_attn_normalize_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 136 | `layer1.block5.gamma_rescale_sf` | `ctct_gamma_attn_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 137 | `layer1.block5.wffn1_rescale_sf` | `ctct_wffn1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 138 | `layer1.block5.gelu_power_rescale_sf_0` | `ctct_gelu_x2` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 139 | `layer1.block5.gelu_power_rescale_sf_1` | `ctct_gelu_x3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 140 | `layer1.block5.gelu_power_rescale_sf_2` | `ctct_gelu_x4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 141 | `layer1.block5.gelu_coeff_mul_rescale_sf_0` | `ctct_gelu_b_x` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 142 | `layer1.block5.gelu_coeff_mul_rescale_sf_1` | `ctct_gelu_c_x2` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 143 | `layer1.block5.gelu_coeff_mul_rescale_sf_2` | `ctct_gelu_d_x3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 144 | `layer1.block5.gelu_coeff_mul_rescale_sf_3` | `ctct_gelu_e_x4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 145 | `layer1.block5.output_truncation_k` | `block5_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 146 | `layer2.block1.gelu_out_sf` | `ctpt_gelu_out` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 147 | `layer2.block1.wffn2_sf` | `ctpt_ffn2` | `encoding` | 4 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 148 | `layer2.block1.mean_inv_d_sf` | `ctpt_inv_d_1` | `scalar` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 149 | `layer2.block1.var_inv_d_sf` | `ctpt_inv_d_2` | `scalar` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 150 | `layer2.block1.wffn2_rescale_sf` | `ctct_ffn2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 151 | `layer2.block1.mean_rescale_sf` | `ctct_mean_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 152 | `layer2.block1.square_rescale_sf` | `ctct_ext_square` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 153 | `layer2.block1.var_rescale_sf` | `ctct_var_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 154 | `layer2.block1.output_truncation_k` | `block1_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 155 | `layer2.block2.inv_std_fresh_sf` | `ctpt_inv_std` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 156 | `layer2.block2.x_centered_fresh_sf` | `ctpt_x_centered` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 157 | `layer2.block2.gamma_sf` | `ctpt_gamma` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 158 | `layer2.block2.wq_sf` | `ctpt_wq` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 159 | `layer2.block2.wk_sf` | `ctpt_wk` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 160 | `layer2.block2.wv_sf` | `ctpt_wv` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 161 | `layer2.block2.kt_mask1_sf` | `ctpt_kt_mask1` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 162 | `layer2.block2.kt_mask2_sf` | `ctpt_kt_mask2` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 163 | `layer2.block2.q_mask1_sf` | `ctpt_q_mask1` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 164 | `layer2.block2.q_mask2_sf` | `ctpt_q_mask2` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 165 | `layer2.block2.qkt_merge_mask_sf` | `ctpt_qkt_merge_mask` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 166 | `layer2.block2.normalize_rescale_sf` | `ctct_normalize_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 167 | `layer2.block2.gamma_rescale_sf` | `ctct_gamma_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 168 | `layer2.block2.wk_rescale_sf` | `ctct_wk_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 169 | `layer2.block2.wq_rescale_sf` | `ctct_wq_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 170 | `layer2.block2.wv_rescale_sf` | `ctct_wv_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 171 | `layer2.block2.kt_mask1_rescale_sf` | `ctct_kt_mask1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 172 | `layer2.block2.kt_mask2_rescale_sf` | `ctct_kt_mask2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 173 | `layer2.block2.q_mask1_rescale_sf` | `ctct_q_mask1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 174 | `layer2.block2.q_mask2_rescale_sf` | `ctct_q_mask2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 175 | `layer2.block2.qkt_matmul_rescale_sf` | `ctct_qkt_matmul_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 176 | `layer2.block2.qkt_merge_mask_rescale_sf` | `ctct_qkt_merge_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 177 | `layer2.block2.output_truncation_k` | `block2_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 178 | `layer2.block3.x_fresh_sf` | `ctpt_softmax_x` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 179 | `layer2.block3.inv_2n_sf` | `ctpt_softmax_inv_2n` | `scalar` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 180 | `layer2.block3.x_inv_2n_rescale_sf` | `ctct_softmax_x_inv_2n_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 181 | `layer2.block3.square_rescale_sf_0` | `ctct_softmax_pow_s1` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 182 | `layer2.block3.square_rescale_sf_1` | `ctct_softmax_pow_s2` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 183 | `layer2.block3.square_rescale_sf_2` | `ctct_softmax_pow_s3` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 184 | `layer2.block3.square_rescale_sf_3` | `ctct_softmax_pow_s4` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 185 | `layer2.block3.output_truncation_k` | `block3_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 186 | `layer2.block4.softmax_out_fresh_sf` | `ctpt_softmax_out` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 187 | `layer2.block4.v_fresh_sf` | `ctpt_v` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 188 | `layer2.block4.softmax_out_mask_sf` | `ctpt_softmax_out_mask` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 189 | `layer2.block4.v_mask_sf` | `ctpt_v_mask` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 190 | `layer2.block4.softmax_v_mask_sf` | `ctpt_softmax_v_mask` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 191 | `layer2.block4.ln_mean_inv_d_sf` | `ctpt_inv_d_attn_mean` | `scalar` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 192 | `layer2.block4.ln_var_inv_d_sf` | `ctpt_inv_d_attn_var` | `scalar` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 193 | `layer2.block4.wo_sf` | `ctpt_wo` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 194 | `layer2.block4.softmax_out_mask_rescale_sf` | `ctct_softmax_out_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 195 | `layer2.block4.v_mask_rescale_sf` | `ctct_v_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 196 | `layer2.block4.softmax_v_matmul_rescale_sf` | `ctct_softmax_v_matmul_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 197 | `layer2.block4.softmax_v_mask_rescale_sf` | `ctct_softmax_v_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 198 | `layer2.block4.wo_rescale_sf` | `ctct_wo_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 199 | `layer2.block4.ln_mean_rescale_sf` | `ctct_attn_mean_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 200 | `layer2.block4.ln_square_rescale_sf` | `ctct_attn_square_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 201 | `layer2.block4.ln_var_rescale_sf` | `ctct_attn_var_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 202 | `layer2.block4.output_truncation_k` | `block4_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 203 | `layer2.block5.inv_std_fresh_sf` | `ctpt_attn_inv_std` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 204 | `layer2.block5.x_centered_fresh_sf` | `ctpt_attn_x_centered` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 205 | `layer2.block5.gamma_sf` | `ctpt_gamma_attn` | `encoding` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 206 | `layer2.block5.wffn1_sf` | `ctpt_wffn1` | `encoding` | 4 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 207 | `layer2.block5.gelu_coeff_sf` | `ctpt_gelu_coeff` | `encoding` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 208 | `layer2.block5.normalize_rescale_sf` | `ctct_attn_normalize_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 209 | `layer2.block5.gamma_rescale_sf` | `ctct_gamma_attn_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 210 | `layer2.block5.wffn1_rescale_sf` | `ctct_wffn1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 211 | `layer2.block5.gelu_power_rescale_sf_0` | `ctct_gelu_x2` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 212 | `layer2.block5.gelu_power_rescale_sf_1` | `ctct_gelu_x3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 213 | `layer2.block5.gelu_power_rescale_sf_2` | `ctct_gelu_x4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 214 | `layer2.block5.gelu_coeff_mul_rescale_sf_0` | `ctct_gelu_b_x` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 215 | `layer2.block5.gelu_coeff_mul_rescale_sf_1` | `ctct_gelu_c_x2` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 216 | `layer2.block5.gelu_coeff_mul_rescale_sf_2` | `ctct_gelu_d_x3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 217 | `layer2.block5.gelu_coeff_mul_rescale_sf_3` | `ctct_gelu_e_x4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 218 | `layer2.block5.output_truncation_k` | `block5_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 219 | `layer3.block1.gelu_out_sf` | `ctpt_gelu_out` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 220 | `layer3.block1.wffn2_sf` | `ctpt_ffn2` | `encoding` | 4 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 221 | `layer3.block1.mean_inv_d_sf` | `ctpt_inv_d_1` | `scalar` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 222 | `layer3.block1.var_inv_d_sf` | `ctpt_inv_d_2` | `scalar` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 223 | `layer3.block1.wffn2_rescale_sf` | `ctct_ffn2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 224 | `layer3.block1.mean_rescale_sf` | `ctct_mean_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 225 | `layer3.block1.square_rescale_sf` | `ctct_ext_square` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 226 | `layer3.block1.var_rescale_sf` | `ctct_var_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 227 | `layer3.block1.output_truncation_k` | `block1_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 228 | `layer3.block2.inv_std_fresh_sf` | `ctpt_inv_std` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 229 | `layer3.block2.x_centered_fresh_sf` | `ctpt_x_centered` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 230 | `layer3.block2.gamma_sf` | `ctpt_gamma` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 231 | `layer3.block2.wq_sf` | `ctpt_wq` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 232 | `layer3.block2.wk_sf` | `ctpt_wk` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 233 | `layer3.block2.wv_sf` | `ctpt_wv` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 234 | `layer3.block2.kt_mask1_sf` | `ctpt_kt_mask1` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 235 | `layer3.block2.kt_mask2_sf` | `ctpt_kt_mask2` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 236 | `layer3.block2.q_mask1_sf` | `ctpt_q_mask1` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 237 | `layer3.block2.q_mask2_sf` | `ctpt_q_mask2` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 238 | `layer3.block2.qkt_merge_mask_sf` | `ctpt_qkt_merge_mask` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 239 | `layer3.block2.normalize_rescale_sf` | `ctct_normalize_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 240 | `layer3.block2.gamma_rescale_sf` | `ctct_gamma_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 241 | `layer3.block2.wk_rescale_sf` | `ctct_wk_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 242 | `layer3.block2.wq_rescale_sf` | `ctct_wq_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 243 | `layer3.block2.wv_rescale_sf` | `ctct_wv_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 244 | `layer3.block2.kt_mask1_rescale_sf` | `ctct_kt_mask1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 245 | `layer3.block2.kt_mask2_rescale_sf` | `ctct_kt_mask2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 246 | `layer3.block2.q_mask1_rescale_sf` | `ctct_q_mask1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 247 | `layer3.block2.q_mask2_rescale_sf` | `ctct_q_mask2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 248 | `layer3.block2.qkt_matmul_rescale_sf` | `ctct_qkt_matmul_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 249 | `layer3.block2.qkt_merge_mask_rescale_sf` | `ctct_qkt_merge_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 250 | `layer3.block2.output_truncation_k` | `block2_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 251 | `layer3.block3.x_fresh_sf` | `ctpt_softmax_x` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 252 | `layer3.block3.inv_2n_sf` | `ctpt_softmax_inv_2n` | `scalar` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 253 | `layer3.block3.x_inv_2n_rescale_sf` | `ctct_softmax_x_inv_2n_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 254 | `layer3.block3.square_rescale_sf_0` | `ctct_softmax_pow_s1` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 255 | `layer3.block3.square_rescale_sf_1` | `ctct_softmax_pow_s2` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 256 | `layer3.block3.square_rescale_sf_2` | `ctct_softmax_pow_s3` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 257 | `layer3.block3.square_rescale_sf_3` | `ctct_softmax_pow_s4` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 258 | `layer3.block3.output_truncation_k` | `block3_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 259 | `layer3.block4.softmax_out_fresh_sf` | `ctpt_softmax_out` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 260 | `layer3.block4.v_fresh_sf` | `ctpt_v` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 261 | `layer3.block4.softmax_out_mask_sf` | `ctpt_softmax_out_mask` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 262 | `layer3.block4.v_mask_sf` | `ctpt_v_mask` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 263 | `layer3.block4.softmax_v_mask_sf` | `ctpt_softmax_v_mask` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 264 | `layer3.block4.ln_mean_inv_d_sf` | `ctpt_inv_d_attn_mean` | `scalar` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 265 | `layer3.block4.ln_var_inv_d_sf` | `ctpt_inv_d_attn_var` | `scalar` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 266 | `layer3.block4.wo_sf` | `ctpt_wo` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 267 | `layer3.block4.softmax_out_mask_rescale_sf` | `ctct_softmax_out_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 268 | `layer3.block4.v_mask_rescale_sf` | `ctct_v_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 269 | `layer3.block4.softmax_v_matmul_rescale_sf` | `ctct_softmax_v_matmul_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 270 | `layer3.block4.softmax_v_mask_rescale_sf` | `ctct_softmax_v_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 271 | `layer3.block4.wo_rescale_sf` | `ctct_wo_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 272 | `layer3.block4.ln_mean_rescale_sf` | `ctct_attn_mean_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 273 | `layer3.block4.ln_square_rescale_sf` | `ctct_attn_square_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 274 | `layer3.block4.ln_var_rescale_sf` | `ctct_attn_var_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 275 | `layer3.block4.output_truncation_k` | `block4_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 276 | `layer3.block5.inv_std_fresh_sf` | `ctpt_attn_inv_std` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 277 | `layer3.block5.x_centered_fresh_sf` | `ctpt_attn_x_centered` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 278 | `layer3.block5.gamma_sf` | `ctpt_gamma_attn` | `encoding` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 279 | `layer3.block5.wffn1_sf` | `ctpt_wffn1` | `encoding` | 4 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 280 | `layer3.block5.gelu_coeff_sf` | `ctpt_gelu_coeff` | `encoding` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 281 | `layer3.block5.normalize_rescale_sf` | `ctct_attn_normalize_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 282 | `layer3.block5.gamma_rescale_sf` | `ctct_gamma_attn_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 283 | `layer3.block5.wffn1_rescale_sf` | `ctct_wffn1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 284 | `layer3.block5.gelu_power_rescale_sf_0` | `ctct_gelu_x2` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 285 | `layer3.block5.gelu_power_rescale_sf_1` | `ctct_gelu_x3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 286 | `layer3.block5.gelu_power_rescale_sf_2` | `ctct_gelu_x4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 287 | `layer3.block5.gelu_coeff_mul_rescale_sf_0` | `ctct_gelu_b_x` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 288 | `layer3.block5.gelu_coeff_mul_rescale_sf_1` | `ctct_gelu_c_x2` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 289 | `layer3.block5.gelu_coeff_mul_rescale_sf_2` | `ctct_gelu_d_x3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 290 | `layer3.block5.gelu_coeff_mul_rescale_sf_3` | `ctct_gelu_e_x4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 291 | `layer3.block5.output_truncation_k` | `block5_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 292 | `layer4.block1.gelu_out_sf` | `ctpt_gelu_out` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 293 | `layer4.block1.wffn2_sf` | `ctpt_ffn2` | `encoding` | 4 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 294 | `layer4.block1.mean_inv_d_sf` | `ctpt_inv_d_1` | `scalar` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 295 | `layer4.block1.var_inv_d_sf` | `ctpt_inv_d_2` | `scalar` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 296 | `layer4.block1.wffn2_rescale_sf` | `ctct_ffn2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 297 | `layer4.block1.mean_rescale_sf` | `ctct_mean_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 298 | `layer4.block1.square_rescale_sf` | `ctct_ext_square` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 299 | `layer4.block1.var_rescale_sf` | `ctct_var_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 300 | `layer4.block1.output_truncation_k` | `block1_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 301 | `layer4.block2.inv_std_fresh_sf` | `ctpt_inv_std` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 302 | `layer4.block2.x_centered_fresh_sf` | `ctpt_x_centered` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 303 | `layer4.block2.gamma_sf` | `ctpt_gamma` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 304 | `layer4.block2.wq_sf` | `ctpt_wq` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 305 | `layer4.block2.wk_sf` | `ctpt_wk` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 306 | `layer4.block2.wv_sf` | `ctpt_wv` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 307 | `layer4.block2.kt_mask1_sf` | `ctpt_kt_mask1` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 308 | `layer4.block2.kt_mask2_sf` | `ctpt_kt_mask2` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 309 | `layer4.block2.q_mask1_sf` | `ctpt_q_mask1` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 310 | `layer4.block2.q_mask2_sf` | `ctpt_q_mask2` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 311 | `layer4.block2.qkt_merge_mask_sf` | `ctpt_qkt_merge_mask` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 312 | `layer4.block2.normalize_rescale_sf` | `ctct_normalize_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 313 | `layer4.block2.gamma_rescale_sf` | `ctct_gamma_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 314 | `layer4.block2.wk_rescale_sf` | `ctct_wk_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 315 | `layer4.block2.wq_rescale_sf` | `ctct_wq_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 316 | `layer4.block2.wv_rescale_sf` | `ctct_wv_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 317 | `layer4.block2.kt_mask1_rescale_sf` | `ctct_kt_mask1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 318 | `layer4.block2.kt_mask2_rescale_sf` | `ctct_kt_mask2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 319 | `layer4.block2.q_mask1_rescale_sf` | `ctct_q_mask1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 320 | `layer4.block2.q_mask2_rescale_sf` | `ctct_q_mask2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 321 | `layer4.block2.qkt_matmul_rescale_sf` | `ctct_qkt_matmul_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 322 | `layer4.block2.qkt_merge_mask_rescale_sf` | `ctct_qkt_merge_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 323 | `layer4.block2.output_truncation_k` | `block2_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 324 | `layer4.block3.x_fresh_sf` | `ctpt_softmax_x` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 325 | `layer4.block3.inv_2n_sf` | `ctpt_softmax_inv_2n` | `scalar` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 326 | `layer4.block3.x_inv_2n_rescale_sf` | `ctct_softmax_x_inv_2n_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 327 | `layer4.block3.square_rescale_sf_0` | `ctct_softmax_pow_s1` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 328 | `layer4.block3.square_rescale_sf_1` | `ctct_softmax_pow_s2` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 329 | `layer4.block3.square_rescale_sf_2` | `ctct_softmax_pow_s3` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 330 | `layer4.block3.square_rescale_sf_3` | `ctct_softmax_pow_s4` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 331 | `layer4.block3.output_truncation_k` | `block3_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 332 | `layer4.block4.softmax_out_fresh_sf` | `ctpt_softmax_out` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 333 | `layer4.block4.v_fresh_sf` | `ctpt_v` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 334 | `layer4.block4.softmax_out_mask_sf` | `ctpt_softmax_out_mask` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 335 | `layer4.block4.v_mask_sf` | `ctpt_v_mask` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 336 | `layer4.block4.softmax_v_mask_sf` | `ctpt_softmax_v_mask` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 337 | `layer4.block4.ln_mean_inv_d_sf` | `ctpt_inv_d_attn_mean` | `scalar` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 338 | `layer4.block4.ln_var_inv_d_sf` | `ctpt_inv_d_attn_var` | `scalar` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 339 | `layer4.block4.wo_sf` | `ctpt_wo` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 340 | `layer4.block4.softmax_out_mask_rescale_sf` | `ctct_softmax_out_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 341 | `layer4.block4.v_mask_rescale_sf` | `ctct_v_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 342 | `layer4.block4.softmax_v_matmul_rescale_sf` | `ctct_softmax_v_matmul_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 343 | `layer4.block4.softmax_v_mask_rescale_sf` | `ctct_softmax_v_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 344 | `layer4.block4.wo_rescale_sf` | `ctct_wo_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 345 | `layer4.block4.ln_mean_rescale_sf` | `ctct_attn_mean_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 346 | `layer4.block4.ln_square_rescale_sf` | `ctct_attn_square_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 347 | `layer4.block4.ln_var_rescale_sf` | `ctct_attn_var_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 348 | `layer4.block4.output_truncation_k` | `block4_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 349 | `layer4.block5.inv_std_fresh_sf` | `ctpt_attn_inv_std` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 350 | `layer4.block5.x_centered_fresh_sf` | `ctpt_attn_x_centered` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 351 | `layer4.block5.gamma_sf` | `ctpt_gamma_attn` | `encoding` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 352 | `layer4.block5.wffn1_sf` | `ctpt_wffn1` | `encoding` | 4 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 353 | `layer4.block5.gelu_coeff_sf` | `ctpt_gelu_coeff` | `encoding` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 354 | `layer4.block5.normalize_rescale_sf` | `ctct_attn_normalize_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 355 | `layer4.block5.gamma_rescale_sf` | `ctct_gamma_attn_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 356 | `layer4.block5.wffn1_rescale_sf` | `ctct_wffn1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 357 | `layer4.block5.gelu_power_rescale_sf_0` | `ctct_gelu_x2` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 358 | `layer4.block5.gelu_power_rescale_sf_1` | `ctct_gelu_x3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 359 | `layer4.block5.gelu_power_rescale_sf_2` | `ctct_gelu_x4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 360 | `layer4.block5.gelu_coeff_mul_rescale_sf_0` | `ctct_gelu_b_x` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 361 | `layer4.block5.gelu_coeff_mul_rescale_sf_1` | `ctct_gelu_c_x2` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 362 | `layer4.block5.gelu_coeff_mul_rescale_sf_2` | `ctct_gelu_d_x3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 363 | `layer4.block5.gelu_coeff_mul_rescale_sf_3` | `ctct_gelu_e_x4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 364 | `layer4.block5.output_truncation_k` | `block5_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 365 | `layer5.block1.gelu_out_sf` | `ctpt_gelu_out` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 366 | `layer5.block1.wffn2_sf` | `ctpt_ffn2` | `encoding` | 4 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 367 | `layer5.block1.mean_inv_d_sf` | `ctpt_inv_d_1` | `scalar` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 368 | `layer5.block1.var_inv_d_sf` | `ctpt_inv_d_2` | `scalar` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 369 | `layer5.block1.wffn2_rescale_sf` | `ctct_ffn2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 370 | `layer5.block1.mean_rescale_sf` | `ctct_mean_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 371 | `layer5.block1.square_rescale_sf` | `ctct_ext_square` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 372 | `layer5.block1.var_rescale_sf` | `ctct_var_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 373 | `layer5.block1.output_truncation_k` | `block1_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 374 | `layer5.block2.inv_std_fresh_sf` | `ctpt_inv_std` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 375 | `layer5.block2.x_centered_fresh_sf` | `ctpt_x_centered` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 376 | `layer5.block2.gamma_sf` | `ctpt_gamma` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 377 | `layer5.block2.wq_sf` | `ctpt_wq` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 378 | `layer5.block2.wk_sf` | `ctpt_wk` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 379 | `layer5.block2.wv_sf` | `ctpt_wv` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 380 | `layer5.block2.kt_mask1_sf` | `ctpt_kt_mask1` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 381 | `layer5.block2.kt_mask2_sf` | `ctpt_kt_mask2` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 382 | `layer5.block2.q_mask1_sf` | `ctpt_q_mask1` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 383 | `layer5.block2.q_mask2_sf` | `ctpt_q_mask2` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 384 | `layer5.block2.qkt_merge_mask_sf` | `ctpt_qkt_merge_mask` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 385 | `layer5.block2.normalize_rescale_sf` | `ctct_normalize_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 386 | `layer5.block2.gamma_rescale_sf` | `ctct_gamma_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 387 | `layer5.block2.wk_rescale_sf` | `ctct_wk_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 388 | `layer5.block2.wq_rescale_sf` | `ctct_wq_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 389 | `layer5.block2.wv_rescale_sf` | `ctct_wv_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 390 | `layer5.block2.kt_mask1_rescale_sf` | `ctct_kt_mask1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 391 | `layer5.block2.kt_mask2_rescale_sf` | `ctct_kt_mask2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 392 | `layer5.block2.q_mask1_rescale_sf` | `ctct_q_mask1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 393 | `layer5.block2.q_mask2_rescale_sf` | `ctct_q_mask2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 394 | `layer5.block2.qkt_matmul_rescale_sf` | `ctct_qkt_matmul_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 395 | `layer5.block2.qkt_merge_mask_rescale_sf` | `ctct_qkt_merge_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 396 | `layer5.block2.output_truncation_k` | `block2_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 397 | `layer5.block3.x_fresh_sf` | `ctpt_softmax_x` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 398 | `layer5.block3.inv_2n_sf` | `ctpt_softmax_inv_2n` | `scalar` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 399 | `layer5.block3.x_inv_2n_rescale_sf` | `ctct_softmax_x_inv_2n_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 400 | `layer5.block3.square_rescale_sf_0` | `ctct_softmax_pow_s1` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 401 | `layer5.block3.square_rescale_sf_1` | `ctct_softmax_pow_s2` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 402 | `layer5.block3.square_rescale_sf_2` | `ctct_softmax_pow_s3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | softmax degree 2 does not use this square-rescale slot |
| 403 | `layer5.block3.square_rescale_sf_3` | `ctct_softmax_pow_s4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | softmax degree 2 does not use this square-rescale slot |
| 404 | `layer5.block3.output_truncation_k` | `block3_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 405 | `layer5.block4.softmax_out_fresh_sf` | `ctpt_softmax_out` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 406 | `layer5.block4.v_fresh_sf` | `ctpt_v` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 407 | `layer5.block4.softmax_out_mask_sf` | `ctpt_softmax_out_mask` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 408 | `layer5.block4.v_mask_sf` | `ctpt_v_mask` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 409 | `layer5.block4.softmax_v_mask_sf` | `ctpt_softmax_v_mask` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 410 | `layer5.block4.ln_mean_inv_d_sf` | `ctpt_inv_d_attn_mean` | `scalar` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 411 | `layer5.block4.ln_var_inv_d_sf` | `ctpt_inv_d_attn_var` | `scalar` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 412 | `layer5.block4.wo_sf` | `ctpt_wo` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 413 | `layer5.block4.softmax_out_mask_rescale_sf` | `ctct_softmax_out_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 414 | `layer5.block4.v_mask_rescale_sf` | `ctct_v_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 415 | `layer5.block4.softmax_v_matmul_rescale_sf` | `ctct_softmax_v_matmul_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 416 | `layer5.block4.softmax_v_mask_rescale_sf` | `ctct_softmax_v_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 417 | `layer5.block4.wo_rescale_sf` | `ctct_wo_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 418 | `layer5.block4.ln_mean_rescale_sf` | `ctct_attn_mean_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 419 | `layer5.block4.ln_square_rescale_sf` | `ctct_attn_square_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 420 | `layer5.block4.ln_var_rescale_sf` | `ctct_attn_var_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 421 | `layer5.block4.output_truncation_k` | `block4_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 422 | `layer5.block5.inv_std_fresh_sf` | `ctpt_attn_inv_std` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 423 | `layer5.block5.x_centered_fresh_sf` | `ctpt_attn_x_centered` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 424 | `layer5.block5.gamma_sf` | `ctpt_gamma_attn` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 425 | `layer5.block5.wffn1_sf` | `ctpt_wffn1` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 426 | `layer5.block5.gelu_coeff_sf` | `ctpt_gelu_coeff` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 427 | `layer5.block5.normalize_rescale_sf` | `ctct_attn_normalize_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 428 | `layer5.block5.gamma_rescale_sf` | `ctct_gamma_attn_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 429 | `layer5.block5.wffn1_rescale_sf` | `ctct_wffn1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 430 | `layer5.block5.gelu_power_rescale_sf_0` | `ctct_gelu_x2` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 431 | `layer5.block5.gelu_power_rescale_sf_1` | `ctct_gelu_x3` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 432 | `layer5.block5.gelu_power_rescale_sf_2` | `ctct_gelu_x4` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 433 | `layer5.block5.gelu_coeff_mul_rescale_sf_0` | `ctct_gelu_b_x` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 434 | `layer5.block5.gelu_coeff_mul_rescale_sf_1` | `ctct_gelu_c_x2` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 435 | `layer5.block5.gelu_coeff_mul_rescale_sf_2` | `ctct_gelu_d_x3` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 436 | `layer5.block5.gelu_coeff_mul_rescale_sf_3` | `ctct_gelu_e_x4` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 437 | `layer5.block5.output_truncation_k` | `block5_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 438 | `layer6.block1.gelu_out_sf` | `ctpt_gelu_out` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 439 | `layer6.block1.wffn2_sf` | `ctpt_ffn2` | `encoding` | 4 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 440 | `layer6.block1.mean_inv_d_sf` | `ctpt_inv_d_1` | `scalar` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 441 | `layer6.block1.var_inv_d_sf` | `ctpt_inv_d_2` | `scalar` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 442 | `layer6.block1.wffn2_rescale_sf` | `ctct_ffn2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 443 | `layer6.block1.mean_rescale_sf` | `ctct_mean_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 444 | `layer6.block1.square_rescale_sf` | `ctct_ext_square` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 445 | `layer6.block1.var_rescale_sf` | `ctct_var_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 446 | `layer6.block1.output_truncation_k` | `block1_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 447 | `layer6.block2.inv_std_fresh_sf` | `ctpt_inv_std` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 448 | `layer6.block2.x_centered_fresh_sf` | `ctpt_x_centered` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 449 | `layer6.block2.gamma_sf` | `ctpt_gamma` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 450 | `layer6.block2.wq_sf` | `ctpt_wq` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 451 | `layer6.block2.wk_sf` | `ctpt_wk` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 452 | `layer6.block2.wv_sf` | `ctpt_wv` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 453 | `layer6.block2.kt_mask1_sf` | `ctpt_kt_mask1` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 454 | `layer6.block2.kt_mask2_sf` | `ctpt_kt_mask2` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 455 | `layer6.block2.q_mask1_sf` | `ctpt_q_mask1` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 456 | `layer6.block2.q_mask2_sf` | `ctpt_q_mask2` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 457 | `layer6.block2.qkt_merge_mask_sf` | `ctpt_qkt_merge_mask` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 458 | `layer6.block2.normalize_rescale_sf` | `ctct_normalize_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 459 | `layer6.block2.gamma_rescale_sf` | `ctct_gamma_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 460 | `layer6.block2.wk_rescale_sf` | `ctct_wk_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 461 | `layer6.block2.wq_rescale_sf` | `ctct_wq_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 462 | `layer6.block2.wv_rescale_sf` | `ctct_wv_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 463 | `layer6.block2.kt_mask1_rescale_sf` | `ctct_kt_mask1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 464 | `layer6.block2.kt_mask2_rescale_sf` | `ctct_kt_mask2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 465 | `layer6.block2.q_mask1_rescale_sf` | `ctct_q_mask1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 466 | `layer6.block2.q_mask2_rescale_sf` | `ctct_q_mask2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 467 | `layer6.block2.qkt_matmul_rescale_sf` | `ctct_qkt_matmul_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 468 | `layer6.block2.qkt_merge_mask_rescale_sf` | `ctct_qkt_merge_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 469 | `layer6.block2.output_truncation_k` | `block2_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 470 | `layer6.block3.x_fresh_sf` | `ctpt_softmax_x` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 471 | `layer6.block3.inv_2n_sf` | `ctpt_softmax_inv_2n` | `scalar` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 472 | `layer6.block3.x_inv_2n_rescale_sf` | `ctct_softmax_x_inv_2n_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 473 | `layer6.block3.square_rescale_sf_0` | `ctct_softmax_pow_s1` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 474 | `layer6.block3.square_rescale_sf_1` | `ctct_softmax_pow_s2` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 475 | `layer6.block3.square_rescale_sf_2` | `ctct_softmax_pow_s3` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 476 | `layer6.block3.square_rescale_sf_3` | `ctct_softmax_pow_s4` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 477 | `layer6.block3.output_truncation_k` | `block3_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 478 | `layer6.block4.softmax_out_fresh_sf` | `ctpt_softmax_out` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 479 | `layer6.block4.v_fresh_sf` | `ctpt_v` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 480 | `layer6.block4.softmax_out_mask_sf` | `ctpt_softmax_out_mask` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 481 | `layer6.block4.v_mask_sf` | `ctpt_v_mask` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 482 | `layer6.block4.softmax_v_mask_sf` | `ctpt_softmax_v_mask` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 483 | `layer6.block4.ln_mean_inv_d_sf` | `ctpt_inv_d_attn_mean` | `scalar` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 484 | `layer6.block4.ln_var_inv_d_sf` | `ctpt_inv_d_attn_var` | `scalar` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 485 | `layer6.block4.wo_sf` | `ctpt_wo` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 486 | `layer6.block4.softmax_out_mask_rescale_sf` | `ctct_softmax_out_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 487 | `layer6.block4.v_mask_rescale_sf` | `ctct_v_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 488 | `layer6.block4.softmax_v_matmul_rescale_sf` | `ctct_softmax_v_matmul_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 489 | `layer6.block4.softmax_v_mask_rescale_sf` | `ctct_softmax_v_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 490 | `layer6.block4.wo_rescale_sf` | `ctct_wo_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 491 | `layer6.block4.ln_mean_rescale_sf` | `ctct_attn_mean_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 492 | `layer6.block4.ln_square_rescale_sf` | `ctct_attn_square_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 493 | `layer6.block4.ln_var_rescale_sf` | `ctct_attn_var_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 494 | `layer6.block4.output_truncation_k` | `block4_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 495 | `layer6.block5.inv_std_fresh_sf` | `ctpt_attn_inv_std` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 496 | `layer6.block5.x_centered_fresh_sf` | `ctpt_attn_x_centered` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 497 | `layer6.block5.gamma_sf` | `ctpt_gamma_attn` | `encoding` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 498 | `layer6.block5.wffn1_sf` | `ctpt_wffn1` | `encoding` | 4 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 499 | `layer6.block5.gelu_coeff_sf` | `ctpt_gelu_coeff` | `encoding` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 500 | `layer6.block5.normalize_rescale_sf` | `ctct_attn_normalize_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 501 | `layer6.block5.gamma_rescale_sf` | `ctct_gamma_attn_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 502 | `layer6.block5.wffn1_rescale_sf` | `ctct_wffn1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 503 | `layer6.block5.gelu_power_rescale_sf_0` | `ctct_gelu_x2` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 504 | `layer6.block5.gelu_power_rescale_sf_1` | `ctct_gelu_x3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 505 | `layer6.block5.gelu_power_rescale_sf_2` | `ctct_gelu_x4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 506 | `layer6.block5.gelu_coeff_mul_rescale_sf_0` | `ctct_gelu_b_x` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 507 | `layer6.block5.gelu_coeff_mul_rescale_sf_1` | `ctct_gelu_c_x2` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 508 | `layer6.block5.gelu_coeff_mul_rescale_sf_2` | `ctct_gelu_d_x3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 509 | `layer6.block5.gelu_coeff_mul_rescale_sf_3` | `ctct_gelu_e_x4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 510 | `layer6.block5.output_truncation_k` | `block5_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 511 | `layer7.block1.gelu_out_sf` | `ctpt_gelu_out` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 512 | `layer7.block1.wffn2_sf` | `ctpt_ffn2` | `encoding` | 4 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 513 | `layer7.block1.mean_inv_d_sf` | `ctpt_inv_d_1` | `scalar` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 514 | `layer7.block1.var_inv_d_sf` | `ctpt_inv_d_2` | `scalar` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 515 | `layer7.block1.wffn2_rescale_sf` | `ctct_ffn2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 516 | `layer7.block1.mean_rescale_sf` | `ctct_mean_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 517 | `layer7.block1.square_rescale_sf` | `ctct_ext_square` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 518 | `layer7.block1.var_rescale_sf` | `ctct_var_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 519 | `layer7.block1.output_truncation_k` | `block1_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 520 | `layer7.block2.inv_std_fresh_sf` | `ctpt_inv_std` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 521 | `layer7.block2.x_centered_fresh_sf` | `ctpt_x_centered` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 522 | `layer7.block2.gamma_sf` | `ctpt_gamma` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 523 | `layer7.block2.wq_sf` | `ctpt_wq` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 524 | `layer7.block2.wk_sf` | `ctpt_wk` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 525 | `layer7.block2.wv_sf` | `ctpt_wv` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 526 | `layer7.block2.kt_mask1_sf` | `ctpt_kt_mask1` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 527 | `layer7.block2.kt_mask2_sf` | `ctpt_kt_mask2` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 528 | `layer7.block2.q_mask1_sf` | `ctpt_q_mask1` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 529 | `layer7.block2.q_mask2_sf` | `ctpt_q_mask2` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 530 | `layer7.block2.qkt_merge_mask_sf` | `ctpt_qkt_merge_mask` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 531 | `layer7.block2.normalize_rescale_sf` | `ctct_normalize_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 532 | `layer7.block2.gamma_rescale_sf` | `ctct_gamma_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 533 | `layer7.block2.wk_rescale_sf` | `ctct_wk_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 534 | `layer7.block2.wq_rescale_sf` | `ctct_wq_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 535 | `layer7.block2.wv_rescale_sf` | `ctct_wv_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 536 | `layer7.block2.kt_mask1_rescale_sf` | `ctct_kt_mask1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 537 | `layer7.block2.kt_mask2_rescale_sf` | `ctct_kt_mask2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 538 | `layer7.block2.q_mask1_rescale_sf` | `ctct_q_mask1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 539 | `layer7.block2.q_mask2_rescale_sf` | `ctct_q_mask2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 540 | `layer7.block2.qkt_matmul_rescale_sf` | `ctct_qkt_matmul_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 541 | `layer7.block2.qkt_merge_mask_rescale_sf` | `ctct_qkt_merge_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 542 | `layer7.block2.output_truncation_k` | `block2_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 543 | `layer7.block3.x_fresh_sf` | `ctpt_softmax_x` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 544 | `layer7.block3.inv_2n_sf` | `ctpt_softmax_inv_2n` | `scalar` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 545 | `layer7.block3.x_inv_2n_rescale_sf` | `ctct_softmax_x_inv_2n_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 546 | `layer7.block3.square_rescale_sf_0` | `ctct_softmax_pow_s1` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 547 | `layer7.block3.square_rescale_sf_1` | `ctct_softmax_pow_s2` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 548 | `layer7.block3.square_rescale_sf_2` | `ctct_softmax_pow_s3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | softmax degree 2 does not use this square-rescale slot |
| 549 | `layer7.block3.square_rescale_sf_3` | `ctct_softmax_pow_s4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | softmax degree 2 does not use this square-rescale slot |
| 550 | `layer7.block3.output_truncation_k` | `block3_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 551 | `layer7.block4.softmax_out_fresh_sf` | `ctpt_softmax_out` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 552 | `layer7.block4.v_fresh_sf` | `ctpt_v` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 553 | `layer7.block4.softmax_out_mask_sf` | `ctpt_softmax_out_mask` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 554 | `layer7.block4.v_mask_sf` | `ctpt_v_mask` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 555 | `layer7.block4.softmax_v_mask_sf` | `ctpt_softmax_v_mask` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 556 | `layer7.block4.ln_mean_inv_d_sf` | `ctpt_inv_d_attn_mean` | `scalar` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 557 | `layer7.block4.ln_var_inv_d_sf` | `ctpt_inv_d_attn_var` | `scalar` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 558 | `layer7.block4.wo_sf` | `ctpt_wo` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 559 | `layer7.block4.softmax_out_mask_rescale_sf` | `ctct_softmax_out_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 560 | `layer7.block4.v_mask_rescale_sf` | `ctct_v_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 561 | `layer7.block4.softmax_v_matmul_rescale_sf` | `ctct_softmax_v_matmul_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 562 | `layer7.block4.softmax_v_mask_rescale_sf` | `ctct_softmax_v_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 563 | `layer7.block4.wo_rescale_sf` | `ctct_wo_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 564 | `layer7.block4.ln_mean_rescale_sf` | `ctct_attn_mean_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 565 | `layer7.block4.ln_square_rescale_sf` | `ctct_attn_square_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 566 | `layer7.block4.ln_var_rescale_sf` | `ctct_attn_var_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 567 | `layer7.block4.output_truncation_k` | `block4_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 568 | `layer7.block5.inv_std_fresh_sf` | `ctpt_attn_inv_std` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 569 | `layer7.block5.x_centered_fresh_sf` | `ctpt_attn_x_centered` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 570 | `layer7.block5.gamma_sf` | `ctpt_gamma_attn` | `encoding` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 571 | `layer7.block5.wffn1_sf` | `ctpt_wffn1` | `encoding` | 4 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 572 | `layer7.block5.gelu_coeff_sf` | `ctpt_gelu_coeff` | `encoding` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 573 | `layer7.block5.normalize_rescale_sf` | `ctct_attn_normalize_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 574 | `layer7.block5.gamma_rescale_sf` | `ctct_gamma_attn_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 575 | `layer7.block5.wffn1_rescale_sf` | `ctct_wffn1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 576 | `layer7.block5.gelu_power_rescale_sf_0` | `ctct_gelu_x2` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 577 | `layer7.block5.gelu_power_rescale_sf_1` | `ctct_gelu_x3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 578 | `layer7.block5.gelu_power_rescale_sf_2` | `ctct_gelu_x4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 579 | `layer7.block5.gelu_coeff_mul_rescale_sf_0` | `ctct_gelu_b_x` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 580 | `layer7.block5.gelu_coeff_mul_rescale_sf_1` | `ctct_gelu_c_x2` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 581 | `layer7.block5.gelu_coeff_mul_rescale_sf_2` | `ctct_gelu_d_x3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 582 | `layer7.block5.gelu_coeff_mul_rescale_sf_3` | `ctct_gelu_e_x4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 583 | `layer7.block5.output_truncation_k` | `block5_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 584 | `layer8.block1.gelu_out_sf` | `ctpt_gelu_out` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 585 | `layer8.block1.wffn2_sf` | `ctpt_ffn2` | `encoding` | 4 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 586 | `layer8.block1.mean_inv_d_sf` | `ctpt_inv_d_1` | `scalar` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 587 | `layer8.block1.var_inv_d_sf` | `ctpt_inv_d_2` | `scalar` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 588 | `layer8.block1.wffn2_rescale_sf` | `ctct_ffn2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 589 | `layer8.block1.mean_rescale_sf` | `ctct_mean_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 590 | `layer8.block1.square_rescale_sf` | `ctct_ext_square` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 591 | `layer8.block1.var_rescale_sf` | `ctct_var_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 592 | `layer8.block1.output_truncation_k` | `block1_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 593 | `layer8.block2.inv_std_fresh_sf` | `ctpt_inv_std` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 594 | `layer8.block2.x_centered_fresh_sf` | `ctpt_x_centered` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 595 | `layer8.block2.gamma_sf` | `ctpt_gamma` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 596 | `layer8.block2.wq_sf` | `ctpt_wq` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 597 | `layer8.block2.wk_sf` | `ctpt_wk` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 598 | `layer8.block2.wv_sf` | `ctpt_wv` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 599 | `layer8.block2.kt_mask1_sf` | `ctpt_kt_mask1` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 600 | `layer8.block2.kt_mask2_sf` | `ctpt_kt_mask2` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 601 | `layer8.block2.q_mask1_sf` | `ctpt_q_mask1` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 602 | `layer8.block2.q_mask2_sf` | `ctpt_q_mask2` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 603 | `layer8.block2.qkt_merge_mask_sf` | `ctpt_qkt_merge_mask` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 604 | `layer8.block2.normalize_rescale_sf` | `ctct_normalize_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 605 | `layer8.block2.gamma_rescale_sf` | `ctct_gamma_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 606 | `layer8.block2.wk_rescale_sf` | `ctct_wk_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 607 | `layer8.block2.wq_rescale_sf` | `ctct_wq_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 608 | `layer8.block2.wv_rescale_sf` | `ctct_wv_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 609 | `layer8.block2.kt_mask1_rescale_sf` | `ctct_kt_mask1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 610 | `layer8.block2.kt_mask2_rescale_sf` | `ctct_kt_mask2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 611 | `layer8.block2.q_mask1_rescale_sf` | `ctct_q_mask1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 612 | `layer8.block2.q_mask2_rescale_sf` | `ctct_q_mask2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 613 | `layer8.block2.qkt_matmul_rescale_sf` | `ctct_qkt_matmul_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 614 | `layer8.block2.qkt_merge_mask_rescale_sf` | `ctct_qkt_merge_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 615 | `layer8.block2.output_truncation_k` | `block2_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 616 | `layer8.block3.x_fresh_sf` | `ctpt_softmax_x` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 617 | `layer8.block3.inv_2n_sf` | `ctpt_softmax_inv_2n` | `scalar` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 618 | `layer8.block3.x_inv_2n_rescale_sf` | `ctct_softmax_x_inv_2n_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 619 | `layer8.block3.square_rescale_sf_0` | `ctct_softmax_pow_s1` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 620 | `layer8.block3.square_rescale_sf_1` | `ctct_softmax_pow_s2` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 621 | `layer8.block3.square_rescale_sf_2` | `ctct_softmax_pow_s3` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 622 | `layer8.block3.square_rescale_sf_3` | `ctct_softmax_pow_s4` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 623 | `layer8.block3.output_truncation_k` | `block3_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 624 | `layer8.block4.softmax_out_fresh_sf` | `ctpt_softmax_out` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 625 | `layer8.block4.v_fresh_sf` | `ctpt_v` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 626 | `layer8.block4.softmax_out_mask_sf` | `ctpt_softmax_out_mask` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 627 | `layer8.block4.v_mask_sf` | `ctpt_v_mask` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 628 | `layer8.block4.softmax_v_mask_sf` | `ctpt_softmax_v_mask` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 629 | `layer8.block4.ln_mean_inv_d_sf` | `ctpt_inv_d_attn_mean` | `scalar` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 630 | `layer8.block4.ln_var_inv_d_sf` | `ctpt_inv_d_attn_var` | `scalar` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 631 | `layer8.block4.wo_sf` | `ctpt_wo` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 632 | `layer8.block4.softmax_out_mask_rescale_sf` | `ctct_softmax_out_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 633 | `layer8.block4.v_mask_rescale_sf` | `ctct_v_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 634 | `layer8.block4.softmax_v_matmul_rescale_sf` | `ctct_softmax_v_matmul_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 635 | `layer8.block4.softmax_v_mask_rescale_sf` | `ctct_softmax_v_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 636 | `layer8.block4.wo_rescale_sf` | `ctct_wo_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 637 | `layer8.block4.ln_mean_rescale_sf` | `ctct_attn_mean_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 638 | `layer8.block4.ln_square_rescale_sf` | `ctct_attn_square_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 639 | `layer8.block4.ln_var_rescale_sf` | `ctct_attn_var_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 640 | `layer8.block4.output_truncation_k` | `block4_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 641 | `layer8.block5.inv_std_fresh_sf` | `ctpt_attn_inv_std` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 642 | `layer8.block5.x_centered_fresh_sf` | `ctpt_attn_x_centered` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 643 | `layer8.block5.gamma_sf` | `ctpt_gamma_attn` | `encoding` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 644 | `layer8.block5.wffn1_sf` | `ctpt_wffn1` | `encoding` | 4 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 645 | `layer8.block5.gelu_coeff_sf` | `ctpt_gelu_coeff` | `encoding` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 646 | `layer8.block5.normalize_rescale_sf` | `ctct_attn_normalize_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 647 | `layer8.block5.gamma_rescale_sf` | `ctct_gamma_attn_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 648 | `layer8.block5.wffn1_rescale_sf` | `ctct_wffn1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 649 | `layer8.block5.gelu_power_rescale_sf_0` | `ctct_gelu_x2` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 650 | `layer8.block5.gelu_power_rescale_sf_1` | `ctct_gelu_x3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 651 | `layer8.block5.gelu_power_rescale_sf_2` | `ctct_gelu_x4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 652 | `layer8.block5.gelu_coeff_mul_rescale_sf_0` | `ctct_gelu_b_x` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 653 | `layer8.block5.gelu_coeff_mul_rescale_sf_1` | `ctct_gelu_c_x2` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 654 | `layer8.block5.gelu_coeff_mul_rescale_sf_2` | `ctct_gelu_d_x3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 655 | `layer8.block5.gelu_coeff_mul_rescale_sf_3` | `ctct_gelu_e_x4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 656 | `layer8.block5.output_truncation_k` | `block5_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 657 | `layer9.block1.gelu_out_sf` | `ctpt_gelu_out` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 658 | `layer9.block1.wffn2_sf` | `ctpt_ffn2` | `encoding` | 4 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 659 | `layer9.block1.mean_inv_d_sf` | `ctpt_inv_d_1` | `scalar` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 660 | `layer9.block1.var_inv_d_sf` | `ctpt_inv_d_2` | `scalar` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 661 | `layer9.block1.wffn2_rescale_sf` | `ctct_ffn2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 662 | `layer9.block1.mean_rescale_sf` | `ctct_mean_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 663 | `layer9.block1.square_rescale_sf` | `ctct_ext_square` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 664 | `layer9.block1.var_rescale_sf` | `ctct_var_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 665 | `layer9.block1.output_truncation_k` | `block1_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 666 | `layer9.block2.inv_std_fresh_sf` | `ctpt_inv_std` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 667 | `layer9.block2.x_centered_fresh_sf` | `ctpt_x_centered` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 668 | `layer9.block2.gamma_sf` | `ctpt_gamma` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 669 | `layer9.block2.wq_sf` | `ctpt_wq` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 670 | `layer9.block2.wk_sf` | `ctpt_wk` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 671 | `layer9.block2.wv_sf` | `ctpt_wv` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 672 | `layer9.block2.kt_mask1_sf` | `ctpt_kt_mask1` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 673 | `layer9.block2.kt_mask2_sf` | `ctpt_kt_mask2` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 674 | `layer9.block2.q_mask1_sf` | `ctpt_q_mask1` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 675 | `layer9.block2.q_mask2_sf` | `ctpt_q_mask2` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 676 | `layer9.block2.qkt_merge_mask_sf` | `ctpt_qkt_merge_mask` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 677 | `layer9.block2.normalize_rescale_sf` | `ctct_normalize_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 678 | `layer9.block2.gamma_rescale_sf` | `ctct_gamma_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 679 | `layer9.block2.wk_rescale_sf` | `ctct_wk_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 680 | `layer9.block2.wq_rescale_sf` | `ctct_wq_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 681 | `layer9.block2.wv_rescale_sf` | `ctct_wv_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 682 | `layer9.block2.kt_mask1_rescale_sf` | `ctct_kt_mask1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 683 | `layer9.block2.kt_mask2_rescale_sf` | `ctct_kt_mask2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 684 | `layer9.block2.q_mask1_rescale_sf` | `ctct_q_mask1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 685 | `layer9.block2.q_mask2_rescale_sf` | `ctct_q_mask2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 686 | `layer9.block2.qkt_matmul_rescale_sf` | `ctct_qkt_matmul_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 687 | `layer9.block2.qkt_merge_mask_rescale_sf` | `ctct_qkt_merge_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 688 | `layer9.block2.output_truncation_k` | `block2_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 689 | `layer9.block3.x_fresh_sf` | `ctpt_softmax_x` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 690 | `layer9.block3.inv_2n_sf` | `ctpt_softmax_inv_2n` | `scalar` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 691 | `layer9.block3.x_inv_2n_rescale_sf` | `ctct_softmax_x_inv_2n_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 692 | `layer9.block3.square_rescale_sf_0` | `ctct_softmax_pow_s1` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 693 | `layer9.block3.square_rescale_sf_1` | `ctct_softmax_pow_s2` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 694 | `layer9.block3.square_rescale_sf_2` | `ctct_softmax_pow_s3` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 695 | `layer9.block3.square_rescale_sf_3` | `ctct_softmax_pow_s4` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 696 | `layer9.block3.output_truncation_k` | `block3_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 697 | `layer9.block4.softmax_out_fresh_sf` | `ctpt_softmax_out` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 698 | `layer9.block4.v_fresh_sf` | `ctpt_v` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 699 | `layer9.block4.softmax_out_mask_sf` | `ctpt_softmax_out_mask` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 700 | `layer9.block4.v_mask_sf` | `ctpt_v_mask` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 701 | `layer9.block4.softmax_v_mask_sf` | `ctpt_softmax_v_mask` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 702 | `layer9.block4.ln_mean_inv_d_sf` | `ctpt_inv_d_attn_mean` | `scalar` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 703 | `layer9.block4.ln_var_inv_d_sf` | `ctpt_inv_d_attn_var` | `scalar` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 704 | `layer9.block4.wo_sf` | `ctpt_wo` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 705 | `layer9.block4.softmax_out_mask_rescale_sf` | `ctct_softmax_out_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 706 | `layer9.block4.v_mask_rescale_sf` | `ctct_v_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 707 | `layer9.block4.softmax_v_matmul_rescale_sf` | `ctct_softmax_v_matmul_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 708 | `layer9.block4.softmax_v_mask_rescale_sf` | `ctct_softmax_v_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 709 | `layer9.block4.wo_rescale_sf` | `ctct_wo_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 710 | `layer9.block4.ln_mean_rescale_sf` | `ctct_attn_mean_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 711 | `layer9.block4.ln_square_rescale_sf` | `ctct_attn_square_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 712 | `layer9.block4.ln_var_rescale_sf` | `ctct_attn_var_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 713 | `layer9.block4.output_truncation_k` | `block4_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 714 | `layer9.block5.inv_std_fresh_sf` | `ctpt_attn_inv_std` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 715 | `layer9.block5.x_centered_fresh_sf` | `ctpt_attn_x_centered` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 716 | `layer9.block5.gamma_sf` | `ctpt_gamma_attn` | `encoding` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 717 | `layer9.block5.wffn1_sf` | `ctpt_wffn1` | `encoding` | 4 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 718 | `layer9.block5.gelu_coeff_sf` | `ctpt_gelu_coeff` | `encoding` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 719 | `layer9.block5.normalize_rescale_sf` | `ctct_attn_normalize_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 720 | `layer9.block5.gamma_rescale_sf` | `ctct_gamma_attn_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 721 | `layer9.block5.wffn1_rescale_sf` | `ctct_wffn1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 722 | `layer9.block5.gelu_power_rescale_sf_0` | `ctct_gelu_x2` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 723 | `layer9.block5.gelu_power_rescale_sf_1` | `ctct_gelu_x3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 724 | `layer9.block5.gelu_power_rescale_sf_2` | `ctct_gelu_x4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 725 | `layer9.block5.gelu_coeff_mul_rescale_sf_0` | `ctct_gelu_b_x` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 726 | `layer9.block5.gelu_coeff_mul_rescale_sf_1` | `ctct_gelu_c_x2` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 727 | `layer9.block5.gelu_coeff_mul_rescale_sf_2` | `ctct_gelu_d_x3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 728 | `layer9.block5.gelu_coeff_mul_rescale_sf_3` | `ctct_gelu_e_x4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 729 | `layer9.block5.output_truncation_k` | `block5_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 730 | `layer10.block1.gelu_out_sf` | `ctpt_gelu_out` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 731 | `layer10.block1.wffn2_sf` | `ctpt_ffn2` | `encoding` | 4 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 732 | `layer10.block1.mean_inv_d_sf` | `ctpt_inv_d_1` | `scalar` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 733 | `layer10.block1.var_inv_d_sf` | `ctpt_inv_d_2` | `scalar` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 734 | `layer10.block1.wffn2_rescale_sf` | `ctct_ffn2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 735 | `layer10.block1.mean_rescale_sf` | `ctct_mean_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 736 | `layer10.block1.square_rescale_sf` | `ctct_ext_square` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 737 | `layer10.block1.var_rescale_sf` | `ctct_var_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 738 | `layer10.block1.output_truncation_k` | `block1_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 739 | `layer10.block2.inv_std_fresh_sf` | `ctpt_inv_std` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 740 | `layer10.block2.x_centered_fresh_sf` | `ctpt_x_centered` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 741 | `layer10.block2.gamma_sf` | `ctpt_gamma` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 742 | `layer10.block2.wq_sf` | `ctpt_wq` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 743 | `layer10.block2.wk_sf` | `ctpt_wk` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 744 | `layer10.block2.wv_sf` | `ctpt_wv` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 745 | `layer10.block2.kt_mask1_sf` | `ctpt_kt_mask1` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 746 | `layer10.block2.kt_mask2_sf` | `ctpt_kt_mask2` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 747 | `layer10.block2.q_mask1_sf` | `ctpt_q_mask1` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 748 | `layer10.block2.q_mask2_sf` | `ctpt_q_mask2` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 749 | `layer10.block2.qkt_merge_mask_sf` | `ctpt_qkt_merge_mask` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 750 | `layer10.block2.normalize_rescale_sf` | `ctct_normalize_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 751 | `layer10.block2.gamma_rescale_sf` | `ctct_gamma_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 752 | `layer10.block2.wk_rescale_sf` | `ctct_wk_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 753 | `layer10.block2.wq_rescale_sf` | `ctct_wq_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 754 | `layer10.block2.wv_rescale_sf` | `ctct_wv_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 755 | `layer10.block2.kt_mask1_rescale_sf` | `ctct_kt_mask1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 756 | `layer10.block2.kt_mask2_rescale_sf` | `ctct_kt_mask2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 757 | `layer10.block2.q_mask1_rescale_sf` | `ctct_q_mask1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 758 | `layer10.block2.q_mask2_rescale_sf` | `ctct_q_mask2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 759 | `layer10.block2.qkt_matmul_rescale_sf` | `ctct_qkt_matmul_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 760 | `layer10.block2.qkt_merge_mask_rescale_sf` | `ctct_qkt_merge_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 761 | `layer10.block2.output_truncation_k` | `block2_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 762 | `layer10.block3.x_fresh_sf` | `ctpt_softmax_x` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 763 | `layer10.block3.inv_2n_sf` | `ctpt_softmax_inv_2n` | `scalar` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 764 | `layer10.block3.x_inv_2n_rescale_sf` | `ctct_softmax_x_inv_2n_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 765 | `layer10.block3.square_rescale_sf_0` | `ctct_softmax_pow_s1` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 766 | `layer10.block3.square_rescale_sf_1` | `ctct_softmax_pow_s2` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 767 | `layer10.block3.square_rescale_sf_2` | `ctct_softmax_pow_s3` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 768 | `layer10.block3.square_rescale_sf_3` | `ctct_softmax_pow_s4` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 769 | `layer10.block3.output_truncation_k` | `block3_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 770 | `layer10.block4.softmax_out_fresh_sf` | `ctpt_softmax_out` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 771 | `layer10.block4.v_fresh_sf` | `ctpt_v` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 772 | `layer10.block4.softmax_out_mask_sf` | `ctpt_softmax_out_mask` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 773 | `layer10.block4.v_mask_sf` | `ctpt_v_mask` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 774 | `layer10.block4.softmax_v_mask_sf` | `ctpt_softmax_v_mask` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 775 | `layer10.block4.ln_mean_inv_d_sf` | `ctpt_inv_d_attn_mean` | `scalar` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 776 | `layer10.block4.ln_var_inv_d_sf` | `ctpt_inv_d_attn_var` | `scalar` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 777 | `layer10.block4.wo_sf` | `ctpt_wo` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 778 | `layer10.block4.softmax_out_mask_rescale_sf` | `ctct_softmax_out_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 779 | `layer10.block4.v_mask_rescale_sf` | `ctct_v_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 780 | `layer10.block4.softmax_v_matmul_rescale_sf` | `ctct_softmax_v_matmul_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 781 | `layer10.block4.softmax_v_mask_rescale_sf` | `ctct_softmax_v_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 782 | `layer10.block4.wo_rescale_sf` | `ctct_wo_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 783 | `layer10.block4.ln_mean_rescale_sf` | `ctct_attn_mean_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 784 | `layer10.block4.ln_square_rescale_sf` | `ctct_attn_square_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 785 | `layer10.block4.ln_var_rescale_sf` | `ctct_attn_var_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 786 | `layer10.block4.output_truncation_k` | `block4_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 787 | `layer10.block5.inv_std_fresh_sf` | `ctpt_attn_inv_std` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 788 | `layer10.block5.x_centered_fresh_sf` | `ctpt_attn_x_centered` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 789 | `layer10.block5.gamma_sf` | `ctpt_gamma_attn` | `encoding` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 790 | `layer10.block5.wffn1_sf` | `ctpt_wffn1` | `encoding` | 4 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 791 | `layer10.block5.gelu_coeff_sf` | `ctpt_gelu_coeff` | `encoding` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 792 | `layer10.block5.normalize_rescale_sf` | `ctct_attn_normalize_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 793 | `layer10.block5.gamma_rescale_sf` | `ctct_gamma_attn_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 794 | `layer10.block5.wffn1_rescale_sf` | `ctct_wffn1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 795 | `layer10.block5.gelu_power_rescale_sf_0` | `ctct_gelu_x2` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 796 | `layer10.block5.gelu_power_rescale_sf_1` | `ctct_gelu_x3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 797 | `layer10.block5.gelu_power_rescale_sf_2` | `ctct_gelu_x4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 798 | `layer10.block5.gelu_coeff_mul_rescale_sf_0` | `ctct_gelu_b_x` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 799 | `layer10.block5.gelu_coeff_mul_rescale_sf_1` | `ctct_gelu_c_x2` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 800 | `layer10.block5.gelu_coeff_mul_rescale_sf_2` | `ctct_gelu_d_x3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 801 | `layer10.block5.gelu_coeff_mul_rescale_sf_3` | `ctct_gelu_e_x4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 802 | `layer10.block5.output_truncation_k` | `block5_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 803 | `layer11.block1.gelu_out_sf` | `ctpt_gelu_out` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 804 | `layer11.block1.wffn2_sf` | `ctpt_ffn2` | `encoding` | 4 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 805 | `layer11.block1.mean_inv_d_sf` | `ctpt_inv_d_1` | `scalar` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 806 | `layer11.block1.var_inv_d_sf` | `ctpt_inv_d_2` | `scalar` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 807 | `layer11.block1.wffn2_rescale_sf` | `ctct_ffn2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 808 | `layer11.block1.mean_rescale_sf` | `ctct_mean_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 809 | `layer11.block1.square_rescale_sf` | `ctct_ext_square` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 810 | `layer11.block1.var_rescale_sf` | `ctct_var_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 811 | `layer11.block1.output_truncation_k` | `block1_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 812 | `layer11.block2.inv_std_fresh_sf` | `ctpt_inv_std` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 813 | `layer11.block2.x_centered_fresh_sf` | `ctpt_x_centered` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 814 | `layer11.block2.gamma_sf` | `ctpt_gamma` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 815 | `layer11.block2.wq_sf` | `ctpt_wq` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 816 | `layer11.block2.wk_sf` | `ctpt_wk` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 817 | `layer11.block2.wv_sf` | `ctpt_wv` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 818 | `layer11.block2.kt_mask1_sf` | `ctpt_kt_mask1` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 819 | `layer11.block2.kt_mask2_sf` | `ctpt_kt_mask2` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 820 | `layer11.block2.q_mask1_sf` | `ctpt_q_mask1` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 821 | `layer11.block2.q_mask2_sf` | `ctpt_q_mask2` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 822 | `layer11.block2.qkt_merge_mask_sf` | `ctpt_qkt_merge_mask` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 823 | `layer11.block2.normalize_rescale_sf` | `ctct_normalize_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 824 | `layer11.block2.gamma_rescale_sf` | `ctct_gamma_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 825 | `layer11.block2.wk_rescale_sf` | `ctct_wk_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 826 | `layer11.block2.wq_rescale_sf` | `ctct_wq_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 827 | `layer11.block2.wv_rescale_sf` | `ctct_wv_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 828 | `layer11.block2.kt_mask1_rescale_sf` | `ctct_kt_mask1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 829 | `layer11.block2.kt_mask2_rescale_sf` | `ctct_kt_mask2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 830 | `layer11.block2.q_mask1_rescale_sf` | `ctct_q_mask1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 831 | `layer11.block2.q_mask2_rescale_sf` | `ctct_q_mask2_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 832 | `layer11.block2.qkt_matmul_rescale_sf` | `ctct_qkt_matmul_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 833 | `layer11.block2.qkt_merge_mask_rescale_sf` | `ctct_qkt_merge_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 834 | `layer11.block2.output_truncation_k` | `block2_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 835 | `layer11.block3.x_fresh_sf` | `ctpt_softmax_x` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 836 | `layer11.block3.inv_2n_sf` | `ctpt_softmax_inv_2n` | `scalar` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 837 | `layer11.block3.x_inv_2n_rescale_sf` | `ctct_softmax_x_inv_2n_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 838 | `layer11.block3.square_rescale_sf_0` | `ctct_softmax_pow_s1` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 839 | `layer11.block3.square_rescale_sf_1` | `ctct_softmax_pow_s2` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 840 | `layer11.block3.square_rescale_sf_2` | `ctct_softmax_pow_s3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | softmax degree 2 does not use this square-rescale slot |
| 841 | `layer11.block3.square_rescale_sf_3` | `ctct_softmax_pow_s4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | softmax degree 2 does not use this square-rescale slot |
| 842 | `layer11.block3.output_truncation_k` | `block3_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 843 | `layer11.block4.softmax_out_fresh_sf` | `ctpt_softmax_out` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 844 | `layer11.block4.v_fresh_sf` | `ctpt_v` | `fresh` | 4 | `scaling_factor` | 30 | True | 16384 | 30 |  |
| 845 | `layer11.block4.softmax_out_mask_sf` | `ctpt_softmax_out_mask` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 846 | `layer11.block4.v_mask_sf` | `ctpt_v_mask` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 847 | `layer11.block4.softmax_v_mask_sf` | `ctpt_softmax_v_mask` | `encoding` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 848 | `layer11.block4.ln_mean_inv_d_sf` | `ctpt_inv_d_attn_mean` | `scalar` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 849 | `layer11.block4.ln_var_inv_d_sf` | `ctpt_inv_d_attn_var` | `scalar` | 2 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 850 | `layer11.block4.wo_sf` | `ctpt_wo` | `encoding` | 4 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 851 | `layer11.block4.softmax_out_mask_rescale_sf` | `ctct_softmax_out_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 852 | `layer11.block4.v_mask_rescale_sf` | `ctct_v_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 853 | `layer11.block4.softmax_v_matmul_rescale_sf` | `ctct_softmax_v_matmul_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 854 | `layer11.block4.softmax_v_mask_rescale_sf` | `ctct_softmax_v_mask_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 855 | `layer11.block4.wo_rescale_sf` | `ctct_wo_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 856 | `layer11.block4.ln_mean_rescale_sf` | `ctct_attn_mean_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 857 | `layer11.block4.ln_square_rescale_sf` | `ctct_attn_square_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 858 | `layer11.block4.ln_var_rescale_sf` | `ctct_attn_var_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 16384 | 22 |  |
| 859 | `layer11.block4.output_truncation_k` | `block4_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 16384 |  |  |
| 860 | `layer11.block5.inv_std_fresh_sf` | `ctpt_attn_inv_std` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 861 | `layer11.block5.x_centered_fresh_sf` | `ctpt_attn_x_centered` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |
| 862 | `layer11.block5.gamma_sf` | `ctpt_gamma_attn` | `encoding` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 863 | `layer11.block5.wffn1_sf` | `ctpt_wffn1` | `encoding` | 4 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 864 | `layer11.block5.gelu_coeff_sf` | `ctpt_gelu_coeff` | `encoding` | 2 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 865 | `layer11.block5.normalize_rescale_sf` | `ctct_attn_normalize_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 866 | `layer11.block5.gamma_rescale_sf` | `ctct_gamma_attn_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 867 | `layer11.block5.wffn1_rescale_sf` | `ctct_wffn1_rescale` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 868 | `layer11.block5.gelu_power_rescale_sf_0` | `ctct_gelu_x2` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 869 | `layer11.block5.gelu_power_rescale_sf_1` | `ctct_gelu_x3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 870 | `layer11.block5.gelu_power_rescale_sf_2` | `ctct_gelu_x4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this power-rescale slot |
| 871 | `layer11.block5.gelu_coeff_mul_rescale_sf_0` | `ctct_gelu_b_x` | `rescale` | 3 | `scaling_factor` | 22 | True | 8192 | 22 |  |
| 872 | `layer11.block5.gelu_coeff_mul_rescale_sf_1` | `ctct_gelu_c_x2` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 873 | `layer11.block5.gelu_coeff_mul_rescale_sf_2` | `ctct_gelu_d_x3` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 874 | `layer11.block5.gelu_coeff_mul_rescale_sf_3` | `ctct_gelu_e_x4` | `rescale` | 3 | `scaling_factor` |  | False | 8192 | 22 | GELU degree 1 does not use this coefficient-rescale slot |
| 875 | `layer11.block5.output_truncation_k` | `block5_output_truncation` | `truncation` | 3 | `truncation_k` | 13 | True | 8192 |  |  |
| 876 | `first_input.layer0` | `first_input_fresh` | `fresh` | 4 | `scaling_factor` | 30 | True | 8192 | 30 |  |