# BLB Stage-2 Rotation 派生噪声点

Rotation 不是独立 RL action slot，不计入 action vector 长度。它通常紧跟 fresh 或 rescale 点，scaling factor 继承绑定源。

| user group | current code flag | inherits scale from | 语义 |
|---|---|---|---|
| `B1.rot1` | `rotation_after_gelu_out_fresh` | `gelu_out_fresh` | GELU fresh 后的 rotation。 |
| `B1.rot2` | `rotation_after_wffn2_rescale_a` | `wffn2_result_rescale` | Wffn2 rescale 后的第一次 rotation。 |
| `B1.rot3` | `rotation_after_wffn2_rescale_b` | `wffn2_result_rescale` | Wffn2 rescale 后的第二次连续 rotation。 |
| `B1.rot4` | `rotation_after_square_rescale` | `square_result_rescale` | square rescale 后的 rotation。 |
| `B2.rot1` | `rotation_after_gamma_rescale` | `gamma_result_rescale` | gamma/y 乘法后的 rescale 后 rotation。 |
| `B2.rot2_group` | `rotation_after_wq_rescale` | `wq_result_rescale` | WqX 后 rescale 后 rotation。 |
| `B2.rot2_group` | `rotation_after_wk_rescale` | `wk_result_rescale` | WkX 后 rescale 后 rotation。 |
| `B2.rot2_group` | `rotation_after_wv_rescale` | `wv_result_rescale` | WvX 后 rescale 后 rotation。 |
| `B2.rot3_group` | `rotation_after_q_mask1_rescale` | `q_mask1_result_rescale` | Q 第一个 mask rescale 后 rotation。 |
| `B2.rot3_group` | `rotation_after_kt_mask1_rescale` | `kt_mask1_result_rescale` | K/KT 第一个 mask rescale 后 rotation。 |
| `B2.rot4_group` | `rotation_after_q_mask2_rescale` | `q_mask2_result_rescale` | Q 第二个 mask rescale 后 rotation。 |
| `B2.rot4_group` | `rotation_after_kt_mask2_rescale` | `kt_mask2_result_rescale` | K/KT 第二个 mask rescale 后 rotation。 |
| `B2.rot5` | `rotation_after_qkt_matmul_rescale` | `qkt_matmul_result_rescale` | QK^T matmul rescale 后 rotation。 |
| `B4.rot1` | `rotation_after_softmax_out_mask_rescale` | `softmax_out_mask_result_rescale` | Softmax mask 后 rescale 后 rotation。 |
| `B4.rot2` | `rotation_after_v_mask_rescale` | `v_mask_result_rescale` | V mask 后 rescale 后 rotation。 |
| `B4.rot3` | `rotation_after_softmax_v_matmul_rescale` | `softmax_v_matmul_result_rescale` | Softmax*V matmul rescale 后 rotation。 |
| `B4.rot4` | `rotation_after_softmax_v_mask_rescale` | `softmax_v_mask_result_rescale` | Softmax*V mask rescale 后 rotation。 |
| `B4.rot5` | `rotation_after_wo_rescale` | `wo_result_rescale` | Wo rescale 后 rotation。 |
| `B4.rot6` | `rotation_after_ln_square_rescale` | `ln_square_result_rescale` | LayerNorm square rescale 后 rotation。 |
| `B5.rot1` | `rotation_after_gamma_rescale` | `gamma_result_rescale` | gamma/y 乘法后 rescale 后 rotation。 |
| `B5.rot2` | `rotation_after_wffn1_rescale` | `wffn1_result_rescale` | Wffn1*X 后 rescale 后 rotation。 |
