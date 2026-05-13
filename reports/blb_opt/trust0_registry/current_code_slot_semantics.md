# BLB Stage-2 当前代码槽位语义

说明：RL 输出 action index，不直接输出 scaling factor，也不决定操作是否存在；mask/curriculum 只能限制某个槽位允许的 index。

## Block 1

| field | kind | semantic_type | 用户语义 |
|---|---|---|---|
| `gelu_out_sf` | `F` | `fresh` | GELU 输出结果上的 fresh 噪声 scaling factor。 |
| `wffn2_sf` | `W` | `weight_encode` | Wffn2 权重 encode 噪声 scaling factor。 |
| `mean_inv_d_sf` | `S` | `scalar_encode` | LayerNorm mean 中 1/D 标量 encode 噪声 scaling factor。 |
| `var_inv_d_sf` | `S` | `scalar_encode` | LayerNorm variance 中 1/D 标量 encode 噪声 scaling factor。 |
| `wffn2_rescale_sf` | `R` | `rescale` | GELU_out * Wffn2 乘法结果后的 rescale scaling factor。 |
| `mean_rescale_sf` | `R` | `rescale` | mean 计算相关乘法结果后的 rescale scaling factor。 |
| `square_rescale_sf` | `R` | `rescale` | (X - mean)^2 平方操作结果后的 rescale scaling factor。 |
| `var_rescale_sf` | `R` | `rescale` | variance 计算结果后的 rescale scaling factor。 |
| `output_truncation_k` | `K` | `truncation` | Block 1 末尾 CKKS/MPC 转换模拟的 truncation bit。 |

## Block 2

| field | kind | semantic_type | 用户语义 |
|---|---|---|---|
| `inv_std_fresh_sf` | `F` | `fresh` | LayerNorm 中 1/std 结果上的 fresh 噪声 scaling factor。 |
| `x_centered_fresh_sf` | `F` | `fresh` | LayerNorm 中 X - mean 结果上的 fresh 噪声 scaling factor。 |
| `gamma_sf` | `M` | `mask_encode` | LayerNorm gamma 参数 encode scaling factor。 |
| `wq_sf` | `W` | `weight_encode` | Query 投影权重 Wq encode scaling factor。 |
| `wk_sf` | `W` | `weight_encode` | Key 投影权重 Wk encode scaling factor。 |
| `wv_sf` | `W` | `weight_encode` | Value 投影权重 Wv encode scaling factor。 |
| `kt_mask1_sf` | `M` | `mask_encode` | K/KT BSGS 第一个 mask 矩阵 encode scaling factor。 |
| `kt_mask2_sf` | `M` | `mask_encode` | K/KT BSGS 第二个 mask 矩阵 encode scaling factor。 |
| `q_mask1_sf` | `M` | `mask_encode` | Q BSGS 第一个 mask 矩阵 encode scaling factor。 |
| `q_mask2_sf` | `M` | `mask_encode` | Q BSGS 第二个 mask 矩阵 encode scaling factor。 |
| `qkt_merge_mask_sf` | `M` | `mask_encode` | QK^T 后 merge mask 矩阵 encode scaling factor。 |
| `normalize_rescale_sf` | `R` | `rescale` | (X-mean) * (1/std) normalize 结果后的 rescale scaling factor。 |
| `gamma_rescale_sf` | `R` | `rescale` | normalize 结果乘 gamma 后的 rescale scaling factor。 |
| `wk_rescale_sf` | `R` | `rescale` | X * Wk 得到 K 后的 rescale scaling factor。 |
| `wq_rescale_sf` | `R` | `rescale` | X * Wq 得到 Q 后的 rescale scaling factor。 |
| `wv_rescale_sf` | `R` | `rescale` | X * Wv 得到 V 后的 rescale scaling factor。 |
| `kt_mask1_rescale_sf` | `R` | `rescale` | K/KT 乘第一个 mask 后的 rescale scaling factor。 |
| `kt_mask2_rescale_sf` | `R` | `rescale` | K/KT 乘第二个 mask 后的 rescale scaling factor。 |
| `q_mask1_rescale_sf` | `R` | `rescale` | Q 乘第一个 mask 后的 rescale scaling factor。 |
| `q_mask2_rescale_sf` | `R` | `rescale` | Q 乘第二个 mask 后的 rescale scaling factor。 |
| `qkt_matmul_rescale_sf` | `R` | `rescale` | Q 和 K/KT 相乘得到 QK^T 后的 rescale scaling factor。 |
| `qkt_merge_mask_rescale_sf` | `R` | `rescale` | QK^T 乘 merge mask 后的 rescale scaling factor。 |
| `output_truncation_k` | `K` | `truncation` | Block 2 末尾 CKKS/MPC 转换模拟的 truncation bit。 |

## Block 3

| field | kind | semantic_type | 用户语义 |
|---|---|---|---|
| `x_fresh_sf` | `F` | `fresh` | Softmax 输入 x 的 fresh 噪声 scaling factor。 |
| `inv_2n_sf` | `S` | `scalar_encode` | Softmax exp 近似中 1/2^n 标量 encode scaling factor。 |
| `x_inv_2n_rescale_sf` | `R` | `rescale` | x * 1/2^n 后的 rescale scaling factor。 |
| `square_rescale_sf_0` | `R` | `rescale` | Softmax 指数近似第 1 次平方后的 rescale scaling factor。 |
| `square_rescale_sf_1` | `R` | `rescale` | Softmax 指数近似第 2 次平方后的 rescale scaling factor，degree 条件启用。 |
| `square_rescale_sf_2` | `R` | `rescale` | Softmax 指数近似第 3 次平方后的 rescale scaling factor，degree 条件启用。 |
| `square_rescale_sf_3` | `R` | `rescale` | Softmax 指数近似第 4 次平方后的 rescale scaling factor，degree 条件启用。 |
| `output_truncation_k` | `K` | `truncation` | Block 3 末尾 CKKS/MPC 转换模拟的 truncation bit。 |

## Block 4

| field | kind | semantic_type | 用户语义 |
|---|---|---|---|
| `softmax_out_fresh_sf` | `F` | `fresh` | Softmax 输出结果 fresh 噪声 scaling factor。 |
| `v_fresh_sf` | `F` | `fresh` | V 矩阵 fresh 噪声 scaling factor。 |
| `softmax_out_mask_sf` | `M` | `mask_encode` | Softmax 输出乘全 1 mask 的 mask encode scaling factor。 |
| `v_mask_sf` | `M` | `mask_encode` | V 乘全 1 mask 的 mask encode scaling factor。 |
| `softmax_v_mask_sf` | `M` | `mask_encode` | Softmax*V 之后再乘 mask 的 mask encode scaling factor。 |
| `ln_mean_inv_d_sf` | `S` | `scalar_encode` | attention LayerNorm mean 中 1/D 标量 encode scaling factor。 |
| `ln_var_inv_d_sf` | `S` | `scalar_encode` | attention LayerNorm variance 中 1/D 标量 encode scaling factor。 |
| `wo_sf` | `W` | `weight_encode` | Wo 权重 encode scaling factor。 |
| `softmax_out_mask_rescale_sf` | `R` | `rescale` | Softmax output mask 乘法后的 rescale scaling factor。 |
| `v_mask_rescale_sf` | `R` | `rescale` | V mask 乘法后的 rescale scaling factor。 |
| `softmax_v_matmul_rescale_sf` | `R` | `rescale` | Softmax * V matmul 后的 rescale scaling factor。 |
| `softmax_v_mask_rescale_sf` | `R` | `rescale` | Softmax*V mask 乘法后的 rescale scaling factor。 |
| `wo_rescale_sf` | `R` | `rescale` | Wo 乘法后的 rescale scaling factor。 |
| `ln_mean_rescale_sf` | `R` | `rescale` | LayerNorm mean 计算后的 rescale scaling factor。 |
| `ln_square_rescale_sf` | `R` | `rescale` | LayerNorm 中 (X - mean)^2 后的 rescale scaling factor。 |
| `ln_var_rescale_sf` | `R` | `rescale` | LayerNorm variance 计算后的 rescale scaling factor。 |
| `output_truncation_k` | `K` | `truncation` | Block 4 末尾 CKKS/MPC 转换模拟的 truncation bit。 |

## Block 5

| field | kind | semantic_type | 用户语义 |
|---|---|---|---|
| `inv_std_fresh_sf` | `F` | `fresh` | LayerNorm tail 中 1/std fresh 噪声 scaling factor。 |
| `x_centered_fresh_sf` | `F` | `fresh` | LayerNorm tail 中 X_centered fresh 噪声 scaling factor。 |
| `gamma_sf` | `M` | `mask_encode` | LayerNorm gamma / 逐元素参数 encode scaling factor。 |
| `wffn1_sf` | `W` | `weight_encode` | FFN 第一层 Wffn1 权重 encode scaling factor。 |
| `gelu_coeff_sf` | `M` | `mask_encode` | GELU 多项式系数共享 encode scaling factor。 |
| `normalize_rescale_sf` | `R` | `rescale` | normalize 后的 rescale scaling factor。 |
| `gamma_rescale_sf` | `R` | `rescale` | gamma 乘法后的 rescale scaling factor。 |
| `wffn1_rescale_sf` | `R` | `rescale` | Wffn1 * X 后的 rescale scaling factor。 |
| `gelu_power_rescale_sf_0` | `R` | `rescale` | GELU x^2 幂次计算后的 rescale scaling factor。 |
| `gelu_power_rescale_sf_1` | `R` | `rescale` | GELU x^3 幂次计算后的 rescale scaling factor，degree 条件启用。 |
| `gelu_power_rescale_sf_2` | `R` | `rescale` | GELU x^4 幂次计算后的 rescale scaling factor，degree 条件启用。 |
| `gelu_coeff_mul_rescale_sf_0` | `R` | `rescale` | 第 0 个 GELU 系数乘法结果后的 rescale scaling factor。 |
| `gelu_coeff_mul_rescale_sf_1` | `R` | `rescale` | 第 1 个 GELU 系数乘法结果后的 rescale scaling factor。 |
| `gelu_coeff_mul_rescale_sf_2` | `R` | `rescale` | 第 2 个 GELU 系数乘法结果后的 rescale scaling factor。 |
| `gelu_coeff_mul_rescale_sf_3` | `R` | `rescale` | 第 3 个 GELU 系数乘法结果后的 rescale scaling factor。 |
| `output_truncation_k` | `K` | `truncation` | Block 5 末尾 CKKS/MPC 转换模拟的 truncation bit。 |

## first-input

完整 action vector 尾部有一个 `first_input_sf`，用于第 0 层 embedding 输入进入 BLB 模拟时的 fresh 噪声入口。

## S-kind 注意

`kind=S` 的语义是 scalar encode，例如 `1/D`、`1/2^n`、LayerNorm 标量。当前代码描述中的 noise lookup distribution 可能仍需从实际 `NoisePoint` 验证；未知处在 JSON 中标记为 `unknown`。
