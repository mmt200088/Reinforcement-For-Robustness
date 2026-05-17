# BLB invalid-block diagnosis

- action_config: `Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.005_s2t0.005_s2st0.005/stage2_noise/progress/diagnostics/best_action_vec.json`
- profile: `mrpc` · num_layers: `12`

## Summary

- configs evaluated: **59**
- valid: **43**
- invalid: **16**
- any_invalid: **True**
- total_bits_sum: **14638**
- total_fusion_count: **13**
- avg_k in action: **10.102**

## Invalid blocks

| (L, B) | graph_key | total_bits | fusion | reason |
|:-------|:----------|-----------:|-------:|:-------|
| `L00-B5` | `block5_n4` | 263 | 0 | q_head_bits=60; q_bits=[27, 42, 23, 51]; q_tail_bits=60; total_bits=263; R=4 |
| `L01-B3` | `block3_exp_n4` | 244 | 1 | q_head_bits=60; q_bits=[37, 60, 27]; q_tail_bits=60; total_bits=244; R=3 |
| `L01-B5` | `block5_n4` | 271 | 0 | q_head_bits=60; q_bits=[27, 42, 31, 51]; q_tail_bits=60; total_bits=271; R=4 |
| `L02-B3` | `block3_exp_n4` | 252 | 1 | q_head_bits=60; q_bits=[45, 60, 27]; q_tail_bits=60; total_bits=252; R=3 |
| `L02-B5` | `block5_n4` | 255 | 0 | q_head_bits=60; q_bits=[27, 34, 27, 47]; q_tail_bits=60; total_bits=255; R=4 |
| `L03-B5` | `block5_n4` | 257 | 0 | q_head_bits=60; q_bits=[27, 36, 31, 43]; q_tail_bits=60; total_bits=257; R=4 |
| `L04-B5` | `block5_n4` | 253 | 0 | q_head_bits=60; q_bits=[27, 36, 27, 43]; q_tail_bits=60; total_bits=253; R=4 |
| `L06-B3` | `block3_exp_n4` | 252 | 1 | q_head_bits=60; q_bits=[45, 60, 27]; q_tail_bits=60; total_bits=252; R=3 |
| `L07-B3` | `block3_exp_n4` | 252 | 1 | q_head_bits=60; q_bits=[45, 60, 27]; q_tail_bits=60; total_bits=252; R=3 |
| `L07-B5` | `block5_n4` | 259 | 0 | q_head_bits=60; q_bits=[27, 38, 23, 51]; q_tail_bits=60; total_bits=259; R=4 |
| `L08-B3` | `block3_exp_n4` | 252 | 1 | q_head_bits=60; q_bits=[45, 60, 27]; q_tail_bits=60; total_bits=252; R=3 |
| `L08-B5` | `block5_n4` | 259 | 0 | q_head_bits=60; q_bits=[27, 38, 23, 51]; q_tail_bits=60; total_bits=259; R=4 |
| `L09-B5` | `block5_n4` | 259 | 0 | q_head_bits=60; q_bits=[27, 38, 23, 51]; q_tail_bits=60; total_bits=259; R=4 |
| `L10-B5` | `block5_n4` | 255 | 0 | q_head_bits=60; q_bits=[27, 38, 27, 43]; q_tail_bits=60; total_bits=255; R=4 |
| `L11-B3` | `block3_exp_n4` | 260 | 1 | q_head_bits=60; q_bits=[53, 60, 27]; q_tail_bits=60; total_bits=260; R=3 |
| `L11-B5` | `block5_n4` | 263 | 0 | q_head_bits=60; q_bits=[27, 46, 27, 43]; q_tail_bits=60; total_bits=263; R=4 |

## Slot configs of invalid blocks

### L00-B5 (`block5_n4`)

- invalid reason: `q_head_bits=60; q_bits=[27, 42, 23, 51]; q_tail_bits=60; total_bits=263; R=4`
- slots:
  - `F.inv_std_fresh_sf scaling_factor=24`
  - `F.x_centered_fresh_sf scaling_factor=29`
  - `M.gamma_sf scaling_factor=20`
  - `W.wffn1_sf scaling_factor=18`
  - `M.gelu_coeff_sf scaling_factor=20`
  - `R.normalize_rescale_sf scaling_factor=off`
  - `R.gamma_rescale_sf scaling_factor=off`
  - `R.wffn1_rescale_sf scaling_factor=27`
  - `R.gelu_power_rescale_sf_0 scaling_factor=31`
  - `K.output_truncation_k truncation_bits=9`

### L01-B3 (`block3_exp_n4`)

- invalid reason: `q_head_bits=60; q_bits=[37, 60, 27]; q_tail_bits=60; total_bits=244; R=3`
- slots:
  - `F.x_fresh_sf scaling_factor=22`
  - `S.inv_2n_sf scaling_factor=11`
  - `R.square_rescale_sf_0 scaling_factor=29`
  - `R.square_rescale_sf_1 scaling_factor=off`
  - `R.square_rescale_sf_2 scaling_factor=29`
  - `R.square_rescale_sf_3 scaling_factor=off`
  - `K.output_truncation_k truncation_bits=12`

### L01-B5 (`block5_n4`)

- invalid reason: `q_head_bits=60; q_bits=[27, 42, 31, 51]; q_tail_bits=60; total_bits=271; R=4`
- slots:
  - `F.inv_std_fresh_sf scaling_factor=26`
  - `F.x_centered_fresh_sf scaling_factor=29`
  - `M.gamma_sf scaling_factor=20`
  - `W.wffn1_sf scaling_factor=22`
  - `M.gelu_coeff_sf scaling_factor=20`
  - `R.normalize_rescale_sf scaling_factor=off`
  - `R.gamma_rescale_sf scaling_factor=22`
  - `R.wffn1_rescale_sf scaling_factor=31`
  - `R.gelu_power_rescale_sf_0 scaling_factor=31`
  - `K.output_truncation_k truncation_bits=9`

### L02-B3 (`block3_exp_n4`)

- invalid reason: `q_head_bits=60; q_bits=[45, 60, 27]; q_tail_bits=60; total_bits=252; R=3`
- slots:
  - `F.x_fresh_sf scaling_factor=22`
  - `S.inv_2n_sf scaling_factor=15`
  - `R.square_rescale_sf_0 scaling_factor=29`
  - `R.square_rescale_sf_1 scaling_factor=off`
  - `R.square_rescale_sf_2 scaling_factor=29`
  - `R.square_rescale_sf_3 scaling_factor=off`
  - `K.output_truncation_k truncation_bits=12`

### L02-B5 (`block5_n4`)

- invalid reason: `q_head_bits=60; q_bits=[27, 34, 27, 47]; q_tail_bits=60; total_bits=255; R=4`
- slots:
  - `F.inv_std_fresh_sf scaling_factor=28`
  - `F.x_centered_fresh_sf scaling_factor=29`
  - `M.gamma_sf scaling_factor=18`
  - `W.wffn1_sf scaling_factor=14`
  - `M.gelu_coeff_sf scaling_factor=16`
  - `R.normalize_rescale_sf scaling_factor=off`
  - `R.gamma_rescale_sf scaling_factor=22`
  - `R.wffn1_rescale_sf scaling_factor=29`
  - `R.gelu_power_rescale_sf_0 scaling_factor=off`
  - `K.output_truncation_k truncation_bits=12`

### L03-B5 (`block5_n4`)

- invalid reason: `q_head_bits=60; q_bits=[27, 36, 31, 43]; q_tail_bits=60; total_bits=257; R=4`
- slots:
  - `F.inv_std_fresh_sf scaling_factor=24`
  - `F.x_centered_fresh_sf scaling_factor=29`
  - `M.gamma_sf scaling_factor=20`
  - `W.wffn1_sf scaling_factor=14`
  - `M.gelu_coeff_sf scaling_factor=20`
  - `R.normalize_rescale_sf scaling_factor=off`
  - `R.gamma_rescale_sf scaling_factor=22`
  - `R.wffn1_rescale_sf scaling_factor=29`
  - `R.gelu_power_rescale_sf_0 scaling_factor=27`
  - `K.output_truncation_k truncation_bits=8`

### L04-B5 (`block5_n4`)

- invalid reason: `q_head_bits=60; q_bits=[27, 36, 27, 43]; q_tail_bits=60; total_bits=253; R=4`
- slots:
  - `F.inv_std_fresh_sf scaling_factor=24`
  - `F.x_centered_fresh_sf scaling_factor=29`
  - `M.gamma_sf scaling_factor=16`
  - `W.wffn1_sf scaling_factor=16`
  - `M.gelu_coeff_sf scaling_factor=20`
  - `R.normalize_rescale_sf scaling_factor=off`
  - `R.gamma_rescale_sf scaling_factor=22`
  - `R.wffn1_rescale_sf scaling_factor=27`
  - `R.gelu_power_rescale_sf_0 scaling_factor=27`
  - `K.output_truncation_k truncation_bits=12`

### L06-B3 (`block3_exp_n4`)

- invalid reason: `q_head_bits=60; q_bits=[45, 60, 27]; q_tail_bits=60; total_bits=252; R=3`
- slots:
  - `F.x_fresh_sf scaling_factor=22`
  - `S.inv_2n_sf scaling_factor=15`
  - `R.square_rescale_sf_0 scaling_factor=29`
  - `R.square_rescale_sf_1 scaling_factor=off`
  - `R.square_rescale_sf_2 scaling_factor=29`
  - `R.square_rescale_sf_3 scaling_factor=off`
  - `K.output_truncation_k truncation_bits=12`

### L07-B3 (`block3_exp_n4`)

- invalid reason: `q_head_bits=60; q_bits=[45, 60, 27]; q_tail_bits=60; total_bits=252; R=3`
- slots:
  - `F.x_fresh_sf scaling_factor=22`
  - `S.inv_2n_sf scaling_factor=15`
  - `R.square_rescale_sf_0 scaling_factor=29`
  - `R.square_rescale_sf_1 scaling_factor=off`
  - `R.square_rescale_sf_2 scaling_factor=29`
  - `R.square_rescale_sf_3 scaling_factor=off`
  - `K.output_truncation_k truncation_bits=12`

### L07-B5 (`block5_n4`)

- invalid reason: `q_head_bits=60; q_bits=[27, 38, 23, 51]; q_tail_bits=60; total_bits=259; R=4`
- slots:
  - `F.inv_std_fresh_sf scaling_factor=24`
  - `F.x_centered_fresh_sf scaling_factor=29`
  - `M.gamma_sf scaling_factor=16`
  - `W.wffn1_sf scaling_factor=18`
  - `M.gelu_coeff_sf scaling_factor=20`
  - `R.normalize_rescale_sf scaling_factor=off`
  - `R.gamma_rescale_sf scaling_factor=off`
  - `R.wffn1_rescale_sf scaling_factor=27`
  - `R.gelu_power_rescale_sf_0 scaling_factor=off`
  - `K.output_truncation_k truncation_bits=12`

### L08-B3 (`block3_exp_n4`)

- invalid reason: `q_head_bits=60; q_bits=[45, 60, 27]; q_tail_bits=60; total_bits=252; R=3`
- slots:
  - `F.x_fresh_sf scaling_factor=22`
  - `S.inv_2n_sf scaling_factor=15`
  - `R.square_rescale_sf_0 scaling_factor=29`
  - `R.square_rescale_sf_1 scaling_factor=off`
  - `R.square_rescale_sf_2 scaling_factor=29`
  - `R.square_rescale_sf_3 scaling_factor=off`
  - `K.output_truncation_k truncation_bits=12`

### L08-B5 (`block5_n4`)

- invalid reason: `q_head_bits=60; q_bits=[27, 38, 23, 51]; q_tail_bits=60; total_bits=259; R=4`
- slots:
  - `F.inv_std_fresh_sf scaling_factor=28`
  - `F.x_centered_fresh_sf scaling_factor=29`
  - `M.gamma_sf scaling_factor=20`
  - `W.wffn1_sf scaling_factor=14`
  - `M.gelu_coeff_sf scaling_factor=20`
  - `R.normalize_rescale_sf scaling_factor=off`
  - `R.gamma_rescale_sf scaling_factor=off`
  - `R.wffn1_rescale_sf scaling_factor=27`
  - `R.gelu_power_rescale_sf_0 scaling_factor=31`
  - `K.output_truncation_k truncation_bits=12`

### L09-B5 (`block5_n4`)

- invalid reason: `q_head_bits=60; q_bits=[27, 38, 23, 51]; q_tail_bits=60; total_bits=259; R=4`
- slots:
  - `F.inv_std_fresh_sf scaling_factor=28`
  - `F.x_centered_fresh_sf scaling_factor=29`
  - `M.gamma_sf scaling_factor=20`
  - `W.wffn1_sf scaling_factor=14`
  - `M.gelu_coeff_sf scaling_factor=20`
  - `R.normalize_rescale_sf scaling_factor=off`
  - `R.gamma_rescale_sf scaling_factor=off`
  - `R.wffn1_rescale_sf scaling_factor=27`
  - `R.gelu_power_rescale_sf_0 scaling_factor=31`
  - `K.output_truncation_k truncation_bits=12`

### L10-B5 (`block5_n4`)

- invalid reason: `q_head_bits=60; q_bits=[27, 38, 27, 43]; q_tail_bits=60; total_bits=255; R=4`
- slots:
  - `F.inv_std_fresh_sf scaling_factor=26`
  - `F.x_centered_fresh_sf scaling_factor=29`
  - `M.gamma_sf scaling_factor=16`
  - `W.wffn1_sf scaling_factor=18`
  - `M.gelu_coeff_sf scaling_factor=20`
  - `R.normalize_rescale_sf scaling_factor=off`
  - `R.gamma_rescale_sf scaling_factor=off`
  - `R.wffn1_rescale_sf scaling_factor=27`
  - `R.gelu_power_rescale_sf_0 scaling_factor=27`
  - `K.output_truncation_k truncation_bits=12`

### L11-B3 (`block3_exp_n4`)

- invalid reason: `q_head_bits=60; q_bits=[53, 60, 27]; q_tail_bits=60; total_bits=260; R=3`
- slots:
  - `F.x_fresh_sf scaling_factor=26`
  - `S.inv_2n_sf scaling_factor=15`
  - `R.square_rescale_sf_0 scaling_factor=29`
  - `R.square_rescale_sf_1 scaling_factor=off`
  - `R.square_rescale_sf_2 scaling_factor=29`
  - `R.square_rescale_sf_3 scaling_factor=off`
  - `K.output_truncation_k truncation_bits=12`

### L11-B5 (`block5_n4`)

- invalid reason: `q_head_bits=60; q_bits=[27, 46, 27, 43]; q_tail_bits=60; total_bits=263; R=4`
- slots:
  - `F.inv_std_fresh_sf scaling_factor=26`
  - `F.x_centered_fresh_sf scaling_factor=29`
  - `M.gamma_sf scaling_factor=20`
  - `W.wffn1_sf scaling_factor=22`
  - `M.gelu_coeff_sf scaling_factor=20`
  - `R.normalize_rescale_sf scaling_factor=off`
  - `R.gamma_rescale_sf scaling_factor=20`
  - `R.wffn1_rescale_sf scaling_factor=27`
  - `R.gelu_power_rescale_sf_0 scaling_factor=27`
  - `K.output_truncation_k truncation_bits=12`
