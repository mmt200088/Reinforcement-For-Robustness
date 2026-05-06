# BLB Action Final Evaluation Report

- dataset: `mrpc`
- split: `validation_full`
- selected_source: `blb_action(stage1=json)`
- repeat_n: `50`
- rescale_optimizer: `in_process`
- rescale_optimizer_root: `/var/tmp/root-home/Reinforcement-For-Robustness/Rescale_optimizer`
- json: `/var/tmp/root-home/Reinforcement-For-Robustness/Paean/outputs/mrpc/rl/mrpc_blb_baseline_truncation_sweep/final_eval/blb_action_final_eval_results_mrpc.json`

## Baseline

- clean baseline loss: `0.380749`
- clean baseline Acc.: `0.879902`
- clean baseline F1: `0.877442`

## Group Comparison

| group | truncation k | effective K positions | loss mean | loss std | Acc. mean | Acc. std | F1 mean | F1 std | time mean ms | total bits | fusion | model cfg verified |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `ActionGrid_truncation8` | 8 | 59 | 0.373200 | 0.001205 | 0.876912 | 0.001978 | 0.875319 | 0.001988 | 279.313 | 14989 | 0 | True |
| `ActionGrid_truncation9` | 9 | 59 | 0.373608 | 0.001248 | 0.878333 | 0.002123 | 0.876583 | 0.002171 | 241.840 | 14989 | 0 | True |
| `ActionGrid_truncation10` | 10 | 59 | 0.374350 | 0.001489 | 0.879020 | 0.002179 | 0.877225 | 0.002145 | 233.603 | 14989 | 0 | True |
| `ActionGrid_truncation11` | 11 | 59 | 0.374602 | 0.001369 | 0.880049 | 0.002045 | 0.878199 | 0.001980 | 233.651 | 14989 | 0 | True |
| `ActionGrid_truncation12` | 12 | 59 | 0.374796 | 0.001218 | 0.880441 | 0.001572 | 0.878572 | 0.001528 | 233.507 | 14989 | 0 | True |
| `ActionGrid_truncation13` | 13 | 59 | 0.374784 | 0.001432 | 0.880637 | 0.001850 | 0.878752 | 0.001802 | 233.567 | 14989 | 0 | True |

## Configuration Details

### ActionGrid_truncation8

- action overrides: `{'truncation': 8}`
- base action: BLB RL baseline action: model-side non-truncation fields use highest selectable action-space SF; Rescale_optimizer mode=baseline.
- first_input_sf: `30`
- truncation summary: `8`; effective positions = `59`; skipped = `[{'block': 'block1', 'layer': 0}]`
- model cfg verified before forward: `True`
- handler active layers match expected: `True`
- handler cfg object identity match: `True`
- rescale optimizer: `{'invoker_kind': 'in_process', 'root': '/var/tmp/root-home/Reinforcement-For-Robustness/Rescale_optimizer', 'mode': 'baseline', 'request_count': 60, 'valid_count': 60, 'invalid_count': 0, 't_new_sources': ['unknown']}`

Non-truncation unique scaling factors:

```json
{
  "block1": {
    "gelu_out_fresh": [
      30
    ],
    "mean_inv_d_encode": [
      22
    ],
    "mean_result_rescale": [
      22
    ],
    "square_result_rescale": [
      22
    ],
    "var_inv_d_encode": [
      22
    ],
    "var_result_rescale": [
      22
    ],
    "wffn2_encode": [
      22
    ],
    "wffn2_result_rescale": [
      22
    ]
  },
  "block2": {
    "gamma_encode": [
      22
    ],
    "gamma_result_rescale": [
      22
    ],
    "inv_std_fresh": [
      30
    ],
    "kt_mask1_encode": [
      22
    ],
    "kt_mask1_result_rescale": [
      22
    ],
    "kt_mask2_encode": [
      22
    ],
    "kt_mask2_result_rescale": [
      22
    ],
    "normalize_result_rescale": [
      22
    ],
    "q_mask1_encode": [
      22
    ],
    "q_mask1_result_rescale": [
      22
    ],
    "q_mask2_encode": [
      22
    ],
    "q_mask2_result_rescale": [
      22
    ],
    "qkt_matmul_result_rescale": [
      22
    ],
    "qkt_merge_mask_encode": [
      22
    ],
    "qkt_merge_mask_result_rescale": [
      22
    ],
    "wk_encode": [
      22
    ],
    "wk_result_rescale": [
      22
    ],
    "wq_encode": [
      22
    ],
    "wq_result_rescale": [
      22
    ],
    "wv_encode": [
      22
    ],
    "wv_result_rescale": [
      22
    ],
    "x_centered_fresh": [
      30
    ]
  },
  "block3": {
    "inv_2n_encode": [
      22
    ],
    "square_rescales": [
      22
    ],
    "x_fresh": [
      30
    ],
    "x_inv_2n_result_rescale": [
      22
    ]
  },
  "block4": {
    "ln_mean_inv_d_encode": [
      22
    ],
    "ln_mean_result_rescale": [
      22
    ],
    "ln_square_result_rescale": [
      22
    ],
    "ln_var_inv_d_encode": [
      22
    ],
    "ln_var_result_rescale": [
      22
    ],
    "softmax_out_fresh": [
      30
    ],
    "softmax_out_mask_encode": [
      22
    ],
    "softmax_out_mask_rescale": [
      22
    ],
    "softmax_v_mask_encode": [
      22
    ],
    "softmax_v_mask_rescale": [
      22
    ],
    "softmax_v_matmul_rescale": [
      22
    ],
    "v_fresh": [
      30
    ],
    "v_mask_encode": [
      22
    ],
    "v_mask_rescale": [
      22
    ],
    "wo_encode": [
      22
    ],
    "wo_result_rescale": [
      22
    ]
  },
  "block5": {
    "gamma_encode": [
      22
    ],
    "gamma_result_rescale": [
      22
    ],
    "gelu_coeff_encode": [
      22
    ],
    "gelu_coeff_mul_rescales": [
      22
    ],
    "gelu_power_rescales": [
      22
    ],
    "inv_std_fresh": [
      30
    ],
    "normalize_result_rescale": [
      22
    ],
    "wffn1_encode": [
      22
    ],
    "wffn1_result_rescale": [
      22
    ],
    "x_centered_fresh": [
      30
    ]
  }
}
```

### ActionGrid_truncation9

- action overrides: `{'truncation': 9}`
- base action: BLB RL baseline action: model-side non-truncation fields use highest selectable action-space SF; Rescale_optimizer mode=baseline.
- first_input_sf: `30`
- truncation summary: `9`; effective positions = `59`; skipped = `[{'block': 'block1', 'layer': 0}]`
- model cfg verified before forward: `True`
- handler active layers match expected: `True`
- handler cfg object identity match: `True`
- rescale optimizer: `{'invoker_kind': 'in_process', 'root': '/var/tmp/root-home/Reinforcement-For-Robustness/Rescale_optimizer', 'mode': 'baseline', 'request_count': 60, 'valid_count': 60, 'invalid_count': 0, 't_new_sources': ['unknown']}`

Non-truncation unique scaling factors:

```json
{
  "block1": {
    "gelu_out_fresh": [
      30
    ],
    "mean_inv_d_encode": [
      22
    ],
    "mean_result_rescale": [
      22
    ],
    "square_result_rescale": [
      22
    ],
    "var_inv_d_encode": [
      22
    ],
    "var_result_rescale": [
      22
    ],
    "wffn2_encode": [
      22
    ],
    "wffn2_result_rescale": [
      22
    ]
  },
  "block2": {
    "gamma_encode": [
      22
    ],
    "gamma_result_rescale": [
      22
    ],
    "inv_std_fresh": [
      30
    ],
    "kt_mask1_encode": [
      22
    ],
    "kt_mask1_result_rescale": [
      22
    ],
    "kt_mask2_encode": [
      22
    ],
    "kt_mask2_result_rescale": [
      22
    ],
    "normalize_result_rescale": [
      22
    ],
    "q_mask1_encode": [
      22
    ],
    "q_mask1_result_rescale": [
      22
    ],
    "q_mask2_encode": [
      22
    ],
    "q_mask2_result_rescale": [
      22
    ],
    "qkt_matmul_result_rescale": [
      22
    ],
    "qkt_merge_mask_encode": [
      22
    ],
    "qkt_merge_mask_result_rescale": [
      22
    ],
    "wk_encode": [
      22
    ],
    "wk_result_rescale": [
      22
    ],
    "wq_encode": [
      22
    ],
    "wq_result_rescale": [
      22
    ],
    "wv_encode": [
      22
    ],
    "wv_result_rescale": [
      22
    ],
    "x_centered_fresh": [
      30
    ]
  },
  "block3": {
    "inv_2n_encode": [
      22
    ],
    "square_rescales": [
      22
    ],
    "x_fresh": [
      30
    ],
    "x_inv_2n_result_rescale": [
      22
    ]
  },
  "block4": {
    "ln_mean_inv_d_encode": [
      22
    ],
    "ln_mean_result_rescale": [
      22
    ],
    "ln_square_result_rescale": [
      22
    ],
    "ln_var_inv_d_encode": [
      22
    ],
    "ln_var_result_rescale": [
      22
    ],
    "softmax_out_fresh": [
      30
    ],
    "softmax_out_mask_encode": [
      22
    ],
    "softmax_out_mask_rescale": [
      22
    ],
    "softmax_v_mask_encode": [
      22
    ],
    "softmax_v_mask_rescale": [
      22
    ],
    "softmax_v_matmul_rescale": [
      22
    ],
    "v_fresh": [
      30
    ],
    "v_mask_encode": [
      22
    ],
    "v_mask_rescale": [
      22
    ],
    "wo_encode": [
      22
    ],
    "wo_result_rescale": [
      22
    ]
  },
  "block5": {
    "gamma_encode": [
      22
    ],
    "gamma_result_rescale": [
      22
    ],
    "gelu_coeff_encode": [
      22
    ],
    "gelu_coeff_mul_rescales": [
      22
    ],
    "gelu_power_rescales": [
      22
    ],
    "inv_std_fresh": [
      30
    ],
    "normalize_result_rescale": [
      22
    ],
    "wffn1_encode": [
      22
    ],
    "wffn1_result_rescale": [
      22
    ],
    "x_centered_fresh": [
      30
    ]
  }
}
```

### ActionGrid_truncation10

- action overrides: `{'truncation': 10}`
- base action: BLB RL baseline action: model-side non-truncation fields use highest selectable action-space SF; Rescale_optimizer mode=baseline.
- first_input_sf: `30`
- truncation summary: `10`; effective positions = `59`; skipped = `[{'block': 'block1', 'layer': 0}]`
- model cfg verified before forward: `True`
- handler active layers match expected: `True`
- handler cfg object identity match: `True`
- rescale optimizer: `{'invoker_kind': 'in_process', 'root': '/var/tmp/root-home/Reinforcement-For-Robustness/Rescale_optimizer', 'mode': 'baseline', 'request_count': 60, 'valid_count': 60, 'invalid_count': 0, 't_new_sources': ['unknown']}`

Non-truncation unique scaling factors:

```json
{
  "block1": {
    "gelu_out_fresh": [
      30
    ],
    "mean_inv_d_encode": [
      22
    ],
    "mean_result_rescale": [
      22
    ],
    "square_result_rescale": [
      22
    ],
    "var_inv_d_encode": [
      22
    ],
    "var_result_rescale": [
      22
    ],
    "wffn2_encode": [
      22
    ],
    "wffn2_result_rescale": [
      22
    ]
  },
  "block2": {
    "gamma_encode": [
      22
    ],
    "gamma_result_rescale": [
      22
    ],
    "inv_std_fresh": [
      30
    ],
    "kt_mask1_encode": [
      22
    ],
    "kt_mask1_result_rescale": [
      22
    ],
    "kt_mask2_encode": [
      22
    ],
    "kt_mask2_result_rescale": [
      22
    ],
    "normalize_result_rescale": [
      22
    ],
    "q_mask1_encode": [
      22
    ],
    "q_mask1_result_rescale": [
      22
    ],
    "q_mask2_encode": [
      22
    ],
    "q_mask2_result_rescale": [
      22
    ],
    "qkt_matmul_result_rescale": [
      22
    ],
    "qkt_merge_mask_encode": [
      22
    ],
    "qkt_merge_mask_result_rescale": [
      22
    ],
    "wk_encode": [
      22
    ],
    "wk_result_rescale": [
      22
    ],
    "wq_encode": [
      22
    ],
    "wq_result_rescale": [
      22
    ],
    "wv_encode": [
      22
    ],
    "wv_result_rescale": [
      22
    ],
    "x_centered_fresh": [
      30
    ]
  },
  "block3": {
    "inv_2n_encode": [
      22
    ],
    "square_rescales": [
      22
    ],
    "x_fresh": [
      30
    ],
    "x_inv_2n_result_rescale": [
      22
    ]
  },
  "block4": {
    "ln_mean_inv_d_encode": [
      22
    ],
    "ln_mean_result_rescale": [
      22
    ],
    "ln_square_result_rescale": [
      22
    ],
    "ln_var_inv_d_encode": [
      22
    ],
    "ln_var_result_rescale": [
      22
    ],
    "softmax_out_fresh": [
      30
    ],
    "softmax_out_mask_encode": [
      22
    ],
    "softmax_out_mask_rescale": [
      22
    ],
    "softmax_v_mask_encode": [
      22
    ],
    "softmax_v_mask_rescale": [
      22
    ],
    "softmax_v_matmul_rescale": [
      22
    ],
    "v_fresh": [
      30
    ],
    "v_mask_encode": [
      22
    ],
    "v_mask_rescale": [
      22
    ],
    "wo_encode": [
      22
    ],
    "wo_result_rescale": [
      22
    ]
  },
  "block5": {
    "gamma_encode": [
      22
    ],
    "gamma_result_rescale": [
      22
    ],
    "gelu_coeff_encode": [
      22
    ],
    "gelu_coeff_mul_rescales": [
      22
    ],
    "gelu_power_rescales": [
      22
    ],
    "inv_std_fresh": [
      30
    ],
    "normalize_result_rescale": [
      22
    ],
    "wffn1_encode": [
      22
    ],
    "wffn1_result_rescale": [
      22
    ],
    "x_centered_fresh": [
      30
    ]
  }
}
```

### ActionGrid_truncation11

- action overrides: `{'truncation': 11}`
- base action: BLB RL baseline action: model-side non-truncation fields use highest selectable action-space SF; Rescale_optimizer mode=baseline.
- first_input_sf: `30`
- truncation summary: `11`; effective positions = `59`; skipped = `[{'block': 'block1', 'layer': 0}]`
- model cfg verified before forward: `True`
- handler active layers match expected: `True`
- handler cfg object identity match: `True`
- rescale optimizer: `{'invoker_kind': 'in_process', 'root': '/var/tmp/root-home/Reinforcement-For-Robustness/Rescale_optimizer', 'mode': 'baseline', 'request_count': 60, 'valid_count': 60, 'invalid_count': 0, 't_new_sources': ['unknown']}`

Non-truncation unique scaling factors:

```json
{
  "block1": {
    "gelu_out_fresh": [
      30
    ],
    "mean_inv_d_encode": [
      22
    ],
    "mean_result_rescale": [
      22
    ],
    "square_result_rescale": [
      22
    ],
    "var_inv_d_encode": [
      22
    ],
    "var_result_rescale": [
      22
    ],
    "wffn2_encode": [
      22
    ],
    "wffn2_result_rescale": [
      22
    ]
  },
  "block2": {
    "gamma_encode": [
      22
    ],
    "gamma_result_rescale": [
      22
    ],
    "inv_std_fresh": [
      30
    ],
    "kt_mask1_encode": [
      22
    ],
    "kt_mask1_result_rescale": [
      22
    ],
    "kt_mask2_encode": [
      22
    ],
    "kt_mask2_result_rescale": [
      22
    ],
    "normalize_result_rescale": [
      22
    ],
    "q_mask1_encode": [
      22
    ],
    "q_mask1_result_rescale": [
      22
    ],
    "q_mask2_encode": [
      22
    ],
    "q_mask2_result_rescale": [
      22
    ],
    "qkt_matmul_result_rescale": [
      22
    ],
    "qkt_merge_mask_encode": [
      22
    ],
    "qkt_merge_mask_result_rescale": [
      22
    ],
    "wk_encode": [
      22
    ],
    "wk_result_rescale": [
      22
    ],
    "wq_encode": [
      22
    ],
    "wq_result_rescale": [
      22
    ],
    "wv_encode": [
      22
    ],
    "wv_result_rescale": [
      22
    ],
    "x_centered_fresh": [
      30
    ]
  },
  "block3": {
    "inv_2n_encode": [
      22
    ],
    "square_rescales": [
      22
    ],
    "x_fresh": [
      30
    ],
    "x_inv_2n_result_rescale": [
      22
    ]
  },
  "block4": {
    "ln_mean_inv_d_encode": [
      22
    ],
    "ln_mean_result_rescale": [
      22
    ],
    "ln_square_result_rescale": [
      22
    ],
    "ln_var_inv_d_encode": [
      22
    ],
    "ln_var_result_rescale": [
      22
    ],
    "softmax_out_fresh": [
      30
    ],
    "softmax_out_mask_encode": [
      22
    ],
    "softmax_out_mask_rescale": [
      22
    ],
    "softmax_v_mask_encode": [
      22
    ],
    "softmax_v_mask_rescale": [
      22
    ],
    "softmax_v_matmul_rescale": [
      22
    ],
    "v_fresh": [
      30
    ],
    "v_mask_encode": [
      22
    ],
    "v_mask_rescale": [
      22
    ],
    "wo_encode": [
      22
    ],
    "wo_result_rescale": [
      22
    ]
  },
  "block5": {
    "gamma_encode": [
      22
    ],
    "gamma_result_rescale": [
      22
    ],
    "gelu_coeff_encode": [
      22
    ],
    "gelu_coeff_mul_rescales": [
      22
    ],
    "gelu_power_rescales": [
      22
    ],
    "inv_std_fresh": [
      30
    ],
    "normalize_result_rescale": [
      22
    ],
    "wffn1_encode": [
      22
    ],
    "wffn1_result_rescale": [
      22
    ],
    "x_centered_fresh": [
      30
    ]
  }
}
```

### ActionGrid_truncation12

- action overrides: `{'truncation': 12}`
- base action: BLB RL baseline action: model-side non-truncation fields use highest selectable action-space SF; Rescale_optimizer mode=baseline.
- first_input_sf: `30`
- truncation summary: `12`; effective positions = `59`; skipped = `[{'block': 'block1', 'layer': 0}]`
- model cfg verified before forward: `True`
- handler active layers match expected: `True`
- handler cfg object identity match: `True`
- rescale optimizer: `{'invoker_kind': 'in_process', 'root': '/var/tmp/root-home/Reinforcement-For-Robustness/Rescale_optimizer', 'mode': 'baseline', 'request_count': 60, 'valid_count': 60, 'invalid_count': 0, 't_new_sources': ['unknown']}`

Non-truncation unique scaling factors:

```json
{
  "block1": {
    "gelu_out_fresh": [
      30
    ],
    "mean_inv_d_encode": [
      22
    ],
    "mean_result_rescale": [
      22
    ],
    "square_result_rescale": [
      22
    ],
    "var_inv_d_encode": [
      22
    ],
    "var_result_rescale": [
      22
    ],
    "wffn2_encode": [
      22
    ],
    "wffn2_result_rescale": [
      22
    ]
  },
  "block2": {
    "gamma_encode": [
      22
    ],
    "gamma_result_rescale": [
      22
    ],
    "inv_std_fresh": [
      30
    ],
    "kt_mask1_encode": [
      22
    ],
    "kt_mask1_result_rescale": [
      22
    ],
    "kt_mask2_encode": [
      22
    ],
    "kt_mask2_result_rescale": [
      22
    ],
    "normalize_result_rescale": [
      22
    ],
    "q_mask1_encode": [
      22
    ],
    "q_mask1_result_rescale": [
      22
    ],
    "q_mask2_encode": [
      22
    ],
    "q_mask2_result_rescale": [
      22
    ],
    "qkt_matmul_result_rescale": [
      22
    ],
    "qkt_merge_mask_encode": [
      22
    ],
    "qkt_merge_mask_result_rescale": [
      22
    ],
    "wk_encode": [
      22
    ],
    "wk_result_rescale": [
      22
    ],
    "wq_encode": [
      22
    ],
    "wq_result_rescale": [
      22
    ],
    "wv_encode": [
      22
    ],
    "wv_result_rescale": [
      22
    ],
    "x_centered_fresh": [
      30
    ]
  },
  "block3": {
    "inv_2n_encode": [
      22
    ],
    "square_rescales": [
      22
    ],
    "x_fresh": [
      30
    ],
    "x_inv_2n_result_rescale": [
      22
    ]
  },
  "block4": {
    "ln_mean_inv_d_encode": [
      22
    ],
    "ln_mean_result_rescale": [
      22
    ],
    "ln_square_result_rescale": [
      22
    ],
    "ln_var_inv_d_encode": [
      22
    ],
    "ln_var_result_rescale": [
      22
    ],
    "softmax_out_fresh": [
      30
    ],
    "softmax_out_mask_encode": [
      22
    ],
    "softmax_out_mask_rescale": [
      22
    ],
    "softmax_v_mask_encode": [
      22
    ],
    "softmax_v_mask_rescale": [
      22
    ],
    "softmax_v_matmul_rescale": [
      22
    ],
    "v_fresh": [
      30
    ],
    "v_mask_encode": [
      22
    ],
    "v_mask_rescale": [
      22
    ],
    "wo_encode": [
      22
    ],
    "wo_result_rescale": [
      22
    ]
  },
  "block5": {
    "gamma_encode": [
      22
    ],
    "gamma_result_rescale": [
      22
    ],
    "gelu_coeff_encode": [
      22
    ],
    "gelu_coeff_mul_rescales": [
      22
    ],
    "gelu_power_rescales": [
      22
    ],
    "inv_std_fresh": [
      30
    ],
    "normalize_result_rescale": [
      22
    ],
    "wffn1_encode": [
      22
    ],
    "wffn1_result_rescale": [
      22
    ],
    "x_centered_fresh": [
      30
    ]
  }
}
```

### ActionGrid_truncation13

- action overrides: `{'truncation': 13}`
- base action: BLB RL baseline action: model-side non-truncation fields use highest selectable action-space SF; Rescale_optimizer mode=baseline.
- first_input_sf: `30`
- truncation summary: `13`; effective positions = `59`; skipped = `[{'block': 'block1', 'layer': 0}]`
- model cfg verified before forward: `True`
- handler active layers match expected: `True`
- handler cfg object identity match: `True`
- rescale optimizer: `{'invoker_kind': 'in_process', 'root': '/var/tmp/root-home/Reinforcement-For-Robustness/Rescale_optimizer', 'mode': 'baseline', 'request_count': 60, 'valid_count': 60, 'invalid_count': 0, 't_new_sources': ['unknown']}`

Non-truncation unique scaling factors:

```json
{
  "block1": {
    "gelu_out_fresh": [
      30
    ],
    "mean_inv_d_encode": [
      22
    ],
    "mean_result_rescale": [
      22
    ],
    "square_result_rescale": [
      22
    ],
    "var_inv_d_encode": [
      22
    ],
    "var_result_rescale": [
      22
    ],
    "wffn2_encode": [
      22
    ],
    "wffn2_result_rescale": [
      22
    ]
  },
  "block2": {
    "gamma_encode": [
      22
    ],
    "gamma_result_rescale": [
      22
    ],
    "inv_std_fresh": [
      30
    ],
    "kt_mask1_encode": [
      22
    ],
    "kt_mask1_result_rescale": [
      22
    ],
    "kt_mask2_encode": [
      22
    ],
    "kt_mask2_result_rescale": [
      22
    ],
    "normalize_result_rescale": [
      22
    ],
    "q_mask1_encode": [
      22
    ],
    "q_mask1_result_rescale": [
      22
    ],
    "q_mask2_encode": [
      22
    ],
    "q_mask2_result_rescale": [
      22
    ],
    "qkt_matmul_result_rescale": [
      22
    ],
    "qkt_merge_mask_encode": [
      22
    ],
    "qkt_merge_mask_result_rescale": [
      22
    ],
    "wk_encode": [
      22
    ],
    "wk_result_rescale": [
      22
    ],
    "wq_encode": [
      22
    ],
    "wq_result_rescale": [
      22
    ],
    "wv_encode": [
      22
    ],
    "wv_result_rescale": [
      22
    ],
    "x_centered_fresh": [
      30
    ]
  },
  "block3": {
    "inv_2n_encode": [
      22
    ],
    "square_rescales": [
      22
    ],
    "x_fresh": [
      30
    ],
    "x_inv_2n_result_rescale": [
      22
    ]
  },
  "block4": {
    "ln_mean_inv_d_encode": [
      22
    ],
    "ln_mean_result_rescale": [
      22
    ],
    "ln_square_result_rescale": [
      22
    ],
    "ln_var_inv_d_encode": [
      22
    ],
    "ln_var_result_rescale": [
      22
    ],
    "softmax_out_fresh": [
      30
    ],
    "softmax_out_mask_encode": [
      22
    ],
    "softmax_out_mask_rescale": [
      22
    ],
    "softmax_v_mask_encode": [
      22
    ],
    "softmax_v_mask_rescale": [
      22
    ],
    "softmax_v_matmul_rescale": [
      22
    ],
    "v_fresh": [
      30
    ],
    "v_mask_encode": [
      22
    ],
    "v_mask_rescale": [
      22
    ],
    "wo_encode": [
      22
    ],
    "wo_result_rescale": [
      22
    ]
  },
  "block5": {
    "gamma_encode": [
      22
    ],
    "gamma_result_rescale": [
      22
    ],
    "gelu_coeff_encode": [
      22
    ],
    "gelu_coeff_mul_rescales": [
      22
    ],
    "gelu_power_rescales": [
      22
    ],
    "inv_std_fresh": [
      30
    ],
    "normalize_result_rescale": [
      22
    ],
    "wffn1_encode": [
      22
    ],
    "wffn1_result_rescale": [
      22
    ],
    "x_centered_fresh": [
      30
    ]
  }
}
```

