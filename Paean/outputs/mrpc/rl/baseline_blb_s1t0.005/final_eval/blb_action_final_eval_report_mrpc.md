# BLB Action Final Evaluation Report

- dataset: `mrpc`
- split: `validation_full`
- selected_source: `blb_action(stage1=json)`
- repeat_n: `50`
- rescale_optimizer: `in_process`
- rescale_optimizer_root: `/var/tmp/root-home/Reinforcement-For-Robustness/Rescale_optimizer`
- json: `/var/tmp/root-home/Reinforcement-For-Robustness/Paean/outputs/mrpc/rl/baseline_blb_s1t0.005/final_eval/blb_action_final_eval_results_mrpc.json`

## Baseline

- clean baseline loss: `0.380749`
- clean baseline Acc.: `0.879902`
- clean baseline F1: `0.877442`

## Group Comparison

| group | truncation k | effective K positions | loss mean | loss std | Acc. mean | Acc. std | F1 mean | F1 std | time mean ms | total bits | fusion | model cfg verified |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `ActionSelected` | 10,13 | 59 | 0.374301 | 0.002312 | 0.878578 | 0.003439 | 0.876524 | 0.003530 | 393.753 | 14779 | 0 | False |

## Configuration Details

### ActionSelected

- action overrides: `{}`
- base action: BLB RL baseline action: model-side non-truncation fields use highest selectable action-space SF; Rescale_optimizer mode=baseline.
- first_input_sf: `0`
- truncation summary: `10,13`; effective positions = `59`; skipped = `[]`
- model cfg verified before forward: `False`
- handler active layers match expected: `False`
- handler cfg object identity match: `True`
- rescale optimizer: `{'invoker_kind': 'in_process', 'root': '/var/tmp/root-home/Reinforcement-For-Robustness/Rescale_optimizer', 'mode': 'baseline', 'request_count': 59, 'valid_count': 59, 'invalid_count': 0, 't_new_sources': ['unknown']}`

Non-truncation unique scaling factors:

```json
{
  "block1": {
    "gelu_out_fresh": [
      30
    ],
    "mean_inv_d_encode": [
      20
    ],
    "mean_result_rescale": [
      34
    ],
    "var_inv_d_encode": [
      20
    ],
    "var_result_rescale": [
      34
    ],
    "wffn2_encode": [
      20
    ]
  },
  "block2": {
    "gamma_encode": [
      20
    ],
    "gamma_result_rescale": [
      31
    ],
    "inv_std_fresh": [
      31
    ],
    "kt_mask1_encode": [
      22
    ],
    "kt_mask2_encode": [
      22
    ],
    "kt_mask2_result_rescale": [
      31
    ],
    "q_mask1_encode": [
      22
    ],
    "q_mask2_encode": [
      22
    ],
    "qkt_merge_mask_encode": [
      22
    ],
    "qkt_merge_mask_result_rescale": [
      28
    ],
    "wk_encode": [
      22
    ],
    "wq_encode": [
      22
    ],
    "wv_encode": [
      22
    ],
    "x_centered_fresh": [
      30
    ]
  },
  "block3": {
    "inv_2n_encode": [
      15
    ],
    "square_rescales": [
      31
    ],
    "x_fresh": [
      28
    ]
  },
  "block4": {
    "ln_mean_inv_d_encode": [
      20
    ],
    "ln_mean_result_rescale": [
      31
    ],
    "ln_var_inv_d_encode": [
      20
    ],
    "ln_var_result_rescale": [
      28
    ],
    "softmax_out_fresh": [
      35
    ],
    "softmax_out_mask_encode": [
      14
    ],
    "softmax_v_mask_encode": [
      14
    ],
    "softmax_v_matmul_rescale": [
      31
    ],
    "v_fresh": [
      30
    ],
    "v_mask_encode": [
      22
    ],
    "wo_encode": [
      22
    ]
  },
  "block5": {
    "gamma_encode": [
      20
    ],
    "gamma_result_rescale": [
      22
    ],
    "gelu_coeff_encode": [
      20
    ],
    "gelu_power_rescales": [
      31
    ],
    "inv_std_fresh": [
      30
    ],
    "normalize_result_rescale": [
      31
    ],
    "wffn1_encode": [
      22
    ],
    "wffn1_result_rescale": [
      31
    ],
    "x_centered_fresh": [
      31
    ]
  }
}
```

Full noise and truncation configuration:

| path | type | distribution | N | scaling_factor | truncation_k | value | active |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| first_input.fresh | scaling_factor | fresh | 8192 | 0 |  |  | False |
| layer1.block1.gelu_out_fresh | scaling_factor | fresh | 8192 | 30 |  | 30 | True |
| layer1.block1.wffn2_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer1.block1.mean_inv_d_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer1.block1.var_inv_d_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer1.block1.wffn2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer1.block1.mean_result_rescale | scaling_factor | rescale | 8192 | 34 |  | 34 | True |
| layer1.block1.square_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer1.block1.var_result_rescale | scaling_factor | rescale | 8192 | 34 |  | 34 | True |
| layer1.block1.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer2.block1.gelu_out_fresh | scaling_factor | fresh | 8192 | 30 |  | 30 | True |
| layer2.block1.wffn2_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer2.block1.mean_inv_d_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer2.block1.var_inv_d_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer2.block1.wffn2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer2.block1.mean_result_rescale | scaling_factor | rescale | 8192 | 34 |  | 34 | True |
| layer2.block1.square_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer2.block1.var_result_rescale | scaling_factor | rescale | 8192 | 34 |  | 34 | True |
| layer2.block1.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer3.block1.gelu_out_fresh | scaling_factor | fresh | 8192 | 30 |  | 30 | True |
| layer3.block1.wffn2_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer3.block1.mean_inv_d_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer3.block1.var_inv_d_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer3.block1.wffn2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer3.block1.mean_result_rescale | scaling_factor | rescale | 8192 | 34 |  | 34 | True |
| layer3.block1.square_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer3.block1.var_result_rescale | scaling_factor | rescale | 8192 | 34 |  | 34 | True |
| layer3.block1.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer4.block1.gelu_out_fresh | scaling_factor | fresh | 8192 | 30 |  | 30 | True |
| layer4.block1.wffn2_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer4.block1.mean_inv_d_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer4.block1.var_inv_d_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer4.block1.wffn2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer4.block1.mean_result_rescale | scaling_factor | rescale | 8192 | 34 |  | 34 | True |
| layer4.block1.square_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer4.block1.var_result_rescale | scaling_factor | rescale | 8192 | 34 |  | 34 | True |
| layer4.block1.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer5.block1.gelu_out_fresh | scaling_factor | fresh | 8192 | 30 |  | 30 | True |
| layer5.block1.wffn2_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer5.block1.mean_inv_d_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer5.block1.var_inv_d_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer5.block1.wffn2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer5.block1.mean_result_rescale | scaling_factor | rescale | 8192 | 34 |  | 34 | True |
| layer5.block1.square_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer5.block1.var_result_rescale | scaling_factor | rescale | 8192 | 34 |  | 34 | True |
| layer5.block1.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer6.block1.gelu_out_fresh | scaling_factor | fresh | 8192 | 30 |  | 30 | True |
| layer6.block1.wffn2_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer6.block1.mean_inv_d_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer6.block1.var_inv_d_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer6.block1.wffn2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer6.block1.mean_result_rescale | scaling_factor | rescale | 8192 | 34 |  | 34 | True |
| layer6.block1.square_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer6.block1.var_result_rescale | scaling_factor | rescale | 8192 | 34 |  | 34 | True |
| layer6.block1.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer7.block1.gelu_out_fresh | scaling_factor | fresh | 8192 | 30 |  | 30 | True |
| layer7.block1.wffn2_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer7.block1.mean_inv_d_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer7.block1.var_inv_d_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer7.block1.wffn2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer7.block1.mean_result_rescale | scaling_factor | rescale | 8192 | 34 |  | 34 | True |
| layer7.block1.square_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer7.block1.var_result_rescale | scaling_factor | rescale | 8192 | 34 |  | 34 | True |
| layer7.block1.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer8.block1.gelu_out_fresh | scaling_factor | fresh | 8192 | 30 |  | 30 | True |
| layer8.block1.wffn2_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer8.block1.mean_inv_d_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer8.block1.var_inv_d_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer8.block1.wffn2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer8.block1.mean_result_rescale | scaling_factor | rescale | 8192 | 34 |  | 34 | True |
| layer8.block1.square_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer8.block1.var_result_rescale | scaling_factor | rescale | 8192 | 34 |  | 34 | True |
| layer8.block1.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer9.block1.gelu_out_fresh | scaling_factor | fresh | 8192 | 30 |  | 30 | True |
| layer9.block1.wffn2_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer9.block1.mean_inv_d_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer9.block1.var_inv_d_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer9.block1.wffn2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer9.block1.mean_result_rescale | scaling_factor | rescale | 8192 | 34 |  | 34 | True |
| layer9.block1.square_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer9.block1.var_result_rescale | scaling_factor | rescale | 8192 | 34 |  | 34 | True |
| layer9.block1.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer10.block1.gelu_out_fresh | scaling_factor | fresh | 8192 | 30 |  | 30 | True |
| layer10.block1.wffn2_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer10.block1.mean_inv_d_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer10.block1.var_inv_d_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer10.block1.wffn2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer10.block1.mean_result_rescale | scaling_factor | rescale | 8192 | 34 |  | 34 | True |
| layer10.block1.square_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer10.block1.var_result_rescale | scaling_factor | rescale | 8192 | 34 |  | 34 | True |
| layer10.block1.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer11.block1.gelu_out_fresh | scaling_factor | fresh | 8192 | 30 |  | 30 | True |
| layer11.block1.wffn2_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer11.block1.mean_inv_d_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer11.block1.var_inv_d_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer11.block1.wffn2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer11.block1.mean_result_rescale | scaling_factor | rescale | 8192 | 34 |  | 34 | True |
| layer11.block1.square_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer11.block1.var_result_rescale | scaling_factor | rescale | 8192 | 34 |  | 34 | True |
| layer11.block1.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer0.block2.inv_std_fresh | scaling_factor | fresh | 16384 | 31 |  | 31 | True |
| layer0.block2.x_centered_fresh | scaling_factor | fresh | 16384 | 30 |  | 30 | True |
| layer0.block2.gamma_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer0.block2.wk_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer0.block2.kt_mask1_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer0.block2.kt_mask2_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer0.block2.wq_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer0.block2.q_mask1_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer0.block2.q_mask2_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer0.block2.wv_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer0.block2.qkt_merge_mask_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer0.block2.normalize_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer0.block2.gamma_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer0.block2.wk_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer0.block2.kt_mask1_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer0.block2.kt_mask2_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer0.block2.wq_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer0.block2.q_mask1_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer0.block2.q_mask2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer0.block2.wv_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer0.block2.qkt_matmul_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer0.block2.qkt_merge_mask_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer0.block2.output_truncation_k | truncation |  |  |  | 10 | 10 | True |
| layer1.block2.inv_std_fresh | scaling_factor | fresh | 16384 | 31 |  | 31 | True |
| layer1.block2.x_centered_fresh | scaling_factor | fresh | 16384 | 30 |  | 30 | True |
| layer1.block2.gamma_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer1.block2.wk_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer1.block2.kt_mask1_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer1.block2.kt_mask2_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer1.block2.wq_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer1.block2.q_mask1_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer1.block2.q_mask2_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer1.block2.wv_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer1.block2.qkt_merge_mask_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer1.block2.normalize_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer1.block2.gamma_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer1.block2.wk_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer1.block2.kt_mask1_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer1.block2.kt_mask2_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer1.block2.wq_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer1.block2.q_mask1_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer1.block2.q_mask2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer1.block2.wv_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer1.block2.qkt_matmul_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer1.block2.qkt_merge_mask_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer1.block2.output_truncation_k | truncation |  |  |  | 10 | 10 | True |
| layer2.block2.inv_std_fresh | scaling_factor | fresh | 16384 | 31 |  | 31 | True |
| layer2.block2.x_centered_fresh | scaling_factor | fresh | 16384 | 30 |  | 30 | True |
| layer2.block2.gamma_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer2.block2.wk_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer2.block2.kt_mask1_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer2.block2.kt_mask2_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer2.block2.wq_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer2.block2.q_mask1_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer2.block2.q_mask2_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer2.block2.wv_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer2.block2.qkt_merge_mask_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer2.block2.normalize_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer2.block2.gamma_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer2.block2.wk_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer2.block2.kt_mask1_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer2.block2.kt_mask2_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer2.block2.wq_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer2.block2.q_mask1_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer2.block2.q_mask2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer2.block2.wv_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer2.block2.qkt_matmul_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer2.block2.qkt_merge_mask_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer2.block2.output_truncation_k | truncation |  |  |  | 10 | 10 | True |
| layer3.block2.inv_std_fresh | scaling_factor | fresh | 16384 | 31 |  | 31 | True |
| layer3.block2.x_centered_fresh | scaling_factor | fresh | 16384 | 30 |  | 30 | True |
| layer3.block2.gamma_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer3.block2.wk_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer3.block2.kt_mask1_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer3.block2.kt_mask2_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer3.block2.wq_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer3.block2.q_mask1_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer3.block2.q_mask2_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer3.block2.wv_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer3.block2.qkt_merge_mask_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer3.block2.normalize_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer3.block2.gamma_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer3.block2.wk_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer3.block2.kt_mask1_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer3.block2.kt_mask2_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer3.block2.wq_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer3.block2.q_mask1_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer3.block2.q_mask2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer3.block2.wv_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer3.block2.qkt_matmul_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer3.block2.qkt_merge_mask_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer3.block2.output_truncation_k | truncation |  |  |  | 10 | 10 | True |
| layer4.block2.inv_std_fresh | scaling_factor | fresh | 16384 | 31 |  | 31 | True |
| layer4.block2.x_centered_fresh | scaling_factor | fresh | 16384 | 30 |  | 30 | True |
| layer4.block2.gamma_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer4.block2.wk_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer4.block2.kt_mask1_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer4.block2.kt_mask2_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer4.block2.wq_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer4.block2.q_mask1_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer4.block2.q_mask2_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer4.block2.wv_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer4.block2.qkt_merge_mask_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer4.block2.normalize_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer4.block2.gamma_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer4.block2.wk_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer4.block2.kt_mask1_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer4.block2.kt_mask2_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer4.block2.wq_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer4.block2.q_mask1_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer4.block2.q_mask2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer4.block2.wv_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer4.block2.qkt_matmul_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer4.block2.qkt_merge_mask_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer4.block2.output_truncation_k | truncation |  |  |  | 10 | 10 | True |
| layer5.block2.inv_std_fresh | scaling_factor | fresh | 16384 | 31 |  | 31 | True |
| layer5.block2.x_centered_fresh | scaling_factor | fresh | 16384 | 30 |  | 30 | True |
| layer5.block2.gamma_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer5.block2.wk_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer5.block2.kt_mask1_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer5.block2.kt_mask2_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer5.block2.wq_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer5.block2.q_mask1_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer5.block2.q_mask2_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer5.block2.wv_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer5.block2.qkt_merge_mask_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer5.block2.normalize_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer5.block2.gamma_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer5.block2.wk_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer5.block2.kt_mask1_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer5.block2.kt_mask2_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer5.block2.wq_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer5.block2.q_mask1_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer5.block2.q_mask2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer5.block2.wv_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer5.block2.qkt_matmul_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer5.block2.qkt_merge_mask_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer5.block2.output_truncation_k | truncation |  |  |  | 10 | 10 | True |
| layer6.block2.inv_std_fresh | scaling_factor | fresh | 16384 | 31 |  | 31 | True |
| layer6.block2.x_centered_fresh | scaling_factor | fresh | 16384 | 30 |  | 30 | True |
| layer6.block2.gamma_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer6.block2.wk_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer6.block2.kt_mask1_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer6.block2.kt_mask2_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer6.block2.wq_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer6.block2.q_mask1_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer6.block2.q_mask2_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer6.block2.wv_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer6.block2.qkt_merge_mask_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer6.block2.normalize_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer6.block2.gamma_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer6.block2.wk_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer6.block2.kt_mask1_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer6.block2.kt_mask2_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer6.block2.wq_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer6.block2.q_mask1_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer6.block2.q_mask2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer6.block2.wv_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer6.block2.qkt_matmul_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer6.block2.qkt_merge_mask_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer6.block2.output_truncation_k | truncation |  |  |  | 10 | 10 | True |
| layer7.block2.inv_std_fresh | scaling_factor | fresh | 16384 | 31 |  | 31 | True |
| layer7.block2.x_centered_fresh | scaling_factor | fresh | 16384 | 30 |  | 30 | True |
| layer7.block2.gamma_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer7.block2.wk_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer7.block2.kt_mask1_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer7.block2.kt_mask2_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer7.block2.wq_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer7.block2.q_mask1_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer7.block2.q_mask2_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer7.block2.wv_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer7.block2.qkt_merge_mask_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer7.block2.normalize_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer7.block2.gamma_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer7.block2.wk_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer7.block2.kt_mask1_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer7.block2.kt_mask2_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer7.block2.wq_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer7.block2.q_mask1_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer7.block2.q_mask2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer7.block2.wv_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer7.block2.qkt_matmul_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer7.block2.qkt_merge_mask_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer7.block2.output_truncation_k | truncation |  |  |  | 10 | 10 | True |
| layer8.block2.inv_std_fresh | scaling_factor | fresh | 16384 | 31 |  | 31 | True |
| layer8.block2.x_centered_fresh | scaling_factor | fresh | 16384 | 30 |  | 30 | True |
| layer8.block2.gamma_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer8.block2.wk_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer8.block2.kt_mask1_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer8.block2.kt_mask2_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer8.block2.wq_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer8.block2.q_mask1_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer8.block2.q_mask2_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer8.block2.wv_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer8.block2.qkt_merge_mask_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer8.block2.normalize_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer8.block2.gamma_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer8.block2.wk_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer8.block2.kt_mask1_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer8.block2.kt_mask2_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer8.block2.wq_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer8.block2.q_mask1_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer8.block2.q_mask2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer8.block2.wv_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer8.block2.qkt_matmul_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer8.block2.qkt_merge_mask_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer8.block2.output_truncation_k | truncation |  |  |  | 10 | 10 | True |
| layer9.block2.inv_std_fresh | scaling_factor | fresh | 16384 | 31 |  | 31 | True |
| layer9.block2.x_centered_fresh | scaling_factor | fresh | 16384 | 30 |  | 30 | True |
| layer9.block2.gamma_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer9.block2.wk_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer9.block2.kt_mask1_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer9.block2.kt_mask2_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer9.block2.wq_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer9.block2.q_mask1_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer9.block2.q_mask2_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer9.block2.wv_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer9.block2.qkt_merge_mask_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer9.block2.normalize_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer9.block2.gamma_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer9.block2.wk_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer9.block2.kt_mask1_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer9.block2.kt_mask2_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer9.block2.wq_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer9.block2.q_mask1_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer9.block2.q_mask2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer9.block2.wv_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer9.block2.qkt_matmul_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer9.block2.qkt_merge_mask_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer9.block2.output_truncation_k | truncation |  |  |  | 10 | 10 | True |
| layer10.block2.inv_std_fresh | scaling_factor | fresh | 16384 | 31 |  | 31 | True |
| layer10.block2.x_centered_fresh | scaling_factor | fresh | 16384 | 30 |  | 30 | True |
| layer10.block2.gamma_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer10.block2.wk_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer10.block2.kt_mask1_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer10.block2.kt_mask2_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer10.block2.wq_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer10.block2.q_mask1_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer10.block2.q_mask2_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer10.block2.wv_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer10.block2.qkt_merge_mask_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer10.block2.normalize_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer10.block2.gamma_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer10.block2.wk_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer10.block2.kt_mask1_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer10.block2.kt_mask2_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer10.block2.wq_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer10.block2.q_mask1_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer10.block2.q_mask2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer10.block2.wv_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer10.block2.qkt_matmul_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer10.block2.qkt_merge_mask_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer10.block2.output_truncation_k | truncation |  |  |  | 10 | 10 | True |
| layer11.block2.inv_std_fresh | scaling_factor | fresh | 16384 | 31 |  | 31 | True |
| layer11.block2.x_centered_fresh | scaling_factor | fresh | 16384 | 30 |  | 30 | True |
| layer11.block2.gamma_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer11.block2.wk_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer11.block2.kt_mask1_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer11.block2.kt_mask2_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer11.block2.wq_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer11.block2.q_mask1_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer11.block2.q_mask2_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer11.block2.wv_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer11.block2.qkt_merge_mask_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer11.block2.normalize_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer11.block2.gamma_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer11.block2.wk_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer11.block2.kt_mask1_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer11.block2.kt_mask2_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer11.block2.wq_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer11.block2.q_mask1_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer11.block2.q_mask2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer11.block2.wv_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer11.block2.qkt_matmul_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer11.block2.qkt_merge_mask_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer11.block2.output_truncation_k | truncation |  |  |  | 10 | 10 | True |
| layer0.block3.degree | parameter |  |  |  |  | 2 | True |
| layer0.block3.x_fresh | scaling_factor | fresh | 8192 | 28 |  | 28 | True |
| layer0.block3.inv_2n_encode | scaling_factor | encoding | 8192 | 15 |  | 15 | True |
| layer0.block3.x_inv_2n_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer0.block3.square_rescales[0] | scaling_factor | rescale | 8192 | 31 |  | 31 | True |
| layer0.block3.square_rescales[1] | scaling_factor | rescale | 8192 | 31 |  | 31 | True |
| layer0.block3.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer1.block3.degree | parameter |  |  |  |  | 2 | True |
| layer1.block3.x_fresh | scaling_factor | fresh | 8192 | 28 |  | 28 | True |
| layer1.block3.inv_2n_encode | scaling_factor | encoding | 8192 | 15 |  | 15 | True |
| layer1.block3.x_inv_2n_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer1.block3.square_rescales[0] | scaling_factor | rescale | 8192 | 31 |  | 31 | True |
| layer1.block3.square_rescales[1] | scaling_factor | rescale | 8192 | 31 |  | 31 | True |
| layer1.block3.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer2.block3.degree | parameter |  |  |  |  | 5 | True |
| layer2.block3.x_fresh | scaling_factor | fresh | 16384 | 28 |  | 28 | True |
| layer2.block3.inv_2n_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer2.block3.x_inv_2n_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer2.block3.square_rescales[0] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer2.block3.square_rescales[1] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer2.block3.square_rescales[2] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer2.block3.square_rescales[3] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer2.block3.square_rescales[4] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer2.block3.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer3.block3.degree | parameter |  |  |  |  | 5 | True |
| layer3.block3.x_fresh | scaling_factor | fresh | 16384 | 28 |  | 28 | True |
| layer3.block3.inv_2n_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer3.block3.x_inv_2n_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer3.block3.square_rescales[0] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer3.block3.square_rescales[1] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer3.block3.square_rescales[2] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer3.block3.square_rescales[3] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer3.block3.square_rescales[4] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer3.block3.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer4.block3.degree | parameter |  |  |  |  | 5 | True |
| layer4.block3.x_fresh | scaling_factor | fresh | 16384 | 28 |  | 28 | True |
| layer4.block3.inv_2n_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer4.block3.x_inv_2n_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer4.block3.square_rescales[0] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer4.block3.square_rescales[1] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer4.block3.square_rescales[2] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer4.block3.square_rescales[3] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer4.block3.square_rescales[4] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer4.block3.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer5.block3.degree | parameter |  |  |  |  | 2 | True |
| layer5.block3.x_fresh | scaling_factor | fresh | 8192 | 28 |  | 28 | True |
| layer5.block3.inv_2n_encode | scaling_factor | encoding | 8192 | 15 |  | 15 | True |
| layer5.block3.x_inv_2n_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer5.block3.square_rescales[0] | scaling_factor | rescale | 8192 | 31 |  | 31 | True |
| layer5.block3.square_rescales[1] | scaling_factor | rescale | 8192 | 31 |  | 31 | True |
| layer5.block3.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer6.block3.degree | parameter |  |  |  |  | 5 | True |
| layer6.block3.x_fresh | scaling_factor | fresh | 16384 | 28 |  | 28 | True |
| layer6.block3.inv_2n_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer6.block3.x_inv_2n_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer6.block3.square_rescales[0] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer6.block3.square_rescales[1] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer6.block3.square_rescales[2] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer6.block3.square_rescales[3] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer6.block3.square_rescales[4] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer6.block3.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer7.block3.degree | parameter |  |  |  |  | 2 | True |
| layer7.block3.x_fresh | scaling_factor | fresh | 8192 | 28 |  | 28 | True |
| layer7.block3.inv_2n_encode | scaling_factor | encoding | 8192 | 15 |  | 15 | True |
| layer7.block3.x_inv_2n_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer7.block3.square_rescales[0] | scaling_factor | rescale | 8192 | 31 |  | 31 | True |
| layer7.block3.square_rescales[1] | scaling_factor | rescale | 8192 | 31 |  | 31 | True |
| layer7.block3.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer8.block3.degree | parameter |  |  |  |  | 5 | True |
| layer8.block3.x_fresh | scaling_factor | fresh | 16384 | 28 |  | 28 | True |
| layer8.block3.inv_2n_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer8.block3.x_inv_2n_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer8.block3.square_rescales[0] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer8.block3.square_rescales[1] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer8.block3.square_rescales[2] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer8.block3.square_rescales[3] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer8.block3.square_rescales[4] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer8.block3.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer9.block3.degree | parameter |  |  |  |  | 5 | True |
| layer9.block3.x_fresh | scaling_factor | fresh | 16384 | 28 |  | 28 | True |
| layer9.block3.inv_2n_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer9.block3.x_inv_2n_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer9.block3.square_rescales[0] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer9.block3.square_rescales[1] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer9.block3.square_rescales[2] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer9.block3.square_rescales[3] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer9.block3.square_rescales[4] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer9.block3.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer10.block3.degree | parameter |  |  |  |  | 6 | True |
| layer10.block3.x_fresh | scaling_factor | fresh | 16384 | 28 |  | 28 | True |
| layer10.block3.inv_2n_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer10.block3.x_inv_2n_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer10.block3.square_rescales[0] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer10.block3.square_rescales[1] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer10.block3.square_rescales[2] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer10.block3.square_rescales[3] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer10.block3.square_rescales[4] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer10.block3.square_rescales[5] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer10.block3.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer11.block3.degree | parameter |  |  |  |  | 2 | True |
| layer11.block3.x_fresh | scaling_factor | fresh | 8192 | 28 |  | 28 | True |
| layer11.block3.inv_2n_encode | scaling_factor | encoding | 8192 | 15 |  | 15 | True |
| layer11.block3.x_inv_2n_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer11.block3.square_rescales[0] | scaling_factor | rescale | 8192 | 31 |  | 31 | True |
| layer11.block3.square_rescales[1] | scaling_factor | rescale | 8192 | 31 |  | 31 | True |
| layer11.block3.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer0.block4.softmax_out_fresh | scaling_factor | fresh | 16384 | 35 |  | 35 | True |
| layer0.block4.softmax_out_mask_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
| layer0.block4.v_fresh | scaling_factor | fresh | 16384 | 30 |  | 30 | True |
| layer0.block4.v_mask_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer0.block4.softmax_v_mask_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
| layer0.block4.wo_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer0.block4.ln_mean_inv_d_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer0.block4.ln_var_inv_d_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer0.block4.softmax_out_mask_rescale | scaling_factor |  |  |  |  |  | False |
| layer0.block4.v_mask_rescale | scaling_factor |  |  |  |  |  | False |
| layer0.block4.softmax_v_matmul_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer0.block4.softmax_v_mask_rescale | scaling_factor |  |  |  |  |  | False |
| layer0.block4.wo_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer0.block4.ln_mean_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer0.block4.ln_square_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer0.block4.ln_var_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer0.block4.output_truncation_k | truncation |  |  |  | 10 | 10 | True |
| layer1.block4.softmax_out_fresh | scaling_factor | fresh | 16384 | 35 |  | 35 | True |
| layer1.block4.softmax_out_mask_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
| layer1.block4.v_fresh | scaling_factor | fresh | 16384 | 30 |  | 30 | True |
| layer1.block4.v_mask_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer1.block4.softmax_v_mask_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
| layer1.block4.wo_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer1.block4.ln_mean_inv_d_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer1.block4.ln_var_inv_d_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer1.block4.softmax_out_mask_rescale | scaling_factor |  |  |  |  |  | False |
| layer1.block4.v_mask_rescale | scaling_factor |  |  |  |  |  | False |
| layer1.block4.softmax_v_matmul_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer1.block4.softmax_v_mask_rescale | scaling_factor |  |  |  |  |  | False |
| layer1.block4.wo_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer1.block4.ln_mean_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer1.block4.ln_square_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer1.block4.ln_var_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer1.block4.output_truncation_k | truncation |  |  |  | 10 | 10 | True |
| layer2.block4.softmax_out_fresh | scaling_factor | fresh | 16384 | 35 |  | 35 | True |
| layer2.block4.softmax_out_mask_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
| layer2.block4.v_fresh | scaling_factor | fresh | 16384 | 30 |  | 30 | True |
| layer2.block4.v_mask_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer2.block4.softmax_v_mask_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
| layer2.block4.wo_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer2.block4.ln_mean_inv_d_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer2.block4.ln_var_inv_d_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer2.block4.softmax_out_mask_rescale | scaling_factor |  |  |  |  |  | False |
| layer2.block4.v_mask_rescale | scaling_factor |  |  |  |  |  | False |
| layer2.block4.softmax_v_matmul_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer2.block4.softmax_v_mask_rescale | scaling_factor |  |  |  |  |  | False |
| layer2.block4.wo_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer2.block4.ln_mean_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer2.block4.ln_square_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer2.block4.ln_var_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer2.block4.output_truncation_k | truncation |  |  |  | 10 | 10 | True |
| layer3.block4.softmax_out_fresh | scaling_factor | fresh | 16384 | 35 |  | 35 | True |
| layer3.block4.softmax_out_mask_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
| layer3.block4.v_fresh | scaling_factor | fresh | 16384 | 30 |  | 30 | True |
| layer3.block4.v_mask_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer3.block4.softmax_v_mask_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
| layer3.block4.wo_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer3.block4.ln_mean_inv_d_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer3.block4.ln_var_inv_d_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer3.block4.softmax_out_mask_rescale | scaling_factor |  |  |  |  |  | False |
| layer3.block4.v_mask_rescale | scaling_factor |  |  |  |  |  | False |
| layer3.block4.softmax_v_matmul_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer3.block4.softmax_v_mask_rescale | scaling_factor |  |  |  |  |  | False |
| layer3.block4.wo_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer3.block4.ln_mean_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer3.block4.ln_square_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer3.block4.ln_var_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer3.block4.output_truncation_k | truncation |  |  |  | 10 | 10 | True |
| layer4.block4.softmax_out_fresh | scaling_factor | fresh | 16384 | 35 |  | 35 | True |
| layer4.block4.softmax_out_mask_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
| layer4.block4.v_fresh | scaling_factor | fresh | 16384 | 30 |  | 30 | True |
| layer4.block4.v_mask_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer4.block4.softmax_v_mask_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
| layer4.block4.wo_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer4.block4.ln_mean_inv_d_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer4.block4.ln_var_inv_d_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer4.block4.softmax_out_mask_rescale | scaling_factor |  |  |  |  |  | False |
| layer4.block4.v_mask_rescale | scaling_factor |  |  |  |  |  | False |
| layer4.block4.softmax_v_matmul_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer4.block4.softmax_v_mask_rescale | scaling_factor |  |  |  |  |  | False |
| layer4.block4.wo_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer4.block4.ln_mean_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer4.block4.ln_square_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer4.block4.ln_var_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer4.block4.output_truncation_k | truncation |  |  |  | 10 | 10 | True |
| layer5.block4.softmax_out_fresh | scaling_factor | fresh | 16384 | 35 |  | 35 | True |
| layer5.block4.softmax_out_mask_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
| layer5.block4.v_fresh | scaling_factor | fresh | 16384 | 30 |  | 30 | True |
| layer5.block4.v_mask_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer5.block4.softmax_v_mask_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
| layer5.block4.wo_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer5.block4.ln_mean_inv_d_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer5.block4.ln_var_inv_d_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer5.block4.softmax_out_mask_rescale | scaling_factor |  |  |  |  |  | False |
| layer5.block4.v_mask_rescale | scaling_factor |  |  |  |  |  | False |
| layer5.block4.softmax_v_matmul_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer5.block4.softmax_v_mask_rescale | scaling_factor |  |  |  |  |  | False |
| layer5.block4.wo_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer5.block4.ln_mean_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer5.block4.ln_square_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer5.block4.ln_var_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer5.block4.output_truncation_k | truncation |  |  |  | 10 | 10 | True |
| layer6.block4.softmax_out_fresh | scaling_factor | fresh | 16384 | 35 |  | 35 | True |
| layer6.block4.softmax_out_mask_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
| layer6.block4.v_fresh | scaling_factor | fresh | 16384 | 30 |  | 30 | True |
| layer6.block4.v_mask_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer6.block4.softmax_v_mask_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
| layer6.block4.wo_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer6.block4.ln_mean_inv_d_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer6.block4.ln_var_inv_d_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer6.block4.softmax_out_mask_rescale | scaling_factor |  |  |  |  |  | False |
| layer6.block4.v_mask_rescale | scaling_factor |  |  |  |  |  | False |
| layer6.block4.softmax_v_matmul_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer6.block4.softmax_v_mask_rescale | scaling_factor |  |  |  |  |  | False |
| layer6.block4.wo_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer6.block4.ln_mean_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer6.block4.ln_square_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer6.block4.ln_var_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer6.block4.output_truncation_k | truncation |  |  |  | 10 | 10 | True |
| layer7.block4.softmax_out_fresh | scaling_factor | fresh | 16384 | 35 |  | 35 | True |
| layer7.block4.softmax_out_mask_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
| layer7.block4.v_fresh | scaling_factor | fresh | 16384 | 30 |  | 30 | True |
| layer7.block4.v_mask_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer7.block4.softmax_v_mask_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
| layer7.block4.wo_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer7.block4.ln_mean_inv_d_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer7.block4.ln_var_inv_d_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer7.block4.softmax_out_mask_rescale | scaling_factor |  |  |  |  |  | False |
| layer7.block4.v_mask_rescale | scaling_factor |  |  |  |  |  | False |
| layer7.block4.softmax_v_matmul_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer7.block4.softmax_v_mask_rescale | scaling_factor |  |  |  |  |  | False |
| layer7.block4.wo_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer7.block4.ln_mean_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer7.block4.ln_square_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer7.block4.ln_var_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer7.block4.output_truncation_k | truncation |  |  |  | 10 | 10 | True |
| layer8.block4.softmax_out_fresh | scaling_factor | fresh | 16384 | 35 |  | 35 | True |
| layer8.block4.softmax_out_mask_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
| layer8.block4.v_fresh | scaling_factor | fresh | 16384 | 30 |  | 30 | True |
| layer8.block4.v_mask_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer8.block4.softmax_v_mask_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
| layer8.block4.wo_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer8.block4.ln_mean_inv_d_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer8.block4.ln_var_inv_d_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer8.block4.softmax_out_mask_rescale | scaling_factor |  |  |  |  |  | False |
| layer8.block4.v_mask_rescale | scaling_factor |  |  |  |  |  | False |
| layer8.block4.softmax_v_matmul_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer8.block4.softmax_v_mask_rescale | scaling_factor |  |  |  |  |  | False |
| layer8.block4.wo_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer8.block4.ln_mean_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer8.block4.ln_square_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer8.block4.ln_var_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer8.block4.output_truncation_k | truncation |  |  |  | 10 | 10 | True |
| layer9.block4.softmax_out_fresh | scaling_factor | fresh | 16384 | 35 |  | 35 | True |
| layer9.block4.softmax_out_mask_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
| layer9.block4.v_fresh | scaling_factor | fresh | 16384 | 30 |  | 30 | True |
| layer9.block4.v_mask_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer9.block4.softmax_v_mask_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
| layer9.block4.wo_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer9.block4.ln_mean_inv_d_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer9.block4.ln_var_inv_d_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer9.block4.softmax_out_mask_rescale | scaling_factor |  |  |  |  |  | False |
| layer9.block4.v_mask_rescale | scaling_factor |  |  |  |  |  | False |
| layer9.block4.softmax_v_matmul_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer9.block4.softmax_v_mask_rescale | scaling_factor |  |  |  |  |  | False |
| layer9.block4.wo_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer9.block4.ln_mean_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer9.block4.ln_square_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer9.block4.ln_var_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer9.block4.output_truncation_k | truncation |  |  |  | 10 | 10 | True |
| layer10.block4.softmax_out_fresh | scaling_factor | fresh | 16384 | 35 |  | 35 | True |
| layer10.block4.softmax_out_mask_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
| layer10.block4.v_fresh | scaling_factor | fresh | 16384 | 30 |  | 30 | True |
| layer10.block4.v_mask_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer10.block4.softmax_v_mask_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
| layer10.block4.wo_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer10.block4.ln_mean_inv_d_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer10.block4.ln_var_inv_d_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer10.block4.softmax_out_mask_rescale | scaling_factor |  |  |  |  |  | False |
| layer10.block4.v_mask_rescale | scaling_factor |  |  |  |  |  | False |
| layer10.block4.softmax_v_matmul_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer10.block4.softmax_v_mask_rescale | scaling_factor |  |  |  |  |  | False |
| layer10.block4.wo_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer10.block4.ln_mean_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer10.block4.ln_square_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer10.block4.ln_var_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer10.block4.output_truncation_k | truncation |  |  |  | 10 | 10 | True |
| layer11.block4.softmax_out_fresh | scaling_factor | fresh | 16384 | 35 |  | 35 | True |
| layer11.block4.softmax_out_mask_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
| layer11.block4.v_fresh | scaling_factor | fresh | 16384 | 30 |  | 30 | True |
| layer11.block4.v_mask_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer11.block4.softmax_v_mask_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
| layer11.block4.wo_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer11.block4.ln_mean_inv_d_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer11.block4.ln_var_inv_d_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer11.block4.softmax_out_mask_rescale | scaling_factor |  |  |  |  |  | False |
| layer11.block4.v_mask_rescale | scaling_factor |  |  |  |  |  | False |
| layer11.block4.softmax_v_matmul_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer11.block4.softmax_v_mask_rescale | scaling_factor |  |  |  |  |  | False |
| layer11.block4.wo_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer11.block4.ln_mean_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer11.block4.ln_square_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer11.block4.ln_var_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer11.block4.output_truncation_k | truncation |  |  |  | 10 | 10 | True |
| layer0.block5.inv_std_fresh | scaling_factor | fresh | 8192 | 30 |  | 30 | True |
| layer0.block5.x_centered_fresh | scaling_factor | fresh | 8192 | 31 |  | 31 | True |
| layer0.block5.gamma_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer0.block5.wffn1_encode | scaling_factor | encoding | 8192 | 22 |  | 22 | True |
| layer0.block5.gelu_degree | parameter |  |  |  |  | 1 | True |
| layer0.block5.gelu_coeff_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer0.block5.normalize_result_rescale | scaling_factor | rescale | 8192 | 31 |  | 31 | True |
| layer0.block5.gamma_result_rescale | scaling_factor | rescale | 8192 | 22 |  | 22 | True |
| layer0.block5.wffn1_result_rescale | scaling_factor | rescale | 8192 | 31 |  | 31 | True |
| layer0.block5.gelu_power_rescales | scaling_factor_tuple |  |  |  |  | [] | False |
| layer0.block5.gelu_coeff_mul_rescales[0] | scaling_factor |  |  |  |  |  | False |
| layer0.block5.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer1.block5.inv_std_fresh | scaling_factor | fresh | 8192 | 30 |  | 30 | True |
| layer1.block5.x_centered_fresh | scaling_factor | fresh | 8192 | 31 |  | 31 | True |
| layer1.block5.gamma_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer1.block5.wffn1_encode | scaling_factor | encoding | 8192 | 22 |  | 22 | True |
| layer1.block5.gelu_degree | parameter |  |  |  |  | 1 | True |
| layer1.block5.gelu_coeff_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer1.block5.normalize_result_rescale | scaling_factor | rescale | 8192 | 31 |  | 31 | True |
| layer1.block5.gamma_result_rescale | scaling_factor | rescale | 8192 | 22 |  | 22 | True |
| layer1.block5.wffn1_result_rescale | scaling_factor | rescale | 8192 | 31 |  | 31 | True |
| layer1.block5.gelu_power_rescales | scaling_factor_tuple |  |  |  |  | [] | False |
| layer1.block5.gelu_coeff_mul_rescales[0] | scaling_factor |  |  |  |  |  | False |
| layer1.block5.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer2.block5.inv_std_fresh | scaling_factor | fresh | 8192 | 30 |  | 30 | True |
| layer2.block5.x_centered_fresh | scaling_factor | fresh | 8192 | 31 |  | 31 | True |
| layer2.block5.gamma_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer2.block5.wffn1_encode | scaling_factor | encoding | 8192 | 22 |  | 22 | True |
| layer2.block5.gelu_degree | parameter |  |  |  |  | 1 | True |
| layer2.block5.gelu_coeff_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer2.block5.normalize_result_rescale | scaling_factor | rescale | 8192 | 31 |  | 31 | True |
| layer2.block5.gamma_result_rescale | scaling_factor | rescale | 8192 | 22 |  | 22 | True |
| layer2.block5.wffn1_result_rescale | scaling_factor | rescale | 8192 | 31 |  | 31 | True |
| layer2.block5.gelu_power_rescales | scaling_factor_tuple |  |  |  |  | [] | False |
| layer2.block5.gelu_coeff_mul_rescales[0] | scaling_factor |  |  |  |  |  | False |
| layer2.block5.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer3.block5.inv_std_fresh | scaling_factor | fresh | 8192 | 30 |  | 30 | True |
| layer3.block5.x_centered_fresh | scaling_factor | fresh | 8192 | 31 |  | 31 | True |
| layer3.block5.gamma_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer3.block5.wffn1_encode | scaling_factor | encoding | 8192 | 22 |  | 22 | True |
| layer3.block5.gelu_degree | parameter |  |  |  |  | 1 | True |
| layer3.block5.gelu_coeff_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer3.block5.normalize_result_rescale | scaling_factor | rescale | 8192 | 31 |  | 31 | True |
| layer3.block5.gamma_result_rescale | scaling_factor | rescale | 8192 | 22 |  | 22 | True |
| layer3.block5.wffn1_result_rescale | scaling_factor | rescale | 8192 | 31 |  | 31 | True |
| layer3.block5.gelu_power_rescales | scaling_factor_tuple |  |  |  |  | [] | False |
| layer3.block5.gelu_coeff_mul_rescales[0] | scaling_factor |  |  |  |  |  | False |
| layer3.block5.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer4.block5.inv_std_fresh | scaling_factor | fresh | 8192 | 30 |  | 30 | True |
| layer4.block5.x_centered_fresh | scaling_factor | fresh | 8192 | 31 |  | 31 | True |
| layer4.block5.gamma_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer4.block5.wffn1_encode | scaling_factor | encoding | 8192 | 22 |  | 22 | True |
| layer4.block5.gelu_degree | parameter |  |  |  |  | 1 | True |
| layer4.block5.gelu_coeff_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer4.block5.normalize_result_rescale | scaling_factor | rescale | 8192 | 31 |  | 31 | True |
| layer4.block5.gamma_result_rescale | scaling_factor | rescale | 8192 | 22 |  | 22 | True |
| layer4.block5.wffn1_result_rescale | scaling_factor | rescale | 8192 | 31 |  | 31 | True |
| layer4.block5.gelu_power_rescales | scaling_factor_tuple |  |  |  |  | [] | False |
| layer4.block5.gelu_coeff_mul_rescales[0] | scaling_factor |  |  |  |  |  | False |
| layer4.block5.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer5.block5.inv_std_fresh | scaling_factor | fresh | 16384 | 30 |  | 30 | True |
| layer5.block5.x_centered_fresh | scaling_factor | fresh | 16384 | 31 |  | 31 | True |
| layer5.block5.gamma_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer5.block5.wffn1_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer5.block5.gelu_degree | parameter |  |  |  |  | 4 | True |
| layer5.block5.gelu_coeff_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer5.block5.normalize_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer5.block5.gamma_result_rescale | scaling_factor | rescale | 16384 | 22 |  | 22 | True |
| layer5.block5.wffn1_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer5.block5.gelu_power_rescales[0] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer5.block5.gelu_power_rescales[1] | scaling_factor |  |  |  |  |  | False |
| layer5.block5.gelu_power_rescales[2] | scaling_factor |  |  |  |  |  | False |
| layer5.block5.gelu_coeff_mul_rescales[0] | scaling_factor |  |  |  |  |  | False |
| layer5.block5.gelu_coeff_mul_rescales[1] | scaling_factor |  |  |  |  |  | False |
| layer5.block5.gelu_coeff_mul_rescales[2] | scaling_factor |  |  |  |  |  | False |
| layer5.block5.gelu_coeff_mul_rescales[3] | scaling_factor |  |  |  |  |  | False |
| layer5.block5.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer6.block5.inv_std_fresh | scaling_factor | fresh | 8192 | 30 |  | 30 | True |
| layer6.block5.x_centered_fresh | scaling_factor | fresh | 8192 | 31 |  | 31 | True |
| layer6.block5.gamma_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer6.block5.wffn1_encode | scaling_factor | encoding | 8192 | 22 |  | 22 | True |
| layer6.block5.gelu_degree | parameter |  |  |  |  | 1 | True |
| layer6.block5.gelu_coeff_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer6.block5.normalize_result_rescale | scaling_factor | rescale | 8192 | 31 |  | 31 | True |
| layer6.block5.gamma_result_rescale | scaling_factor | rescale | 8192 | 22 |  | 22 | True |
| layer6.block5.wffn1_result_rescale | scaling_factor | rescale | 8192 | 31 |  | 31 | True |
| layer6.block5.gelu_power_rescales | scaling_factor_tuple |  |  |  |  | [] | False |
| layer6.block5.gelu_coeff_mul_rescales[0] | scaling_factor |  |  |  |  |  | False |
| layer6.block5.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer7.block5.inv_std_fresh | scaling_factor | fresh | 8192 | 30 |  | 30 | True |
| layer7.block5.x_centered_fresh | scaling_factor | fresh | 8192 | 31 |  | 31 | True |
| layer7.block5.gamma_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer7.block5.wffn1_encode | scaling_factor | encoding | 8192 | 22 |  | 22 | True |
| layer7.block5.gelu_degree | parameter |  |  |  |  | 1 | True |
| layer7.block5.gelu_coeff_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer7.block5.normalize_result_rescale | scaling_factor | rescale | 8192 | 31 |  | 31 | True |
| layer7.block5.gamma_result_rescale | scaling_factor | rescale | 8192 | 22 |  | 22 | True |
| layer7.block5.wffn1_result_rescale | scaling_factor | rescale | 8192 | 31 |  | 31 | True |
| layer7.block5.gelu_power_rescales | scaling_factor_tuple |  |  |  |  | [] | False |
| layer7.block5.gelu_coeff_mul_rescales[0] | scaling_factor |  |  |  |  |  | False |
| layer7.block5.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer8.block5.inv_std_fresh | scaling_factor | fresh | 8192 | 30 |  | 30 | True |
| layer8.block5.x_centered_fresh | scaling_factor | fresh | 8192 | 31 |  | 31 | True |
| layer8.block5.gamma_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer8.block5.wffn1_encode | scaling_factor | encoding | 8192 | 22 |  | 22 | True |
| layer8.block5.gelu_degree | parameter |  |  |  |  | 1 | True |
| layer8.block5.gelu_coeff_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer8.block5.normalize_result_rescale | scaling_factor | rescale | 8192 | 31 |  | 31 | True |
| layer8.block5.gamma_result_rescale | scaling_factor | rescale | 8192 | 22 |  | 22 | True |
| layer8.block5.wffn1_result_rescale | scaling_factor | rescale | 8192 | 31 |  | 31 | True |
| layer8.block5.gelu_power_rescales | scaling_factor_tuple |  |  |  |  | [] | False |
| layer8.block5.gelu_coeff_mul_rescales[0] | scaling_factor |  |  |  |  |  | False |
| layer8.block5.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer9.block5.inv_std_fresh | scaling_factor | fresh | 8192 | 30 |  | 30 | True |
| layer9.block5.x_centered_fresh | scaling_factor | fresh | 8192 | 31 |  | 31 | True |
| layer9.block5.gamma_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer9.block5.wffn1_encode | scaling_factor | encoding | 8192 | 22 |  | 22 | True |
| layer9.block5.gelu_degree | parameter |  |  |  |  | 1 | True |
| layer9.block5.gelu_coeff_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer9.block5.normalize_result_rescale | scaling_factor | rescale | 8192 | 31 |  | 31 | True |
| layer9.block5.gamma_result_rescale | scaling_factor | rescale | 8192 | 22 |  | 22 | True |
| layer9.block5.wffn1_result_rescale | scaling_factor | rescale | 8192 | 31 |  | 31 | True |
| layer9.block5.gelu_power_rescales | scaling_factor_tuple |  |  |  |  | [] | False |
| layer9.block5.gelu_coeff_mul_rescales[0] | scaling_factor |  |  |  |  |  | False |
| layer9.block5.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer10.block5.inv_std_fresh | scaling_factor | fresh | 8192 | 30 |  | 30 | True |
| layer10.block5.x_centered_fresh | scaling_factor | fresh | 8192 | 31 |  | 31 | True |
| layer10.block5.gamma_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer10.block5.wffn1_encode | scaling_factor | encoding | 8192 | 22 |  | 22 | True |
| layer10.block5.gelu_degree | parameter |  |  |  |  | 1 | True |
| layer10.block5.gelu_coeff_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer10.block5.normalize_result_rescale | scaling_factor | rescale | 8192 | 31 |  | 31 | True |
| layer10.block5.gamma_result_rescale | scaling_factor | rescale | 8192 | 22 |  | 22 | True |
| layer10.block5.wffn1_result_rescale | scaling_factor | rescale | 8192 | 31 |  | 31 | True |
| layer10.block5.gelu_power_rescales | scaling_factor_tuple |  |  |  |  | [] | False |
| layer10.block5.gelu_coeff_mul_rescales[0] | scaling_factor |  |  |  |  |  | False |
| layer10.block5.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer11.block5.inv_std_fresh | scaling_factor | fresh | 8192 | 30 |  | 30 | True |
| layer11.block5.x_centered_fresh | scaling_factor | fresh | 8192 | 31 |  | 31 | True |
| layer11.block5.gamma_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer11.block5.wffn1_encode | scaling_factor | encoding | 8192 | 22 |  | 22 | True |
| layer11.block5.gelu_degree | parameter |  |  |  |  | 1 | True |
| layer11.block5.gelu_coeff_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer11.block5.normalize_result_rescale | scaling_factor | rescale | 8192 | 31 |  | 31 | True |
| layer11.block5.gamma_result_rescale | scaling_factor | rescale | 8192 | 22 |  | 22 | True |
| layer11.block5.wffn1_result_rescale | scaling_factor | rescale | 8192 | 31 |  | 31 | True |
| layer11.block5.gelu_power_rescales | scaling_factor_tuple |  |  |  |  | [] | False |
| layer11.block5.gelu_coeff_mul_rescales[0] | scaling_factor |  |  |  |  |  | False |
| layer11.block5.output_truncation_k | truncation |  |  |  | 13 | 13 | True |

