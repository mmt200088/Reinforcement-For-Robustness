# BLB Action Final Evaluation Report

- dataset: `mrpc`
- split: `validation_full`
- selected_source: `blb_action(stage1=manual)`
- repeat_n: `5`
- rescale_optimizer: `in_process`
- rescale_optimizer_root: `/hy-tmp/fusion_block_eval_rlpath_8896716_20260702_030732/src/Rescale_optimizer`
- json: `/hy-tmp/fusion_block_eval_rlpath_8896716_20260702_030732/src/experiments/server_command_runs/fusion_count_block_eval_rlpath_8896716_20260702_031349/paean_outputs/mrpc/rl/block4_fusionmax_2_layers/final_eval/blb_action_final_eval_results_mrpc.json`

## Baseline

- clean baseline loss: `0.383963`
- clean baseline Acc.: `0.877451`
- clean baseline F1: `0.874422`

## Cost-Matched Random Sampling

- target total_bits_sum: `11237`
- target total_fusion_count: `2`
- target sum_truncation_k: `767`
- requested: `50` configs
- accepted: `0` configs in `5000`/`5000` attempts
- rejection breakdown: invalid=`0`, cost_mismatch=`0`, avg_k_prefilter=`5000`

## Selected vs Cost-Matched Random Comparison

- selected (`ActionSelected`): loss=0.384564 ± 0.003139, Acc.=0.877451 ± 0.003797, F1=0.875815 ± 0.003738, total_bits=11237, fusion=2, avg_k=13.000

## Group Comparison

| group | truncation k | effective K positions | loss mean | loss std | Acc. mean | Acc. std | F1 mean | F1 std | time mean ms | total bits | fusion | replan applied | model cfg verified |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `ActionSelected` | 13 | 59 | 0.384564 | 0.003139 | 0.877451 | 0.003797 | 0.875815 | 0.003738 | 118.812 | 11237 | 2 | True | True |

## Configuration Details

### ActionSelected

- action overrides: `{}`
- base action: BLB RL baseline action: model-side non-truncation fields use highest selectable action-space SF; Rescale_optimizer mode=cfg_derived.
- truncation summary: `13`; effective positions = `59`; skipped = `[]`
- model cfg verified before forward: `True`
- replan cfg applied before forward: `True`
- replan application summary: `{'applied_before_forward': True, 'model_uses_replan_config': True, 'expected_config_count': 47, 'applied_config_count': 47, 'invalid_config_count': 0, 'missing_compact_config_count': 0, 'missing_decoded_cfg_count': 0, 'apply_error_count': 0, 'override_total': 2}`
- fusion group diagnostics: `{'group_name': 'block4_fusionmax_2_layers', 'declared_total_fusion_count': 1, 'realized_total_fusion_count': 2, 'declared_by_graph': {'block1_mrpc': 0, 'block2_mrpc': 0, 'block4': 0, 'block5_n0': 0, 'block5_n1': 0, 'block5_n2': 0, 'block5_n4': 0, 'block4_selected_layers': 1}, 'matches_realized_total': False}`
- handler active layers match expected: `True`
- handler cfg object identity match: `True`
- rescale optimizer: `{'invoker_kind': 'in_process', 'root': '/hy-tmp/fusion_block_eval_rlpath_8896716_20260702_030732/src/Rescale_optimizer', 'mode': 'cfg_derived', 'request_count': 47, 'valid_count': 47, 'invalid_count': 0, 't_new_sources': ['cfg_derived']}`

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
      30
    ],
    "var_inv_d_encode": [
      20
    ],
    "var_result_rescale": [
      27
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
      28
    ],
    "inv_std_fresh": [
      28
    ],
    "kt_mask1_encode": [
      15
    ],
    "kt_mask1_result_rescale": [
      28
    ],
    "kt_mask2_encode": [
      15
    ],
    "q_mask1_encode": [
      15
    ],
    "q_mask1_result_rescale": [
      28
    ],
    "q_mask2_encode": [
      15
    ],
    "qkt_matmul_result_rescale": [
      28
    ],
    "qkt_merge_mask_encode": [
      15
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
      28
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
      14,
      20
    ],
    "ln_mean_result_rescale": [
      31,
      46
    ],
    "ln_square_result_rescale": [
      31,
      32
    ],
    "ln_var_inv_d_encode": [
      20,
      21
    ],
    "softmax_out_fresh": [
      21,
      35
    ],
    "softmax_out_mask_encode": [
      13,
      14
    ],
    "softmax_v_mask_encode": [
      14
    ],
    "softmax_v_matmul_rescale": [
      31
    ],
    "v_fresh": [
      17,
      25
    ],
    "v_mask_encode": [
      13,
      14
    ],
    "wo_encode": [
      14,
      22
    ]
  },
  "block5": {
    "gamma_encode": [
      20
    ],
    "gelu_coeff_encode": [
      20
    ],
    "gelu_coeff_mul_rescales": [
      31
    ],
    "inv_std_fresh": [
      31
    ],
    "normalize_result_rescale": [
      28,
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
| layer1.block1.gelu_out_fresh | scaling_factor | fresh | 8192 | 30 |  | 30 | True |
| layer1.block1.wffn2_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer1.block1.mean_inv_d_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer1.block1.var_inv_d_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer1.block1.wffn2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer1.block1.mean_result_rescale | scaling_factor | rescale | 8192 | 30 |  | 30 | True |
| layer1.block1.square_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer1.block1.var_result_rescale | scaling_factor | rescale | 8192 | 27 |  | 27 | True |
| layer1.block1.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer2.block1.gelu_out_fresh | scaling_factor | fresh | 8192 | 30 |  | 30 | True |
| layer2.block1.wffn2_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer2.block1.mean_inv_d_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer2.block1.var_inv_d_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer2.block1.wffn2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer2.block1.mean_result_rescale | scaling_factor | rescale | 8192 | 30 |  | 30 | True |
| layer2.block1.square_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer2.block1.var_result_rescale | scaling_factor | rescale | 8192 | 27 |  | 27 | True |
| layer2.block1.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer3.block1.gelu_out_fresh | scaling_factor | fresh | 8192 | 30 |  | 30 | True |
| layer3.block1.wffn2_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer3.block1.mean_inv_d_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer3.block1.var_inv_d_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer3.block1.wffn2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer3.block1.mean_result_rescale | scaling_factor | rescale | 8192 | 30 |  | 30 | True |
| layer3.block1.square_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer3.block1.var_result_rescale | scaling_factor | rescale | 8192 | 27 |  | 27 | True |
| layer3.block1.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer4.block1.gelu_out_fresh | scaling_factor | fresh | 8192 | 30 |  | 30 | True |
| layer4.block1.wffn2_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer4.block1.mean_inv_d_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer4.block1.var_inv_d_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer4.block1.wffn2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer4.block1.mean_result_rescale | scaling_factor | rescale | 8192 | 30 |  | 30 | True |
| layer4.block1.square_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer4.block1.var_result_rescale | scaling_factor | rescale | 8192 | 27 |  | 27 | True |
| layer4.block1.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer5.block1.gelu_out_fresh | scaling_factor | fresh | 8192 | 30 |  | 30 | True |
| layer5.block1.wffn2_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer5.block1.mean_inv_d_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer5.block1.var_inv_d_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer5.block1.wffn2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer5.block1.mean_result_rescale | scaling_factor | rescale | 8192 | 30 |  | 30 | True |
| layer5.block1.square_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer5.block1.var_result_rescale | scaling_factor | rescale | 8192 | 27 |  | 27 | True |
| layer5.block1.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer6.block1.gelu_out_fresh | scaling_factor | fresh | 8192 | 30 |  | 30 | True |
| layer6.block1.wffn2_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer6.block1.mean_inv_d_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer6.block1.var_inv_d_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer6.block1.wffn2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer6.block1.mean_result_rescale | scaling_factor | rescale | 8192 | 30 |  | 30 | True |
| layer6.block1.square_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer6.block1.var_result_rescale | scaling_factor | rescale | 8192 | 27 |  | 27 | True |
| layer6.block1.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer7.block1.gelu_out_fresh | scaling_factor | fresh | 8192 | 30 |  | 30 | True |
| layer7.block1.wffn2_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer7.block1.mean_inv_d_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer7.block1.var_inv_d_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer7.block1.wffn2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer7.block1.mean_result_rescale | scaling_factor | rescale | 8192 | 30 |  | 30 | True |
| layer7.block1.square_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer7.block1.var_result_rescale | scaling_factor | rescale | 8192 | 27 |  | 27 | True |
| layer7.block1.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer8.block1.gelu_out_fresh | scaling_factor | fresh | 8192 | 30 |  | 30 | True |
| layer8.block1.wffn2_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer8.block1.mean_inv_d_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer8.block1.var_inv_d_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer8.block1.wffn2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer8.block1.mean_result_rescale | scaling_factor | rescale | 8192 | 30 |  | 30 | True |
| layer8.block1.square_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer8.block1.var_result_rescale | scaling_factor | rescale | 8192 | 27 |  | 27 | True |
| layer8.block1.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer9.block1.gelu_out_fresh | scaling_factor | fresh | 8192 | 30 |  | 30 | True |
| layer9.block1.wffn2_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer9.block1.mean_inv_d_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer9.block1.var_inv_d_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer9.block1.wffn2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer9.block1.mean_result_rescale | scaling_factor | rescale | 8192 | 30 |  | 30 | True |
| layer9.block1.square_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer9.block1.var_result_rescale | scaling_factor | rescale | 8192 | 27 |  | 27 | True |
| layer9.block1.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer10.block1.gelu_out_fresh | scaling_factor | fresh | 8192 | 30 |  | 30 | True |
| layer10.block1.wffn2_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer10.block1.mean_inv_d_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer10.block1.var_inv_d_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer10.block1.wffn2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer10.block1.mean_result_rescale | scaling_factor | rescale | 8192 | 30 |  | 30 | True |
| layer10.block1.square_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer10.block1.var_result_rescale | scaling_factor | rescale | 8192 | 27 |  | 27 | True |
| layer10.block1.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer11.block1.gelu_out_fresh | scaling_factor | fresh | 8192 | 30 |  | 30 | True |
| layer11.block1.wffn2_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer11.block1.mean_inv_d_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer11.block1.var_inv_d_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer11.block1.wffn2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer11.block1.mean_result_rescale | scaling_factor | rescale | 8192 | 30 |  | 30 | True |
| layer11.block1.square_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer11.block1.var_result_rescale | scaling_factor | rescale | 8192 | 27 |  | 27 | True |
| layer11.block1.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer0.block2.inv_std_fresh | scaling_factor | fresh | 16384 | 28 |  | 28 | True |
| layer0.block2.x_centered_fresh | scaling_factor | fresh | 16384 | 28 |  | 28 | True |
| layer0.block2.gamma_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer0.block2.wk_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer0.block2.kt_mask1_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer0.block2.kt_mask2_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer0.block2.wq_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer0.block2.q_mask1_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer0.block2.q_mask2_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer0.block2.wv_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer0.block2.qkt_merge_mask_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer0.block2.normalize_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer0.block2.gamma_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer0.block2.wk_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer0.block2.kt_mask1_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer0.block2.kt_mask2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer0.block2.wq_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer0.block2.q_mask1_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer0.block2.q_mask2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer0.block2.wv_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer0.block2.qkt_matmul_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer0.block2.qkt_merge_mask_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer0.block2.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer1.block2.inv_std_fresh | scaling_factor | fresh | 16384 | 28 |  | 28 | True |
| layer1.block2.x_centered_fresh | scaling_factor | fresh | 16384 | 28 |  | 28 | True |
| layer1.block2.gamma_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer1.block2.wk_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer1.block2.kt_mask1_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer1.block2.kt_mask2_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer1.block2.wq_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer1.block2.q_mask1_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer1.block2.q_mask2_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer1.block2.wv_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer1.block2.qkt_merge_mask_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer1.block2.normalize_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer1.block2.gamma_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer1.block2.wk_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer1.block2.kt_mask1_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer1.block2.kt_mask2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer1.block2.wq_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer1.block2.q_mask1_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer1.block2.q_mask2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer1.block2.wv_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer1.block2.qkt_matmul_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer1.block2.qkt_merge_mask_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer1.block2.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer2.block2.inv_std_fresh | scaling_factor | fresh | 16384 | 28 |  | 28 | True |
| layer2.block2.x_centered_fresh | scaling_factor | fresh | 16384 | 28 |  | 28 | True |
| layer2.block2.gamma_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer2.block2.wk_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer2.block2.kt_mask1_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer2.block2.kt_mask2_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer2.block2.wq_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer2.block2.q_mask1_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer2.block2.q_mask2_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer2.block2.wv_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer2.block2.qkt_merge_mask_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer2.block2.normalize_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer2.block2.gamma_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer2.block2.wk_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer2.block2.kt_mask1_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer2.block2.kt_mask2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer2.block2.wq_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer2.block2.q_mask1_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer2.block2.q_mask2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer2.block2.wv_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer2.block2.qkt_matmul_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer2.block2.qkt_merge_mask_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer2.block2.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer3.block2.inv_std_fresh | scaling_factor | fresh | 16384 | 28 |  | 28 | True |
| layer3.block2.x_centered_fresh | scaling_factor | fresh | 16384 | 28 |  | 28 | True |
| layer3.block2.gamma_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer3.block2.wk_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer3.block2.kt_mask1_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer3.block2.kt_mask2_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer3.block2.wq_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer3.block2.q_mask1_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer3.block2.q_mask2_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer3.block2.wv_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer3.block2.qkt_merge_mask_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer3.block2.normalize_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer3.block2.gamma_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer3.block2.wk_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer3.block2.kt_mask1_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer3.block2.kt_mask2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer3.block2.wq_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer3.block2.q_mask1_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer3.block2.q_mask2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer3.block2.wv_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer3.block2.qkt_matmul_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer3.block2.qkt_merge_mask_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer3.block2.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer4.block2.inv_std_fresh | scaling_factor | fresh | 16384 | 28 |  | 28 | True |
| layer4.block2.x_centered_fresh | scaling_factor | fresh | 16384 | 28 |  | 28 | True |
| layer4.block2.gamma_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer4.block2.wk_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer4.block2.kt_mask1_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer4.block2.kt_mask2_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer4.block2.wq_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer4.block2.q_mask1_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer4.block2.q_mask2_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer4.block2.wv_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer4.block2.qkt_merge_mask_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer4.block2.normalize_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer4.block2.gamma_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer4.block2.wk_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer4.block2.kt_mask1_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer4.block2.kt_mask2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer4.block2.wq_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer4.block2.q_mask1_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer4.block2.q_mask2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer4.block2.wv_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer4.block2.qkt_matmul_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer4.block2.qkt_merge_mask_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer4.block2.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer5.block2.inv_std_fresh | scaling_factor | fresh | 16384 | 28 |  | 28 | True |
| layer5.block2.x_centered_fresh | scaling_factor | fresh | 16384 | 28 |  | 28 | True |
| layer5.block2.gamma_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer5.block2.wk_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer5.block2.kt_mask1_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer5.block2.kt_mask2_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer5.block2.wq_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer5.block2.q_mask1_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer5.block2.q_mask2_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer5.block2.wv_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer5.block2.qkt_merge_mask_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer5.block2.normalize_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer5.block2.gamma_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer5.block2.wk_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer5.block2.kt_mask1_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer5.block2.kt_mask2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer5.block2.wq_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer5.block2.q_mask1_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer5.block2.q_mask2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer5.block2.wv_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer5.block2.qkt_matmul_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer5.block2.qkt_merge_mask_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer5.block2.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer6.block2.inv_std_fresh | scaling_factor | fresh | 16384 | 28 |  | 28 | True |
| layer6.block2.x_centered_fresh | scaling_factor | fresh | 16384 | 28 |  | 28 | True |
| layer6.block2.gamma_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer6.block2.wk_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer6.block2.kt_mask1_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer6.block2.kt_mask2_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer6.block2.wq_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer6.block2.q_mask1_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer6.block2.q_mask2_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer6.block2.wv_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer6.block2.qkt_merge_mask_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer6.block2.normalize_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer6.block2.gamma_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer6.block2.wk_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer6.block2.kt_mask1_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer6.block2.kt_mask2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer6.block2.wq_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer6.block2.q_mask1_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer6.block2.q_mask2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer6.block2.wv_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer6.block2.qkt_matmul_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer6.block2.qkt_merge_mask_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer6.block2.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer7.block2.inv_std_fresh | scaling_factor | fresh | 16384 | 28 |  | 28 | True |
| layer7.block2.x_centered_fresh | scaling_factor | fresh | 16384 | 28 |  | 28 | True |
| layer7.block2.gamma_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer7.block2.wk_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer7.block2.kt_mask1_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer7.block2.kt_mask2_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer7.block2.wq_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer7.block2.q_mask1_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer7.block2.q_mask2_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer7.block2.wv_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer7.block2.qkt_merge_mask_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer7.block2.normalize_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer7.block2.gamma_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer7.block2.wk_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer7.block2.kt_mask1_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer7.block2.kt_mask2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer7.block2.wq_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer7.block2.q_mask1_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer7.block2.q_mask2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer7.block2.wv_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer7.block2.qkt_matmul_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer7.block2.qkt_merge_mask_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer7.block2.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer8.block2.inv_std_fresh | scaling_factor | fresh | 16384 | 28 |  | 28 | True |
| layer8.block2.x_centered_fresh | scaling_factor | fresh | 16384 | 28 |  | 28 | True |
| layer8.block2.gamma_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer8.block2.wk_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer8.block2.kt_mask1_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer8.block2.kt_mask2_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer8.block2.wq_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer8.block2.q_mask1_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer8.block2.q_mask2_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer8.block2.wv_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer8.block2.qkt_merge_mask_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer8.block2.normalize_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer8.block2.gamma_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer8.block2.wk_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer8.block2.kt_mask1_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer8.block2.kt_mask2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer8.block2.wq_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer8.block2.q_mask1_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer8.block2.q_mask2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer8.block2.wv_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer8.block2.qkt_matmul_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer8.block2.qkt_merge_mask_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer8.block2.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer9.block2.inv_std_fresh | scaling_factor | fresh | 16384 | 28 |  | 28 | True |
| layer9.block2.x_centered_fresh | scaling_factor | fresh | 16384 | 28 |  | 28 | True |
| layer9.block2.gamma_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer9.block2.wk_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer9.block2.kt_mask1_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer9.block2.kt_mask2_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer9.block2.wq_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer9.block2.q_mask1_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer9.block2.q_mask2_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer9.block2.wv_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer9.block2.qkt_merge_mask_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer9.block2.normalize_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer9.block2.gamma_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer9.block2.wk_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer9.block2.kt_mask1_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer9.block2.kt_mask2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer9.block2.wq_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer9.block2.q_mask1_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer9.block2.q_mask2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer9.block2.wv_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer9.block2.qkt_matmul_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer9.block2.qkt_merge_mask_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer9.block2.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer10.block2.inv_std_fresh | scaling_factor | fresh | 16384 | 28 |  | 28 | True |
| layer10.block2.x_centered_fresh | scaling_factor | fresh | 16384 | 28 |  | 28 | True |
| layer10.block2.gamma_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer10.block2.wk_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer10.block2.kt_mask1_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer10.block2.kt_mask2_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer10.block2.wq_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer10.block2.q_mask1_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer10.block2.q_mask2_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer10.block2.wv_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer10.block2.qkt_merge_mask_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer10.block2.normalize_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer10.block2.gamma_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer10.block2.wk_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer10.block2.kt_mask1_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer10.block2.kt_mask2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer10.block2.wq_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer10.block2.q_mask1_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer10.block2.q_mask2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer10.block2.wv_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer10.block2.qkt_matmul_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer10.block2.qkt_merge_mask_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer10.block2.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer11.block2.inv_std_fresh | scaling_factor | fresh | 16384 | 28 |  | 28 | True |
| layer11.block2.x_centered_fresh | scaling_factor | fresh | 16384 | 28 |  | 28 | True |
| layer11.block2.gamma_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer11.block2.wk_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer11.block2.kt_mask1_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer11.block2.kt_mask2_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer11.block2.wq_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer11.block2.q_mask1_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer11.block2.q_mask2_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer11.block2.wv_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer11.block2.qkt_merge_mask_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer11.block2.normalize_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer11.block2.gamma_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer11.block2.wk_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer11.block2.kt_mask1_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer11.block2.kt_mask2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer11.block2.wq_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer11.block2.q_mask1_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer11.block2.q_mask2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer11.block2.wv_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer11.block2.qkt_matmul_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer11.block2.qkt_merge_mask_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer11.block2.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer0.block3.degree | parameter |  |  |  |  | 6 | True |
| layer0.block3.x_fresh | scaling_factor | fresh | 16384 | 28 |  | 28 | True |
| layer0.block3.inv_2n_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer0.block3.x_inv_2n_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer0.block3.square_rescales[0] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer0.block3.square_rescales[1] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer0.block3.square_rescales[2] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer0.block3.square_rescales[3] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer0.block3.square_rescales[4] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer0.block3.square_rescales[5] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer0.block3.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer1.block3.degree | parameter |  |  |  |  | 6 | True |
| layer1.block3.x_fresh | scaling_factor | fresh | 16384 | 28 |  | 28 | True |
| layer1.block3.inv_2n_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer1.block3.x_inv_2n_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer1.block3.square_rescales[0] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer1.block3.square_rescales[1] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer1.block3.square_rescales[2] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer1.block3.square_rescales[3] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer1.block3.square_rescales[4] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer1.block3.square_rescales[5] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer1.block3.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer2.block3.degree | parameter |  |  |  |  | 6 | True |
| layer2.block3.x_fresh | scaling_factor | fresh | 16384 | 28 |  | 28 | True |
| layer2.block3.inv_2n_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer2.block3.x_inv_2n_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer2.block3.square_rescales[0] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer2.block3.square_rescales[1] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer2.block3.square_rescales[2] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer2.block3.square_rescales[3] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer2.block3.square_rescales[4] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer2.block3.square_rescales[5] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer2.block3.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer3.block3.degree | parameter |  |  |  |  | 6 | True |
| layer3.block3.x_fresh | scaling_factor | fresh | 16384 | 28 |  | 28 | True |
| layer3.block3.inv_2n_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer3.block3.x_inv_2n_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer3.block3.square_rescales[0] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer3.block3.square_rescales[1] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer3.block3.square_rescales[2] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer3.block3.square_rescales[3] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer3.block3.square_rescales[4] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer3.block3.square_rescales[5] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer3.block3.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer4.block3.degree | parameter |  |  |  |  | 6 | True |
| layer4.block3.x_fresh | scaling_factor | fresh | 16384 | 28 |  | 28 | True |
| layer4.block3.inv_2n_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer4.block3.x_inv_2n_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer4.block3.square_rescales[0] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer4.block3.square_rescales[1] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer4.block3.square_rescales[2] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer4.block3.square_rescales[3] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer4.block3.square_rescales[4] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer4.block3.square_rescales[5] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer4.block3.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer5.block3.degree | parameter |  |  |  |  | 6 | True |
| layer5.block3.x_fresh | scaling_factor | fresh | 16384 | 28 |  | 28 | True |
| layer5.block3.inv_2n_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer5.block3.x_inv_2n_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer5.block3.square_rescales[0] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer5.block3.square_rescales[1] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer5.block3.square_rescales[2] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer5.block3.square_rescales[3] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer5.block3.square_rescales[4] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer5.block3.square_rescales[5] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer5.block3.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer6.block3.degree | parameter |  |  |  |  | 6 | True |
| layer6.block3.x_fresh | scaling_factor | fresh | 16384 | 28 |  | 28 | True |
| layer6.block3.inv_2n_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer6.block3.x_inv_2n_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer6.block3.square_rescales[0] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer6.block3.square_rescales[1] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer6.block3.square_rescales[2] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer6.block3.square_rescales[3] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer6.block3.square_rescales[4] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer6.block3.square_rescales[5] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer6.block3.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer7.block3.degree | parameter |  |  |  |  | 6 | True |
| layer7.block3.x_fresh | scaling_factor | fresh | 16384 | 28 |  | 28 | True |
| layer7.block3.inv_2n_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer7.block3.x_inv_2n_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer7.block3.square_rescales[0] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer7.block3.square_rescales[1] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer7.block3.square_rescales[2] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer7.block3.square_rescales[3] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer7.block3.square_rescales[4] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer7.block3.square_rescales[5] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer7.block3.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer8.block3.degree | parameter |  |  |  |  | 6 | True |
| layer8.block3.x_fresh | scaling_factor | fresh | 16384 | 28 |  | 28 | True |
| layer8.block3.inv_2n_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer8.block3.x_inv_2n_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer8.block3.square_rescales[0] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer8.block3.square_rescales[1] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer8.block3.square_rescales[2] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer8.block3.square_rescales[3] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer8.block3.square_rescales[4] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer8.block3.square_rescales[5] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer8.block3.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer9.block3.degree | parameter |  |  |  |  | 6 | True |
| layer9.block3.x_fresh | scaling_factor | fresh | 16384 | 28 |  | 28 | True |
| layer9.block3.inv_2n_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer9.block3.x_inv_2n_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer9.block3.square_rescales[0] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer9.block3.square_rescales[1] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer9.block3.square_rescales[2] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer9.block3.square_rescales[3] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer9.block3.square_rescales[4] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer9.block3.square_rescales[5] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
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
| layer11.block3.degree | parameter |  |  |  |  | 6 | True |
| layer11.block3.x_fresh | scaling_factor | fresh | 16384 | 28 |  | 28 | True |
| layer11.block3.inv_2n_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer11.block3.x_inv_2n_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer11.block3.square_rescales[0] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer11.block3.square_rescales[1] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer11.block3.square_rescales[2] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer11.block3.square_rescales[3] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer11.block3.square_rescales[4] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer11.block3.square_rescales[5] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer11.block3.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer0.block4.softmax_out_fresh | scaling_factor | fresh | 16384 | 21 |  | 21 | True |
| layer0.block4.softmax_out_mask_encode | scaling_factor | encoding | 16384 | 13 |  | 13 | True |
| layer0.block4.v_fresh | scaling_factor | fresh | 16384 | 17 |  | 17 | True |
| layer0.block4.v_mask_encode | scaling_factor | encoding | 16384 | 13 |  | 13 | True |
| layer0.block4.softmax_v_mask_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
| layer0.block4.wo_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
| layer0.block4.ln_mean_inv_d_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
| layer0.block4.ln_var_inv_d_encode | scaling_factor | encoding | 16384 | 21 |  | 21 | True |
| layer0.block4.softmax_out_mask_rescale | scaling_factor |  |  |  |  |  | False |
| layer0.block4.v_mask_rescale | scaling_factor |  |  |  |  |  | False |
| layer0.block4.softmax_v_matmul_rescale | scaling_factor |  |  |  |  |  | False |
| layer0.block4.softmax_v_mask_rescale | scaling_factor |  |  |  |  |  | False |
| layer0.block4.wo_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer0.block4.ln_mean_result_rescale | scaling_factor | rescale | 16384 | 46 |  | 46 | True |
| layer0.block4.ln_square_result_rescale | scaling_factor | rescale | 16384 | 32 |  | 32 | True |
| layer0.block4.ln_var_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer0.block4.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer1.block4.softmax_out_fresh | scaling_factor | fresh | 16384 | 35 |  | 35 | True |
| layer1.block4.softmax_out_mask_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
| layer1.block4.v_fresh | scaling_factor | fresh | 16384 | 25 |  | 25 | True |
| layer1.block4.v_mask_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
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
| layer1.block4.ln_square_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer1.block4.ln_var_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer1.block4.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer2.block4.softmax_out_fresh | scaling_factor | fresh | 16384 | 35 |  | 35 | True |
| layer2.block4.softmax_out_mask_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
| layer2.block4.v_fresh | scaling_factor | fresh | 16384 | 25 |  | 25 | True |
| layer2.block4.v_mask_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
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
| layer2.block4.ln_square_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer2.block4.ln_var_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer2.block4.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer3.block4.softmax_out_fresh | scaling_factor | fresh | 16384 | 35 |  | 35 | True |
| layer3.block4.softmax_out_mask_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
| layer3.block4.v_fresh | scaling_factor | fresh | 16384 | 25 |  | 25 | True |
| layer3.block4.v_mask_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
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
| layer3.block4.ln_square_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer3.block4.ln_var_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer3.block4.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer4.block4.softmax_out_fresh | scaling_factor | fresh | 16384 | 35 |  | 35 | True |
| layer4.block4.softmax_out_mask_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
| layer4.block4.v_fresh | scaling_factor | fresh | 16384 | 25 |  | 25 | True |
| layer4.block4.v_mask_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
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
| layer4.block4.ln_square_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer4.block4.ln_var_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer4.block4.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer5.block4.softmax_out_fresh | scaling_factor | fresh | 16384 | 35 |  | 35 | True |
| layer5.block4.softmax_out_mask_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
| layer5.block4.v_fresh | scaling_factor | fresh | 16384 | 25 |  | 25 | True |
| layer5.block4.v_mask_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
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
| layer5.block4.ln_square_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer5.block4.ln_var_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer5.block4.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer6.block4.softmax_out_fresh | scaling_factor | fresh | 16384 | 21 |  | 21 | True |
| layer6.block4.softmax_out_mask_encode | scaling_factor | encoding | 16384 | 13 |  | 13 | True |
| layer6.block4.v_fresh | scaling_factor | fresh | 16384 | 17 |  | 17 | True |
| layer6.block4.v_mask_encode | scaling_factor | encoding | 16384 | 13 |  | 13 | True |
| layer6.block4.softmax_v_mask_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
| layer6.block4.wo_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
| layer6.block4.ln_mean_inv_d_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
| layer6.block4.ln_var_inv_d_encode | scaling_factor | encoding | 16384 | 21 |  | 21 | True |
| layer6.block4.softmax_out_mask_rescale | scaling_factor |  |  |  |  |  | False |
| layer6.block4.v_mask_rescale | scaling_factor |  |  |  |  |  | False |
| layer6.block4.softmax_v_matmul_rescale | scaling_factor |  |  |  |  |  | False |
| layer6.block4.softmax_v_mask_rescale | scaling_factor |  |  |  |  |  | False |
| layer6.block4.wo_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer6.block4.ln_mean_result_rescale | scaling_factor | rescale | 16384 | 46 |  | 46 | True |
| layer6.block4.ln_square_result_rescale | scaling_factor | rescale | 16384 | 32 |  | 32 | True |
| layer6.block4.ln_var_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer6.block4.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer7.block4.softmax_out_fresh | scaling_factor | fresh | 16384 | 35 |  | 35 | True |
| layer7.block4.softmax_out_mask_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
| layer7.block4.v_fresh | scaling_factor | fresh | 16384 | 25 |  | 25 | True |
| layer7.block4.v_mask_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
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
| layer7.block4.ln_square_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer7.block4.ln_var_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer7.block4.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer8.block4.softmax_out_fresh | scaling_factor | fresh | 16384 | 35 |  | 35 | True |
| layer8.block4.softmax_out_mask_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
| layer8.block4.v_fresh | scaling_factor | fresh | 16384 | 25 |  | 25 | True |
| layer8.block4.v_mask_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
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
| layer8.block4.ln_square_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer8.block4.ln_var_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer8.block4.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer9.block4.softmax_out_fresh | scaling_factor | fresh | 16384 | 35 |  | 35 | True |
| layer9.block4.softmax_out_mask_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
| layer9.block4.v_fresh | scaling_factor | fresh | 16384 | 25 |  | 25 | True |
| layer9.block4.v_mask_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
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
| layer9.block4.ln_square_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer9.block4.ln_var_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer9.block4.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer10.block4.softmax_out_fresh | scaling_factor | fresh | 16384 | 35 |  | 35 | True |
| layer10.block4.softmax_out_mask_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
| layer10.block4.v_fresh | scaling_factor | fresh | 16384 | 25 |  | 25 | True |
| layer10.block4.v_mask_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
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
| layer10.block4.ln_square_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer10.block4.ln_var_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer10.block4.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer11.block4.softmax_out_fresh | scaling_factor | fresh | 16384 | 35 |  | 35 | True |
| layer11.block4.softmax_out_mask_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
| layer11.block4.v_fresh | scaling_factor | fresh | 16384 | 25 |  | 25 | True |
| layer11.block4.v_mask_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
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
| layer11.block4.ln_square_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer11.block4.ln_var_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer11.block4.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer0.block5.inv_std_fresh | scaling_factor | fresh | 8192 | 31 |  | 31 | True |
| layer0.block5.x_centered_fresh | scaling_factor | fresh | 8192 | 31 |  | 31 | True |
| layer0.block5.gamma_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer0.block5.wffn1_encode | scaling_factor | encoding | 8192 | 22 |  | 22 | True |
| layer0.block5.gelu_degree | parameter |  |  |  |  | 1 | True |
| layer0.block5.gelu_coeff_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer0.block5.normalize_result_rescale | scaling_factor | rescale | 8192 | 28 |  | 28 | True |
| layer0.block5.gamma_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer0.block5.wffn1_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer0.block5.gelu_power_rescales | scaling_factor_tuple |  |  |  |  | [] | False |
| layer0.block5.gelu_coeff_mul_rescales[0] | scaling_factor | rescale | 8192 | 31 |  | 31 | True |
| layer0.block5.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer1.block5.inv_std_fresh | scaling_factor | fresh | 16384 | 31 |  | 31 | True |
| layer1.block5.x_centered_fresh | scaling_factor | fresh | 16384 | 31 |  | 31 | True |
| layer1.block5.gamma_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer1.block5.wffn1_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer1.block5.gelu_degree | parameter |  |  |  |  | 2 | True |
| layer1.block5.gelu_coeff_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer1.block5.normalize_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer1.block5.gamma_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer1.block5.wffn1_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer1.block5.gelu_power_rescales[0] | scaling_factor |  |  |  |  |  | False |
| layer1.block5.gelu_coeff_mul_rescales[0] | scaling_factor |  |  |  |  |  | False |
| layer1.block5.gelu_coeff_mul_rescales[1] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer1.block5.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer2.block5.inv_std_fresh | scaling_factor | fresh | 8192 | 31 |  | 31 | True |
| layer2.block5.x_centered_fresh | scaling_factor | fresh | 8192 | 31 |  | 31 | True |
| layer2.block5.gamma_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer2.block5.wffn1_encode | scaling_factor | encoding | 8192 | 22 |  | 22 | True |
| layer2.block5.gelu_degree | parameter |  |  |  |  | 1 | True |
| layer2.block5.gelu_coeff_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer2.block5.normalize_result_rescale | scaling_factor | rescale | 8192 | 28 |  | 28 | True |
| layer2.block5.gamma_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer2.block5.wffn1_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer2.block5.gelu_power_rescales | scaling_factor_tuple |  |  |  |  | [] | False |
| layer2.block5.gelu_coeff_mul_rescales[0] | scaling_factor | rescale | 8192 | 31 |  | 31 | True |
| layer2.block5.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer3.block5.inv_std_fresh | scaling_factor | fresh | 8192 | 31 |  | 31 | True |
| layer3.block5.x_centered_fresh | scaling_factor | fresh | 8192 | 31 |  | 31 | True |
| layer3.block5.gamma_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer3.block5.wffn1_encode | scaling_factor | encoding | 8192 | 22 |  | 22 | True |
| layer3.block5.gelu_degree | parameter |  |  |  |  | 1 | True |
| layer3.block5.gelu_coeff_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer3.block5.normalize_result_rescale | scaling_factor | rescale | 8192 | 28 |  | 28 | True |
| layer3.block5.gamma_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer3.block5.wffn1_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer3.block5.gelu_power_rescales | scaling_factor_tuple |  |  |  |  | [] | False |
| layer3.block5.gelu_coeff_mul_rescales[0] | scaling_factor | rescale | 8192 | 31 |  | 31 | True |
| layer3.block5.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer4.block5.inv_std_fresh | scaling_factor | fresh | 8192 | 31 |  | 31 | True |
| layer4.block5.x_centered_fresh | scaling_factor | fresh | 8192 | 31 |  | 31 | True |
| layer4.block5.gamma_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer4.block5.wffn1_encode | scaling_factor | encoding | 8192 | 22 |  | 22 | True |
| layer4.block5.gelu_degree | parameter |  |  |  |  | 1 | True |
| layer4.block5.gelu_coeff_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer4.block5.normalize_result_rescale | scaling_factor | rescale | 8192 | 28 |  | 28 | True |
| layer4.block5.gamma_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer4.block5.wffn1_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer4.block5.gelu_power_rescales | scaling_factor_tuple |  |  |  |  | [] | False |
| layer4.block5.gelu_coeff_mul_rescales[0] | scaling_factor | rescale | 8192 | 31 |  | 31 | True |
| layer4.block5.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer5.block5.inv_std_fresh | scaling_factor | fresh | 8192 | 31 |  | 31 | True |
| layer5.block5.x_centered_fresh | scaling_factor | fresh | 8192 | 31 |  | 31 | True |
| layer5.block5.gamma_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer5.block5.wffn1_encode | scaling_factor | encoding | 8192 | 22 |  | 22 | True |
| layer5.block5.gelu_degree | parameter |  |  |  |  | 1 | True |
| layer5.block5.gelu_coeff_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer5.block5.normalize_result_rescale | scaling_factor | rescale | 8192 | 28 |  | 28 | True |
| layer5.block5.gamma_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer5.block5.wffn1_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer5.block5.gelu_power_rescales | scaling_factor_tuple |  |  |  |  | [] | False |
| layer5.block5.gelu_coeff_mul_rescales[0] | scaling_factor | rescale | 8192 | 31 |  | 31 | True |
| layer5.block5.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer6.block5.inv_std_fresh | scaling_factor | fresh | 8192 | 31 |  | 31 | True |
| layer6.block5.x_centered_fresh | scaling_factor | fresh | 8192 | 31 |  | 31 | True |
| layer6.block5.gamma_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer6.block5.wffn1_encode | scaling_factor | encoding | 8192 | 22 |  | 22 | True |
| layer6.block5.gelu_degree | parameter |  |  |  |  | 1 | True |
| layer6.block5.gelu_coeff_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer6.block5.normalize_result_rescale | scaling_factor | rescale | 8192 | 28 |  | 28 | True |
| layer6.block5.gamma_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer6.block5.wffn1_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer6.block5.gelu_power_rescales | scaling_factor_tuple |  |  |  |  | [] | False |
| layer6.block5.gelu_coeff_mul_rescales[0] | scaling_factor | rescale | 8192 | 31 |  | 31 | True |
| layer6.block5.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer7.block5.inv_std_fresh | scaling_factor | fresh | 8192 | 31 |  | 31 | True |
| layer7.block5.x_centered_fresh | scaling_factor | fresh | 8192 | 31 |  | 31 | True |
| layer7.block5.gamma_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer7.block5.wffn1_encode | scaling_factor | encoding | 8192 | 22 |  | 22 | True |
| layer7.block5.gelu_degree | parameter |  |  |  |  | 1 | True |
| layer7.block5.gelu_coeff_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer7.block5.normalize_result_rescale | scaling_factor | rescale | 8192 | 28 |  | 28 | True |
| layer7.block5.gamma_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer7.block5.wffn1_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer7.block5.gelu_power_rescales | scaling_factor_tuple |  |  |  |  | [] | False |
| layer7.block5.gelu_coeff_mul_rescales[0] | scaling_factor | rescale | 8192 | 31 |  | 31 | True |
| layer7.block5.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer8.block5.inv_std_fresh | scaling_factor | fresh | 16384 | 31 |  | 31 | True |
| layer8.block5.x_centered_fresh | scaling_factor | fresh | 16384 | 31 |  | 31 | True |
| layer8.block5.gamma_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer8.block5.wffn1_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer8.block5.gelu_degree | parameter |  |  |  |  | 2 | True |
| layer8.block5.gelu_coeff_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer8.block5.normalize_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer8.block5.gamma_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer8.block5.wffn1_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer8.block5.gelu_power_rescales[0] | scaling_factor |  |  |  |  |  | False |
| layer8.block5.gelu_coeff_mul_rescales[0] | scaling_factor |  |  |  |  |  | False |
| layer8.block5.gelu_coeff_mul_rescales[1] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer8.block5.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer9.block5.inv_std_fresh | scaling_factor | fresh | 8192 | 31 |  | 31 | True |
| layer9.block5.x_centered_fresh | scaling_factor | fresh | 8192 | 31 |  | 31 | True |
| layer9.block5.gamma_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer9.block5.wffn1_encode | scaling_factor | encoding | 8192 | 22 |  | 22 | True |
| layer9.block5.gelu_degree | parameter |  |  |  |  | 1 | True |
| layer9.block5.gelu_coeff_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer9.block5.normalize_result_rescale | scaling_factor | rescale | 8192 | 28 |  | 28 | True |
| layer9.block5.gamma_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer9.block5.wffn1_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer9.block5.gelu_power_rescales | scaling_factor_tuple |  |  |  |  | [] | False |
| layer9.block5.gelu_coeff_mul_rescales[0] | scaling_factor | rescale | 8192 | 31 |  | 31 | True |
| layer9.block5.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer10.block5.inv_std_fresh | scaling_factor | fresh | 8192 | 31 |  | 31 | True |
| layer10.block5.x_centered_fresh | scaling_factor | fresh | 8192 | 31 |  | 31 | True |
| layer10.block5.gamma_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer10.block5.wffn1_encode | scaling_factor | encoding | 8192 | 22 |  | 22 | True |
| layer10.block5.gelu_degree | parameter |  |  |  |  | 1 | True |
| layer10.block5.gelu_coeff_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer10.block5.normalize_result_rescale | scaling_factor | rescale | 8192 | 28 |  | 28 | True |
| layer10.block5.gamma_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer10.block5.wffn1_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer10.block5.gelu_power_rescales | scaling_factor_tuple |  |  |  |  | [] | False |
| layer10.block5.gelu_coeff_mul_rescales[0] | scaling_factor | rescale | 8192 | 31 |  | 31 | True |
| layer10.block5.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer11.block5.inv_std_fresh | scaling_factor | fresh | 8192 | 31 |  | 31 | True |
| layer11.block5.x_centered_fresh | scaling_factor | fresh | 8192 | 31 |  | 31 | True |
| layer11.block5.gamma_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer11.block5.wffn1_encode | scaling_factor | encoding | 8192 | 22 |  | 22 | True |
| layer11.block5.gelu_degree | parameter |  |  |  |  | 1 | True |
| layer11.block5.gelu_coeff_encode | scaling_factor | encoding | 8192 | 20 |  | 20 | True |
| layer11.block5.normalize_result_rescale | scaling_factor | rescale | 8192 | 28 |  | 28 | True |
| layer11.block5.gamma_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer11.block5.wffn1_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer11.block5.gelu_power_rescales | scaling_factor_tuple |  |  |  |  | [] | False |
| layer11.block5.gelu_coeff_mul_rescales[0] | scaling_factor | rescale | 8192 | 31 |  | 31 | True |
| layer11.block5.output_truncation_k | truncation |  |  |  | 13 | 13 | True |

