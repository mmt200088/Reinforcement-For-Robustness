# BLB Action Final Evaluation Report

- dataset: `mrpc`
- split: `validation_full`
- selected_source: `blb_action(stage1=json)`
- repeat_n: `1`
- rescale_optimizer: `in_process`
- rescale_optimizer_root: `/hy-tmp/RFR_k6_13_eb19c6af_20260727/Rescale_optimizer`
- json: `/hy-tmp/stage2_k6_k7_gate_74c5bda4_20260727_214945/paean/mrpc/rl/k6-k7-production-gate-b7ce0bfd/final_eval/blb_action_final_eval_results_mrpc.json`

## Baseline

- clean baseline loss: `0.383963`
- clean baseline Acc.: `0.877451`
- clean baseline F1: `0.874422`
- clean baseline protocol: `single_validation_full`
- clean baseline loss std: `0.000000`
- clean baseline Acc. std: `0.000000`
- clean baseline F1 std: `0.000000`

## Selected vs Cost-Matched Random Comparison

- selected (`ActionSelected`): loss=0.369498 ± 0.000000, Acc.=0.887255 ± 0.000000, F1=0.885099 ± 0.000000, total_bits=14461, fusion=0, avg_k=12.783

## Group Comparison

| group | truncation k | effective K positions | loss mean | loss std | Acc. mean | Acc. std | F1 mean | F1 std | time mean ms | total bits | fusion | replan applied | model cfg verified |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `ActionSelected` | 6,7,13 | 60 | 0.369498 | 0.000000 | 0.887255 | 0.000000 | 0.885099 | 0.000000 | 337.722 | 14461 | 0 | True | True |

## Configuration Details

### ActionSelected

- action overrides: `{}`
- base action: BLB RL baseline action: model-side non-truncation fields use highest selectable action-space SF; Rescale_optimizer mode=cfg_derived.
- truncation summary: `6,7,13`; effective positions = `60`; skipped = `[]`
- model cfg verified before forward: `True`
- replan cfg applied before forward: `True`
- replan application summary: `{'applied_before_forward': True, 'model_uses_replan_config': True, 'expected_config_count': 59, 'applied_config_count': 59, 'invalid_config_count': 0, 'missing_compact_config_count': 0, 'missing_decoded_cfg_count': 0, 'apply_error_count': 0, 'override_total': 146, 'optimizer_cfg_overrides': {'block2_mrpc_L0': [('rotation_after_gamma_rescale', 'rotation_flag', False, True), ('rotation_after_kt_mask1_rescale', 'rotation_flag', False, True), ('rotation_after_q_mask1_rescale', 'rotation_flag', False, True), ('rotation_after_qkt_matmul_rescale', 'rotation_flag', False, True), ('rotation_repeat_counts.rotation_after_gamma_rescale', 'rotation_count', 0, 1), ('rotation_repeat_counts.rotation_after_kt_mask1_rescale', 'rotation_count', 0, 1), ('rotation_repeat_counts.rotation_after_q_mask1_rescale', 'rotation_count', 0, 1), ('rotation_repeat_counts.rotation_after_qkt_matmul_rescale', 'rotation_count', 0, 1)], 'block2_mrpc_L1': [('rotation_after_gamma_rescale', 'rotation_flag', False, True), ('rotation_after_kt_mask1_rescale', 'rotation_flag', False, True), ('rotation_after_q_mask1_rescale', 'rotation_flag', False, True), ('rotation_after_qkt_matmul_rescale', 'rotation_flag', False, True), ('rotation_repeat_counts.rotation_after_gamma_rescale', 'rotation_count', 0, 1), ('rotation_repeat_counts.rotation_after_kt_mask1_rescale', 'rotation_count', 0, 1), ('rotation_repeat_counts.rotation_after_q_mask1_rescale', 'rotation_count', 0, 1), ('rotation_repeat_counts.rotation_after_qkt_matmul_rescale', 'rotation_count', 0, 1)], 'block2_mrpc_L2': [('rotation_after_gamma_rescale', 'rotation_flag', False, True), ('rotation_after_kt_mask1_rescale', 'rotation_flag', False, True), ('rotation_after_q_mask1_rescale', 'rotation_flag', False, True), ('rotation_after_qkt_matmul_rescale', 'rotation_flag', False, True), ('rotation_repeat_counts.rotation_after_gamma_rescale', 'rotation_count', 0, 1), ('rotation_repeat_counts.rotation_after_kt_mask1_rescale', 'rotation_count', 0, 1), ('rotation_repeat_counts.rotation_after_q_mask1_rescale', 'rotation_count', 0, 1), ('rotation_repeat_counts.rotation_after_qkt_matmul_rescale', 'rotation_count', 0, 1)], 'block2_mrpc_L3': [('rotation_after_gamma_rescale', 'rotation_flag', False, True), ('rotation_after_kt_mask1_rescale', 'rotation_flag', False, True), ('rotation_after_q_mask1_rescale', 'rotation_flag', False, True), ('rotation_after_qkt_matmul_rescale', 'rotation_flag', False, True), ('rotation_repeat_counts.rotation_after_gamma_rescale', 'rotation_count', 0, 1), ('rotation_repeat_counts.rotation_after_kt_mask1_rescale', 'rotation_count', 0, 1), ('rotation_repeat_counts.rotation_after_q_mask1_rescale', 'rotation_count', 0, 1), ('rotation_repeat_counts.rotation_after_qkt_matmul_rescale', 'rotation_count', 0, 1)], 'block2_mrpc_L4': [('rotation_after_gamma_rescale', 'rotation_flag', False, True), ('rotation_after_kt_mask1_rescale', 'rotation_flag', False, True), ('rotation_after_q_mask1_rescale', 'rotation_flag', False, True), ('rotation_after_qkt_matmul_rescale', 'rotation_flag', False, True), ('rotation_repeat_counts.rotation_after_gamma_rescale', 'rotation_count', 0, 1), ('rotation_repeat_counts.rotation_after_kt_mask1_rescale', 'rotation_count', 0, 1), ('rotation_repeat_counts.rotation_after_q_mask1_rescale', 'rotation_count', 0, 1), ('rotation_repeat_counts.rotation_after_qkt_matmul_rescale', 'rotation_count', 0, 1)], 'block2_mrpc_L5': [('rotation_after_gamma_rescale', 'rotation_flag', False, True), ('rotation_after_kt_mask1_rescale', 'rotation_flag', False, True), ('rotation_after_q_mask1_rescale', 'rotation_flag', False, True), ('rotation_after_qkt_matmul_rescale', 'rotation_flag', False, True), ('rotation_repeat_counts.rotation_after_gamma_rescale', 'rotation_count', 0, 1), ('rotation_repeat_counts.rotation_after_kt_mask1_rescale', 'rotation_count', 0, 1), ('rotation_repeat_counts.rotation_after_q_mask1_rescale', 'rotation_count', 0, 1), ('rotation_repeat_counts.rotation_after_qkt_matmul_rescale', 'rotation_count', 0, 1)], 'block2_mrpc_L6': [('rotation_after_gamma_rescale', 'rotation_flag', False, True), ('rotation_after_kt_mask1_rescale', 'rotation_flag', False, True), ('rotation_after_q_mask1_rescale', 'rotation_flag', False, True), ('rotation_after_qkt_matmul_rescale', 'rotation_flag', False, True), ('rotation_repeat_counts.rotation_after_gamma_rescale', 'rotation_count', 0, 1), ('rotation_repeat_counts.rotation_after_kt_mask1_rescale', 'rotation_count', 0, 1), ('rotation_repeat_counts.rotation_after_q_mask1_rescale', 'rotation_count', 0, 1), ('rotation_repeat_counts.rotation_after_qkt_matmul_rescale', 'rotation_count', 0, 1)], 'block2_mrpc_L7': [('rotation_after_gamma_rescale', 'rotation_flag', False, True), ('rotation_after_kt_mask1_rescale', 'rotation_flag', False, True), ('rotation_after_q_mask1_rescale', 'rotation_flag', False, True), ('rotation_after_qkt_matmul_rescale', 'rotation_flag', False, True), ('rotation_repeat_counts.rotation_after_gamma_rescale', 'rotation_count', 0, 1), ('rotation_repeat_counts.rotation_after_kt_mask1_rescale', 'rotation_count', 0, 1), ('rotation_repeat_counts.rotation_after_q_mask1_rescale', 'rotation_count', 0, 1), ('rotation_repeat_counts.rotation_after_qkt_matmul_rescale', 'rotation_count', 0, 1)], 'block2_mrpc_L8': [('rotation_after_gamma_rescale', 'rotation_flag', False, True), ('rotation_after_kt_mask1_rescale', 'rotation_flag', False, True), ('rotation_after_q_mask1_rescale', 'rotation_flag', False, True), ('rotation_after_qkt_matmul_rescale', 'rotation_flag', False, True), ('rotation_repeat_counts.rotation_after_gamma_rescale', 'rotation_count', 0, 1), ('rotation_repeat_counts.rotation_after_kt_mask1_rescale', 'rotation_count', 0, 1), ('rotation_repeat_counts.rotation_after_q_mask1_rescale', 'rotation_count', 0, 1), ('rotation_repeat_counts.rotation_after_qkt_matmul_rescale', 'rotation_count', 0, 1)], 'block2_mrpc_L9': [('rotation_after_gamma_rescale', 'rotation_flag', False, True), ('rotation_after_kt_mask1_rescale', 'rotation_flag', False, True), ('rotation_after_q_mask1_rescale', 'rotation_flag', False, True), ('rotation_after_qkt_matmul_rescale', 'rotation_flag', False, True), ('rotation_repeat_counts.rotation_after_gamma_rescale', 'rotation_count', 0, 1), ('rotation_repeat_counts.rotation_after_kt_mask1_rescale', 'rotation_count', 0, 1), ('rotation_repeat_counts.rotation_after_q_mask1_rescale', 'rotation_count', 0, 1), ('rotation_repeat_counts.rotation_after_qkt_matmul_rescale', 'rotation_count', 0, 1)], 'block2_mrpc_L10': [('rotation_after_gamma_rescale', 'rotation_flag', False, True), ('rotation_after_kt_mask1_rescale', 'rotation_flag', False, True), ('rotation_after_q_mask1_rescale', 'rotation_flag', False, True), ('rotation_after_qkt_matmul_rescale', 'rotation_flag', False, True), ('rotation_repeat_counts.rotation_after_gamma_rescale', 'rotation_count', 0, 1), ('rotation_repeat_counts.rotation_after_kt_mask1_rescale', 'rotation_count', 0, 1), ('rotation_repeat_counts.rotation_after_q_mask1_rescale', 'rotation_count', 0, 1), ('rotation_repeat_counts.rotation_after_qkt_matmul_rescale', 'rotation_count', 0, 1)], 'block2_mrpc_L11': [('rotation_after_gamma_rescale', 'rotation_flag', False, True), ('rotation_after_kt_mask1_rescale', 'rotation_flag', False, True), ('rotation_after_q_mask1_rescale', 'rotation_flag', False, True), ('rotation_after_qkt_matmul_rescale', 'rotation_flag', False, True), ('rotation_repeat_counts.rotation_after_gamma_rescale', 'rotation_count', 0, 1), ('rotation_repeat_counts.rotation_after_kt_mask1_rescale', 'rotation_count', 0, 1), ('rotation_repeat_counts.rotation_after_q_mask1_rescale', 'rotation_count', 0, 1), ('rotation_repeat_counts.rotation_after_qkt_matmul_rescale', 'rotation_count', 0, 1)], 'block4_L0': [('rotation_after_ln_square_rescale', 'rotation_flag', False, True), ('rotation_after_softmax_v_matmul_rescale', 'rotation_flag', False, True), ('rotation_repeat_counts.rotation_after_ln_square_rescale', 'rotation_count', 0, 3), ('rotation_repeat_counts.rotation_after_softmax_v_matmul_rescale', 'rotation_count', 0, 1)], 'block4_L1': [('rotation_after_ln_square_rescale', 'rotation_flag', False, True), ('rotation_after_softmax_v_matmul_rescale', 'rotation_flag', False, True), ('rotation_repeat_counts.rotation_after_ln_square_rescale', 'rotation_count', 0, 3), ('rotation_repeat_counts.rotation_after_softmax_v_matmul_rescale', 'rotation_count', 0, 1)], 'block4_L2': [('rotation_after_ln_square_rescale', 'rotation_flag', False, True), ('rotation_after_softmax_v_matmul_rescale', 'rotation_flag', False, True), ('rotation_repeat_counts.rotation_after_ln_square_rescale', 'rotation_count', 0, 3), ('rotation_repeat_counts.rotation_after_softmax_v_matmul_rescale', 'rotation_count', 0, 1)], 'block4_L3': [('rotation_after_ln_square_rescale', 'rotation_flag', False, True), ('rotation_after_softmax_v_matmul_rescale', 'rotation_flag', False, True), ('rotation_repeat_counts.rotation_after_ln_square_rescale', 'rotation_count', 0, 3), ('rotation_repeat_counts.rotation_after_softmax_v_matmul_rescale', 'rotation_count', 0, 1)], 'block4_L4': [('rotation_after_ln_square_rescale', 'rotation_flag', False, True), ('rotation_after_softmax_v_matmul_rescale', 'rotation_flag', False, True), ('rotation_repeat_counts.rotation_after_ln_square_rescale', 'rotation_count', 0, 3), ('rotation_repeat_counts.rotation_after_softmax_v_matmul_rescale', 'rotation_count', 0, 1)], 'block4_L5': [('rotation_after_ln_square_rescale', 'rotation_flag', False, True), ('rotation_after_softmax_v_matmul_rescale', 'rotation_flag', False, True), ('rotation_repeat_counts.rotation_after_ln_square_rescale', 'rotation_count', 0, 3), ('rotation_repeat_counts.rotation_after_softmax_v_matmul_rescale', 'rotation_count', 0, 1)], 'block4_L6': [('rotation_after_ln_square_rescale', 'rotation_flag', False, True), ('rotation_after_softmax_v_matmul_rescale', 'rotation_flag', False, True), ('rotation_repeat_counts.rotation_after_ln_square_rescale', 'rotation_count', 0, 3), ('rotation_repeat_counts.rotation_after_softmax_v_matmul_rescale', 'rotation_count', 0, 1)], 'block4_L7': [('rotation_after_ln_square_rescale', 'rotation_flag', False, True), ('rotation_after_softmax_v_matmul_rescale', 'rotation_flag', False, True), ('rotation_repeat_counts.rotation_after_ln_square_rescale', 'rotation_count', 0, 3), ('rotation_repeat_counts.rotation_after_softmax_v_matmul_rescale', 'rotation_count', 0, 1)], 'block4_L8': [('rotation_after_ln_square_rescale', 'rotation_flag', False, True), ('rotation_after_softmax_v_matmul_rescale', 'rotation_flag', False, True), ('rotation_repeat_counts.rotation_after_ln_square_rescale', 'rotation_count', 0, 3), ('rotation_repeat_counts.rotation_after_softmax_v_matmul_rescale', 'rotation_count', 0, 1)], 'block4_L9': [('rotation_after_ln_square_rescale', 'rotation_flag', False, True), ('rotation_after_softmax_v_matmul_rescale', 'rotation_flag', False, True), ('rotation_repeat_counts.rotation_after_ln_square_rescale', 'rotation_count', 0, 3), ('rotation_repeat_counts.rotation_after_softmax_v_matmul_rescale', 'rotation_count', 0, 1)], 'block4_L10': [('rotation_after_ln_square_rescale', 'rotation_flag', False, True), ('rotation_after_softmax_v_matmul_rescale', 'rotation_flag', False, True), ('rotation_repeat_counts.rotation_after_ln_square_rescale', 'rotation_count', 0, 3), ('rotation_repeat_counts.rotation_after_softmax_v_matmul_rescale', 'rotation_count', 0, 1)], 'block4_L11': [('rotation_after_ln_square_rescale', 'rotation_flag', False, True), ('rotation_after_softmax_v_matmul_rescale', 'rotation_flag', False, True), ('rotation_repeat_counts.rotation_after_ln_square_rescale', 'rotation_count', 0, 3), ('rotation_repeat_counts.rotation_after_softmax_v_matmul_rescale', 'rotation_count', 0, 1)], 'block5_n4_L5': [('rotation_after_wffn1_rescale', 'rotation_flag', False, True), ('rotation_repeat_counts.rotation_after_wffn1_rescale', 'rotation_count', 0, 1)]}, 'expected_output_names': ['block1_mrpc_L1', 'block1_mrpc_L10', 'block1_mrpc_L11', 'block1_mrpc_L2', 'block1_mrpc_L3', 'block1_mrpc_L4', 'block1_mrpc_L5', 'block1_mrpc_L6', 'block1_mrpc_L7', 'block1_mrpc_L8', 'block1_mrpc_L9', 'block2_mrpc_L0', 'block2_mrpc_L1', 'block2_mrpc_L10', 'block2_mrpc_L11', 'block2_mrpc_L2', 'block2_mrpc_L3', 'block2_mrpc_L4', 'block2_mrpc_L5', 'block2_mrpc_L6', 'block2_mrpc_L7', 'block2_mrpc_L8', 'block2_mrpc_L9', 'block3_exp_n2_L0', 'block3_exp_n2_L1', 'block3_exp_n2_L11', 'block3_exp_n2_L5', 'block3_exp_n2_L7', 'block3_exp_n5_L2', 'block3_exp_n5_L3', 'block3_exp_n5_L4', 'block3_exp_n5_L6', 'block3_exp_n5_L8', 'block3_exp_n5_L9', 'block3_exp_n6_L10', 'block4_L0', 'block4_L1', 'block4_L10', 'block4_L11', 'block4_L2', 'block4_L3', 'block4_L4', 'block4_L5', 'block4_L6', 'block4_L7', 'block4_L8', 'block4_L9', 'block5_n1_L0', 'block5_n1_L1', 'block5_n1_L10', 'block5_n1_L11', 'block5_n1_L2', 'block5_n1_L3', 'block5_n1_L4', 'block5_n1_L6', 'block5_n1_L7', 'block5_n1_L8', 'block5_n1_L9', 'block5_n4_L5'], 'actual_output_names': ['block1_mrpc_L1', 'block1_mrpc_L10', 'block1_mrpc_L11', 'block1_mrpc_L2', 'block1_mrpc_L3', 'block1_mrpc_L4', 'block1_mrpc_L5', 'block1_mrpc_L6', 'block1_mrpc_L7', 'block1_mrpc_L8', 'block1_mrpc_L9', 'block2_mrpc_L0', 'block2_mrpc_L1', 'block2_mrpc_L10', 'block2_mrpc_L11', 'block2_mrpc_L2', 'block2_mrpc_L3', 'block2_mrpc_L4', 'block2_mrpc_L5', 'block2_mrpc_L6', 'block2_mrpc_L7', 'block2_mrpc_L8', 'block2_mrpc_L9', 'block3_exp_n2_L0', 'block3_exp_n2_L1', 'block3_exp_n2_L11', 'block3_exp_n2_L5', 'block3_exp_n2_L7', 'block3_exp_n5_L2', 'block3_exp_n5_L3', 'block3_exp_n5_L4', 'block3_exp_n5_L6', 'block3_exp_n5_L8', 'block3_exp_n5_L9', 'block3_exp_n6_L10', 'block4_L0', 'block4_L1', 'block4_L10', 'block4_L11', 'block4_L2', 'block4_L3', 'block4_L4', 'block4_L5', 'block4_L6', 'block4_L7', 'block4_L8', 'block4_L9', 'block5_n1_L0', 'block5_n1_L1', 'block5_n1_L10', 'block5_n1_L11', 'block5_n1_L2', 'block5_n1_L3', 'block5_n1_L4', 'block5_n1_L6', 'block5_n1_L7', 'block5_n1_L8', 'block5_n1_L9', 'block5_n4_L5'], 'missing_optimizer_outputs': [], 'unexpected_optimizer_outputs': [], 'optimizer_output_set_matches': True, 'missing_baseline_skeletons': [], 'all_baseline_skeletons_available': True}`
- fusion group diagnostics: `{}`
- handler active layers match expected: `True`
- handler cfg object identity match: `True`
- rescale optimizer: `{'invoker_kind': 'in_process', 'root': '/hy-tmp/RFR_k6_13_eb19c6af_20260727/Rescale_optimizer', 'mode': 'cfg_derived', 'request_count': 59, 'valid_count': 59, 'invalid_count': 0, 't_new_sources': ['cfg_derived']}`

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
      30,
      34
    ],
    "var_inv_d_encode": [
      20
    ],
    "var_result_rescale": [
      27,
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
      15,
      16
    ],
    "square_rescales": [
      31,
      34,
      35
    ],
    "x_fresh": [
      27,
      28,
      31
    ]
  },
  "block4": {
    "ln_mean_inv_d_encode": [
      20
    ],
    "ln_mean_result_rescale": [
      31
    ],
    "ln_square_result_rescale": [
      31
    ],
    "ln_var_inv_d_encode": [
      20
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
      25
    ],
    "v_mask_encode": [
      14
    ],
    "wo_encode": [
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
    "gelu_power_rescales": [
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
| layer0.block1.gelu_out_fresh | scaling_factor | fresh | 16384 | 30 |  | 30 | True |
| layer0.block1.wffn2_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer0.block1.mean_inv_d_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer0.block1.var_inv_d_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer0.block1.wffn2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer0.block1.mean_result_rescale | scaling_factor | rescale | 16384 | 34 |  | 34 | True |
| layer0.block1.square_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer0.block1.var_result_rescale | scaling_factor | rescale | 16384 | 34 |  | 34 | True |
| layer0.block1.output_truncation_k | truncation |  |  |  | 6 | 6 | True |
| layer1.block1.gelu_out_fresh | scaling_factor | fresh | 16384 | 30 |  | 30 | True |
| layer1.block1.wffn2_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer1.block1.mean_inv_d_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer1.block1.var_inv_d_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer1.block1.wffn2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer1.block1.mean_result_rescale | scaling_factor | rescale | 16384 | 30 |  | 30 | True |
| layer1.block1.square_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer1.block1.var_result_rescale | scaling_factor | rescale | 16384 | 27 |  | 27 | True |
| layer1.block1.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer2.block1.gelu_out_fresh | scaling_factor | fresh | 16384 | 30 |  | 30 | True |
| layer2.block1.wffn2_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer2.block1.mean_inv_d_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer2.block1.var_inv_d_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer2.block1.wffn2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer2.block1.mean_result_rescale | scaling_factor | rescale | 16384 | 30 |  | 30 | True |
| layer2.block1.square_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer2.block1.var_result_rescale | scaling_factor | rescale | 16384 | 27 |  | 27 | True |
| layer2.block1.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer3.block1.gelu_out_fresh | scaling_factor | fresh | 16384 | 30 |  | 30 | True |
| layer3.block1.wffn2_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer3.block1.mean_inv_d_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer3.block1.var_inv_d_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer3.block1.wffn2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer3.block1.mean_result_rescale | scaling_factor | rescale | 16384 | 30 |  | 30 | True |
| layer3.block1.square_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer3.block1.var_result_rescale | scaling_factor | rescale | 16384 | 27 |  | 27 | True |
| layer3.block1.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer4.block1.gelu_out_fresh | scaling_factor | fresh | 16384 | 30 |  | 30 | True |
| layer4.block1.wffn2_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer4.block1.mean_inv_d_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer4.block1.var_inv_d_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer4.block1.wffn2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer4.block1.mean_result_rescale | scaling_factor | rescale | 16384 | 30 |  | 30 | True |
| layer4.block1.square_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer4.block1.var_result_rescale | scaling_factor | rescale | 16384 | 27 |  | 27 | True |
| layer4.block1.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer5.block1.gelu_out_fresh | scaling_factor | fresh | 16384 | 30 |  | 30 | True |
| layer5.block1.wffn2_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer5.block1.mean_inv_d_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer5.block1.var_inv_d_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer5.block1.wffn2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer5.block1.mean_result_rescale | scaling_factor | rescale | 16384 | 30 |  | 30 | True |
| layer5.block1.square_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer5.block1.var_result_rescale | scaling_factor | rescale | 16384 | 27 |  | 27 | True |
| layer5.block1.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer6.block1.gelu_out_fresh | scaling_factor | fresh | 16384 | 30 |  | 30 | True |
| layer6.block1.wffn2_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer6.block1.mean_inv_d_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer6.block1.var_inv_d_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer6.block1.wffn2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer6.block1.mean_result_rescale | scaling_factor | rescale | 16384 | 30 |  | 30 | True |
| layer6.block1.square_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer6.block1.var_result_rescale | scaling_factor | rescale | 16384 | 27 |  | 27 | True |
| layer6.block1.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer7.block1.gelu_out_fresh | scaling_factor | fresh | 16384 | 30 |  | 30 | True |
| layer7.block1.wffn2_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer7.block1.mean_inv_d_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer7.block1.var_inv_d_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer7.block1.wffn2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer7.block1.mean_result_rescale | scaling_factor | rescale | 16384 | 30 |  | 30 | True |
| layer7.block1.square_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer7.block1.var_result_rescale | scaling_factor | rescale | 16384 | 27 |  | 27 | True |
| layer7.block1.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer8.block1.gelu_out_fresh | scaling_factor | fresh | 16384 | 30 |  | 30 | True |
| layer8.block1.wffn2_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer8.block1.mean_inv_d_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer8.block1.var_inv_d_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer8.block1.wffn2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer8.block1.mean_result_rescale | scaling_factor | rescale | 16384 | 30 |  | 30 | True |
| layer8.block1.square_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer8.block1.var_result_rescale | scaling_factor | rescale | 16384 | 27 |  | 27 | True |
| layer8.block1.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer9.block1.gelu_out_fresh | scaling_factor | fresh | 16384 | 30 |  | 30 | True |
| layer9.block1.wffn2_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer9.block1.mean_inv_d_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer9.block1.var_inv_d_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer9.block1.wffn2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer9.block1.mean_result_rescale | scaling_factor | rescale | 16384 | 30 |  | 30 | True |
| layer9.block1.square_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer9.block1.var_result_rescale | scaling_factor | rescale | 16384 | 27 |  | 27 | True |
| layer9.block1.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer10.block1.gelu_out_fresh | scaling_factor | fresh | 16384 | 30 |  | 30 | True |
| layer10.block1.wffn2_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer10.block1.mean_inv_d_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer10.block1.var_inv_d_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer10.block1.wffn2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer10.block1.mean_result_rescale | scaling_factor | rescale | 16384 | 30 |  | 30 | True |
| layer10.block1.square_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer10.block1.var_result_rescale | scaling_factor | rescale | 16384 | 27 |  | 27 | True |
| layer10.block1.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer11.block1.gelu_out_fresh | scaling_factor | fresh | 16384 | 30 |  | 30 | True |
| layer11.block1.wffn2_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer11.block1.mean_inv_d_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer11.block1.var_inv_d_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer11.block1.wffn2_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer11.block1.mean_result_rescale | scaling_factor | rescale | 16384 | 30 |  | 30 | True |
| layer11.block1.square_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer11.block1.var_result_rescale | scaling_factor | rescale | 16384 | 27 |  | 27 | True |
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
| layer0.block2.output_truncation_k | truncation |  |  |  | 7 | 7 | True |
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
| layer0.block3.degree | parameter |  |  |  |  | 2 | True |
| layer0.block3.x_fresh | scaling_factor | fresh | 16384 | 27 |  | 27 | True |
| layer0.block3.inv_2n_encode | scaling_factor | encoding | 16384 | 16 |  | 16 | True |
| layer0.block3.x_inv_2n_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer0.block3.square_rescales[0] | scaling_factor | rescale | 16384 | 34 |  | 34 | True |
| layer0.block3.square_rescales[1] | scaling_factor | rescale | 16384 | 34 |  | 34 | True |
| layer0.block3.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer1.block3.degree | parameter |  |  |  |  | 2 | True |
| layer1.block3.x_fresh | scaling_factor | fresh | 16384 | 27 |  | 27 | True |
| layer1.block3.inv_2n_encode | scaling_factor | encoding | 16384 | 16 |  | 16 | True |
| layer1.block3.x_inv_2n_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer1.block3.square_rescales[0] | scaling_factor | rescale | 16384 | 34 |  | 34 | True |
| layer1.block3.square_rescales[1] | scaling_factor | rescale | 16384 | 34 |  | 34 | True |
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
| layer5.block3.x_fresh | scaling_factor | fresh | 16384 | 27 |  | 27 | True |
| layer5.block3.inv_2n_encode | scaling_factor | encoding | 16384 | 16 |  | 16 | True |
| layer5.block3.x_inv_2n_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer5.block3.square_rescales[0] | scaling_factor | rescale | 16384 | 34 |  | 34 | True |
| layer5.block3.square_rescales[1] | scaling_factor | rescale | 16384 | 34 |  | 34 | True |
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
| layer7.block3.x_fresh | scaling_factor | fresh | 16384 | 27 |  | 27 | True |
| layer7.block3.inv_2n_encode | scaling_factor | encoding | 16384 | 16 |  | 16 | True |
| layer7.block3.x_inv_2n_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer7.block3.square_rescales[0] | scaling_factor | rescale | 16384 | 34 |  | 34 | True |
| layer7.block3.square_rescales[1] | scaling_factor | rescale | 16384 | 34 |  | 34 | True |
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
| layer10.block3.x_fresh | scaling_factor | fresh | 16384 | 31 |  | 31 | True |
| layer10.block3.inv_2n_encode | scaling_factor | encoding | 16384 | 15 |  | 15 | True |
| layer10.block3.x_inv_2n_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer10.block3.square_rescales[0] | scaling_factor | rescale | 16384 | 35 |  | 35 | True |
| layer10.block3.square_rescales[1] | scaling_factor | rescale | 16384 | 35 |  | 35 | True |
| layer10.block3.square_rescales[2] | scaling_factor | rescale | 16384 | 35 |  | 35 | True |
| layer10.block3.square_rescales[3] | scaling_factor | rescale | 16384 | 35 |  | 35 | True |
| layer10.block3.square_rescales[4] | scaling_factor | rescale | 16384 | 35 |  | 35 | True |
| layer10.block3.square_rescales[5] | scaling_factor | rescale | 16384 | 35 |  | 35 | True |
| layer10.block3.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer11.block3.degree | parameter |  |  |  |  | 2 | True |
| layer11.block3.x_fresh | scaling_factor | fresh | 16384 | 27 |  | 27 | True |
| layer11.block3.inv_2n_encode | scaling_factor | encoding | 16384 | 16 |  | 16 | True |
| layer11.block3.x_inv_2n_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer11.block3.square_rescales[0] | scaling_factor | rescale | 16384 | 34 |  | 34 | True |
| layer11.block3.square_rescales[1] | scaling_factor | rescale | 16384 | 34 |  | 34 | True |
| layer11.block3.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer0.block4.softmax_out_fresh | scaling_factor | fresh | 16384 | 35 |  | 35 | True |
| layer0.block4.softmax_out_mask_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
| layer0.block4.v_fresh | scaling_factor | fresh | 16384 | 25 |  | 25 | True |
| layer0.block4.v_mask_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
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
| layer0.block4.ln_square_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
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
| layer6.block4.softmax_out_fresh | scaling_factor | fresh | 16384 | 35 |  | 35 | True |
| layer6.block4.softmax_out_mask_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
| layer6.block4.v_fresh | scaling_factor | fresh | 16384 | 25 |  | 25 | True |
| layer6.block4.v_mask_encode | scaling_factor | encoding | 16384 | 14 |  | 14 | True |
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
| layer6.block4.ln_square_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
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
| layer0.block5.inv_std_fresh | scaling_factor | fresh | 16384 | 31 |  | 31 | True |
| layer0.block5.x_centered_fresh | scaling_factor | fresh | 16384 | 31 |  | 31 | True |
| layer0.block5.gamma_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer0.block5.wffn1_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer0.block5.gelu_degree | parameter |  |  |  |  | 1 | True |
| layer0.block5.gelu_coeff_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer0.block5.normalize_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer0.block5.gamma_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer0.block5.wffn1_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer0.block5.gelu_power_rescales | scaling_factor_tuple |  |  |  |  | [] | False |
| layer0.block5.gelu_coeff_mul_rescales[0] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer0.block5.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer1.block5.inv_std_fresh | scaling_factor | fresh | 16384 | 31 |  | 31 | True |
| layer1.block5.x_centered_fresh | scaling_factor | fresh | 16384 | 31 |  | 31 | True |
| layer1.block5.gamma_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer1.block5.wffn1_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer1.block5.gelu_degree | parameter |  |  |  |  | 1 | True |
| layer1.block5.gelu_coeff_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer1.block5.normalize_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer1.block5.gamma_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer1.block5.wffn1_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer1.block5.gelu_power_rescales | scaling_factor_tuple |  |  |  |  | [] | False |
| layer1.block5.gelu_coeff_mul_rescales[0] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer1.block5.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer2.block5.inv_std_fresh | scaling_factor | fresh | 16384 | 31 |  | 31 | True |
| layer2.block5.x_centered_fresh | scaling_factor | fresh | 16384 | 31 |  | 31 | True |
| layer2.block5.gamma_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer2.block5.wffn1_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer2.block5.gelu_degree | parameter |  |  |  |  | 1 | True |
| layer2.block5.gelu_coeff_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer2.block5.normalize_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer2.block5.gamma_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer2.block5.wffn1_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer2.block5.gelu_power_rescales | scaling_factor_tuple |  |  |  |  | [] | False |
| layer2.block5.gelu_coeff_mul_rescales[0] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer2.block5.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer3.block5.inv_std_fresh | scaling_factor | fresh | 16384 | 31 |  | 31 | True |
| layer3.block5.x_centered_fresh | scaling_factor | fresh | 16384 | 31 |  | 31 | True |
| layer3.block5.gamma_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer3.block5.wffn1_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer3.block5.gelu_degree | parameter |  |  |  |  | 1 | True |
| layer3.block5.gelu_coeff_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer3.block5.normalize_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer3.block5.gamma_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer3.block5.wffn1_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer3.block5.gelu_power_rescales | scaling_factor_tuple |  |  |  |  | [] | False |
| layer3.block5.gelu_coeff_mul_rescales[0] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer3.block5.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer4.block5.inv_std_fresh | scaling_factor | fresh | 16384 | 31 |  | 31 | True |
| layer4.block5.x_centered_fresh | scaling_factor | fresh | 16384 | 31 |  | 31 | True |
| layer4.block5.gamma_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer4.block5.wffn1_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer4.block5.gelu_degree | parameter |  |  |  |  | 1 | True |
| layer4.block5.gelu_coeff_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer4.block5.normalize_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer4.block5.gamma_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer4.block5.wffn1_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer4.block5.gelu_power_rescales | scaling_factor_tuple |  |  |  |  | [] | False |
| layer4.block5.gelu_coeff_mul_rescales[0] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer4.block5.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer5.block5.inv_std_fresh | scaling_factor | fresh | 16384 | 31 |  | 31 | True |
| layer5.block5.x_centered_fresh | scaling_factor | fresh | 16384 | 31 |  | 31 | True |
| layer5.block5.gamma_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer5.block5.wffn1_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer5.block5.gelu_degree | parameter |  |  |  |  | 4 | True |
| layer5.block5.gelu_coeff_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer5.block5.normalize_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer5.block5.gamma_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer5.block5.wffn1_result_rescale | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer5.block5.gelu_power_rescales[0] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer5.block5.gelu_power_rescales[1] | scaling_factor |  |  |  |  |  | False |
| layer5.block5.gelu_power_rescales[2] | scaling_factor |  |  |  |  |  | False |
| layer5.block5.gelu_coeff_mul_rescales[0] | scaling_factor |  |  |  |  |  | False |
| layer5.block5.gelu_coeff_mul_rescales[1] | scaling_factor |  |  |  |  |  | False |
| layer5.block5.gelu_coeff_mul_rescales[2] | scaling_factor |  |  |  |  |  | False |
| layer5.block5.gelu_coeff_mul_rescales[3] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer5.block5.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer6.block5.inv_std_fresh | scaling_factor | fresh | 16384 | 31 |  | 31 | True |
| layer6.block5.x_centered_fresh | scaling_factor | fresh | 16384 | 31 |  | 31 | True |
| layer6.block5.gamma_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer6.block5.wffn1_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer6.block5.gelu_degree | parameter |  |  |  |  | 1 | True |
| layer6.block5.gelu_coeff_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer6.block5.normalize_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer6.block5.gamma_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer6.block5.wffn1_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer6.block5.gelu_power_rescales | scaling_factor_tuple |  |  |  |  | [] | False |
| layer6.block5.gelu_coeff_mul_rescales[0] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer6.block5.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer7.block5.inv_std_fresh | scaling_factor | fresh | 16384 | 31 |  | 31 | True |
| layer7.block5.x_centered_fresh | scaling_factor | fresh | 16384 | 31 |  | 31 | True |
| layer7.block5.gamma_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer7.block5.wffn1_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer7.block5.gelu_degree | parameter |  |  |  |  | 1 | True |
| layer7.block5.gelu_coeff_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer7.block5.normalize_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer7.block5.gamma_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer7.block5.wffn1_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer7.block5.gelu_power_rescales | scaling_factor_tuple |  |  |  |  | [] | False |
| layer7.block5.gelu_coeff_mul_rescales[0] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer7.block5.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer8.block5.inv_std_fresh | scaling_factor | fresh | 16384 | 31 |  | 31 | True |
| layer8.block5.x_centered_fresh | scaling_factor | fresh | 16384 | 31 |  | 31 | True |
| layer8.block5.gamma_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer8.block5.wffn1_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer8.block5.gelu_degree | parameter |  |  |  |  | 1 | True |
| layer8.block5.gelu_coeff_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer8.block5.normalize_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer8.block5.gamma_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer8.block5.wffn1_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer8.block5.gelu_power_rescales | scaling_factor_tuple |  |  |  |  | [] | False |
| layer8.block5.gelu_coeff_mul_rescales[0] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer8.block5.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer9.block5.inv_std_fresh | scaling_factor | fresh | 16384 | 31 |  | 31 | True |
| layer9.block5.x_centered_fresh | scaling_factor | fresh | 16384 | 31 |  | 31 | True |
| layer9.block5.gamma_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer9.block5.wffn1_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer9.block5.gelu_degree | parameter |  |  |  |  | 1 | True |
| layer9.block5.gelu_coeff_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer9.block5.normalize_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer9.block5.gamma_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer9.block5.wffn1_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer9.block5.gelu_power_rescales | scaling_factor_tuple |  |  |  |  | [] | False |
| layer9.block5.gelu_coeff_mul_rescales[0] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer9.block5.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer10.block5.inv_std_fresh | scaling_factor | fresh | 16384 | 31 |  | 31 | True |
| layer10.block5.x_centered_fresh | scaling_factor | fresh | 16384 | 31 |  | 31 | True |
| layer10.block5.gamma_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer10.block5.wffn1_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer10.block5.gelu_degree | parameter |  |  |  |  | 1 | True |
| layer10.block5.gelu_coeff_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer10.block5.normalize_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer10.block5.gamma_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer10.block5.wffn1_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer10.block5.gelu_power_rescales | scaling_factor_tuple |  |  |  |  | [] | False |
| layer10.block5.gelu_coeff_mul_rescales[0] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer10.block5.output_truncation_k | truncation |  |  |  | 13 | 13 | True |
| layer11.block5.inv_std_fresh | scaling_factor | fresh | 16384 | 31 |  | 31 | True |
| layer11.block5.x_centered_fresh | scaling_factor | fresh | 16384 | 31 |  | 31 | True |
| layer11.block5.gamma_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer11.block5.wffn1_encode | scaling_factor | encoding | 16384 | 22 |  | 22 | True |
| layer11.block5.gelu_degree | parameter |  |  |  |  | 1 | True |
| layer11.block5.gelu_coeff_encode | scaling_factor | encoding | 16384 | 20 |  | 20 | True |
| layer11.block5.normalize_result_rescale | scaling_factor | rescale | 16384 | 28 |  | 28 | True |
| layer11.block5.gamma_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer11.block5.wffn1_result_rescale | scaling_factor |  |  |  |  |  | False |
| layer11.block5.gelu_power_rescales | scaling_factor_tuple |  |  |  |  | [] | False |
| layer11.block5.gelu_coeff_mul_rescales[0] | scaling_factor | rescale | 16384 | 31 |  | 31 | True |
| layer11.block5.output_truncation_k | truncation |  |  |  | 13 | 13 | True |

