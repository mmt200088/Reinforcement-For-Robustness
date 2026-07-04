# Stage-1 Semantics Gate Evidence

Final source commit: `dbd1b6f`

This evidence covers the Stage-1 focused semantic gate for optimization work
that touched shared evaluation and GELU polynomial fast paths. The first server
gate exposed two regressions at source head `8faf478`:

- `PolynomialGELU._poly()` used Horner evaluation, which avoided the old powers
  stack but exceeded the existing fp32 equivalence tolerance for degree 4,
  sign 1.
- Stage-1 `_run_evaluation()` delegated to the shared installed-model helper,
  which changed Stage-1 loss averaging and weighted-F1 precision relative to
  the historical per-batch-sync loop.

The final source keeps the allocation improvements while restoring the Stage-1
semantic locks:

- `PolynomialGELU._poly()` evaluates in coefficient order with one reusable
  power tensor, avoiding the `(degree+1)` stacked powers tensor while matching
  the existing tolerance gate.
- `run_installed_model_on_dataloader()` keeps default sample-weighted loss for
  Stage-2/Paean/shared users, but supports `loss_average="batch"`.
- Stage-1 `_run_evaluation()` passes `loss_average="batch"` and recomputes the
  legacy sklearn metric pair from the shared helper's collected logits/labels,
  preserving the old exact Stage-1 result semantics.

## Server Verification

- Initial RED run:
  `/hy-tmp/rfr_stage1_semantics_8faf478_20260704_075601`
- Focused loss-average RED run:
  `/hy-tmp/rfr_stage1_gate_red_8faf478_20260704_075934`
- Intermediate GREEN attempt:
  `/hy-tmp/rfr_stage1_gate_green_20260704_080029`
- Final GREEN run:
  `/hy-tmp/rfr_stage1_gate_green_final_dbd1b6f_20260704_080621`

Initial RED command:

```bash
python3 -m unittest tests.test_stage1_eval_accel tests.test_stage1_parallel_semantics -v
```

Initial RED result: `TEST_RC=1`, with failures in
`HornerPolyEquivalenceTest.test_poly_matches_stacked_reference_all_degrees_and_signs`
and `RunEvaluationDeferredSyncTest.test_bit_identical_to_per_batch_sync_loop`.

Focused RED command:

```bash
python3 -m unittest tests.test_blb_inference_eval_shared.SharedInstalledInferenceEvalTest.test_probe_trial_and_full_eval_share_metric_semantics -v
```

Focused RED result: `RED_RC=1` because the old helper did not accept
`loss_average="batch"`.

Final GREEN commands:

```bash
python3 -m py_compile function_handler.py blb_stage2_rl/inference_eval.py layer_importance_evaluator.py tests/test_stage1_eval_accel.py tests/test_stage1_parallel_semantics.py tests/test_blb_inference_eval_shared.py
python3 -m unittest tests.test_stage1_eval_accel.HornerPolyEquivalenceTest tests.test_stage1_eval_accel.RunEvaluationDeferredSyncTest tests.test_blb_inference_eval_shared.SharedInstalledInferenceEvalTest.test_probe_trial_and_full_eval_share_metric_semantics -v
python3 -m unittest tests.test_stage1_eval_accel tests.test_stage1_parallel_semantics tests.test_blb_inference_eval_shared -v
```

Final GREEN result: `PY_COMPILE_RC=0`, focused tests passed (`FOCUS_RC=0`),
full gate passed (`FULL_RC=0`).

## Local Contents

- `initial_stage1_semantics_red.log`: initial server RED gate.
- `stage1_failure_probe.log`: diagnostic quantification of Horner vs
  coefficient-order polynomial differences.
- `loss_average_param_red.log`: focused RED for the new helper parameter.
- `intermediate_green_metric_red.log`: intermediate server attempt that fixed
  loss but still exposed the exact weighted-F1 precision mismatch.
- `precommit_green_validation.log`: successful server validation of the final
  file content before the final source commit was created.
- `green_validation.log`: final server GREEN gate.
- `source_snapshot/`: final source/test snapshots from `dbd1b6f`.
