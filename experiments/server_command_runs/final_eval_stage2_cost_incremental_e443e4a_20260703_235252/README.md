# Final-Eval Stage-2 Cost-Matched Search Evidence

Source commit: `e443e4a` (`Optimize final eval cost matching`)

This directory contains server-side red/green evidence for optimizing
`UnifiedFinalEvaluationModule._stage2_cost_matched_array()`.

## Red

Directory:
`rfr_final_eval_stage2_cost_incremental_red_db96be7_20260703_235029/`

Command target:
`python -m unittest tests.test_final_eval_normalize_arrays -v`

Result:
`red_rc=1`

The new hot-path guard failed on the pre-change implementation because the
inner 500-step adjustment loop recomputed the full configuration cost with
`sum(cost_map[int(d)] for d in cfg)` on every attempted mutation.

## Green

Directory:
`rfr_final_eval_stage2_cost_incremental_green_db96be7_20260703_235252/`

Command targets:

- `python -m py_compile final_evaluation_module.py tests/test_final_eval_normalize_arrays.py`
- `python -m unittest tests.test_final_eval_normalize_arrays tests.test_paean_action_grid -v`

Result:

- `green_py_compile_rc=0`
- `green_unittest_rc=0`
- `Ran 7 tests ... OK`

The green run verifies the cost-matched search keeps current cost
incrementally while preserving the existing final-eval normalization and Paean
action-grid cache behavior.
