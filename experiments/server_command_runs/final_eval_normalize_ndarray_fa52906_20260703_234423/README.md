# Final-Eval ndarray Normalization Evidence

Source commit: `fa52906` (`Optimize final eval array normalization`)

This directory contains server-side red/green evidence for avoiding extra
`list()` materialization when final-eval configuration normalization receives
NumPy arrays.

## Red

Directory:
`rfr_final_eval_normalize_ndarray_red_b46b894_20260703_234201/`

Command target:
`python -m unittest tests.test_final_eval_normalize_arrays -v`

Result:
`red_rc=1`

The new tests failed on the pre-change implementation because both
`_normalize_config_array()` and `_normalize_noise_array()` called
`np.asarray(list(values), dtype=int)` for ndarray inputs.

## Green

Directory:
`rfr_final_eval_normalize_ndarray_green_b46b894_20260703_234423/`

Command targets:

- `python -m py_compile final_evaluation_module.py tests/test_final_eval_normalize_arrays.py`
- `python -m unittest tests.test_final_eval_normalize_arrays tests.test_paean_action_grid -v`

Result:

- `green_py_compile_rc=0`
- `green_unittest_rc=0`
- `Ran 6 tests ... OK`

The green run keeps existing Paean action-grid cache behavior covered while
verifying ndarray normalization avoids unnecessary Python list allocation.
