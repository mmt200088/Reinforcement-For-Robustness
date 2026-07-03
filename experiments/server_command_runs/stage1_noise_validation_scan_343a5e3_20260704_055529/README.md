# stage1_noise_validation_scan_343a5e3_20260704_055529

Source commit: `343a5e3`

Optimization: shared Stage-1/layer-evaluator noise scaling validation now uses
`_unsupported_int_values()` to scan normalized numpy arrays for unsupported
integer scaling factors. This replaces repeated `arr.tolist()` materialization
and set construction in input-noise, weight-noise, and softmax/value-noise
validation paths.

Server workflow:

- Red package: `/hy-tmp/rfr_stage1_noise_validation_red_55ab9ed_20260704_`
- Green package: `/hy-tmp/rfr_stage1_noise_validation_green_20260704_`
- Server canonical worktree was not modified.

Verification:

- `red_unittest.log`: expected failure on
  `test_noise_scaling_validation_scans_arrays_without_tolist_materialization`,
  proving the helper path was absent before the optimization.
- `green_validation.log`: `python3 -m py_compile` passed,
  `tests.test_stage1_parallel_semantics` passed 14 tests, and the source guard
  confirmed the old validation materialization statements are gone.
