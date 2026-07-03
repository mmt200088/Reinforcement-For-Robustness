# final_eval_invalid_values_d76cf29_20260704_054400

Source commit: `d76cf29`

Optimization: Paean/final-eval config validation now scans integer arrays for
unsupported values through `_unsupported_int_values()` instead of materializing
the entire normalized numpy array with `arr.tolist()` and building a set from
that list.

Server workflow:

- Red package: `/hy-tmp/rfr_final_invalid_red_8d42b41_20260704_`
- Green package: `/hy-tmp/rfr_final_invalid_green_20260704_`
- Server canonical worktree was not modified.

Verification:

- `red_unittest.log`: expected failure on
  `test_invalid_value_checks_scan_arrays_without_tolist_materialization`, proving
  the old validation path still used `arr.tolist()`.
- `green_validation.log`: `python3 -m py_compile` passed, target unittest passed
  4 tests, and the source guard confirmed the old materialization statement is
  gone while the helper path is present.
