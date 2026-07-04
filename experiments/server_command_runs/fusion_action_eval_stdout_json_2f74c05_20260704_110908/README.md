# Fusion Action Eval Stdout JSON Streaming

Source commit: `2f74c05`

This run verifies that `scripts/run_fusion_count_action_eval.py` streams its
final CLI summary directly through `json.dump(..., sys.stdout, ...)` instead of
materializing a full `json.dumps(...)` string before printing.

Server temporary sources:

- RED: `/hy-tmp/fusion_action_eval_stdout_red_be789a5_20260704_110827`
- GREEN: `/hy-tmp/fusion_action_eval_stdout_green_be789a5_20260704_110908`

Verification:

- `red.rc`: `1`, expected failure on old source because `main()` still used
  `print(json.dumps(...))` for the final summary.
- `green.rc`: `py_compile_rc=0`, `test_rc=0`.
- `green.log`: full `tests.test_run_fusion_count_action_eval` passed with 6
  tests.
