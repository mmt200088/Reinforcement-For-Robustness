# Fusion RL-Path Eval Stdout JSON Streaming

Source commit: `d125fd4`

This run verifies that `scripts/run_fusion_count_action_eval_rlpath.py` streams
its final CLI summary directly through `json.dump(..., sys.stdout, ...)`
instead of materializing a full `json.dumps(...)` string before printing.

Server temporary sources:

- RED: `/hy-tmp/fusion_rlpath_stdout_red_0a4ba12_20260704_111224`
- GREEN: `/hy-tmp/fusion_rlpath_stdout_green_0a4ba12_20260704_111300`

Verification:

- `red.rc`: `1`, expected failure on old source because `main()` still used
  `print(json.dumps(...))` for the final summary.
- `green.rc`: `py_compile_rc=0`, `test_rc=0`.
- `green.log`: full `tests.test_run_fusion_count_action_eval_rlpath` passed
  with 13 tests.
