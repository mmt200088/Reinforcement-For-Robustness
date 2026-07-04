# BLB Eval Action Stdout JSON Streaming

Source commit: `1b15448`

This run verifies that `scripts/blb_eval_action.py` streams its CLI stdout
candidate record directly through `json.dump(record, sys.stdout, ...)` instead
of materializing a full `json.dumps(...)` string before printing.

Server temporary sources:

- RED: `/hy-tmp/blb_eval_action_stdout_red_63c9710_20260704_110439`
- GREEN: `/hy-tmp/blb_eval_action_stdout_green_63c9710_20260704_110534`

Verification:

- `red.rc`: `1`, expected failure on old source because `main()` still used
  `print(json.dumps(record, ...))`.
- `green.rc`: `py_compile_rc=0`, `test_rc=0`.
- `green.log`: `tests.test_blb_eval_action_static` passed.
