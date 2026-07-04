# Optimizer Mode Comparison Stdout JSON Streaming

Source commit: `46df691`

This run verifies that `scripts/blb_compare_optimizer_modes.py` streams its CLI
summary directly through `json.dump(..., sys.stdout, ...)` instead of
materializing a full `json.dumps(...)` string before printing.

Server temporary sources:

- RED: `/hy-tmp/optimizer_modes_stdout_red_85e8778_20260704_112416`
- GREEN: `/hy-tmp/optimizer_modes_stdout_green_85e8778_20260704_112453`

Verification:

- `red.rc`: `1`, expected failure on old source because `main()` still used
  `print(json.dumps(...))`.
- `green.rc`: `py_compile_rc=0`, `test_rc=0`.
- `green.log`: `tests.test_blb_compare_optimizer_modes_static` passed.
