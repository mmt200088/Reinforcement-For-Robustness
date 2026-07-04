# Block4 Fusion Diagnosis Stdout JSON Streaming

Source commit: `032d2c1`

This run verifies that `scripts/diagnose_block4_fusion_install.py` streams its
CLI output summary directly through `json.dump(..., sys.stdout, ...)` instead
of materializing a full `json.dumps(...)` string before printing.

Server temporary sources:

- RED: `/hy-tmp/block4_diag_stdout_red_a737ec6_20260704_112742`
- GREEN: `/hy-tmp/block4_diag_stdout_green_a737ec6_20260704_112821`

Verification:

- `red.rc`: `1`, expected failure on old source because `main()` still used
  `print(json.dumps(...))`.
- `green.rc`: `py_compile_rc=0`, `test_rc=0`.
- `green.log`: `tests.test_diagnose_block4_fusion_install_static` passed.
