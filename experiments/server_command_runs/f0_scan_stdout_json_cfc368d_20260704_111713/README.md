# F0 Scan Stdout JSON Streaming

Source commit: `cfc368d`

This run verifies that `scripts/blb_f0_scan_feasible_domain.py` streams its
final CLI summary directly through `json.dump(..., sys.stdout, ...)` instead of
materializing a full `json.dumps(...)` string before printing.

Server temporary sources:

- RED: `/hy-tmp/f0_scan_stdout_red_f938b98_20260704_111632`
- GREEN: `/hy-tmp/f0_scan_stdout_green_f938b98_20260704_111713`

Verification:

- `red.rc`: `1`, expected failure on old source because `main()` still used
  `print(json.dumps(...))` for the final summary.
- `green.rc`: `py_compile_rc=0`, `test_rc=0`.
- `green.log`: full `tests.test_blb_f0_scan` passed with 8 tests.
