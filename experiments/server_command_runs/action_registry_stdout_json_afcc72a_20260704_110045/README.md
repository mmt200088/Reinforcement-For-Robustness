# Action Registry Stdout JSON Streaming

Source commit: `afcc72a`

This run verifies that `scripts/blb_export_action_registry.py` streams its CLI
stdout path summary directly through `json.dump(paths, sys.stdout, ...)`
instead of materializing a full `json.dumps(...)` string before printing.

Server temporary sources:

- RED: `/hy-tmp/action_registry_stdout_json_red_607932e_20260704_110000`
- GREEN: `/hy-tmp/action_registry_stdout_json_green_607932e_20260704_110045`

Verification:

- `red.rc`: `1`, expected failure on old source because `main()` still used
  `print(json.dumps(paths, ...))`.
- `green.rc`: `py_compile_rc=0`, `test_rc=0`.
- `green.log`: full `tests.test_blb_export_action_registry_light` passed with
  4 tests.
