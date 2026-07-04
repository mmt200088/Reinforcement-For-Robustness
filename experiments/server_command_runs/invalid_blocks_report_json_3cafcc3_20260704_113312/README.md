# Invalid Blocks Diagnosis Report JSON Streaming

Source commit: `3cafcc3`

This run verifies that `scripts/blb_diagnose_invalid_blocks.py` writes
`report.json` through the shared `write_json_file()` helper instead of
materializing a full `json.dumps(...)` string and passing it to
`Path.write_text()`.

Server temporary sources:

- RED: `/hy-tmp/invalid_blocks_report_red_ab6e95d_20260704_113055`
- GREEN: `/hy-tmp/invalid_blocks_report_green2_ab6e95d_20260704_113312`

Verification:

- `red.rc`: `1`, expected failure on old source because `report.json` still
  used `write_text(json.dumps(...))`.
- `green.rc`: `py_compile_rc=0`, `test_rc=0`.
- `green.log`: `tests.test_blb_diagnose_invalid_blocks_static` passed.
