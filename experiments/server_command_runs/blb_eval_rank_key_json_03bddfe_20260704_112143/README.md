# BLB Eval Rank Key JSON Streaming

Source commit: `03bddfe`

This run verifies that `scripts/blb_eval_action.py` writes `rank_key.json`
through the shared `write_json_file()` helper instead of materializing a full
`json.dumps(...)` string and passing it to `Path.write_text()`.

Server temporary sources:

- RED: `/hy-tmp/blb_eval_rank_key_red_bef48c8_20260704_112033`
- GREEN: `/hy-tmp/blb_eval_rank_key_green2_bef48c8_20260704_112143`

Verification:

- `red.rc`: `1`, expected failure on old source because `rank_key.json` still
  used `write_text(json.dumps(...))`.
- `green.rc`: `py_compile_rc=0`, `test_rc=0`.
- `green.log`: full `tests.test_blb_eval_action_static` passed with 2 tests.
