# BLB JSON Log Encoder Reuse

Source commit: `5f877ca`

This run verifies that `blb_stage2_rl/logging_helpers.py` reuses one module-level
`json.JSONEncoder` for structured JSON log records. JSON logging can run for
long training/diagnostic processes; reusing the encoder avoids constructing a
new encoder through `json.dumps(...)` for every formatted record.

Server temporary sources:

- RED: `/hy-tmp/logging_json_encoder_red_48c8214_20260704_124000`
- GREEN: `/hy-tmp/logging_json_encoder_green_20260704_124000`

Verification:

- `red.rc`: `1`, expected failure on old source because `_JSONFormatter`
  still returned `json.dumps(payload, ensure_ascii=False, default=str)`.
- `green.rc`: `py_compile_rc=0`, `test_rc=0`.
- `green.log`: `tests.test_logging_helpers_static` passed.
