# Experiments Register JSON Streaming Verification

Source commit: `c5424cd`

Server temp root: `/hy-tmp/rfr_experiments_log_register_json_stream_20260704_102430`

Purpose:

- Avoid materializing the complete CLI JSON string for
  `python -m tools.experiments_log register`.
- `tools/experiments_log.py` now writes the register result directly to stdout
  through the shared `json.dump(..., sys.stdout)` helper.
- Registry append behavior, emitted JSON fields, and trailing newline stay
  unchanged.

Server verification:

- RED package: old `tools/experiments_log.py` plus the new regression test.
  - Command: `python3 -m unittest
    tests.test_experiments_log.ExperimentsLogTest.test_register_json_output_streams_without_json_dumps
    -v`
  - Result: expected failure at the old `print(json.dumps(rec, ...))` path.
  - Log: `red.log`
- GREEN package: source commit `c5424cd` changes plus tests.
  - Command: same single regression test.
  - Result: OK.
  - Log: `green.log`
- Wider GREEN:
  - Command: `python3 -m py_compile tools/experiments_log.py
    tests/test_experiments_log.py jsonl_utils.py json_utils.py &&
    python3 -m unittest tests.test_experiments_log -v`
  - Result: OK, 16 tests.
  - Log: `green_full.log`
