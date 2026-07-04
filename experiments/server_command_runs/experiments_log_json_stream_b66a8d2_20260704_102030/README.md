# Experiments Query JSON Streaming Verification

Source commit: `b66a8d2`

Server temp root: `/hy-tmp/rfr_experiments_log_json_stream_20260704_102030`

Purpose:

- Avoid materializing the complete CLI JSON string for
  `python -m tools.experiments_log query --format json`.
- `tools/experiments_log.py` now writes query JSON directly to stdout through
  `json.dump(..., sys.stdout)` and then writes the trailing newline.
- Query filtering, returned rows, and JSON indentation remain unchanged.

Server verification:

- RED package: old `tools/experiments_log.py` plus the new regression test.
  - Command: `python3 -m unittest
    tests.test_experiments_log.ExperimentsLogTest.test_query_json_output_streams_without_json_dumps
    -v`
  - Result: expected failure at the old `print(json.dumps(rows, ...))` path.
  - Log: `red.log`
- GREEN package: source commit `b66a8d2` changes plus tests.
  - Command: same single regression test.
  - Result: OK.
  - Log: `green.log`
- Wider GREEN:
  - Command: `python3 -m py_compile tools/experiments_log.py
    tests/test_experiments_log.py jsonl_utils.py json_utils.py &&
    python3 -m unittest tests.test_experiments_log -v`
  - Result: OK, 15 tests.
  - Log: `green_full.log`
