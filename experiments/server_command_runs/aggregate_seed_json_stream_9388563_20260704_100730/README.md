# Aggregate Seed JSON Streaming Verification

Source commit: `9388563`

Server temp root: `/hy-tmp/rfr_aggregate_seed_json_stream_20260704_100730`

Purpose:

- Avoid building the full `seed_summary.json` payload as
  `[asdict(row) for row in seed_rows]` before writing multi-seed summaries.
- `tools/aggregate_seeds.py` now streams the top-level JSON array row by row
  through `json.JSONEncoder.iterencode()`.
- The Markdown summary path and JSON field schema stay unchanged.

Server verification:

- RED package: old `tools/aggregate_seeds.py` plus the new regression test.
  - Command: `python3 -m unittest
    tests.test_aggregate_seeds.AggregateSeedsFinalEvalTest.test_main_writes_json_without_materializing_full_row_list
    -v`
  - Result: expected failure at the old full-list `json.dump(...)` path.
  - Log: `red.log`
- GREEN package: source commit `9388563` changes plus tests.
  - Command: same single regression test.
  - Result: OK.
  - Log: `green.log`
- Wider GREEN:
  - Command: `python3 -m py_compile tools/aggregate_seeds.py
    tests/test_aggregate_seeds.py && python3 -m unittest
    tests.test_aggregate_seeds -v`
  - Result: OK, 8 tests.
  - Log: `green_full.log`
