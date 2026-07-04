# GPU Utilization Markdown Streaming Evidence

Source commit: `6875648`

Optimization:

- `scripts/gpu_utilization_report.py` now writes `--out-md` reports through
  `write_markdown_file()` and `iter_markdown_lines()` instead of materializing
  one full `render_markdown()` string and writing it with `Path.write_text()`.
- `render_markdown()` remains available for stdout and compatibility paths.
- The source also restores the narrow `_iter_jsonl()` compatibility wrapper so
  the existing GPU utilization report unit tests can run as a full module.

Server paths:

- RED: `/hy-tmp/gpu_util_md_stream_red_db550c2_20260704_154530`
- GREEN: `/hy-tmp/gpu_util_md_stream_green_20260704_154500`

Verification:

- RED: focused `python3 -m unittest` for the new CLI streaming test and adjacent
  Markdown behavior tests
  - `red/red.rc`: `1`
  - Expected failure:
    `AssertionError: CLI markdown output should not render one full string`
- GREEN: `python3 -m py_compile scripts/gpu_utilization_report.py tests/test_gpu_utilization_report.py`
  - `green/py_compile.rc`: `0`
- GREEN: `python3 -m unittest tests.test_gpu_utilization_report -v`
  - `green/green.rc`: `0`
  - `Ran 16 tests`
