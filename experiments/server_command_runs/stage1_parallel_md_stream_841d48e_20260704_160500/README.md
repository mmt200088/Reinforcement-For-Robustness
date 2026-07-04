# Stage-1 Parallel Markdown Streaming Evidence

Source commit: `841d48e`

Optimization:

- `scripts/stage1_parallel_report.py` now writes `--out-md` reports through
  `write_markdown_file()` and `iter_markdown_lines()` instead of materializing
  one full `render_markdown()` string and writing it with `Path.write_text()`.
- `render_markdown()` remains available for stdout and compatibility paths.

Server paths:

- RED: `/hy-tmp/stage1_parallel_md_stream_red_0e14a02_20260704_160500`
- GREEN: `/hy-tmp/stage1_parallel_md_stream_green_20260704_160500`

Verification:

- RED: `python3 -m unittest tests.test_stage1_parallel_report -v`
  - `red/red.rc`: `1`
  - Expected failure:
    `AssertionError: CLI markdown output should not render one full string`
- GREEN: `python3 -m py_compile scripts/stage1_parallel_report.py tests/test_stage1_parallel_report.py`
  - `green/py_compile.rc`: `0`
- GREEN: `python3 -m unittest tests.test_stage1_parallel_report -v`
  - `green/green.rc`: `0`
  - `Ran 10 tests`
