# Server Resource Snapshot Markdown Streaming Evidence

Source commit: `bd99c65`

Optimization:

- `scripts/server_resource_snapshot.py` now writes `--out-md` reports through
  `write_markdown_file()` and `iter_markdown_lines()` instead of materializing
  one full `render_markdown()` string and writing it with `Path.write_text()`.
- `render_markdown()` remains available for stdout and compatibility paths.

Server paths:

- RED: `/hy-tmp/server_snapshot_md_stream_red_0cf4551_20260704_153000`
- GREEN: `/hy-tmp/server_snapshot_md_stream_green_20260704_153000`

Verification:

- RED: `python3 -m unittest tests.test_server_resource_snapshot -v`
  - `red/red.rc`: `1`
  - Expected failure:
    `AssertionError: CLI markdown output should not render one full string`
- GREEN: `python3 -m py_compile scripts/server_resource_snapshot.py tests/test_server_resource_snapshot.py`
  - `green/py_compile.rc`: `0`
- GREEN: `python3 -m unittest tests.test_server_resource_snapshot -v`
  - `green/green.rc`: `0`
  - `Ran 9 tests`
