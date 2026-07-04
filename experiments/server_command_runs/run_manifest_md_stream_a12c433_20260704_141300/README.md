# Run Manifest Markdown Streaming Evidence

Source commit: `a12c433`

Optimization:
- `scripts/blb_make_run_manifest.py` now writes `run_manifest.md` through `iter_manifest_markdown_lines()` and `_write_lines()`.
- The JSON artifact still uses the shared `write_json_file()` helper; only the Markdown summary path changed.

Server paths:
- RED: `/hy-tmp/run_manifest_md_stream_red_bda066f_20260704_141300`
- GREEN: `/hy-tmp/run_manifest_md_stream_green_20260704_141300`

Verification:
- RED: `python3 -m unittest tests.test_blb_make_run_manifest -v`
  - `red/red.rc`: `1`
  - Expected failure: `AssertionError: manifest markdown should stream file writes`
  - The failing line was the old `md_path.write_text("\n".join(lines) + "\n", ...)`.
- GREEN: `python3 -m py_compile scripts/blb_make_run_manifest.py`
  - `green/py_compile.rc`: `0`
- GREEN: `python3 -m unittest tests.test_blb_make_run_manifest -v`
  - `green/green.rc`: `0`
  - `Ran 12 tests`
