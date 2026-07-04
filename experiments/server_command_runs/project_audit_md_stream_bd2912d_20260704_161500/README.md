# Project Optimization Audit Markdown Streaming Evidence

Source commit: `bd2912d`

Optimization:

- `scripts/project_optimization_audit.py` now writes `--out-md` reports through
  `write_markdown_file()` and `iter_markdown_lines()` instead of materializing
  one full `render_markdown()` string and writing it with `Path.write_text()`.
- `render_markdown()` remains available for stdout and compatibility paths.

Server paths:

- RED: `/hy-tmp/project_audit_md_stream_red_c655efb_20260704_161500`
- GREEN: `/hy-tmp/project_audit_md_stream_green_20260704_161500`

Verification:

- RED: `python3 -m unittest tests.test_project_optimization_audit -v`
  - `red/red.rc`: `1`
  - Expected failure:
    `AssertionError: CLI markdown output should not render one full string`
- GREEN: `python3 -m py_compile scripts/project_optimization_audit.py tests/test_project_optimization_audit.py`
  - `green/py_compile.rc`: `0`
- GREEN: `python3 -m unittest tests.test_project_optimization_audit -v`
  - `green/green.rc`: `0`
  - `Ran 9 tests`
