# Optimizer Modes Markdown Streaming Evidence

Source commit: `83a317e`

Optimization:
- `scripts/blb_compare_optimizer_modes.py` now writes `optimizer_mode_comparison.md` directly through a file handle instead of building a full `lines` list and writing `"\n".join(lines)` with `Path.write_text()`.
- The JSON artifact and stdout JSON summary path are unchanged.

Server paths:
- RED: `/hy-tmp/optimizer_modes_md_stream_red_1dab527_20260704_142248`
- GREEN: `/hy-tmp/optimizer_modes_md_stream_green_20260704_142248`

Verification:
- RED: `python3 -m unittest tests.test_blb_compare_optimizer_modes_static -v`
  - `red/red.rc`: `1`
  - Expected failure: `_write_markdown()` lacked `.open()` and still used joined `Path.write_text()`.
- GREEN: `python3 -m py_compile scripts/blb_compare_optimizer_modes.py`
  - `green/py_compile.rc`: `0`
- GREEN: `python3 -m unittest tests.test_blb_compare_optimizer_modes_static -v`
  - `green/green.rc`: `0`
  - `Ran 2 tests`
