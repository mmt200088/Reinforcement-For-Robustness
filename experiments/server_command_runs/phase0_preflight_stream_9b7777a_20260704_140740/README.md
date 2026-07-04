# Phase0 Preflight Streaming Evidence

Source commit: `9b7777a`

Optimization:
- `scripts/blb_phase0_preflight.py` now streams Phase-0 preflight output files line by line instead of joining full output lists before `Path.write_text()`.
- `phase0_entrypoints.md` is written from `iter_phase0_entrypoint_report_lines()` so the CLI path does not need the compatibility `build_phase0_entrypoint_report()` string helper.

Server paths:
- RED: `/hy-tmp/phase0_preflight_stream_red_bb67b9c_20260704_140740`
- GREEN: `/hy-tmp/phase0_preflight_stream_green_20260704_140740`

Verification:
- RED: `python3 -m unittest tests.test_blb_phase0_preflight -v`
  - `red/red.rc`: `1`
  - Expected failures: old `_write_lines()` still used `Path.write_text()`, and old `phase0_entrypoints.md` writing still called `build_phase0_entrypoint_report()`.
- GREEN: `python3 -m py_compile scripts/blb_phase0_preflight.py`
  - `green/py_compile.rc`: `0`
- GREEN: `python3 -m unittest tests.test_blb_phase0_preflight -v`
  - `green/green.rc`: `0`
  - `Ran 4 tests`
