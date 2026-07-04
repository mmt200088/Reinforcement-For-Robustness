# F0 Scan Markdown Streaming Evidence

Source commit: `2edeb35`

Optimization:
- `scripts/blb_f0_scan_feasible_domain.py` now writes `per_slot_summary.md` and `suggested_action_mask.md` through line iterators and `_write_lines()`.
- F0 scan JSON, JSONL, CSV, cost ranking, mask semantics, and random-scan outputs are unchanged.

Server paths:
- RED: `/hy-tmp/f0_scan_md_stream_red_bf78781_20260704_141650`
- GREEN: `/hy-tmp/f0_scan_md_stream_green_20260704_141650`

Verification:
- RED: `python3 -m unittest tests.test_blb_f0_scan -v`
  - `red/red.rc`: `1`
  - Expected failure: `AssertionError: F0 markdown outputs should stream file writes`
  - The failing line was the old `per_slot_summary.md` `Path.write_text("\n".join(...))` path.
- GREEN: `python3 -m py_compile scripts/blb_f0_scan_feasible_domain.py`
  - `green/py_compile.rc`: `0`
- GREEN: `python3 -m unittest tests.test_blb_f0_scan -v`
  - `green/green.rc`: `0`
  - `Ran 9 tests`
