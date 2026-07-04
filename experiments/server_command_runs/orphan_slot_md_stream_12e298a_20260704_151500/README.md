# Orphan Slot Audit Markdown Streaming Evidence

Source commit: `12e298a`

Optimization:

- `scripts/blb_orphan_slot_audit.py` now writes the top-level Markdown report
  through `_MarkdownLinesWriter` instead of accumulating the full report line
  list and calling `Path.write_text("\n".join(lines))`.
- `tests/test_blb_orphan_slot_audit.py` adds a regression test that rejects
  `Path.write_text` for `audit_mrpc.md` while preserving JSON output.

Server paths:

- RED: `/hy-tmp/orphan_slot_md_stream_red_00ab5c7_20260704_151500`
- GREEN: `/hy-tmp/orphan_slot_md_stream_green_20260704_151500`

Verification:

- RED: `python3 -m unittest tests.test_blb_orphan_slot_audit -v`
  - `red/red.rc`: `1`
  - Expected failure:
    `AssertionError: markdown report should stream through Path.open`
- GREEN: `python3 -m py_compile scripts/blb_orphan_slot_audit.py tests/test_blb_orphan_slot_audit.py`
  - `green/py_compile.rc`: `0`
- GREEN: `python3 -m unittest tests.test_blb_orphan_slot_audit -v`
  - `green/green.rc`: `0`
  - `Ran 5 tests`
