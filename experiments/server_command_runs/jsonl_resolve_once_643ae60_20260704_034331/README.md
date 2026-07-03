# JSONL Single Path Resolution Evidence

Source commit: `643ae60`

Purpose: verify that shared JSONL readers resolve `.jsonl` / `.jsonl.gz` paths once and open the resolved path directly.

Server runs:
- Red: `/hy-tmp/rfr_jsonl_resolve_red_3f3db4f_20260704_043430`
- Green: `/hy-tmp/rfr_jsonl_resolve_green_3f3db4f_20260704_034331`

Green command coverage:
- `python -m py_compile jsonl_utils.py`
- `python -m unittest tests.test_jsonl_utils -v`
- Source guard confirming `iter_jsonl_records()` uses `_open_resolved_jsonl()`
