# Experiments Log Rebuild Sort Evidence

Source commit: `9cabfaa`

Optimization: `tools.experiments_log._rebuild_index()` now sorts the
`_latest_by_run_id(...).values()` view directly with `sorted(...)` instead of
first copying it through `list(...)` and then sorting in place.

Server evidence:

- RED: `/hy-tmp/experiments_log_rebuild_sort_red_28ec18d_20260704_131500`
  ran the new focused test against the previous source and failed with
  `red.rc=1` because the old implementation still used `latest = list(...)`
  followed by `latest.sort(...)`.
- GREEN: `/hy-tmp/experiments_log_rebuild_sort_green_20260704_131500`
  ran `python3 -m py_compile tools/experiments_log.py
  tests/test_experiments_log.py jsonl_utils.py json_utils.py` and the complete
  `python3 -m unittest tests.test_experiments_log -v` suite.
  `py_compile.rc=0`, `green.rc=0`, 17 tests passed.
