# Stage-1 Report Regex Dispatch Evidence

Source commit: `5901ffb`
Base red snapshot: `d96cf43`

This evidence captures the red/green server verification for dispatching
Stage-1 parallel report log lines by marker before running regex parsers.

## Red

- Run directory: `rfr_stage1_report_regex_dispatch_red_d96cf43_20260704_002317`
- Command: `PYTHONPATH="$PWD" python -m unittest tests.test_stage1_parallel_report.Stage1ParallelReportTest.test_parse_log_lines_dispatches_total_lines_without_worker_regex -v`
- Status: `red_rc=1`
- Expected failure: old parsing sent `[stage1-rollout-total]` lines through
  `ROLLOUT_RE.search()` before the total parser.

## Green

- Run directory: `rfr_stage1_report_regex_dispatch_green_d96cf43_20260704_002526`
- Compile command: `PYTHONPATH="$PWD" python -m py_compile scripts/stage1_parallel_report.py tests/test_stage1_parallel_report.py`
- Test command: `PYTHONPATH="$PWD" python -m unittest tests.test_stage1_parallel_report -v`
- Status: `green_py_compile_rc=0`, `green_unittest_rc=0`
- Result: 8 tests passed.
