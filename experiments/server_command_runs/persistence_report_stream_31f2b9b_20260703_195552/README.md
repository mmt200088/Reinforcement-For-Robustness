# Stage-2 Persistence Report Streaming Verification

- Date: 2026-07-03
- Server temp run: `/hy-tmp/persistence_report_stream_31f2b9b_20260703_195552`
- Green commit: `31f2b9bf9b85ca89fa214c60e6d53e38b4e782ee`
- Red baseline commit: `ce030a264eacda70d9579a6db030565dfcde24f0`

## Result

- Red check on the old persistence writers: `red_old_rc=0`
  - Confirms the old action/final/crash report writers materialized full joined documents.
- Green check on the current head: `green_head_unittest_rc=0`
  - Command: `python3 -m unittest tests.test_blb_stage2_rl_regressions.BLBTraceWriterRegressionTests.test_persistence_report_writers_stream_line_outputs -v`
  - Result: `Ran 1 test in 0.021s`, `OK`
- Server GitHub reachability check: `git_ls_remote_rc=124`
  - The server-side `git ls-remote` hit the 20-second timeout; the verified source package was built locally from the already-pushed commits and copied to the server temp directory.

## Files

- `logs/red_old.log`
- `logs/green_head_unittest.log`
- `logs/summary.json`
- `persistence_report_stream_31f2b9b_20260703_195552_red.py`
- `persistence_report_stream_31f2b9b_20260703_195552_verify.sh`
