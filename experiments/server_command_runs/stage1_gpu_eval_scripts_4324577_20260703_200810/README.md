# Stage-1 GPU Eval Script Transfer/Sync Verification

- Date: 2026-07-03
- Server temp run: `/hy-tmp/stage1_gpu_eval_scripts_4324577_20260703_200810`
- Green commit: `43245772e6b91012e1d01fced6aebfa35d4dbc20`
- Red baseline commit: `9d013d2a8dd471c165eadc01a7fc3adf8e007b34`

## Result

- Red check on the old Stage-1 eval scripts: `red_old_rc=0`
  - Raw old-source unittest exit: `red_old_unittest_raw_rc=1`
  - Confirms the old scripts lacked the pinned-memory / non-blocking transfer and deferred-sync source contract.
- Green check on the current head: `green_head_unittest_rc=0`
  - Command: `python3 -m unittest tests.test_stage1_eval_accel.Stage1GpuEvalScriptSourceTest -v`
  - Result: `Ran 2 tests in 0.000s`, `OK`
- Server GitHub reachability check: `git_ls_remote_rc=124`
  - The server-side `git ls-remote` hit the 20-second timeout; the verified source package was built locally from the already-pushed commits and copied to the server temp directory.

## Files

- `logs/red_old_unittest.log`
- `logs/green_head_unittest.log`
- `logs/summary.json`
- `stage1_gpu_eval_scripts_4324577_20260703_200810_verify.sh`
