# Reward-probe scalar sync batching evidence

Source commit: `497ecda`

Purpose: verify that `run_installed_probe_trial()` batches scalar tensor
conversion for `loss`, `metric1`, and `metric2` instead of synchronizing each
field through the legacy per-sequence helper.

Server runs:

- Red: `red/red_status.txt` records `red_rc=1`; `red/logs/red_unittest.log`
  shows the source guard failing against the old three-call conversion path.
- Green: `green/green_status.txt` records `py_compile_rc=0`,
  `unittest_rc=0`, and `source_guard_rc=0`.

Green command coverage:

- `python -m py_compile blb_stage2_rl/inference_eval.py`
- `python -m unittest tests.test_blb_inference_eval_shared -v`
- source guard confirming the batched helper call is present and the three old
  per-field scalar conversion calls are absent from `run_installed_probe_trial()`
