# Reward-probe count-weight reuse evidence

Source commit: `da02fca`

Purpose: verify that `weighted_probe_batch_means()` builds the probe batch
count weights once and reuses them for loss, metric1, and metric2 means.

Server runs:

- Red: `red/red_status.txt` records `red_rc=1`; `red/logs/red_unittest.log`
  shows the old implementation iterated `counts` again on the second metric.
- Green: `green/green_status.txt` records `py_compile_rc=0`,
  `unittest_rc=0`, and `source_guard_rc=0`.

Green command coverage:

- `python -m py_compile blb_stage2_rl/eval_metrics.py`
- `python -m unittest tests.test_blb_eval_metrics_shared -v`
- source guard confirming `weighted_probe_batch_means()` builds weights once and
  no longer calls `sample_weighted_mean()` internally
