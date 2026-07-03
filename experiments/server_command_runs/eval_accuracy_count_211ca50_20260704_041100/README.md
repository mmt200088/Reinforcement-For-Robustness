# Reward-Probe Accuracy Count Evidence

Source commit: `211ca50`

Purpose: verify that classification accuracy uses direct match counting instead
of generic `np.mean()` over a boolean mask.

Server runs:

- Red: `/hy-tmp/rfr_accuracy_count_red_8bf11a1_20260704_041000`
- Green: `/hy-tmp/rfr_accuracy_count_green_8bf11a1_20260704_041100`

Green command coverage:

- `python -m py_compile blb_stage2_rl/eval_metrics.py`
- `python -m unittest tests.test_blb_eval_metrics_shared -v`
- Source guard confirming `accuracy_from_labels()` uses `np.count_nonzero()`
  and does not call `np.mean()`
