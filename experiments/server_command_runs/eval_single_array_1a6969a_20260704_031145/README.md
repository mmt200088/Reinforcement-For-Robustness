# Reward-Probe Single-Array Flatten Evidence

Source commit: `1a6969a`

Purpose: verify that already-packed reward-probe prediction/label arrays are
reshaped directly instead of copied through `np.concatenate()`.

Server runs:

- Red: `/hy-tmp/rfr_eval_single_array_red2_5616253_20260704_031130`
- Green: `/hy-tmp/rfr_eval_single_array_green2_5616253_20260704_031145`

Green command coverage:

- `python -m py_compile blb_stage2_rl/eval_metrics.py`
- `python -m unittest tests.test_blb_eval_metrics_shared -v`
- Source guard confirming `_flatten_probe_arrays()` single-array fast path and
  finalizer helper usage
