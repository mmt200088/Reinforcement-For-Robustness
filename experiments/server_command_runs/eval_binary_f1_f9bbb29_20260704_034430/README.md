# Reward-Probe Binary Weighted-F1 Evidence

Source commit: `f9bbb29`

Purpose: verify that standard 0/1 MRPC/QQP weighted-F1 arrays use direct
count reductions instead of sorting a class union.

Server runs:

- Red: `/hy-tmp/rfr_binary_f1_red_b743386_20260704_034330`
- Green: `/hy-tmp/rfr_binary_f1_green_b743386_20260704_034430`

Green command coverage:

- `python -m py_compile blb_stage2_rl/eval_metrics.py`
- `python -m unittest tests.test_blb_eval_metrics_shared -v`
- Source guard confirming the binary fast path appears before the
  `np.union1d()` fallback
