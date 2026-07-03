# Reward-Probe Binary MCC Evidence

Source commit: `d0e8b8c`

Purpose: verify that standard 0/1 CoLA-style Matthews correlation arrays use
direct count reductions instead of sorting a class union.

Server runs:

- Red: `/hy-tmp/rfr_binary_mcc_red_0928a4a_20260704_035330`
- Green: `/hy-tmp/rfr_binary_mcc_green_0928a4a_20260704_035430`

Green command coverage:

- `python -m py_compile blb_stage2_rl/eval_metrics.py`
- `python -m unittest tests.test_blb_eval_metrics_shared -v`
- Source guard confirming the binary MCC fast path appears before the
  `np.union1d()` fallback
