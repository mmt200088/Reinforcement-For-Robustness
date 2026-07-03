# Reward-probe accuracy-only prediction-array skip evidence

Source commit: `b5dfff5`

Purpose: verify that `run_installed_probe_trial()` does not retain and transfer
prediction/label tensors for accuracy-only metric profiles, where per-batch
sample-weighted accuracy already provides both returned metrics.

Server runs:

- Red: `red/red_status.txt` records `red_rc=1`; `red/logs/red_unittest.log`
  shows the old SST-2 path still called `tensor_values_to_numpy_arrays()`.
- Green: `green/green_status.txt` records `py_compile_rc=0`,
  `unittest_rc=0`, and `source_guard_rc=0`.

Green command coverage:

- `python -m py_compile blb_stage2_rl/inference_eval.py`
- `python -m unittest tests.test_blb_inference_eval_shared -v`
- source guard confirming `need_prediction_arrays` gates prediction/label tensor
  retention and numpy transfer
