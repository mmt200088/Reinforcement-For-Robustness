# Reward-probe tensor array packed transfer evidence

Source commit: `2d98907`

Purpose: verify that tensor-backed reward-probe prediction/label arrays are
concatenated on-device before a single CPU/numpy transfer when full arrays are
needed for MRPC/QQP weighted-F1 or regression metrics.

Server runs:

- Red: `red/red_status.txt` records `red_rc=1`; `red/logs/red_unittest.log`
  shows the old helper returned two arrays for two same-device tensors.
- Green: `green/green_status.txt` records `py_compile_rc=0`,
  `unittest_rc=0`, and `source_guard_rc=0`.

Green command coverage:

- `python -m py_compile blb_stage2_rl/inference_eval.py`
- `python -m unittest tests.test_blb_inference_eval_shared -v`
- source guard confirming `tensor_values_to_numpy_arrays()` uses the same-device
  tensor `torch.cat(...).cpu().numpy()` fast path
