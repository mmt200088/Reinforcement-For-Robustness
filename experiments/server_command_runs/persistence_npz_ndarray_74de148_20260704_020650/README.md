# Persistence NPZ ndarray fast-path evidence

Source commit: `74de148`

Purpose: verify that Stage-2 `write_training_curves()` writes mandatory NPZ
curve artifacts without copying ndarray-backed series through `list()`.

Server runs:

- Red: `red/red_status.txt` records `red_rc=1`; the new regression test failed
  on the old implementation because blocking `list(seq)` prevented `out["npz"]`
  from being written.
- Green: `green/green_status.txt` records `py_compile_rc=0`,
  `unittest_rc=0`, and `source_guard_rc=0`.

Green command coverage:

- `python -m py_compile blb_stage2_rl/persistence.py`
- `python -m unittest tests.test_blb_stage2_outputs.UpgradedCurvesTest -v`
- Source guard confirming the NPZ writer uses `_float_array(seq)` and no longer
  contains `values = list(seq)` in the `write_training_curves()` NPZ branch.
