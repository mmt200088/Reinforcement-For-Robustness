# Persistence entropy ndarray fast-path evidence

Source commit: `47782e9`

Purpose: verify that Stage-2 entropy PNG rendering avoids copying ndarray-backed
`entropy_series` and matching `entropy_episodes` through `list()`.

Server runs:

- Red: `red/red_status.txt` records `red_rc=1`; the new regression test failed
  on the old implementation because the blocked `list(entropy_series)` copy was
  caught by the entropy-rendering guard and `entropy_png` was not written.
- Green: `green/green_status.txt` records `py_compile_rc=0`,
  `unittest_rc=0`, and `source_guard_rc=0`.

Green command coverage:

- `python -m py_compile blb_stage2_rl/persistence.py`
- `python -m unittest tests.test_blb_stage2_outputs.UpgradedCurvesTest -v`
- Source guard confirming the entropy branch uses `_float_array()` for
  `entropy_series` and `entropy_episodes`, with no direct `list()` conversion
  for either ndarray-backed series.
