# Stage-2 Curve ndarray Fast-Path Evidence

Source commit: `bd4ca26` (`Avoid curve ndarray list materialization`)

Optimization: `blb_stage2_rl/persistence.py` now routes curve values through
`_float_array()`. Iterator inputs keep the existing one-materialization path,
while ndarray-backed inputs go directly through `np.asarray(values,
dtype=float)`. This avoids a full Python-list copy in `_ema_smooth()` and
`_moving_average()` when Stage-2 curve/report pipelines already have numpy
arrays.

Server evidence:

- `rfr_persistence_curve_ndarray_red_a680835_20260704_015230/red_status.txt`:
  `red_rc=1`. The regression test failed on the old implementation because
  `_ema_smooth()` called `list(values)` for ndarray input.
- `rfr_persistence_curve_ndarray_green_a680835_20260704_015320/green_status.txt`:
  `py_compile_rc=0`, `unittest_rc=0`, `source_guard_rc=0`. The green run passed
  all nine `UpgradedCurvesTest` tests and confirmed the ndarray fast-path helper
  is present.

The evidence bundle excludes the temporary server source tree and keeps only
status files, logs, and source snapshots needed for audit.
