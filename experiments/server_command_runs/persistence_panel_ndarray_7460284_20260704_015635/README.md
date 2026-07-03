# Stage-2 Panel ndarray Fast-Path Evidence

Source commit: `7460284` (`Avoid panel ndarray list materialization`)

Optimization: `_stage1_style_panel()` in `blb_stage2_rl/persistence.py` now
uses `_float_array(raw)` instead of `np.asarray(list(raw), dtype=float)`. This
preserves iterator behavior while avoiding a full Python-list copy when
Stage-2 report regeneration or training-curve rendering passes ndarray-backed
reward/loss/metric series.

Server evidence:

- `rfr_persistence_panel_ndarray_red_6a741a1_20260704_015550/red_status.txt`:
  `red_rc=1`. The regression test failed on the old implementation because
  panel raw ndarray input still called `list(raw)`.
- `rfr_persistence_panel_ndarray_green_6a741a1_20260704_015635/green_status.txt`:
  `py_compile_rc=0`, `unittest_rc=0`, `source_guard_rc=0`. The green run passed
  all ten `UpgradedCurvesTest` tests and confirmed the panel raw series source
  path now uses `_float_array(raw)`.

The evidence bundle excludes the temporary server source tree and keeps only
status files, logs, and source snapshots needed for audit.
