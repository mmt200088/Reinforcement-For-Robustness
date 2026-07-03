# Action-mask degree-vector ndarray fast-path evidence

Source commit: `522b42f`

Purpose: verify that `blb_stage2_rl.action_mask._degree_vector()` handles
ndarray-backed GELU/attention degree vectors without copying them through
`list(raw)` first.

Server runs:

- Red: `red/red_status.txt` records `red_rc=1`; the regression test failed on
  the old implementation because `_degree_vector()` called `list(raw)` for an
  ndarray.
- Green: `green/green_status.txt` records `py_compile_rc=0`,
  `unittest_rc=0`, and `source_guard_rc=0`.

Green command coverage:

- `python -m py_compile blb_stage2_rl/action_mask.py`
- `python -m unittest tests.test_sequential_smoke.ForbiddenActionMaskTest.test_degree_vector_accepts_ndarray_without_list_materialization tests.test_sequential_smoke.ForbiddenActionMaskTest.test_roundtrip_via_minimal_import_shim -v`
- Source guard confirming `_degree_vector()` has an ndarray branch based on
  `np.asarray(...).reshape(-1)` and that branch does not call `list(raw)`.
