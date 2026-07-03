# Paean Base Action ndarray Normalization Evidence

Source commit: `85c03b9` (`Avoid Paean base action list copy`)

Optimization: `Paean/action_grid.py` now normalizes non-string
`base_action_vec` inputs with `np.asarray(base_action_vec, dtype=int)` directly
instead of first materializing `list(base_action_vec)`. This avoids a full
Python-list copy when Paean final-eval callers already hold ndarray-backed
action vectors, while preserving size validation, bounds checks, and the final
`arr.copy()` return behavior.

Server evidence:

- `rfr_paean_base_action_ndarray_red_1530619_20260704_014345/red_status.txt`:
  `red_rc=1`. The regression test failed on the old implementation because
  ndarray input still called `list(base_action_vec)`.
- `rfr_paean_base_action_ndarray_green_1530619_20260704_014440/green_status.txt`:
  `py_compile_rc=0`, `unittest_rc=0`, `source_guard_rc=0`. The green run passed
  all seven `tests.test_paean_action_grid` tests and confirmed the list-copy
  source pattern is gone.

The evidence bundle excludes the temporary server source tree and keeps only
status files, logs, and source snapshots needed for audit.
