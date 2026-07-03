# Paean Legacy Action Vector Parse Evidence

Source commit: `a600b79` (`Avoid Paean legacy action vector list copy`)

Optimization: `Paean/action_grid.py` now parses legacy list-form
`action_vec` / `base_action_vec` config payloads with `np.asarray(base_raw,
dtype=int)` directly instead of first copying the complete vector through
`list(base_raw)`. This preserves legacy action-config behavior while avoiding
one full-vector Python-list copy during final-eval action-config loading.

Server evidence:

- `rfr_paean_parse_action_vec_red_7f2d1f5_20260704_014630/red_status.txt`:
  `red_rc=1`. The regression test failed on the old implementation because
  list-form `action_vec` still called `list(base_raw)`.
- `rfr_paean_parse_action_vec_green_7f2d1f5_20260704_014720/green_status.txt`:
  `py_compile_rc=0`, `unittest_rc=0`, `source_guard_rc=0`. The green run passed
  all eight `tests.test_paean_action_grid` tests and confirmed the legacy
  list-copy source pattern is gone.

The evidence bundle excludes the temporary server source tree and keeps only
status files, logs, and source snapshots needed for audit.
