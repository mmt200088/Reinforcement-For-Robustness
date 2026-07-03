# Paean Cost-Match Degree Array Reuse Evidence

Source commit: `94f1aad` (`Reuse Paean cost-match degree arrays`)

Optimization: `Paean/action_grid.py` now normalizes GELU/Softmax degree arrays
once before the cost-matched random action reject-sampling loop and reuses them
for every `action_vector_to_cfgs()` decode attempt. The same change also
precomputes the target sum-K, total-bit, and fusion-count integers once before
the loop. Sampling order, fixed override application, optimizer requests, and
cost-match filtering semantics are unchanged.

Server evidence:

- `rfr_paean_degree_array_red_1a803f0_20260704_013820/red_status.txt`:
  `red_rc=1`. The new regression test failed on the old implementation because
  repeated decode attempts received distinct degree array objects.
- `rfr_paean_degree_array_green_1a803f0_20260704_013930/green_status.txt`:
  `py_compile_rc=0`, `unittest_rc=0`, `source_guard_rc=0`. The green run passed
  all six `tests.test_paean_action_grid` tests and confirmed the loop no longer
  contains per-decode `np.asarray(gelu_degree/attn_degree, dtype=int)` calls.

The evidence bundle excludes the temporary server source tree and keeps only
status files, logs, and source snapshots needed for audit.
