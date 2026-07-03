# diagnostic_curve_array_cache_00bc7e8_20260704_050000

Source commit: `00bc7e8`
Base commit for red test: `392708f`

Optimization:
- `blb_stage2_rl.persistence.write_diagnostic_curves()` now routes diagnostic
  series through `_float_array()` instead of calling `list(seq)` directly.
- The diagnostic render pass caches converted arrays by input object id, so
  repeated checks and panels reuse the same numpy array.
- This keeps Stage-2 collapse-diagnostic PNG/report generation cheaper for
  already-materialized list/tuple/range series while preserving iterator
  fallback semantics through `_float_array()`.

Server workflow:
- Red snapshot: `/hy-tmp/rfr_diag_tuple_red_392708f_20260704_045900`
- Green snapshot: `/hy-tmp/rfr_diag_tuple_green_392708f_20260704_050000`
- Server temp snapshots only ran code; canonical source was changed locally
  and then pushed through git.

Verification:
- Red: the new unittest failed against the base source because diagnostic tuple
  series still hit `list(seq)`.
- Green: `python3 -m py_compile blb_stage2_rl/persistence.py tests/test_blb_stage2_outputs.py`
  exited 0.
- Green: `python3 -m unittest tests.test_blb_stage2_outputs -v` ran 30 tests
  and exited 0.
- Green source guard confirmed the diagnostic array cache and `_float_array()`
  path, with no `values = list(seq)` call left.
