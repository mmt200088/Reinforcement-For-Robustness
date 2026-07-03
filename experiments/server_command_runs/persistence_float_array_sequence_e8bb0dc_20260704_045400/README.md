# persistence_float_array_sequence_e8bb0dc_20260704_045400

Source commit: `e8bb0dc`
Base commit for red test: `62ee57c`

Optimization:
- `blb_stage2_rl.persistence._float_array()` now sends already-materialized
  list/tuple/range curve inputs directly to `numpy.asarray()`.
- Generator and iterator inputs retain the previous `list(values)` fallback.
- This removes one unnecessary Python sequence copy from Stage-2 training
  curve, entropy curve, diagnostic curve, and NPZ/report regeneration paths
  when callers pass ordinary lists or tuples.

Server workflow:
- Red snapshot: `/hy-tmp/rfr_persistence_float_tuple_red_62ee57c_20260704_045300`
- Green snapshot: `/hy-tmp/rfr_persistence_float_tuple_green_62ee57c_20260704_045400`
- Server temp snapshots only ran code; canonical source was changed locally
  and then pushed through git.

Verification:
- Red: the new unittest failed against the base source because tuple input
  still called `list(values)`.
- Green: `python3 -m py_compile blb_stage2_rl/persistence.py tests/test_blb_stage2_outputs.py`
  exited 0.
- Green: `python3 -m unittest tests.test_blb_stage2_outputs -v` ran 29 tests
  and exited 0.
- Green source guard confirmed the cached direct sequence type fast path.
