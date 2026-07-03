# action_k_accum_direct_4db8e02_20260704_044800

Source commit: `4db8e02`
Base commit for red test: `4b07227`

Optimization:
- `avg_truncation_k_in_action()` and `sum_truncation_k_in_action()` now share a
  direct sum/count accumulator for effective truncation K values.
- Per-layer K slot positions are cached as a tuple, so repeated Stage-2/Paean
  cost prefilters avoid rebuilding the same positions list.
- `_gather_effective_k_values_in_action()` remains available for compatibility,
  but the hot avg/sum helpers no longer allocate that gathered K list.

Server workflow:
- Red snapshot: `/hy-tmp/rfr_action_k_accum_red_4b07227_20260704_044600`
- Green snapshot: `/hy-tmp/rfr_action_k_accum_green_4b07227_20260704_044800`
- Server temp snapshots only ran code; canonical source was changed locally
  and then pushed through git.

Verification:
- Red: the new unittest failed against the base source because the hot helper
  still called `_gather_effective_k_values_in_action()`.
- Green: `python3 -m py_compile blb_stage2_rl/action_space.py tests/test_blb_cost_semantics.py`
  exited 0.
- Green: `python3 -m unittest tests.test_blb_cost_semantics -v` ran 6 tests
  and exited 0.
- Green source guard: fixed-string grep found no hot-path gather-list calls.
