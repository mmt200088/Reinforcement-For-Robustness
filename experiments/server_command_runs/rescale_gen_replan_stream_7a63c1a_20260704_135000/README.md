# Rescale Replan Action Formatting Evidence

Source commit: `7a63c1a`

Optimization: `Rescale_optimizer/scripts/gen_replan_actions.py` now formats
`delta_overrides` by streaming `delta_overrides.items()` and using the mapping
length to detect the final row, instead of copying the whole mapping through
`list(delta_overrides.items())`.

Server evidence:

- RED: `/hy-tmp/rescale_gen_replan_stream_red_7283c48_20260704_135000` ran the
  new focused action-template formatting test against the previous source and
  failed with `red.rc=1` because the old implementation still used
  `items = list(delta_overrides.items())`.
- GREEN: `/hy-tmp/rescale_gen_replan_stream_green_20260704_135000` ran
  `python3 -m py_compile Rescale_optimizer/scripts/gen_replan_actions.py
  tests/test_rescale_gen_replan_actions.py` and
  `python3 -m unittest tests.test_rescale_gen_replan_actions -v`.
  `py_compile.rc=0`, `green.rc=0`, 1 test passed.
