# Rescale Replan JSON Dict Streaming Evidence

Source commit: `8fe0b03`

Optimization: `Rescale_optimizer/scripts/replan_what_if.py` now formats
multi-line compact JSON dictionaries by streaming `obj.items()` and using
`len(obj)` to detect the final row, instead of copying the whole mapping
through `list(obj.items())` before rendering.

Server evidence:

- RED: `/hy-tmp/rescale_replan_json_stream_red_ed69d4e_20260704_140500` ran
  the new focused compact-JSON test against the previous source and failed with
  `red.rc=1` because the old implementation still used
  `items = list(obj.items())`.
- GREEN: `/hy-tmp/rescale_replan_json_stream_green_20260704_140500` ran
  `python3 -m py_compile Rescale_optimizer/scripts/replan_what_if.py
  tests/test_rescale_replan_what_if.py` and
  `python3 -m unittest tests.test_rescale_replan_what_if -v`.
  `py_compile.rc=0`, `green.rc=0`, 1 test passed.
