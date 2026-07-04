# Fusion Report Graph-Key View Evidence

Source commit: `61185e9`

Optimization: `scripts/report_fusion_count_map.py` now reuses the
`graphs.keys()` view in `_group_specs()` instead of copying graph names through
`list(graphs.keys())` before building fusion-count report group specs.

Server evidence:

- RED: `/hy-tmp/fusion_report_graph_keys_red4_931561f_20260704_142000` ran the
  new focused static test against the previous source and failed with
  `red.rc=1` because the old implementation still used
  `graph_order = list(graphs.keys())`.
- GREEN: `/hy-tmp/fusion_report_graph_keys_green_20260704_142000` ran
  `python3 -m py_compile scripts/report_fusion_count_map.py
  tests/test_report_fusion_count_map.py` and the complete
  `python3 -m unittest tests.test_report_fusion_count_map -v` suite.
  `py_compile.rc=0`, `green.rc=0`, 23 tests passed.
