# Fusion Report Block-Action Cache Evidence

Source commit: `82b83ca` (`Cache fusion report block actions`)

This run verifies a Rescale/fusion report-path optimization in
`scripts/report_fusion_count_map.py`: action-config generation now caches the
adjusted per-graph/per-option block action tuple after converting
`action_indices` and restoring the baseline K slot, instead of rebuilding that
list for every schedule step.

## Server Verification

- Red package: `/hy-tmp/rfr_fusion_block_actions_red_4931191_20260704_052547`
- Green package: `/hy-tmp/rfr_fusion_block_actions_green_20260704_052649`
- Red test:
  `python3 -m unittest tests.test_report_fusion_count_map.FusionCountMapReportTest.test_splice_group_action_reuses_adjusted_block_actions -v`
- Green gate:
  `python3 -m py_compile scripts/report_fusion_count_map.py tests/test_report_fusion_count_map.py`
- Green gate:
  `python3 -m unittest tests.test_report_fusion_count_map -v`
- Source guards confirmed `_write_action_configs()` builds
  `block_actions_by_option` once and passes it to `_splice_group_action()`.

## Result

- Red: failed because old `_splice_group_action()` iterated the same
  `action_indices` twice for two same-option schedule steps.
- Green: passed `py_compile`, all 12 `tests.test_report_fusion_count_map`
  tests, and source guards.
