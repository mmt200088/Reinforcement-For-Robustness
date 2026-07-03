# Fusion Report Field-Kind Lookup Evidence

Source commit: `71fbfc6` (`Reuse fusion slot field kind lookups`)

This run verifies a Rescale/fusion report-path optimization in
`scripts/report_fusion_count_map.py`: slot-form action-config generation now
reuses per-block field-name to kind lookups instead of rebuilding the same
dictionary for every schedule step.

## Server Verification

- Red package: `/hy-tmp/rfr_fusion_field_kinds_red_1d1f15d_20260704_052146`
- Green package: `/hy-tmp/rfr_fusion_field_kinds_green_20260704_052234`
- Red test:
  `python3 -m unittest tests.test_report_fusion_count_map.FusionCountMapReportTest.test_splice_group_slots_reuses_field_kind_lookup -v`
- Green gate:
  `python3 -m py_compile scripts/report_fusion_count_map.py tests/test_report_fusion_count_map.py`
- Green gate:
  `python3 -m unittest tests.test_report_fusion_count_map -v`
- Source guards confirmed `_write_action_configs()` builds
  `field_kinds_by_block` once and passes it to `_splice_group_slots()`.

## Result

- Red: failed because old `_splice_group_slots()` iterated block fields twice
  for two same-block schedule steps.
- Green: passed `py_compile`, all 11 `tests.test_report_fusion_count_map`
  tests, and source guards.
