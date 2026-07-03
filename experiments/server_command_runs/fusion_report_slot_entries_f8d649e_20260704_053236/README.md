# Fusion Report Slot-Entry Cache Evidence

Source commit: `f8d649e` (`Cache fusion report slot entries`)

This run verifies a Rescale/fusion report-path optimization in
`scripts/report_fusion_count_map.py`: slot-form action-config generation now
caches bound, filtered, sorted slot entries per graph option, then only stamps
the layer-specific label for each schedule occurrence.

## Server Verification

- Red package: `/hy-tmp/rfr_fusion_slot_entries_red_5c6e81b_20260704_053119`
- Green package: `/hy-tmp/rfr_fusion_slot_entries_green_20260704_053236`
- Red test:
  `python3 -m unittest tests.test_report_fusion_count_map.FusionCountMapReportTest.test_splice_group_slots_reuses_bound_slot_entries -v`
- Green gate:
  `python3 -m py_compile scripts/report_fusion_count_map.py tests/test_report_fusion_count_map.py`
- Green gate:
  `python3 -m unittest tests.test_report_fusion_count_map -v`
- Source guards confirmed `_write_action_configs()` builds
  `slot_entries_by_option` once and passes it to `_splice_group_slots()`.

## Result

- Red: failed because old `_splice_group_slots()` called `_bound_slot_values()`
  twice for two same-option schedule steps.
- Green: passed `py_compile`, all 13 `tests.test_report_fusion_count_map`
  tests, and source guards.
