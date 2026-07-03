# Fusion Report Option-Index Evidence

Source commit: `ccbfc5f` (`Index fusion report options for action configs`)

This run verifies a Rescale/fusion report-path optimization in
`scripts/report_fusion_count_map.py`: action-config generation now builds one
`option_id -> option` index per graph and reuses it for both action-vector and
slot-form config splicing instead of linearly scanning graph options for every
schedule step.

## Server Verification

- Red package: `/hy-tmp/rfr_fusion_option_index_red_b6ff631_20260704_051728`
- Green package: `/hy-tmp/rfr_fusion_option_index_green_20260704_051827`
- Red test:
  `python3 -m unittest tests.test_report_fusion_count_map.FusionCountMapReportTest.test_write_action_configs_uses_prebuilt_option_index -v`
- Green gate:
  `python3 -m py_compile scripts/report_fusion_count_map.py tests/test_report_fusion_count_map.py`
- Green gate:
  `python3 -m unittest tests.test_report_fusion_count_map -v`
- Source guards confirmed `_write_action_configs()` builds
  `option_index_by_graph` once and passes it to both splice helpers.

## Result

- Red: failed because old `_write_action_configs()` reached `_option_by_id()`.
- Green: passed `py_compile`, all 10 `tests.test_report_fusion_count_map`
  tests, and source guards.
