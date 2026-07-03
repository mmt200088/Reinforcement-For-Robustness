# Fusion Report Slot Mapping Copy Evidence

Source commit: `269ba69`

This evidence covers a fusion-count report optimization in
`scripts/report_fusion_count_map.py`: option slot mappings are now normalized by
iterating mapping `.items()` directly instead of copying them through
`dict(...).items()` first. Non-mapping inputs still fall back to `dict(...)` for
compatibility.

The affected paths feed option summaries and report payload generation, so this
removes one short-lived dictionary copy per option/base slot map in report
generation without changing output fields.

## Server Verification

- RED run directory:
  `/hy-tmp/rfr_fusion_report_slot_copy_red_4ea82b8_20260704_072514`
- GREEN run directory:
  `/hy-tmp/rfr_fusion_report_slot_copy_green_20260704_072557`

RED command:

```bash
python3 -m unittest \
  tests.test_report_fusion_count_map.FusionCountMapReportTest.test_option_slot_summary_uses_mapping_items_without_dict_copy \
  tests.test_report_fusion_count_map.FusionCountMapReportTest.test_build_report_payload_normalizes_base_slots_without_dict_copy -v
```

RED result: failed as expected on the pre-change source because
`_option_slot_summary()` and `_build_report_payload()` still copied slot
mappings through `dict(...)`.

GREEN commands:

```bash
python3 -m py_compile scripts/report_fusion_count_map.py tests/test_report_fusion_count_map.py
python3 -m unittest \
  tests.test_report_fusion_count_map.FusionCountMapReportTest.test_option_slot_summary_uses_mapping_items_without_dict_copy \
  tests.test_report_fusion_count_map.FusionCountMapReportTest.test_build_report_payload_normalizes_base_slots_without_dict_copy -v
```

GREEN result: `PY_COMPILE_RC=0`, both focused unittests passed, `TEST_RC=0`,
and the server wrapper exited 0.

## Local Contents

- `red_unittest.log`: server RED focused unittest log.
- `green_validation.log`: server GREEN py-compile and focused unittest log.
- `source_snapshot/report_fusion_count_map.py`: source snapshot from
  `269ba69`.
- `source_snapshot/test_report_fusion_count_map.py`: focused test snapshot from
  `269ba69`.
