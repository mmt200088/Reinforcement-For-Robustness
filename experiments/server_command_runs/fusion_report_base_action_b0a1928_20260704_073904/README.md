# Fusion Report Base Action Copy Evidence

Source commit: `b0a1928`

This evidence covers a fusion-count report optimization in
`scripts/report_fusion_count_map.py`: `_build_report_payload()` now passes the
base option `action_indices` sequence directly into `_option_slot_summary()`
instead of copying and converting the full sequence once per graph.

`_option_slot_summary()` still converts indexed values with `int(...)` when
constructing rows and changed-slot summaries, so output semantics are unchanged.

## Server Verification

- RED run directory:
  `/hy-tmp/rfr_fusion_report_base_action_red_0e55f9d_20260704_073831`
- GREEN run directory:
  `/hy-tmp/rfr_fusion_report_base_action_green_20260704_073904`

RED command:

```bash
python3 -m unittest tests.test_report_fusion_count_map.FusionCountMapReportTest.test_build_report_payload_passes_base_action_sequence_without_copy -v
```

RED result: failed as expected on the pre-change source because
`_build_report_payload()` still used a full-list `base_action` comprehension.

GREEN commands:

```bash
python3 -m py_compile scripts/report_fusion_count_map.py tests/test_report_fusion_count_map.py
python3 -m unittest tests.test_report_fusion_count_map.FusionCountMapReportTest.test_build_report_payload_passes_base_action_sequence_without_copy -v
```

GREEN result: `PY_COMPILE_RC=0`, focused unittest passed, `TEST_RC=0`, and the
server wrapper exited 0.

## Local Contents

- `red_unittest.log`: server RED focused unittest log.
- `green_validation.log`: server GREEN py-compile and focused unittest log.
- `source_snapshot/report_fusion_count_map.py`: source snapshot from
  `b0a1928`.
- `source_snapshot/test_report_fusion_count_map.py`: focused test snapshot from
  `b0a1928`.
