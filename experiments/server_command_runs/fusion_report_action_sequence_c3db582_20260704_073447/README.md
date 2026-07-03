# Fusion Report Action Sequence Copy Evidence

Source commit: `c3db582`

This evidence covers a fusion-count report optimization in
`scripts/report_fusion_count_map.py`: `_option_slot_summary()` now indexes the
option and base `action_indices` sequences directly instead of copying and
converting the entire sequence up front with list comprehensions.

The function still converts each indexed value with `int(...)` at use sites, so
the emitted slot rows and changed-slot summaries are unchanged.

## Server Verification

- RED run directory:
  `/hy-tmp/rfr_fusion_report_action_seq_red_2f059d1_20260704_073416`
- GREEN run directory:
  `/hy-tmp/rfr_fusion_report_action_seq_green_20260704_073447`

RED command:

```bash
python3 -m unittest tests.test_report_fusion_count_map.FusionCountMapReportTest.test_option_slot_summary_indexes_action_sequences_without_list_copy -v
```

RED result: failed as expected on the pre-change source because
`_option_slot_summary()` still used full-list `action_indices` comprehensions.

GREEN commands:

```bash
python3 -m py_compile scripts/report_fusion_count_map.py tests/test_report_fusion_count_map.py
python3 -m unittest tests.test_report_fusion_count_map.FusionCountMapReportTest.test_option_slot_summary_indexes_action_sequences_without_list_copy -v
```

GREEN result: `PY_COMPILE_RC=0`, focused unittest passed, `TEST_RC=0`, and the
server wrapper exited 0.

## Local Contents

- `red_unittest.log`: server RED focused unittest log.
- `green_validation.log`: server GREEN py-compile and focused unittest log.
- `source_snapshot/report_fusion_count_map.py`: source snapshot from
  `c3db582`.
- `source_snapshot/test_report_fusion_count_map.py`: focused test snapshot from
  `c3db582`.
