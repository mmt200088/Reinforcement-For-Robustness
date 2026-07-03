# Fusion Report Occurrence Set Accumulation Evidence

Source commit: `74d5d28`

This evidence covers a fusion-count report optimization in
`scripts/report_fusion_count_map.py`: `_graph_occurrences()` now accumulates
unique layer indices in sets while scanning the schedule instead of first
building per-graph lists and then calling `sorted(set(v))`.

The returned report payload is unchanged: graph keys remain sorted and each
graph still maps to a sorted list of unique layer indices.

## Server Verification

- RED run directory:
  `/hy-tmp/rfr_fusion_report_occurrences_red_ee490c8_20260704_072956`
- GREEN run directory:
  `/hy-tmp/rfr_fusion_report_occurrences_green_20260704_073030`

RED command:

```bash
python3 -m unittest tests.test_report_fusion_count_map.FusionCountMapReportTest.test_graph_occurrences_accumulates_unique_layers_without_list_dedupe -v
```

RED result: failed as expected on the pre-change source because
`_graph_occurrences()` still used `append(int(step["layer_idx"]))` and
`sorted(set(v))`.

GREEN commands:

```bash
python3 -m py_compile scripts/report_fusion_count_map.py tests/test_report_fusion_count_map.py
python3 -m unittest tests.test_report_fusion_count_map.FusionCountMapReportTest.test_graph_occurrences_accumulates_unique_layers_without_list_dedupe -v
```

GREEN result: `PY_COMPILE_RC=0`, focused unittest passed, `TEST_RC=0`, and the
server wrapper exited 0.

## Local Contents

- `red_unittest.log`: server RED focused unittest log.
- `green_validation.log`: server GREEN py-compile and focused unittest log.
- `source_snapshot/report_fusion_count_map.py`: source snapshot from
  `74d5d28`.
- `source_snapshot/test_report_fusion_count_map.py`: focused test snapshot from
  `74d5d28`.
