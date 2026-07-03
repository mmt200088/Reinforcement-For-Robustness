# Fusion Slots Option Index Evidence

Source commit: `1410ba0` (`Cache fusion slot report option lookups`)

Base source for red test: `3bbf1bf`

## Optimization

`scripts/render_fusion_count_slots_eval_report.py` now builds a per-options
`option_id -> option` index the first time a fusion graph's options are queried.
Repeated lookups in action-slot, selected-option, boost-audit, and boost-summary
report sections reuse that index instead of linearly scanning the same options
list for every lookup.

## Red Gate

Directory:
`rfr_fusion_slots_option_index_red_3bbf1bf_20260704_005753/`

Command:

```bash
PYTHONPATH="$PWD" python -m unittest tests.test_render_fusion_count_slots_eval_report.FusionCountSlotsEvalReportTest.test_option_lookup_indexes_graph_options_once -v
```

Status: `red_rc=1`

Expected failure: the old `_option_by_id()` scans the same options list on the
second lookup and triggers
`AssertionError: option lookup should reuse an index after first scan`.

## Green Gate

Directory:
`rfr_fusion_slots_option_index_green_3bbf1bf_20260704_010024/`

Commands:

```bash
PYTHONPATH="$PWD" python -m py_compile scripts/render_fusion_count_slots_eval_report.py tests/test_render_fusion_count_slots_eval_report.py
PYTHONPATH="$PWD" python -m unittest tests.test_render_fusion_count_slots_eval_report -v
```

Status: `py_compile_rc=0`, `unittest_rc=0`

Result: 2 fusion slots eval report tests passed, including the option-index
reuse test.
