# Fusion Report Stdout JSON Streaming

- Source commit: `0980322`
- Remote RED package: `/hy-tmp/fusion_report_stdout_json_red_20260704_104648`
- Remote GREEN package: `/hy-tmp/fusion_report_stdout_json_green_20260704_104732`
- Scope: `scripts/report_fusion_count_map.py` CLI stdout summary.

## RED

Command:

```bash
python3 -m unittest tests.test_report_fusion_count_map.FusionCountMapReportTest.test_main_streams_stdout_json_summary -v
```

Result: `red.rc=1`.

Expected failure: old source still used `print(json.dumps(...))`, which materialized the complete CLI stdout summary before writing it.

## GREEN

Commands:

```bash
python3 -m py_compile scripts/report_fusion_count_map.py
python3 -m unittest tests.test_report_fusion_count_map -v
```

Result: `green.rc=0`.

Evidence: `green.log` shows `py_compile_rc=0`, `tests.test_report_fusion_count_map` passing all 22 tests, and `final_rc=0`.
