# Fusion Map Report Option Scan Evidence

Source commit: `5760c6d`
Base red snapshot: `4018434`

This evidence captures the red/green server verification for reducing repeated
ordered option scans in `scripts/report_fusion_count_map.py`.

## Red

- Run directory: `rfr_fusion_report_option_scan_red_4018434_20260704_001505`
- Command: `PYTHONPATH="$PWD" python -m unittest tests.test_report_fusion_count_map.FusionCountMapReportTest.test_build_report_payload_avoids_repeated_ordered_option_scans -v`
- Status: `red_rc=1`
- Expected failure: old report payload generation iterated the ordered options
  list 4 times, exceeding the new `<= 2` scan guard.

## Green

- Run directory: `rfr_fusion_report_option_scan_green_4018434_20260704_001718`
- Compile command: `PYTHONPATH="$PWD" python -m py_compile scripts/report_fusion_count_map.py tests/test_report_fusion_count_map.py`
- Test command: `PYTHONPATH="$PWD" python -m unittest tests.test_report_fusion_count_map -v`
- Status: `green_py_compile_rc=0`, `green_unittest_rc=0`
- Result: 9 tests passed.
