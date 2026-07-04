# Fusion Report Default JSON Evidence

Source commit: `6137460`

Optimization: `scripts/report_fusion_count_map.py` now computes the fixed
default GELU and Softmax degree JSON strings once at module load and reuses
those strings for argparse defaults. This avoids repeating the same
`json.dumps()` calls whenever the fusion-count map report CLI builds its
parser.

Server evidence:

- RED: `/hy-tmp/fusion_report_defaults_red_3140454_20260704_125319` ran the
  new focused test against the previous source and failed with `red.rc=1`
  because the old implementation still dumped the fixed defaults inline inside
  `main()`.
- GREEN: `/hy-tmp/fusion_report_defaults_green_20260704_125319` ran
  `python3 -m py_compile scripts/report_fusion_count_map.py` and
  `python3 -m unittest tests.test_report_fusion_count_map -v`.
  `py_compile.rc=0`, `green.rc=0`, 24 tests passed.
