# Rescale Noise Dict Formatting Evidence

Source commit: `4b9a144`

Optimization: `Rescale_optimizer/scripts/update_noise_tables_from_csv.py`
now formats noise-table dictionaries by streaming `d.items()` with a bounded
per-line buffer instead of copying every `(sf, value)` pair through
`list(d.items())` and slicing that list.

Server evidence:

- RED: `/hy-tmp/rescale_noise_fmt_stream_red_8c55616_20260704_134000` ran the
  new focused formatter test against the previous source and failed with
  `red.rc=1` because the old implementation still used
  `items = list(d.items())`.
- GREEN: `/hy-tmp/rescale_noise_fmt_stream_green_20260704_134000` ran
  `python3 -m py_compile Rescale_optimizer/scripts/update_noise_tables_from_csv.py
  tests/test_rescale_update_noise_tables.py` and the complete
  `python3 -m unittest tests.test_rescale_update_noise_tables -v` suite.
  `py_compile.rc=0`, `green.rc=0`, 3 tests passed.
