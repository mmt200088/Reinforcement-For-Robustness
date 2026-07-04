# Shared Mean Helper Streaming

Source commit: `65481ad`

This run verifies that `stats_utils.mean_or_none()` keeps the existing
`math.fsum()` numeric behavior while streaming float conversion and count
tracking, avoiding the intermediate `vals = [...]` list used by multiple report
and diagnostic paths.

Server temporary sources:

- RED: `/hy-tmp/stats_utils_mean_stream_red_5d34fef_20260704_120000`
- GREEN: `/hy-tmp/stats_utils_mean_stream_green_20260704_120000`

Verification:

- `red_focused.rc`: `1`, expected failure on old source because
  `stats_utils.py` still contained `vals = [float(value) for value in values]`.
- `green.rc`: `py_compile_rc=0`, `test_rc=0`.
- `green.log`: `tests.test_stats_utils` passed with 7 tests.
