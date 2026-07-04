# Fusion Map Report HTML Streaming Evidence

Source commit: `c0d2a92`

Optimization: `scripts/report_fusion_count_map.py` now keeps the existing
`_render_html()` string-returning helper for compatibility, but the CLI path
uses `write_rendered_html()` and `_HtmlPartsWriter` to stream the fusion-count
map HTML report directly to the output file. This avoids materializing the full
map report as one large string before writing.

Server evidence:

- RED: `/hy-tmp/fusion_map_report_html_stream_red_d2dd9e9_20260704_134100`
  ran the new focused test against the previous source and failed with
  `red.rc=1` because the old implementation had no `write_rendered_html()`
  streaming writer and still wrote `html_path.write_text(_render_html(payload))`.
- GREEN: `/hy-tmp/fusion_map_report_html_stream_green_20260704_134100` ran
  `python3 -m py_compile scripts/report_fusion_count_map.py` and
  `python3 -m unittest tests.test_report_fusion_count_map -v`.
  `py_compile.rc=0`, `green.rc=0`, 25 tests passed.
