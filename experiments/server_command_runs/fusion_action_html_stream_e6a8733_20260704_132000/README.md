# Fusion Action Eval HTML Streaming Evidence

Source commit: `e6a8733`

Optimization: `scripts/run_fusion_count_action_eval.py` now keeps the existing
`_render_html()` string-returning helper for compatibility, but the CLI path
uses `write_rendered_html()` and `_HtmlPartsWriter` to stream the comparison
HTML directly to the output file. This avoids materializing the full fixed
fusion-count action-eval HTML report as one large string before writing.

Server evidence:

- RED: `/hy-tmp/fusion_action_html_stream_red_93b9a32_20260704_132000` ran the
  new focused test against the previous source and failed with `red.rc=1`
  because the old implementation had no `write_rendered_html()` streaming
  writer and still wrote `output_html.write_text(_render_html(combined))`.
- GREEN: `/hy-tmp/fusion_action_html_stream_green_20260704_132000` ran
  `python3 -m py_compile scripts/run_fusion_count_action_eval.py` and
  `python3 -m unittest tests.test_run_fusion_count_action_eval -v`.
  `py_compile.rc=0`, `green.rc=0`, 8 tests passed.
