# Fusion RL-Path HTML Streaming Evidence

Source commit: `6d43e0d`

Optimization: `scripts/run_fusion_count_action_eval_rlpath.py` now keeps the
existing `_render_html()` string-returning helper for compatibility, but the
CLI path uses `write_rendered_html()` and `_HtmlPartsWriter` to stream the
RL-path comparison HTML directly to the output file. This avoids materializing
the full RL install-path comparison report as one large string before writing.

Server evidence:

- RED: `/hy-tmp/fusion_rlpath_html_stream_red_fde11b1_20260704_132900` ran the
  new focused test against the previous source and failed with `red.rc=1`
  because the old implementation had no `write_rendered_html()` streaming
  writer and still wrote `output_html.write_text(_render_html(combined))`.
- GREEN: `/hy-tmp/fusion_rlpath_html_stream_green_20260704_132900` ran
  `python3 -m py_compile scripts/run_fusion_count_action_eval_rlpath.py` and
  `python3 -m unittest tests.test_run_fusion_count_action_eval_rlpath -v`.
  `py_compile.rc=0`, `green.rc=0`, 15 tests passed.
