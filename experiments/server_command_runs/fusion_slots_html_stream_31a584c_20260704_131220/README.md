# Fusion Slots Eval HTML Streaming Evidence

Source commit: `31a584c`

Optimization: `scripts/render_fusion_count_slots_eval_report.py` now keeps the
existing `render()` string-returning API for compatibility, but the CLI path
uses `write_rendered_html()` and `_HtmlPartsWriter` to stream HTML fragments
directly to the output file. This avoids materializing the full detailed
fusion-count slot-eval report as one large string before writing.

Server evidence:

- RED: `/hy-tmp/fusion_slots_html_stream_red_04abcde_20260704_131220` ran the
  new focused test against the previous source and failed with `red.rc=1`
  because the old CLI path had no `write_rendered_html()` streaming writer and
  still wrote `output_html.write_text(render(...))`.
- GREEN: `/hy-tmp/fusion_slots_html_stream_green_20260704_131220` ran
  `python3 -m py_compile scripts/render_fusion_count_slots_eval_report.py` and
  `python3 -m unittest tests.test_render_fusion_count_slots_eval_report -v`.
  `py_compile.rc=0`, `green.rc=0`, 3 tests passed.
