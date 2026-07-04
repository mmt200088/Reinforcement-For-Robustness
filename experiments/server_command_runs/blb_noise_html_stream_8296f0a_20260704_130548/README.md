# BLB Noise Verifier HTML Streaming Evidence

Source commit: `8296f0a`

Optimization: `scripts/blb_verify_noise_install.py` now writes smoke and full
noise-install HTML report fragments through `_HtmlPartsWriter` as they are
generated instead of accumulating a full `parts` list and materializing
`"\n".join(parts)` before writing. This reduces report-generation peak memory
for large full-mode verifier reports.

Server evidence:

- RED: `/hy-tmp/blb_noise_html_stream_red_5e2ad74_20260704_130548` ran the new
  focused static test against the previous source and failed with `red.rc=1`
  because the old implementation still used a full `parts` list and joined it
  before `Path.write_text()`.
- GREEN: `/hy-tmp/blb_noise_html_stream_green_20260704_130548` ran
  `python3 -m py_compile scripts/blb_verify_noise_install.py` and
  `python3 -m unittest tests.test_blb_verify_noise_install -v`.
  `py_compile.rc=0`, `green.rc=0`, 3 tests passed.
