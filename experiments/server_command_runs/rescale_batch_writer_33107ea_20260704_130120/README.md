# Rescale Batch Archive Writer Evidence

Source commit: `33107ea`

Optimization: `Rescale_optimizer/scripts/batch_run_configs.py` now streams the
compact static-skeleton archive directly to the output file handle through
`_write_doc()` instead of first materializing the whole archive string and then
calling `Path.write_text()`. `_format_doc()` remains available for callers that
need the string form.

Server evidence:

- RED: `/hy-tmp/rescale_batch_writer_red_344ec28_20260704_130120` ran the new
  focused test against the previous source and failed with `red.rc=1` because
  `main()` still used `_format_doc(entries...)` plus `write_text(out_text, ...)`.
- GREEN: `/hy-tmp/rescale_batch_writer_green_20260704_130120` ran
  `python3 -m py_compile Rescale_optimizer/scripts/batch_run_configs.py` and
  `python3 -m unittest tests.test_rescale_config_discovery -v`.
  `py_compile.rc=0`, `green.rc=0`, 3 tests passed.
