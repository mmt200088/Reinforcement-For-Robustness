# Final Eval Plot Color Map Reuse Evidence

Source commit: `37890c5`

This evidence covers a unified final-eval report optimization: `_plot_results()`
and `_plot_variance_results()` now use the module-level `_FAMILY_COLOR_MAP`
directly for internal read-only color lookups instead of calling
`_family_colors()` and copying the dictionary for each plot render.

The public `_family_colors()` helper remains available and still returns a fresh
dictionary.

## Server Verification

- RED run directory:
  `/hy-tmp/rfr_final_eval_color_map_red_1edf4aa_20260704_071529`
- GREEN run directory:
  `/hy-tmp/rfr_final_eval_color_map_green_20260704_071600`

RED command:

```bash
python3 -m unittest tests.test_final_evaluation_config_cache.FinalEvaluationConfigCacheTest.test_final_eval_plots_reuse_static_family_color_map -v
```

RED result: failed as expected on the pre-change source because `_plot_results()`
still called `self._family_colors()`.

GREEN commands:

```bash
python3 -m py_compile final_evaluation_module.py tests/test_final_evaluation_config_cache.py
python3 -m unittest tests.test_final_evaluation_config_cache.FinalEvaluationConfigCacheTest.test_final_eval_plots_reuse_static_family_color_map -v
```

GREEN result: `PY_COMPILE_RC=0`, focused unittest passed, `TEST_RC=0`, and the
server wrapper exited 0.

## Local Contents

- `red_unittest.log`: server RED focused unittest log.
- `green_validation.log`: server GREEN py-compile and focused unittest log.
- `source_snapshot/final_evaluation_module.py`: source snapshot from
  `37890c5`.
- `source_snapshot/test_final_evaluation_config_cache.py`: focused source guard
  snapshot from `37890c5`.
