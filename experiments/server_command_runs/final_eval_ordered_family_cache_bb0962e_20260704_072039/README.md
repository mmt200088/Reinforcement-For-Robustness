# Final Eval Ordered Family Cache Evidence

Source commit: `bb0962e`

This evidence covers a unified final-eval report optimization: `_plot_results()`
now computes the ordered random-result family list once after grouping random
rows and reuses that order across the three comparison panels.

`_plot_variance_results()` already had the same cached-family pattern, so this
change aligns the comparison plot with the variance plot without changing panel
order or plotted values.

## Server Verification

- RED run directory:
  `/hy-tmp/rfr_final_eval_order_cache_red_743e572_20260704_072006`
- GREEN run directory:
  `/hy-tmp/rfr_final_eval_order_cache_green_20260704_072039`

RED command:

```bash
python3 -m unittest tests.test_final_evaluation_config_cache.FinalEvaluationConfigCacheTest.test_final_eval_comparison_plot_reuses_ordered_families -v
```

RED result: failed as expected on the pre-change source because `_plot_results()`
still had `for fam in self._ordered_families(grouped):` inside the panel loop.

GREEN commands:

```bash
python3 -m py_compile final_evaluation_module.py tests/test_final_evaluation_config_cache.py
python3 -m unittest tests.test_final_evaluation_config_cache.FinalEvaluationConfigCacheTest.test_final_eval_comparison_plot_reuses_ordered_families -v
```

GREEN result: `PY_COMPILE_RC=0`, focused unittest passed, `TEST_RC=0`, and the
server wrapper exited 0.

## Local Contents

- `red_unittest.log`: server RED focused unittest log.
- `green_validation.log`: server GREEN py-compile and focused unittest log.
- `source_snapshot/final_evaluation_module.py`: source snapshot from
  `bb0962e`.
- `source_snapshot/test_final_evaluation_config_cache.py`: focused source guard
  snapshot from `bb0962e`.
