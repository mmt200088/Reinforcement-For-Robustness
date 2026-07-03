# final_comparison_scatter_scan_c85b896_20260704_061000

Purpose: verify the main final-evaluation comparison scatter panels scan each random result once per metric panel instead of building separate `xs` and `ys` list comprehensions over the same rows.

Source commit: c85b896
Base evidence commit before source change: 1856182
Target branch: jk_standard_rl

Red gate:
- Ran targeted unittest against old `_plot_results()` scatter aggregation.
- Expected failure: a guarded random result raises when `total_cost` is read more than once per metric panel, proving the old `xs` and `ys` comprehensions scanned the same item repeatedly.

Green gate:
- `python -m py_compile final_evaluation_module.py tests/test_final_evaluation_config_cache.py`
- `python -m unittest tests.test_final_evaluation_config_cache -v`
- Source guard confirmed the old `xs`/`ys` list-comprehension patterns are absent and the implementation uses one `for it in items` loop with a single `cost = it.get("total_cost")` read per point/panel.
