# final_variance_scatter_scan_e4c3d47_20260704_054500

Purpose: verify final-evaluation variance scatter panels scan each random result once per metric panel instead of building separate `xs` and `ys` list comprehensions over the same items.

Source commit: e4c3d47
Base evidence commit before source change: 50cabd3
Target branch: jk_standard_rl

Red gate:
- Ran targeted unittest against old `_plot_variance_results()` scatter aggregation.
- Expected failure: a guarded random result raises when `total_cost` is read more than once per variance panel, proving the old `xs` and `ys` comprehensions scanned the same item repeatedly.

Green gate:
- `python -m py_compile final_evaluation_module.py tests/test_final_evaluation_config_cache.py`
- `python -m unittest tests.test_final_evaluation_config_cache -v`
- Source guard confirmed the old `xs`/`ys` list-comprehension patterns are absent and the implementation uses one `for item in items` loop with a single `cost = item.get("total_cost")` read per point/panel.
