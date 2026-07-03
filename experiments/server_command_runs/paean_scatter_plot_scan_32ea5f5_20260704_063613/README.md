# Paean BLB Final-Eval Scatter Input Streaming Evidence

- Source commit: `32ea5f5`
- Red server run: `/hy-tmp/rfr_paean_scatter_plot_red_3ba750f_20260704_063519`
- Green server run: `/hy-tmp/rfr_paean_scatter_plot_green_20260704_063613`
- Scope: `Paean/blb_action_eval.py` BLB action final-eval selected-vs-random scatter plot data preparation.

## Verification

- RED: `red_unittest.log` ran the new source guard before implementation and failed because `_save_scatter_plot()` still used `_xs_ys()` plus secondary-metric list comprehensions over `selected_results` / `random_results`.
- GREEN: `green_validation.log` passed `python3 -m py_compile Paean/blb_action_eval.py tests/test_blb_final_eval_fusion_fixed_action.py`.
- GREEN: `test_scatter_plot_scans_result_rows_once_per_group` passed.

The implementation scans each result group once to collect primary and secondary scatter columns, then reuses those columns for both panels.
