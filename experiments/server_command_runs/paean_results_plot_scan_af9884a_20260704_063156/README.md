# Paean BLB Final-Eval Plot Input Streaming Evidence

- Source commit: `af9884a`
- Red server run: `/hy-tmp/rfr_paean_results_plot_red_edd08a1_20260704_063052`
- Green server run: `/hy-tmp/rfr_paean_results_plot_green_20260704_063156`
- Scope: `Paean/blb_action_eval.py` BLB action final-eval comparison plot data preparation.

## Verification

- RED: `red_unittest.log` ran the new source guard before implementation and failed because `_save_results_plot()` still built six separate `np.asarray([... for r in candidate_results])` arrays.
- GREEN: `green_validation.log` passed `python3 -m py_compile Paean/blb_action_eval.py tests/test_blb_final_eval_fusion_fixed_action.py`.
- GREEN: `test_results_plot_scans_candidate_rows_once` passed.

The implementation scans `candidate_results` once to collect labels, loss, loss std, metric, metric std, total bits, and timing columns before converting each column to numpy arrays for matplotlib.
