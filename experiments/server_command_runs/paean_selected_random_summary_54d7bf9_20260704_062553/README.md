# Paean Selected-vs-Random Summary Streaming Evidence

- Source commit: `54d7bf9`
- Red server run: `/hy-tmp/rfr_paean_selected_random_summary_red2_0114a09_20260704_062357`
- Green server run: `/hy-tmp/rfr_paean_selected_random_summary_green_20260704_062553`
- Scope: `Paean/blb_action_eval.py` selected-vs-random final-eval report summary.

## Verification

- RED: `red_target_unittest.log` ran the new source guard before implementation and failed because `_summarize_selected_vs_random()` still built per-field numpy arrays and separate rank lists.
- GREEN: `green_validation.log` passed `python3 -m py_compile Paean/blb_action_eval.py tests/test_blb_final_eval_fusion_fixed_action.py`.
- GREEN: the two new focused tests passed:
  - `test_selected_vs_random_summary_keeps_existing_statistics`
  - `test_selected_vs_random_summary_streams_random_rows_once`

The implementation keeps the output summary schema and statistics semantics, while scanning `random_results` once to accumulate field stats and anchor ranks.
