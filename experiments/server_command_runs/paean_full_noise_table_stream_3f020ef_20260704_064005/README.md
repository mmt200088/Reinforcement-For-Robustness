# Paean BLB Full Noise Markdown Table Streaming Evidence

- Source commit: `3f020ef`
- Red server run: `/hy-tmp/rfr_paean_full_noise_table_red_97150f4_20260704_063919`
- Green server run: `/hy-tmp/rfr_paean_full_noise_table_green_20260704_064005`
- Scope: `Paean/blb_action_eval.py` BLB action final-eval full noise/truncation Markdown table generation.

## Verification

- RED: `red_unittest.log` ran the new source guard before implementation and failed because `_full_noise_config_markdown_table()` copied entries with `entries = list(...)`.
- GREEN: `green_validation.log` passed `python3 -m py_compile Paean/blb_action_eval.py tests/test_blb_final_eval_fusion_fixed_action.py`.
- GREEN: `test_full_noise_markdown_table_streams_entries_without_copy` passed.

The implementation iterates the `entries` sequence/iterator directly, avoiding a full temporary list per candidate configuration detail table.
