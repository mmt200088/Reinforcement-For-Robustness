# Paean Fusion Fixed-Action Decode Copy Wrapper Evidence

Source commit: `5fe7760`

Scope: `Paean/blb_action_eval.py`
`BLBActionFinalEvaluationModule._decode_fusion_count_fixed_action()`.

Optimization:

- Iterate `group.option_by_graph` / `group.option_by_step` mappings directly
  instead of copying them through `dict(...)` before normalization.
- Build per-step block slices with `np.take(base_arr, block_offsets)` instead
  of materializing `list(block_offsets)` for numpy indexing.
- Iterate the selected fusion option field mapping directly instead of copying
  it through `dict(option_fields).items()`.

This trims short-lived Python list/dict wrappers from each fusion fixed-action
decode while preserving the same normalized metadata dictionaries, decoded
field updates, and selected K restoration logic.

Server RED:

- Run directory:
  `/hy-tmp/rfr_paean_fusion_decode_copy_red_62cae98_20260704_072200`
- Command:
  `python3 -m unittest tests.test_blb_final_eval_fusion_fixed_action.FusionCountFixedActionDecodeTest.test_fusion_fixed_action_decode_avoids_step_copy_wrappers -v`
- Result: expected failure because the old decode path still contained
  `base_arr[list(block_offsets)]`.

Server GREEN:

- Run directory:
  `/hy-tmp/rfr_paean_fusion_decode_copy_green_20260704_072520`
- Commands:
  `python3 -m py_compile Paean/blb_action_eval.py tests/test_blb_final_eval_fusion_fixed_action.py`
  and the focused copy-wrapper guard.
- Result: `PY_COMPILE_RC=0`, `TEST_RC=0`, `GREEN_RC=0`.

Baseline note:

- The existing `test_per_step_fusion_option_replay_preserves_rl_selected_k`
  fails on clean source commit `62cae98` with `13 != 14`; this is recorded in
  `baseline_existing_failure.log` and is not used as the gate for this
  optimization.
