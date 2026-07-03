# Final Eval Repeat Install Reuse Verification

- Optimization source commit: `567ad752b0ba706f921cd97c88ad00ee97d98aeb`
- Red-test commit: `37974157f0d5d38111031df9ac26f11d6f78d21b`
- Server run directory: `/hy-tmp/rfr_final_eval_repeat_red_49d4734`
- Scope: Paean BLB action final-eval repeat handling.

## Red

Command ran on the server against the red-test source package:

`python -m unittest tests.test_blb_stage2_rl_regressions.BLBActionFinalEvalRegressionTests.test_clean_baseline_reuses_single_configuration_install_without_eval_cache tests.test_blb_stage2_rl_regressions.BLBActionFinalEvalRegressionTests.test_blb_repeat_reuses_single_bridge_install_without_eval_cache`

Result: `red_fixedtest_rc=1`. Both tests failed because repeat handling still
installed the same clean baseline / BLB candidate three times for `repeat_n=3`.

## Green

Command ran on the server against source commit `567ad75`:

`python -m py_compile Paean/blb_action_eval.py tests/test_blb_stage2_rl_regressions.py`

`python -m unittest tests.test_blb_stage2_rl_regressions.BLBActionFinalEvalRegressionTests.test_clean_baseline_reuses_single_configuration_install_without_eval_cache tests.test_blb_stage2_rl_regressions.BLBActionFinalEvalRegressionTests.test_blb_repeat_reuses_single_bridge_install_without_eval_cache`

Result: `green_rc=0`, two tests OK.
