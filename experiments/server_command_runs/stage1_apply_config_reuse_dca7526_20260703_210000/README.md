# Stage-1 Apply Configuration Reuse Verification

- Optimization source commit: `dca752672f180679c67864b3eb50612170464213`
- Red-test commit: `b501db7b9fdfb3c7d3b97d5540ecca60ea7417e7`
- Server run directory: `/hy-tmp/rfr_stage1_apply_config_reuse_b501db7_20260703_210000`
- Scope: `LayerImportanceEvaluator.apply_configuration()` repeated GELU/Softmax installs.

## Red

Command ran on the server against the red-test source package:

`python -m unittest tests.test_stage1_eval_accel.Stage1ApplyConfigurationReuseTest.test_repeated_same_config_skips_handler_reinstall_but_keeps_eval_mode`

Result: `red_rc=1`. The test failed because the second identical
configuration call appended six additional handler restore/replace calls.

## Green

Command ran on the server against source commit `dca7526`:

`python -m py_compile layer_importance_evaluator.py tests/test_stage1_eval_accel.py`

`python -m unittest tests.test_stage1_eval_accel.Stage1ApplyConfigurationReuseTest.test_repeated_same_config_skips_handler_reinstall_but_keeps_eval_mode`

Result: `green_rc=0`, one test OK.
