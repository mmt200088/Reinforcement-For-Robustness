# Stage-1 Worker Apply Configuration Reuse Verification

- Optimization source commit: `5d15e6ca5cb663fe165b922eae7ba3c028f2af0b`
- Red-test commit: `a720977c90e836f818a4f7edb6ba34a10008df6c`
- Server run directory: `/hy-tmp/rfr_stage1_worker_apply_config_reuse_a720977_20260703_211000`
- Scope: `_stage1_evaluate_on_model()` repeated GELU/Softmax installs on a worker handler.

## Red

Command ran on the server against the red-test source package:

`python -m unittest tests.test_stage1_eval_accel.Stage1ApplyConfigurationReuseTest.test_worker_eval_path_reuses_handler_install_without_eval_cache`

Result: `red_rc=1`. The test failed because the second identical worker
configuration call appended six additional handler restore/replace calls.

## Green

Command ran on the server against source commit `5d15e6c`:

`python -m py_compile layer_importance_evaluator.py tests/test_stage1_eval_accel.py`

`python -m unittest tests.test_stage1_eval_accel.Stage1ApplyConfigurationReuseTest.test_worker_eval_path_reuses_handler_install_without_eval_cache`

Result: `green_rc=0`, one test OK.
