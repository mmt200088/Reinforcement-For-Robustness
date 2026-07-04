# Server Verification

RED:

- Temporary source package: `/hy-tmp/rfr_stage1_default_red_ab9ed62_20260704_092505`
- Command: `python3 -m unittest tests.test_stage1_launcher_defaults -v`
- Expected failure: missing `STAGE1_RL_DEFAULT_BATCH_SIZE="128"` in the old
  launcher source.

GREEN:

- Temporary source package: `/hy-tmp/rfr_stage1_default_green_20260704_092556`
- Command: `python3 -m unittest tests.test_stage1_launcher_defaults tests.test_launcher_gpu_audit -v`
- Result: `Ran 12 tests in 0.052s`, `OK`.

Runtime gate:

- Temporary source package: `/hy-tmp/rfr_stage1_default_batch_ab_9f3864d_20260704_092741`
- Command shape: 170-episode Stage-1 MRPC 1GPU vs 4GPU A/B, no explicit
  `--batch-size` in the launcher command.
- Result: both runs completed, launcher defaulted to Python `--batch_size 128`,
  and `g4` reached `9007.153` ep/h versus `7558.355` ep/h for `g1`.
