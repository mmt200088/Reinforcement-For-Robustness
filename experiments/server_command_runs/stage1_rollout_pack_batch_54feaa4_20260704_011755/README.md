# Stage-1 Rollout Scalar Pack Batch Evidence

Source commit: `54feaa4` (`Batch Stage-1 rollout scalar tensor packing`)

## Optimization

`RecurrentRolloutBuffer.get_batch()` now converts Stage-1 rollout `logprobs`
and `values` through `_stage1_scalar_episode_values_to_numpy()`. When the
stored values are tensors, the helper stacks the whole rollout batch and
performs one CPU transfer for the field instead of calling `.item()` for every
episode step. Non-tensor and mixed inputs keep the old scalar conversion
fallback.

## Server Evidence

- Red: `rfr_stage1_rollout_pack_red_8336eef_20260704_011436/red_status.txt`
  has `red_rc=1`. The added source guard failed against the pre-optimization
  worktree.
- Green: `rfr_stage1_rollout_pack_green_focus_8336eef_20260704_011755/green_status.txt`
  has `py_compile_rc=0`, `rollout_pack_rc=0`, and `functional_rc=0`.
- Baseline context: `rfr_stage1_eval_accel_base_8336eef_20260704_011649/base_status.txt`
  has `base_rc=1` for two pre-existing `tests.test_stage1_eval_accel` numeric
  failures. Those failures reproduce on clean `8336eef` and are not caused by
  this rollout packing change.

No server source tree is included in this evidence directory.
