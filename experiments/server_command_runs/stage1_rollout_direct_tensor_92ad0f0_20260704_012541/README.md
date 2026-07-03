# Stage-1 Rollout Direct Tensor Pack Evidence

Source commit: `92ad0f0` (`Pack Stage-1 rollout scalar tensors on target device`)

## Optimization

Stage-1 `RecurrentRolloutBuffer.get_batch()` now packs rollout `logprobs` and
`values` directly as target-device `torch.float32` tensors. Tensor-backed
rollout fields are stacked once and moved with `stacked.to(device=..., dtype=...)`;
float and mixed fallback inputs preserve scalar conversion semantics.

This supersedes the earlier `54feaa4` CPU-batched path by avoiding the
`torch.from_numpy(logprobs_np).to(device)` and
`torch.from_numpy(values_np).to(device)` round trip before PPO updates.

## Server Evidence

- Red: `rfr_stage1_rollout_direct_tensor_red_d7f3f3d_20260704_012351/red_status.txt`
  has `red_rc=1` against the previous source.
- Green: `rfr_stage1_rollout_direct_tensor_green_d7f3f3d_20260704_012541/green_status.txt`
  has `py_compile_rc=0`, `eval_accel_rc=0`, `parallel_semantics_rc=0`, and
  `functional_rc=0`.
- Functional evidence: `get_batch_functional.log` reports
  `get_batch_direct_tensor_ok cuda:0 ... cuda:0 cuda:0`, proving the packed
  `logprobs` and `values` tensors stayed on the CUDA target device.

No server source tree is included in this evidence directory.
