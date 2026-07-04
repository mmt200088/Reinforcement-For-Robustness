# Stage-1 Batch-128 GPU A/B Probe (`ab9ed62`)

Purpose: test whether the Stage-1 4GPU slowdown from the previous formal gate
was caused by tiny validation batches. This run used the same 170-episode MRPC
Stage-1 PPO gate as `stage1_gpu_ab_180d319_20260704_090423`, but explicitly
passed `--batch-size 128` to both the 1GPU and 4GPU runs.

Source commit: `ab9ed62`

Server run: `/hy-tmp/rfr_stage1_batch128_ab_ab9ed62_20260704_091809`

Result:

- Both `g1` and `g4` completed with `launcher_rc=0`, `wait_rc=0`, and
  `COMPLETED`.
- `g4` used `cuda:0..3` and reported 170 episodes with worker counts
  `43/43/42/42`.
- `g1` stayed effectively flat versus the batch-16 gate: wall `102s` vs `107s`,
  parser throughput `7548.287` vs `7469.518` ep/h.
- `g4` improved materially: wall `91s` vs `197s`, parser throughput
  `8810.700` vs `3742.936` ep/h, and model-forward timing `175.841s` vs
  `569.828s`.
- 4GPU/1GPU wall speedup changed from `0.543x` at batch 16 to `1.121x` at
  batch 128. Parser-throughput speedup changed from `0.501x` to `1.167x`.

Conclusion: Stage-1 RL should not keep the launcher's generic batch-size 16
default when running `run rl --mode stage1-only` without a user-specified
`--batch-size`. Promote a Stage-1-only default of 128 while preserving explicit
user batch-size overrides.
