# Stage-2 Action Materialization and Truncation Verification

## Source identity

- Source commit: `1a27544f28e9977430e4d91d23e529d2aa4c8431`
- Server runroot:
  `/hy-tmp/stage2_action_materialization_1a27544f_20260721_213839`
- The SHA-256 values of all 25 changed production and test files matched the
  local worktree exactly.

## Server verification

The server used Python 3.11.12 and Torch 2.9.1+cu128. Tests ran with
`CUDA_VISIBLE_DEVICES` empty so they could not allocate GPU memory or interfere
with the existing five-GPU 150,000-episode training process (PID 22449).

- Core unittest suite: 247 tests passed, 2 skipped. The two skips require two
  visible CUDA devices. Exit code: 0.
- Robust-baseline and robust-reward pytest suite: 49 tests passed, 3 dependency
  deprecation warnings. Exit code: 0.
- The core suite includes a real MRPC materialization regression using the
  checked-in Rescale Optimizer archive. It verifies 59/59 optimizer writebacks,
  all five block families, 12 Block3 requests, selected Block3 K=13, model-ready
  output, and a final post-replan configuration fingerprint.

## Evidence files

- `SOURCE_SYNC_COMMIT`: exact source commit marker.
- `source_sha256.txt`: server-side hashes for the 25 audited files.
- `server_environment.txt`: server runtime and the pre-existing RL process that
  was observed but not modified.
- `server_unittest_final.log` and `.rc`: core suite output and exit code.
- `server_pytest_final.log` and `.rc`: reward suite output and exit code.

The optional `stochastic_ring` truncation backend was verified but remains
disabled. The runtime default remains the historical `binary` implementation.
