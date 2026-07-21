# Stage-2 rotation materialization verification

This bundle records the TDD and server verification for canonical
Rescale-Optimizer rotation materialization.

## Scope

- Source RED commit: `ae7f1d8b`.
- Source GREEN commit: `d3fa013b6207b300b7556ad46699f85901b9bb43`.
- Server: five-GPU host `i-1.gpushare.com`; tests ran with
  `CUDA_VISIBLE_DEVICES=""` and `nice -n 10` so the unrelated active 150k RL
  process was not stopped or assigned additional GPU work.
- PID 22449 was alive during the test check and disappeared independently
  during evidence collection; no signal or mutation was issued from this task.
- The final documentation-only follow-up does not change the tested runtime
  semantics.

## Result

1. RED: 5 focused tests produced the intended 4 failures and 1 error. They
   exposed missing default optimizer-to-model rotation mappings, collapsed
   repeat counts, silent acceptance of unknown rotations, no repeat-noise
   executor, and absent rotations in a real MRPC materialization.
2. Focused GREEN: all 5 tests passed.
3. Broad GREEN: 357 unittest tests passed; 2 tests were skipped because GPUs
   were deliberately hidden.
4. Reward GREEN: 49 robust-baseline/reward pytest tests passed with only 3
   third-party deprecation warnings.

The broad snapshot initially lacked the tracked `tools/` directory, which made
four artifact/preset tests fail for packaging reasons. The directory was
overlaid from the same `d3fa013b` Git archive and the exact broad command was
rerun; `broad_green.log` is that clean rerun.

## Runtime contract verified

- Default graph-rotation mappings are shared by every canonical materializer.
- One graph rotation can enable multiple concrete model branches.
- Optimizer `count` values survive materialization and cause one independent
  Gaussian sample per concrete rotation.
- Unknown/malformed rotations fail closed before inference.
- Real MRPC all-max materialization installs rotations and preserves Block3 K.
- Existing Block3, all-block K, final-eval/fixed-action/GLUE route, reward, and
  default-binary truncation contracts remain green.

## Files

- `red_tests.log`, `red_tests.rc`: pre-fix evidence.
- `focused_green.log`, `focused_green.rc`: five focused post-fix tests.
- `broad_green.log`, `broad_green.rc`: 357-test expanded regression.
- `reward_green.log`, `reward_green.rc`: 49-test reward regression.
- `source_sha256.txt`: hashes from the tested server snapshot.
- `server_environment.txt`: Python/Torch and untouched active-job evidence.
- `server_snapshot_paths.txt`: exact temporary server snapshot paths.
