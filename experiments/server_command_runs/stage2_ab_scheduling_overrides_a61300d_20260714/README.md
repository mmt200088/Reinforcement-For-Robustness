# Stage-2 A/B scheduling-command readiness evidence

This bundle records a no-GPU RED/GREEN correction and readiness audit for the
pending production 1GPU-versus-5GPU Stage-2 gate.

## Problem and fix

The A/B harness accepted and logged `ONE_WORKERS_PER_DEVICE`,
`MANY_WORKERS_PER_DEVICE`, `POLICY_DEVICE`, and `DYNAMIC_ASSIGNMENT`, but its
effective command omitted all four values. Source `9fe4d0c` added a preflight
contract using `one_wpd=2`, `many_wpd=3`, `policy=cpu`, and `dynamic=0`; it
failed because the first policy environment field was absent.

Source `a61300d` now builds one `CASE_ENV` array containing CUDA visibility,
policy placement, dynamic assignment, and timeout. Both command recording and
real execution consume that same array. The case-specific worker count is also
forwarded through `--stage2-workers-per-device`.

## Verification

| Gate | Result | Evidence |
| --- | --- | --- |
| RED focused contract | expected failure, rc 1 | `red/test.log` |
| GREEN focused contract | 1 passed | `green/targeted_test.log` |
| Bash syntax | rc 0 | `green/bash_n.rc` |
| Launcher/comparator modules | 18 passed | `green/related_tests.log` |
| Full pytest | 1,599 passed, 6 skipped | `green/full_pytest.log` |
| Non-default no-GPU preflight | rc 0; one=2 workers, many=3 workers | `green/preflight/effective_commands.txt` |
| Occupied-host idle gate | expected rc 20; no new training PID | `idle_gate_wrapper.log`, `rl_tune_processes_after_idle_gate.txt` |

The idle gate sampled all five RTX 5090s above the 2,048 MiB threshold and
stopped before model loading or launcher execution. The only `rl_tune.py`
process afterward was the pre-existing formal PID `10089`.

## Production-path interpretation

The current production gate is layerwise and uses `reward_devices` to split
the deterministic K reward trials. Source tracing shows that
`stage2_workers_per_device`, policy placement, and dynamic assignment are
consumed by the mutually exclusive `stage2_rl_devices` episode-parallel path.
Consequently, this correction proves command integrity but is not itself a
speedup result. The final production A/B must run once with fixed
`1:worker:1`, and its speed claim must come only from the one-device versus
five-device reward-trial split. Redundant candidate sweeps would not test the
production mechanism.

## Server prerequisites

The generic defaults `/hy-tmp/hf_cache` and `/hy-tmp/glue_data` are absent on
this host. The active formal process uses these existing run-local paths, which
the final A/B should reuse explicitly and read-only:

- `/hy-tmp/stage2_layerwise_robust_24e919c_20260713_175542/hf_cache`
- `/hy-tmp/stage2_layerwise_robust_24e919c_20260713_175542/local_glue`

At capture time the formal run was healthy at 15,720/60,000 episodes (26.2%),
terminal priority P3, `last_invalid=false`, with approximately 20.0 hours
remaining. `/hy-tmp` had about 40 GiB free.

This bundle does not claim the final parity or throughput result. The
transferred archive
`/hy-tmp/stage2_ab_scheduling_overrides_a61300d_20260714.tar.gz` has SHA-256
`8ab81ceebceed69890ac46f56b630383bcf31fd5aa91146f8201f20ca0f22273`.
