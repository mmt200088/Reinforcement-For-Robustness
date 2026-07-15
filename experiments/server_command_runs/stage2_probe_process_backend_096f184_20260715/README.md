# Stage-2 persistent reward-probe processes

This evidence set records the diagnosis and first real-GPU validation of the
persistent-process reward-probe backend.

## Diagnosis

Five independent one-GPU processes at source `bff6403` each ran 170 episodes
with two online trials per candidate. Their mean per-trial latency was
`0.524324s`, or `0.9964x` the sequential one-GPU reference (`0.526223s`) and
`0.3988x` the old five-thread critical path (`1.314906s`). This confirms that
same-process Python dispatch contention, rather than CUDA compute or BLB
installation, caused the multi-GPU slowdown.

The two failed attempts are retained separately. The first lacked the offline
Hugging Face endpoint/cache settings. The second reached real inference but
used one candidate-assessment trial, while the statistical contract requires
at least two. Neither failure is evidence against the process design.

## Implementation gate

- RED source `ddb328d`: seven focused tests failed on the missing process
  scheduler/backend contracts.
- GREEN source `096f184`: the CPU-focused suite passed `40` tests with the two
  CUDA-only tests skipped; the two-GPU suite passed all `42` tests.
- `BLB_STAGE2_PROBE_BACKEND=process` is the default. The legacy in-process
  implementation remains available through `BLB_STAGE2_PROBE_BACKEND=thread`.

## Real 170-episode result

The five-GPU process run used source `096f184`, seed 42, K=5, probe size 256,
and the production layerwise robust-constrained command. It completed in
`171s`; the previously verified one-GPU reference completed in `582s`, giving
`3.404x` end-to-end speedup. Research outputs, strict diagnostics, and both PPO
updates compared exactly. Probe latency fell from `2.6311s` to `0.5290s` per
episode. All five RTX 5090s were sampled active, with mean utilization between
`59.60%` and `64.93%` over startup, training, and shutdown.

This is not the final acceptance run: the one-GPU side was reused from source
`9099cfa`. A fresh 600-episode one-versus-five comparison from the same final
source must still preserve exact episode/PPO outputs and speedup of at least
`3.4x`.

## Contents

- `hypothesis_k2_success/` and `hypothesis_k2_raw_training_data/`: independent
  process timing evidence and complete structured mirrors.
- `hypothesis_failed_*` and `failed_k1_raw_training_data/`: precondition-failure
  audit artifacts.
- `red_test/`, `green_cpu_tests.log`, `green_two_gpu_tests.log`: TDD evidence.
- `real_170ep/`: compact A/B artifacts, full structured raw mirror, GPU samples,
  exact-comparison verdict, and runtime logs. Checkpoints are excluded.
