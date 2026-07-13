# Sampled GPU Activity Classification Evidence

This bundle verifies the low-conflict Stage-2 GPU diagnostics change at source
commit `19c5a10986ceefe690d45083cdd2f354fd07b2b1` on the five-RTX-5090 server.
It does not claim that the still-pending 1GPU-versus-5GPU equality and speed
gate has passed.

## Result

- RED commit `a732054f229be3ce7229de64a04abba07cd9013c` failed because a sampled-active
  `cuda:0` was incorrectly included in `idle_visible_devices`.
- GREEN passed all 20 related report and evidence-bundle tests plus
  `py_compile` with CUDA hidden from the test process.
- The GREEN reporter processed 7,744 real episode rows and the 120-sample GPU
  CSV in `1.07s`. It classified all five GPUs as sampled-active, none as idle,
  and retained all five as unattributed because the concurrent layerwise run
  did not record `terminal_probe_devices`.
- The passive profile observed the concurrent Stage-2 source commit `24e919c`
  for 137 seconds. It added 64 rows (`1,681.75` rows/hour); per-GPU mean
  utilization was `30.90%` to `31.89%`, with about `3.0` to `3.2 GiB` used per
  GPU. This is profiling evidence only because the formal run occupied every
  GPU throughout the sample.

## Server Context

- GPU: 5 x NVIDIA GeForce RTX 5090, 32,607 MiB each.
- Runtime: PyTorch `2.9.1+cu128`; all five `sm_120` devices were visible.
- CPU/RAM: 256 logical CPUs and 629 GiB RAM.
- The preflight passed 17 Stage-2 N-GPU comparator/runner tests, Bash syntax
  checks, and Python compilation without using CUDA.
- After the evidence commits, server head `d9ee378` passed 29 project-audit,
  GPU-report, and evidence-bundle tests, compiled all three completion tools,
  verified every checksum, and produced a six-stage project audit.

## Files

- `red/`: focused failing test and clean Git status at the RED commit.
- `green/`: focused GREEN tests, real report, compile result, and timing.
- `passive_profile/`: raw `nvidia-smi` samples and bounded summary from the
  active concurrent run; its report is the pre-fix output that falsely labels
  all five sampled-active GPUs idle.
- `preflight/`: server environment and A/B harness readiness checks.
- `post_sync/`: final-head focused regression, checksum, and six-stage audit
  evidence before the evidence-only closeout commit.
