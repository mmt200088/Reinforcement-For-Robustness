# Stage-2 18-Group Precision/Stability Evaluation

This directory contains the compact, reproducible evidence from the BERT-base
MRPC validation-full experiment requested on 2026-07-27.

- `aggregate/stage2_18_group_precision_stability.html`: human-readable report.
- `aggregate/stage2_18_group_precision_stability.json`: aggregate metrics,
  paired comparisons, constraints, installed K evidence, and provenance.
- `seed_*/precision_stability_grid_seed_result.json`: all five raw trials for
  every one of the 18 groups under one experiment seed, including full boosted
  overrides and post-materialization K audits.
- `seed_*.log`, `seed_*.rc`, and `seed_*.gpu`: worker logs and exit/device
  records.
- `manifest.txt`, `events.log`, `status.txt`, `aggregate.log`, and
  `aggregate.rc`: exact source identity and orchestration evidence.

The run used source commit `7b5abc8b149aaeabdcab7a1a2ab0386172c44fb7`
and tree `ea28f74d2b4cf8d090370c15e9c540fa66716a3d`. Five experiment
seeds each evaluated five paired noise trials per group on the 408-example MRPC
validation set. Every group passed the runtime installation audit: 60
post-materialization K slots, a stable non-empty model-configuration
fingerprint, a real forward pass, and exact fusion totals of 0, 24, or 36.

Four server CUDA devices were usable. The fifth physical GPU was visible to
`nvidia-smi` but rejected CUDA initialization; the container lacked permission
to reset it. Four seeds ran concurrently and the fifth started immediately
after GPU 0 became free. This changes wall time only, not trial semantics.
