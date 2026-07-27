# Stage-2 27-Group Precision/Stability Evaluation

This directory contains the compact, reproducible evidence from the BERT-base
MRPC validation-full experiment requested on 2026-07-27.

- `aggregate/stage2_27_group_precision_stability.html`: human-readable report.
- `aggregate/stage2_27_group_precision_stability.json`: aggregate metrics,
  paired comparisons, constraints, installed K evidence, and provenance.
- `seed_*/precision_stability_grid_seed_result.json`: all five raw trials for
  every one of the 27 groups under one experiment seed, including full boosted
  overrides and post-materialization K audits.
- `seed_*.log`, `seed_*.rc`, and `seed_*.gpu`: worker logs and exit/device
  records.
- `manifest.txt`, `events.log`, `status.txt`, `aggregate.log`, and
  `aggregate.rc`: exact source identity and orchestration evidence.

The run used source commit `6ca65e22449030108cba70c5d42b2143f315a43e`
and tree `3886850ab21ecf434ae4f2602470a8545b8e84e9`. Five experiment
seeds each evaluated five paired noise trials per group on the 408-example MRPC
validation set, for 675 model inferences in total. Every group passed the
runtime installation audit: 60 post-materialization K slots, one stable
non-empty model-configuration fingerprint, a real forward pass, and exact
fusion totals of 0, 24, or 36.

The two added schedules use one-based human layer numbering. Layers
1/3/5/7/9/11 use the high or medium profile, while layers 2/4/6/8/10/12 use
the low profile. The exact 12-layer schedules and all 60 installed K values are
stored in every seed JSON and in the aggregate JSON.

Four server CUDA devices were usable. The fifth physical GPU was visible to
`nvidia-smi` but rejected CUDA initialization; the container lacked permission
to reset it. Four seeds ran concurrently and the fifth started immediately
after GPU 0 became free. This changes wall time only, not trial semantics.
