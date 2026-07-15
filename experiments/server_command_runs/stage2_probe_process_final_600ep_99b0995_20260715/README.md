# Stage-2 persistent reward-probe final gate

This bundle records the final same-source 600-episode acceptance run for the
persistent multi-process reward-probe backend. Both sides ran from clean source
`99b0995512831904d3f84dfe0051a90551ff95db` on the same five-RTX-5090 server.

## Acceptance result

| Gate | One GPU | Five GPUs | Result |
| --- | ---: | ---: | --- |
| Episodes | 600 | 600 | complete |
| PPO updates | 5 | 5 | exact equality |
| Wall time | 1,951s | 521s | `3.745x` speedup |
| Throughput | 1,107.125 ep/h | 4,145.873 ep/h | `3.745x` speedup |
| Terminal probe mean | 2.632s/episode | 0.530s/episode | `4.97x` reduction |
| Quality/effect rows | reference | exact match | pass |
| Strict diagnostics | reference | exact match | pass |

The required end-to-end threshold was `3.4x`. The five-GPU run assigned one of
the five trials to each device on every episode, so every GPU completed exactly
600 trials. Reward, constraints, action selection, trial count, PPO updates,
and validation semantics were unchanged.

## Hardware evidence

| Device | Mean utilization | Active-sample rate | Peak utilization |
| --- | ---: | ---: | ---: |
| `cuda:0` | 73.55% | 93.50% | 99% |
| `cuda:1` | 71.25% | 82.93% | 99% |
| `cuda:2` | 73.60% | 83.74% | 99% |
| `cuda:3` | 72.47% | 83.74% | 99% |
| `cuda:4` | 69.02% | 83.33% | 99% |

No requested device was idle. After both runs exited, the resource snapshot
found all five GPUs at 0 MiB and 0% utilization, no remaining probe process,
and a clean server checkout at the tested source commit.

## Regression and audit

- Full server pytest with CUDA intentionally hidden: `1611 passed, 8 skipped,
  5 warnings` in 77.63 seconds, exit 0.
- Whole-project optimization audit: 30/30 expected flow files present and no
  missing artifact-evidence class.
- Transferred archive SHA-256:
  `80fa6c6816c256a377cef859d8d58a63f1d55ba2fa2d4df1aea24137fb3042e6`.
- All 85 transferred checksum entries passed again after local extraction.

The eight skips require visible CUDA devices or Python 3.9. Real CUDA behavior
is covered by this 600-episode A/B run and the earlier focused two-GPU process
suite.

## Contents

- `stage2_ngpu_gate_verdict.txt` and `stage2_ngpu_speed_ab_stdout.log`: strict
  equality and speed-gate output.
- `one_*` and `many_*`: episode/PPO rows, wall times, launch logs, GPU samples,
  and utilization summaries.
- `persistent_one/` and `persistent_many/`: compact run manifests, logs,
  summaries, action diagnostics, and curves.
- `raw_training_data/`: both complete structured training-data mirrors. They
  are also promoted under `rl_training_data_points/stage2/bert-base/mrpc/`.
- `full_pytest.*`, `project_optimization_audit.*`, and
  `post_run_resource_snapshot.*`: final acceptance evidence.
- `MANIFEST.txt`, `SHA256SUMS`, and `SHA256SUMS.verify.log`: transfer manifest
  and integrity record.

The compact bundle excludes model checkpoints, candidate stores, and redundant
large candidate/frontier files. Those files are not needed to reproduce the
performance verdict or redraw the recorded training curves.
