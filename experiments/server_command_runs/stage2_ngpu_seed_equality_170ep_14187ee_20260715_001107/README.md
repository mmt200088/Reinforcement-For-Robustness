# Stage-2 seeded 1-GPU versus 5-GPU smoke: red evidence

This bundle records the 170-episode production correctness smoke at source
`14187ee1c1778f4dee598cb755017effc8332869`. It follows the first strict red
gate at `ba8bb14` and tests the TDD fix that makes `ProbeWorker.run_trial()`
seed the dedicated per-device BLB-noise generator without mutating global
PyTorch or NumPy RNG state.

## Seed fix verification

The server-side CUDA regression suite passed all 13 deterministic-lock tests.
The focused probe/seed suite passed 37 tests with 22 unrelated deselections.
A direct fake noisy-worker check replayed the same `(base_seed, trial_idx)` on
`cuda:0` and `cuda:1` and obtained the exact same loss, accuracy, and F1 tuple:

```text
(0.9047113656997681, 0.5, 0.4874874874874875)
```

The RED, GREEN, focused-suite, and cross-GPU logs are included as
`probe_worker_*.log`.

## Full-path smoke result

Both cases used the same production layerwise robust PPO contract: batch 64,
rollout and PPO interval 120, five online trials, probe size 256, all-4 GELU,
degree-6 Softmax, seed 42, and 170 fresh episodes. Only the requested reward
devices changed from logical device `0` to `0,1,2,3,4`.

| Case | Episodes | PPO updates | Wall time | Episodes/hour |
| --- | ---: | ---: | ---: | ---: |
| 1 GPU | 170 | 2 | 642 s | 953.271 |
| 5 GPU | 170 | 2 | 331 s | 1,848.943 |

The measured speedup was `1.940x`. Every requested physical GPU was sampled
active, but mean utilization was only `28.38%`-`30.33%` per card in the 5-GPU
case, versus `89.25%` for the 1-GPU case. This remains below the final `3.4x`
acceptance target.

Exact episode and PPO equality still failed. Episode 0 has identical actions,
trial seeds, accuracy, and F1 samples, but loss differs by roughly `3e-5` to
`8e-5` per trial. Source tracing found a second deterministic boundary:
`build_probe_runner()` enables TF32/high matmul precision only when the
multi-GPU probe runner is constructed, while the single-device sequential path
does not enable the same process-global mode before its baseline and probes.
This also explains the small robust-baseline loss-threshold difference before
PPO divergence. The next fix must enable one common fast-matmul mode before
both paths evaluate any Stage-2 baseline; disabling the optimized mode only on
the multi-GPU path would sacrifice the throughput objective.

## Diagnostics gap

The raw environment already measures probe install, probe forward, probe clear,
cost evaluation, and per-device work. The layerwise episode recorder currently
drops those fields, so this bundle's utilization reports show zero component
timings and no attributed probe devices even though sampled `nvidia-smi`
activity proves all five cards ran. Propagating those existing fields is the
next diagnostics-only step before changing the install hot path.

## Evidence

`stage2_ngpu_gate_verdict.txt`, both episode/PPO JSONL files, wall files,
utilization reports, sampled `nvidia-smi` CSVs, exact commands, and launcher
logs are the primary evidence. `SHA256SUMS.server` verifies the transferred
top-level bundle. `rl_training_data_points_SHA256SUMS.server` verifies both
project-root structured mirrors committed under
`rl_training_data_points/stage2/bert-base/mrpc/`.

This is intentionally a red intermediate checkpoint. It proves the dedicated
BLB-noise seed fix, but it does not prove full-path 1-GPU/5-GPU equality or the
required final speedup.
