# Stage-2 strict 1-GPU versus 5-GPU gate: red evidence

This bundle records the first isolated 600-episode production A/B at source
`ba8bb140723a6300c9ee14aadeb7e073737424a9`. The formal Stage-2 process and
its post-run work had exited, all five RTX 5090s were at `0 MiB`, and no other
GPU compute process was present before launch.

## Contract

Both cases used the same layerwise robust PPO command: batch 64, rollout and
PPO update interval 120, five online noise trials, probe size 256, all-4 fixed
GELU with degree-6 Softmax, seed 42, and 600 fresh episodes. The only intended
throughput variable was `--blb-v3-reward-devices`: logical device `0` versus
logical devices `0,1,2,3,4`. Acceptance required all of the following:

- exactly 600 episode rows and 5 PPO updates per case;
- exact quality/effect and PPO equality;
- end-to-end wall-clock speedup of at least `3.4x`;
- sampled activity on every requested physical GPU.

## Result

The gate correctly failed.

| Case | Episodes | PPO updates | Wall time | Episodes/hour |
| --- | ---: | ---: | ---: | ---: |
| 1 GPU | 600 | 5 | 2,251 s | 959.574 |
| 5 GPU | 600 | 5 | 1,171 s | 1,844.577 |

Measured speedup was only `1.922x`, below `3.4x`. Quality/effect equality and
PPO equality also failed. The independent comparator rerun exited `2`; see
`stage2_ngpu_gate_recheck.rc` and `stage2_ngpu_gate_verdict_recheck.txt`.

Physical-GPU coverage itself passed. The single-card run averaged 92.73% on
`cuda:0`. The five-card run sampled all requested cards active, but mean
utilization was only 30.71%-32.48% per GPU, with active-sample rates of
86.70%-95.99%. This is real multi-GPU execution, but it is not efficient
enough and is not result-equivalent.

## Root-cause trace

The one-card and five-card robust baseline records contain the same 25
predicted trial seeds, in the same group and trial order, but produce different
loss/metric samples and therefore different constraint thresholds from episode
0. Source tracing identifies the broken boundary:

- the single-device deterministic path calls `noise_rng_scope()` and
  `reseed_noise_rng_for_device()` before every trial;
- BLB noise sampling uses `function_handler`'s independent per-device
  `torch.Generator`, intentionally isolated from PyTorch's global RNG;
- multi-GPU `ProbeWorker.run_trial()` records the derived seed but only calls
  `torch.manual_seed()`, `numpy.random.seed()`, and
  `torch.cuda.manual_seed()`; it never reseeds the independent BLB generator.

Thus the multi-GPU seed diagnostics describe the intended streams, not the
noise streams actually consumed by model forwards. This explains the baseline,
episode, and PPO divergence. The performance evidence separately shows a
large multi-worker setup/host overhead: aggregate GPU work rises, but each card
falls from a 92.73% single-card mean to about 31%. That bottleneck needs direct
timing evidence before optimization; it is not being conflated with the RNG
correctness fix.

## Evidence

Primary files are `stage2_ngpu_gate_verdict.txt`,
`stage2_ngpu_gate_verdict_recheck.txt`, `one_episodes.jsonl`,
`many_episodes.jsonl`, both PPO JSONL files, wall-time files, utilization
JSON/Markdown reports, sampled `nvidia-smi` CSVs, effective commands, and
launcher logs. `SHA256SUMS.server` verifies every transferred top-level file.
`rl_training_data_points_SHA256SUMS.server` verifies both project-root raw
training-data mirrors, which are committed under
`rl_training_data_points/stage2/bert-base/mrpc/`.

This bundle is intentionally a red checkpoint. It does not close the runtime
optimization goal and must not be cited as proof that five-GPU production is
equivalent or meets the throughput target.
