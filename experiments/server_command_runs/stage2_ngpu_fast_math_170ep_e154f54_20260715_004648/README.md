# Stage-2 fast-matmul parity smoke

This server run validates source commit
`e154f54484f017555264d515df87967c9bf24dab` after enabling the same CUDA
fast-matmul mode before both the one-GPU and five-GPU evaluator setup paths.

## Result

- Episodes: 170 one-GPU and 170 five-GPU.
- Episode quality/effect equality: PASS.
- Strict diagnostic equality: PASS.
- PPO update equality: PASS (2 updates per run).
- Wall time: 571 seconds one-GPU, 331 seconds five-GPU.
- Throughput speedup: 1.725x.
- Sampled active devices: all five GPUs in the five-GPU run.

The equality fix is accepted. The 1.725x speedup is not the final performance
acceptance result; the project target remains at least 3.4x on the strict
600-episode gate.

## Diagnostic gap

The GPU sampler measured mean utilization near 30-31% on all five cards, but
the layerwise episode writer recorded terminal probe timing and device fields
as zero or empty. The underlying environment already exposes those values;
the next optimization step is to propagate them through
`LayerwiseEpisodeRecord` into the existing `EpisodeStats` fields before
changing any runtime behavior.

## Included evidence

- `stage2_ngpu_gate_verdict.txt`: strict parity and timing verdict.
- `one_episodes.jsonl`, `many_episodes.jsonl`: copied diagnostic episodes.
- `one_ppo_updates.jsonl`, `many_ppo_updates.jsonl`: PPO diagnostics.
- `*_gpu_utilization.*`, `*_nvidia_smi.csv`: sampled GPU evidence.
- `stage2_fast_math_red_7f945ec.log`: server RED contract test.
- `stage2_fast_math_green_e154f54.log`: server GREEN focused tests.
- `stage2_fast_math_cuda_determinism_e154f54.log`: CUDA determinism tests.
- Project-root `rl_training_data_points/`: both required raw data mirrors.

Temporary model checkpoints were intentionally excluded.
