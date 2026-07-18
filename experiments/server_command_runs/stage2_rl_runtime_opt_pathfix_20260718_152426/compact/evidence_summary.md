# Stage-2 Runtime Optimization Evidence

- Source: `de8bb2d276818ab1e6c362103c1d6c3652a83bda` vs baseline `48b03e869934aa8b3aa904a1fe8b611a1e2d618a`
- Semantic parity: **PASS** (600 episodes, 5 PPO updates, normalized candidate evidence exact)
- Wall time: 1255s -> 1209s (1.038x; 3.67% less wall time)
- Summed per-GPU sampled peak memory: 32054 -> 18165 MiB (43.33% lower)
- Probe replica processes: 8 -> 4 (50% fewer)
- Candidate store: 24816152 -> 9838606 bytes (60.35% lower)
- Mean sampled GPU utilization: 82.23% -> 85.39% (+3.15 points)
- Episode 458: both `failed_probability_gate`, candidate key and all 25 F4 seeds exact

Baseline 48b03e8 does not emit parent RSS telemetry. The optimized parent peaked at 3.670 GiB; no unsupported RSS reduction claim is made.
