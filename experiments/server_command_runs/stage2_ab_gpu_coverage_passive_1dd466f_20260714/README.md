# Passive Five-GPU Coverage Gate

This bundle validates the strict physical-GPU activity gate from source
`1dd466fdee44b9b393cbd95fd0e8f7bb86190cb9` against real server telemetry. It
is passive readiness evidence, not the pending 1GPU-versus-5GPU speed result.

## Method

- Snapshot one complete episode from the active formal Stage-2 run. Its
  `terminal_probe_devices` field is empty, matching the production diagnostic
  condition that motivated the sampled-activity fallback.
- Poll all five physical GPUs once per second for 60 seconds. The CSV contains
  300 GPU samples plus its header.
- Run `scripts/gpu_utilization_report.py` with visible devices `0,1,2,3,4` and
  `--require-all-visible-sampled-active` at the default 10% threshold.
- Compare process inventories before and after sampling. Formal PID `10089` is
  the only RL/A/B process, and it remains the only compute process on every GPU.

## Result

The strict gate exited `0`. All five GPUs were sampled active despite absent
episode-level device attribution.

| GPU | Mean utilization | Maximum utilization | Active sample rate | Maximum memory |
| --- | ---: | ---: | ---: | ---: |
| cuda:0 | 33.85% | 43% | 100.00% | 3,245 MiB |
| cuda:1 | 34.95% | 59% | 93.33% | 3,065 MiB |
| cuda:2 | 34.48% | 52% | 91.67% | 3,065 MiB |
| cuda:3 | 33.95% | 49% | 91.67% | 3,065 MiB |
| cuda:4 | 33.13% | 42% | 91.67% | 3,065 MiB |

Primary evidence is in `gpu_utilization.json`, `gpu_utilization.md`,
`nvidia_smi_60s.csv`, `strict_gate.rc`, `processes_before.txt`,
`processes_after.txt`, and `compute_apps_after.csv`. `SHA256SUMS.server`
verifies the 11 files pulled from the server.

This proves that the final A/B harness can enforce real five-card activity
without trusting missing diagnostic fields. It does not prove equality or
speedup; those still require the isolated 600-episode production A/B after the
formal run and its post-run work release the GPUs.
