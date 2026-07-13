# GPU Utilization Report

Episodes: 1
Visible devices: cuda:0, cuda:1, cuda:2, cuda:3, cuda:4
Used probe devices: none
Sampled active devices: cuda:0, cuda:1, cuda:2, cuda:3, cuda:4
Unattributed visible devices: cuda:0, cuda:1, cuda:2, cuda:3, cuda:4
Idle visible devices: none

## Probe Timing
Terminal probe mean seconds: 0.0
Policy rollout mean seconds: 0.0
Replan/optimizer mean seconds: 0.0

## Probe Wall By Device
- none

## Trial Balance
- none

## Nvidia SMI
- cuda:0: max_util_pct=43.0, mean_util_pct=33.85, active_sample_rate=1.0, max_memory_mib=3245.0
- cuda:1: max_util_pct=59.0, mean_util_pct=34.95, active_sample_rate=0.9333333333333333, max_memory_mib=3065.0
- cuda:2: max_util_pct=52.0, mean_util_pct=34.483333333333334, active_sample_rate=0.9166666666666666, max_memory_mib=3065.0
- cuda:3: max_util_pct=49.0, mean_util_pct=33.95, active_sample_rate=0.9166666666666666, max_memory_mib=3065.0
- cuda:4: max_util_pct=42.0, mean_util_pct=33.13333333333333, active_sample_rate=0.9166666666666666, max_memory_mib=3065.0

## Warnings
- No terminal_probe_devices were recorded in episode diagnostics.

## Recommendations
- Enable terminal reward-probe diagnostics before judging GPU utilization.
