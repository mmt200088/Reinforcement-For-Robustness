# GPU Utilization Report

Episodes: 7744
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
- cuda:0: max_util_pct=51.0, mean_util_pct=31.875, active_sample_rate=0.9666666666666667, max_memory_mib=3245.0
- cuda:1: max_util_pct=87.0, mean_util_pct=30.9, active_sample_rate=0.9, max_memory_mib=3065.0
- cuda:2: max_util_pct=50.0, mean_util_pct=31.091666666666665, active_sample_rate=0.8333333333333334, max_memory_mib=3065.0
- cuda:3: max_util_pct=78.0, mean_util_pct=31.891666666666666, active_sample_rate=0.8916666666666667, max_memory_mib=3065.0
- cuda:4: max_util_pct=51.0, mean_util_pct=31.65, active_sample_rate=0.8833333333333333, max_memory_mib=3065.0

## Warnings
- No terminal_probe_devices were recorded in episode diagnostics.

## Recommendations
- Enable terminal reward-probe diagnostics before judging GPU utilization.
