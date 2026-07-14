# GPU Utilization Report

Episodes: 600
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
- cuda:0: max_util_pct=69.0, mean_util_pct=32.47723132969035, active_sample_rate=0.9599271402550091, max_memory_mib=3245.0
- cuda:1: max_util_pct=82.0, mean_util_pct=31.721311475409838, active_sample_rate=0.8961748633879781, max_memory_mib=3065.0
- cuda:2: max_util_pct=49.0, mean_util_pct=31.105646630236794, active_sample_rate=0.8670309653916212, max_memory_mib=3065.0
- cuda:3: max_util_pct=48.0, mean_util_pct=30.714025500910747, active_sample_rate=0.8761384335154827, max_memory_mib=3065.0
- cuda:4: max_util_pct=98.0, mean_util_pct=31.55191256830601, active_sample_rate=0.8907103825136612, max_memory_mib=3065.0

## Warnings
- No terminal_probe_devices were recorded in episode diagnostics.

## Recommendations
- Enable terminal reward-probe diagnostics before judging GPU utilization.
