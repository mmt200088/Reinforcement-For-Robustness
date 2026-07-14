# GPU Utilization Report

Episodes: 170
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
- cuda:0: max_util_pct=76.0, mean_util_pct=30.84516129032258, active_sample_rate=0.9096774193548387, max_memory_mib=3245.0
- cuda:1: max_util_pct=58.0, mean_util_pct=31.393548387096775, active_sample_rate=0.8516129032258064, max_memory_mib=3065.0
- cuda:2: max_util_pct=52.0, mean_util_pct=31.43225806451613, active_sample_rate=0.8516129032258064, max_memory_mib=3065.0
- cuda:3: max_util_pct=51.0, mean_util_pct=30.0, active_sample_rate=0.8516129032258064, max_memory_mib=3065.0
- cuda:4: max_util_pct=58.0, mean_util_pct=29.974193548387095, active_sample_rate=0.8064516129032258, max_memory_mib=3065.0

## Warnings
- No terminal_probe_devices were recorded in episode diagnostics.

## Recommendations
- Enable terminal reward-probe diagnostics before judging GPU utilization.
