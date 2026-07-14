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
- cuda:0: max_util_pct=65.0, mean_util_pct=30.329032258064515, active_sample_rate=0.896774193548387, max_memory_mib=3245.0
- cuda:1: max_util_pct=62.0, mean_util_pct=29.741935483870968, active_sample_rate=0.8064516129032258, max_memory_mib=3065.0
- cuda:2: max_util_pct=58.0, mean_util_pct=30.258064516129032, active_sample_rate=0.7935483870967742, max_memory_mib=3065.0
- cuda:3: max_util_pct=61.0, mean_util_pct=29.625806451612902, active_sample_rate=0.8193548387096774, max_memory_mib=3065.0
- cuda:4: max_util_pct=68.0, mean_util_pct=28.38064516129032, active_sample_rate=0.8064516129032258, max_memory_mib=3065.0

## Warnings
- No terminal_probe_devices were recorded in episode diagnostics.

## Recommendations
- Enable terminal reward-probe diagnostics before judging GPU utilization.
