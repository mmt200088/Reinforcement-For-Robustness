# GPU Utilization Report

Episodes: 170
Visible devices: cuda:0
Used probe devices: cuda:0
Sampled active devices: cuda:0
Unattributed visible devices: none
Idle visible devices: none

## Probe Timing
Terminal probe mean seconds: 2.6311162712093554
Policy rollout mean seconds: 0.0
Replan/optimizer mean seconds: 0.0

## Probe Wall By Device
- cuda:0: episodes=170, mean_s=2.6311162712093554, min_s=2.6256923810578883, max_s=2.6339686261489987

## Trial Balance
- cuda:0: 850

## Nvidia SMI
- cuda:0: max_util_pct=99.0, mean_util_pct=88.70149253731343, active_sample_rate=0.9477611940298507, max_memory_mib=3091.0
- cuda:1: max_util_pct=0.0, mean_util_pct=0.0, active_sample_rate=0.0, max_memory_mib=0.0
- cuda:2: max_util_pct=0.0, mean_util_pct=0.0, active_sample_rate=0.0, max_memory_mib=0.0
- cuda:3: max_util_pct=0.0, mean_util_pct=0.0, active_sample_rate=0.0, max_memory_mib=0.0
- cuda:4: max_util_pct=0.0, mean_util_pct=0.0, active_sample_rate=0.0, max_memory_mib=0.0

## Warnings
- cuda:1 max utilization 0.0% below 10.0%.
- cuda:2 max utilization 0.0% below 10.0%.
- cuda:3 max utilization 0.0% below 10.0%.
- cuda:4 max utilization 0.0% below 10.0%.

## Recommendations
- Check whether reward probes are balanced across visible GPUs.
