# GPU Utilization Report

Episodes: 600
Visible devices: cuda:0
Used probe devices: none
Sampled active devices: cuda:0
Unattributed visible devices: cuda:0
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
- cuda:0: max_util_pct=99.0, mean_util_pct=92.72806171648988, active_sample_rate=0.9816779170684667, max_memory_mib=3085.0
- cuda:1: max_util_pct=0.0, mean_util_pct=0.0, active_sample_rate=0.0, max_memory_mib=0.0
- cuda:2: max_util_pct=0.0, mean_util_pct=0.0, active_sample_rate=0.0, max_memory_mib=0.0
- cuda:3: max_util_pct=0.0, mean_util_pct=0.0, active_sample_rate=0.0, max_memory_mib=0.0
- cuda:4: max_util_pct=0.0, mean_util_pct=0.0, active_sample_rate=0.0, max_memory_mib=0.0

## Warnings
- No terminal_probe_devices were recorded in episode diagnostics.
- cuda:1 max utilization 0.0% below 10.0%.
- cuda:2 max utilization 0.0% below 10.0%.
- cuda:3 max utilization 0.0% below 10.0%.
- cuda:4 max utilization 0.0% below 10.0%.

## Recommendations
- Check whether reward probes are balanced across visible GPUs.
- Enable terminal reward-probe diagnostics before judging GPU utilization.
