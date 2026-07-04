# Stage-1 Parallel Report

Windows: 1
Episodes: 170
Total wall seconds: 163.508
Throughput: 3742.936 ep/h
Mean worker speedup: 3.98

## Worker Balance
- cuda:0: 43
- cuda:1: 43
- cuda:2: 42
- cuda:3: 42

## Component Wall Seconds
- collect: 160.204 (0.980)
- replay: 0.005 (0.000)
- detail: 0.329 (0.002)
- ppo_update: 2.941 (0.018)
- other: 0.029 (0.000)

## Nested Timing Seconds
- model_forward: 569.828
- report_write: 0.329

## Eval Cache
Hits: 0
Misses: 170
Distinct: 170
Hit rate: 0.0

## Warnings
- none
