# Stage-1 Parallel Report

Windows: 1
Episodes: 170
Total wall seconds: 67.946
Throughput: 9007.153 ep/h
Mean worker speedup: 3.98

## Worker Balance
- cuda:0: 43
- cuda:1: 43
- cuda:2: 42
- cuda:3: 42

## Component Wall Seconds
- collect: 64.3 (0.946)
- replay: 0.004 (0.000)
- detail: 0.33 (0.005)
- ppo_update: 3.283 (0.048)
- other: 0.029 (0.000)

## Nested Timing Seconds
- model_forward: 174.264
- report_write: 0.33

## Eval Cache
Hits: 0
Misses: 170
Distinct: 170
Hit rate: 0.0

## Warnings
- none
