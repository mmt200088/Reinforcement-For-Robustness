# Stage-1 Parallel Report

Windows: 1
Episodes: 168
Total wall seconds: 161.021
Throughput: 3756.032 ep/h
Mean worker speedup: 3.98

## Worker Balance
- cuda:0: 43
- cuda:1: 43
- cuda:2: 42
- cuda:3: 42

## Component Wall Seconds
- collect: 157.55 (0.978)
- replay: 0.004 (0.000)
- detail: 0.329 (0.002)
- ppo_update: 3.108 (0.019)
- other: 0.029 (0.000)

## Nested Timing Seconds
- model_forward: 559.123
- report_write: 0.329

## Eval Cache
Hits: 0
Misses: 170
Distinct: 170
Hit rate: 0.0

## Warnings
- none
