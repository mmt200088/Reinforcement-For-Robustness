# Replan Clean-State Restore Evidence

## Change

Source commit `f5d45c5` tracks whether each preloaded graph is already at its
baseline propagation-delta state. A successful session replan restores the
graph before returning and marks it clean, so the next call skips the redundant
entry restore. A call that starts after an abnormal interruption sees a dirty
marker and still restores before evaluating the next action.

The graph is restored on every normal return exactly as before. A dedicated
server recovery check corrupted a graph, marked it dirty, and proved the next
compact result exactly matched the clean baseline.

## TDD And Gates

- RED commit: `24806c2`; two compact calls performed four restores.
- GREEN commit: `f5d45c5`; two compact calls performed two restores and ended
  clean.
- Related Rescale/fusion gate: 132 tests passed, with two existing optional
  torch-bridge skips.
- Python compilation passed.
- Dirty-state recovery parity passed.

## Hot-Loop A/B

The real `block1_mrpc` template enumerated ranks `[0, 100000)` eleven times
per source.

| Source | Median / 100k | Mean / 100k |
| --- | ---: | ---: |
| Parent `24806c2` | 1.246418 s | 1.247128 s |
| Production `f5d45c5` | 1.218758 s | 1.217658 s |

Production is `1.0227x` faster with identical valid count, reduced rows, and
result SHA256
`8d76a3ab286630d6ee68f7a341961c0673c0c34c264bccb293e8db790bd41b62`.
The production profile performs 200,000 restores for 200,000 combinations,
down from 400,000.

## Builder A/B

The real 20-worker builder enumerated all 3,913,140 `block1_mrpc`
combinations three times per source.

| Source | Wall samples | Median | Median user CPU |
| --- | --- | ---: | ---: |
| Parent | 11.36s, 11.36s, 11.36s | 11.36s | 192.74s |
| Production | 11.00s, 11.05s, 11.21s | 11.05s | 186.47s |

End-to-end wall improves `1.0281x`, and median user CPU drops by `6.27s`.
All generated maps are equal after excluding wall metadata; canonical map
SHA256 is
`b96031eeff4f44ac76a1df3930493ef766847cf4cab23b59e5eed9911f10fc26`.

Raw RED/GREEN logs, recovery parity, microbenchmarks, cProfile, builder runs,
maps, hardware inventory, and hashes are retained here.
