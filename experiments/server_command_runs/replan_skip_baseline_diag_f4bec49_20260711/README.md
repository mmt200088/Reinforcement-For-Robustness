# Compact Replan Baseline-Diagnostic Evidence

## Change

Source commit `f4bec49` passes `baseline_q_bits=None` only when
`ReplanSession.replan_compact()` invokes the generic replan core. Compact
fusion-map enumeration never consumes `delta_q_vs_baseline`, so this avoids a
baseline list copy plus a per-stage subtraction list for every combination.
Full replan, CLI, and diagnostic callers retain the original baseline vector
and diagnostic output.

## Candidate Selection

Same-process alternating screens compared three low-conflict candidates on the
real `block1_mrpc` template:

| Candidate | Median speedup / 100k |
| --- | ---: |
| Cache default fusion-policy list | 1.0088x |
| Cache policy list and normalized set | 1.0146x |
| Skip unused compact baseline diagnostics | 1.0181x |

The third candidate was selected because it had the highest screen result and
the smallest state/contract surface. All screen result signatures matched.

## TDD And Gates

- RED commit: `a43f2f2`; the new contract test observed compact
  `baseline_q_bits=[57, 56, 31]` instead of `None`.
- GREEN commit: `f4bec49`; the focused test passed while the full replan path
  continued receiving the real baseline vector.
- Related Rescale/fusion gate: 135 tests passed, with two existing optional
  torch-bridge skips.
- Python compilation passed.

## Hot-Loop A/B

The real `block1_mrpc` template enumerated ranks `[0, 100000)` eleven times
per source.

| Source | Median / 100k | Mean / 100k |
| --- | ---: | ---: |
| Parent `a43f2f2` | 1.170702 s | 1.173134 s |
| Production `f4bec49` | 1.115188 s | 1.119432 s |

Production is `1.0498x` faster with identical valid count, reduced rows, and
result SHA256
`bb33ac65f751f6900b1a9bba8e5513b7743b3a8c3b1582dd385c290da49acbc6`.
The 200,000-combination production profile recorded about 400,000 fewer
Python calls than the preceding compact profile because the baseline
subtraction list and its iterator are absent.

## Builder A/B

The real 20-worker builder enumerated all 3,913,140 `block1_mrpc`
combinations three times per source.

| Source | Wall samples | Median | Median user CPU |
| --- | --- | ---: | ---: |
| Parent | 10.99s, 11.01s, 11.03s | 11.01s | 183.43s |
| Production | 10.84s, 10.82s, 10.95s | 10.84s | 183.14s |

End-to-end wall improves `1.0157x`, and median user CPU drops by `0.29s`.
All generated maps are equal after excluding wall metadata; canonical map
SHA256 is
`b96031eeff4f44ac76a1df3930493ef766847cf4cab23b59e5eed9911f10fc26`.

Raw candidate screens, RED/GREEN logs, microbenchmarks, cProfile, builder runs,
maps, hardware inventory, and hashes are retained here.
