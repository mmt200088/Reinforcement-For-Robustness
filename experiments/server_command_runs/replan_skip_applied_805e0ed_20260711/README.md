# Compact Replan Applied-Delta Evidence

## Change

Source commit `805e0ed` stops materializing the per-action
`applied_delta_overrides` echo dictionary when `ReplanSession.replan_compact()`
is used by fast fusion-map enumeration. Delta lookup, node mutation, and type
validation still execute exactly as before. Full replan, CLI, and diagnostic
callers keep recording and returning the applied overrides by default.

## TDD And Gates

- RED commit: `06ceb25`; the compact path had no way to disable the unused
  record and both new tests failed.
- GREEN commit: `805e0ed`; both focused tests passed.
- Related Rescale/fusion gate: 134 tests passed, with two existing optional
  torch-bridge skips.
- Python compilation passed.

## Hot-Loop A/B

The real `block1_mrpc` template enumerated ranks `[0, 100000)` eleven times
per source.

| Source | Median / 100k | Mean / 100k |
| --- | ---: | ---: |
| Parent `06ceb25` | 1.197590 s | 1.197035 s |
| Production `805e0ed` | 1.179304 s | 1.179694 s |

Production is `1.0155x` faster with identical valid count, reduced rows, and
result SHA256
`8d76a3ab286630d6ee68f7a341961c0673c0c34c264bccb293e8db790bd41b62`.

## Builder A/B

The real 20-worker builder enumerated all 3,913,140 `block1_mrpc`
combinations three times per source.

| Source | Wall samples | Median | Median user CPU |
| --- | --- | ---: | ---: |
| Parent | 10.98s, 11.05s, 11.13s | 11.05s | 186.81s |
| Production | 11.09s, 11.01s, 11.14s | 11.09s | 186.35s |

The short end-to-end wall measurement is neutral (`0.9964x`, a 0.04s
difference), while median user CPU drops by `0.46s`. This is consistent with
the small per-combination allocation saving being divided across 20 workers;
it is not presented as a builder wall-time speedup. All generated maps are
equal after excluding wall metadata; canonical map SHA256 is
`b96031eeff4f44ac76a1df3930493ef766847cf4cab23b59e5eed9911f10fc26`.

Raw RED/GREEN logs, microbenchmarks, cProfile, builder runs, maps, hardware
inventory, and hashes are retained here.
