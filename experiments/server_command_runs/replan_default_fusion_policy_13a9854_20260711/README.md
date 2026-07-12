# Replan Default Fusion-Policy Cache Evidence

## Change

Source commit `13a9854` resolves and validates each graph's default fusion
policy once during `ReplanSession` construction. The session retains two
representations:

- the original ordered list used by full JSON and diagnostic output;
- an internal immutable normalized set used by repeated replan math.

Explicit custom policies such as `"none"`, `"all"`, or caller-provided pairs
still use the original per-call parser and validator.

## TDD And Gates

- RED commit: `4ea2997`; two default session replans increased resolver and
  normalizer call counts from `(0, 0)` to `(2, 2)`.
- GREEN commit: `13a9854`; default calls reused the construction-time policy,
  while an explicit custom policy still invoked both paths once.
- Related Rescale/fusion gate: 138 tests passed, with two existing optional
  torch-bridge skips.
- Python compilation passed.
- Full suite: 1,230 tests; its 19-item Stage-2 failure set was exactly equal to
  the `9ab82f0` audit, with zero added or removed failures.

## Hot-Loop A/B

The real `block1_mrpc` template enumerated ranks `[0, 100000)` eleven times
per source.

| Source | Median / 100k | Mean / 100k |
| --- | ---: | ---: |
| Parent `4ea2997` | 1.043888 s | 1.043641 s |
| Production `13a9854` | 1.015084 s | 1.015409 s |

Production is `1.0284x` faster with identical valid count, reduced rows, and
result SHA256
`bb33ac65f751f6900b1a9bba8e5513b7743b3a8c3b1582dd385c290da49acbc6`.
The 200,000-combination profile records 11,855,282 calls, 400,000 fewer than
the preceding 12,255,282-call profile because the resolver and generic
normalizer no longer run for each default-policy combination.

## Builder A/B

The real 20-worker builder enumerated all 3,913,140 `block1_mrpc`
combinations three times per source.

| Source | Wall samples | Median | Median user CPU |
| --- | --- | ---: | ---: |
| Parent | 10.51s, 10.55s, 10.59s | 10.55s | 177.34s |
| Production | 10.46s, 10.56s, 10.59s | 10.56s | 176.35s |

The short wall measurement is neutral (`0.9991x`, a 0.01s difference), while
median user CPU drops by `0.99s`. It is not presented as an end-to-end wall
speedup. All generated maps are equal after excluding wall metadata;
canonical map SHA256 is
`b96031eeff4f44ac76a1df3930493ef766847cf4cab23b59e5eed9911f10fc26`.

Raw screen, RED/GREEN logs, full-suite comparison, microbenchmarks, cProfile,
builder runs, maps, hardware inventory, and hashes are retained here.
