# Replan Single-Pass Drop-Bounds Evidence

## Change

Source commit `3e99adc` validates non-positive and over-`q_max` initial drops
in one loop. The old path first evaluated `any(...)` over every drop and then,
for normal inputs, scanned the same vector again with a list comprehension.
The new loop preserves the original error precedence: the first non-positive
drop still returns immediately, while an otherwise positive vector retains
the complete 1-indexed list of over-limit stages.

## TDD And Gates

- RED commit: `66cb6f0`; the source guard found both old scans.
- GREEN commit: `3e99adc`; the focused guard passed.
- Related Rescale/fusion gate: 136 tests passed, with two existing optional
  torch-bridge skips.
- Python compilation passed.
- Parent/production parity was byte-identical for `[0, 61]`, `[31, 61]`,
  `[61, 62]`, and `[-1, 31]`, including messages, stage indices, chains, and
  the non-positive-first precedence.

## Hot-Loop A/B

The real `block1_mrpc` template enumerated ranks `[0, 100000)` eleven times
per source.

| Source | Median / 100k | Mean / 100k |
| --- | ---: | ---: |
| Parent `66cb6f0` | 1.107091 s | 1.105417 s |
| Production `3e99adc` | 1.027356 s | 1.027104 s |

Production is `1.0776x` faster with identical valid count, reduced rows, and
result SHA256
`bb33ac65f751f6900b1a9bba8e5513b7743b3a8c3b1582dd385c290da49acbc6`.
The 200,000-combination profile fell from 13,252,147 calls in the preceding
source profile to 12,255,282 calls, removing 996,865 calls while retaining the
same output signature.

## Builder A/B

The real 20-worker builder enumerated all 3,913,140 `block1_mrpc`
combinations three times per source.

| Source | Wall samples | Median | Median user CPU |
| --- | --- | ---: | ---: |
| Parent | 10.72s, 10.84s, 10.84s | 10.84s | 181.86s |
| Production | 10.62s, 10.63s, 10.83s | 10.63s | 178.91s |

End-to-end wall improves `1.0198x`, and median user CPU drops by `2.95s`.
All generated maps are equal after excluding wall metadata; canonical map
SHA256 is
`b96031eeff4f44ac76a1df3930493ef766847cf4cab23b59e5eed9911f10fc26`.

Raw RED/GREEN logs, invalid-bound parity payloads, microbenchmarks, cProfile,
builder runs, maps, hardware inventory, and hashes are retained here.
