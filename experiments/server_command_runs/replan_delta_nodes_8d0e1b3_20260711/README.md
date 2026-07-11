# Cached Replan Delta-Node Evidence

## Change

Source commit `8d0e1b3` caches each preloaded graph's multiplication-node
`name -> node` mapping in `ReplanSession`. Repeated propagation-delta updates
reuse that mapping instead of scanning every graph node and rebuilding the
same dictionary for every fusion-map combination.

The cache contains references to the original nodes. Every action still runs
the existing type validation and writes the requested CTPT/CTCT delta to the
same graph object. The standalone replan API and callers without a session
cache retain the original lookup construction.

## TDD And Gates

- RED commit: `d76baef`; the focused test failed because the session did not
  pass a cached `delta_nodes` mapping.
- GREEN commit: `8d0e1b3`; focused tests passed.
- Related Rescale/fusion gate: 130 tests passed, with two existing optional
  torch-bridge skips.
- Python compilation passed for both production files and the regression test.

## Hot-Loop A/B

The same real `block1_mrpc` template enumerated ranks `[0, 100000)` eleven
times per source.

| Source | Median / 100k | Mean / 100k |
| --- | ---: | ---: |
| Parent `d76baef` | 1.422549 s | 1.423451 s |
| Production `8d0e1b3` | 1.301853 s | 1.302967 s |

Production is `1.0927x` faster. Both paths returned 769 valid combinations,
one reduced row, and result SHA256
`8d76a3ab286630d6ee68f7a341961c0673c0c34c264bccb293e8db790bd41b62`.
The production profile removes the per-call multiplication-node dictionary
comprehension while retaining `_apply_delta_overrides()` validation.

## Builder A/B

The real 20-worker builder enumerated all 3,913,140 `block1_mrpc`
combinations three times per source.

| Source | Wall samples | Median | Median user CPU |
| --- | --- | ---: | ---: |
| Parent | 12.02s, 12.10s, 12.12s | 12.10s | 207.59s |
| Production | 11.52s, 11.54s, 11.55s | 11.54s | 195.11s |

End-to-end wall improves `1.0485x`, and median user CPU falls by `12.48s`.
All generated maps are equal after excluding measured wall metadata; canonical
map SHA256 is
`b96031eeff4f44ac76a1df3930493ef766847cf4cab23b59e5eed9911f10fc26`.

Raw RED/GREEN logs, eleven-round microbenchmarks, cProfile output, three-round
builder timings, generated maps, hardware inventory, and hashes are retained
in this directory.
