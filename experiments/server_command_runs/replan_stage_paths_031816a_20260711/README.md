# Cached Replan Stage-Path Evidence

## Change

Source commit `031816a` precomputes the node sequences between adjacent cut
points of each `ReplanSession` baseline skeleton. Repeated replans reuse those
tuples instead of rebuilding the same `graph.nodes_between()` lists for every
Cartesian-product combination.

The cached paths hold references to the original graph nodes. Per-action
propagation-delta updates therefore remain visible to `propagate_scale()`;
only the repeated topology traversal and list allocation are removed. The
single-shot replan API and callers without a session cache keep the original
path construction behavior.

## TDD And Gates

- RED commit: `0293ccf`; the focused test failed because the second-level
  replan path called `graph.nodes_between()`.
- GREEN commit: `031816a`; the focused tests passed.
- Related Rescale/fusion gate: 129 tests passed, with two existing optional
  torch-bridge skips.
- Python compilation passed for both production files and the regression test.

## Hot-Loop A/B

The same real `block1_mrpc` template enumerated ranks `[0, 100000)` eleven
times per source on the 20-CPU replacement server.

| Source | Median / 100k | Mean / 100k |
| --- | ---: | ---: |
| Parent `0293ccf` | 1.516123 s | 1.517626 s |
| Production `031816a` | 1.423941 s | 1.426765 s |

Production is `1.0647x` faster. Both sources returned 769 valid combinations,
one reduced row, and result SHA256
`8d76a3ab286630d6ee68f7a341961c0673c0c34c264bccb293e8db790bd41b62`.
The production cProfile no longer contains `graph.nodes_between()` in the
200,000-combination hot path.

## Builder A/B

The real builder enumerated all 3,913,140 `block1_mrpc` combinations with 20
workers three times per source.

| Source | Wall samples | Median | Median user CPU |
| --- | --- | ---: | ---: |
| Parent | 12.35s, 12.32s, 12.40s | 12.35s | 211.82s |
| Production | 12.09s, 12.17s, 12.19s | 12.17s | 207.67s |

End-to-end wall improves `1.0148x`, while median user CPU drops by `4.15s`.
All production maps equal the parent map after removing measured wall-time
metadata; canonical map SHA256 is
`b96031eeff4f44ac76a1df3930493ef766847cf4cab23b59e5eed9911f10fc26`.

`comparison.json`, raw RED/GREEN logs, profiles, timing samples, hardware
inventory, and generated maps are retained in this directory.
