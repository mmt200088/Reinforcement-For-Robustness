# Rescale Incremental Compact Propagation Evidence

Source commit: `85c81a2`

Optimization:
- `build_new_compact_config()` now carries the current scale through each
  adjacent `stage_node_lists` segment exactly once.
- At a retained rescale cut point it records the propagated `sf_pre`, emits the
  existing `sf_post`, and resets the running scale to that post-rescale value.
- The previous implementation rebuilt `nodes_between()` lists and replayed the
  path from the latest retained rescale for every cut point.

Server:
- Host: `f1ac06029e4a`
- GPU inventory: one NVIDIA GeForce RTX 4090 with 24564 MiB
- CPU inventory: 20 logical CPUs
- Python: `3.10.19` from the `llm_ist` environment

TDD and verification:
- RED test commit: `57f67d9`.
- RED focused test: `red/focused.rc` is `1`; the old compact builder called a
  patched `graph.nodes_between()` and raised `replayed cut-point path`.
- GREEN compile and focused return codes are `0`.
- GREEN directly related suites: `green/core.rc` is `0` (`21` tests covering
  Rescale hot paths, bridge/replan behavior, real fused-rescale installation,
  optimizer cost consistency, and multi-profile precision-boost topology).
- Before and after compact JSON for all 12 MRPC graphs is byte-identical.
- `tests.test_blb_fusion_count_map` still has its one pre-existing stale
  `block4 should be degenerate` failure; the unchanged-main reproduction is in
  the preceding `rescale_scalar_restore_2907e63_20260710_204509` evidence.
- Transformer imports emitted existing deprecation `FutureWarning` messages.

Benchmark:
- Workload: MRPC `block4`, three repeats per path.
- Compact construction, 100,000 calls: median `1.499434683995787s` before and
  `1.0715171089977957s` after (`1.3993567357951273x`).
- Full dict-and-compact replan, 50,000 calls: median
  `1.7351223709993064s` before and `1.432390601024963s` after
  (`1.2113472189483234x`).
- The block4 compact SHA-256, full semantic signature, and all 12 graph outputs
  match exactly.
- A 10,000-call cProfile dropped from 2,040,001 to 1,550,001 calls;
  `build_new_compact_config()` cumulative time fell from `0.341s` to `0.216s`.

Scope:
- This is a profiled pure-CPU Rescale enumeration optimization. It preserves
  the compact handoff consumed by Stage-2 and Paean and makes no GPU claim.
