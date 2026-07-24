# Stage-2 Integrated Runtime-Efficiency Verification

Tested source: `43045ba760288b4ea01ddab19785b220ff56bccc`

This server run verified the integrated Stage-2 runtime changes:

- one shared F1/F4 probe-worker pool with complete child cleanup;
- bounded in-memory diagnostics with exact JSONL resume state;
- compact v2 candidate evidence combined with the O(1) latest-promotion index;
- v1/v2 mixed-store replay, recovery, and logical-evidence parity.

The focused candidate and layerwise suite passed completely. The broad Stage-2
suite had the same nine pre-existing failures before and after this change, so
the optimization introduced no new failure. Four healthy GPUs passed 80 CUDA,
probe-lock, and multi-device tests. The remaining five-device integration gate
could not run because GPU 3 reports `GPU requires reset`.

The 600-group persistence microbenchmark retained all 3,000 logical trials and
the same latest promotion status. Compact storage reduced bytes by 69.46% and
made this isolated write workload 1.239x faster. This is a persistence
microbenchmark, not an end-to-end RL throughput claim.
