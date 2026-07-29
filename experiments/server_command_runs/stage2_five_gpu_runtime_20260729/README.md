# Stage-2 Five-GPU Runtime Evidence

- Performance source: `b9e1e50f03c61e95e3bbde3a46a09a4bf3092a9b`
- Final test source: `63a80e2c474afafd4ab22e38a3c67fcdd45422a8`
- Matched 1 GPU: 10356s (41.715 episodes/hour)
- Matched 5 GPU: 2232s (193.548 episodes/hour)
- Matched scaling: 4.640x (92.8% of ideal)
- Runtime-only gain vs previous source: 1.0801x single GPU, 1.0762x five GPU
- Strict equality: `True` with `0` differences
- All healthy GPUs used: cuda:0,1,2,3,4; no quarantine or restart
- Full unittest: 1933 passed, 3 skipped

See `summary.json` and the referenced gate, utilization, health, and test files.
