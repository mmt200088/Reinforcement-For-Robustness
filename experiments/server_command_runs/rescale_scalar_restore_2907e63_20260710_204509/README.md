# Rescale Scalar State Restore Evidence

Source commit: `2907e63`

Optimization:
- `ReplanSession` snapshots and restores `scale_delta_bits: int` and
  `other_ct_scale_bits: Optional[int]` with direct scalar assignment.
- The previous path called `copy.deepcopy()` once per node during the initial
  snapshot and twice per node for every repeated replan.
- Graph loading, feasibility-DAG construction, fusion policy, compact output,
  and optimizer semantics are unchanged.

Server:
- Host: `f1ac06029e4a`
- GPU inventory: one NVIDIA GeForce RTX 4090 with 24564 MiB
- CPU inventory: 20 logical CPUs
- Python: `3.10.19` from the `llm_ist` environment

TDD and verification:
- RED test commit: `0fc84b6`.
- RED focused test: `red/focused.rc` is `1`; patched `copy.deepcopy()` raised
  from the old snapshot path.
- GREEN compile: `green/py_compile.rc` is `0`.
- GREEN focused test: `green/focused.rc` is `0`.
- GREEN directly related suites: `green/core.rc` is `0` (`8` tests covering
  Rescale hot paths, bridge caching, and replan output).
- The broader GREEN command ran `32` tests and had `31` pass. Its only failure
  was the existing `block4 should be degenerate` assertion in
  `tests.test_blb_fusion_count_map`.
- Unchanged main commit `0cee1b7` reproduced the same fusion-map failure in
  `meta/baseline_fusion_suite.log` (`23/24` passed), so it is not a regression
  from this optimization. Transformer imports also emitted existing
  deprecation `FutureWarning` messages.

Benchmark:
- Workload: MRPC `block4`, 14 graph nodes, baseline-valid replan result.
- Restore-only, 200,000 calls, three repeats: median `0.6161715029738843s`
  before and `0.16961117298342288s` after (`3.632847365745808x`).
- Full replan with dict and compact output, 50,000 calls, three repeats: median
  `2.050284032942727s` before and `1.7536860059481114s` after
  (`1.169128353644051x`).
- The before/after semantic signatures match exactly: validity, fusion count,
  total bits, q vectors, t vector, compact skeleton, and drop order.
- A 10,000-call cProfile dropped from 3,440,001 to 2,040,001 function calls;
  the former 280,000 `deepcopy` calls are absent after the change.

Scope:
- This is a profiled pure-CPU Rescale enumeration optimization. It does not
  claim that this workload is suitable for GPU execution.
