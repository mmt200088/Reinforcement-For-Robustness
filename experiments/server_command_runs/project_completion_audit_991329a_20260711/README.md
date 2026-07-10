# Runtime Optimization Completion Audit

This directory records the completion audit run on the replacement server at
source commit `991329ac3876b859ca7cf123207031dfb0ac724b`.

## Server

- Host: `100.64.229.185:8722`
- GPU: one NVIDIA GeForce RTX 4090 (24 GiB)
- CPU: 20 logical CPUs
- Server worktree was clean at the tested commit.

## Results

- Whole-flow inventory: 30/30 expected files across launcher, Stage-1,
  Stage-2, Rescale/fusion maps, Paean, and structured artifacts.
- Artifact inventory: all required evidence categories were present.
- Project-audit tests: 9/9 passed.
- Structured-artifact contract tests: 35/35 passed.
- Completion-tool compilation: passed.
- Test-isolation reproductions: both passed, and neither dirtied the server
  worktree.
- Full suite: 1,221 tests in 17.867 seconds; 18 failures, 1 error, 2 skips.

The full-suite run is intentionally recorded as red. All 19 remaining cases
are Stage-2 contract paths changed by concurrent integration commit `16b68e3`.
They include eight runtime guards for cached masks/static tensors, deferred
GPU scalar synchronization, causal-prefix rollout, and probe scheduling. No
Stage-2 RL algorithm file was changed during this audit because that area is
owned by the concurrent Stage-2 agent.

## Audit Fix

Commit `991329a` fixes two test-process side effects found by the initial
completion run:

1. The experiments-log smoke now runs its registration subprocess from a
   temporary working directory, so the default generated index cannot replace
   tracked `experiments/index.md`.
2. The truncated-policy test preserves an already-loaded real PyTorch module
   and only removes test stubs, preventing later Stage-1 and Stage-2 tests from
   failing during an unsupported in-process PyTorch re-import.

## Remaining Gates

1. Coordinate a Stage-2 handoff, reconcile the 19 contract failures, and rerun
   the full suite without changing research semantics.
2. Run the formal Stage-2 1-GPU versus N-GPU parity and throughput gate on a
   server with at least two visible GPUs before promoting any GPU default.

See `full_tests_summary.md` for the exact failing tests and `audit.md` for the
six-stage flow inventory.
