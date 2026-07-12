# Runtime Optimization Completion Audit Refresh

This directory records the whole-project audit on replacement server
`100.64.229.185:8722` at source commit
`9ab82f04375689aeefdb89dac22cca5cdd0ceb7a`.

## Server

- GPU: one NVIDIA GeForce RTX 4090 (24 GiB).
- CPU: 20 logical CPUs.
- Server worktree was clean before and after the test process.

## Green Gates

- Whole-flow inventory: 30/30 expected files across launcher, Stage-1,
  Stage-2, Rescale/fusion maps, Paean, and structured artifacts.
- Artifact inventory: all required evidence categories were present.
- Project-audit tests: 9/9 passed.
- Structured-artifact contract tests: 35/35 passed.
- Completion-tool compilation passed.
- The corrected `ReplanSession` construction fixture passed independently.

## Full Suite

The full process ran 1,229 tests in 16.991 seconds and reported 18 failures,
one error, and two skips. The 19-item failure set is exactly equal to the
earlier `991329a` audit: zero failures were added and zero were removed. All
remaining cases are Stage-2 contract paths owned by the concurrent Stage-2
agent after integration commit `16b68e3`.

An initial audit at `9c2a3a4` exposed one additional non-Stage-2 fixture error:
the fake baseline in `test_replan_session_construction.py` omitted the required
`skeleton` field now read by session stage-path precomputation. Commit
`9ab82f0` aligned that fixture with `BaselineRecord`; the final full run proves
the extra error is gone.

The suite is intentionally recorded as red. This audit does not relabel the
19 Stage-2 failures as acceptable or edit the concurrently owned algorithm
files.

## Remaining Gates

1. Receive the Stage-2 agent handoff, reconcile the same 19 contract failures,
   and rerun the full suite.
2. Run the formal Stage-2 1-GPU versus N-GPU parity and throughput gate on a
   server exposing at least two GPUs. The current one-GPU server cannot close
   that gate.

`full_tests_summary.md` lists every remaining test. The compressed raw log,
six-stage audit, resource snapshot, focused gates, worktree status, and hashes
are retained here.
