# Stage-2 Action Materialization and Optional Truncation Implementation Plan

**Goal:** Close every Stage-2 action-to-model gap and provide a tested,
default-off stochastic ring truncation backend while preserving current runtime
semantics by default.

**Architecture:** `blb_stage2_rl.optimizer_cost` owns canonical action
materialization and final-config identity. Model-evaluation callers consume the
materialized object and fail closed before inference. `function_handler.py`
retains legacy truncation and hosts an isolated optional stochastic-ring
backend selected by explicit Stage-2 configuration.

---

### Task 1: Lock chain behavior with failing tests

- [x] Add tests for valid-but-incomplete replan write-back failing closed.
- [x] Add tests proving boosted SF changes final-config identity.
- [x] Add tests proving every Block K reaches an installed runtime operation.
- [x] Add route-parity guards for RL, Paean, fixed-action, and GLUE callers.

### Task 2: Canonical materialization

- [x] Add a structured materialized-action result.
- [x] Centralize decode, boost reconstruction, optimizer replan, write-back,
      validation, and final-config fingerprinting.
- [x] Migrate online/prepared RL, sequential evaluation, and Paean to it.
- [x] Change persistent model-install caching to final-config identity.
- [x] Persist materialization diagnostics and fingerprint in evaluation output.

### Task 3: Optional stochastic-ring truncation

- [x] Add an isolated truncation RNG stream with deterministic seed derivation.
- [x] Implement signed ring encode/wrap, arithmetic shift, probabilistic
      rounding, and target-K decode.
- [x] Preserve legacy `binary` and `decimal` code paths exactly.
- [x] Add explicit default-`binary` Stage-2 configuration/CLI plumbing.
- [x] Persist backend parameters and include them in final-config identity.

### Task 4: Verification and delivery

- [x] Run focused static/torch-free checks locally where available.
- [x] Commit and push source from the clean worktree.
- [x] Run Torch-backed tests on the server from the exact pushed commit while
      keeping the GPUs isolated from the active training job.
- [x] Capture route parity, all-block K, fail-closed, legacy parity, stochastic
      statistics, and Gaussian-RNG-isolation evidence.
- [x] Pull compact evidence locally.
- [x] Commit/push the evidence and verify local/remote synchronization.
