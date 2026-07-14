# Stage-2 K Convergence Implementation Plan

**Goal:** Make the 12-step Stage-2 PPO learn and converge its 59 truncation-K
decisions while preserving robust precision/stability semantics and enforcing
the requested `K - 2 == Block4 fusion + 1` cost equivalence.

**Base:** `origin/jk_standard_rl` at `5964eef`.

## Task 1: Lock The New Cost Algebra

**Files:**
- Modify: `blb_stage2_rl/layerwise_action.py`
- Modify: `tests/test_blb_layerwise_action.py`
- Modify: `tests/test_blb_fusion_reward.py`

1. Add failing tests for exact per-coordinate equivalence, all-59-slot
   accounting, layer-local terms, and `[0, 1]` normalization.
2. Implement raw fusion/K units and layer-local normalized cost terms.
3. Update old equal-family assertions without changing action decoding.
4. Run the focused action/reward tests.

## Task 2: Redistribute P3 Cost Credit

**Files:**
- Modify: `blb_stage2_rl/layerwise_env.py`
- Modify: `blb_stage2_rl/layerwise_runner.py`
- Modify: `tests/test_blb_layerwise_env.py`
- Modify: `tests/test_blb_layerwise_runner.py`

1. Add failing tests that P3 layer rewards sum to terminal `C`, P1/P2 get no
   cost, and the episode return is unchanged.
2. Persist raw units and per-layer terms in terminal handoff metadata.
3. Store zero during collection, then backfill terminal constraint reward plus
   P3-only layer-local critic costs after priority is known.
4. Give the factorized actor the shared terminal constraint return plus only
   its own slot cost, preventing sibling-cost credit noise.
5. Run focused environment/runner tests.

## Task 3: Add Factorized MultiDiscrete PPO Clipping

**Files:**
- Modify: `blb_stage2_rl/sequential_policy.py`
- Modify: `blb_stage2_rl/sequential_runner.py`
- Modify: `tests/test_blb_layerwise_policy.py`
- Modify: `tests/test_blb_sequential_policy.py`

1. Add failing tests for per-slot log probabilities, masked slots, sibling
   clipping independence, and legacy joint-PPO compatibility.
2. Add backward-compatible per-slot statistics from `evaluate_action`.
3. Add opt-in factorized actor clipping and normalized active-slot mean entropy
   objective; keep the scalar critic and legacy joint mode unchanged.
4. Enable factorized mode only for the layerwise robust branch.
5. Run focused policy tests.

## Task 4: Make The Layerwise Entropy Schedule Convergent

**Files:**
- Modify: `blb_stage2_rl/sequential_runner.py`
- Modify: `tests/test_blb_layerwise_runner.py`

1. Add failing tests for exact zero at the planned horizon and all extensions.
2. Set layerwise cosine end and lower bound to zero, end decay at 85% of the
   planned horizon, and preserve the safe initial prior/exploration plateau.
3. Persist the effective schedule values in the run manifest and PPO updates.
4. Run focused runner tests.

## Task 5: Verify Compatibility And Convergence

1. Run `py_compile` for all touched modules.
2. Run the focused layerwise, robust-reward, fusion-reward, persistence, and
   sequential PPO tests.
3. Run the broader torch-free Stage-2 test selection.
4. Run a deterministic synthetic factorized-bandit convergence test and assert
   both action-family entropies fall below `0.1` at Block4 fusion `1` and the
   decoded production optimum `K=8`.
5. Package the exact commit to the server and run the project-integrated short
   smoke; inspect reward, per-family entropy, P1/P2/P3, invalids, and K action
   movement before approving another 60k launch.

## Task 6: Version Resume And Candidate Cost Semantics

**Files:**
- Modify: `blb_stage2_rl/layerwise_action.py`
- Modify: `blb_stage2_rl/layerwise_runner.py`
- Modify: `blb_stage2_rl/sequential_runner.py`
- Modify: `tests/test_blb_layerwise_policy.py`
- Modify: `tests/test_blb_layerwise_runner.py`

1. Bind candidate identity to the exact `K_LEVELS` ordering and fusion/K
   cost-model revision.
2. Recompute restored candidate cost from the canonical 12x6 action matrix.
3. Store and validate an algorithm-contract hash before mutating the run
   manifest; reject all pre-v3 checkpoints.
4. Bind checkpoints to the full experiment context (model/profile, fixed
   Stage-1 configuration, maps/skeletons, baseline limits, trial counts, and
   probability gates).
5. Fingerprint the committed prefixes of the candidate store plus primary and
   mirrored episode/update JSONL files; validate them before loading policy
   state or rolling any file back.
6. Correct the synthetic convergence fixture to reward decoded K values rather
   than category indices, and verify it converges to real `K=8` on server
   Torch.

## Task 7: Persist Behavior Policy And Crash-Safe Run State

**Files:**
- Modify: `blb_stage2_rl/sequential_policy.py`
- Modify: `blb_stage2_rl/layerwise_runner.py`
- Modify: `blb_stage2_rl/sequential_runner.py`
- Modify: `tests/test_blb_layerwise_policy.py`
- Modify: `tests/test_blb_layerwise_runner.py`

1. Store factorized behavior log probabilities at action-sampling time and use
   those immutable values for PPO ratios after earlier optimizer updates.
2. Reject factorized updates that lack sampling-time per-slot evidence and
   verify its sum against the stored joint behavior probability.
3. Add a stable-parent single-writer lock acquired by the launcher before
   `--fresh` cleanup and held by Python across baseline probing and either
   Stage-2 branch; reject stale fresh-run artifacts and write an episode-zero
   checkpoint before collection.
4. Replace repeated full-prefix hashing with resumable incremental SHA-256
   tracking while preserving committed-prefix validation on resume.
5. Persist and restore the exact PPO update count instead of deriving it from
   episode count.
6. Bump the algorithm revision to `factorized_slot_credit_v4`, rerun the
   server Torch suite, and complete at least three PPO updates in the five-GPU
   integrated smoke.
