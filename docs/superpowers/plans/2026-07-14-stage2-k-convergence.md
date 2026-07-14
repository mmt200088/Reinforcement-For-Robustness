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
   both action-family entropies fall below `0.1` at the known optimum.
5. Package the exact commit to the server and run the project-integrated short
   smoke; inspect reward, per-family entropy, P1/P2/P3, invalids, and K action
   movement before approving another 60k launch.
