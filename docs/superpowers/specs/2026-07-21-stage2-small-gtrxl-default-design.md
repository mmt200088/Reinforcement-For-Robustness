# Stage-2 Small GTrXL Default Design

> **Superseded action-domain snapshot:** The six-level K action space,
> `6^59`, and parameter count below describe the 2026-07-21 design point. The
> current K domain is `K=6..13` with ordered
> `K_LEVELS=(8,9,11,13,10,12,6,7)`, so the current factorized K space is
> `8^60`. The historical parameter count is retained for provenance and must
> not be treated as a current count without measuring the current policy.

## Context

The Stage-2 layerwise policy makes 12 sequential decisions. Each layer exposes
one binary Block4 fusion action and up to five six-level truncation-K actions,
for 12 binary factors and 59 active K factors overall. The factorized action
space is `2^12 * 6^59`, but the policy learns shared conditional structure
rather than enumerating complete configurations.

The current shared actor-critic uses `d_model=256`, four GTrXL blocks, eight
attention heads, and `d_ff=512`, totaling 5,330,461 parameters. Two long runs
reached 114,240 and 231,960 episodes without satisfying strict convergence.
Those results do not establish any causal link between network size and the
lack of convergence. The reason for reducing the network is narrower: if this
task does not need 5.33M policy/value parameters, their additional forward,
backward, optimizer, checkpoint, and synchronization cost is avoidable training
time. Network downsizing is therefore a runtime-efficiency hypothesis, not a
convergence fix.

## Decision

Add `shared_gtrxl_small_v1` with:

- `d_model=128`
- `n_heads=4`
- `n_layers=2`
- `d_ff=256`
- unchanged embeddings, actor/value head widths, dropout, PPO, reward,
  constraints, action masks, candidate promotion, and convergence rules

This architecture has 680,221 parameters under the production layerwise
configuration, an 87.2% reduction from the existing shared GTrXL network. It
becomes the default for fresh Stage-2 runs.

## Compatibility And Rollback

The existing `shared_gtrxl_v1` remains byte-for-byte architecture-compatible
with historical checkpoints and remains selectable explicitly. The existing
`separate_critic_gtrxl_v1` and `separate_critic_mlp_v1` variants are unchanged.

The pre-change selectable large-network implementation is frozen at Git tag
`stage2-rl-v10-large-network-ablation-20260721`. The earlier original shared
baseline remains frozen at `stage2-rl-v10-shared-baseline-20260721`.

Checkpoint metadata must identify the network variant. Legacy checkpoints
without an explicit network field are inferred as `shared_gtrxl_v1` only. A
small-network run must fail before state mutation if asked to resume any large
network checkpoint, and vice versa.

## Contract

`shared_gtrxl_v1` keeps the legacy algorithm contract unchanged. The new small
variant receives a distinct `rl_variant` and an explicit architecture payload
in the algorithm contract so run-context hashes, checkpoint contracts, and
candidate evidence cannot be mixed with the large network.

All manifests, live status, checkpoints, structured training data, and final
summaries continue to record the selected variant and parameter partition.

## Verification

Tests must prove:

1. Fresh configuration defaults to `shared_gtrxl_small_v1`.
2. The small production network has exactly 680,221 parameters and the agreed
   `128/4/2/256` architecture.
3. Explicit `shared_gtrxl_v1` still has exactly 5,330,461 parameters.
4. Historical actor initialization and logits remain unchanged when the large
   variant is selected explicitly.
5. Old checkpoints are accepted only by the large variant.
6. Cross-size checkpoint resume is rejected before loading policy, optimizer,
   RNG, or structured-writer state.
7. Every retained variant completes a real factorized PPO update.
8. Launcher defaults, explicit overrides, metadata, and resume guards agree.

## Experimental Expectation

The expected useful convergence window for the small network is 100,000 to
150,000 episodes. A controlled run may continue to 200,000 episodes, but model
size alone is not claimed to guarantee convergence. Failure to stabilize by
that point should trigger analysis of reward identifiability, equivalent K
actions, stochastic terminal evidence, and actor-credit quality rather than an
unbounded increase in episode count.

## Network-Size Decision Rule

Run the new small shared network first and compare it with the retained large
shared-network evidence under the same evaluation contract. Network size must
be selected from result quality and wall-clock cost, not from an assumption
that the large model caused non-convergence.

1. If the small network is not materially worse on strict-feasible success,
   robust objective quality, convergence evidence, or final evaluation, use
   the small architecture as the base for later network/reward ablations. Do
   not repeat the ablation matrix with the large architecture.
2. If the small network is materially worse, test one intermediate-size GTrXL
   architecture under the same contract. Its exact dimensions require a
   separate controlled design; they are not implicitly selected here.
3. If the intermediate network is acceptable, use it as the base for later
   ablations. Keep the large network and its tags only for reproducibility and
   rollback rather than as the intended production choice.

"Materially worse" must be judged using matched seeds and evaluation budgets;
a single run or a small reward difference is not enough to trigger a larger
network.

## Follow-Up Experiment Backlog

These are retained candidate improvements, not mandatory changes to the
current implementation. Each experiment must change one factor at a time.

1. Add diagnostics before changing reward: critic explained variance,
   held-out value error, actor/critic gradient cosine, per-action-head advantage
   SNR, KL/clip/entropy, and P1/P2/P3 plus reward-component distributions.
2. After choosing the small or intermediate base size, compare its shared
   actor-critic against the same-size actor with an independent critic. Test a
   wider network only if the independent critic still provides evidence of
   underfitting.
3. If reward still plateaus, keep v10 as the control and separately test either
   a conservative-confidence-bound dense constraint signal or primal-dual
   constrained PPO. Do not combine these changes in the first experiment.
4. Screen with `3 seeds x 30k episodes`; do not draw final conclusions from the
   screening stage. Promote candidates to at least five seeds and train to the
   predefined convergence condition or a 150k maximum budget.
5. For publication-level comparison at matched evaluation budget, compare PPO
   with random search, greedy/local search, or CEM. Report strict-feasible
   success rate, robust floor, sample efficiency, wall time, IQM, and 95%
   bootstrap confidence intervals.
6. Validate finalists on the real system rather than only fusion-count/K
   surrogates: measure compute latency, communication bytes, end-to-end time
   under multiple network conditions, and the resulting Pareto frontier.

The earlier 200k cap applies only to the first one-off small-network assessment.
The formal multi-seed ablation protocol uses the 150k maximum above so its
comparison cost remains controlled.

## Non-Goals

- Do not delete or rewrite any large-network implementation or checkpoint.
- Do not change reward, PPO hyperparameters, action semantics, constraints,
  evidence tiers, or termination logic.
- Do not add an actor MLP or tiny GTrXL variant in this change.
- Do not claim convergence before a long controlled server experiment.
