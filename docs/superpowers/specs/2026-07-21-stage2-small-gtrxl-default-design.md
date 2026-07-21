# Stage-2 Small GTrXL Default Design

## Context

The Stage-2 layerwise policy makes 12 sequential decisions. Each layer exposes
one binary Block4 fusion action and up to five six-level truncation-K actions,
for 12 binary factors and 59 active K factors overall. The factorized action
space is `2^12 * 6^59`, but the policy learns shared conditional structure
rather than enumerating complete configurations.

The current shared actor-critic uses `d_model=256`, four GTrXL blocks, eight
attention heads, and `d_ff=512`, totaling 5,330,461 parameters. Two long runs
reached 114,240 and 231,960 episodes without satisfying strict convergence.
This does not prove that model size is the only cause, but the network is large
relative to the 136-dimensional state, 12-step horizon, and factorized heads.

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

## Non-Goals

- Do not delete or rewrite any large-network implementation or checkpoint.
- Do not change reward, PPO hyperparameters, action semantics, constraints,
  evidence tiers, or termination logic.
- Do not add an actor MLP or tiny GTrXL variant in this change.
- Do not claim convergence before a long controlled server experiment.
