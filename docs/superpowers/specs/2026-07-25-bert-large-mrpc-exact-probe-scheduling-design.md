# BERT-Large MRPC Exact Probe Scheduling Design

## Objective

Reduce Stage-2 layerwise RL wall time on the current four-healthy-GPU server
without changing the sampled actions, K=5 evidence, trial seeds or values,
terminal metrics, rewards, candidate decisions, PPO inputs or updates,
checkpoint state, or scientific conclusions.

## Measured Bottleneck

The stopped `e1a1bba9` BERT-large MRPC run completed a durable checkpoint at
episode 10,920 and PPO update 91. Its ordinary episode wall time was about
4.7 seconds, of which the terminal probe consumed about 4.0 seconds. Policy,
cost evaluation, install, candidate indexing, and PPO amortization were not the
dominant costs.

The active reward pool has four healthy workers and each action requires five
complete trials. The current per-action round-robin split is therefore
`[2, 1, 1, 1]`; three GPUs become idle while one GPU runs the fifth trial.
For four consecutive actions the critical path is eight trial durations even
though the 20 independent action/trial tasks need only five trial durations
when balanced across four workers.

## Decision

Use the existing `terminal_eval_batch_size` runtime setting as an exact
layerwise scheduling control:

- `1` keeps the current per-episode path.
- A value greater than one permits deferred online terminal evaluation only
  when the exact scheduler is eligible.
- With K=5, four workers, and the existing default value `4`, collect four
  actions under the unchanged policy, flatten their 20 complete trials in
  action-major order, and round-robin those tasks across the four workers.
- Reassemble every action's five results in original trial-index order before
  computing metrics or reward.
- If K is already divisible by the worker count, use the existing path. Thus a
  repaired five-worker K=5 pool automatically avoids unnecessary deferral.

The scheduler batches independent complete trials. It does not concatenate
probe examples, change model batch size, reduce K, cache a stochastic reward,
or reshape noise tensors.

## Exactness Invariants

For absolute episode `e` and trial index `t`, both paths must use:

```text
action(e)          = unchanged policy sample
base_seed(e)       = derive_layerwise_episode_probe_seed(run_seed, e, K)
trial_seed(e, t)   = derive_probe_trial_seed(base_seed(e), t)
trial_indices(e)   = [0, 1, 2, 3, 4]
```

The following order remains authoritative:

1. Sample actions and append rollout transitions in episode/step order.
2. Return each action's trial values in trial-index order.
3. Finalize reward and add terminal credit in episode order.
4. Append candidate evidence and run promotion in episode order.
5. Fire episode callbacks in episode order.
6. Run PPO only after the same 120 completed episodes.
7. Save checkpoints only after the same PPO boundary.

Batch collection never crosses a PPO boundary, the bounded run end, or a
convergence boundary. Layerwise convergence is observed only at PPO updates, so
this preserves the point at which sampling can stop.

## Probe Worker Protocol

Add one process command that receives an ordered list of action/trial tasks.
Each task carries:

- action index within the terminal batch
- original trial index
- original per-action base seed
- decoded action
- probe batch-set key

Worker task lists retain action-major order. A worker installs an action once
before its consecutive trials for that action, then runs each trial through the
existing `ProbeWorker.run_trial()`. The parent stores results by
`(action_index, trial_index)` and reconstructs each K-result list in
trial-index order.

Per-action diagnostics report actual worker assignments and deterministic trial
seeds. Timing fields may change because timing is the intended effect; action,
trial, metric, reward, candidate, and PPO fields may not.

## Layerwise Deferral

`BLBStage2LayerwiseEnv` gains an internal deferred-terminal mode. At the final
layer it performs the unchanged action materialization and resource-objective
calculation, but calls `prepare_action_for_terminal_probe()` instead of the
synchronous base `step()`. The training loop captures that prepared payload
before resetting the environment for the next action.

After a terminal batch is evaluated, the training loop uses the returned
base-env result as the episode's runtime terminal information and executes the
existing reward redistribution, candidate persistence, promotion, record, and
PPO logic in original episode order.

The old synchronous path remains unchanged and is the fallback whenever the
environment or probe runner does not advertise the exact batched API.

## Eligibility And Fallback

Exact batching is enabled only when all conditions hold:

- `terminal_eval_batch_size > 1`
- a multi-worker `ProbeRunner` is active
- online K is positive and not divisible by the worker count
- persistent probe installation is enabled
- the layerwise environment supports deferred terminal preparation
- the runner supports exact multi-action K-trial evaluation

The effective batch size is capped by the requested size, the minimum action
count needed to balance `K * actions` across workers, the remaining episodes in
the PPO window, and the remaining bounded episode budget. Otherwise the
effective size is one and execution follows the old path.

## Failure Semantics

No scientific fallback may silently reuse, drop, or regenerate evidence.
Missing tasks, duplicate task identities, missing per-action seeds, or worker
errors fail the exact batch. Existing invalid-action short-circuiting remains
per action and does not launch a model forward.

## Verification Gate

The change is promoted only if all of the following pass on the server:

1. Focused scheduling, process-protocol, environment, and layerwise-order tests.
2. Existing Stage-2 contract tests with no new failures.
3. Profile-off A/B from the same source, model, dataset, seed, K=5, four reward
   devices, PPO interval, and episode budget; only
   `terminal_eval_batch_size=1` versus `4` differs.
4. Exact equality (`atol=0`) for episode actions, trial seeds and values,
   metrics, rewards, candidate/promotion fields, and PPO update records.
5. At least `1.2x` end-to-end episodes/hour improvement over at least two PPO
   windows, plus activity on all four healthy GPUs.

If one semantic field differs or throughput is below `1.2x`, keep the runtime
setting at one and revert the exact scheduler from the production aggregate.

## Source And Deployment Protocol

All source edits are made in the isolated local worktree. Each completed change
is committed and pushed before the server fetches/checks out that commit.
Before final deployment, refresh every remote agent head, integrate all
completed non-superseded work, and verify identical local, remote, and server
commit and source-tree IDs. Server run artifacts remain untracked and intact.
