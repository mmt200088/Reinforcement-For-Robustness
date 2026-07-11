# Stage-2 Three-Group Fusion Stability Experiment Design

## Goal

Extend the existing BERT-base MRPC fixed-fusion comparison with a third group
that enables fusion count 1 for Blocks 2, 4, and 5 in every layer. Rerun all
three groups under one controlled protocol instead of combining a new result
with historical control/treatment data.

The experiment measures both average validation quality and Gaussian-noise
stability. It must reuse the canonical Stage-2 RL-path evaluator so the fixed
actions reach the model through the same replan, precision-boost, noise-install,
and truncation path used by Stage-2 RL and final evaluation.

## Fixed Model And Evaluation Protocol

- model/profile: BERT-base MRPC;
- evaluation split: full MRPC validation set, 408 examples;
- Stage-1 GELU degrees: `[4] * 12`;
- Stage-1 Softmax degrees: `[6] * 12`;
- truncation K: `13` at every applicable decision;
- independent experiment runs: `5`;
- repeated inference trials per group and experiment: `5`;
- total full-validation evaluations: `5 experiments * 3 groups * 5 trials = 75`.

The five experiment seeds remain the established schedule:
`20260721`, `20261721`, `20262721`, `20263721`, and `20264721`. Within one
experiment, all groups use matching trial indices and deterministic probe-seed
derivation. This common-random-number pairing reduces variance when comparing
groups while the five experiment seeds preserve independent replication.

## Groups

All fusion choices apply to all 12 encoder layers. Block 1 remains at fusion
count 0.

| Group | Block 2 | Block 4 | Block 5 | Total fusion count |
|---|---:|---:|---:|---:|
| Control (`all_fusion0`) | 0 | 0 | 0 | 0 |
| Existing treatment (`block2_block5_all_layers_fusionmax`) | 1 | 0 | 1 | 24 |
| New treatment (`block2_block4_block5_all_layers_fusion1`) | 1 | 1 | 1 | 36 |

The control is the Stage-2 all-fusion-zero configuration with K=13. It is not
the original noise-free plaintext model.

Current canonical maps are expected to expose fusion counts `[0, 1]` for
Blocks 2, 4, and 5. Generation and evaluation must fail rather than silently
clamp or substitute an action if fusion count 1 is unavailable.

## Canonical Data Flow

Each action configuration must use the existing merged evaluator:

`action JSON -> BLBStage2SequentialEnv.evaluate_step -> commit_step ->`
`BLBStage2Env.step(boosted_overrides) -> canonical replan -> optimizer override`
`-> bridge.apply(noise configuration) -> K=13 MPC truncation -> validation_full`

Action generation may describe the three fixed fusion schedules, but it must
not implement model evaluation, replan, noise installation, or truncation.
`scripts/run_fusion_count_action_eval_rlpath.py` remains the sole experiment
evaluation entry point.

## Implementation Boundaries

1. Extend canonical action-config generation with the new all-Blocks-2/4/5
   fusion-one group.
2. Keep the existing control and Block2/Block5 treatment definitions unchanged.
3. Extend comparison aggregation/reporting to support all three groups and
   preserve numeric per-run and pooled statistics.
4. Run one evaluator invocation per independent experiment, passing all three
   action JSON files together with `repeat=5`.
5. Do not introduce a separate inference implementation or reconstruct actions
   inside reporting code.

## Validation Gates

Every experiment must satisfy all of the following before inclusion:

- exactly three unique action configurations are loaded;
- exactly five completed trials exist per group;
- all 47 Stage-2 decisions are valid;
- `invalid_block_count == 0`;
- install verification confirms the model uses the selected replan config;
- replan application confirms the model uses the canonical replan result;
- K equals 13 at every applicable decision;
- per-block fusion counts exactly match `(0,0,0)`, `(1,0,1)`, and `(1,1,1)`;
- total fusion counts equal 0, 24, and 36 respectively;
- Block 5 resolves through the MRPC GELU4 `block5_n4` map;
- all loss, accuracy, Weighted F1, and standard-deviation fields are finite.

Any gate failure aborts the affected run and prevents a successful aggregate
report. There is no fallback to historical results or server-handwritten action
files.

## Results And Comparisons

Preserve raw per-trial metrics and produce:

- per-experiment mean, standard deviation, extrema, trial seeds, and action
  verification for all three groups;
- pooled 25-trial mean and standard deviation for loss, accuracy, and Weighted
  F1;
- experiment-level variability across the five group means;
- paired deltas for existing treatment minus control;
- paired deltas for new treatment minus control;
- paired deltas for new treatment minus existing treatment, isolating the
  effect of enabling Block 4;
- concise JSON plus a readable HTML report and a chat summary.

The report must label the control as a noisy Stage-2 fusion-zero control and
show GELU, Softmax, K, per-block fusion decisions, total fusion count, repeat
count, and experiment seeds directly rather than relying on nested JSON.

## Server And Git Workflow

- Source and report-code changes are made only in the clean local worktree.
- Each runnable source snapshot is committed and pushed before server use.
- The school server checks out the exact pushed commit in an isolated runroot.
- Before launch, inspect GPU/process use and do not stop or alter other users'
  jobs.
- The server only runs tests and evaluations and creates artifacts; it does not
  edit source.
- Compact artifacts return to the local worktree, then are committed and
  pushed from local.

## Verification Strategy

Use focused test-first coverage for:

- generation of the new action and exact layer/block fusion schedule;
- rejection when a required fusion-one option is missing;
- three-group aggregation and all three paired comparisons;
- five-experiment and five-trial completeness gates;
- report rendering of means, standard deviations, K, and per-block fusion
  decisions.

Run focused tests on the server from the exact pushed commit. Then run the 75
full-validation evaluations, validate the result schema and gates, generate the
aggregate artifacts, and sync them through Git.

## Non-Goals

- No Stage-2 RL training.
- No reward, policy, fusion-map, replan, precision-boost, noise, or truncation
  semantic changes.
- No use of historical control/treatment metrics in the new aggregate.
- No evaluation on the training set or a validation proxy.
- No interference with unrelated server jobs.
