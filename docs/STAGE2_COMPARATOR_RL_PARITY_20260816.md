# Stage-2 Comparator and RL Parity Audit

## Scope

This audit covers the current `bo_rf`, `greedy`, and `coinn_ga` two-stage
comparators against the current layerwise Stage-2 PPO path. Algorithm-specific
proposal, update, and termination rules are intentionally excluded from parity.

## Shared scientific contract

| Contract | PPO | BO-RF / Greedy / COINN-GA |
| --- | --- | --- |
| Model and data | BERT-base MRPC, 12 layers | Same |
| Stage-1 binding | Selected Stage-1 result | Each backend binds its own selected Stage-1 result |
| Stage-2 action | Layerwise Block-4 fusion count plus precision preset | Same six-valued action |
| Materialization | Shared Rescale optimizer and model-forward path | Same |
| Online evidence | Stratified 256-example F1 probe, K=3 | Same |
| Baseline | 5 groups x 3 trials | Same |
| Precision tolerance | 0.001 | Same |
| Stability multiplier | 2.0 | Same |
| Resource objective | Shared compute and communication axes, ratio 1.0 | Same |
| Strict evidence | Full 408-example validation set | Same |
| Strict banks | A/B/C, 15 trials each | Same |
| Strict gate | Joint six-point gate plus compute and communication counterfactual gates | Same |
| Materialized result | Shared action decode, fingerprint, candidate store, and certification | Same |

Comparator top-5 selection is the algorithm candidate-shortlist budget. It is
not a substitute for PPO rollout behavior and must not be changed to PPO's
default report-selection count.

## Corrected mismatch

The aligned Stage-1 reruns intentionally use batch size 16 because the
historical Stage-1 RL reference used batch 16. The two-stage comparator process
previously carried that global batch into Stage-2. This was not equivalent to
the authoritative Stage-2 RL run, whose exact launch used batch 64:

- source commit: `8c2a526dbf793c95c388b5f8544a793e83c733dc`
- artifact: `rl_training_data_points/stage2/archives/bert-base-mrpc-60k-20260805/small_files/run/launch_evidence/launch_command.sh`

MRPC loss is aggregated per batch in this project, so batch partitioning is a
scientific parameter rather than a throughput-only knob. The comparator now:

1. keeps Stage-1 loaders and deterministic caches at batch 16;
2. switches all evaluator loaders to batch 64 at the Stage-2 boundary;
3. clears Stage-1 deterministic evaluation caches at that boundary;
4. constructs both F1 and F4 loaders at batch 64 without changing sample order;
5. binds batch 64 into comparator invocation, candidate identity, manifests,
   result metadata, and resume validation;
6. rejects formal comparator command-line overrides of either stage batch.

The formal aliases also lock the remaining shared RL evidence contract:
layerwise robust fusion-count materialization, sequential mode, calibration 8,
online K=3, bootstrap 4096, confidence thresholds 0.50/0.80/0.95, protected-K1
off, the legacy reporting tolerance 1.2, and the effective stability multiplier
2.0. Python boundary validation repeats the scientifically relevant checks so a
preset or direct entrypoint cannot bypass the shell guard.

The PPO identity format remains unchanged for backward-compatible checkpoint
resume. The new batch identity field is comparator-only because the comparator
introduces a stage-specific batch distinct from its global Stage-1 batch.
The Python boundary rejects a distinct Stage-2-only batch for PPO, so callers
cannot change PPO numerical partitioning without a corresponding historical
identity contract.

## Algorithm-only differences

- BO-RF uses categorical random-forest acquisition and its own stopping rule.
- Greedy exhaustively evaluates its 1-opt and 2-opt neighborhoods.
- COINN-GA uses population evolution, elitism, crossover, and mutation.
- Comparator online seeds are action-keyed to make evidence independent of
  proposal order; PPO online seeds are episode-keyed. The fixed strict banks
  remain identical and authoritative for both paths.

No reward, action, trial order, strict rule, cost model, model materialization,
or dataset membership is changed by the batch-boundary correction.
