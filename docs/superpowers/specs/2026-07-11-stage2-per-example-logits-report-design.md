# Stage-2 Per-Example Correctness And Logits Report Design

## Goal

Extend the completed BERT-base MRPC three-group fixed-fusion stability
experiment with per-example predictions. The enriched artifacts must identify
which validation examples are correct or incorrect and retain the exact final
classification logits for every one of the existing 75 noisy inference trials.

The experiment protocol remains unchanged:

- Stage-1 GELU: `[4] * 12`;
- Stage-1 Softmax: `[6] * 12`;
- truncation K: `13` at all 47 Stage-2 decisions;
- groups: fusion totals `0`, `24`, and `36` for B2/B4/B5 patterns
  `(0,0,0)`, `(1,0,1)`, and `(1,1,1)`;
- run seeds: `20260721`, `20261721`, `20262721`, `20263721`, and
  `20264721`;
- five deterministic noise trials per group and run seed;
- MRPC `validation_full`, 408 examples per trial;
- total detailed prediction rows: `5 * 3 * 5 * 408 = 30,600`.

This is fixed-action evaluation only. It must not start Stage-2 RL.

## Chosen Capture Boundary

Use a read-only top-level model forward hook owned by
`scripts/run_fusion_count_action_eval_rlpath.py`. Enable it only for the
terminal deterministic probe forward of a fixed group and disable it
immediately afterwards.

This preserves the canonical execution chain:

`action JSON -> BLBStage2SequentialEnv.evaluate_step -> commit_step ->`
`BLBStage2Env.step(boosted_overrides) -> canonical replan -> bridge.apply ->`
`installed noisy model forward -> K=13 MPC truncation -> validation_full`

The hook observes the exact model keyword arguments and returned logits. It
must not replace, repeat, mutate, or short-circuit the model forward. The
Stage-2 environment and RL training hot path remain unchanged.

Rejected alternatives:

- adding a recorder to the core Stage-2 probe loop would increase conflict and
  semantic risk in an actively modified RL module;
- running a separate post-evaluation inference pass would use a different
  noise stream, so its logits would not explain the reported metrics.

## Example Identity

Record both meanings of input identity requested by the user:

1. `dataset_idx`: the original stable GLUE MRPC validation `idx`;
2. `input_ids`: the complete padded token-ID vector actually passed to the
   model.

The fixed evaluator builds a read-only lookup from tokenized validation rows
before formatting. Lookup keys include token IDs, token-type IDs, and the gold
label. Duplicate keys map to ordered lists of MRPC indices and are consumed
with a per-trial occurrence counter. Every trial must resolve exactly 408
unique dataset indices; ambiguity, missing IDs, or reuse is a hard failure.

## Detailed Row Schema

Each prediction row contains:

- `run_seed`, `group`, `trial_index`, and deterministic `trial_seed`;
- `probe_position` and `dataset_idx`;
- complete padded `input_ids`, `attention_mask`, and `token_type_ids` when
  present;
- `gold_label`, `predicted_label`, and `correct`;
- final two-class `logits` as finite FP32 numbers.

Rows are ordered by run seed, group, trial index, and probe position. The
recorder validates the number of captured model calls against
`trial_count * probe_batch_count`, then partitions calls without guessing.

## Structured Artifacts

Each run writes one compact JSONL file containing its 6,120 detailed rows.
The final aggregate writes:

- a manifest with row counts and hashes;
- per-input summaries for each group over five trials per run;
- per-input summaries for each group over all 25 trials;
- correct and incorrect `dataset_idx` lists for every run/group/trial;
- mean and population-standard-deviation logits, correctness count, and
  correctness rate for every dataset input.

Raw per-trial metrics and existing replan/model-install evidence remain in the
original result JSON files.

## Self-Contained HTML

Generate one self-contained HTML report with all 30,600 detailed rows embedded
as JSON. The page provides:

- protocol and gate status;
- group-level loss, Accuracy, and Weighted F1 results;
- per-input correctness summaries;
- correct/incorrect ID lists for each run/group/trial;
- a paginated detail table showing dataset ID, full token IDs, labels,
  correctness, and logits;
- filters for seed, group, trial, correctness, and dataset ID.

The initial DOM renders only one page of rows; filtering and pagination happen
client-side so the complete standalone report remains usable despite the
embedded dataset.

The final user-facing copy is:

`/Users/pengjunkai/Desktop/20260711_mrpc_three_group_per_example_logits.html`

## Validation Gates

The enriched run is accepted only when all of the following pass:

- five exact run seeds, three exact groups, five exact trials, and 408 exact
  examples per trial;
- 30,600 total detailed rows;
- all logits are finite and contain exactly two values;
- predictions equal `argmax(logits)` and `correct` equals prediction versus
  gold label;
- each trial contains every expected MRPC `dataset_idx` exactly once;
- groups in one run use identical trial-seed streams;
- every fixed-action result still has 47 valid K=13 steps, expected fusion
  totals, and `model_uses_replan_config=true`;
- Accuracy recomputed from detailed rows equals the evaluator's per-trial
  Accuracy;
- recomputed Weighted F1 and loss-derived aggregate checks match the existing
  evaluator result where applicable;
- every prior per-trial loss, Accuracy, and Weighted F1 value matches the
  already committed three-group experiment within `1e-9`;
- the output JSONL/hash manifest and copied desktop HTML pass integrity checks.

Any mismatch aborts aggregation. Historical values must not be substituted for
a failed rerun.

## Testing And Server Workflow

Use test-first development for:

- duplicate-safe MRPC ID lookup;
- forward-capture partitioning and row construction;
- prediction/correctness/logit validation;
- 30,600-row aggregate accounting;
- prior-result equivalence gates;
- HTML filters, pagination data, and visible protocol fields.

Source edits occur only in the local isolated worktree and are committed and
pushed before execution. All project tests and the 75 inference evaluations run
on the school GPU server from an exact Git snapshot. The server only produces
artifacts. Compact artifacts are pulled back, hash-verified, committed, and
pushed from the local worktree.
