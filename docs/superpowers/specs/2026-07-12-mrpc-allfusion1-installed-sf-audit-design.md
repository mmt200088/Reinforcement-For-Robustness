# MRPC All-Fusion1 Installed SF Audit Design

## Goal

Audit the BERT-base MRPC fixed action with Block 2, Block 4, and Block 5
fusion-count set to `1` in all 12 layers. The audit must prove that noise is
installed through the canonical Stage-2 action path and report only scaling
factors from the final post-replan configuration actually passed to the model.

The prediction section uses stable MRPC validation row numbers `0..407`. GLUE
source `idx`, tokenizer `input_ids`, and shuffled probe positions are not
user-facing identifiers in this report.

This is fixed-action evaluation only. It must not start or update RL.

## Authoritative Runtime Boundary

The authoritative SF source is the argument received by
`BLBNoiseRLBridge.apply()` after the complete canonical path:

`fusion option -> precision-boost explicit SF -> SF-direct rebuild -> replan ->`
`optimizer write-back -> fused-rescale nulling -> binding synchronization ->`
`bridge.apply -> function_handler -> noisy model forward -> K truncation`.

The audit wraps the existing bridge instance after environment construction.
The wrapper calls the original `apply` method, then verifies that every active
handler configuration object is the same object supplied to `bridge.apply`.
Only after this identity check passes may a value be labelled as installed SF.

Map JSON `slots`, action-report bindings, and pre-replan explicit field values
are non-authoritative and must never populate the installed-SF table. A field
that replan fuses away is reported as `OFF / fused away`, not with its earlier
proposal SF.

## Evaluation Protocol

- model/profile: BERT-base MRPC;
- Stage-1 GELU degrees: `[4] * 12`;
- Stage-1 Softmax degrees: `[6] * 12`;
- action: B2/B4/B5 fusion-count `1` for every layer;
- truncation: `K=13` for every Stage-2 decision;
- data: complete MRPC validation split, 408 examples;
- audit inference: one canonical full-validation noisy trial;
- historical outcome source: the committed 25-trial-per-group prediction
  artifacts from `stage2_three_group_per_example_logits_20260712_171826`.

The single audit inference proves the live install chain. The existing 25
draws remain the source for per-row correctness frequency and logits because
rerunning them would create a different noise sample rather than improve the
installation proof.

## Stable Validation Row IDs

Load the original, unshuffled MRPC validation split and assign its enumeration
ordinal as `validation_row_id`. Build a one-to-one lookup from GLUE `idx` to
that ordinal. Every historical prediction row is translated through this
lookup and the final report exposes only `validation_row_id` values `0..407`.

Acceptance requires exactly 408 unique source rows, a bijective mapping, and
exactly 25 prediction outcomes for every validation row in every comparison
group.

## Captured Installed Configuration

For each of B2, B4, and the active B5 degree-specific configuration in every
layer, serialize:

- layer and block;
- configuration field and tuple index;
- noise-point type;
- final installed scaling factor, or null when no noise point is installed;
- truncation K when present;
- source marker `post_replan_bridge_apply`;
- handler object-identity verification result.

The serializer must retain tuple elements independently. It must not infer or
copy a missing value from a bound sibling field.

## Report

Produce one self-contained HTML file containing:

1. a pass/fail install-chain verdict;
2. the exact protocol and source commits;
3. a compact layer-by-layer B2/B4/B5 installed-SF table;
4. explicit fused-away entries;
5. a validation table keyed only by `0..407`, with correct count out of 25 for
   each of the three historical groups;
6. expandable per-trial labels, predictions, correctness, and final logits.

The visible page must not use the labels `dataset_idx`, `input_ids`, or
`probe_position`.

Final user-facing path:

`/Users/pengjunkai/Desktop/20260712_mrpc_allfusion1_actual_sf_audit.html`

## Hard Gates

The report is accepted only when:

- the bridge wrapper observes the terminal candidate installation;
- B2/B4/B5 active handler layers are exactly `0..11`;
- installed handler cfg objects are identical to the captured bridge args;
- all 36 selected block positions have fusion-count `1`, boost enabled, and
  `model_uses_replan_config=true`;
- all 47 Stage-2 steps use `K=13` and are valid;
- the full-validation audit returns finite metrics;
- every installed-SF row is marked `post_replan_bridge_apply`;
- the MRPC validation mapping is exactly `0..407` and bijective;
- every row/group has exactly 25 historical outcomes;
- predictions equal `argmax(logits)` and correctness matches the gold label;
- the HTML omits obsolete identifier labels and contains rows `0` and `407`.

## Workflow

All source changes are made and tested in the local isolated worktree, then
committed and pushed. The GPU server checks out the exact verified snapshot and
only runs tests/evaluation and produces artifacts. Compact results are copied
back, verified, committed, pushed, and copied to the Desktop.
