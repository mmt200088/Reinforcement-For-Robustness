# Six-Profile Train-Probe Data Protocol Design

## Status

Approved for implementation on 2026-08-24.

This specification covers the functional code changes only. Moving historical
training data, results, logs, PDFs, and intermediate versions to an archive
branch is explicitly deferred until the functional implementation has passed
all tests and six real-GPU smoke runs.

The completed implementation will be integrated into the existing
`jk_standard_rl` canonical branch through the mandatory multi-agent Git
protocol. No replacement canonical or release branch will be introduced.

## Goals

1. Restrict the supported scientific matrix to six profiles:
   - BERT-base: MRPC, RTE, SST-2.
   - BERT-large: MRPC, RTE, SST-2.
2. Build one deterministic 256-example probe from each GLUE training split.
3. Reuse the exact same ordered examples for cleartext profiling, Stage-1,
   Stage-2, all search-time baselines, promotion gates, and strict top-5
   selection.
4. Reserve the complete GLUE validation split for the final plaintext
   evaluation after search has terminated.
5. Persist enough sample identity and protocol metadata to prove that all
   search components used the same data.
6. Reject old checkpoints and result artifacts that used validation-derived
   search probes.
7. Remove executable support and dedicated configuration for every dataset and
   model family outside the six supported profiles.

## Non-Goals

- The external 512-example encrypted-inference validation set is not built or
  consumed by this repository.
- Historical data and result cleanup is not part of this phase.
- The README rewrite, final-result curation, and repository publication cleanup
  are not part of this phase.
- This change does not alter actions, rewards, cost models, fusion maps,
  precision presets, trial counts, or final metric definitions.
- This change does not make old and new scientific results comparable.

## Supported Matrix

The runtime registry contains only the following combinations:

| Model | Dataset | Input fields | Task type |
| --- | --- | --- | --- |
| BERT-base | MRPC | `sentence1`, `sentence2` | binary classification |
| BERT-base | RTE | `sentence1`, `sentence2` | binary classification |
| BERT-base | SST-2 | `sentence` | binary classification |
| BERT-large | MRPC | `sentence1`, `sentence2` | binary classification |
| BERT-large | RTE | `sentence1`, `sentence2` | binary classification |
| BERT-large | SST-2 | `sentence` | binary classification |

Unsupported dataset or model values fail in the launcher before model or
dataset loading. CoLA, QNLI, WNLI, STS-B, MNLI, GPT-2, WikiText, regression
metrics, MNLI matched/mismatched handling, and their dedicated presets or
registries are removed from the active runtime surface.

Generic infrastructure is retained only when one of the six profiles still
calls it. Historical output directories are left untouched until the later
cleanup phase.

## Shared Data Protocol

### Constants

The shared protocol defines:

- schema: `glue_train_probe_protocol_v1`;
- source split: `train`;
- registered search split: `train_probe`;
- probe size: 256;
- selection seed: 42;
- final evaluation split: `validation_full`;
- supported datasets: MRPC, RTE, SST-2;
- supported model families: BERT-base, BERT-large.

The probe size and selection seed are scientific constants for formal runs,
not launcher tuning parameters.

### Selection Algorithm

For each supported dataset:

1. Enumerate the raw GLUE training split in its canonical physical order.
2. Extract the binary label for every row.
3. Apply the existing stratified selection behavior with
   `train_test_split(indices, train_size=256, random_state=42,
   shuffle=True, stratify=labels)`.
4. Sort the selected physical positions in ascending order.
5. Select those rows without replacement and preserve that sorted order in all
   consumers.

All three supported training splits must contain at least 256 examples and both
labels. Missing labels, duplicate stable IDs, insufficient rows, or failed
stratification are protocol errors. Formal runs do not silently switch to an
unstratified sample.

The shared selector returns both the dataset view and an identity record:

- physical row positions;
- raw `idx` values when the dataset provides them;
- label histogram;
- ordered identity SHA-256;
- source dataset revision;
- selection seed and size.

The selected physical positions depend only on dataset and protocol constants,
not on model size. BERT-base and BERT-large therefore consume identical raw
examples in identical order.

## Runtime Data Flow

```text
GLUE train split
  -> deterministic stratified selector
  -> train_probe (256 ordered examples)
       -> cleartext Profile
       -> Stage-1 baseline and constraints
       -> Stage-1 PPO / GA / Greedy / BO-RF candidate evaluation
       -> Stage-2 baseline calibration and online reward probe
       -> Stage-2 promotion banks and strict top-5

GLUE validation split
  -> validation_full
       -> final plaintext evaluation after search termination only
```

No code path in this repository creates or consumes the external 512-example
encrypted-inference validation set.

## Stage-1 Contract

- `train_probe` is the only search-time metric source.
- The clean baseline and all Stage-1 constraint thresholds are calculated on
  `train_probe`.
- PPO online rewards and every non-RL comparator candidate are evaluated on
  `train_probe`.
- Stage-1 manifests and structured data identify the split as `train_probe`,
  never `validation_full`.
- The selected Stage-1 configuration is not re-ranked using validation data.
- The complete validation split is used only by the post-search final
  plaintext evaluation.

## Stage-2 Contract

- The online reward probe uses the already materialized `train_probe`; it does
  not sample again.
- Baseline calibration groups use the same probe examples and ordering.
- Promotion Bank A/B/C and strict top-5 use the same examples; independent
  trials vary only noise seeds.
- Search evidence tiers and serialized field names state `train_probe` as the
  source. Names such as `F4_validation_full` are removed or versioned because
  they no longer describe the data.
- The selected Stage-2 configuration is not re-ranked using validation data.
- Paean or unified final evaluation runs on `validation_full` only after the
  search result has been fixed.

## Profile Contract

Both cleartext distribution profilers call the same shared selector before
tokenization. Formal BERT profiling always consumes the 256-example
`train_probe`; `max_samples` cannot silently produce a different formal sample.

The base and large variants for the same dataset persist the same identity
hash. Profile output records the protocol schema, dataset revision, selected
positions, and ordered identity hash.

## Dataset Loading and Identity

The dataset loader continues to load the full `train` and `validation` splits:

- full `train` remains available to the existing model-training surface;
- `train_probe` is derived once and registered with the evaluator;
- `validation_full` keeps the current tokenization, ordering, batch size, and
  loss aggregation semantics for final evaluation.

The current MRPC validation fixture no longer supplies a search stability
probe. Its validation-order role is either retained behind the generic
protocol or migrated to a generic validation identity record. RTE and SST-2
use the same generic interfaces; no dataset receives a private search path.

## Persistence and Resume Safety

Every formal run writes `dataset_protocol.json` before the first search
evaluation. It contains:

- protocol schema and supported profile;
- dataset repository and revision;
- source and final split names;
- probe size and seed;
- ordered positions and available raw IDs;
- label histogram;
- ordered identity hash.

The protocol payload or its stable hash is included in:

- run manifest;
- Stage-1 and Stage-2 invocation contracts;
- candidate identity context;
- checkpoint metadata;
- resume validation;
- final two-stage result provenance.

Existing checkpoints without `glue_train_probe_protocol_v1`, or with a
different identity hash, fail closed with a migration error. They are not
replayed or resumed under the new protocol.

## Unsupported-Code Removal

Implementation removes unsupported entries and branches from at least these
active surfaces:

- launcher dataset/model parsing and compatibility matrix;
- GLUE dataset loading and tokenization dispatch;
- metric dispatch;
- cleartext Profile task registries;
- Stage-1/Stage-2 evaluator task metadata;
- Paean task/model registries and presets;
- general-policy task validation;
- final-evaluation task dispatch;
- dataset-specific presets, fixtures, and tests whose only target is an
  unsupported profile.

Shared primitives remain when required by MRPC, RTE, or SST-2. Bulk deletion
of historical result directories remains deferred.

## Error Handling

The implementation fails before expensive work when:

- dataset or model family is unsupported;
- a requested model/dataset combination is outside the six-profile matrix;
- the training split has fewer than 256 labeled examples;
- stratified selection cannot produce exactly 256 unique positions;
- Profile, Stage-1, and Stage-2 identity hashes disagree;
- search code attempts to use `validation_full`;
- final evaluation cannot access `validation_full`;
- a checkpoint has no protocol hash or a mismatched hash.

## Verification

### Unit and Integration Tests

Tests prove:

1. MRPC, RTE, and SST-2 each produce 256 unique, ordered, deterministic
   training positions with the expected stratified label histogram.
2. BERT-base and BERT-large produce identical probe identities per dataset.
3. Profile, Stage-1, and Stage-2 receive the same identity hash.
4. Stage-1 baseline, online reward, PPO, GA, Greedy, and BO-RF never access
   validation data during search.
5. Stage-2 calibration, online probe, promotion banks, and strict top-5 never
   access validation data during search.
6. Final plaintext evaluation requires and consumes all validation examples.
7. Old checkpoint and artifact schemas are rejected.
8. Unsupported datasets and GPT-2 fail at the public entrypoint.
9. Active runtime registries and formal presets contain only the six supported
   profiles.
10. Action materialization, fusion/K mapping, rewards, trial ordering, and
    final metric calculations remain unchanged for the supported profiles.

### Server Gates

Server verification runs:

- all affected unit tests;
- the complete project test suite after unsupported-code removal;
- Python compilation and Bash syntax checks;
- dataset identity generation and cross-model hash comparison;
- one minimal cleartext Profile run for each of the six profiles;
- one minimal Stage-1 real-model run for each profile;
- one minimal Stage-2 real-model run for each profile;
- final validation-full evaluation canaries for each profile.

The smoke runs must prove real model forward execution, exact sample counts,
correct split provenance, unchanged action materialization, and no access to
unsupported datasets.

## Scientific Consequences

This is an intentional research-protocol change. Results produced with the old
validation-derived search probe remain historical evidence and cannot be
combined with or compared directly to new train-probe results without an
explicit protocol distinction. All new reports must state the protocol schema
and identity hash.

## Deferred Cleanup Phase

After this implementation is complete and verified, a separate approved task
will:

- create the dedicated archive branch for all removed data and results;
- curate the two authoritative six-model HTML reports and final RL configs;
- remove historical outputs, old versions, PDFs, and personal metadata from
  the current canonical tree;
- replace the README with a concise English Stage-1/Stage-2 run guide.
