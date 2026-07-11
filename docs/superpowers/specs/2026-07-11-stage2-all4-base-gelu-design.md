# Stage-2 Configurable All-GELU4 Base Design

## Goal

Use GELU degree 4 in every layer as the current Stage-2 prerequisite Stage-1
configuration, while preserving an explicit, supported path back to a searched
Stage-1 configuration later.

The resolved Stage-1 configuration must be shared by Stage-2 training, fusion
action mapping, model evaluation, and the corresponding final evaluation. A
consumer must not independently fall back to an older Stage-1 result.

## Configuration Contract

Extend `--stage2-fixed-config-source` with a new `all4` source:

```text
all4 | stage1_result | json | manual
```

`all4` becomes the current default for Stage-2. It resolves to:

```text
GELU    = [4] * num_layers
Softmax = [6] * num_layers
```

The setting is reversible. Passing `stage1_result` restores the previous
searched-Stage-1 behavior. `json` and `manual` remain available for explicit
reproduction and controlled experiments.

The launcher rejects path or manual-array arguments that do not belong to the
selected source, so there is one unambiguous source of truth per run.

## Runtime Data Flow

The Stage-2 prerequisite resolver produces one immutable GELU/Softmax pair and
a source label. That pair is passed through the existing Stage-2 runner into:

1. plaintext function installation;
2. baseline bootstrap;
3. the layer/block action schedule;
4. fusion-count action-to-configuration mapping;
5. noisy model inference and reward calculation;
6. best-action final evaluation and reporting.

No downstream component re-resolves the prerequisite configuration.

The manifest and logs record the resolved arrays and a source label such as
`stage2_all4`, making the active base configuration auditable.

## Stage-2 Block Alignment

With `all4`, every layer's Block 5 schedule entry must select `block5_n4`.
Baseline bootstrap must independently confirm the same graph key, and the
selected profile must contain a loadable `block5_n4.json` map.

Block 2 and Block 4 keep their existing shared graph keys. Their current maps
already declare GELU degree 4 metadata, so this change does not require an
unrelated rebuild. Existing `block5_n1` and `block5_n2` maps remain committed
for historical reproduction and future `stage1_result` runs.

## Fixed Evaluation Tools

Stage-2 fixed-action and diagnostic tools that currently embed the old MRPC
Stage-1 best GELU vector switch their default to all 4. Tools that expose an
explicit GELU argument continue to accept it. This keeps the default experiment
aligned with Stage-2 RL without removing controlled override capability.

## Compatibility And Failure Behavior

- Existing explicit `stage1_result`, `json`, and `manual` runs retain their
  prior semantics.
- Historical result files are not rewritten.
- An unsupported source fails at launcher validation.
- `all4` fails before GPU work if an incompatible manual/path override is also
  provided, the resolved vector has the wrong length, or `block5_n4` is absent.
- Final evaluation fails rather than silently switching to a different GELU
  vector.

## Verification

Focused tests must cover:

1. launcher default resolution to `all4`;
2. explicit `stage1_result` restoring the old path;
3. resolver output `[4] * L` and `[6] * L`;
4. all Block 5 schedule entries resolving to `block5_n4`;
5. baseline bootstrap/map alignment for GELU4;
6. the same resolved arrays reaching Stage-2 training and final evaluation;
7. fixed-action script defaults changing from the old PPO vector to all 4.

After the source is committed and pushed, the exact snapshot is verified on the
GPU server with focused tests and a short Stage-2 startup gate. No local project
runtime is used.
