# Stage-2 Truncation-K 6-13 Design

## Goal

Extend every active Stage-2 truncation-K action from the current supported
values `8..13` to `6..13`, while preserving the categorical index meaning of
all existing actions and keeping training, reward, final evaluation, reports,
and model installation on one shared K-domain definition.

## Accepted Compatibility Contract

The canonical K table becomes:

```text
(8, 9, 11, 13, 10, 12, 6, 7)
```

Existing categorical indices remain unchanged:

| Index | K before | K after |
|---:|---:|---:|
| 0 | 8 | 8 |
| 1 | 9 | 9 |
| 2 | 11 | 11 |
| 3 | 13 | 13 |
| 4 | 10 | 10 |
| 5 | 12 | 12 |
| 6 | unavailable | 6 |
| 7 | unavailable | 7 |

This preserves the meaning of historical action vectors and fusion-map K
indices. The baseline remains `K=13` at index `3`.

The policy output head changes from six K categories to eight. A six-category
PPO checkpoint therefore cannot be resumed into the new policy. The existing
algorithm-contract validation must reject such a resume, and new eight-level
training must start fresh. No silent tensor padding or optimizer-state migration
is permitted because it would mix two different scientific action spaces.

## Shared Module Boundary

Create a torch-free `blb_stage2_rl.truncation_levels` module as the only owner
of:

- the legacy-compatible default K ordering;
- the supported value set `{6, 7, 8, 9, 10, 11, 12, 13}`;
- `K_MIN_BITS=6` and `K_MAX_BITS=13`;
- environment-variable parsing and duplicate validation;
- exact-domain validation for the canonical layerwise RL path;
- baseline-index lookup by decoded value.

`action_space`, `layerwise_action`, `reward`, and `fusion_cost` must import these
definitions instead of maintaining independent constants. The module remains
torch-free so layerwise action tests and cost/reward tests continue to run on a
local machine without PyTorch.

The existing `BLB_TRUNCATION_K_LEVELS` override remains available. Legacy
action-space utilities may consume a custom unique table, while the canonical
layerwise RL path continues to require exactly the supported `6..13` values in
any order. This keeps experimental override behavior explicit without allowing
a malformed production policy domain.

## Policy Shape And Initialization

Each Transformer layer still has six policy slots:

```text
Block4 fusion, Block1 K, Block2 K, Block3 K, Block4 K, Block5 K
```

Only the category count of the five K slots changes:

```text
slot_dims = (2, 8, 8, 8, 8, 8)
max_step_dim = 6
max_num_levels = 8
```

The old K prior distribution is scaled by `0.95`, preserving its internal
ratios, and the new values receive low but nonzero probability:

| K | Initial probability |
|---:|---:|
| 13 | 0.475 |
| 12 | 0.190 |
| 11 | 0.114 |
| 10 | 0.076 |
| 9 | 0.057 |
| 8 | 0.038 |
| 7 | 0.030 |
| 6 | 0.020 |

The probabilities sum to `1.0`. They are specified by decoded K value, then
mapped through the canonical non-monotonic table, so index ordering cannot
silently alter the intended prior.

## Cost And Reward Semantics

`K=13` remains the zero-saving baseline. The maximum removable truncation bits
per active K slot changes from five to seven:

```text
removed_bits = 13 - K
max_removed_bits_per_slot = 13 - 6 = 7
```

Layerwise communication-cost normalization, robust reward normalization, and
fusion-cost truncation normalization must all use the shared `K_MIN_BITS=6` and
`K_MAX_BITS=13`. This ensures:

- `K=13` produces zero communication saving;
- `K=6` produces normalized communication saving `1.0`;
- intermediate K values are linear in removed bits;
- slot-level Shapley credits still sum to the episode resource score.

No Gaussian-noise, fusion-count, replan, or truncation backend semantics change.
The selected K still reaches the same shared action materialization, bridge
installation, and configured truncation executor used by online RL and final
evaluation.

## Artifact Compatibility

Fusion maps do not require rebuilding because their existing action indices
`0..5` retain identical decoded K values. New fixed-action artifacts may use
indices `6` and `7`.

Saved algorithm contracts and reports must record the eight-value `k_levels`
table and policy `max_num_levels=8`. Reports must continue to display decoded K
values, not infer K from a sorted index.

Historical artifacts remain immutable and continue to describe the six-level
action space used by their original runs.

## Verification

Implementation follows test-driven development.

Torch-free tests must first fail against the six-level implementation and then
cover:

1. the exact canonical table and preservation of indices `0..5`;
2. decoding indices `6` and `7` to K6 and K7;
3. exact-domain validation for values `6..13`;
4. layerwise slot dimensions `(2, 8, 8, 8, 8, 8)`;
5. policy configuration `max_num_levels=8`;
6. initial probabilities, including positive K6/K7 mass and sum `1.0`;
7. `K=6` communication saving and seven-bit normalization;
8. fusion-cost and reward K bounds `6..13`;
9. neighbor generation reaching both new categories;
10. unchanged baseline index `3` and unchanged decoding of historical action
    vectors;
11. algorithm-contract metadata containing the new table and policy width;
12. rejection of incompatible old checkpoint metadata rather than silent
    migration.

After focused tests pass, run the broader torch-free Stage-2 action, policy,
reward, materialization, report, and static single-path suites. Server
Torch/CUDA tests are required before claiming that K6/K7 execute correctly in a
real model forward.
