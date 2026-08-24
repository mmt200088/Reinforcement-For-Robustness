# MPC Truncation Paper Semantics Design

## Goal

Expose the paper's H/M/L MPC precision choices as the public Stage-2 meaning
without changing any truncation value that reaches the cleartext simulation
model. After this change, an identical RL action must produce the same legacy
full vector, installed `output_truncation_k` values, model outputs, rewards,
and checkpoint execution behavior as before.

## Non-Negotiable Invariant

The executable cleartext truncation tuples remain byte-for-byte unchanged:

| Preset | Block 1 | Block 2 | Block 3 | Block 4 | Block 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| High | 11 | 10 | 10 | 12 | 11 |
| Medium | 9 | 8 | 8 | 10 | 9 |
| Low | 7 | 6 | 6 | 8 | 7 |

`output_truncation_k` continues to mean this executable cleartext simulation
value. The implementation must not repurpose that field for the paper-facing
precision. CPU truncation, fused CUDA truncation, online reward probes, final
evaluation, fixed-action experiments, and legacy full action vectors continue
to consume the same values through their existing paths.

## Public Paper Semantics

The public Stage-2 preset meaning follows the paper:

| Preset | Ciphertext K by Block | Ring Bits |
| --- | --- | ---: |
| High | `(13, 13, 13, 13, 13)` | 40 |
| Medium | `(12, 12, 12, 12, 12)` | 39 |
| Low | `(11, 11, 11, 12, 11)` | 38 |

Each preset also exposes the explicit per-block reserve:

```text
reserve_bits = ciphertext_k - simulation_k
```

| Preset | Reserve Bits by Block |
| --- | --- |
| High | `(2, 3, 3, 1, 2)` |
| Medium | `(3, 4, 4, 2, 3)` |
| Low | `(4, 5, 5, 4, 4)` |

These reserve bits are metadata that explain the conservative cleartext
simulation. They are not an additional runtime operation and must not be
subtracted again inside `function_handler.py`.

## Data Model

`blb_stage2_rl.precision_presets.PrecisionPreset` becomes the single source of
truth for both semantic planes:

- `ciphertext_k_by_block`: paper-facing deployment precision.
- `simulation_k_by_block`: executable cleartext precision, preserving current
  behavior.
- `reserve_bits_by_block`: derived property equal to the element-wise
  difference above.
- `ciphertext_ring_bits`: paper-facing 40/39/38-bit ring choice.
- `communication_utility`: existing reward/cost utility, unchanged.

The compatibility property `k_by_block` continues to return
`simulation_k_by_block`. This protects existing executable consumers while new
code must use explicit names. Validation requires five values, supported
simulation K values, non-negative reserve bits, and exact reconstruction of
the simulation tuple.

The existing executable preset version remains unchanged because the policy
action, full vector, reward semantics, and model behavior do not change. A new
paper-semantics metadata version identifies the added reporting contract.

## Action and Runtime Flow

The compact per-layer RL action remains `(block4_fusion, preset_index)`. The
preset index still has cardinality three and retains the order High, Medium,
Low.

`apply_layer_action()` continues to splice `simulation_k_by_block` into the
legacy full vector. `LayerwiseDecodedAction.k_by_block` therefore remains the
executable simulation tuple. No reserve metadata is installed into model
configuration objects, and `output_truncation_k` is not changed.

The paper-facing tuple is recovered from `precision_preset_index`, not inferred
from mutable model configuration. This keeps public meaning attached to the RL
choice while preventing it from affecting inference.

## Cost and Reward

The existing H/M/L communication utilities remain `0.0`, `0.5`, and `1.0`.
PPO reward, hard priorities, resource score, Shapley-compatible diagnostics,
and executable removed-bit diagnostics remain unchanged.

New reporting diagnostics may expose ciphertext removed bits and reserve bits,
but they must not feed reward or candidate ranking in this task.

## Reporting Contract

`describe_layerwise_action_matrix()` and persisted installed-action summaries
must report, per layer:

```json
{
  "precision_preset_index": 0,
  "precision_preset_name": "high",
  "ciphertext_truncation_k_by_block": {
    "block1": 13,
    "block2": 13,
    "block3": 13,
    "block4": 13,
    "block5": 13
  },
  "cleartext_simulation_k_by_block": {
    "block1": 11,
    "block2": 10,
    "block3": 10,
    "block4": 12,
    "block5": 11
  },
  "reserve_bits_by_block": {
    "block1": 2,
    "block2": 3,
    "block3": 3,
    "block4": 1,
    "block5": 2
  },
  "ciphertext_ring_bits": 40
}
```

For compatibility, the existing `truncation_k_by_block` field remains present
and continues to contain the executable simulation tuple. This field must be
documented as a compatibility alias rather than relabeled as ciphertext K.

## Compatibility

- No policy-head shape change.
- No compact action-matrix change.
- No legacy full-vector change.
- No installed model configuration change.
- No reward or ranking change.
- No checkpoint execution change.
- No fusion-map rebuild.
- No truncation backend change.

Only the paper-facing metadata schema and human-readable semantics are new.

## Verification

Tests must lock the following properties:

1. The paper tuples and ring widths exactly match the paper.
2. `ciphertext_k - reserve_bits == simulation_k` for all 15 block/preset
   positions.
3. `k_by_block` remains the original executable tuple.
4. Every H/M/L action produces the same legacy full vector as a frozen
   pre-change golden vector.
5. Every installed `output_truncation_k` remains the original executable value
   for all five blocks.
6. Existing variable-cost and reward-facing values remain unchanged.
7. Action descriptions expose both semantic planes and the reserve.
8. Static shared-path tests continue to prove training, final evaluation, and
   experiment callers use canonical materialization.

GPU verification is required only to reconfirm that existing fused CUDA tests
remain green; this task intentionally does not modify CUDA or truncation
kernels.
