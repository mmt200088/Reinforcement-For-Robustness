# Stage-2 Block4-Only Fusion Action Design

## Goal

Change Stage-2 fusion-count RL so that only Block4 fusion count is selected by
the policy. Every Block2 and Block5 occurrence must use `fusion_count=1`.
Truncation K remains independently selectable at every existing block step.

## Action Semantics

The sequential schedule and two-slot policy shape remain unchanged. Each step
still presents `(fusion_choice, k_index)` to the policy, but the fusion choice
uses a per-step local option domain:

- Block2: one local fusion choice, mapped to the map option whose declared
  `fusion_count` is exactly `1`.
- Block4: all map options remain selectable by the policy.
- Block5: one local fusion choice, mapped to the map option whose declared
  `fusion_count` is exactly `1`.
- Other scheduled blocks retain their existing behavior.
- K retains the existing six-level domain and is not fixed by this change.

The schedule must fail before training if a Block2 or Block5 graph does not
contain exactly one option with `fusion_count=1`. The implementation must not
assume that `option_id == fusion_count`.

## Canonical Resolution Path

`FusionStepSpec` carries the policy-local-to-map option mapping. A single helper
resolves a policy-local fusion choice to the real map option ID. The resolved
option is then used consistently for:

1. map expansion into the full block action vector;
2. precision-boost explicit field values;
3. Rescale Optimizer replan and canonical cfg write-back;
4. model noise installation and terminal inference;
5. fusion-cost/reward bookkeeping;
6. persisted action logs and human-readable reports.

No runner-only mask or final-eval-only translation is allowed. Fixed-action
experiments must enter through the same Stage-2 RL terminal install path.

## MRPC BERT-Base Evaluation

Run four full-validation groups. Every group uses Softmax degree 6 in all 12
layers, truncation K=13 at every block, five independent inference repeats, and
the same deterministic seed schedule.

| Pair | GELU configuration | Group | Block2 | Block4 | Block5 |
|---|---|---|---:|---:|---:|
| A | MRPC Stage-1 best `[1,2,1,1,1,1,1,1,2,1,1,1]` | control | 0 | 0 | 0 |
| A | same | fixed B2/B5 fusion | 1 | 0 | 1 |
| B | `[4]*12` | control | 0 | 0 | 0 |
| B | same | fixed B2/B5 fusion | 1 | 0 | 1 |

The report must show per-repeat and mean loss, Accuracy, and Weighted F1,
together with absolute and percentage deltas within each pair. It must also show
the effective per-layer fusion count and K values, plus installation evidence
that the selected configs reached the model.

## Other Profile Map Completion

Build or recover complete Block2, Block4, and all Stage-1-reachable Block5 graph
maps for:

- BERT-base RTE and SST-2;
- BERT-large MRPC, RTE, and SST-2.

For every generated graph, record available fusion counts, option count,
precision-boost state, K-independence, baseline-option equivalence, and build
provenance. A graph with any `fusion_count > 1` is an anomaly for this task: do
not silently activate it in RL, and report its profile, graph key, option, and
underlying fused rescale set for review.

## Verification

Tests must prove:

- Block2 and Block5 expose one policy-local fusion choice and resolve it to the
  real `fusion_count=1` option.
- Block4 retains its full option domain.
- K choices remain unchanged and independently applied.
- effective option logging, reward bookkeeping, precision boost, replan, and
  terminal model install all use the same resolved option.
- missing or ambiguous `fusion_count=1` options fail before training.

All project execution and tests run on a verified server snapshot. Local work is
limited to source edits, static checks, git commits, and artifact integration.
