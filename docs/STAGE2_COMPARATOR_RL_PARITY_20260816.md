# Stage-2 Comparator and PPO Parity Audit

## Status and scope

This document audits the current two-stage `bo_rf`, `greedy`, and `coinn_ga`
paths against the authoritative layerwise Stage-2 PPO implementation. It
separates three different claims:

1. shared scientific machinery, which must be identical;
2. optimizer-specific proposal, update, repeat, shortlist, and termination
   behavior, which is intentionally different;
3. historical facts that are not recoverable from archived evidence and must
   not be presented as proven.

The local static audit and parity corrections are complete. On 2026-08-17, the
exact aggregate commit passed the focused and full server test gates, Python
compilation, launcher syntax, and tracked-clean checks. A real-BERT same-action
dynamic parity gate is still required before canonical promotion or a formal
comparator launch; it is currently blocked by the server NVIDIA driver/library
mismatch described below.

## Current server verification

The tested code-bearing server checkout was synchronized through Git to commit
`d6d1fa1fa51ac6a82d82c1302150d026f9d7bdbd`, tree
`9ffb3931966302abc4f4bb213f29e7ad206278bb`, and remained tracked-clean.

- Focused Stage-2 parity gate: 382 passed, 240 subtests passed.
- Same-action PPO/comparator seed and metric gate: 248 passed, 47 subtests
  passed, including the direct same-action path comparison.
- Full project pytest gate: 2360 passed, 16 skipped, 613 subtests passed.
- Python `compileall`, `bash -n llama_7B_LayerImportance.sh`, and
  `git diff --check`: passed.
- Server evidence:
  `/hy-tmp/rfr_stage2_comparator_parity_gate_20260817/focused_stage2_v3.xml`
  `/hy-tmp/rfr_stage2_comparator_parity_gate_20260817/same_action_seed_focused_d6d1fa1f.xml`,
  and
  `/hy-tmp/rfr_stage2_comparator_parity_gate_20260817/full_pytest_d6d1fa1f.xml`.

The 16 full-suite skips include the CUDA-specific gates. The server kernel has
NVIDIA module `580.173.02`, while `libnvidia-ml.so.1` and `libcuda.so.1` resolve
to `580.159.03`. Consequently, `nvidia-smi` reports a driver/library version
mismatch and PyTorch reports CUDA error 803 with zero visible devices. No
matching `580.173.02` user-space library is installed, so this cannot be fixed
with a non-persistent library-path override. The audit did not install drivers,
change system libraries, or reboot the server.

## Authoritative references

- Historical Stage-2 PPO source: `5c222da6186b8a60244b46029bbc8dac79befb34`
- Archived run-evidence commit: `8c2a526dbf793c95c388b5f8544a793e83c733dc`
- Archived launch evidence:
  `rl_training_data_points/stage2/archives/bert-base-mrpc-60k-20260805/small_files/run/launch_evidence/launch_command.sh`
- Current canonical base for this audit:
  `9d833d90760b1bf85fca4c8650e8149f61119ad2`

The archive commit contains evidence; it is not the PPO source commit. The
historical launch used a manual Stage-1 prerequisite rather than a dynamically
loaded Stage-1 result:

- GELU: `[1,2,1,1,1,1,1,1,2,1,1,1]`
- Softmax: `[6,6,6,6,6,6,6,6,6,6,6,6]`

Each comparator intentionally binds its own selected Stage-1 result. Therefore
an end-to-end two-stage comparator differs from historical PPO both in search
algorithm and, when selected vectors differ, in the Stage-1 prerequisite. This
is the defined two-stage comparison protocol; it is not an optimizer-only
Stage-2 ablation.

## Shared setup before optimizer dispatch

Both PPO and all three comparators construct the following objects before the
code dispatches on `search_backend`:

- the same BERT-base MRPC evaluator and Stage-2 batch-64 loaders;
- the same calibrated baseline action, fusion map, maximum-SF table, and real
  in-process Rescale optimizer;
- the same layerwise environment and 12-layer schedule;
- the same 256-example stratified F1 probe environment;
- the same full 408-example F4 validation environment;
- the same robust F1 baseline reference and strict A/B/C references;
- the same GPU reward-probe owner and model-install path.

Formal comparator validation additionally requires the pinned MRPC fixture,
model ID, model revision, tokenizer revision, canonical-row hash, full
validation-order hash, and probe-order hash. These identities are persisted in
the Stage-2 invocation contract.

## Baseline contract

There are three baseline-related mechanisms. They must not be conflated.

| Mechanism | Data and repeats | Role |
| --- | --- | --- |
| Cost calibration | `calibrate_baseline_samples=8` | Estimates structural cost normalizers; it is not the statistical accuracy/stability baseline. |
| Online F1 robust baseline | 256 stratified validation examples; minimum 5 groups x 3 trials | Builds the online bootstrap reference used by both PPO and comparators. If any channel has zero/invalid sample variance after 15 trials, collection continues deterministically, up to 10 groups x 3 trials. |
| Strict F4 banks | Full 408 validation examples; A, B, C each exactly 5 groups x 3 trials | A is a 15-trial early gate; A+B is the 30-trial promotion reference; A+B+C is the 45-trial final reference. No variance-extension groups are permitted. |

All statistical baselines evaluate the same canonical all-max BLB baseline
action with real noise installed. The default run seed is 42. Strict baseline
group indices start at 1000, 2000, and 3000 for A, B, and C respectively, so
their trial streams are deterministic and disjoint.

Consequently, the earlier shorthand statement "the baseline is exactly 5x3"
was incomplete. It is exact for each strict bank and only a minimum for the
online F1 baseline.

## Precision and stability constraints

The active robust reference uses pooled trial values and sample standard
deviation (`ddof=1`). With precision tolerance `t=0.001` and effective
stability multiplier `m=2.0`, its six limits are:

```text
loss_mean_limit    = baseline_loss_mean    * (1 + t)
accuracy_limit     = baseline_accuracy     * (1 - t)
weighted_f1_limit  = baseline_weighted_f1  * (1 - t)
loss_std_limit     = baseline_loss_std     * m
accuracy_std_limit = baseline_accuracy_std * m
f1_std_limit       = baseline_f1_std       * m
```

MRPC loss is sample-count weighted across inference batches. Accuracy is
computed globally from concatenated predictions and labels. Weighted F1 is
also computed globally. Candidate means are arithmetic means across trials and
candidate standard deviations use `ddof=1`.

Online F1 evaluates every candidate with K=3 trials and 4096 deterministic
bootstrap resamples. It computes six pass probabilities. The online hard gate
requires the minimum of the three precision probabilities and the minimum of
the three stability probabilities to each be at least 0.50.

Strict F4 does not use the 0.80 and 0.95 bootstrap probabilities as hard
scientific gates. Those probabilities are retained as diagnostics and
tie-break evidence. The hard strict gate applies the six point limits above to
joint, compute-only, and communication-only materializations:

- Bank A point gate;
- pooled A+B point gate;
- held-out pooled A+B+C point gate;
- the same point gates for both isolated resource axes.

The historical `stage2_stability_tolerance=1.2` is retained in the invocation
for provenance and legacy compatibility. In robust-constrained layerwise mode,
the active statistical limits use `stage2_stability_multiplier=2.0`; the raw
1.2 value is not the active robust gate.

## Candidate and repeat behavior

The shared action space contains one atomic six-valued gene per layer:

```text
(Block4 fusion count 0 or 1) x (high, medium, or low precision)
```

For 12 layers, all four algorithms therefore operate over the same `6^12`
materializable policy space. A candidate is optimizer-valid only when it can
produce a complete legacy action, pass real replan/materialization, install the
configuration, and execute a model forward.

Candidate repetition is intentionally not identical:

| Behavior | PPO | BO-RF / Greedy / COINN-GA |
| --- | --- | --- |
| Repeated online action | Re-evaluated with a fresh K=3 group in every episode | Returned from the exact-action cache; no second model evaluation |
| Online evidence pooling | Repeated F1 groups are pooled in the candidate store, up to the promotion target | One K=3 result per unique action |
| Seed index | PPO episode index | Global index of actual unique model evaluations |
| Strict evidence | Fixed F4 banks cached by candidate identity | The same fixed F4 banks cached by candidate identity |

PPO and all three comparators now call one shared seed-plan function for the
episode reset seed and K=3 probe seed. A direct regression gate executes the
same action through the PPO episode collector and comparator runtime evaluator
at the same global stream index, then requires exact equality for the compact
action, materialized full vector, final fingerprint, trial seeds, each raw
loss/accuracy/F1 trial, and all six aggregate metric values.

Proposal order still differs by algorithm, and comparator cache deduplication
means the optimizer families do not consume an identical online trial
sequence. Therefore, the same action encountered at different online stream
indices intentionally receives different noise trials and need not have equal
online K=3 values. Cross-algorithm result comparison must use the fixed strict
F4 A/B/C banks. Their seed order is optimizer-independent, so the same complete
Stage-1 plus Stage-2 configuration must produce exactly equal strict trial
values and aggregate metrics; any mismatch invalidates the run rather than
being accepted as an optimizer difference.

Online comparator ranking uses the same six bootstrap probabilities and the
same resource objective. Feasible candidates rank ahead of infeasible ones.
The strict shortlist contains exactly the best five optimizer-valid,
materializable, model-forward candidates; if fewer than five are online
feasible, valid online-infeasible candidates fill the remaining positions.

PPO does not use a fixed top-five shortlist. It admits fresh P3 candidates on
its resource frontier and certifies the current strict winner. Comparators pass
`priority=3` to the shared strict machinery deliberately, because their formal
protocol says every eligible top-five candidate must receive strict
validation. This is an explicit algorithm/protocol difference.

Per shortlisted candidate, the maximum strict work is:

```text
joint A+B+C                  45 trials
compute-only A+B+C          45 trials
communication-only A+B+C    45 trials
maximum per candidate       135 trials
maximum for top five        675 trials
```

Point-gate failures short-circuit later banks, and persisted evidence is reused,
so actual trial counts can be lower. Infrastructure failures remain
`search_complete_pending_strict`; they cannot be exported as least-violating
scientific results. If all five complete strict evaluation but none is
feasible, the comparator may export the strictly evaluated least-violating
candidate with `formal_feasible=false`. PPO has no corresponding final
least-violating export rule.

## Fusion-count, SF, K, and model-install chain

The H/M/L precision presets are shared by PPO and comparators:

| Preset | Block1 | Block2 | Block3 | Block4 | Block5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| high | 11 | 10 | 10 | 12 | 11 |
| medium | 9 | 8 | 8 | 10 | 9 |
| low | 7 | 6 | 6 | 8 | 7 |

For every layer, the exact materialization chain is:

1. Decode the atomic layer gene into Block4 fusion count and one H/M/L preset.
2. Resolve exactly one fusion-map option matching the requested fusion count.
3. Keep Block2 and Block5 at fusion count 1; select Block4 at 0 or 1.
4. Expand each selected option to all underlying SF action indices, then
   overwrite its K slot with the selected preset value.
5. Keep Block1 and Block3 SF values on the calibrated baseline and replace only
   their K slots.
6. Build the complete legacy action vector for all 12 layers.
7. Decode all five block families, run the real in-process Rescale optimizer,
   write optimizer outputs back into the installable configurations, and
   compute the final configuration fingerprint.
8. Install Block1 through Block5 configurations through the shared reversible
   bridge and execute the real model forward.

Binary K truncation uses `trunc(x * 2^K) / 2^K`. The recorded ring size 43 and
source fractional bits 24 identify the launch contract but do not replace that
binary runtime operation.

The strict gate derives three configurations from the same candidate:

- `joint`: selected fusion and selected K;
- `compute_only`: selected fusion with every K reset to baseline K=13;
- `communication_only`: baseline fusion/SF with only selected K installed.

All three go through the same replan and model-forward path. Stored fingerprints
are recomputed before strict selection, so stale or mismatched materialization
cannot be selected.

Stage-1 GELU and Softmax vectors are also part of the candidate identity. The
Stage-1 result byte SHA-256 and semantic selection hash are now checked at the
Stage-2 boundary and included in strict candidate identity. Replacing a result
file at the same path therefore cannot silently reuse old strict evidence.

## Optimizer-specific behavior that remains different

- PPO samples sequential actions, records every episode, pools repeated-action
  online evidence, updates one policy/value network, and uses frontier-based
  strict promotion.
- BO-RF uses a categorical random forest, probability of feasibility times
  deterministic expected improvement, and a deterministic violation ordering
  before any feasible point exists.
- Greedy exhaustively verifies 1-opt, then 2-opt, and returns to 1-opt after an
  accepted 2-opt move.
- Current Stage-2 COINN-GA uses tournament parent selection, seven elites,
  forced adjacent replacement mutation over at most four layers, and a
  deterministic complete adjacent-neighborhood fallback after collision
  retries. It does not use crossover or immigrants.
- Current Stage-2 COINN-GA has population 64, an 800-generation safety cap, a
  45,664-inference cap, and a five-generation no-incumbent-improvement stop.
  The separate Stage-1 200-generation no-early-stop contract does not apply to
  Stage-2.

These differences define the algorithms and must not be removed in the name of
parity.

## Corrections to the earlier audit

The earlier review contained several unsupported or incorrect statements:

1. It described every baseline as exactly 5x3 and omitted the deterministic
   online variance-extension path.
2. It said Stage-2 COINN-GA used crossover. That was inferred from a generic
   codec/helper that supports crossover; the actual Stage-2 GA update loop does
   not call it.
3. It described comparator online seeds as action-keyed. The corrected path is
   global-evaluation-index keyed, matching PPO's seed derivation for actual
   evaluations.
4. It overclaimed identical online trial order. Optimizer proposal order and
   comparator deduplication make that impossible; only strict bank order is
   fixed and identical.
5. It treated the archived evidence commit as the PPO source commit. They are
   different commits and are now recorded separately.
6. It did not distinguish the raw legacy stability value 1.2 from the active
   robust multiplier 2.0.
7. It did not state that historical PPO used a manual Stage-1 vector while each
   comparator binds its own Stage-1 result.
8. It recorded `terminal_eval_batch_size=4` as though comparators batch four
   proposed actions together. The comparator evaluator currently evaluates one
   proposed action at a time; the value is pinned for launch/topology parity but
   is not a four-action comparator batching mechanism.
9. It claimed historical dataset row/revision identity without archived proof.
   The current comparator fixture proves its own exact rows and order, but the
   old PPO archive predates this fixture and does not by itself prove the exact
   historical dataset revision.

## Remaining proof obligations

Before launch, the server must still prove all of the following from the exact
aggregate commit and tree:

- real BERT accepts only the pinned 12-layer MRPC model and fixture;
- a fixed action produces identical PPO/comparator full vectors, replan
  fingerprints, installed Block1-Block5 configurations, trial seeds, raw trial
  metrics, six online probabilities, and strict A/B/C evidence;
- fresh and resumed comparator runs preserve ordered observations and strict
  evidence without duplicate model evaluations;
- optional Paean failure leaves the authoritative strict-F4 result unchanged;
- PPO checkpoints and previously frozen PPO results remain readable and
  unchanged.

Until those server gates pass, this audit is a code-level alignment result, not
a completed scientific parity claim.
