# Production Code Cleanup Design

## Objective

Reduce the repository to the code required to run the final research system,
without changing its scientific behavior. The resulting source must expose one
unambiguous implementation for each production function, contain no personal
information, and document Stage 1 and Stage 2 with a short English README.

The cleanup starts from canonical commit
`f86359bce6ac432cd7f431c21627efd5930f9e02`. The cloud-only recovery branch
`codex/archive-pre-thorough-code-cleanup-20260825-024715` preserves that state;
its archive commit is `88aa21283a5c433792bcdf9801b04c290e752b9a`
and its parent is the exact canonical commit.

## Non-goals

This task does not delete or reorganize training data, checkpoints, experiment
outputs, server backups, or historical result bundles. In particular, it does
not remove content under the following result-oriented trees:

- `server_backups/`
- `rl_training_data_points/`
- `Parting Chapter/`, `Prelude Chapter/`, `Previous Chapter/`, and
  `Previous Chapter Server Reserve/`
- `Paean/outputs/`
- `experiment/outputs/`
- every result subdirectory under `experiments/`
- `gelu_analysis/`
- `glue_submission/`
- `reports/`
- `Model_analysis/model_statistics/weight_hist_out/`
- `Rescale_optimizer/diagnose_certacc_output/`
- root-level `glue_final_configs_best_*.json` result files

Generated report PDFs inside those preserved result trees are not treated as
AI design guidance and remain unchanged.

## Production Feature Matrix

The cleaned source supports exactly these model and dataset combinations:

- BERT-base with MRPC, RTE, or SST-2
- BERT-large with MRPC, RTE, or SST-2

The retained production functions are:

1. deterministic Profile generation;
2. Stage-1 RL search over GELU and Softmax approximation degrees;
3. Stage-2 layerwise PPO search over fusion and MPC truncation precision;
4. BO-RF, Greedy, and COINN-GA comparator searches;
5. Paean final evaluation;
6. fusion-map and required profile/fixture construction;
7. real in-process Rescale optimizer materialization;
8. current checkpoint save, graceful stop, and resume;
9. elastic multi-GPU reward-trial execution;
10. tests and the generic multi-agent Git safety protocol.

No compatibility claim is made for removed models, datasets, network variants,
checkpoint schemas, command aliases, or research ablations.

## Integration Order

The cleanup branch is based on the canonical MPC truncation semantics, so the
paper-facing ciphertext K metadata and the unchanged executable simulation K
are present before cleanup begins.

Before deleting source, integrate the latest train-probe implementation from
source commit `d67100a7cb0444275498b780c10b46631a7577c1`. Its merge base with the
current canonical is `480e154053b1303e140077a05c46295cab95ef0a`; the only overlapping
path is `AGENTS.md`. Resolve that document by retaining both current MPC
semantics and the train-probe contract, then later replace its historical
content with the concise generic agent guide defined below.

The integration must produce a dedicated commit and pass the complete CPU and
static project gates before any deletion commit is made. GPU verification may
be deferred while the current server driver is unavailable, but final
canonical advancement is blocked until the required GPU smokes pass.

## Data Protocol

All six profiles use one pinned GLUE revision and one deterministic probe
identity per dataset.

1. Read the GLUE `train` split.
2. Shuffle with seed 42.
3. Select exactly 256 examples with stratification by binary label.
4. Sort selected shuffled positions to fix order.
5. Persist raw IDs, labels, positions, label histogram, and identity hash.
6. Reuse the exact materialized probe batches for Profile, Stage 1, Stage 2,
   comparators, baseline calibration, candidate promotion, and strict top-5.

Stage-2 online evaluation uses three noise trials. Search-gate Bank A, B, and C
use disjoint trial seeds over the same 256 examples. A+B forms the promotion
reference and A+B+C forms final search certification. They are repeated-trial
banks, not separate dataset splits.

The complete GLUE `validation` split is used only after search has fixed a
configuration. Final-evaluation metrics cannot update PPO, reward thresholds,
candidate ranking, convergence, or strict selection.

During cleanup, misleading internal `validation_banks` terminology should be
renamed to `search_gate_banks` where this does not change serialized scientific
records. A current schema may be revised because all pre-train-probe
checkpoints are explicitly incompatible; no permissive legacy migration is
introduced.

## Retained Stage-2 Semantics

Stage 2 has one policy implementation:

- layerwise causal shared GTrXL actor-critic;
- the current small architecture only;
- robust constrained reward only;
- current fusion-count and H/M/L precision action matrix;
- current candidate store, A/B/C promotion, strict top-5, and Bank-C
  certification;
- current deterministic seeding and checkpoint schema;
- current process-based elastic reward-probe backend.

Paper-facing MPC precision remains distinct from executable simulation:

- ciphertext H/M/L K and ring width are report metadata;
- `output_truncation_k` remains the cleartext simulation K;
- reserve bits are metadata derived as ciphertext K minus simulation K;
- reward, cost, action shape, fusion mapping, materialized configs, and final
  evaluation continue to consume simulation K.

## Source to Remove

Remove implementations that exist only for rollback, ablation, reproduction,
or one-off diagnosis:

- General-RL training and search;
- legacy Stage-2 v2 noise RL;
- large shared GTrXL and separate-critic network variants;
- single-shot, retired per-block, substage, and OSR execution paths;
- non-production reward variants and warmup ablations;
- heuristic, stub, and subprocess Rescale fallbacks;
- episode-parallel and thread-fallback paths superseded by reward-trial process
  parallelism;
- obsolete CLI aliases, compatibility flags, and old checkpoint inference;
- source files with `.bak` or dated rollback suffixes;
- one-off A/B, benchmark, sweep, watchdog, report, and diagnosis scripts;
- obsolete configuration files and presets for removed paths;
- stale reports and design notes that describe retired behavior.

The following unused submodules and their `.gitmodules` entries are removed:

- `EzPC`
- `LLM-Adapters`
- `importance-aware-sparse-tuning-IST-paper`

Required environment setup, train-probe fixture construction, Profile,
fusion-map construction, Rescale configuration validation, and Git guard tools
remain.

## Tests and Development Protocol

Tests are retained as verification infrastructure. Tests whose only purpose is
to preserve deleted behavior are removed. Tests for retained behavior are
updated to use the final API and must not rely on source-text counts or stale
compatibility aliases when a behavioral assertion is possible.

Keep concise, generic versions of:

- `AGENTS.md`
- `CLAUDE.md`
- `docs/GIT_MULTI_AGENT_PROTOCOL.md`
- `agent_handoffs/README.md`
- `agent_handoffs/schema.json`

Remove historical task handoffs, prior aggregate manifests, Superpowers plans
and specifications, agent-specific readmes, server command handoffs, and other
process narratives. The cleanup task's own handoff and final aggregate
manifest are added by the protocol after the source cleanup is verified.

## Personal Information and Documentation

Active source, configuration, comments, README files, and retained protocol
documents must not contain personal names, usernames, workstation paths,
private repository owner names, server addresses, SSH commands, credentials,
or messaging identifiers.

The main README is replaced with a concise English document that contains:

- supported profiles;
- prerequisites;
- Stage-1 first-run and resume commands;
- Stage-2 first-run and resume commands;
- comparator commands;
- final-evaluation command;
- the fixed production configuration and output locations.

Remove AI-oriented guidance documents and AI method-design PDFs outside the
preserved result trees. Do not remove ordinary scientific references required
to understand or reproduce the method.

## Comment Policy

Do not mechanically erase every comment. Remove comments and docstrings that
describe prompts, user requests, chronological debugging, abandoned options,
or obvious syntax. Preserve shebangs, type/lint pragmas, licenses, and comments
that encode non-obvious scientific or systems invariants.

Write a small number of concise English comments for:

- train-probe identity and split isolation;
- action index versus decoded value semantics;
- ciphertext K versus simulation K;
- hard-priority reward constraints;
- deterministic trial seeding;
- Rescale materialization ownership;
- checkpoint atomicity and resume boundaries;
- elastic GPU assignment and failure handling.

Runtime log and artifact wording is changed only when it is stale, personal,
or refers to removed behavior. Scientific fields are not renamed merely for
style.

## Verification

Verification is staged so a broad deletion cannot hide the first regression.

### Integration gate

Before cleanup:

- validate the six-profile registry and pinned train-probe identities;
- run the focused Stage-1, Stage-2, comparator, checkpoint, and final-eval
  suites;
- run full `unittest` and `pytest` on the server;
- compile Python and validate shell scripts;
- confirm MPC execution snapshot parity remains at
  `68a50ef270d894f3995bd01437b6febcb0bd2b3c757b42edb03485ad2ceb63e7`.

### Deletion gates

After each deletion group:

- scan imports, dynamic module names, shell commands, presets, and config paths;
- run focused tests for affected owners;
- reject any remaining reference to a removed module or option;
- confirm tracked data/result paths are unchanged.

### Final acceptance

- complete `unittest` and `pytest` pass on the exact source commit;
- all retained Python modules compile and shell entrypoints pass syntax checks;
- six Profile, Stage-1, and Stage-2 GPU smokes run on real CUDA;
- BO-RF, Greedy, and COINN-GA minimal searches run through the shared model and
  data paths;
- fixed pre/post-cleanup inputs produce identical action, materialization,
  reward, cost, candidate, checkpoint, and final-evaluation scientific state;
- personal-information and stale-reference scans return no active-source hits;
- README commands are exercised from a clean Git checkout;
- local canonical, remote canonical, and server canonical have identical full
  commit and tree IDs and are tracked-clean.

## Git Workflow

All source edits occur in `codex/task-production-code-cleanup-20260825`.
Each coherent change is committed and pushed. Server source is obtained only
through Git. After a completed task handoff, the authorized aggregator reviews
all remote heads, creates one clean aggregate, runs final server validation,
fast-forwards `jk_standard_rl`, and verifies local/Git/server parity.
