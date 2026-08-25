# Comparator Single-GPU Optimization Design

## Status and scope

This design covers one-card execution optimization for all three two-stage comparator backends:

- `bo_rf`
- `greedy`
- `coinn_ga`

The work starts from canonical commit `5f8441a8d30cd5cf9fac871d4240f1de61328b97`, whose tree `95c0fd1b1cb9d5448f29d8ad7a181ffee6108253` contains the completed comparator-correctness task. The initial implementation occurs only on ordinary task branch `codex/task-comparator-single-gpu-optimization-20260810`. Any later retention-policy source adjustment occurs in its own ordinary task branch after server evidence; it is never folded back into the initial implementation task. COINN-GA candidate-level multi-GPU execution remains explicitly deferred.

The task changes execution mechanics only. It must not change any scientific search configuration, action space, candidate-generation order, candidate-evaluation order, random-number call sequence, random-number draw count, final RNG state, deterministic replay, model input, optimizer request, trial seed, callback order, durable append order, persistence order, budget, stopping rule, strict-validation rule, or final-selection rule.

## Locked scientific behavior

### Shared two-stage contract

1. One backend drives both Stage-1 and Stage-2.
2. Stage-1 continues to evaluate the real `validation_full` split with `use_train=False`.
3. Stage-1 searches GELU degrees only; every layer's Softmax degree remains fixed at 6.
4. Stage-2 continues through compact layer-action decode, production materialization, real `Rescale_optimizer`, optimizer write-back and binding restoration, final configuration fingerprinting, `BLBNoiseRLBridge` installation, and real repeated-trial model forward.
5. Search observations remain durably committed in request order. Cache hits do not consume real-evaluation budgets. Completed resume performs zero new model forwards. Stage-1 search evaluations are not assumed to equal physical forwards because the existing plaintext evaluation cache may satisfy a request. Stage-2 separately preserves total observation count and inference-performing evaluation count; optimizer/materialization-invalid observations increment only the former.
6. Strict top-five A/B/C validation, joint/compute-only/communication-only families, point/family gates, and deterministic least-violating fallback remain unchanged.
7. Performance mode, phase timing, counters, and benchmark files are execution metadata. Algorithms never read them, and they are excluded from checkpoint identity, resume contracts, ranking, feasibility, completion, and final selection.

### Backend-specific invariants

- **BO-RF:** retain the exact 64-point initial design, exact 2,048-action pool, full ordered-prefix RF refit before every selected evaluation, 128-tree RF parameters and seed, one selected candidate per iteration, patience 100, and maximum 50,000 real evaluations. Preserve Stage-1's distinct design/pool RNGs and Stage-2's one continuous initial-design/pool generator without adding draws. No warm start, stale model, q-batch, speculation, alternate RF, or asynchronous persistence.
- **Greedy:** retain complete deterministic 1-opt scans, complete 2-opt scans only after 1-opt has no improvement, best-improvement selection after the complete scan, restart at full 1-opt after every accepted move, and verified-local-optimum termination only after both complete neighborhoods fail to improve. Stage-2's six-valued `(Block4 fusion, precision preset)` layer gene remains indivisible.
- **COINN-GA:** retain population 64, elites 7, exactly 57 inference-performing offspring per complete generation, maximum 800 generations, patience 5, mutation-only reproduction, no crossover or immigrants, per-layer mutation probability `1/12`, at least one and at most four changed layers, and no partial generation when fewer than 57 inference evaluations remain. Stage-1 generates all 57 unique unseen children before evaluating the generation. Stage-2 deliberately generates and evaluates children sequentially because invalid feedback changes its forbidden set and future duplicate handling; invalid or materialization-skipped observations remain durable and forbidden but do not fill one of the 57 inference-performing offspring slots. A Stage-2 generation may therefore contain more than 57 observations, while population and stagnation state change only after exactly 57 inference-performing offspring complete.

## Execution profiles

A single typed execution-only selector, `ComparatorExecutionProfile`, provides reproducible A/B runs and accepts exactly these serialized values:

- `reference`: current full reinstall/clear behavior and current BO matrix construction, with no performance timing or detailed event stream.
- `timing_only`: reference scientific behavior plus aggregate phase timing; detailed event JSONL is written only while `ComparatorRetentionPolicy.timing_only_event_stream` is true.
- `optimized`: reference scientific behavior routed through exactly the optimization mechanisms whose static retention booleans are true. The initial benchmark source enables Stage-1 delta installation, the BO ordered workspace, and Stage-2 persistent full apply so each candidate mechanism can be measured before any retention verdict.

The selector is threaded explicitly through the Stage-1 and Stage-2 comparator runners; it is not inferred from an environment variable. It is not a scientific parameter and must not enter scientific manifests or resume equality. The selected profile, implementation commit/tree, Python version, NumPy version, scikit-learn version, and joblib version are execution-only benchmark metadata so exact-sequence claims are scoped to a recorded numerical environment; they never enter scientific metadata. Every summary sidecar and benchmark summary exposes the exact same six-field identity mapping, while detailed event JSONL rows need not duplicate it. The benchmark parent supplies the already-verified source commit/tree to every child recorder and rejects missing or unavailable identity values before exact comparison; package versions are read through distribution metadata without importing the numerical libraries. A non-benchmark runner may record `unavailable` for commit/tree, and independently for NumPy, scikit-learn, or joblib only when that distribution-metadata lookup fails. Its `python_version` is always concrete and nonempty. These telemetry limitations never change scientific execution.
The production default remains `reference` throughout this task.
Any future switch to `optimized` requires a separate ordinary task, a new source handoff, explicit aggregation, exact canonical commit/tree parity, and server verification. `reference` remains available for diagnosis and future regression checks.

A separate torch-free `ComparatorRetentionPolicy` statically controls which already-implemented mechanisms the `optimized` profile may route through. Its four booleans are `stage1_delta_install`, `bo_ordered_workspace`, `stage2_persistent_install`, and `timing_only_event_stream`. The initial implementation task sets all four to true so the canonical benchmark can measure every candidate mechanism. After exact server evidence, a separate ordinary TDD task must set every failed boolean to false. A false optimization boolean routes `optimized` through the corresponding tested reference implementation; a false telemetry boolean keeps aggregate counters/timings but disables normal detailed event JSONL. This policy is execution-only: it is excluded from search configs, scientific manifests, run locks, checkpoints, resume identity, ranking, feasibility, completion, and the production profile default.

The initial result branch publishes a self-contained immutable `comparator_single_gpu_trace_bundle_v1`. Its manifest hashes every Stage-1 and Stage-2 trace JSON. If a retention boolean changes, the final canonical benchmark writes a separate final search-core output for exact sequence/artifact comparison; it must not overwrite, replace, mutate, or regenerate the initial bundle. Final model replay reads actions only from the pinned initial result branch, result tip, trace path, and manifest hash. Model replay never regenerates actions. If all booleans pass, the initial exact canonical run is the final evidence and no retention source or second benchmark task is created.

## Architecture

### 1. Execution-only phase telemetry

Add one small, best-effort performance recorder shared by both stages. It uses monotonic clocks, never raises into scientific execution, and writes only performance-sidecar files. Reference execution, normal telemetry, and benchmark timing mode do not introduce CUDA synchronization. Only benchmark `model-replay --model-replay-mode verify` may synchronize, and only at documented exact tensor/logit capture boundaries used to prove scientific equality.

The two fixed sidecar paths describe one **current live invocation**, not accumulated scientific resume state. Completed resumes and Stage-2 `search_complete_pending_strict` resumes do not construct a recorder and leave the existing sidecar byte-for-byte untouched. A genuinely new live invocation starts only after scientific collision/journal validation; its recorder best-effort removes the prior summary and detailed-event files that it owns, starts detailed `event_index` at zero, and publishes only that invocation's aggregate timing. If an old detailed stream cannot be removed, detailed recording is disabled for the invocation and telemetry is marked degraded rather than appending a duplicate zero-based sequence. No prior timing or event row is replayed into scientific state, and no reset failure may change search, persistence, ranking, or resume behavior. Stage-1 assigns an execution-only request index from a dedicated counter at every real adapter entry, including failed requests; the existing scientific evaluation index/count still advances only for successfully constructed evaluations. A runner restores any transient recorder binding before best-effort-closing only a recorder it created, and an unexpected telemetry-close failure may not replace a scientific return or exception.

Stage-1 events record:

- requested action and cache-hit status;
- full versus delta installation;
- changed GELU and Softmax layer counts;
- installation wall time;
- validation forward plus metric-finalization wall time;
- observation callback/persistence wall time.

Stage-2 events record:

- compact action decode;
- the 12 layer decisions and 59 active block decisions;
- logical optimizer evaluations, optimizer-cache hits/misses, and physical replans separately;
- all 60 materialization calls and the terminal write-back/fingerprint phase;
- full install, same-fingerprint install skip, clear, and hard-cleanup counts;
- all three online trial indices, action-keyed seeds, and wall times;
- aggregation and observation callback/persistence wall time.

Search-core events record:

- BO ordered-matrix view, RF fit, pool generation/encoding, prediction, and acquisition/resource ordering;
- Greedy neighborhood generation, duplicate/cache handling, and complete-scan bookkeeping;
- COINN-GA mutation, duplicate/forbidden handling, inference-performing offspring count, and generation bookkeeping.

Telemetry overhead must be at most 3% in the representative reference workload. If detailed recording exceeds that bound, the separate retention task sets `timing_only_event_stream=False`: normal `timing_only` and `optimized` execution keep aggregate counters/timings but write no detailed event JSONL. Benchmark artifacts from the initial run remain immutable evidence of why the stream was disabled.

### 2. Stage-1 comparator-scoped delta installation

All three backends share the same real Stage-1 evaluator and therefore receive the same optimization.

The exact original-function baseline runs before the comparator installation scope. The scope is created lazily only when the search requires a new real evaluation; completed resume and an all-preloaded completion path do not touch model handlers.

Inside the scope:

1. The first live comparator candidate performs a complete install: Softmax degree 6 on every layer and the candidate's full GELU vector.
2. The installed signature records only operations that completed successfully. It contains model identity, handler identity, full GELU vector, and full Softmax vector; the presence of that signature object is the validity token, so there is no second Boolean that can diverge.
3. A later live candidate groups and applies only layers whose requested degree differs from the last successfully installed vector.
4. Because every comparator candidate requests Softmax degree 6, later candidates make no Softmax handler call while the exclusive scope remains valid.
5. A search cache hit performs no install and does not alter the installed signature.
6. Original-function sentinel transitions use the existing per-layer restore methods; approximation transitions use the existing replacement methods and their module-freshness checks.
7. Model or handler replacement, any installation exception, explicit restoration, or loss of exclusive scope invalidates the session signature. The four low-level GELU/Softmax replace/restore entry points clear the handler's composite Stage-1 signature before mutation. The ordinary evaluator installer becomes idempotent only when **both** `LayerImportanceEvaluator._last_applied_config` and `ReversibleLayerHandler._last_stage1_applied_config` equal the requested signature; divergence forces a full repair install. The evaluator/session writes both standard signatures only after a complete install succeeds.
8. The new session signature is committed only after every changed-layer operation succeeds.

Scope teardown independently attempts restoration of original GELU and Softmax functions, clears both standard signatures in all success and failure cases, preserves `Stage1EvalCache`, and propagates the first restoration failure after attempting both families. Clearing the evaluator sentinel is what prevents the current direct installer from returning early; clearing the handler sentinel also preserves session/worker ownership and lets the new composite gate detect direct low-level mutations. A failed close is terminal for that session. If search and close both fail, the search failure remains primary and the close failure is chained rather than silently replacing it.

Production continuation is path-specific rather than an unconditional finalizer:

- fresh or pending-strict canonical sequential Stage-2 calls the shared materialization setup, which installs `fixed_gelu/fixed_softmax` before building probe batches or installing BLB noise;
- completed canonical sequential resume intentionally performs no Stage-1 install and no forward; the selected Stage-1 configuration must instead be installed by the first enabled final-evaluation boundary, or by outer normal completion when final evaluation is skipped/ineligible;
- BLB-v3 non-sequential/single-shot and opt-in substage setup both explicitly install the selected Stage-1 configuration before their first Stage-2 use; this does not refer to historical `legacy_v2`;
- the layerwise sequential cleanup path reapplies selected Stage-1 state from a real `finally`; traditional sequential, non-sequential, substage, Unified final evaluation, Paean final evaluation, and outer evaluator binding retain their checked-in normal-tail or candidate-level cleanup semantics. This optimization does not add a new global guarantee that every exceptional exit leaves the selected Stage-1 configuration installed.

Therefore the comparator scope may restore original functions without weakening Stage-1-to-Stage-2 binding only when teardown clears stale signatures and every forward-capable continuation proves an explicit selected-config install before its first forward. Tests cover fresh, completed-resume, skipped/ineligible final-eval, non-sequential, substage, Unified-final-eval, and Paean continuations, plus direct low-level mutation repair and teardown failures.

The optimization is acceptable only if reference full-layer installation and optimized delta installation produce exactly equal per-batch logits, loss accumulation, final metrics, selected actions, and normalized artifacts for the same actions and seeds.

### 3. Retain the existing validation data path

The evaluator already collates `validation_full` once into an ordered pinned-CPU tuple, moves tensors with `non_blocking=True`, keeps the model on the selected device, and defers GPU-to-CPU synchronization. Keep that production path unchanged.

Do not add device-resident validation batches, alternate-stream prefetch, rebatching, example combination, reordering, dropping, or dynamic padding in this task. Historical exact-path evidence measured only about `57.9µs` per evaluation beyond the pinned tuple, far below a real full-validation forward. This decision may be revisited only if a fresh exact one-card profile overturns that measurement.

### 4. BO-RF ordered CPU workspace

BO remains sequential because every RF fit depends on the complete ordered observation prefix. Optimize allocation and repeated deterministic feature work, not the dependency graph.

- Maintain the ordered feature matrix and multi-output target matrix incrementally as float64 C-contiguous storage.
- Reconstruct that storage exactly once from the durable ordered prefix on resume, exposing only the replay-consumed prefix to each RF fit.
- Cache immutable one-hot feature rows by exact action tuple and reuse the existing Stage-1 simulated-cost cache. Do not add a Stage-2 resource cache in this task; benchmark evidence may justify a separately designed cache later.
- Reuse the selected candidate's pool feature row when appending its observation.
- Bound any pool-derived cache by deterministic capacity. Eviction may affect performance only; it must not consume RNG, reorder iteration, alter cache-visible search state, or change a score.
- Preserve exact row order and exact NumPy values supplied to every RF fit and acquisition calculation.

The shared prediction helper preserves estimator order and reconstructs the complete per-tree prediction tensor before the unchanged NumPy mean/std reduction. Production remains serial with `estimator_workers=1` throughout this task. The benchmark may measure fixed contiguous estimator ranges with more than one CPU worker, but that measurement can report only `eligible_for_future_task`; it cannot enable parallel tree prediction here. Any future production parallelism requires a separate ordinary task and must first prove:

1. the complete per-tree prediction tensor is `np.array_equal` to the serial tensor;
2. every acquisition key, selected pool index, and selected action is identical across fresh and resumed runs; and
3. end-to-end BO wall time improves.

### 5. Stage-2 persistent full apply

The comparator's current `LayerwiseRuntimeEvaluator` clears BLB state before and after every candidate. This disables the production environment's existing `persistent_probe_install` capability on one GPU.

The optimized session wraps the shared `run_search` call, not any backend-specific loop:

1. Require the true single-GPU path (`probe_runner is None`); reject the `optimized` Stage-2 persistent-install route rather than silently changing multi-GPU behavior.
2. Perform one independent hard cleanup at session entry.
3. Enable `persistent_probe_install` only for the online search evaluator.
4. Do not perform per-candidate pre-clear or final-clear in that session.
5. Every candidate still executes all 12 layer decisions, all 59 active block decisions, all 118 logical optimizer evaluations, all 60 materialization calls, terminal optimizer write-back/binding restoration, and fingerprint construction.
6. An optimizer-invalid or materialization-skipped candidate still performs no model forward and remains a durable forbidden observation.
7. If the post-replan fingerprint equals the currently installed fingerprint, skip installation only; run the same three action-keyed trials.
8. If the fingerprint differs, apply the complete materialized configuration through the production bridge. Do not infer changed blocks from the requested six-valued layer action.
9. Hard-clean at session exit before strict validation or any other shared-model consumer begins, restore the prior persistence flag, and invalidate the installed fingerprint.

Strict validation retains its existing independent complete-family installation semantics. Persistent online state never crosses into A/B/C or axis-family evaluation.

No candidate batching, combined trial forward, changed-layer Stage-2 apply, trial-seed merge, noisy-metric cache, device-resident probe-batch rewrite, or new per-block materialization cache is allowed in the initial implementation. The existing deterministic `RescaleOptimizerBridge` LRU remains unchanged; telemetry may justify a separately designed cache later, but this task does not broaden its key or semantics.

### 6. Independent Stage-2 hard cleanup

`BLBNoiseRLBridge.clear()` uses its `_installed` ledger. A handler can mutate part of the model and then raise before the bridge records that block, so ordinary clear alone is not sufficient for persistent reuse.

Add a narrow hard-cleanup operation to `BLBNoiseRLBridge`:

- it does not read `_installed` to decide what to restore;
- it attempts every block restoration in reverse dataflow order: Block5, Block4, Block3, Block2, Block1, first input;
- it targets every layer through the existing restore methods;
- it attempts all restorations even if one fails, then reports the aggregate failure;
- only successful cleanup clears the ledger;
- `BLBStage2Env` always invalidates `_installed_config_fingerprint` when hard cleanup is requested;
- cleanup failure is an infrastructure error and is never converted into an invalid scientific candidate.

Do not use `ReversibleLayerHandler.restore_all()`: it deep-copies and replaces the handler's model reference, which is too broad, expensive, and unsafe for the evaluator's shared model reference.

Bridge apply exceptions, trial exceptions, optimized-session teardown, and resume boundaries invoke hard cleanup. A partial-install regression test must prove that cleanup works even when `_installed` is still empty.

### 7. Greedy and COINN-GA control paths

Greedy and COINN-GA receive speedup only through the shared Stage-1 delta session and Stage-2 persistent full-apply session. Their algorithm loops remain scalar and synchronous. This task introduces no fourth optimization family: no new neighborhood cache, mutation cache, tuple-conversion cache, resource cache, duplicate-membership cache, or other Greedy/GA loop cache. Any such mechanism requires a separately designed ordinary task with its own exact-order and performance evidence.

There is no speculative 2-opt generation, speculative GA generation, candidate batching, or multi-GPU dispatch in this task. Neighborhood and mutation generation order, RNG calls and draw count, existing tie-break traversal, cache-hit versus real-evaluation accounting, the 57-inference complete-generation barrier, callback order, and durable persistence order remain unchanged.

## Failure and recovery semantics

- Performance telemetry is best-effort and cannot block or validate scientific completion.
- Infrastructure failures are not converted into invalid scientific candidates.
- A failed Stage-1 delta install invalidates the execution signature, restores the original functions, and preserves the existing exception behavior.
- A failed Stage-2 install or trial invokes independent hard cleanup, invalidates the fingerprint, and preserves the existing infrastructure-error behavior.
- Durable observations remain the authoritative resume prefix. Unpersisted work may be recomputed; persisted work receives zero new forwards.
- Existing append, flush, file-`fsync`, atomic-publication, torn-tail repair, and corruption-rejection behavior is not redesigned or batched in this performance task.
- Resume reconstructs BO workspace and action caches only from the durable ordered prefix without extra RNG draws.
- Execution-profile differences are written to the performance sidecar and excluded from backend/seed/action/scientific configuration equality.
- The reference path remains available for A/B verification and diagnosis.

## Verification strategy

### Torch-free control equivalence

For Stage-1 and Stage-2 search cores, compare `reference` and `optimized` for:

- complete initial-design order;
- BO pool actions, RF seeds, feature/target matrices, per-tree predictions, acquisition keys, selected indices/actions, patience, and termination;
- Greedy requested/cache-hit sequence, neighborhood completion, accepted moves, restarts, and local-optimum proof;
- COINN-GA parents, mutations, duplicate/forbidden handling, 57-offspring generation barriers, populations, elites, incumbent, stagnation, and termination;
- observation/inference counters and persistence callback order;
- fresh, interrupted, resumed, and completed-resume runs.

### Model-level equivalence

- Stage-1 first full install followed by fixed-Softmax and changed-GELU delta installs.
- Stage-1 cache hit between two live candidates.
- Stage-1 partial-install exception followed by full repair.
- Stage-1 scope restoration followed by selected Stage-1 reinstallation before Stage-2.
- Stage-2 reference clear/reinstall versus persistent full apply.
- Repeated fingerprints, one-layer and two-layer action changes, valid-invalid-valid sequences, and injected partial-install failure.
- Exact logical optimizer/materialization counts, optimizer request/result payloads, final fingerprints, action/trial seeds, per-trial arrays, aggregate metrics, priority/rank, and selected action.
- Hard cleanup leaves no active BLB hooks, wrappers, cfg maps, bridge ledger entries, or installed fingerprint.

### Resume and artifact equivalence

- Crash before and after ordered observation append/fsync.
- Truncated final JSONL repair and middle-corruption rejection.
- Resume reconstructs BO ordered matrices and caches without changing the next pool or selected action.
- Completed resume constructs no live execution session and performs zero model forwards.
- Reference and optimized artifacts are byte-identical after removing declared performance-only files and existing nondeterministic timing fields.

## Benchmark and retention gates

Run benchmarks on one visible GPU from an exact canonical commit/tree. Capture the `reference` baseline first, then run the initial all-enabled candidate routes under `optimized`; no mechanism is called retained until the report applies its independent retention gate. Real-model `reference`, `timing_only`, and `optimized` replays run in separate processes so each profile owns an independent model, handler, CUDA state, and evaluator cache. Search-core generates the immutable action trace once; model replay verifies or times only those recorded actions and never generates replacements. Model verification covers the complete action prefix later used for timing: the first eight Stage-1 actions and first four Stage-2 actions per backend, with the same adjacent Stage-2 repeat expansion. Timing may repeat that already-verified workload five times, but it may not time a longer or different action prefix. Search-core, model-verification, and model-timing reports each pass an immediate exact gate before the next phase starts. The final retention report merges those exact-gated suites and requires the benchmark source receipt as an input; receipt schema and six-field identity are validated against all suites before any timing, GPU, phase, or retention payload is opened. BO ordered-workspace inputs come only from search-core, Stage-1 delta-install and Stage-2 persistent-install inputs come only from model replay, and telemetry overhead is computed from explicitly matched workload cells. No suite fabricates a phase that it did not execute.

A benchmark observation limit is only a stop request; it is not permitted to interrupt an algorithm at an incoherent point. BO-RF honors it only after the complete initial design or after a later selected observation plus incumbent/patience bookkeeping, COINN-GA honors it only after a complete initial population or a complete 57-inference generation plus population/elite/stagnation/history/RNG bookkeeping, and Greedy honors it only at its natural verified-local-optimum boundary after the full neighborhood scan. The callback receives a recursively owned backend snapshot; it cannot mutate live scientific history, population, elite, incumbent, or RNG-state containers. The benchmark callback never raises from the durable per-observation callback. It returns through the ordinary result-construction path with a benchmark-only termination reason, while production callers pass no benchmark callback. Bounded model replay takes the first `N` actions in immutable trace order, preserves duplicates, and never sorts, samples, or substitutes actions; Stage-2 repeat requests are adjacent copies of each selected action.

Report for every backend and stage:

- cold and warm median/p90/p99 candidate latency;
- phase fractions;
- candidates or real inferences per hour;
- GPU utilization and peak memory;
- model install, install-skip, clear, hard-cleanup, and cache-hit counts;
- BO RF fit/predict/pool time by observation-prefix size;
- end-to-end wall time for identical bounded workloads;
- normalized scientific-artifact equality.

Retention requirements:

- every scientific field, candidate order, selected action, trial seed/array, and final fingerprint is exact for both `reference == timing_only` and `reference == optimized`;
- Stage-2 retains 118 logical optimizer evaluations, 60 materializations, and three trial boundaries per valid candidate;
- each Stage-2 observation records the ordered optimizer accounting row, with `logical_optimizer_evaluations = optimizer_cache_hits + optimizer_cache_misses` and `physical_replans = optimizer_cache_misses`; reference and optimized rows must be exact because this task adds no Stage-2 optimizer/resource cache;
- invalid/materialization-skipped candidates perform no forward;
- every accepted real evaluation produces exactly one durable observation callback;
- hard cleanup leaves no residual Stage-2 mutation;
- Stage-1 delta installation independently improves its targeted install phase by at least 10%, or the representative Stage-1 end-to-end workload by at least 5%;
- the BO ordered workspace independently improves its targeted CPU phase by at least 10%, or the representative BO end-to-end workload by at least 5%;
- Stage-2 persistent warm lifecycle improves its targeted install/clear phase by at least 25%, and end-to-end wall time does not regress by more than 2%;
- detailed profiler overhead is at most 3%.

An optimization that fails exact equivalence is removed. An optimization that is exact but misses its performance gate remains disabled and is documented rather than forced into production. "Removed" here means its execution-only retention boolean is set to false in a separate TDD source task; the tested implementation may remain available behind the false route for reproducibility and later diagnosis. A disabled Stage-1/BO/Stage-2 mechanism must route `optimized` to `reference`, remain exact, and add no more than 2% end-to-end overhead in the final replay. Disabled detailed telemetry must route to aggregate-only recording and remain within 3% overhead.

The report schema records, for every mechanism, `configured`, `execution_route`, exactness, measured improvements/regression, `keep`, and final `acceptance`. The environment section records exactly one visible GPU, production estimator workers equal to one, GPU sampling evidence, and the four-value static retention policy. Initial retention is recomputed independently from raw report fields before any source route changes. Final acceptance is recomputed independently after any changed policy is aggregated and rerun. For a mechanism spanning multiple backend/stage cells, each repetition first sums the relevant phase or end-to-end seconds across those cells; the report then takes the median of those repetition totals. It never averages already-computed percentages.

The report exposes those aggregation inputs under `retention_inputs` as ordered per-repetition totals, not percentages or precomputed verdicts. `stage1_delta_install` and `bo_ordered_workspace` each carry reference/candidate targeted-phase seconds and reference/candidate end-to-end seconds; `stage2_persistent_install` carries reference/candidate install-plus-clear seconds and reference/candidate end-to-end seconds; `telemetry` carries reference and timing-only end-to-end seconds. Each array element is already the sum of every applicable backend/stage cell in that repetition. Independent consumers take the median of each array, recompute the improvement or regression from those medians, then apply the locked thresholds. They never trust `keep`, `acceptance`, or an already-calculated percentage as the authority.

Every real Stage-2 model-replay report cell exposes ordered `operation_counts_per_valid_candidate` and `operation_counts_per_invalid_candidate` arrays for every compared profile. A valid row contains all six counters (`logical_optimizer_evaluations`, optimizer cache hits/misses, `physical_replans`, `materialization_calls`, and `online_trial_boundaries`) and satisfies the 118/60/3 plus accounting invariants. Under the current production chain, a normally returned optimizer-invalid or materialization-skipped row still completes all 118 logical optimizer evaluations and all 60 materialization boundaries before it is classified, but records `online_trial_boundaries == 0`; therefore its fixed contract is 118/60/0. A smaller normally returned prefix is an infrastructure-shape failure, not another scientific candidate class. The ordered valid and invalid row arrays must match exactly across `reference`, `timing_only`, and `optimized`. Search-core is intentionally torch-free and performs no production replan, materialization, or trial; therefore both operation-count arrays are `null` for both Stage-1 and Stage-2 search-core cells. Search-core must never synthesize production operation rows.

Exactness is a hard read barrier. On any candidate/RNG/callback/artifact/operation-row or source-receipt identity mismatch, the report sets `status="failed_exact_equivalence"`, `exact.overall=false`, and `retention`, `retention_inputs`, `phase_statistics`, every latency/throughput field, and every retention decision to `null`. No consumer may inspect or publish performance conclusions from that report. Compact failure-evidence collection is non-throwing for missing, truncated, malformed, wrong-workload, or structurally invalid reports: it retains path/hash/parse diagnostics and never turns a report failure into missing failure evidence.

## Git and server boundary

1. Implement and verify locally on the initial ordinary source task branch.
2. Commit and push source changes to that branch only. The source task does not modify `SERVER_COMMAND.md`.
3. Before marking any ordinary task completed, run its required project gates on the server from an isolated Git checkout of the exact remotely fetchable task source commit/tree. The authorized invocation supplies those reviewed commit/tree/branch literals; it never derives the expected identity from the remote being checked. Bring back one structured receipt plus its complete log, recompute the log hash locally, refresh all remote heads, and record any canonical movement.
4. Publish a completed handoff-only tip after that server evidence; do not advance `jk_standard_rl`.
5. Only an explicit aggregator may advance canonical, and only after assembling the required all-head manifest and completing exact local/server candidate verification. Every aggregate uses a fresh remotely absent branch name so the aggregate branch itself cannot be hidden from the all-head review.
6. Re-run local sync guards and prove the exact source canonical commit/tree.
7. Publish any benchmark command only through a later bridge-control ordinary task that changes `SERVER_COMMAND.md` and its own handoff metadata, then server-verify, complete, and aggregate that control task through the same boundary.
8. The server obtains source only through Git at the exact pre-authorized benchmark canonical commit/tree; it never receives a direct source patch. Benchmark launch commands compare the remote canonical tip to the supplied literals before both synchronizing and rechecking.
9. Run the initial one-GPU `reference`/`timing_only`/`optimized` benchmark and publish its immutable trace bundle through one compact result-only branch. Recover the result from the exact result tip, prove changed-path scope, and bind run source, performance source, report, raw retention inputs, trace bundle, hashes, branch, and tip in one verified receipt.
10. Prove exact equivalence before reading latency, then mechanically recompute every retention formula from the ordered raw `retention_inputs` arrays.
11. If any retention boolean is false, change only `ComparatorRetentionPolicy` through a separate ordinary RED/GREEN source task, exact task-source server gate, completed handoff, all-head aggregate, and exact canonical gate. The production profile default remains `reference`.
12. For a changed policy, publish a separate final bridge-control ordinary task, rerun search-core into a new output without modifying the immutable initial trace bundle, replay the pinned initial model trace on the exact adjusted canonical source, recover exact final evidence, and retire the bridge through another control-only task. If no policy changed, the initial run is final evidence.
13. Keep four identities distinct: `FINAL_PERFORMANCE_SOURCE_*` is the retained-policy implementation actually measured; `FINAL_CONTROL_SOURCE_*` is the active bridge/control canonical; `FINAL_RESULT_SOURCE_*` is the exact canonical that launched the result; `FINAL_CANONICAL_*` is the bridge-closed inert canonical. On the no-change route the performance source may be an ancestor of the final canonical.
14. Every bridge retirement follows ordinary server gate, handoff, and explicit aggregate protocol; source tasks never smuggle bridge edits into their tips. After the final inert canonical advances, synchronize the server to that exact commit/tree and verify the command is inert.
15. Finish with a fresh canonical test gate, a structured final-server-inert receipt, exact Git-tip-derived result/source receipts, result-only changed-path proof, and a final assertion that the production default remains `reference`.

No password, SSH credential, or token is written to repository files, handoffs, logs, prompts, or performance artifacts.
