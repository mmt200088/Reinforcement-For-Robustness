# Comparator Single-GPU Optimization Design

## Status and scope

This design covers one-card execution optimization for all three two-stage comparator backends:

- `bo_rf`
- `greedy`
- `coinn_ga`

The work starts from canonical commit `5f8441a8d30cd5cf9fac871d4240f1de61328b97`, whose tree `95c0fd1b1cb9d5448f29d8ad7a181ffee6108253` contains the completed comparator-correctness task. Implementation occurs only on ordinary task branch `codex/task-comparator-single-gpu-optimization-20260810`. COINN-GA candidate-level multi-GPU execution remains explicitly deferred.

The task changes execution mechanics only. It must not change any scientific search configuration, action space, candidate order, random-number consumption, model input, optimizer request, trial seed, persistence order, stopping rule, strict-validation rule, or final-selection rule.

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

- `reference`: current full reinstall/clear behavior and current BO matrix construction.
- `timing_only`: reference behavior plus detailed phase telemetry.
- `optimized`: Stage-1 delta installation, retained BO CPU optimizations, and any Stage-2 persistent-install optimization that has passed its retention gate.

The selector is threaded explicitly through the Stage-1 and Stage-2 comparator runners; it is not inferred from an environment variable. It is not a scientific parameter and must not enter scientific manifests or resume equality. The selected profile, implementation revision, Python version, NumPy version, scikit-learn version, and joblib version are written only to a performance sidecar so exact-sequence claims are scoped to a recorded numerical environment. The production default changes from `reference` to `optimized` only after exact-equivalence tests and the canonical one-GPU benchmark pass. `reference` remains available for diagnosis and future regression checks.

## Architecture

### 1. Execution-only phase telemetry

Add one small, best-effort performance recorder shared by both stages. It uses monotonic clocks, never raises into scientific execution, and writes only performance-sidecar files. Normal timing does not introduce CUDA synchronization. Explicit benchmark mode may synchronize only at a documented measurement boundary.

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

Telemetry overhead must be at most 3% in the representative reference workload. If detailed recording exceeds that bound, detailed events remain benchmark-only and normal optimized execution keeps counters only.

### 2. Stage-1 comparator-scoped delta installation

All three backends share the same real Stage-1 evaluator and therefore receive the same optimization.

The exact original-function baseline runs before the comparator installation scope. The scope is created lazily only when the search requires a new real evaluation; completed resume and an all-preloaded completion path do not touch model handlers.

Inside the scope:

1. The first live comparator candidate performs a complete install: Softmax degree 6 on every layer and the candidate's full GELU vector.
2. The installed signature records only operations that completed successfully. It contains model identity, handler identity, full GELU vector, full Softmax vector, and a validity token.
3. A later live candidate groups and applies only layers whose requested degree differs from the last successfully installed vector.
4. Because every comparator candidate requests Softmax degree 6, later candidates make no Softmax handler call while the exclusive scope remains valid.
5. A search cache hit performs no install and does not alter the installed signature.
6. Original-function sentinel transitions use the existing per-layer restore methods; approximation transitions use the existing replacement methods and their module-freshness checks.
7. Model or handler replacement, any installation exception, explicit restoration, or loss of exclusive scope invalidates the signature. The next live evaluation performs a complete repair install.
8. The new signature is committed only after every changed-layer operation succeeds.

Scope teardown restores the original GELU and Softmax functions and invalidates both evaluator and handler configuration caches. Cache invalidation is mandatory: the selected comparator action is often the last evaluated action, and leaving its signature cached after restoration would make the subsequent Stage-2 `apply_configuration` incorrectly return early.

The existing Stage-2 lifecycle then deterministically installs the selected Stage-1 configuration again:

- the shared sequential materialization setup calls `ev.apply_configuration(fixed_gelu, fixed_softmax)` before constructing the Stage-2 environment;
- the legacy single-shot Stage-2 setup does the same;
- Stage-2 teardown and the outer evaluator finalization reapply the selected configuration again.

Therefore the comparator scope may cleanly restore original functions without weakening Stage-1-to-Stage-2 binding, provided teardown clears stale signatures. Tests must cover this exact restoration-then-selected-reinstall sequence.

The optimization is acceptable only if reference full-layer installation and optimized delta installation produce exactly equal per-batch logits, loss accumulation, final metrics, selected actions, and normalized artifacts for the same actions and seeds.

### 3. Retain the existing validation data path

The evaluator already collates `validation_full` once into an ordered pinned-CPU tuple, moves tensors with `non_blocking=True`, keeps the model on the selected device, and defers GPU-to-CPU synchronization. Keep that production path unchanged.

Do not add device-resident validation batches, alternate-stream prefetch, rebatching, example combination, reordering, dropping, or dynamic padding in this task. Historical exact-path evidence measured only about `57.9µs` per evaluation beyond the pinned tuple, far below a real full-validation forward. This decision may be revisited only if a fresh exact one-card profile overturns that measurement.

### 4. BO-RF ordered CPU workspace

BO remains sequential because every RF fit depends on the complete ordered observation prefix. Optimize allocation and repeated deterministic feature work, not the dependency graph.

- Maintain the ordered feature matrix and multi-output target matrix incrementally as float64 C-contiguous storage.
- Reconstruct that storage exactly once from the durable ordered prefix on resume, exposing only the replay-consumed prefix to each RF fit.
- Cache immutable one-hot feature rows by exact action tuple. Reuse the existing Stage-1 simulated-cost cache and add a Stage-2 exact-action resource cache only if profiling shows repeated pure resource computation is material.
- Reuse the selected candidate's pool feature row when appending its observation.
- Bound any pool-derived cache by deterministic capacity. Eviction may affect performance only; it must not consume RNG, reorder iteration, alter cache-visible search state, or change a score.
- Preserve exact row order and exact NumPy values supplied to every RF fit and acquisition calculation.

Ordered per-tree prediction is an optional retained sub-optimization. Fixed contiguous estimator ranges may be evaluated concurrently on CPU, but the result tensor must be restored to original `estimators_` order before the unchanged NumPy mean/std reduction. It is retained only if:

1. the complete per-tree prediction tensor is `np.array_equal` to the reference tensor;
2. every acquisition key, selected pool index, and selected action is identical across fresh and resumed runs; and
3. end-to-end BO wall time improves.

Otherwise the serial ordered prediction path remains canonical.

### 5. Stage-2 persistent full apply

The comparator's current `LayerwiseRuntimeEvaluator` clears BLB state before and after every candidate. This disables the production environment's existing `persistent_probe_install` capability on one GPU.

The optimized session wraps the shared `run_search` call, not any backend-specific loop:

1. Require the true single-GPU path (`probe_runner is None`); reject the persistent profile rather than silently changing multi-GPU behavior.
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

Greedy and COINN-GA receive their primary speedup through the shared Stage-1 delta session and Stage-2 persistent full-apply session. Their algorithm loops remain scalar and synchronous.

Torch-free profiling may justify exact immutable-action caches for neighborhood/mutation encoding, tuple conversion, deterministic resource lookup, or duplicate membership. Such a cache is retained only when measurable and must be keyed by the complete immutable action. It must not change:

- neighborhood or mutation generation order;
- RNG calls or draw count;
- set/dictionary traversal used by an existing tie-break;
- cache-hit versus real-evaluation accounting;
- the 57-inference complete-generation barrier;
- persistence callback order.

There is no speculative 2-opt generation, speculative GA generation, candidate batching, or multi-GPU dispatch in this task.

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

Run benchmarks on one visible GPU from an exact canonical commit/tree. Establish a `reference` baseline before enabling each retained optimization.

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

- every scientific field, candidate order, selected action, trial seed/array, and final fingerprint is exact;
- Stage-2 retains 118 logical optimizer evaluations, 60 materializations, and three trial boundaries per valid candidate;
- invalid/materialization-skipped candidates perform no forward;
- every accepted real evaluation produces exactly one durable observation callback;
- hard cleanup leaves no residual Stage-2 mutation;
- Stage-1 delta installation or BO CPU work improves its targeted phase by at least 10%, or the representative end-to-end workload by at least 5%;
- Stage-2 persistent warm lifecycle improves its targeted install/clear phase by at least 25%, and end-to-end wall time does not regress by more than 2%;
- detailed profiler overhead is at most 3%.

An optimization that fails exact equivalence is removed. An optimization that is exact but misses its performance gate remains disabled and is documented rather than forced into production.

## Git and server boundary

1. Implement and verify locally on the ordinary task branch.
2. Commit and push source changes to that branch only.
3. Publish a normal task handoff; do not advance `jk_standard_rl`.
4. An explicit aggregator advances canonical.
5. Re-run local sync guards and prove exact canonical commit/tree.
6. The server obtains source only through Git at that canonical commit/tree.
7. Run one-GPU reference/optimized benchmarks and backend acceptance runs.
8. If a retention decision changes production defaults, make that source change locally in an ordinary task branch, aggregate it separately, and repeat exact parity before the final server run.

No password, SSH credential, or token is written to repository files, handoffs, logs, prompts, or performance artifacts.
