# ADR-010: Step-1 ×15 SF grid + direct-replan fast enumeration

Date: 2026-06-11
Status: Accepted (user spec)
Supersedes: the SF-grid portion of ADR-008's decode notes (hybrid 2/1 ×10,
2026-06-04) and closes the grid churn recorded in ADR-009 D3 (uniform-2 /
floor-12, withdrawn same day).

## Context

The user requires the fusion-count enumeration to cover the FULL integer SF
range per slot — uniform step-1 spacing, up to 15 levels, no floor beyond the
physical noise-table minimum, exhaustively enumerated (no probes/shortcuts).
That multiplies the cartesian products to ~3–6×10⁹ combos total (block4
≤4.9e9, block5_n4 ≈9.1e8, block2 ≤2.1e8 — projected exactly from the
committed maps' baselines), which is infeasible on the golden per-combo
pipeline (~0.6–1 ms) but fine if each combo costs little more than the bare
replan call (~0.06 ms measured).

Profiling facts: the training/builder replan interaction was ALREADY
in-process (`InProcessInvoker.replan_variables → ReplanSession.replan`; the
JSON-file layer exists only in the debug `SubprocessInvoker`). The golden
per-combo cost was dominated by intermediates around that call: cfg-object
construction (`action_vector_to_cfgs`), the bridge's per-call cache key +
`copy.deepcopy` store (an enumeration's combos are all distinct, so that
cache can never hit), `RescaleOptimizerOutput` wrapping, optimizer-output→cfg
write-back + `sync_block*`, and cfg introspection for installed points.

GPU batching was evaluated and rejected: `ReplanSession` is pure-Python
integer modulus-chain planning (sequential algorithm). Running it on CUDA
means writing a second implementation of the cost source of truth, which the
project's correctness rule forbids (every cost number must come from real
replan). The parallel axis is CPU processes.

## Decision

1. **Grid** (`action_space`): `LEVELS_F=W=MS=R=15`; `sf_from` = baseline −
   dist (full integer range `[baseline-14, baseline]`); no floor;
   `_snap_to_table` still bounds at the table min 10 and
   `distinct_sf_level_indices` keeps one (lowest) index per decoded value, so
   snap-duplicates are never enumerated. `option0 == baseline` holds for any
   dataset/baseline. All committed fusion maps are stale and rebuilt under
   this grid (`block5_n0.json` stays dormant-stale until degree-0 returns).
2. **Direct-replan fast path** (`blb_stage2_rl/fusion_enum_fast.py`; builder
   default `--enum-path fast`): per block-type a TEMPLATE is derived once in
   the main process from the golden machinery itself — each enum slot probed
   through the golden decode at two levels, the resulting
   `(t_new, delta_overrides)` diffs must be an exact identity map of the
   decoded SF; Q/K-style mirrors discovered by sentinel probes through the
   real `sync_block*` functions; installed-point specs classified as
   source / rescale (fused-away = absent) / encode (settled
   `propagation_deltas`) / slot / const, with rotations taken from
   `effective_rotations` exactly as the golden helper does. The torch-free
   hot loop then patches the two replan inputs per combo, calls
   `ReplanSession.replan` (same function, same `DEFAULT_FUSION_POLICY`), and
   assembles points straight from the raw output: **0.099 ms/combo measured**
   (~10× over golden). Workers consume contiguous combo-rank ranges via
   mixed-radix unranking (no `i % shards` skip-spinning, which costs real
   minutes per worker at 1e9+ combos) and report progress/ETA.
3. **Correctness gates (mandatory, layered)** — the acceleration may never
   change an answer:
   * template derivation raises on any wiring that is not an exact identity
     map, on inconsistent wiring across levels, or on cfg point-layout drift;
   * `verify_template` evaluates baseline + all-min corner + N random combos
     (server command: 128) through BOTH paths and requires exact equality of
     (valid, fusion_count, total_bits, installed signature, total variance);
     any mismatch aborts the build;
   * `--enum-path both` runs golden AND fast to completion and requires the
     final option lists to match item-for-item (variance compared with
     float-sum-order tolerance only) — applied to the small block-types
     (block1, block5_n1) on every rebuild;
   * the streaming per-fusion-count min-variance reducer is shared with the
     golden shards, and range-vs-stride sharding exactness is unit-locked.
4. **Throughput**: builder workers default to `nproc-1` (the old SERVER
   command's 16-worker cap was an accident and is removed; cap 128). The
   fast workers import only `fusion_enum_fast` + `ReplanSession` — no torch.

## Consequences

* Full step-1 ×15 enumeration of all 6 block-types lands in roughly an hour
  on a many-core box instead of 10–30+ hours.
* The fast path is structurally incapable of silent drift: its wiring comes
  from the golden code, and every build re-proves equivalence empirically
  before and during enumeration.
* Future grid changes remain cheap (the template re-derives itself), but any
  change to `sf_from` / level counts still invalidates committed maps.
* `tests/test_blb_fusion_enum_fast.py` locks the unranking/odometer ordering,
  shard-union exactness, and live-session hot-loop behavior (the in-repo
  Rescale_optimizer is torch-free, so these run locally and in CI).
