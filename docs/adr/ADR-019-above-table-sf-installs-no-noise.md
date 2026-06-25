# ADR-019: A scaling factor above the noise-table max installs NO noise (open the ≤46 precision-boost cap to q_max)

- **Status**: Accepted
- **Date**: 2026-06-25
- **Supersedes**: the "≤46 install cap" of the phase-2 precision-boost design
  (`docs/superpowers/specs/2026-06-23-stage2-precision-boost-phase2-design.md` §3),
  which is now a noise floor, not a hard install limit.

## Context

The noise model (`function_handler.NOISE_VARIANCE_TABLE_BY_N`, mirrored torch-free in
`noise_tables.NOISE_VARIANCE_TABLE_BY_N`) is tabulated for scaling factors in `[10, 46]`.
Both the model-forward install (`get_input_noise_variance_by_N`) and the min-noise metric
(`noise_tables.variance`) **raised** for any SF outside the table.

The phase-2 precision boost ("二阶段加大精度") raises a block's output scale to its
config-derived ceiling. For **block4**, doing so the way the user specified — drop the final
encode (the last `1/d`, `ln_var_inv_d`) toward the floor 15 and shift the freed precision onto
the output rescale's `sf_post` — requires raising the *only* node that feeds the output
rescale's pre-scale, `ln_mean_rescale` (already at 45), to 49. Under the ≤46 cap that was
**uninstallable**, so block4's min-noise composition was forced onto the final-encode-**increase**
route (`1/d` 20→21) and the user's intended decrease never happened. Similarly **block5_n1**'s
output (a single rescale, no final encode to split) was clamped from its config ceiling 48 down
to 46.

The user observed the block4 `1/d`=21 and asked whether the decrease logic was missing. It was
not — it is implemented and general (verified with the cap lifted) — it was structurally blocked
by the ≤46 cap. The user then proposed the fix below.

## Decision

**A scaling factor ABOVE the noise-table max installs no measurable noise — treat it as 0
(no noise) instead of an error — and raise the precision-boost install limit from 46 to the
modulus limit q_max (60).**

Rationale (the user's, verified): `var(46) ≈ 2.8e-25` and each `+1` SF is `×0.25`, so
`var(49) ≈ 4.3e-27` — far below fp64 precision (`1e-16`), let alone fp32. A point at SF>46 is
already noiseless to machine precision, so injecting nothing is exact, not an approximation that
loses anything. Points in `(46, q_max]` install no noise; only `> q_max` is a real modulus
violation (also rejected by `replan`).

This is the smallest change that lets the precision boost reach its true optimum: with it, the
block4 min-noise winner **is** the decrease route (`1/d`=15, `sf_post`=38, `ln_mean_rescale`=49),
and its summed installed variance (1.95e-05) is **lower** than the cap-blocked increase route
(5.26e-05) — because the decrease forces raising the dominant high-noise upstream encodes
(`sf`=13, `var`≈2e-5 each) far more than the negligible noise added at `1/d`=15. So the user's
intuition holds AND the result is strictly more faithful.

## What changed

- `function_handler.get_input_noise_variance_by_N`: SF **above** the table max returns `0.0`
  (no noise) instead of raising. SF **below** the min still raises (that high-noise regime must
  be snapped to the table min upstream, never silently dropped).
- `noise_tables.variance`: same — SF above the table max returns `0.0`; below-min still raises.
- `precision_boost`: the phase-2 install limit (`generate_phase2_candidates.max_installed_sf`,
  `effective_output_target.max_installed_sf`, `boost_option_phase2`) defaults to `DEFAULT_Q_MAX`
  (60), not `MAX_ENCODE_SF` (46). `effective_output_target` therefore clamps only at q_max, so
  block5_n1 reaches 48.
- `fusion_enum.boost_options_for_block` build guard: the over-cap abort uses `> topo.q_max`
  (60), not `> 46`.
- SERVER_COMMAND phase-2 gate: TARGETS `block5_n1` 46→48; the map-content over-cap check uses
  `> topo.q_max`.

**Not changed (deliberately scoped):** phase-1 (`generate_candidates`) keeps its `MAX_ENCODE_SF`
cap — its installed points sit far below 46, so the cap never binds; opening it would change
phase-1 results for every block with no benefit. `MAX_ENCODE_SF` (46) remains as the *noise
floor* constant (the table max), used by phase-1 and by the variance lookups' branch point.

## Realized phase-2 outputs (mrpc, local real-replan)

| block | output | `1/d` / notes |
|-------|--------|---------------|
| block2 | 46 | unchanged (no >46 point) |
| block4 | 53 | **`1/d` 20→15**, `sf_post`→38, `ln_mean_rescale`→49 (>46 = no noise); lower total noise |
| block5_n1 | 31→**48** | reaches full config ceiling (was clamped to 46) |
| block5_n2 | 43 | unchanged |
| block5_n4 | 43 | unchanged (middle prime 31 kept) |

## Consequences

- **Determinism preserved**: `>46 → 0` is a pure deterministic function, so 1==N and the
  device-/worker-count invariance gate are unaffected.
- **Maps must be rebuilt**: re-apply phase-2 on the server (`blb_apply_precision_boost.py`); the
  new boosted options carry >46 points which the runtime now installs as no-noise. An in-flight
  run keeps its old maps in memory — restart to pick up the new ones.
- The model forward no longer guards against >46 as a sanity check; this is intentional —
  `replan`/q_max bounds the chain at 60, and a >46 point is always an intentional
  high-precision (noiseless) point.
- Tests: the ≤46-cap tests are rewritten to the new behavior
  (`test_block4_decrease_available_under_default_cap`,
  `test_block4_user_special_case_now_installable`,
  `EffectiveTargetTest`, `test_block5_n1_phase2_reaches_48`,
  `AboveTableNoiseSemanticsTest`, `FunctionHandlerAboveTableTest`).
