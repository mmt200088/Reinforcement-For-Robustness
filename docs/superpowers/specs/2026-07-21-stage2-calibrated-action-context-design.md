# Stage-2 Calibrated Action Context Design

## Goal

Make Stage-2 training, Paean final evaluation, action-grid loading, human slot
export, and GLUE submission decode the same action against the same calibrated
Rescale Optimizer baseline. A flat action vector or fusion-count option must
produce the same final per-layer configuration in every runtime surface.

## Root Cause

The production RL path derives a per-layer `MaxSFsTable` from the selected
Stage-1 GELU/Softmax degrees and `static_skeletons_<dataset>.json`. Paean,
action-grid loading, and GLUE submission still call `load_max_sfs(profile)` and
cache only by profile. That generic table lacks the per-layer static-skeleton
calibration, so identical action indices can decode to different SF values.

## Shared Context

Add one immutable calibrated action context in `baseline_bootstrap.py`. Its
inputs are:

- Rescale Optimizer root;
- dataset/profile;
- layer count;
- every layer's GELU degree;
- every layer's Softmax degree;
- whether SF values are snapped to the noise table.

It reuses `load_static_skeletons_baseline` and
`static_skeletons_baseline_to_action` and returns the baseline object,
calibrated baseline action vector, calibrated `MaxSFsTable`, cost statistics,
diagnostics, and provenance. Exact replay uses
`snap_sf_to_noise_table=False`, matching the production Stage-2 runner.

## Data Flow

1. Resolve Stage-1 degrees before loading any Stage-2 action configuration.
2. Build the calibrated context once from the static-skeleton archive.
3. Pass its `max_sfs` explicitly into action-grid slot decoding, candidate
   decoding, human slot export, final evaluation, and GLUE submission.
4. Decode fusion metadata, apply precision-boost overrides, run replan, write
   optimizer outputs back, and install the resulting configuration exactly as
   before.
5. Persist the static-skeleton archive path and calibrated-context fingerprint
   with evaluation/submission evidence.

No replan, reward, action-space, fusion-map, noise, or truncation semantics are
changed.

## Compatibility And Failure Handling

- Exact Stage-2 replay must fail when the static-skeleton archive or selected
  Stage-1 degrees are unavailable. It must not silently fall back to the
  profile-only table.
- Low-level generic action utilities may retain explicit legacy behavior for
  callers that deliberately pass `load_max_sfs(profile)`, but Paean and GLUE
  entrypoints must use the calibrated context.
- Context caches must include profile, layer count, GELU degrees, Softmax
  degrees, Rescale Optimizer root, and archive identity. A profile-only cache
  is invalid.
- Existing fusion maps do not require rebuilding when their stored provenance
  matches the same static-skeleton baseline; the fix aligns consumers with the
  map-building path.

## Verification

- Red test: profile-only Paean decoding differs from the calibrated MRPC
  Block3 baseline.
- Unit test: context construction reproduces the calibrated baseline action and
  per-layer `max_sfs` used by the production runner.
- Unit test: Paean candidate decode and human slot export receive the same
  calibrated table.
- Unit test: action-grid slot-form loading accepts an explicitly supplied
  calibrated table and does not call its profile-only loader.
- Unit test: GLUE submission passes the same calibrated table through slot
  loading and model configuration decode.
- Integration gate: production RL, Paean, and GLUE replay produce identical
  per-layer/block configuration fingerprints, including MRPC Block3
  `x_fresh=31`, `inv_2n=15`, six square-rescale SFs of `35`, and the selected
  truncation K.

