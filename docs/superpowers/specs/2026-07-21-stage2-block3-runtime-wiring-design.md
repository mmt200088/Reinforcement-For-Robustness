# Stage-2 Block3 Runtime Wiring Design

## Goal

Make every Stage-2 Block3 truncation-K action affect the actual plaintext MPC
simulation while keeping all Block3 fusion/SF choices permanently fixed to the
Rescale Optimizer baseline.

## Locked Semantics

- Block3 exposes no fusion-count action and no SF action.
- Every Block3 fresh/encode/rescale SF comes from the existing
  `static_skeletons`/Rescale Optimizer baseline loader. RL must not overwrite
  these values.
- Each layer keeps one learnable Block3 `output_truncation_k`, using the same
  `K_LEVELS`, action-matrix encoding, reporting, and reward accounting as the
  other blocks.
- Rescale Optimizer does not optimize K. It receives the Block3 configuration
  so it can validate and replan the fixed baseline SF chain; K passes through
  unchanged.
- The optimizer write-back mutates only SF/rescale fields. The selected K must
  remain unchanged before and after write-back.
- `BLBNoiseRLBridge.apply()` installs the final Block3 configuration through
  `replace_layer_block3_noise(cfg_per_layer=...)`, exactly as Block5 handles
  degree-dependent per-layer configurations.
- The installed Block3 approximation executes `_apply_truncation` after the
  polynomial exponential, so changing only Block3 K changes actual model
  inference.
- Training, multi-GPU probes, promotion/final revalidation, fixed-action final
  evaluation, and GLUE generation must continue to use the shared decode,
  optimizer-write-back, and bridge-install path.

## Data Flow

1. The layerwise policy selects Block3 K in slot 3 of each layer action.
2. `apply_layer_action` writes only that K index into Block3's legacy vector
   slice; the remaining Block3 indices stay at the RO baseline.
3. `action_vector_to_cfgs` builds `Block3NoiseConfig` from baseline SF values
   plus the selected K.
4. `build_optimizer_requests` includes Block3 instead of discarding it.
5. Rescale Optimizer evaluates/replans the Block3 SF chain.
6. `apply_optimizer_outputs_to_cfgs` writes final SFs back into the same
   `Block3NoiseConfig`, preserving K.
7. `BLBNoiseRLBridge.apply` installs the per-layer Block3 configs.
8. The model applies Block3 noise and truncation, produces metrics, and those
   metrics determine the reward and PPO update.

## Failure Handling

- A missing Block3 baseline graph or invalid Block3 replan is handled by the
  existing canonical invalid-chain path; it must not silently skip Block3.
- A degree/config mismatch remains a hard installation error.
- Tests must fail if Block3 is removed from optimizer requests, ignored by the
  bridge, or if optimizer write-back changes K.

## Verification

- Unit test: optimizer requests contain all 12 Block3 configs.
- Unit test: bridge installs and clears Block3 with `cfg_per_layer`.
- Chain test: Block3 SFs start from the baseline loader and remain
  non-policy-controlled.
- Chain test: two otherwise identical actions with different Block3 K values
  install different K values while preserving identical Block3 SFs.
- Model test: deterministic Block3 polynomial input produces different outputs
  for different K values, proving real truncation execution.
- Server test: run the narrow Block3 chain gate with torch/model support and
  capture the installed config plus forward evidence.

