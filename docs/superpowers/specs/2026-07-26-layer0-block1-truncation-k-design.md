# Layer-0 Block1 Truncation-K Design

## Goal

Make `L0.B1.output_truncation_k` a real Stage-2 layerwise RL action with the
same K support, PPO semantics, communication-cost accounting, materialization,
installation, and runtime truncation point as every later-layer Block1 K.

## Scope

- Enable the existing `block1_k` policy coordinate at layer 0.
- Decode and account for five Block1 K coordinates per five model blocks in
  every layer, including `L0.B1.K`.
- Materialize `block1_cfgs[0]` and install it through the canonical
  `BLBNoiseRLBridge` path used by training, fixed-action evaluation, Paean
  final evaluation, and GLUE generation.
- Execute layer-0 Block1 K at the existing Block1 boundary: LayerNorm variance
  after the post-FFN head and before `rsqrt`.
- Keep the current layer-0 Block1 Gaussian-noise semantics unchanged. This
  task adds truncation only; it does not add layer-0 Block1 SF/fusion search.

## Design

`Block1NoiseConfig` gains a `noise_enabled` flag. All existing callers default
to `True`. The canonical action decoder constructs layer-0 Block1 through the
same builder as later layers, with `noise_enabled=False`; later layers remain
`True`. `NoisyBlock1LayerNorm` computes the clean mean/variance path when noise
is disabled and still calls the shared configured truncation executor. The
Block1 FFN2 wrapper likewise executes the clean projection when noise is
disabled.

This keeps one runtime module and one truncation executor for all Block1 K
values. The only distinction is whether the independent Gaussian-noise family
is enabled, which is orthogonal to K.

The Rescale Optimizer baseline handover remains allowed to omit `(block=1,
layer=0)`: K does not participate in SF replan, and requiring a new RO graph
entry would incorrectly couple this truncation-only change to the SF chain.

## Action And Cost Contract

- Layerwise policy mask: all six slots are active at every layer.
- Active K slots: `5 * num_layers`.
- Maximum communication saving:
  `5 * num_layers * (13 - 8)`.
- Neighbor generation may vary `L0.B1.K`.
- Checkpoint/action-space identity is bumped so an old checkpoint cannot be
  resumed under the changed action geometry.

## Verification

1. Torch-free codec tests prove `L0.B1.K` is active, decoded, spliced into the
   legacy vector, included in neighbors, and included in communication cost.
2. Materialization tests prove `block1_cfgs[0]` carries the selected K while
   `noise_enabled=False`.
3. Bridge tests prove layer 0 is no longer filtered and is installed.
4. Torch runtime tests prove changing only `L0.B1.K` changes the actual
   variance-before-rsqrt truncation while Gaussian sampling remains disabled.
5. Existing shared-path static tests continue to prove training, fixed-action
   evaluation, Paean, and GLUE consume canonical materialized configs.
