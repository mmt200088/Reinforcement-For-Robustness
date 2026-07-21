# Stage-2 Action Materialization and Optional MPC Truncation Design

## Goal

Make every Stage-2 model evaluation consume one canonical, fully materialized
configuration, and add a more protocol-aligned plaintext truncation simulator
that remains dormant unless explicitly selected.

The current deterministic truncation behavior remains the default. This change
must not alter existing RL, final evaluation, fixed-action evaluation, or GLUE
results unless the new truncation backend is explicitly enabled.

## Canonical Runtime Contract

Every path that evaluates a Stage-2 action against the model follows this order:

1. Decode the flat policy action with the calibrated Rescale Optimizer baseline.
2. Resolve fusion-count map options for Blocks 1, 2, 4, and 5.
3. Apply any fusion precision-boost overrides as explicit SF values.
4. Preserve Block3 baseline SFs while applying the selected Block3 K.
5. Build and execute Rescale Optimizer replan requests for all five blocks.
6. Write every valid optimizer result back into the decoded model configs.
7. Verify that every expected valid config was updated successfully.
8. Compute a stable fingerprint from the final post-replan configs.
9. Install exactly those configs into the model bridge.
10. Run inference, derive metrics, compute reward, and feed that reward to PPO.

The output of steps 1-8 is a single materialized-action object. Online RL,
parallel probes, promotion/revalidation, Paean final evaluation, fixed-action
experiments, and GLUE generation must use this object instead of reimplementing
decode/replan/write-back logic.

## Failure Semantics

- A genuinely optimizer-invalid action remains a normal invalid RL action. It
  does not run a model forward pass and receives the existing invalid outcome.
- If optimizer outputs are valid but any expected config is absent, cannot be
  written back, or cannot be verified, materialization fails closed. No model
  forward pass is allowed with a partially updated configuration.
- If a graph's Rescale Optimizer baseline skeleton is unavailable,
  materialization also fails closed because fused-away rescale positions cannot
  be identified safely.
- A bridge-install failure also prevents inference and is surfaced explicitly.
- The policy action hash remains the search identity. A separate final-config
  fingerprint is the model-install/cache identity.
- Persistent install caching compares the final-config fingerprint, never just
  the flat action hash. This prevents boosted SF overrides from aliasing an
  unboosted installation.

## Block3 Invariants

- Block3 exposes no fusion-count or SF policy action.
- Its SF chain is loaded once from the calibrated Rescale Optimizer baseline.
- The layerwise policy changes only Block3 truncation K.
- Replan validates and may reproduce Block3 SFs but must preserve its selected K.
- The final materialized fingerprint includes Block3 SFs and K.

## Truncation Backends

### Legacy backend (default)

`binary` remains exactly:

```text
trunc(x * 2^K) / 2^K
```

`decimal` remains available for compatibility. Neither consumes truncation RNG.
No existing default, preset, launcher, or final-eval behavior changes.

### Optional `stochastic_ring` backend

This backend approximates the numerical semantics that matter for plaintext MPC
simulation without pretending to implement a secure MPC protocol.

Defaults:

- signed two's-complement ring: `Z_(2^43)`
- source fixed-point fractional bits: `24`
- target fractional bits: the selected action K

The source fractional width must be non-negative and strictly smaller than the
ring width; target K must be between zero and the source fractional width.

For each finite tensor value `x`:

1. Encode `X = round(x * 2^24)` as a signed integer.
2. Wrap `X` into `Z_(2^43)` and decode it as a signed two's-complement integer.
3. Let `shift = 24 - K`.
4. Compute arithmetic quotient `q = floor(X / 2^shift)` and non-negative
   remainder `r = X - q * 2^shift`.
5. Sample `b ~ Bernoulli(r / 2^shift)` from a dedicated truncation RNG stream.
6. Return `(q + b) / 2^K`.

This is unbiased for both positive and negative encoded values, up to the source
fixed-point encoding error. It models signed arithmetic shift, probabilistic
rounding, finite ring wrap, and K-dependent scale reduction.

It intentionally omits secret shares, communication, masking, field/ring
conversion protocols, security failure probabilities, and the protocol's
security-extension bits. It is a protocol-aligned numerical simulator, not a
cryptographic MPC implementation.

## RNG Isolation

Stochastic truncation uses a generator stream separate from Gaussian BLB noise.
The existing trial seed deterministically derives both streams with distinct
domain-separation salts. Drawing truncation randomness must never advance or
change Gaussian samples. Legacy backends consume no truncation RNG.

## Activation

- `binary` is the explicit and default runtime value.
- `stochastic_ring` is selectable only through an explicit Stage-2 runtime
  configuration/CLI value.
- No environment-variable fallback is allowed.
- The selected backend and ring/source-bit parameters are persisted in run
  metadata and included in the final-config fingerprint.

## Verification

- Legacy positive/negative outputs remain bit-for-bit unchanged.
- `stochastic_ring` produces only adjacent K-grid outcomes and is empirically
  unbiased for positive and negative inputs.
- Ring wrap and two's-complement decode are tested at boundaries.
- Same seed reproduces the truncation stream; different seeds vary it.
- Truncation draws do not change Gaussian samples.
- All five block K settings alter their actual installed runtime outputs.
- Block3 K changes while its baseline SF chain remains identical.
- Valid-but-incomplete replan write-back prevents every forward path.
- Boosted and unboosted final configs cannot share an install fingerprint.
- Production RL, Paean/final evaluation, fixed-action evaluation, and GLUE
  produce the same final-config fingerprint for the same calibrated action.
