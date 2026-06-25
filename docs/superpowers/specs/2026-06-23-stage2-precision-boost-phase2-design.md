# Stage-2 Fusion-Option Precision Boost — PHASE 2 ("二阶段加大精度") — design

Status: approved mechanism (2026-06-23). Builds on phase 1
(`2026-06-19-stage2-precision-boost-design.md`). Implemented + locally verified vs real replan
(`tests/test_blb_precision_boost_phase2.py`); the server applies it to the committed maps
(`scripts/blb_apply_precision_boost.py`, now chains phase-1 + phase-2). Rollout:
**block2 / block4 / block5_n1 / block5_n2 / block5_n4**. block5_n1 gets phase 2 even though it
has no phase 1 (its fused chain is already all-q_max — no short prime to raise — but its output
scale can still be lifted).

## 1. Problem & rationale

Phase 1 raises the **intermediate short modulus primes** of a fusion option to `q_max`, but leaves the
**final output scale fixed**. The "last node SF" — the scale of the block's output ciphertext — is

```
output_sf = last_rescale.sf_post + final_encode_SF
```

(`final_encode` = the encode after the last rescale, feeding `q_tail`; block5 has none → `+0`). After
phase 1 the committed maps sit at:

| block | last sf_post + final_encode = output | config target | achieved (install-clamped) |
|-------|--------------------------------------|---------------|----------------------------|
| block2 | 28 + 15 = **43** | 46 | 46 |
| block4 | 31 + 20 = **51** | 53 | 53 |
| block5_n1 | 31 + 0 = **31** | 48 | **46** (clamped — see §3) |
| block5_n2 | 31 + 0 = **31** | 43 | 43 |
| block5_n4 | 31 + 0 = **31** | 43 | 43 |

The output can be higher. The ceiling is set by the block's amplitude budget for the final cut point:

```
target_output_sf = q_tail_bits - amplitude_budgets[-1] - h_sf
```

all three read from `Rescale_optimizer/configs/<profile>/<graph_key>.json` (general — a changed JSON
yields a changed target; **not hardcoded**). For mrpc: `60 - {12,5,10,15,15} - 2 = {46,53,48,43,43}`
for block2 / block4 / block5_n1 / block5_n2 / block5_n4.

Phase 2 is a deterministic **post-phase-1** step that raises `output_sf` to that ceiling at minimum
installed noise, keeping `fusion_count` and every prior prime unchanged. (`replan` does NOT enforce the
ceiling — `merge=19 → output 47` replans "valid" — so phase 2 enforces `output == target` itself.)

## 2. Composition space

Raise `output_sf` by `Δ = target - base_output`. Parameterize by the final encode SF:

* `final_encode ∈ [FINAL_ENCODE_MIN(=15, hardcoded), base_final + Δ]`, with
  `sf_post = target - final_encode` (so `sf_post ≥ base_sf_post`, i.e. the pre-scale only rises).
  * Final encode can **increase** (block2's only option — its encode is already at the floor 15) OR
    **decrease to 15** (block4's user-spec special case: drop the final encode, push the freed precision
    onto `sf_post`).
  * Blocks with no final encode (block5) → `final_encode = 0`, `sf_post = target`.
* Raising `sf_post` by `δ_r` needs the pre-scale entering the last rescale to rise `δ_r`. That rise is
  supplied by the upstream encodes through the `×2` doublings, with every intermediate prime
  compensated — **exactly the phase-1 machinery** (`_resolve_geometry` on the last rescale,
  `_addable_bit_weights`, `_max_reachable_fill`, `_simulate_rescale_edits`). The last prime is kept as
  high as possible (`≤ q_max`): more upstream supply = higher prime = lower noise, so min-noise prefers
  the max prime (replan rejects > q_max). block4's last prime is even LIFTED 59→60 by some compositions.

The minimum-installed-noise survivor (same `SummedInstalledVariance` metric as the fusion-count enum and
phase 1) becomes the option's `explicit_field_values`.

## 3. The ≤46 install cap (hard constraint) — SUPERSEDED by ADR-019 (2026-06-25)

> **SUPERSEDED**: ADR-019 makes a scaling factor above the noise-table max (46) install
> **no noise** (var(49)≈4e-27 ≪ fp precision), so the install limit is now q_max (60), not 46.
> The two block-specific consequences below NO LONGER hold: block4's final encode (1/d) DOES
> decrease to 15 (`ln_mean_rescale`→49 installs no noise; lower total noise than the increase
> route), and block5_n1 reaches its full ceiling 48 (not clamped to 46). The section is kept
> for historical context; read ADR-019 for the current behavior.

The model's noise install (`function_handler.get_input_noise_variance_by_N`) **raises** for any SF above
the noise table's max (**46**) — encodes, fresh, AND rescales. So every installed point in a phase-2
composition must be `≤ 46`, or the model crashes. `generate_phase2_candidates(max_installed_sf=46)`
drops any candidate whose installed encode/rescale SF would exceed it, with **no lower-prime fallback**
(the prime stays high or the composition is dropped).

Consequence for **block4**: its compensation rescale `ln_mean_rescale` is already at 45 (one bit of
headroom), and it is the *only* path feeding the last rescale's pre-scale (the chain is
`ln_mean_rescale → ×2 → ln_square_rescale`, no encode between). So `sf_post` can rise by at most ~2
before `ln_mean_rescale` hits 47. The user's worked special case (final_encode 20→15, sf_post 31→38)
needs `ln_mean_rescale → 49` → **uninstallable**; block4's min-noise therefore uses the final-encode
*increase* route (e.g. final_encode 21, sf_post 32, `ln_mean_rescale` 46) and never decreases the final
encode. The decrease mechanism is general and *is* exercised when a block has headroom (verified with the
cap lifted) — block4 just doesn't have it. block2/block5 stay well under 46.

Consequence for **block5_n1**: its output is a *single* rescale (no final encode to split the scale
across two ≤46 points), so the output itself is bounded by 46. Its config ceiling is 48, but the install
cap binds at **46**. `effective_output_target(topology, config_target)` clamps the goal:
`min(config_target, 46 × (2 if final_encode else 1))` — only no-final-encode blocks above 46 are
affected (block5_n1: 48 → 46; block5_n2/n4 at 43 and the final-encode blocks are unchanged). n1 still
gains +15 on its output (31 → 46), distributed across `gamma`/`wffn1`/`gelu_coeff` (all weight 1, no
compensation), q_final preserved. (Reaching 48 would require noise-table entries above 46 — a
model-level change, out of scope.)

## 4. Concrete min-noise results (mrpc, local real-replan)

| block | output | q_final (phase1 → phase2) | boost |
|-------|--------|---------------------------|-------|
| block2 | 43→46 | (60,60) → (60,60) | final_encode 15→16, sf_post 28→30, gamma +1 |
| block4 | 51→53 | (60,59) → (60,60) | final_encode 20→21, sf_post 31→32, ln_mean_inv_d +1 |
| block5_n1 | 31→46 | (60,) → (60,) | sf_post 31→46 via gamma/wffn1/gelu_coeff (no phase 1) |
| block5_n2 | 31→43 | (60,60) → (60,60) | sf_post 31→43 via gamma/wffn1/gelu_coeff |
| block5_n4 | 31→43 | (60,31,60) → (60,31,60) | sf_post 31→43 (middle prime 31 kept) |

(min-noise composition depends on the metric; the builder's `SummedInstalledVariance` may pick a
different but equal-or-lower-noise split — the build guard accepts any composition that hits the target
with prior primes preserved and all SF ≤46.)

## 5. Code

* `blb_stage2_rl/precision_boost.py`:
  * `target_output_sf(graph_key, profile, root)` — the config-derived ceiling.
  * `effective_output_target(topology, config_target)` — clamps the ceiling to the install cap (46 for
    no-final-encode blocks, 92 otherwise); only block5_n1 is affected (48 → 46).
  * `FINAL_ENCODE_MIN = 15`.
  * `BLOCK5_N1_MRPC_TOPOLOGY` — new (degree-1 GELU: no gelu ×2, all encodes weight 1, no compensation).
  * `_last_rescale_and_final_encode(topology)` — locates the last rescale + the final encode (if any).
  * `generate_phase2_candidates(..., max_installed_sf=46)` — the composition enumeration + install cap.
  * `boost_option_phase2(...)` → `Phase2Result` — replan-verified min-noise driver (clamps target internally).
  * `ReplanProbe.t_final` — added so the driver can read the achieved `sf_post` and check the output.
* `blb_stage2_rl/fusion_enum.py`: `boost_options_for_block` now chains phase-1 → phase-2; the build guard
  verifies valid + `fusion_count` + prior primes + `output == target` + all installed `≤ 46`; the option
  stores `output_sf` and a combined `boost_description` (`p1:…; p2:…`). `BlockTypeBuildContext` gained
  `rescale_optimizer_root` (to read the config target). `_eval_block_from_field_values` returns `t_final`.
* `scripts/blb_apply_precision_boost.py` — unchanged entrypoint; applies both phases (since
  `boost_options_for_block` does), records `precision_boost_phase2_applied`.

## 6. Tests

`tests/test_blb_precision_boost_phase2.py` — target formula (all blocks), composition enumeration (the 3
methods + the decrease-to-15 mechanism uncapped + the cap excluding it for block4 + block5's sf_post-only),
and real-replan driver for block2/block4/block5_n2/block5_n4 (reaches target, prior primes preserved,
min-noise verified independently, all installed ≤46). The phase-1 file's `BoostOptionsForBlockGuardTest`
was updated to drive the full builder phase-1+phase-2 path. Torch-free core + real-replan lanes (the
latter skip where `rescale_optimizer` isn't importable).

## 7. Invariants preserved

`option0` (fc=0 baseline) never touched; the SF-direct boosted runtime path (phase-1 data model) is
reused as-is (above-baseline SF has no action index → boosted option stores `explicit_field_values`); the
build is replan-guarded (any drift aborts); changing the grid still invalidates maps (rebuild). Phase 2
**raises FUSION options' installed precision** → it shifts the fusion reward/accuracy landscape (more
faithful); the no-fusion baseline is unchanged. An in-flight run keeps its old maps in memory — restart to
pick up phase-2.
