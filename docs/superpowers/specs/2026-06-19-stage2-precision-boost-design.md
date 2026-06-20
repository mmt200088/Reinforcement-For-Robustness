# Stage-2 Fusion-Option Precision Boost ("加大精度") — design

Status: approved mechanism (2026-06-19). Rollout: **block2 done**, **block4 done**, block5 next.
(block2 + block4 are implemented + locally verified vs real replan; the server rebuilds the maps.)

## 1. Problem & rationale

The fusion-count map (`scripts/blb_build_fusion_count_map.py` → `blb_stage2_rl/fusion_enum*.py`)
enumerates each block-type's SF grid, groups configs by realized `fusion_count`, and keeps the
**exact minimum installed-noise** config per group (`group_min_noise_options`, `noise_tol=1e-18`).

Two structural facts make the kept non-zero-fusion option land on a **short modulus prime**:

1. The SF grid **sweeps each slot down from its baseline** (`sf_from`, anchor = calibrated
   `MaxSFsTable`). It can never produce an *above-baseline* SF.
2. To fuse, the option lowers several encode SFs jointly, which shifts the scale accounting so one
   non-fused rescale consumes a prime **< `q_max`** (e.g. block2 fc=1: chain `60 29 31 58 60`,
   29+31 fuse → `60 60 58 60`). Filling that `58` to `60` needs *more* SF than the baseline gives
   at the segment feeding it — i.e. **above-baseline**, which the grid structurally cannot reach.

So the enum keeps the short-prime config. "加大精度" is a deterministic **post-enum** step that, for
each non-zero-fusion option, raises the precision (SF) at specific points so **every prime hits
`q_max`** (`60 60 …60`), keeping the same `fusion_count`, and picks the **minimum-noise** way to do it.
This trades total_bits (higher, accepted) for accuracy headroom (the noise floor drops at the boosted
points). `total_bits` is not in the reward; the noise floor is what matters.

## 2. Chain semantics (from the graph + real replan)

Main-chain scale accounting (derivable from the graph node types):
- `ctct_*` (ciphertext×ciphertext, delta `"x2"`) → scale **×2**.
- `ctpt_*` (ciphertext×plaintext = encode) → scale **+= that encode's SF** (a "side node").
- each `cut_point` → **rescale**: consumes a prime = `(scale before rescale) − sf_post`.

`ReplanSession.replan(..., return_dict=True)["result"]` exposes everything needed (verified in
`Rescale_optimizer/rescale_optimizer/replan.py`):
- `q_initial` — per-stage primes **before** fusion (`[29,31,58]`),
- `q_final` — per-stage primes **after** fusion (`[60,58]`),
- `fusions` — which stages fused (`fused_position`, `fused_into`, …),
- `t_final`, `delta_q_vs_baseline`, `chain`, `new_compact_config`.

**Short prime** = any `q_final[s] < q_max` (`q_max = ModulusChain.q_head_bits = 60`).

## 3. The boost mechanism (confirmed)

Work on the **pre-fusion chain** (the graph's natural one-rescale-per-cut_point segments); let replan
re-fuse + verify. For each short prime (stage `s`, deficit `Δ = q_max − q_final[s]`):

- **SF→bits conversion:** adding `δ` SF at a node that passes through `c` `ctct` doublings before
  reaching stage `s`'s rescale raises that rescale's drop by `δ·2^c` bits. To fill exactly:
  `δ = Δ / 2^c` (must be a positive integer; else that node can't fill stage `s` and is skipped).
- **Placement candidates:**
  - **(a) same segment, no compensation** — an encode side-node in stage `s`'s own segment
    (after the previous surviving rescale). Bump `+δ`.
  - **(b) earlier segment, with compensation** — an encode side-node in an earlier segment, `+δ`,
    **and bump the `sf_post` of every surviving rescale between that node and stage `s` by `+δ`**
    (keeps those already-formed primes' drops constant; the raised `sf_post` carries `+δ` downstream,
    where the `ctct` doublings scale it up to fill stage `s`). Pre-fusion has more intervening
    rescales (compensate each); post-fusion they may be merged (compensate the fused one) — equivalent.
- **Off-limits placement:** any `ctct`/×2 node and the first side-node (the input ×2).
- **Multiple short primes:** generate candidates that fill **all** short primes jointly.
- **Fill-to-max, not always `q_max`.** A short prime fed through `c` `×2` doublings is raised by
  `2**c · S` where `S = floor(deficit / 2**c)` (the budget, in chain-accumulation units). It reaches
  `q_max` only when `deficit` is divisible by `2**c`. block2 fc=1: `deficit=2, c=1 → S=1 → 60`. block4
  fc=1: `deficit=29, c=1 → S=14 → 59` (an odd 60 is unreachable: `2×(integer)` can't be odd).
- **Distribute `S` by minimum noise.** `S` is spread across the addable upstream encodes; block2's
  single placement is the `S=1` case, block4's `S=14` is a real partition. A **binding multiplier**
  handles a shared-SF node (block4 `softmax_out_mask` is bound to `v_mask` → +1 SF costs 2 of `S`).
- **Cascade compensation (the rule actually used):** every rescale strictly before `R_target` absorbs
  the chain-weighted additions **upstream of it** (`sf_post += that`), keeping *every* intermediate
  prime — hence the whole fusion structure — constant. So every budget distribution stays at the same
  `fusion_count` and is replan-valid; fused rescales install no noise, so this never perturbs the
  objective. (A "compensate only the rescale before the ×2" variant drops valid candidates: block2's
  `gamma` would make two 30-primes that replan refuses to fuse, since fusion triggers only on a prime
  below `q_min`.)
- **Off-limits placement:** any `×2`/`fresh` node, and any encode after `R_target` (feeds the fixed
  `q_tail`: block2 `qkt_merge`, block4 `ln_var`).
- **Verify + cost every candidate via real replan:** keep only `{valid ∧ fusion_count unchanged ∧
  Σ q_final == base + total_fill}`; noise = Σ installed-point variance (`SummedInstalledVariance`,
  same as enum). The **min-noise** survivor wins. (Noise-increasing fills lose automatically.)
- **Multiple short primes:** cartesian of per-prime candidates with conflict-free merged edits.
- **Fallback:** if no fillable short prime or no candidate verifies, keep the original option (log it).

Worked example (block2 fc=1), the 4 candidates = `{kt_mask2 (no-comp, between R_pre and the qkt ×2)`,
`gamma`, `wk`, `kt_mask1` (cascade-comp)}`; `S=1`, min-noise wins (kt_mask2).
Worked example (block4 fc=1): `S=14` distributed over `{softmax_out_mask (×2 cost), softmax_v_mask, wo,
ln_mean}`, `ln_mean_rescale.sf_post += 14`; 372 distributions, all reach `[60,59]`, min-noise wins.

## 4. Per-block ChainTopology ("按 block 写死，不写死 SF/位置")

Per block-type, a small **ChainTopology**: the ordered nodes, each `{kind: ×2 | encode(graph_node,
cfg_field, action_slot) | rescale(cfg_field, action_slot)}`, derived once from the graph
(`default_block{N}_cfg_to_delta` + `GRAPH_NODE_TO_CFG_ATTR` + the skeleton cut_points) and frozen as a
per-block descriptor. One **generic** algorithm consumes ChainTopology + the replan chain; it is not
hardcoded per SF or per chain instance, so it handles any fc action group of that block.

**block2 topology (validated against the map SFs + the user's worked chains):**
```
fresh(inv_std_fresh)  ×2(ctct_x_mean_over_std, off-limits)
encode gamma (ctpt_gama1 / gamma_encode)            ─┐ segment 1
rescale gamma_rescale                                ┘
encode wk   (ctpt_wq_wk / wk_encode)                ─┐ segment 2
encode kt_mask1 (ctpt_rotKT_mask1 / kt_mask1_encode) │
rescale kt_mask1_rescale  (mirror q_mask1_rescale)   ┘
encode kt_mask2 (ctpt_rotKT_mask2 / kt_mask2_encode) ─┐ segment 3
×2 (ctct_preprocess_qkt, off-limits)                  │
encode qkt_merge (ctpt_mask / qkt_merge_mask_encode)  │
rescale qkt_matmul_rescale  ← short prime here        ┘
```
**Q/K binding:** `wk`/`kt_mask1_rescale` mirror to the Q side via `_build_block2_action` (and
`sync_block2_qk_binding`); the SF-direct build inherits this, so the boost sets only the K-side slot.

**block4 topology (validated against block4.json fc=1 + real replan):**
```
fresh(softmax_out_fresh, off-limits)
encode softmax_out_mask (ctpt_mask2)        ── addable, binding ×2 (bound to v_mask)
additive_ctct ctct_rot_softmax_mul_v        ── += v_fresh + v_mask  (NOT a doubling; off-limits)
rescale softmax_v_matmul_rescale            ── fuses away
encode softmax_v_mask (ctpt_mask)           ── addable
encode wo (ctpt_wo_attnout)                 ── addable
encode ln_mean (ctpt_inv_d_1)               ── addable
rescale ln_mean_rescale                     ── R_pre (fused prime)
×2 ctct_square (LN (X−μ)², off-limits)
rescale ln_square_rescale  ← short prime here
encode ln_var (ctpt_inv_d_2)                ── after R_target → feeds q_tail, off-limits
```
`q_initial [27,33,31] → fuse 1&2 → q_final [60,31]`; the `31` fills to `59` (deficit 29 odd). The
`softmax_out_mask → v_mask` binding (`_build_block4_action`, `sync_block4_v_mask_binding`) makes
`softmax_out_mask` cost 2/SF and is inherited by the SF-direct build (sets only `softmax_out_mask_sf`).

block5 topology: TBD next (same contract; `default_block5_cfg_to_delta` gives the wiring).

## 5. Data model + runtime (decision: explicit-SF option)

A boosted option abandons the down-sweep `action_indices` (it carries above-baseline SFs) and stores an
**explicit per-block SF vector**. Map JSON option gains `"boosted": true` + `"explicit_slots": {field: sf}`
(the K slot stays separately decided via `K_LEVELS`).

Runtime: `FusionCountMap.expand`/the env branch on `boosted`. For a boosted option, build the block cfg
**directly from the explicit SFs** (a new SF-direct path that bypasses `sf_from`/the grid), then run the
existing override + `sync_block2_qk_binding` + install pipeline unchanged. Non-boosted options are
byte-for-byte unchanged.

## 6. Integration & rollout

- **Where:** in the builder, after `group_min_noise_options`, before writing the map JSON. For each
  option with `fusion_count != 0`, run the boost; replace the stored option with its boosted (explicit-SF)
  form. `option0` (baseline, fc=0) is never touched.
- **Verification source of truth:** real replan (every candidate). Builder asserts the boosted option's
  `q_final` is all-`q_max` and `fusion_count` is unchanged.
- **Rebuild:** block2's map is rebuilt first; block4/5 follow. Existing per-slot path + non-fusion
  options are unaffected.
- **Rollout order:** block2 → block4 → block5.

## 7. Verification / tests

- Torch-free core (local, against real `ReplanSession`): block2 fc=1 → boost → replan `q_final` all 60,
  `fusion_count` still 1, and the chosen config is the min-noise of the 4 candidates.
- SF-direct decode == explicit SFs; boosted option's installed noise points match the builder's record.
- Server: rebuild block2 map + the existing 1==N / map gates.
