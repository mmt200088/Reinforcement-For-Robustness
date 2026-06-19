# Stage-2 Fusion-Option Precision Boost ("加大精度") — design

Status: approved mechanism (2026-06-19). Rollout: **block2 first**, then block4, block5.

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
- **Verify + cost every candidate via real replan:** keep only `{valid ∧ fusion_count unchanged ∧
  q_final all == q_max}`; noise = Σ installed-point variance (`SummedInstalledVariance`, same as enum).
  The **min-noise** survivor wins. (Noise-increasing fills — e.g. lowering a rescale `sf_post` — lose
  automatically; no need to forbid them.)
- **Fallback:** if no valid all-`q_max` candidate exists, keep the original option unchanged (log it).

Worked example (block2 fc=1), the 4 candidates = `{kt_mask2 (no-comp, before the qkt ×2, +1 SF→+2 bit)`,
`gamma (+comp on the rescales between it and stage 3)`, `wk (+comp)`, `kt_mask1 (+comp)}`; min-noise wins.

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
**Q/K binding:** bumping `kt_mask1_rescale` (or `wk`) must mirror to the Q side via
`sync_block2_qk_binding` (K-side is authoritative). Any boosted cfg must run the block2 sync.

block4 / block5 topologies: TBD when we get there (same contract; `default_block4/5_cfg_to_delta`
already give the node→cfg-field wiring).

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
