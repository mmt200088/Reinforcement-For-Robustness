#!/usr/bin/env python3
"""Focused diagnostic for the block2 precision-boost `output_sf != target` failure.

The 5-profile build emits a block2 fc=1 option that golden replan confirms IS
fusion 1 (so the enumeration is NOT mislabelling fc), yet precision boost only
reaches output_sf=43 (target 46). The fc=1 representative's three rescale slots
(gamma/kt_mask1/qkt_matmul) sit at the lex-min action index while mrpc's reaches
46 with them at the baseline index — so the boost is sensitive to the (noise-
irrelevant?) rescale base. Earlier torch-free reconstructions disagreed with the
real decode, so this dumps the GROUND TRUTH from the real pipeline:

  1. the cfg rescale SFs the runtime decode (action_vector_to_cfgs) produces;
  2. the t_new the bridge derives + the golden replan (fc, q_final, total_bits);
  3. the boost's own decoded base (_decode_block_field_values) + what it boosts to;
  4. a CANONICALISED variant (the 3 rescale slots forced to their baseline index):
     its cfg / fc / installed-noise signature / boost output — i.e. does resetting
     the rescales to baseline keep fc + noise but let the boost reach the target?

Run on the server (CPU is fine: CUDA_VISIBLE_DEVICES=""; needs torch +
rescale_optimizer). Reads the already-built map for the kept option, or accepts
--action-indices.

Usage:
  python scripts/blb_diag_block2_boost.py --profile rte \
      --maps-dir blb_stage2_rl/fusion_maps/rte --num-layers 12
"""
from __future__ import annotations

import argparse
import pathlib
import sys

_REPO = pathlib.Path(__file__).resolve().parents[1]
for _p in (str(_REPO), str(_REPO / "blb_stage2_rl"), str(_REPO / "Rescale_optimizer")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

from cli_parse_utils import parse_int_list_text  # noqa: E402
import fusion_enum  # noqa: E402
from json_utils import read_json_file  # noqa: E402
import precision_boost as pb  # noqa: E402


# block2 topology rescale cfg_fields -> their action-vector slot index (_BLOCK2_FIELDS order)
_BLOCK2_RESCALE_SLOTS = {
    "gamma_rescale_sf": 8,
    "kt_mask1_rescale_sf": 18,
    "qkt_matmul_rescale_sf": 21,
}


def _cfg_rescale_sfs(cfg) -> dict:
    out = {}
    for attr in ("gamma_result_rescale", "kt_mask1_result_rescale", "qkt_matmul_result_rescale",
                 "kt_mask2_result_rescale", "q_mask2_result_rescale"):
        pt = getattr(cfg, attr, None)
        out[attr] = None if pt is None else int(getattr(pt, "scaling_factor", -1))
    return out


def _report(ctx, label, action_indices):
    from action_space import _decode_block_field_values, action_vector_to_cfgs
    from rescale_optimizer_bridge import cfg_to_t_new_from_table

    blk = int(ctx.block_idx)
    print(f"\n================ {label} ================")
    print(f"action_indices = {list(action_indices)}")
    print(f"  rescale slot idx: gamma(8)={action_indices[8]} kt_mask1(18)={action_indices[18]} "
          f"qkt(21)={action_indices[21]}")

    # (a) boost's own decode
    base_fv = _decode_block_field_values(
        layer_idx=int(ctx.ref_layer), block_idx=blk,
        action_slice=np.asarray(action_indices, dtype=int),
        max_sfs=ctx.max_sfs, attn_degree=int(ctx.attn_per_layer[ctx.ref_layer]),
        gelu_degree=int(ctx.gelu_per_layer[ctx.ref_layer]),
    )
    print("  [_decode_block_field_values] rescales:",
          {k: base_fv.get(k) for k in ("gamma_rescale_sf", "kt_mask1_rescale_sf", "qkt_matmul_rescale_sf")})

    # (b) runtime cfg + bridge t_new
    full = ctx.baseline_full.copy()
    full[ctx.block_offset: ctx.block_offset + ctx.block_num_slots] = np.asarray(action_indices, dtype=int)
    decoded = action_vector_to_cfgs(
        full, ctx.max_sfs, num_layers=ctx.num_layers,
        gelu_degree=ctx.gelu_per_layer, attn_degree=ctx.attn_per_layer,
        only=(int(ctx.ref_layer), blk),
    )
    cfg = decoded.cfgs_dict()[f"block{blk}"][ctx.ref_layer]
    print("  [runtime cfg] rescale SFs:", _cfg_rescale_sfs(cfg))
    t = cfg_to_t_new_from_table(ctx.graph_key, cfg,
                                baseline_t_new=ctx.bridge._lookup_baseline_t_new(ctx.graph_key),
                                table=ctx.bridge._cfg_to_t_new_table)
    print("  [bridge t_new] =", list(t or []))

    # (c) golden eval
    g = fusion_enum._eval_block(ctx, list(action_indices))
    if g.get("valid"):
        sig = fusion_enum._installed_signature(g["points"])
        print(f"  [golden _eval_block] valid fc={g['fusion_count']} total_bits={g['total_bits']} "
              f"n_points={len(g['points'])} sig_hash={hash(sig) & 0xffffff:06x}")
    else:
        print("  [golden _eval_block] INVALID")

    # (d) boost
    opt = {"option_id": 1, "fusion_count": int(g.get("fusion_count", 1)),
           "action_indices": list(action_indices), "slots": {},
           "total_bits": int(g.get("total_bits", 0)), "total_variance": 0.0}
    out_sf = None
    try:
        boosted = fusion_enum.boost_options_for_block(ctx, [dict(opt)])[0]
        out_sf = boosted.get("output_sf")
        print(f"  [boost] output_sf={out_sf} boosted={boosted.get('boosted')} "
              f"desc={boosted.get('boost_description')}")
    except Exception as exc:  # boost guard may raise (e.g. fc changed / inconsistent)
        print(f"  [boost] RAISED: {type(exc).__name__}: {exc}")
    sig = fusion_enum._installed_signature(g["points"]) if g.get("valid") else None
    return g.get("fusion_count"), sig, out_sf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="rte")
    ap.add_argument("--maps-dir", default="blb_stage2_rl/fusion_maps/rte")
    ap.add_argument("--rescale-optimizer-root", default="Rescale_optimizer")
    ap.add_argument("--num-layers", type=int, default=12)
    ap.add_argument("--ref-layer", type=int, default=1)
    ap.add_argument("--action-indices", default="", help="comma list; default reads the built map's fc=1 option")
    args = ap.parse_args()

    gk = f"block2_{args.profile}"
    ctx = fusion_enum.prepare_block_type_context(
        graph_key=gk, block_idx=2, gelu_degree=4, attn_degree=2,
        profile=args.profile, rescale_optimizer_root=args.rescale_optimizer_root,
        num_layers=int(args.num_layers), ref_layer=int(args.ref_layer),
    )

    if args.action_indices:
        ai = parse_int_list_text(args.action_indices, allow_semicolon=False)
    else:
        mp = pathlib.Path(args.maps_dir) / f"{gk}.json"
        payload = read_json_file(mp)
        fc1 = [o for o in payload["options"] if int(o.get("fusion_count", 0)) == 1]
        if not fc1:
            print(f"[diag] no fc=1 option in {mp}")
            return 1
        ai = [int(x) for x in fc1[0]["action_indices"]]

    fc0, sig0, out0 = _report(ctx, "AS-BUILT fc=1 option", ai)

    # canonicalise: force the 3 topology rescale slots to their BASELINE index.
    base_block = [int(x) for x in ctx.baseline_block_indices]
    ai_canon = list(ai)
    for _f, pos in _BLOCK2_RESCALE_SLOTS.items():
        ai_canon[pos] = int(base_block[pos])
    fc1c, sig1, out1 = _report(ctx, "CANONICALISED (rescales -> baseline idx)", ai_canon)

    print("\n================ VERDICT ================")
    print(f"as-built : fc={fc0} boost_output={out0}")
    print(f"canonical: fc={fc1c} boost_output={out1}")
    print(f"fc preserved by canonicalisation : {fc0 == fc1c}")
    print(f"installed-noise signature preserved: {sig0 == sig1}")
    print(f"canonical reaches target 46       : {out1 == 46}")
    if fc0 == fc1c and sig0 == sig1 and out1 == 46:
        print("=> FIX CONFIRMED: canonicalising the noise-irrelevant rescales to baseline is "
              "noise+fc preserving AND lets the boost reach the target.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
