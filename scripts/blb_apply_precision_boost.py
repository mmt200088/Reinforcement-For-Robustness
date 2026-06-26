"""Apply the fusion-option precision boost ("加大精度", phase 1 + phase 2) to the
committed maps.

The boost (``fusion_enum.boost_options_for_block``) is a POST-enumeration step:
the builder runs ``enum → group_min_noise_options → boost_options_for_block``.
``boost_options_for_block`` now chains BOTH stages — phase 1 raises the
intermediate short primes, phase 2 raises the final output scale to its
config-derived ceiling — so this one script applies both. The committed
``blb_stage2_rl/fusion_maps/<profile>/*.json`` were built before phase 2 existed
(they hold phase-1-only options); re-running the full enumeration just to apply a
deterministic post-process would cost ~minutes to ~1h (block4), so this script
loads each committed map, re-derives the boost from each option's ``action_indices``
(the pre-boost base — never rewritten), and rewrites it — byte-equivalent to a
full builder rebuild's final step, but in seconds.

Needs torch + rescale_optimizer (the boost builds real cfgs and replan-verifies);
run on the server. Idempotent: re-running on an already-boosted map re-derives the
same phase-1+phase-2 options.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

_REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (str(_REPO), str(_REPO / "blb_stage2_rl"), str(_REPO / "Rescale_optimizer")):
    if p not in sys.path:
        sys.path.insert(0, p)

# (graph_key, block_idx, gelu_degree, attn_degree) for each boostable block-type.
# Mirrors scripts/blb_build_fusion_count_map.py's block_types_for_profile degrees.
# block5_n1 gets NO phase-1 boost (its fused chain is already all-q_max) but DOES get
# phase-2 (raise the output scale toward the ceiling; install limit is q_max=60 since
# ADR-019). block2's graph key is profile-suffixed (block2_<profile>); block4 /
# block5_n* are shared names — so this generalizes to any fine-tuned profile.
def boost_targets_for_profile(profile: str):
    p = str(profile)
    return [
        (f"block2_{p}", 2, 4, 2),
        ("block4", 4, 4, 2),
        ("block5_n1", 5, 1, 2),
        ("block5_n2", 5, 2, 2),
        ("block5_n4", 5, 4, 2),
    ]


# Back-compat default (mrpc); main() rebuilds this per --profile.
BOOST_TARGETS = boost_targets_for_profile("mrpc")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="mrpc")
    ap.add_argument("--maps-dir", default=str(_REPO / "blb_stage2_rl" / "fusion_maps" / "mrpc"))
    ap.add_argument("--rescale-optimizer-root", default=str(_REPO / "Rescale_optimizer"))
    ap.add_argument("--num-layers", type=int, default=12,
                    help="transformer layers (12 bert-base / 24 bert-large); matches the build")
    ap.add_argument("--ref-layer", type=int, default=1, help="reference layer for the block-type context")
    ap.add_argument("--only", default="", help="comma list of graph_keys (default: all boostable)")
    ap.add_argument("--dry-run", action="store_true", help="report, do not rewrite JSON")
    args = ap.parse_args()

    import fusion_enum

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    maps_dir = pathlib.Path(args.maps_dir)
    rc = 0
    for graph_key, block_idx, gelu, attn in boost_targets_for_profile(args.profile):
        if only and graph_key not in only:
            continue
        path = maps_dir / f"{graph_key}.json"
        if not path.exists():
            print(f"[skip] {graph_key}: no map at {path}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        options = payload["options"]
        before = [(o["option_id"], o["fusion_count"], o.get("boosted", False),
                   o["total_bits"], o["total_variance"]) for o in options]

        ctx = fusion_enum.prepare_block_type_context(
            graph_key=graph_key, block_idx=block_idx, gelu_degree=gelu, attn_degree=attn,
            profile=args.profile, rescale_optimizer_root=args.rescale_optimizer_root,
            num_layers=int(args.num_layers), ref_layer=int(args.ref_layer),
        )
        options = fusion_enum.boost_options_for_block(ctx, options)
        n_boosted = sum(1 for o in options if o.get("boosted"))
        print(f"\n=== {graph_key} ===")
        for o in options:
            tag = ""
            if o.get("boosted"):
                tag = (f"  BOOSTED out_sf={o.get('output_sf', '?')} "
                       f"({o.get('boost_description', '')})")
            print(f"  opt {o['option_id']} fc={o['fusion_count']} bits={o['total_bits']} "
                  f"var={o['total_variance']:.3e}{tag}")
        print(f"  -> {n_boosted}/{len(options)} options boosted")

        if not args.dry_run:
            payload["options"] = options
            meta = payload.setdefault("build_meta", {})
            meta["precision_boost_applied"] = True
            meta["precision_boost_phase2_applied"] = True
            path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            print(f"  wrote {path}")
    print("\n[done] precision boost applied" + (" (dry-run)" if args.dry_run else ""))
    return rc


if __name__ == "__main__":
    sys.exit(main())
