"""Apply the fusion-option precision boost ("加大精度") to the committed maps.

The boost (``fusion_enum.boost_options_for_block``) is a POST-enumeration step:
the builder runs ``enum → group_min_noise_options → boost_options_for_block``.
The committed ``blb_stage2_rl/fusion_maps/<profile>/*.json`` were built before the
boost existed, so they hold the pre-boost (enum→group) options. Re-running the
full enumeration just to apply a deterministic post-process would cost ~minutes
to ~1h (block4); since the boost only transforms the already-grouped options, this
script loads each committed map, applies the boost to its options, and rewrites it
— byte-equivalent to a full builder rebuild's final step, but in seconds.

Needs torch + rescale_optimizer (the boost builds real cfgs and replan-verifies);
run on the server. Idempotent: re-running on an already-boosted map re-derives the
same boosted options (the base SFs come from each option's action_indices, which
the boost never rewrites).
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
# Mirrors scripts/blb_build_fusion_count_map.py's BLOCK_TYPES degrees; block5_n1
# is intentionally absent (its fused chain is already 60/60/60 → no boost).
BOOST_TARGETS = [
    ("block2_mrpc", 2, 4, 2),
    ("block4", 4, 4, 2),
    ("block5_n2", 5, 2, 2),
    ("block5_n4", 5, 4, 2),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="mrpc")
    ap.add_argument("--maps-dir", default=str(_REPO / "blb_stage2_rl" / "fusion_maps" / "mrpc"))
    ap.add_argument("--rescale-optimizer-root", default=str(_REPO / "Rescale_optimizer"))
    ap.add_argument("--only", default="", help="comma list of graph_keys (default: all boostable)")
    ap.add_argument("--dry-run", action="store_true", help="report, do not rewrite JSON")
    args = ap.parse_args()

    import fusion_enum

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    maps_dir = pathlib.Path(args.maps_dir)
    rc = 0
    for graph_key, block_idx, gelu, attn in BOOST_TARGETS:
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
            num_layers=12, ref_layer=1,
        )
        options = fusion_enum.boost_options_for_block(ctx, options)
        n_boosted = sum(1 for o in options if o.get("boosted"))
        print(f"\n=== {graph_key} ===")
        for o in options:
            tag = ""
            if o.get("boosted"):
                tag = f"  BOOSTED ({o.get('boost_description', '')})"
            print(f"  opt {o['option_id']} fc={o['fusion_count']} bits={o['total_bits']} "
                  f"var={o['total_variance']:.3e}{tag}")
        print(f"  -> {n_boosted}/{len(options)} options boosted")

        if not args.dry_run:
            payload["options"] = options
            payload.setdefault("build_meta", {})["precision_boost_applied"] = True
            path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            print(f"  wrote {path}")
    print("\n[done] precision boost applied" + (" (dry-run)" if args.dry_run else ""))
    return rc


if __name__ == "__main__":
    sys.exit(main())
