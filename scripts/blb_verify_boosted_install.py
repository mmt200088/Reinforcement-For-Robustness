#!/usr/bin/env python3
"""Verify the RUNTIME install path for precision-boosted fusion options.

This answers the two user questions the map-content gate (step [5] of the
precision-boost SERVER_COMMAND) does NOT:

  Q1. After RL picks a fusion option, is the action group SENT TO THE MODEL the
      phase-2 precision-boosted one (not the lower-precision in-grid decode)?
  Q2. Has that installed action group's rescale undergone modulus-chain fusion
      processing (the fused rescale nulled so it installs NO noise)?

It drives the EXACT runtime path ``BLBStage2Env.step`` uses on each committed
map's boosted fusion option:

    evaluate_action_for_cost(boosted_overrides=...)   # SF-direct rebuild + replan
      -> apply_optimizer_output_to_cfg(...) + sync_block*   # optimizer write-back
                                                             # (incl. fused-rescale nulling)

then inspects the resulting cfg — the one ``bridge.apply`` installs noise from.

  * Q1 is proven by comparing the installed SF projection WITH the boost vs the
    plain in-grid decode of the SAME option: the boosted install must carry
    strictly more installed precision (higher summed SF) — i.e. the model gets
    the boosted group, not the grid group.
  * Q2 is proven directly from the optimizer write-back overrides: when the
    option fuses (fusion_count >= 1) at least one rescale field must be nulled
    with ``source == "rescale_fused_away"``.

Needs torch + rescale_optimizer (the install path imports function_handler).
Run on the server (CPU is fine: ``CUDA_VISIBLE_DEVICES=""``). Exits non-zero on
any violation so the gate can hard-fail on it.

Usage:
    python scripts/blb_verify_boosted_install.py \
        --profile mrpc --rescale-optimizer-root Rescale_optimizer \
        --maps-dir blb_stage2_rl/fusion_maps/mrpc
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

_REPO = pathlib.Path(__file__).resolve().parents[1]
for _p in (str(_REPO), str(_REPO / "blb_stage2_rl"), str(_REPO / "Rescale_optimizer")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from action_space import parse_config_name  # noqa: E402
import fusion_enum  # noqa: E402
import numpy as np  # noqa: E402
from optimizer_cost import evaluate_action_for_cost  # noqa: E402

from rescale_optimizer_bridge import (  # noqa: E402
    _strip_layer_suffix,
    apply_optimizer_output_to_cfg,
    sync_block2_aux_fresh_binding,
    sync_block2_qk_binding,
    sync_block4_v_mask_binding,
    sync_block5_aux_fresh_binding,
)

# Degenerate / dormant maps carry no boosted fusion option to install. block1's
# key is profile-suffixed (block1_<profile>) and is fusion-degenerate (never
# boosted); block5_n0 is the dormant degree-0 map.
_SKIP_GRAPH_KEYS = {"block5_n0"}


def _is_skipped_graph_key(graph_key: str) -> bool:
    gk = str(graph_key)
    return gk in _SKIP_GRAPH_KEYS or gk.startswith("block1_")


def _cfg_sf_projection(cfg) -> dict:
    """Stable {cfg_attr: scaling_factor} projection of a block NoiseConfig.

    A noise point set to ``None`` (e.g. a fused-away rescale) installs nothing
    and is therefore absent from the projection. Mirrors the helper in
    tests/test_blb_precision_boost.py."""
    out = {}
    for name in vars(cfg):
        val = getattr(cfg, name)
        for i, pt in enumerate(val if isinstance(val, tuple) else (val,)):
            sf = getattr(pt, "scaling_factor", None)
            if sf is not None:
                out[f"{name}[{i}]" if isinstance(val, tuple) else name] = int(sf)
    return out


def _install_and_inspect(ev, ctx):
    """Run env.step's optimizer write-back loop on the target (block, layer) and
    return ``(installed_cfg, override_entries)``. Faithful to BLBStage2Env.step."""
    invoker = getattr(ctx.bridge, "invoker", None)
    invoker_baselines = getattr(invoker, "baselines", {}) or {}
    target_cfg = None
    overrides: list = []
    for cn, out in ev.outputs.items():
        block_idx, _profile, layer_idx = parse_config_name(cn)
        if layer_idx < 0 or block_idx != int(ctx.block_idx) or layer_idx != int(ctx.ref_layer):
            continue
        cfg = ev.cfgs_dict[f"block{block_idx}"][int(layer_idx)]
        graph_key, _ = _strip_layer_suffix(cn)
        baseline_entry = invoker_baselines.get(graph_key)
        baseline_skeleton = list(baseline_entry[0]) if baseline_entry else []
        ov = apply_optimizer_output_to_cfg(
            cfg,
            output_raw=out.raw,
            block_idx=int(block_idx),
            graph_key=graph_key,
            baseline_skeleton=baseline_skeleton,
            rotation_name_map={},  # rotations don't affect Q1 (output SF) / Q2 (fused rescale)
        )
        if int(block_idx) == 2:
            ov = list(ov) + sync_block2_qk_binding(cfg) + sync_block2_aux_fresh_binding(cfg)
        elif int(block_idx) == 4:
            ov = list(ov) + sync_block4_v_mask_binding(cfg)
        elif int(block_idx) == 5:
            ov = list(ov) + sync_block5_aux_fresh_binding(cfg)
        target_cfg, overrides = cfg, list(ov)
    return target_cfg, overrides


def _evaluate(ctx, action_indices, boosted_overrides=None):
    full = np.asarray(ctx.baseline_full, dtype=int).copy()
    full[ctx.block_offset: ctx.block_offset + ctx.block_num_slots] = np.asarray(action_indices, dtype=int)
    return evaluate_action_for_cost(
        full,
        profile=ctx.profile,
        num_layers=int(ctx.num_layers),
        max_sfs=ctx.max_sfs,
        rescale_bridge=ctx.bridge,
        gelu_degree=ctx.gelu_per_layer,
        attn_degree=ctx.attn_per_layer,
        boosted_overrides=boosted_overrides,
    )


def verify_map(path: pathlib.Path, profile: str, ro_root: str, num_layers: int) -> tuple:
    """Returns (checked, problems) for one map file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    graph_key = str(payload["graph_key"])
    block_idx = int(payload["block_idx"])
    gelu_degree = int(payload.get("gelu_degree", 4))
    attn_degree = int(payload.get("attn_degree", 2))
    boosted_opts = [
        o for o in payload["options"]
        if int(o.get("fusion_count", 0)) >= 1 and o.get("boosted") and o.get("explicit_field_values")
    ]
    if not boosted_opts:
        print(f"[skip] {graph_key}: no boosted fusion option")
        return 0, 0

    ctx = fusion_enum.prepare_block_type_context(
        graph_key=graph_key, block_idx=block_idx, gelu_degree=gelu_degree, attn_degree=attn_degree,
        profile=profile, rescale_optimizer_root=ro_root, num_layers=num_layers, ref_layer=1,
    )
    checked = 0
    problems = 0
    for o in boosted_opts:
        oid = int(o.get("option_id", -1))
        fc = int(o["fusion_count"])
        explicit = {str(k): int(v) for k, v in dict(o["explicit_field_values"]).items()}
        action_indices = [int(x) for x in o["action_indices"]]

        ev_b = _evaluate(ctx, action_indices, boosted_overrides={(block_idx, int(ctx.ref_layer)): explicit})
        ev_p = _evaluate(ctx, action_indices, boosted_overrides=None)
        cfg_b, ov_b = _install_and_inspect(ev_b, ctx)
        cfg_p, _ov_p = _install_and_inspect(ev_p, ctx)
        if cfg_b is None or cfg_p is None:
            print(f"[BAD] {graph_key} opt{oid}: target cfg not found in optimizer outputs")
            problems += 1
            continue

        proj_b = _cfg_sf_projection(cfg_b)
        proj_p = _cfg_sf_projection(cfg_p)
        sum_b = sum(proj_b.values())
        sum_p = sum(proj_p.values())

        # Q1: the model installs the BOOSTED (phase-2, higher-precision) group,
        # not the plain in-grid decode of the same option.
        q1 = sum_b > sum_p
        # Q2: when the option fuses, at least one rescale is nulled (fusion
        # processing applied to the installed group). Read straight off the
        # optimizer write-back overrides.
        fused = [e for e in ov_b if getattr(e, "source", "") == "rescale_fused_away"]
        q2 = (fc < 1) or (len(fused) >= 1)

        if not q1:
            print(f"[BAD] {graph_key} opt{oid}: Q1 FAIL — installed boosted sum {sum_b} "
                  f"!> in-grid sum {sum_p} (model not getting the boosted group)")
            problems += 1
        if not q2:
            print(f"[BAD] {graph_key} opt{oid}: Q2 FAIL — fusion_count={fc} but NO rescale "
                  f"nulled (rescale_fused_away) in the installed group")
            problems += 1
        if q1 and q2:
            print(f"[OK] {graph_key} opt{oid}: Q1 boosted-install sum {sum_p}->{sum_b} "
                  f"(+{sum_b - sum_p}); Q2 fused rescales={len(fused)} "
                  f"({', '.join(e.cfg_attr for e in fused) or 'n/a'})")
        checked += 1
    return checked, problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="mrpc")
    ap.add_argument("--rescale-optimizer-root", default="Rescale_optimizer")
    ap.add_argument("--maps-dir", default="blb_stage2_rl/fusion_maps/mrpc")
    ap.add_argument("--num-layers", type=int, default=12)
    args = ap.parse_args()

    mdir = pathlib.Path(args.maps_dir)
    files = sorted(p for p in mdir.glob("*.json") if not p.name.startswith("_"))
    total_checked = 0
    total_problems = 0
    for p in files:
        try:
            gk = json.loads(p.read_text(encoding="utf-8")).get("graph_key", p.stem)
        except Exception:
            gk = p.stem
        if _is_skipped_graph_key(gk):
            print(f"[skip] {gk}: degenerate/dormant map (no boosted fusion option)")
            continue
        c, pr = verify_map(p, args.profile, args.rescale_optimizer_root, args.num_layers)
        total_checked += c
        total_problems += pr

    print(f"\nchecked {total_checked} boosted option(s); {total_problems} problem(s)")
    if total_checked == 0:
        print("VERIFY_FAIL (no boosted option found — maps not phase-2-applied?)")
        return 1
    print("VERIFY_OK" if total_problems == 0 else f"VERIFY_FAIL ({total_problems} problems)")
    return 0 if total_problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
