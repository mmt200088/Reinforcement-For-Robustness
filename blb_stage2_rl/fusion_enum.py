"""Offline fusion-count map builder core (spec §3.2/§3.3/§3.4).

Two layers:

* **Pure core** (torch-free, locally testable): :func:`group_min_noise_options`
  takes already-evaluated per-block SF configs and produces the ordered option
  list — group by realized ``fusion_count``, keep the minimum-installed-noise set
  per group (dedup by installed plan), force ``option 0 = baseline`` by
  construction.

* **Enumeration driver** (server-only): :func:`build_block_type` enumerates the
  effective-chain slots of one block-type, runs real ``replan`` + optimizer
  override, computes installed variance, and feeds the pure core. Its torch /
  Rescale_optimizer imports are lazy (inside the function), so importing this
  module stays torch-free — mirrors ``blb_verify_noise_install.run_full``.

See docs/superpowers/specs/2026-06-03-stage2-fusion-count-action-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
from typing import Any, Dict, Hashable, List, Mapping, Sequence, Tuple

import numpy as np

try:  # torch-free test lane (blb_stage2_rl on sys.path)
    import fusion_count_map as fcm
except ImportError:  # package context
    from . import fusion_count_map as fcm  # type: ignore

InstalledNoisePoint = fcm.InstalledNoisePoint
NoiseOrder = fcm.NoiseOrder


# ---------------------------------------------------------------------------
# Pure core — group/dedup/order (torch-free, locally testable)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EvaluatedConfig:
    """One enumerated, valid SF config for a block-type after real replan.

    ``action_indices`` is the FULL per-block slot index vector (K left at the
    baseline placeholder). ``installed_signature`` is a hashable digest of the
    post-override installed noise plan, used to dedup configs that install the
    identical noise (and hence are interchangeable).
    """

    action_indices: Tuple[int, ...]
    fusion_count: int
    total_bits: int
    total_variance: float
    installed_signature: Hashable
    slots: Dict[str, int] = field(default_factory=dict)


def group_min_noise_options(
    evaluated: Sequence[EvaluatedConfig],
    baseline_action_indices: Sequence[int],
    *,
    noise_tol: float = 1e-18,
) -> List[Dict[str, Any]]:
    """Group valid configs by ``fusion_count``; per group keep the minimum-noise
    set (within ``noise_tol``, deduped by installed plan); return ordered options
    with ``option 0 == baseline`` by construction (spec §3.4).
    """
    baseline_key = tuple(int(x) for x in baseline_action_indices)

    by_fc: Dict[int, List[EvaluatedConfig]] = {}
    for ec in evaluated:
        by_fc.setdefault(int(ec.fusion_count), []).append(ec)

    kept: List[Tuple[EvaluatedConfig, int]] = []  # (config, tie_index)
    for _fc, members in by_fc.items():
        min_var = min(m.total_variance for m in members)
        at_min = [m for m in members if m.total_variance <= min_var + noise_tol]
        # dedup configs that install the identical noise plan: keep the cheapest
        # (lowest total_bits, then lexicographically smallest action vector).
        best_by_sig: Dict[Hashable, EvaluatedConfig] = {}
        for m in at_min:
            cur = best_by_sig.get(m.installed_signature)
            if cur is None or (m.total_bits, tuple(m.action_indices)) < (cur.total_bits, tuple(cur.action_indices)):
                best_by_sig[m.installed_signature] = m
        uniq = sorted(
            best_by_sig.values(),
            key=lambda m: (m.total_variance, m.total_bits, tuple(m.action_indices)),
        )
        for tie_idx, m in enumerate(uniq):
            kept.append((m, tie_idx))

    baseline_entry = next(((m, t) for (m, t) in kept if tuple(m.action_indices) == baseline_key), None)
    if baseline_entry is None:
        raise ValueError(
            "baseline action vector not found among the kept minimum-noise configs — "
            "baseline must be valid and the global minimum-variance config (spec §3.4)"
        )
    others = [(m, t) for (m, t) in kept if tuple(m.action_indices) != baseline_key]
    others.sort(
        key=lambda mt: (mt[0].fusion_count, mt[0].total_variance, mt[0].total_bits, tuple(mt[0].action_indices))
    )

    ordered = [baseline_entry, *others]
    options: List[Dict[str, Any]] = []
    for opt_id, (m, tie_idx) in enumerate(ordered):
        options.append(
            {
                "option_id": int(opt_id),
                "fusion_count": int(m.fusion_count),
                "tie_index": int(tie_idx),
                "total_variance": float(m.total_variance),
                "total_bits": int(m.total_bits),
                "slots": dict(m.slots),
                "action_indices": [int(x) for x in m.action_indices],
            }
        )
    return options


# ---------------------------------------------------------------------------
# Enumeration driver (server-only; torch + Rescale_optimizer imported lazily)
# ---------------------------------------------------------------------------
_NOISE_DISTS = frozenset({"fresh", "encoding", "rescale", "rotation"})


@dataclass
class BlockTypeBuildContext:
    """Everything one worker needs to enumerate one block-type's chain slots.

    Built once per process (each worker owns its own bridge — ``ReplanSession``
    is not picklable). ``enum_positions`` are the per-block slot indices that
    materially change the optimizer outcome (rescales + any non-rescale slot a
    full single-axis probe proves changes ``(fusion_count, total_bits)``);
    everything else is pinned at its baseline (max-SF = minimum noise).
    """

    graph_key: str
    block_idx: int
    profile: str
    num_layers: int
    ref_layer: int
    N_block: int
    bridge: Any
    max_sfs: Any
    baseline_full: np.ndarray
    block_offset: int
    block_num_slots: int
    k_slot_index: int
    gelu_per_layer: List[int]
    attn_per_layer: List[int]
    baseline_skeleton: List[Any]
    baseline_block_indices: Tuple[int, ...]
    enum_positions: List[int] = field(default_factory=list)
    enum_levels: List[int] = field(default_factory=list)
    pinned_positions: List[int] = field(default_factory=list)
    active_rescale_fields: List[str] = field(default_factory=list)

    def enum_total(self) -> int:
        total = 1
        for n in self.enum_levels:
            total *= int(n)
        return int(total)


def _installed_noise_points(cfg: Any, out_raw: Mapping, N_block: int) -> List[Any]:
    """Post-override installed Gaussian noise points (spec §3.3 / G7).

    Mirrors ``blb_verify_noise_install._enumerate_cfg_noise_points`` for the
    cfg fresh/encode/rescale fields, and additionally quantifies bound rotations
    from the optimizer's ``effective_rotations`` (which carry sf + count). K
    (truncation) has no noise distribution and is filtered out.
    """
    pts: List[Any] = []
    for name in vars(cfg):
        if name.startswith("rotation_after_") or name == "output_truncation_mode":
            continue
        value = getattr(cfg, name)
        candidates = value if isinstance(value, tuple) else (value,)
        for point in candidates:
            if point is None or not hasattr(point, "scaling_factor"):
                continue
            dist = str(getattr(point, "distribution", "")).lower()
            if dist not in _NOISE_DISTS:
                continue
            N = int(getattr(point, "N", 0) or N_block)
            pts.append(InstalledNoisePoint(int(point.scaling_factor), dist, N))
    compact = (out_raw or {}).get("new_compact_config") or {}
    for rot in compact.get("effective_rotations") or []:
        sf = rot.get("sf")
        if sf is None:
            continue
        count = int(rot.get("count", 1) or 1)
        for _ in range(max(1, count)):
            pts.append(InstalledNoisePoint(int(sf), "rotation", int(N_block)))
    return pts


def _installed_signature(points: Sequence[Any]) -> Tuple:
    return tuple(sorted((int(p.scaling_factor), str(p.distribution), int(p.N)) for p in points))


def _eval_block(ctx: BlockTypeBuildContext, block_indices: Sequence[int]) -> Dict[str, Any]:
    """Decode one block-slot vector exactly as the runtime env does, run real
    replan, and (if valid) return the post-override installed noise plan."""
    from action_space import action_vector_to_cfgs

    from rescale_optimizer_bridge import (
        apply_optimizer_output_to_cfg,
        sync_block2_aux_fresh_binding,
        sync_block2_qk_binding,
        sync_block4_v_mask_binding,
        sync_block5_aux_fresh_binding,
    )

    full = ctx.baseline_full.copy()
    full[ctx.block_offset : ctx.block_offset + ctx.block_num_slots] = np.asarray(block_indices, dtype=int)
    decoded = action_vector_to_cfgs(
        full,
        ctx.max_sfs,
        num_layers=ctx.num_layers,
        gelu_degree=ctx.gelu_per_layer,
        attn_degree=ctx.attn_per_layer,
    )
    cfg = decoded.cfgs_dict()[f"block{ctx.block_idx}"][ctx.ref_layer]
    out = ctx.bridge.evaluate(
        config_name=f"{ctx.graph_key}_L{ctx.ref_layer}",
        block_name=f"block{ctx.block_idx}",
        cfg=cfg,
    )
    if not bool(out.valid):
        return {"valid": False}
    apply_optimizer_output_to_cfg(
        cfg,
        output_raw=out.raw,
        block_idx=int(ctx.block_idx),
        graph_key=ctx.graph_key,
        baseline_skeleton=ctx.baseline_skeleton,
        rotation_name_map=None,
    )
    if ctx.block_idx == 2:
        sync_block2_qk_binding(cfg)
        sync_block2_aux_fresh_binding(cfg)
    elif ctx.block_idx == 4:
        sync_block4_v_mask_binding(cfg)
    elif ctx.block_idx == 5:
        sync_block5_aux_fresh_binding(cfg)
    points = _installed_noise_points(cfg, out.raw, ctx.N_block)
    return {
        "valid": True,
        "fusion_count": int(out.fusion_count),
        "total_bits": int(out.total_bits),
        "points": points,
    }


def prepare_block_type_context(
    *,
    graph_key: str,
    block_idx: int,
    gelu_degree: int,
    attn_degree: int,
    profile: str,
    rescale_optimizer_root: str,
    num_layers: int = 12,
    ref_layer: int = 1,
) -> BlockTypeBuildContext:
    """Bootstrap the calibrated baseline + bridge and classify which effective
    non-K slots to enumerate (vs pin at max). Uses the SAME calibrated max_sfs
    path as the runner so decoded SFs match runtime (avoids the generic
    ``load_max_sfs`` mismatch that makes degree-1 all-max look invalid)."""
    import json as _json
    import os as _os

    import action_space as _action_space
    from action_space import (
        _BLOCK_SPECS,
        NUM_LEVELS_PER_DIM_BY_BLOCK_KIND,
        _block_default_N,
        _full_vec_offset_for_block,
        _is_action_field_effective,
    )

    from rescale_optimizer_bridge import InProcessInvoker, RescaleOptimizerBridge

    try:
        from skeleton_stage_map import build_stage_plans_from_archive as _build_stage_plans
    except ImportError:
        from blb_stage2_rl.skeleton_stage_map import build_stage_plans_from_archive as _build_stage_plans

    try:
        from baseline_bootstrap import (
            load_static_skeletons_baseline,
            static_skeletons_baseline_to_action,
        )
    except ImportError:
        from blb_stage2_rl.baseline_bootstrap import (  # type: ignore
            load_static_skeletons_baseline,
            static_skeletons_baseline_to_action,
        )

    gelu_per_layer = [int(gelu_degree)] * int(num_layers)
    attn_per_layer = [int(attn_degree)] * int(num_layers)
    ss = load_static_skeletons_baseline(
        rescale_optimizer_root=str(rescale_optimizer_root),
        dataset=str(profile),
        num_layers=int(num_layers),
        gelu_per_layer=gelu_per_layer,
        softmax_per_layer=attn_per_layer,
    )
    baseline_full, max_sfs, _cost, _diag = static_skeletons_baseline_to_action(
        ss,
        snap_sf_to_noise_table=False,
    )
    baseline_full = np.asarray(baseline_full, dtype=int)

    invoker = InProcessInvoker.from_profile(
        rescale_optimizer_root=str(rescale_optimizer_root),
        profile=str(profile),
    )
    bridge = RescaleOptimizerBridge(invoker=invoker)
    baseline_entry = (getattr(invoker, "baselines", {}) or {}).get(graph_key)
    baseline_skeleton = list(baseline_entry[0]) if baseline_entry else []

    # Seed action_space's active-rescale cache from the EXPLICIT ro_root, so R-slot
    # effectiveness never depends on action_space's __file__-relative archive load
    # (``_load_active_rescale_sets`` silently returns {} on any path failure — the
    # server temp-dir build hit exactly that, judged every rescale non-effective,
    # never enumerated rescales, and produced rescale-free maps with fusion stuck
    # at 0). The bootstrap above already proved this ro_root resolves the archive.
    _arch_path = _os.path.join(str(rescale_optimizer_root), "configs", str(profile), f"static_skeletons_{profile}.json")
    with open(_arch_path, encoding="utf-8") as _f:
        _archive = _json.load(_f)
    _action_space._ACTIVE_RESCALE_SETS_CACHE = {
        gk: frozenset(p.active_rescale_rl_fields) for gk, p in _build_stage_plans(_archive).items()
    }
    active_rescale_fields = sorted(_action_space._ACTIVE_RESCALE_SETS_CACHE.get(graph_key, ()))
    if not active_rescale_fields:
        raise RuntimeError(
            f"{graph_key}: no active rescale slots derived from {_arch_path}; the fusion-count map "
            "would have no rescale lever (fusion stuck at 0). Check the archive / rescale_optimizer_root."
        )

    fields = _BLOCK_SPECS[int(block_idx)].fields
    block_num_slots = len(fields)
    block_offset = _full_vec_offset_for_block(int(num_layers), int(ref_layer), int(block_idx))
    k_slot_index = next(i for i, (_n, k, _m) in enumerate(fields) if k == "K")
    N_block = int(_block_default_N(int(block_idx), gelu_degree=int(gelu_degree), attn_degree=int(attn_degree)))

    ctx = BlockTypeBuildContext(
        graph_key=str(graph_key),
        block_idx=int(block_idx),
        profile=str(profile),
        num_layers=int(num_layers),
        ref_layer=int(ref_layer),
        N_block=N_block,
        bridge=bridge,
        max_sfs=max_sfs,
        baseline_full=baseline_full,
        block_offset=int(block_offset),
        block_num_slots=int(block_num_slots),
        k_slot_index=int(k_slot_index),
        gelu_per_layer=gelu_per_layer,
        attn_per_layer=attn_per_layer,
        baseline_skeleton=baseline_skeleton,
        baseline_block_indices=tuple(int(x) for x in baseline_full[block_offset : block_offset + block_num_slots]),
        active_rescale_fields=active_rescale_fields,
    )

    base_block = np.asarray(ctx.baseline_block_indices, dtype=int)
    base_res = _eval_block(ctx, base_block)
    if not base_res.get("valid"):
        raise RuntimeError(f"{graph_key}: baseline (all-max) block config is invalid under replan")
    base_key = (base_res["fusion_count"], base_res["total_bits"])

    # Classify each effective non-K slot: enumerate rescales always; for the
    # rest, full single-axis scan — pin at baseline (max) iff every level leaves
    # (fusion_count, total_bits) unchanged (slot affects only its own noise, so
    # its minimum-noise value is the max-SF baseline). Robust to non-monotonic
    # effects because it scans every level, not just one probe.
    for pos, (fname, kind, _maxsf) in enumerate(fields):
        if kind == "K":
            continue
        eff, _why = _is_action_field_effective(
            layer_idx=int(ref_layer),
            block_idx=int(block_idx),
            field_name=str(fname),
            attn_degree=int(attn_degree),
            gelu_degree=int(gelu_degree),
            profile=str(profile),
        )
        if not eff:
            continue
        levels = int(NUM_LEVELS_PER_DIM_BY_BLOCK_KIND[kind])
        if kind == "R":
            ctx.enum_positions.append(pos)
            ctx.enum_levels.append(levels)
            continue
        constant = True
        for lvl in range(levels):
            if lvl == int(base_block[pos]):
                continue
            probe = base_block.copy()
            probe[pos] = lvl
            pres = _eval_block(ctx, probe)
            if (not pres.get("valid")) or (pres["fusion_count"], pres["total_bits"]) != base_key:
                constant = False
                break
        if constant:
            ctx.pinned_positions.append(pos)
        else:
            ctx.enum_positions.append(pos)
            ctx.enum_levels.append(levels)
    return ctx


def enumerate_shard(
    ctx: BlockTypeBuildContext,
    *,
    shard_idx: int,
    num_shards: int,
    noise_order: Any,
) -> List[EvaluatedConfig]:
    """Enumerate this worker's stride of the chain-slot cartesian product."""
    base_block = np.asarray(ctx.baseline_block_indices, dtype=int)
    out: List[EvaluatedConfig] = []
    ranges = [range(n) for n in ctx.enum_levels]
    for i, combo in enumerate(itertools.product(*ranges)):
        if (i % int(num_shards)) != int(shard_idx):
            continue
        block = base_block.copy()
        for pos, lvl in zip(ctx.enum_positions, combo, strict=True):
            block[pos] = int(lvl)
        res = _eval_block(ctx, block)
        if not res.get("valid"):
            continue
        points = res["points"]
        out.append(
            EvaluatedConfig(
                action_indices=tuple(int(x) for x in block),
                fusion_count=int(res["fusion_count"]),
                total_bits=int(res["total_bits"]),
                total_variance=float(noise_order.total_variance(points)),
                installed_signature=_installed_signature(points),
                slots={},
            )
        )
    return out


def check_k_independence(
    ctx: BlockTypeBuildContext,
    *,
    sample_configs: Sequence[Sequence[int]],
) -> Dict[str, Any]:
    """For a few sample block configs, vary the K slot over all K levels and
    confirm ``fusion_count`` does not change (spec §3.6). K is decided
    separately, so the map (built at baseline K) is only valid if K does not
    move fusion."""
    from action_space import K_LEVELS

    violations: List[Dict[str, Any]] = []
    for cfg_indices in sample_configs:
        fusion_seen = set()
        for k_idx in range(len(K_LEVELS)):
            block = np.asarray(cfg_indices, dtype=int).copy()
            block[ctx.k_slot_index] = int(k_idx)
            res = _eval_block(ctx, block)
            if res.get("valid"):
                fusion_seen.add(int(res["fusion_count"]))
        if len(fusion_seen) > 1:
            violations.append(
                {
                    "action_indices": [int(x) for x in cfg_indices],
                    "fusion_counts_over_k": sorted(fusion_seen),
                }
            )
    return {"k_independent": not violations, "violations": violations, "samples_checked": len(list(sample_configs))}


def decode_block_slots(ctx: BlockTypeBuildContext, block_indices: Sequence[int]) -> Dict[str, int]:
    """SF/K-first human view {field_name: decoded SF} for one block config —
    used to populate the final options' ``slots`` (kept options only)."""
    from action_space import _BLOCK_SPECS, action_vector_to_cfgs

    full = ctx.baseline_full.copy()
    full[ctx.block_offset : ctx.block_offset + ctx.block_num_slots] = np.asarray(block_indices, dtype=int)
    decoded = action_vector_to_cfgs(
        full,
        ctx.max_sfs,
        num_layers=ctx.num_layers,
        gelu_degree=ctx.gelu_per_layer,
        attn_degree=ctx.attn_per_layer,
    )
    cfg = decoded.cfgs_dict()[f"block{ctx.block_idx}"][ctx.ref_layer]
    slots: Dict[str, int] = {}
    for fname, kind, _m in _BLOCK_SPECS[ctx.block_idx].fields:
        if kind == "K":
            continue
        value = getattr(cfg, fname, None)
        sf = getattr(value, "scaling_factor", None)
        if sf is not None:
            slots[str(fname)] = int(sf)
    return slots
