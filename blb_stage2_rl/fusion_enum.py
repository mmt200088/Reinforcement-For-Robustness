"""Offline fusion-count map builder core (spec §3.2/§3.3/§3.4).

Two layers:

* **Pure core** (torch-free, locally testable): :func:`group_min_noise_options`
  takes already-evaluated per-block SF configs and produces the ordered option
  list — group by realized ``fusion_count``, keep the minimum-installed-noise set
  per group (dedup by installed plan), order by (fusion, variance, bits, lex) so
  the all-max baseline lands at option 0 (it is the lowest-fusion global minimum
  once rescale-None is excluded from the enumeration).

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
    set (within ``noise_tol``, deduped by installed plan); order by (fusion,
    variance, bits, lex). The all-max baseline is the lowest-fusion global minimum
    (rescale-None excluded), so it lands at option 0; a guard asserts this (spec §3.4).
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

    # Order globally by (fusion asc, variance asc, bits asc, lexicographic). With
    # rescale-None excluded from the enumeration, the all-max baseline is the
    # lowest-fusion, globally-minimum-variance config, so it sorts to option 0.
    kept.sort(key=lambda mt: (mt[0].fusion_count, mt[0].total_variance, mt[0].total_bits, tuple(mt[0].action_indices)))
    options: List[Dict[str, Any]] = []
    for opt_id, (m, tie_idx) in enumerate(kept):
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
    # Loud guard: option 0 must be the baseline. A failure means a non-baseline
    # config reached lower fusion or lower installed variance than all-max — i.e.
    # the "baseline = lowest-fusion minimum-noise" invariant broke (re-examine the
    # noise order or the enumeration domain).
    if options and tuple(options[0]["action_indices"]) != baseline_key:
        raise ValueError(
            f"option 0 {options[0]['action_indices']} != baseline {list(baseline_key)}: the all-max baseline "
            "is not the lowest-fusion minimum-noise config (expected after excluding rescale-None from the enum)."
        )
    return options


class _MinNoiseReducer:
    """Streaming per-``fusion_count`` minimum-installed-variance reducer.

    Used INSIDE each shard worker so it retains only O(distinct min-variance
    signatures) configs instead of all its valid configs — block4's ~3e8 valid
    configs across 96 workers would otherwise OOM the box and blow multiprocessing
    pickle traffic. Keeps, per fusion_count, the configs within ``noise_tol`` of the
    running minimum (deduped by installed_signature, cheapest bits), resetting that
    fusion_count's set when a strictly-lower minimum (> tol below) appears.

    Soundness: the union of all shards' kept sets is a SUPERSET of the global
    minimum-variance set. A globally-at-min config c (variance ≤ G+tol) sits in some
    shard whose running min, when c is processed, is ≥ the shard's final min ≥ G, so
    c is within tol of the running min → kept; and no later config can reset it
    (any resetting config d has d.var >= shard_min >= G >= c.var - tol, so d cannot be
    > tol below the running min that c established). The final
    ``group_min_noise_options`` over the union therefore reproduces the exact result
    of grouping every valid config.
    """

    def __init__(self, noise_tol: float = 1e-18) -> None:
        self.noise_tol = float(noise_tol)
        self.num_valid = 0  # total configs seen (diagnostic; the kept set is small)
        self._min: Dict[int, float] = {}
        self._by_sig: Dict[int, Dict[Hashable, EvaluatedConfig]] = {}

    def add(self, ec: EvaluatedConfig) -> None:
        self.num_valid += 1
        fc = int(ec.fusion_count)
        cur = self._min.get(fc)
        if cur is None or ec.total_variance < cur - self.noise_tol:
            # strictly new minimum (or first) → this fc's old kept set is now > tol
            # above the new min and is discarded.
            self._min[fc] = ec.total_variance
            self._by_sig[fc] = {ec.installed_signature: ec}
            return
        if ec.total_variance <= cur + self.noise_tol:
            self._min[fc] = min(cur, ec.total_variance)
            bucket = self._by_sig[fc]
            prev = bucket.get(ec.installed_signature)
            if prev is None or (ec.total_bits, ec.action_indices) < (prev.total_bits, prev.action_indices):
                bucket[ec.installed_signature] = ec

    def results(self) -> List[EvaluatedConfig]:
        out: List[EvaluatedConfig] = []
        for bucket in self._by_sig.values():
            out.extend(bucket.values())
        return out


def _level_breaks_pin(probe_result: Mapping[str, Any], base_key: Tuple[int, int]) -> bool:
    """One probed slot level forces ENUMERATION (the slot cannot be pinned) iff it
    is invalid or changes the baseline ``(fusion_count, total_bits)``.

    The ``total_bits`` half is a build-time over-enumeration PROXY, NOT a reward
    term (bits left the reward 2026-06-03). It is required for soundness: fusion is
    driven by the JOINT lowering of several non-rescale encode SFs (committed
    ground-truth maps: each block's fusion>0 option lowers 2-4 encodes together
    while all rescales stay at baseline). No single encode moves fusion alone, so a
    fusion-only predicate returns False for every one of them, pins them all, and
    the map collapses to fusion={0}. Lowering any encode SF lowers total_bits, so
    the (fusion, bits) predicate keeps every such encode enumerated; the cartesian
    product over enumerated slots then recovers the joint fusion configs. See
    ``prepare_block_type_context`` for the full rationale.
    """
    if not probe_result.get("valid"):
        return True
    return (int(probe_result["fusion_count"]), int(probe_result["total_bits"])) != tuple(base_key)


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
    enum_choices: List[List[int]] = field(default_factory=list)
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
    # only=(ref_layer, block): the enumeration consumes exactly one cfg, and the
    # full 12-layer decode dominated per-combo cost (the replan itself is sub-ms).
    # Per-(layer, block) decode independence makes this bit-identical.
    decoded = action_vector_to_cfgs(
        full,
        ctx.max_sfs,
        num_layers=ctx.num_layers,
        gelu_degree=ctx.gelu_per_layer,
        attn_degree=ctx.attn_per_layer,
        only=(int(ctx.ref_layer), int(ctx.block_idx)),
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


def _eval_block_from_field_values(ctx: "BlockTypeBuildContext", field_values: Mapping[str, Any]) -> Dict[str, Any]:
    """Like ``_eval_block`` but builds the cfg SF-direct from explicit
    ``field_values`` (above-baseline SF allowed) — the precision boost evaluator.
    Returns valid / fusion_count / total_bits / q_initial / q_final / fusions /
    points (the chain fields come straight from the real replan result)."""
    from action_space import build_block_cfg_from_field_values

    from rescale_optimizer_bridge import (
        apply_optimizer_output_to_cfg,
        sync_block2_aux_fresh_binding,
        sync_block2_qk_binding,
        sync_block4_v_mask_binding,
        sync_block5_aux_fresh_binding,
    )

    li_gelu = int(ctx.gelu_per_layer[ctx.ref_layer])
    li_attn = int(ctx.attn_per_layer[ctx.ref_layer])
    cfg = build_block_cfg_from_field_values(
        int(ctx.block_idx), int(ctx.ref_layer), dict(field_values),
        N=int(ctx.N_block), gelu_degree=li_gelu, attn_degree=li_attn,
    )
    out = ctx.bridge.evaluate(
        config_name=f"{ctx.graph_key}_L{ctx.ref_layer}",
        block_name=f"block{ctx.block_idx}",
        cfg=cfg,
    )
    if not bool(out.valid):
        return {"valid": False}
    apply_optimizer_output_to_cfg(
        cfg, output_raw=out.raw, block_idx=int(ctx.block_idx), graph_key=ctx.graph_key,
        baseline_skeleton=ctx.baseline_skeleton, rotation_name_map=None,
    )
    if ctx.block_idx == 2:
        sync_block2_qk_binding(cfg)
        sync_block2_aux_fresh_binding(cfg)
    elif ctx.block_idx == 4:
        sync_block4_v_mask_binding(cfg)
    elif ctx.block_idx == 5:
        sync_block5_aux_fresh_binding(cfg)
    points = _installed_noise_points(cfg, out.raw, ctx.N_block)
    r = (out.raw or {}).get("result", {}) or {}
    return {
        "valid": True,
        "fusion_count": int(out.fusion_count),
        "total_bits": int(out.total_bits),
        "q_initial": tuple(int(x) for x in r.get("q_initial", ())),
        "q_final": tuple(int(x) for x in r.get("q_final", ())),
        "fusions": tuple(r.get("fusions", ())),
        "points": points,
    }


def boost_options_for_block(ctx: "BlockTypeBuildContext", options: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply the precision boost ("加大精度") to each non-zero-fusion option.

    For every option with ``fusion_count != 0`` whose block-type has a registered
    ``ChainTopology``: raise its short modulus primes to ``q_max`` at minimum
    installed noise (``precision_boost.boost_option``, replan-verified), and rewrite
    the option as a boosted explicit-SF option (``boosted=True`` +
    ``explicit_field_values``). Options with no topology, no short prime, or no
    valid all-``q_max`` candidate are left unchanged. ``option0`` (fc=0) is never
    touched. Mutates + returns ``options``.
    """
    try:
        import fusion_count_map as _fcm
        import precision_boost as _pb
    except ImportError:
        from . import fusion_count_map as _fcm, precision_boost as _pb

    from action_space import _decode_block_field_values

    topo = _pb.TOPOLOGIES.get(ctx.graph_key)
    if topo is None:
        return options  # block-type topology not registered yet (e.g. block4/5 TBD)
    noise_order = _fcm.SummedInstalledVariance()
    li_gelu = int(ctx.gelu_per_layer[ctx.ref_layer])
    li_attn = int(ctx.attn_per_layer[ctx.ref_layer])

    def _make_probe(slots: Mapping[str, Any]) -> Any:
        r = _eval_block_from_field_values(ctx, slots)
        if not r.get("valid"):
            return _pb.ReplanProbe(valid=False, fusion_count=0, q_initial=(), q_final=())
        return _pb.ReplanProbe(
            valid=True, fusion_count=int(r["fusion_count"]),
            q_initial=r["q_initial"], q_final=r["q_final"], fusions=r["fusions"],
            extra=r["points"],
        )

    def _noise(_slots: Mapping[str, Any], probe: Any) -> float:
        return float(noise_order.total_variance(probe.extra))

    for opt in options:
        if int(opt.get("fusion_count", 0)) == 0:
            continue
        base_fv_raw = _decode_block_field_values(
            layer_idx=int(ctx.ref_layer), block_idx=int(ctx.block_idx),
            action_slice=np.asarray(opt["action_indices"], dtype=int),
            max_sfs=ctx.max_sfs, attn_degree=li_attn, gelu_degree=li_gelu,
        )
        # keep ints (incl. output_truncation_k); drop inactive-rescale None fields
        # (their _build_block*_action reads are guarded by the active set).
        base_fv = {k: int(v) for k, v in base_fv_raw.items() if v is not None}
        res = _pb.boost_option(
            topology=topo, base_slots=base_fv,
            replan_fn=_make_probe, noise_fn=_noise, q_max=int(topo.q_max),
        )
        if res is None:
            continue
        final = _eval_block_from_field_values(ctx, res.boosted_slots)
        # hard build-time guard: a stored boost MUST be valid, keep fusion_count,
        # and have every prime at q_max (the boost_option contract — re-checked
        # here against an independent replan so a broken topology aborts the build).
        if (
            not final.get("valid")
            or int(final.get("fusion_count", -1)) != int(opt["fusion_count"])
            or any(int(q) != int(topo.q_max) for q in final.get("q_final", ()))
        ):
            raise RuntimeError(
                f"{ctx.graph_key}: precision boost produced an inconsistent option "
                f"(fc={final.get('fusion_count')} q_final={final.get('q_final')} "
                f"expected fc={opt['fusion_count']} all={topo.q_max}); aborting build"
            )
        opt["boosted"] = True
        opt["explicit_field_values"] = {k: int(v) for k, v in res.boosted_slots.items()}
        opt["total_variance"] = float(res.total_variance)
        opt["total_bits"] = int(final.get("total_bits", opt.get("total_bits", 0)))
        opt["boost_description"] = str(res.description)
        # refresh the human SF view for the topology's named slots
        slots_view = opt.get("slots") or {}
        for node in topo.nodes:
            if node.cfg_field and node.cfg_field in res.boosted_slots and node.cfg_field in slots_view:
                slots_view[node.cfg_field] = int(res.boosted_slots[node.cfg_field])
        opt["slots"] = slots_view
    return options


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
        distinct_sf_level_indices,
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
    base_key = (int(base_res["fusion_count"]), int(base_res["total_bits"]))

    # Classify each effective non-K slot.
    #
    # Rescale (R) slots: enumerate SF VALUES ONLY (action indices 1..levels-1),
    # never index 0 (= None = "drop this rescale"). Per the CKKS mental model
    # (CLAUDE.md item 2) RL never decides whether a must-exist operation happens —
    # the rescale points are fixed; the optimizer decides fusion via its response
    # to the SF schedule. Enumerating index 0 let RL "drop" a rescale, which the
    # optimizer honoured into a strictly-lower-noise / same-bits config that
    # dominated the all-max baseline (2026-06-03 block1 crash).
    #
    # Non-rescale (F/W/M/S) slots: single-axis scan — pin at baseline (max) iff
    # every level keeps the baseline ``(fusion_count, total_bits)``. The total_bits
    # half is a build-time over-enumeration PROXY, not a reward term (bits left the
    # reward 2026-06-03). It is REQUIRED for soundness: the committed ground-truth
    # maps show every block's fusion>0 option is reached by lowering 2-4 non-rescale
    # ENCODE SFs *jointly* while ALL rescales stay at baseline (e.g. block2 fc=1 =
    # {inv_std_fresh 28->20, gamma 20->16, wk 22->16}; every block5_n* the same).
    # No single encode moves fusion alone, so a fusion-only solo-probe pins every
    # one of them and the map collapses to fusion={0} — the 2026-06-04 relaxation
    # did exactly that and is REVERTED here. Because lowering any encode SF lowers
    # total_bits, the (fusion, bits) probe keeps all such encodes in the enumerated
    # set, and the downstream cartesian product over enumerated slots recovers the
    # joint fusion configs. Pinning is thus restricted to slots that move NEITHER
    # fusion nor bits (effectively inert). Scans every level (robust to non-monotone
    # effects). Do NOT re-relax this to fusion-only without enumerating slots jointly.
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
        # Acceleration (result-equivalent): enumerate only DISTINCT decoded
        # values per slot. Under the hybrid sweep a low-baseline slot's deep
        # levels all snap to the table-min SF=10 and decode to identical cfgs
        # → identical replans; the lowest index per value is kept, which is
        # exactly the lex-min representative the post-eval installed-signature
        # dedup would have selected, so the emitted options (incl. their
        # action_indices) match a full enumeration — only the valid_configs
        # diagnostic count shrinks. R slots exclude idx 0 (= None/drop).
        field_max_sf = int(ctx.max_sfs.get(int(block_idx), str(fname), layer_idx=int(ref_layer)))
        distinct = distinct_sf_level_indices(
            kind=str(kind), levels=levels, max_sf=field_max_sf, N=int(ctx.N_block),
        )
        if kind == "R":
            ctx.enum_positions.append(pos)
            ctx.enum_choices.append(list(distinct))
            ctx.enum_levels.append(len(distinct))
            continue
        constant = True
        for lvl in distinct:
            if lvl == int(base_block[pos]):
                continue
            probe = base_block.copy()
            probe[pos] = lvl
            if _level_breaks_pin(_eval_block(ctx, probe), base_key):
                constant = False
                break
        if constant:
            ctx.pinned_positions.append(pos)
        else:
            ctx.enum_positions.append(pos)
            ctx.enum_choices.append(list(distinct))
            ctx.enum_levels.append(len(distinct))
    return ctx


def enumerate_shard(
    ctx: BlockTypeBuildContext,
    *,
    shard_idx: int,
    num_shards: int,
    noise_order: Any,
    noise_tol: float = 1e-18,
) -> Tuple[List[EvaluatedConfig], int]:
    """Enumerate this worker's stride of the chain-slot cartesian product, returning
    ``(per-fusion_count minimum-installed-variance set, num_valid_seen)``.

    The streaming :class:`_MinNoiseReducer` means a worker holds O(distinct min-var
    signatures) configs, not all its valid configs — mandatory for block4 (~3e8
    valid across 96 workers would OOM the box + blow pickle traffic). The reduced
    set is a superset of the global minimum, so the main-process
    ``group_min_noise_options`` over all shards' sets is still exact. ``num_valid``
    is returned separately so the build still reports the true valid count.
    """
    base_block = np.asarray(ctx.baseline_block_indices, dtype=int)
    reducer = _MinNoiseReducer(noise_tol=float(noise_tol))
    for i, combo in enumerate(itertools.product(*ctx.enum_choices)):
        if (i % int(num_shards)) != int(shard_idx):
            continue
        block = base_block.copy()
        for pos, lvl in zip(ctx.enum_positions, combo, strict=True):
            block[pos] = int(lvl)
        res = _eval_block(ctx, block)
        if not res.get("valid"):
            continue
        points = res["points"]
        reducer.add(
            EvaluatedConfig(
                action_indices=tuple(int(x) for x in block),
                fusion_count=int(res["fusion_count"]),
                total_bits=int(res["total_bits"]),
                total_variance=float(noise_order.total_variance(points)),
                installed_signature=_installed_signature(points),
                slots={},
            )
        )
    return reducer.results(), reducer.num_valid


def degeneracy_probe(
    ctx: BlockTypeBuildContext,
    *,
    num_random: int = 2000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Cheap evidence that a block-type is fusion-degenerate over its enumerated
    grid — for blocks too large to enumerate fully.

    The (fusion, total_bits) classification deliberately over-enumerates encodes
    that move bits but not fusion (e.g. block4's ~1e6 encode combos that, per the
    near-exhaustive committed map, never fuse). When such a block exceeds the build
    budget, this probe decides whether the all-max baseline is its only option.

    Evaluates the all-MIN-SF corner (the maximally-fused config under the monotone
    "lower SF => more fusion" structure of rescale fusion — pulling EVERY lever to
    its deepest level) plus ``num_random`` uniform samples of the enumerated
    cartesian product. ``degenerate`` is True iff every valid probe keeps the
    baseline ``fusion_count``. A True result is strong (the corner maximally fuses);
    a False result is conclusive (a concrete fusing config exists) and must block
    any degenerate shortcut.
    """
    base_block = np.asarray(ctx.baseline_block_indices, dtype=int)
    base_res = _eval_block(ctx, base_block)
    base_fc = int(base_res["fusion_count"]) if base_res.get("valid") else None

    def _with(combo: Sequence[int]) -> np.ndarray:
        blk = base_block.copy()
        for pos, lvl in zip(ctx.enum_positions, combo, strict=True):
            blk[pos] = int(lvl)
        return blk

    corner = [ch[0] for ch in ctx.enum_choices]  # deepest level of every enum slot
    corner_res = _eval_block(ctx, _with(corner))
    rng = np.random.default_rng(int(seed))
    probes = [corner] + [[int(rng.choice(ch)) for ch in ctx.enum_choices] for _ in range(int(num_random))]

    fusion_seen: set = set()
    checked = 0
    for combo in probes:
        r = _eval_block(ctx, _with(combo))
        if r.get("valid"):
            fusion_seen.add(int(r["fusion_count"]))
            checked += 1
    degenerate = (base_fc is not None) and fusion_seen.issubset({base_fc})
    return {
        "degenerate": bool(degenerate),
        "base_fc": base_fc,
        "fusion_seen": sorted(fusion_seen),
        "corner_valid": bool(corner_res.get("valid")),
        "corner_fusion": int(corner_res["fusion_count"]) if corner_res.get("valid") else None,
        "samples_checked": int(checked),
        "num_random": int(num_random),
    }


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
    """SF/K-first human view {action_field_name: decoded SF} for one block config.

    Decodes each effective non-K slot's action index straight to its scaling
    factor via ``_field_level_values`` (the canonical per-slot decode, using the
    calibrated max_sfs). This avoids the action-field-name vs cfg-attr-name
    mismatch that previously left ``slots`` empty.
    """
    from action_space import (
        _BLOCK_SPECS,
        NUM_LEVELS_PER_DIM_BY_BLOCK_KIND,
        _block_default_N,
        _field_level_values,
        _is_action_field_effective,
    )

    gelu = int(ctx.gelu_per_layer[ctx.ref_layer])
    attn = int(ctx.attn_per_layer[ctx.ref_layer])
    N = int(_block_default_N(ctx.block_idx, gelu_degree=gelu, attn_degree=attn))
    slots: Dict[str, int] = {}
    for pos, (fname, kind, _m) in enumerate(_BLOCK_SPECS[ctx.block_idx].fields):
        if kind == "K":
            continue
        eff, _why = _is_action_field_effective(
            layer_idx=ctx.ref_layer,
            block_idx=ctx.block_idx,
            field_name=str(fname),
            attn_degree=attn,
            gelu_degree=gelu,
            profile=ctx.profile,
        )
        if not eff:
            continue
        idx = int(block_indices[pos])
        max_sf = int(ctx.max_sfs.get(ctx.block_idx, str(fname), layer_idx=ctx.ref_layer))
        levels = int(NUM_LEVELS_PER_DIM_BY_BLOCK_KIND[kind])
        vals = _field_level_values(kind=kind, levels=levels, max_sf=max_sf, N=N)
        sf = vals[idx] if 0 <= idx < len(vals) else None
        if sf is not None:
            slots[str(fname)] = int(sf)
    return slots
