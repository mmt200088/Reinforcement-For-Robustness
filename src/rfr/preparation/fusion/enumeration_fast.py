"""Direct-replan fast path for fusion-count map enumeration.

Why: a single in-process ``ReplanSession.replan`` costs ~0.06 ms, but the
golden per-combo pipeline (``fusion_enum._eval_block``) pays ~10x that in
intermediates — cfg-object construction (``action_vector_to_cfgs``), the
bridge's per-call cache key + ``copy.deepcopy`` store (the enumeration's
combos are all distinct, so that cache can never hit), ``RescaleOptimizerOutput``
wrapping, optimizer-output→cfg write-back, and cfg introspection for the
installed-noise points. Under the step-1 x15 grid the full
enumeration is ~3-6e9 combos, so those intermediates are the difference
between ~1 hour and ~10-30 hours.

What: a per-block-type **template** is derived once in the main process
(torch side, against the SAME golden machinery), then a torch-free **hot
loop** evaluates combos by patching ``(t_new, delta_overrides)`` directly,
calling ``ReplanSession.replan`` (the exact function the golden path bottoms
out in — same DEFAULT_FUSION_POLICY, same math), and assembling the installed
noise plan straight from the raw output.

Correctness contract:

* the template's slot→input wiring is DERIVED, not hand-written: each enum
  slot is probed through the golden decode at two levels and the produced
  ``(t_new, delta_overrides)`` diffs must equal the decoded SF exactly;
  Q/K-style mirrors are discovered by sentinel probes through the SAME
  ``sync_block*`` functions the golden path calls;
* ``verify_template`` (mandatory before enumeration; the builder aborts on
  any mismatch) evaluates baseline + all-min corner + N random combos through
  BOTH paths and requires exact equality of (valid, fusion_count, total_bits,
  installed-signature, total_variance);
* invalid combos short-circuit exactly like the golden path (no points).

GPU note: the evaluation core is ``ReplanSession`` — pure-Python integer
modulus-chain planning. It cannot run on CUDA without reimplementing the
cost source of truth (forbidden: every cost number must come from real
replan), so the parallel axis is CPU processes; this module's job is to make
each process's combo loop as close to bare ``replan`` cost as possible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import count_map as fcm
from . import enumeration as fusion_enum

import noise_tables

InstalledNoisePoint = fcm.InstalledNoisePoint
EvaluatedConfig = fusion_enum.EvaluatedConfig
_NOISE_DISTS = fusion_enum._NOISE_DISTS


def unrank_combo(choices_lens: Sequence[int], rank: int) -> List[int]:
    """Mixed-radix unranking: rank → per-slot choice positions (row-major,
    last slot fastest — identical ordering to ``itertools.product``)."""
    out = [0] * len(choices_lens)
    r = int(rank)
    for i in range(len(choices_lens) - 1, -1, -1):
        base = int(choices_lens[i])
        out[i] = r % base
        r //= base
    return out


def iter_combo_range(choices_lens: Sequence[int], start: int, stop: int):
    """Yield per-slot choice-position vectors for ranks [start, stop) in
    ``itertools.product`` order, via an O(1)-amortized odometer (each worker
    touches only its own contiguous range — no skip-spinning over the full
    product, which at 1e9+ combos costs real minutes per worker)."""
    n = len(choices_lens)
    lens = [int(x) for x in choices_lens]
    cur = unrank_combo(lens, start)
    for _ in range(int(stop) - int(start)):
        yield cur
        for i in range(n - 1, -1, -1):
            cur[i] += 1
            if cur[i] < lens[i]:
                break
            cur[i] = 0


@dataclass
class PointSpec:
    """How one installed Gaussian point's SF is resolved per combo.

    kind:
      * "source"      — cut_point_sf entry of skeleton position 0, key "sf".
      * "rescale"     — skeleton position ``skel_pos``: absent from
                        cut_point_sf → fused away (no point); "sf_post"
                        present → that value; present without "sf_post" →
                        the PROPOSED value (= the combo value of ``slot_idx``).
      * "encode"      — propagation_deltas[name]: int delta → that value,
                        else the proposed value (slot combo value or const).
      * "slot"        — always the combo value of ``slot_idx`` (model-noise
                        -only field with no graph node, e.g. block2 wv).
      * "const"       — fixed baseline SF (pinned / non-action field).
    """

    kind: str
    distribution: str
    N: int
    skel_pos: int = -1
    node: str = ""
    slot_idx: int = -1
    const_sf: int = 0


@dataclass
class DeltaTermSpec:
    """One term in a derived propagation delta.

    Most graph deltas are fed by a single action slot. Block4 has one compound
    CTCT delta, ``ctct_rot_softmax_mul_v``, whose value is the accumulated SF of
    two inputs (V fresh + mask2). That must be recomputed from the current combo
    instead of treated as a single-slot identity target.
    """

    slot_idx: int = -1
    const_sf: int = 0


@dataclass
class DerivedDeltaSpec:
    node: str
    terms: List[DeltaTermSpec]


@dataclass
class FastEnumTemplate:
    graph_key: str
    block_idx: int
    n_block: int
    baseline_t_new: List[int]
    baseline_deltas: Dict[str, Any]
    skeleton_node_ids: List[int]

    slot_t_targets: List[List[int]]
    slot_d_targets: List[List[str]]
    slot_choice_sfs: List[List[int]]
    enum_positions: List[int]
    enum_choices: List[List[int]]
    baseline_block_indices: List[int]
    points: List[PointSpec] = field(default_factory=list)
    derived_deltas: List[DerivedDeltaSpec] = field(default_factory=list)


def _build_var_lut() -> Dict[Tuple[int, str], Dict[int, float]]:
    lut: Dict[Tuple[int, str], Dict[int, float]] = {}
    for N in noise_tables.ALLOWED_N:
        for dist in ("fresh", "encoding", "rescale", "rotation"):
            per: Dict[int, float] = {}
            for sf in noise_tables.ALLOWED_SCALING_FACTORS_BY_N.get(N, ()):  # type: ignore[attr-defined]
                try:
                    per[int(sf)] = float(noise_tables.variance(int(N), int(sf), dist))
                except Exception:
                    continue
            lut[(int(N), dist)] = per
    return lut


def _var_of(lut: Mapping[Tuple[int, str], Mapping[int, float]], N: int, sf: int, dist: str) -> float:
    per = lut.get((int(N), str(dist)))
    if per is not None:
        v = per.get(int(sf))
        if v is not None:
            return v


    return float(noise_tables.variance(int(N), int(sf), str(dist)))


def eval_combo_fast(
    template: FastEnumTemplate,
    session: Any,
    combo_positions: Sequence[int],
    *,
    var_lut: Optional[Mapping[Tuple[int, str], Mapping[int, float]]] = None,
) -> Dict[str, Any]:
    """Evaluate ONE combo through the direct replan path.

    Returns the same shape ``fusion_enum._eval_block`` produces:
    ``{"valid": False}`` or ``{"valid": True, "fusion_count", "total_bits",
    "points": [InstalledNoisePoint, ...]}`` — plus ``"_sf"`` (per-slot decoded
    SFs) for diagnostics.
    """
    t_new = list(template.baseline_t_new)
    deltas = dict(template.baseline_deltas)
    sf_by_slot: List[int] = [0] * len(template.slot_choice_sfs)
    for e, cpos in enumerate(combo_positions):
        sf = int(template.slot_choice_sfs[e][int(cpos)])
        sf_by_slot[e] = sf
        for ti in template.slot_t_targets[e]:
            t_new[ti] = sf
        for dk in template.slot_d_targets[e]:
            deltas[dk] = sf
    for spec in template.derived_deltas:
        total = 0
        for term in spec.terms:
            if int(term.slot_idx) >= 0:
                total += int(sf_by_slot[int(term.slot_idx)])
            else:
                total += int(term.const_sf)
        deltas[str(spec.node)] = int(total)

    raw = session.replan_compact(
        template.graph_key, t_new=t_new, delta_overrides=deltas,
    )

    if not raw.valid:
        return {"valid": False}
    total_bits = int(raw.total_bits)
    fusion_count = int(raw.fusion_count)

    compact = raw.compact_config or {}
    cut_points: Dict[int, Mapping[str, Any]] = {}
    for entry in compact.get("cut_point_sf", []) or []:
        if isinstance(entry, Mapping) and "i" in entry:
            cut_points[int(entry["i"])] = entry
    prop: Dict[str, Any] = {}
    for entry in compact.get("propagation_deltas", []) or []:
        if isinstance(entry, Mapping):
            prop[str(entry.get("name", ""))] = entry.get("delta")

    lut = var_lut if var_lut is not None else _VAR_LUT
    skel = template.skeleton_node_ids
    points: List[InstalledNoisePoint] = []
    for spec in template.points:
        if spec.kind == "source":
            cpt = cut_points.get(int(skel[0])) if skel else None
            sf = int(cpt["sf"]) if (cpt is not None and "sf" in cpt) else (
                sf_by_slot[spec.slot_idx] if spec.slot_idx >= 0 else spec.const_sf
            )
        elif spec.kind == "rescale":
            node_id = int(skel[spec.skel_pos])
            cpt = cut_points.get(node_id)
            if cpt is None:
                continue
            if "sf_post" in cpt:
                sf = int(cpt["sf_post"])
            else:
                sf = sf_by_slot[spec.slot_idx] if spec.slot_idx >= 0 else spec.const_sf
        elif spec.kind == "encode":
            delta = prop.get(spec.node)
            if isinstance(delta, int):
                sf = int(delta)
            else:
                sf = sf_by_slot[spec.slot_idx] if spec.slot_idx >= 0 else spec.const_sf
        elif spec.kind == "slot":
            sf = sf_by_slot[spec.slot_idx]
        else:
            sf = spec.const_sf
        points.append(InstalledNoisePoint(int(sf), spec.distribution, int(spec.N)))

    for entry in compact.get("effective_rotations", []) or []:
        if not isinstance(entry, Mapping):
            continue
        sf = entry.get("sf")
        if sf is None:
            continue
        count = int(entry.get("count", 1) or 1)
        for _ in range(max(1, count)):
            points.append(InstalledNoisePoint(int(sf), "rotation", int(template.n_block)))

    total_variance = 0.0
    for p in points:
        total_variance += _var_of(lut, p.N, p.scaling_factor, p.distribution)

    return {
        "valid": True,
        "fusion_count": fusion_count,
        "total_bits": total_bits,
        "points": points,
        "total_variance": float(total_variance),
        "_sf": list(sf_by_slot),
    }


_VAR_LUT = _build_var_lut()


def enumerate_range_fast(
    template: FastEnumTemplate,
    session: Any,
    *,
    start: int,
    stop: int,
    noise_tol: float = 1e-18,
) -> Tuple[List[EvaluatedConfig], int]:
    """Enumerate combo ranks [start, stop) through the fast path, reduced via
    the same streaming per-fusion_count min-variance reducer the golden shards
    use (union of range results == union of stride results == exact)."""
    reducer = fusion_enum._MinNoiseReducer(noise_tol=float(noise_tol))
    base_block = list(template.baseline_block_indices)
    choices = template.enum_choices
    positions = template.enum_positions
    lens = [len(c) for c in choices]
    for combo_pos in iter_combo_range(lens, start, stop):
        res = eval_combo_fast(template, session, combo_pos, var_lut=_VAR_LUT)
        if not res.get("valid"):
            continue
        block = list(base_block)
        for e, cpos in enumerate(combo_pos):
            block[positions[e]] = int(choices[e][int(cpos)])
        reducer.add(
            EvaluatedConfig(
                action_indices=tuple(int(x) for x in block),
                fusion_count=int(res["fusion_count"]),
                total_bits=int(res["total_bits"]),
                total_variance=float(res["total_variance"]),
                installed_signature=fusion_enum._installed_signature(res["points"]),
                slots={},
            )
        )
    return reducer.results(), reducer.num_valid


def _walk_cfg_points_with_attrs(cfg: Any, n_block: int) -> List[Tuple[str, int, str, int]]:
    """[(attr_path, sf, distribution, N)] over the cfg's noise points.

    Field rules mirror ``fusion_enum._installed_noise_points`` exactly (skip
    rotation flags / truncation mode; tuple fields expand per member; only
    fresh/encoding/rescale/rotation distributions count).
    """
    out: List[Tuple[str, int, str, int]] = []
    for name in vars(cfg):
        if name.startswith("rotation_after_") or name == "output_truncation_mode":
            continue
        value = getattr(cfg, name)
        candidates = value if isinstance(value, tuple) else (value,)
        for j, point in enumerate(candidates):
            if point is None or not hasattr(point, "scaling_factor"):
                continue
            dist = str(getattr(point, "distribution", "")).lower()
            if dist not in _NOISE_DISTS:
                continue
            attr = f"{name}[{j}]" if isinstance(value, tuple) else name
            N = int(getattr(point, "N", 0) or n_block)
            out.append((attr, int(point.scaling_factor), dist, N))
    return out


def build_fast_template(ctx: Any) -> FastEnumTemplate:
    """Derive the direct-replan template from a prepared golden context.

    Probes every enum slot through the GOLDEN decode at two levels and diffs
    the produced ``(t_new, delta_overrides)`` — the wiring is discovered from
    the same code that drives the golden path, never hand-written. Mirrors
    (block2 Q←K etc.) are discovered by sentinel probes through the same
    ``sync_block*`` functions. Raises on any wiring that does not behave as
    an exact identity map of the decoded SF.
    """
    from blb_stage2_rl.action_space import (
        NUM_LEVELS_PER_DIM_BY_BLOCK_KIND,
        _BLOCK_SPECS,
        _field_level_values,
        action_vector_to_cfgs,
    )

    from rescale_optimizer_bridge import (
        GRAPH_NODE_TO_CFG_ATTR,
        cfg_to_t_new_from_table,
        sync_block2_aux_fresh_binding,
        sync_block2_qk_binding,
        sync_block4_v_mask_binding,
        sync_block5_aux_fresh_binding,
    )

    bridge = ctx.bridge
    graph = str(ctx.graph_key)
    blk = int(ctx.block_idx)
    bname = f"block{blk}"
    baseline_t = None
    try:
        baseline_t = bridge._lookup_baseline_t_new(graph)
    except Exception:
        baseline_t = None

    def _decode(block_indices: Sequence[int]):
        full = ctx.baseline_full.copy()
        full[ctx.block_offset: ctx.block_offset + ctx.block_num_slots] = np.asarray(
            block_indices, dtype=int
        )
        decoded = action_vector_to_cfgs(
            full, ctx.max_sfs, num_layers=ctx.num_layers,
            gelu_degree=ctx.gelu_per_layer, attn_degree=ctx.attn_per_layer,
            only=(int(ctx.ref_layer), blk),
        )
        return decoded.cfgs_dict()[bname][ctx.ref_layer]

    def _inputs(cfg) -> Tuple[List[int], Dict[str, Any]]:
        t = cfg_to_t_new_from_table(
            graph, cfg, baseline_t_new=baseline_t, table=bridge._cfg_to_t_new_table,
        )
        d = bridge.cfg_to_delta_overrides(bname, cfg)
        return list(t or []), dict(d)

    base_block = list(int(x) for x in ctx.baseline_block_indices)
    base_cfg = _decode(base_block)
    t0, d0 = _inputs(base_cfg)
    pts0 = _walk_cfg_points_with_attrs(base_cfg, int(ctx.N_block))

    fields = _BLOCK_SPECS[blk].fields
    n_slots = len(ctx.enum_positions)
    slot_t_targets: List[List[int]] = []
    slot_d_targets: List[List[str]] = []
    slot_choice_sfs: List[List[int]] = []
    attr_to_slot: Dict[str, int] = {}
    field_to_slot: Dict[str, int] = {}
    derived_delta_names = (
        {"ctct_rot_softmax_mul_v"}
        if blk == 4 and "ctct_rot_softmax_mul_v" in d0
        else set()
    )

    for e in range(n_slots):
        pos = int(ctx.enum_positions[e])
        fname, kind, _maxsf = fields[pos]
        levels = int(NUM_LEVELS_PER_DIM_BY_BLOCK_KIND[kind])
        max_sf = int(ctx.max_sfs.get(blk, str(fname), layer_idx=int(ctx.ref_layer)))
        vals = _field_level_values(kind=str(kind), levels=levels, max_sf=max_sf, N=int(ctx.N_block))
        choice_sfs = [int(vals[idx]) for idx in ctx.enum_choices[e]]
        slot_choice_sfs.append(choice_sfs)
        field_to_slot[str(fname)] = int(e)

        probe_levels = [c for c in ctx.enum_choices[e] if c != base_block[pos]][:2]
        if not probe_levels:
            raise RuntimeError(f"{graph}: enum slot {fname} has no non-baseline level to probe")
        t_targets: Optional[List[int]] = None
        d_targets: Optional[List[str]] = None
        for lvl in probe_levels:
            probe = list(base_block)
            probe[pos] = int(lvl)
            cfg_v = _decode(probe)
            t_v, d_v = _inputs(cfg_v)
            expect = int(vals[int(lvl)])
            tt = [i for i in range(len(t0)) if t_v[i] != t0[i]]
            dd = [
                k for k in d0
                if k not in derived_delta_names and d_v.get(k) != d0[k]
            ]
            for i in tt:
                if int(t_v[i]) != expect:
                    raise RuntimeError(
                        f"{graph}: slot {fname} level {lvl}: t_new[{i}]={t_v[i]} != decoded SF {expect}"
                    )
            for k in dd:
                if int(d_v[k]) != expect:
                    raise RuntimeError(
                        f"{graph}: slot {fname} level {lvl}: delta[{k}]={d_v[k]} != decoded SF {expect}"
                    )
            if t_targets is None:
                t_targets, d_targets = tt, dd
            elif (tt, dd) != (t_targets, d_targets):
                raise RuntimeError(
                    f"{graph}: slot {fname}: inconsistent wiring across levels "
                    f"({(tt, dd)} vs {(t_targets, d_targets)})"
                )

            pts_v = _walk_cfg_points_with_attrs(cfg_v, int(ctx.N_block))
            if len(pts_v) != len(pts0):
                raise RuntimeError(
                    f"{graph}: slot {fname}: cfg point count changed across decode "
                    f"({len(pts_v)} vs {len(pts0)})"
                )
            for (attr, sf, _dist, _N), (attr0, sf0, _dist0, _N0) in zip(pts_v, pts0, strict=True):
                if attr != attr0:
                    raise RuntimeError(f"{graph}: cfg point layout changed across decode ({attr} vs {attr0})")
                if sf != sf0:
                    if sf != expect:
                        raise RuntimeError(
                            f"{graph}: slot {fname}: point {attr} moved to {sf}, expected {expect}"
                        )
                    attr_to_slot.setdefault(attr, e)
        slot_t_targets.append(list(t_targets or []))
        slot_d_targets.append(list(d_targets or []))


    sync_fns = {
        2: (sync_block2_qk_binding, sync_block2_aux_fresh_binding),
        4: (sync_block4_v_mask_binding,),
        5: (sync_block5_aux_fresh_binding,),
    }.get(blk, ())
    mirror_of: Dict[str, str] = {}
    if sync_fns:
        probe_cfg = _decode(base_block)
        sentinel = {}
        v = 90
        for attr, _sf, _dist, _N in _walk_cfg_points_with_attrs(probe_cfg, int(ctx.N_block)):
            if "[" in attr:
                continue
            point = getattr(probe_cfg, attr, None)
            if point is not None and hasattr(point, "scaling_factor"):
                point.scaling_factor = v
                sentinel[v] = attr
                v += 1
        for fn in sync_fns:
            fn(probe_cfg)
        for attr, sf, _dist, _N in _walk_cfg_points_with_attrs(probe_cfg, int(ctx.N_block)):
            src = sentinel.get(int(sf))
            if src is not None and src != attr:
                mirror_of[attr] = src


    skel_entries = list(bridge._cfg_to_t_new_table.get(graph, ()))
    attr_of_skel: Dict[int, str] = {}
    for r, entry in enumerate(skel_entries):
        cfg_field = entry.cfg_field
        attr = cfg_field if entry.tuple_index is None else f"{cfg_field}[{entry.tuple_index}]"
        attr_of_skel[r] = attr

    node_to_attr = GRAPH_NODE_TO_CFG_ATTR.get(blk, {})
    attr_to_node = {str(v): str(k) for k, v in node_to_attr.items()}

    points: List[PointSpec] = []
    skel_attr_by_pos = {attr: r for r, attr in attr_of_skel.items()}
    for attr, sf, dist, N in pts0:
        eff_attr = mirror_of.get(attr, attr)
        slot_idx = attr_to_slot.get(attr, attr_to_slot.get(eff_attr, -1))
        if attr in skel_attr_by_pos or eff_attr in skel_attr_by_pos:
            r = skel_attr_by_pos.get(attr, skel_attr_by_pos.get(eff_attr))
            if int(r) == 0:
                points.append(PointSpec(
                    kind="source", distribution=dist, N=N,
                    slot_idx=slot_idx, const_sf=sf,
                ))
            else:
                points.append(PointSpec(
                    kind="rescale", distribution=dist, N=N, skel_pos=int(r),
                    slot_idx=slot_idx, const_sf=sf,
                ))
        elif eff_attr in attr_to_node:
            points.append(PointSpec(
                kind="encode", distribution=dist, N=N,
                node=attr_to_node[eff_attr], slot_idx=slot_idx, const_sf=sf,
            ))
        elif slot_idx >= 0:
            points.append(PointSpec(kind="slot", distribution=dist, N=N, slot_idx=slot_idx))
        else:
            points.append(PointSpec(kind="const", distribution=dist, N=N, const_sf=sf))

    derived_deltas: List[DerivedDeltaSpec] = []
    if blk == 4 and "ctct_rot_softmax_mul_v" in d0:
        def _term(field_name: str, cfg_attr: str) -> DeltaTermSpec:
            slot = int(field_to_slot.get(field_name, -1))
            if slot >= 0:
                return DeltaTermSpec(slot_idx=slot)
            point = getattr(base_cfg, cfg_attr, None)
            if point is None or not hasattr(point, "scaling_factor"):
                raise RuntimeError(
                    f"{graph}: cannot derive ctct_rot_softmax_mul_v term {field_name}/{cfg_attr}"
                )
            return DeltaTermSpec(slot_idx=-1, const_sf=int(point.scaling_factor))


        derived_deltas.append(DerivedDeltaSpec(
            node="ctct_rot_softmax_mul_v",
            terms=[
                _term("v_fresh_sf", "v_fresh"),
                _term("softmax_out_mask_sf", "softmax_out_mask_encode"),
            ],
        ))

    return FastEnumTemplate(
        graph_key=graph,
        block_idx=blk,
        n_block=int(ctx.N_block),
        baseline_t_new=list(int(x) for x in t0),
        baseline_deltas=dict(d0),
        skeleton_node_ids=[int(x) for x in ctx.baseline_skeleton],
        slot_t_targets=slot_t_targets,
        slot_d_targets=slot_d_targets,
        slot_choice_sfs=slot_choice_sfs,
        enum_positions=[int(p) for p in ctx.enum_positions],
        enum_choices=[[int(c) for c in ch] for ch in ctx.enum_choices],
        baseline_block_indices=list(int(x) for x in ctx.baseline_block_indices),
        points=points,
        derived_deltas=derived_deltas,
    )


def verify_template(
    template: FastEnumTemplate,
    ctx: Any,
    *,
    num_random: int = 64,
    seed: int = 0,
) -> Dict[str, Any]:
    """Evaluate baseline + all-min corner + ``num_random`` random combos through
    BOTH paths; raise on the FIRST mismatch of (valid, fusion_count,
    total_bits, installed signature, total variance). The builder must call
    this before enumeration and abort on failure — it is the contract that the
    fast path can never change the enumeration's answers.
    """
    session = ctx.bridge.invoker.session
    noise_order = fcm.SummedInstalledVariance()
    lens = [len(c) for c in template.enum_choices]
    rng = np.random.default_rng(int(seed))

    base_pos = []
    for e, ch in enumerate(template.enum_choices):
        base_idx = template.baseline_block_indices[template.enum_positions[e]]
        base_pos.append(ch.index(int(base_idx)) if int(base_idx) in ch else len(ch) - 1)
    probes: List[List[int]] = [base_pos, [0] * len(lens)]
    for _ in range(int(num_random)):
        probes.append([int(rng.integers(0, n)) for n in lens])

    checked = 0
    for combo_pos in probes:
        block = list(template.baseline_block_indices)
        for e, cpos in enumerate(combo_pos):
            block[template.enum_positions[e]] = int(template.enum_choices[e][int(cpos)])
        golden = fusion_enum._eval_block(ctx, block)
        fast = eval_combo_fast(template, session, combo_pos)
        if bool(golden.get("valid")) != bool(fast.get("valid")):
            raise RuntimeError(
                f"{template.graph_key}: fast-path mismatch on validity at {block}: "
                f"golden={golden.get('valid')} fast={fast.get('valid')}"
            )
        if not golden.get("valid"):
            checked += 1
            continue
        g_sig = fusion_enum._installed_signature(golden["points"])
        f_sig = fusion_enum._installed_signature(fast["points"])
        g_var = float(noise_order.total_variance(golden["points"]))
        f_var = float(fast["total_variance"])
        mism = []
        if int(golden["fusion_count"]) != int(fast["fusion_count"]):
            mism.append(f"fusion {golden['fusion_count']} != {fast['fusion_count']}")
        if int(golden["total_bits"]) != int(fast["total_bits"]):
            mism.append(f"bits {golden['total_bits']} != {fast['total_bits']}")
        if g_sig != f_sig:
            mism.append(f"signature {g_sig} != {f_sig}")
        if abs(g_var - f_var) > 1e-12 * max(1.0, abs(g_var)):
            mism.append(f"variance {g_var!r} != {f_var!r}")
        if mism:
            raise RuntimeError(
                f"{template.graph_key}: fast-path mismatch at block={block}: " + "; ".join(mism)
            )
        checked += 1
    return {"checked": int(checked), "num_random": int(num_random)}


def enumerate_range_worker(payload: Mapping[str, Any]) -> Tuple[int, List[Tuple], float, int]:
    """Multiprocessing worker over a contiguous combo-rank range.

    ``payload`` carries the pickled template + (start, stop) + the
    Rescale_optimizer root/profile. Returns ``(num_valid, reduced_tuples,
    wall_seconds, combo_count)``. The worker process imports only this module
    + ReplanSession — no torch (faster spawn, smaller footprint).
    """
    import sys

    root = str(payload["ro_root"])
    if root not in sys.path:
        sys.path.insert(0, root)
    from rescale_optimizer import ReplanSession

    template: FastEnumTemplate = payload["template"]
    session = ReplanSession.from_profile(profile=str(payload["profile"]), root=root)
    start = int(payload["start"])
    stop = int(payload["stop"])
    t0 = time.perf_counter()
    ecs, num_valid = enumerate_range_fast(
        template, session,
        start=start, stop=stop,
        noise_tol=float(payload.get("noise_tol", 1e-18)),
    )
    wall = time.perf_counter() - t0
    rows = [
        (ec.action_indices, ec.fusion_count, ec.total_bits, ec.total_variance, ec.installed_signature)
        for ec in ecs
    ]
    return num_valid, rows, float(wall), int(stop - start)
