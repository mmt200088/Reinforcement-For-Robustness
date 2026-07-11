"""
rescale_optimizer/replan.py

What-if re-planning with **user-supplied scaling factors** + a new
*fusion-tolerant* feasibility check.

Workflow
--------

Given:

* a baseline graph (already loaded from a config),
* a baseline ``skeleton`` (e.g. the DP output, or any other valid one),
* the baseline ``t`` vector  (post-rescale scale at each stage anchor,
  i.e. ``t[r] = s_r``'s working scale after rescale; ``t[0]`` is the
  source scale),
* a **new** ``t'`` vector (one entry per skeleton stage) that the user
  pre-selects,

we re-run scale propagation along the same skeleton with the new
``t'`` to obtain the new modulus chain:

    s_pre'_r  = PropagateScale(t'_{r-1}, nodes_between(s_{r-1}, s_r))
    q'_r      = s_pre'_r - t'_r                  (new drop bits @ s_r)
    Δ q_r     = q'_r - q_r                       (per-stage delta)

The chain layout is still ``[q_head=60, q'_1, ..., q'_R, q_tail=60]``.

Modified feasibility check (fusion-tolerant)
--------------------------------------------

Standard rule: every ``q'_r`` must lie in [q_legal_min, q_legal_max] —
typically [30, 60].

Modified rule (this module):

    if some q'_r < q_legal_min:
        try to fuse q'_r with one of its neighbours q'_{r-1} or q'_{r+1}:

            (a) prefer the side where the merged size satisfies
                q'_r + q1' <= q_legal_max  (=60)
            (b) the merge **removes the small q'_r and its rescale
                step at s_r** from the skeleton; the surviving
                neighbour absorbs the bits:

                    q1''  =  q'_r + q1'                 (其位置不变)
                    skeleton:  ..., s_{r-1}, s_r, s_{r+1}, ...
                                 ↓  remove s_r
                               ..., s_{r-1}, s_{r+1}, ...

        repeat recursively until either:
          * every remaining q'' ∈ [q_legal_min, q_legal_max] ⇒  VALID
          * some q'' < q_legal_min still exists and no admissible
            neighbour ⇒  INVALID (and report the offending chain)

Output
------

A ``ReplanResult`` dataclass holding:

* ``valid`` flag,
* the (possibly fused) skeleton,
* the (possibly fused) modulus chain,
* the per-stage drop bits before fusion (``q_initial``) and after
  (``q_final``),
* the recomputed ``t'`` after fusion (one entry per surviving stage),
* the number of fusions performed,
* per-fusion log entries (which q was absorbed into which side),
* a diagnostic ``invalid_chain`` field set to the chain at the point
  fusion gave up (only if ``valid == False``).

The check **does not** alter the underlying ``RescaleGraph`` — every
mutation happens on local copies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .graph import ComputeNode, NodeType, RescaleGraph, propagate_scale
from .modulus_chain import ModulusChain


# ---------------------------------------------------------------------------
# Inputs / outputs
# ---------------------------------------------------------------------------


@dataclass
class ReplanInputs:
    """User-supplied data for one what-if replan call."""

    #: Baseline skeleton, e.g. ``[0, 1, 3, 5, M+1]``.  The final entry must
    #: be ``graph.dummy_sink_index``.  Length R+2 where R = number of rescales.
    skeleton: List[int]

    #: Baseline t vector, length R+1 (one entry per stage anchor s_0..s_R).
    #: t_baseline[r] is the scale at s_r right after the rescale (or the
    #: source scale at s_0).  Optional; supplied for reference / logging.
    t_baseline: Optional[List[int]] = None

    #: NEW pre-selected t vector, length R+1.  ``t_new[r]`` is the desired
    #: post-rescale scale at the r-th rescale point (or source scale at r=0).
    #: This is the user "action": the outgoing scaling factor at each
    #: rescale stage.
    t_new: List[int] = field(default_factory=list)

    #: Optional override: q_head_bits / q_tail_bits.  Default = 60 / 60.
    q_head_bits: int = 60
    q_tail_bits: int = 60

    #: Optional propagation override, keyed by multiplication node name.
    #:
    #: For CTPT_MUL:
    #:   value must be int -> node.scale_delta_bits = value
    #:
    #: For CTCT_MUL:
    #:   value == "x2"      -> symmetric mode (other_ct_scale_bits = None)
    #:   value is int       -> asymmetric mode (other_ct_scale_bits = value)
    #:
    #: This lets users co-optimize stage t and per-op propagation deltas.
    delta_overrides: Optional[Dict[str, Union[int, str]]] = None

    #: Optional legal rescale-fusion pairs, using 1-indexed stage positions.
    #: For example, [(1, 2)] means only the first and second rescale
    #: stages may be fused. None keeps the legacy behaviour where any
    #: adjacent pair is considered; [] disables fusion entirely.
    allowed_fusion_pairs: Optional[Sequence[Tuple[int, int]]] = None


@dataclass
class FusionEvent:
    """One fusion operation log entry."""

    #: Position (skeleton index r) of the small prime that got fused away.
    fused_position: int
    #: Side it was merged into: "prev" (r-1) or "next" (r+1).
    fused_into: str
    #: Bit-widths involved.
    small_q: int
    neighbour_q_before: int
    neighbour_q_after: int

    def __repr__(self) -> str:
        return (f"FusionEvent(pos={self.fused_position}, side={self.fused_into}, "
                f"{self.small_q}+{self.neighbour_q_before}={self.neighbour_q_after})")


@dataclass
class ReplanResult:
    """Outcome of one replan + fusion-feasibility pass."""

    valid: bool = False

    #: The skeleton AFTER fusion (with the dummy-sink as last entry).
    skeleton: List[int] = field(default_factory=list)

    #: The chain AFTER fusion.  May not satisfy [q_min, q_max] when
    #: ``valid == False`` (in which case ``invalid_chain`` is also set).
    chain: Optional[ModulusChain] = None

    #: Per-stage drops as initially computed from ``t_new`` (before any fusion).
    #: Length = R_initial.
    q_initial: List[int] = field(default_factory=list)

    #: Per-stage drops AFTER all fusions.  Length = R_final.
    q_final: List[int] = field(default_factory=list)

    #: New t vector after fusions (length R_final + 1).
    t_final: List[int] = field(default_factory=list)

    #: Per-stage drop deltas vs the *baseline* drops, BEFORE fusion (one
    #: entry per baseline stage).  Useful for diagnostics.
    delta_q_vs_baseline: List[int] = field(default_factory=list)

    #: List of fusion events (in chronological order).
    fusions: List[FusionEvent] = field(default_factory=list)

    #: When valid is False, the chain at the moment fusion gave up.
    invalid_chain: Optional[ModulusChain] = None

    #: Human-readable status / error.
    message: str = ""

    #: Echo of user-supplied propagation overrides actually applied.
    applied_delta_overrides: Dict[str, Union[int, str]] = field(default_factory=dict)

    @property
    def fusion_count(self) -> int:
        return len(self.fusions)

    def summary(self) -> str:
        lines = [
            "=" * 64,
            "ReplanResult",
            "=" * 64,
            f"  valid           : {self.valid}",
            f"  message         : {self.message}",
            f"  fusion_count    : {self.fusion_count}",
            f"  skeleton (final): {self.skeleton}",
            f"  q_initial       : {self.q_initial}",
            f"  q_final         : {self.q_final}",
            f"  t_final         : {self.t_final}",
            f"  Δq vs baseline  : {self.delta_q_vs_baseline}",
        ]
        if self.applied_delta_overrides:
            lines.append(f"  delta_overrides : {self.applied_delta_overrides}")
        if self.fusions:
            lines.append("  fusions         :")
            for ev in self.fusions:
                lines.append(f"    - {ev}")
        if self.chain is not None:
            lines.append(f"  chain           : {self.chain.summary()}")
        if not self.valid and self.invalid_chain is not None:
            lines.append(f"  invalid_chain   : {self.invalid_chain.summary()}")
        lines.append("=" * 64)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 1: recompute drops from new t
# ---------------------------------------------------------------------------


def _recompute_drops(
    graph: RescaleGraph,
    skeleton: Sequence[int],
    t_new: Sequence[int],
    *,
    stage_paths: Optional[Sequence[Sequence[ComputeNode]]] = None,
) -> List[int]:
    """
    For each stage r = 1..R, compute the new drop bits

        q'_r  =  PropagateScale(t_new[r-1],  nodes_between(s_{r-1}, s_r))  -  t_new[r]

    Returns a list of length R = len(skeleton) - 2.

    Negative drops are returned as-is (caller must validate).
    """
    M = graph.M
    if skeleton[-1] != M + 1:
        raise ValueError(
            f"replan: skeleton must end with dummy-sink index {M + 1}, "
            f"got {skeleton[-1]}"
        )
    R = len(skeleton) - 2
    if len(t_new) != R + 1:
        raise ValueError(
            f"replan: t_new length must be R+1 = {R + 1}, got {len(t_new)}"
        )
    if stage_paths is not None and len(stage_paths) != R:
        raise ValueError(
            f"replan: stage_paths length must be R = {R}, got {len(stage_paths)}"
        )

    drops: List[int] = []
    for r in range(1, R + 1):
        s_prev = skeleton[r - 1]
        s_curr = skeleton[r]
        if not (0 <= s_prev <= s_curr <= M):
            raise ValueError(
                f"replan: bad skeleton segment ({s_prev} -> {s_curr})"
            )
        path = (
            stage_paths[r - 1]
            if stage_paths is not None
            else graph.nodes_between(s_prev, s_curr)
        )
        s_pre = propagate_scale(int(t_new[r - 1]), path)
        d = s_pre - int(t_new[r])
        drops.append(int(d))
    return drops


def _apply_delta_overrides(
    graph: RescaleGraph,
    delta_overrides: Optional[Dict[str, Union[int, str]]],
    *,
    delta_nodes: Optional[Mapping[str, ComputeNode]] = None,
    collect_applied: bool = True,
) -> Dict[str, Union[int, str]]:
    """
    Apply user-provided propagation delta overrides in-place on ``graph``.

    Returns a normalized ``name -> delta`` dict that was applied.
    Raises ValueError on unknown nodes or invalid value/type combinations.
    """
    if not delta_overrides:
        return {}

    by_name = delta_nodes if delta_nodes is not None else {
        n.name: n
        for n in graph.nodes
        if n.node_type in (NodeType.CTPT_MUL, NodeType.CTCT_MUL)
    }
    applied: Optional[Dict[str, Union[int, str]]] = {} if collect_applied else None

    for name, raw in delta_overrides.items():
        node = by_name.get(name)
        if node is None:
            raise ValueError(f"delta_overrides: unknown multiplication node '{name}'")

        if node.node_type == NodeType.CTPT_MUL:
            if not isinstance(raw, int):
                raise ValueError(
                    f"delta_overrides[{name}] for CTPT_MUL must be int, got {raw!r}"
                )
            node.scale_delta_bits = int(raw)
            if applied is not None:
                applied[name] = int(raw)
            continue

        # CTCT_MUL
        if raw == "x2":
            node.other_ct_scale_bits = None
            if applied is not None:
                applied[name] = "x2"
        elif isinstance(raw, int):
            node.other_ct_scale_bits = int(raw)
            if applied is not None:
                applied[name] = int(raw)
        else:
            raise ValueError(
                f"delta_overrides[{name}] for CTCT_MUL must be int or 'x2', got {raw!r}"
            )

    return applied if applied is not None else {}


# ---------------------------------------------------------------------------
# Step 2: fusion-tolerant feasibility check
# ---------------------------------------------------------------------------


def _normalize_allowed_fusion_pairs(
    allowed_fusion_pairs: Optional[Sequence[Tuple[int, int]]],
) -> Optional[set[Tuple[int, int]]]:
    if allowed_fusion_pairs is None:
        return None

    out: set[Tuple[int, int]] = set()
    for raw_pair in allowed_fusion_pairs:
        if len(raw_pair) != 2:
            raise ValueError(f"allowed fusion pair must have length 2, got {raw_pair!r}")
        a, b = int(raw_pair[0]), int(raw_pair[1])
        if a == b:
            raise ValueError(f"allowed fusion pair cannot fuse a stage with itself: {raw_pair!r}")
        if a > b:
            a, b = b, a
        out.add((a, b))
    return out


def _is_fusion_boundary_allowed(
    allowed_pairs: Optional[set[Tuple[int, int]]],
    left_group: Sequence[int],
    right_group: Sequence[int],
) -> bool:
    if allowed_pairs is None:
        return True
    if not allowed_pairs:
        return False
    a, b = int(left_group[-1]), int(right_group[0])
    if a > b:
        a, b = b, a
    return (a, b) in allowed_pairs


def _fuse_chain(
    skeleton: Sequence[int],
    q_bits: Sequence[int],
    t_vec: Sequence[int],
    q_min: int,
    q_max: int,
    allowed_fusion_pairs: Optional[Sequence[Tuple[int, int]]] = None,
) -> Tuple[bool, List[int], List[int], List[int], List[FusionEvent], Optional[ModulusChain]]:
    """
    Apply rescale-fusion until no q < q_min remains, OR until fusion is
    impossible.

    Parameters
    ----------
    skeleton : list of int, length R+2 (with trailing dummy-sink)
    q_bits   : list of int, length R     — initial drop bits per stage
    t_vec    : list of int, length R+1  — post-rescale scales (or source)
    allowed_fusion_pairs : optional list of 1-indexed original stage pairs.
        ``None`` keeps the legacy all-adjacent behaviour; ``[]`` disables
        fusion.  After a fusion, later decisions still use the original
        stage boundary, so allowing only ``(1, 2)`` cannot accidentally permit
        a second fusion across the new shifted ``(1, 2)`` boundary.

    Returns
    -------
    (valid, skeleton', q_bits', t_vec', fusions, invalid_chain_if_any)

    The fusion rule (matching the spec):

        find smallest r with q_bits[r-1] < q_min        # 1-indexed stage r
        if no such r:  done.
        else:
            for side in (prefer "next", then "prev"):
                neighbour = q_bits[r-1+1]  if side=="next"
                          = q_bits[r-1-1]  if side=="prev"
                if r is at the boundary, skip that side.
                if neighbour + q_bits[r-1] <= q_max:
                    merge:
                       remove q_bits[r-1] and skeleton[r] (the rescale at s_r);
                       neighbour absorbs:  q_bits[neighbour_idx] += q_bits[r-1]
                       (position of neighbour stays the same)
                       t_vec: drop t_vec[r] (its rescale is gone), t_vec[v]
                              for v > r shifts down by one in indexing.
                    record fusion event, restart the search.
                    break
            else:
                # neither neighbour admits the merge → INVALID
                return False, skeleton, q_bits, t_vec, fusions, ModulusChain(...)

    """
    # work on local mutable copies
    skel = list(skeleton)
    q = list(q_bits)
    t = list(t_vec)
    events: List[FusionEvent] = []
    allowed_pairs = _normalize_allowed_fusion_pairs(allowed_fusion_pairs)
    # Each q slot tracks the original stage ids it has absorbed.  This keeps
    # legal-fusion decisions tied to the baseline graph, not to shifted indices
    # after an earlier fusion.
    stage_groups: List[List[int]] = [[r] for r in range(1, len(q) + 1)]

    # ``r`` is the 1-indexed stage; q_bits index is r-1
    while True:
        # find first r with q < q_min
        r_bad: Optional[int] = None
        for r in range(1, len(q) + 1):
            if q[r - 1] < q_min:
                r_bad = r
                break

        if r_bad is None:
            return True, skel, q, t, events, None

        small = q[r_bad - 1]

        # try fuse with NEXT first (preferred — the spec says
        # "去掉前面进行rescale 的那个 q, 后面的模数 + 前面的去掉的q",
        # i.e. the small one on the LEFT is removed and its bits go
        # into the LATER neighbour's slot, position 不变).
        side = None
        idx_neighbour = -1
        if (
            r_bad < len(q)
            and _is_fusion_boundary_allowed(
                allowed_pairs, stage_groups[r_bad - 1], stage_groups[r_bad]
            )
            and (q[r_bad] + small) <= q_max
        ):
            side = "next"
            idx_neighbour = r_bad   # 0-indexed in q
        elif (
            r_bad > 1
            and _is_fusion_boundary_allowed(
                allowed_pairs, stage_groups[r_bad - 2], stage_groups[r_bad - 1]
            )
            and (q[r_bad - 2] + small) <= q_max
        ):
            side = "prev"
            idx_neighbour = r_bad - 2

        if side is None:
            chain_now = ModulusChain(q_head_bits=60, q_bits=list(q), q_tail_bits=60)
            return False, skel, q, t, events, chain_now

        if side == "next":
            before = q[idx_neighbour]
            after = before + small
            q[idx_neighbour] = after
            stage_groups[idx_neighbour] = stage_groups[r_bad - 1] + stage_groups[idx_neighbour]
            del q[r_bad - 1]
            del stage_groups[r_bad - 1]
            # remove the rescale at s_{r_bad} from the skeleton
            del skel[r_bad]
            # remove t at the rescale-r_bad point: t had length R+1,
            # one slot per stage anchor. Stage r_bad's anchor is gone.
            del t[r_bad]
            events.append(FusionEvent(
                fused_position=r_bad, fused_into="next",
                small_q=small,
                neighbour_q_before=before,
                neighbour_q_after=after,
            ))
        else:  # "prev"
            before = q[idx_neighbour]
            after = before + small
            q[idx_neighbour] = after
            stage_groups[idx_neighbour] = stage_groups[idx_neighbour] + stage_groups[r_bad - 1]
            del q[r_bad - 1]
            del stage_groups[r_bad - 1]
            # the small q' is at position r_bad in the skeleton; its
            # rescale step is removed.
            del skel[r_bad]
            del t[r_bad]
            events.append(FusionEvent(
                fused_position=r_bad, fused_into="prev",
                small_q=small,
                neighbour_q_before=before,
                neighbour_q_after=after,
            ))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def replan_with_user_actions(
    graph: RescaleGraph,
    inputs: ReplanInputs,
    baseline_q_bits: Optional[Sequence[int]] = None,
    *,
    stage_paths: Optional[Sequence[Sequence[ComputeNode]]] = None,
    delta_nodes: Optional[Mapping[str, ComputeNode]] = None,
    record_applied_delta_overrides: bool = True,
) -> ReplanResult:
    """
    Re-run scale propagation under user-supplied ``t_new``, then run
    fusion-tolerant feasibility.

    Parameters
    ----------
    graph : RescaleGraph
        Must already have ``stage_node_lists`` populated (i.e. you called
        :func:`build_feasibility_dag` once on it).
    inputs : ReplanInputs
        See dataclass docs.
    baseline_q_bits : optional
        The original chain's per-stage drops (length R) — used only to
        compute ``delta_q_vs_baseline`` for diagnostics.

    Returns
    -------
    ReplanResult
    """
    M = graph.M
    skeleton = list(inputs.skeleton)
    if not skeleton:
        return ReplanResult(message="empty skeleton")
    if skeleton[-1] != M + 1:
        skeleton = skeleton + [M + 1]
    if not graph.stage_node_lists:
        return ReplanResult(
            message=(
                "graph.stage_node_lists is empty. Call build_feasibility_dag(graph) "
                "before replan_with_user_actions()."
            )
        )

    R = len(skeleton) - 2
    if len(inputs.t_new) != R + 1:
        return ReplanResult(
            message=(
                f"t_new length must be R+1 = {R + 1} (skeleton has R={R} stages); "
                f"got {len(inputs.t_new)}."
            )
        )

    try:
        applied_delta_overrides = _apply_delta_overrides(
            graph,
            inputs.delta_overrides,
            delta_nodes=delta_nodes,
            collect_applied=record_applied_delta_overrides,
        )
    except ValueError as e:
        return ReplanResult(message=f"delta override failed: {e}")

    try:
        q_initial = _recompute_drops(
            graph,
            skeleton,
            inputs.t_new,
            stage_paths=stage_paths,
        )
    except ValueError as e:
        return ReplanResult(
            message=f"recompute_drops failed: {e}",
            applied_delta_overrides=applied_delta_overrides,
        )

    # delta vs baseline (if provided)
    delta_q: List[int] = []
    if baseline_q_bits is not None and len(baseline_q_bits) == R:
        delta_q = [int(qn) - int(qb) for qn, qb in zip(q_initial, baseline_q_bits)]

    q_min = graph.q_legal_min
    q_max = graph.q_legal_max
    has_non_positive = False
    too_big: Optional[List[int]] = None
    for r, q in enumerate(q_initial, start=1):
        if q <= 0:
            has_non_positive = True
            break
        if q > q_max:
            if too_big is None:
                too_big = []
            too_big.append(r)

    # Any non-positive drop is unfixable by fusion semantically. Report it.
    if has_non_positive:
        bad_chain = ModulusChain(
            q_head_bits=inputs.q_head_bits,
            q_bits=list(q_initial),
            q_tail_bits=inputs.q_tail_bits,
        )
        return ReplanResult(
            valid=False,
            skeleton=skeleton,
            chain=None,
            q_initial=q_initial,
            q_final=q_initial,
            t_final=list(inputs.t_new),
            delta_q_vs_baseline=delta_q,
            invalid_chain=bad_chain,
            applied_delta_overrides=applied_delta_overrides,
            message=(
                f"new chain contains non-positive drops {q_initial}; "
                "the chosen t_new makes some rescale stage redundant or inverted."
            ),
        )

    # Quickly check oversized drops; fusion cannot fix q > q_max.
    if too_big:
        bad_chain = ModulusChain(
            q_head_bits=inputs.q_head_bits,
            q_bits=list(q_initial),
            q_tail_bits=inputs.q_tail_bits,
        )
        return ReplanResult(
            valid=False,
            skeleton=skeleton,
            chain=None,
            q_initial=q_initial,
            q_final=q_initial,
            t_final=list(inputs.t_new),
            delta_q_vs_baseline=delta_q,
            invalid_chain=bad_chain,
            applied_delta_overrides=applied_delta_overrides,
            message=(
                f"new chain has prime(s) > q_max={q_max} at stage(s) {too_big}; "
                "fusion cannot reduce. Reject."
            ),
        )

    # run fusion
    try:
        valid, new_skel, q_after, t_after, events, bad = _fuse_chain(
            skeleton, q_initial, list(inputs.t_new), q_min, q_max,
            allowed_fusion_pairs=inputs.allowed_fusion_pairs,
        )
    except ValueError as e:
        return ReplanResult(
            valid=False,
            skeleton=skeleton,
            chain=None,
            q_initial=q_initial,
            q_final=q_initial,
            t_final=list(inputs.t_new),
            delta_q_vs_baseline=delta_q,
            applied_delta_overrides=applied_delta_overrides,
            message=f"fusion policy failed: {e}",
        )

    if valid:
        chain_final = ModulusChain(
            q_head_bits=inputs.q_head_bits,
            q_bits=list(q_after),
            q_tail_bits=inputs.q_tail_bits,
        )
        return ReplanResult(
            valid=True,
            skeleton=new_skel,
            chain=chain_final,
            q_initial=q_initial,
            q_final=q_after,
            t_final=t_after,
            delta_q_vs_baseline=delta_q,
            fusions=events,
            applied_delta_overrides=applied_delta_overrides,
            message=(
                f"replan OK after {len(events)} fusion(s). "
                f"R: {len(q_initial)} -> {len(q_after)}."
            ),
        )

    return ReplanResult(
        valid=False,
        skeleton=new_skel,
        chain=ModulusChain(
            q_head_bits=inputs.q_head_bits,
            q_bits=list(q_after),
            q_tail_bits=inputs.q_tail_bits,
        ),
        q_initial=q_initial,
        q_final=q_after,
        t_final=t_after,
        delta_q_vs_baseline=delta_q,
        fusions=events,
        invalid_chain=bad,
        applied_delta_overrides=applied_delta_overrides,
        message=(
            f"replan FAILED: a prime < q_min={q_min} could not be fused "
            f"after {len(events)} successful fusion(s)."
        ),
    )
