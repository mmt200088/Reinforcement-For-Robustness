"""Fusion-option precision boost ("加大精度") — torch-free core.

After the fusion-count enumeration keeps the minimum-noise config per
``fusion_count`` (``fusion_enum.group_min_noise_options``), a non-zero-fusion
option lands on a modulus chain with at least one **short prime** (a stage whose
drop is ``< q_max``), e.g. block2 fc=1: ``q_final = [60, 58]``. The enum cannot
fix this — its SF grid only sweeps *down* from each slot's baseline, while
filling the short prime needs *above-baseline* SF at the segment feeding it.

This module raises those short primes to ``q_max`` ("加大精度") for a single
block-type, deterministically (route 2 — structural, per-block topology; NOT a
search), then verifies every candidate via REAL replan and keeps the
minimum-noise one. The result carries above-baseline SFs, so it is stored as an
explicit-SF "boosted" option (see the design spec
``docs/superpowers/specs/2026-06-19-stage2-precision-boost-design.md``).

Mechanism (per the spec / confirmed worked examples), applied on the **pre-fusion
chain** (the graph's natural one-rescale-per-cut_point segments; replan re-fuses):
for a short prime at pre-fusion rescale ``s`` (deficit ``Δ = q_max − drop``),
adding ``δ`` SF at an encode side-node that passes ``c`` ``ctct`` (×2) doublings
before reaching rescale ``s`` raises that prime by ``δ·2^c`` bits → ``δ = Δ/2^c``.
Placement is either (a) an encode in stage ``s``'s own segment (no compensation),
or (b) an encode in an earlier segment, ``+δ``, with every surviving rescale
between it and ``s`` also ``+δ`` (keeps the already-formed primes constant). The
first ×2 / ``ctct`` nodes are off-limits.

torch-free: the structural candidate generation lives here; ``replan`` and the
noise metric are injected, so the builder supplies the real cfg-based ones while
tests drive it with a real ``ReplanSession``.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

DEFAULT_Q_MAX = 60


# ---------------------------------------------------------------------------
# Per-block chain topology (hardcoded per block, NOT per SF/position)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ChainNode:
    """One node in a block's scale-accumulation chain, in chain order.

    ``kind``:
      * ``"fresh"``   — the source fresh ciphertext (off-limits placement).
      * ``"x2"``      — a ciphertext×ciphertext multiply (scale doubles);
                        off-limits placement.
      * ``"encode"``  — a ciphertext×plaintext multiply (scale += this SF); a
                        candidate headroom placement iff ``can_place``.
      * ``"rescale""``— a cut_point that consumes a prime = scale − sf_post.

    ``cfg_field`` is the option-``slots`` key (e.g. ``gamma_sf`` /
    ``kt_mask1_rescale_sf``); ``mirror_fields`` are slot keys that must move with
    it (block2 Q/K binding: bumping ``kt_mask1_rescale_sf`` mirrors
    ``q_mask1_rescale_sf``).
    """

    kind: str
    cfg_field: Optional[str] = None
    can_place: bool = False
    mirror_fields: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ChainTopology:
    graph_key: str
    nodes: Tuple[ChainNode, ...]
    q_max: int = DEFAULT_Q_MAX

    def rescale_positions(self) -> List[int]:
        return [i for i, n in enumerate(self.nodes) if n.kind == "rescale"]


# block2 (mrpc) — validated against the committed map SFs, the user's worked
# chains, and real replan (q_initial [29,31,58] → fuse 1&2 → q_final [60,58]).
BLOCK2_MRPC_TOPOLOGY = ChainTopology(
    graph_key="block2_mrpc",
    nodes=(
        ChainNode("fresh", "inv_std_fresh_sf"),
        ChainNode("x2"),  # ctct_x_mean_over_std (input ×2, off-limits)
        ChainNode("encode", "gamma_sf", can_place=True),
        ChainNode("rescale", "gamma_rescale_sf"),
        ChainNode("encode", "wk_sf", can_place=True),
        ChainNode("encode", "kt_mask1_sf", can_place=True),
        ChainNode("rescale", "kt_mask1_rescale_sf", mirror_fields=("q_mask1_rescale_sf",)),
        ChainNode("encode", "kt_mask2_sf", can_place=True),
        ChainNode("x2"),  # ctct_preprocess_qkt (QK ×2, off-limits)
        ChainNode("rescale", "qkt_matmul_rescale_sf"),
        # qkt_merge is consumed AFTER the qkt_matmul rescale (it feeds the fixed
        # q_tail, not any rescale drop), so it can never fill a short prime —
        # ordered last so it is never a candidate. Verified via replan: bumping
        # it leaves q_final unchanged at [.., 58].
        ChainNode("encode", "qkt_merge_mask_sf", can_place=False),
    ),
)

TOPOLOGIES: Dict[str, ChainTopology] = {
    "block2_mrpc": BLOCK2_MRPC_TOPOLOGY,
}


# ---------------------------------------------------------------------------
# Replan probe (injected) + candidate model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ReplanProbe:
    """Result of replanning one candidate's named SFs (injected by the caller)."""

    valid: bool
    fusion_count: int
    q_initial: Tuple[int, ...]  # per pre-fusion stage (one per rescale)
    q_final: Tuple[int, ...]  # per post-fusion stage
    fusions: Tuple[dict, ...] = ()
    extra: Any = None  # opaque payload for noise_fn (e.g. installed points) — lets
    #                    replan_fn and noise_fn share one evaluation (no double replan).


@dataclass(frozen=True)
class Candidate:
    edits: Dict[str, int]  # slot field -> new SF (above-baseline allowed)
    description: str


@dataclass
class BoostResult:
    boosted_slots: Dict[str, int]
    total_variance: float
    description: str
    base_q_final: Tuple[int, ...]
    boosted_q_final: Tuple[int, ...]
    candidates_tried: int
    candidates_valid: int


# ---------------------------------------------------------------------------
# Short-prime detection (map post-fusion q_final back to pre-fusion stages)
# ---------------------------------------------------------------------------
def _qfinal_to_pre_stage(probe: ReplanProbe) -> List[List[int]]:
    """Map each post-fusion ``q_final`` index to the list of pre-fusion stage
    indices it covers. A fusion event ``fused_position p`` (1-indexed) into the
    next surviving stage merges pre-stage ``p`` into its neighbour. We replay the
    events on the identity grouping to recover the post→pre mapping.
    """
    n_pre = len(probe.q_initial)
    groups: List[List[int]] = [[i] for i in range(n_pre)]
    # apply fusions: each absorbs position p (1-indexed) into the next group.
    for ev in probe.fusions:
        p = int(ev.get("fused_position", 0)) - 1  # to 0-indexed pre-stage
        # find the group currently holding pre-stage p, merge into the next group
        gi = next((k for k, g in enumerate(groups) if p in g), None)
        if gi is None:
            continue
        into = str(ev.get("fused_into", "next"))
        tgt = gi + 1 if into == "next" else gi - 1
        if 0 <= tgt < len(groups):
            groups[tgt] = sorted(groups[gi] + groups[tgt])
            groups.pop(gi)
    return groups


def find_short_primes(probe: ReplanProbe, q_max: int) -> List[Tuple[int, int]]:
    """Return ``[(pre_fusion_rescale_idx, deficit), ...]`` for every post-fusion
    stage whose drop ``< q_max``. A short post-fusion stage is always a single
    unfused pre-fusion stage (fused stages sum to exactly ``q_max``)."""
    groups = _qfinal_to_pre_stage(probe)
    out: List[Tuple[int, int]] = []
    for post_idx, qf in enumerate(probe.q_final):
        if int(qf) >= int(q_max):
            continue
        if post_idx >= len(groups):
            continue
        grp = groups[post_idx]
        # short stage = single unfused pre-stage
        if len(grp) != 1:
            # a fused-yet-short stage is unexpected; skip (replan-verify is the
            # ultimate guard — a candidate for it just won't reach all-q_max).
            continue
        out.append((int(grp[0]), int(q_max) - int(qf)))
    return out


# ---------------------------------------------------------------------------
# Candidate generation (structural, per the topology)
# ---------------------------------------------------------------------------
def _candidates_for_short_prime(
        topology: ChainTopology,
        base_slots: Dict[str, int],
        pre_rescale_idx: int,
        deficit: int,
        ) -> List[Candidate]:
    """All placements that fill ONE short prime (at pre-fusion rescale
    ``pre_rescale_idx``) by ``deficit`` bits, per the mechanism."""
    nodes = topology.nodes
    rpos = topology.rescale_positions()
    if not (0 <= pre_rescale_idx < len(rpos)):
        return []
    target_pos = rpos[pre_rescale_idx]
    out: List[Candidate] = []
    for p in range(target_pos):
        node = nodes[p]
        if node.kind != "encode" or not node.can_place or not node.cfg_field:
            continue
        # ctct doublings strictly between this encode and the target rescale
        c = sum(1 for q in range(p + 1, target_pos) if nodes[q].kind == "x2")
        if deficit % (2 ** c) != 0:
            continue  # cannot fill exactly from here
        delta = deficit // (2 ** c)
        if delta <= 0:
            continue
        edits: Dict[str, int] = {node.cfg_field: int(base_slots[node.cfg_field]) + delta}
        # compensate every surviving rescale strictly between p and the target
        # (keep their already-formed drops constant); mirror Q/K rescales too.
        comp: List[str] = []
        for q in range(p + 1, target_pos):
            if nodes[q].kind == "rescale" and nodes[q].cfg_field:
                edits[nodes[q].cfg_field] = int(base_slots[nodes[q].cfg_field]) + delta
                comp.append(nodes[q].cfg_field)
                for mf in nodes[q].mirror_fields:
                    if mf in base_slots:
                        edits[mf] = int(base_slots[mf]) + delta
        desc = f"{node.cfg_field}+{delta}" + (f" comp[{','.join(comp)}]+{delta}" if comp else " (no-comp)")
        out.append(Candidate(edits=edits, description=desc))
    return out


def generate_candidates(
        topology: ChainTopology,
        base_slots: Dict[str, int],
        short_primes: Sequence[Tuple[int, int]],
        ) -> List[Candidate]:
    """All candidate edit-sets that fill ALL short primes. With one short prime
    this is the per-prime list; with several it is the cartesian product (each
    short prime filled by one of its placements), edits merged."""
    if not short_primes:
        return []
    per_prime = [_candidates_for_short_prime(topology, base_slots, idx, d) for idx, d in short_primes]
    if any(len(c) == 0 for c in per_prime):
        # at least one short prime has no fillable placement → no full fix
        return []
    out: List[Candidate] = []
    for combo in itertools.product(*per_prime):
        merged: Dict[str, int] = {}
        ok = True
        descs: List[str] = []
        for cand in combo:
            for k, v in cand.edits.items():
                if k in merged and merged[k] != v:
                    ok = False  # conflicting edits on the same slot → drop combo
                    break
                merged[k] = v
            if not ok:
                break
            descs.append(cand.description)
        if ok and merged:
            out.append(Candidate(edits=merged, description=" & ".join(descs)))
    return out


# ---------------------------------------------------------------------------
# Boost driver
# ---------------------------------------------------------------------------
def boost_option(
        *,
        topology: ChainTopology,
        base_slots: Dict[str, int],
        replan_fn: Callable[[Dict[str, int]], ReplanProbe],
        noise_fn: Callable[[Dict[str, int], ReplanProbe], float],
        q_max: int = DEFAULT_Q_MAX,
        ) -> Optional[BoostResult]:
    """Raise every short prime of ``base_slots`` to ``q_max`` at minimum noise.

    Returns the min-noise boosted slots (same ``fusion_count``, all primes
    ``== q_max``, replan-valid), or ``None`` if the base has no short prime or no
    candidate reaches all-``q_max`` (caller keeps the original option).
    """
    base = replan_fn(dict(base_slots))
    if not base.valid:
        return None
    base_fc = int(base.fusion_count)
    short = find_short_primes(base, q_max)
    if not short:
        return None  # already all-q_max — nothing to boost
    candidates = generate_candidates(topology, base_slots, short)
    best: Optional[BoostResult] = None
    n_valid = 0
    for cand in candidates:
        slots = dict(base_slots)
        slots.update(cand.edits)
        probe = replan_fn(slots)
        if not probe.valid or int(probe.fusion_count) != base_fc:
            continue
        if any(int(q) != int(q_max) for q in probe.q_final):
            continue
        n_valid += 1
        var = float(noise_fn(slots, probe))
        if best is None or var < best.total_variance:
            best = BoostResult(
                boosted_slots=slots,
                total_variance=var,
                description=cand.description,
                base_q_final=tuple(int(x) for x in base.q_final),
                boosted_q_final=tuple(int(x) for x in probe.q_final),
                candidates_tried=len(candidates),
                candidates_valid=0,  # filled below
            )
    if best is not None:
        best.candidates_valid = n_valid
    return best
