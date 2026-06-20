"""Fusion-option precision boost ("加大精度") — torch-free core.

After the fusion-count enumeration keeps the minimum-noise config per
``fusion_count`` (``fusion_enum.group_min_noise_options``), a non-zero-fusion
option lands on a modulus chain with at least one **short prime** (a stage whose
drop is ``< q_max``). The enum cannot fix this — its SF grid only sweeps *down*
from each slot's baseline, while filling the short prime needs *above-baseline*
SF at the segment feeding it.

This module raises those short primes as high as possible (≤ ``q_max``) for a
single block-type, deterministically (route 2 — structural per-block topology +
ONE generic algorithm; NOT a search), then verifies every candidate via REAL
replan and keeps the minimum-installed-noise one. The result carries
above-baseline SFs → stored as an explicit-SF "boosted" option (see
``docs/superpowers/specs/2026-06-19-stage2-precision-boost-design.md``).

Mechanism (generic; confirmed against real replan for block2 fc=1 and block4
fc=1). A short prime sits at a rescale ``R_target``, fed through ``c`` ``ctct``
self-multiply (``×2``) doublings from the rescale immediately before them,
``R_pre``. Adding one SF (in chain-accumulation units) anywhere upstream that
reaches the ``×2`` input raises ``R_target``'s drop by ``2**c`` bits, so the
maximum integer fill is ``S = floor(deficit / 2**c)`` and the prime reaches
``base + 2**c * S`` (== ``q_max`` only when ``deficit`` is divisible by
``2**c``; e.g. block2 fc=1 reaches 60, but block4 fc=1's 31 reaches only 59
because 29 is odd). The ``S`` SF are **distributed across the addable upstream
encodes by minimum noise**:

* encodes BEFORE ``R_pre`` need ``R_pre.sf_post += (their chain-weighted total)``
  so the fused group's modulus stays constant (the earlier prime grows by the
  same amount ``R_pre``'s shrinks → replan re-fuses to the same total);
* encodes BETWEEN ``R_pre`` and the ``×2`` raise the ``×2`` input directly (no
  compensation);
* a binding multiplier handles encodes whose SF is shared by two graph nodes
  (block4 ``softmax_out_mask`` is bound to ``v_mask``, so +1 SF adds +2 to the
  chain — it costs 2 of the budget ``S`` per SF).

block2 (``S=1``) is the degenerate single-SF case of this distribution; block4
(``S=14``, 4 nodes, one binding-double) exercises the full partition.

torch-free: structural candidate generation lives here; ``replan`` and the noise
metric are injected, so the builder supplies the real cfg-based ones while tests
drive it with a real ``ReplanSession``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

DEFAULT_Q_MAX = 60
# Safety cap on the distribution enumeration (block4 fc=1 = 372; this guards a
# pathological large-S block-type from exploding the build).
MAX_DISTRIBUTIONS = 200_000
# Encodes are never raised above the noise table's max representable SF.
MAX_ENCODE_SF = 46


# ---------------------------------------------------------------------------
# Per-block chain topology (hardcoded per block, NOT per SF/position)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ChainNode:
    """One node in a block's scale-accumulation chain, in chain order.

    ``kind``:
      * ``"fresh"``         — source fresh ciphertext (off-limits placement).
      * ``"encode"``        — ciphertext×plaintext (scale += this SF); a
                              headroom-placement candidate iff ``addable``.
      * ``"x2"``            — ciphertext×ciphertext SELF multiply (scale ×2);
                              off-limits, and counts toward ``c``.
      * ``"additive_ctct"`` — ciphertext×ciphertext with a different operand
                              (scale += operand scale, NOT a doubling); a
                              structural passthrough, off-limits, NOT counted
                              in ``c`` (block4 ``ctct_rot_softmax_mul_v``).
      * ``"rescale"``       — cut_point consuming a prime = scale − sf_post.

    ``cfg_field`` is the option-``slots`` key (e.g. ``gamma_sf`` /
    ``ln_mean_rescale_sf``). ``binding_multiplier`` is the chain-accumulation
    cost per SF for an encode whose SF is shared with a bound node (block4
    ``softmax_out_mask_sf`` → 2, because ``v_mask`` mirrors it). The bound
    counterpart's cfg field is set automatically by ``_build_block*_action``, so
    it is NOT listed as a separate node.
    """

    kind: str
    cfg_field: Optional[str] = None
    addable: bool = False
    binding_multiplier: int = 1


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
        ChainNode("encode", "gamma_sf", addable=True),
        ChainNode("rescale", "gamma_rescale_sf"),
        ChainNode("encode", "wk_sf", addable=True),
        ChainNode("encode", "kt_mask1_sf", addable=True),
        ChainNode("rescale", "kt_mask1_rescale_sf"),  # R_pre for the qkt short prime
        ChainNode("encode", "kt_mask2_sf", addable=True),
        ChainNode("x2"),  # ctct_preprocess_qkt (QK ×2, off-limits)
        ChainNode("rescale", "qkt_matmul_rescale_sf"),  # R_target (short prime)
        ChainNode("encode", "qkt_merge_mask_sf", addable=False),  # feeds q_tail
    ),
)

# block4 (mrpc) — validated against block4.json fc=1 and real replan
# (q_initial [27,33,31] → fuse 1&2 → q_final [60,31]; the 31 fills only to 59).
# ctct_rot_softmax_mul_v is an ADDITIVE ctct (scale += v_fresh + v_mask), NOT a
# doubling; softmax_out_mask is bound to v_mask (binding_multiplier=2).
BLOCK4_MRPC_TOPOLOGY = ChainTopology(
    graph_key="block4",
    nodes=(
        ChainNode("fresh", "softmax_out_fresh_sf"),
        ChainNode("encode", "softmax_out_mask_sf", addable=True, binding_multiplier=2),
        ChainNode("additive_ctct"),  # ctct_rot_softmax_mul_v (+= v_fresh + v_mask)
        ChainNode("rescale", "softmax_v_matmul_rescale_sf"),  # fuses away
        ChainNode("encode", "softmax_v_mask_sf", addable=True),
        ChainNode("encode", "wo_sf", addable=True),
        ChainNode("encode", "ln_mean_inv_d_sf", addable=True),
        ChainNode("rescale", "ln_mean_rescale_sf"),  # R_pre for the ln_square short prime
        ChainNode("x2"),  # ctct_square (LN (X−μ)², off-limits)
        ChainNode("rescale", "ln_square_rescale_sf"),  # R_target (short prime)
        ChainNode("encode", "ln_var_inv_d_sf", addable=False),  # feeds q_tail
    ),
)

# block5 n=2 (mrpc) — validated against block5_n2.json fc=1 and real replan
# (q_initial [21,39,51] → fuse 1&2 → q_final [60,51]). The short prime (51) has an
# encode (gelu_coeff) AFTER the ctct_gelu_x2 ×2 (c=0, weight 1), so an ODD deficit
# (9) can be filled exactly → 60 (unlike block4). x_centered_fresh is the bound
# initial ×2 operand (off-limits, like block2's). n=1 needs no boost (its fused
# chain is already 60/60/60), so it has no topology and the builder leaves it.
BLOCK5_N2_MRPC_TOPOLOGY = ChainTopology(
    graph_key="block5_n2",
    nodes=(
        ChainNode("fresh", "x_centered_fresh_sf"),
        ChainNode("x2"),  # ctct_xmean_over_std (initial ×2, off-limits)
        ChainNode("rescale", "normalize_rescale_sf"),  # fuses away
        ChainNode("encode", "gamma_sf", addable=True),
        ChainNode("encode", "wffn1_sf", addable=True),
        ChainNode("rescale", "wffn1_rescale_sf"),  # R_pre (fused prime)
        ChainNode("x2"),  # ctct_gelu_x2 (off-limits)
        ChainNode("encode", "gelu_coeff_sf", addable=True),  # after the ×2 → c=0
        ChainNode("rescale", "gelu_coeff_mul_rescale_sf_0"),  # R_target (short prime)
    ),
)

# block5 n=4 (mrpc) — validated against block5_n4.json fc=1 and real replan
# (q_initial [21,39,31,51] → fuse 1&2 → q_final [60,31,51]). TWO ×2 (ctct_gelu_x2,
# ctct_gelu_x4) sit between the before-encodes and the short prime, so gamma/wffn1
# have c=2 (bit-weight 4 — 4 bits/SF) and the compensation must propagate through
# both doublings (forward-simulated). The short prime is the LAST one (51 → 60);
# the intermediate 31 (gelu_power) is left as-is (user spec). gelu_coeff (after
# both ×2) has c=0 (weight 1) → makes the odd deficit 9 reach 60.
BLOCK5_N4_MRPC_TOPOLOGY = ChainTopology(
    graph_key="block5_n4",
    nodes=(
        ChainNode("fresh", "x_centered_fresh_sf"),
        ChainNode("x2"),  # ctct_xmean_over_std (initial ×2, off-limits)
        ChainNode("rescale", "normalize_rescale_sf"),  # fuses away
        ChainNode("encode", "gamma_sf", addable=True),
        ChainNode("encode", "wffn1_sf", addable=True),
        ChainNode("rescale", "wffn1_rescale_sf"),  # fused prime
        ChainNode("x2"),  # ctct_gelu_x2 (off-limits)
        ChainNode("rescale", "gelu_power_rescale_sf_0"),  # prime 3 (31) — kept, compensated
        ChainNode("x2"),  # ctct_gelu_x4 (off-limits)
        ChainNode("encode", "gelu_coeff_sf", addable=True),  # after both ×2 → c=0
        ChainNode("rescale", "gelu_coeff_mul_rescale_sf_0"),  # R_target (short prime 4)
    ),
)

TOPOLOGIES: Dict[str, ChainTopology] = {
    "block2_mrpc": BLOCK2_MRPC_TOPOLOGY,
    "block4": BLOCK4_MRPC_TOPOLOGY,
    "block5_n2": BLOCK5_N2_MRPC_TOPOLOGY,
    "block5_n4": BLOCK5_N4_MRPC_TOPOLOGY,
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
    """Map each post-fusion ``q_final`` index to the pre-fusion stage indices it
    covers, by replaying the fusion events on the identity grouping."""
    n_pre = len(probe.q_initial)
    groups: List[List[int]] = [[i] for i in range(n_pre)]
    for ev in probe.fusions:
        p = int(ev.get("fused_position", 0)) - 1  # to 0-indexed pre-stage
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
    unfused pre-fusion stage (a fused stage sums to exactly ``q_max``)."""
    groups = _qfinal_to_pre_stage(probe)
    out: List[Tuple[int, int]] = []
    for post_idx, qf in enumerate(probe.q_final):
        if int(qf) >= int(q_max) or post_idx >= len(groups):
            continue
        grp = groups[post_idx]
        if len(grp) != 1:
            # a fused-yet-short stage is unexpected; skip (replan-verify guards).
            continue
        out.append((int(grp[0]), int(q_max) - int(qf)))
    return out


# ---------------------------------------------------------------------------
# Chain geometry + candidate generation (structural, per the topology)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Geometry:
    target_pos: int
    r_pre_pos: Optional[int]
    c: int  # ×2 doublings between R_pre and R_target
    # addable encodes: (cfg_field, binding_multiplier, c_i, position)
    addable: Tuple[Tuple[str, int, int, int], ...]
    # rescales strictly before R_target: (cfg_field, position)
    rescales_before: Tuple[Tuple[str, int], ...]


def _resolve_geometry(topology: ChainTopology, pre_rescale_idx: int) -> Optional[_Geometry]:
    rpos = topology.rescale_positions()
    if not (0 <= pre_rescale_idx < len(rpos)):
        return None
    target_pos = rpos[pre_rescale_idx]
    c = 0
    r_pre_pos: Optional[int] = None
    for p in range(target_pos - 1, -1, -1):
        n = topology.nodes[p]
        if n.kind == "x2":
            c += 1
        elif n.kind == "rescale":
            r_pre_pos = p
            break
    addable: List[Tuple[str, int, int, int]] = []
    rescales_before: List[Tuple[str, int]] = []
    for p in range(target_pos):
        n = topology.nodes[p]
        if n.kind == "encode" and n.addable and n.cfg_field:
            c_i = sum(1 for q in range(p + 1, target_pos) if topology.nodes[q].kind == "x2")
            addable.append((n.cfg_field, int(n.binding_multiplier), int(c_i), int(p)))
        elif n.kind == "rescale" and n.cfg_field:
            rescales_before.append((n.cfg_field, int(p)))
    return _Geometry(
        target_pos=target_pos, r_pre_pos=r_pre_pos, c=c,
        addable=tuple(addable), rescales_before=tuple(rescales_before),
    )


def _addable_bit_weights(geo: "_Geometry") -> List[Tuple[str, int, int]]:
    """``[(cfg_field, bit_weight, position), ...]`` for each addable encode.

    ``bit_weight = binding_multiplier * 2**c_i`` is how many bits one SF of that
    encode adds to the short prime: it passes through ``c_i`` ``×2`` doublings on
    the way, and a bound encode (binding_multiplier=2) moves two graph nodes. An
    encode AFTER the ×2 has ``c_i=0`` (weight 1) → it can fill an ODD deficit
    (block5_n2 reaches 60 this way; block4 has no such encode and stops at 59).
    """
    return [(f, m * (2 ** c_i), pos) for (f, m, c_i, pos) in geo.addable]


def _max_reachable_fill(bit_weights: Sequence[int], deficit: int) -> int:
    """Largest ``Σ w_i·a_i ≤ deficit`` (a_i ≥ 0, unlimited) — the coin problem.
    With a weight-1 encode this is ``deficit``; with only even weights it is the
    largest even ``≤ deficit``."""
    if deficit <= 0:
        return 0
    reach = [False] * (deficit + 1)
    reach[0] = True
    for v in range(1, deficit + 1):
        for w in bit_weights:
            if 0 < w <= v and reach[v - w]:
                reach[v] = True
                break
    for v in range(deficit, -1, -1):
        if reach[v]:
            return v
    return 0


def short_prime_fill(topology: ChainTopology, short_prime: Tuple[int, int]) -> Optional[int]:
    """Max bits one short prime can be raised (≤ its deficit), or None if not
    fillable. ``current + fill ≤ q_max``; equals the deficit when an odd fill is
    reachable, else the largest even ≤ deficit."""
    geo = _resolve_geometry(topology, int(short_prime[0]))
    if geo is None or geo.r_pre_pos is None or not geo.addable:
        return None
    weights = [w for _f, w, _pos in _addable_bit_weights(geo)]
    fill = _max_reachable_fill(weights, int(short_prime[1]))
    return fill if fill > 0 else None


def _enumerate_distributions(
        weighted_fields: Sequence[Tuple[str, int]],
        budget: int,
        ) -> List[Dict[str, int]]:
    """All ``{field: a}`` with ``Σ multiplier_i * a_i == budget`` (a_i ≥ 0).

    Each enumerates the budget exactly (under-filling never reaches the target
    prime; over-filling overshoots q_max / breaks fusion → replan rejects).
    """
    fields = [(str(f), int(m)) for f, m in weighted_fields]
    out: List[Dict[str, int]] = []

    def rec(i: int, rem: int, acc: Dict[str, int]) -> None:
        if len(out) > MAX_DISTRIBUTIONS:
            return
        if i == len(fields):
            if rem == 0:
                out.append(dict(acc))
            return
        field, mult = fields[i]
        a = 0
        while mult * a <= rem:
            if a > 0:
                acc[field] = a
            rec(i + 1, rem - mult * a, acc)
            if a > 0:
                del acc[field]
            a += 1

    rec(0, int(budget), {})
    return out


def _simulate_rescale_edits(
        topology: ChainTopology,
        base_slots: Mapping[str, int],
        dist: Mapping[str, int],
        target_pos: int,
        ) -> Tuple[Dict[str, int], int]:
    """Forward "delta" simulation: walk the chain accumulating ``delta`` (how much
    the scale rose vs base), and at every rescale BEFORE ``target_pos`` set
    ``sf_post += delta`` so that rescale's prime stays at its base value (keeps
    the whole fusion structure intact). A ``×2`` doubles ``delta`` (so an encode
    upstream of N doublings contributes ``2**N`` bits to the short prime, and the
    compensation of an intermediate rescale picks up that amplification — block5_n4
    has TWO doublings, which a flat "sum upstream additions" rule got wrong).

    Returns ``({rescale_field: new_sf_post}, short_fill)`` where ``short_fill`` is
    the ``delta`` reaching ``target_pos`` (the bits the short prime gains).

    Binding note: a bound encode (``binding_multiplier=2``) adds its chain delta
    at the encode position; this is exact only when no ``×2`` sits between the two
    bound graph nodes (true for block4's ``softmax_out_mask``/``v_mask``).
    """
    edits: Dict[str, int] = {}
    delta = 0
    for pos in range(int(target_pos)):
        node = topology.nodes[pos]
        if node.kind == "x2":
            delta *= 2
        elif node.kind == "encode" and node.addable and node.cfg_field:
            a = int(dist.get(node.cfg_field, 0))
            if a:
                delta += int(node.binding_multiplier) * a
        elif node.kind == "rescale" and node.cfg_field and delta:
            edits[node.cfg_field] = int(base_slots[node.cfg_field]) + delta
        # fresh / additive_ctct / non-addable encode: delta unchanged
    return edits, delta


def _candidates_for_short_prime(
        topology: ChainTopology,
        base_slots: Mapping[str, int],
        short_prime: Tuple[int, int],
        ) -> List[Candidate]:
    """All candidate edit-sets that fill ONE short prime to its max.

    Distributes ``max_fill`` bits across the addable encodes by their per-SF bit
    weight (``binding_multiplier * 2**c_i``), then forward-simulates the rescale
    ``sf_post`` compensations (``_simulate_rescale_edits``) so every prime before
    the short one stays constant through any number of ``×2`` doublings. Mixed
    ``c_i`` (an encode after the ×2 has weight 1, ones before have weight ≥2) is
    handled directly — no uniform-c restriction — so an odd deficit can be filled
    exactly when a weight-1 encode exists (block5_n2/n4 reach 60; block4 stops at 59).
    """
    geo = _resolve_geometry(topology, int(short_prime[0]))
    max_fill = short_prime_fill(topology, short_prime)
    if geo is None or geo.r_pre_pos is None or max_fill is None:
        return []
    weighted = [(f, w) for f, w, _pos in _addable_bit_weights(geo)]

    out: List[Candidate] = []
    for dist in _enumerate_distributions(weighted, max_fill):
        edits: Dict[str, int] = {}
        ok = True
        for f, a in dist.items():
            new_sf = int(base_slots[f]) + int(a)
            if new_sf > MAX_ENCODE_SF:
                ok = False
                break
            edits[f] = new_sf
        if not ok:
            continue
        rescale_edits, fill = _simulate_rescale_edits(topology, base_slots, dist, geo.target_pos)
        if fill != max_fill:
            continue  # sanity: the bit-weighted budget must match the chain sim
        edits.update(rescale_edits)
        if not edits:
            continue
        desc = "+".join(f"{f}:{dist[f]}" for f in sorted(dist))
        out.append(Candidate(edits=edits, description=desc))
    return out


def generate_candidates(
        topology: ChainTopology,
        base_slots: Mapping[str, int],
        short_primes: Sequence[Tuple[int, int]],
        ) -> List[Candidate]:
    """Candidate edit-sets filling ALL short primes. One short prime → that
    prime's list; several → the cartesian product with merged (conflict-free)
    edits."""
    if not short_primes:
        return []
    per_prime = [_candidates_for_short_prime(topology, base_slots, sp) for sp in short_primes]
    if any(len(c) == 0 for c in per_prime):
        return []
    if len(per_prime) == 1:
        return per_prime[0]
    import itertools
    out: List[Candidate] = []
    for combo in itertools.product(*per_prime):
        merged: Dict[str, int] = {}
        descs: List[str] = []
        ok = True
        for cand in combo:
            for k, v in cand.edits.items():
                if k in merged and merged[k] != v:
                    ok = False
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
    """Raise the LAST short prime of ``base_slots`` as high as possible
    (≤ ``q_max``) at minimum installed noise.

    Only the prime at the final rescale (the one feeding ``q_tail``) is boosted —
    per the user spec, intermediate short primes are left as-is (block5_n4 keeps
    its middle ``31`` while raising the trailing ``51`` to ``60``). Returns the
    min-noise boosted slots (same ``fusion_count``, replan-valid), or ``None`` if
    the last prime is not short/fillable or no candidate verifies (caller keeps
    the original option).
    """
    base = replan_fn(dict(base_slots))
    if not base.valid:
        return None
    base_fc = int(base.fusion_count)
    base_sum = sum(int(x) for x in base.q_final)
    # the last rescale is the highest pre-fusion rescale index in the topology.
    last_rescale_idx = len(topology.rescale_positions()) - 1
    shorts = find_short_primes(base, q_max)
    fillable = [
        sp for sp in shorts
        if int(sp[0]) == last_rescale_idx and short_prime_fill(topology, sp) is not None
    ]
    if not fillable:
        return None
    total_fill = int(short_prime_fill(topology, fillable[0]))
    candidates = generate_candidates(topology, base_slots, fillable)
    best: Optional[BoostResult] = None
    n_valid = 0
    for cand in candidates:
        slots = dict(base_slots)
        slots.update(cand.edits)
        probe = replan_fn(slots)
        if not probe.valid or int(probe.fusion_count) != base_fc:
            continue
        # every valid distribution must reach the SAME boosted chain (each short
        # prime raised by its fill; other stages preserved) → total prime bits up
        # by exactly total_fill. This is the replan-verified success criterion.
        if sum(int(x) for x in probe.q_final) != base_sum + total_fill:
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
                candidates_valid=0,
            )
    if best is not None:
        best.candidates_valid = n_valid
    return best
