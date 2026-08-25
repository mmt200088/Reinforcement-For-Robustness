"""Fusion-option precision boost ("加大精度") — torch-free core.

After the fusion-count enumeration keeps the minimum-noise config per
``fusion_count`` (``fusion_enum.group_min_noise_options``), a non-zero-fusion
option lands on a modulus chain with at least one **short prime** (a stage whose
drop is ``< q_max``). The enum cannot fix this — its SF grid only sweeps *down*
from each slot's baseline, while filling the short prime needs *above-baseline*
SF at the segment feeding it.

This module raises short primes as high as possible (≤ ``q_max``) for one
block type, verifies every candidate through real replan, and keeps the
minimum-installed-noise option. Above-baseline SFs are stored explicitly.

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


MAX_DISTRIBUTIONS = 200_000

MAX_ENCODE_SF = 46


FINAL_ENCODE_MIN = 15


def _phase2_final_encode_floor(
        topology: "ChainTopology",
        final_field: Optional[str],
        base_final: int,
        ) -> int:
    """Lower bound for phase-2 final-encode redistribution.

    Most final encodes may trade precision for the last rescale's ``sf_post`` down
    to ``FINAL_ENCODE_MIN``. Block4's ``ln_var_inv_d_sf`` is different: it feeds the
    LayerNorm variance inverse path and is marked non-addable in the topology, so
    phase 2 must not lower it below the pre-boost/baseline value.
    """
    if topology.graph_key == "block4" and final_field == "ln_var_inv_d_sf":
        return int(base_final)
    return int(FINAL_ENCODE_MIN)


def target_output_sf(graph_key: str, profile: str, root: str) -> int:
    """The phase-2 output-SF ceiling for one block type, read from its
    Rescale_optimizer config: ``q_tail_bits - amplitude_budgets[-1] - h_sf``.

    General, not hardcoded — a changed JSON yields a changed target (the config
    file is ``<root>/configs/<profile>/<graph_key>.json``; the file name matches
    the graph key for every block type, e.g. ``block2_mrpc`` / ``block4`` /
    ``block5_n2``)."""
    import json as _json
    import os as _os

    path = _os.path.join(str(root), "configs", str(profile), f"{graph_key}.json")
    with open(path, encoding="utf-8") as fh:
        cfg = _json.load(fh)
    q_tail = int(cfg["optimization"]["q_tail_bits"])
    h_sf = int(cfg["global"]["h_sf"])
    amp_last = int(cfg["amplitude_budgets"][-1])
    return q_tail - amp_last - h_sf


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


BLOCK2_MRPC_TOPOLOGY = ChainTopology(
    graph_key="block2_mrpc",
    nodes=(
        ChainNode("fresh", "inv_std_fresh_sf"),
        ChainNode("x2"),
        ChainNode("encode", "gamma_sf", addable=True),
        ChainNode("rescale", "gamma_rescale_sf"),
        ChainNode("encode", "wk_sf", addable=True),
        ChainNode("encode", "kt_mask1_sf", addable=True),
        ChainNode("rescale", "kt_mask1_rescale_sf"),
        ChainNode("encode", "kt_mask2_sf", addable=True),
        ChainNode("x2"),
        ChainNode("rescale", "qkt_matmul_rescale_sf"),
        ChainNode("encode", "qkt_merge_mask_sf", addable=False),
    ),
)


BLOCK4_MRPC_TOPOLOGY = ChainTopology(
    graph_key="block4",
    nodes=(
        ChainNode("fresh", "softmax_out_fresh_sf"),
        ChainNode("encode", "softmax_out_mask_sf", addable=True, binding_multiplier=2),
        ChainNode("additive_ctct"),
        ChainNode("rescale", "softmax_v_matmul_rescale_sf"),
        ChainNode("encode", "softmax_v_mask_sf", addable=True),
        ChainNode("encode", "wo_sf", addable=True),
        ChainNode("encode", "ln_mean_inv_d_sf", addable=True),
        ChainNode("rescale", "ln_mean_rescale_sf"),
        ChainNode("x2"),
        ChainNode("rescale", "ln_square_rescale_sf"),
        ChainNode("encode", "ln_var_inv_d_sf", addable=False),
    ),
)


BLOCK5_N2_MRPC_TOPOLOGY = ChainTopology(
    graph_key="block5_n2",
    nodes=(
        ChainNode("fresh", "x_centered_fresh_sf"),
        ChainNode("x2"),
        ChainNode("rescale", "normalize_rescale_sf"),
        ChainNode("encode", "gamma_sf", addable=True),
        ChainNode("encode", "wffn1_sf", addable=True),
        ChainNode("rescale", "wffn1_rescale_sf"),
        ChainNode("x2"),
        ChainNode("encode", "gelu_coeff_sf", addable=True),
        ChainNode("rescale", "gelu_coeff_mul_rescale_sf_0"),
    ),
)


BLOCK5_N4_MRPC_TOPOLOGY = ChainTopology(
    graph_key="block5_n4",
    nodes=(
        ChainNode("fresh", "x_centered_fresh_sf"),
        ChainNode("x2"),
        ChainNode("rescale", "normalize_rescale_sf"),
        ChainNode("encode", "gamma_sf", addable=True),
        ChainNode("encode", "wffn1_sf", addable=True),
        ChainNode("rescale", "wffn1_rescale_sf"),
        ChainNode("x2"),
        ChainNode("rescale", "gelu_power_rescale_sf_0"),
        ChainNode("x2"),
        ChainNode("encode", "gelu_coeff_sf", addable=True),
        ChainNode("rescale", "gelu_coeff_mul_rescale_sf_0"),
    ),
)


BLOCK5_N1_MRPC_TOPOLOGY = ChainTopology(
    graph_key="block5_n1",
    nodes=(
        ChainNode("fresh", "x_centered_fresh_sf"),
        ChainNode("x2"),
        ChainNode("rescale", "normalize_rescale_sf"),
        ChainNode("encode", "gamma_sf", addable=True),
        ChainNode("encode", "wffn1_sf", addable=True),
        ChainNode("encode", "gelu_coeff_sf", addable=True),
        ChainNode("rescale", "gelu_coeff_mul_rescale_sf_0"),
    ),
)

TOPOLOGIES: Dict[str, ChainTopology] = {
    "block2_mrpc": BLOCK2_MRPC_TOPOLOGY,
    "block4": BLOCK4_MRPC_TOPOLOGY,
    "block5_n1": BLOCK5_N1_MRPC_TOPOLOGY,
    "block5_n2": BLOCK5_N2_MRPC_TOPOLOGY,
    "block5_n4": BLOCK5_N4_MRPC_TOPOLOGY,
}


def topology_for_graph_key(graph_key: str) -> Optional[ChainTopology]:
    """Resolve a Rescale_optimizer graph key to its boost topology, generalizing
    across fine-tuned profiles.

    block4 / block5_n* keys are profile-agnostic and match ``TOPOLOGIES`` exactly.
    block2's key is profile-suffixed (``block2_<profile>``), but its modulus-chain
    STRUCTURE is profile-independent (the ``cut_point_sf`` / ``propagation_deltas``
    node lists are byte-identical across mrpc / rte / sst2 + their ``_large``
    variants — only the SF values differ), so every ``block2_*`` resolves to the
    shared block2 topology. The topology is used for structure only (nodes, q_max,
    last-rescale lookup); replan always runs against the caller's profile-correct
    ``ctx.graph_key``, so reusing the block2 structure across profiles is safe.

    Returns ``None`` for keys with no topology (block1 — fusion-degenerate, never
    boosted and block3 is frozen, so the boost leaves
    those options untouched.
    """
    gk = str(graph_key)
    topo = TOPOLOGIES.get(gk)
    if topo is not None:
        return topo
    if gk.startswith("block2_"):
        return BLOCK2_MRPC_TOPOLOGY
    return None


def canonicalize_noise_irrelevant_rescales(
        base_slots: Mapping[str, int],
        topology: ChainTopology,
        baseline_sfs: Mapping[str, int],
        *,
        probe_fn: Callable[[Dict[str, int]], Any],
        sig_fn: Callable[[Any], Any],
        ) -> Dict[str, int]:
    """Reset each topology rescale in ``base_slots`` to its baseline SF when doing
    so preserves ``(valid, fusion_count, installed-noise signature)`` — i.e. the
    rescale is NOISE-IRRELEVANT and the chain is unchanged at runtime.

    Why the boost needs this (server-confirmed, block2 rte/sst2 fc=1): some
    topology rescales are NOT noise-install points at runtime — their cfg field is
    ``None``, so the bridge's ``t_new`` falls back to the baseline ``sf_post`` and
    the runtime replan is INSENSITIVE to their action SF. But the boost decodes them
    (``_decode_block_field_values``) to their raw action SF, which can sit BELOW
    baseline (the dedup keeps a lex-min representative of a noise-tie). The boost
    then replans a lower-precision chain than the runtime installs and cannot raise
    the output to the ceiling (decode SF 15 → boost stalls at 43; baseline 28 → 46;
    both replan identically at runtime, ``t_new=[..,28,28,28]``). Resetting such a
    rescale to baseline aligns the boost base with the runtime.

    The guard (same fusion_count + identical installed signature) leaves a genuinely
    noise-relevant rescale untouched — raising its ``sf_post`` would move an
    installed point and change the signature, so the trial is rejected. Already-at-
    or-above-baseline rescales are skipped. Pure: ``probe_fn(slots) -> probe`` (with
    ``.valid`` / ``.fusion_count``) and ``sig_fn(probe) -> hashable`` are injected.
    """
    base0 = probe_fn(dict(base_slots))
    if not getattr(base0, "valid", False):
        return dict(base_slots)
    sig0 = sig_fn(base0)
    fc0 = int(base0.fusion_count)
    out: Dict[str, int] = dict(base_slots)
    for node in topology.nodes:
        if node.kind != "rescale" or not node.cfg_field or node.cfg_field not in out:
            continue
        bsf = baseline_sfs.get(node.cfg_field)
        if bsf is None or int(out[node.cfg_field]) >= int(bsf):
            continue
        trial = dict(out)
        trial[node.cfg_field] = int(bsf)
        tp = probe_fn(trial)
        if getattr(tp, "valid", False) and int(tp.fusion_count) == fc0 and sig_fn(tp) == sig0:
            out = trial
    return out


@dataclass(frozen=True)
class ReplanProbe:
    """Result of replanning one candidate's named SFs (injected by the caller)."""

    valid: bool
    fusion_count: int
    q_initial: Tuple[int, ...]
    q_final: Tuple[int, ...]
    fusions: Tuple[dict, ...] = ()
    extra: Any = None

    t_final: Tuple[int, ...] = ()


@dataclass(frozen=True)
class Candidate:
    edits: Dict[str, int]
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


def _qfinal_to_pre_stage(probe: ReplanProbe) -> List[List[int]]:
    """Map each post-fusion ``q_final`` index to the pre-fusion stage indices it
    covers, by replaying the fusion events on the identity grouping."""
    n_pre = len(probe.q_initial)
    groups: List[List[int]] = [[i] for i in range(n_pre)]
    for ev in probe.fusions:
        p = int(ev.get("fused_position", 0)) - 1
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

            continue
        out.append((int(grp[0]), int(q_max) - int(qf)))
    return out


@dataclass(frozen=True)
class _Geometry:
    target_pos: int
    r_pre_pos: Optional[int]
    c: int

    addable: Tuple[Tuple[str, int, int, int], ...]

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
            continue
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


def _last_rescale_and_final_encode(topology: ChainTopology) -> Tuple[int, str, Optional[str]]:
    """``(last_rescale_idx, last_rescale_field, final_encode_field_or_None)``.

    The "last node SF" is ``last_rescale.sf_post + final_encode_SF``. The final
    encode is the encode node AFTER the last rescale (the one feeding q_tail);
    block2/block4 have one (``qkt_merge_mask`` / ``ln_var_inv_d``), block5 has
    none (the last rescale IS the last node)."""
    rpos = topology.rescale_positions()
    last_idx = len(rpos) - 1
    last_pos = rpos[last_idx]
    last_field = str(topology.nodes[last_pos].cfg_field)
    final_field: Optional[str] = None
    for p in range(last_pos + 1, len(topology.nodes)):
        n = topology.nodes[p]
        if n.kind == "encode" and n.cfg_field:
            final_field = n.cfg_field
            break
    return last_idx, last_field, final_field


def effective_output_target(
        topology: ChainTopology,
        config_target: int,
        max_installed_sf: int = DEFAULT_Q_MAX,
        ) -> int:
    """The achievable output-SF target after the install limit.

    The output is ``last_rescale.sf_post (+ final_encode)``. Since the ADR (SF>46 =
    no noise), an installed point may run up to the modulus limit ``max_installed_sf``
    (q_max=60, NOT the noise-table max 46 — points in (46, 60] just install no noise).
    So the max installable output is ``q_max`` for a block with no final encode (the
    single rescale carries it all — block5), or ``2*q_max`` for one with a final
    encode. Every mrpc config ceiling (``q_tail - amplitude - h_sf``; <=53, n1's 48)
    is <= q_max, so none are clamped — block5_n1 now reaches its full 48 (was clamped
    to 46 under the old <=46 install cap). The clamp only bites a config beyond q_max."""
    _last_idx, _last_field, final_field = _last_rescale_and_final_encode(topology)
    ceiling = int(max_installed_sf) * (2 if final_field is not None else 1)
    return min(int(config_target), ceiling)


def generate_phase2_candidates(
        topology: ChainTopology,
        base_slots: Mapping[str, int],
        target_output_sf: int,
        base_last_prime: int,
        q_max: int = DEFAULT_Q_MAX,
        max_installed_sf: int = DEFAULT_Q_MAX,
        ) -> List[Candidate]:
    """All candidate edit-sets that raise the final output scale to
    ``target_output_sf`` (== ``last_rescale.sf_post + final_encode``).

    The composition is parameterized by the final encode SF: it may rise OR fall
    (down to ``FINAL_ENCODE_MIN``, except protected fields such as block4's
    ``ln_var_inv_d_sf`` which stay at or above their base value), with
    ``sf_post = target − final_encode`` taking the remainder. Raising ``sf_post``
    needs the pre-scale entering the last rescale to rise; that rise is supplied
    by the upstream encodes (REUSING the phase-1 geometry: bit-weights through the
    ``×2`` doublings + ``_simulate_rescale_edits`` compensation of every earlier
    prime), maximizing the last prime ``≤ q_max`` (== ``min noise`` on the
    upstream encodes). Each ``final_encode`` × upstream-distribution is a distinct
    candidate; the driver replan-verifies and keeps the minimum-noise one.

    Blocks with no final encode (block5) get ``final_encode = 0`` → the whole
    target lands on ``sf_post``.

    Install limit: since the ADR (SF>46 = no noise), an installed point may run up to
    ``max_installed_sf`` = the modulus limit q_max (60), NOT the noise-table max 46 —
    a point in (46, 60] just installs no noise (negligible). So a composition whose
    compensation pushes e.g. block4's ``ln_mean_rescale`` to 49 is now KEPT (it was
    DROPPED under the old <=46 cap, which is why block4's final encode could not drop
    below its base). Only points beyond q_max (a real modulus violation, also rejected
    by replan) are dropped; no lower-prime fallback is generated.
    """
    last_idx, last_field, final_field = _last_rescale_and_final_encode(topology)
    geo = _resolve_geometry(topology, last_idx)
    if geo is None:
        return []
    base_sf_post = int(base_slots[last_field])
    base_final = int(base_slots[final_field]) if final_field else 0
    delta = int(target_output_sf) - (base_sf_post + base_final)
    if delta <= 0:
        return []
    weighted = [(f, w) for f, w, _pos in _addable_bit_weights(geo)]
    weights = [w for _f, w in weighted]


    if final_field is None:
        fe_values: Sequence[int] = (0,)
    else:
        fe_floor = _phase2_final_encode_floor(topology, final_field, base_final)
        fe_values = range(fe_floor, base_final + delta + 1)

    out: List[Candidate] = []
    seen = set()
    for fe in fe_values:
        sf_post_target = int(target_output_sf) - int(fe)
        if sf_post_target < base_sf_post or sf_post_target > q_max:
            continue
        if final_field is not None and int(fe) > max_installed_sf:
            continue
        sf_post_rise = sf_post_target - base_sf_post

        budget = sf_post_rise + (int(q_max) - int(base_last_prime))
        max_fill = _max_reachable_fill(weights, budget) if (weights and budget > 0) else 0
        dists = _enumerate_distributions(weighted, max_fill) if (weighted and max_fill > 0) else [{}]
        for dist in dists:
            edits: Dict[str, int] = {}
            for f, a in dist.items():
                edits[str(f)] = int(base_slots[f]) + int(a)
            rescale_edits, fill = _simulate_rescale_edits(topology, base_slots, dist, geo.target_pos)
            if fill != max_fill:
                continue
            edits.update(rescale_edits)
            edits[last_field] = sf_post_target
            if final_field is not None:
                edits[final_field] = int(fe)


            if any(int(v) > int(max_installed_sf) for v in edits.values()):
                continue
            key = tuple(sorted(edits.items()))
            if key in seen:
                continue
            seen.add(key)
            up = "+".join(f"{f}:{dist[f]}" for f in sorted(dist))
            desc = f"final_encode={fe},sf_post={sf_post_target}" + (f",{up}" if up else "")
            out.append(Candidate(edits=edits, description=desc))
    return out


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


@dataclass
class Phase2Result:
    boosted_slots: Dict[str, int]
    total_variance: float
    description: str
    output_sf: int
    base_q_final: Tuple[int, ...]
    boosted_q_final: Tuple[int, ...]
    candidates_tried: int
    candidates_valid: int


def boost_option_phase2(
        *,
        topology: ChainTopology,
        base_slots: Dict[str, int],
        target_output_sf: int,
        replan_fn: Callable[[Dict[str, int]], ReplanProbe],
        noise_fn: Callable[[Dict[str, int], ReplanProbe], float],
        q_max: int = DEFAULT_Q_MAX,
        ) -> Optional["Phase2Result"]:
    """Raise the final OUTPUT scale of ``base_slots`` (a phase-1-boosted option)
    to ``target_output_sf`` at minimum installed noise.

    The output scale is ``last_rescale.sf_post + final_encode_SF``. Candidates
    (``generate_phase2_candidates``) split the gain between the final encode (which
    may drop to ``FINAL_ENCODE_MIN``) and the last rescale's ``sf_post`` (sourced
    upstream, last prime maximized). Each is replan-verified — valid, same
    ``fusion_count``, every PRIOR prime unchanged, and the achieved output
    (``t_final[-1] + final_encode``) equal to the target — and the minimum-noise
    survivor wins. Returns ``None`` if the output is already at/above target or no
    candidate verifies (caller keeps the phase-1 option)."""
    base = replan_fn(dict(base_slots))
    if not base.valid or not base.q_final:
        return None
    base_fc = int(base.fusion_count)
    base_qf = tuple(int(x) for x in base.q_final)
    base_last_prime = base_qf[-1]
    base_prior = base_qf[:-1]
    _last_idx, _last_field, final_field = _last_rescale_and_final_encode(topology)


    target = effective_output_target(topology, int(target_output_sf), int(q_max))

    candidates = generate_phase2_candidates(
        topology, base_slots, target, base_last_prime, q_max=int(q_max),
        max_installed_sf=int(q_max),
    )
    best: Optional[Phase2Result] = None
    n_valid = 0
    for cand in candidates:
        slots = dict(base_slots)
        slots.update(cand.edits)
        probe = replan_fn(slots)
        if not probe.valid or int(probe.fusion_count) != base_fc or not probe.t_final:
            continue
        qf = tuple(int(x) for x in probe.q_final)
        if qf[:-1] != base_prior:
            continue
        fe = int(slots[final_field]) if final_field else 0
        out_sf = int(probe.t_final[-1]) + fe
        if out_sf != target:
            continue
        n_valid += 1
        var = float(noise_fn(slots, probe))
        if best is None or var < best.total_variance:
            best = Phase2Result(
                boosted_slots=slots,
                total_variance=var,
                description=cand.description,
                output_sf=out_sf,
                base_q_final=base_qf,
                boosted_q_final=qf,
                candidates_tried=len(candidates),
                candidates_valid=0,
            )
    if best is not None:
        best.candidates_valid = n_valid
    return best
