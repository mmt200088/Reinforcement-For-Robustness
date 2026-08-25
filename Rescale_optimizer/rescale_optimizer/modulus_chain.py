"""
rescale_optimizer/modulus_chain.py

Algorithms 4–8:
    * Alg 4  ConstructModulusChain         (top-level)
    * Alg 5  RepairChain                    (iterative repair)
    * Alg 6  BestFirstRepairableSkeleton   (K-best fallback)
    * Alg 7  CompressHeadroom              (post-repair shrink)
    * Alg 8  ValidateCutPoints             (validity predicate)

Modulus chain layout
--------------------
    q = [q_head, q_1, q_2, ..., q_R, q_tail]

``q_tail`` is the special prime for key-switching / rotation and is
EXCLUDED from ``ActiveBits`` — it is not consumed by any rescale.

Conventions
-----------
    Skeleton S*  = [s_0, s_1, ..., s_R, M+1]        (last entry = dummy sink)
    t_r          = s_r's post-rescale target scale (stage-r scale)
    t_r^base     = FindMinSF(s_r) (without headroom)
    t_0          is the scale of the source c_0; there is still a
                 "q_1" between stages 0 and 1 in our indexing.

ActiveBits at a node j in stage r (s_r ≤ j < s_{r+1}):

    ActiveBits(j) = bits(q_head) + Σ_{u=r+1..R} bits(q_u)

(Excluding q_tail.)
"""

from __future__ import annotations

import copy
import heapq
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .backward_level_dp import (
    DPResult, build_dp_table, backtrack_from, deviate_at,
    stage_edge_cost, tail_edge_cost,
)
from .graph import CostParams, RescaleGraph, propagate_scale
from .reachability import Reachability

logger = logging.getLogger("rescale_optimizer")

INF = float("inf")


@dataclass
class ModulusChain:
    """
    Modulus chain.

    Attributes
    ----------
    q_head_bits : int
        bit-width of q_head.
    q_bits : List[int]
        [bits(q_1), bits(q_2), ..., bits(q_R)] — rescale primes.
    q_tail_bits : int
        bit-width of q_tail (NOT counted in ActiveBits).
    """
    q_head_bits: int = 60
    q_bits: List[int] = field(default_factory=list)
    q_tail_bits: int = 60

    @property
    def R(self) -> int:
        return len(self.q_bits)

    @property
    def total_bits(self) -> int:
        return self.q_head_bits + sum(self.q_bits) + self.q_tail_bits

    def active_bits(self, r: int) -> int:
        """
        ActiveBits for a node in stage r (0 ≤ r ≤ R).

            = bits(q_head) + Σ_{u=r+1..R} bits(q_u)

        (``q_u`` is consumed by the rescale at ``s_u``.)
        """
        return self.q_head_bits + sum(self.q_bits[r:])

    def copy(self) -> "ModulusChain":
        return ModulusChain(
            q_head_bits=self.q_head_bits,
            q_bits=list(self.q_bits),
            q_tail_bits=self.q_tail_bits,
        )

    def summary(self) -> str:
        parts = [f"q_head={self.q_head_bits}"]
        parts += [f"q_{i + 1}={b}" for i, b in enumerate(self.q_bits)]
        parts += [f"q_tail={self.q_tail_bits}"]
        return (f"ModulusChain({', '.join(parts)})  "
                f"total={self.total_bits} bits, R={self.R}")


def _stage_index_of(skeleton: Sequence[int], j: int) -> int:
    """
    Given a skeleton S* = [s_0, s_1, ..., s_R, M+1] and a cut-point j
    with 0 ≤ j ≤ M, return r = max{u : s_u ≤ j}.

    (We assume s_0 = 0 so the answer is always ≥ 0.)
    """
    r = 0
    for u in range(len(skeleton) - 1):
        if skeleton[u] <= j:
            r = u
        else:
            break
    return r


def validate_cut_points(
    graph: RescaleGraph,
    skeleton: Sequence[int],
    chain: ModulusChain,
    t: Sequence[int],
    A_budgets: Sequence[int],
) -> Tuple[bool, Optional[int], Optional[int], int]:
    """
    Check the amplitude-budget / active-bits invariant at every real
    cut point j ∈ 0..M.

    For each j:
        r        = stage index of j in the skeleton
        s_hat_j  = PropagateScale(t_r, nodes_{s_r → j})
        B_j^act  = chain.active_bits(r)
        Δ_j      = s_hat_j + A_j^budget - B_j^act

    Returns
    -------
    (valid, j*, r*, Δ*)
        valid == False iff any Δ_j > 0  (that j is the first violator).
    """
    M = graph.M

    for j in range(M + 1):
        r = _stage_index_of(skeleton, j)
        t_r = int(t[r])
        s_r = skeleton[r]
        path = graph.nodes_between(s_r, j)
        s_hat = propagate_scale(t_r, path)
        B_act = chain.active_bits(r)
        A_j = int(A_budgets[j])
        delta = s_hat + A_j - B_act
        if delta > 0:
            logger.debug("ValidateCutPoints: violation at j=%d (r=%d): "
                         "s_hat=%d, A=%d, B_act=%d, Δ=%d",
                         j, r, s_hat, A_j, B_act, delta)
            return False, j, r, int(delta)

    return True, None, None, 0


def _next_larger_legal(bits: int, q_legal_min: int, q_legal_max: int) -> Optional[int]:
    """The next integer > bits in [q_legal_min, q_legal_max]; None if none."""
    nb = bits + 1
    if nb < q_legal_min:
        nb = q_legal_min
    if nb > q_legal_max:
        return None
    return nb


def repair_chain(
    graph: RescaleGraph,
    skeleton: Sequence[int],
    chain: ModulusChain,
    t: List[int],
    t_base: Sequence[int],
    A_budgets: Sequence[int],
) -> Tuple[ModulusChain, List[int], bool]:
    """
    Alg 5. Iteratively try to repair the modulus chain.

    On each violation at (j*, r*, Δ*): walk u = r*+1..R and try to bump
    ``bits(q_u)`` up to the next legal size.

    After bumping ``q_u`` we **re-walk** ``t_v`` (v = u..R) using
    ``propagate_scale`` so that the chain-consistency invariant
    ``t_v = PropagateScale(t_{v-1}, path_v) - bits(q_v)`` is preserved.
    This is essential when the path between two skeleton points
    contains a symmetric ``CTCT_MUL`` (s → 2s): a bump of ``q_u`` by
    ``δ`` shrinks ``t_u`` by ``δ`` but ``t_{u+1}`` by ``2^k · δ``
    (k = #symmetric CTCTs on the path between u and u+1), which the
    old linear formula ``t_v -= δ`` got wrong.

    Returns
    -------
    (chain, t, valid_flag) where ``valid_flag`` is True iff the chain
    could be repaired to pass ValidateCutPoints.
    """
    R = chain.R
    q_min = graph.q_legal_min
    q_max = graph.q_legal_max

    chain = chain.copy()
    t = list(t)
    skel = list(skeleton)


    max_iter = (R + 1) * (q_max - q_min + 1) * 4
    for _ in range(max_iter):
        valid, j_star, r_star, delta = validate_cut_points(
            graph, skel, chain, t, A_budgets,
        )
        if valid:
            return chain, t, True

        repaired = False
        for u in range(r_star + 1, R + 1):
            current_bits = int(chain.q_bits[u - 1])
            new_bits = _next_larger_legal(current_bits, q_min, q_max)
            if new_bits is None:
                continue


            tentative_q = list(chain.q_bits)
            tentative_q[u - 1] = new_bits
            tentative_t = list(t[:u])
            ok = True
            for v in range(u, R + 1):
                path = graph.nodes_between(skel[v - 1], skel[v])
                sf_pre = propagate_scale(tentative_t[v - 1], path)
                tv = int(sf_pre) - int(tentative_q[v - 1])
                if tv < int(t_base[v]):
                    ok = False
                    break
                tentative_t.append(tv)
            if not ok:
                continue

            chain.q_bits[u - 1] = new_bits
            t = tentative_t
            repaired = True
            break

        if not repaired:
            return chain, t, False

    logger.warning("RepairChain: hit iteration bound")
    return chain, t, False


def compress_headroom(
    graph: RescaleGraph,
    skeleton: Sequence[int],
    t: Sequence[int],
    t_base: Sequence[int],
    chain: ModulusChain,
    q_legal_min: int,
    q_legal_max: int,
) -> Tuple[List[int], ModulusChain]:
    """
    Alg 7 (chain-consistent). Shrink unused headroom while keeping
    ``t`` and ``q`` self-consistent for direct SEAL/OpenFHE deployment.

    Walks the skeleton forward.  At stage r, given the already-decided
    ``t_new[r-1]``, computes the *true* pre-rescale scale via
    ``PropagateScale`` (so symmetric ``CTCT_MUL`` doublings are honored),
    then picks the smallest legal prime ``q_r`` such that the post-rescale
    scale is at least ``t_base[r]``::

        sf_pre  = PropagateScale(t_new[r-1], path_r)
        q_cand  = sf_pre - t_base[r]               # max compression
        q_r'    = clip(q_cand, q_legal_min, min(q_legal_max, bits(q_r)))
        t_new[r]= sf_pre - q_r'

    Three branches:

      1. ``q_legal_min ≤ q_cand ≤ q_upper``: max compression feasible,
         ``t_new[r] = t_base[r]``.
      2. ``q_cand > q_upper``: cannot drop enough at this stage; clip
         at ``q_upper`` and keep the residual headroom in ``t_new[r]``.
      3. ``q_cand < q_legal_min``: drop too small even before
         compression; revert this stage to the pre-compress values
         (no compression here).

    The post-condition is::

        t_new[r] = PropagateScale(t_new[r-1], path_r) - q_new[r-1]

    so that walking the chain forward from ``t_new[0]`` reproduces the
    returned ``t_new`` exactly — which is what SEAL would actually
    produce after each rescale.

    Compared to the previous linear-formula version, this fixes the
    ``(2^k - 1) · c[r-1]``-bit slack that accrued whenever path
    contained ``k`` symmetric CTCTs.
    """
    R = chain.R
    assert len(t) == R + 1 == len(t_base)
    skel = list(skeleton)
    t_old = [int(x) for x in t]
    tb = [int(x) for x in t_base]

    new_q: List[int] = list(int(x) for x in chain.q_bits)
    t_new: List[int] = [0] * (R + 1)


    t_new[0] = tb[0]

    for r in range(1, R + 1):
        path = graph.nodes_between(skel[r - 1], skel[r])
        sf_pre_new = int(propagate_scale(t_new[r - 1], path))

        q_old = int(chain.q_bits[r - 1])
        q_lower = int(q_legal_min)
        q_upper = min(int(q_legal_max), q_old)
        q_candidate = sf_pre_new - tb[r]

        if q_lower <= q_candidate <= q_upper:
            new_q[r - 1] = q_candidate
            t_new[r] = tb[r]
        elif q_candidate > q_upper:
            new_q[r - 1] = q_upper
            t_new[r] = sf_pre_new - q_upper
        else:
            new_q[r - 1] = q_old
            t_new[r] = sf_pre_new - q_old
            if t_new[r] < tb[r]:


                t_new[r] = max(t_old[r], sf_pre_new - q_old)
                new_q[r - 1] = q_old

    new_chain = ModulusChain(
        q_head_bits=int(chain.q_head_bits),
        q_bits=list(new_q),
        q_tail_bits=int(chain.q_tail_bits),
    )
    return t_new, new_chain


def derive_stage_parameters(
    graph: RescaleGraph,
    dp_result: DPResult,
) -> Tuple[List[int], List[int]]:
    """
    From a DP result, derive:
        t_base[r]  = baseline_scale_bits of skeleton[r]     (r = 0..R)
        d_hat[r]   = drop_bits of stage edge (s_{r-1} → s_r) (r = 1..R)

    Notes
    -----
    * The skeleton's last entry is the dummy sink (M+1), which has no
      t_base / drop_bits; the loops stop at s_R.
    * drop_bits list excludes the final tail-edge drop (it is None).
    """
    skel = dp_result.skeleton

    R = len(skel) - 2
    t_base = [int(graph.cut_points[skel[r]].baseline_scale_bits)
              for r in range(R + 1)]
    d_hat: List[int] = []
    for k, (kind, _, _) in enumerate(dp_result.edges):
        if kind == "stage":
            d_hat.append(int(dp_result.drop_bits[k]))
    return t_base, d_hat


def initial_chain_from_skeleton(
    graph: RescaleGraph,
    dp_result: DPResult,
    q_head_bits: int,
    q_tail_bits: int,
) -> Tuple[ModulusChain, List[int], List[int]]:
    """
    Given a DP skeleton, build the *initial* modulus chain and the
    (t, t_base) vectors.  Sets:

        bits(q_r)   = d_hat_r                       (r = 1..R)
        t_0         = t_0^base + h_sf
        t_r         = PropagateScale(t_{r-1}, path_r) - bits(q_r)   (r ≥ 1)

    Note
    ----
    The vector ``t`` is *chain-consistent by construction*: walking
    forward from ``t_0`` using the chain primes ``q_1..q_R`` reproduces
    the very same ``t_r``.  This is the invariant maintained by
    ``repair_chain`` and ``compress_headroom`` so that the final
    output ``(t, q)`` can be deployed directly to SEAL/OpenFHE.
    """
    t_base, d_hat = derive_stage_parameters(graph, dp_result)
    h_sf = graph.h_sf
    skel = list(dp_result.skeleton)
    R = len(t_base) - 1

    chain = ModulusChain(
        q_head_bits=q_head_bits,
        q_bits=list(d_hat),
        q_tail_bits=q_tail_bits,
    )

    t: List[int] = [int(t_base[0]) + int(h_sf)]
    for r in range(1, R + 1):
        path = graph.nodes_between(skel[r - 1], skel[r])
        sf_pre = propagate_scale(t[r - 1], path)
        t.append(int(sf_pre) - int(chain.q_bits[r - 1]))

    return chain, t, t_base


def _skeleton_key(dp_result: DPResult) -> Tuple:
    """A hashable key for a skeleton + L_star (for the visited set)."""
    return (dp_result.L_star, tuple(dp_result.skeleton),
            tuple(dp_result.edges))


def best_first_repairable_skeleton(
    graph: RescaleGraph,
    reach: Reachability,
    cost: CostParams,
    failed_dp: DPResult,
    A_budgets: Sequence[int],
    q_head_bits: int,
    q_tail_bits: int,
    max_expansions: int = 64,
) -> Tuple[Optional[List[int]], Optional[List[int]],
           Optional[ModulusChain], Optional[List[int]], bool]:
    """
    Alg 6. Return the *next cheapest* skeleton whose chain is repairable.

    Parameters
    ----------
    failed_dp : DPResult
        The initial best skeleton that failed Alg 5.
    max_expansions : int
        Safety cap on number of heap expansions.

    Returns
    -------
    (skeleton, t_base, chain, t, valid)

        valid=False → no repairable skeleton found; all other fields None.
    """
    heap: List[Tuple[float, int, DPResult]] = []
    visited = {_skeleton_key(failed_dp)}
    counter = 0

    def enqueue_deviations(src: DPResult) -> None:
        nonlocal counter
        for t_idx in range(1, len(src.edges) + 1):
            dev = deviate_at(graph, reach, cost, src, t_idx)
            if not dev.is_feasible:
                continue
            key = _skeleton_key(dev)
            if key in visited:
                continue
            visited.add(key)
            heapq.heappush(heap, (dev.total_cost, counter, dev))
            counter += 1

    enqueue_deviations(failed_dp)

    expansions = 0
    while heap and expansions < max_expansions:
        expansions += 1
        _, _, cand = heapq.heappop(heap)

        chain, t, t_base = initial_chain_from_skeleton(
            graph, cand, q_head_bits, q_tail_bits,
        )
        chain2, t2, ok = repair_chain(
            graph, cand.skeleton, chain, t, t_base, A_budgets,
        )
        if ok:
            logger.info("BestFirstRepairableSkeleton: repaired after "
                        "%d expansion(s): cost=%.3f, skeleton=%s",
                        expansions, cand.total_cost, cand.skeleton)
            return list(cand.skeleton), list(t_base), chain2, t2, True

        enqueue_deviations(cand)

    if expansions == 0:
        logger.warning("BestFirstRepairableSkeleton: no feasible deviation "
                       "skeletons exist (the failed skeleton is the only "
                       "L-reachable path)")
    else:
        logger.warning("BestFirstRepairableSkeleton: no repairable skeleton "
                       "found after %d expansions", expansions)
    return None, None, None, None, False


@dataclass
class ChainResult:
    skeleton: List[int] = field(default_factory=list)
    chain: Optional[ModulusChain] = None
    t: List[int] = field(default_factory=list)
    t_base: List[int] = field(default_factory=list)
    valid: bool = False
    used_fallback: bool = False

    def __repr__(self) -> str:
        tag = "VALID" if self.valid else "INVALID"
        return (f"ChainResult({tag}, fallback={self.used_fallback}, "
                f"skeleton={self.skeleton}, "
                f"chain={self.chain.summary() if self.chain else None})")


def construct_modulus_chain(
    graph: RescaleGraph,
    reach: Reachability,
    cost: CostParams,
    dp_result: DPResult,
    A_budgets: Sequence[int],
    q_head_bits: int = 60,
    q_tail_bits: int = 60,
    max_expansions: int = 64,
) -> ChainResult:
    """
    Alg 4. Top-level orchestrator for building + validating the
    modulus chain.

    Pipeline:
        1. Initialise the chain from dp_result (S*, d_hat, t_base).
        2. Try RepairChain on S*.
        3. On failure, BestFirstRepairableSkeleton finds an alternative.
        4. CompressHeadroom on the repairable (S, t_base, q, t).
        5. Final ValidateCutPoints on (S, q_final, t_final).
    """

    chain0, t0, t_base0 = initial_chain_from_skeleton(
        graph, dp_result, q_head_bits, q_tail_bits
    )
    logger.info("Initial chain: %s", chain0.summary())


    chain_r, t_r, ok = repair_chain(
        graph, dp_result.skeleton, chain0, t0, t_base0, A_budgets,
    )
    if ok:
        S_cand = list(dp_result.skeleton)
        t_base_cand = list(t_base0)
        chain_cand = chain_r
        t_cand = t_r
        used_fallback = False
        logger.info("RepairChain: OK on initial skeleton")
    else:
        logger.info("RepairChain failed on initial skeleton; "
                    "falling back to BestFirstRepairableSkeleton")

        S_cand, t_base_cand, chain_cand, t_cand, ok2 =\
            best_first_repairable_skeleton(
                graph, reach, cost, dp_result, A_budgets,
                q_head_bits, q_tail_bits, max_expansions,
            )
        if not ok2:
            return ChainResult(skeleton=[], chain=None, t=[], t_base=[],
                               valid=False, used_fallback=True)
        used_fallback = True


    t_final, chain_final = compress_headroom(
        graph, S_cand, t_cand, t_base_cand, chain_cand,
        graph.q_legal_min, graph.q_legal_max,
    )
    logger.info("After CompressHeadroom: %s", chain_final.summary())


    valid, j_star, r_star, delta = validate_cut_points(
        graph, S_cand, chain_final, t_final, A_budgets,
    )
    if not valid:
        logger.warning("Final ValidateCutPoints FAILED at j=%d (Δ=%d)",
                       j_star, delta)
        return ChainResult(skeleton=[], chain=chain_final, t=list(t_final),
                           t_base=list(t_base_cand),
                           valid=False, used_fallback=used_fallback)

    return ChainResult(
        skeleton=list(S_cand),
        chain=chain_final,
        t=list(t_final),
        t_base=list(t_base_cand),
        valid=True,
        used_fallback=used_fallback,
    )
