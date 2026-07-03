"""
rescale_optimizer/backward_level_dp.py

Alg 3: Backward Level DP for cost-optimal rescale placement.

State
-----
    DP[i, l]   =   minimum cost to reach the DUMMY_SINK c_{M+1}
                   from cut point c_i using exactly l remaining rescales.

Terminal
--------
    DP[M+1, 0] = 0

Transitions at (i, l)
---------------------
    stage edge (i → j), j ≤ M :
        requires  l ≥ 1  and  Feas(i, j, l) == True
        next state: (j, l - 1)
        cost:  ~C(i,j,l) = λ₀ + λ₁·l + α·E(i,j,l) + β·d(i,j)

    tail edge  (i → M+1) :
        requires  l == 0  and  TailFeas(i, M+1, 0) == True
        next state: (M+1, 0)                   (terminal)
        cost:  ~C(i,M+1,0) = λ₀ + α·E_tail(i, 0)

Starting L
----------
The DP is initialised for every L ∈ FwdSteps[M+1].  We pick the L
minimising DP[0, L] and backtrack from (0, L*).

Public API
----------
    run_backward_dp(graph, reach, cost)              → DPResult
    build_dp_table(graph, reach, cost,               → (DP, NEXT)
                   forbidden_edge_at_state=None)
    backtrack_from(DP, NEXT, graph, cost, i, l)      → DPResult
    deviate_at(graph, reach, cost, dp_result, t)     → DPResult  (for Alg 6)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .graph import CostParams, RescaleGraph, StageEdge, TailEdge
from .reachability import Reachability

logger = logging.getLogger("rescale_optimizer")

INF = float("inf")


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------

@dataclass
class DPResult:
    """
    Result of one backward-DP solve (or one deviation).

    Fields
    ------
    skeleton : List[int]
        [c_0, i_1, ..., i_R, c_{M+1}]  — cut-point indices, in order.
    edges : List[(kind, i, j)]
        kind ∈ {"stage", "tail"}; i and j are cut-point indices.
    total_cost : float
        Sum of ~C(·) over all edges.
    L_star : int
        Starting remaining-rescale level at c_0 (= number of stage edges).
    drop_bits : List[Optional[int]]
        d(i,j) for stage edges; None for the tail edge.
    per_edge_cost : List[float]
        ~C(·) for each edge, in order.
    """
    skeleton: List[int] = field(default_factory=list)
    edges: List[Tuple[str, int, int]] = field(default_factory=list)
    total_cost: float = INF
    L_star: int = 0
    drop_bits: List[Optional[int]] = field(default_factory=list)
    per_edge_cost: List[float] = field(default_factory=list)

    @property
    def is_feasible(self) -> bool:
        return self.total_cost < INF and len(self.skeleton) > 0

    def __repr__(self) -> str:
        return (f"DPResult(L={self.L_star}, cost={self.total_cost:.3f}, "
                f"skeleton={self.skeleton})")


# ---------------------------------------------------------------------------
# Cost helpers
# ---------------------------------------------------------------------------

def stage_edge_cost(edge: StageEdge, l: int, p: CostParams) -> float:
    """~C(i,j,l) for a stage edge."""
    return (p.lambda_0
            + p.lambda_1 * l
            + p.alpha * edge.E(l)
            + p.beta * edge.drop_bits)


def tail_edge_cost(edge: TailEdge, p: CostParams) -> float:
    """~C(i, M+1, 0) for a tail edge."""
    return p.lambda_0 + p.alpha * edge.E()


# ---------------------------------------------------------------------------
# DP table construction
# ---------------------------------------------------------------------------

def build_dp_table(
    graph: RescaleGraph,
    reach: Reachability,
    cost: CostParams,
    forbidden_edge_at_state: Optional[Tuple[int, int, int, str]] = None,
) -> Tuple[Dict[Tuple[int, int], float],
           Dict[Tuple[int, int], Tuple[str, int, int]]]:
    """
    Build the full DP table.

    Parameters
    ----------
    forbidden_edge_at_state : (i, l, j, kind) or None
        If set, the transition (i, l) -- (kind) --> j is not allowed.
        Used for deviation-constrained subproblems in Alg 6.

    Returns
    -------
    DP   : (i, l) -> minimum cost to reach sink
    NEXT : (i, l) -> (kind, next_i, next_l) chosen transition
    """
    M = graph.M
    sink = graph.dummy_sink_index

    DP: Dict[Tuple[int, int], float] = {(sink, 0): 0.0}
    NEXT: Dict[Tuple[int, int], Tuple[str, int, int]] = {}
    stage_successor_edges: Dict[int, List[Tuple[int, StageEdge]]] = {}
    for (ii, j), edge in graph.stage_edges.items():
        stage_successor_edges.setdefault(ii, []).append((j, edge))

    # Process cut points in reverse topological order for every l.
    for i in range(M, -1, -1):
        for l in sorted(reach.bwd_steps.get(i, set())):
            best = INF
            best_tr: Optional[Tuple[str, int, int]] = None

            # stage edges
            if l >= 1:
                for j, edge in stage_successor_edges.get(i, ()):
                    if not reach.feas_stage(i, j, l):
                        continue
                    if (forbidden_edge_at_state is not None and
                            forbidden_edge_at_state == (i, l, j, "stage")):
                        continue
                    c_edge = stage_edge_cost(edge, l, cost)
                    nxt = DP.get((j, l - 1), INF)
                    total = c_edge + nxt
                    if total < best:
                        best = total
                        best_tr = ("stage", j, l - 1)

            # tail edge
            if l == 0 and i in graph.tail_edges:
                not_forbidden = (forbidden_edge_at_state is None or
                                 forbidden_edge_at_state != (i, 0, sink, "tail"))
                if not_forbidden:
                    edge_t = graph.tail_edges[i]
                    c_edge = tail_edge_cost(edge_t, cost)
                    total = c_edge  # DP[(sink, 0)] == 0
                    if total < best:
                        best = total
                        best_tr = ("tail", sink, 0)

            if best < INF:
                DP[(i, l)] = best
                if best_tr is not None:
                    NEXT[(i, l)] = best_tr

    return DP, NEXT


# ---------------------------------------------------------------------------
# Back-tracking
# ---------------------------------------------------------------------------

def backtrack_from(
    DP: Dict[Tuple[int, int], float],
    NEXT: Dict[Tuple[int, int], Tuple[str, int, int]],
    graph: RescaleGraph,
    cost: CostParams,
    i_start: int,
    l_start: int,
) -> DPResult:
    """Follow NEXT pointers from (i_start, l_start) to DUMMY_SINK."""
    sink = graph.dummy_sink_index
    if (i_start, l_start) not in DP or DP[(i_start, l_start)] == INF:
        return DPResult()

    skeleton = [i_start]
    edges: List[Tuple[str, int, int]] = []
    drops: List[Optional[int]] = []
    costs: List[float] = []

    i, l = i_start, l_start
    while (i, l) in NEXT:
        kind, j, nxt_l = NEXT[(i, l)]
        edges.append((kind, i, j))

        if kind == "stage":
            edge = graph.stage_edges[(i, j)]
            drops.append(edge.drop_bits)
            costs.append(stage_edge_cost(edge, l, cost))
        else:
            edge_t = graph.tail_edges[i]
            drops.append(None)
            costs.append(tail_edge_cost(edge_t, cost))

        skeleton.append(j)
        i, l = j, nxt_l
        if i == sink:
            break

    return DPResult(
        skeleton=skeleton,
        edges=edges,
        total_cost=sum(costs),
        L_star=sum(1 for k, _, _ in edges if k == "stage"),
        drop_bits=drops,
        per_edge_cost=costs,
    )


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def run_backward_dp(graph: RescaleGraph,
                    reach: Reachability,
                    cost: CostParams) -> DPResult:
    """Run backward DP and return the best (lowest-cost) skeleton."""
    sink = graph.dummy_sink_index
    L_choices = sorted(reach.fwd_steps.get(sink, set()))
    if not L_choices:
        logger.warning("No reachable (0 → sink) paths — DP infeasible.")
        return DPResult()

    DP, NEXT = build_dp_table(graph, reach, cost)

    best_L: Optional[int] = None
    best_total = INF
    for L in L_choices:
        c = DP.get((0, L), INF)
        if c < best_total:
            best_total = c
            best_L = L

    if best_L is None:
        return DPResult()

    result = backtrack_from(DP, NEXT, graph, cost, 0, best_L)
    return result


# ---------------------------------------------------------------------------
# Deviation (used by Alg 6 BestFirstRepairableSkeleton)
# ---------------------------------------------------------------------------

def deviate_at(graph: RescaleGraph,
               reach: Reachability,
               cost: CostParams,
               source: DPResult,
               t: int) -> DPResult:
    """
    Deviate from `source.skeleton` at the t-th edge (1-indexed):

        * force edges e_1, e_2, ..., e_{t-1}  as prefix
        * forbid edge e_t at its state (i, l)
        * let the DP pick the cheapest continuation

    Returns ``DPResult()`` (infeasible) if no deviation exists.
    """
    if t < 1 or t > len(source.edges):
        return DPResult()

    sink = graph.dummy_sink_index

    # ---------------- walk the forced prefix -------------------------
    i, l = 0, source.L_star
    prefix_skel: List[int] = [0]
    prefix_edges: List[Tuple[str, int, int]] = []
    prefix_drops: List[Optional[int]] = []
    prefix_costs: List[float] = []
    prefix_cost = 0.0

    for k in range(t - 1):
        kind, ei, ej = source.edges[k]
        if ei != i:
            return DPResult()
        if kind == "stage":
            edge = graph.stage_edges.get((ei, ej))
            if edge is None or l < 1:
                return DPResult()
            c = stage_edge_cost(edge, l, cost)
            prefix_drops.append(edge.drop_bits)
            i, l = ej, l - 1
        else:
            # A tail edge in the middle of the prefix is impossible
            # because tail edges immediately terminate at the sink.
            return DPResult()

        prefix_cost += c
        prefix_costs.append(c)
        prefix_edges.append((kind, ei, ej))
        prefix_skel.append(ej)

    # ---------------- forbid e_t at state (i, l) ---------------------
    fk, fi, fj = source.edges[t - 1]
    if fi != i:
        return DPResult()
    forbidden = (i, l, fj, fk)

    DP, NEXT = build_dp_table(graph, reach, cost,
                              forbidden_edge_at_state=forbidden)
    if (i, l) not in DP or DP[(i, l)] == INF:
        return DPResult()

    suffix = backtrack_from(DP, NEXT, graph, cost, i, l)
    if not suffix.is_feasible:
        return DPResult()

    return DPResult(
        skeleton=prefix_skel + suffix.skeleton[1:],
        edges=prefix_edges + suffix.edges,
        total_cost=prefix_cost + suffix.total_cost,
        L_star=source.L_star,
        drop_bits=prefix_drops + suffix.drop_bits,
        per_edge_cost=prefix_costs + suffix.per_edge_cost,
    )
