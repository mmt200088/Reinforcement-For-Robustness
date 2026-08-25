"""
rescale_optimizer/reachability.py

Alg 2: Reachability analysis on the Feasibility-DAG.

Two quantities are computed:

    FwdSteps[j]  :  set of integers r such that there exists a path
                    c_0 → ... → c_j using exactly r rescales.

    BwdSteps[j]  :  set of integers r such that there exists a path
                    c_j → ... → c_{M+1} using exactly r remaining rescales.

Edge semantics:
    * stage edge (i, j)    contributes +1 rescale.
    * tail   edge (i, M+1) contributes 0 (no final rescale).

The three-argument feasibility predicate used downstream is

    Feas(i, j, l) =  True iff edge (i, j) is usable *and* the remaining
                     level bookkeeping is consistent, i.e.

        stage edge (i, j):  l ∈ BwdSteps[i]  and  (l - 1) ∈ BwdSteps[j]
        tail  edge (i, M+1): l ∈ BwdSteps[i]  and  l == 0

    (equivalently, we can just answer: "after taking this edge from
    state (i, l), can we still reach the sink?")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from .graph import RescaleGraph

logger = logging.getLogger("rescale_optimizer")


@dataclass
class Reachability:
    """
    Result of reachability analysis.

    Fields
    ------
    fwd_steps : Dict[int, Set[int]]
        j -> set of r for which c_0 →^r c_j reachable.
        Includes j = M+1 (DUMMY_SINK).
    bwd_steps : Dict[int, Set[int]]
        j -> set of r for which c_j →^r c_{M+1} reachable.
    """
    fwd_steps: Dict[int, Set[int]] = field(default_factory=dict)
    bwd_steps: Dict[int, Set[int]] = field(default_factory=dict)


    def is_reachable_fwd(self, j: int, r: int) -> bool:
        return r in self.fwd_steps.get(j, ())

    def is_reachable_bwd(self, j: int, r: int) -> bool:
        return r in self.bwd_steps.get(j, ())

    def feas_stage(self, i: int, j: int, l: int) -> bool:
        """
        Stage edge (i,j) with remaining levels l at i.

        Usable iff:  l ∈ BwdSteps[i] AND (l-1) ∈ BwdSteps[j] AND l ≥ 1.
        """
        if l < 1:
            return False
        return (l in self.bwd_steps.get(i, ())) and\
               ((l - 1) in self.bwd_steps.get(j, ()))

    def feas_tail(self, i: int, dummy_sink: int, l: int) -> bool:
        """Tail edge usable only when l == 0 (no levels left)."""
        return l == 0 and (0 in self.bwd_steps.get(i, ()))

    def valid_L_choices(self) -> List[int]:
        """
        Every L such that there exists a path c_0 →^L c_{M+1}.

        These are the starting levels that the Backward DP should try.
        """
        return sorted(self.fwd_steps.get(
            max(self.bwd_steps.keys()) if self.bwd_steps else 0, set()
        ))


def compute_reachability(graph: RescaleGraph) -> Reachability:
    """
    Build Reachability object for a Feasibility-DAG.

    Works by:
      - Forward DP in topo order, scanning (i, j) stage edges and tail
        edges to update FwdSteps[j] (and FwdSteps[M+1]).
      - Backward DP in reverse topo order for BwdSteps.
    """
    M = graph.M
    sink = graph.dummy_sink_index
    stage_successors: Dict[int, List[int]] = {}
    for ii, v in graph.stage_edges:
        stage_successors.setdefault(ii, []).append(v)


    fwd: Dict[int, Set[int]] = {j: set() for j in range(M + 2)}
    fwd[0].add(0)

    for j in range(M + 2):
        if not fwd[j]:
            continue

        if j <= M:
            for v in stage_successors.get(j, ()):
                for r in fwd[j]:
                    fwd[v].add(r + 1)
            if j in graph.tail_edges:
                for r in fwd[j]:
                    fwd[sink].add(r)


    bwd: Dict[int, Set[int]] = {j: set() for j in range(M + 2)}
    bwd[sink].add(0)

    for j in range(M, -1, -1):
        out_sets = bwd[j]

        for v in stage_successors.get(j, ()):
            if bwd[v]:
                for r in bwd[v]:
                    out_sets.add(r + 1)

        if j in graph.tail_edges:
            out_sets.add(0)

    logger.info("Reachability: FwdSteps[M+1]=%s,  BwdSteps[0]=%s",
                sorted(fwd[sink]) or "∅",
                sorted(bwd[0]) or "∅")

    return Reachability(fwd_steps=fwd, bwd_steps=bwd)
