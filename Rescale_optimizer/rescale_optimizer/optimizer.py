"""
rescale_optimizer/optimizer.py

Top-level orchestrator for the 4-stage rescale optimization.

Pipeline:

    ┌──────────────────────────────────────────────────────────────────┐
    │ Stage 1: Feasibility-DAG construction   (build_feasibility_dag)  │
    │   → t_j^{base}, t_j = t_j^{base} + h_sf, stage & tail edges      │
    │                                                                  │
    │ Stage 2: Reachability analysis          (compute_reachability)   │
    │   → FwdSteps, BwdSteps, Feas predicate                           │
    │                                                                  │
    │ Stage 3: Backward Level-DP              (run_backward_dp)        │
    │   → (S*, cost, L*)                                               │
    │                                                                  │
    │ Stage 4: Modulus chain                  (construct_modulus_chain)│
    │   → RepairChain, [BestFirstRepairableSkeleton], CompressHeadroom,│
    │     final ValidateCutPoints.                                     │
    └──────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .backward_level_dp import DPResult, run_backward_dp
from .feasibility import build_feasibility_dag
from .graph import CostParams, RescaleGraph
from .modulus_chain import (
    ChainResult, ModulusChain, construct_modulus_chain,
)
from .reachability import Reachability, compute_reachability

logger = logging.getLogger("rescale_optimizer")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class OptimizationConfig:
    """
    Top-level optimization config.

    Attributes
    ----------
    cost_params : CostParams
        Cost-function parameters (λ₀, λ₁, α, β) for Alg 3.
    q_head_bits : int
        Bit-width of q_head (the head prime included in ActiveBits).
    q_tail_bits : int
        Bit-width of q_tail (special prime, excluded from ActiveBits).
    max_best_first_expansions : int
        Cap on Alg 6 BestFirstRepairableSkeleton heap expansions.
    """
    cost_params: CostParams = field(default_factory=CostParams)
    q_head_bits: int = 60
    q_tail_bits: int = 60
    max_best_first_expansions: int = 64


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    """
    Full result of `optimize_rescale`.

    Attributes
    ----------
    success : bool
    reachability : Reachability
    dp : DPResult
        Outcome of Alg 3 backward DP (before chain construction).
    chain_result : ChainResult
        Outcome of Alg 4 construct_modulus_chain (with repair / fallback /
        compression / final validation).
    message : str
    """
    success: bool = False
    reachability: Optional[Reachability] = None
    dp: DPResult = field(default_factory=DPResult)
    chain_result: ChainResult = field(default_factory=ChainResult)
    message: str = ""

    # convenience shortcuts --------------------------------------------

    @property
    def skeleton(self) -> List[int]:
        return self.chain_result.skeleton if self.chain_result.valid \
            else self.dp.skeleton

    @property
    def modulus_chain(self) -> Optional[ModulusChain]:
        return self.chain_result.chain

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "Rescale Optimization Result",
            "=" * 60,
            f"  success        : {self.success}",
            f"  DP cost        : {self.dp.total_cost:.4f}",
            f"  DP L*          : {self.dp.L_star}",
            f"  DP skeleton    : {self.dp.skeleton}",
            f"  DP edges       : {self.dp.edges}",
            f"  DP drop_bits   : {self.dp.drop_bits}",
        ]
        cr = self.chain_result
        lines.extend([
            f"  chain valid    : {cr.valid}",
            f"  used fallback  : {cr.used_fallback}",
            f"  final skeleton : {cr.skeleton}",
        ])
        if cr.chain is not None:
            lines.append(f"  {cr.chain.summary()}")
            lines.append(f"  final t        : {cr.t}")
            lines.append(f"  baseline t     : {cr.t_base}")
            lines.append("")
            lines.extend(_format_modulus_chain(cr.chain, cr.skeleton))
        if self.message:
            lines.append(f"  message        : {self.message}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def optimize_rescale(
    graph: RescaleGraph,
    config: Optional[OptimizationConfig] = None,
    amplitude_budgets: Optional[Sequence[int]] = None,
) -> OptimizationResult:
    """
    Run the 4-stage rescale optimization.

    Parameters
    ----------
    graph : RescaleGraph
        Must have its nodes / cut_points / noise_table / h_sf /
        q_legal_min / q_legal_max / cut-point metadata already filled
        in.  The feasibility DAG is (re)built here.
    config : OptimizationConfig, optional
    amplitude_budgets : list of int, optional
        Overrides each cut_point's ``amplitude_budget_bits``.  If None,
        uses the per-cut-point field on the graph.

    Returns
    -------
    OptimizationResult
    """
    if config is None:
        config = OptimizationConfig()

    M = graph.M
    if amplitude_budgets is None:
        amplitude_budgets = [int(graph.cut_points[j].amplitude_budget_bits)
                             for j in range(M + 1)]

    result = OptimizationResult()

    # ---- Stage 1: Feasibility DAG ------------------------------------
    logger.info("=" * 58)
    logger.info("Stage 1: Feasibility-DAG construction")
    build_feasibility_dag(graph)
    logger.info(graph.summary())

    if not graph.stage_edges and not graph.tail_edges:
        result.message = "Feasibility DAG empty — no edges satisfy Q_legal."
        logger.error(result.message)
        return result

    # ---- Stage 2: Reachability ---------------------------------------
    logger.info("=" * 58)
    logger.info("Stage 2: Reachability analysis")
    reach = compute_reachability(graph)
    result.reachability = reach

    L_choices = sorted(reach.fwd_steps.get(graph.dummy_sink_index, set()))
    if not L_choices:
        result.message = ("Reachability: no path from c_0 to c_{M+1}. "
                          "Adjust Q_legal, amplitude budgets or the graph.")
        logger.error(result.message)
        return result

    # ---- Stage 3: Backward Level-DP ----------------------------------
    logger.info("=" * 58)
    logger.info("Stage 3: Backward Level-DP (valid L's: %s)", L_choices)
    dp_result = run_backward_dp(graph, reach, config.cost_params)
    result.dp = dp_result

    if not dp_result.is_feasible:
        result.message = "Backward DP infeasible."
        logger.error(result.message)
        return result

    logger.info("Best DP: %s", dp_result)

    # ---- Stage 4: Modulus chain construction -------------------------
    logger.info("=" * 58)
    logger.info("Stage 4: Modulus chain construction (+ repair/compress)")
    chain_result = construct_modulus_chain(
        graph, reach, config.cost_params, dp_result,
        A_budgets=amplitude_budgets,
        q_head_bits=config.q_head_bits,
        q_tail_bits=config.q_tail_bits,
        max_expansions=config.max_best_first_expansions,
    )
    result.chain_result = chain_result

    if not chain_result.valid:
        result.message = ("Chain construction FAILED. "
                          "Consider increasing h_sf, relaxing budgets, "
                          "or widening Q_legal.")
        logger.warning(result.message)
        return result

    result.success = True
    result.message = ("Optimization successful."
                      + (" (via fallback)" if chain_result.used_fallback else ""))
    logger.info(result.message)
    return result


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def print_result(result: OptimizationResult) -> None:
    print(result.summary())


# ---------------------------------------------------------------------------
# Pretty printing — modulus chain + SEAL CoeffModulus ordering
# ---------------------------------------------------------------------------

def _format_modulus_chain(
    chain: ModulusChain,
    skeleton: Sequence[int],
) -> List[str]:
    """
    Build a two-view rendering of the final chain:

    1. *Algorithmic view* — the order in which rescale consumes primes:
       ``[q_head, q_1, q_2, ..., q_R, q_tail]``.  This is how Alg 4 /
       Alg 7 / ValidateCutPoints reason about the chain.

    2. *SEAL view* — the order expected by
       ``seal::CoeffModulus::Create(poly_modulus_degree, bit_sizes)``.
       SEAL's implementation rescales by *dropping the last coefficient
       prime first*, so the caller must list the primes so that the
       "first to be consumed" appears last in the middle segment, i.e.:

           ``[q_head, q_R, q_{R-1}, ..., q_1, q_tail]``

       (the head and tail keep their outer positions; only the
       intermediate primes are reversed relative to the algorithmic
       order).
    """
    R = chain.R
    total = chain.total_bits
    lines: List[str] = []
    lines.append("  Final modulus chain")
    lines.append("  " + "-" * 58)

    # --- algorithmic view ---------------------------------------------
    lines.append(
        f"  layout  : q = [q_head, q_1, ..., q_{R}, q_tail]  "
        f"(R = {R}, total = {total} bits)"
    )
    lines.append(
        f"    q_head  =  {chain.q_head_bits} bits   "
        f"(head prime; kept in ActiveBits, not consumed by rescale)"
    )
    # s_1..s_R are skeleton[1..R]
    for i, b in enumerate(chain.q_bits, start=1):
        s_i = skeleton[i] if i < len(skeleton) else "?"
        lines.append(
            f"    q_{i}     =  {b} bits   "
            f"(consumed by rescale at s_{i} = c_{s_i})"
        )
    lines.append(
        f"    q_tail  =  {chain.q_tail_bits} bits   "
        f"(special prime for key-switching / rotation; excluded from ActiveBits)"
    )

    # --- SEAL view -----------------------------------------------------
    seal_bits: List[int] = (
        [chain.q_head_bits] + list(reversed(chain.q_bits)) + [chain.q_tail_bits]
    )
    seal_labels: List[str] = (
        ["q_head"]
        + [f"q_{i}" for i in range(R, 0, -1)]
        + ["q_tail"]
    )
    seal_bits_str = ", ".join(str(b) for b in seal_bits)
    seal_labels_str = ", ".join(seal_labels)
    lines.append("")
    lines.append(
        "  SEAL CoeffModulus bit-sizes  "
        "(pass to CoeffModulus.Create in this order):"
    )
    lines.append(
        f"    [ {seal_bits_str} ]   "
        f"({len(seal_bits)} primes, total = {sum(seal_bits)} bits)"
    )
    lines.append(f"    layout:  [ {seal_labels_str} ]")
    lines.append(
        "    note  :  middle segment is reversed vs. the algorithmic view "
        "because SEAL rescales drop the last prime first."
    )
    return lines
