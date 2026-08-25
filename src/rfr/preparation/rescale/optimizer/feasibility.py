"""
rescale_optimizer/feasibility.py

Alg 1: Feasibility-DAG construction.

Pipeline:
    1. For each cut point j ∈ 0..M:
         t_j^{base} = FindMinSF(Amp_j, SNR_j, noise_table, op_type_j)
         t_j        = t_j^{base} + h_sf
    2. For every ordered pair (i, j) with 0 ≤ i < j ≤ M:
         compute s_pre(i,j)  = PropagateScale(t_i, path_{i→j})
         compute d(i,j)      = s_pre(i,j) − t_j
         stage edge feasible ⇔  d(i,j) ∈ Q_legal = [q_legal_min, q_legal_max]
    3. For every i ∈ 0..M:
         γ_tail(i) = max over v ∈ (i, M] of
                       ( PropagateScale(t_i, path_{i→v}) + A_v^{budget} )
         tail edge feasible ⇔  γ_tail(i) < q_max

Inputs are a set of ComputeNode's with already-set
`scale_delta_bits` / `stage_anchor` / `topo_order`, plus the cut-point
metadata (amp / snr / op_type / amplitude_budget_bits).  This module
populates `baseline_scale_bits`, `target_scale_bits`, `stage_edges`
and `tail_edges` on the RescaleGraph.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from .graph import (
    AmplitudeProfile,
    ComputeNode,
    CutPoint,
    NodeType,
    NoiseLookupTable,
    RescaleGraph,
    SNRRequirement,
    StageEdge,
    TailEdge,
    propagate_scale,
)

logger = logging.getLogger("rescale_optimizer")


def find_min_sf(
    amplitude: AmplitudeProfile,
    snr: SNRRequirement,
    noise_table: NoiseLookupTable,
    op_type: str = "rescale",
) -> int:
    """
    Return the smallest ``sf_bits`` satisfying the relative-error bound.

    To protect a fraction ``p`` of the data::

        p           = snr.percentile
        a_threshold = amplitude.get_value_at(1 - p)
        noise(sf) / a_threshold <= max_relative_error

    The ``(1-p)`` quantile is the smallest protected magnitude. Every value
    with ``|x| >= a_threshold`` then satisfies ``max_relative_error``.

    If the table is empty or no entry satisfies the bound, return the largest
    available ``sf_bits`` and emit a warning.
    """


    q = max(0.0, min(1.0, 1.0 - snr.percentile))
    a = amplitude.get_value_at(q)
    max_err = snr.max_relative_error

    available = noise_table.available_sf_bits(op_type)
    if not available:
        logger.warning("Noise table has no entries for op_type='%s'; "
                       "returning 0", op_type)
        return 0
    if a <= 0:
        logger.warning("Amplitude at p=%.2f is %.2e ≤ 0; "
                       "returning max sf=%d", snr.percentile, a, available[-1])
        return int(available[-1])

    for sf_bits in available:
        noise = noise_table.lookup(op_type, sf_bits)
        if noise is None:
            continue
        if noise / a <= max_err:
            return int(sf_bits)

    logger.warning("No sf satisfies SNR for op='%s' amp=%.2e max_err=%.4f; "
                   "using max sf=%d", op_type, a, max_err, available[-1])
    return int(available[-1])


def build_feasibility_dag(graph: RescaleGraph) -> RescaleGraph:
    """
    Populate baseline scales, target scales, stage edges, and tail edges.

    Preconditions:
        ``graph.nodes`` is sorted by ``topo_order``.
        ``graph.cut_points`` contains ``c_0..c_M`` and a dummy ``c_{M+1}``.
        Every node has its anchor, scale delta, count, and cost fields set.
        The noise table, headroom, and legal modulus range are configured.
    """
    M = graph.M
    nt = graph.noise_table
    h_sf = graph.h_sf
    q_min = graph.q_legal_min
    q_max = graph.q_legal_max


    for j in range(M + 1):
        cp = graph.cut_points[j]
        t_base = find_min_sf(cp.amplitude_profile, cp.snr_requirement,
                             nt, cp.op_type)
        cp.baseline_scale_bits = int(t_base)
        cp.target_scale_bits = int(t_base) + int(h_sf)
        logger.debug("cut_point[%d] '%s': t_base=%d, t=%d",
                     j, cp.node.name, cp.baseline_scale_bits,
                     cp.target_scale_bits)


    dummy = graph.cut_points[M + 1]
    dummy.baseline_scale_bits = 0
    dummy.target_scale_bits = 0


    cut_point_index_by_node_id: Dict[int, int] = {}
    for cp in graph.cut_points:
        cut_point_index_by_node_id.setdefault(id(cp.node), cp.index)
    stage_nodes: List[List[ComputeNode]] = [[] for _ in range(M + 1)]
    for node in graph.nodes:
        if node.node_type == NodeType.DUMMY_SINK:
            continue
        if node.is_cut_point:

            idx = cut_point_index_by_node_id.get(id(node))
            if idx is None:
                continue
            if idx >= 1:
                stage_nodes[idx - 1].append(node)
        else:
            k = node.stage_anchor
            if 0 <= k < M + 1:
                stage_nodes[k].append(node)
            else:
                logger.warning("Node %s has stage_anchor=%d out of range",
                               node.name, k)


    for k in range(M + 1):
        stage_nodes[k].sort(key=lambda n: n.topo_order)


    graph.stage_node_lists = stage_nodes


    graph.stage_edges.clear()
    graph.tail_edges.clear()

    for i in range(M + 1):
        t_i = graph.cut_points[i].target_scale_bits

        cumulative_nodes: List[ComputeNode] = []
        cumulative_slope = 0.0
        cumulative_intercept = 0.0
        s_pre = int(t_i)

        for v in range(i + 1, M + 1):

            stage_chunk = stage_nodes[v - 1]
            if stage_chunk:
                cumulative_nodes.extend(stage_chunk)
                s_pre = propagate_scale(s_pre, stage_chunk)
                for n in stage_chunk:
                    cumulative_slope += n.count * n.cost_slope
                    cumulative_intercept += n.count * n.cost_intercept

            t_v = graph.cut_points[v].target_scale_bits
            d = s_pre - t_v


            if q_min <= d <= q_max:
                edge = StageEdge(
                    start=i, end=v,
                    nodes_in_stage=list(cumulative_nodes),
                    pre_rescale_scale_bits=int(s_pre),
                    drop_bits=int(d),
                    total_cost_slope=float(cumulative_slope),
                    total_cost_intercept=float(cumulative_intercept),
                )
                graph.stage_edges[(i, v)] = edge


        tail_nodes: List[ComputeNode] = []
        tail_intercept = 0.0
        tail_scale = int(t_i)
        gamma_tail = -1
        for v in range(i + 1, M + 1):
            stage_chunk = stage_nodes[v - 1]
            if stage_chunk:
                tail_nodes.extend(stage_chunk)
                tail_scale = propagate_scale(tail_scale, stage_chunk)
                for n in stage_chunk:
                    tail_intercept += n.count * n.cost_intercept
            A_v = graph.cut_points[v].amplitude_budget_bits
            val = tail_scale + A_v
            if val > gamma_tail:
                gamma_tail = val


        if i == M:
            gamma_tail = 0

        if gamma_tail < q_max:
            graph.tail_edges[i] = TailEdge(
                start=i,
                nodes_in_tail=list(tail_nodes),
                gamma=int(gamma_tail),
                total_cost_intercept=float(tail_intercept),
            )

    logger.info("FeasibilityDAG: %d stage edges, %d tail edges",
                len(graph.stage_edges), len(graph.tail_edges))
    return graph
