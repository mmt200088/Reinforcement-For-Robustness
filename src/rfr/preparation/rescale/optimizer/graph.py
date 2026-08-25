"""
rescale_optimizer/graph.py

Core data structures for the HE rescale optimizer (new 7-algorithm pipeline).

Taxonomy of computation nodes
-----------------------------
    SOURCE     : c_0           (initial operand — "fresh ct" or PT-like;
                                 has a base scaling factor t_0 but CANNOT
                                 be rescaled; starting state of the DAG)
    CTCT_MUL   : ct × ct       (multiplication; a cut point, rescalable)
    CTPT_MUL   : ct × pt       (multiplication; a cut point, rescalable)
    ROTATION   : ct rotation   (non-multiplication; has compute cost)
    PT_OP      : pt operation  (non-multiplication; has compute cost)
    PT         : plaintext operand (leaf, carries a scaling factor Δ)
    DUMMY_SINK : c_{M+1}       (virtual node past the last multiplication;
                                 needed so that paths may finish without
                                 a final rescale — tail edges)

The last real cut point c_M is simply *the last multiplication in the
graph* (a CTCT_MUL or CTPT_MUL).  There is no separate ``SINK`` type.

Cut-point indexing
------------------
Cut-point indices 0..M are assigned in topological order:

    index 0 : SOURCE (c_0, not rescalable)
    index 1 : first multiplication (first rescalable position)
    ...
    index M : last multiplication
    index M+1 : DUMMY_SINK (virtual; for tail edges)

Every non-cut-point node belongs to exactly one *stage* (c_k, c_{k+1}],
identified by ``stage_anchor = k``.

Scale propagation
-----------------
Every node carries a ``scale_delta_bits`` — the amount the scale of
the "main ct on the path" grows when this node executes:

    SOURCE / ROTATION / PT_OP / PT / DUMMY_SINK    → 0
    CTPT_MUL                                       → Δ   (pt operand's sf bits)
    CTCT_MUL                                       → **ignored**.  The true
        delta is computed dynamically by ``propagate_scale``:
          * default (symmetric squaring): ``s → 2·s`` — both operands at
            the working scale;
          * if the node carries ``other_ct_scale_bits``: ``s → s + other``
            — the external ciphertext enters at that fixed scale.

For a stage edge c_i → c_j:

    s_pre(i, j) = t_i + Σ_{v ∈ (c_i, c_j]} v.scale_delta_bits
                       (path excludes c_i, includes c_j)

c_i contributes only via the starting scale t_i (i.e., c_i's post-
rescale scale when i ≥ 1, or the source's base scale when i = 0).

Rescale semantics along a skeleton
----------------------------------
A skeleton S* = [s_0=0, s_1, ..., s_R, M+1] has:
    * R stage edges (s_0→s_1, ..., s_{R-1}→s_R) — one rescale each,
      performed AT the destination s_k (k = 1..R).  The rescale at
      s_0 = 0 does NOT exist; c_0 is only the starting position.
    * 1 tail edge (s_R → M+1) — no rescale.
So R rescales total, all at multiplications.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Sequence, Tuple


class NodeType(Enum):
    """
    Classification of computation nodes.

    Notes
    -----
    * ``SOURCE`` is the initial operand (fresh ct or pt-like).  It
      carries a baseline scaling factor t_0 but **cannot** be
      rescaled; the first rescale can only happen at the first
      multiplication.
    * There is no dedicated "SINK" type — the last multiplication
      (CTCT_MUL / CTPT_MUL) is simply called c_M.
    * ``DUMMY_SINK`` is a virtual node at index M+1, used as the
      endpoint of tail edges when the computation finishes without a
      final rescale.
    """
    SOURCE     = auto()
    DUMMY_SINK = auto()
    CTCT_MUL   = auto()
    CTPT_MUL   = auto()
    ROTATION   = auto()
    PT_OP      = auto()
    PT         = auto()


MULTIPLICATION_TYPES = frozenset({
    NodeType.CTCT_MUL, NodeType.CTPT_MUL,
})


CUT_POINT_TYPES = frozenset({NodeType.SOURCE}) | MULTIPLICATION_TYPES


@dataclass
class AmplitudeProfile:
    """Sorted amplitude quantiles used by the SNR feasibility calculation."""
    percentiles: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)

    def get_value_at(self, percentile: float) -> float:
        """Interpolate the amplitude at a requested quantile."""
        if not self.percentiles or not self.values:
            return 0.0
        if percentile <= self.percentiles[0]:
            return self.values[0]
        if percentile >= self.percentiles[-1]:
            return self.values[-1]
        for k in range(len(self.percentiles) - 1):
            p0, v0 = self.percentiles[k], self.values[k]
            p1, v1 = self.percentiles[k + 1], self.values[k + 1]
            if p0 <= percentile <= p1:
                t = (percentile - p0) / (p1 - p0) if p1 > p0 else 0.0
                return v0 + t * (v1 - v0)
        return self.values[-1]

    @property
    def max_value(self) -> float:
        return max(self.values) if self.values else 0.0

    @staticmethod
    def constant(value: float) -> "AmplitudeProfile":
        return AmplitudeProfile(percentiles=[0.0, 1.0], values=[value, value])

    def __repr__(self) -> str:
        if not self.percentiles:
            return "AmplitudeProfile(empty)"
        parts = [f"{p:.0%}≤{v:.2e}"
                 for p, v in zip(self.percentiles, self.values)]
        return f"AmplitudeProfile({', '.join(parts)})"


@dataclass
class SNRRequirement:
    """Relative-error requirement for a protected fraction of the data."""
    percentile: float = 0.8
    max_relative_error: float = 0.01

    def __repr__(self) -> str:
        return (f"SNRRequirement(p={self.percentile:.0%}, "
                f"ε≤{self.max_relative_error:.2%})")


@dataclass
class NoiseLookupTable:
    """Noise bounds indexed by operation type and scaling-factor bits."""
    table: Dict[str, Dict[int, float]] = field(default_factory=dict)

    def lookup(self, op_type: str, sf_bits: int) -> Optional[float]:
        op_entry = self.table.get(op_type)
        if op_entry is None:
            return None
        return op_entry.get(sf_bits)

    def available_sf_bits(self, op_type: str) -> List[int]:
        return sorted(self.table.get(op_type, {}).keys())

    def __repr__(self) -> str:
        return f"NoiseLookupTable(ops={list(self.table.keys())})"


@dataclass
class ComputeNode:
    """One operation in the Rescale optimization graph."""
    node_id: int
    name: str
    node_type: NodeType
    topo_order: int = 0
    stage_anchor: int = 0
    scale_delta_bits: int = 0
    count: int = 1
    cost_slope: float = 0.0
    cost_intercept: float = 0.0
    other_ct_scale_bits: Optional[int] = None


    @property
    def is_rescalable(self) -> bool:
        """True iff a rescale can be inserted right after this node's
        execution — i.e. this is a multiplication (CTCT_MUL / CTPT_MUL).
        SOURCE is NOT rescalable."""
        return self.node_type in MULTIPLICATION_TYPES

    @property
    def is_cut_point(self) -> bool:
        """True iff this node carries a cut-point index (0..M).
        SOURCE + all multiplications return True; rotation / pt_op / pt
        return False; DUMMY_SINK is virtual (handled separately)."""
        return self.node_type in CUT_POINT_TYPES

    @property
    def is_multiplication(self) -> bool:
        return self.node_type in MULTIPLICATION_TYPES

    def unit_cost(self, level: int) -> float:
        return self.cost_slope * level + self.cost_intercept

    def weighted_cost(self, level: int) -> float:
        return self.count * self.unit_cost(level)


@dataclass
class CutPoint:
    """A legal rescale location in topological order."""
    index: int
    node: ComputeNode
    amplitude_profile: AmplitudeProfile = field(default_factory=AmplitudeProfile)
    snr_requirement: SNRRequirement = field(default_factory=SNRRequirement)
    op_type: str = "rescale"
    amplitude_budget_bits: int = 15
    baseline_scale_bits: int = 0
    target_scale_bits: int = 0

    @property
    def is_dummy_sink(self) -> bool:
        return self.node.node_type == NodeType.DUMMY_SINK


@dataclass
class StageEdge:
    """A feasible rescale stage between two real cut points."""
    start: int
    end: int
    nodes_in_stage: List[ComputeNode] = field(default_factory=list)
    pre_rescale_scale_bits: int = 0
    drop_bits: int = 0
    total_cost_slope: float = 0.0
    total_cost_intercept: float = 0.0

    def E(self, level: int) -> float:
        return self.total_cost_slope * level + self.total_cost_intercept


@dataclass
class TailEdge:
    """A feasible path from a real cut point to the sink without another rescale."""
    start: int
    nodes_in_tail: List[ComputeNode] = field(default_factory=list)
    gamma: int = 0
    total_cost_intercept: float = 0.0

    def E(self) -> float:
        return self.total_cost_intercept


@dataclass
class CostParams:
    """Weights for stage, level, execution, and modulus-drop costs."""
    lambda_0: float = 1.0
    lambda_1: float = 0.0
    alpha: float = 1.0
    beta: float = 0.1


def propagate_scale(start_scale_bits: int,
                    path_nodes: Sequence[ComputeNode]) -> int:
    """Propagate scale bits through a sequence of plaintext and ciphertext operations."""
    s = int(start_scale_bits)
    for node in path_nodes:
        if node.node_type == NodeType.CTCT_MUL:
            if node.other_ct_scale_bits is not None:
                s = s + int(node.other_ct_scale_bits)
            else:
                s = 2 * s
        else:
            s += int(node.scale_delta_bits)
    return s


@dataclass
class RescaleGraph:
    """Complete feasibility and cost graph for Rescale placement."""
    nodes: List[ComputeNode] = field(default_factory=list)
    cut_points: List[CutPoint] = field(default_factory=list)
    stage_edges: Dict[Tuple[int, int], StageEdge] = field(default_factory=dict)
    tail_edges: Dict[int, TailEdge] = field(default_factory=dict)


    stage_node_lists: List[List[ComputeNode]] = field(default_factory=list)

    noise_table: NoiseLookupTable = field(default_factory=NoiseLookupTable)

    h_sf: int = 10
    q_legal_min: int = 30
    q_legal_max: int = 60


    @property
    def M(self) -> int:
        """Index of the last real cut point."""
        return len(self.cut_points) - 2

    @property
    def dummy_sink_index(self) -> int:
        return self.M + 1

    @property
    def num_real_cut_points(self) -> int:
        return self.M + 1

    @property
    def q_max(self) -> int:
        return self.q_legal_max


    def get_cut_point(self, index: int) -> CutPoint:
        return self.cut_points[index]

    def get_stage_edge(self, i: int, j: int) -> Optional[StageEdge]:
        return self.stage_edges.get((i, j))

    def get_tail_edge(self, i: int) -> Optional[TailEdge]:
        return self.tail_edges.get(i)

    def is_stage_feasible(self, i: int, j: int) -> bool:
        return (i, j) in self.stage_edges

    def is_tail_feasible(self, i: int) -> bool:
        return i in self.tail_edges

    def feasible_successors(self, i: int) -> List[int]:
        """Return feasible DAG successors of a cut point."""
        succ: List[int] = sorted(j for (ii, j) in self.stage_edges if ii == i)
        if i in self.tail_edges:
            succ.append(self.dummy_sink_index)
        return succ

    def feasible_predecessors(self, j: int) -> List[int]:
        """Return feasible DAG predecessors of a cut point."""
        if j == self.dummy_sink_index:
            return sorted(self.tail_edges.keys())
        return sorted(ii for (ii, jj) in self.stage_edges if jj == j)

    def nodes_between(self, a: int, b: int) -> List[ComputeNode]:
        """Return graph nodes in the half-open cut-point interval ``(a, b]``."""
        if a == b:
            return []
        if a > b or a < 0 or b > self.M:
            raise ValueError(f"nodes_between: invalid range a={a}, b={b}, M={self.M}")
        out: List[ComputeNode] = []
        for k in range(a, b):
            out.extend(self.stage_node_lists[k])
        return out


    def summary(self) -> str:
        lines = [
            f"RescaleGraph: M={self.M}  (c_0 .. c_{self.M} + dummy c_{self.M + 1})",
            f"  nodes             : {len(self.nodes)}",
            f"  stage edges       : {len(self.stage_edges)}",
            f"  tail  edges       : {len(self.tail_edges)}",
            f"  Q_legal           : [{self.q_legal_min}, {self.q_legal_max}]",
            f"  h_sf              : {self.h_sf}",
            f"  noise_table       : {self.noise_table}",
        ]
        return "\n".join(lines)
