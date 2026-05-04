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


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

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
    SOURCE     = auto()   # c_0  initial operand (not rescalable)
    DUMMY_SINK = auto()   # c_{M+1}  virtual terminal
    CTCT_MUL   = auto()   # ct × ct  – rescalable multiplication
    CTPT_MUL   = auto()   # ct × pt  – rescalable multiplication
    ROTATION   = auto()   # rotation (non-mul, has compute cost)
    PT_OP      = auto()   # plaintext arithmetic (non-mul, has compute cost)
    PT         = auto()   # plaintext leaf (no cost)


#: Multiplication-like node types — the only positions where a rescale
#: operation can be inserted.
MULTIPLICATION_TYPES = frozenset({
    NodeType.CTCT_MUL, NodeType.CTPT_MUL,
})

#: Node types that occupy a cut-point index (0..M).  Includes SOURCE
#: (index 0, not rescalable) plus all multiplications (indices 1..M).
#: ``DUMMY_SINK`` (index M+1) is handled separately because it is
#: purely virtual.
CUT_POINT_TYPES = frozenset({NodeType.SOURCE}) | MULTIPLICATION_TYPES


# ---------------------------------------------------------------------------
# Amplitude Profile / SNR Requirement / Noise Lookup Table
# ---------------------------------------------------------------------------

@dataclass
class AmplitudeProfile:
    """
    幅度分布：两个等长升序数组。

        percentiles : List[float]   ∈ (0, 1)
        values      : List[float]   升序

    percentiles[k] 比例的元素的幅度 ≤ values[k]。
    """
    percentiles: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)

    def get_value_at(self, percentile: float) -> float:
        """线性插值查询给定百分位的幅度值。"""
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
    """
    信噪比要求。

    Semantics
    ---------
    ``percentile``  = fraction of data we want to protect
                      (e.g. 0.8 ⇒ "80% of data must satisfy the bound").
    ``max_relative_error``  = ε，noise / |x| 上限。

    Inside :func:`find_min_sf`, the worst-case amplitude used as the
    divisor is ``amplitude.get_value_at(1 - percentile)`` — i.e. the
    (1−p) quantile of the CDF, which is the *smallest* magnitude among
    the protected p fraction.  Larger p ⇒ smaller reference magnitude
    ⇒ stricter sf requirement.
    """
    percentile: float = 0.8
    max_relative_error: float = 0.01

    def __repr__(self) -> str:
        return (f"SNRRequirement(p={self.percentile:.0%}, "
                f"ε≤{self.max_relative_error:.2%})")


@dataclass
class NoiseLookupTable:
    """
    噪声查表：  (op_type, sf_bits) → noise_bound

        table = {
            "rescale": {30: 1e-9, 35: 3e-10, 40: 1e-10, ...},
            "ctmul":   {...},
            "ctpt":    {...},
        }
    """
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


# ---------------------------------------------------------------------------
# Compute Node
# ---------------------------------------------------------------------------

@dataclass
class ComputeNode:
    """
    计算图中的一个运算节点。

    Attributes
    ----------
    node_id : int
    name : str
    node_type : NodeType
    topo_order : int
        全局拓扑排序位置（所有节点，包括 pt / rotation / 乘法）。
    stage_anchor : int
        对 *非* cut-point 节点：它落在 (c_{stage_anchor}, c_{stage_anchor+1}] 之间。
        对 cut-point 节点：stage_anchor = 其 cut-point index。
    scale_delta_bits : int
        该节点为"路径上的主 ct 的 scale"所增加的 bit 数。
          SOURCE / PT / ROTATION / PT_OP / DUMMY_SINK : 0
          CTPT_MUL : Δ  (pt 操作数的 scaling factor，由 SNR 决定)
          CTCT_MUL : **忽略**。真实 delta 由 ``propagate_scale`` 动态
                     决定 —— 见 ``other_ct_scale_bits``。
    other_ct_scale_bits : Optional[int]
        仅对 ``CTCT_MUL`` 有意义；其它节点应为 ``None``。
          * ``None`` (缺省): 两个操作数都假设在当前工作 scale 上，
            ``s → 2·s`` (对称 squaring 语义)。
          * 非 ``None``: 外部 ct 以 ``other_ct_scale_bits`` 位的 scale
            进来，``s → s + other_ct_scale_bits`` (非对称 CTCT)。
    count : int
        该节点执行次数（权重，用于 cost 计算）。
    cost_slope, cost_intercept : float
        每执行一次的成本： cost_slope * level + cost_intercept
        其中 level = 在此节点执行时剩余的 rescale levels（= 模数链剩余长度
        等效量）。非 ct 节点允许 cost_slope = 0。
    """
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

    # ------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Cut Point
# ---------------------------------------------------------------------------

@dataclass
class CutPoint:
    """
    拓扑序上的一个 cut point 位置。

    索引约定： c_0 (SOURCE) < c_1 (第一个乘法) < ... < c_M (最后一个乘法)
    另外有虚拟的 c_{M+1} (DUMMY_SINK)。

    注意 c_0 本身**不能 rescale**（它只是初始操作数的位置，携带
    一个 baseline scaling factor t_0）。第一次 rescale 只能发生在
    c_1（第一个乘法）之后。

    Attributes
    ----------
    index : int
        cut-point index ∈ [0, M+1]。
    node : ComputeNode
        关联的计算节点（index = M+1 时是 DUMMY_SINK）。
    amplitude_profile : AmplitudeProfile
    snr_requirement : SNRRequirement
    op_type : str
        用于查噪声表的 op 名，默认 "rescale"。
    amplitude_budget_bits : int
        A_j^{budget}  — 该 cut point 的中间幅度预算（bit）。
    baseline_scale_bits : int
        t_j^{base} — FindMinSF 的输出（不含 h_sf headroom）。
    target_scale_bits : int
        t_j = t_j^{base} + h_sf
    """
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


# ---------------------------------------------------------------------------
# Edges: stage edge + tail edge
# ---------------------------------------------------------------------------

@dataclass
class StageEdge:
    """
    可行的 stage 边 (i, j) — 两端都是普通 cut point (0 ≤ i < j ≤ M)。

    Attributes
    ----------
    start, end : int                        i, j
    nodes_in_stage : List[ComputeNode]      c_i 以后, c_j 以及中间所有节点
                                            （不含 c_i 自身；含 c_j 本身）
    pre_rescale_scale_bits : int            s_pre(i,j)
    drop_bits : int                         d(i,j) = s_pre(i,j) − t_j
    total_cost_slope : float                Σ count·cost_slope
    total_cost_intercept : float            Σ count·cost_intercept
    """
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
    """
    Tail 边 (i, M+1) — 从 c_i 一路计算到 DUMMY_SINK，**不再做 rescale**。

    Attributes
    ----------
    start : int                              i
    nodes_in_tail : List[ComputeNode]        c_i 以后、所有中间节点 + c_M
                                             （不含 c_i，不含 DUMMY_SINK，
                                             因 DUMMY_SINK 无 cost/贡献）
    gamma : int                              γ_tail(i) = max over v ∈ (i, M]
                                             of (s_hat(i,v) + A_v^{budget})
    total_cost_intercept : float             Σ count·cost_intercept
                                             (tail 上 level = 0, 斜率项 = 0)
    """
    start: int
    nodes_in_tail: List[ComputeNode] = field(default_factory=list)
    gamma: int = 0
    total_cost_intercept: float = 0.0

    def E(self) -> float:
        return self.total_cost_intercept


# ---------------------------------------------------------------------------
# Cost parameters
# ---------------------------------------------------------------------------

@dataclass
class CostParams:
    """
    统一边成本:
        ~C(i,j,l) = λ₀ + λ₁·l + α·E(i,j,l) + β·d(i,j)
                           (stage edge,  next level rule: l − 1)

        ~C(i,M+1,0) = λ₀ + α·E_tail(i,0)
                           (tail edge,   next level rule: 0)

    其中 l 是 *remaining rescale levels*（backward 视角）。
    """
    lambda_0: float = 1.0
    lambda_1: float = 0.0
    alpha: float = 1.0
    beta: float = 0.1


# ---------------------------------------------------------------------------
# PropagateScale
# ---------------------------------------------------------------------------

def propagate_scale(start_scale_bits: int,
                    path_nodes: Sequence[ComputeNode]) -> int:
    """
    沿路径累加 scale，按节点类型 dispatch:

    * **Non-mul / CTPT_MUL** : delta = ``node.scale_delta_bits``
                               (plaintext operand's scaling factor)
    * **CTCT_MUL**    :
        - if ``node.other_ct_scale_bits`` is not ``None``:
            delta = ``node.other_ct_scale_bits``   (asymmetric CTCT;
            the external ciphertext enters at this fixed scale),
            i.e. ``s → s + other_ct_scale_bits``.
        - else (default, symmetric squaring):
            ``s → 2·s`` (both operands at the working scale).
        ``node.scale_delta_bits`` is **ignored** for CTCT_MUL.

    Parameters
    ----------
    start_scale_bits : int
        路径起点的 scale bits（通常是 c_i 刚做完 rescale 后的 t_i）。
    path_nodes : sequence of ComputeNode
        路径上 c_i 之后的所有节点（含终点 c_j）。

    Returns
    -------
    int
        路径终点处的 scale bits（若终点是乘法节点，结果包含该乘法）。
    """
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


# ---------------------------------------------------------------------------
# RescaleGraph — 顶层容器
# ---------------------------------------------------------------------------

@dataclass
class RescaleGraph:
    """
    完整的 rescale 优化图。

    Fields
    ------
    nodes : List[ComputeNode]
        全部计算节点，topo_order 升序。
    cut_points : List[CutPoint]
        长度 = M + 2。  index 0..M = 真实 cut points；index M+1 = DUMMY_SINK。
    stage_edges : Dict[(i,j), StageEdge]
        所有 *可行* stage 边（d(i,j) ∈ Q_legal）。
    tail_edges : Dict[i, TailEdge]
        所有 *可行* tail 边（γ_tail(i) < q_max）。
    noise_table : NoiseLookupTable

    Global parameters
    -----------------
    h_sf : int           — uniform scale headroom（加到每个 t_j 上）
    q_legal_min : int    — Q_legal 的下界（如 30）
    q_legal_max : int    — Q_legal 的上界 (= q_max，如 60）
    """
    nodes: List[ComputeNode] = field(default_factory=list)
    cut_points: List[CutPoint] = field(default_factory=list)
    stage_edges: Dict[Tuple[int, int], StageEdge] = field(default_factory=dict)
    tail_edges: Dict[int, TailEdge] = field(default_factory=dict)

    #: ``stage_node_lists[k]`` 是 (c_k, c_{k+1}] 之间的节点列表（含 c_{k+1}
    #: 自身，不含 c_k）。  由 Feasibility-DAG 构建过程写入；供 Alg 8
    #: ValidateCutPoints 在任意两个 cut point 之间做 PropagateScale 使用。
    stage_node_lists: List[List[ComputeNode]] = field(default_factory=list)

    noise_table: NoiseLookupTable = field(default_factory=NoiseLookupTable)

    h_sf: int = 10
    q_legal_min: int = 30
    q_legal_max: int = 60

    # ---- sizes --------------------------------------------------------

    @property
    def M(self) -> int:
        """最后一个 *真实* cut point 的 index（= len(cut_points) - 2）。"""
        return len(self.cut_points) - 2

    @property
    def dummy_sink_index(self) -> int:
        return self.M + 1

    @property
    def num_real_cut_points(self) -> int:
        return self.M + 1     # c_0 ... c_M

    @property
    def q_max(self) -> int:
        return self.q_legal_max

    # ---- queries ------------------------------------------------------

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
        """c_i 在可行 DAG 上的所有后继（含 DUMMY_SINK 若有 tail 边）。"""
        succ: List[int] = sorted(j for (ii, j) in self.stage_edges if ii == i)
        if i in self.tail_edges:
            succ.append(self.dummy_sink_index)
        return succ

    def feasible_predecessors(self, j: int) -> List[int]:
        """c_j 在可行 DAG 上的所有前驱。  j 可取 M+1（DUMMY_SINK）。"""
        if j == self.dummy_sink_index:
            return sorted(self.tail_edges.keys())
        return sorted(ii for (ii, jj) in self.stage_edges if jj == j)

    def nodes_between(self, a: int, b: int) -> List[ComputeNode]:
        """
        返回 cut point c_a 与 c_b 之间的节点列表 (c_a, c_b]。

        Requires 0 ≤ a ≤ b ≤ M.  When a == b the list is empty.
        """
        if a == b:
            return []
        if a > b or a < 0 or b > self.M:
            raise ValueError(f"nodes_between: invalid range a={a}, b={b}, M={self.M}")
        out: List[ComputeNode] = []
        for k in range(a, b):
            out.extend(self.stage_node_lists[k])
        return out

    # ---- summary ------------------------------------------------------

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
