"""Per-block weighted fusion and truncation saving for the P3 reward.

The fixed block1:block2:block4:block5:truncation weights are
``80:150:130:40:50``. ``total_bits`` is a ranking diagnostic, not part of the
reward scalar.

The module is **pure / torch-free** (dataclasses + arithmetic plus the shared
truncation-domain constants). It imports only the torch-free ``truncation_levels``
sibling;
callers pass pre-extracted per-block choices.

Semantics:

  per block b of block-type t(b), with chosen (fusion_option, K):
    fusion_saving_b = fusion_count / max_fusion(t)   # 0 when max_fusion == 0 (block1/block4)
    trunc_saving_b  = (K_MAX - K) / (K_MAX - K_MIN)   # smaller K = bigger saving
    actual = Σ FUSION_W[t(b)] * fusion_saving_b + Σ TRUNC_W * trunc_saving_b
    cost_norm = actual / MAX_ACTUAL                    # in [0, 1]
    cost_rank = actual                                 # unbounded, candidate ranking only

``MAX_ACTUAL`` only counts the per-block weights that can actually move: a block
whose ``max_fusion == 0`` (block1/block4 in the current mrpc maps) contributes its
truncation weight but **not** its fusion weight, so its inert 80/130 fusion weight
does not dilute the normalization.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Mapping, Sequence

from .truncation_levels import K_MAX_BITS, K_MIN_BITS


@dataclass(frozen=True)
class BlockChoice:
    """One per-block decision realized during a fusion-count episode.

    ``block_idx`` is the block-type (1/2/4/5 after stripping the ``_L<i>`` suffix);
    ``max_fusion`` is the largest ``fusion_count`` reachable for that block-type's map
    (0 => the block has no fusion lever, e.g. block1/block4 in the mrpc maps).
    """
    block_idx: int
    graph_key: str
    fusion_count: int
    max_fusion: int
    k_value: int


@dataclass
class FusionCostResult:
    """Output of :func:`compute_fusion_cost_saving`.

    ``fusion_norm`` and ``trunc_norm`` use independent maxima so the larger
    truncation range cannot suppress the fusion signal. ``cost_norm`` is the
    shared diagnostic normalization; ``cost_rank`` remains unbounded.
    """
    cost_norm: float
    cost_rank: float
    max_actual: float
    fusion_norm: float = 0.0


    fusion_norm_saturated: float = 0.0
    trunc_norm: float = 0.0
    fusion_actual: float = 0.0
    trunc_actual: float = 0.0
    fusion_max_actual: float = 0.0
    trunc_max_actual: float = 0.0
    per_block: List[Dict[str, Any]] = field(default_factory=list)


def saturate_fusion(x: float, tau: float) -> float:
    """Concave saturating transform on ``x in [0, 1]``.

    ``sat(0)=0``, ``sat(1)=1``, with a steep initial slope that flattens past a
    knee controlled by ``tau`` (smaller ``tau`` saturates earlier). Used to turn
    the LINEAR weighted fusion saving into one with diminishing returns: each
    additional fused block adds less reward, so the deterministic monotone fusion
    incentive no longer pushes the policy past a healthy fusion level into the
    noisy accuracy boundary (the 4th-60k hot collapse, fusion 8→35).

    ``tau <= 0`` selects the identity transform.
    """
    x = min(1.0, max(0.0, float(x)))
    t = float(tau)
    if t <= 0.0:
        return x
    denom = 1.0 - math.exp(-1.0 / t)
    if denom <= 0.0:
        return x
    return (1.0 - math.exp(-x / t)) / denom


def _fusion_saving(fusion_count: int, max_fusion: int) -> float:
    if int(max_fusion) <= 0:
        return 0.0
    return min(1.0, max(0.0, float(fusion_count) / float(max_fusion)))


def _trunc_saving(k_value: int, k_max: int, k_min: int) -> float:
    if int(k_max) <= int(k_min):
        return 0.0
    s = (float(k_max) - float(k_value)) / (float(k_max) - float(k_min))
    return min(1.0, max(0.0, s))


def max_actual_for_choices(
        choices: Sequence[BlockChoice],
        *,
        fusion_w: Mapping[int, float],
        trunc_w: float,
        ) -> float:
    """Maximum achievable weighted saving for this schedule.

    A block contributes its fusion weight only if it actually has a fusion lever
    (``max_fusion > 0``); every block contributes one truncation weight.
    """
    total = 0.0
    for c in choices:
        if int(c.max_fusion) > 0:
            total += float(fusion_w.get(int(c.block_idx), 0.0))
        total += float(trunc_w)
    return total


def compute_fusion_cost_saving(
        choices: Sequence[BlockChoice],
        *,
        fusion_w: Mapping[int, float],
        trunc_w: float,
        k_max: int = K_MAX_BITS,
        k_min: int = K_MIN_BITS,
        max_actual: float | None = None,
        fusion_saturation_tau: float = 0.0,
        ) -> FusionCostResult:
    """Per-block weighted fusion + truncation saving.

    Returns ``cost_norm in [0, 1]`` (the bounded P3 PPO cost factor, to be scaled by
    ``p3_cost_budget``) and the unbounded ``cost_rank`` (candidate/frontier ranking
    only — never the PPO scalar). ``max_actual`` may be precomputed and passed to keep
    normalization stable across episodes; if ``None`` it is derived from ``choices``.
    """
    actual = 0.0
    fusion_actual = 0.0
    trunc_actual = 0.0
    fusion_max = 0.0
    trunc_max = 0.0
    per_block: List[Dict[str, Any]] = []
    for c in choices:
        fs = _fusion_saving(c.fusion_count, c.max_fusion)
        ts = _trunc_saving(c.k_value, k_max, k_min)
        w_f = float(fusion_w.get(int(c.block_idx), 0.0))
        f_contrib = w_f * fs
        t_contrib = float(trunc_w) * ts
        actual += f_contrib + t_contrib
        fusion_actual += f_contrib
        trunc_actual += t_contrib
        if int(c.max_fusion) > 0:
            fusion_max += w_f
        trunc_max += float(trunc_w)
        per_block.append({
            "block_idx": int(c.block_idx),
            "graph_key": str(c.graph_key),
            "fusion_count": int(c.fusion_count),
            "max_fusion": int(c.max_fusion),
            "k_value": int(c.k_value),
            "fusion_saving": fs,
            "trunc_saving": ts,
            "fusion_contrib": f_contrib,
            "trunc_contrib": t_contrib,
        })

    denom = (
        float(max_actual)
        if max_actual is not None
        else fusion_max + trunc_max
    )
    cost_norm = min(1.0, max(0.0, actual / denom)) if denom > 0.0 else 0.0
    fusion_norm = (
        min(1.0, max(0.0, fusion_actual / fusion_max)) if fusion_max > 0.0 else 0.0
    )
    trunc_norm = (
        min(1.0, max(0.0, trunc_actual / trunc_max)) if trunc_max > 0.0 else 0.0
    )
    fusion_norm_saturated = saturate_fusion(fusion_norm, fusion_saturation_tau)
    return FusionCostResult(
        cost_norm=float(cost_norm),
        cost_rank=float(actual),
        max_actual=float(denom),
        fusion_norm=float(fusion_norm),
        fusion_norm_saturated=float(fusion_norm_saturated),
        trunc_norm=float(trunc_norm),
        fusion_actual=float(fusion_actual),
        trunc_actual=float(trunc_actual),
        fusion_max_actual=float(fusion_max),
        trunc_max_actual=float(trunc_max),
        per_block=per_block,
    )
