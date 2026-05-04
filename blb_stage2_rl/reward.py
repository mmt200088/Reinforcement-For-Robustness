"""BLB Stage 2 RL 奖励函数（spec §6 三层优先级）。

设计思想：
  优先级 1 - 精度约束：``acc < ACC_THRESHOLD`` → 直接重罚 + 距离 dense 引导
  优先级 2 - 稳定性约束：``loss_std > STAB_THRESHOLD`` → 中重罚 + dense 引导
  优先级 3 - cost 优化：``r_bits + r_fusion + r_k`` 加权求和

硬约束惩罚量级 >> cost reward 量级（差 1-2 个数量级），保证 RL 永远先把硬约束做满
再去抠 cost。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# 默认权重 / 阈值（spec §6.4 起步建议；可在 RL 训练前 sweep）
# ---------------------------------------------------------------------------
DEFAULT_S = 1.0                          # 单位奖励量级
DEFAULT_PRIORITY1_PENALTY = 100.0        # 精度违反基础罚
DEFAULT_PRIORITY1_SCALE = 200.0          # 精度差距 dense 引导
DEFAULT_PRIORITY2_PENALTY = 50.0         # 稳定性违反基础罚
DEFAULT_PRIORITY2_SCALE = 100.0          # 稳定性差距 dense 引导
DEFAULT_INVALID_PENALTY = 30.0           # invalid_chain 直接判死
DEFAULT_BASELINE_AVG_K = 13.0            # 全 max-k baseline 的 avg_k


@dataclass
class BaselineCostStats:
    """RL 训练前一次性算出的 baseline cost stats（全 max-action / 全 max-K）。

    Attributes:
        total_bits_sum:    所有 (block, layer) ``Rescale_optimizer`` total_bits 之和
        total_fusion_count: 所有 (block, layer) fusion_count 之和
        avg_k:             baseline 平均 truncation k（默认 13.0）
        loss_mean:         baseline 在 probe 上的 loss 均值
        loss_std:          baseline 在 probe 上的 loss 多次 trial 的 std
        metric1_mean:      第一指标均值 (e.g. accuracy)
        metric2_mean:      第二指标均值 (e.g. F1)
        metric1_std / metric2_std: 多次 trial 的 std（可空，未跑则 0）
    """
    total_bits_sum: int = 0
    total_fusion_count: int = 0
    avg_k: float = DEFAULT_BASELINE_AVG_K
    loss_mean: float = 0.0
    loss_std: float = 0.0
    metric1_mean: float = 0.0
    metric2_mean: float = 0.0
    metric1_std: float = 0.0
    metric2_std: float = 0.0
    # 反推权重时用到的"典型 drop 量"
    typical_bits_drop: float = 1.0
    typical_fusion_count: float = 1.0
    typical_k_drop: float = 1.0


@dataclass
class RewardWeights:
    """三类 cost reward 项的权重（按 spec §6.4 反推得到的工作权重）。

    ``cost_reward_mode`` 控制三项 cost 的语义：
      * ``"differential"``（默认，与 spec §11 sanity 一致）：
            r_bits   = w_bits * (baseline.total_bits_sum - opt.total_bits_sum)
            r_fusion = w_fusion * (baseline.total_fusion_count - opt.total_fusion_count)
            r_k      = w_k * (baseline.avg_k - action_avg_k)
        amax baseline 时 reward == 0；更小 SF / 更少 rescale → reward 增加。
      * ``"absolute"``（与 spec §6.2 公式严格一致，但 amax 时 reward < 0）：
            r_bits   = w_bits * (baseline.total_bits_sum - opt.total_bits_sum)
            r_fusion = -w_fusion * opt.total_fusion_count
            r_k      = w_k * (baseline.avg_k - action_avg_k)
    """
    w_bits: float = DEFAULT_S / 30.0    # 起步：r_bits ≈ S/30
    w_fusion: float = DEFAULT_S
    w_k: float = DEFAULT_S
    invalid_penalty: float = DEFAULT_INVALID_PENALTY
    priority1_penalty: float = DEFAULT_PRIORITY1_PENALTY
    priority1_scale: float = DEFAULT_PRIORITY1_SCALE
    priority2_penalty: float = DEFAULT_PRIORITY2_PENALTY
    priority2_scale: float = DEFAULT_PRIORITY2_SCALE
    cost_reward_mode: str = "differential"   # "differential" | "absolute"


def calibrate_weights_from_baseline(
        baseline: BaselineCostStats,
        s: float = DEFAULT_S,
        ) -> RewardWeights:
    """按 spec §6.4 反推权重：让 ``r_bits ≈ S/30``、``r_fusion / r_k ≈ S``。

    具体公式：
        w_fusion = S
        w_k      = S
        w_bits   = (S / 30) / max(1, typical_bits_drop)
    """
    w_fusion = float(s)
    w_k = float(s)
    bits_drop = float(max(1.0, baseline.typical_bits_drop))
    w_bits = (float(s) / 30.0) / bits_drop
    return RewardWeights(w_bits=w_bits, w_fusion=w_fusion, w_k=w_k)


@dataclass
class EpisodeMetrics:
    """单回合 K trials 评估得到的精度 / 稳定性结果。

    Attributes:
        loss_mean:        K trials loss 均值（越低越好）
        loss_std:         K trials loss std
        metric1_mean:     第一指标均值（越高越好）
        metric2_mean:     第二指标均值（越高越好）
        loss_max:         K trials loss 最大值（worst case）
        metric1_min:      K trials 第一指标最小值（worst case）
        metric2_min:      K trials 第二指标最小值（worst case）
    """
    loss_mean: float = 0.0
    loss_std: float = 0.0
    metric1_mean: float = 0.0
    metric2_mean: float = 0.0
    loss_max: float = 0.0
    metric1_min: float = 0.0
    metric2_min: float = 0.0


@dataclass
class RewardBreakdown:
    """``compute_reward`` 的明细返回值。"""
    reward: float
    priority: int                       # 1=acc, 2=stab, 3=cost
    invalid: bool                       # invalid_chain
    r_bits: float = 0.0
    r_fusion: float = 0.0
    r_k: float = 0.0
    bits_drop: float = 0.0
    k_drop: float = 0.0
    fusion_count: float = 0.0
    acc_violation: float = 0.0          # max(0, threshold - mean_metric)
    stab_violation: float = 0.0         # max(0, std - threshold)


# ---------------------------------------------------------------------------
# 单步 reward 计算
# ---------------------------------------------------------------------------
def _resolve_metric_for_threshold(
        metrics: EpisodeMetrics,
        prefer_metric: str = "accuracy",
        ) -> float:
    """选哪个标量代表"精度"。BLB stage2 上层固定用 metric1 即可（数据集相关）。"""
    return float(metrics.metric1_mean)


def compute_reward(
        metrics: EpisodeMetrics,
        opt_signals: Any,                 # rescale_optimizer_bridge.OptimizerRewardSignals
        action_avg_k: float,
        baseline: BaselineCostStats,
        *,
        weights: Optional[RewardWeights] = None,
        acc_threshold: float = 0.0,
        stab_threshold: float = float("inf"),
        any_invalid: Optional[bool] = None,
        ) -> RewardBreakdown:
    """三层优先级奖励（spec §6.1-§6.5）。

    Args:
        metrics:       本步 K trials 评估指标
        opt_signals:   ``rescale_optimizer_bridge.aggregate_optimizer_signals`` 输出
        action_avg_k:  本步动作的平均 truncation k
        baseline:      ``BaselineCostStats`` (全 max baseline)
        weights:       reward 权重；None ⇒ DEFAULT 起步值
        acc_threshold: 精度硬阈值（达不到 → 优先级 1 罚）
        stab_threshold:稳定性硬阈值（loss_std > 阈值 → 优先级 2 罚）
        any_invalid:   ``opt_signals.any_invalid`` 的显式覆盖；None=直接读 signals

    Returns:
        ``RewardBreakdown``
    """
    weights = weights or RewardWeights()

    # 优先级 1：精度
    acc = _resolve_metric_for_threshold(metrics)
    acc_violation = max(0.0, float(acc_threshold) - float(acc))
    if acc_violation > 0:
        # -PEN + (acc - threshold) * SCALE  （acc < threshold 时 (acc - threshold) < 0）
        r = -float(weights.priority1_penalty) + (float(acc) - float(acc_threshold)) * float(weights.priority1_scale)
        return RewardBreakdown(
            reward=float(r), priority=1, invalid=False, acc_violation=acc_violation,
        )

    # 优先级 2：稳定性
    loss_std = float(metrics.loss_std)
    stab_violation = max(0.0, loss_std - float(stab_threshold))
    if stab_violation > 0:
        r = -float(weights.priority2_penalty) + (float(stab_threshold) - loss_std) * float(weights.priority2_scale)
        return RewardBreakdown(
            reward=float(r), priority=2, invalid=False, stab_violation=stab_violation,
        )

    # 优先级 3：cost
    invalid = bool(any_invalid) if any_invalid is not None else bool(getattr(opt_signals, "any_invalid", False))
    if invalid:
        return RewardBreakdown(
            reward=-float(weights.invalid_penalty), priority=3, invalid=True,
        )

    bits_drop = float(baseline.total_bits_sum) - float(getattr(opt_signals, "total_bits_sum", 0))
    fusion_count = float(getattr(opt_signals, "total_fusion_count", 0))
    fusion_drop = float(baseline.total_fusion_count) - fusion_count
    k_drop = float(baseline.avg_k) - float(action_avg_k)

    r_bits = float(weights.w_bits) * bits_drop
    r_k = float(weights.w_k) * k_drop
    if str(getattr(weights, "cost_reward_mode", "differential")).lower() == "absolute":
        r_fusion = float(weights.w_fusion) * (-fusion_count)
    else:
        r_fusion = float(weights.w_fusion) * fusion_drop

    total = r_bits + r_fusion + r_k
    return RewardBreakdown(
        reward=float(total),
        priority=3,
        invalid=False,
        r_bits=r_bits, r_fusion=r_fusion, r_k=r_k,
        bits_drop=bits_drop, k_drop=k_drop, fusion_count=fusion_count,
    )
