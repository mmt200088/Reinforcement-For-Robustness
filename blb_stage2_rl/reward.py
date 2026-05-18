"""BLB Stage 2 RL 奖励函数（v2-style clipped shaping + tier bonus）。

2026-05-18 重写：从 ADR-002 的 hard-priority (-50/-100/-200 大惩罚) 切换到
v2 (``noise_rl_module_v2``) 的设计——shaping 项夹紧到 [-5, +5]，优先级靠
tier_bonus +20/+40 的大跳变体现。详见 ``docs/adr/ADR-007-v2-style-clipped-tier-reward.md``。

核心公式：

  margin_acc       = (acc - acc_threshold) / max(|baseline_acc - acc_threshold|, 0.01)
  cost_score       = (bits_score + fusion_score + k_score) / 3   # 每项 dimensionless
  stab_excess      = max(0, loss_std - stab_threshold)
  stability_penalty = -lambda_stab * stab_excess

  metric_ok        = (acc_violation == 0) AND not invalid
  stab_ok          = (stab_excess == 0)

  shaping          = margin_acc - invalid_penalty*invalid
                   + (cost_score + stability_penalty) IF metric_ok ELSE 0
  shaping_clipped  = clip(shaping, -5, +5)

  tier_bonus       = 0
  if metric_ok:    tier_bonus += 20
    if stab_ok:    tier_bonus += 20

  total_reward     = shaping_clipped + tier_bonus

reward range:
  · 全部 fail (invalid 或 acc < threshold)   → [-5, 0]
  · metric OK + stab fail                       → [+15, +25]
  · metric OK + stab OK                          → [+35, +45]

3 个 tier 之间至少差 ~15 reward，PPO 看到的 advantage 信号清晰；同时单 episode 的
reward 永远 bounded，cost 优化在 metric/stab 都通过后才以 ≤±0.5 的小幅度差分驱动
"已经合格" 的候选互相挤名次——这就是 v2 在 stage1 上工作良好的关键。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# v2-style reward constants（与 noise_rl_module_v2.py 等价）
# ---------------------------------------------------------------------------
DEFAULT_REWARD_CLIP_MIN = -5.0
DEFAULT_REWARD_CLIP_MAX = 5.0
DEFAULT_TIER_METRIC_BONUS = 20.0
DEFAULT_TIER_STABILITY_BONUS = 20.0
DEFAULT_LAMBDA_STAB = 5.0
DEFAULT_INVALID_PENALTY = 5.0          # 单次 invalid_chain 罚 -5（足以打满 clip）
DEFAULT_MARGIN_DENOM_FLOOR = 0.01      # 防 baseline_acc ≈ acc_threshold 时分母 0
DEFAULT_BASELINE_AVG_K = 13.0          # 全 max-k baseline 的 avg_k

# Legacy fields kept for any caller / log line that still references them.
DEFAULT_S = 1.0
DEFAULT_PRIORITY1_PENALTY = 100.0
DEFAULT_PRIORITY1_SCALE = 200.0
DEFAULT_PRIORITY2_PENALTY = 50.0
DEFAULT_PRIORITY2_SCALE = 100.0


@dataclass
class BaselineCostStats:
    """RL 训练前一次性算出的 baseline cost stats（全 max-action / 全 max-K）。

    v2-style 用到的额外字段（typical_*_drop）保留旧名字，由
    ``estimate_baseline_cost_stats`` 在 setup 阶段从 random samples 估出，
    作为 cost_score normalizer。
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
    # cost normalizers (set in baseline calibration; 1.0 fallback keeps the
    # ratio finite even if estimation produced no valid samples)
    typical_bits_drop: float = 1.0
    typical_fusion_count: float = 1.0
    typical_k_drop: float = 1.0


@dataclass
class RewardWeights:
    """v2-style reward 的可调权重。

    Args:
        cost_weight:       cost_score 整体乘子（默认 1.0）
        lambda_stab:       stab_excess → penalty 的乘子（默认 5.0；
                           v2 的 NOISE_STAGE_STABILITY_LAMBDA 同名）
        invalid_penalty:   any_invalid 时一次性罚（默认 5.0，足够打满 clip）
        reward_clip_min:   shaping 下限（默认 -5.0）
        reward_clip_max:   shaping 上限（默认 +5.0）
        tier_metric_bonus: metric_ok 时加（默认 +20）
        tier_stability_bonus: 进一步 stab_ok 时再加（默认 +20）
        margin_denom_floor: margin 计算的分母下限
        baseline_metric1:  baseline accuracy 的 margin denominator 来源
                           （runner 把 baseline.metric1_mean 灌进来即可）
        # legacy fields kept for backward compatibility with older callers /
        # diagnostics that read them — currently unused by compute_reward.
        w_bits / w_fusion / w_k / priority1_* / priority2_* / cost_reward_mode
    """
    cost_weight: float = 1.0
    lambda_stab: float = DEFAULT_LAMBDA_STAB
    invalid_penalty: float = DEFAULT_INVALID_PENALTY
    reward_clip_min: float = DEFAULT_REWARD_CLIP_MIN
    reward_clip_max: float = DEFAULT_REWARD_CLIP_MAX
    tier_metric_bonus: float = DEFAULT_TIER_METRIC_BONUS
    tier_stability_bonus: float = DEFAULT_TIER_STABILITY_BONUS
    margin_denom_floor: float = DEFAULT_MARGIN_DENOM_FLOOR
    baseline_metric1: float = 0.0
    # legacy fields (kept for backward-compat, never read by compute_reward):
    w_bits: float = DEFAULT_S / 30.0
    w_fusion: float = DEFAULT_S
    w_k: float = DEFAULT_S
    priority1_penalty: float = DEFAULT_PRIORITY1_PENALTY
    priority1_scale: float = DEFAULT_PRIORITY1_SCALE
    priority2_penalty: float = DEFAULT_PRIORITY2_PENALTY
    priority2_scale: float = DEFAULT_PRIORITY2_SCALE
    cost_reward_mode: str = "differential"


def calibrate_weights_from_baseline(
        baseline: BaselineCostStats,
        s: float = DEFAULT_S,
        ) -> RewardWeights:
    """v2-style 反推：保留旧 API 名字但写入 baseline_metric1。

    其他权重（cost_weight / lambda_stab 等）走 RewardWeights 默认值——这些
    是 v2 验证过的 sweet spot，不需要再校准。``s`` 仅作向后兼容，不影响
    新公式。
    """
    return RewardWeights(
        baseline_metric1=float(getattr(baseline, "metric1_mean", 0.0) or 0.0),
    )


@dataclass
class EpisodeMetrics:
    """单回合 K trials 评估得到的精度 / 稳定性结果。"""
    loss_mean: float = 0.0
    loss_std: float = 0.0
    metric1_mean: float = 0.0
    metric2_mean: float = 0.0
    loss_max: float = 0.0
    metric1_min: float = 0.0
    metric2_min: float = 0.0


@dataclass
class RewardBreakdown:
    """``compute_reward`` 的明细返回值（v2-style 字段 + 旧字段兼容）。"""
    reward: float
    priority: int                       # 1=acc/invalid, 2=stab, 3=cost (报告用)
    invalid: bool                       # opt_signals.any_invalid 或 acc_violation
    # v2-style 拆解
    shaping_raw: float = 0.0
    shaping_clipped: float = 0.0
    tier_bonus: float = 0.0
    margin_acc: float = 0.0
    cost_score: float = 0.0
    stability_penalty: float = 0.0
    invalid_term: float = 0.0
    metric_ok: bool = False
    stab_ok: bool = False
    # 兼容字段（runner / 诊断 / persistence 仍在读）
    r_bits: float = 0.0
    r_fusion: float = 0.0
    r_k: float = 0.0
    r_invalid: float = 0.0
    bits_drop: float = 0.0
    k_drop: float = 0.0
    fusion_count: float = 0.0
    acc_violation: float = 0.0
    stab_violation: float = 0.0
    optimizer_cost_terms: Any = field(default_factory=lambda: ["total_bits_sum", "fusion_count"])
    optimizer_validity_terms: Any = field(default_factory=lambda: ["invalid_chain", "optimizer_valid", "any_invalid"])
    optimizer_diagnostic_terms: Any = field(default_factory=lambda: ["q_bits", "q_head_bits", "q_tail_bits"])
    mpc_truncation_cost_enabled: bool = True
    mpc_truncation_term: str = "avg_k"


def _resolve_metric_for_threshold(
        metrics: EpisodeMetrics,
        prefer_metric: str = "accuracy",
        ) -> float:
    return float(metrics.metric1_mean)


def compute_reward(
        metrics: EpisodeMetrics,
        opt_signals: Any,
        action_avg_k: float,
        baseline: BaselineCostStats,
        *,
        weights: Optional[RewardWeights] = None,
        acc_threshold: float = 0.0,
        stab_threshold: float = float("inf"),
        any_invalid: Optional[bool] = None,
        ) -> RewardBreakdown:
    """v2-style clipped-shaping + tier-bonus reward.

    Args:
        metrics:       本步 K trials 评估指标
        opt_signals:   ``rescale_optimizer_bridge.aggregate_optimizer_signals`` 输出
        action_avg_k:  本步动作的平均 truncation k
        baseline:      ``BaselineCostStats`` (全 max baseline)
        weights:       RewardWeights；None ⇒ v2 默认值
        acc_threshold: 精度硬阈值（acc < threshold ⇒ metric_ok = False）
        stab_threshold: 稳定性 soft cap（loss_std > threshold ⇒ continuous penalty）
        any_invalid:   优化器 invalid_chain 显式覆盖；None=直接读 signals

    Returns:
        ``RewardBreakdown``
    """
    weights = weights or RewardWeights()

    invalid = bool(any_invalid) if any_invalid is not None else bool(
        getattr(opt_signals, "any_invalid", False)
    )

    acc = _resolve_metric_for_threshold(metrics)
    if not math.isfinite(float(acc)):
        acc = 0.0
    loss_std = float(metrics.loss_std)

    # === 1. margin_acc (dimensionless, signed) ===
    baseline_acc = float(weights.baseline_metric1 or 0.0)
    denom_acc = max(
        abs(baseline_acc - float(acc_threshold)),
        float(weights.margin_denom_floor),
    )
    margin_acc = (float(acc) - float(acc_threshold)) / denom_acc
    acc_violation = max(0.0, float(acc_threshold) - float(acc))

    # === 2. cost_score (dimensionless, signed; 0 = baseline) ===
    bits_drop = float(baseline.total_bits_sum) - float(
        getattr(opt_signals, "total_bits_sum", 0)
    )
    fusion_count = float(getattr(opt_signals, "total_fusion_count", 0))
    fusion_drop = float(baseline.total_fusion_count) - fusion_count
    k_drop = float(baseline.avg_k) - float(action_avg_k)

    typical_bits = max(abs(float(baseline.typical_bits_drop)), 1.0)
    typical_fusion = max(abs(float(baseline.typical_fusion_count)), 1.0)
    typical_k = max(abs(float(baseline.typical_k_drop)), 1.0)
    bits_score = bits_drop / typical_bits
    fusion_score = fusion_drop / typical_fusion
    k_score = k_drop / typical_k
    cost_score_raw = float(weights.cost_weight) * (
        bits_score + fusion_score + k_score
    ) / 3.0

    # === 3. stability_penalty (soft, continuous, ≤ 0) ===
    if math.isfinite(loss_std):
        stab_excess = max(0.0, float(loss_std) - float(stab_threshold))
    else:
        stab_excess = 1.0   # non-finite loss_std treated as severe
    stability_penalty = -float(weights.lambda_stab) * stab_excess

    # === 4. eligibility gates ===
    metric_ok = (acc_violation == 0.0) and not invalid
    stab_ok = (stab_excess == 0.0)

    # cost & stability only contribute when metric_ok — v2 line 1647 同款语义
    # "未满足 metric 时不启用 stability 信号（避免 PPO 在不可行区间被 std 分散注意力）"
    effective_cost_score = cost_score_raw if metric_ok else 0.0
    effective_stab_penalty = stability_penalty if metric_ok else 0.0
    invalid_term = -float(weights.invalid_penalty) if invalid else 0.0

    # === 5. shaping (clipped to [-5, +5]) ===
    shaping_raw = (
        float(margin_acc)
        + invalid_term
        + effective_cost_score
        + effective_stab_penalty
    )
    shaping_clipped = float(
        np.clip(shaping_raw, float(weights.reward_clip_min), float(weights.reward_clip_max))
    )

    # === 6. tier_bonus (hard-priority via large jumps) ===
    tier_bonus = 0.0
    if metric_ok:
        tier_bonus += float(weights.tier_metric_bonus)
        if stab_ok:
            tier_bonus += float(weights.tier_stability_bonus)

    total = float(shaping_clipped + tier_bonus)

    # === 7. legacy priority label (for reporting) ===
    # Mirrors ADR-002 semantics: priority is purely a function of
    # (acc_violation, stab_excess). invalid is orthogonal — it appears in
    # ``RewardBreakdown.invalid`` and contributes ``invalid_term`` to
    # shaping, but it doesn't bump the priority label. This keeps the
    # priority field meaningful as a "which gate did this action trip"
    # diagnostic; the reward magnitude already encodes the invalid cost
    # via the clipped shaping.
    if acc_violation > 0:
        priority = 1
    elif stab_excess > 0:
        priority = 2
    else:
        priority = 3

    return RewardBreakdown(
        reward=float(total),
        priority=int(priority),
        invalid=invalid,
        shaping_raw=float(shaping_raw),
        shaping_clipped=float(shaping_clipped),
        tier_bonus=float(tier_bonus),
        margin_acc=float(margin_acc),
        cost_score=float(effective_cost_score),
        stability_penalty=float(effective_stab_penalty),
        invalid_term=float(invalid_term),
        metric_ok=bool(metric_ok),
        stab_ok=bool(stab_ok),
        # legacy fields populated for backward-compatible diagnostics
        r_bits=float(bits_score / 3.0 * weights.cost_weight) if metric_ok else 0.0,
        r_fusion=float(fusion_score / 3.0 * weights.cost_weight) if metric_ok else 0.0,
        r_k=float(k_score / 3.0 * weights.cost_weight) if metric_ok else 0.0,
        r_invalid=float(invalid_term),
        bits_drop=float(bits_drop),
        k_drop=float(k_drop),
        fusion_count=float(fusion_count),
        acc_violation=float(acc_violation),
        stab_violation=float(stab_excess),
    )
