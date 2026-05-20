"""BLB Stage 2 RL 奖励函数（v3：m1+m2 双指标 + Pareto-only P3 cost）。

2026-05-20 v3 重写（保留 ADR-007 的 clipped-shaping + tier-bonus 框架）：

  * 指标 gate 同时看 m1、m2：以前只看 m1 → 现在两者都要过自己的阈值
    (``baseline.metric{i}_mean * (1 - acc_tolerance)``)。margin_acc 取
    两者归一化后的均值。
  * 稳定性 gate 看 m1_std、m2_std、loss_std 三个方差，按 30:30:1 加权
    （和指标重要性一致：m1=m2>>loss）。
  * 顺序 Stage-2 路径的 cost_score 改为 Pareto-only：
      - 只有 P3（metric_ok 且 stab_ok 且非 invalid）进入 cost archive。
      - objective vector 直接最大化 raw gains：
        ``fusion_gain = action_fusion - baseline_fusion``、
        ``k_gain = baseline_avg_k - action_avg_k``、
        ``bits_gain = baseline_bits - action_bits``。
      - PPO 仍需要 scalar；scalar 只来自 frontier/dominance/duplicate 事件，
        不用 ``typical_*`` 或人工加权标量决定 P3 cost 排名。
  * 优先级硬序：tier_bonus 0/+20/+40 锁住 metric_ok / stab_ok 三档，
    cost_score 总 ≤ 1.0 永远拉不动 tier 边界，所以 cost 不可能压过指标
    和稳定性 —— 即便所有 cost 维度同时打满。

核心公式：

  acc_thr_m1 = baseline.metric1_mean * (1 - acc_tolerance)   # e.g. 0.88 * 0.995
  acc_thr_m2 = baseline.metric2_mean * (1 - acc_tolerance)
  acc_violation = max(0, acc_thr_m1 - m1, acc_thr_m2 - m2)
  margin_acc = ((m1 - thr_m1)/denom_m1 + (m2 - thr_m2)/denom_m2) / 2

  stab_thr_X = baseline.X_std * (1 + stab_tolerance) + stab_floor   # X ∈ {m1,m2,loss}
  excess_X = max(0, X_std - stab_thr_X)
  norm_X = excess_X / max(baseline.X_std, stab_floor)
  combined_stab_excess = (30·norm_m1 + 30·norm_m2 + 1·norm_loss) / 61
  stability_penalty = -lambda_stab · combined_stab_excess

  P3 cost_vector = (fusion_gain, k_gain, bits_gain)
  pareto_event ∈ {frontier_expansion, frontier_member, dominated, duplicate}

  metric_ok = (acc_violation == 0) AND not invalid
  stab_ok = (combined_stab_excess == 0)

  shaping_raw = margin_acc + invalid_term + (pareto_cost_score IF P3 ELSE 0) + (stab_penalty IF metric_ok ELSE 0)
  shaping_clipped = clip(shaping_raw, -5, +5)
  tier_bonus = 20·metric_ok + 20·(metric_ok AND stab_ok)
  total = shaping_clipped + tier_bonus

reward range:
  · 全部 fail (invalid 或 acc < threshold)   → [-5, 0]
  · metric OK + stab fail                       → [+15, +25]
  · metric OK + stab OK                          → [+35, +45]

3 个 tier 之间至少差 ~15 reward，PPO 看到的 advantage 信号清晰；同时单 episode 的
reward 永远 bounded，cost 优化在 metric/stab 都通过后才以 ≤±1 的小幅度差分驱动
"已经合格" 的候选互相挤名次。
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
DEFAULT_LAMBDA_STAB = 1.0              # 2026-05-20 v3：从 5.0 降到 1.0；用户 spec
                                       # 中 "loss 可牺牲" + 三方 stab 已内部 30:30:1
                                       # 加权过，lambda_stab 是全局放大；保持 ≤ 1.0
                                       # 让 stab penalty 不要在 metric_ok 后压过 cost_score。
DEFAULT_INVALID_PENALTY = 5.0          # 单次 invalid_chain 罚 -5（足以打满 clip）
DEFAULT_MARGIN_DENOM_FLOOR = 0.01      # 防 baseline_acc ≈ acc_threshold 时分母 0
DEFAULT_BASELINE_AVG_K = 13.0          # 全 max-k baseline 的 avg_k

# Per-axis importance weights inside cost_score and stability_penalty:
# user spec — fusion = truncation(K) ≈ 30 × total_bits in CKKS overhead semantics,
# and mirrored for m1_std = m2_std ≈ 30 × loss_std in the stability path.
DEFAULT_COST_W_FUSION = 30.0
DEFAULT_COST_W_K = 30.0
DEFAULT_COST_W_BITS = 1.0
DEFAULT_STAB_W_M1 = 30.0
DEFAULT_STAB_W_M2 = 30.0
DEFAULT_STAB_W_LOSS = 1.0

# Tolerances driving the per-metric thresholds. ``acc_tolerance`` is the
# relative drop you allow from baseline.metric{i}_mean (0.5% by default —
# matches the 2026-05-18 noisy-baseline preflight constant in sequential_runner).
# ``stab_tolerance`` is the relative slack you allow above baseline.X_std before
# declaring an excess. ``stab_floor`` is deliberately 1e-2, not 1e-3: MRPC
# metric1/metric2 are estimated from only 5 stochastic probe trials over a small
# validation subset, so their standard deviations are quantized by a few samples.
# A 1e-3 floor made normal sampling jitter randomly remove the +20 stability
# tier and collapse rolling reward averages without any accuracy/loss failure.
DEFAULT_ACC_TOLERANCE = 0.005
DEFAULT_STAB_TOLERANCE = 0.5
DEFAULT_STAB_FLOOR = 1.0e-2

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
    """v3 reward 的可调权重。

    Args:
        cost_weight:       cost_score 整体乘子（默认 1.0）
        lambda_stab:       combined_stab_excess → penalty 的乘子（默认 1.0；
                           v2 的 5.0 在 m1_std/m2_std 加进来后会过强）
        invalid_penalty:   any_invalid 时一次性罚（默认 5.0，足够打满 clip）
        reward_clip_min:   shaping 下限（默认 -5.0）
        reward_clip_max:   shaping 上限（默认 +5.0）
        tier_metric_bonus: metric_ok 时加（默认 +20）
        tier_stability_bonus: 进一步 stab_ok 时再加（默认 +20）
        margin_denom_floor: margin 计算的分母下限
        baseline_metric1:  baseline metric1 的 margin denominator 来源
                           （runner 把 baseline.metric1_mean 灌进来即可）
        baseline_metric2:  baseline metric2 的 margin denominator 来源
        acc_tolerance:     metric_ok 的容忍百分比；阈值 = baseline_X * (1 - tol)
                           caller 显式传 acc_threshold_m{1,2} 时覆盖该 fallback
        stab_tolerance:    stability gate 的容忍百分比；阈值 = baseline.X_std * (1 + tol)
        stab_floor:        stability 阈值的最小绝对值（防 baseline.std≈0 失稳）
        cost_w_fusion / cost_w_k / cost_w_bits:
                           cost_score 内部 fusion / k / bits 三项的权重
                           （默认 30:30:1，用户 spec）
        stab_w_m1 / stab_w_m2 / stab_w_loss:
                           combined_stab_excess 内部三方差的权重（30:30:1）
        # legacy fields kept for backward-compatibility:
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
    baseline_metric2: float = 0.0
    acc_tolerance: float = DEFAULT_ACC_TOLERANCE
    stab_tolerance: float = DEFAULT_STAB_TOLERANCE
    stab_floor: float = DEFAULT_STAB_FLOOR
    cost_w_fusion: float = DEFAULT_COST_W_FUSION
    cost_w_k: float = DEFAULT_COST_W_K
    cost_w_bits: float = DEFAULT_COST_W_BITS
    stab_w_m1: float = DEFAULT_STAB_W_M1
    stab_w_m2: float = DEFAULT_STAB_W_M2
    stab_w_loss: float = DEFAULT_STAB_W_LOSS
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
    """v3 反推：写入 baseline_metric1 / baseline_metric2。

    其他权重（cost_weight / lambda_stab 等）走 RewardWeights 默认值——这些
    是用户在 2026-05-20 spec 里直接拍的 sweet spot，不需要再校准。``s`` 仅作
    向后兼容，不影响新公式。
    """
    return RewardWeights(
        baseline_metric1=float(getattr(baseline, "metric1_mean", 0.0) or 0.0),
        baseline_metric2=float(getattr(baseline, "metric2_mean", 0.0) or 0.0),
    )


@dataclass
class EpisodeMetrics:
    """单回合 K trials 评估得到的精度 / 稳定性结果。

    2026-05-20 v3：新增 ``metric1_std`` / ``metric2_std`` —— v3 stability gate
    需要 m1、m2 的 trial 间方差，配合 loss_std 共同进入 combined_stab_excess。
    旧 caller 不填这两个字段时 default=0.0 等价于"trial 间无差异"（不触发 stab
    excess），保持向后兼容。
    """
    loss_mean: float = 0.0
    loss_std: float = 0.0
    metric1_mean: float = 0.0
    metric2_mean: float = 0.0
    metric1_std: float = 0.0
    metric2_std: float = 0.0
    loss_max: float = 0.0
    metric1_min: float = 0.0
    metric2_min: float = 0.0


@dataclass
class RewardBreakdown:
    """``compute_reward`` 的明细返回值（v3 字段 + 旧字段兼容）。"""
    reward: float
    priority: int                       # 1=acc/invalid, 2=stab, 3=cost (报告用)
    invalid: bool                       # opt_signals.any_invalid 或 acc_violation
    # v3 拆解
    shaping_raw: float = 0.0
    shaping_clipped: float = 0.0
    tier_bonus: float = 0.0
    margin_acc: float = 0.0
    cost_score: float = 0.0
    stability_penalty: float = 0.0
    invalid_term: float = 0.0
    metric_ok: bool = False
    stab_ok: bool = False
    # v3 per-axis breakdown (helpful for diagnostics / artifacts)
    acc_violation_m1: float = 0.0
    acc_violation_m2: float = 0.0
    margin_m1: float = 0.0
    margin_m2: float = 0.0
    bits_norm: float = 0.0
    fusion_norm: float = 0.0
    k_norm: float = 0.0
    stab_excess_m1: float = 0.0
    stab_excess_m2: float = 0.0
    stab_excess_loss: float = 0.0
    fusion_gain: float = 0.0
    pareto_event_kind: str = ""
    pareto_action_hash: str = ""
    pareto_frontier_removed: int = 0
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


@dataclass(frozen=True)
class ParetoCostEntry:
    """One raw-gain point on the P3 cost Pareto frontier."""
    action_hash: str
    fusion_gain: float
    k_gain: float
    bits_gain: float

    @property
    def gains(self) -> tuple:
        return (self.fusion_gain, self.k_gain, self.bits_gain)


@dataclass(frozen=True)
class ParetoCostEvent:
    """Bounded scalar shaping emitted by :class:`ParetoCostArchive.add`."""
    kind: str
    shaping: float
    action_hash: str
    entry: Optional[ParetoCostEntry] = None
    removed: int = 0


class ParetoCostArchive:
    """P3-only Pareto archive over raw cost gains.

    Ranking maximizes ``fusion_gain``, ``k_gain`` and ``bits_gain`` directly.
    ``BaselineCostStats.typical_*`` normalizers may be supplied by callers for
    surrounding reward code, but this archive deliberately does not read them.
    """

    def __init__(
            self,
            *,
            baseline: Optional[BaselineCostStats] = None,
            max_abs_shaping: float = 0.25,
            frontier_member_shaping: float = 0.025,
            duplicate_shaping: float = -0.005,
            dominated_shaping: float = -0.05,
            expansion_base_shaping: float = 0.10,
            expansion_removed_bonus: float = 0.05,
            ) -> None:
        self.baseline = baseline
        self.max_abs_shaping = abs(float(max_abs_shaping))
        self.frontier_member_shaping = float(frontier_member_shaping)
        self.duplicate_shaping = float(duplicate_shaping)
        self.dominated_shaping = float(dominated_shaping)
        self.expansion_base_shaping = float(expansion_base_shaping)
        self.expansion_removed_bonus = float(expansion_removed_bonus)
        self._frontier: list[ParetoCostEntry] = []
        self._seen_hashes: set[str] = set()

    @property
    def frontier(self) -> tuple:
        return tuple(self._frontier)

    def add(self, action_hash: str, breakdown: RewardBreakdown) -> ParetoCostEvent:
        action_hash = str(action_hash)
        if not self._is_p3_candidate(breakdown):
            return ParetoCostEvent(
                kind="excluded",
                shaping=0.0,
                action_hash=action_hash,
            )
        if action_hash in self._seen_hashes:
            return ParetoCostEvent(
                kind="duplicate",
                shaping=self._bounded(self.duplicate_shaping),
                action_hash=action_hash,
            )

        entry = ParetoCostEntry(
            action_hash=action_hash,
            fusion_gain=_safe_float(getattr(breakdown, "fusion_gain", 0.0), 0.0),
            k_gain=_safe_float(getattr(breakdown, "k_drop", 0.0), 0.0),
            bits_gain=_safe_float(getattr(breakdown, "bits_drop", 0.0), 0.0),
        )
        self._seen_hashes.add(action_hash)

        if any(self._dominates(existing, entry) for existing in self._frontier):
            return ParetoCostEvent(
                kind="dominated",
                shaping=self._bounded(self.dominated_shaping),
                action_hash=action_hash,
                entry=entry,
            )

        kept = []
        removed = 0
        for existing in self._frontier:
            if self._dominates(entry, existing):
                removed += 1
            else:
                kept.append(existing)
        kept.append(entry)
        self._frontier = kept

        if removed > 0 or len(self._frontier) == 1:
            shaping = self.expansion_base_shaping + self.expansion_removed_bonus * float(removed)
            kind = "frontier_expansion"
        else:
            shaping = self.frontier_member_shaping
            kind = "frontier_member"
        return ParetoCostEvent(
            kind=kind,
            shaping=self._bounded(shaping),
            action_hash=action_hash,
            entry=entry,
            removed=removed,
        )

    def _bounded(self, value: float) -> float:
        if self.max_abs_shaping <= 0.0:
            return 0.0
        return float(np.clip(float(value), -self.max_abs_shaping, self.max_abs_shaping))

    @staticmethod
    def _is_p3_candidate(breakdown: RewardBreakdown) -> bool:
        return (
            int(getattr(breakdown, "priority", 0)) == 3
            and not bool(getattr(breakdown, "invalid", False))
            and bool(getattr(breakdown, "metric_ok", False))
            and bool(getattr(breakdown, "stab_ok", False))
        )

    @staticmethod
    def _dominates(left: ParetoCostEntry, right: ParetoCostEntry) -> bool:
        left_gains = left.gains
        right_gains = right.gains
        return (
            all(l >= r for l, r in zip(left_gains, right_gains))
            and any(l > r for l, r in zip(left_gains, right_gains))
        )


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Coerce to a finite float; non-finite / non-numeric → ``default``."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(v):
        return float(default)
    return v


def _resolve_metric_for_threshold(
        metrics: EpisodeMetrics,
        prefer_metric: str = "accuracy",
        ) -> float:
    """Legacy single-metric resolver kept for callers that haven't moved to v3."""
    return float(metrics.metric1_mean)


def _resolve_acc_threshold(
        explicit: Optional[float],
        baseline_value: float,
        weights: RewardWeights,
        ) -> float:
    """Resolve the m1 or m2 acc threshold.

    Priority: ``explicit`` (caller-supplied via ``acc_threshold`` /
    ``acc_threshold_m2``) → ``baseline_value * (1 - acc_tolerance)`` if the
    baseline value is meaningful (> margin_denom_floor) → 0.0 as the last
    resort (legacy callers that don't have baseline metric data).
    """
    if explicit is not None:
        return float(explicit)
    bv = _safe_float(baseline_value, 0.0)
    if abs(bv) > float(weights.margin_denom_floor):
        return bv * (1.0 - float(weights.acc_tolerance))
    return 0.0


def compute_reward(
        metrics: EpisodeMetrics,
        opt_signals: Any,
        action_avg_k: float,
        baseline: BaselineCostStats,
        *,
        weights: Optional[RewardWeights] = None,
        acc_threshold: Optional[float] = None,
        acc_threshold_m2: Optional[float] = None,
        stab_threshold: Optional[float] = None,
        any_invalid: Optional[bool] = None,
        pareto_archive: Optional[ParetoCostArchive] = None,
        action_hash: Optional[str] = None,
        ) -> RewardBreakdown:
    """v3 clipped-shaping + tier-bonus reward with m1+m2 gate and 30:30:1 cost weights.

    Args:
        metrics:           本步 K trials 评估指标（v3 期望 metric1_std / metric2_std
                           已填；旧 caller 不填 default=0 等价于"trial 间无差异"）
        opt_signals:       ``rescale_optimizer_bridge.aggregate_optimizer_signals`` 输出
        action_avg_k:      本步动作的平均 truncation k（小 = 更激进 = 通讯更少 = 好）
        baseline:          ``BaselineCostStats`` (全 max baseline)
        weights:           ``RewardWeights``；None ⇒ v3 默认值
        acc_threshold:     m1 硬阈值；None ⇒ 由 baseline.metric1_mean * (1-tol) 派生
        acc_threshold_m2:  m2 硬阈值；None ⇒ 由 baseline.metric2_mean * (1-tol) 派生
        stab_threshold:    loss_std 阈值；v3 中 m1_std/m2_std/loss_std 各自有阈值
                           （baseline.X_std*(1+stab_tol)+stab_floor），传入的
                           ``stab_threshold`` 仅 override loss 那一项以兼容老 caller
        any_invalid:       优化器 invalid_chain 显式覆盖；None=直接读 signals
        pareto_archive:    可选 P3-only cost archive。传入后，P3 cost scalar
                           只来自 Pareto event shaping，不读取 typical_* 排名。
        action_hash:       Pareto archive 的候选身份；与 pareto_archive 同时传入
                           才启用 Pareto-only cost shaping。

    Returns:
        ``RewardBreakdown``
    """
    weights = weights or RewardWeights()

    invalid = bool(any_invalid) if any_invalid is not None else bool(
        getattr(opt_signals, "any_invalid", False)
    )

    m1 = _safe_float(metrics.metric1_mean, 0.0)
    m2 = _safe_float(metrics.metric2_mean, 0.0)
    # Non-finite std values mean the probe produced inf/nan losses — treat as
    # max-severity instability rather than silently clamping to 0 (which would
    # let runaway candidates earn stab_ok and a full tier_bonus). The original
    # v2 reward formula did the same via ``stab_excess = 1.0`` when loss_std
    # was non-finite; we preserve that semantics for all three stab channels.
    loss_std_raw = float(metrics.loss_std)
    m1_std_raw = float(getattr(metrics, "metric1_std", 0.0))
    m2_std_raw = float(getattr(metrics, "metric2_std", 0.0))
    loss_std = loss_std_raw if math.isfinite(loss_std_raw) else float("inf")
    m1_std = m1_std_raw if math.isfinite(m1_std_raw) else float("inf")
    m2_std = m2_std_raw if math.isfinite(m2_std_raw) else float("inf")

    # === 1. Per-metric thresholds (m1, m2) ===
    # A metric is "active" only if its baseline reference is non-trivial.
    # When callers don't calibrate one of the baselines (e.g. legacy tests that
    # only set baseline.metric1_mean), the inactive channel is skipped from
    # both the gate and the margin so its zero baseline doesn't blow up the
    # margin denominator. This preserves v2-style "single-metric" semantics for
    # backward compatibility while v3-aware callers (sequential_runner /
    # runner.py) get both channels by setting baseline.metric{1,2}_mean.
    baseline_m1 = _safe_float(weights.baseline_metric1, 0.0)
    baseline_m2 = _safe_float(weights.baseline_metric2, 0.0)
    m1_active = abs(baseline_m1) > float(weights.margin_denom_floor)
    m2_active = abs(baseline_m2) > float(weights.margin_denom_floor)

    thr_m1 = _resolve_acc_threshold(acc_threshold, weights.baseline_metric1, weights)
    thr_m2 = _resolve_acc_threshold(acc_threshold_m2, weights.baseline_metric2, weights)

    if m1_active:
        denom_m1 = max(abs(baseline_m1 - thr_m1), float(weights.margin_denom_floor))
        margin_m1 = (m1 - thr_m1) / denom_m1
        acc_violation_m1 = max(0.0, thr_m1 - m1)
    else:
        margin_m1 = 0.0
        acc_violation_m1 = 0.0
    if m2_active:
        denom_m2 = max(abs(baseline_m2 - thr_m2), float(weights.margin_denom_floor))
        margin_m2 = (m2 - thr_m2) / denom_m2
        acc_violation_m2 = max(0.0, thr_m2 - m2)
    else:
        margin_m2 = 0.0
        acc_violation_m2 = 0.0

    active_metric_count = (1 if m1_active else 0) + (1 if m2_active else 0)
    if active_metric_count > 0:
        margin_acc = (margin_m1 + margin_m2) / float(active_metric_count)
    else:
        # Neither baseline calibrated; fall back to "no margin signal" so
        # legacy callers that just want the tier_bonus path still work.
        margin_acc = 0.0
    combined_acc_violation = max(acc_violation_m1, acc_violation_m2)

    # === 2. cost_score with corrected fusion sign and importance weights ===
    opt_total_bits = _safe_float(getattr(opt_signals, "total_bits_sum", 0), 0.0)
    fusion_count = _safe_float(getattr(opt_signals, "total_fusion_count", 0), 0.0)

    bits_gain = float(baseline.total_bits_sum) - opt_total_bits
    fusion_gain = fusion_count - float(baseline.total_fusion_count)  # v3 sign flip
    k_gain = float(baseline.avg_k) - float(action_avg_k)             # smaller K = better

    typical_bits = max(abs(float(baseline.typical_bits_drop)), 1.0)
    typical_fusion = max(abs(float(baseline.typical_fusion_count)), 1.0)
    typical_k = max(abs(float(baseline.typical_k_drop)), 1.0)

    bits_norm = bits_gain / typical_bits
    fusion_norm = fusion_gain / typical_fusion
    k_norm = k_gain / typical_k

    cost_total_w = float(weights.cost_w_fusion + weights.cost_w_k + weights.cost_w_bits)
    if cost_total_w <= 0.0:
        cost_total_w = 1.0
    cost_score_raw = float(weights.cost_weight) * (
        float(weights.cost_w_fusion) * fusion_norm
        + float(weights.cost_w_k) * k_norm
        + float(weights.cost_w_bits) * bits_norm
    ) / cost_total_w

    # === 3. combined stability_excess (m1_std, m2_std, loss_std) ===
    def _stab_threshold(baseline_std: float) -> float:
        return float(baseline_std) * (1.0 + float(weights.stab_tolerance)) + float(weights.stab_floor)

    stab_thr_m1 = _stab_threshold(baseline.metric1_std)
    stab_thr_m2 = _stab_threshold(baseline.metric2_std)
    if stab_threshold is not None:
        # honour legacy single-channel loss_std override when caller explicitly
        # passes one; otherwise derive from baseline.loss_std same as m1/m2.
        stab_thr_loss = float(stab_threshold)
    else:
        stab_thr_loss = _stab_threshold(baseline.loss_std)

    # Non-finite raw std → infinite excess → forced "max" normalized excess
    # so the priority drops to P2 even with very lenient baseline thresholds.
    def _channel_excess(observed: float, threshold: float, denom: float) -> tuple:
        if not math.isfinite(observed):
            return 1.0, 1.0  # excess=1.0 sentinel + max-norm contribution
        raw_excess = max(0.0, observed - threshold)
        return raw_excess, raw_excess / max(abs(denom), float(weights.stab_floor))

    denom_m1_std = float(baseline.metric1_std)
    denom_m2_std = float(baseline.metric2_std)
    denom_loss_std = float(baseline.loss_std)
    stab_excess_m1, stab_norm_m1 = _channel_excess(m1_std, stab_thr_m1, denom_m1_std)
    stab_excess_m2, stab_norm_m2 = _channel_excess(m2_std, stab_thr_m2, denom_m2_std)
    stab_excess_loss, stab_norm_loss = _channel_excess(loss_std, stab_thr_loss, denom_loss_std)

    stab_total_w = float(weights.stab_w_m1 + weights.stab_w_m2 + weights.stab_w_loss)
    if stab_total_w <= 0.0:
        stab_total_w = 1.0
    combined_stab_excess = (
        float(weights.stab_w_m1) * stab_norm_m1
        + float(weights.stab_w_m2) * stab_norm_m2
        + float(weights.stab_w_loss) * stab_norm_loss
    ) / stab_total_w
    stability_penalty = -float(weights.lambda_stab) * combined_stab_excess

    # === 4. eligibility gates ===
    metric_ok = (combined_acc_violation == 0.0) and not invalid
    stab_ok = (combined_stab_excess == 0.0)

    # Priority label follows the hard-ordering contract: invalid/accuracy
    # failure is P1, stability failure after accuracy passes is P2, and only
    # accuracy+stability-passing candidates enter P3 cost search.
    if invalid or combined_acc_violation > 0:
        priority = 1
    elif combined_stab_excess > 0:
        priority = 2
    else:
        priority = 3

    # === 4.5. Optional Pareto-only P3 cost shaping ===
    # Cost must never help P1/P2 candidates. When a Pareto archive is wired
    # (the sequential Stage-2 path), the three raw gains are only converted to
    # PPO's required scalar through frontier/dominance events. The weighted
    # typical_* scalar above is retained as a legacy fallback for code paths
    # that have not been switched to a stateful archive yet.
    pareto_event: Optional[ParetoCostEvent] = None
    use_pareto = pareto_archive is not None and action_hash is not None
    if use_pareto:
        pareto_candidate = RewardBreakdown(
            reward=0.0,
            priority=int(priority),
            invalid=bool(invalid),
            metric_ok=bool(metric_ok),
            stab_ok=bool(stab_ok),
            fusion_gain=float(fusion_gain),
            bits_drop=float(bits_gain),
            k_drop=float(k_gain),
        )
        pareto_event = pareto_archive.add(str(action_hash), pareto_candidate)
        cost_score_raw = float(pareto_event.shaping) if priority == 3 else 0.0

    # Cost & stability only contribute to shaping when metric_ok. Cost is even
    # stricter: it only contributes in P3 (metric_ok AND stab_ok), matching the
    # "accuracy/stability as hard constraints, cost only after both pass" rule.
    effective_cost_score = cost_score_raw if (metric_ok and stab_ok) else 0.0
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

    # === 6. tier_bonus (hard-priority via large jumps; +20/+40 dominates cost) ===
    tier_bonus = 0.0
    if metric_ok:
        tier_bonus += float(weights.tier_metric_bonus)
        if stab_ok:
            tier_bonus += float(weights.tier_stability_bonus)

    total = float(shaping_clipped + tier_bonus)

    # Per-axis raw cost components for downstream diagnostics
    r_fusion = float(
        weights.cost_w_fusion * fusion_norm / cost_total_w * weights.cost_weight
    ) if (metric_ok and stab_ok and not use_pareto) else 0.0
    r_k = float(
        weights.cost_w_k * k_norm / cost_total_w * weights.cost_weight
    ) if (metric_ok and stab_ok and not use_pareto) else 0.0
    r_bits = float(
        weights.cost_w_bits * bits_norm / cost_total_w * weights.cost_weight
    ) if (metric_ok and stab_ok and not use_pareto) else 0.0

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
        # v3 per-axis breakdown
        acc_violation_m1=float(acc_violation_m1),
        acc_violation_m2=float(acc_violation_m2),
        margin_m1=float(margin_m1),
        margin_m2=float(margin_m2),
        bits_norm=float(bits_norm),
        fusion_norm=float(fusion_norm),
        k_norm=float(k_norm),
        stab_excess_m1=float(stab_excess_m1),
        stab_excess_m2=float(stab_excess_m2),
        stab_excess_loss=float(stab_excess_loss),
        fusion_gain=float(fusion_gain),
        pareto_event_kind=str(getattr(pareto_event, "kind", "") or ""),
        pareto_action_hash=str(action_hash or ""),
        pareto_frontier_removed=int(getattr(pareto_event, "removed", 0) or 0),
        # legacy / back-compat fields
        r_bits=r_bits,
        r_fusion=r_fusion,
        r_k=r_k,
        r_invalid=float(invalid_term),
        bits_drop=float(bits_gain),     # name kept for compat; semantic = bits_gain
        k_drop=float(k_gain),           # name kept for compat; semantic = k_gain
        fusion_count=float(fusion_count),
        acc_violation=float(combined_acc_violation),
        stab_violation=float(combined_stab_excess),
    )
