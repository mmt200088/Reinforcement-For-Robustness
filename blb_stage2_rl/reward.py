"""BLB Stage 2 RL 奖励函数（v3：m1+m2 双指标 + adaptive scalar P3 cost）。

2026-05-20 v3 重写（保留 ADR-007 的 clipped-shaping + tier-bonus 框架）：

  * 指标 gate 同时看 m1、m2：以前只看 m1 → 现在两者都要过自己的阈值
    (``baseline.metric{i}_mean * (1 - acc_tolerance)``)。margin_acc 取
    两者归一化后的均值。
  * 稳定性 gate 看 m1_std、m2_std、loss_std 三个方差，按 30:30:1 加权
    （和指标重要性一致：m1=m2>>loss）。
  * 顺序 Stage-2 路径的 cost_score 回到 budgeted adaptive scalar：
      - 只有 P3（metric_ok 且 stab_ok 且非 invalid）能吃到 cost reward。
      - P3 内部把 metric margin 和 cost 分开预算：metric 余量只占一个小
        budget，不能挤掉 cost 优化空间。
      - fusion_count 和 truncation/K reduction 是区间式奖励：多一个 fusion
        或多一个 decoded K-bit unit 改善，都会产生清晰的 scalar jump。
      - total_bits 是单独 clip 的弱线性 tie-breaker，不像 fusion/K 那样形成
        明显 tier jump，也不能靠大 bits_gain 主导 cost ranking。
      - Pareto archive 仍可记录 P3 frontier 用于诊断/探索统计，但不再是
        PPO cost scalar 的来源。
  * 优先级硬序：tier_bonus 0/+20/+40 锁住 metric_ok / stab_ok 三档，
    cost_score bounded 且最终 shaping 仍 clip 到 [-5,+5]，所以 cost
    不可能压过指标和稳定性 —— 即便所有 cost 维度同时打满。

核心公式：

  acc_thr_m1 = baseline.metric1_mean * (1 - acc_tolerance)   # e.g. 0.88 * 0.995
  acc_thr_m2 = baseline.metric2_mean * (1 - acc_tolerance)
  acc_violation = max(0, acc_thr_m1 - m1, acc_thr_m2 - m2)
  margin_acc = ((m1 - thr_m1)/denom_m1 + (m2 - thr_m2)/denom_m2) / 2

  stab_thr_X = max(baseline.X_std * stab_tolerance, stab_floor)     # X ∈ {m1,m2,loss}; tol is a MULTIPLIER
  excess_X = max(0, X_std - stab_thr_X)
  norm_X = excess_X / max(baseline.X_std, stab_floor)
  combined_stab_excess = (30·norm_m1 + 30·norm_m2 + 1·norm_loss) / 61
  stability_penalty = -lambda_stab · combined_stab_excess

  P3 budgeted adaptive cost:
    p3_metric_margin = clip(margin_acc, 0, 1) * p3_metric_margin_budget
    fusion_bonus = floor(max(fusion_gain, 0)) * fusion_step_bonus
    truncation_step_gain = max(k_gain, 0) / k_step_size   # avg-K -> coarse K tier units
    k_bonus = floor(truncation_step_gain) * k_step_bonus
    bits_tiebreaker = clip(bits_linear_scale * bits_gain / typical_bits,
                           -bits_tiebreaker_clip, +bits_tiebreaker_clip)
    cost_score = clip(fusion_bonus + k_bonus + bits_tiebreaker,
                      cost_score_clip_min, p3_cost_budget)

  metric_ok = (acc_violation == 0) AND not invalid
  stab_ok = (combined_stab_excess == 0)

  shaping_raw =
      P1: margin_acc + invalid_term
      P2: margin_acc + stab_penalty
      P3: p3_metric_margin + adaptive_cost_score
  shaping_clipped = clip(shaping_raw, -5, +5)
  tier_bonus = 20·metric_ok + 20·(metric_ok AND stab_ok)
  total = shaping_clipped + tier_bonus

reward range:
  · 全部 fail (invalid 或 acc < threshold)   → [-5, 0]
  · metric OK + stab fail                       → [+15, +25]
  · metric OK + stab OK                          → [+35, +45]

3 个 tier 之间至少差 ~15 reward，PPO 看到的 advantage 信号清晰；同时单 episode 的
reward 永远 bounded，cost 优化只在 metric/stab 都通过后驱动候选排序。
"""
from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import numpy as np


def _load_truncation_bounds() -> tuple[int, int]:
    if __package__:
        from .truncation_levels import K_MAX_BITS, K_MIN_BITS

        return K_MAX_BITS, K_MIN_BITS

    try:
        from truncation_levels import K_MAX_BITS, K_MIN_BITS
    except ModuleNotFoundError as exc:
        if exc.name != "truncation_levels":
            raise
    else:
        return K_MAX_BITS, K_MIN_BITS

    sibling = Path(__file__).with_name("truncation_levels.py")
    spec = importlib.util.spec_from_file_location(
        f"_{Path(__file__).stem}_standalone_truncation_levels",
        sibling,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load truncation bounds from {sibling}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.K_MAX_BITS, module.K_MIN_BITS


K_MAX_BITS, K_MIN_BITS = _load_truncation_bounds()


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
DEFAULT_P3_METRIC_MARGIN_BUDGET = 0.50
DEFAULT_P3_COST_BUDGET = 4.50
DEFAULT_COST_FUSION_STEP_BONUS = 0.35
DEFAULT_COST_K_STEP_BONUS = 0.35
DEFAULT_COST_K_STEP_SIZE = 1.0 / 12.0
DEFAULT_COST_BITS_LINEAR_SCALE = 0.10
DEFAULT_COST_BITS_TIEBREAKER_CLIP = 0.25
DEFAULT_COST_SCORE_CLIP_MIN = -0.50
DEFAULT_COST_SCORE_CLIP_MAX = DEFAULT_P3_COST_BUDGET
DEFAULT_STAB_W_M1 = 30.0
DEFAULT_STAB_W_M2 = 30.0
DEFAULT_STAB_W_LOSS = 1.0

# ---------------------------------------------------------------------------
# Fusion-count action P3 cost (2026-06-03 redesign; ADR-008 follow-up).
# ---------------------------------------------------------------------------
# Per-block-TYPE fusion weights + a per-block truncation weight (user ratio
# block1:block2:block4:block5:truncation = 80:150:130:40:50). These drive the
# pure helper ``blb_stage2_rl.fusion_cost.compute_fusion_cost_saving``; the result
# enters ``compute_reward`` as ``external_cost_score`` (bounded P3 cost factor) +
# ``external_cost_rank`` (unbounded ranking). ``total_bits`` is intentionally NOT
# part of the fusion reward scalar. With the current mrpc maps block1/block4 are
# fusion-degenerate, so their 80/130 weights are inert (those blocks tune only K).
FUSION_COST_W = {1: 80.0, 2: 150.0, 4: 130.0, 5: 40.0}
TRUNC_COST_W = 50.0
# The P3 cost budget is split equally between fusion and truncation/K, each
# normalized over its own maximum. Fusion still keeps the per-block-type ratio
# in FUSION_COST_W, while K contributes through TRUNC_COST_W with lower K better.
FUSION_COST_BUDGET_FRACTION = 0.5

# ADR-014 (2026-06-14): STRUCTURAL anti-runaway. The 4th 60k (ADR-013 barrier,
# 4e3aec0) STILL collapsed HOT — fusion marched monotonically 8 -> 35, all P1,
# watchdog-killed at 40320. Root cause: the probe's metric std in the fusion
# regime (~0.0155 at the best point) EXCEEDS the feasibility margin (~0.013),
# ~8.6x the 0.0018 baseline sigma ADR-013 calibrated MARGIN_REF against. So the
# log-barrier's headroom is noise-drowned and cannot form a measurable restoring
# attractor, while the LINEAR fusion cost reward is deterministic & monotone and
# wins. Fix (user-approved, probe size/K kept unchanged): make the fusion cost
# CONCAVE/saturating so its marginal reward -> ~0 past a healthy knee (~fusion 8,
# safely below the noisy boundary ~10-13). ``saturate_fusion`` in fusion_cost.py
# does ``(1-exp(-x/tau))/(1-exp(-1/tau))`` on ``fusion_norm`` in [0,1]; tau=0.15
# gives ~80% of the fusion reward by fusion_norm~0.23 (fusion ~8), steep initial
# slope (still pulls UP to the knee -> no cold collapse) then flat (no incentive
# to over-fuse -> no hot collapse). tau<=0 disables it (bit-for-bit ADR-013).
FUSION_SATURATION_TAU = 0.15

# ADR-015 (2026-06-14): CONTINUOUS bounded reward — a faithful port of Stage-1's
# `_compute_final_reward` design plus the original Stage-2's std-based stability
# constraint, replacing the tier 0/+20/+40 structure. The 4 collapses all share
# the same disease: at the feasibility boundary the policy straddles tiers and the
# reward swings ±40 (the user's "变化幅度太大"), and the 500% stability tolerance
# made the stability gate vacuous ("没看到稳定性约束"). Stage-1 avoids this with a
# SMOOTH reward = accuracy log-barrier + cost, normalized + clipped to [-5,+5]; we
# add a stability log-barrier (the original Stage-2's std constraint). Hard priority
# (item 7) holds via WEIGHTING — a violated barrier (-VIO·exp) dwarfs the P3-gated
# cost — plus strict feasibility selection, NOT via tiers. The strict std gate also
# doubles as the principled anti-runaway brake: high fusion/deep-K raises std past
# the threshold → P2 (not P3) → no cost reward → fusion cannot run away (retires
# the ADR-014 saturation hack). ``reward_design="tiered"`` restores the ADR-014
# path (kept for A/B + rollback). Constants mirror Stage-1
# (layer_importance_evaluator.py:480-396). The active/default Stage-2 objective
# is the stricter Stage-1-aligned variant below: precision barriers first,
# stability barriers second, then cost reward.
DEFAULT_REWARD_DESIGN = "stage1_aligned"
CONT_BARRIER_VIOLATION_SCALE = 10.0    # invalid-case diagnostic magnitude (line ~826)
# ADR-016 (2026-06-16): the violated barrier is now LINEAR in the violation depth,
# not -VIO*exp(-m*STEEP). The exponential exploded → the reward clip flattened it to
# -5 for ANY margin < ~-0.25, so a mild violation (m1=0.84) and a catastrophic one
# (m1=0.63) earned the SAME -5 → zero recovery gradient → the 5th 60k froze in deep
# P1 at max fusion. A linear penalty gives a CONSTANT recovery gradient across the
# realistic violation range so a milder violation always scores strictly higher and
# the policy can climb back.
CONT_BARRIER_VIOLATION_SLOPE = 4.0     # ADR-016 linear violation slope (per unit margin); calibrated by replay
CONT_BARRIER_SATISFACTION_SCALE = 0.5  # Stage-1 LOG_BARRIER_SATISFACTION_SCALE
CONT_REWARD_NORM = 20.0                # Stage-1 REWARD_NORMALIZATION_SCALE
CONT_W_ACC = 1.0                       # accuracy-barrier weight in raw
CONT_W_STAB = 1.0                      # stability-barrier weight (= acc; both hard constraints)
# Max positive a fully-feasible (P3) config earns from cost saving (cost_frac∈[0,1]
# × headroom × this), added AFTER /NORM. Kept < CLIP_MAX so P3 stays bounded.
CONT_W_COST = 4.0
# ADR-016: the cost reward is scaled by the worst-margin HEADROOM so the optimum sits
# at a SAFE positive margin, not the knife-edge boundary. headroom = clip(worst_margin
# / MARGIN_REF, 0, 1): full cost only at margin >= MARGIN_REF; ramps smoothly to 0 as
# the worst margin -> 0 (the boundary), so pushing fusion toward the boundary loses
# cost reward -> a restoring force -> a stable interior optimum (no cliff, no runaway).
# Calibrated by the offline reward-landscape replay so the peak lands at a healthy
# moderate fusion.
CONT_COST_HEADROOM_MARGIN_REF = 1.0

# Stage-1 exact reward constants. These mirror layer_importance_evaluator.py:
# log barriers over hard constraints, cost_saving * 20, divide by 20, clip [-5,5].
STAGE1_REWARD_COST_WEIGHT = 20.0
STAGE1_REWARD_DENSE_SCALE = 0.1
STAGE1_REWARD_NORMALIZATION_SCALE = 20.0
STAGE1_LOG_BARRIER_VIOLATION_SCALE = 10.0
STAGE1_LOG_BARRIER_VIOLATION_STEEPNESS = 20.0
STAGE1_LOG_BARRIER_SATISFACTION_SCALE = 0.5

# ---------------------------------------------------------------------------
# ADR-013 (2026-06-13): Stage-1-style two-piece log-barrier on the accuracy
# margin. REPLACES the ADR-012 graded near-miss tier (P1 shaping) AND the
# linear ``_p3_metric_margin_reward`` (P3 shaping). The 3rd 60k run
# (stage2_grid_gate_60k_20260612_191530) collapsed HOT: fusion marched
# monotonically 1.4 -> 35, metric1 0.866 -> 0.690, and the back half of the
# run froze flat at -6.95 (every episode catastrophic P1) because nothing
# created a restoring force at the feasible frontier and the cliff floor had
# no gradient to climb back from. Stage-1's log-barrier
# (layer_importance_evaluator.py:log_barrier_reward) is the proven shape:
#
#   satisfied (mu >= 0): SAT * (log(mu+eps) - log(MARGIN_REF+eps)), penalty
#       ONLY below the headroom MARGIN_REF, 0 beyond. Its slope SAT/mu -> inf
#       as mu -> 0, so cost(fusion) + barrier(margin) has an interior peak at
#       a POSITIVE margin -> the policy is pushed back instead of overshooting.
#   violated  (mu <  0): a smooth monotone descent (continuous at mu=0, no
#       flat plateau over the realistic collapse depth) so a collapsed policy
#       always sees a gradient toward feasibility -> recovery.
#
# ``mu`` is the worst per-channel SIGNED margin in |baseline - threshold|
# units (same coordinate as the old near_miss_band). MARGIN_REF ~= 2-3x the
# probe-noise sigma so the stable optimum lands where the 256-sample probe can
# actually resolve it (the 3rd-60k optimum sat at margin 0.0003 << sigma 0.0018
# -> a coin flip). Selection / priority / rank-key are UNCHANGED (barrier only
# rewrites the PPO scalar); violated barrier stays < the P3 tier floor so cost
# can never offset an accuracy violation (mental-model item 7).
# MARGIN_REF is THE aggressiveness knob: a probe-noise safety buffer ABOVE the
# (already 0.5%-tolerance-adjusted) accuracy threshold, in |baseline-threshold|
# units where one probe sigma ~= 0.14 of that unit (sigma 0.0018 / denom 0.013
# for mrpc). The stable optimum sits ~at MARGIN_REF, so 0.25 ~= 1.8 sigma. Lower
# it for more aggressive fusion (closer to the boundary, more borderline-P1
# noise — the 3rd-60k optimum at margin ~0.02 was sub-sigma, a coin flip); raise
# it for a safer lower-fusion point. Server sweep range: {0.15, 0.25, 0.35}.
# Stably reaching the knife-edge 22-fusion regime is a probe-resolution problem
# (bigger probe), not a barrier-tuning one.
DEFAULT_ACC_BARRIER_ENABLED = True
DEFAULT_ACC_BARRIER_SAT_SCALE = 0.5      # satisfied-side restoring-force strength
# ADR-014: raised 0.25 -> 0.5. The 4th-60k fusion-regime probe sigma (~0.0155)
# is ~8.6x the 0.0018 baseline sigma ADR-013 assumed, so 0.25 headroom was
# sub-sigma (a coin flip). A larger headroom makes the restoring penalty kick in
# earlier; together with the concave fusion saturation (FUSION_SATURATION_TAU)
# the stable point sits at a moderate, probe-resolvable positive margin.
DEFAULT_ACC_BARRIER_MARGIN_REF = 0.5     # headroom target (|baseline-thr| units)
DEFAULT_ACC_BARRIER_VIO_SCALE = 0.30     # violated-side slope (recovery gradient)
DEFAULT_ACC_BARRIER_FLOOR = -10.0        # lower bound (below realistic collapse depth)
DEFAULT_ACC_BARRIER_EPS = 1.0e-3

# Tolerances driving the per-metric thresholds. ``acc_tolerance`` is the
# relative drop you allow from baseline.metric{i}_mean (0.5% by default —
# matches the 2026-05-18 noisy-baseline preflight constant in sequential_runner).
# ``stab_tolerance`` is a MULTIPLIER on baseline.X_std (2026-06-15 user spec):
# the per-metric std constraint is ``baseline.X_std × tol`` (tol=1.0 → 100% =
# baseline std; tol=5.0 → 5×, deliberately lenient but a REAL gate, never
# vacuous). ``stab_floor`` is deliberately 1e-2, not 1e-3: MRPC
# metric1/metric2 are estimated from only 5 stochastic probe trials over a small
# validation subset, so their standard deviations are quantized by a few samples.
# A 1e-3 floor made normal sampling jitter randomly remove the +20 stability
# tier and collapse rolling reward averages without any accuracy/loss failure.
DEFAULT_ACC_TOLERANCE = 0.005
DEFAULT_STAB_TOLERANCE = 1.2
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
        stab_tolerance:    stability gate 的倍率（MULTIPLIER）；阈值 = baseline.X_std × tol
        stab_floor:        stability 阈值的最小绝对值（防 baseline.std≈0 失稳）
        p3_metric_margin_budget:
                           P3 内 accuracy margin 可占的最大 shaping 空间。
                           该预算故意很小，避免更高精度余量挤掉 cost 优化。
        p3_cost_budget:    P3 内 cost shaping 的最大空间。fusion/K step bonus
                           和 bits tie-breaker 的和会 clip 到此预算内。
        cost_w_fusion / cost_w_k / cost_w_bits:
                           旧配置兼容字段；默认 adaptive scalar P3 cost
                           不再读取这些权重。
        cost_fusion_step_bonus:
                           P3 内每新增 1 个 fusion_gain 的区间式奖励。
        cost_k_step_bonus:
                           P3 内每新增 1 个 truncation/K step 的区间式奖励。
        cost_k_step_size:  ``k_gain`` 是 episode 平均 K drop；默认 1/12 把
                           truncation/K 改善按 layer-equivalent 粗粒度分档。
                           早期 1/59 会把单个 slot 的 K 降低 1 档当成完整
                           reward unit，真实 60k 前序样本中约 27.5% 的 P3
                           候选过早打满 cost clip，削弱 fusion/K 的继续排序。
        cost_bits_linear_scale:
                           total_bits 的弱线性权重；它只做细粒度排序。
        cost_bits_tiebreaker_clip:
                           total_bits 弱线性项的独立 clip，保证 bits-only
                           改善不能接近一个 fusion/K 区间级跳变。
        cost_score_clip_min / max:
                           P3 cost scalar 的独立 clip，仍会再被总 shaping
                           clip 到 [-5,+5]，因此不能跨越 P1/P2 tier。
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
    p3_metric_margin_budget: float = DEFAULT_P3_METRIC_MARGIN_BUDGET
    p3_cost_budget: float = DEFAULT_P3_COST_BUDGET
    # ADR-012 graded near-miss tier (2026-06-12). The 2nd 60k run proved the
    # P1 cliff (-46 vs P3) makes the accuracy-boundary region unnavigable:
    # every fusion-era episode whose 256-sample probe landed a hair below the
    # acc threshold ate the full cliff (8.4% of fusion attempts; ALL of them
    # borderline m1>=0.83, ZERO catastrophic), so expected fusion advantage was
    # +0.117 - 0.084*46 ~= -3.8 and the policy rationally killed fusion. A
    # metric fail that is NOT invalid and whose worst per-channel deficit is
    # within ``near_miss_band`` (in units of |baseline - threshold|) now earns
    # a graded tier between cap (deficit -> 0) and floor (deficit = band)
    # instead of the cliff. Priority stays 1 (selection/rank semantics are
    # untouched - a near-miss can never beat ANY P3 in either rank or scalar:
    # cap 35 < P3 floor 40). Deeper fails keep the old cliff.
    near_miss_tier_cap: float = 35.0
    near_miss_tier_floor: float = 15.0
    near_miss_band: float = 1.0
    # ADR-013 (2026-06-13): Stage-1-style log-barrier accuracy margin. When
    # ``acc_barrier_enabled`` (default), this REPLACES the near_miss_* tier
    # (P1) and the linear p3_metric_margin (P3). Set False to fall back to the
    # ADR-012 near-miss path (kept for the NearMissGradedTierTest A/B).
    acc_barrier_enabled: bool = DEFAULT_ACC_BARRIER_ENABLED
    acc_barrier_sat_scale: float = DEFAULT_ACC_BARRIER_SAT_SCALE
    acc_barrier_margin_ref: float = DEFAULT_ACC_BARRIER_MARGIN_REF
    acc_barrier_vio_scale: float = DEFAULT_ACC_BARRIER_VIO_SCALE
    acc_barrier_floor: float = DEFAULT_ACC_BARRIER_FLOOR
    acc_barrier_eps: float = DEFAULT_ACC_BARRIER_EPS
    # ADR-014 concave/saturating fusion cost — RETIRED by ADR-015 (default 0 =
    # off). The continuous reward's strict std stability gate is now the
    # principled anti-runaway brake (high fusion → high std → P2 → no cost
    # reward), so the saturation hack is no longer needed. The saturate_fusion
    # code stays dormant (tau=0 = identity) for the tiered A/B rollback path.
    fusion_saturation_tau: float = 0.0
    # ADR-015/Stage-1 alignment: the active default is "stage1_aligned".
    # "continuous" remains available for historical A/B; "tiered" restores the
    # ADR-014 path.
    reward_design: str = DEFAULT_REWARD_DESIGN
    cont_barrier_violation_scale: float = CONT_BARRIER_VIOLATION_SCALE
    cont_barrier_violation_slope: float = CONT_BARRIER_VIOLATION_SLOPE
    cont_barrier_satisfaction_scale: float = CONT_BARRIER_SATISFACTION_SCALE
    cont_reward_norm: float = CONT_REWARD_NORM
    cont_w_acc: float = CONT_W_ACC
    cont_w_stab: float = CONT_W_STAB
    cont_w_cost: float = CONT_W_COST
    cont_cost_headroom_margin_ref: float = CONT_COST_HEADROOM_MARGIN_REF
    cost_w_fusion: float = DEFAULT_COST_W_FUSION
    cost_w_k: float = DEFAULT_COST_W_K
    cost_w_bits: float = DEFAULT_COST_W_BITS
    cost_fusion_step_bonus: float = DEFAULT_COST_FUSION_STEP_BONUS
    cost_k_step_bonus: float = DEFAULT_COST_K_STEP_BONUS
    cost_k_step_size: float = DEFAULT_COST_K_STEP_SIZE
    cost_bits_linear_scale: float = DEFAULT_COST_BITS_LINEAR_SCALE
    cost_bits_tiebreaker_clip: float = DEFAULT_COST_BITS_TIEBREAKER_CLIP
    cost_score_clip_min: float = DEFAULT_COST_SCORE_CLIP_MIN
    cost_score_clip_max: float = DEFAULT_COST_SCORE_CLIP_MAX
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
    cost_reward_mode: str = "adaptive_scalar"


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
    loss_trials: Tuple[float, ...] = field(default_factory=tuple)
    metric1_trials: Tuple[float, ...] = field(default_factory=tuple)
    metric2_trials: Tuple[float, ...] = field(default_factory=tuple)
    trial_seeds: Tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.loss_trials = tuple(float(value) for value in self.loss_trials)
        self.metric1_trials = tuple(float(value) for value in self.metric1_trials)
        self.metric2_trials = tuple(float(value) for value in self.metric2_trials)
        self.trial_seeds = tuple(int(value) for value in self.trial_seeds)


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
    p3_metric_margin_reward: float = 0.0
    stability_penalty: float = 0.0
    invalid_term: float = 0.0
    metric_ok: bool = False
    stab_ok: bool = False
    # 2026-06-15: loss_mean lower-better gate (continuous path only). True when
    # loss_mean is within tolerance (or the gate is inactive). Folded into
    # metric_ok in the continuous path; diagnostics elsewhere.
    loss_ok: bool = True
    # ADR-012: graded near-miss (metric fail within near_miss_band of the
    # threshold, not invalid) — priority stays 1, tier is graded not cliff.
    near_miss: bool = False
    acc_worst_deficit_norm: float = 0.0
    # ADR-013: Stage-1-style log-barrier accuracy term (diagnostics). Exactly
    # one is non-zero per episode: sat (mu>=0, the P2/P3 restoring penalty) or
    # vio (mu<0, the P1 recovery-gradient penalty). worst_signed_margin is the
    # barrier coordinate (>=0 feasible w/ headroom, <0 violated).
    acc_barrier_sat: float = 0.0
    acc_barrier_vio: float = 0.0
    # ADR-015: continuous-reward stability barrier (mean Stage-1 log-barrier over
    # the loss/m1/m2 std margins). Diagnostics only; 0 in the tiered path.
    stab_barrier: float = 0.0
    worst_signed_margin: float = 0.0
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
    cost_fusion_bonus: float = 0.0
    cost_truncation_bonus: float = 0.0
    cost_bits_tiebreaker: float = 0.0
    cost_truncation_step_gain: float = 0.0
    # Unbounded P3-only ranking signal. PPO still uses bounded cost_score;
    # candidate selection/archive diagnostics use this to distinguish high-cost
    # improvements after the PPO shaping budget saturates.
    cost_rank_score: float = 0.0
    cost_rank_fusion: float = 0.0
    cost_rank_truncation: float = 0.0
    cost_rank_bits: float = 0.0
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
    loss_precision_probability: float = 0.0
    metric1_precision_probability: float = 0.0
    metric2_precision_probability: float = 0.0
    loss_stability_probability: float = 0.0
    metric1_stability_probability: float = 0.0
    metric2_stability_probability: float = 0.0
    q_precision: float = 0.0
    q_stability: float = 0.0
    precision_signal: float = 0.0
    stability_signal: float = 0.0
    variable_cost: float = 0.0
    compute_saving: float = 0.0
    communication_saving: float = 0.0
    robust_floor: float = 0.0
    secondary_progress: float = 0.0
    ppo_resource_score: float = 0.0
    compute_shapley_credit: float = 0.0
    communication_shapley_credit: float = 0.0
    layer_resource_rewards: Any = field(default_factory=list)
    slot_resource_rewards: Any = field(default_factory=list)
    constraint_policy: str = ""


def boundary_signal(p: float, eps: float = 1.0e-8) -> float:
    """Return the clipped log-distance from the online probability gate."""
    probability = float(p)
    epsilon = float(eps)
    if not math.isfinite(probability):
        raise ValueError("p must be finite")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("eps must be finite and positive")
    probability = float(np.clip(probability, 0.0, 1.0))
    return float(np.clip(
        math.log((probability + epsilon) / (0.5 + epsilon)), -1.0, 1.0,
    ))


_ROBUST_PROBABILITY_FIELDS = (
    "loss_precision_probability",
    "metric1_precision_probability",
    "metric2_precision_probability",
    "loss_stability_probability",
    "metric1_stability_probability",
    "metric2_stability_probability",
)


def _finite_unit_interval(name: str, value: Any) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and in [0, 1]") from exc
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return normalized


def _resource_objective_diagnostics(
        objective: Optional[Mapping[str, Any]],
        ppo_resource_score: float,
        ) -> Mapping[str, Any]:
    """Validate and own the optional layerwise dual-resource diagnostics."""
    score = _finite_unit_interval("ppo_resource_score", ppo_resource_score)
    if objective is None:
        return {
            "compute_saving": 0.0,
            "communication_saving": 0.0,
            "robust_floor": 0.0,
            "secondary_progress": 0.0,
            "ppo_resource_score": score,
            "compute_shapley_credit": 0.0,
            "communication_shapley_credit": 0.0,
            "layer_resource_rewards": [],
            "slot_resource_rewards": [],
        }

    scalar_fields = (
        "compute_saving",
        "communication_saving",
        "robust_floor",
        "secondary_progress",
        "ppo_resource_score",
        "compute_shapley_credit",
        "communication_shapley_credit",
    )
    diagnostics = {
        field_name: _finite_unit_interval(field_name, objective[field_name])
        for field_name in scalar_fields
    }
    if not math.isclose(
            diagnostics["ppo_resource_score"], score,
            rel_tol=0.0, abs_tol=1.0e-12,
    ):
        raise ValueError(
            "external_resource_objective ppo_resource_score does not match "
            "external_cost_score"
        )
    diagnostics["layer_resource_rewards"] = [
        float(value) for value in objective.get("layer_resource_rewards", ())
    ]
    diagnostics["slot_resource_rewards"] = [
        [float(value) for value in row]
        for row in objective.get("slot_resource_rewards", ())
    ]
    return diagnostics


def robust_constrained_reward(
        assessment: Any,
        invalid: bool,
        variable_cost: float,
        eps: float = 1.0e-8,
        ) -> Tuple[float, int, float, float]:
    """Return robust reward, priority, and both boundary signals."""
    cost = _finite_unit_interval("variable_cost", variable_cost)

    if assessment is None:
        if not invalid:
            raise ValueError("robust constrained reward requires a statistical assessment")
        probabilities = {field_name: 0.0 for field_name in _ROBUST_PROBABILITY_FIELDS}
    else:
        probabilities = {
            field_name: _finite_unit_interval(field_name, getattr(assessment, field_name))
            for field_name in _ROBUST_PROBABILITY_FIELDS
        }

    q_precision = min(
        probabilities[field_name] for field_name in _ROBUST_PROBABILITY_FIELDS[:3]
    )
    q_stability = min(
        probabilities[field_name] for field_name in _ROBUST_PROBABILITY_FIELDS[3:]
    )
    precision_signal = boundary_signal(q_precision, eps)
    stability_signal = boundary_signal(q_stability, eps)

    if invalid:
        reward = -5.0
        priority = 1
    elif q_precision < 0.5:
        reward = -3.0 + 0.5 * precision_signal
        priority = 1
    elif q_stability < 0.5:
        reward = -1.5 + 0.5 * stability_signal
        priority = 2
    else:
        reward = 1.0 + cost + 0.0005 * (precision_signal + stability_signal)
        priority = 3
    return (
        float(reward),
        int(priority),
        float(precision_signal),
        float(stability_signal),
    )


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
            max_abs_shaping: float = 0.35,
            frontier_member_shaping: float = 0.05,
            duplicate_shaping: float = -0.025,
            dominated_shaping: float = -0.10,
            expansion_base_shaping: float = 0.20,
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


def stage1_dense_cost_reward(
        total_bits: float,
        baseline_total_bits: float,
        *,
        scale: float = STAGE1_REWARD_DENSE_SCALE,
        ) -> float:
    """Stage-1-style positive dense cost reward for one Stage-2 block step."""
    base = _safe_float(baseline_total_bits, 0.0)
    if base <= 0.0:
        return 0.0
    saving = max(0.0, base - _safe_float(total_bits, base)) / base
    return float(scale) * float(saving)


def stage1_exact_log_barrier(
        curr_value: float,
        limit_value: float,
        *,
        is_upper_bound: bool,
        ) -> float:
    """Exact Stage-1 log-barrier shape from layer_importance_evaluator.py."""
    limit = _safe_float(limit_value, 0.0)
    try:
        curr = float(curr_value)
    except (TypeError, ValueError):
        curr = float("inf") if bool(is_upper_bound) else float("-inf")
    if not math.isfinite(curr):
        curr = float("inf") if bool(is_upper_bound) else float("-inf")
    margin = (limit - curr) if bool(is_upper_bound) else (curr - limit)
    if margin < 0.0:
        exponent = min(
            60.0,
            -float(margin) * STAGE1_LOG_BARRIER_VIOLATION_STEEPNESS,
        )
        return -STAGE1_LOG_BARRIER_VIOLATION_SCALE * math.exp(exponent)
    return STAGE1_LOG_BARRIER_SATISFACTION_SCALE * math.log(margin + 1.0e-5)


def _positive_step_bonus(value: float, *, step_size: float, step_bonus: float) -> float:
    """Interval bonus for a positive cost gain.

    ``value`` is already a gain where larger is better. A step bonus is paid
    only after crossing a full interval; fractional progress is deliberately
    not rewarded here so fusion/K improvements stand out as discrete jumps.
    """
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        return 0.0
    step = max(float(step_size), 1.0e-12)
    units = math.floor(float(value) / step + 1.0e-9)
    return float(max(0, units)) * float(step_bonus)


def _adaptive_scalar_cost_score(
        *,
        fusion_gain: float,
        k_gain: float,
        bits_gain: float,
        bits_norm: float,
        weights: RewardWeights,
        ) -> tuple:
    """P3-only budgeted adaptive scalar cost reward.

    Fusion and truncation/K gains use interval jumps. ``k_gain`` arrives as an
    average over active K slots, so ``cost_k_step_size`` converts it into a
    coarser layer-equivalent K tier. This keeps truncation comparable to fusion
    without letting many tiny per-slot K changes saturate the P3 cost budget too
    early. ``total_bits`` remains a separately clipped weak linear tie-breaker.
    """
    fusion_step = _positive_step_bonus(
        fusion_gain,
        step_size=1.0,
        step_bonus=float(weights.cost_fusion_step_bonus),
    )
    k_step_size = max(float(weights.cost_k_step_size), 1.0e-12)
    truncation_step_gain = max(0.0, float(k_gain)) / k_step_size
    k_step = _positive_step_bonus(
        k_gain,
        step_size=k_step_size,
        step_bonus=float(weights.cost_k_step_bonus),
    )
    bits_linear_raw = float(weights.cost_bits_linear_scale) * float(bits_norm)
    bits_linear = float(np.clip(
        bits_linear_raw,
        -float(weights.cost_bits_tiebreaker_clip),
        float(weights.cost_bits_tiebreaker_clip),
    ))
    raw = float(weights.cost_weight) * (fusion_step + k_step + bits_linear)
    clipped = float(np.clip(
        raw,
        float(weights.cost_score_clip_min),
        min(float(weights.cost_score_clip_max), float(weights.p3_cost_budget)),
    ))
    cost_rank_score = float(fusion_step + k_step + bits_linear_raw)
    return (
        clipped,
        fusion_step,
        k_step,
        bits_linear,
        truncation_step_gain,
        cost_rank_score,
        fusion_step,
        k_step,
        bits_linear_raw,
    )


def _p3_metric_margin_reward(margin_acc: float, weights: RewardWeights) -> float:
    """Small P3-only margin budget that cannot crowd out cost ranking."""
    return float(np.clip(float(margin_acc), 0.0, 1.0)) * float(
        max(0.0, weights.p3_metric_margin_budget)
    )


def accuracy_margin_barrier(worst_signed_margin: float, weights: RewardWeights) -> float:
    """ADR-013 Stage-1-style two-piece log-barrier on the accuracy margin.

    ``worst_signed_margin`` (``mu``) is the worst per-channel signed margin in
    ``|baseline - threshold|`` units: ``mu >= 0`` means feasible (with that
    much headroom), ``mu < 0`` means the accuracy constraint is violated by
    that much.

    Returns a value in ``[acc_barrier_floor, 0]``:

    * **mu >= MARGIN_REF** (comfortable headroom) -> ``0`` (no penalty; cost
      reward alone decides among comfortable P3 configs).
    * **0 <= mu < MARGIN_REF** -> ``SAT*(log(mu+eps) - log(MARGIN_REF+eps))``,
      a NEGATIVE restoring penalty that steepens (slope ``SAT/mu`` -> inf) as
      the margin thins. Combined with the rising cost reward this puts the
      reward peak at a positive interior margin -> the policy is pushed back
      from the boundary instead of overshooting it.
    * **mu < 0** -> a continuous monotone descent ``b0 - VIO*(-mu)`` where
      ``b0`` is the satisfied-side value at ``mu=0`` (continuity). Linear (not
      exp) so it never flattens over the realistic collapse depth: a collapsed
      policy always sees a gradient toward feasibility -> recovery, which the
      flat ``-6.95`` cliff of the 3rd-60k run did not provide.

    The output stays <= 0 (and is clamped >= ``acc_barrier_floor``), so when it
    feeds the P1 shaping it is always far below the P3 tier floor (item 7), and
    when it feeds the P3 shaping it only ever reduces (never inflates) the
    cost-led ranking.
    """
    mu = float(worst_signed_margin)
    eps = float(weights.acc_barrier_eps)
    ref = float(weights.acc_barrier_margin_ref)
    sat = float(weights.acc_barrier_sat_scale)
    floor = float(weights.acc_barrier_floor)
    log_ref = math.log(ref + eps)
    if mu >= 0.0:
        raw = sat * (math.log(mu + eps) - log_ref)
        val = min(0.0, raw)          # penalty only below headroom; 0 beyond
    else:
        b0 = sat * (math.log(eps) - log_ref)            # satisfied-side value at mu=0
        val = b0 - float(weights.acc_barrier_vio_scale) * (-mu)
    return max(floor, val)


def stage1_log_barrier(margin: float, weights: RewardWeights) -> float:
    """ADR-015 Stage-1-style log-barrier on ONE signed normalized margin.

    Faithful port of ``layer_importance_evaluator.log_barrier_reward``:
      * ``margin >= 0`` (satisfied): ``SAT * log(margin + 1e-5)`` — diminishing
        reward for extra headroom (can be slightly negative for margin < 1).
      * ``margin < 0`` (violated): ``-VIO * exp(-margin * STEEPNESS)`` — a smooth
        exponential penalty that grows steeply as the violation deepens (the
        exponent is clamped so it never overflows; the caller's reward clip then
        bounds it to ``reward_clip_min``).
    Same shape for the accuracy AND stability constraints (the original Stage-2
    used an std-based stability penalty; here it is a log-barrier for symmetry).
    """
    m = float(margin)
    if not math.isfinite(m):
        m = -1.0e9
    if m < 0.0:
        # ADR-016: LINEAR violation penalty (was -VIO*exp(-m*STEEP), which exploded
        # → the reward clip flattened it to clip_min for ANY margin < ~-0.25, so a
        # mild violation and a catastrophic one scored identically → zero recovery
        # gradient → the policy froze in deep P1 at max fusion). Linear in the
        # violation depth gives a CONSTANT recovery gradient across the realistic
        # violation range, so a milder violation always scores strictly higher and
        # the policy can climb back toward feasibility. Bounded below only by the
        # caller's reward clip (which now only engages at extreme depth).
        return float(weights.cont_barrier_violation_slope) * m  # m<0 → negative; deeper → lower
    return float(weights.cont_barrier_satisfaction_scale) * math.log(m + 1.0e-5)


def _continuous_reward(
        *,
        acc_margins: Sequence[float],
        std_margins: Sequence[float],
        effective_cost_score: float,
        invalid: bool,
        weights: RewardWeights,
        ) -> tuple:
    """ADR-015 continuous bounded reward (Stage-1 design + std stability).

    ``raw = W_acc*mean(acc_barrier) + W_stab*mean(stab_barrier)``; the scalar is
    ``clip(raw/NORM + W_cost*cost_frac*headroom, CLIP_MIN, CLIP_MAX)``. ``cost_frac``
    is the P3-gated saving in ``[0,1]``; ``headroom = clip(worst_margin/MARGIN_REF,
    0, 1)`` (ADR-016) fades the cost to 0 as the worst margin → 0 and is 0 in
    violation. Hard priority / item 7 holds two ways: a violation drives the worst
    margin < 0 ⇒ headroom = 0 ⇒ NO cost, and the violated barrier (linear,
    ``SLOPE*m`` per channel) is negative ⇒ a violation always scores below any
    fully-feasible config. Returns ``(scalar, acc_barrier, stab_barrier)``.
    """
    clip_min = float(weights.reward_clip_min)
    clip_max = float(weights.reward_clip_max)
    norm = float(weights.cont_reward_norm) or 1.0
    if invalid:
        # Invalid chain: the model forward is skipped, metrics are unreliable, so
        # treat it as a hard violation pinned at the floor (bounded, like the rest).
        return float(clip_min), float(-weights.cont_barrier_violation_scale), 0.0
    acc_vals = [stage1_log_barrier(m, weights) for m in acc_margins]
    stab_vals = [stage1_log_barrier(m, weights) for m in std_margins]
    acc_b = (sum(acc_vals) / len(acc_vals)) if acc_vals else 0.0
    stab_b = (sum(stab_vals) / len(stab_vals)) if stab_vals else 0.0
    barrier_raw = float(weights.cont_w_acc) * acc_b + float(weights.cont_w_stab) * stab_b
    cost_frac = float(np.clip(
        float(effective_cost_score) / max(float(weights.p3_cost_budget), 1e-8), 0.0, 1.0,
    ))
    # ADR-016: scale the cost reward by the worst-margin HEADROOM so the optimum sits
    # at a SAFE positive margin, not the knife-edge boundary. In the feasible region
    # the cost lure used to dominate the (tiny, /NORM) barrier (cost≈2.4 ≫ barrier≈
    # 0.02), pulling fusion right up to the boundary; then P3-gating cliffed cost to 0
    # there. headroom smoothly takes the cost to 0 as the worst margin → 0, so pushing
    # fusion toward the boundary loses cost reward → a restoring force → a stable
    # interior optimum (no cliff, no runaway). worst margin < 0 → headroom = 0 → cost
    # = 0 (item 7: a violation never earns cost, on top of the upstream P3-gate).
    worst_overall = min((*acc_margins, *std_margins), default=0.0)
    headroom = float(np.clip(
        worst_overall / max(float(weights.cont_cost_headroom_margin_ref), 1e-8), 0.0, 1.0,
    ))
    scalar = barrier_raw / norm + float(weights.cont_w_cost) * cost_frac * headroom
    return float(np.clip(scalar, clip_min, clip_max)), float(acc_b), float(stab_b)


def _stage1_aligned_cost_fraction(
        *,
        external_cost_score: Optional[float],
        baseline: BaselineCostStats,
        opt_total_bits: float,
        action_avg_k: float,
        weights: RewardWeights,
        ) -> float:
    """Resolve Stage-2 cost savings to the Stage-1 ``cost_saving`` coordinate."""
    if external_cost_score is not None:
        return float(np.clip(
            _safe_float(external_cost_score, 0.0)
            / max(_safe_float(weights.p3_cost_budget, 4.5), 1.0e-8),
            0.0,
            1.0,
        ))
    bits_frac = 0.0
    base_bits = _safe_float(baseline.total_bits_sum, 0.0)
    if base_bits > 0.0:
        bits_frac = max(0.0, base_bits - _safe_float(opt_total_bits, base_bits)) / base_bits
    k_frac = 0.0
    base_k = _safe_float(baseline.avg_k, DEFAULT_BASELINE_AVG_K)
    k_denom = max(base_k - K_MIN_BITS, 1.0e-8)
    if base_k > K_MIN_BITS:
        k_frac = max(0.0, base_k - _safe_float(action_avg_k, base_k)) / k_denom
    return float(np.clip(max(bits_frac, k_frac), 0.0, 1.0))


def _stage1_aligned_terminal_reward(
        *,
        metrics: EpisodeMetrics,
        baseline: BaselineCostStats,
        weights: RewardWeights,
        thr_m1: float,
        thr_m2: float,
        m1_active: bool,
        m2_active: bool,
        stab_thr_m1: float,
        stab_thr_m2: float,
        stab_thr_loss: float,
        loss_threshold: Optional[float] = None,
        invalid: bool,
        cost_fraction: float,
        ) -> tuple:
    """Stage-1 terminal reward plus Stage-2 std stability barriers."""
    if invalid:
        return (
            float(weights.reward_clip_min),
            0.0,
            0.0,
            False,
            False,
            False,
            0.0,
            0.0,
        )

    baseline_loss = _safe_float(baseline.loss_mean, 0.0)
    if loss_threshold is not None and math.isfinite(float(loss_threshold)):
        loss_limit = float(loss_threshold)
        loss_active = True
    else:
        loss_limit = baseline_loss * (1.0 + float(weights.acc_tolerance))
        loss_active = baseline_loss > float(weights.margin_denom_floor)
    constraint_terms = []
    if loss_active:
        constraint_terms.append(stage1_exact_log_barrier(
            metrics.loss_mean, loss_limit, is_upper_bound=True,
        ))
    if m1_active:
        constraint_terms.append(stage1_exact_log_barrier(
            metrics.metric1_mean, thr_m1, is_upper_bound=False,
        ))
    if m2_active:
        constraint_terms.append(stage1_exact_log_barrier(
            metrics.metric2_mean, thr_m2, is_upper_bound=False,
        ))

    stability_terms = [
        stage1_exact_log_barrier(metrics.metric1_std, stab_thr_m1, is_upper_bound=True),
        stage1_exact_log_barrier(metrics.metric2_std, stab_thr_m2, is_upper_bound=True),
        stage1_exact_log_barrier(metrics.loss_std, stab_thr_loss, is_upper_bound=True),
    ]
    constraint_barrier = (
        float(sum(constraint_terms) / len(constraint_terms))
        if constraint_terms else 0.0
    )
    stability_barrier = float(sum(stability_terms) / len(stability_terms))

    loss_ok = (not loss_active) or (_safe_float(metrics.loss_mean, float("inf")) <= loss_limit)
    metric_ok = bool(loss_ok)
    if m1_active:
        metric_ok = metric_ok and (_safe_float(metrics.metric1_mean, 0.0) >= float(thr_m1))
    if m2_active:
        metric_ok = metric_ok and (_safe_float(metrics.metric2_mean, 0.0) >= float(thr_m2))
    stab_ok = (
        math.isfinite(float(metrics.metric1_std))
        and math.isfinite(float(metrics.metric2_std))
        and math.isfinite(float(metrics.loss_std))
        and float(metrics.metric1_std) <= float(stab_thr_m1)
        and float(metrics.metric2_std) <= float(stab_thr_m2)
        and float(metrics.loss_std) <= float(stab_thr_loss)
    )
    cost_reward = (
        STAGE1_REWARD_COST_WEIGHT * float(np.clip(cost_fraction, 0.0, 1.0))
        if (metric_ok and stab_ok)
        else 0.0
    )
    if not metric_ok:
        raw = constraint_barrier
    elif not stab_ok:
        raw = constraint_barrier + stability_barrier
    else:
        raw = constraint_barrier + stability_barrier + cost_reward
    scalar = raw / STAGE1_REWARD_NORMALIZATION_SCALE
    return (
        float(np.clip(scalar, float(weights.reward_clip_min), float(weights.reward_clip_max))),
        float(constraint_barrier),
        float(stability_barrier),
        bool(loss_ok),
        bool(metric_ok),
        bool(stab_ok),
        float(cost_reward),
        float(raw - cost_reward),
    )


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
        external_cost_score: Optional[float] = None,
        external_cost_rank: Optional[float] = None,
        external_resource_objective: Optional[Mapping[str, Any]] = None,
        loss_threshold: Optional[float] = None,
        constraint_assessment: Any = None,
        ) -> RewardBreakdown:
    """v3 reward with loss/m1/m2 precision gates and std stability gates.

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
        pareto_archive:    可选 P3 cost archive。传入后仍只收 P3 候选，用于
                           frontier 诊断/探索统计；默认不再决定 PPO scalar。
        action_hash:       Pareto archive 的候选身份；与 pareto_archive 同时传入
                           才记录 P3 frontier 诊断事件。
        external_cost_score: fusion-count 路径专用。调用方（env + fusion map）算好的
                           per-block 加权 cost 节省（已是 [0, p3_cost_budget] 量级的有界
                           标量），非 None 时直接替掉聚合 fusion/K/bits cost_score；
                           仅在 P3（metric_ok 且 stab_ok）生效，total_bits 不参与。
        external_cost_rank: 对应的无界排序值（候选/前沿排序用，永不进 PPO 标量）。
        external_resource_objective: layerwise 双资源目标的显式诊断字段；其
                           ``ppo_resource_score`` 必须与 external_cost_score 一致。

    Returns:
        ``RewardBreakdown``
    """
    weights = weights or RewardWeights()

    invalid = bool(any_invalid) if any_invalid is not None else bool(
        getattr(opt_signals, "any_invalid", False)
    )

    if str(getattr(weights, "reward_design", "tiered")).strip().lower() == "robust_constrained":
        if not invalid and external_cost_score is None:
            raise ValueError(
                "robust_constrained valid candidate requires external_cost_score"
            )
        raw_variable_cost = 0.0 if external_cost_score is None else external_cost_score
        reward, priority, precision_signal, stability_signal = robust_constrained_reward(
            constraint_assessment,
            invalid=invalid,
            variable_cost=raw_variable_cost,
        )
        variable_cost = float(raw_variable_cost)
        resource_diagnostics = _resource_objective_diagnostics(
            external_resource_objective, variable_cost,
        )
        probabilities = (
            {field_name: 0.0 for field_name in _ROBUST_PROBABILITY_FIELDS}
            if constraint_assessment is None
            else {
                field_name: float(getattr(constraint_assessment, field_name))
                for field_name in _ROBUST_PROBABILITY_FIELDS
            }
        )
        q_precision = min(
            probabilities[field_name] for field_name in _ROBUST_PROBABILITY_FIELDS[:3]
        )
        q_stability = min(
            probabilities[field_name] for field_name in _ROBUST_PROBABILITY_FIELDS[3:]
        )
        metric_ok = bool(priority >= 2) and not invalid
        stab_ok = bool(priority == 3) and metric_ok
        return RewardBreakdown(
            reward=float(reward),
            priority=int(priority),
            invalid=invalid,
            metric_ok=metric_ok,
            stab_ok=stab_ok,
            loss_ok=bool(probabilities["loss_precision_probability"] >= 0.5) and not invalid,
            cost_score=float(variable_cost) if stab_ok else 0.0,
            variable_cost=float(variable_cost),
            constraint_policy="bootstrap_5x5_v1",
            q_precision=float(q_precision),
            q_stability=float(q_stability),
            precision_signal=float(precision_signal),
            stability_signal=float(stability_signal),
            **resource_diagnostics,
            **probabilities,
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

    # loss_mean (LOWER-better) diagnostic for legacy continuous mode. The active
    # stage1_aligned branch below folds loss_threshold into the per-episode
    # precision barrier and gate, matching Stage-1's loss/m1/m2 objective.
    _continuous_design = str(getattr(weights, "reward_design", "tiered")) == "continuous"
    baseline_loss_mean = _safe_float(getattr(baseline, "loss_mean", 0.0), 0.0)
    loss_mean_active = _continuous_design and (
        abs(baseline_loss_mean) > float(weights.margin_denom_floor)
    )
    if loss_mean_active:
        loss_mean_val = _safe_float(metrics.loss_mean, 0.0)
        thr_loss_mean_diag = baseline_loss_mean * (1.0 + float(weights.acc_tolerance))
        loss_mean_ok = (loss_mean_val <= thr_loss_mean_diag)
    else:
        loss_mean_ok = True

    active_metric_count = (1 if m1_active else 0) + (1 if m2_active else 0)
    if active_metric_count > 0:
        margin_acc = (margin_m1 + margin_m2) / float(active_metric_count)
    else:
        # Neither baseline calibrated; fall back to "no margin signal" so
        # legacy callers that just want the tier_bonus path still work.
        margin_acc = 0.0
    # Legacy tiered/continuous code below uses m1/m2 for priority. The active
    # stage1_aligned branch recomputes metric_ok with loss_ok included.
    combined_acc_violation = max(acc_violation_m1, acc_violation_m2)
    # ADR-012: worst per-channel deficit normalized by that channel's
    # |baseline - threshold| width — the near-miss grading coordinate.
    _deficits = []
    if m1_active:
        _deficits.append(acc_violation_m1 / denom_m1)
    if m2_active:
        _deficits.append(acc_violation_m2 / denom_m2)
    acc_worst_deficit_norm = max(_deficits) if _deficits else 0.0

    # ADR-013: worst per-channel SIGNED normalized margin (>=0 feasible w/
    # headroom, <0 violated) — the log-barrier coordinate. Uses the same
    # |baseline - threshold| normalization as the deficit/near_miss_band.
    _signed_margins = []
    if m1_active:
        _signed_margins.append(margin_m1)
    if m2_active:
        _signed_margins.append(margin_m2)
    # loss_mean intentionally excluded from the per-episode barrier (see determinism
    # note at combined_acc_violation): its noisy reference is not cross-GPU-identical.
    worst_signed_margin = min(_signed_margins) if _signed_margins else 0.0

    # === 2. Raw cost gains; scalar cost_score is legacy-only unless a Pareto archive is absent. ===
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

    cost_score_raw = 0.0

    # === 3. combined stability_excess (m1_std, m2_std, loss_std) ===
    def _stab_threshold(baseline_std: float) -> float:
        # 2026-06-15 (user spec): ``stab_tolerance`` is a MULTIPLIER on the
        # baseline std — the per-metric std constraint is ``baseline.X_std × tol``
        # (tol=1.0 → 100% = baseline; tol=5.0 → 5×, a deliberately LENIENT but
        # real gate, not an empty one). Was ``baseline_std·(1+tol)+floor``
        # (fractional-slack semantics), which both mislabeled a lenient 6× setting
        # as "vacuous" and broke the "× value" interpretation. ``stab_floor`` is
        # kept only as an absolute minimum for the degenerate near-zero-baseline
        # (probe-quantization) case; set it to 0 for a pure multiplier.
        return max(
            float(baseline_std) * float(weights.stab_tolerance),
            float(weights.stab_floor),
        )

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

    # === 4.5. Optional Pareto archive diagnostics + adaptive scalar P3 cost ===
    # Cost must never help P1/P2 candidates. The Pareto archive may still be
    # wired for diagnostics / frontier-neighbor exploration, but PPO's scalar
    # cost signal is now adaptive scalar unless weights.cost_reward_mode is
    # explicitly set to "pareto_only".
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

    (
        _rank_cost_score_raw,
        _rank_fusion_raw,
        _rank_k_raw,
        _rank_bits_clipped_raw,
        _rank_truncation_step_gain_raw,
        cost_rank_score_raw,
        cost_rank_fusion_raw,
        cost_rank_truncation_raw,
        cost_rank_bits_raw,
    ) = _adaptive_scalar_cost_score(
        fusion_gain=float(fusion_gain),
        k_gain=float(k_gain),
        bits_gain=float(bits_gain),
        bits_norm=float(bits_norm),
        weights=weights,
    )

    if str(getattr(weights, "cost_reward_mode", "adaptive_scalar")) == "pareto_only":
        cost_score_raw = (
            float(getattr(pareto_event, "shaping", 0.0) or 0.0)
            if (priority == 3 and pareto_event is not None)
            else 0.0
        )
        r_fusion_raw = 0.0
        r_k_raw = 0.0
        r_bits_raw = 0.0
        truncation_step_gain_raw = max(0.0, float(k_gain)) / max(
            float(weights.cost_k_step_size), 1.0e-12
        )
    else:
        (
            cost_score_raw,
            r_fusion_raw,
            r_k_raw,
            r_bits_raw,
            truncation_step_gain_raw,
            _cost_rank_score_unused,
            _cost_rank_fusion_unused,
            _cost_rank_truncation_unused,
            _cost_rank_bits_unused,
        ) = _adaptive_scalar_cost_score(
            fusion_gain=float(fusion_gain),
            k_gain=float(k_gain),
            bits_gain=float(bits_gain),
            bits_norm=float(bits_norm),
            weights=weights,
        )

    # Fusion-count action path: the P3 cost is a per-block weighted saving computed
    # outside (action_space + fusion map), passed in as external_cost_score (already a
    # bounded [0, p3_cost_budget] scalar) + external_cost_rank (unbounded ranking). It
    # replaces the aggregate fusion/K/bits scalar; total_bits is not part of the fusion
    # reward. Injected here, before the P3 gate below, so it still only fires in P3.
    if external_cost_score is not None:
        cost_score_raw = float(np.clip(
            float(external_cost_score), 0.0, float(weights.p3_cost_budget)
        ))
        cost_rank_score_raw = (
            float(external_cost_rank) if external_cost_rank is not None else 0.0
        )
        r_fusion_raw = 0.0
        r_k_raw = 0.0
        r_bits_raw = 0.0
        truncation_step_gain_raw = 0.0
        cost_rank_fusion_raw = 0.0
        cost_rank_truncation_raw = 0.0
        cost_rank_bits_raw = 0.0

    # Cost & stability only contribute to shaping when metric_ok. Cost is even
    # stricter: it only contributes in P3 (metric_ok AND stab_ok), matching the
    # "accuracy/stability as hard constraints, cost only after both pass" rule.
    effective_cost_score = cost_score_raw if (metric_ok and stab_ok) else 0.0
    effective_cost_rank_score = cost_rank_score_raw if (metric_ok and stab_ok) else 0.0
    effective_cost_rank_fusion = cost_rank_fusion_raw if (metric_ok and stab_ok) else 0.0
    effective_cost_rank_truncation = (
        cost_rank_truncation_raw if (metric_ok and stab_ok) else 0.0
    )
    effective_cost_rank_bits = cost_rank_bits_raw if (metric_ok and stab_ok) else 0.0
    effective_p3_margin = (
        _p3_metric_margin_reward(margin_acc, weights)
        if (metric_ok and stab_ok)
        else 0.0
    )
    effective_stab_penalty = stability_penalty if (metric_ok and not stab_ok) else 0.0
    invalid_term = -float(weights.invalid_penalty) if invalid else 0.0

    # === 5. shaping ===
    _clip_min = float(weights.reward_clip_min)
    _clip_max = float(weights.reward_clip_max)
    acc_barrier_sat = 0.0
    acc_barrier_vio = 0.0
    near_miss = False
    stab_barrier = 0.0  # ADR-015 continuous-reward stability barrier (diagnostics)
    # Barrier handles the non-invalid accuracy dimension; invalid episodes keep
    # the legacy invalid_term shaping (their metrics are unreliable — the model
    # forward is skipped on an invalid chain).
    barrier_on = bool(getattr(weights, "acc_barrier_enabled", False)) and not invalid
    if barrier_on:
        # ADR-013: Stage-1-style log-barrier replaces both the linear P3 margin
        # and the ADR-012 near-miss/cliff. Satisfied side (P2/P3) is a <=0
        # restoring penalty below MARGIN_REF; violated side (P1) is a smooth
        # monotone descent with a recovery gradient.
        _barrier = accuracy_margin_barrier(worst_signed_margin, weights)
        if worst_signed_margin >= 0.0:
            acc_barrier_sat = _barrier
        else:
            acc_barrier_vio = _barrier
        if metric_ok and stab_ok:            # P3: barrier (<=0) + cost-led ranking
            shaping_raw = float(acc_barrier_sat) + float(effective_cost_score)
            shaping_clipped = float(np.clip(shaping_raw, _clip_min, _clip_max))
        elif metric_ok and not stab_ok:      # P2: barrier (<=0) + stability penalty
            shaping_raw = float(acc_barrier_sat) + float(effective_stab_penalty)
            shaping_clipped = float(np.clip(shaping_raw, _clip_min, _clip_max))
        else:                                # P1 (accuracy violated, not invalid)
            # Clamp to the barrier floor (NOT -5) so the violated region keeps
            # its monotone gradient — the missing recovery path of the 3rd 60k.
            shaping_raw = float(acc_barrier_vio)
            shaping_clipped = float(
                np.clip(shaping_raw, float(weights.acc_barrier_floor), 0.0)
            )
    else:
        # Legacy ADR-012 path (near-miss tier + linear P3 margin) — also the
        # invalid-episode path.
        if metric_ok and stab_ok:
            shaping_raw = float(effective_p3_margin) + float(effective_cost_score)
        else:
            shaping_raw = float(margin_acc) + invalid_term + effective_stab_penalty
        shaping_clipped = float(np.clip(shaping_raw, _clip_min, _clip_max))

    # === 6. tier_bonus (hard-priority via large jumps; +20/+40 dominates cost) ===
    tier_bonus = 0.0
    if metric_ok:
        tier_bonus += float(weights.tier_metric_bonus)
        if stab_ok:
            tier_bonus += float(weights.tier_stability_bonus)
    elif (
            not barrier_on
            and not invalid
            and combined_acc_violation > 0.0
            and float(weights.near_miss_band) > 0.0
            and acc_worst_deficit_norm <= float(weights.near_miss_band)
    ):
        # ADR-012 graded near-miss (only when the barrier is disabled — the
        # barrier provides its own smooth P1 shaping, no tier on top). Stays
        # strictly below the P3 floor; priority stays 1 so selection is unchanged.
        near_miss = True
        cap = float(weights.near_miss_tier_cap)
        floor = float(weights.near_miss_tier_floor)
        frac = acc_worst_deficit_norm / float(weights.near_miss_band)
        tier_bonus += cap - (cap - floor) * float(np.clip(frac, 0.0, 1.0))

    total = float(shaping_clipped + tier_bonus)

    # === ADR-015: CONTINUOUS reward override (replaces the tier total) ===
    # When reward_design="continuous" (default), the smooth log-barrier reward
    # supersedes the tier 0/+20/+40 structure: accuracy log-barrier + stability
    # log-barrier + P3-gated cost, all clipped to [-5,+5] (Stage-1's design + the
    # original Stage-2's std stability). The tiered total above is computed but
    # discarded (cheap); priority / cost gating / breakdown share the same path.
    # Hard priority (item 7) holds via weighting: a violated barrier pins the
    # scalar at CLIP_MIN while cost can only lift a fully-feasible config by W_cost.
    if str(getattr(weights, "reward_design", "tiered")) == "continuous":
        def _std_margin(observed: float, threshold: float, denom: float) -> float:
            if not math.isfinite(observed):
                return -1.0e9
            return (float(threshold) - float(observed)) / max(
                abs(float(denom)), float(weights.stab_floor)
            )
        _acc_margins = []
        if m1_active:
            _acc_margins.append(margin_m1)
        if m2_active:
            _acc_margins.append(margin_m2)
        # loss_mean excluded from the continuous barrier scalar (determinism: its
        # noisy reference is not cross-GPU-identical). Enforced at strict selection.
        _std_margins = [
            _std_margin(m1_std, stab_thr_m1, denom_m1_std),
            _std_margin(m2_std, stab_thr_m2, denom_m2_std),
            _std_margin(loss_std, stab_thr_loss, denom_loss_std),
        ]
        _scalar, _acc_b, _stab_b = _continuous_reward(
            acc_margins=_acc_margins,
            std_margins=_std_margins,
            effective_cost_score=effective_cost_score,
            invalid=invalid,
            weights=weights,
        )
        acc_barrier_sat = _acc_b if worst_signed_margin >= 0.0 else 0.0
        acc_barrier_vio = _acc_b if worst_signed_margin < 0.0 else 0.0
        stab_barrier = float(_stab_b)
        near_miss = False
        tier_bonus = 0.0
        shaping_raw = float(_scalar)
        shaping_clipped = float(_scalar)
        total = float(_scalar)

    if str(getattr(weights, "reward_design", "tiered")) == "stage1_aligned":
        cost_fraction = _stage1_aligned_cost_fraction(
            external_cost_score=external_cost_score,
            baseline=baseline,
            opt_total_bits=opt_total_bits,
            action_avg_k=action_avg_k,
            weights=weights,
        )
        (
            _scalar,
            _constraint_barrier,
            _stab_barrier,
            _loss_ok,
            _metric_ok,
            _stab_ok,
            _stage1_cost_reward,
            _stage1_combined_barrier,
        ) = _stage1_aligned_terminal_reward(
            metrics=metrics,
            baseline=baseline,
            weights=weights,
            thr_m1=thr_m1,
            thr_m2=thr_m2,
            m1_active=m1_active,
            m2_active=m2_active,
            stab_thr_m1=stab_thr_m1,
            stab_thr_m2=stab_thr_m2,
            stab_thr_loss=stab_thr_loss,
            loss_threshold=loss_threshold,
            invalid=invalid,
            cost_fraction=cost_fraction,
        )
        loss_mean_ok = bool(_loss_ok)
        metric_ok = bool(_metric_ok) and not invalid
        stab_ok = bool(_stab_ok)
        if invalid or not metric_ok:
            priority = 1
        elif not stab_ok:
            priority = 2
        else:
            priority = 3
        effective_cost_score = float(cost_fraction) if (metric_ok and stab_ok) else 0.0
        effective_cost_rank_score = (
            float(external_cost_rank)
            if (external_cost_rank is not None and metric_ok and stab_ok)
            else float(cost_fraction)
            if (metric_ok and stab_ok)
            else 0.0
        )
        effective_cost_rank_fusion = 0.0
        effective_cost_rank_truncation = 0.0
        effective_cost_rank_bits = 0.0
        acc_barrier_sat = float(_constraint_barrier) if metric_ok else 0.0
        acc_barrier_vio = float(_constraint_barrier) if not metric_ok else 0.0
        stab_barrier = float(_stab_barrier)
        near_miss = False
        tier_bonus = 0.0
        shaping_raw = float(_stage1_combined_barrier + _stage1_cost_reward)
        shaping_clipped = float(_scalar)
        total = float(_scalar)

    # Per-axis raw cost components for downstream diagnostics
    r_fusion = float(r_fusion_raw) if (metric_ok and stab_ok) else 0.0
    r_k = float(r_k_raw) if (metric_ok and stab_ok) else 0.0
    r_bits = float(r_bits_raw) if (metric_ok and stab_ok) else 0.0

    return RewardBreakdown(
        reward=float(total),
        priority=int(priority),
        invalid=invalid,
        shaping_raw=float(shaping_raw),
        shaping_clipped=float(shaping_clipped),
        tier_bonus=float(tier_bonus),
        margin_acc=float(margin_acc),
        cost_score=float(effective_cost_score),
        # When the barrier is on it replaces the linear P3 margin; report 0 so
        # the field never double-counts what is actually in ``total``.
        p3_metric_margin_reward=(0.0 if barrier_on else float(effective_p3_margin)),
        stability_penalty=float(effective_stab_penalty),
        invalid_term=float(invalid_term),
        metric_ok=bool(metric_ok),
        stab_ok=bool(stab_ok),
        loss_ok=bool(loss_mean_ok),
        near_miss=bool(near_miss),
        acc_worst_deficit_norm=float(acc_worst_deficit_norm),
        acc_barrier_sat=float(acc_barrier_sat),
        acc_barrier_vio=float(acc_barrier_vio),
        stab_barrier=float(stab_barrier),
        worst_signed_margin=float(worst_signed_margin),
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
        cost_fusion_bonus=float(r_fusion),
        cost_truncation_bonus=float(r_k),
        cost_bits_tiebreaker=float(r_bits),
        cost_truncation_step_gain=(
            float(truncation_step_gain_raw) if (metric_ok and stab_ok) else 0.0
        ),
        cost_rank_score=float(effective_cost_rank_score),
        cost_rank_fusion=float(effective_cost_rank_fusion),
        cost_rank_truncation=float(effective_cost_rank_truncation),
        cost_rank_bits=float(effective_cost_rank_bits),
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
