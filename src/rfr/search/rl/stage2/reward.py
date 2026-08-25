"""Stage-2 reward calculation and diagnostic breakdowns.

Every reward design orders precision before stability and resource savings.
The production robust-constrained path uses six bootstrap probabilities and a
bounded dual-resource score, so infeasible candidates cannot gain cost credit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

import numpy as np

from rfr.search.common.truncation_levels import K_MAX_BITS, K_MIN_BITS


DEFAULT_REWARD_CLIP_MIN = -5.0
DEFAULT_REWARD_CLIP_MAX = 5.0
DEFAULT_TIER_METRIC_BONUS = 20.0
DEFAULT_TIER_STABILITY_BONUS = 20.0
DEFAULT_LAMBDA_STAB = 1.0


DEFAULT_INVALID_PENALTY = 5.0
DEFAULT_MARGIN_DENOM_FLOOR = 0.01
DEFAULT_BASELINE_AVG_K = 13.0


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


FUSION_COST_W = {1: 80.0, 2: 150.0, 4: 130.0, 5: 40.0}
TRUNC_COST_W = 50.0


FUSION_COST_BUDGET_FRACTION = 0.5


FUSION_SATURATION_TAU = 0.15


DEFAULT_REWARD_DESIGN = "stage1_aligned"
CONT_BARRIER_VIOLATION_SCALE = 10.0


CONT_BARRIER_VIOLATION_SLOPE = 4.0
CONT_BARRIER_SATISFACTION_SCALE = 0.5
CONT_REWARD_NORM = 20.0
CONT_W_ACC = 1.0
CONT_W_STAB = 1.0


CONT_W_COST = 4.0


CONT_COST_HEADROOM_MARGIN_REF = 1.0


STAGE1_REWARD_COST_WEIGHT = 20.0
STAGE1_REWARD_DENSE_SCALE = 0.1
STAGE1_REWARD_NORMALIZATION_SCALE = 20.0
STAGE1_LOG_BARRIER_VIOLATION_SCALE = 10.0
STAGE1_LOG_BARRIER_VIOLATION_STEEPNESS = 20.0
STAGE1_LOG_BARRIER_SATISFACTION_SCALE = 0.5


DEFAULT_ACC_BARRIER_ENABLED = True
DEFAULT_ACC_BARRIER_SAT_SCALE = 0.5


DEFAULT_ACC_BARRIER_MARGIN_REF = 0.5
DEFAULT_ACC_BARRIER_VIO_SCALE = 0.30
DEFAULT_ACC_BARRIER_FLOOR = -10.0
DEFAULT_ACC_BARRIER_EPS = 1.0e-3


DEFAULT_ACC_TOLERANCE = 0.005
DEFAULT_STAB_TOLERANCE = 1.2
DEFAULT_STAB_FLOOR = 1.0e-2


DEFAULT_S = 1.0
DEFAULT_PRIORITY1_PENALTY = 100.0
DEFAULT_PRIORITY1_SCALE = 200.0
DEFAULT_PRIORITY2_PENALTY = 50.0
DEFAULT_PRIORITY2_SCALE = 100.0


@dataclass
class BaselineCostStats:
    """Metrics and cost normalizers measured from the all-maximum baseline."""
    total_bits_sum: int = 0
    total_fusion_count: int = 0
    avg_k: float = DEFAULT_BASELINE_AVG_K
    loss_mean: float = 0.0
    loss_std: float = 0.0
    metric1_mean: float = 0.0
    metric2_mean: float = 0.0
    metric1_std: float = 0.0
    metric2_std: float = 0.0


    typical_bits_drop: float = 1.0
    typical_fusion_count: float = 1.0
    typical_k_drop: float = 1.0


@dataclass
class RewardWeights:
    """Weights, thresholds, and bounded budgets used by reward designs."""
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


    near_miss_tier_cap: float = 35.0
    near_miss_tier_floor: float = 15.0
    near_miss_band: float = 1.0


    acc_barrier_enabled: bool = DEFAULT_ACC_BARRIER_ENABLED
    acc_barrier_sat_scale: float = DEFAULT_ACC_BARRIER_SAT_SCALE
    acc_barrier_margin_ref: float = DEFAULT_ACC_BARRIER_MARGIN_REF
    acc_barrier_vio_scale: float = DEFAULT_ACC_BARRIER_VIO_SCALE
    acc_barrier_floor: float = DEFAULT_ACC_BARRIER_FLOOR
    acc_barrier_eps: float = DEFAULT_ACC_BARRIER_EPS


    fusion_saturation_tau: float = 0.0


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
    """Bind baseline metric denominators while keeping production defaults."""
    return RewardWeights(
        baseline_metric1=float(getattr(baseline, "metric1_mean", 0.0) or 0.0),
        baseline_metric2=float(getattr(baseline, "metric2_mean", 0.0) or 0.0),
    )


@dataclass
class EpisodeMetrics:
    """Precision and cross-trial stability metrics for one episode."""
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
    """Complete scalar and diagnostic output from ``compute_reward``."""
    reward: float
    priority: int
    invalid: bool

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


    loss_ok: bool = True


    near_miss: bool = False
    acc_worst_deficit_norm: float = 0.0


    acc_barrier_sat: float = 0.0
    acc_barrier_vio: float = 0.0


    stab_barrier: float = 0.0
    worst_signed_margin: float = 0.0

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


    cost_rank_score: float = 0.0
    cost_rank_fusion: float = 0.0
    cost_rank_truncation: float = 0.0
    cost_rank_bits: float = 0.0
    pareto_event_kind: str = ""
    pareto_action_hash: str = ""
    pareto_frontier_removed: int = 0

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

    # Accuracy and stability violations never receive resource-saving credit.
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
        val = min(0.0, raw)
    else:
        b0 = sat * (math.log(eps) - log_ref)
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


        return float(weights.cont_barrier_violation_slope) * m
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


        return float(clip_min), float(-weights.cont_barrier_violation_scale), 0.0
    acc_vals = [stage1_log_barrier(m, weights) for m in acc_margins]
    stab_vals = [stage1_log_barrier(m, weights) for m in std_margins]
    acc_b = (sum(acc_vals) / len(acc_vals)) if acc_vals else 0.0
    stab_b = (sum(stab_vals) / len(stab_vals)) if stab_vals else 0.0
    barrier_raw = float(weights.cont_w_acc) * acc_b + float(weights.cont_w_stab) * stab_b
    cost_frac = float(np.clip(
        float(effective_cost_score) / max(float(weights.p3_cost_budget), 1e-8), 0.0, 1.0,
    ))


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
    """Resolve the primary metric for single-metric threshold callers."""
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
    resort when baseline metric data is unavailable.
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
            constraint_policy="bootstrap_5x3_v1",
            q_precision=float(q_precision),
            q_stability=float(q_stability),
            precision_signal=float(precision_signal),
            stability_signal=float(stability_signal),
            **resource_diagnostics,
            **probabilities,
        )

    m1 = _safe_float(metrics.metric1_mean, 0.0)
    m2 = _safe_float(metrics.metric2_mean, 0.0)


    loss_std_raw = float(metrics.loss_std)
    m1_std_raw = float(getattr(metrics, "metric1_std", 0.0))
    m2_std_raw = float(getattr(metrics, "metric2_std", 0.0))
    loss_std = loss_std_raw if math.isfinite(loss_std_raw) else float("inf")
    m1_std = m1_std_raw if math.isfinite(m1_std_raw) else float("inf")
    m2_std = m2_std_raw if math.isfinite(m2_std_raw) else float("inf")


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


        margin_acc = 0.0


    combined_acc_violation = max(acc_violation_m1, acc_violation_m2)


    _deficits = []
    if m1_active:
        _deficits.append(acc_violation_m1 / denom_m1)
    if m2_active:
        _deficits.append(acc_violation_m2 / denom_m2)
    acc_worst_deficit_norm = max(_deficits) if _deficits else 0.0


    _signed_margins = []
    if m1_active:
        _signed_margins.append(margin_m1)
    if m2_active:
        _signed_margins.append(margin_m2)


    worst_signed_margin = min(_signed_margins) if _signed_margins else 0.0


    opt_total_bits = _safe_float(getattr(opt_signals, "total_bits_sum", 0), 0.0)
    fusion_count = _safe_float(getattr(opt_signals, "total_fusion_count", 0), 0.0)

    bits_gain = float(baseline.total_bits_sum) - opt_total_bits
    fusion_gain = fusion_count - float(baseline.total_fusion_count)
    k_gain = float(baseline.avg_k) - float(action_avg_k)

    typical_bits = max(abs(float(baseline.typical_bits_drop)), 1.0)
    typical_fusion = max(abs(float(baseline.typical_fusion_count)), 1.0)
    typical_k = max(abs(float(baseline.typical_k_drop)), 1.0)

    bits_norm = bits_gain / typical_bits
    fusion_norm = fusion_gain / typical_fusion
    k_norm = k_gain / typical_k

    cost_score_raw = 0.0


    def _stab_threshold(baseline_std: float) -> float:


        return max(
            float(baseline_std) * float(weights.stab_tolerance),
            float(weights.stab_floor),
        )

    stab_thr_m1 = _stab_threshold(baseline.metric1_std)
    stab_thr_m2 = _stab_threshold(baseline.metric2_std)
    if stab_threshold is not None:


        stab_thr_loss = float(stab_threshold)
    else:
        stab_thr_loss = _stab_threshold(baseline.loss_std)


    def _channel_excess(observed: float, threshold: float, denom: float) -> tuple:
        if not math.isfinite(observed):
            return 1.0, 1.0
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


    metric_ok = (combined_acc_violation == 0.0) and not invalid
    stab_ok = (combined_stab_excess == 0.0)


    if invalid or combined_acc_violation > 0:
        priority = 1
    elif combined_stab_excess > 0:
        priority = 2
    else:
        priority = 3


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


    _clip_min = float(weights.reward_clip_min)
    _clip_max = float(weights.reward_clip_max)
    acc_barrier_sat = 0.0
    acc_barrier_vio = 0.0
    near_miss = False
    stab_barrier = 0.0


    barrier_on = bool(getattr(weights, "acc_barrier_enabled", False)) and not invalid
    if barrier_on:


        _barrier = accuracy_margin_barrier(worst_signed_margin, weights)
        if worst_signed_margin >= 0.0:
            acc_barrier_sat = _barrier
        else:
            acc_barrier_vio = _barrier
        if metric_ok and stab_ok:
            shaping_raw = float(acc_barrier_sat) + float(effective_cost_score)
            shaping_clipped = float(np.clip(shaping_raw, _clip_min, _clip_max))
        elif metric_ok and not stab_ok:
            shaping_raw = float(acc_barrier_sat) + float(effective_stab_penalty)
            shaping_clipped = float(np.clip(shaping_raw, _clip_min, _clip_max))
        else:


            shaping_raw = float(acc_barrier_vio)
            shaping_clipped = float(
                np.clip(shaping_raw, float(weights.acc_barrier_floor), 0.0)
            )
    else:


        if metric_ok and stab_ok:
            shaping_raw = float(effective_p3_margin) + float(effective_cost_score)
        else:
            shaping_raw = float(margin_acc) + invalid_term + effective_stab_penalty
        shaping_clipped = float(np.clip(shaping_raw, _clip_min, _clip_max))


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


        near_miss = True
        cap = float(weights.near_miss_tier_cap)
        floor = float(weights.near_miss_tier_floor)
        frac = acc_worst_deficit_norm / float(weights.near_miss_band)
        tier_bonus += cap - (cap - floor) * float(np.clip(frac, 0.0, 1.0))

    total = float(shaping_clipped + tier_bonus)


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

        r_bits=r_bits,
        r_fusion=r_fusion,
        r_k=r_k,
        r_invalid=float(invalid_term),
        bits_drop=float(bits_gain),
        k_drop=float(k_gain),
        fusion_count=float(fusion_count),
        acc_violation=float(combined_acc_violation),
        stab_violation=float(combined_stab_excess),
    )
