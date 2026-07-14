"""Twelve-step robust PPO training for the Stage-2 layerwise action space."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import math
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from .candidate_store import CandidateStore, CandidateTrialEvidence, sha256_json
from .statistical_constraints import (
    TrialSeries,
    assess_candidate,
)


_PROBABILITY_FIELDS = (
    "loss_precision_probability",
    "metric1_precision_probability",
    "metric2_precision_probability",
    "loss_stability_probability",
    "metric1_stability_probability",
    "metric2_stability_probability",
)
_DECISION_GRANULARITIES = frozenset(("layer", "block"))
_REWARD_DESIGNS = frozenset((
    "robust_constrained", "stage1_aligned", "continuous", "tiered",
))


def normalize_decision_granularity(value: Any) -> str:
    normalized = str(value or "block").strip().lower()
    if normalized not in _DECISION_GRANULARITIES:
        raise ValueError(
            "decision_granularity must be 'layer' or 'block', "
            f"got {value!r}"
        )
    return normalized


def normalize_reward_design(value: Any) -> str:
    normalized = str(value or "stage1_aligned").strip().lower()
    if normalized not in _REWARD_DESIGNS:
        allowed = ", ".join(sorted(_REWARD_DESIGNS))
        raise ValueError(f"reward_design must be one of {allowed}; got {value!r}")
    return normalized


def apply_public_stage2_decision_config(evaluator: Any, config: Any) -> Any:
    """Apply and validate the public evaluator fields used for dispatch."""
    granularity = getattr(evaluator, "blb_v3_decision_granularity", None)
    reward_design = getattr(evaluator, "blb_v3_reward_design", None)
    config.decision_granularity = normalize_decision_granularity(
        config.decision_granularity if granularity in (None, "") else granularity
    )
    config.reward_design = normalize_reward_design(
        config.reward_design if reward_design in (None, "") else reward_design
    )
    return config


def resolve_decision_path(
        *,
        fusion_count_action: bool,
        decision_granularity: str,
        reward_design: str,
        ) -> str:
    """Validate Stage-2 decision granularity and return the training path."""
    granularity = normalize_decision_granularity(decision_granularity)
    normalized_reward = normalize_reward_design(reward_design)
    robust = normalized_reward == "robust_constrained"
    if robust and granularity != "layer":
        raise ValueError(
            "robust_constrained Stage-2 training requires decision_granularity='layer'"
        )
    if granularity == "layer" and not robust:
        raise ValueError(
            "decision_granularity='layer' requires reward_design='robust_constrained'"
        )
    if granularity == "layer" and not bool(fusion_count_action):
        raise ValueError(
            "layer decision granularity requires the fusion-count action map"
        )
    return "layerwise" if granularity == "layer" else "block"


def initialize_layerwise_policy(policy: Any) -> None:
    """Install the accepted decoded-value priors on all six slot heads."""
    from .layerwise_action import K_LEVELS

    k_probabilities = {
        13: 0.50,
        12: 0.20,
        11: 0.12,
        10: 0.08,
        9: 0.06,
        8: 0.04,
    }
    policy.set_initial_slot_probabilities(
        [{0: 0.60, 1: 0.40}] + [dict(k_probabilities) for _ in range(5)],
        [(0, 1)] + [tuple(K_LEVELS) for _ in range(5)],
    )


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _finite(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _assessment_probabilities(assessment: Any) -> tuple[float, ...]:
    if assessment is None:
        raise ValueError("strict ranking requires an assessment")
    probabilities = tuple(
        _finite(_field(assessment, name), name=name)
        for name in _PROBABILITY_FIELDS
    )
    if any(value < 0.0 or value > 1.0 for value in probabilities):
        raise ValueError("assessment probabilities must be in [0, 1]")
    return probabilities


def strict_rank_key(candidate: Any) -> tuple[float, ...]:
    """Ascending sort key for robust-feasible layerwise candidates."""
    assessment = _field(candidate, "assessment")
    metrics = _field(candidate, "metrics")
    probabilities = _assessment_probabilities(assessment)
    return (
        -_finite(_field(candidate, "variable_cost"), name="variable_cost"),
        -min(probabilities),
        _finite(_field(metrics, "loss_mean"), name="loss_mean"),
        -_finite(_field(metrics, "metric1_mean"), name="metric1_mean"),
        -_finite(_field(metrics, "metric2_mean"), name="metric2_mean"),
    )


def _strict_best_snapshot(
        accepted_candidates: Mapping[str, Mapping[str, Any]],
        ) -> Optional[dict[str, Any]]:
    """Return a detached snapshot of the current feasible frontier winner."""
    if not accepted_candidates:
        return None
    best = min(accepted_candidates.values(), key=strict_rank_key)
    promotion_trials = best.get("promotion_trials")
    promotion_evidence = None
    if isinstance(promotion_trials, TrialSeries):
        promotion_evidence = {
            "status": "promoted",
            "trial_count": len(promotion_trials.loss),
            "trials": {
                "loss": list(promotion_trials.loss),
                "metric1": list(promotion_trials.metric1),
                "metric2": list(promotion_trials.metric2),
                "seeds": list(promotion_trials.seeds),
            },
        }
    return {
        "rank_key": list(strict_rank_key(best)),
        "action_matrix": [list(row) for row in best["action_matrix"]],
        "full_vector": list(best["full_vector"]),
        "assessment": _to_plain_mapping(best["assessment"]),
        "metrics": dict(best["metrics"]),
        "variable_cost": float(best["variable_cost"]),
        "reward": (
            None if best.get("reward") is None else float(best["reward"])
        ),
        "boosted_overrides": copy.deepcopy(best["boosted_overrides"]),
        "promotion_evidence": promotion_evidence,
    }


def normalized_entropy_snapshot(
        entropy_per_slot: Any,
        slot_masks: Any,
        per_slot_num_levels: Any,
        ) -> dict[str, float | int | None]:
    """Normalize active per-slot entropy and split Block4 from K slots."""
    entropy = np.asarray(entropy_per_slot, dtype=np.float64)
    masks = np.asarray(slot_masks, dtype=bool)
    levels = np.asarray(per_slot_num_levels, dtype=np.int64)
    if entropy.ndim != 2 or entropy.shape != masks.shape or entropy.shape != levels.shape:
        raise ValueError("entropy, masks, and levels must be aligned 2-D arrays")
    if entropy.shape[1] != 6:
        raise ValueError("layerwise entropy requires exactly six slots")

    active = masks & (levels > 1)
    normalized = np.zeros_like(entropy, dtype=np.float64)
    normalized[active] = entropy[active] / np.log(levels[active].astype(np.float64))
    block4_values = normalized[:, 0][active[:, 0]]
    k_values = normalized[:, 1:][active[:, 1:]]
    return {
        "block4": (
            float(np.mean(block4_values)) if block4_values.size else None
        ),
        "k": float(np.mean(k_values)) if k_values.size else None,
        "block4_slot_count": int(block4_values.size),
        "k_slot_count": int(k_values.size),
    }


def redistribute_layerwise_rewards(
        *,
        terminal_reward: float,
        priority: int,
        variable_cost: float,
        layer_cost_rewards: Sequence[float],
        ) -> tuple[float, ...]:
    """Move P3 cost credit to its twelve source layer transitions.

    The returned rewards always sum to ``terminal_reward``. Precision/stability
    failures retain terminal-only credit, so cost cannot leak into P1 or P2.
    """
    reward = _finite(terminal_reward, name="terminal_reward")
    cost = _finite(variable_cost, name="variable_cost")
    layer_costs = tuple(
        _finite(value, name=f"layer_cost_rewards[{index}]")
        for index, value in enumerate(layer_cost_rewards)
    )
    if len(layer_costs) != 12:
        raise ValueError(
            f"layer_cost_rewards must contain 12 values, got {len(layer_costs)}"
        )
    if cost < 0.0 or cost > 1.0:
        raise ValueError(f"variable_cost must be in [0, 1], got {cost}")
    if not math.isclose(sum(layer_costs), cost, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(
            "layer_cost_rewards sum must equal variable_cost: "
            f"{sum(layer_costs)} != {cost}"
        )
    if int(priority) != 3:
        return (0.0,) * 11 + (reward,)

    redistributed = list(layer_costs)
    redistributed[-1] += reward - cost
    if not math.isclose(sum(redistributed), reward, rel_tol=0.0, abs_tol=1.0e-9):
        raise RuntimeError("layerwise reward redistribution changed episode return")
    return tuple(float(value) for value in redistributed)


@dataclass(frozen=True)
class LayerwiseConvergenceState:
    completed_episodes: int
    block4_entropy: Optional[float]
    k_entropy: Optional[float]
    stall_update_windows: int
    converged: bool
    extension_required: bool
    best_robust_feasible_cost: Optional[float]


class LayerwiseConvergenceTracker:
    """Track entropy and frontier-stall gates across PPO update windows."""

    def __init__(self) -> None:
        self._best_cost: Optional[float] = None
        self._current_frontier_cost: Optional[float] = None
        self._stall_windows = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "best_robust_feasible_cost": self._best_cost,
            "current_robust_feasible_cost": self._current_frontier_cost,
            "stall_update_windows": int(self._stall_windows),
        }

    def load_state_dict(self, state: Optional[Mapping[str, Any]]) -> None:
        if not isinstance(state, Mapping):
            return
        best_cost = state.get("best_robust_feasible_cost")
        self._best_cost = (
            None if best_cost is None else _finite(best_cost, name="best_robust_feasible_cost")
        )
        current_cost = state.get("current_robust_feasible_cost")
        self._current_frontier_cost = (
            None if current_cost is None
            else _finite(current_cost, name="current_robust_feasible_cost")
        )
        stall_windows = int(state.get("stall_update_windows", 0))
        if stall_windows < 0:
            raise ValueError("stall_update_windows must be nonnegative")
        self._stall_windows = stall_windows

    def reconcile_frontier(self, robust_feasible_cost: Optional[float]) -> None:
        """Align restored convergence state with the revalidated frontier."""
        if robust_feasible_cost is None:
            self._current_frontier_cost = None
            self._stall_windows = 0
            return
        cost = _finite(robust_feasible_cost, name="robust_feasible_cost")
        if (
                self._current_frontier_cost is None
                or not math.isclose(
                    cost, self._current_frontier_cost, rel_tol=0.0, abs_tol=1.0e-12,
                )
        ):
            self._best_cost = cost
            self._stall_windows = 0
        self._current_frontier_cost = cost

    def observe_update(
            self,
            *,
            completed_episodes: int,
            block4_entropy: Optional[float],
            k_entropy: Optional[float],
            robust_feasible_cost: Optional[float],
            ) -> LayerwiseConvergenceState:
        episodes = int(completed_episodes)
        if robust_feasible_cost is None:
            self._current_frontier_cost = None
            self._stall_windows = 0
        else:
            cost = _finite(robust_feasible_cost, name="robust_feasible_cost")
            frontier_restarted = self._current_frontier_cost is None
            frontier_retracted = bool(
                self._current_frontier_cost is not None
                and cost < self._current_frontier_cost - 1.0e-12
            )
            if frontier_restarted or frontier_retracted:
                self._best_cost = cost
                self._stall_windows = 0
            elif self._best_cost is None or cost > self._best_cost + 1.0e-12:
                self._best_cost = cost
                self._stall_windows = 0
            else:
                self._stall_windows += 1
            self._current_frontier_cost = cost

        b4 = None if block4_entropy is None else _finite(
            block4_entropy, name="block4_entropy",
        )
        k_value = None if k_entropy is None else _finite(k_entropy, name="k_entropy")
        converged = bool(
            episodes >= 30_000
            and b4 is not None and b4 < 0.1
            and k_value is not None and k_value < 0.1
            and robust_feasible_cost is not None
            and self._best_cost is not None
            and self._stall_windows >= 100
        )
        return LayerwiseConvergenceState(
            completed_episodes=episodes,
            block4_entropy=b4,
            k_entropy=k_value,
            stall_update_windows=int(self._stall_windows),
            best_robust_feasible_cost=self._best_cost,
            converged=converged,
            extension_required=bool(episodes >= 60_000 and not converged),
        )


@dataclass(frozen=True)
class PromotionResult:
    status: str
    trial_count: int
    fresh_trial_count: int
    evidence: Optional[CandidateTrialEvidence]
    assessment: Optional[Any]
    metrics: Optional[Mapping[str, float]]


@dataclass(frozen=True)
class LayerwiseEpisodeRecord:
    episode_index: int
    reward: float
    priority: int
    action_matrix: tuple[tuple[int, ...], ...]
    pending_full_vector: tuple[int, ...]
    variable_cost: float
    raw_trials: Optional[TrialSeries]
    pooled_trials: Optional[TrialSeries]
    fresh_trial_count: int
    pooled_trial_count: int
    reward_evidence: str
    ranking_evidence: str
    fresh_assessment: Optional[Mapping[str, Any]]
    assessment: Optional[Any]
    metrics: Mapping[str, float]
    pooled_metrics: Optional[Mapping[str, float]]
    promoted_trial_count: int
    promotion_status: str
    invalid_steps: int
    step_count: int
    block4_entropy: Optional[float]
    k_entropy: Optional[float]
    stall_update_windows: int
    converged: bool
    extension_required: bool
    best_robust_feasible_cost: Optional[float]


def _to_plain_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def _trial_series_from_info(
        info: Any,
        *,
        required: bool = False,
        expected_count: Optional[int] = None,
        context: str = "terminal",
        ) -> Optional[TrialSeries]:
    if not isinstance(info, Mapping):
        if required:
            raise RuntimeError(f"{context} info is required for raw evidence")
        return None
    raw = info.get("statistical_trials")
    if not isinstance(raw, Mapping):
        if required:
            raise RuntimeError(f"{context} statistical_trials are required")
        return None
    trials = TrialSeries(
        loss=raw.get("loss", ()),
        metric1=raw.get("metric1", ()),
        metric2=raw.get("metric2", ()),
        seeds=raw.get("seeds", ()),
    )
    if not trials.seeds or len(trials.seeds) != len(trials.loss):
        raise ValueError(f"{context} requires nonempty aligned trial seeds")
    if expected_count is not None and len(trials.loss) != int(expected_count):
        raise ValueError(
            f"{context} expected exactly {int(expected_count)} raw trials; "
            f"received {len(trials.loss)}"
        )
    return trials


def _metrics_from_trials(trials: TrialSeries) -> dict[str, float]:
    loss = np.asarray(trials.loss, dtype=np.float64)
    metric1 = np.asarray(trials.metric1, dtype=np.float64)
    metric2 = np.asarray(trials.metric2, dtype=np.float64)
    return {
        "loss_mean": float(np.mean(loss)),
        "loss_std": float(np.std(loss, ddof=1)) if loss.size > 1 else 0.0,
        "metric1_mean": float(np.mean(metric1)),
        "metric1_std": float(np.std(metric1, ddof=1)) if metric1.size > 1 else 0.0,
        "metric2_mean": float(np.mean(metric2)),
        "metric2_std": float(np.std(metric2, ddof=1)) if metric2.size > 1 else 0.0,
    }


def _assessment_passes(assessment: Any, threshold: float) -> bool:
    try:
        return min(_assessment_probabilities(assessment)) >= float(threshold)
    except (TypeError, ValueError):
        return False


def _append_promotion_status(
        store: CandidateStore,
        action_indices: Sequence[int],
        identity_context: Mapping[str, Any],
        *,
        status: str,
        metadata: Optional[Mapping[str, Any]] = None,
        ) -> None:
    store.append({
        "record_type": "candidate_promotion_status_v1",
        "action_indices": list(action_indices),
        "effective_action_indices": list(action_indices),
        "identity_context": dict(identity_context),
        "fidelity": "F4" if status == "promoted" else "F1",
        "valid": status == "promoted",
        "promotion_status": str(status),
        "promotion_metadata": dict(metadata or {}),
    })


def _serialize_boosted_overrides(overrides: Mapping[Any, Any]) -> list[dict[str, Any]]:
    rows = []
    for key, values in sorted(
            dict(overrides).items(),
            key=lambda item: (int(item[0][1]), int(item[0][0])),
    ):
        block_idx, layer_idx = key
        rows.append({
            "block_idx": int(block_idx),
            "layer_idx": int(layer_idx),
            "field_values": {
                str(name): int(value) for name, value in dict(values).items()
            },
        })
    return rows


def _deserialize_boosted_overrides(rows: Any) -> dict[tuple[int, int], dict[str, int]]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("boosted_overrides must be a sequence of rows")
    overrides: dict[tuple[int, int], dict[str, int]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("boosted_overrides rows must be mappings")
        values = row.get("field_values")
        if not isinstance(values, Mapping):
            raise ValueError("boosted_overrides field_values must be a mapping")
        key = (int(row["block_idx"]), int(row["layer_idx"]))
        overrides[key] = {str(name): int(value) for name, value in values.items()}
    return overrides


def restore_promoted_candidates(
        *,
        candidate_store: CandidateStore,
        identity_context: Mapping[str, Any],
        statistical_reference: Any,
        assess_candidate_fn: Callable[..., Any] = assess_candidate,
        promotion_probability: float = 0.80,
        assessment_trial_limit: int = 25,
        ) -> dict[str, dict[str, Any]]:
    """Rebuild the current promoted frontier from append-only raw evidence."""
    latest_status: dict[
        str, tuple[str, tuple[int, ...], dict[str, Any]]
    ] = {}
    wanted_context_hash = sha256_json(dict(identity_context))
    for record in candidate_store.iter_active_records():
        if record.get("record_type") != "candidate_promotion_status_v1":
            continue
        if str(record.get("identity_context_hash", "")) != wanted_context_hash:
            continue
        key = str(record.get("candidate_key", ""))
        if not key:
            continue
        latest_status[key] = (
            str(record.get("promotion_status", "")),
            tuple(int(value) for value in record.get("action_indices", ())),
            dict(record.get("promotion_metadata") or {}),
        )

    restored: dict[str, dict[str, Any]] = {}
    for key, (status, action_indices, promotion_metadata) in latest_status.items():
        if status != "promoted" or not action_indices:
            continue
        evidence = candidate_store.trial_evidence_for_action(
            action_indices, identity_context,
            max_trials=int(assessment_trial_limit),
        )
        if evidence is None or not evidence.promoted:
            continue
        metadata: dict[str, Any] = {}
        for group in evidence.groups:
            for name in (
                    "action_matrix", "variable_cost", "episode_reward",
                    "assessment_bootstrap_seed", "boosted_overrides",
            ):
                if name in group:
                    metadata[name] = group[name]
        for name in (
                "action_matrix", "variable_cost", "episode_reward",
                "assessment_bootstrap_seed", "boosted_overrides",
        ):
            if name in promotion_metadata:
                metadata[name] = promotion_metadata[name]
        if not all(name in metadata for name in (
                "action_matrix", "variable_cost", "boosted_overrides",
        )):
            continue
        assessment = assess_candidate_fn(
            evidence.trials,
            statistical_reference,
            gate_probability=float(promotion_probability),
            bootstrap_seed=int(metadata.get("assessment_bootstrap_seed", 0)),
        )
        if not _assessment_passes(assessment, promotion_probability):
            continue
        action_matrix = tuple(
            tuple(int(value) for value in row)
            for row in metadata["action_matrix"]
        )
        if len(action_matrix) != 12 or any(len(row) != 6 for row in action_matrix):
            raise ValueError("persisted layerwise action_matrix must be 12x6")
        reward = metadata.get("episode_reward")
        restored[key] = {
            "variable_cost": _finite(metadata["variable_cost"], name="variable_cost"),
            "assessment": assessment,
            "metrics": _metrics_from_trials(evidence.trials),
            "action_matrix": action_matrix,
            "full_vector": tuple(action_indices),
            "boosted_overrides": _deserialize_boosted_overrides(
                metadata["boosted_overrides"]
            ),
            "reward": None if reward is None else _finite(reward, name="episode_reward"),
            "promotion_trials": evidence.trials,
        }
    return restored


def _promotion_probe_seed(
        candidate_key_value: str,
        bootstrap_seed: int,
        existing_trial_count: int,
        existing_seeds: Sequence[int],
        fresh_trial_count: int,
        ) -> tuple[int, tuple[int, ...]]:
    from .seed_utils import (
        derive_layerwise_promotion_probe_seed,
        derive_probe_trial_seed,
    )

    material = (
        f"layerwise-promotion:{candidate_key_value}:{int(bootstrap_seed)}:"
        f"{int(existing_trial_count)}"
    ).encode("utf-8")
    seed_material = (
        int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
        & 0x7FFFFFFFFFFFFFFF
    )
    occupied = {int(seed) for seed in existing_seeds}
    # Attempt domains are pairwise disjoint. With N occupied seeds, N+1
    # attempts guarantee at least one trial set has no overlap.
    for attempt_idx in range(len(occupied) + 1):
        probe_seed = derive_layerwise_promotion_probe_seed(
            seed_material,
            attempt_idx,
            trial_count=int(fresh_trial_count),
        )
        predicted = tuple(
            derive_probe_trial_seed(probe_seed, trial_idx)
            for trial_idx in range(int(fresh_trial_count))
        )
        if occupied.isdisjoint(predicted):
            return probe_seed, predicted
    raise RuntimeError("could not allocate disjoint layerwise promotion trial seeds")


def promote_candidate_if_eligible(
        *,
        env: Any,
        candidate_store: CandidateStore,
        action_indices: Sequence[int],
        identity_context: Mapping[str, Any],
        action_matrix: Sequence[Sequence[int]],
        assessment: Any,
        priority: int,
        variable_cost: float,
        frontier_cost: Optional[float],
        boosted_overrides: Mapping[Any, Any],
        bootstrap_seed: int,
        episode_reward: Optional[float] = None,
        assess_candidate_fn: Callable[..., Any] = assess_candidate,
        promotion_probability: float = 0.80,
        target_trial_count: int = 25,
        ) -> PromotionResult:
    """Promote one robust frontier improvement using fresh real probes."""
    evidence = candidate_store.trial_evidence_for_action(
        action_indices, identity_context,
        max_trials=int(target_trial_count),
    )
    trial_count = candidate_store.trial_count_for_action(
        action_indices, identity_context,
    )
    pooled_metrics = _metrics_from_trials(evidence.trials) if evidence is not None else None
    if int(priority) != 3:
        return PromotionResult(
            "priority_not_p3", trial_count, 0, evidence, assessment, pooled_metrics,
        )
    if not _assessment_passes(assessment, promotion_probability):
        return PromotionResult(
            "promotion_probability_below_gate", trial_count, 0,
            evidence, assessment, pooled_metrics,
        )
    cost = _finite(variable_cost, name="variable_cost")
    if frontier_cost is not None and cost <= float(frontier_cost) + 1.0e-12:
        return PromotionResult(
            "not_frontier_improvement", trial_count, 0,
            evidence, assessment, pooled_metrics,
        )
    if evidence is None:
        raise ValueError("promotion requires existing raw candidate evidence")
    if evidence.promoted:
        return PromotionResult(
            "already_promoted", trial_count, 0, evidence, assessment, pooled_metrics,
        )
    target = int(target_trial_count)
    if target <= 0:
        raise ValueError("target_trial_count must be positive")
    pending_reassessment = bool(
        evidence.promotion_attempted
        and not evidence.promotion_status
        and trial_count >= target
    )
    if evidence.promotion_attempted and not pending_reassessment:
        return PromotionResult(
            "promotion_already_attempted", trial_count, 0,
            evidence, assessment, pooled_metrics,
        )

    fresh_count = max(0, target - trial_count)
    status_metadata = {
        "existing_trial_count": int(trial_count),
        "requested_fresh_trial_count": int(fresh_count),
        "variable_cost": float(cost),
        "assessment_bootstrap_seed": int(bootstrap_seed),
        "action_matrix": [list(map(int, row)) for row in action_matrix],
        "boosted_overrides": _serialize_boosted_overrides(boosted_overrides),
    }
    if episode_reward is not None:
        status_metadata["episode_reward"] = float(episode_reward)
    promotion_probe_seed: Optional[int] = None
    predicted_trial_seeds: tuple[int, ...] = ()
    if fresh_count:
        promotion_probe_seed, predicted_trial_seeds = _promotion_probe_seed(
            evidence.candidate_key,
            bootstrap_seed,
            trial_count,
            evidence.trials.seeds,
            fresh_count,
        )
    try:
        if fresh_count:
            previous_probe_seed = getattr(env.base, "probe_noise_seed", None)
            env.base.probe_noise_seed = promotion_probe_seed
            try:
                prepared = env.base.prepare_action_for_terminal_probe(
                    list(action_indices),
                    external_cost_score=cost,
                    external_cost_rank=cost,
                    boosted_overrides=copy.deepcopy(dict(boosted_overrides)),
                )
                evaluated = env.base.evaluate_prepared_terminal_batch(
                    [prepared],
                    num_trials_per_action=fresh_count,
                    validation_required=True,
                )
            finally:
                env.base.probe_noise_seed = previous_probe_seed
            if len(evaluated) != 1:
                raise RuntimeError(
                    f"promotion expected one terminal result, received {len(evaluated)}"
                )
            terminal_info = evaluated[0][3]
            if not isinstance(terminal_info, Mapping) or bool(terminal_info.get("invalid", False)):
                raise RuntimeError("promotion terminal evaluation was invalid")
            fresh_trials = _trial_series_from_info(
                terminal_info,
                required=True,
                expected_count=fresh_count,
                context="promotion terminal",
            )
            if tuple(fresh_trials.seeds) != predicted_trial_seeds:
                raise RuntimeError(
                    "promotion terminal trial seeds did not match the predicted fresh set"
                )
            candidate_store.append_trial_group(
                action_indices,
                fresh_trials,
                {
                    "identity_context": dict(identity_context),
                    "fidelity": "F4",
                    "variable_cost": float(cost),
                    "action_matrix": [list(map(int, row)) for row in action_matrix],
                    "boosted_overrides_hash": sha256_json(boosted_overrides),
                    "boosted_overrides": _serialize_boosted_overrides(boosted_overrides),
                    "boosted_overrides_provenance": "layerwise_env",
                    "assessment_bootstrap_seed": int(bootstrap_seed),
                    "promotion_marker": "fresh_top_up",
                    "promotion_status": "pending_reassessment",
                },
            )
        evidence = candidate_store.trial_evidence_for_action(
            action_indices, identity_context,
            max_trials=target,
        )
        trial_count = candidate_store.trial_count_for_action(
            action_indices, identity_context,
        )
        if evidence is None or trial_count < target:
            raise RuntimeError(
                f"promotion evidence count {trial_count} "
                f"is below target {target}"
            )
        pooled_assessment = assess_candidate_fn(
            evidence.trials,
            env.base.statistical_reference,
            gate_probability=float(promotion_probability),
            bootstrap_seed=int(bootstrap_seed),
        )
        promotion_status = (
            "promoted" if _assessment_passes(pooled_assessment, promotion_probability)
            else "failed_probability_gate"
        )
    except Exception as exc:
        promotion_status = "failed_evaluation"
        pooled_assessment = assessment
        status_metadata["error"] = str(exc)

    _append_promotion_status(
        candidate_store,
        action_indices,
        identity_context,
        status=promotion_status,
        metadata=status_metadata,
    )
    evidence = candidate_store.trial_evidence_for_action(
        action_indices, identity_context,
        max_trials=target,
    )
    total_trial_count = candidate_store.trial_count_for_action(
        action_indices, identity_context,
    )
    return PromotionResult(
        status=promotion_status,
        trial_count=total_trial_count,
        fresh_trial_count=(fresh_count if evidence is not None and total_trial_count >= target else 0),
        evidence=evidence,
        assessment=pooled_assessment,
        metrics=(_metrics_from_trials(evidence.trials) if evidence is not None else pooled_metrics),
    )


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def _first_detached_scalar(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.reshape(-1)[0].detach().reshape(())
    return float(np.asarray(value).reshape(-1)[0])


def _policy_input(value: np.ndarray, device: Any) -> Any:
    try:
        import torch
    except ImportError:
        return value
    return torch.as_tensor(value, device=device)


def _cosine_entropy_coefficient(train_cfg: Any, completed_episodes: int) -> float:
    total = max(1, int(getattr(
        train_cfg,
        "planned_total_episodes",
        getattr(train_cfg, "total_episodes", 1),
    )))
    progress = float(completed_episodes) / float(total)
    plateau = min(1.0, max(0.0, float(
        getattr(train_cfg, "ent_coef_cosine_plateau", 0.25)
    )))
    decay_end = min(1.0, max(plateau, float(
        getattr(train_cfg, "ent_coef_cosine_decay_end", 1.0)
    )))
    start = float(getattr(train_cfg, "ent_coef_cosine_start", 0.05))
    end = float(getattr(train_cfg, "ent_coef_cosine_end", 0.001))
    lower_bound = float(getattr(train_cfg, "ent_coef_cosine_lower_bound", 0.012))
    if progress <= plateau:
        value = start
    elif progress >= decay_end:
        value = end
    else:
        phase = min(1.0, max(
            0.0,
            (progress - plateau) / max(1.0e-8, decay_end - plateau),
        ))
        value = end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * phase))
    return max(lower_bound, value)


def _current_policy_entropy(
        policy: Any,
        samples: Sequence[Mapping[str, np.ndarray]],
        device: Any,
        ) -> dict[str, float | int | None]:
    if not samples or not hasattr(policy, "evaluate_action"):
        return {
            "block4": None, "k": None,
            "block4_slot_count": 0, "k_slot_count": 0,
        }
    import torch

    states = torch.as_tensor(np.stack([row["state"] for row in samples]), device=device)
    actions = torch.as_tensor(np.stack([row["action"] for row in samples]), device=device)
    masks = torch.as_tensor(np.stack([row["slot_mask"] for row in samples]), device=device)
    levels = torch.as_tensor(np.stack([row["levels"] for row in samples]), device=device)
    with torch.no_grad():
        evaluated = policy.evaluate_action(
            states,
            actions,
            masks,
            levels,
            return_per_slot_entropy=True,
        )
    entropy_per_slot = _as_numpy(evaluated[3])
    return normalized_entropy_snapshot(
        entropy_per_slot,
        _as_numpy(masks),
        _as_numpy(levels),
    )


def train_layerwise(
        *,
        env: Any,
        policy: Any,
        train_cfg: Any,
        candidate_store: CandidateStore,
        identity_context: Optional[Mapping[str, Any]] = None,
        on_episode_end: Optional[Callable[[LayerwiseEpisodeRecord], None]] = None,
        on_ppo_update_end: Optional[
            Callable[[Mapping[str, Any], int, LayerwiseEpisodeRecord], None]
        ] = None,
        device: Any = None,
        optimizer: Any = None,
        rollout_buffer: Any = None,
        ppo_update_fn: Optional[Callable[..., Mapping[str, Any]]] = None,
        assess_candidate_fn: Callable[..., Any] = assess_candidate,
        step_adapter_fn: Optional[Callable[[Any, int, int], tuple[np.ndarray, np.ndarray]]] = None,
        ) -> dict[str, Any]:
    """Collect 12-step layerwise episodes and update the shared PPO policy."""
    if identity_context is None:
        raise ValueError("layerwise training requires a CandidateStore identity_context")
    if int(getattr(env, "horizon", 0)) != 12 or int(getattr(env, "max_step_dim", 0)) != 6:
        raise ValueError("layerwise training requires horizon=12 and max_step_dim=6")
    if device is None:
        try:
            device = next(policy.parameters()).device
        except (AttributeError, StopIteration, TypeError):
            device = None
    if optimizer is None:
        import torch

        optimizer = torch.optim.Adam(
            policy.parameters(), lr=float(getattr(train_cfg.ppo, "lr", 5.0e-5)),
        )
    if rollout_buffer is None:
        from .sequential_policy import SequentialRolloutBuffer

        rollout_buffer = SequentialRolloutBuffer()
    if ppo_update_fn is None:
        from .sequential_policy import sequential_ppo_update

        ppo_update_fn = sequential_ppo_update
    if step_adapter_fn is None:
        from .sequential_policy import step_to_mask_and_levels

        step_adapter_fn = step_to_mask_and_levels

    ppo_cfg = copy.copy(train_cfg.ppo)
    ppo_cfg.gamma = 1.0
    ppo_cfg.gae_lambda = 1.0
    update_window = max(1, int(getattr(train_cfg, "update_every_n_episodes", 120)))
    total_episodes = max(0, int(getattr(train_cfg, "total_episodes", 0)))
    absolute_start = int(getattr(train_cfg, "absolute_episode_start", 0))
    base_seed = getattr(train_cfg, "seed", None)
    expected_online_trials = int(
        getattr(train_cfg, "online_num_trials_per_step", 5)
    )
    if expected_online_trials <= 0:
        raise ValueError("online_num_trials_per_step must be positive")
    online_probability = float(
        getattr(train_cfg, "online_constraint_probability", 0.50)
    )
    promotion_probability = float(
        getattr(train_cfg, "promotion_constraint_probability", 0.80)
    )
    promotion_trials = int(getattr(train_cfg, "promotion_validation_trials", 25))
    if not 0.0 < online_probability <= promotion_probability <= 1.0:
        raise ValueError(
            "constraint probabilities must satisfy 0 < online <= promotion <= 1"
        )
    if promotion_trials < expected_online_trials:
        raise ValueError("promotion_validation_trials must cover the online trial group")
    policy.eval()

    records: list[LayerwiseEpisodeRecord] = []
    rewards: list[float] = []
    ppo_diagnostics: list[dict[str, Any]] = []
    accepted_candidates = restore_promoted_candidates(
        candidate_store=candidate_store,
        identity_context=identity_context,
        statistical_reference=env.base.statistical_reference,
        assess_candidate_fn=assess_candidate_fn,
        promotion_probability=promotion_probability,
        assessment_trial_limit=promotion_trials,
    )
    convergence_resume_state = getattr(train_cfg, "convergence_resume_state", None)
    convergence_resume_state = (
        dict(convergence_resume_state)
        if isinstance(convergence_resume_state, Mapping) else {}
    )
    convergence_tracker = LayerwiseConvergenceTracker()
    convergence_tracker.load_state_dict(convergence_resume_state)
    restored_frontier_cost = (
        max(row["variable_cost"] for row in accepted_candidates.values())
        if accepted_candidates else None
    )
    convergence_tracker.reconcile_frontier(restored_frontier_cost)
    restored_tracker_state = convergence_tracker.state_dict()
    restored_block4_entropy = convergence_resume_state.get("block4_entropy")
    restored_k_entropy = convergence_resume_state.get("k_entropy")
    restored_converged = bool(
        restored_frontier_cost is not None
        and absolute_start >= 30_000
        and restored_block4_entropy is not None
        and float(restored_block4_entropy) < 0.1
        and restored_k_entropy is not None
        and float(restored_k_entropy) < 0.1
        and int(restored_tracker_state["stall_update_windows"]) >= 100
        and convergence_resume_state.get("converged", False)
    )
    convergence_state = LayerwiseConvergenceState(
        completed_episodes=absolute_start,
        block4_entropy=(
            None if restored_block4_entropy is None
            else _finite(restored_block4_entropy, name="block4_entropy")
        ),
        k_entropy=(
            None if restored_k_entropy is None
            else _finite(restored_k_entropy, name="k_entropy")
        ),
        stall_update_windows=int(restored_tracker_state["stall_update_windows"]),
        best_robust_feasible_cost=restored_tracker_state["best_robust_feasible_cost"],
        converged=restored_converged,
        extension_required=bool(
            convergence_resume_state.get("extension_required", False)
            or (absolute_start >= 60_000 and not restored_converged)
        ),
    )
    entropy_samples: list[dict[str, np.ndarray]] = []

    for local_episode in range(total_episodes):
        absolute_episode = absolute_start + local_episode
        state = env.reset(
            seed=(None if base_seed is None else int(base_seed) + absolute_episode)
        )
        if base_seed is not None and hasattr(env, "base"):
            from .seed_utils import derive_layerwise_episode_probe_seed

            env.base.probe_noise_seed = derive_layerwise_episode_probe_seed(
                int(base_seed),
                absolute_episode,
                trial_count=expected_online_trials,
            )
        step_infos: list[Mapping[str, Any]] = []
        transition_indices: list[int] = []
        terminal_info: Optional[Mapping[str, Any]] = None
        episode_reward = 0.0
        for step_idx in range(12):
            spec = env.current_spec()
            slot_mask, levels = step_adapter_fn(spec, 6, 6)
            state_np = np.asarray(state, dtype=np.float32)
            actions_raw, log_prob_raw, value_raw = policy.sample_action(
                _policy_input(state_np[None, ...], device),
                _policy_input(slot_mask[None, ...], device),
                _policy_input(levels[None, ...], device),
                deterministic=False,
                baseline_prior_scale=0.0,
            )
            action = _as_numpy(actions_raw).reshape(-1).astype(np.int64)
            action[~slot_mask] = 0
            log_prob = _first_detached_scalar(log_prob_raw)
            value = _first_detached_scalar(value_raw)
            next_state, reward, done, info = env.step(action.tolist())
            expected_done = step_idx == 11
            if bool(done) != expected_done:
                raise RuntimeError(
                    f"layerwise episode termination mismatch at step {step_idx}: done={done}"
                )
            transition_index = rollout_buffer.add(
                state=state_np,
                action=action,
                slot_mask=slot_mask,
                per_slot_num_levels=levels,
                action_level_mask=None,
                log_prob=log_prob,
                value=value,
                reward=0.0,
                done=bool(done),
                baseline_prior_scale=0.0,
            )
            transition_indices.append(int(transition_index))
            entropy_samples.append({
                "state": state_np.copy(),
                "action": action.copy(),
                "slot_mask": slot_mask.copy(),
                "levels": levels.copy(),
            })
            step_infos.append(info if isinstance(info, Mapping) else {})
            state = next_state
            if expected_done:
                terminal_info = info
                episode_reward = float(reward)

        if terminal_info is None:
            raise RuntimeError("layerwise episode completed without terminal info")
        runtime_info = getattr(env, "runtime_terminal_info", None)
        runtime_info = runtime_info if isinstance(runtime_info, Mapping) else {}
        action_matrix = tuple(
            tuple(int(value) for value in row)
            for row in terminal_info.get("policy_actions", ())
        )
        if len(action_matrix) != 12 or any(len(row) != 6 for row in action_matrix):
            raise RuntimeError("layerwise terminal policy_actions must be a 12x6 matrix")
        full_vector = tuple(
            int(value) for value in terminal_info.get("pending_full_vector", ())
        )
        if not full_vector:
            raise RuntimeError("layerwise terminal pending_full_vector is required")
        variable_cost = _finite(
            _field(terminal_info.get("variable_cost", {}), "normalized"),
            name="variable_cost",
        )
        layer_cost_rewards = tuple(
            float(value) for value in _field(
                terminal_info.get("variable_cost", {}),
                "layer_cost_rewards",
                (),
            )
        )
        slot_cost_rewards = tuple(
            tuple(float(value) for value in row)
            for row in _field(
                terminal_info.get("variable_cost", {}),
                "slot_cost_rewards",
                (),
            )
        )
        if len(slot_cost_rewards) != 12 or any(
                len(row) != 6 for row in slot_cost_rewards
        ):
            raise RuntimeError("layerwise terminal slot_cost_rewards must be a 12x6 matrix")
        for layer_idx, (layer_cost, slot_costs) in enumerate(zip(
                layer_cost_rewards, slot_cost_rewards,
        )):
            if not math.isclose(
                    sum(slot_costs), layer_cost, rel_tol=0.0, abs_tol=1.0e-9,
            ):
                raise RuntimeError(
                    f"layer {layer_idx} slot cost sum does not match layer cost"
                )
        breakdown = runtime_info.get("reward_breakdown")
        priority = int(_field(breakdown, "priority", runtime_info.get("priority", 0)))
        invalid_terminal = bool(runtime_info.get("invalid", False))
        if invalid_terminal and not math.isclose(episode_reward, -5.0, abs_tol=1.0e-9):
            raise RuntimeError(
                f"invalid layerwise terminal reward must be -5, got {episode_reward}"
            )
        reward_priority = 0 if invalid_terminal else priority
        redistributed_rewards = redistribute_layerwise_rewards(
            terminal_reward=episode_reward,
            priority=reward_priority,
            variable_cost=variable_cost,
            layer_cost_rewards=layer_cost_rewards,
        )
        zero_slot_costs = ((0.0,) * 6,) * 12
        actor_slot_costs = (
            slot_cost_rewards if reward_priority == 3 else zero_slot_costs
        )
        actor_shared_return = (
            episode_reward - variable_cost
            if reward_priority == 3 else episode_reward
        )
        for transition_index, reward_delta, per_slot_cost in zip(
                transition_indices, redistributed_rewards, actor_slot_costs,
        ):
            rollout_buffer.add_reward_at(transition_index, reward_delta)
            rollout_buffer.set_actor_cost_at(transition_index, per_slot_cost)
            rollout_buffer.set_actor_shared_return_at(
                transition_index, actor_shared_return,
            )
        raw_trials = (
            None
            if invalid_terminal
            else _trial_series_from_info(
                runtime_info,
                required=True,
                expected_count=expected_online_trials,
                context="valid robust terminal",
            )
        )
        fresh_assessment = _to_plain_mapping(runtime_info.get("statistical_assessment"))
        metrics = _to_plain_mapping(runtime_info.get("metrics"))
        bootstrap_seed = int(fresh_assessment.get("bootstrap_seed", 0))
        pooled_assessment = None
        pooled_metrics = None
        pooled_trials = None
        promotion = PromotionResult(
            "invalid_terminal" if invalid_terminal else "not_evaluated",
            0, 0, None, None, None,
        )
        if raw_trials is not None:
            candidate_store.append_trial_group(
                full_vector,
                raw_trials,
                {
                    "identity_context": dict(identity_context),
                    "fidelity": "F1",
                    "episode_index": int(absolute_episode),
                    "variable_cost": float(variable_cost),
                    "action_matrix": [list(row) for row in action_matrix],
                    "boosted_overrides_hash": sha256_json(
                        getattr(env, "boosted_overrides", {})
                    ),
                    "boosted_overrides": _serialize_boosted_overrides(
                        getattr(env, "boosted_overrides", {})
                    ),
                    "boosted_overrides_provenance": "layerwise_env",
                    "assessment_bootstrap_seed": int(bootstrap_seed),
                    "episode_reward": float(episode_reward),
                    "promotion_marker": "online_group",
                },
            )
            evidence = candidate_store.trial_evidence_for_action(
                full_vector, identity_context,
                max_trials=promotion_trials,
            )
            if evidence is None:
                raise RuntimeError("candidate evidence append was not readable")
            pooled_assessment = assess_candidate_fn(
                evidence.trials,
                env.base.statistical_reference,
                gate_probability=promotion_probability,
                bootstrap_seed=bootstrap_seed,
            )
            pooled_metrics = _metrics_from_trials(evidence.trials)
            pooled_trials = evidence.trials
            frontier_cost = (
                max(row["variable_cost"] for row in accepted_candidates.values())
                if accepted_candidates else None
            )
            promotion = promote_candidate_if_eligible(
                env=env,
                candidate_store=candidate_store,
                action_indices=full_vector,
                identity_context=identity_context,
                action_matrix=action_matrix,
                assessment=pooled_assessment,
                priority=priority,
                variable_cost=variable_cost,
                frontier_cost=frontier_cost,
                boosted_overrides=getattr(env, "boosted_overrides", {}),
                bootstrap_seed=bootstrap_seed,
                episode_reward=episode_reward,
                assess_candidate_fn=assess_candidate_fn,
                promotion_probability=promotion_probability,
                target_trial_count=promotion_trials,
            )
            if promotion.assessment is not None:
                pooled_assessment = promotion.assessment
            if promotion.metrics is not None:
                pooled_metrics = promotion.metrics
            candidate_key_value = evidence.candidate_key
            promotion_evidence = promotion.evidence or evidence
            pooled_trials = promotion_evidence.trials
            if (
                    promotion_evidence.promoted
                    and promotion_evidence.trial_count >= promotion_trials
                    and _assessment_passes(pooled_assessment, promotion_probability)
            ):
                existing_candidate = accepted_candidates.get(candidate_key_value)
                accepted_candidates[candidate_key_value] = {
                    "variable_cost": float(variable_cost),
                    "assessment": pooled_assessment,
                    "metrics": dict(pooled_metrics or {}),
                    "action_matrix": action_matrix,
                    "full_vector": full_vector,
                    "boosted_overrides": copy.deepcopy(
                        getattr(env, "boosted_overrides", {})
                    ),
                    "reward": (
                        float(existing_candidate["reward"])
                        if existing_candidate is not None
                        and existing_candidate.get("reward") is not None
                        else float(episode_reward)
                    ),
                    "promotion_trials": promotion_evidence.trials,
                }
            else:
                accepted_candidates.pop(candidate_key_value, None)

        completed = local_episode + 1
        # The environment's direct terminal return is the PPO source of truth.
        rewards.append(episode_reward)
        entropy_snapshot = {
            "block4": None, "k": None,
            "block4_slot_count": 0, "k_slot_count": 0,
        }
        update_due = completed % update_window == 0 or completed == total_episodes
        ppo_metrics: Optional[dict[str, Any]] = None
        if update_due:
            ent_coef = _cosine_entropy_coefficient(
                train_cfg, absolute_start + completed,
            )
            ppo_metrics = dict(ppo_update_fn(
                policy,
                optimizer,
                rollout_buffer,
                ppo_cfg,
                device,
                ent_coef_override=ent_coef,
            ))
            entropy_snapshot = _current_policy_entropy(policy, entropy_samples, device)
            best_cost = (
                max(row["variable_cost"] for row in accepted_candidates.values())
                if accepted_candidates else None
            )
            convergence_state = convergence_tracker.observe_update(
                completed_episodes=absolute_start + completed,
                block4_entropy=entropy_snapshot["block4"],
                k_entropy=entropy_snapshot["k"],
                robust_feasible_cost=best_cost,
            )
            persisted_convergence_state = {
                **convergence_tracker.state_dict(),
                "block4_entropy": convergence_state.block4_entropy,
                "k_entropy": convergence_state.k_entropy,
                "converged": convergence_state.converged,
                "extension_required": convergence_state.extension_required,
            }
            ppo_metrics.update({
                "completed_episodes": absolute_start + completed,
                "block4_entropy": entropy_snapshot["block4"],
                "k_entropy": entropy_snapshot["k"],
                "stall_update_windows": convergence_state.stall_update_windows,
                "converged": convergence_state.converged,
                "extension_required": convergence_state.extension_required,
                "best_robust_feasible_cost": convergence_state.best_robust_feasible_cost,
                "convergence_state": persisted_convergence_state,
                "strict_best": _strict_best_snapshot(accepted_candidates),
            })
            ppo_diagnostics.append(ppo_metrics)

        invalid_steps = sum(
            not bool(_field(info.get("layer_summary", {}), "all_valid", True))
            for info in step_infos
        )
        record = LayerwiseEpisodeRecord(
            episode_index=absolute_episode,
            reward=float(rewards[-1]),
            priority=priority,
            action_matrix=action_matrix,
            pending_full_vector=full_vector,
            variable_cost=float(variable_cost),
            raw_trials=raw_trials,
            pooled_trials=pooled_trials,
            fresh_trial_count=(0 if raw_trials is None else len(raw_trials.loss)),
            pooled_trial_count=(0 if pooled_trials is None else len(pooled_trials.loss)),
            reward_evidence="fresh_trials",
            ranking_evidence="pooled_prefix_trials",
            fresh_assessment=fresh_assessment or None,
            assessment=pooled_assessment,
            metrics={
                name: float(value)
                for name, value in metrics.items()
                if name in (
                    "loss_mean", "loss_std", "metric1_mean", "metric1_std",
                    "metric2_mean", "metric2_std",
                )
            },
            pooled_metrics=pooled_metrics,
            promoted_trial_count=int(promotion.trial_count),
            promotion_status=str(promotion.status),
            invalid_steps=int(invalid_steps),
            step_count=12,
            block4_entropy=entropy_snapshot["block4"],
            k_entropy=entropy_snapshot["k"],
            stall_update_windows=int(convergence_state.stall_update_windows),
            converged=bool(convergence_state.converged),
            extension_required=bool(convergence_state.extension_required),
            best_robust_feasible_cost=convergence_state.best_robust_feasible_cost,
        )
        records.append(record)
        if on_episode_end is not None:
            on_episode_end(record)
        if update_due:
            if on_ppo_update_end is not None and ppo_metrics is not None:
                on_ppo_update_end(ppo_metrics, absolute_start + completed, record)
            rollout_buffer.clear()
            entropy_samples.clear()

    strict_best = _strict_best_snapshot(accepted_candidates)
    final_convergence_state = {
        **convergence_tracker.state_dict(),
        "block4_entropy": convergence_state.block4_entropy,
        "k_entropy": convergence_state.k_entropy,
        "converged": convergence_state.converged,
        "extension_required": convergence_state.extension_required,
    }
    return {
        "strict_best": strict_best,
        "convergence_state": final_convergence_state,
        "best_action": (
            list(strict_best["full_vector"]) if strict_best is not None else None
        ),
        "best_action_matrix": (
            [list(row) for row in strict_best["action_matrix"]]
            if strict_best is not None else None
        ),
        "best_full_vector": (
            list(strict_best["full_vector"]) if strict_best is not None else None
        ),
        "best_assessment": (
            dict(strict_best["assessment"]) if strict_best is not None else None
        ),
        "best_metrics": (
            dict(strict_best["metrics"]) if strict_best is not None else None
        ),
        "best_variable_cost": (
            float(strict_best["variable_cost"]) if strict_best is not None else None
        ),
        "best_reward": (
            None if strict_best is None or strict_best.get("reward") is None
            else float(strict_best["reward"])
        ),
        "best_boosted_overrides": (
            copy.deepcopy(strict_best["boosted_overrides"])
            if strict_best is not None else None
        ),
        "best_promotion_evidence": (
            copy.deepcopy(strict_best.get("promotion_evidence"))
            if strict_best is not None else None
        ),
        "episode_rewards": rewards,
        "ppo_metrics": ppo_diagnostics,
        "episode_records": records,
        "block4_entropy": convergence_state.block4_entropy,
        "k_entropy": convergence_state.k_entropy,
        "stall_update_windows": convergence_state.stall_update_windows,
        "converged": convergence_state.converged,
        "extension_required": convergence_state.extension_required,
        "recommended_extension_episodes": (
            12_000 if convergence_state.extension_required else 0
        ),
        "completed_episodes": absolute_start + total_episodes,
    }
