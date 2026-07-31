"""Constrained non-RL baselines for the canonical Stage-2 layerwise action.

The optimizers in this module are deliberately torch-free.  They operate on
one ``(Block4 fusion, H/M/L truncation preset)`` row per Transformer layer and
delegate every model interaction to a caller-supplied evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import math
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence

import numpy as np

from .layerwise_action import compute_variable_cost_from_action_matrix
from .precision_presets import validate_communication_importance_ratio


ActionMatrix = tuple[tuple[int, int], ...]
EvaluationFn = Callable[[ActionMatrix], "SearchEvaluation"]
SurrogateFactory = Callable[[int], Any]

SUPPORTED_SEARCH_BACKENDS = ("ppo", "bo_rf", "greedy", "coinn_ga")
CONSTRAINT_NAMES = (
    "loss_mean",
    "metric1_mean",
    "metric2_mean",
    "loss_std",
    "metric1_std",
    "metric2_std",
)
CONSTRAINT_PROBABILITY_NAMES = (
    "loss_precision_probability",
    "metric1_precision_probability",
    "metric2_precision_probability",
    "loss_stability_probability",
    "metric1_stability_probability",
    "metric2_stability_probability",
)
_EPS = 1.0e-12


def normalize_search_backend(value: Any) -> str:
    normalized = str(value or "ppo").strip().lower().replace("-", "_")
    aliases = {
        "bayes_rf": "bo_rf",
        "bayesian_rf": "bo_rf",
        "smac": "bo_rf",
        "smac_rf": "bo_rf",
        "ga": "coinn_ga",
        "genetic": "coinn_ga",
        "coinn": "coinn_ga",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_SEARCH_BACKENDS:
        raise ValueError(
            f"unsupported Stage-2 search backend {value!r}; expected one of "
            f"{SUPPORTED_SEARCH_BACKENDS}"
        )
    return normalized


@dataclass(frozen=True)
class ConstraintLimits:
    loss_max: float
    metric1_min: float
    metric2_min: float
    loss_std_max: float
    metric1_std_max: float
    metric2_std_max: float

    def __post_init__(self) -> None:
        for name, value in self.as_dict().items():
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        for name in ("loss_std_max", "metric1_std_max", "metric2_std_max"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be nonnegative")

    def as_dict(self) -> dict[str, float]:
        return {
            "loss_max": float(self.loss_max),
            "metric1_min": float(self.metric1_min),
            "metric2_min": float(self.metric2_min),
            "loss_std_max": float(self.loss_std_max),
            "metric1_std_max": float(self.metric1_std_max),
            "metric2_std_max": float(self.metric2_std_max),
        }


@dataclass(frozen=True)
class SearchMetrics:
    loss_mean: float
    metric1_mean: float
    metric2_mean: float
    loss_std: float
    metric1_std: float
    metric2_std: float

    def __post_init__(self) -> None:
        for name, value in self.as_dict().items():
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        for name in ("loss_std", "metric1_std", "metric2_std"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be nonnegative")

    def as_dict(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name))
            for name in CONSTRAINT_NAMES
        }


def _owned_action_matrix(action_matrix: Sequence[Sequence[int]]) -> ActionMatrix:
    return tuple(
        (int(row[0]), int(row[1]))
        for row in action_matrix
    )


@dataclass(frozen=True)
class SearchEvaluation:
    action_matrix: ActionMatrix
    metrics: SearchMetrics
    limits: ConstraintLimits
    valid: bool = True
    reward: Optional[float] = None
    communication_importance_ratio: float = 1.0
    constraint_probabilities: tuple[float, ...] = ()
    gate_probability: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        action = _owned_action_matrix(self.action_matrix)
        LayerwiseSearchSpace(len(action)).validate(action)
        object.__setattr__(self, "action_matrix", action)
        object.__setattr__(self, "valid", bool(self.valid))
        object.__setattr__(
            self,
            "communication_importance_ratio",
            validate_communication_importance_ratio(
                self.communication_importance_ratio,
            ),
        )
        probabilities = tuple(
            float(value) for value in self.constraint_probabilities
        )
        gate = self.gate_probability
        if probabilities or gate is not None:
            if len(probabilities) != len(CONSTRAINT_PROBABILITY_NAMES):
                raise ValueError(
                    "constraint_probabilities must contain all six precision "
                    "and stability probabilities"
                )
            if not all(
                    math.isfinite(value) and 0.0 <= value <= 1.0
                    for value in probabilities
            ):
                raise ValueError(
                    "constraint probabilities must be finite and in [0, 1]"
                )
            gate = float(gate)
            if not math.isfinite(gate) or not 0.0 < gate <= 1.0:
                raise ValueError("gate_probability must be in (0, 1]")
        object.__setattr__(self, "constraint_probabilities", probabilities)
        object.__setattr__(self, "gate_probability", gate)
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.reward is not None and not math.isfinite(float(self.reward)):
            raise ValueError("reward must be finite when provided")

    @property
    def raw_margins(self) -> tuple[float, ...]:
        return (
            float(self.limits.loss_max - self.metrics.loss_mean),
            float(self.metrics.metric1_mean - self.limits.metric1_min),
            float(self.metrics.metric2_mean - self.limits.metric2_min),
            float(self.limits.loss_std_max - self.metrics.loss_std),
            float(self.limits.metric1_std_max - self.metrics.metric1_std),
            float(self.limits.metric2_std_max - self.metrics.metric2_std),
        )

    @property
    def normalized_margins(self) -> tuple[float, ...]:
        scales = (
            max(abs(float(self.limits.loss_max)), 1.0e-6),
            max(abs(float(self.limits.metric1_min)), 1.0e-6),
            max(abs(float(self.limits.metric2_min)), 1.0e-6),
            max(abs(float(self.limits.loss_std_max)), 1.0e-6),
            max(abs(float(self.limits.metric1_std_max)), 1.0e-6),
            max(abs(float(self.limits.metric2_std_max)), 1.0e-6),
        )
        margins = tuple(
            float(value / scale)
            for value, scale in zip(self.raw_margins, scales)
        )
        if self.valid:
            return margins
        return tuple(min(value, -1.0) for value in margins)

    @property
    def feasible(self) -> bool:
        return bool(
            self.valid
            and all(value >= -_EPS for value in self.constraint_margins)
        )

    @property
    def normalized_violation(self) -> float:
        return float(sum(max(0.0, -value) for value in self.constraint_margins))

    @property
    def constraint_margins(self) -> tuple[float, ...]:
        """Return the six margins used by PPO's active feasibility gate."""
        if self.constraint_probabilities:
            gate = float(self.gate_probability)
            margins = tuple(
                float(value - gate)
                for value in self.constraint_probabilities
            )
            if self.valid:
                return margins
            return tuple(min(value, -1.0) for value in margins)
        return self.normalized_margins

    @property
    def resource(self) -> Any:
        return compute_variable_cost_from_action_matrix(
            self.action_matrix,
            communication_importance_ratio=self.communication_importance_ratio,
        )

    def as_dict(self) -> dict[str, Any]:
        resource = self.resource
        return {
            "action_matrix": [list(row) for row in self.action_matrix],
            "metrics": self.metrics.as_dict(),
            "limits": self.limits.as_dict(),
            "valid": bool(self.valid),
            "feasible": bool(self.feasible),
            "raw_margins": [float(value) for value in self.raw_margins],
            "normalized_margins": [
                float(value) for value in self.normalized_margins
            ],
            "constraint_probabilities": {
                name: float(value)
                for name, value in zip(
                    CONSTRAINT_PROBABILITY_NAMES,
                    self.constraint_probabilities,
                )
            },
            "gate_probability": (
                None
                if self.gate_probability is None
                else float(self.gate_probability)
            ),
            "constraint_margins": [
                float(value) for value in self.constraint_margins
            ],
            "normalized_violation": float(self.normalized_violation),
            "reward": None if self.reward is None else float(self.reward),
            "communication_importance_ratio": float(
                self.communication_importance_ratio
            ),
            "resource": {
                "compute_saving": float(resource.compute_saving),
                "communication_saving": float(resource.communication_saving),
                "robust_floor": float(resource.robust_floor),
                "secondary_progress": float(resource.secondary_progress),
                "ppo_resource_score": float(resource.ppo_resource_score),
                "fusion_count": int(resource.fusion_count),
                "removed_k_bits": int(resource.removed_k_bits),
            },
            "metadata": dict(self.metadata),
        }


class LayerwiseSearchSpace:
    """Discrete ``num_layers x 2`` Stage-2 policy space."""

    def __init__(self, num_layers: int):
        self.num_layers = int(num_layers)
        if self.num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.dimensions = tuple(
            value
            for _layer_idx in range(self.num_layers)
            for value in (2, 3)
        )

    @property
    def cardinality(self) -> int:
        return int(math.prod(self.dimensions))

    @property
    def safe_action(self) -> ActionMatrix:
        return tuple((0, 0) for _ in range(self.num_layers))

    @property
    def max_resource_action(self) -> ActionMatrix:
        return tuple((1, 2) for _ in range(self.num_layers))

    def validate(self, action_matrix: Sequence[Sequence[int]]) -> ActionMatrix:
        rows = tuple(tuple(int(value) for value in row) for row in action_matrix)
        if len(rows) != self.num_layers:
            raise ValueError(
                f"action_matrix must contain {self.num_layers} layers, "
                f"got {len(rows)}"
            )
        for layer_idx, row in enumerate(rows):
            if len(row) != 2:
                raise ValueError(
                    f"action_matrix layer {layer_idx} must contain two coordinates"
                )
            for slot_idx, (value, dimension) in enumerate(zip(row, (2, 3))):
                if not 0 <= value < dimension:
                    raise ValueError(
                        f"action_matrix[{layer_idx}][{slot_idx}]={value} "
                        f"outside [0, {dimension})"
                    )
        return _owned_action_matrix(rows)

    def flatten(self, action_matrix: Sequence[Sequence[int]]) -> tuple[int, ...]:
        action = self.validate(action_matrix)
        return tuple(value for row in action for value in row)

    def unflatten(self, values: Sequence[int]) -> ActionMatrix:
        flat = tuple(int(value) for value in values)
        if len(flat) != 2 * self.num_layers:
            raise ValueError(
                f"flat action must contain {2 * self.num_layers} values"
            )
        return self.validate(
            tuple((flat[2 * index], flat[2 * index + 1])
                  for index in range(self.num_layers))
        )

    def random_action(self, rng: np.random.Generator) -> ActionMatrix:
        return self.unflatten(tuple(
            int(rng.integers(dimension))
            for dimension in self.dimensions
        ))

    def all_actions(self, *, max_cardinality: int = 100_000) -> Iterator[ActionMatrix]:
        if self.cardinality > int(max_cardinality):
            raise ValueError(
                f"action space cardinality {self.cardinality} exceeds "
                f"enumeration cap {int(max_cardinality)}"
            )
        for values in itertools.product(
                *(range(dimension) for dimension in self.dimensions)
        ):
            yield self.unflatten(values)

    def neighbors(self, action_matrix: Sequence[Sequence[int]]) -> Iterator[ActionMatrix]:
        flat = list(self.flatten(action_matrix))
        for index, (current, dimension) in enumerate(zip(flat, self.dimensions)):
            if index % 2 == 0:
                alternatives: Iterable[int] = (1 - current,)
            else:
                alternatives = (
                    value
                    for value in (current - 1, current + 1)
                    if 0 <= value < dimension
                )
            for alternative in alternatives:
                candidate = flat[:]
                candidate[index] = int(alternative)
                yield self.unflatten(candidate)

    def mutate(
            self,
            action_matrix: Sequence[Sequence[int]],
            rng: np.random.Generator,
            *,
            max_coordinates: int = 1,
            ) -> ActionMatrix:
        candidate = self.validate(action_matrix)
        count = int(rng.integers(1, max(1, int(max_coordinates)) + 1))
        for _ in range(count):
            neighbors = tuple(self.neighbors(candidate))
            if not neighbors:
                break
            candidate = neighbors[int(rng.integers(len(neighbors)))]
        return candidate


def candidate_rank_key(evaluation: SearchEvaluation) -> tuple[float, ...]:
    resource = evaluation.resource
    lexicographic = tuple(
        -float(value)
        for value in LayerwiseSearchSpace(
            len(evaluation.action_matrix)
        ).flatten(evaluation.action_matrix)
    )
    constraint_margins = tuple(sorted(evaluation.constraint_margins))
    point_margins = tuple(sorted(evaluation.normalized_margins))
    confidence = tuple(sorted(evaluation.constraint_probabilities))
    if evaluation.feasible:
        return (
            1.0,
            float(resource.ppo_resource_score),
            float(resource.robust_floor),
            *confidence,
            *point_margins,
            *lexicographic,
        )
    return (
        0.0,
        -float(evaluation.normalized_violation),
        *constraint_margins,
        float(resource.ppo_resource_score),
        float(resource.robust_floor),
        *lexicographic,
    )


@dataclass(frozen=True)
class SearchConfig:
    evaluation_budget: int
    seed: int = 42
    initial_design_size: int = 8
    candidate_pool_size: int = 512
    population_size: int = 24
    patience_generations: int = 5
    mutation_max_coordinates: int = 3
    rf_n_estimators: int = 128
    rf_min_samples_leaf: int = 2
    acquisition_exploration: float = 0.05
    communication_importance_ratio: float = 1.0

    def __post_init__(self) -> None:
        for name in (
                "evaluation_budget", "initial_design_size",
                "candidate_pool_size", "population_size",
                "patience_generations", "mutation_max_coordinates",
                "rf_n_estimators", "rf_min_samples_leaf",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if not math.isfinite(float(self.acquisition_exploration)):
            raise ValueError("acquisition_exploration must be finite")
        if float(self.acquisition_exploration) < 0.0:
            raise ValueError("acquisition_exploration must be nonnegative")
        object.__setattr__(
            self,
            "communication_importance_ratio",
            validate_communication_importance_ratio(
                self.communication_importance_ratio,
            ),
        )


@dataclass(frozen=True)
class SearchResult:
    algorithm: str
    best: SearchEvaluation
    observations: tuple[SearchEvaluation, ...]
    history: tuple[Mapping[str, Any], ...]
    termination_reason: str

    @property
    def evaluation_count(self) -> int:
        return len(self.observations)

    def as_dict(self) -> dict[str, Any]:
        return {
            "algorithm": str(self.algorithm),
            "evaluation_count": int(self.evaluation_count),
            "termination_reason": str(self.termination_reason),
            "best": self.best.as_dict(),
            "history": [dict(row) for row in self.history],
        }


class _EvaluationCache:
    def __init__(
            self,
            space: LayerwiseSearchSpace,
            evaluator: EvaluationFn,
            budget: int,
            ):
        self.space = space
        self.evaluator = evaluator
        self.budget = min(int(budget), int(space.cardinality))
        self._by_action: dict[ActionMatrix, SearchEvaluation] = {}
        self._ordered: list[SearchEvaluation] = []

    @property
    def remaining(self) -> int:
        return max(0, self.budget - len(self._ordered))

    @property
    def observations(self) -> tuple[SearchEvaluation, ...]:
        return tuple(self._ordered)

    def contains(self, action: Sequence[Sequence[int]]) -> bool:
        return self.space.validate(action) in self._by_action

    def evaluate(self, action: Sequence[Sequence[int]]) -> SearchEvaluation:
        owned = self.space.validate(action)
        cached = self._by_action.get(owned)
        if cached is not None:
            return cached
        if self.remaining <= 0:
            raise RuntimeError("search evaluation budget exhausted")
        observed = self.evaluator(owned)
        if not isinstance(observed, SearchEvaluation):
            raise TypeError("search evaluator must return SearchEvaluation")
        if observed.action_matrix != owned:
            raise ValueError(
                "search evaluator returned metrics for a different action"
            )
        self._by_action[owned] = observed
        self._ordered.append(observed)
        return observed

    def best(self) -> SearchEvaluation:
        if not self._ordered:
            raise RuntimeError("search produced no observations")
        return max(self._ordered, key=candidate_rank_key)


def _resource_score(
        action: ActionMatrix,
        communication_importance_ratio: float,
        ) -> float:
    return float(
        compute_variable_cost_from_action_matrix(
            action,
            communication_importance_ratio=communication_importance_ratio,
        ).ppo_resource_score
    )


def _initial_design(
        space: LayerwiseSearchSpace,
        rng: np.random.Generator,
        count: int,
        ) -> list[ActionMatrix]:
    target = min(int(count), int(space.cardinality))
    candidates: list[ActionMatrix] = []
    seen: set[ActionMatrix] = set()

    def add(action: ActionMatrix) -> None:
        owned = space.validate(action)
        if owned not in seen and len(candidates) < target:
            candidates.append(owned)
            seen.add(owned)

    add(space.safe_action)
    add(space.max_resource_action)
    add(tuple((1, 0) for _ in range(space.num_layers)))
    add(tuple((0, 2) for _ in range(space.num_layers)))
    add(tuple(
        (layer_idx % 2, 1)
        for layer_idx in range(space.num_layers)
    ))

    if space.cardinality <= max(256, 4 * target):
        for action in space.all_actions(max_cardinality=max(256, space.cardinality)):
            add(action)
            if len(candidates) >= target:
                break
    attempts = 0
    while len(candidates) < target and attempts < max(100, 20 * target):
        attempts += 1
        add(space.random_action(rng))
    return candidates


def _best_history_row(
        cache: _EvaluationCache,
        *,
        phase: str,
        iteration: int,
        **extra: Any,
        ) -> dict[str, Any]:
    best = cache.best()
    resource = best.resource
    return {
        "iteration": int(iteration),
        "phase": str(phase),
        "evaluations": len(cache.observations),
        "best_action_matrix": [list(row) for row in best.action_matrix],
        "best_feasible": bool(best.feasible),
        "best_violation": float(best.normalized_violation),
        "best_resource_score": float(resource.ppo_resource_score),
        "best_robust_floor": float(resource.robust_floor),
        **extra,
    }


def _run_greedy(
        space: LayerwiseSearchSpace,
        evaluator: EvaluationFn,
        config: SearchConfig,
        ) -> SearchResult:
    cache = _EvaluationCache(space, evaluator, config.evaluation_budget)
    current = cache.evaluate(space.safe_action)
    history = [_best_history_row(cache, phase="initial", iteration=0)]
    iteration = 0
    termination = "local_optimum"
    while cache.remaining > 0:
        iteration += 1
        current_score = _resource_score(
            current.action_matrix,
            config.communication_importance_ratio,
        )
        neighbors = [
            action
            for action in space.neighbors(current.action_matrix)
            if _resource_score(
                action, config.communication_importance_ratio,
            ) > current_score + _EPS
            and not cache.contains(action)
        ]
        neighbors.sort(
            key=lambda action: (
                _resource_score(
                    action, config.communication_importance_ratio,
                ),
                tuple(-value for value in space.flatten(action)),
            ),
            reverse=True,
        )
        evaluated = [
            cache.evaluate(action)
            for action in neighbors[:cache.remaining]
        ]
        if not evaluated:
            break

        if current.feasible:
            improvements = [
                item for item in evaluated
                if item.feasible
                and candidate_rank_key(item) > candidate_rank_key(current)
            ]
        else:
            improvements = [
                item for item in evaluated
                if candidate_rank_key(item) > candidate_rank_key(current)
            ]
        if not improvements:
            history.append(_best_history_row(
                cache, phase="neighbor_scan", iteration=iteration,
                accepted=False,
            ))
            break
        current = max(improvements, key=candidate_rank_key)
        history.append(_best_history_row(
            cache, phase="neighbor_scan", iteration=iteration,
            accepted=True,
        ))
    if cache.remaining <= 0:
        termination = "evaluation_budget"
    return SearchResult(
        algorithm="greedy",
        best=cache.best(),
        observations=cache.observations,
        history=tuple(history),
        termination_reason=termination,
    )


def _default_surrogate_factory(
        config: SearchConfig,
        ) -> SurrogateFactory:
    def factory(seed: int) -> Any:
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError as exc:  # pragma: no cover - exercised on server
            raise RuntimeError(
                "BO-RF requires scikit-learn; install it in the runtime "
                "environment or provide surrogate_factory"
            ) from exc
        return RandomForestRegressor(
            n_estimators=int(config.rf_n_estimators),
            min_samples_leaf=int(config.rf_min_samples_leaf),
            max_features=0.75,
            bootstrap=True,
            random_state=int(seed),
            n_jobs=-1,
        )
    return factory


def _candidate_pool(
        space: LayerwiseSearchSpace,
        cache: _EvaluationCache,
        rng: np.random.Generator,
        *,
        pool_size: int,
        ) -> list[ActionMatrix]:
    target = min(int(pool_size), int(space.cardinality) - len(cache.observations))
    if target <= 0:
        return []
    if space.cardinality <= 100_000:
        return [
            action for action in space.all_actions()
            if not cache.contains(action)
        ][:target]

    pool: list[ActionMatrix] = []
    seen: set[ActionMatrix] = set()

    def add(action: ActionMatrix) -> None:
        owned = space.validate(action)
        if (
                owned not in seen
                and not cache.contains(owned)
                and len(pool) < target
        ):
            pool.append(owned)
            seen.add(owned)

    ranked = sorted(
        cache.observations,
        key=candidate_rank_key,
        reverse=True,
    )
    for observation in ranked[:8]:
        for neighbor in space.neighbors(observation.action_matrix):
            add(neighbor)
    attempts = 0
    while len(pool) < target and attempts < max(1000, 30 * target):
        attempts += 1
        add(space.random_action(rng))
    return pool


def _tree_predictions(model: Any, features: np.ndarray) -> np.ndarray:
    estimators = tuple(getattr(model, "estimators_", ()) or ())
    if estimators:
        predictions = [
            np.asarray(estimator.predict(features), dtype=float)
            for estimator in estimators
        ]
        return np.stack(predictions, axis=0)
    prediction = np.asarray(model.predict(features), dtype=float)
    return prediction.reshape(1, prediction.shape[0], -1)


def _run_bo_rf(
        space: LayerwiseSearchSpace,
        evaluator: EvaluationFn,
        config: SearchConfig,
        surrogate_factory: Optional[SurrogateFactory],
        ) -> SearchResult:
    rng = np.random.default_rng(int(config.seed))
    cache = _EvaluationCache(space, evaluator, config.evaluation_budget)
    history: list[dict[str, Any]] = []
    design = _initial_design(
        space,
        rng,
        min(config.initial_design_size, cache.budget),
    )
    for action in design:
        if cache.remaining <= 0:
            break
        cache.evaluate(action)
    history.append(_best_history_row(cache, phase="initial_design", iteration=0))

    factory = surrogate_factory or _default_surrogate_factory(config)
    iteration = 0
    while cache.remaining > 0:
        iteration += 1
        observations = cache.observations
        features = np.asarray(
            [space.flatten(item.action_matrix) for item in observations],
            dtype=float,
        )
        targets = np.asarray(
            [item.constraint_margins for item in observations],
            dtype=float,
        )
        model = factory(int(config.seed) + iteration)
        model.fit(features, targets)
        pool = _candidate_pool(
            space,
            cache,
            rng,
            pool_size=config.candidate_pool_size,
        )
        if not pool:
            break
        pool_features = np.asarray(
            [space.flatten(action) for action in pool],
            dtype=float,
        )
        tree_predictions = _tree_predictions(model, pool_features)
        if tree_predictions.ndim != 3 or tree_predictions.shape[2] != 6:
            raise RuntimeError(
                "BO-RF surrogate must predict six constraint margins"
            )
        feasible_probability = np.mean(
            np.all(tree_predictions >= 0.0, axis=2),
            axis=0,
        )
        predicted_mean = tree_predictions.mean(axis=0)
        predicted_std = tree_predictions.std(axis=0)
        feasible_observations = [item for item in observations if item.feasible]
        incumbent_resource = max(
            (_resource_score(
                item.action_matrix, config.communication_importance_ratio,
            )
             for item in feasible_observations),
            default=0.0,
        )
        acquisition = []
        for index, action in enumerate(pool):
            resource = _resource_score(
                action, config.communication_importance_ratio,
            )
            probability = float(feasible_probability[index])
            uncertainty = float(np.mean(predicted_std[index]))
            predicted_floor = float(np.min(predicted_mean[index]))
            if feasible_observations:
                improvement = max(0.0, resource - incumbent_resource)
                value = probability * (
                    improvement
                    + float(config.acquisition_exploration) * uncertainty
                    + 1.0e-9
                )
                value += 1.0e-6 * resource
            else:
                value = (
                    probability
                    + 0.05 * predicted_floor
                    + 0.01 * resource
                    + float(config.acquisition_exploration) * uncertainty
                )
            acquisition.append(float(value))
        selected_index = max(
            range(len(pool)),
            key=lambda index: (
                acquisition[index],
                _resource_score(
                    pool[index], config.communication_importance_ratio,
                ),
                tuple(-value for value in space.flatten(pool[index])),
            ),
        )
        selected = pool[selected_index]
        cache.evaluate(selected)
        history.append(_best_history_row(
            cache,
            phase="constrained_ei",
            iteration=iteration,
            acquisition=float(acquisition[selected_index]),
            predicted_feasibility=float(
                feasible_probability[selected_index]
            ),
        ))
    termination = (
        "evaluation_budget" if cache.remaining <= 0 else "candidate_space_exhausted"
    )
    return SearchResult(
        algorithm="bo_rf",
        best=cache.best(),
        observations=cache.observations,
        history=tuple(history),
        termination_reason=termination,
    )


def _ga_fitness(evaluation: SearchEvaluation) -> float:
    if evaluation.feasible:
        return 2.0 + float(evaluation.resource.ppo_resource_score)
    return max(1.0e-9, math.exp(-evaluation.normalized_violation))


def _run_coinn_ga(
        space: LayerwiseSearchSpace,
        evaluator: EvaluationFn,
        config: SearchConfig,
        ) -> SearchResult:
    rng = np.random.default_rng(int(config.seed))
    cache = _EvaluationCache(space, evaluator, config.evaluation_budget)
    population_target = min(
        int(config.population_size),
        int(space.cardinality),
        int(cache.budget),
    )
    population = _initial_design(space, rng, population_target)
    history: list[dict[str, Any]] = []
    incumbent: Optional[SearchEvaluation] = None
    stagnation = 0
    generation = 0
    termination = "evaluation_budget"

    while population and cache.remaining > 0:
        generation += 1
        generation_observations = [
            cache.evaluate(action)
            for action in population[:cache.remaining]
        ]
        if not generation_observations:
            break
        generation_best = max(
            generation_observations,
            key=candidate_rank_key,
        )
        improved = bool(
            incumbent is None
            or candidate_rank_key(generation_best)
            > candidate_rank_key(incumbent)
        )
        if improved:
            incumbent = generation_best
            stagnation = 0
        else:
            stagnation += 1
        history.append(_best_history_row(
            cache,
            phase="population",
            iteration=generation,
            generation=int(generation),
            population_evaluated=len(generation_observations),
            improved=bool(improved),
            stagnation=int(stagnation),
        ))
        if cache.remaining <= 0:
            break
        if stagnation >= int(config.patience_generations):
            termination = "coinn_stagnation"
            break

        ranked = sorted(
            generation_observations,
            key=candidate_rank_key,
            reverse=True,
        )
        elite_count = min(max(1, population_target // 5), len(ranked))
        next_population: list[ActionMatrix] = []
        next_seen: set[ActionMatrix] = set()

        def add(action: ActionMatrix) -> None:
            owned = space.validate(action)
            if (
                    owned not in next_seen
                    and not cache.contains(owned)
                    and len(next_population) < population_target
            ):
                next_population.append(owned)
                next_seen.add(owned)

        for elite in ranked[:elite_count]:
            for neighbor in space.neighbors(elite.action_matrix):
                add(neighbor)
                if len(next_population) >= elite_count:
                    break

        fitness = np.asarray(
            [_ga_fitness(item) for item in generation_observations],
            dtype=float,
        )
        probabilities = fitness / fitness.sum()
        attempts = 0
        while (
                len(next_population) < population_target
                and attempts < max(200, 40 * population_target)
        ):
            attempts += 1
            parent = generation_observations[
                int(rng.choice(len(generation_observations), p=probabilities))
            ]
            child = space.mutate(
                parent.action_matrix,
                rng,
                max_coordinates=config.mutation_max_coordinates,
            )
            add(child)
        while len(next_population) < population_target:
            candidate = space.random_action(rng)
            before = len(next_population)
            add(candidate)
            if len(next_population) == before and (
                    len(cache.observations) + len(next_population)
                    >= space.cardinality
            ):
                break
        population = next_population

    if cache.remaining > 0 and not population:
        termination = "candidate_space_exhausted"
    return SearchResult(
        algorithm="coinn_ga",
        best=cache.best(),
        observations=cache.observations,
        history=tuple(history),
        termination_reason=termination,
    )


def run_search(
        backend: str,
        space: LayerwiseSearchSpace,
        evaluator: EvaluationFn,
        config: SearchConfig,
        *,
        surrogate_factory: Optional[SurrogateFactory] = None,
        ) -> SearchResult:
    normalized = normalize_search_backend(backend)
    if normalized == "ppo":
        raise ValueError(
            "run_search implements non-RL baselines only; PPO uses the "
            "existing layerwise trainer"
        )
    if normalized == "greedy":
        return _run_greedy(space, evaluator, config)
    if normalized == "bo_rf":
        return _run_bo_rf(
            space, evaluator, config, surrogate_factory,
        )
    if normalized == "coinn_ga":
        return _run_coinn_ga(space, evaluator, config)
    raise AssertionError(f"unhandled search backend {normalized}")


__all__ = [
    "ActionMatrix",
    "CONSTRAINT_NAMES",
    "CONSTRAINT_PROBABILITY_NAMES",
    "ConstraintLimits",
    "LayerwiseSearchSpace",
    "SUPPORTED_SEARCH_BACKENDS",
    "SearchConfig",
    "SearchEvaluation",
    "SearchMetrics",
    "SearchResult",
    "candidate_rank_key",
    "normalize_search_backend",
    "run_search",
]
