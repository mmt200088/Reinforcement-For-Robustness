"""Canonical torch-free search baselines for Stage-1 GELU-only search.

The search chromosome is an immutable tuple with one categorical gene per
Transformer layer.  Categories ``0, 1, 2`` decode to GELU degrees ``4, 2, 1``;
Softmax is not searched and is always fixed to degree 6 in every layer.

All model interaction is delegated to a caller-supplied evaluator.  This module
contains no torch import and imports scikit-learn only lazily when the default
random-forest surrogate is requested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
import heapq
import itertools
import math
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence

import numpy as np


Stage1Action = tuple[int, ...]
EvaluationFn = Callable[[Stage1Action], "SearchEvaluation"]
SurrogateFactory = Callable[[int], Any]
CheckpointCallback = Callable[[tuple["SearchEvaluation", ...]], None]
IncrementalCheckpointCallback = Callable[["SearchEvaluation", int], None]

GENE_CATEGORIES = (0, 1, 2)
GELU_DEGREES = (4, 2, 1)
FIXED_SOFTMAX_DEGREE = 6
SUPPORTED_SEARCH_BACKENDS = ("bo_rf", "greedy", "coinn_ga")
_EPS = 1.0e-12


def normalize_search_backend(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "bayes": "bo_rf",
        "bayesian": "bo_rf",
        "bayes_rf": "bo_rf",
        "bayesian_rf": "bo_rf",
        "rf_bo": "bo_rf",
        "ga": "coinn_ga",
        "genetic": "coinn_ga",
        "coinn": "coinn_ga",
        "local": "greedy",
        "local_search": "greedy",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_SEARCH_BACKENDS:
        raise ValueError(
            f"unsupported Stage-1 search backend {value!r}; expected one of "
            f"{SUPPORTED_SEARCH_BACKENDS}"
        )
    return normalized


def _finite(name: str, value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _float_tuple(name: str, values: Sequence[Any]) -> tuple[float, ...]:
    result = tuple(_finite(f"{name}[{index}]", value) for index, value in enumerate(values))
    if not result:
        raise ValueError(f"{name} must contain at least one value")
    return result


def _broadcast_tolerance(value: Any, count: int, *, name: str) -> tuple[float, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        tolerances = tuple(float(item) for item in value)
    else:
        tolerances = (float(value),) * int(count)
    if len(tolerances) != int(count):
        raise ValueError(f"{name} must contain {count} values")
    if not all(math.isfinite(item) and item >= 0.0 for item in tolerances):
        raise ValueError(f"{name} values must be finite and nonnegative")
    return tolerances


@dataclass(frozen=True)
class Stage1Constraints:
    """Exact baseline values and the hard limits derived from them.

    ``baseline_metrics`` and ``metric_mins`` may contain either one or two
    metrics.  Their names are carried for persistence only; feasibility always
    uses every supplied metric.
    """

    baseline_loss: float
    baseline_metrics: tuple[float, ...]
    loss_max: float
    metric_mins: tuple[float, ...]
    metric_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        baseline_loss = _finite("baseline_loss", self.baseline_loss)
        baseline_metrics = _float_tuple("baseline_metrics", self.baseline_metrics)
        loss_max = _finite("loss_max", self.loss_max)
        metric_mins = _float_tuple("metric_mins", self.metric_mins)
        if len(baseline_metrics) != len(metric_mins):
            raise ValueError("baseline_metrics and metric_mins must have equal length")
        if len(metric_mins) not in (1, 2):
            raise ValueError("Stage-1 search supports one or two metrics")
        names = tuple(str(name) for name in self.metric_names)
        if not names:
            names = tuple(f"metric{index + 1}" for index in range(len(metric_mins)))
        if len(names) != len(metric_mins):
            raise ValueError("metric_names must match the supplied metrics")
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("metric_names must be non-empty and unique")
        object.__setattr__(self, "baseline_loss", baseline_loss)
        object.__setattr__(self, "baseline_metrics", baseline_metrics)
        object.__setattr__(self, "loss_max", loss_max)
        object.__setattr__(self, "metric_mins", metric_mins)
        object.__setattr__(self, "metric_names", names)

    @classmethod
    def from_baseline(
            cls,
            *,
            baseline_loss: float,
            baseline_metrics: Sequence[float],
            loss_relative_tolerance: float,
            metric_relative_tolerance: Any,
            metric_names: Sequence[str] = (),
            ) -> "Stage1Constraints":
        metrics = _float_tuple("baseline_metrics", baseline_metrics)
        loss_tolerance = float(loss_relative_tolerance)
        if not math.isfinite(loss_tolerance) or loss_tolerance < 0.0:
            raise ValueError("loss_relative_tolerance must be finite and nonnegative")
        metric_tolerances = _broadcast_tolerance(
            metric_relative_tolerance,
            len(metrics),
            name="metric_relative_tolerance",
        )
        return cls(
            baseline_loss=float(baseline_loss),
            baseline_metrics=metrics,
            loss_max=float(baseline_loss) * (1.0 + loss_tolerance),
            metric_mins=tuple(
                metric * (1.0 - tolerance)
                for metric, tolerance in zip(metrics, metric_tolerances)
            ),
            metric_names=tuple(metric_names),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "baseline_loss": float(self.baseline_loss),
            "baseline_metrics": [float(value) for value in self.baseline_metrics],
            "loss_max": float(self.loss_max),
            "metric_mins": [float(value) for value in self.metric_mins],
            "metric_names": list(self.metric_names),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Stage1Constraints":
        return cls(
            baseline_loss=float(payload["baseline_loss"]),
            baseline_metrics=tuple(float(value) for value in payload["baseline_metrics"]),
            loss_max=float(payload["loss_max"]),
            metric_mins=tuple(float(value) for value in payload["metric_mins"]),
            metric_names=tuple(str(value) for value in payload.get("metric_names", ())),
        )


class Stage1SearchSpace:
    """Immutable categorical ``3**L`` GELU-only search space."""

    def __init__(self, num_layers: int):
        self.num_layers = int(num_layers)
        if self.num_layers < 1:
            raise ValueError("num_layers must be positive")

    @property
    def cardinality(self) -> int:
        return 3 ** self.num_layers

    @property
    def all4_action(self) -> Stage1Action:
        return (0,) * self.num_layers

    @property
    def all2_action(self) -> Stage1Action:
        return (1,) * self.num_layers

    @property
    def all1_action(self) -> Stage1Action:
        return (2,) * self.num_layers

    @property
    def anchors(self) -> tuple[Stage1Action, ...]:
        return (self.all4_action, self.all2_action, self.all1_action)

    def validate(self, action: Sequence[int]) -> Stage1Action:
        owned = tuple(int(value) for value in action)
        if len(owned) != self.num_layers:
            raise ValueError(
                f"Stage-1 action must contain {self.num_layers} genes, got {len(owned)}"
            )
        for layer_idx, category in enumerate(owned):
            if category not in GENE_CATEGORIES:
                raise ValueError(
                    f"Stage-1 action[{layer_idx}]={category} is not a category in "
                    f"{GENE_CATEGORIES}"
                )
        return owned

    def decode_gelu(self, action: Sequence[int]) -> tuple[int, ...]:
        return tuple(GELU_DEGREES[value] for value in self.validate(action))

    def fixed_softmax(self) -> tuple[int, ...]:
        return (FIXED_SOFTMAX_DEGREE,) * self.num_layers

    def action_from_index(self, index: int) -> Stage1Action:
        value = int(index)
        if not 0 <= value < self.cardinality:
            raise ValueError("action index outside search-space cardinality")
        genes = [0] * self.num_layers
        for position in range(self.num_layers - 1, -1, -1):
            value, genes[position] = divmod(value, 3)
        return tuple(genes)

    def random_action(self, rng: np.random.Generator) -> Stage1Action:
        return tuple(int(value) for value in rng.integers(0, 3, size=self.num_layers))

    def all_actions(self, *, max_cardinality: int = 1_000_000) -> Iterator[Stage1Action]:
        if self.cardinality > int(max_cardinality):
            raise ValueError(
                f"search-space cardinality {self.cardinality} exceeds enumeration cap "
                f"{int(max_cardinality)}"
            )
        yield from itertools.product(GENE_CATEGORIES, repeat=self.num_layers)

    def one_opt_neighbors(self, action: Sequence[int]) -> Iterator[Stage1Action]:
        current = self.validate(action)
        for layer_idx, category in enumerate(current):
            for replacement in GENE_CATEGORIES:
                if replacement == category:
                    continue
                candidate = list(current)
                candidate[layer_idx] = replacement
                yield tuple(candidate)

    def two_opt_neighbors(self, action: Sequence[int]) -> Iterator[Stage1Action]:
        current = self.validate(action)
        for first in range(self.num_layers):
            for second in range(first + 1, self.num_layers):
                for first_value in GENE_CATEGORIES:
                    if first_value == current[first]:
                        continue
                    for second_value in GENE_CATEGORIES:
                        if second_value == current[second]:
                            continue
                        candidate = list(current)
                        candidate[first] = first_value
                        candidate[second] = second_value
                        yield tuple(candidate)

    def one_hot(self, actions: Sequence[Sequence[int]]) -> np.ndarray:
        rows = [self.validate(action) for action in actions]
        features = np.zeros((len(rows), self.num_layers * 3), dtype=float)
        for row_idx, action in enumerate(rows):
            for layer_idx, category in enumerate(action):
                features[row_idx, layer_idx * 3 + category] = 1.0
        return features

    def hamming_distance(self, first: Sequence[int], second: Sequence[int]) -> int:
        lhs = self.validate(first)
        rhs = self.validate(second)
        return sum(int(a != b) for a, b in zip(lhs, rhs))


@dataclass(frozen=True)
class SearchEvaluation:
    action: Stage1Action
    loss: float
    metrics: tuple[float, ...]
    cost: float
    constraints: Stage1Constraints
    valid: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        action = tuple(int(value) for value in self.action)
        Stage1SearchSpace(len(action)).validate(action)
        loss = _finite("loss", self.loss)
        metrics = _float_tuple("metrics", self.metrics)
        cost = _finite("cost", self.cost)
        if cost < 0.0:
            raise ValueError("cost must be nonnegative")
        if len(metrics) != len(self.constraints.metric_mins):
            raise ValueError("evaluation metrics must match the constraint metric count")
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "loss", loss)
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "cost", cost)
        object.__setattr__(self, "valid", bool(self.valid))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @cached_property
    def gelu_degrees(self) -> tuple[int, ...]:
        return Stage1SearchSpace(len(self.action)).decode_gelu(self.action)

    @cached_property
    def softmax_degrees(self) -> tuple[int, ...]:
        return Stage1SearchSpace(len(self.action)).fixed_softmax()

    @cached_property
    def raw_margins(self) -> tuple[float, ...]:
        return (
            float(self.constraints.loss_max - self.loss),
            *(float(value - minimum)
              for value, minimum in zip(self.metrics, self.constraints.metric_mins)),
        )

    @cached_property
    def normalized_margins(self) -> tuple[float, ...]:
        scales = (
            max(abs(float(self.constraints.loss_max)), 1.0e-12),
            *(max(abs(float(value)), 1.0e-12)
              for value in self.constraints.metric_mins),
        )
        return tuple(margin / scale for margin, scale in zip(self.raw_margins, scales))

    @cached_property
    def constraint_margins(self) -> tuple[float, ...]:
        margins = self.normalized_margins
        if self.valid:
            return margins
        return tuple(min(value, -1.0) for value in margins)

    @cached_property
    def violations(self) -> tuple[float, ...]:
        return tuple(max(0.0, -margin) for margin in self.constraint_margins)

    @cached_property
    def failed_constraint_count(self) -> int:
        return sum(int(value > _EPS) for value in self.violations)

    @cached_property
    def total_violation(self) -> float:
        return float(sum(self.violations))

    @cached_property
    def worst_violation(self) -> float:
        return float(max(self.violations, default=0.0))

    @cached_property
    def feasible(self) -> bool:
        return bool(self.valid and self.failed_constraint_count == 0)

    @property
    def metric1(self) -> float:
        return float(self.metrics[0])

    @property
    def metric2(self) -> Optional[float]:
        return None if len(self.metrics) < 2 else float(self.metrics[1])

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": list(self.action),
            "gelu_degrees": list(self.gelu_degrees),
            "softmax_degrees": list(self.softmax_degrees),
            "loss": float(self.loss),
            "metrics": [float(value) for value in self.metrics],
            "metric_values": {
                name: float(value)
                for name, value in zip(self.constraints.metric_names, self.metrics)
            },
            "cost": float(self.cost),
            "constraints": self.constraints.as_dict(),
            "valid": bool(self.valid),
            "feasible": bool(self.feasible),
            "raw_margins": [float(value) for value in self.raw_margins],
            "normalized_margins": [float(value) for value in self.normalized_margins],
            "constraint_margins": [float(value) for value in self.constraint_margins],
            "failed_constraint_count": int(self.failed_constraint_count),
            "total_violation": float(self.total_violation),
            "worst_violation": float(self.worst_violation),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SearchEvaluation":
        return cls(
            action=tuple(int(value) for value in payload["action"]),
            loss=float(payload["loss"]),
            metrics=tuple(float(value) for value in payload["metrics"]),
            cost=float(payload["cost"]),
            constraints=Stage1Constraints.from_dict(payload["constraints"]),
            valid=bool(payload.get("valid", True)),
            metadata=dict(payload.get("metadata") or {}),
        )


def candidate_rank_key(evaluation: SearchEvaluation) -> tuple[Any, ...]:
    """Return a max-is-better feasible-first deterministic rank.

    Valid infeasible candidates use the required least-violating order:
    failed constraint count, total violation, worst violation, cost, then the
    lexicographically smallest categorical chromosome.  Invalid evaluations are
    always below valid evaluations.
    """

    deterministic = tuple(-int(value) for value in evaluation.action)
    if evaluation.feasible:
        sorted_margins = tuple(sorted(float(value) for value in evaluation.normalized_margins))
        return (
            2,
            -float(evaluation.cost),
            *sorted_margins,
            *deterministic,
        )
    if evaluation.valid:
        return (
            1,
            -int(evaluation.failed_constraint_count),
            -float(evaluation.total_violation),
            -float(evaluation.worst_violation),
            -float(evaluation.cost),
            *deterministic,
        )
    return (
        0,
        -int(evaluation.failed_constraint_count),
        -float(evaluation.total_violation),
        -float(evaluation.worst_violation),
        -float(evaluation.cost),
        *deterministic,
    )


@dataclass(frozen=True)
class SearchConfig:
    """Serializable configuration for all three canonical baselines."""

    evaluation_cap: int = 45_664
    seed: int = 42

    bo_initial_design_size: int = 64
    bo_candidate_pool_size: int = 2048
    bo_no_improvement_patience: int = 100
    rf_n_estimators: int = 256
    rf_min_samples_leaf: int = 2
    acquisition_exploration: float = 0.05

    greedy_max_starts: int = 3

    ga_population_size: int = 64
    ga_elite_count: int = 7
    ga_update_generations: int = 800
    ga_no_improvement_patience: int = 5
    ga_stop_on_no_improvement: bool = True
    # Compatibility-only fields retained in serialized historical configs.
    ga_tournament_size: int = 3
    ga_crossover_probability: float = 0.0
    ga_mutation_max_layers: int = 4
    ga_duplicate_attempts: int = 64
    ga_unique_ratio_threshold: float = 0.60
    ga_mean_distance_threshold: float = 2.0
    ga_immigrant_fraction: float = 0.0
    maximin_candidate_pool_size: int = 1024

    def __post_init__(self) -> None:
        positive = (
            "evaluation_cap",
            "bo_initial_design_size",
            "bo_candidate_pool_size",
            "bo_no_improvement_patience",
            "rf_n_estimators",
            "rf_min_samples_leaf",
            "greedy_max_starts",
            "ga_population_size",
            "ga_elite_count",
            "ga_update_generations",
            "ga_no_improvement_patience",
            "ga_tournament_size",
            "ga_mutation_max_layers",
            "ga_duplicate_attempts",
            "maximin_candidate_pool_size",
        )
        for name in positive:
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if type(self.ga_stop_on_no_improvement) is not bool:
            raise ValueError("ga_stop_on_no_improvement must be a boolean")
        if int(self.ga_elite_count) >= int(self.ga_population_size):
            raise ValueError("ga_elite_count must be smaller than ga_population_size")
        for name in (
                "ga_crossover_probability",
                "ga_unique_ratio_threshold",
                "ga_immigrant_fraction",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("acquisition_exploration", "ga_mean_distance_threshold"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")

    @property
    def canonical_ga_target_evaluations(self) -> int:
        return int(
            self.ga_population_size
            + self.ga_update_generations
            * (self.ga_population_size - self.ga_elite_count)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SearchConfig":
        names = set(cls.__dataclass_fields__)
        return cls(**{name: payload[name] for name in names if name in payload})


STAGE1_COMPARATOR_NUM_LAYERS = 12
STAGE1_COMPARATOR_SPLIT = "validation_full"
STAGE1_COMPARATOR_USE_TRAIN = False
STAGE1_COMPARATOR_LOSS_RELATIVE_TOLERANCE = 0.001
STAGE1_COMPARATOR_METRIC_RELATIVE_TOLERANCE = 0.001
STAGE1_COMPARATOR_METRIC_NAMES = ("accuracy", "weighted_f1")


def _stage1_comparator_error(reason: str) -> RuntimeError:
    return RuntimeError(f"Stage-1 comparator protocol violation: {reason}")


def stage1_comparator_search_config(backend: Any) -> SearchConfig:
    """Return the reproducible MRPC search parameters for one comparator."""

    normalized = normalize_search_backend(backend)
    ga_update_generations = 200 if normalized == "coinn_ga" else 800
    evaluation_caps = {
        "bo_rf": 50_000,
        "greedy": 3 ** STAGE1_COMPARATOR_NUM_LAYERS,
        "coinn_ga": 64 + ga_update_generations * (64 - 7),
    }
    return SearchConfig(
        evaluation_cap=evaluation_caps[normalized],
        seed=42,
        bo_initial_design_size=64,
        bo_candidate_pool_size=2_048,
        bo_no_improvement_patience=100,
        rf_n_estimators=128,
        rf_min_samples_leaf=2,
        acquisition_exploration=0.05,
        greedy_max_starts=3,
        ga_population_size=64,
        ga_elite_count=7,
        ga_update_generations=ga_update_generations,
        ga_no_improvement_patience=5,
        ga_stop_on_no_improvement=normalized != "coinn_ga",
        ga_tournament_size=3,
        ga_crossover_probability=0.0,
        ga_mutation_max_layers=4,
        ga_duplicate_attempts=64,
        ga_unique_ratio_threshold=0.60,
        ga_mean_distance_threshold=2.0,
        ga_immigrant_fraction=0.0,
        maximin_candidate_pool_size=1_024,
    )


def validate_stage1_comparator_constraints(
        constraints: Stage1Constraints,
        ) -> dict[str, Any]:
    """Validate the scientific MRPC limits without hashes or authority state."""

    if not isinstance(constraints, Stage1Constraints):
        raise _stage1_comparator_error("constraints are not Stage1Constraints")
    if constraints.metric_names != STAGE1_COMPARATOR_METRIC_NAMES:
        raise _stage1_comparator_error(
            "metric names must be accuracy and weighted_f1"
        )
    if len(constraints.baseline_metrics) != 2:
        raise _stage1_comparator_error("MRPC requires two metrics")
    expected = Stage1Constraints.from_baseline(
        baseline_loss=constraints.baseline_loss,
        baseline_metrics=constraints.baseline_metrics,
        loss_relative_tolerance=STAGE1_COMPARATOR_LOSS_RELATIVE_TOLERANCE,
        metric_relative_tolerance=STAGE1_COMPARATOR_METRIC_RELATIVE_TOLERANCE,
        metric_names=STAGE1_COMPARATOR_METRIC_NAMES,
    )
    if constraints.as_dict() != expected.as_dict():
        raise _stage1_comparator_error(
            "constraint thresholds do not use the exact 0.1% MRPC tolerances"
        )
    return expected.as_dict()


def validate_stage1_comparator_setup(
        *,
        backend: Any,
        config: SearchConfig,
        num_layers: int,
        constraints: Stage1Constraints,
        split: str = STAGE1_COMPARATOR_SPLIT,
        use_train: bool = STAGE1_COMPARATOR_USE_TRAIN,
        ) -> None:
    """Validate reproducible comparator parameters using direct value checks."""

    normalized = normalize_search_backend(backend)
    expected_config = stage1_comparator_search_config(normalized)
    if not isinstance(config, SearchConfig) or config.as_dict() != expected_config.as_dict():
        raise _stage1_comparator_error(
            f"{normalized} search configuration is not the comparator preset"
        )
    if (
            isinstance(num_layers, bool)
            or not isinstance(num_layers, int)
            or num_layers != STAGE1_COMPARATOR_NUM_LAYERS
    ):
        raise _stage1_comparator_error("the action space must have 12 layers")
    if split != STAGE1_COMPARATOR_SPLIT or use_train is not False:
        raise _stage1_comparator_error(
            "evaluation must use validation_full with use_train=False"
        )
    validate_stage1_comparator_constraints(constraints)


@dataclass(frozen=True)
class SearchResult:
    algorithm: str
    config: SearchConfig
    best: SearchEvaluation
    observations: tuple[SearchEvaluation, ...]
    history: tuple[Mapping[str, Any], ...]
    termination_reason: str

    @property
    def evaluation_count(self) -> int:
        return len(self.observations)

    @property
    def unique_evaluation_count(self) -> int:
        return len({item.action for item in self.observations})

    def as_dict(
            self,
            *,
            include_observations: bool = True,
            include_history: bool = True,
            ) -> dict[str, Any]:
        payload = {
            "schema_version": "stage1_gelu_search_result_v1",
            "algorithm": str(self.algorithm),
            "config": self.config.as_dict(),
            "evaluation_count": int(self.evaluation_count),
            "unique_evaluation_count": int(self.unique_evaluation_count),
            "termination_reason": str(self.termination_reason),
            "best": self.best.as_dict(),
        }
        if include_observations:
            payload["observations"] = [
                item.as_dict() for item in self.observations
            ]
        if include_history:
            payload["history"] = [dict(row) for row in self.history]
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SearchResult":
        observations = tuple(
            SearchEvaluation.from_dict(item)
            for item in payload.get("observations", ())
        )
        best_payload = payload.get("best")
        if best_payload is None:
            if not observations:
                raise ValueError("serialized SearchResult contains no best or observations")
            best = max(observations, key=candidate_rank_key)
        else:
            best = SearchEvaluation.from_dict(best_payload)
        return cls(
            algorithm=normalize_search_backend(payload["algorithm"]),
            config=SearchConfig.from_dict(payload.get("config") or {}),
            best=best,
            observations=observations,
            history=tuple(dict(row) for row in payload.get("history", ())),
            termination_reason=str(payload.get("termination_reason", "unknown")),
        )


class _EvaluationCache:
    def __init__(
            self,
            *,
            space: Stage1SearchSpace,
            evaluator: EvaluationFn,
            evaluation_cap: int,
            preload: Iterable[SearchEvaluation] = (),
            checkpoint_callback: Optional[CheckpointCallback] = None,
            incremental_checkpoint_callback: Optional[
                IncrementalCheckpointCallback
            ] = None,
            replay_preload_in_order: bool = True,
            ):
        self.space = space
        self.evaluator = evaluator
        self.cap = min(int(evaluation_cap), int(space.cardinality))
        self.checkpoint_callback = checkpoint_callback
        self.incremental_checkpoint_callback = incremental_checkpoint_callback
        self._by_action: dict[Stage1Action, SearchEvaluation] = {}
        self._ordered: list[SearchEvaluation] = []
        self._best: Optional[SearchEvaluation] = None
        self._replay_preload = tuple(preload)
        self._replay_index = 0
        self._replay_preload_in_order = bool(replay_preload_in_order)
        if self._replay_preload and not self._replay_preload_in_order:
            raise ValueError(
                "unordered Stage-1 preload is unsupported; exact replay is required"
            )
        seen: set[Stage1Action] = set()
        for evaluation in self._replay_preload:
            if not isinstance(evaluation, SearchEvaluation):
                raise TypeError(
                    "preload entries must be SearchEvaluation instances"
                )
            action = self.space.validate(evaluation.action)
            if action in seen:
                raise ValueError(
                    "ordered preload contains a duplicate action"
                )
            seen.add(action)
        if len(self._replay_preload) > self.cap:
            raise ValueError("preloaded observations exceed the evaluation cap")

    @property
    def remaining(self) -> int:
        return max(0, self.cap - len(self._ordered))

    @property
    def observation_count(self) -> int:
        return len(self._ordered)

    @property
    def observations(self) -> tuple[SearchEvaluation, ...]:
        return tuple(self._ordered)

    @property
    def replay_complete(self) -> bool:
        return self._replay_index == len(self._replay_preload)

    def assert_replay_consumed(self) -> None:
        if not self.replay_complete:
            next_action = self._replay_preload[self._replay_index].action
            raise RuntimeError(
                "exact Stage-1 search replay terminated before consuming "
                f"persisted observation {self._replay_index}: {next_action!r}"
            )

    def contains(self, action: Sequence[int]) -> bool:
        return self.space.validate(action) in self._by_action

    def get(self, action: Sequence[int]) -> Optional[SearchEvaluation]:
        return self._by_action.get(self.space.validate(action))

    def _record(self, evaluation: SearchEvaluation) -> None:
        self._by_action[evaluation.action] = evaluation
        self._ordered.append(evaluation)
        if (
                self._best is None
                or candidate_rank_key(evaluation) > candidate_rank_key(self._best)
        ):
            self._best = evaluation

    def add_preloaded(self, evaluation: SearchEvaluation) -> None:
        if not isinstance(evaluation, SearchEvaluation):
            raise TypeError("preload entries must be SearchEvaluation instances")
        action = self.space.validate(evaluation.action)
        previous = self._by_action.get(action)
        if previous is not None:
            if previous.as_dict() != evaluation.as_dict():
                raise ValueError("conflicting preloaded evaluations for one action")
            return
        self._record(evaluation)

    def evaluate(self, action: Sequence[int]) -> SearchEvaluation:
        owned = self.space.validate(action)
        cached = self._by_action.get(owned)
        if cached is not None:
            return cached
        if (
                self._replay_preload_in_order
                and self._replay_index < len(self._replay_preload)
        ):
            evaluation = self._replay_preload[self._replay_index]
            if evaluation.action != owned:
                raise RuntimeError(
                    "exact Stage-1 search replay diverged at persisted "
                    f"observation {self._replay_index}: expected "
                    f"{evaluation.action!r}, requested {owned!r}"
                )
            self._replay_index += 1
            self._record(evaluation)
            return evaluation
        if self.remaining <= 0:
            raise RuntimeError("Stage-1 search evaluation cap exhausted")
        evaluation = self.evaluator(owned)
        if not isinstance(evaluation, SearchEvaluation):
            raise TypeError("Stage-1 search evaluator must return SearchEvaluation")
        if evaluation.action != owned:
            raise ValueError("Stage-1 evaluator returned a different action")
        self._record(evaluation)
        if self.incremental_checkpoint_callback is not None:
            self.incremental_checkpoint_callback(evaluation, len(self._ordered))
        if self.checkpoint_callback is not None:
            self.checkpoint_callback(self.observations)
        return evaluation

    def best(self) -> SearchEvaluation:
        if self._best is None:
            raise RuntimeError("Stage-1 search produced no observations")
        return self._best


# ---------------------------------------------------------------------------
# Deterministic structured and maximin design helpers
# ---------------------------------------------------------------------------


def _candidate_stream(
        space: Stage1SearchSpace,
        rng: np.random.Generator,
        *,
        target: int,
        excluded: set[Stage1Action],
        ) -> list[Stage1Action]:
    candidates: list[Stage1Action] = []
    seen = set(excluded)

    def add(action: Sequence[int]) -> None:
        owned = space.validate(action)
        if owned not in seen and len(candidates) < int(target):
            candidates.append(owned)
            seen.add(owned)

    for offset in GENE_CATEGORIES:
        add(tuple((layer_idx + offset) % 3 for layer_idx in range(space.num_layers)))
        add(tuple((layer_idx // 2 + offset) % 3 for layer_idx in range(space.num_layers)))
        add(tuple((layer_idx // 3 + offset) % 3 for layer_idx in range(space.num_layers)))
    for period in (2, 3, 4, 6):
        for offset in GENE_CATEGORIES:
            add(tuple(((layer_idx % period) + offset) % 3 for layer_idx in range(space.num_layers)))
    attempts = 0
    max_attempts = max(1000, 40 * int(target))
    while len(candidates) < int(target) and attempts < max_attempts:
        attempts += 1
        add(space.random_action(rng))
    if len(candidates) < int(target) and space.cardinality <= 1_000_000:
        for action in space.all_actions():
            add(action)
            if len(candidates) >= int(target):
                break
    return candidates


def _maximin_fill(
        space: Stage1SearchSpace,
        selected: Sequence[Stage1Action],
        *,
        count: int,
        rng: np.random.Generator,
        excluded: Iterable[Stage1Action] = (),
        pool_size: int,
        ) -> list[Stage1Action]:
    chosen = [space.validate(action) for action in selected]
    blocked = {space.validate(action) for action in excluded}
    blocked.update(chosen)
    needed = min(int(count), int(space.cardinality) - len(blocked))
    result: list[Stage1Action] = []
    target_pool = min(
        max(int(pool_size), 16 * needed),
        int(space.cardinality) - len(blocked),
    )
    pool = _candidate_stream(
        space,
        rng,
        target=target_pool,
        excluded=blocked,
    )
    while len(result) < needed:
        if not pool:
            pool = _candidate_stream(
                space,
                rng,
                target=min(
                    max(64, 16 * (needed - len(result))),
                    int(space.cardinality) - len(blocked),
                ),
                excluded=blocked,
            )
            if not pool:
                break
        references = chosen + result
        if references:
            def maximin_key(action: Stage1Action) -> tuple[Any, ...]:
                distances = tuple(
                    sum(int(lhs != rhs) for lhs, rhs in zip(action, other))
                    for other in references
                )
                return (
                    min(distances),
                    sum(distances),
                    tuple(-value for value in action),
                )

            best = max(pool, key=maximin_key)
        else:
            best = min(pool)
        result.append(best)
        blocked.add(best)
        pool.remove(best)
    return result


def structured_maximin_initial_design(
        space: Stage1SearchSpace,
        *,
        count: int,
        seed: int,
        maximin_candidate_pool_size: int = 1024,
        ) -> tuple[Stage1Action, ...]:
    """Build anchors, all one-layer all4 reductions, then maximin actions.

    For ``L=12, count=64`` this is exactly three anchors, 24 one-layer
    reductions from all4, and 37 categorical maximin actions.
    """

    target = min(int(count), int(space.cardinality))
    selected: list[Stage1Action] = []
    seen: set[Stage1Action] = set()

    def add(action: Sequence[int]) -> None:
        owned = space.validate(action)
        if owned not in seen and len(selected) < target:
            selected.append(owned)
            seen.add(owned)

    for anchor in space.anchors:
        add(anchor)
    for layer_idx in range(space.num_layers):
        for replacement in (1, 2):
            candidate = list(space.all4_action)
            candidate[layer_idx] = replacement
            add(candidate)
    if len(selected) < target:
        rng = np.random.default_rng(int(seed))
        for action in _maximin_fill(
                space,
                selected,
                count=target - len(selected),
                rng=rng,
                excluded=(),
                pool_size=int(maximin_candidate_pool_size),
        ):
            add(action)
    return tuple(selected)


def _history_row(
        cache: _EvaluationCache,
        *,
        phase: str,
        iteration: int,
        **extra: Any,
        ) -> dict[str, Any]:
    best = cache.best()
    return {
        "phase": str(phase),
        "iteration": int(iteration),
        "evaluations": int(cache.observation_count),
        "best_action": list(best.action),
        "best_gelu_degrees": list(best.gelu_degrees),
        "best_feasible": bool(best.feasible),
        "best_valid": bool(best.valid),
        "best_cost": float(best.cost),
        "best_failed_constraint_count": int(best.failed_constraint_count),
        "best_total_violation": float(best.total_violation),
        "best_worst_violation": float(best.worst_violation),
        **extra,
    }


# ---------------------------------------------------------------------------
# Deterministic multi-start 1-opt / 2-opt best-improvement search
# ---------------------------------------------------------------------------


def _scan_neighborhood(
        cache: _EvaluationCache,
        actions: Iterable[Stage1Action],
        ) -> tuple[list[SearchEvaluation], bool]:
    evaluations: list[SearchEvaluation] = []
    complete = True
    for action in actions:
        cached = cache.get(action)
        if cached is not None:
            evaluations.append(cached)
            continue
        if cache.remaining <= 0:
            complete = False
            break
        evaluations.append(cache.evaluate(action))
    return evaluations, complete


def _run_greedy(
        *,
        space: Stage1SearchSpace,
        evaluator: EvaluationFn,
        config: SearchConfig,
        preload: Iterable[SearchEvaluation],
        checkpoint_callback: Optional[CheckpointCallback],
        incremental_checkpoint_callback: Optional[
            IncrementalCheckpointCallback
        ],
        replay_preload_in_order: bool = True,
        ) -> SearchResult:
    cache = _EvaluationCache(
        space=space,
        evaluator=evaluator,
        evaluation_cap=config.evaluation_cap,
        preload=preload,
        checkpoint_callback=checkpoint_callback,
        incremental_checkpoint_callback=incremental_checkpoint_callback,
        replay_preload_in_order=replay_preload_in_order,
    )
    history: list[dict[str, Any]] = []
    all_verified = True
    starts = space.anchors[:min(int(config.greedy_max_starts), len(space.anchors))]
    for start_idx, start in enumerate(starts):
        if cache.get(start) is None and cache.remaining <= 0:
            all_verified = False
            break
        current = cache.evaluate(start)
        iteration = 0
        history.append(_history_row(
            cache,
            phase="start",
            iteration=iteration,
            start_index=int(start_idx),
            current_action=list(current.action),
        ))
        while True:
            iteration += 1
            one_evaluations, complete = _scan_neighborhood(
                cache, space.one_opt_neighbors(current.action),
            )
            if not complete:
                all_verified = False
                history.append(_history_row(
                    cache,
                    phase="one_opt",
                    iteration=iteration,
                    start_index=int(start_idx),
                    neighborhood_complete=False,
                    accepted=False,
                ))
                break
            one_best = max(one_evaluations, key=candidate_rank_key, default=current)
            if candidate_rank_key(one_best) > candidate_rank_key(current):
                current = one_best
                history.append(_history_row(
                    cache,
                    phase="one_opt",
                    iteration=iteration,
                    start_index=int(start_idx),
                    neighborhood_complete=True,
                    accepted=True,
                    current_action=list(current.action),
                ))
                continue

            two_evaluations, complete = _scan_neighborhood(
                cache, space.two_opt_neighbors(current.action),
            )
            if not complete:
                all_verified = False
                history.append(_history_row(
                    cache,
                    phase="two_opt",
                    iteration=iteration,
                    start_index=int(start_idx),
                    neighborhood_complete=False,
                    accepted=False,
                ))
                break
            two_best = max(two_evaluations, key=candidate_rank_key, default=current)
            if candidate_rank_key(two_best) > candidate_rank_key(current):
                current = two_best
                history.append(_history_row(
                    cache,
                    phase="two_opt",
                    iteration=iteration,
                    start_index=int(start_idx),
                    neighborhood_complete=True,
                    accepted=True,
                    return_to_one_opt=True,
                    current_action=list(current.action),
                ))
                continue
            history.append(_history_row(
                cache,
                phase="verified_local_optimum",
                iteration=iteration,
                start_index=int(start_idx),
                one_opt_verified=True,
                two_opt_verified=True,
                current_action=list(current.action),
            ))
            break
        if not all_verified and cache.remaining <= 0:
            break
    cache.assert_replay_consumed()
    termination = (
        "verified_local_optimum"
        if all_verified
        else "evaluation_cap"
    )
    return SearchResult(
        algorithm="greedy",
        config=config,
        best=cache.best(),
        observations=cache.observations,
        history=tuple(history),
        termination_reason=termination,
    )


# ---------------------------------------------------------------------------
# Constrained categorical random-forest Bayesian optimization
# ---------------------------------------------------------------------------


def _default_surrogate_factory(config: SearchConfig) -> SurrogateFactory:
    def factory(seed: int) -> Any:
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError as exc:  # pragma: no cover - depends on runtime image
            raise RuntimeError(
                "Stage-1 BO-RF requires scikit-learn or an injected "
                "surrogate_factory"
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


def _tree_predictions(model: Any, features: np.ndarray) -> np.ndarray:
    estimators = tuple(getattr(model, "estimators_", ()) or ())
    if estimators:
        predictions = [
            np.asarray(estimator.predict(features), dtype=float)
            for estimator in estimators
        ]
        stacked = np.stack(predictions, axis=0)
    else:
        prediction = np.asarray(model.predict(features), dtype=float)
        if prediction.ndim == 1:
            prediction = prediction.reshape(-1, 1)
        stacked = prediction.reshape(1, prediction.shape[0], prediction.shape[1])
    if stacked.ndim == 2:
        stacked = stacked[:, :, np.newaxis]
    return stacked


def _bo_candidate_pool(
        *,
        space: Stage1SearchSpace,
        cache: _EvaluationCache,
        rng: np.random.Generator,
        pool_size: int,
        ) -> list[Stage1Action]:
    remaining_space = int(space.cardinality) - cache.observation_count
    target = min(int(pool_size), remaining_space)
    if target <= 0:
        return []
    if space.cardinality <= max(10_000, 4 * target):
        return [
            action
            for action in space.all_actions(max_cardinality=max(10_000, space.cardinality))
            if not cache.contains(action)
        ]

    pool: list[Stage1Action] = []
    seen: set[Stage1Action] = set(item.action for item in cache.observations)

    def add(action: Sequence[int]) -> None:
        owned = space.validate(action)
        if owned not in seen and len(pool) < target:
            pool.append(owned)
            seen.add(owned)

    ranked = heapq.nlargest(
        min(16, cache.observation_count),
        cache.observations,
        key=candidate_rank_key,
    )
    for observation in ranked:
        for neighbor in space.one_opt_neighbors(observation.action):
            add(neighbor)
        for neighbor in space.two_opt_neighbors(observation.action):
            add(neighbor)
            if len(pool) >= target // 2:
                break
        if len(pool) >= target // 2:
            break
    for action in _candidate_stream(
            space,
            rng,
            target=target - len(pool),
            excluded=seen,
    ):
        add(action)
    return pool


def _cost_hint(evaluator: EvaluationFn, action: Stage1Action) -> float:
    """Return exact adapter cost when available, otherwise a monotonic proxy."""

    cost_for_action = getattr(evaluator, "cost_for_action", None)
    if callable(cost_for_action):
        return float(cost_for_action(action))
    # Categories 0/1/2 decode to degrees 4/2/1, so larger category sums are a
    # deterministic lower-cost proxy when a synthetic evaluator exposes no
    # separate cost-only seam.
    return -float(sum(action))


def _bo_acquisition_key(
        *,
        has_feasible_incumbent: bool,
        probability_of_feasibility: float,
        expected_improvement: float,
        expected_failed_constraints: float,
        expected_total_violation: float,
        expected_worst_violation: float,
        exploration_tiebreak: float,
        objective_tiebreak: float,
        deterministic_tiebreak: tuple[int, ...],
        ) -> tuple[Any, ...]:
    if has_feasible_incumbent:
        return (
            float(probability_of_feasibility) * float(expected_improvement),
            float(probability_of_feasibility),
            float(exploration_tiebreak),
            float(objective_tiebreak),
            deterministic_tiebreak,
        )
    return (
        -float(expected_failed_constraints),
        -float(expected_total_violation),
        -float(expected_worst_violation),
        float(probability_of_feasibility),
        float(exploration_tiebreak),
        float(objective_tiebreak),
        deterministic_tiebreak,
    )


def _run_bo_rf(
        *,
        space: Stage1SearchSpace,
        evaluator: EvaluationFn,
        config: SearchConfig,
        surrogate_factory: Optional[SurrogateFactory],
        preload: Iterable[SearchEvaluation],
        checkpoint_callback: Optional[CheckpointCallback],
        incremental_checkpoint_callback: Optional[
            IncrementalCheckpointCallback
        ],
        ) -> SearchResult:
    rng = np.random.default_rng(int(config.seed))
    cache = _EvaluationCache(
        space=space,
        evaluator=evaluator,
        evaluation_cap=config.evaluation_cap,
        preload=preload,
        checkpoint_callback=checkpoint_callback,
        incremental_checkpoint_callback=incremental_checkpoint_callback,
        replay_preload_in_order=True,
    )
    history: list[dict[str, Any]] = []
    design = structured_maximin_initial_design(
        space,
        count=min(int(config.bo_initial_design_size), cache.cap),
        seed=int(config.seed),
        maximin_candidate_pool_size=int(config.maximin_candidate_pool_size),
    )
    for action in design:
        if cache.get(action) is None and cache.remaining <= 0:
            break
        cache.evaluate(action)
    history.append(_history_row(cache, phase="structured_maximin_initial_design", iteration=0))

    factory = surrogate_factory or _default_surrogate_factory(config)
    incumbent_key = candidate_rank_key(cache.best())
    no_improvement = 0
    iteration = 0
    termination = "candidate_space_exhausted"
    while cache.remaining > 0:
        if no_improvement >= int(config.bo_no_improvement_patience):
            termination = "no_improvement_convergence"
            break
        iteration += 1
        observations = cache.observations
        features = space.one_hot([item.action for item in observations])
        targets = np.asarray(
            [item.constraint_margins for item in observations],
            dtype=float,
        )
        model = factory(int(config.seed) + iteration)
        model.fit(features, targets)
        pool = _bo_candidate_pool(
            space=space,
            cache=cache,
            rng=rng,
            pool_size=int(config.bo_candidate_pool_size),
        )
        if not pool:
            termination = "candidate_space_exhausted"
            break
        tree_predictions = _tree_predictions(model, space.one_hot(pool))
        if tree_predictions.ndim != 3 or tree_predictions.shape[2] != targets.shape[1]:
            raise RuntimeError(
                "Stage-1 BO-RF surrogate must predict one margin for loss and "
                "each active metric"
            )
        feasibility_probability = np.mean(
            np.all(tree_predictions >= 0.0, axis=2), axis=0,
        )
        predicted_std = tree_predictions.std(axis=0)
        predicted_violations = np.maximum(0.0, -tree_predictions)
        predicted_failed = np.mean(
            np.sum(tree_predictions < 0.0, axis=2), axis=0,
        )
        predicted_total_violation = np.mean(
            np.sum(predicted_violations, axis=2), axis=0,
        )
        predicted_worst_violation = np.mean(
            np.max(predicted_violations, axis=2), axis=0,
        )
        feasible_observations = [item for item in observations if item.feasible]
        incumbent_cost = min(
            (_cost_hint(evaluator, item.action) for item in feasible_observations),
            default=math.inf,
        )
        candidate_costs = [_cost_hint(evaluator, action) for action in pool]
        acquisition: list[float] = []
        acquisition_keys: list[tuple[Any, ...]] = []
        for index, action in enumerate(pool):
            probability = float(feasibility_probability[index])
            uncertainty = float(np.mean(predicted_std[index]))
            candidate_cost = float(candidate_costs[index])
            deterministic = tuple(-value for value in action)
            improvement = max(0.0, incumbent_cost - candidate_cost)
            value = (
                probability * improvement
                if feasible_observations
                else -float(predicted_failed[index])
            )
            key = _bo_acquisition_key(
                has_feasible_incumbent=bool(feasible_observations),
                probability_of_feasibility=probability,
                expected_improvement=improvement,
                expected_failed_constraints=float(predicted_failed[index]),
                expected_total_violation=float(
                    predicted_total_violation[index]
                ),
                expected_worst_violation=float(
                    predicted_worst_violation[index]
                ),
                exploration_tiebreak=(
                    float(config.acquisition_exploration) * uncertainty
                ),
                objective_tiebreak=-candidate_cost,
                deterministic_tiebreak=deterministic,
            )
            acquisition.append(float(value))
            acquisition_keys.append(key)
        selected_index = max(
            range(len(pool)), key=acquisition_keys.__getitem__,
        )
        selected = cache.evaluate(pool[selected_index])
        new_key = candidate_rank_key(cache.best())
        improved = new_key > incumbent_key
        if improved:
            incumbent_key = new_key
            no_improvement = 0
        else:
            no_improvement += 1
        history.append(_history_row(
            cache,
            phase="feasibility_aware_acquisition",
            iteration=iteration,
            selected_action=list(selected.action),
            acquisition=float(acquisition[selected_index]),
            acquisition_mode=(
                "probability_of_feasibility_times_expected_improvement"
                if feasible_observations
                else "lexicographic_predicted_violation"
            ),
            predicted_feasibility=float(feasibility_probability[selected_index]),
            predicted_failed_constraints=float(predicted_failed[selected_index]),
            predicted_total_violation=float(
                predicted_total_violation[selected_index]
            ),
            predicted_worst_violation=float(
                predicted_worst_violation[selected_index]
            ),
            no_improvement=int(no_improvement),
            improved=bool(improved),
        ))
    if cache.remaining <= 0:
        termination = "evaluation_cap"
    cache.assert_replay_consumed()
    return SearchResult(
        algorithm="bo_rf",
        config=config,
        best=cache.best(),
        observations=cache.observations,
        history=tuple(history),
        termination_reason=termination,
    )


# ---------------------------------------------------------------------------
# Structured elitist Stage-1 GA (P64/E7, 800 update generations by default)
# ---------------------------------------------------------------------------


def _population_diversity(
        space: Stage1SearchSpace,
        population: Sequence[SearchEvaluation],
        ) -> tuple[float, float]:
    if not population:
        return 0.0, 0.0
    unique_ratio = len({item.action for item in population}) / float(len(population))
    distances = [
        space.hamming_distance(population[first].action, population[second].action)
        for first in range(len(population))
        for second in range(first + 1, len(population))
    ]
    mean_distance = float(np.mean(distances)) if distances else 0.0
    return float(unique_ratio), mean_distance


def _select_hamming_diverse_elites(
        space: Stage1SearchSpace,
        population: Sequence[SearchEvaluation],
        elite_count: int,
        ) -> list[SearchEvaluation]:
    """Keep the best incumbent, then prefer feasible elites at distance >= 2."""

    target = min(int(elite_count), len(population))
    if target <= 0:
        return []
    ranked = sorted(population, key=candidate_rank_key, reverse=True)
    feasible = [item for item in ranked if item.feasible]
    pools = (feasible, ranked)
    selected: list[SearchEvaluation] = []
    selected_actions: set[Stage1Action] = set()
    for pool in pools:
        remaining = [
            item for item in pool if item.action not in selected_actions
        ]
        while remaining and len(selected) < target:
            if not selected:
                chosen = remaining[0]
            else:
                distance_two = [
                    item for item in remaining
                    if all(
                        space.hamming_distance(item.action, owned.action) >= 2
                        for owned in selected
                    )
                ]
                distance_one = [
                    item for item in remaining
                    if all(
                        space.hamming_distance(item.action, owned.action) >= 1
                        for owned in selected
                    )
                ]
                chosen = (distance_two or distance_one or remaining)[0]
            selected.append(chosen)
            selected_actions.add(chosen.action)
            remaining = [
                item for item in remaining if item.action != chosen.action
            ]
        if len(selected) >= target:
            break
    return selected


def _ga_parent_weights(
        population: Sequence[SearchEvaluation],
        ) -> tuple[float, ...]:
    """Return feasibility-aware positive COINN parent fitness weights."""

    if not population:
        raise ValueError("GA parent population must not be empty")

    def inverse_penalty(item: SearchEvaluation) -> float:
        components = (
            1.0,
            float(not item.valid),
            float(item.failed_constraint_count),
            *item.violations,
            float(item.worst_violation),
        )
        scale = max(components)
        if not math.isfinite(scale):
            return math.nextafter(0.0, 1.0)
        scaled_total = sum(value / scale for value in components)
        return max(
            (1.0 / scale) / scaled_total,
            math.nextafter(0.0, 1.0),
        )

    feasible_weights = tuple(
        1.0 / max(float(item.cost), _EPS) if item.feasible else 0.0
        for item in population
    )
    infeasible_weights = tuple(
        0.0 if item.feasible else inverse_penalty(item)
        for item in population
    )
    feasible_total = float(sum(feasible_weights))
    infeasible_total = float(sum(infeasible_weights))
    if feasible_total > 0.0 and infeasible_total > 0.0:
        return tuple(
            0.90 * (feasible / feasible_total)
            + 0.10 * (infeasible / infeasible_total)
            for feasible, infeasible in zip(  # noqa: B905 - Python 3.9
                feasible_weights, infeasible_weights,
            )
        )
    if feasible_total > 0.0:
        return feasible_weights
    return infeasible_weights


def _tournament(
        population: Sequence[SearchEvaluation],
        rng: np.random.Generator,
        tournament_size: int,
        *,
        diverse_from: Optional[Stage1Action] = None,
        ) -> SearchEvaluation:
    """Compatibility-named fitness-proportional COINN parent selector."""

    del tournament_size, diverse_from
    weights = np.asarray(_ga_parent_weights(population), dtype=float)
    probabilities = weights / float(np.sum(weights))
    index = int(rng.choice(len(population), p=probabilities))
    return population[index]


def _mutate_action(
        space: Stage1SearchSpace,
        action: Stage1Action,
        rng: np.random.Generator,
        *,
        max_layers: int,
        force: bool,
        ) -> Stage1Action:
    candidate = list(space.validate(action))
    probability = 1.0 / float(space.num_layers)
    selected = [
        layer_idx
        for layer_idx in range(space.num_layers)
        if float(rng.random()) < probability
    ]
    if len(selected) > int(max_layers):
        selected = sorted(
            int(value)
            for value in rng.choice(selected, size=int(max_layers), replace=False)
        )
    if force and not selected:
        selected = [int(rng.integers(space.num_layers))]
    for layer_idx in selected:
        current = candidate[layer_idx]
        alternatives = tuple(
            value for value in GENE_CATEGORIES
            if value != current and abs(value - current) == 1
        )
        candidate[layer_idx] = alternatives[
            int(rng.integers(len(alternatives)))
        ]
    return tuple(candidate)


def _crossover(
        space: Stage1SearchSpace,
        first: Stage1Action,
        second: Stage1Action,
        rng: np.random.Generator,
        ) -> Stage1Action:
    lhs = space.validate(first)
    rhs = space.validate(second)
    if space.num_layers == 1:
        return lhs if float(rng.random()) < 0.5 else rhs
    if float(rng.random()) < 0.5:
        # Atomic two-point crossover: boundaries are between whole layer genes.
        boundaries = sorted(
            int(value)
            for value in rng.choice(space.num_layers + 1, size=2, replace=False)
        )
        start, stop = boundaries
        child = list(lhs)
        child[start:stop] = rhs[start:stop]
        return tuple(child)
    mask = rng.random(space.num_layers) < 0.5
    return tuple(rhs[index] if mask[index] else lhs[index] for index in range(space.num_layers))


def _maximin_immigrant(
        *,
        space: Stage1SearchSpace,
        cache: _EvaluationCache,
        references: Sequence[Stage1Action],
        blocked: set[Stage1Action],
        rng: np.random.Generator,
        pool_size: int,
        ) -> Optional[Stage1Action]:
    excluded = set(blocked)
    excluded.update(item.action for item in cache.observations)
    immigrants = _maximin_fill(
        space,
        references,
        count=1,
        rng=rng,
        excluded=excluded,
        pool_size=int(pool_size),
    )
    return None if not immigrants else immigrants[0]


def _breed_unique_child(
        *,
        space: Stage1SearchSpace,
        population: Sequence[SearchEvaluation],
        cache: _EvaluationCache,
        blocked: set[Stage1Action],
        rng: np.random.Generator,
        config: SearchConfig,
        ) -> tuple[Optional[Stage1Action], bool]:
    for _attempt in range(int(config.ga_duplicate_attempts)):
        parent = _tournament(
            population, rng, int(config.ga_tournament_size),
        )
        child = _mutate_action(
            space,
            parent.action,
            rng,
            max_layers=int(config.ga_mutation_max_layers),
            force=True,
        )
        if child in blocked or cache.contains(child):
            child = _mutate_action(
                space,
                parent.action,
                rng,
                max_layers=int(config.ga_mutation_max_layers),
                force=True,
            )
        if child not in blocked and not cache.contains(child):
            return child, False

    # A finite random retry budget proves only that those draws collided, not
    # that the population has no unseen mesh neighbor. Preserve the random
    # path above, then fail closed only after checking every allowed adjacent
    # mutation deterministically.
    max_changed_layers = min(
        int(config.ga_mutation_max_layers),
        int(space.num_layers),
    )
    for changed_layer_count in range(1, max_changed_layers + 1):
        for parent in population:
            parent_action = space.validate(parent.action)
            for layer_indices in itertools.combinations(
                    range(space.num_layers), changed_layer_count,
            ):
                adjacent_values = [
                    tuple(
                        value for value in GENE_CATEGORIES
                        if abs(value - parent_action[layer_idx]) == 1
                    )
                    for layer_idx in layer_indices
                ]
                for replacements in itertools.product(*adjacent_values):
                    candidate = list(parent_action)
                    for layer_idx, replacement in zip(  # noqa: B905 - Python 3.9
                            layer_indices, replacements,
                    ):
                        candidate[layer_idx] = replacement
                    child = tuple(candidate)
                    if child not in blocked and not cache.contains(child):
                        return child, False
    return None, False


def _run_coinn_ga(
        *,
        space: Stage1SearchSpace,
        evaluator: EvaluationFn,
        config: SearchConfig,
        preload: Iterable[SearchEvaluation],
        checkpoint_callback: Optional[CheckpointCallback],
        incremental_checkpoint_callback: Optional[
            IncrementalCheckpointCallback
        ],
        ) -> SearchResult:
    rng = np.random.default_rng(int(config.seed))
    cache = _EvaluationCache(
        space=space,
        evaluator=evaluator,
        evaluation_cap=config.evaluation_cap,
        preload=preload,
        checkpoint_callback=checkpoint_callback,
        incremental_checkpoint_callback=incremental_checkpoint_callback,
        replay_preload_in_order=True,
    )
    population_size = min(
        int(config.ga_population_size),
        int(space.cardinality),
        int(cache.cap),
    )
    elite_count = min(int(config.ga_elite_count), max(0, population_size - 1))
    initial_actions = structured_maximin_initial_design(
        space,
        count=population_size,
        seed=int(config.seed),
        maximin_candidate_pool_size=int(config.maximin_candidate_pool_size),
    )
    population: list[SearchEvaluation] = []
    for action in initial_actions:
        if cache.get(action) is None and cache.remaining <= 0:
            break
        population.append(cache.evaluate(action))
    if not population:
        raise RuntimeError("Stage-1 GA could not evaluate its initial population")
    history: list[dict[str, Any]] = []
    unique_ratio, mean_distance = _population_diversity(space, population)
    history.append(_history_row(
        cache,
        phase="initial_population",
        iteration=0,
        generation=0,
        population_size=len(population),
        anchors=min(3, len(initial_actions)),
        one_layer_reductions=min(2 * space.num_layers, max(0, len(initial_actions) - 3)),
        maximin_count=max(0, len(initial_actions) - 3 - 2 * space.num_layers),
        unique_ratio=float(unique_ratio),
        mean_pairwise_distance=float(mean_distance),
    ))

    termination = "completed_generations"
    completed_generations = 0
    incumbent_key = candidate_rank_key(max(
        population, key=candidate_rank_key,
    ))
    no_improvement_generations = 0
    offspring_target = population_size - elite_count
    for generation in range(1, int(config.ga_update_generations) + 1):
        if cache.remaining < offspring_target:
            termination = "evaluation_cap"
            break
        elites = _select_hamming_diverse_elites(
            space, population, elite_count,
        )
        actions: list[Stage1Action] = []
        blocked: set[Stage1Action] = set()
        while len(actions) < offspring_target:
            child, _used_immigrant = _breed_unique_child(
                space=space,
                population=population,
                cache=cache,
                blocked=blocked,
                rng=rng,
                config=config,
            )
            if child is None:
                break
            actions.append(child)
            blocked.add(child)
        if len(actions) < offspring_target:
            termination = "candidate_space_exhausted"
            break
        offspring = [cache.evaluate(action) for action in actions]
        population = elites + offspring
        completed_generations = generation
        next_incumbent_key = candidate_rank_key(max(
            population, key=candidate_rank_key,
        ))
        improved = next_incumbent_key > incumbent_key
        no_improvement_generations = (
            0 if improved else no_improvement_generations + 1
        )
        incumbent_key = next_incumbent_key
        next_unique_ratio, next_mean_distance = _population_diversity(space, population)
        history.append(_history_row(
            cache,
            phase="elitist_update",
            iteration=generation,
            generation=generation,
            elite_count=len(elites),
            feasible_elite_count=sum(item.feasible for item in elites),
            elite_actions=[list(item.action) for item in elites],
            elite_policy="best_incumbent_then_hamming_distance_2",
            new_unique_evaluations=len(offspring),
            improved=bool(improved),
            no_improvement_generations=int(no_improvement_generations),
            diversity_triggered=False,
            scheduled_immigrants=0,
            fallback_immigrants=0,
            replaced_worst_nonelite_actions=[],
            immigrant_actions=[],
            unique_ratio=float(next_unique_ratio),
            mean_pairwise_distance=float(next_mean_distance),
            post_update_unique_ratio=float(next_unique_ratio),
            post_update_mean_pairwise_distance=float(next_mean_distance),
        ))
        if (
            config.ga_stop_on_no_improvement
            and no_improvement_generations >= int(
                config.ga_no_improvement_patience
            )
        ):
            termination = "ga_no_incumbent_improvement"
            break
    if (
            termination == "completed_generations"
            and completed_generations < int(config.ga_update_generations)
            and cache.remaining <= 0
    ):
        termination = "evaluation_cap"
    cache.assert_replay_consumed()
    return SearchResult(
        algorithm="coinn_ga",
        config=config,
        best=cache.best(),
        observations=cache.observations,
        history=tuple(history),
        termination_reason=termination,
    )


def run_search(
        backend: str,
        space: Stage1SearchSpace,
        evaluator: EvaluationFn,
        config: Optional[SearchConfig] = None,
        *,
        surrogate_factory: Optional[SurrogateFactory] = None,
        preload: Iterable[SearchEvaluation] = (),
        checkpoint_callback: Optional[CheckpointCallback] = None,
        incremental_checkpoint_callback: Optional[
            IncrementalCheckpointCallback
        ] = None,
        replay_greedy_preload_in_order: bool = True,
        ) -> SearchResult:
    """Run one canonical non-RL Stage-1 search backend."""

    normalized = normalize_search_backend(backend)
    cfg = config or SearchConfig()
    preload_tuple = tuple(preload)
    if normalized == "greedy":
        return _run_greedy(
            space=space,
            evaluator=evaluator,
            config=cfg,
            preload=preload_tuple,
            checkpoint_callback=checkpoint_callback,
            incremental_checkpoint_callback=incremental_checkpoint_callback,
            replay_preload_in_order=replay_greedy_preload_in_order,
        )
    if normalized == "bo_rf":
        return _run_bo_rf(
            space=space,
            evaluator=evaluator,
            config=cfg,
            surrogate_factory=surrogate_factory,
            preload=preload_tuple,
            checkpoint_callback=checkpoint_callback,
            incremental_checkpoint_callback=incremental_checkpoint_callback,
        )
    if normalized == "coinn_ga":
        return _run_coinn_ga(
            space=space,
            evaluator=evaluator,
            config=cfg,
            preload=preload_tuple,
            checkpoint_callback=checkpoint_callback,
            incremental_checkpoint_callback=incremental_checkpoint_callback,
        )
    raise AssertionError(f"unhandled Stage-1 search backend {normalized}")


__all__ = [
    "CheckpointCallback",
    "EvaluationFn",
    "FIXED_SOFTMAX_DEGREE",
    "GENE_CATEGORIES",
    "GELU_DEGREES",
    "SUPPORTED_SEARCH_BACKENDS",
    "SearchConfig",
    "SearchEvaluation",
    "SearchResult",
    "Stage1Action",
    "Stage1Constraints",
    "Stage1SearchSpace",
    "SurrogateFactory",
    "candidate_rank_key",
    "normalize_search_backend",
    "run_search",
    "stage1_comparator_search_config",
    "structured_maximin_initial_design",
    "validate_stage1_comparator_constraints",
    "validate_stage1_comparator_setup",
]
