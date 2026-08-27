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
import itertools
import math
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence

import numpy as np

from rfr.preparation.data.protocol import TRAIN_PROBE_SPLIT


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

    seed: int = 42

    bo_initial_design_size: int = 64
    bo_candidate_pool_size: int = 2048
    bo_no_improvement_patience: int = 1_000
    rf_n_estimators: int = 128
    rf_min_samples_leaf: int = 2
    acquisition_exploration: float = 0.05

    greedy_max_starts: int = 3
    greedy_no_improvement_rounds: int = 1

    ga_population_size: int = 64
    ga_elite_count: int = 7
    ga_update_generations: int = 200

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
            "bo_initial_design_size",
            "bo_candidate_pool_size",
            "bo_no_improvement_patience",
            "rf_n_estimators",
            "rf_min_samples_leaf",
            "greedy_max_starts",
            "greedy_no_improvement_rounds",
            "ga_population_size",
            "ga_elite_count",
            "ga_update_generations",
            "ga_tournament_size",
            "ga_mutation_max_layers",
            "ga_duplicate_attempts",
            "maximin_candidate_pool_size",
        )
        for name in positive:
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
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
STAGE1_COMPARATOR_SPLIT = TRAIN_PROBE_SPLIT
STAGE1_COMPARATOR_USE_TRAIN = False
STAGE1_COMPARATOR_LOSS_RELATIVE_TOLERANCE = 0.001
STAGE1_COMPARATOR_METRIC_RELATIVE_TOLERANCE = 0.001
STAGE1_COMPARATOR_METRIC_NAMES = ("accuracy", "weighted_f1")


def _stage1_comparator_error(reason: str) -> RuntimeError:
    return RuntimeError(f"Stage-1 comparator protocol violation: {reason}")


def stage1_comparator_search_config(
        backend: Any,
        *,
        bo_no_improvement_patience: int = 1_000,
        greedy_no_improvement_rounds: int = 1,
        ga_update_generations: int = 200,
        ) -> SearchConfig:
    """Return the reproducible MRPC search parameters for one comparator."""

    normalize_search_backend(backend)
    return SearchConfig(
        seed=42,
        bo_initial_design_size=64,
        bo_candidate_pool_size=2_048,
        bo_no_improvement_patience=int(bo_no_improvement_patience),
        rf_n_estimators=128,
        rf_min_samples_leaf=2,
        acquisition_exploration=0.05,
        greedy_max_starts=3,
        greedy_no_improvement_rounds=int(greedy_no_improvement_rounds),
        ga_population_size=64,
        ga_elite_count=7,
        ga_update_generations=int(ga_update_generations),
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
    expected_config = stage1_comparator_search_config(
        normalized,
        bo_no_improvement_patience=int(config.bo_no_improvement_patience),
        greedy_no_improvement_rounds=int(config.greedy_no_improvement_rounds),
        ga_update_generations=int(config.ga_update_generations),
    )
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
            "evaluation must use train_probe with use_train=False"
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
            preload: Iterable[SearchEvaluation] = (),
            checkpoint_callback: Optional[CheckpointCallback] = None,
            incremental_checkpoint_callback: Optional[
                IncrementalCheckpointCallback
            ] = None,
            replay_preload_in_order: bool = True,
            ):
        self.space = space
        self.evaluator = evaluator
        self.cap = int(space.cardinality)
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
            raise ValueError("preloaded observations exceed the action space")

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
            raise RuntimeError("Stage-1 search action space exhausted")
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
        from rfr.search.comparators.greedy.stage1 import run

        return run(
            space=space,
            evaluator=evaluator,
            config=cfg,
            preload=preload_tuple,
            checkpoint_callback=checkpoint_callback,
            incremental_checkpoint_callback=incremental_checkpoint_callback,
            replay_preload_in_order=replay_greedy_preload_in_order,
        )
    if normalized == "bo_rf":
        from rfr.search.comparators.bo_rf.stage1 import run

        return run(
            space=space,
            evaluator=evaluator,
            config=cfg,
            surrogate_factory=surrogate_factory,
            preload=preload_tuple,
            checkpoint_callback=checkpoint_callback,
            incremental_checkpoint_callback=incremental_checkpoint_callback,
        )
    if normalized == "coinn_ga":
        from rfr.search.comparators.coinn_ga.stage1 import run

        return run(
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
