"""Constrained non-RL baselines for the canonical Stage-2 layerwise action.

The public runtime action remains one ``(Block4 fusion, H/M/L preset)`` row per
Transformer layer. Search operators encode each row as one atomic six-valued
categorical gene so crossover, mutation, and neighborhoods never split a
layer's coupled decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
import heapq
import itertools
import math
import re
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence

import numpy as np

from .layerwise_action import (
    LAYER_GENE_CARDINALITY,
    compute_variable_cost_from_action_matrix,
    decode_layer_gene,
    decode_layerwise_action_genes,
    encode_layerwise_action_matrix,
)
from .precision_presets import validate_communication_importance_ratio


ActionMatrix = tuple[tuple[int, int], ...]
EvaluationFn = Callable[[ActionMatrix], "SearchEvaluation"]
SurrogateFactory = Callable[[int], Any]
CheckpointCallback = Callable[[tuple["SearchEvaluation", ...]], None]
IncrementalCheckpointCallback = Callable[["SearchEvaluation", int], None]

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
_GA_OFFSPRING_ATTEMPTS = 64


def normalize_search_backend(value: Any) -> str:
    normalized = re.sub(
        r"[^a-z0-9]+", "_", str(value or "ppo").strip().lower(),
    ).strip("_")
    aliases = {
        "policy": "ppo",
        "policy_gradient": "ppo",
        "bayes": "bo_rf",
        "bayes_rf": "bo_rf",
        "bayesian": "bo_rf",
        "bayesian_optimization": "bo_rf",
        "bayesian_rf": "bo_rf",
        "bo": "bo_rf",
        "borf": "bo_rf",
        "random_forest_bo": "bo_rf",
        "rf_bo": "bo_rf",
        "smac": "bo_rf",
        "smac_rf": "bo_rf",
        "greedy_search": "greedy",
        "hill_climb": "greedy",
        "hill_climbing": "greedy",
        "coinn": "coinn_ga",
        "coinnga": "coinn_ga",
        "coinn_style_ga": "coinn_ga",
        "ga": "coinn_ga",
        "genetic": "coinn_ga",
        "genetic_algorithm": "coinn_ga",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_SEARCH_BACKENDS:
        raise ValueError(
            f"unsupported Stage-2 search backend {value!r}; expected one of "
            f"{SUPPORTED_SEARCH_BACKENDS}"
        )
    return normalized


def validate_comparator_scientific_parameters(
    *,
    communication_importance_ratio: float,
    truncation_backend: str,
    truncation_ring_bits: int,
    truncation_source_fractional_bits: int,
) -> None:
    """Reject comparator settings that alter the locked scientific protocol."""
    parameters = (
        float(communication_importance_ratio),
        str(truncation_backend or "binary").strip().lower(),
        int(truncation_ring_bits),
        int(truncation_source_fractional_bits),
    )
    if parameters != (1.0, "binary", 43, 24):
        raise ValueError(
            "two-stage comparators require canonical Stage-2 scientific "
            "parameters (communication importance ratio 1.0, binary truncation, "
            "ring bits 43, source fractional bits 24)"
        )


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

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConstraintLimits":
        return cls(**{
            name: float(payload[name])
            for name in cls.__dataclass_fields__
        })


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

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SearchMetrics":
        return cls(**{
            name: float(payload[name])
            for name in CONSTRAINT_NAMES
        })


def _owned_action_matrix(action_matrix: Sequence[Sequence[int]]) -> ActionMatrix:
    rows = tuple(tuple(int(value) for value in row) for row in action_matrix)
    if any(len(row) != 2 for row in rows):
        raise ValueError("action_matrix rows must contain two coordinates")
    return tuple((row[0], row[1]) for row in rows)


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

    @cached_property
    def raw_margins(self) -> tuple[float, ...]:
        return (
            float(self.limits.loss_max - self.metrics.loss_mean),
            float(self.metrics.metric1_mean - self.limits.metric1_min),
            float(self.metrics.metric2_mean - self.limits.metric2_min),
            float(self.limits.loss_std_max - self.metrics.loss_std),
            float(self.limits.metric1_std_max - self.metrics.metric1_std),
            float(self.limits.metric2_std_max - self.metrics.metric2_std),
        )

    @cached_property
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

    @cached_property
    def inference_performed(self) -> bool:
        """Whether this unique candidate consumed one model-inference quota."""
        return bool(self.metadata.get("inference_performed", True))

    @cached_property
    def feasible(self) -> bool:
        return bool(
            self.valid
            and all(value >= -_EPS for value in self.constraint_margins)
        )

    @cached_property
    def normalized_violation(self) -> float:
        return float(sum(
            max(0.0, -value) for value in self.constraint_margins
        ))

    @cached_property
    def failed_constraint_count(self) -> int:
        return int(sum(
            value < -_EPS for value in self.constraint_margins
        ))

    @cached_property
    def worst_normalized_violation(self) -> float:
        return float(max(
            (max(0.0, -value) for value in self.constraint_margins),
            default=0.0,
        ))

    @cached_property
    def confidence_margins(self) -> tuple[float, ...]:
        """Return bootstrap-confidence margins for diagnostics and tie-breaks."""
        if not self.constraint_probabilities:
            return ()
        gate = float(self.gate_probability)
        margins = tuple(
            float(value - gate)
            for value in self.constraint_probabilities
        )
        if self.valid:
            return margins
        return tuple(min(value, -1.0) for value in margins)

    @cached_property
    def constraint_margins(self) -> tuple[float, ...]:
        """Return the six margins used by PPO's active online gate."""
        if self.constraint_probabilities:
            return self.confidence_margins
        return self.normalized_margins

    @cached_property
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
            "inference_performed": bool(self.inference_performed),
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
            "constraint_margin_basis": (
                "bootstrap_probability_minus_gate"
                if self.constraint_probabilities
                else "normalized_point_limit"
            ),
            "confidence_margins": [
                float(value) for value in self.confidence_margins
            ],
            "failed_constraint_count": int(self.failed_constraint_count),
            "normalized_violation": float(self.normalized_violation),
            "worst_normalized_violation": float(
                self.worst_normalized_violation
            ),
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

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SearchEvaluation":
        probability_payload = payload.get("constraint_probabilities") or {}
        if isinstance(probability_payload, Mapping):
            probabilities = tuple(
                float(probability_payload[name])
                for name in CONSTRAINT_PROBABILITY_NAMES
                if name in probability_payload
            )
        else:
            probabilities = tuple(float(value) for value in probability_payload)
        return cls(
            action_matrix=tuple(
                tuple(int(value) for value in row)
                for row in payload["action_matrix"]
            ),
            metrics=SearchMetrics.from_dict(payload["metrics"]),
            limits=ConstraintLimits.from_dict(payload["limits"]),
            valid=bool(payload.get("valid", True)),
            reward=(
                None
                if payload.get("reward") is None
                else float(payload["reward"])
            ),
            communication_importance_ratio=float(
                payload.get("communication_importance_ratio", 1.0)
            ),
            constraint_probabilities=probabilities,
            gate_probability=(
                None
                if payload.get("gate_probability") is None
                else float(payload["gate_probability"])
            ),
            metadata=dict(payload.get("metadata") or {}),
        )


class LayerwiseSearchSpace:
    """Discrete Stage-2 space with atomic six-valued per-layer genes."""

    def __init__(self, num_layers: int):
        self.num_layers = int(num_layers)
        if self.num_layers < 1:
            raise ValueError("num_layers must be positive")
        # Preserve the legacy flattened runtime-coordinate contract.
        self.dimensions = tuple(
            value
            for _layer_idx in range(self.num_layers)
            for value in (2, 3)
        )
        self.gene_dimensions = (LAYER_GENE_CARDINALITY,) * self.num_layers

    @property
    def cardinality(self) -> int:
        return int(LAYER_GENE_CARDINALITY ** self.num_layers)

    @property
    def safe_action(self) -> ActionMatrix:
        return tuple((0, 0) for _ in range(self.num_layers))

    @property
    def max_resource_action(self) -> ActionMatrix:
        return tuple((1, 2) for _ in range(self.num_layers))

    @property
    def uniform_anchors(self) -> tuple[ActionMatrix, ...]:
        return tuple(
            self.from_genes((gene,) * self.num_layers)
            for gene in range(LAYER_GENE_CARDINALITY)
        )

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
        """Return the legacy ``2 * num_layers`` coordinate representation."""
        action = self.validate(action_matrix)
        return tuple(value for row in action for value in row)

    def unflatten(self, values: Sequence[int]) -> ActionMatrix:
        """Decode legacy coordinates, or atomic genes when one per layer."""
        flat = tuple(int(value) for value in values)
        if len(flat) == self.num_layers:
            return self.from_genes(flat)
        if len(flat) != 2 * self.num_layers:
            raise ValueError(
                f"flat action must contain {self.num_layers} genes or "
                f"{2 * self.num_layers} legacy values"
            )
        return self.validate(tuple(
            (flat[2 * index], flat[2 * index + 1])
            for index in range(self.num_layers)
        ))

    def genes(self, action_matrix: Sequence[Sequence[int]]) -> tuple[int, ...]:
        return encode_layerwise_action_matrix(self.validate(action_matrix))

    def from_genes(self, genes: Sequence[int]) -> ActionMatrix:
        values = tuple(int(value) for value in genes)
        if len(values) != self.num_layers:
            raise ValueError(
                f"atomic action must contain {self.num_layers} layer genes, "
                f"got {len(values)}"
            )
        return self.validate(decode_layerwise_action_genes(values))

    def one_hot(self, action_matrix: Sequence[Sequence[int]]) -> np.ndarray:
        genes = self.genes(action_matrix)
        features = np.zeros(
            self.num_layers * LAYER_GENE_CARDINALITY,
            dtype=float,
        )
        for layer_idx, gene in enumerate(genes):
            features[layer_idx * LAYER_GENE_CARDINALITY + gene] = 1.0
        return features

    def random_action(self, rng: np.random.Generator) -> ActionMatrix:
        return self.from_genes(tuple(
            int(rng.integers(LAYER_GENE_CARDINALITY))
            for _ in range(self.num_layers)
        ))

    def all_actions(self, *, max_cardinality: int = 100_000) -> Iterator[ActionMatrix]:
        if self.cardinality > int(max_cardinality):
            raise ValueError(
                f"action space cardinality {self.cardinality} exceeds "
                f"enumeration cap {int(max_cardinality)}"
            )
        for genes in itertools.product(
                range(LAYER_GENE_CARDINALITY), repeat=self.num_layers,
        ):
            yield self.from_genes(genes)

    def neighbors(self, action_matrix: Sequence[Sequence[int]]) -> Iterator[ActionMatrix]:
        """Yield full-layer 1-opt alternatives, five per layer."""
        genes = list(self.genes(action_matrix))
        for layer_idx, current in enumerate(genes):
            for alternative in range(LAYER_GENE_CARDINALITY):
                if alternative == current:
                    continue
                candidate = genes[:]
                candidate[layer_idx] = alternative
                yield self.from_genes(candidate)

    def two_opt_neighbors(
            self,
            action_matrix: Sequence[Sequence[int]],
            ) -> Iterator[ActionMatrix]:
        """Yield exhaustive alternatives changing exactly two whole layers."""
        genes = list(self.genes(action_matrix))
        for left, right in itertools.combinations(range(self.num_layers), 2):
            for left_gene in range(LAYER_GENE_CARDINALITY):
                if left_gene == genes[left]:
                    continue
                for right_gene in range(LAYER_GENE_CARDINALITY):
                    if right_gene == genes[right]:
                        continue
                    candidate = genes[:]
                    candidate[left] = left_gene
                    candidate[right] = right_gene
                    yield self.from_genes(candidate)

    def mutate(
            self,
            action_matrix: Sequence[Sequence[int]],
            rng: np.random.Generator,
            *,
            max_coordinates: int = 1,
            ) -> ActionMatrix:
        """Replacement-mutate one or more whole layer genes."""
        genes = list(self.genes(action_matrix))
        maximum = min(self.num_layers, max(1, int(max_coordinates)))
        count = int(rng.integers(1, maximum + 1))
        indices = np.asarray(
            rng.choice(self.num_layers, size=count, replace=False),
            dtype=int,
        ).reshape(-1)
        for layer_idx in indices:
            current = genes[int(layer_idx)]
            replacement = int(rng.integers(LAYER_GENE_CARDINALITY - 1))
            if replacement >= current:
                replacement += 1
            genes[int(layer_idx)] = replacement
        return self.from_genes(genes)

    def crossover(
            self,
            first: Sequence[Sequence[int]],
            second: Sequence[Sequence[int]],
            rng: np.random.Generator,
            *,
            mode: str,
            ) -> ActionMatrix:
        """Cross parents only at whole-layer gene boundaries."""
        left = np.asarray(self.genes(first), dtype=int)
        right = np.asarray(self.genes(second), dtype=int)
        if mode == "uniform":
            mask = np.asarray(rng.random(self.num_layers) < 0.5, dtype=bool)
            child = np.where(mask, right, left)
        elif mode == "two_point":
            if self.num_layers < 2:
                child = np.asarray(
                    [right[0] if rng.random() < 0.5 else left[0]],
                    dtype=int,
                )
            else:
                cuts = np.sort(np.asarray(
                    rng.choice(
                        self.num_layers + 1, size=2, replace=False,
                    ),
                    dtype=int,
                ))
                start, stop = int(cuts[0]), int(cuts[1])
                child = left.copy()
                child[start:stop] = right[start:stop]
        else:
            raise ValueError("crossover mode must be 'two_point' or 'uniform'")
        return self.from_genes(tuple(int(value) for value in child))


def candidate_rank_key(evaluation: SearchEvaluation) -> tuple[float, ...]:
    resource = evaluation.resource
    lexicographic = tuple(
        -float(value)
        for value in LayerwiseSearchSpace(
            len(evaluation.action_matrix)
        ).genes(evaluation.action_matrix)
    )
    point_margins = tuple(sorted(evaluation.normalized_margins))
    confidence = tuple(sorted(evaluation.confidence_margins))
    if evaluation.feasible:
        return (
            2.0,
            float(resource.ppo_resource_score),
            float(resource.robust_floor),
            *confidence,
            *point_margins,
            *lexicographic,
        )
    return (
        1.0 if evaluation.valid else 0.0,
        -float(evaluation.failed_constraint_count),
        -float(evaluation.normalized_violation),
        -float(evaluation.worst_normalized_violation),
        float(resource.ppo_resource_score),
        float(resource.robust_floor),
        *lexicographic,
    )


@dataclass(frozen=True)
class SearchConfig:
    evaluation_budget: int
    seed: int = 42
    initial_design_size: int = 64
    candidate_pool_size: int = 512
    # Kept for runner/config compatibility. The Stage-2 GA uses ga_population_size.
    population_size: int = 64
    patience_generations: int = 20
    mutation_max_coordinates: int = 4
    rf_n_estimators: int = 128
    rf_min_samples_leaf: int = 2
    acquisition_exploration: float = 0.05
    communication_importance_ratio: float = 1.0
    ga_population_size: int = 64
    ga_elite_count: int = 7
    ga_generations: int = 800
    observation_attempt_limit: Optional[int] = None

    def __post_init__(self) -> None:
        for name in (
                "evaluation_budget", "initial_design_size",
                "candidate_pool_size", "population_size",
                "patience_generations", "mutation_max_coordinates",
                "rf_n_estimators", "rf_min_samples_leaf",
                "ga_population_size", "ga_elite_count", "ga_generations",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(self.mutation_max_coordinates) > 4:
            raise ValueError("mutation_max_coordinates must be at most 4")
        if int(self.ga_elite_count) >= int(self.ga_population_size):
            raise ValueError("ga_elite_count must be smaller than ga_population_size")
        if (
                self.observation_attempt_limit is not None
                and int(self.observation_attempt_limit) <= 0
        ):
            raise ValueError("observation_attempt_limit must be positive")
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

    def as_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SearchConfig":
        names = set(cls.__dataclass_fields__)
        return cls(**{
            name: payload[name]
            for name in names
            if name in payload
        })


@dataclass(frozen=True)
class SearchResult:
    algorithm: str
    best: SearchEvaluation
    observations: tuple[SearchEvaluation, ...]
    history: tuple[Mapping[str, Any], ...]
    termination_reason: str

    @property
    def evaluation_count(self) -> int:
        return int(sum(
            item.inference_performed for item in self.observations
        ))

    @property
    def observation_count(self) -> int:
        return len(self.observations)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "stage2_layerwise_search_result_v2",
            "algorithm": str(self.algorithm),
            "evaluation_count": int(self.evaluation_count),
            "observation_count": int(self.observation_count),
            "termination_reason": str(self.termination_reason),
            "best": self.best.as_dict(),
            "history": [dict(row) for row in self.history],
        }

    @classmethod
    def from_dict(
            cls,
            payload: Mapping[str, Any],
            *,
            observations: Iterable[SearchEvaluation] = (),
            ) -> "SearchResult":
        rows = tuple(observations)
        best_payload = payload.get("best")
        if best_payload is None:
            if not rows:
                raise ValueError("serialized search result has no best candidate")
            best = max(rows, key=candidate_rank_key)
        else:
            best = SearchEvaluation.from_dict(best_payload)
        return cls(
            algorithm=normalize_search_backend(payload["algorithm"]),
            best=best,
            observations=rows,
            history=tuple(dict(row) for row in payload.get("history", ())),
            termination_reason=str(payload.get("termination_reason", "unknown")),
        )


class _EvaluationCache:
    def __init__(
            self,
            space: LayerwiseSearchSpace,
            evaluator: EvaluationFn,
            budget: int,
            observation_attempt_limit: Optional[int] = None,
            *,
            preload: Iterable[SearchEvaluation] = (),
            checkpoint_callback: Optional[CheckpointCallback] = None,
            incremental_checkpoint_callback: Optional[
                IncrementalCheckpointCallback
            ] = None,
            ):
        self.space = space
        self.evaluator = evaluator
        self.budget = min(int(budget), int(space.cardinality))
        default_attempt_limit = max(1024, 10 * self.budget)
        configured_limit = (
            default_attempt_limit
            if observation_attempt_limit is None
            else int(observation_attempt_limit)
        )
        self.observation_attempt_limit = min(
            int(space.cardinality), configured_limit,
        )
        self._by_action: dict[ActionMatrix, SearchEvaluation] = {}
        self._ordered: list[SearchEvaluation] = []
        self._best: Optional[SearchEvaluation] = None
        self._inference_count = 0
        self._replay: list[SearchEvaluation] = []
        self._replay_by_action: dict[ActionMatrix, SearchEvaluation] = {}
        self._replay_index = 0
        self.checkpoint_callback = checkpoint_callback
        self.incremental_checkpoint_callback = incremental_checkpoint_callback
        for evaluation in preload:
            self.add_preloaded(evaluation)
        if sum(item.inference_performed for item in self._replay) > self.budget:
            raise ValueError("preloaded observations exceed the inference budget")
        if len(self._replay) > self.observation_attempt_limit:
            raise ValueError("preloaded observations exceed the observation guard")

    def _record(self, evaluation: SearchEvaluation) -> None:
        self._by_action[evaluation.action_matrix] = evaluation
        self._ordered.append(evaluation)
        if evaluation.inference_performed:
            self._inference_count += 1
        if (
                self._best is None
                or candidate_rank_key(evaluation) > candidate_rank_key(self._best)
        ):
            self._best = evaluation

    def add_preloaded(self, evaluation: SearchEvaluation) -> None:
        if not isinstance(evaluation, SearchEvaluation):
            raise TypeError("preloaded rows must be SearchEvaluation objects")
        owned = self.space.validate(evaluation.action_matrix)
        previous = self._replay_by_action.get(owned)
        if previous is not None:
            if previous.as_dict() != evaluation.as_dict():
                raise ValueError("conflicting preloaded evaluation for one action")
            raise ValueError("duplicate preloaded evaluation for one action")
        self._replay.append(evaluation)
        self._replay_by_action[owned] = evaluation

    def assert_replay_consumed(self) -> None:
        if self._replay_index != len(self._replay):
            next_action = self._replay[self._replay_index].action_matrix
            raise RuntimeError(
                "exact search replay terminated before consuming persisted "
                f"observation {self._replay_index}: {next_action!r}"
            )

    @property
    def remaining(self) -> int:
        return max(0, self.budget - self._inference_count)

    @property
    def evaluation_count(self) -> int:
        return int(self._inference_count)

    @property
    def observation_count(self) -> int:
        return len(self._ordered)

    @property
    def observation_guard_reached(self) -> bool:
        return self.observation_count >= self.observation_attempt_limit

    @property
    def can_observe(self) -> bool:
        return bool(self.remaining > 0 and not self.observation_guard_reached)

    @property
    def observations(self) -> tuple[SearchEvaluation, ...]:
        return tuple(self._ordered)

    def contains(self, action: Sequence[Sequence[int]]) -> bool:
        return self.space.validate(action) in self._by_action

    def get(self, action: Sequence[Sequence[int]]) -> Optional[SearchEvaluation]:
        return self._by_action.get(self.space.validate(action))

    def evaluate(self, action: Sequence[Sequence[int]]) -> SearchEvaluation:
        owned = self.space.validate(action)
        cached = self._by_action.get(owned)
        if cached is not None:
            return cached
        if self.remaining <= 0:
            raise RuntimeError("search model-inference budget exhausted")
        if self.observation_guard_reached:
            raise RuntimeError("search observation attempt guard exhausted")
        replayed = self._replay_index < len(self._replay)
        if replayed:
            observed = self._replay[self._replay_index]
            if observed.action_matrix != owned:
                raise RuntimeError(
                    "exact search replay diverged at persisted observation "
                    f"{self._replay_index}: expected {observed.action_matrix!r}, "
                    f"requested {owned!r}"
                )
            self._replay_index += 1
        else:
            observed = self.evaluator(owned)
        if not isinstance(observed, SearchEvaluation):
            raise TypeError("search evaluator must return SearchEvaluation")
        if observed.action_matrix != owned:
            raise ValueError(
                "search evaluator returned metrics for a different action"
            )
        self._record(observed)
        if not replayed and self.incremental_checkpoint_callback is not None:
            self.incremental_checkpoint_callback(observed, len(self._ordered))
        if not replayed and self.checkpoint_callback is not None:
            self.checkpoint_callback(self.observations)
        return observed

    def best(self) -> SearchEvaluation:
        if self._best is None:
            raise RuntimeError("search produced no observations")
        return self._best


def _cache_stop_reason(cache: _EvaluationCache) -> str:
    if cache.remaining <= 0:
        return "evaluation_budget"
    if cache.observation_guard_reached:
        return "observation_attempt_guard"
    return "candidate_space_exhausted"


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


def _hamming_distance(
        space: LayerwiseSearchSpace,
        first: ActionMatrix,
        second: ActionMatrix,
        ) -> int:
    return int(sum(
        left != right
        for left, right in zip(space.genes(first), space.genes(second))
    ))


def _maximin_candidate(
        space: LayerwiseSearchSpace,
        selected: Sequence[ActionMatrix],
        forbidden: set[ActionMatrix],
        rng: np.random.Generator,
        *,
        sample_size: int = 512,
        ) -> Optional[ActionMatrix]:
    candidates: list[ActionMatrix] = []
    if space.cardinality <= 100_000:
        candidates = [
            action for action in space.all_actions()
            if action not in forbidden
        ]
    else:
        seen: set[ActionMatrix] = set()
        attempts = 0
        target = min(int(sample_size), space.cardinality - len(forbidden))
        while len(candidates) < target and attempts < max(1000, 30 * target):
            attempts += 1
            action = space.random_action(rng)
            if action in forbidden or action in seen:
                continue
            candidates.append(action)
            seen.add(action)
    if not candidates:
        return None
    candidate_genes = np.asarray(
        [space.genes(action) for action in candidates], dtype=np.int8,
    )
    if selected:
        reference_genes = np.asarray(
            [space.genes(action) for action in selected], dtype=np.int8,
        )
        distances = np.sum(
            candidate_genes[:, None, :] != reference_genes[None, :, :],
            axis=2,
        )
        minimum_distances = distances.min(axis=1)
        total_distances = distances.sum(axis=1)
    else:
        minimum_distances = np.full(
            len(candidates), space.num_layers, dtype=int,
        )
        total_distances = np.zeros(len(candidates), dtype=int)
    selected_index = max(
        range(len(candidates)),
        key=lambda index: (
            int(minimum_distances[index]),
            int(total_distances[index]),
            tuple(-int(value) for value in candidate_genes[index]),
        ),
    )
    return candidates[selected_index]


def _balanced_one_layer_actions(
        space: LayerwiseSearchSpace,
        ) -> Iterator[ActionMatrix]:
    base = [0] * space.num_layers
    layer_counts = [0] * space.num_layers
    alternative_counts = [0] * (LAYER_GENE_CARDINALITY - 1)
    unused = {
        (layer_idx, alternative)
        for layer_idx in range(space.num_layers)
        for alternative in range(1, LAYER_GENE_CARDINALITY)
    }
    while unused:
        layer_idx, alternative = min(
            unused,
            key=lambda pair: (
                layer_counts[pair[0]],
                alternative_counts[pair[1] - 1],
                ((pair[1] - 1) - pair[0])
                % (LAYER_GENE_CARDINALITY - 1),
                pair[0],
                pair[1],
            ),
        )
        unused.remove((layer_idx, alternative))
        layer_counts[layer_idx] += 1
        alternative_counts[alternative - 1] += 1
        genes = base[:]
        genes[layer_idx] = alternative
        yield space.from_genes(genes)


def _structured_initial_design(
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

    for anchor in space.uniform_anchors:
        add(anchor)
    for action in itertools.islice(_balanced_one_layer_actions(space), 30):
        add(action)
    while len(candidates) < target:
        candidate = _maximin_candidate(
            space, candidates, seen, rng,
        )
        if candidate is None:
            break
        add(candidate)
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
        "evaluations": int(cache.evaluation_count),
        "observations": int(cache.observation_count),
        "best_action_matrix": [list(row) for row in best.action_matrix],
        "best_feasible": bool(best.feasible),
        "best_valid": bool(best.valid),
        "best_failed_constraints": int(best.failed_constraint_count),
        "best_violation": float(best.normalized_violation),
        "best_worst_violation": float(best.worst_normalized_violation),
        "best_resource_score": float(resource.ppo_resource_score),
        "best_robust_floor": float(resource.robust_floor),
        **extra,
    }


def _evaluate_full_neighborhood(
        cache: _EvaluationCache,
        actions: Iterable[ActionMatrix],
        ) -> tuple[list[SearchEvaluation], bool]:
    neighborhood = [cache.space.validate(action) for action in actions]
    pending = [action for action in neighborhood if not cache.contains(action)]
    for action in pending:
        if not cache.can_observe:
            break
        cache.evaluate(action)
    complete = all(cache.contains(action) for action in neighborhood)
    observed = [
        item
        for action in neighborhood
        for item in (cache.get(action),)
        if item is not None
    ]
    return observed, complete


def _run_greedy(
        space: LayerwiseSearchSpace,
        evaluator: EvaluationFn,
        config: SearchConfig,
        *,
        preload: Iterable[SearchEvaluation] = (),
        checkpoint_callback: Optional[CheckpointCallback] = None,
        ) -> SearchResult:
    cache = _EvaluationCache(
        space,
        evaluator,
        config.evaluation_budget,
        config.observation_attempt_limit,
        preload=preload,
        checkpoint_callback=checkpoint_callback,
    )
    history: list[dict[str, Any]] = []
    anchors: list[SearchEvaluation] = []
    for anchor_idx, action in enumerate(space.uniform_anchors):
        cached = cache.get(action)
        if cached is None:
            if not cache.can_observe:
                break
            cached = cache.evaluate(action)
        anchors.append(cached)
        history.append(_best_history_row(
            cache,
            phase="uniform_anchor",
            iteration=anchor_idx,
            anchor_gene=int(anchor_idx),
        ))
    if not anchors:
        raise RuntimeError("greedy search could not evaluate an anchor")

    termination = "verified_local_optima"
    scan_index = 0
    for start_index, start in enumerate(anchors):
        current = start
        while True:
            scan_index += 1
            one_evaluated, one_complete = _evaluate_full_neighborhood(
                cache, space.neighbors(current.action_matrix),
            )
            one_candidates = [current, *one_evaluated]
            one_best = max(one_candidates, key=candidate_rank_key)
            one_improved = (
                candidate_rank_key(one_best) > candidate_rank_key(current)
            )
            history.append(_best_history_row(
                cache,
                phase="greedy_1opt",
                iteration=scan_index,
                start_index=int(start_index),
                neighborhood_complete=bool(one_complete),
                accepted=bool(one_improved and one_complete),
            ))
            if not one_complete:
                termination = _cache_stop_reason(cache)
                break
            if one_improved:
                current = one_best
                continue

            two_evaluated, two_complete = _evaluate_full_neighborhood(
                cache, space.two_opt_neighbors(current.action_matrix),
            )
            two_candidates = [current, *two_evaluated]
            two_best = max(two_candidates, key=candidate_rank_key)
            two_improved = (
                candidate_rank_key(two_best) > candidate_rank_key(current)
            )
            history.append(_best_history_row(
                cache,
                phase="greedy_2opt",
                iteration=scan_index,
                start_index=int(start_index),
                neighborhood_complete=bool(two_complete),
                accepted=bool(two_improved and two_complete),
            ))
            if not two_complete:
                termination = _cache_stop_reason(cache)
                break
            if two_improved:
                current = two_best
                # The next loop always returns to exhaustive 1-opt.
                continue

            # Both neighborhoods for this exact point were exhaustively scanned.
            history.append(_best_history_row(
                cache,
                phase="greedy_final_verification",
                iteration=scan_index,
                start_index=int(start_index),
                one_opt_improvement=False,
                two_opt_improvement=False,
            ))
            break
        if termination in ("evaluation_budget", "observation_attempt_guard"):
            break
    if termination != "verified_local_optima" and not cache.can_observe:
        termination = _cache_stop_reason(cache)
    cache.assert_replay_consumed()
    return SearchResult(
        algorithm="greedy",
        best=cache.best(),
        observations=cache.observations,
        history=tuple(history),
        termination_reason=termination,
    )


def _default_surrogate_factory(config: SearchConfig) -> SurrogateFactory:
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
    unseen_count = int(space.cardinality) - len(cache.observations)
    target = min(int(pool_size), unseen_count)
    if target <= 0:
        return []
    if unseen_count <= target and space.cardinality <= 100_000:
        pool = [
            action for action in space.all_actions()
            if not cache.contains(action)
        ]
        order = np.asarray(rng.permutation(len(pool)), dtype=int).reshape(-1)
        return [pool[int(index)] for index in order]

    pool: list[ActionMatrix] = []
    seen: set[ActionMatrix] = set()

    def add(action: ActionMatrix) -> None:
        owned = space.validate(action)
        if owned not in seen and not cache.contains(owned) and len(pool) < target:
            pool.append(owned)
            seen.add(owned)

    # Half of every pool is globally uniform, avoiding categorical prefix bias.
    global_target = max(1, target // 2)
    attempts = 0
    while len(pool) < global_target and attempts < max(1000, 40 * target):
        attempts += 1
        add(space.random_action(rng))

    # One quarter is atomic local search around feasibility-ranked incumbents.
    ranked = heapq.nlargest(
        min(16, len(cache.observations)),
        cache.observations,
        key=candidate_rank_key,
    )
    local_candidates: list[ActionMatrix] = []
    for observation in ranked:
        local_candidates.extend(space.neighbors(observation.action_matrix))
    if local_candidates:
        order = np.asarray(
            rng.permutation(len(local_candidates)), dtype=int,
        ).reshape(-1)
        local_limit = min(target, global_target + max(1, target // 4))
        for index in order:
            add(local_candidates[int(index)])
            if len(pool) >= local_limit:
                break

    # The remainder is uniform again, independent of incumbent quality.
    attempts = 0
    while len(pool) < target and attempts < max(2000, 80 * target):
        attempts += 1
        add(space.random_action(rng))
    if len(pool) < target and space.cardinality <= 100_000:
        remaining = [
            action for action in space.all_actions()
            if action not in seen and not cache.contains(action)
        ]
        order = np.asarray(rng.permutation(len(remaining)), dtype=int).reshape(-1)
        for index in order:
            add(remaining[int(index)])
            if len(pool) >= target:
                break
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
        space: LayerwiseSearchSpace,
        evaluator: EvaluationFn,
        config: SearchConfig,
        surrogate_factory: Optional[SurrogateFactory],
        *,
        preload: Iterable[SearchEvaluation] = (),
        checkpoint_callback: Optional[CheckpointCallback] = None,
        ) -> SearchResult:
    rng = np.random.default_rng(int(config.seed))
    cache = _EvaluationCache(
        space,
        evaluator,
        config.evaluation_budget,
        config.observation_attempt_limit,
        preload=preload,
        checkpoint_callback=checkpoint_callback,
    )
    history: list[dict[str, Any]] = []
    design = _structured_initial_design(
        space,
        rng,
        min(config.initial_design_size, cache.budget),
    )
    for action in design:
        if not cache.can_observe:
            break
        cache.evaluate(action)
    history.append(_best_history_row(
        cache, phase="structured_initial_design", iteration=0,
        design_size=int(cache.observation_count),
        design_inferences=int(cache.evaluation_count),
    ))

    factory = surrogate_factory or _default_surrogate_factory(config)
    iteration = 0
    no_improvement = 0
    termination = "candidate_space_exhausted"
    while cache.can_observe:
        iteration += 1
        observations = cache.observations
        features = np.asarray(
            [space.one_hot(item.action_matrix) for item in observations],
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
            [space.one_hot(action) for action in pool],
            dtype=float,
        )
        tree_predictions = _tree_predictions(model, pool_features)
        if tree_predictions.ndim != 3 or tree_predictions.shape[2] != 6:
            raise RuntimeError(
                "BO-RF surrogate must predict six constraint margins"
            )
        feasible_probability = np.mean(
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
        incumbent_resource = max(
            (_resource_score(
                item.action_matrix, config.communication_importance_ratio,
            ) for item in feasible_observations),
            default=0.0,
        )
        acquisition: list[float] = []
        acquisition_keys: list[tuple[Any, ...]] = []
        for index, action in enumerate(pool):
            resource = _resource_score(
                action, config.communication_importance_ratio,
            )
            probability = float(feasible_probability[index])
            uncertainty = float(np.mean(predicted_std[index]))
            deterministic = tuple(-value for value in space.genes(action))
            improvement = max(0.0, resource - incumbent_resource)
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
                objective_tiebreak=resource,
                deterministic_tiebreak=deterministic,
            )
            acquisition.append(float(value))
            acquisition_keys.append(key)
        selected_index = max(
            range(len(pool)), key=acquisition_keys.__getitem__,
        )
        previous_best = cache.best()
        selected = pool[selected_index]
        observed = cache.evaluate(selected)
        improved = (
            candidate_rank_key(cache.best()) > candidate_rank_key(previous_best)
        )
        if observed.inference_performed:
            no_improvement = 0 if improved else no_improvement + 1
        history.append(_best_history_row(
            cache,
            phase="feasibility_aware_acquisition",
            iteration=iteration,
            acquisition=float(acquisition[selected_index]),
            acquisition_mode=(
                "probability_of_feasibility_times_expected_improvement"
                if feasible_observations
                else "lexicographic_predicted_violation"
            ),
            predicted_feasibility=float(feasible_probability[selected_index]),
            predicted_failed_constraints=float(predicted_failed[selected_index]),
            predicted_total_violation=float(
                predicted_total_violation[selected_index]
            ),
            predicted_worst_violation=float(
                predicted_worst_violation[selected_index]
            ),
            improved=bool(improved),
            inference_performed=bool(observed.inference_performed),
            no_improvement_iterations=int(no_improvement),
        ))
        if no_improvement >= int(config.patience_generations):
            termination = "bo_no_improvement"
            break
    if cache.remaining <= 0:
        termination = "evaluation_budget"
    elif cache.observation_guard_reached:
        termination = "observation_attempt_guard"
    cache.assert_replay_consumed()
    return SearchResult(
        algorithm="bo_rf",
        best=cache.best(),
        observations=cache.observations,
        history=tuple(history),
        termination_reason=termination,
    )


def _ga_parent_weights(
        population: Sequence[SearchEvaluation],
        ) -> tuple[float, ...]:
    """Return feasibility-aware positive COINN parent fitness weights."""

    if not population:
        raise ValueError("GA parent population must not be empty")
    feasible_scores = [
        float(item.resource.ppo_resource_score)
        for item in population
        if item.feasible
    ]
    if feasible_scores:
        minimum = min(feasible_scores)
        return tuple(
            float(item.resource.ppo_resource_score) - minimum + 1.0
            if item.feasible else 0.0
            for item in population
        )
    return tuple(
        1.0 / (
            1.0
            + float(not item.valid)
            + float(item.failed_constraint_count)
            + float(item.normalized_violation)
            + float(item.worst_normalized_violation)
        )
        for item in population
    )


def _tournament_parent(
        population: Sequence[SearchEvaluation],
        rng: np.random.Generator,
        ) -> SearchEvaluation:
    """Compatibility-named fitness-proportional COINN parent selector."""

    weights = np.asarray(_ga_parent_weights(population), dtype=float)
    probabilities = weights / float(np.sum(weights))
    index = int(rng.choice(len(population), p=probabilities))
    return population[index]


def _diverse_second_parent(
        space: LayerwiseSearchSpace,
        population: Sequence[SearchEvaluation],
        first: SearchEvaluation,
        rng: np.random.Generator,
        ) -> SearchEvaluation:
    distances = [
        (
            item,
            _hamming_distance(
                space, first.action_matrix, item.action_matrix,
            ),
        )
        for item in population
        if item.action_matrix != first.action_matrix
    ]
    eligible = [item for item, distance in distances if distance >= 2]
    if not eligible:
        eligible = [item for item, distance in distances if distance >= 1]
    if not eligible:
        return first
    size = min(3, len(eligible))
    indices = np.asarray(
        rng.choice(len(eligible), size=size, replace=False), dtype=int,
    ).reshape(-1)
    sampled = [eligible[int(index)] for index in indices]
    return max(sampled, key=candidate_rank_key)


def _mesh_adjacent_gene_values(gene: int) -> tuple[int, ...]:
    current = decode_layer_gene(gene)
    return tuple(
        candidate
        for candidate in range(LAYER_GENE_CARDINALITY)
        if candidate != gene
        and max(
            abs(current[0] - decode_layer_gene(candidate)[0]),
            abs(current[1] - decode_layer_gene(candidate)[1]),
        ) <= 1
    )


def _replacement_mutation(
        space: LayerwiseSearchSpace,
        action: ActionMatrix,
        rng: np.random.Generator,
        *,
        force: bool,
        max_layers: int,
        ) -> ActionMatrix:
    genes = list(space.genes(action))
    probability = 1.0 / float(space.num_layers)
    selected = [
        index for index in range(space.num_layers)
        if float(rng.random()) < probability
    ]
    maximum = min(space.num_layers, max(1, int(max_layers)))
    if len(selected) > maximum:
        chosen = np.asarray(
            rng.choice(
                selected, size=maximum, replace=False,
            ),
            dtype=int,
        ).reshape(-1)
        selected = [int(value) for value in chosen]
    if force and not selected:
        selected = [int(rng.integers(space.num_layers))]
    for layer_idx in selected:
        alternatives = _mesh_adjacent_gene_values(genes[layer_idx])
        genes[layer_idx] = alternatives[int(rng.integers(len(alternatives)))]
    return space.from_genes(genes)


def _mean_pairwise_distance(
        space: LayerwiseSearchSpace,
        actions: Sequence[ActionMatrix],
        ) -> float:
    if len(actions) < 2:
        return 0.0
    total = 0
    count = 0
    genes = [space.genes(action) for action in actions]
    for left, right in itertools.combinations(genes, 2):
        total += sum(a != b for a, b in zip(left, right))
        count += 1
    return float(total) / float(count)


def _population_diversity(
        space: LayerwiseSearchSpace,
        population: Sequence[SearchEvaluation],
        ) -> tuple[float, float]:
    actions = [item.action_matrix for item in population]
    unique_ratio = float(len(set(actions))) / float(max(1, len(actions)))
    return unique_ratio, _mean_pairwise_distance(space, actions)


def _select_hamming_diverse_elites(
        space: LayerwiseSearchSpace,
        population: Sequence[SearchEvaluation],
        elite_count: int,
        ) -> list[SearchEvaluation]:
    """Keep the best incumbent, then prefer feasible elites at distance >= 2."""

    target = min(int(elite_count), len(population))
    if target <= 0:
        return []
    ranked = sorted(population, key=candidate_rank_key, reverse=True)
    feasible = [item for item in ranked if item.feasible]
    selected: list[SearchEvaluation] = []
    selected_actions: set[ActionMatrix] = set()
    for pool in (feasible, ranked):
        remaining = [
            item for item in pool
            if item.action_matrix not in selected_actions
        ]
        while remaining and len(selected) < target:
            if not selected:
                chosen = remaining[0]
            else:
                distance_two = [
                    item for item in remaining
                    if all(
                        _hamming_distance(
                            space, item.action_matrix, owned.action_matrix,
                        ) >= 2
                        for owned in selected
                    )
                ]
                distance_one = [
                    item for item in remaining
                    if all(
                        _hamming_distance(
                            space, item.action_matrix, owned.action_matrix,
                        ) >= 1
                        for owned in selected
                    )
                ]
                chosen = (distance_two or distance_one or remaining)[0]
            selected.append(chosen)
            selected_actions.add(chosen.action_matrix)
            remaining = [
                item for item in remaining
                if item.action_matrix != chosen.action_matrix
            ]
        if len(selected) >= target:
            break
    return selected


def _make_ga_child(
        space: LayerwiseSearchSpace,
        population: Sequence[SearchEvaluation],
        rng: np.random.Generator,
        forbidden: set[ActionMatrix],
        *,
        mutation_max_layers: int,
        ) -> tuple[Optional[ActionMatrix], bool]:
    for _attempt in range(_GA_OFFSPRING_ATTEMPTS):
        parent = _tournament_parent(population, rng)
        child = _replacement_mutation(
            space,
            parent.action_matrix,
            rng,
            force=True,
            max_layers=int(mutation_max_layers),
        )
        if child in forbidden:
            child = _replacement_mutation(
                space,
                parent.action_matrix,
                rng,
                force=True,
                max_layers=int(mutation_max_layers),
            )
        if child not in forbidden:
            return child, False
    maximum = min(space.num_layers, max(1, int(mutation_max_layers)))
    parents = sorted(population, key=candidate_rank_key, reverse=True)
    for parent in parents:
        parent_genes = space.genes(parent.action_matrix)
        for changed_count in range(1, maximum + 1):
            for coordinates in itertools.combinations(
                    range(space.num_layers), changed_count,
            ):
                alternatives = tuple(
                    _mesh_adjacent_gene_values(parent_genes[index])
                    for index in coordinates
                )
                for replacements in itertools.product(*alternatives):
                    genes = list(parent_genes)
                    for index, replacement in zip(coordinates, replacements):
                        genes[index] = int(replacement)
                    child = space.from_genes(genes)
                    if child not in forbidden:
                        return child, False
    return None, False


def _run_coinn_ga(
        space: LayerwiseSearchSpace,
        evaluator: EvaluationFn,
        config: SearchConfig,
        *,
        preload: Iterable[SearchEvaluation] = (),
        checkpoint_callback: Optional[CheckpointCallback] = None,
        ) -> SearchResult:
    rng = np.random.default_rng(int(config.seed))
    cache = _EvaluationCache(
        space,
        evaluator,
        config.evaluation_budget,
        config.observation_attempt_limit,
        preload=preload,
        checkpoint_callback=checkpoint_callback,
    )
    population_target = min(
        int(config.ga_population_size),
        int(space.cardinality),
        int(cache.budget),
    )
    elite_count = min(
        int(config.ga_elite_count), max(0, population_target - 1),
    )
    structured = _structured_initial_design(
        space, rng, population_target,
    )
    population: list[SearchEvaluation] = []
    structured_index = 0
    while len(population) < population_target and cache.can_observe:
        if structured_index < len(structured):
            action = structured[structured_index]
            structured_index += 1
        else:
            forbidden = {
                item.action_matrix for item in cache.observations
            }
            action = _maximin_candidate(
                space,
                [item.action_matrix for item in population],
                forbidden,
                rng,
            )
            if action is None:
                break
        observed = cache.evaluate(action)
        if observed.inference_performed:
            population.append(observed)

    history: list[dict[str, Any]] = [_best_history_row(
        cache,
        phase="ga_initial_population",
        iteration=0,
        population_size=len(population),
        population_target=int(population_target),
        elite_count=int(elite_count),
        structured_observations=int(min(structured_index, len(structured))),
        initialization_provenance=[
            {
                "action_matrix": [list(row) for row in action],
                "source": (
                    "uniform_anchor"
                    if index < 6
                    else (
                        "balanced_one_layer_from_all_0H"
                        if index < 36
                        else "categorical_maximin"
                    )
                ),
            }
            for index, action in enumerate(structured)
        ],
        non_inference_observations=int(
            cache.observation_count - cache.evaluation_count
        ),
    )]
    observed_actions = {
        item.action_matrix for item in cache.observations
    }
    completed_generations = 0
    no_improvement_generations = 0
    termination = "generation_limit"
    if len(population) < population_target:
        termination = _cache_stop_reason(cache)

    while (
            len(population) == population_target
            and completed_generations < int(config.ga_generations)
    ):
        elites = _select_hamming_diverse_elites(
            space, population, elite_count,
        )
        offspring_target = population_target - elite_count
        if offspring_target <= 0:
            termination = "candidate_space_exhausted"
            break
        if cache.remaining < offspring_target:
            termination = "evaluation_budget"
            break
        if space.cardinality - cache.observation_count < offspring_target:
            termination = "candidate_space_exhausted"
            break

        unique_ratio, mean_distance = _population_diversity(space, population)
        offspring: list[SearchEvaluation] = []
        forbidden = observed_actions
        observation_start = cache.observation_count
        previous_best = cache.best()
        generation_failed = False

        while len(offspring) < offspring_target:
            if not cache.can_observe:
                generation_failed = True
                break
            child, _used_immigrant = _make_ga_child(
                space,
                population,
                rng,
                forbidden,
                mutation_max_layers=int(config.mutation_max_coordinates),
            )
            if child is None:
                termination = "mutation_neighborhood_exhausted"
                generation_failed = True
                break
            forbidden.add(child)
            observed = cache.evaluate(child)
            if observed.inference_performed:
                offspring.append(observed)

        if generation_failed or len(offspring) != offspring_target:
            if termination not in {
                    "candidate_space_exhausted",
                    "mutation_neighborhood_exhausted",
            }:
                termination = _cache_stop_reason(cache)
            history.append(_best_history_row(
                cache,
                phase="ga_generation_aborted",
                iteration=completed_generations + 1,
                generation=int(completed_generations + 1),
                inference_reaching_offspring=int(len(offspring)),
                offspring_target=int(offspring_target),
                generation_observations=int(
                    cache.observation_count - observation_start
                ),
            ))
            break

        population = [*elites, *offspring]
        if len(population) != population_target:
            raise RuntimeError("GA population refill did not preserve its size")
        if any(not item.inference_performed for item in population):
            raise RuntimeError("GA parent population contains a non-inference candidate")
        completed_generations += 1
        improved = (
            candidate_rank_key(cache.best()) > candidate_rank_key(previous_best)
        )
        no_improvement_generations = (
            0 if improved else no_improvement_generations + 1
        )
        post_unique_ratio, post_mean_distance = _population_diversity(
            space, population,
        )
        generation_observations = cache.observation_count - observation_start
        history.append(_best_history_row(
            cache,
            phase="ga_update_generation",
            iteration=completed_generations,
            generation=int(completed_generations),
            population_size=int(population_target),
            elite_count=int(elite_count),
            feasible_elite_count=sum(item.feasible for item in elites),
            elite_policy="best_incumbent_then_hamming_distance_2",
            elite_actions=[
                [list(row) for row in item.action_matrix]
                for item in elites
            ],
            offspring_evaluated=int(len(offspring)),
            offspring_observations=int(generation_observations),
            non_inference_offspring_observations=int(
                generation_observations - len(offspring)
            ),
            expected_evaluations=int(
                population_target + completed_generations * offspring_target
            ),
            improved=bool(improved),
            no_improvement_generations=int(no_improvement_generations),
            unique_ratio=float(unique_ratio),
            mean_pairwise_distance=float(mean_distance),
            diversity_triggered=False,
            diversity_immigrants=0,
            fallback_immigrants=0,
            replaced_worst_nonelite_actions=[],
            immigrant_actions=[],
            post_update_unique_ratio=float(post_unique_ratio),
            post_update_mean_pairwise_distance=float(post_mean_distance),
        ))
        if no_improvement_generations >= int(config.patience_generations):
            termination = "ga_no_incumbent_improvement"
            break

    if (
            termination == "generation_limit"
            and completed_generations >= int(config.ga_generations)
    ):
        termination = "generation_limit"
    cache.assert_replay_consumed()
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
        preload: Iterable[SearchEvaluation] = (),
        checkpoint_callback: Optional[CheckpointCallback] = None,
        ) -> SearchResult:
    normalized = normalize_search_backend(backend)
    preloaded_by_action: dict[ActionMatrix, SearchEvaluation] = {}
    for evaluation in preload:
        owned = space.validate(evaluation.action_matrix)
        previous = preloaded_by_action.get(owned)
        if previous is not None:
            if previous.as_dict() != evaluation.as_dict():
                raise ValueError("conflicting preloaded evaluation for one action")
            raise ValueError("duplicate preloaded evaluation for one action")
        preloaded_by_action[owned] = evaluation
    preload_rows = tuple(preloaded_by_action.values())
    if normalized == "ppo":
        raise ValueError(
            "run_search implements non-RL baselines only; PPO uses the "
            "existing layerwise trainer"
        )
    if normalized == "greedy":
        return _run_greedy(
            space,
            evaluator,
            config,
            preload=preload_rows,
            checkpoint_callback=checkpoint_callback,
        )
    if normalized == "bo_rf":
        return _run_bo_rf(
            space,
            evaluator,
            config,
            surrogate_factory,
            preload=preload_rows,
            checkpoint_callback=checkpoint_callback,
        )
    if normalized == "coinn_ga":
        return _run_coinn_ga(
            space,
            evaluator,
            config,
            preload=preload_rows,
            checkpoint_callback=checkpoint_callback,
        )
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
