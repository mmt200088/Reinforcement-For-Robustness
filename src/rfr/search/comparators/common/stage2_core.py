"""Constrained non-RL baselines for the canonical Stage-2 layerwise action.

The public runtime action remains one ``(Block4 fusion, H/M/L preset)`` row per
Transformer layer. Search operators encode each row as one atomic six-valued
categorical gene so crossover, mutation, and neighborhoods never split a
layer's coupled decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
import itertools
import math
import re
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence

import numpy as np

from rfr.search.common.layerwise_action import (
    LAYER_GENE_CARDINALITY,
    compute_variable_cost_from_action_matrix,
    decode_layer_gene,
    decode_layerwise_action_genes,
    encode_layerwise_action_matrix,
)
from rfr.search.common.precision_presets import validate_communication_importance_ratio


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
        """Return the pairwise ``2 * num_layers`` coordinate representation."""
        action = self.validate(action_matrix)
        return tuple(value for row in action for value in row)

    def unflatten(self, values: Sequence[int]) -> ActionMatrix:
        """Decode pairwise coordinates or one atomic gene per layer."""
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
    seed: int = 42
    initial_design_size: int = 64
    candidate_pool_size: int = 512

    bo_no_improvement_patience: int = 2_000
    greedy_no_improvement_rounds: int = 1
    mutation_max_coordinates: int = 4
    rf_n_estimators: int = 128
    rf_min_samples_leaf: int = 2
    acquisition_exploration: float = 0.05
    communication_importance_ratio: float = 1.0
    ga_population_size: int = 64
    ga_elite_count: int = 7
    ga_generations: int = 200

    def __post_init__(self) -> None:
        for name in (
                "initial_design_size", "candidate_pool_size",
                "bo_no_improvement_patience",
                "greedy_no_improvement_rounds",
                "mutation_max_coordinates",
                "rf_n_estimators", "rf_min_samples_leaf",
                "ga_population_size", "ga_elite_count", "ga_generations",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(self.mutation_max_coordinates) > 4:
            raise ValueError("mutation_max_coordinates must be at most 4")
        if int(self.ga_elite_count) >= int(self.ga_population_size):
            raise ValueError("ga_elite_count must be smaller than ga_population_size")
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
            *,
            preload: Iterable[SearchEvaluation] = (),
            checkpoint_callback: Optional[CheckpointCallback] = None,
            incremental_checkpoint_callback: Optional[
                IncrementalCheckpointCallback
            ] = None,
            ):
        self.space = space
        self.evaluator = evaluator
        self.capacity = int(space.cardinality)
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
        if len(self._replay) > self.capacity:
            raise ValueError("preloaded observations exceed the action space")

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
        return max(0, self.capacity - len(self._ordered))

    @property
    def evaluation_count(self) -> int:
        return int(self._inference_count)

    @property
    def observation_count(self) -> int:
        return len(self._ordered)

    @property
    def can_observe(self) -> bool:
        return self.remaining > 0

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
            raise RuntimeError("search action space exhausted")
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
        from rfr.search.comparators.greedy.stage2 import run

        return run(
            space,
            evaluator,
            config,
            preload=preload_rows,
            checkpoint_callback=checkpoint_callback,
        )
    if normalized == "bo_rf":
        from rfr.search.comparators.bo_rf.stage2 import run

        return run(
            space,
            evaluator,
            config,
            surrogate_factory,
            preload=preload_rows,
            checkpoint_callback=checkpoint_callback,
        )
    if normalized == "coinn_ga":
        from rfr.search.comparators.coinn_ga.stage2 import run

        return run(
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
