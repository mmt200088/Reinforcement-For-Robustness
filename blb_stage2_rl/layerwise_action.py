"""Torch-free codec for the canonical Stage-2 layerwise policy.

The legacy full action vector remains the runtime interchange format.  This
module owns only the compact policy action: one Block4 fusion choice and five
truncation-K choices per Transformer layer.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence, Tuple

import numpy as np

try:
    from .truncation_levels import (
        K_LEVELS,
        K_MAX_BITS,
        K_MIN_BITS,
        validate_exact_k_domain,
    )
except ImportError:  # pragma: no cover - legacy top-level import compatibility
    from truncation_levels import (
        K_LEVELS,
        K_MAX_BITS,
        K_MIN_BITS,
        validate_exact_k_domain,
    )


LAYERWISE_SLOT_NAMES = (
    "block4_fusion",
    "block1_k",
    "block2_k",
    "block3_k",
    "block4_k",
    "block5_k",
)

def _validate_k_levels() -> Tuple[int, ...]:
    return validate_exact_k_domain(K_LEVELS)

# These are the stable legacy action_space._BLOCK_SPECS field counts, in block
# order.  All blocks put output_truncation_k in their last slot.
_BLOCK_SLOT_COUNTS = {1: 9, 2: 23, 3: 8, 4: 17, 5: 16}
_BLOCK_STARTS = {1: 0, 2: 9, 3: 32, 4: 40, 5: 57}
_LAYER_WIDTH = sum(_BLOCK_SLOT_COUNTS.values())
_BLOCK_ORDER = (1, 2, 3, 4, 5)


def _validated_num_layers(num_layers: int) -> int:
    layers = int(num_layers)
    if layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {layers}")
    return layers


def layerwise_action_space_version(num_layers: int) -> str:
    """Return the persisted action-space identity for one model depth."""
    layers = _validated_num_layers(num_layers)
    return f"stage2_layerwise_{layers}x{len(LAYERWISE_SLOT_NAMES)}_v2"


def max_compute_saving_units(num_layers: int) -> float:
    """Maximum learnable Block4 fusion count for the model depth."""
    return float(_validated_num_layers(num_layers))


def max_communication_saving_units(num_layers: int) -> float:
    """Maximum removed K bits across all active per-block K slots."""
    layers = _validated_num_layers(num_layers)
    return float(5 * layers) * float(K_MAX_BITS - K_MIN_BITS)


# Backward-compatible BERT-base constants. New callers with a model instance
# must use the layer-count helpers above.
MAX_COMPUTE_SAVING_UNITS = max_compute_saving_units(12)
MAX_COMMUNICATION_SAVING_UNITS = max_communication_saving_units(12)
RESOURCE_SECONDARY_EPSILON = 1.0e-4
LAYERWISE_DECODE_VERSION = "layerwise_action_v2"
LAYERWISE_COST_MODEL_REVISION = "dual_resource_maxmin_shapley_v2"


@dataclass(frozen=True)
class LayerwiseStepSpec:
    step_idx: int
    layer_idx: int
    slot_dims: Tuple[int, ...]
    slot_mask: Tuple[bool, ...]
    terminal: bool
    num_layers: int
    graph_keys_by_block: Tuple[Tuple[int, str], ...]


@dataclass(frozen=True)
class LayerwiseDecodedAction:
    block4_fusion: int
    k_by_block: Mapping[int, int]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "k_by_block",
            MappingProxyType({int(block): int(k_value) for block, k_value in self.k_by_block.items()}),
        )


@dataclass(frozen=True)
class LayerActionApplication:
    full_vector: np.ndarray
    decoded: LayerwiseDecodedAction
    fusion_option_ids: Mapping[int, int]
    boosted_field_values_by_block: Mapping[int, Mapping[str, int]]

    def __post_init__(self) -> None:
        full_vector = np.asarray(self.full_vector, dtype=int).reshape(-1).copy()
        full_vector.setflags(write=False)
        object.__setattr__(self, "full_vector", full_vector)
        object.__setattr__(
            self,
            "decoded",
            LayerwiseDecodedAction(self.decoded.block4_fusion, self.decoded.k_by_block),
        )
        object.__setattr__(
            self,
            "fusion_option_ids",
            MappingProxyType({int(block): int(option_id) for block, option_id in self.fusion_option_ids.items()}),
        )
        object.__setattr__(
            self,
            "boosted_field_values_by_block",
            MappingProxyType({
                int(block): MappingProxyType({str(name): int(value) for name, value in values.items()})
                for block, values in self.boosted_field_values_by_block.items()
            }),
        )


@dataclass(frozen=True)
class VariableCost:
    compute_saving: float
    communication_saving: float
    robust_floor: float
    secondary_progress: float
    ppo_resource_score: float
    compute_shapley_credit: float
    communication_shapley_credit: float
    fusion_count: int
    removed_k_bits: int
    layer_resource_rewards: Tuple[float, ...]
    slot_resource_rewards: Tuple[Tuple[float, ...], ...]

    @property
    def fusion_saving(self) -> float:
        return self.compute_saving

    @property
    def truncation_saving(self) -> float:
        return self.communication_saving

    @property
    def normalized(self) -> float:
        return self.ppo_resource_score

    @property
    def fusion_units(self) -> float:
        return float(self.fusion_count)

    @property
    def truncation_units(self) -> float:
        return float(self.removed_k_bits)

    @property
    def layer_cost_rewards(self) -> Tuple[float, ...]:
        return self.layer_resource_rewards

    @property
    def slot_cost_rewards(self) -> Tuple[Tuple[float, ...], ...]:
        return self.slot_resource_rewards


def _unit_interval(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1], got {value!r}")
    return result


def dual_resource_score(
        compute_saving: float,
        communication_saving: float,
        ) -> Tuple[float, float, float]:
    """Return robust floor, secondary progress, and the bounded PPO score."""
    compute = _unit_interval("compute_saving", compute_saving)
    communication = _unit_interval("communication_saving", communication_saving)
    robust_floor = min(compute, communication)
    secondary_progress = 0.5 * (compute + communication)
    score = (
        robust_floor + RESOURCE_SECONDARY_EPSILON * secondary_progress
    ) / (1.0 + RESOURCE_SECONDARY_EPSILON)
    return float(robust_floor), float(secondary_progress), float(score)


def resource_shapley_credits(
        compute_saving: float,
        communication_saving: float,
        ) -> Tuple[float, float]:
    """Split the coupled PPO score between compute and communication."""
    compute = _unit_interval("compute_saving", compute_saving)
    communication = _unit_interval("communication_saving", communication_saving)

    def value(compute_value: float, communication_value: float) -> float:
        return dual_resource_score(compute_value, communication_value)[2]

    empty = value(0.0, 0.0)
    compute_credit = 0.5 * (value(compute, 0.0) - empty) + 0.5 * (
        value(compute, communication) - value(0.0, communication)
    )
    total = value(compute, communication)
    if abs(compute_credit) < 1.0e-15:
        compute_credit = 0.0
    communication_credit = total - compute_credit
    if abs(communication_credit) < 1.0e-15:
        communication_credit = 0.0
    if compute_credit < 0.0 or communication_credit < 0.0:
        raise RuntimeError("dual-resource Shapley credits must be nonnegative")
    return float(compute_credit), float(communication_credit)


def _graph_keys(layer_idx: int, profile: str, gelu_degree: int) -> Tuple[Tuple[int, str], ...]:
    pairs = []
    if layer_idx > 0:
        pairs.append((1, f"block1_{profile}"))
    pairs.extend(((2, f"block2_{profile}"), (4, "block4"), (5, f"block5_n{gelu_degree}")))
    return tuple(pairs)


def _unique_option_for_fusion_count(fusion_map: Any, graph_key: str, fusion_count: int) -> Any:
    matches = [
        option for option in fusion_map.options(graph_key)
        if int(option.fusion_count) == int(fusion_count)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{graph_key}: requires exactly one fusion_count={fusion_count} option, "
            f"found {[int(option.option_id) for option in matches]}"
        )
    return matches[0]


def _validate_graph_options(graph_key: str, graph: Any, expected_slots: int) -> None:
    k_slot_index = int(graph.k_slot_index)
    if not 0 <= k_slot_index < expected_slots:
        raise ValueError(
            f"{graph_key}: K slot {k_slot_index} outside [0, {expected_slots})"
        )
    if k_slot_index != expected_slots - 1:
        raise ValueError(
            f"{graph_key}: K slot {k_slot_index} is not legacy slot {expected_slots - 1}"
        )
    option_ids = [int(option.option_id) for option in graph.options]
    if len(set(option_ids)) != len(option_ids):
        raise ValueError(f"{graph_key}: duplicate option_id values {option_ids}")
    for option in graph.options:
        vector = np.asarray(option.action_indices, dtype=int).reshape(-1)
        if vector.size != expected_slots:
            raise ValueError(
                f"{graph_key} option {option.option_id}: action_indices has {vector.size} slots, "
                f"expected {expected_slots}"
            )
        if bool(getattr(option, "boosted", False)) and not getattr(option, "explicit_field_values", None):
            raise ValueError(
                f"{graph_key} option {option.option_id}: boosted options require explicit_field_values"
            )


def _validate_graphs(spec: LayerwiseStepSpec, fusion_map: Any) -> None:
    for block_idx, graph_key in spec.graph_keys_by_block:
        # Block 1 has no RL-selectable fusion dimension. Its SF chain stays on
        # the calibrated RO baseline and only its K slot is changed, exactly
        # like Block 3. Large-profile map bundles therefore need no Block 1 map.
        if block_idx == 1:
            continue
        if graph_key not in fusion_map.graphs:
            raise KeyError(f"fusion map has no graph {graph_key!r} for block {block_idx}")
        graph = fusion_map.graphs[graph_key]
        expected_slots = _BLOCK_SLOT_COUNTS[block_idx]
        if int(graph.block_num_slots) != expected_slots:
            raise ValueError(
                f"{graph_key}: map has {graph.block_num_slots} slots, expected {expected_slots}"
            )
        _validate_graph_options(graph_key, graph, expected_slots)
        required_count = 0 if block_idx == 1 else 1 if block_idx in (2, 5) else None
        if required_count is not None:
            _unique_option_for_fusion_count(fusion_map, graph_key, required_count)
        elif block_idx == 4:
            _unique_option_for_fusion_count(fusion_map, graph_key, 0)
            _unique_option_for_fusion_count(fusion_map, graph_key, 1)


def layerwise_schedule(
        num_layers: int,
        fusion_map: Any,
        profile: str = "mrpc",
        gelu_degrees: Sequence[int] | None = None,
        ) -> list[LayerwiseStepSpec]:
    """Return one six-slot policy step per Transformer layer."""
    levels = _validate_k_levels()
    layers = int(num_layers)
    if layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {layers}")
    if gelu_degrees is not None and len(gelu_degrees) != layers:
        raise ValueError(f"gelu_degrees has {len(gelu_degrees)} values, expected {layers}")

    slot_dims = (2,) + (len(levels),) * 5
    specs = []
    for layer_idx in range(layers):
        gelu_degree = int(gelu_degrees[layer_idx]) if gelu_degrees is not None else 4
        spec = LayerwiseStepSpec(
            step_idx=layer_idx,
            layer_idx=layer_idx,
            slot_dims=slot_dims,
            slot_mask=(True, True, True, True, True, True),
            terminal=(layer_idx == layers - 1),
            num_layers=layers,
            graph_keys_by_block=_graph_keys(layer_idx, str(profile), gelu_degree),
        )
        _validate_graphs(spec, fusion_map)
        specs.append(spec)
    return specs


def _block_offsets(layer_idx: int, block_idx: int) -> range:
    start = int(layer_idx) * _LAYER_WIDTH + _BLOCK_STARTS[block_idx]
    return range(start, start + _BLOCK_SLOT_COUNTS[block_idx])


def _validate_layer_action(layer_action: Sequence[int], step_spec: LayerwiseStepSpec) -> Tuple[int, ...]:
    values = tuple(int(value) for value in layer_action)
    if len(values) != len(LAYERWISE_SLOT_NAMES):
        raise ValueError(f"layer action expects {len(LAYERWISE_SLOT_NAMES)} slots, got {len(values)}")
    for slot_idx, (value, dim, enabled) in enumerate(zip(values, step_spec.slot_dims, step_spec.slot_mask)):
        if enabled and not 0 <= value < dim:
            raise ValueError(f"step {step_spec.step_idx} slot {slot_idx} index {value} outside [0, {dim})")
    return values


def _splice_mapped_block(
        full_vector: np.ndarray,
        graph: Any,
        graph_key: str,
        option: Any,
        k_index: int,
        layer_idx: int,
        block_idx: int,
        ) -> None:
    # FusionCountMap.expand currently indexes graph.options by option_id.  Keep
    # this codec correct for valid non-contiguous IDs by expanding the resolved
    # option object instead of treating its reporting ID as a list index.
    expanded = np.asarray(option.action_indices, dtype=int).reshape(-1).copy()
    offsets = _block_offsets(layer_idx, block_idx)
    if expanded.size != len(offsets):
        raise ValueError(
            f"{graph_key} option {option.option_id} expanded to {expanded.size} slots, expected {len(offsets)}"
        )
    k_slot_index = int(graph.k_slot_index)
    if not 0 <= k_slot_index < expanded.size:
        raise ValueError(
            f"{graph_key}: K slot {k_slot_index} outside expanded option width {expanded.size}"
        )
    expanded[k_slot_index] = int(k_index)
    full_vector[list(offsets)] = expanded


def apply_layer_action(
        full_vector: Sequence[int],
        layer_action: Sequence[int],
        step_spec: LayerwiseStepSpec,
        fusion_map: Any,
        profile: str = "mrpc",
        gelu_degree: int = 4,
        ) -> LayerActionApplication:
    """Copy a legacy vector and apply exactly one layer's policy action."""
    del profile, gelu_degree  # Graph identity is fixed by the schedule metadata.
    _validate_k_levels()
    action = _validate_layer_action(layer_action, step_spec)
    expected_size = int(step_spec.num_layers) * _LAYER_WIDTH + 1
    result = np.asarray(full_vector, dtype=int).reshape(-1).copy()
    if result.size != expected_size:
        raise ValueError(f"full_vector has {result.size} slots, expected {expected_size}")
    _validate_graphs(step_spec, fusion_map)

    graph_keys = dict(step_spec.graph_keys_by_block)
    k_indices = {1: action[1], 2: action[2], 3: action[3], 4: action[4], 5: action[5]}
    active_blocks = _BLOCK_ORDER
    k_by_block = {block_idx: int(K_LEVELS[k_indices[block_idx]]) for block_idx in active_blocks}
    fusion_option_ids: dict[int, int] = {}
    boosted_values: dict[int, Mapping[str, int]] = {}

    choices = ((2, 1), (4, action[0]), (5, 1))
    for block_idx, fusion_count in choices:
        if block_idx not in active_blocks:
            continue
        graph_key = graph_keys[block_idx]
        option = _unique_option_for_fusion_count(fusion_map, graph_key, fusion_count)
        _splice_mapped_block(
            result, fusion_map.graphs[graph_key], graph_key, option,
            k_indices[block_idx], step_spec.layer_idx, block_idx,
        )
        fusion_option_ids[block_idx] = int(option.option_id)
        if bool(getattr(option, "boosted", False)) and getattr(option, "explicit_field_values", None):
            values = {str(name): int(value) for name, value in option.explicit_field_values.items()}
            values["output_truncation_k"] = k_by_block[block_idx]
            boosted_values[block_idx] = values

    for baseline_owned_block in (1, 3):
        if baseline_owned_block not in active_blocks:
            continue
        k_offset = _block_offsets(
            step_spec.layer_idx, baseline_owned_block,
        ).stop - 1
        result[k_offset] = k_indices[baseline_owned_block]
    return LayerActionApplication(
        full_vector=result,
        decoded=LayerwiseDecodedAction(block4_fusion=int(action[0]), k_by_block=k_by_block),
        fusion_option_ids=fusion_option_ids,
        boosted_field_values_by_block=boosted_values,
    )


def compute_variable_cost(actions: Sequence[LayerwiseDecodedAction]) -> VariableCost:
    """Compute independent compute/communication savings from decoded actions."""
    _validate_k_levels()
    num_layers = len(actions)
    if num_layers < 1:
        raise ValueError("variable cost requires at least one layer action")
    compute_denominator = max_compute_saving_units(num_layers)
    active_k_slots = 5 * num_layers
    communication_denominator = max_communication_saving_units(num_layers)
    fusion_values = []
    k_values = []
    raw_slot_contributions = []
    for layer_idx, action in enumerate(actions):
        expected_blocks = {1, 2, 3, 4, 5}
        actual_blocks = set(action.k_by_block)
        if actual_blocks != expected_blocks:
            raise ValueError(
                f"layer {layer_idx} K blocks {sorted(actual_blocks)} do not match {sorted(expected_blocks)}"
            )
        fusion = int(action.block4_fusion)
        if fusion not in (0, 1):
            raise ValueError(f"layer {layer_idx} Block4 fusion must be 0 or 1, got {fusion}")
        fusion_values.append(fusion)
        for block_idx, k_value in action.k_by_block.items():
            k = int(k_value)
            if k not in K_LEVELS:
                raise ValueError(f"layer {layer_idx} block {block_idx} has unsupported K={k}")
            k_values.append(k)
        current_slot_contributions = [
            float(fusion) / compute_denominator,
        ]
        current_slot_contributions.extend(
            (float(K_MAX_BITS) - float(action.k_by_block[block_idx]))
            / communication_denominator
            if block_idx in action.k_by_block else 0.0
            for block_idx in _BLOCK_ORDER
        )
        raw_slot_contributions.append(current_slot_contributions)
    if len(k_values) != active_k_slots:
        raise RuntimeError(
            f"{num_layers}-layer actions yielded {len(k_values)} active K values, "
            f"expected {active_k_slots}"
        )
    fusion_count = int(sum(fusion_values))
    removed_k_bits = int(sum(K_MAX_BITS - k for k in k_values))
    compute_saving = float(fusion_count) / compute_denominator
    communication_saving = (
        float(removed_k_bits) / communication_denominator
    )
    robust_floor, secondary_progress, ppo_resource_score = dual_resource_score(
        compute_saving, communication_saving,
    )
    compute_credit, communication_credit = resource_shapley_credits(
        compute_saving, communication_saving,
    )

    slot_resource_rewards = []
    for raw_row in raw_slot_contributions:
        row = [
            compute_credit * raw_row[0] / compute_saving
            if compute_saving > 0.0 else 0.0,
        ]
        row.extend(
            communication_credit * value / communication_saving
            if communication_saving > 0.0 else 0.0
            for value in raw_row[1:]
        )
        slot_resource_rewards.append(tuple(float(value) for value in row))
    layer_resource_rewards = tuple(
        float(sum(row)) for row in slot_resource_rewards
    )
    if not math.isclose(
            sum(layer_resource_rewards), ppo_resource_score,
            rel_tol=0.0, abs_tol=1.0e-12,
    ):
        raise RuntimeError("dual-resource slot credits do not sum to PPO score")

    return VariableCost(
        compute_saving=float(compute_saving),
        communication_saving=float(communication_saving),
        robust_floor=float(robust_floor),
        secondary_progress=float(secondary_progress),
        ppo_resource_score=float(ppo_resource_score),
        compute_shapley_credit=float(compute_credit),
        communication_shapley_credit=float(communication_credit),
        fusion_count=int(fusion_count),
        removed_k_bits=int(removed_k_bits),
        layer_resource_rewards=layer_resource_rewards,
        slot_resource_rewards=tuple(slot_resource_rewards),
    )


def compute_variable_cost_from_action_matrix(
        action_matrix: Sequence[Sequence[int]],
        ) -> VariableCost:
    """Decode a canonical ``num_layers x 6`` policy action and compute its cost."""
    levels = _validate_k_levels()
    rows = [tuple(int(value) for value in row) for row in action_matrix]
    if not rows or any(len(row) != len(LAYERWISE_SLOT_NAMES) for row in rows):
        raise ValueError("action_matrix must have shape num_layers x 6")
    decoded = []
    for layer_idx, row in enumerate(rows):
        fusion = int(row[0])
        if fusion not in (0, 1):
            raise ValueError(
                f"action_matrix[{layer_idx}][0]={fusion} outside [0, 2)"
            )
        active_blocks = _BLOCK_ORDER
        k_by_block = {}
        for block_idx in active_blocks:
            k_index = int(row[block_idx])
            if not 0 <= k_index < len(levels):
                raise ValueError(
                    f"action_matrix[{layer_idx}][{block_idx}]={k_index} "
                    f"outside [0, {len(levels)})"
                )
            k_by_block[block_idx] = int(levels[k_index])
        decoded.append(LayerwiseDecodedAction(fusion, k_by_block))
    return compute_variable_cost(decoded)


def one_coordinate_neighbors(action_matrix: Sequence[Sequence[int]]) -> Iterator[list[list[int]]]:
    """Yield every legal one-coordinate alternative for a layerwise policy action."""
    levels = _validate_k_levels()
    rows = [list(map(int, row)) for row in action_matrix]
    if not rows or any(len(row) != len(LAYERWISE_SLOT_NAMES) for row in rows):
        raise ValueError("action_matrix must have shape num_layers x 6")
    dims = (2,) + (len(levels),) * 5
    for layer_idx, row in enumerate(rows):
        for slot_idx, value in enumerate(row):
            if not 0 <= value < dims[slot_idx]:
                raise ValueError(
                    f"action_matrix[{layer_idx}][{slot_idx}]={value} outside [0, {dims[slot_idx]})"
                )

    for layer_idx, row in enumerate(rows):
        for slot_idx, value in enumerate(row):
            for alternative in range(dims[slot_idx]):
                if alternative == value:
                    continue
                neighbor = [candidate[:] for candidate in rows]
                neighbor[layer_idx][slot_idx] = alternative
                yield neighbor
