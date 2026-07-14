"""Torch-free codec for the canonical 12-step Stage-2 layerwise policy.

The legacy full action vector remains the runtime interchange format.  This
module owns only the compact policy action: one Block4 fusion choice and five
truncation-K choices per Transformer layer.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence, Tuple

import numpy as np


LAYERWISE_SLOT_NAMES = (
    "block4_fusion",
    "block1_k",
    "block2_k",
    "block3_k",
    "block4_k",
    "block5_k",
)

# Keep this loader in step with action_space.K_LEVELS without importing
# action_space, which deliberately remains outside the torch-free boundary.
_DEFAULT_K_LEVELS = (8, 9, 11, 13, 10, 12)


def _load_k_levels() -> Tuple[int, ...]:
    raw = str(os.environ.get("BLB_TRUNCATION_K_LEVELS", "") or "").strip()
    if not raw:
        return _DEFAULT_K_LEVELS
    try:
        levels = tuple(int(value.strip()) for value in raw.split(","))
    except ValueError as exc:
        raise ValueError("BLB_TRUNCATION_K_LEVELS must contain only integers") from exc
    if not levels or any(value == "" for value in raw.split(",")):
        raise ValueError("BLB_TRUNCATION_K_LEVELS must contain at least one integer")
    if len(set(levels)) != len(levels):
        raise ValueError(f"BLB_TRUNCATION_K_LEVELS contains duplicate values: {levels}")
    return levels


K_LEVELS = _load_k_levels()
_REQUIRED_K_VALUES = frozenset((8, 9, 10, 11, 12, 13))


def _validate_k_levels() -> Tuple[int, ...]:
    levels = tuple(int(value) for value in K_LEVELS)
    if len(levels) != len(_REQUIRED_K_VALUES) or frozenset(levels) != _REQUIRED_K_VALUES:
        raise ValueError(
            "K_LEVELS must contain each supported K value exactly once: "
            f"{sorted(_REQUIRED_K_VALUES)}, got {levels}"
        )
    return levels

# These are the stable legacy action_space._BLOCK_SPECS field counts, in block
# order.  All blocks put output_truncation_k in their last slot.
_BLOCK_SLOT_COUNTS = {1: 9, 2: 23, 3: 8, 4: 17, 5: 16}
_BLOCK_STARTS = {1: 0, 2: 9, 3: 32, 4: 40, 5: 57}
_LAYER_WIDTH = sum(_BLOCK_SLOT_COUNTS.values())
_BLOCK_ORDER = (1, 2, 3, 4, 5)

BLOCK4_FUSION_COST_UNIT = 1.0
TRUNCATION_COST_UNIT_PER_BIT = 0.5
MAX_FUSION_COST_UNITS = 12.0
MAX_TRUNCATION_COST_UNITS = 0.5 * 59.0 * (13.0 - 8.0)
MAX_VARIABLE_COST_UNITS = MAX_FUSION_COST_UNITS + MAX_TRUNCATION_COST_UNITS


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
    fusion_saving: float
    truncation_saving: float
    normalized: float
    fusion_units: float
    truncation_units: float
    total_units: float
    max_units: float
    layer_cost_rewards: Tuple[float, ...]


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
            slot_mask=(True, layer_idx != 0, True, True, True, True),
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
    active_blocks = (2, 3, 4, 5) if step_spec.layer_idx == 0 else _BLOCK_ORDER
    k_by_block = {block_idx: int(K_LEVELS[k_indices[block_idx]]) for block_idx in active_blocks}
    fusion_option_ids: dict[int, int] = {}
    boosted_values: dict[int, Mapping[str, int]] = {}

    choices = ((1, 0), (2, 1), (4, action[0]), (5, 1))
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

    block3_k_offset = _block_offsets(step_spec.layer_idx, 3).stop - 1
    result[block3_k_offset] = k_indices[3]
    return LayerActionApplication(
        full_vector=result,
        decoded=LayerwiseDecodedAction(block4_fusion=int(action[0]), k_by_block=k_by_block),
        fusion_option_ids=fusion_option_ids,
        boosted_field_values_by_block=boosted_values,
    )


def compute_variable_cost(actions: Sequence[LayerwiseDecodedAction]) -> VariableCost:
    """Compute the learnable BERT-base variable cost from decoded actions."""
    _validate_k_levels()
    if len(actions) != 12:
        raise ValueError(f"variable cost requires 12 layer actions, got {len(actions)}")
    fusion_values = []
    k_values = []
    layer_units = []
    for layer_idx, action in enumerate(actions):
        expected_blocks = {2, 3, 4, 5} if layer_idx == 0 else {1, 2, 3, 4, 5}
        actual_blocks = set(action.k_by_block)
        if actual_blocks != expected_blocks:
            raise ValueError(
                f"layer {layer_idx} K blocks {sorted(actual_blocks)} do not match {sorted(expected_blocks)}"
            )
        fusion = int(action.block4_fusion)
        if fusion not in (0, 1):
            raise ValueError(f"layer {layer_idx} Block4 fusion must be 0 or 1, got {fusion}")
        fusion_values.append(fusion)
        current_layer_k_values = []
        for block_idx, k_value in action.k_by_block.items():
            k = int(k_value)
            if k not in K_LEVELS:
                raise ValueError(f"layer {layer_idx} block {block_idx} has unsupported K={k}")
            k_values.append(k)
            current_layer_k_values.append(k)
        layer_units.append(
            BLOCK4_FUSION_COST_UNIT * float(fusion)
            + TRUNCATION_COST_UNIT_PER_BIT
            * sum(13.0 - float(k) for k in current_layer_k_values)
        )
    if len(k_values) != 59:
        raise RuntimeError(f"BERT-base layer actions yielded {len(k_values)} active K values, expected 59")
    fusion_units = BLOCK4_FUSION_COST_UNIT * float(sum(fusion_values))
    truncation_units = TRUNCATION_COST_UNIT_PER_BIT * float(
        sum(13 - k for k in k_values)
    )
    total_units = fusion_units + truncation_units
    fusion_saving = fusion_units / MAX_FUSION_COST_UNITS
    truncation_saving = truncation_units / MAX_TRUNCATION_COST_UNITS
    return VariableCost(
        fusion_saving=float(fusion_saving),
        truncation_saving=float(truncation_saving),
        normalized=float(total_units / MAX_VARIABLE_COST_UNITS),
        fusion_units=float(fusion_units),
        truncation_units=float(truncation_units),
        total_units=float(total_units),
        max_units=float(MAX_VARIABLE_COST_UNITS),
        layer_cost_rewards=tuple(
            float(value / MAX_VARIABLE_COST_UNITS) for value in layer_units
        ),
    )


def one_coordinate_neighbors(action_matrix: Sequence[Sequence[int]]) -> Iterator[list[list[int]]]:
    """Yield every legal one-coordinate alternative for a 12x6 policy action."""
    levels = _validate_k_levels()
    rows = [list(map(int, row)) for row in action_matrix]
    if len(rows) != 12 or any(len(row) != len(LAYERWISE_SLOT_NAMES) for row in rows):
        raise ValueError("action_matrix must have shape 12x6")
    dims = (2,) + (len(levels),) * 5
    for layer_idx, row in enumerate(rows):
        for slot_idx, value in enumerate(row):
            if layer_idx == 0 and slot_idx == 1:
                continue
            if not 0 <= value < dims[slot_idx]:
                raise ValueError(
                    f"action_matrix[{layer_idx}][{slot_idx}]={value} outside [0, {dims[slot_idx]})"
                )

    for layer_idx, row in enumerate(rows):
        for slot_idx, value in enumerate(row):
            if layer_idx == 0 and slot_idx == 1:
                continue
            for alternative in range(dims[slot_idx]):
                if alternative == value:
                    continue
                neighbor = [candidate[:] for candidate in rows]
                neighbor[layer_idx][slot_idx] = alternative
                yield neighbor
