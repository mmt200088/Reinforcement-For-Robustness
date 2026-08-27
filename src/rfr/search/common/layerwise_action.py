"""Torch-free codec for the canonical Stage-2 layerwise policy.

The full action vector remains the runtime interchange format. This
module owns only the compact policy action: one Block4 fusion choice and one
high/medium/low truncation-precision preset per Transformer layer.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence, Tuple

import numpy as np

from .precision_presets import (
    PRECISION_PRESETS,
    network_axis_weights,
    precision_preset,
    validate_communication_importance_ratio,
)
from .truncation_levels import (
    K_LEVELS,
    K_MAX_BITS,
    validate_exact_k_domain,
)


LAYERWISE_SLOT_NAMES = (
    "block4_fusion",
    "truncation_precision",
)
LAYER_GENE_CARDINALITY = 6


def encode_layer_gene(layer_action: Sequence[int]) -> int:
    """Encode one public ``(fusion, preset)`` row as one atomic gene."""
    values = tuple(int(value) for value in layer_action)
    if len(values) != len(LAYERWISE_SLOT_NAMES):
        raise ValueError(
            f"layer action expects {len(LAYERWISE_SLOT_NAMES)} slots, "
            f"got {len(values)}"
        )
    fusion, preset = values
    if fusion not in (0, 1):
        raise ValueError(f"Block4 fusion index {fusion} outside [0, 2)")
    if not 0 <= preset < len(PRECISION_PRESETS):
        raise ValueError(
            f"precision preset index {preset} outside "
            f"[0, {len(PRECISION_PRESETS)})"
        )
    return int(3 * fusion + preset)


def decode_layer_gene(gene: int) -> Tuple[int, int]:
    """Decode one atomic six-valued gene to the public runtime row."""
    value = int(gene)
    if not 0 <= value < LAYER_GENE_CARDINALITY:
        raise ValueError(
            f"layer gene {value} outside [0, {LAYER_GENE_CARDINALITY})"
        )
    fusion, preset = divmod(value, len(PRECISION_PRESETS))
    return int(fusion), int(preset)


def encode_layerwise_action_matrix(
        action_matrix: Sequence[Sequence[int]],
        ) -> Tuple[int, ...]:
    """Encode a nonempty public action matrix as atomic per-layer genes."""
    rows = tuple(tuple(int(value) for value in row) for row in action_matrix)
    if not rows:
        raise ValueError("action_matrix must contain at least one layer")
    return tuple(encode_layer_gene(row) for row in rows)


def decode_layerwise_action_genes(
        genes: Sequence[int],
        ) -> Tuple[Tuple[int, int], ...]:
    """Decode atomic per-layer genes to the public Stage-2 action matrix."""
    values = tuple(int(value) for value in genes)
    if not values:
        raise ValueError("layer genes must contain at least one layer")
    return tuple(decode_layer_gene(value) for value in values)


def _validate_k_levels() -> Tuple[int, ...]:
    return validate_exact_k_domain(K_LEVELS)


_BLOCK_SLOT_COUNTS = {1: 9, 2: 23, 3: 8, 4: 17, 5: 16}
_BLOCK_STARTS = {1: 0, 2: 9, 3: 32, 4: 40, 5: 57}
_LAYER_WIDTH = sum(_BLOCK_SLOT_COUNTS.values())
_BLOCK_ORDER = (1, 2, 3, 4, 5)


def _validated_num_layers(num_layers: int) -> int:
    layers = int(num_layers)
    if layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {layers}")
    return layers


def truncation_k_summary_from_full_action(
        action_vec: Sequence[int],
        num_layers: int,
        *,
        k_levels: Sequence[int] = K_LEVELS,
        ) -> Tuple[int, int, float]:
    """Decode the five per-layer K slots from one full action vector."""
    layers = _validated_num_layers(num_layers)
    levels = validate_exact_k_domain(k_levels)
    try:
        raw_values = tuple(action_vec)
    except TypeError as exc:
        raise ValueError("Stage-2 full action vector must be a sequence") from exc
    expected_length = layers * _LAYER_WIDTH + 1
    if len(raw_values) != expected_length:
        raise ValueError(
            "Stage-2 full action vector length mismatch: "
            f"expected {expected_length}, got {len(raw_values)}"
        )
    values = []
    try:
        for value in raw_values:
            if isinstance(value, bool) or type(value).__name__ == "bool_":
                raise TypeError("boolean action indices are invalid")
            values.append(operator.index(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Stage-2 full action vector must contain only integer indices"
        ) from exc

    total = 0
    count = 0
    for layer_idx in range(layers):
        layer_start = layer_idx * _LAYER_WIDTH
        for block_idx in _BLOCK_ORDER:
            slot = (
                layer_start
                + _BLOCK_STARTS[block_idx]
                + _BLOCK_SLOT_COUNTS[block_idx]
                - 1
            )
            level_index = values[slot]
            if not 0 <= level_index < len(levels):
                raise ValueError(
                    f"Stage-2 K action index {level_index} is out of range"
                )
            total += int(levels[level_index])
            count += 1
    return int(total), int(count), float(total / count)


def layerwise_action_space_version(num_layers: int) -> str:
    """Return the persisted action-space identity for one model depth."""
    layers = _validated_num_layers(num_layers)
    return f"stage2_layerwise_{layers}x{len(LAYERWISE_SLOT_NAMES)}_hml_v3"


def max_compute_saving_units(num_layers: int) -> float:
    """Maximum learnable Block4 fusion count for the model depth."""
    return float(_validated_num_layers(num_layers))


def max_communication_saving_units(num_layers: int) -> float:
    """Maximum count of per-layer low-precision utility units."""
    return float(_validated_num_layers(num_layers))


MAX_COMPUTE_SAVING_UNITS = max_compute_saving_units(12)
MAX_COMMUNICATION_SAVING_UNITS = max_communication_saving_units(12)
LAYERWISE_DECODE_VERSION = "layerwise_hml_action_v3"
LAYERWISE_COST_MODEL_REVISION = "network_weighted_compute_communication_v3"


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
class FusionMaterializationBlock:
    """One mapped block in the persisted full-action representation."""

    artifact_index: int
    layer_idx: int
    block_idx: int
    graph_key: str
    full_vec_offsets: Tuple[int, ...]


@dataclass(frozen=True)
class LayerwiseDecodedAction:
    block4_fusion: int
    k_by_block: Mapping[int, int]
    precision_preset_index: int = -1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "k_by_block",
            MappingProxyType({int(block): int(k_value) for block, k_value in self.k_by_block.items()}),
        )
        object.__setattr__(self, "precision_preset_index", int(self.precision_preset_index))

    @property
    def precision_preset_name(self) -> str:
        return precision_preset(self.precision_preset_index).name


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
            LayerwiseDecodedAction(
                self.decoded.block4_fusion,
                self.decoded.k_by_block,
                self.decoded.precision_preset_index,
            ),
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
class LayerwiseMaterialization:
    """Materialize one full action exactly for strict evaluation."""

    mode: str
    full_vector: np.ndarray
    action_matrix: Tuple[Tuple[int, int], ...]
    boosted_overrides: Mapping[Tuple[int, int], Mapping[str, int]]

    def __post_init__(self) -> None:
        vector = np.asarray(self.full_vector, dtype=int).reshape(-1).copy()
        vector.setflags(write=False)
        matrix = tuple(
            tuple(int(value) for value in row)
            for row in self.action_matrix
        )
        if not matrix or any(len(row) != len(LAYERWISE_SLOT_NAMES) for row in matrix):
            raise ValueError("materialized action_matrix must have shape num_layers x 2")
        object.__setattr__(self, "mode", str(self.mode))
        object.__setattr__(self, "full_vector", vector)
        object.__setattr__(self, "action_matrix", matrix)
        object.__setattr__(
            self,
            "boosted_overrides",
            MappingProxyType({
                (int(block_idx), int(layer_idx)): MappingProxyType({
                    str(name): int(value)
                    for name, value in fields.items()
                })
                for (block_idx, layer_idx), fields in self.boosted_overrides.items()
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
    compute_weight: float
    communication_weight: float
    communication_importance_ratio: float
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
        communication_importance_ratio: float = 1.0,
        ) -> Tuple[float, float, float]:
    """Return diagnostics plus the network-weighted bounded PPO score."""
    compute = _unit_interval("compute_saving", compute_saving)
    communication = _unit_interval("communication_saving", communication_saving)
    robust_floor = min(compute, communication)
    secondary_progress = 0.5 * (compute + communication)
    compute_weight, communication_weight = network_axis_weights(
        communication_importance_ratio,
    )
    score = compute_weight * compute + communication_weight * communication
    return float(robust_floor), float(secondary_progress), float(score)


def resource_shapley_credits(
        compute_saving: float,
        communication_saving: float,
        communication_importance_ratio: float = 1.0,
        ) -> Tuple[float, float]:
    """Return direct, separable resource-family credits.

    The historical function name remains as a read-only API alias for reports
    and fixtures. No Shapley decomposition is used by the v3 objective.
    """
    compute = _unit_interval("compute_saving", compute_saving)
    communication = _unit_interval("communication_saving", communication_saving)
    compute_weight, communication_weight = network_axis_weights(
        communication_importance_ratio,
    )
    compute_credit = compute_weight * compute
    communication_credit = communication_weight * communication
    return float(compute_credit), float(communication_credit)


def _graph_keys(layer_idx: int, profile: str, gelu_degree: int) -> Tuple[Tuple[int, str], ...]:
    pairs = []
    if layer_idx > 0:
        pairs.append((1, f"block1_{profile}"))
    pairs.extend(((2, f"block2_{profile}"), (4, "block4"), (5, f"block5_n{gelu_degree}")))
    return tuple(pairs)


def fusion_materialization_blocks(
        num_layers: int,
        *,
        profile: str = "mrpc",
        gelu_degrees: Sequence[int] | None = None,
        ) -> Tuple[FusionMaterializationBlock, ...]:
    """Describe mapped blocks without exposing a policy decision schedule."""
    layers = _validated_num_layers(num_layers)
    if gelu_degrees is not None and len(gelu_degrees) != layers:
        raise ValueError(
            f"gelu_degrees has {len(gelu_degrees)} values, expected {layers}"
        )

    blocks = []
    for layer_idx in range(layers):
        gelu_degree = (
            int(gelu_degrees[layer_idx])
            if gelu_degrees is not None
            else 4
        )
        graph_keys = dict(_graph_keys(layer_idx, str(profile), gelu_degree))
        block_order = (2, 4, 5) if layer_idx == 0 else (1, 2, 4, 5)
        for block_idx in block_order:
            blocks.append(FusionMaterializationBlock(
                artifact_index=len(blocks),
                layer_idx=layer_idx,
                block_idx=block_idx,
                graph_key=graph_keys[block_idx],
                full_vec_offsets=tuple(_block_offsets(layer_idx, block_idx)),
            ))
    return tuple(blocks)


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
            f"{graph_key}: K slot {k_slot_index} is not the final slot "
            f"{expected_slots - 1}"
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
    """Return one two-slot policy step per Transformer layer."""
    _validate_k_levels()
    layers = int(num_layers)
    if layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {layers}")
    if gelu_degrees is not None and len(gelu_degrees) != layers:
        raise ValueError(f"gelu_degrees has {len(gelu_degrees)} values, expected {layers}")

    slot_dims = (2, len(PRECISION_PRESETS))
    specs = []
    for layer_idx in range(layers):
        gelu_degree = int(gelu_degrees[layer_idx]) if gelu_degrees is not None else 4
        spec = LayerwiseStepSpec(
            step_idx=layer_idx,
            layer_idx=layer_idx,
            slot_dims=slot_dims,
            slot_mask=(True, True),
            terminal=(layer_idx == layers - 1),
            num_layers=layers,
            graph_keys_by_block=_graph_keys(layer_idx, str(profile), gelu_degree),
        )
        _validate_graphs(spec, fusion_map)
        specs.append(spec)
    return specs


def layerwise_fusion_option_by_step(
        action_matrix: Sequence[Sequence[int]],
        schedule: Sequence[LayerwiseStepSpec],
        fusion_map: Any,
        ) -> Mapping[str, int]:
    """Project the compact action to the executable fusion-step map.

    Stable numeric keys preserve the full-action artifact format: layer 0 owns
    ``B2,B4,B5`` and later layers own ``B1,B2,B4,B5``. Block 2 and Block 5 are
    fixed at fusion-count 1 while Block 4 follows the policy action.
    """
    rows = tuple(tuple(int(value) for value in row) for row in action_matrix)
    specs = tuple(schedule)
    if not rows or len(rows) != len(specs):
        raise ValueError(
            "action_matrix and schedule must contain the same nonzero layer count"
        )
    if any(len(row) != len(LAYERWISE_SLOT_NAMES) for row in rows):
        raise ValueError("action_matrix must have shape num_layers x 2")

    option_by_step: dict[str, int] = {}
    artifact_index = 0
    for row, spec in zip(rows, specs):  # noqa: B905 - lengths checked above
        _validate_layer_action(row, spec)
        graph_keys = dict(spec.graph_keys_by_block)
        block_order = (2, 4, 5) if int(spec.layer_idx) == 0 else (1, 2, 4, 5)
        for block_idx in block_order:
            if block_idx in (2, 4, 5):
                fusion_count = int(row[0]) if block_idx == 4 else 1
                graph_key = graph_keys[block_idx]
                option = _unique_option_for_fusion_count(
                    fusion_map,
                    graph_key,
                    fusion_count,
                )
                option_by_step[str(artifact_index)] = int(option.option_id)
            artifact_index += 1
    return MappingProxyType(option_by_step)


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
    """Copy a full vector and apply exactly one layer's policy action."""
    del profile, gelu_degree
    _validate_k_levels()
    action = _validate_layer_action(layer_action, step_spec)
    expected_size = int(step_spec.num_layers) * _LAYER_WIDTH + 1
    result = np.asarray(full_vector, dtype=int).reshape(-1).copy()
    if result.size != expected_size:
        raise ValueError(f"full_vector has {result.size} slots, expected {expected_size}")
    _validate_graphs(step_spec, fusion_map)

    graph_keys = dict(step_spec.graph_keys_by_block)
    preset_index = int(action[1])
    preset = precision_preset(preset_index)
    k_by_block = {
        block_idx: int(preset.k_by_block[block_idx - 1])
        for block_idx in _BLOCK_ORDER
    }
    k_indices = {
        block_idx: int(K_LEVELS.index(k_by_block[block_idx]))
        for block_idx in _BLOCK_ORDER
    }
    active_blocks = _BLOCK_ORDER
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
        decoded=LayerwiseDecodedAction(
            block4_fusion=int(action[0]),
            k_by_block=k_by_block,
            precision_preset_index=preset_index,
        ),
        fusion_option_ids=fusion_option_ids,
        boosted_field_values_by_block=boosted_values,
    )


def materialize_layerwise_counterfactuals(
        baseline_full_vector: Sequence[int],
        action_matrix: Sequence[Sequence[int]],
        schedule: Sequence[LayerwiseStepSpec],
        fusion_map: Any,
        ) -> Mapping[str, LayerwiseMaterialization]:
    """Materialize joint and isolated-axis configs from one policy action.

    ``compute_only`` preserves every installed fusion option but resets every
    truncation K to the statistical baseline K=13. ``communication_only``
    starts from the statistical baseline vector and changes only the five K
    slots per layer, so no fusion option or boosted noise is installed.
    """
    rows = tuple(tuple(int(value) for value in row) for row in action_matrix)
    specs = tuple(schedule)
    if not rows or len(rows) != len(specs):
        raise ValueError(
            "action_matrix and schedule must contain the same nonzero layer count"
        )
    if any(len(row) != len(LAYERWISE_SLOT_NAMES) for row in rows):
        raise ValueError("action_matrix must have shape num_layers x 2")
    if any(int(spec.step_idx) != index for index, spec in enumerate(specs)):
        raise ValueError("layerwise schedule must be ordered by step_idx")

    baseline = np.asarray(baseline_full_vector, dtype=int).reshape(-1).copy()
    expected_size = len(specs) * _LAYER_WIDTH + 1
    if baseline.size != expected_size:
        raise ValueError(
            f"baseline_full_vector has {baseline.size} slots, expected {expected_size}"
        )

    joint_vector = baseline.copy()
    joint_overrides: dict[Tuple[int, int], dict[str, int]] = {}
    for row, spec in zip(rows, specs):
        application = apply_layer_action(
            joint_vector,
            row,
            spec,
            fusion_map,
        )
        joint_vector = application.full_vector.copy()
        for block_idx, fields in application.boosted_field_values_by_block.items():
            joint_overrides[(int(block_idx), int(spec.layer_idx))] = {
                str(name): int(value) for name, value in fields.items()
            }

    baseline_k_index = int(K_LEVELS.index(K_MAX_BITS))
    compute_vector = joint_vector.copy()
    compute_overrides = {
        key: dict(fields) for key, fields in joint_overrides.items()
    }
    communication_vector = baseline.copy()
    for row, spec in zip(rows, specs):
        preset = precision_preset(row[1])
        for block_idx, k_value in zip(_BLOCK_ORDER, preset.k_by_block):
            k_offset = _block_offsets(spec.layer_idx, block_idx).stop - 1
            compute_vector[k_offset] = baseline_k_index
            communication_vector[k_offset] = int(K_LEVELS.index(int(k_value)))
        for block_idx in _BLOCK_ORDER:
            fields = compute_overrides.get((block_idx, int(spec.layer_idx)))
            if fields is not None:
                fields["output_truncation_k"] = int(K_MAX_BITS)

    return MappingProxyType({
        "joint": LayerwiseMaterialization(
            mode="joint",
            full_vector=joint_vector,
            action_matrix=rows,
            boosted_overrides=joint_overrides,
        ),
        "compute_only": LayerwiseMaterialization(
            mode="compute_only",
            full_vector=compute_vector,
            action_matrix=rows,
            boosted_overrides=compute_overrides,
        ),
        "communication_only": LayerwiseMaterialization(
            mode="communication_only",
            full_vector=communication_vector,
            action_matrix=rows,
            boosted_overrides={},
        ),
    })


def _decoded_preset_index(action: LayerwiseDecodedAction, layer_idx: int) -> int:
    index = int(action.precision_preset_index)
    if 0 <= index < len(PRECISION_PRESETS):
        expected = {
            block_idx: PRECISION_PRESETS[index].k_by_block[block_idx - 1]
            for block_idx in _BLOCK_ORDER
        }
        if dict(action.k_by_block) != expected:
            raise ValueError(
                f"layer {layer_idx} preset {index} does not match decoded K values"
            )
        return index
    observed = tuple(int(action.k_by_block[block]) for block in _BLOCK_ORDER)
    matches = [
        preset_index
        for preset_index, preset in enumerate(PRECISION_PRESETS)
        if tuple(preset.k_by_block) == observed
    ]
    if len(matches) != 1:
        raise ValueError(
            f"layer {layer_idx} K values {observed} do not identify one precision preset"
        )
    return int(matches[0])


def compute_variable_cost(
        actions: Sequence[LayerwiseDecodedAction],
        *,
        communication_importance_ratio: float = 1.0,
        ) -> VariableCost:
    """Compute independent compute/communication savings from decoded actions."""
    _validate_k_levels()
    num_layers = len(actions)
    if num_layers < 1:
        raise ValueError("variable cost requires at least one layer action")
    ratio = validate_communication_importance_ratio(
        communication_importance_ratio,
    )
    compute_weight, communication_weight = network_axis_weights(ratio)
    compute_denominator = max_compute_saving_units(num_layers)
    active_k_slots = 5 * num_layers
    fusion_values = []
    k_values = []
    communication_utilities = []
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
        preset_index = _decoded_preset_index(action, layer_idx)
        communication_utility = float(
            PRECISION_PRESETS[preset_index].communication_utility
        )
        communication_utilities.append(communication_utility)
        for block_idx, k_value in action.k_by_block.items():
            k = int(k_value)
            if k not in K_LEVELS:
                raise ValueError(f"layer {layer_idx} block {block_idx} has unsupported K={k}")
            k_values.append(k)
        current_slot_contributions = [
            compute_weight * float(fusion) / compute_denominator,
            communication_weight * communication_utility / float(num_layers),
        ]
        raw_slot_contributions.append(current_slot_contributions)
    if len(k_values) != active_k_slots:
        raise RuntimeError(
            f"{num_layers}-layer actions yielded {len(k_values)} active K values, "
            f"expected {active_k_slots}"
        )
    fusion_count = int(sum(fusion_values))
    removed_k_bits = int(sum(K_MAX_BITS - k for k in k_values))
    compute_saving = float(fusion_count) / compute_denominator
    communication_saving = float(sum(communication_utilities)) / float(num_layers)
    robust_floor, secondary_progress, ppo_resource_score = dual_resource_score(
        compute_saving,
        communication_saving,
        ratio,
    )
    compute_credit, communication_credit = resource_shapley_credits(
        compute_saving,
        communication_saving,
        ratio,
    )

    slot_resource_rewards = [
        tuple(float(value) for value in raw_row)
        for raw_row in raw_slot_contributions
    ]
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
        compute_weight=float(compute_weight),
        communication_weight=float(communication_weight),
        communication_importance_ratio=float(ratio),
        fusion_count=int(fusion_count),
        removed_k_bits=int(removed_k_bits),
        layer_resource_rewards=layer_resource_rewards,
        slot_resource_rewards=tuple(slot_resource_rewards),
    )


def compute_variable_cost_from_action_matrix(
        action_matrix: Sequence[Sequence[int]],
        *,
        communication_importance_ratio: float = 1.0,
        ) -> VariableCost:
    """Decode a canonical ``num_layers x 2`` policy action and compute its cost."""
    _validate_k_levels()
    rows = [tuple(int(value) for value in row) for row in action_matrix]
    if not rows or any(len(row) != len(LAYERWISE_SLOT_NAMES) for row in rows):
        raise ValueError("action_matrix must have shape num_layers x 2")
    decoded = []
    for layer_idx, row in enumerate(rows):
        fusion = int(row[0])
        if fusion not in (0, 1):
            raise ValueError(
                f"action_matrix[{layer_idx}][0]={fusion} outside [0, 2)"
            )
        preset_index = int(row[1])
        preset = precision_preset(preset_index)
        k_by_block = {
            block_idx: int(preset.k_by_block[block_idx - 1])
            for block_idx in _BLOCK_ORDER
        }
        decoded.append(LayerwiseDecodedAction(fusion, k_by_block, preset_index))
    return compute_variable_cost(
        decoded,
        communication_importance_ratio=communication_importance_ratio,
    )


def describe_layerwise_action_matrix(
        action_matrix: Sequence[Sequence[int]],
        ) -> list[dict[str, Any]]:
    """Return the exact per-layer fusion and truncation configuration."""
    rows = [tuple(int(value) for value in row) for row in action_matrix]
    if not rows or any(len(row) != len(LAYERWISE_SLOT_NAMES) for row in rows):
        raise ValueError("action_matrix must have shape num_layers x 2")
    description = []
    for layer_idx, (fusion, preset_index) in enumerate(rows):
        if fusion not in (0, 1):
            raise ValueError(
                f"action_matrix[{layer_idx}][0]={fusion} outside [0, 2)"
            )
        preset = precision_preset(preset_index)
        simulation_k_by_block = {
            f"block{block_idx}": int(k_value)
            for block_idx, k_value in enumerate(
                preset.simulation_k_by_block, start=1,
            )
        }
        description.append({
            "layer_idx": int(layer_idx),
            "block4_fusion_count": int(fusion),
            "precision_preset_index": int(preset_index),
            "precision_preset_name": str(preset.name),
            "truncation_k_by_block": dict(simulation_k_by_block),
            "cleartext_simulation_k_by_block": dict(simulation_k_by_block),
            "ciphertext_truncation_k_by_block": {
                f"block{block_idx}": int(k_value)
                for block_idx, k_value in enumerate(
                    preset.ciphertext_k_by_block, start=1,
                )
            },
            "reserve_bits_by_block": {
                f"block{block_idx}": int(reserve_bits)
                for block_idx, reserve_bits in enumerate(
                    preset.reserve_bits_by_block, start=1,
                )
            },
            "ciphertext_ring_bits": int(preset.ciphertext_ring_bits),
        })
    return description


def one_coordinate_neighbors(action_matrix: Sequence[Sequence[int]]) -> Iterator[list[list[int]]]:
    """Yield every legal one-coordinate alternative for a layerwise policy action."""
    _validate_k_levels()
    rows = [list(map(int, row)) for row in action_matrix]
    if not rows or any(len(row) != len(LAYERWISE_SLOT_NAMES) for row in rows):
        raise ValueError("action_matrix must have shape num_layers x 2")
    dims = (2, len(PRECISION_PRESETS))
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
