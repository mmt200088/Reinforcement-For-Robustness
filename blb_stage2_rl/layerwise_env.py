"""Twelve-step Stage-2 environment with one policy action per BERT layer."""

from __future__ import annotations

import copy
from dataclasses import dataclass, fields, is_dataclass
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .action_space import make_all_max_action_vector
from .layerwise_action import (
    LAYERWISE_SLOT_NAMES,
    LayerwiseDecodedAction,
    LayerwiseStepSpec,
    apply_layer_action,
    compute_variable_cost,
    layerwise_schedule,
)
from .sequential_env import BlockRuntimeResult, evaluate_block_from_full_vector


_SNAPSHOT_OMIT = object()
_SNAPSHOT_MAX_DEPTH = 5
_SNAPSHOT_MAX_ITEMS = 256
_TERMINAL_INFO_SNAPSHOT_FIELDS = (
    "metrics",
    "reward_breakdown",
    "action_hash",
    "invalid",
    "apply_failed",
    "eval_failed",
    "forward_ran",
    "forward_skipped_reason",
    "error",
    "optimizer_baseline_action",
    "optimizer_eval_mode",
    "optimizer_invalid_summary",
    "model_degree_sync",
    "optimizer_cfg_overrides",
    "opt_outputs_keys",
    "opt_signals",
    "timing",
    "probe_diagnostics",
    "raw_trials",
    "trial_metrics",
    "priority",
    "reward",
    "fusion_count_b2",
    "fusion_count_b4",
    "fusion_count_b5",
    "fusion_action_steps",
    "fusion_cost_fusion_norm",
    "fusion_cost_fusion_norm_saturated",
    "fusion_saturation_tau",
    "fusion_cost_trunc_norm",
    "fusion_cost_rank",
    "fusion_cost_max_actual",
)


def _bounded_json_value(value: Any, *, depth: int = 0) -> Any:
    """Convert one selected diagnostic value without traversing opaque objects."""
    if depth > _SNAPSHOT_MAX_DEPTH:
        return _SNAPSHOT_OMIT
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _bounded_json_value(value.item(), depth=depth)
    if isinstance(value, np.ndarray):
        if int(value.size) > _SNAPSHOT_MAX_ITEMS:
            return _SNAPSHOT_OMIT
        return _bounded_json_value(value.tolist(), depth=depth + 1)
    if is_dataclass(value) and not isinstance(value, type):
        result: Dict[str, Any] = {}
        for field_def in fields(value)[:_SNAPSHOT_MAX_ITEMS]:
            converted = _bounded_json_value(
                getattr(value, field_def.name), depth=depth + 1,
            )
            if converted is not _SNAPSHOT_OMIT:
                result[str(field_def.name)] = converted
        return result
    if isinstance(value, Mapping):
        result = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= _SNAPSHOT_MAX_ITEMS:
                break
            if isinstance(key, np.generic):
                key = key.item()
            if not isinstance(key, (str, bool, int, float)):
                continue
            converted = _bounded_json_value(item, depth=depth + 1)
            if converted is not _SNAPSHOT_OMIT:
                result[str(key)] = converted
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result = []
        for index, item in enumerate(value):
            if index >= _SNAPSHOT_MAX_ITEMS:
                break
            converted = _bounded_json_value(item, depth=depth + 1)
            if converted is not _SNAPSHOT_OMIT:
                result.append(converted)
        return result
    return _SNAPSHOT_OMIT


def _snapshot_terminal_info(terminal_info: Any) -> Dict[str, Any]:
    """Take a compact report snapshot from the production terminal payload."""
    if not isinstance(terminal_info, Mapping):
        return {}
    snapshot: Dict[str, Any] = {}
    for name in _TERMINAL_INFO_SNAPSHOT_FIELDS:
        if name not in terminal_info:
            continue
        converted = _bounded_json_value(terminal_info[name], depth=1)
        if converted is not _SNAPSHOT_OMIT:
            snapshot[name] = converted
    return snapshot


@dataclass(frozen=True)
class LayerwiseEnvConfig:
    """Observation normalization constants for layerwise rollouts."""

    total_bits_scale: float = 1000.0
    fusion_count_scale: float = 10.0


def _degree_vector(value: Any, *, num_layers: int, default: int) -> List[int]:
    try:
        values = np.asarray(value, dtype=int).reshape(-1)
    except Exception:
        return [int(default)] * int(num_layers)
    if values.size == 0:
        return [int(default)] * int(num_layers)
    if values.size == 1:
        return [int(values[0])] * int(num_layers)
    if values.size != int(num_layers):
        raise ValueError(
            f"degree vector has {values.size} values, expected {int(num_layers)}"
        )
    return [int(item) for item in values]


class BLBStage2LayerwiseEnv:
    """Aggregate all active block replans into one outer step per layer."""

    def __init__(
            self,
            *,
            base_env: Any,
            fusion_map: Any,
            env_cfg: Optional[LayerwiseEnvConfig] = None,
            profile: Optional[str] = None,
            ):
        self.base = base_env
        self.fusion_map = fusion_map
        self.cfg = env_cfg or LayerwiseEnvConfig()
        self.num_layers = int(base_env.num_layers)
        if self.num_layers != 12:
            raise ValueError(
                f"BLBStage2LayerwiseEnv requires 12 BERT layers, got {self.num_layers}"
            )
        self.profile = str(profile or base_env.env_cfg.profile)
        self.horizon = self.num_layers
        self._max_step_dim = len(LAYERWISE_SLOT_NAMES)
        self._schedule: List[LayerwiseStepSpec] = []
        self._gelu_degrees: List[int] = []
        self._attn_degrees: List[int] = []
        self._rebuild_schedule()

        self._pending_full_vec = np.asarray(
            make_all_max_action_vector(self.num_layers), dtype=int,
        ).reshape(-1).copy()
        self._step_idx = 0
        self._has_reset = False
        self._done = False
        self._runtime_terminal_info: Optional[Any] = None
        self._action_history: List[List[int]] = []
        self._decoded_actions: List[LayerwiseDecodedAction] = []
        self._fusion_option_ids: List[Dict[int, int]] = []
        self._layer_summaries: List[Dict[str, Any]] = []
        self._boosted_overrides: Dict[Tuple[int, int], Dict[str, int]] = {}
        self._action_obs = np.zeros((self.horizon, self._max_step_dim), dtype=np.float32)
        self._signal_obs = np.zeros((self.horizon, 4), dtype=np.float32)

    @property
    def schedule(self) -> List[LayerwiseStepSpec]:
        return list(self._schedule)

    @property
    def max_step_dim(self) -> int:
        return self._max_step_dim

    @property
    def state_dim(self) -> int:
        return 4 + self.horizon + self.horizon * self._max_step_dim + self.horizon * 4

    @property
    def pending_full_vector(self) -> np.ndarray:
        return self._pending_full_vec.copy()

    @property
    def action_history(self) -> List[List[int]]:
        return [row[:] for row in self._action_history]

    @property
    def optimizer_signal_history(self) -> List[Dict[str, Any]]:
        return [
            {
                "all_valid": bool(row["all_valid"]),
                "total_bits": int(row["total_bits"]),
                "fusion_count": int(row["fusion_count"]),
                "active_block_count": int(row["active_block_count"]),
            }
            for row in self._layer_summaries
        ]

    @property
    def layer_summaries(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self._layer_summaries)

    @property
    def boosted_overrides(self) -> Dict[Tuple[int, int], Dict[str, int]]:
        return copy.deepcopy(self._boosted_overrides)

    @property
    def runtime_terminal_info(self) -> Optional[Any]:
        """Original base-env terminal payload for in-process runtime consumers."""
        return self._runtime_terminal_info

    def current_spec(self) -> LayerwiseStepSpec:
        if self._done or self._step_idx >= self.horizon:
            raise RuntimeError("episode terminated; call reset() before current_spec()")
        return self._schedule[self._step_idx]

    def reset(self, *, seed: Optional[int] = None) -> np.ndarray:
        self.base.reset(seed=seed)
        self._rebuild_schedule()
        self._pending_full_vec = np.asarray(
            make_all_max_action_vector(self.num_layers), dtype=int,
        ).reshape(-1).copy()
        self._step_idx = 0
        self._has_reset = True
        self._done = False
        self._runtime_terminal_info = None
        self._action_history = []
        self._decoded_actions = []
        self._fusion_option_ids = []
        self._layer_summaries = []
        self._boosted_overrides = {}
        self._action_obs = np.zeros((self.horizon, self._max_step_dim), dtype=np.float32)
        self._signal_obs = np.zeros((self.horizon, 4), dtype=np.float32)
        return self._build_obs()

    def step(
            self,
            action: Sequence[int],
            ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if not self._has_reset:
            raise RuntimeError("call reset() before step()")
        if self._done:
            raise RuntimeError("episode already terminated; call reset() before step()")

        spec = self.current_spec()
        owned_action = [int(value) for value in action]
        application = apply_layer_action(
            self._pending_full_vec,
            owned_action,
            spec,
            self.fusion_map,
            profile=self.profile,
            gelu_degree=self._gelu_degrees[spec.layer_idx],
        )
        candidate_vector = np.asarray(application.full_vector, dtype=int).copy()
        graph_keys = dict(spec.graph_keys_by_block)
        graph_keys[3] = f"block3_exp_n{self._attn_degrees[spec.layer_idx]}"
        active_blocks = (2, 3, 4, 5) if spec.layer_idx == 0 else (1, 2, 3, 4, 5)

        runtimes: List[BlockRuntimeResult] = []
        block_summaries: List[Dict[str, Any]] = []
        for block_idx in active_blocks:
            boosted = application.boosted_field_values_by_block.get(block_idx)
            runtime = evaluate_block_from_full_vector(
                base_env=self.base,
                full_vec=candidate_vector,
                layer_idx=int(spec.layer_idx),
                block_idx=int(block_idx),
                graph_key=str(graph_keys[block_idx]),
                boosted_field_values=(dict(boosted) if boosted else None),
            )
            runtimes.append(runtime)
            block_summaries.append(self._runtime_summary(block_idx, runtime))

        all_valid = all(runtime.valid for runtime in runtimes)
        total_bits = sum(runtime.total_bits for runtime in runtimes)
        fusion_count = sum(runtime.fusion_count for runtime in runtimes)
        layer_summary = {
            "layer_idx": int(spec.layer_idx),
            "all_valid": bool(all_valid),
            "total_bits": int(total_bits),
            "fusion_count": int(fusion_count),
            "active_block_count": len(active_blocks),
            "blocks": block_summaries,
        }

        self._pending_full_vec = candidate_vector
        self._action_history.append(owned_action[:])
        self._decoded_actions.append(application.decoded)
        self._fusion_option_ids.append(dict(application.fusion_option_ids))
        self._layer_summaries.append(copy.deepcopy(layer_summary))
        for block_idx, values in application.boosted_field_values_by_block.items():
            self._boosted_overrides[(int(block_idx), int(spec.layer_idx))] = {
                str(name): int(value) for name, value in values.items()
            }
        self._record_observation_row(
            int(spec.layer_idx), owned_action, all_valid, total_bits,
            fusion_count, len(active_blocks),
        )
        self._step_idx += 1

        info: Dict[str, Any] = {
            "step": int(spec.step_idx),
            "layer_idx": int(spec.layer_idx),
            "layer_summary": copy.deepcopy(layer_summary),
        }
        if self._step_idx < self.horizon:
            return self._build_obs(), 0.0, False, info

        variable_cost = compute_variable_cost(self._decoded_actions)
        # Robust layerwise reward contract consumed by Task 6:
        # P3 = 1 + C + 0.0005 * barriers, with C in [0, 1].  Keep this raw
        # normalized C; do not multiply it by the legacy p3_cost_budget.
        external_cost_score = float(variable_cost.normalized)
        external_cost_rank = float(variable_cost.normalized)
        if not 0.0 <= external_cost_score <= 1.0:
            raise RuntimeError(
                f"layerwise normalized variable cost outside [0, 1]: {external_cost_score}"
            )
        terminal_state, terminal_reward, _terminal_done, terminal_info = self.base.step(
            self._pending_full_vec.copy(),
            external_cost_score=external_cost_score,
            external_cost_rank=external_cost_rank,
            boosted_overrides=(copy.deepcopy(self._boosted_overrides) or None),
        )
        del terminal_state
        self._runtime_terminal_info = terminal_info
        self._done = True
        info.update(self._terminal_handoff(
            variable_cost,
            terminal_reward,
            terminal_info,
            external_cost_score=external_cost_score,
            external_cost_rank=external_cost_rank,
        ))
        return self._build_obs(), float(terminal_reward), True, info

    def _rebuild_schedule(self) -> None:
        self._gelu_degrees = _degree_vector(
            self.base.gelu_degree, num_layers=self.num_layers, default=4,
        )
        self._attn_degrees = _degree_vector(
            self.base.attn_degree, num_layers=self.num_layers, default=6,
        )
        self._schedule = layerwise_schedule(
            self.num_layers,
            self.fusion_map,
            profile=self.profile,
            gelu_degrees=self._gelu_degrees,
        )

    def _record_observation_row(
            self,
            layer_idx: int,
            action: Sequence[int],
            all_valid: bool,
            total_bits: int,
            fusion_count: int,
            active_blocks: int,
            ) -> None:
        dims = self._schedule[layer_idx].slot_dims
        for slot_idx, (value, dim) in enumerate(zip(action, dims)):
            self._action_obs[layer_idx, slot_idx] = float(value) / max(1.0, float(dim - 1))
        self._signal_obs[layer_idx] = np.asarray([
            1.0 if all_valid else 0.0,
            float(total_bits) / max(float(self.cfg.total_bits_scale), 1.0),
            float(fusion_count) / max(float(self.cfg.fusion_count_scale), 1.0),
            float(active_blocks) / 5.0,
        ], dtype=np.float32)

    def _build_obs(self) -> np.ndarray:
        if self._done or self._step_idx >= self.horizon:
            return np.zeros(self.state_dim, dtype=np.float32)
        static = np.asarray([
            float(self.num_layers) / 24.0,
            float(np.mean(self._attn_degrees)) / 8.0,
            float(np.mean(self._gelu_degrees)) / 8.0,
            float(self.horizon) / 12.0,
        ], dtype=np.float32)
        layer_identity = np.zeros(self.horizon, dtype=np.float32)
        layer_identity[self._step_idx] = 1.0
        return np.concatenate((
            static,
            layer_identity,
            self._action_obs.reshape(-1),
            self._signal_obs.reshape(-1),
        )).astype(np.float32)

    @staticmethod
    def _runtime_summary(block_idx: int, runtime: BlockRuntimeResult) -> Dict[str, Any]:
        return {
            "block_idx": int(block_idx),
            "graph_key": str(runtime.graph_key),
            "config_name": str(runtime.config_name),
            "valid": bool(runtime.valid),
            "total_bits": int(runtime.total_bits),
            "fusion_count": int(runtime.fusion_count),
            "invalid_chain": copy.deepcopy(runtime.invalid_chain),
            "bridge_error": runtime.bridge_error,
            "bridge_error_type": runtime.bridge_error_type,
            "optimizer_wall_seconds": float(runtime.optimizer_wall_seconds),
            "boosted_field_values": copy.deepcopy(runtime.boosted_field_values),
            "replan_application": copy.deepcopy(runtime.replan_application),
            "optimizer_cfg_overrides": copy.deepcopy(runtime.optimizer_cfg_overrides),
        }

    def _terminal_handoff(
            self,
            variable_cost: Any,
            terminal_reward: float,
            terminal_info: Any,
            *,
            external_cost_score: float,
            external_cost_rank: float,
            ) -> Dict[str, Any]:
        decoded_actions = [
            {
                "block4_fusion": int(action.block4_fusion),
                "k_by_block": {
                    int(block): int(k_value) for block, k_value in action.k_by_block.items()
                },
            }
            for action in self._decoded_actions
        ]
        k_choices = [
            {
                "layer_idx": layer_idx,
                "block_idx": int(block_idx),
                "k_value": int(k_value),
            }
            for layer_idx, action in enumerate(self._decoded_actions)
            for block_idx, k_value in action.k_by_block.items()
        ]
        boosted_override_rows = [
            {
                "block_idx": int(block_idx),
                "layer_idx": int(layer_idx),
                "field_values": {
                    str(name): int(value) for name, value in field_values.items()
                },
            }
            for (block_idx, layer_idx), field_values in sorted(
                self._boosted_overrides.items(),
                key=lambda item: (int(item[0][1]), int(item[0][0])),
            )
        ]
        return {
            "terminal_info": _snapshot_terminal_info(terminal_info),
            "terminal_reward": float(terminal_reward),
            "external_cost_score": float(external_cost_score),
            "external_cost_rank": float(external_cost_rank),
            "variable_cost": {
                "fusion_saving": float(variable_cost.fusion_saving),
                "truncation_saving": float(variable_cost.truncation_saving),
                "normalized": float(variable_cost.normalized),
            },
            "policy_actions": self.action_history,
            "decoded_actions": decoded_actions,
            "layer_summaries": self.layer_summaries,
            "block4_fusion_choices": [
                int(action.block4_fusion) for action in self._decoded_actions
            ],
            "k_choices": k_choices,
            "fusion_option_ids": [dict(row) for row in self._fusion_option_ids],
            # Stable report order: layer first, then block. Tuple keys remain
            # internal-only for the base.step boosted_overrides contract.
            "boosted_overrides": boosted_override_rows,
            "pending_full_vector": [int(value) for value in self._pending_full_vec],
        }
