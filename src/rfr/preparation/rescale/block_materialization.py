"""Decode, replan, and materialize one Stage-2 block configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from rfr.search.common.action_space import (
    _BLOCK_SPECS,
    _block_default_N,
    _degree_for_layer,
    action_vector_to_cfgs,
    build_block_cfg_from_field_values,
    make_all_max_action_vector,
)
from blb_stage2_rl.env import BLBStage2Env
from .optimizer_cost import materialize_decoded_action


_BRIDGE_OPERATIONAL_ERRORS = (RuntimeError, OSError)


@dataclass
class BlockRuntimeResult:
    block_cfg: Optional[Any]
    optimizer_output: Optional[Any]
    valid: bool
    total_bits: int
    fusion_count: int
    invalid_chain: Optional[Any]
    bridge_error: Optional[str]
    bridge_error_type: Optional[str]
    config_name: str
    graph_key: str
    optimizer_wall_seconds: float
    boosted_field_values: Optional[dict[str, int]] = None
    replan_application: dict[str, Any] = field(default_factory=dict)
    optimizer_cfg_overrides: list[Any] = field(default_factory=list)
    materialization_failure_reason: Optional[str] = None
    final_config_fingerprint: str = ""


def evaluate_block_from_full_vector(
    *,
    base_env: BLBStage2Env,
    full_vec: Sequence[int],
    layer_idx: int,
    block_idx: int,
    graph_key: str,
    boosted_field_values: Optional[Mapping[str, int]] = None,
) -> BlockRuntimeResult:
    """Run the shared decode, optimizer, and config-binding path."""
    layer = int(layer_idx)
    block = int(block_idx)
    vector = np.asarray(full_vec, dtype=int).reshape(-1).copy()
    expected_size = np.asarray(
        make_all_max_action_vector(int(base_env.num_layers)), dtype=int
    ).reshape(-1).size
    if vector.size != expected_size:
        raise ValueError(f"full_vec has {vector.size} slots, expected {expected_size}")
    if not 0 <= layer < int(base_env.num_layers):
        raise ValueError(
            f"layer_idx {layer} outside [0, {int(base_env.num_layers)})"
        )
    if block not in _BLOCK_SPECS:
        raise ValueError(f"unsupported block_idx {block}")

    decoded = action_vector_to_cfgs(
        vector,
        base_env.max_sfs,
        num_layers=int(base_env.num_layers),
        gelu_degree=base_env.gelu_degree,
        attn_degree=base_env.attn_degree,
        only=(layer, block),
    )
    block_cfg = decoded.cfgs_dict()[f"block{block}"][layer]

    boosted_copy = None
    if boosted_field_values is not None:
        boosted_copy = {
            str(name): int(value) for name, value in boosted_field_values.items()
        }
        gelu_degree = _degree_for_layer(
            base_env.gelu_degree,
            layer,
            int(base_env.num_layers),
            default=4,
            name="gelu_degree",
        )
        attention_degree = _degree_for_layer(
            base_env.attn_degree,
            layer,
            int(base_env.num_layers),
            default=4,
            name="attn_degree",
        )
        block_cfg = build_block_cfg_from_field_values(
            block,
            layer,
            boosted_copy,
            N=int(
                _block_default_N(
                    block,
                    gelu_degree=gelu_degree,
                    attn_degree=attention_degree,
                )
            ),
            gelu_degree=gelu_degree,
            attn_degree=attention_degree,
        )
        getattr(decoded, f"block{block}_cfgs")[layer] = block_cfg

    config_name = f"{graph_key}_L{layer}"
    optimizer_start = time.perf_counter()
    try:
        evaluate_optimizer = getattr(
            base_env.rescale_bridge,
            "evaluate_readonly",
            base_env.rescale_bridge.evaluate,
        )
        output = evaluate_optimizer(
            config_name=config_name,
            block_name=f"block{block}",
            cfg=block_cfg,
        )
    except _BRIDGE_OPERATIONAL_ERRORS as exc:
        return BlockRuntimeResult(
            block_cfg=None,
            optimizer_output=None,
            valid=False,
            total_bits=0,
            fusion_count=0,
            invalid_chain={"reason": f"bridge_error: {exc}"},
            bridge_error=str(exc),
            bridge_error_type=type(exc).__name__,
            config_name=config_name,
            graph_key=str(graph_key),
            optimizer_wall_seconds=float(time.perf_counter() - optimizer_start),
            boosted_field_values=boosted_copy,
        )
    optimizer_wall_seconds = float(time.perf_counter() - optimizer_start)

    invoker_baselines = getattr(
        base_env.rescale_bridge.invoker, "baselines", {}
    ) or {}

    def rotation_provider(runtime_block_idx: int, profile: str) -> Mapping[str, str]:
        return (base_env.env_cfg.rotation_name_map or {}).get(
            (int(runtime_block_idx), str(profile)), {}
        )

    materialized = materialize_decoded_action(
        action_indices=vector,
        decoded=decoded,
        cfgs_dict={f"block{block}": {layer: block_cfg}},
        outputs={config_name: output},
        signals=SimpleNamespace(any_invalid=not bool(output.valid)),
        profile=str(base_env.env_cfg.profile),
        invoker_baselines=invoker_baselines,
        rotation_name_map_provider=rotation_provider,
        expected_config_names=[config_name],
        truncation_backend=getattr(
            base_env.env_cfg, "truncation_backend", "binary"
        ),
        truncation_ring_bits=int(
            getattr(base_env.env_cfg, "truncation_ring_bits", 43)
        ),
        truncation_source_fractional_bits=int(
            getattr(base_env.env_cfg, "truncation_source_fractional_bits", 24)
        ),
    )
    block_cfg = materialized.cfgs_dict[f"block{block}"][layer]
    replan_application = materialized.replan_application
    optimizer_cfg_overrides = []
    per_config = replan_application.get("optimizer_cfg_overrides", {})
    if per_config:
        optimizer_cfg_overrides = list(per_config.get(config_name, []))
    invalid_chain = output.invalid_chain
    if not materialized.model_ready and not materialized.optimizer_invalid:
        invalid_chain = {
            "reason": str(materialized.failure_reason),
            "replan_application": replan_application,
        }

    return BlockRuntimeResult(
        block_cfg=block_cfg,
        optimizer_output=output,
        valid=bool(materialized.model_ready),
        total_bits=int(output.total_bits),
        fusion_count=int(output.fusion_count),
        invalid_chain=invalid_chain,
        bridge_error=None,
        bridge_error_type=None,
        config_name=config_name,
        graph_key=str(graph_key),
        optimizer_wall_seconds=optimizer_wall_seconds,
        boosted_field_values=boosted_copy,
        replan_application=replan_application,
        optimizer_cfg_overrides=optimizer_cfg_overrides,
        materialization_failure_reason=materialized.failure_reason,
        final_config_fingerprint=materialized.final_config_fingerprint,
    )
