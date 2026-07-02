"""Canonical BLB action cost evaluation helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from rescale_optimizer_bridge import (
    _strip_layer_suffix,
    aggregate_optimizer_signals,
    apply_optimizer_output_to_cfg,
    sync_block2_aux_fresh_binding,
    sync_block2_qk_binding,
    sync_block4_v_mask_binding,
    sync_block5_aux_fresh_binding,
)

try:  # torch-free test lane (blb_stage2_rl on sys.path)
    from action_space import (
        ActionDecodeResult,
        MaxSFsTable,
        action_vector_to_cfgs,
        build_optimizer_requests,
        parse_config_name,
    )
except ImportError:  # package context
    from .action_space import (
        ActionDecodeResult,
        MaxSFsTable,
        action_vector_to_cfgs,
        build_optimizer_requests,
        parse_config_name,
    )


@dataclass
class ActionCostEvaluation:
    action_indices: Sequence[int]
    decoded: ActionDecodeResult
    cfgs_dict: Dict[str, Mapping[int, object]]
    requests: Dict[str, Tuple[str, object]]
    outputs: Mapping[str, Any]
    signals: Any
    optimizer_eval_mode: str = "evaluate_blocks_cfg_path"


def _override_entry_dict(entry: Any) -> Dict[str, Any]:
    return {
        "cfg_attr": str(getattr(entry, "cfg_attr", "")),
        "graph_node": getattr(entry, "graph_node", None),
        "source": str(getattr(entry, "source", "")),
        "old_value": getattr(entry, "old_value", None),
        "new_value": getattr(entry, "new_value", None),
    }


def _override_entry_tuple(entry: Any) -> Tuple[str, str, Any, Any]:
    return (
        str(getattr(entry, "cfg_attr", "")),
        str(getattr(entry, "source", "")),
        getattr(entry, "old_value", None),
        getattr(entry, "new_value", None),
    )


def apply_optimizer_outputs_to_cfgs(
        *,
        profile: str,
        cfgs_dict: Mapping[str, Mapping[int, object]],
        opt_outputs: Mapping[str, Any],
        invoker_baselines: Optional[Mapping[str, Any]] = None,
        rotation_name_map_provider: Optional[Callable[[int, str], Mapping[str, str]]] = None,
        skip_on_any_invalid: bool = True,
        ) -> Dict[str, Any]:
    """Apply Rescale_optimizer/replan outputs to decoded cfg objects in place.

    This is the canonical write-back seam for every executable Stage-2 path:
    online RL terminal eval, sequential per-block eval, Paean final eval, and
    fixed-action experiments.  The action decode proposes cfgs; the optimizer's
    ``new_compact_config`` decides which rescale points survive and what SFs the
    model must actually install.  This helper centralizes that write-back plus
    the block-specific binding mirrors that must remain identical everywhere.
    """
    invoker_baselines = invoker_baselines or {}

    def rotation_map(block_idx: int) -> Mapping[str, str]:
        if rotation_name_map_provider is None:
            return {}
        raw = rotation_name_map_provider(int(block_idx), str(profile))
        return raw if isinstance(raw, Mapping) else {}

    per_config: Dict[str, Dict[str, Any]] = {}
    legacy_overrides: Dict[str, List[Tuple[str, str, Any, Any]]] = {}
    invalid_count = 0
    missing_compact_count = 0
    missing_cfg_count = 0
    apply_error_count = 0
    applied_count = 0
    override_total = 0

    outputs = dict(opt_outputs or {})
    batch_has_invalid = any(
        not bool(getattr(out, "valid", False))
        for out in outputs.values()
    )
    if skip_on_any_invalid and batch_has_invalid:
        for config_name, out in outputs.items():
            valid = bool(getattr(out, "valid", False))
            if not valid:
                invalid_count += 1
            per_config[str(config_name)] = {
                "valid": valid,
                "applied": False,
                "override_count": 0,
                "overrides": [],
                "skipped_reason": (
                    "optimizer_invalid" if not valid else "optimizer_invalid_batch"
                ),
            }
        return {
            "applied_before_forward": False,
            "model_uses_replan_config": False,
            "expected_config_count": int(len(outputs)),
            "applied_config_count": 0,
            "invalid_config_count": int(invalid_count),
            "missing_compact_config_count": 0,
            "missing_decoded_cfg_count": 0,
            "apply_error_count": 0,
            "override_total": 0,
            "per_config": per_config,
            "optimizer_cfg_overrides": legacy_overrides,
        }

    for config_name, out in outputs.items():
        name = str(config_name)
        raw = getattr(out, "raw", {}) or {}
        valid = bool(getattr(out, "valid", False))
        record: Dict[str, Any] = {
            "valid": valid,
            "applied": False,
            "override_count": 0,
            "overrides": [],
        }
        if not valid:
            invalid_count += 1
            record["skipped_reason"] = "optimizer_invalid"
            per_config[name] = record
            continue
        if not isinstance(raw, Mapping) or not isinstance(raw.get("new_compact_config"), Mapping):
            missing_compact_count += 1
            record["skipped_reason"] = "missing_new_compact_config"
            per_config[name] = record
            continue

        try:
            block_idx, _cfg_profile, layer_idx = parse_config_name(name)
        except Exception as exc:
            apply_error_count += 1
            record["skipped_reason"] = "parse_config_name_failed"
            record["error"] = str(exc)
            per_config[name] = record
            continue
        if int(layer_idx) < 0:
            missing_cfg_count += 1
            record["skipped_reason"] = "missing_layer_suffix"
            per_config[name] = record
            continue

        block_key = f"block{int(block_idx)}"
        block_cfgs = cfgs_dict.get(block_key, {}) if isinstance(cfgs_dict, Mapping) else {}
        target_cfg = block_cfgs.get(int(layer_idx)) if isinstance(block_cfgs, Mapping) else None
        if target_cfg is None:
            missing_cfg_count += 1
            record["skipped_reason"] = "decoded_cfg_missing"
            record["block"] = block_key
            record["layer"] = int(layer_idx)
            per_config[name] = record
            continue

        graph_key, _ = _strip_layer_suffix(name)
        baseline_entry = invoker_baselines.get(graph_key)
        baseline_skeleton = list(baseline_entry[0]) if baseline_entry else []
        try:
            overrides = apply_optimizer_output_to_cfg(
                target_cfg,
                output_raw=raw,
                block_idx=int(block_idx),
                graph_key=graph_key,
                baseline_skeleton=baseline_skeleton,
                rotation_name_map=rotation_map(int(block_idx)),
            )
            if int(block_idx) == 2:
                overrides = (
                    list(overrides)
                    + sync_block2_qk_binding(target_cfg)
                    + sync_block2_aux_fresh_binding(target_cfg)
                )
            elif int(block_idx) == 4:
                overrides = list(overrides) + sync_block4_v_mask_binding(target_cfg)
            elif int(block_idx) == 5:
                overrides = list(overrides) + sync_block5_aux_fresh_binding(target_cfg)
        except Exception as exc:
            apply_error_count += 1
            record["skipped_reason"] = "apply_optimizer_output_to_cfg_failed"
            record["error"] = str(exc)
            per_config[name] = record
            continue

        override_rows = [_override_entry_dict(entry) for entry in overrides]
        override_tuples = [_override_entry_tuple(entry) for entry in overrides]
        applied_count += 1
        override_total += len(override_rows)
        if override_tuples:
            legacy_overrides[name] = override_tuples
        record.update(
            {
                "applied": True,
                "block": block_key,
                "layer": int(layer_idx),
                "graph_key": graph_key,
                "baseline_skeleton_available": bool(baseline_skeleton),
                "rotation_name_map_available": bool(rotation_map(int(block_idx))),
                "override_count": int(len(override_rows)),
                "overrides": override_rows,
            }
        )
        per_config[name] = record

    expected = len(outputs)
    fully_applied = (
        expected == applied_count
        and invalid_count == 0
        and missing_compact_count == 0
        and missing_cfg_count == 0
        and apply_error_count == 0
    )
    return {
        "applied_before_forward": True,
        "model_uses_replan_config": bool(fully_applied),
        "expected_config_count": int(expected),
        "applied_config_count": int(applied_count),
        "invalid_config_count": int(invalid_count),
        "missing_compact_config_count": int(missing_compact_count),
        "missing_decoded_cfg_count": int(missing_cfg_count),
        "apply_error_count": int(apply_error_count),
        "override_total": int(override_total),
        "per_config": per_config,
        "optimizer_cfg_overrides": legacy_overrides,
    }


def evaluate_action_for_cost(
        action_vec: Sequence[int],
        *,
        profile: str,
        num_layers: int,
        max_sfs: MaxSFsTable,
        rescale_bridge: Any,
        gelu_degree: Any = 4,
        attn_degree: Any = 4,
        boosted_overrides: "Mapping[Tuple[int, int], Mapping[str, int]] | None" = None,
        ) -> ActionCostEvaluation:
    """Evaluate every action through the same cfg-derived optimizer path.

    This is the canonical convention for candidate ranking, reward baseline,
    F0 scans, and RL env cost comparison.  Even the all-max baseline uses
    ``action_vector_to_cfgs -> build_optimizer_requests -> evaluate_blocks``.
    Optimizer-native empty-payload baselines remain diagnostic-only because
    they may use a different Rescale_optimizer convention.

    ``boosted_overrides`` (加大精度): ``{(block_idx, layer_idx): {field: sf}}`` of
    explicit boosted SFs (above-baseline, no action index). After the grid decode,
    the listed (block, layer) cfgs are rebuilt SF-direct so the ENTIRE downstream
    path — replan cost signals, optimizer override, AND the model noise install —
    uses the boosted action group rather than the index-decoded pre-boost one.
    """
    action_arr = np.asarray(action_vec, dtype=int).reshape(-1)
    decoded = action_vector_to_cfgs(
        action_arr,
        max_sfs,
        num_layers=int(num_layers),
        gelu_degree=gelu_degree,
        attn_degree=attn_degree,
    )
    if boosted_overrides:
        try:  # torch-free test lane (blb_stage2_rl on sys.path)
            from action_space import (
                _block_default_N,
                _degree_for_layer,
                build_block_cfg_from_field_values,
            )
        except ImportError:  # package context
            from .action_space import (
                _block_default_N,
                _degree_for_layer,
                build_block_cfg_from_field_values,
            )
        for (block_idx, layer_idx), field_values in boosted_overrides.items():
            bi, li = int(block_idx), int(layer_idx)
            deg_g = _degree_for_layer(gelu_degree, li, int(num_layers), default=4, name="gelu_degree")
            deg_a = _degree_for_layer(attn_degree, li, int(num_layers), default=4, name="attn_degree")
            cfg = build_block_cfg_from_field_values(
                bi, li, dict(field_values),
                N=int(_block_default_N(bi, gelu_degree=deg_g, attn_degree=deg_a)),
                gelu_degree=deg_g, attn_degree=deg_a,
            )
            getattr(decoded, f"block{bi}_cfgs")[li] = cfg
    cfgs_dict = decoded.cfgs_dict()
    requests = build_optimizer_requests(profile, cfgs_dict)

    outputs = rescale_bridge.evaluate_blocks(requests)

    return ActionCostEvaluation(
        action_indices=[int(x) for x in action_arr.tolist()],
        decoded=decoded,
        cfgs_dict=cfgs_dict,
        requests=requests,
        outputs=outputs,
        signals=aggregate_optimizer_signals(outputs),
    )
