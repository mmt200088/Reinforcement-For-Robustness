"""Canonical BLB action cost evaluation helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from json_utils import bounded_stable_json_hash

from rescale_optimizer_bridge import (
    _strip_layer_suffix,
    aggregate_optimizer_signals,
    apply_optimizer_output_to_cfg,
    default_rotation_name_map,
    sync_block2_aux_fresh_binding,
    sync_block2_qk_binding,
    sync_block4_v_mask_binding,
    sync_block5_aux_fresh_binding,
)

if __package__:  # package/runtime context
    from .action_space import (
        ActionDecodeResult,
        MaxSFsTable,
        action_vector_to_cfgs,
        build_optimizer_requests,
        parse_config_name,
    )
else:  # torch-free test lane (blb_stage2_rl on sys.path)
    from action_space import (
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


@dataclass
class MaterializedStage2Action:
    """One fully resolved Stage-2 action that is safe to install in a model."""

    action_indices: Sequence[int]
    decoded: Any
    cfgs_dict: Dict[str, Mapping[int, object]]
    requests: Dict[str, Tuple[str, object]]
    outputs: Mapping[str, Any]
    signals: Any
    replan_application: Dict[str, Any]
    optimizer_invalid: bool
    model_ready: bool
    failure_reason: Optional[str]
    final_config_fingerprint: str
    optimizer_eval_mode: str = "evaluate_blocks_cfg_path"


_TRUNCATION_BACKENDS = frozenset({"binary", "decimal", "stochastic_ring"})


def configure_truncation_backend(
        cfgs_dict: Mapping[str, Mapping[int, object]],
        *,
        backend: str = "binary",
        ring_bits: int = 43,
        source_fractional_bits: int = 24,
        ) -> None:
    """Attach one explicit truncation backend contract to every block cfg."""
    normalized = str(backend).strip().lower()
    if normalized not in _TRUNCATION_BACKENDS:
        raise ValueError(f"unsupported truncation backend: {backend!r}")
    if normalized == "stochastic_ring":
        if not 2 <= int(ring_bits) <= 62:
            raise ValueError("truncation ring_bits must be in [2, 62]")
        if not 0 <= int(source_fractional_bits) < int(ring_bits):
            raise ValueError(
                "truncation source_fractional_bits must be non-negative and "
                "smaller than ring_bits"
            )

    cfgs = [
        cfg
        for layer_cfgs in cfgs_dict.values()
        if isinstance(layer_cfgs, Mapping)
        for cfg in layer_cfgs.values()
    ]
    if normalized == "stochastic_ring":
        invalid_targets = sorted({
            int(target_k)
            for cfg in cfgs
            for target_k in (getattr(cfg, "output_truncation_k", None),)
            if target_k is not None
            and not 0 <= int(target_k) <= int(source_fractional_bits)
        })
        if invalid_targets:
            raise ValueError(
                "stochastic_ring target truncation K must be in "
                f"[0, {int(source_fractional_bits)}], got {invalid_targets}"
            )

    for cfg in cfgs:
        setattr(cfg, "output_truncation_mode", normalized)
        setattr(cfg, "output_truncation_ring_bits", int(ring_bits))
        setattr(
            cfg,
            "output_truncation_source_fractional_bits",
            int(source_fractional_bits),
        )


def materialized_config_fingerprint(
        cfgs_dict: Mapping[str, Mapping[int, object]],
        ) -> str:
    """Hash the exact post-replan configuration that will be installed."""
    canonical = {
        "schema": "blb_stage2_materialized_config_v1",
        "blocks": {
            str(block_name): {
                str(int(layer_idx)): cfg
                for layer_idx, cfg in sorted(
                    layer_cfgs.items(), key=lambda item: int(item[0]),
                )
            }
            for block_name, layer_cfgs in sorted(
                cfgs_dict.items(), key=lambda item: str(item[0]),
            )
            if isinstance(layer_cfgs, Mapping)
        },
    }
    return bounded_stable_json_hash(canonical)


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
        rotation_name_map_provider: Optional[Callable[[int, str], Mapping[str, Any]]] = None,
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

    def rotation_map(block_idx: int) -> Mapping[str, Any]:
        resolved = dict(default_rotation_name_map(int(block_idx)))
        if rotation_name_map_provider is None:
            return resolved
        raw = rotation_name_map_provider(int(block_idx), str(profile))
        if isinstance(raw, Mapping):
            resolved.update(raw)
        return resolved

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


def materialize_decoded_action(
        *,
        action_indices: Sequence[int],
        decoded: Any,
        cfgs_dict: Mapping[str, Mapping[int, object]],
        outputs: Mapping[str, Any],
        signals: Any,
        profile: str,
        invoker_baselines: Optional[Mapping[str, Any]] = None,
        rotation_name_map_provider: Optional[Callable[[int, str], Mapping[str, Any]]] = None,
        expected_config_names: Optional[Sequence[str]] = None,
        truncation_backend: str = "binary",
        truncation_ring_bits: int = 43,
        truncation_source_fractional_bits: int = 24,
        optimizer_eval_mode: str = "evaluate_blocks_cfg_path",
        ) -> MaterializedStage2Action:
    """Resolve optimizer outputs into the only cfg set allowed to reach a model.

    Optimizer-invalid chains are normal invalid actions.  A valid optimizer
    response that is missing, extra, or cannot be written back completely is a
    plumbing failure and therefore fails closed before model inference.
    """
    mutable_cfgs = dict(cfgs_dict)
    configure_truncation_backend(
        mutable_cfgs,
        backend=truncation_backend,
        ring_bits=int(truncation_ring_bits),
        source_fractional_bits=int(truncation_source_fractional_bits),
    )
    requests = build_optimizer_requests(str(profile), mutable_cfgs)
    expected_names = {
        str(name)
        for name in (
            expected_config_names
            if expected_config_names is not None
            else requests.keys()
        )
    }
    output_names = {str(name) for name in (outputs or {}).keys()}
    missing_outputs = sorted(expected_names - output_names)
    unexpected_outputs = sorted(output_names - expected_names)

    optimizer_invalid = bool(getattr(signals, "any_invalid", False)) or any(
        not bool(getattr(out, "valid", False))
        for out in (outputs or {}).values()
    )
    replan_application = apply_optimizer_outputs_to_cfgs(
        profile=str(profile),
        cfgs_dict=mutable_cfgs,
        opt_outputs=outputs,
        invoker_baselines=invoker_baselines,
        rotation_name_map_provider=rotation_name_map_provider,
        skip_on_any_invalid=True,
    )
    replan_application["expected_output_names"] = sorted(expected_names)
    replan_application["actual_output_names"] = sorted(output_names)
    replan_application["missing_optimizer_outputs"] = missing_outputs
    replan_application["unexpected_optimizer_outputs"] = unexpected_outputs
    replan_application["optimizer_output_set_matches"] = bool(
        not missing_outputs and not unexpected_outputs
    )
    missing_baseline_skeletons = sorted(
        str(name)
        for name in expected_names
        if not bool(
            (
                (invoker_baselines or {}).get(_strip_layer_suffix(str(name))[0])
                or ([], [], [])
            )[0]
        )
    )
    replan_application["missing_baseline_skeletons"] = missing_baseline_skeletons
    replan_application["all_baseline_skeletons_available"] = bool(
        not missing_baseline_skeletons
    )
    if missing_baseline_skeletons:
        replan_application["model_uses_replan_config"] = False

    failure_reason: Optional[str] = None
    if optimizer_invalid:
        failure_reason = "optimizer_invalid_chain"
    elif missing_outputs or unexpected_outputs:
        failure_reason = "optimizer_output_set_mismatch"
    elif not bool(replan_application.get("model_uses_replan_config", False)):
        failure_reason = "replan_config_not_fully_applied"

    model_ready = failure_reason is None
    return MaterializedStage2Action(
        action_indices=[int(value) for value in action_indices],
        decoded=decoded,
        cfgs_dict=mutable_cfgs,
        requests=requests,
        outputs=outputs,
        signals=signals,
        replan_application=replan_application,
        optimizer_invalid=bool(optimizer_invalid),
        model_ready=bool(model_ready),
        failure_reason=failure_reason,
        final_config_fingerprint=(
            materialized_config_fingerprint(mutable_cfgs) if model_ready else ""
        ),
        optimizer_eval_mode=str(optimizer_eval_mode),
    )


def materialize_action_for_model(
        action_vec: Sequence[int],
        *,
        profile: str,
        num_layers: int,
        max_sfs: MaxSFsTable,
        rescale_bridge: Any,
        gelu_degree: Any = 4,
        attn_degree: Any = 4,
        boosted_overrides: "Mapping[Tuple[int, int], Mapping[str, int]] | None" = None,
        invoker_baselines: Optional[Mapping[str, Any]] = None,
        rotation_name_map_provider: Optional[Callable[[int, str], Mapping[str, Any]]] = None,
        truncation_backend: str = "binary",
        truncation_ring_bits: int = 43,
        truncation_source_fractional_bits: int = 24,
        borrow_cached_optimizer_payloads: bool = False,
        ) -> MaterializedStage2Action:
    """Decode, replan, write back, and fingerprint one executable action."""
    evaluated = evaluate_action_for_cost(
        action_vec,
        profile=str(profile),
        num_layers=int(num_layers),
        max_sfs=max_sfs,
        rescale_bridge=rescale_bridge,
        gelu_degree=gelu_degree,
        attn_degree=attn_degree,
        boosted_overrides=boosted_overrides,
        borrow_cached_optimizer_payloads=borrow_cached_optimizer_payloads,
    )
    return materialize_decoded_action(
        action_indices=evaluated.action_indices,
        decoded=evaluated.decoded,
        cfgs_dict=evaluated.cfgs_dict,
        outputs=evaluated.outputs,
        signals=evaluated.signals,
        profile=str(profile),
        invoker_baselines=invoker_baselines,
        rotation_name_map_provider=rotation_name_map_provider,
        expected_config_names=list(evaluated.requests),
        truncation_backend=truncation_backend,
        truncation_ring_bits=int(truncation_ring_bits),
        truncation_source_fractional_bits=int(truncation_source_fractional_bits),
        optimizer_eval_mode=evaluated.optimizer_eval_mode,
    )


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
        borrow_cached_optimizer_payloads: bool = False,
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
        if __package__:  # package/runtime context
            from .action_space import (
                _block_default_N,
                _degree_for_layer,
                build_block_cfg_from_field_values,
            )
        else:  # torch-free test lane (blb_stage2_rl on sys.path)
            from action_space import (
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

    readonly_batch_eval = getattr(
        rescale_bridge, "evaluate_blocks_readonly", None,
    )
    if borrow_cached_optimizer_payloads and callable(readonly_batch_eval):
        outputs = readonly_batch_eval(requests)
    else:
        outputs = rescale_bridge.evaluate_blocks(requests)

    return ActionCostEvaluation(
        action_indices=[int(x) for x in action_arr.tolist()],
        decoded=decoded,
        cfgs_dict=cfgs_dict,
        requests=requests,
        outputs=outputs,
        signals=aggregate_optimizer_signals(outputs),
    )
