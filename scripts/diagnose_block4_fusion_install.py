#!/usr/bin/env python3
"""Diagnose MRPC block4 fusion-option install through the Stage-2 RL path.

The script reuses ``run_fusion_count_action_eval_rlpath`` setup so model,
Stage-1 GELU/Softmax, probe batches, fusion maps, and BLB install semantics match
the online RL path. It evaluates a minimal action: all blocks at option 0 except
layer-0 block4 at option 1, then ablates one boosted block4 field at a time back
to the option-0 baseline value.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from json_utils import write_json_file

from blb_stage2_rl.action_space import (
    K_LEVELS,
    expand_fusion_step_action,
    make_all_max_action_vector,
    splice_fusion_step_into_full_vec,
)

from run_fusion_count_action_eval_rlpath import (
    DEFAULT_STAGE1_GELU,
    DEFAULT_STAGE1_SOFTMAX,
    _base_model,
    _build_evaluator,
    _build_seq_env,
    _json_int_list,
    _jsonable,
    _metric_dict,
    _resolve,
)


BLOCK4_FIELDS = (
    "softmax_out_fresh_sf",
    "v_fresh_sf",
    "softmax_out_mask_sf",
    "v_mask_sf",
    "softmax_v_mask_sf",
    "ln_mean_inv_d_sf",
    "ln_var_inv_d_sf",
    "wo_sf",
    "softmax_v_matmul_rescale_sf",
    "ln_mean_rescale_sf",
    "ln_square_rescale_sf",
    "output_truncation_k",
)


def _sf(point: Any) -> Any:
    if point is None:
        return None
    return int(getattr(point, "scaling_factor"))


def _block4_cfg_snapshot(cfg: Any) -> Dict[str, Any]:
    return {
        "softmax_out_fresh": _sf(getattr(cfg, "softmax_out_fresh", None)),
        "softmax_out_mask_encode": _sf(getattr(cfg, "softmax_out_mask_encode", None)),
        "v_fresh": _sf(getattr(cfg, "v_fresh", None)),
        "v_mask_encode": _sf(getattr(cfg, "v_mask_encode", None)),
        "softmax_v_mask_encode": _sf(getattr(cfg, "softmax_v_mask_encode", None)),
        "wo_encode": _sf(getattr(cfg, "wo_encode", None)),
        "ln_mean_inv_d_encode": _sf(getattr(cfg, "ln_mean_inv_d_encode", None)),
        "ln_var_inv_d_encode": _sf(getattr(cfg, "ln_var_inv_d_encode", None)),
        "softmax_out_mask_rescale": _sf(getattr(cfg, "softmax_out_mask_rescale", None)),
        "v_mask_rescale": _sf(getattr(cfg, "v_mask_rescale", None)),
        "softmax_v_matmul_rescale": _sf(getattr(cfg, "softmax_v_matmul_rescale", None)),
        "softmax_v_mask_rescale": _sf(getattr(cfg, "softmax_v_mask_rescale", None)),
        "wo_result_rescale": _sf(getattr(cfg, "wo_result_rescale", None)),
        "ln_mean_result_rescale": _sf(getattr(cfg, "ln_mean_result_rescale", None)),
        "ln_square_result_rescale": _sf(getattr(cfg, "ln_square_result_rescale", None)),
        "ln_var_result_rescale": _sf(getattr(cfg, "ln_var_result_rescale", None)),
        "output_truncation_k": getattr(cfg, "output_truncation_k", None),
    }


def _one_hot_block4_action(seq_env, *, layer_idx: int, option_id: int, k_index: int) -> np.ndarray:
    vec = make_all_max_action_vector(seq_env.num_layers)
    for spec in seq_env._schedule:
        if int(spec.layer_idx) == int(layer_idx) and int(spec.block_idx) == 4:
            block_vec = expand_fusion_step_action(spec, seq_env._fusion_map, int(option_id), int(k_index))
            splice_fusion_step_into_full_vec(vec, spec, block_vec)
            return vec
    raise RuntimeError(f"no block4 fusion step found for layer {layer_idx}")


def _run_variant(seq_env, *, name: str, action_vec: np.ndarray, boosted_fields: Mapping[str, int] | None, seed: int) -> dict:
    seq_env.base.reset(seed=int(seed))
    seq_env.base.probe_noise_seed = int(seed)
    _state, reward, _done, info = seq_env.base.step(
        action_vec,
        boosted_overrides={(4, 0): dict(boosted_fields)} if boosted_fields else None,
    )
    decoded = info.get("decoded")
    cfg = None
    if decoded is not None:
        cfg = getattr(decoded, "block4_cfgs", {}).get(0)
    metrics = _metric_dict(info.get("metrics"))
    return {
        "name": str(name),
        "reward": float(reward),
        "invalid": bool(info.get("invalid")),
        "forward_ran": bool(info.get("forward_ran")),
        "metrics": metrics,
        "optimizer_cfg_overrides": _jsonable(info.get("optimizer_cfg_overrides") or {}),
        "reward_breakdown": _jsonable(info.get("reward_breakdown") or {}),
        "block4_layer0_installed_cfg": _block4_cfg_snapshot(cfg) if cfg is not None else {},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="mrpc")
    parser.add_argument("--model-type", default="bert-base")
    parser.add_argument("--base-model", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--run-output-dir", default="experiments/server_command_runs/block4_fusion_install_diag_tmp")
    parser.add_argument("--stage1-config-json", default="experiments/server_command_runs/mrpc_stage2_fixed_stage1_rlbest_20260627.json")
    parser.add_argument("--stage1-gelu", default=json.dumps(DEFAULT_STAGE1_GELU))
    parser.add_argument("--stage1-softmax", default=json.dumps(DEFAULT_STAGE1_SOFTMAX))
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--probe-size", type=int, default=408)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--stage2-limit-tolerance", type=float, default=0.001)
    parser.add_argument("--stage2-stability-tolerance", type=float, default=3.5)
    parser.add_argument("--rescale-optimizer-root", default="Rescale_optimizer")
    args = parser.parse_args()

    args.base_model = args.base_model or _base_model(args.model_type, args.dataset)
    output_json = _resolve(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    stage1_gelu = _json_int_list(args.stage1_gelu, default=DEFAULT_STAGE1_GELU, name="--stage1-gelu")
    stage1_softmax = _json_int_list(args.stage1_softmax, default=DEFAULT_STAGE1_SOFTMAX, name="--stage1-softmax")

    ev = _build_evaluator(args, stage1_gelu=stage1_gelu, stage1_softmax=stage1_softmax)
    seq_env, baseline = _build_seq_env(args, ev, stage1_gelu=stage1_gelu, stage1_softmax=stage1_softmax)
    fmap = seq_env._fusion_map
    opt0 = fmap.options("block4")[0]
    opt1 = fmap.options("block4")[1]
    base_fields = {str(k): int(v) for k, v in opt0.slots.items()}
    base_fields["v_mask_sf"] = int(base_fields["softmax_out_mask_sf"])
    base_fields["output_truncation_k"] = int(K_LEVELS[3])
    boosted_fields = {str(k): int(v) for k, v in (opt1.explicit_field_values or {}).items()}
    action0 = make_all_max_action_vector(seq_env.num_layers)
    action1 = _one_hot_block4_action(seq_env, layer_idx=0, option_id=1, k_index=3)

    variants: Dict[str, Dict[str, int] | None] = {
        "all_option0": None,
        "block4_L0_option1_asbuilt": boosted_fields,
    }
    encode_fields = [
        "softmax_out_fresh_sf", "v_fresh_sf", "softmax_out_mask_sf", "v_mask_sf",
        "softmax_v_mask_sf", "ln_mean_inv_d_sf", "ln_var_inv_d_sf", "wo_sf",
    ]
    rescale_fields = [
        "softmax_v_matmul_rescale_sf", "ln_mean_rescale_sf", "ln_square_rescale_sf",
    ]
    for group_name, fields in (("revert_all_fresh_encode", encode_fields), ("revert_all_rescale", rescale_fields)):
        fv = dict(boosted_fields)
        for field in fields:
            if field in base_fields:
                fv[field] = int(base_fields[field])
        variants[group_name] = fv
    for field in BLOCK4_FIELDS:
        if field not in boosted_fields or field not in base_fields:
            continue
        if int(boosted_fields[field]) == int(base_fields[field]):
            continue
        fv = dict(boosted_fields)
        fv[field] = int(base_fields[field])
        variants[f"revert_{field}"] = fv
    for field in BLOCK4_FIELDS:
        if field not in boosted_fields or field not in base_fields:
            continue
        if int(boosted_fields[field]) == int(base_fields[field]):
            continue
        fv = dict(base_fields)
        fv[field] = int(boosted_fields[field])
        variants[f"only_{field}"] = fv

    results = []
    for idx, (name, fields) in enumerate(variants.items()):
        print(f"[run] {name}", flush=True)
        vec = action0 if fields is None else action1
        results.append(_run_variant(
            seq_env,
            name=name,
            action_vec=vec,
            boosted_fields=fields,
            seed=int(args.seed) + idx,
        ))

    payload = {
        "schema": "block4_fusion_install_diag_v1",
        "stage1_gelu": [int(x) for x in stage1_gelu],
        "stage1_softmax": [int(x) for x in stage1_softmax],
        "baseline_metrics": _jsonable(baseline),
        "block4_option0_slots": base_fields,
        "block4_option1_action_indices": [int(x) for x in opt1.action_indices],
        "block4_option1_explicit_field_values": boosted_fields,
        "variants": results,
    }
    write_json_file(output_json, payload)
    print(json.dumps({"output_json": str(output_json)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
