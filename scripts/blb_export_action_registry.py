"""Export BLB Stage-2 action registry artifacts requested by the playbook."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cli_parse_utils import parse_broadcast_int_vector  # noqa: E402


SEMANTIC_TYPE_BY_KIND = {
    "F": "fresh",
    "W": "weight_encode",
    "M": "mask_encode",
    "S": "scalar_encode",
    "R": "rescale",
    "K": "truncation",
}

_ACTION_SPACE_DEPS: Dict[str, Any] | None = None


def _load_action_space_deps() -> Dict[str, Any]:
    """Import action-space helpers only when registry generation needs them."""
    global _ACTION_SPACE_DEPS
    if _ACTION_SPACE_DEPS is None:
        from blb_stage2_rl.action_space import (
            K_LEVELS,
            describe_action_vector,
            load_max_sfs,
            make_all_max_action_vector,
            per_layer_field_offsets,
        )

        _ACTION_SPACE_DEPS = {
            "K_LEVELS": K_LEVELS,
            "describe_action_vector": describe_action_vector,
            "load_max_sfs": load_max_sfs,
            "make_all_max_action_vector": make_all_max_action_vector,
            "per_layer_field_offsets": per_layer_field_offsets,
        }
    return _ACTION_SPACE_DEPS


FIELD_SEMANTICS = {
    (1, "gelu_out_sf"): "GELU 输出结果上的 fresh 噪声 scaling factor。",
    (1, "wffn2_sf"): "Wffn2 权重 encode 噪声 scaling factor。",
    (1, "mean_inv_d_sf"): "LayerNorm mean 中 1/D 标量 encode 噪声 scaling factor。",
    (1, "var_inv_d_sf"): "LayerNorm variance 中 1/D 标量 encode 噪声 scaling factor。",
    (1, "wffn2_rescale_sf"): "GELU_out * Wffn2 乘法结果后的 rescale scaling factor。",
    (1, "mean_rescale_sf"): "mean 计算相关乘法结果后的 rescale scaling factor。",
    (1, "square_rescale_sf"): "(X - mean)^2 平方操作结果后的 rescale scaling factor。",
    (1, "var_rescale_sf"): "variance 计算结果后的 rescale scaling factor。",
    (1, "output_truncation_k"): "Block 1 末尾 CKKS/MPC 转换模拟的 truncation bit。",
    (2, "inv_std_fresh_sf"): "LayerNorm 中 1/std 结果上的 fresh 噪声 scaling factor。",
    (2, "x_centered_fresh_sf"): "LayerNorm 中 X - mean 结果上的 fresh 噪声 scaling factor。",
    (2, "gamma_sf"): "LayerNorm gamma 参数 encode scaling factor。",
    (2, "wq_sf"): "Query 投影权重 Wq encode scaling factor。",
    (2, "wk_sf"): "Key 投影权重 Wk encode scaling factor。",
    (2, "wv_sf"): "Value 投影权重 Wv encode scaling factor。",
    (2, "kt_mask1_sf"): "K/KT BSGS 第一个 mask 矩阵 encode scaling factor。",
    (2, "kt_mask2_sf"): "K/KT BSGS 第二个 mask 矩阵 encode scaling factor。",
    (2, "q_mask1_sf"): "Q BSGS 第一个 mask 矩阵 encode scaling factor。",
    (2, "q_mask2_sf"): "Q BSGS 第二个 mask 矩阵 encode scaling factor。",
    (2, "qkt_merge_mask_sf"): "QK^T 后 merge mask 矩阵 encode scaling factor。",
    (2, "normalize_rescale_sf"): "(X-mean) * (1/std) normalize 结果后的 rescale scaling factor。",
    (2, "gamma_rescale_sf"): "normalize 结果乘 gamma 后的 rescale scaling factor。",
    (2, "wk_rescale_sf"): "X * Wk 得到 K 后的 rescale scaling factor。",
    (2, "wq_rescale_sf"): "X * Wq 得到 Q 后的 rescale scaling factor。",
    (2, "wv_rescale_sf"): "X * Wv 得到 V 后的 rescale scaling factor。",
    (2, "kt_mask1_rescale_sf"): "K/KT 乘第一个 mask 后的 rescale scaling factor。",
    (2, "kt_mask2_rescale_sf"): "K/KT 乘第二个 mask 后的 rescale scaling factor。",
    (2, "q_mask1_rescale_sf"): "Q 乘第一个 mask 后的 rescale scaling factor。",
    (2, "q_mask2_rescale_sf"): "Q 乘第二个 mask 后的 rescale scaling factor。",
    (2, "qkt_matmul_rescale_sf"): "Q 和 K/KT 相乘得到 QK^T 后的 rescale scaling factor。",
    (2, "qkt_merge_mask_rescale_sf"): "QK^T 乘 merge mask 后的 rescale scaling factor。",
    (2, "output_truncation_k"): "Block 2 末尾 CKKS/MPC 转换模拟的 truncation bit。",
    (3, "x_fresh_sf"): "Softmax 输入 x 的 fresh 噪声 scaling factor。",
    (3, "inv_2n_sf"): "Softmax exp 近似中 1/2^n 标量 encode scaling factor。",
    (3, "x_inv_2n_rescale_sf"): "x * 1/2^n 后的 rescale scaling factor。",
    (3, "square_rescale_sf_0"): "Softmax 指数近似第 1 次平方后的 rescale scaling factor。",
    (3, "square_rescale_sf_1"): "Softmax 指数近似第 2 次平方后的 rescale scaling factor，degree 条件启用。",
    (3, "square_rescale_sf_2"): "Softmax 指数近似第 3 次平方后的 rescale scaling factor，degree 条件启用。",
    (3, "square_rescale_sf_3"): "Softmax 指数近似第 4 次平方后的 rescale scaling factor，degree 条件启用。",
    (3, "output_truncation_k"): "Block 3 末尾 CKKS/MPC 转换模拟的 truncation bit。",
    (4, "softmax_out_fresh_sf"): "Softmax 输出结果 fresh 噪声 scaling factor。",
    (4, "v_fresh_sf"): "V 矩阵 fresh 噪声 scaling factor。",
    (4, "softmax_out_mask_sf"): "Softmax 输出乘全 1 mask 的 mask encode scaling factor。",
    (4, "v_mask_sf"): "V 乘全 1 mask 的 mask encode scaling factor。",
    (4, "softmax_v_mask_sf"): "Softmax*V 之后再乘 mask 的 mask encode scaling factor。",
    (4, "ln_mean_inv_d_sf"): "attention LayerNorm mean 中 1/D 标量 encode scaling factor。",
    (4, "ln_var_inv_d_sf"): "attention LayerNorm variance 中 1/D 标量 encode scaling factor。",
    (4, "wo_sf"): "Wo 权重 encode scaling factor。",
    (4, "softmax_out_mask_rescale_sf"): "Softmax output mask 乘法后的 rescale scaling factor。",
    (4, "v_mask_rescale_sf"): "V mask 乘法后的 rescale scaling factor。",
    (4, "softmax_v_matmul_rescale_sf"): "Softmax * V matmul 后的 rescale scaling factor。",
    (4, "softmax_v_mask_rescale_sf"): "Softmax*V mask 乘法后的 rescale scaling factor。",
    (4, "wo_rescale_sf"): "Wo 乘法后的 rescale scaling factor。",
    (4, "ln_mean_rescale_sf"): "LayerNorm mean 计算后的 rescale scaling factor。",
    (4, "ln_square_rescale_sf"): "LayerNorm 中 (X - mean)^2 后的 rescale scaling factor。",
    (4, "ln_var_rescale_sf"): "LayerNorm variance 计算后的 rescale scaling factor。",
    (4, "output_truncation_k"): "Block 4 末尾 CKKS/MPC 转换模拟的 truncation bit。",
    (5, "inv_std_fresh_sf"): "LayerNorm tail 中 1/std fresh 噪声 scaling factor。",
    (5, "x_centered_fresh_sf"): "LayerNorm tail 中 X_centered fresh 噪声 scaling factor。",
    (5, "gamma_sf"): "LayerNorm gamma / 逐元素参数 encode scaling factor。",
    (5, "wffn1_sf"): "FFN 第一层 Wffn1 权重 encode scaling factor。",
    (5, "gelu_coeff_sf"): "GELU 多项式系数共享 encode scaling factor。",
    (5, "normalize_rescale_sf"): "normalize 后的 rescale scaling factor。",
    (5, "gamma_rescale_sf"): "gamma 乘法后的 rescale scaling factor。",
    (5, "wffn1_rescale_sf"): "Wffn1 * X 后的 rescale scaling factor。",
    (5, "gelu_power_rescale_sf_0"): "GELU x^2 幂次计算后的 rescale scaling factor。",
    (5, "gelu_power_rescale_sf_1"): "GELU x^3 幂次计算后的 rescale scaling factor，degree 条件启用。",
    (5, "gelu_power_rescale_sf_2"): "GELU x^4 幂次计算后的 rescale scaling factor，degree 条件启用。",
    (5, "gelu_coeff_mul_rescale_sf_0"): "第 0 个 GELU 系数乘法结果后的 rescale scaling factor。",
    (5, "gelu_coeff_mul_rescale_sf_1"): "第 1 个 GELU 系数乘法结果后的 rescale scaling factor。",
    (5, "gelu_coeff_mul_rescale_sf_2"): "第 2 个 GELU 系数乘法结果后的 rescale scaling factor。",
    (5, "gelu_coeff_mul_rescale_sf_3"): "第 3 个 GELU 系数乘法结果后的 rescale scaling factor。",
    (5, "output_truncation_k"): "Block 5 末尾 CKKS/MPC 转换模拟的 truncation bit。",
}


ROTATION_ROWS = [
    ("B1.rot1", "rotation_after_gelu_out_fresh", "gelu_out_fresh", "GELU fresh 后的 rotation。"),
    ("B1.rot2", "rotation_after_wffn2_rescale_a", "wffn2_result_rescale", "Wffn2 rescale 后的第一次 rotation。"),
    ("B1.rot3", "rotation_after_wffn2_rescale_b", "wffn2_result_rescale", "Wffn2 rescale 后的第二次连续 rotation。"),
    ("B1.rot4", "rotation_after_square_rescale", "square_result_rescale", "square rescale 后的 rotation。"),
    ("B2.rot1", "rotation_after_gamma_rescale", "gamma_result_rescale", "gamma/y 乘法后的 rescale 后 rotation。"),
    ("B2.rot2_group", "rotation_after_wq_rescale", "wq_result_rescale", "WqX 后 rescale 后 rotation。"),
    ("B2.rot2_group", "rotation_after_wk_rescale", "wk_result_rescale", "WkX 后 rescale 后 rotation。"),
    ("B2.rot2_group", "rotation_after_wv_rescale", "wv_result_rescale", "WvX 后 rescale 后 rotation。"),
    ("B2.rot3_group", "rotation_after_q_mask1_rescale", "q_mask1_result_rescale", "Q 第一个 mask rescale 后 rotation。"),
    ("B2.rot3_group", "rotation_after_kt_mask1_rescale", "kt_mask1_result_rescale", "K/KT 第一个 mask rescale 后 rotation。"),
    ("B2.rot4_group", "rotation_after_q_mask2_rescale", "q_mask2_result_rescale", "Q 第二个 mask rescale 后 rotation。"),
    ("B2.rot4_group", "rotation_after_kt_mask2_rescale", "kt_mask2_result_rescale", "K/KT 第二个 mask rescale 后 rotation。"),
    ("B2.rot5", "rotation_after_qkt_matmul_rescale", "qkt_matmul_result_rescale", "QK^T matmul rescale 后 rotation。"),
    ("B4.rot1", "rotation_after_softmax_out_mask_rescale", "softmax_out_mask_result_rescale", "Softmax mask 后 rescale 后 rotation。"),
    ("B4.rot2", "rotation_after_v_mask_rescale", "v_mask_result_rescale", "V mask 后 rescale 后 rotation。"),
    ("B4.rot3", "rotation_after_softmax_v_matmul_rescale", "softmax_v_matmul_result_rescale", "Softmax*V matmul rescale 后 rotation。"),
    ("B4.rot4", "rotation_after_softmax_v_mask_rescale", "softmax_v_mask_result_rescale", "Softmax*V mask rescale 后 rotation。"),
    ("B4.rot5", "rotation_after_wo_rescale", "wo_result_rescale", "Wo rescale 后 rotation。"),
    ("B4.rot6", "rotation_after_ln_square_rescale", "ln_square_result_rescale", "LayerNorm square rescale 后 rotation。"),
    ("B5.rot1", "rotation_after_gamma_rescale", "gamma_result_rescale", "gamma/y 乘法后 rescale 后 rotation。"),
    ("B5.rot2", "rotation_after_wffn1_rescale", "wffn1_result_rescale", "Wffn1*X 后 rescale 后 rotation。"),
]


def _parse_degree_vector(raw: str | Sequence[int] | None, *, num_layers: int, default: int) -> List[int]:
    return parse_broadcast_int_vector(
        raw,
        num_layers=int(num_layers),
        default=int(default),
        name="degree vector",
    )


def _scale_semantics(record: Dict[str, Any]) -> str:
    kind = str(record.get("kind", ""))
    if kind in ("F", "W", "M", "S"):
        return "encode/fresh increases current CKKS scale and controls simulator noise variance"
    if kind == "R":
        return "rescale action selects the target scale after this CKKS rescale point"
    if kind == "K":
        return "truncation action selects MPC/CKKS conversion fractional bits"
    return "see action_space field kind"


def _semantic_type(kind: str) -> str:
    return SEMANTIC_TYPE_BY_KIND.get(str(kind), str(kind))


def _noise_lookup_distribution(record: Dict[str, Any]) -> str:
    if str(record.get("kind")) == "S":
        return "unknown; kind=S is scalar_encode but NoisePoint lookup may still use encoding"
    return str(record.get("distribution", ""))


def _active_condition(record: Dict[str, Any]) -> str:
    if bool(record.get("is_effective", True)):
        return "always" if not str(record.get("ineffective_reason", "")) else "degree-dependent"
    return str(record.get("ineffective_reason", "")) or "inactive in current degree/profile"


def _all_max_action_index(record: Dict[str, Any], *, k_levels: Sequence[int] | None = None) -> int:
    if record.get("value_type") == "truncation_k":
        if k_levels is None:
            k_levels = _load_action_space_deps()["K_LEVELS"]
        return int(list(k_levels).index(max(k_levels)))
    return int(record.get("num_levels", 1)) - 1


def _nullable_int(value: Any) -> Any:
    return None if value is None else int(value)


def _registry_record(record: Dict[str, Any], *, k_levels: Sequence[int] | None = None) -> Dict[str, Any]:
    note = str(record.get("note", "") or "")
    block_idx = record.get("block_index")
    block_key = None if block_idx is None else int(block_idx)
    field = str(record["field"])
    action_values = [_nullable_int(v) for v in record.get("level_values", [])]
    return {
        "slot_id": (
            "first_input"
            if block_key is None
            else f"L{int(record['layer'])}.B{block_key}.{field}"
        ),
        "global_index": int(record["global_index"]),
        "layer": int(record["layer"]),
        "block_index": block_idx,
        "block": str(record["block"]),
        "field": field,
        "kind": str(record["kind"]),
        "semantic_type": _semantic_type(str(record["kind"])),
        "user_prompt_semantics": (
            "第 0 层 embedding 输入进入 BLB 模拟时的 first-input fresh 噪声入口。"
            if block_key is None
            else FIELD_SEMANTICS.get((block_key, field), "")
        ),
        "operation": str(record["operation"]),
        "current_code_operation": str(record["operation"]),
        "location": str(record["location"]),
        "config_name": str(record["config_name"]),
        "is_required": bool(record.get("effective", True)),
        "is_effective": bool(record.get("effective", True)),
        "is_action_slot": True,
        "ineffective_reason": note,
        "active_condition": _active_condition({
            "is_effective": bool(record.get("effective", True)),
            "ineffective_reason": note,
        }),
        "value_type": str(record["value_type"]),
        "N": record.get("N"),
        "max_sf": record.get("max_sf"),
        "num_levels": int(record["num_levels"]),
        "action_index_count": int(record["num_levels"]),
        "action_index": int(record["action_index"]),
        "action_values": action_values,
        "level_values": action_values,
        "index_to_value": {str(i): _nullable_int(v) for i, v in enumerate(action_values)},
        "decoded_value": _nullable_int(record["value"]),
        "effective_value": record.get("effective_value"),
        "all_max_action_index": _all_max_action_index(record, k_levels=k_levels),
        "distribution": str(record.get("distribution", "")),
        "noise_lookup_distribution": _noise_lookup_distribution(record),
        "scale_semantics": _scale_semantics(record),
        "decode_rule": "sf_from(idx,max_sf,levels) then snap_to_table; K uses K_LEVELS",
        "max_sf_source": "max_sfs json or action_space fallback default",
        "N_source": "block default N / degree-aware action_space rule",
        "rotation_dependency": None,
        "notes": note,
        "source": "blb_stage2_rl.action_space.describe_action_vector",
    }


def _required_count_by_layer(records: Sequence[Dict[str, Any]]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for record in records:
        if record["block"] == "first_input":
            continue
        if not record["is_required"]:
            continue
        layer = int(record["layer"])
        counts[layer] = counts.get(layer, 0) + 1
    return counts


def _slot_count_markdown(
        *,
        profile: str,
        num_layers: int,
        expected_slots_per_layer: int,
        records: Sequence[Dict[str, Any]],
        ) -> str:
    counts = _required_count_by_layer(records)
    mismatched = {
        layer: count
        for layer, count in sorted(counts.items())
        if int(count) != int(expected_slots_per_layer)
    }
    status = "match" if not mismatched and len(counts) == int(num_layers) else "mismatch"
    lines = [
        "# BLB Action Registry Current-Code Slot Check",
        "",
        f"- profile: `{profile}`",
        f"- num_layers: `{int(num_layers)}`",
        f"- expected_slots_per_layer: `{int(expected_slots_per_layer)}`",
        f"- status: `{status}`",
        f"- effective_required_total: `{sum(counts.values())}`",
        "",
        "| layer | required_slots | status |",
        "|---:|---:|---|",
    ]
    for layer in range(int(num_layers)):
        count = int(counts.get(layer, 0))
        row_status = "ok" if count == int(expected_slots_per_layer) else "mismatch"
        lines.append(f"| {layer} | {count} | {row_status} |")
    if mismatched:
        lines.extend([
            "",
            "Safe handling: keep every current action field in the registry; mark non-required fields as compat or ineffective extras instead of deleting them.",
        ])
    return "\n".join(lines) + "\n"


def _mapping_markdown(records: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "# BLB Action Index Mapping",
        "",
        "| idx | location | kind | values | all_max_idx | required |",
        "|---:|---|---|---|---:|---|",
    ]
    for record in records:
        values = ",".join(str(v) for v in record["action_values"])
        lines.append(
            f"| {record['global_index']} | `{record['location']}` | `{record['kind']}` | "
            f"`{values}` | {record['all_max_action_index']} | {str(record['is_required']).lower()} |"
        )
    return "\n".join(lines) + "\n"


def _registry_markdown(payload: Dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# BLB Stage-2 当前代码动作 Registry",
        "",
        f"- profile: `{payload['profile']}`",
        f"- num_layers: `{payload['num_layers']}`",
        f"- registry_hash: `{payload['registry_hash']}`",
        f"- 每层槽位: `{summary['per_layer_slot_count']}`",
        f"- 完整 action length: `{summary['full_action_length']}`",
        f"- first-input tail: `{summary['first_input_tail_slots']}`",
        "",
        "| block | slots per layer |",
        "|---|---:|",
    ]
    for block, count in sorted(summary["block_slot_counts_per_layer"].items()):
        lines.append(f"| {block} | {count} |")
    lines.extend([
        "",
        "| idx | slot_id | kind | semantic_type | values | effective |",
        "|---:|---|---|---|---|---|",
    ])
    for record in payload["slot_registry_full"]:
        values = ",".join(str(v) for v in record["action_values"])
        lines.append(
            f"| {record['global_index']} | `{record['slot_id']}` | `{record['kind']}` | "
            f"`{record['semantic_type']}` | `{values}` | {str(record['is_effective']).lower()} |"
        )
    return "\n".join(lines) + "\n"


def _slot_semantics_markdown(payload: Dict[str, Any]) -> str:
    seen = set()
    lines = [
        "# BLB Stage-2 当前代码槽位语义",
        "",
        "说明：RL 输出 action index，不直接输出 scaling factor，也不决定操作是否存在；mask/curriculum 只能限制某个槽位允许的 index。",
        "",
    ]
    prev_block = None
    for record in payload["slot_registry_full"]:
        key = (record.get("block_index"), record["field"])
        if key in seen or record["block"] == "first_input":
            continue
        seen.add(key)
        if len(seen) == 1 or (len(seen) > 1 and record.get("block_index") != prev_block):
            if prev_block is not None:
                lines.append("")
            lines.extend([f"## Block {record.get('block_index')}", "", "| field | kind | semantic_type | 用户语义 |", "|---|---|---|---|"])
        lines.append(
            f"| `{record['field']}` | `{record['kind']}` | `{record['semantic_type']}` | "
            f"{record.get('user_prompt_semantics', '')} |"
        )
        prev_block = record.get("block_index")
    lines.extend([
        "",
        "## first-input",
        "",
        "完整 action vector 尾部有一个 `first_input_sf`，用于第 0 层 embedding 输入进入 BLB 模拟时的 fresh 噪声入口。",
        "",
        "## S-kind 注意",
        "",
        "`kind=S` 的语义是 scalar encode，例如 `1/D`、`1/2^n`、LayerNorm 标量。当前代码描述中的 noise lookup distribution 可能仍需从实际 `NoisePoint` 验证；未知处在 JSON 中标记为 `unknown`。",
        "",
    ])
    return "\n".join(lines)


def _rotation_markdown() -> str:
    lines = [
        "# BLB Stage-2 Rotation 派生噪声点",
        "",
        "Rotation 不是独立 RL action slot，不计入 action vector 长度。它通常紧跟 fresh 或 rescale 点，scaling factor 继承绑定源。",
        "",
        "| user group | current code flag | inherits scale from | 语义 |",
        "|---|---|---|---|",
    ]
    for group, flag, source, text in ROTATION_ROWS:
        lines.append(f"| `{group}` | `{flag}` | `{source}` | {text} |")
    lines.append("")
    return "\n".join(lines)


def build_registry_payload(
        *,
        profile: str,
        num_layers: int,
        gelu_degree: str | Sequence[int] | None = None,
        attn_degree: str | Sequence[int] | None = None,
        expected_slots_per_layer: int | None = None,
        ) -> Dict[str, Any]:
    deps = _load_action_space_deps()
    k_levels = deps["K_LEVELS"]
    describe_action_vector = deps["describe_action_vector"]
    load_max_sfs = deps["load_max_sfs"]
    make_all_max_action_vector = deps["make_all_max_action_vector"]
    per_layer_offsets = list(deps["per_layer_field_offsets"]())

    gelu = _parse_degree_vector(gelu_degree, num_layers=num_layers, default=4)
    attn = _parse_degree_vector(attn_degree, num_layers=num_layers, default=4)
    if expected_slots_per_layer is None:
        expected_slots_per_layer = len(per_layer_offsets)
    action = make_all_max_action_vector(num_layers=num_layers)
    description = describe_action_vector(
        action,
        max_sfs=load_max_sfs(profile),
        num_layers=num_layers,
        gelu_degree=gelu,
        attn_degree=attn,
        profile=profile,
    )
    records = [_registry_record(record, k_levels=k_levels) for record in description["records"]]
    effective = [record for record in records if record["is_required"]]
    slot_check = _slot_count_markdown(
        profile=profile,
        num_layers=num_layers,
        expected_slots_per_layer=expected_slots_per_layer,
        records=records,
    )
    block_counts: Dict[str, int] = {}
    for _block_idx, _field, _kind in per_layer_offsets:
        key = f"block{int(_block_idx)}"
        block_counts[key] = block_counts.get(key, 0) + 1
    registry_hash = hashlib.sha256(
        json.dumps(records, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema": "blb_action_registry_export_v1",
        "action_space_version": "current-code-v1",
        "profile": str(profile),
        "num_layers": int(num_layers),
        "gelu_degree": gelu,
        "attn_degree": attn,
        "expected_slots_per_layer": int(expected_slots_per_layer),
        "registry_hash": registry_hash,
        "summary": {
            "slot_count": len(records),
            "required_slot_count": len(effective),
            "ineffective_or_compat_extra_count": len(records) - len(effective),
            "required_count_by_layer": _required_count_by_layer(records),
            "block_slot_counts_per_layer": block_counts,
            "per_layer_slot_count": int(len(per_layer_offsets)),
            "first_input_tail_slots": 1,
            "full_action_length": int(len(records)),
        },
        "slot_registry_full": records,
        "slot_registry_effective": effective,
        "current_code_slot_check_markdown": slot_check,
        "action_index_mapping_markdown": _mapping_markdown(records),
    }


def write_registry_artifacts(payload: Dict[str, Any], output_dir: os.PathLike[str] | str) -> Dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    full_path = out / "current_code_action_registry.json"
    md_path = out / "current_code_action_registry.md"
    semantics_path = out / "current_code_slot_semantics.md"
    rotation_path = out / "rotation_derived_slots.md"
    slot_registry_full_path = out / "slot_registry_full.json"
    effective_path = out / "slot_registry_effective.json"
    slot_check_path = out / "current_code_slot_check.md"
    mapping_path = out / "action_index_mapping.md"
    full_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(_registry_markdown(payload), encoding="utf-8")
    semantics_path.write_text(_slot_semantics_markdown(payload), encoding="utf-8")
    rotation_path.write_text(_rotation_markdown(), encoding="utf-8")
    slot_registry_full_path.write_text(
        json.dumps(payload["slot_registry_full"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    effective_path.write_text(
        json.dumps(payload["slot_registry_effective"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    slot_check_path.write_text(payload["current_code_slot_check_markdown"], encoding="utf-8")
    mapping_path.write_text(payload["action_index_mapping_markdown"], encoding="utf-8")
    return {
        "current_code_action_registry": str(full_path),
        "current_code_action_registry_md": str(md_path),
        "current_code_slot_semantics": str(semantics_path),
        "rotation_derived_slots": str(rotation_path),
        "slot_registry_full": str(slot_registry_full_path),
        "slot_registry_effective": str(effective_path),
        "current_code_slot_check": str(slot_check_path),
        "action_index_mapping": str(mapping_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="mrpc")
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--fixed-gelu", default="")
    parser.add_argument("--fixed-softmax", default="")
    parser.add_argument("--expected-slots-per-layer", type=int, default=None)
    parser.add_argument("--output-dir", default="reports/blb_opt/trust0_registry")
    args = parser.parse_args(argv)

    payload = build_registry_payload(
        profile=args.profile,
        num_layers=args.num_layers,
        gelu_degree=args.fixed_gelu,
        attn_degree=args.fixed_softmax,
        expected_slots_per_layer=args.expected_slots_per_layer,
    )
    paths = write_registry_artifacts(payload, args.output_dir)
    print(json.dumps(paths, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
