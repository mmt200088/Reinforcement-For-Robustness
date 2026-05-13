"""BLB Stage 2 RL 动作空间定义、SF/k 编解码、cfg 构建。

每个层、每个 BLB block 的离散动作字段（按 spec §4.2）：

| Block | F (fresh) | W (weight) | M / S | R (rescale) | K | 总 SF dim |
| ----- | --------- | ---------- | ----- | ----------- | - | --------- |
| 1     | 1         | 1          | 2     | 4           | 1 | 8 + 1K |
| 2     | 2         | 3          | 6     | 11          | 1 | 22 + 1K |
| 3     | 1         | 0          | 1     | 1+degree    | 1 | 7 + 1K (deg=4) |
| 4     | 2         | 1          | 5     | 8           | 1 | 16 + 1K |
| 5     | 2         | 1          | 2     | 3+(deg-1)+deg | 1 | 15 + 1K (deg=4) |
| first-input | 1   | 0          | 0     | 0           | 0 | 1 |

每层 73 维（softmax_deg=4, gelu_deg=4 各占满），L=12 层 → 73*12 + 1 = 877 维。
（旧注释里 94 维和 1129 维已废弃；以 ``action_dims_for_config(num_layers)`` 实际返回为准。
 槽位数不再采用旧记忆假设，权威分类由 ``scripts/blb_export_action_registry.py`` 从当前代码导出。）

动作向量布局（顺序）：
  对每层 i ∈ [0, L):
    block1 dims (9) | block2 dims (23) | block3 dims (7+1) | block4 dims (17) | block5 dims (16)
  尾部追加 first_input_sf (1 dim)

每个分量都是 [0, num_levels) 的 categorical index；按
``sf_from(idx, max, levels) = max - 2 * (levels - 1 - idx)`` 反推 SF 实际值。
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from blb_rl_bridge import (
    Block1ActionSpec,
    Block2ActionSpec,
    Block3ActionSpec,
    Block4ActionSpec,
    Block5ActionSpec,
    build_block1_cfg_from_action,
    build_block2_cfg_from_action,
    build_block3_cfg_from_action,
    build_block4_cfg_from_action,
    build_block5_cfg_from_action,
)
from function_handler import (
    NOISE_TABLE_ALLOWED_SCALING_FACTORS_BY_N,
    Block1NoiseConfig,
    Block2NoiseConfig,
    Block3NoiseConfig,
    Block4NoiseConfig,
    Block5NoiseConfig,
)


# ---------------------------------------------------------------------------
# 全局常量：每类噪声的离散挡位数（与 spec §4.1 对齐）
# ---------------------------------------------------------------------------
LEVELS_W = 5         # weight encode (5 挡：max-8 .. max)
LEVELS_MS = 3        # mask / scalar encode (3 挡：max-4 .. max)
LEVELS_R = 4         # rescale (4 挡：max-6 .. max)
LEVELS_F = 5         # fresh (5 挡：max-8 .. max)

# Truncation K 挡位默认扩展为 6 档，但保持旧 checkpoint / 旧 action vector 的
# index 语义：0/1/2/3 仍然解码为 8/9/11/13，新挡位 10/12 追加在后面。
# 如需做临时实验，可用环境变量覆盖，例如：
#   BLB_TRUNCATION_K_LEVELS=8,9,11,13,10,12
DEFAULT_K_LEVELS_LEGACY_COMPAT: Tuple[int, ...] = (8, 9, 11, 13, 10, 12)


def _load_k_levels_from_env() -> Tuple[int, ...]:
    raw = str(os.environ.get("BLB_TRUNCATION_K_LEVELS", "") or "").strip()
    if not raw:
        return DEFAULT_K_LEVELS_LEGACY_COMPAT
    values = tuple(int(x.strip()) for x in raw.replace(";", ",").split(",") if x.strip())
    if not values:
        raise ValueError("BLB_TRUNCATION_K_LEVELS must contain at least one integer")
    if len(set(values)) != len(values):
        raise ValueError(f"BLB_TRUNCATION_K_LEVELS contains duplicate values: {values}")
    return values


K_LEVELS: Tuple[int, ...] = _load_k_levels_from_env()
LEVELS_K = len(K_LEVELS)
LEVELS_FIRST_INPUT = 5   # 与 fresh 一致
BLB_FIRST_INPUT_N = 8192

# 离散挡位数（与 cfg 字段一一对应；同时影响 reward / policy 头维度）
NUM_LEVELS_PER_DIM_BY_BLOCK_KIND = {
    "F": LEVELS_F,
    "W": LEVELS_W,
    "M": LEVELS_MS,
    "S": LEVELS_MS,
    "R": LEVELS_R,
    "K": LEVELS_K,
}


# ---------------------------------------------------------------------------
# Block 字段表（顺序极其重要，必须与 build_*_cfg_from_action 的字段顺序对齐）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _BlockFieldSpec:
    """单层中一个 block 的离散动作字段表。

    每个 entry 形如 ``(field_name, kind, default_max_sf)``：
      * ``field_name``：``Block{N}ActionSpec`` 的字段名（用于 setattr）
      * ``kind``：``F/W/M/S/R/K`` 之一（决定挡位数）
      * ``default_max_sf``：用于 max_sfs JSON 缺失节点时的兜底值
                          （来自 ``make_block{N}_default_config`` 的默认 SF）
    """
    fields: Tuple[Tuple[str, str, int], ...]


# Block 1（不含首层 K 字段；首层会在 cfg build 时强制 truncation_k=None）
_BLOCK1_FIELDS = _BlockFieldSpec(
    fields=(
        ("gelu_out_sf",        "F", 30),
        ("wffn2_sf",           "W", 22),
        ("mean_inv_d_sf",      "S", 22),
        ("var_inv_d_sf",       "S", 22),
        ("wffn2_rescale_sf",   "R", 22),
        ("mean_rescale_sf",    "R", 22),
        ("square_rescale_sf",  "R", 22),
        ("var_rescale_sf",     "R", 22),
        ("output_truncation_k","K", 13),
    ),
)


# Block 2
_BLOCK2_FIELDS = _BlockFieldSpec(
    fields=(
        ("inv_std_fresh_sf",            "F", 30),
        ("x_centered_fresh_sf",         "F", 30),
        ("gamma_sf",                    "M", 22),
        ("wq_sf",                       "W", 22),
        ("wk_sf",                       "W", 22),
        ("wv_sf",                       "W", 22),
        ("kt_mask1_sf",                 "M", 22),
        ("kt_mask2_sf",                 "M", 22),
        ("q_mask1_sf",                  "M", 22),
        ("q_mask2_sf",                  "M", 22),
        ("qkt_merge_mask_sf",           "M", 22),
        ("normalize_rescale_sf",        "R", 22),
        ("gamma_rescale_sf",            "R", 22),
        ("wk_rescale_sf",               "R", 22),
        ("wq_rescale_sf",               "R", 22),
        ("wv_rescale_sf",               "R", 22),
        ("kt_mask1_rescale_sf",         "R", 22),
        ("kt_mask2_rescale_sf",         "R", 22),
        ("q_mask1_rescale_sf",          "R", 22),
        ("q_mask2_rescale_sf",          "R", 22),
        ("qkt_matmul_rescale_sf",       "R", 22),
        ("qkt_merge_mask_rescale_sf",   "R", 22),
        ("output_truncation_k",         "K", 13),
    ),
)


# Block 3 字段是 degree-aware 的 —— square_rescale_sfs 长度 == degree。
# 我们让 max degree=4（也是 default），动作里固定占 5 个 R 槽（1 个 x_inv_2n + 4 个 square）。
# build_*_cfg 时按实际 degree 截短；多出来的 dim 在 RL 出动作里占位（不用就掩掉）。
_BLOCK3_R_SLOTS = 1 + 4   # x_inv_2n_rescale + 4 个 square_rescale (max degree=4)
_BLOCK3_FIELDS = _BlockFieldSpec(
    fields=(
        ("x_fresh_sf",              "F", 30),
        ("inv_2n_sf",               "S", 22),
        ("x_inv_2n_rescale_sf",     "R", 22),
        ("square_rescale_sf_0",     "R", 22),
        ("square_rescale_sf_1",     "R", 22),
        ("square_rescale_sf_2",     "R", 22),
        ("square_rescale_sf_3",     "R", 22),
        ("output_truncation_k",     "K", 13),
    ),
)


# Block 4
_BLOCK4_FIELDS = _BlockFieldSpec(
    fields=(
        ("softmax_out_fresh_sf",            "F", 30),
        ("v_fresh_sf",                      "F", 30),
        ("softmax_out_mask_sf",             "M", 22),
        ("v_mask_sf",                       "M", 22),
        ("softmax_v_mask_sf",               "M", 22),
        ("ln_mean_inv_d_sf",                "S", 22),
        ("ln_var_inv_d_sf",                 "S", 22),
        ("wo_sf",                           "W", 22),
        ("softmax_out_mask_rescale_sf",     "R", 22),
        ("v_mask_rescale_sf",               "R", 22),
        ("softmax_v_matmul_rescale_sf",     "R", 22),
        ("softmax_v_mask_rescale_sf",       "R", 22),
        ("wo_rescale_sf",                   "R", 22),
        ("ln_mean_rescale_sf",              "R", 22),
        ("ln_square_rescale_sf",            "R", 22),
        ("ln_var_rescale_sf",               "R", 22),
        ("output_truncation_k",             "K", 13),
    ),
)


# Block 5（GELU degree-aware）：固定按 max gelu_degree=4 占槽
# gelu_power_rescales 长度 = degree-1 = 3
# gelu_coeff_mul_rescales 长度 = degree = 4
# normalize/gamma/wffn1 各 1 个 R = 3
# 加 K = 16 维
_BLOCK5_FIELDS = _BlockFieldSpec(
    fields=(
        ("inv_std_fresh_sf",                "F", 30),
        ("x_centered_fresh_sf",             "F", 30),
        ("gamma_sf",                        "M", 22),
        ("wffn1_sf",                        "W", 22),
        ("gelu_coeff_sf",                   "M", 22),
        ("normalize_rescale_sf",            "R", 22),
        ("gamma_rescale_sf",                "R", 22),
        ("wffn1_rescale_sf",                "R", 22),
        ("gelu_power_rescale_sf_0",         "R", 22),  # x²
        ("gelu_power_rescale_sf_1",         "R", 22),  # x³（degree==4 时启用）
        ("gelu_power_rescale_sf_2",         "R", 22),  # x⁴（degree==4 时启用）
        ("gelu_coeff_mul_rescale_sf_0",     "R", 22),
        ("gelu_coeff_mul_rescale_sf_1",     "R", 22),
        ("gelu_coeff_mul_rescale_sf_2",     "R", 22),
        ("gelu_coeff_mul_rescale_sf_3",     "R", 22),
        ("output_truncation_k",             "K", 13),
    ),
)


_BLOCK_SPECS: Dict[int, _BlockFieldSpec] = {
    1: _BLOCK1_FIELDS,
    2: _BLOCK2_FIELDS,
    3: _BLOCK3_FIELDS,
    4: _BLOCK4_FIELDS,
    5: _BLOCK5_FIELDS,
}


# Per-block "节点名" → block 字段名（用于 max_sfs JSON 反查）。
# 与 ``rescale_optimizer_bridge.default_block*_cfg_to_delta`` 的命名约定保持一致；
# 实际外部 ``static_skeletons_<profile>.json`` 节点名以用户最终使用的 config 为准。
_BLOCK_NODE_NAME_BY_FIELD: Dict[int, Dict[str, str]] = {
    1: {
        "gelu_out_sf":          "ctpt_gelu_out",
        "wffn2_sf":             "ctpt_ffn2",
        "mean_inv_d_sf":        "ctpt_inv_d_1",
        "var_inv_d_sf":         "ctpt_inv_d_2",
        "wffn2_rescale_sf":     "ctct_ffn2_rescale",
        "mean_rescale_sf":      "ctct_mean_rescale",
        "square_rescale_sf":    "ctct_ext_square",
        "var_rescale_sf":       "ctct_var_rescale",
    },
    2: {
        "inv_std_fresh_sf":            "ctpt_inv_std",
        "x_centered_fresh_sf":         "ctpt_x_centered",
        "gamma_sf":                    "ctpt_gamma",
        "wq_sf":                       "ctpt_wq",
        "wk_sf":                       "ctpt_wk",
        "wv_sf":                       "ctpt_wv",
        "kt_mask1_sf":                 "ctpt_kt_mask1",
        "kt_mask2_sf":                 "ctpt_kt_mask2",
        "q_mask1_sf":                  "ctpt_q_mask1",
        "q_mask2_sf":                  "ctpt_q_mask2",
        "qkt_merge_mask_sf":           "ctpt_qkt_merge_mask",
        "normalize_rescale_sf":        "ctct_normalize_rescale",
        "gamma_rescale_sf":            "ctct_gamma_rescale",
        "wk_rescale_sf":               "ctct_wk_rescale",
        "wq_rescale_sf":               "ctct_wq_rescale",
        "wv_rescale_sf":               "ctct_wv_rescale",
        "kt_mask1_rescale_sf":         "ctct_kt_mask1_rescale",
        "kt_mask2_rescale_sf":         "ctct_kt_mask2_rescale",
        "q_mask1_rescale_sf":          "ctct_q_mask1_rescale",
        "q_mask2_rescale_sf":          "ctct_q_mask2_rescale",
        "qkt_matmul_rescale_sf":       "ctct_qkt_matmul_rescale",
        "qkt_merge_mask_rescale_sf":   "ctct_qkt_merge_mask_rescale",
    },
    3: {
        "x_fresh_sf":               "ctpt_softmax_x",
        "inv_2n_sf":                "ctpt_softmax_inv_2n",
        "x_inv_2n_rescale_sf":      "ctct_softmax_x_inv_2n_rescale",
        "square_rescale_sf_0":      "ctct_softmax_pow_s1",
        "square_rescale_sf_1":      "ctct_softmax_pow_s2",
        "square_rescale_sf_2":      "ctct_softmax_pow_s3",
        "square_rescale_sf_3":      "ctct_softmax_pow_s4",
    },
    4: {
        "softmax_out_fresh_sf":         "ctpt_softmax_out",
        "v_fresh_sf":                   "ctpt_v",
        "softmax_out_mask_sf":          "ctpt_softmax_out_mask",
        "v_mask_sf":                    "ctpt_v_mask",
        "softmax_v_mask_sf":            "ctpt_softmax_v_mask",
        "ln_mean_inv_d_sf":             "ctpt_inv_d_attn_mean",
        "ln_var_inv_d_sf":              "ctpt_inv_d_attn_var",
        "wo_sf":                        "ctpt_wo",
        "softmax_out_mask_rescale_sf":  "ctct_softmax_out_mask_rescale",
        "v_mask_rescale_sf":            "ctct_v_mask_rescale",
        "softmax_v_matmul_rescale_sf":  "ctct_softmax_v_matmul_rescale",
        "softmax_v_mask_rescale_sf":    "ctct_softmax_v_mask_rescale",
        "wo_rescale_sf":                "ctct_wo_rescale",
        "ln_mean_rescale_sf":           "ctct_attn_mean_rescale",
        "ln_square_rescale_sf":         "ctct_attn_square_rescale",
        "ln_var_rescale_sf":            "ctct_attn_var_rescale",
    },
    5: {
        "inv_std_fresh_sf":                 "ctpt_attn_inv_std",
        "x_centered_fresh_sf":              "ctpt_attn_x_centered",
        "gamma_sf":                         "ctpt_gamma_attn",
        "wffn1_sf":                         "ctpt_wffn1",
        "gelu_coeff_sf":                    "ctpt_gelu_coeff",
        "normalize_rescale_sf":             "ctct_attn_normalize_rescale",
        "gamma_rescale_sf":                 "ctct_gamma_attn_rescale",
        "wffn1_rescale_sf":                 "ctct_wffn1_rescale",
        "gelu_power_rescale_sf_0":          "ctct_gelu_x2",
        "gelu_power_rescale_sf_1":          "ctct_gelu_x3",
        "gelu_power_rescale_sf_2":          "ctct_gelu_x4",
        "gelu_coeff_mul_rescale_sf_0":      "ctct_gelu_b_x",
        "gelu_coeff_mul_rescale_sf_1":      "ctct_gelu_c_x2",
        "gelu_coeff_mul_rescale_sf_2":      "ctct_gelu_d_x3",
        "gelu_coeff_mul_rescale_sf_3":      "ctct_gelu_e_x4",
    },
}


# ---------------------------------------------------------------------------
# Block N 选取（与 ``make_block*_default_config`` 推荐一致）
# ---------------------------------------------------------------------------
def _block_default_N(block_idx: int, gelu_degree: int = 4, attn_degree: int = 4) -> int:
    """返回 block 默认使用的 CKKS 多项式阶 N（与 ``make_block*_default_config`` 一致）。"""
    if block_idx == 1:
        return 8192
    if block_idx == 5 and int(gelu_degree) == 1:
        return 8192
    if block_idx == 3 and int(attn_degree) == 2:
        return 8192
    return 16384


# ---------------------------------------------------------------------------
# action index ↔ scaling factor 转换
# ---------------------------------------------------------------------------
def sf_from(idx: int, max_sf: int, levels: int) -> int:
    """``sf_from(idx, max, levels) = max - 2 * (levels - 1 - idx)``。

    例如 levels=5, max=30 → idx 0..4 ↔ {22, 24, 26, 28, 30}（与 spec 一致）。
    """
    idx = int(idx)
    levels = int(levels)
    if idx < 0 or idx >= levels:
        raise ValueError(f"action idx {idx} out of [0, {levels})")
    return int(max_sf) - 2 * (levels - 1 - idx)


def _rescale_sf_from_index(idx: int, max_sf: int) -> Optional[int]:
    idx = int(idx)
    if idx <= 0:
        return None
    return sf_from(idx, int(max_sf), LEVELS_R)


def _optional_int(value: object) -> Optional[int]:
    return None if value is None else int(value)


def _snap_to_table(sf: int, N: int) -> int:
    """把 SF 钳到 ``NOISE_VARIANCE_TABLE_BY_N[N]`` 实际存在的 key 上。

    这样即便 max_sf JSON 配错或 sf_from 反推出表外的值，仍能找到最接近的合法 SF
    （取小于等于的最大值，找不到就回退到表里最小值）。
    """
    sf = int(sf)
    allowed = list(NOISE_TABLE_ALLOWED_SCALING_FACTORS_BY_N.get(int(N), ()))
    if not allowed:
        return sf
    if sf in allowed:
        return sf
    smaller = [v for v in allowed if v <= sf]
    if smaller:
        return max(smaller)
    return min(allowed)


# ---------------------------------------------------------------------------
# max_sfs JSON 加载
# ---------------------------------------------------------------------------
@dataclass
class MaxSFsTable:
    """每个 (block_idx, node_name) 的 max scaling factor 缓存。

    构造 RL 动作时按 ``sf_from(action_idx, max_sf, levels)`` 反推；
    JSON 缺失节点时 fallback 到 ``_BlockFieldSpec.fields`` 里的 default_max_sf。
    """
    by_block_node: Dict[Tuple[int, str], int] = field(default_factory=dict)
    by_layer_block_node: Dict[Tuple[int, int, str], int] = field(default_factory=dict)

    def get(
            self,
            block_idx: int,
            field_name: str,
            *,
            layer_idx: Optional[int] = None,
            ) -> int:
        node = _BLOCK_NODE_NAME_BY_FIELD.get(int(block_idx), {}).get(str(field_name))
        if node is not None:
            if layer_idx is not None:
                v = self.by_layer_block_node.get((int(layer_idx), int(block_idx), node))
                if v is not None:
                    return int(v)
            v = self.by_block_node.get((int(block_idx), node))
            if v is not None:
                return int(v)
        # fallback 到 _BLOCK_SPECS 默认
        for fname, kind, default_max_sf in _BLOCK_SPECS[int(block_idx)].fields:
            if fname == field_name:
                return int(default_max_sf)
        return 22  # 终极兜底


def load_max_sfs(profile: str, search_paths: Optional[Sequence[str]] = None) -> MaxSFsTable:
    """从 ``blb_stage2_rl/max_sfs/<profile>.json`` 加载 max SF 表。

    JSON 结构（完全可选；缺字段或文件不存在都允许）：

        {
            "block1": {"ctpt_ffn2": 30, "ctpt_inv_d_1": 22, ...},
            "block2": {...},
            "block3": {...},
            "block4": {...},
            "block5": {...}
        }

    若想从 ``Rescale_optimizer/configs/<profile>/static_skeletons_<profile>.json``
    自动生成，参见 ``docs/BLB_stage2_rl_spec.md`` §4.4。
    """
    profile = str(profile or "default")
    table = MaxSFsTable(by_block_node={})

    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates: List[str] = []
    if search_paths:
        candidates.extend(search_paths)
    candidates.append(os.path.join(base_dir, "max_sfs", f"{profile}.json"))
    candidates.append(os.path.join(base_dir, "max_sfs", "default.json"))

    for path in candidates:
        if not path:
            continue
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        for block_key, mapping in payload.items():
            if not str(block_key).startswith("block"):
                continue
            try:
                block_idx = int(str(block_key)[5:])
            except (ValueError, TypeError):
                continue
            if not isinstance(mapping, dict):
                continue
            for node_name, max_sf in mapping.items():
                try:
                    table.by_block_node[(int(block_idx), str(node_name))] = int(max_sf)
                except (TypeError, ValueError):
                    continue
        # 找到一份就停止；上层覆盖下层
        break

    return table


# ---------------------------------------------------------------------------
# 动作维度 (block-wise + per-layer) 计算
# ---------------------------------------------------------------------------
def block_dims(block_idx: int) -> List[int]:
    """返回单层一个 block 的离散维度数列表（顺序 = ``_BLOCK_SPECS`` 顺序）。"""
    spec = _BLOCK_SPECS[int(block_idx)]
    return [NUM_LEVELS_PER_DIM_BY_BLOCK_KIND[kind] for _name, kind, _max in spec.fields]


def layer_dims() -> List[int]:
    """返回单层全部 5 个 block 的离散维度数列表。"""
    out: List[int] = []
    for b in (1, 2, 3, 4, 5):
        out.extend(block_dims(b))
    return out


def action_dims_for_config(num_layers: int) -> List[int]:
    """返回完整动作向量的 ``MultiDiscrete`` 维度数（含尾部 first_input）。"""
    layer_dim = layer_dims()
    out: List[int] = []
    for _ in range(int(num_layers)):
        out.extend(layer_dim)
    out.append(LEVELS_FIRST_INPUT)
    return out


def per_layer_field_offsets() -> List[Tuple[int, str, str]]:
    """返回单层动作向量内每个分量的 ``(block_idx, field_name, kind)`` 三元组。"""
    out: List[Tuple[int, str, str]] = []
    for b in (1, 2, 3, 4, 5):
        for fname, kind, _max in _BLOCK_SPECS[b].fields:
            out.append((b, fname, kind))
    return out


# ---------------------------------------------------------------------------
# action vector → cfgs
# ---------------------------------------------------------------------------
@dataclass
class ActionDecodeResult:
    """``action_vector_to_cfgs`` 的返回值。"""
    block1_cfgs: Dict[int, Block1NoiseConfig]
    block2_cfgs: Dict[int, Block2NoiseConfig]
    block3_cfgs: Dict[int, Block3NoiseConfig]
    block4_cfgs: Dict[int, Block4NoiseConfig]
    block5_cfgs: Dict[int, Block5NoiseConfig]
    first_input_sf: int
    # 调试用：原始动作 idx → SF 的 per-layer 映射
    per_layer_field_values: List[Dict[str, object]] = field(default_factory=list)

    def cfgs_dict(self) -> Dict[str, Dict[int, object]]:
        """``aggregate_*_signals`` 风格的字典视图。"""
        return {
            "block1": dict(self.block1_cfgs),
            "block2": dict(self.block2_cfgs),
            "block3": dict(self.block3_cfgs),
            "block4": dict(self.block4_cfgs),
            "block5": dict(self.block5_cfgs),
        }


def _build_block1_action(
        layer_idx: int,
        layer_field_values: Dict[str, object],
        is_first_layer: bool,
        ) -> Block1ActionSpec:
    """从单层 block1 字段值构建 ``Block1ActionSpec``。

    首层（layer 0）Block 1 缺失（spec §12 风险 #2）→ 强制 truncation_k=None。
    """
    a = Block1ActionSpec(
        gelu_out_sf=int(layer_field_values["gelu_out_sf"]),
        wffn2_sf=int(layer_field_values["wffn2_sf"]),
        mean_inv_d_sf=int(layer_field_values["mean_inv_d_sf"]),
        var_inv_d_sf=int(layer_field_values["var_inv_d_sf"]),
        wffn2_rescale_sf=_optional_int(layer_field_values["wffn2_rescale_sf"]),
        mean_rescale_sf=_optional_int(layer_field_values["mean_rescale_sf"]),
        square_rescale_sf=_optional_int(layer_field_values["square_rescale_sf"]),
        var_rescale_sf=_optional_int(layer_field_values["var_rescale_sf"]),
        output_truncation_k=(None if is_first_layer else int(layer_field_values["output_truncation_k"])),
    )
    return a


def _build_block2_action(
        layer_idx: int,
        layer_field_values: Dict[str, object],
        ) -> Block2ActionSpec:
    a = Block2ActionSpec(
        inv_std_fresh_sf=int(layer_field_values["inv_std_fresh_sf"]),
        x_centered_fresh_sf=int(layer_field_values["x_centered_fresh_sf"]),
        gamma_sf=int(layer_field_values["gamma_sf"]),
        wk_sf=int(layer_field_values["wk_sf"]),
        kt_mask1_sf=int(layer_field_values["kt_mask1_sf"]),
        kt_mask2_sf=int(layer_field_values["kt_mask2_sf"]),
        wq_sf=int(layer_field_values["wq_sf"]),
        q_mask1_sf=int(layer_field_values["q_mask1_sf"]),
        q_mask2_sf=int(layer_field_values["q_mask2_sf"]),
        wv_sf=int(layer_field_values["wv_sf"]),
        qkt_merge_mask_sf=int(layer_field_values["qkt_merge_mask_sf"]),
        normalize_rescale_sf=_optional_int(layer_field_values["normalize_rescale_sf"]),
        gamma_rescale_sf=_optional_int(layer_field_values["gamma_rescale_sf"]),
        wk_rescale_sf=_optional_int(layer_field_values["wk_rescale_sf"]),
        kt_mask1_rescale_sf=_optional_int(layer_field_values["kt_mask1_rescale_sf"]),
        kt_mask2_rescale_sf=_optional_int(layer_field_values["kt_mask2_rescale_sf"]),
        wq_rescale_sf=_optional_int(layer_field_values["wq_rescale_sf"]),
        q_mask1_rescale_sf=_optional_int(layer_field_values["q_mask1_rescale_sf"]),
        q_mask2_rescale_sf=_optional_int(layer_field_values["q_mask2_rescale_sf"]),
        wv_rescale_sf=_optional_int(layer_field_values["wv_rescale_sf"]),
        qkt_matmul_rescale_sf=_optional_int(layer_field_values["qkt_matmul_rescale_sf"]),
        qkt_merge_mask_rescale_sf=_optional_int(layer_field_values["qkt_merge_mask_rescale_sf"]),
        output_truncation_k=int(layer_field_values["output_truncation_k"]),
    )
    return a


def _build_block3_action(
        layer_idx: int,
        layer_field_values: Dict[str, object],
        attn_degree: int,
        ) -> Block3ActionSpec:
    deg = int(attn_degree)
    if deg < 1:
        deg = 1
    if deg > 6:
        deg = 6
    sq_keys = ("square_rescale_sf_0", "square_rescale_sf_1", "square_rescale_sf_2", "square_rescale_sf_3")
    square_rescale_base = [_optional_int(layer_field_values[key]) for key in sq_keys]
    if deg <= len(square_rescale_base):
        square_rescale_sfs = tuple(square_rescale_base[:deg])
    else:
        # The historical action vector exposes four square-rescale slots.
        # Degree-5/6 softmax configs reuse the last slot for the extra powers.
        square_rescale_sfs = tuple(
            square_rescale_base + [square_rescale_base[-1]] * (deg - len(square_rescale_base))
        )
    return Block3ActionSpec(
        degree=deg,
        x_fresh_sf=int(layer_field_values["x_fresh_sf"]),
        inv_2n_sf=int(layer_field_values["inv_2n_sf"]),
        x_inv_2n_rescale_sf=_optional_int(layer_field_values["x_inv_2n_rescale_sf"]),
        square_rescale_sfs=square_rescale_sfs,
        output_truncation_k=int(layer_field_values["output_truncation_k"]),
    )


def _build_block4_action(
        layer_idx: int,
        layer_field_values: Dict[str, object],
        ) -> Block4ActionSpec:
    return Block4ActionSpec(
        softmax_out_fresh_sf=int(layer_field_values["softmax_out_fresh_sf"]),
        softmax_out_mask_sf=int(layer_field_values["softmax_out_mask_sf"]),
        v_fresh_sf=int(layer_field_values["v_fresh_sf"]),
        v_mask_sf=int(layer_field_values["v_mask_sf"]),
        softmax_v_mask_sf=int(layer_field_values["softmax_v_mask_sf"]),
        wo_sf=int(layer_field_values["wo_sf"]),
        ln_mean_inv_d_sf=int(layer_field_values["ln_mean_inv_d_sf"]),
        ln_var_inv_d_sf=int(layer_field_values["ln_var_inv_d_sf"]),
        softmax_out_mask_rescale_sf=_optional_int(layer_field_values["softmax_out_mask_rescale_sf"]),
        v_mask_rescale_sf=_optional_int(layer_field_values["v_mask_rescale_sf"]),
        softmax_v_matmul_rescale_sf=_optional_int(layer_field_values["softmax_v_matmul_rescale_sf"]),
        softmax_v_mask_rescale_sf=_optional_int(layer_field_values["softmax_v_mask_rescale_sf"]),
        wo_rescale_sf=_optional_int(layer_field_values["wo_rescale_sf"]),
        ln_mean_rescale_sf=_optional_int(layer_field_values["ln_mean_rescale_sf"]),
        ln_square_rescale_sf=_optional_int(layer_field_values["ln_square_rescale_sf"]),
        ln_var_rescale_sf=_optional_int(layer_field_values["ln_var_rescale_sf"]),
        output_truncation_k=int(layer_field_values["output_truncation_k"]),
    )


def _build_block5_action(
        layer_idx: int,
        layer_field_values: Dict[str, object],
        gelu_degree: int,
        ) -> Block5ActionSpec:
    deg = int(gelu_degree)
    # block5 GELU degree 仅支持 {1, 2, 4}
    if deg not in (1, 2, 4):
        deg = 4 if deg >= 4 else (2 if deg >= 2 else 1)
    power_n = max(0, deg - 1)
    power_keys = ("gelu_power_rescale_sf_0", "gelu_power_rescale_sf_1", "gelu_power_rescale_sf_2")
    gelu_power_rescale_sfs = tuple(
        _optional_int(layer_field_values[power_keys[k]]) for k in range(power_n)
    )
    coeff_keys = (
        "gelu_coeff_mul_rescale_sf_0", "gelu_coeff_mul_rescale_sf_1",
        "gelu_coeff_mul_rescale_sf_2", "gelu_coeff_mul_rescale_sf_3",
    )
    gelu_coeff_mul_rescale_sfs = tuple(
        _optional_int(layer_field_values[coeff_keys[k]]) for k in range(deg)
    )
    return Block5ActionSpec(
        gelu_degree=deg,
        inv_std_fresh_sf=int(layer_field_values["inv_std_fresh_sf"]),
        x_centered_fresh_sf=int(layer_field_values["x_centered_fresh_sf"]),
        gamma_sf=int(layer_field_values["gamma_sf"]),
        wffn1_sf=int(layer_field_values["wffn1_sf"]),
        gelu_coeff_sf=int(layer_field_values["gelu_coeff_sf"]),
        normalize_rescale_sf=_optional_int(layer_field_values["normalize_rescale_sf"]),
        gamma_rescale_sf=_optional_int(layer_field_values["gamma_rescale_sf"]),
        wffn1_rescale_sf=_optional_int(layer_field_values["wffn1_rescale_sf"]),
        gelu_power_rescale_sfs=gelu_power_rescale_sfs,
        gelu_coeff_mul_rescale_sfs=gelu_coeff_mul_rescale_sfs,
        output_truncation_k=int(layer_field_values["output_truncation_k"]),
    )


def _decode_block_field_values(
        layer_idx: int,
        block_idx: int,
        action_slice: np.ndarray,
        max_sfs: MaxSFsTable,
        attn_degree: int,
        gelu_degree: int,
        ) -> Dict[str, object]:
    """把单 block 的 action 子段 ↦ ``{field_name: value}`` 字典。

    SF 字段：``sf_from(idx, max_sf, levels)`` 反推；K 字段：``K_LEVELS[idx]``。
    """
    spec = _BLOCK_SPECS[int(block_idx)]
    out: Dict[str, object] = {}
    if action_slice.shape[0] != len(spec.fields):
        raise ValueError(
            f"layer {layer_idx} block {block_idx} expects {len(spec.fields)} dims, "
            f"got {action_slice.shape[0]}"
        )
    for slot_idx, (fname, kind, default_max_sf) in enumerate(spec.fields):
        idx_val = int(action_slice[slot_idx])
        if kind == "K":
            out[fname] = int(K_LEVELS[idx_val])
            continue
        max_sf = max_sfs.get(int(block_idx), fname, layer_idx=int(layer_idx))
        N = _block_default_N(int(block_idx), gelu_degree=gelu_degree, attn_degree=attn_degree)
        if kind == "R":
            sf = _rescale_sf_from_index(idx_val, max_sf)
            out[fname] = None if sf is None else int(_snap_to_table(sf, N))
        else:
            levels = NUM_LEVELS_PER_DIM_BY_BLOCK_KIND[kind]
            sf = sf_from(idx_val, max_sf, levels)
            out[fname] = int(_snap_to_table(sf, N))
    return out


def _decode_first_input_sf(
        action_value: int,
        max_sfs: MaxSFsTable,
        ) -> int:
    """first_input fresh SF：与 fresh 同语义（5 挡）。"""
    # 没有专门的节点表，沿用 block1 fresh 默认 max=30
    max_sf = 30
    sf = sf_from(int(action_value), max_sf, LEVELS_FIRST_INPUT)
    return _snap_to_table(int(sf), BLB_FIRST_INPUT_N)


def _degree_for_layer(
        degrees: object,
        layer_idx: int,
        num_layers: int,
        *,
        default: int,
        name: str,
        ) -> int:
    if degrees is None:
        return int(default)
    if isinstance(degrees, np.ndarray):
        arr = np.asarray(degrees, dtype=int).reshape(-1)
    elif isinstance(degrees, (list, tuple)):
        arr = np.asarray(degrees, dtype=int).reshape(-1)
    else:
        return int(degrees)
    if arr.size == 0:
        return int(default)
    if arr.size == 1:
        return int(arr[0])
    if arr.size != int(num_layers):
        raise ValueError(
            f"{name} length {arr.size} must be 1 or num_layers={int(num_layers)}"
        )
    return int(arr[int(layer_idx)])


def action_vector_to_cfgs(
        action_vec: np.ndarray,
        max_sfs: MaxSFsTable,
        num_layers: int,
        gelu_degree: object = 4,
        attn_degree: object = 4,
        ) -> ActionDecodeResult:
    """``MultiDiscrete`` 风格动作向量 → 每层 5 个 BLB block cfg + first_input SF。

    Args:
        action_vec:    1D ndarray，长度 == ``sum(action_dims_for_config(num_layers))``
        max_sfs:       ``load_max_sfs(profile)`` 加载的 max SF 表
        num_layers:    模型层数 L
        gelu_degree:   Block 5 GELU 多项式 degree (1/2/4)；首层并不影响（每层独立）
        attn_degree:   Block 3 softmax 多项式 degree (1..6)

    Returns:
        ``ActionDecodeResult``
    """
    arr = np.asarray(action_vec, dtype=int).reshape(-1)

    expected_dim = len(action_dims_for_config(num_layers))
    if arr.size != expected_dim:
        raise ValueError(
            f"action_vec length {arr.size} != expected {expected_dim} (num_layers={num_layers})"
        )

    layer_dim_list = layer_dims()
    layer_dim = len(layer_dim_list)

    block1_cfgs: Dict[int, Block1NoiseConfig] = {}
    block2_cfgs: Dict[int, Block2NoiseConfig] = {}
    block3_cfgs: Dict[int, Block3NoiseConfig] = {}
    block4_cfgs: Dict[int, Block4NoiseConfig] = {}
    block5_cfgs: Dict[int, Block5NoiseConfig] = {}
    per_layer_values: List[Dict[str, object]] = []

    for li in range(int(num_layers)):
        li_gelu_degree = _degree_for_layer(
            gelu_degree,
            li,
            num_layers,
            default=4,
            name="gelu_degree",
        )
        li_attn_degree = _degree_for_layer(
            attn_degree,
            li,
            num_layers,
            default=4,
            name="attn_degree",
        )
        slice_start = li * layer_dim
        slice_end = slice_start + layer_dim
        layer_action = arr[slice_start:slice_end]

        # 切出每个 block 的 action 子段
        offset = 0
        layer_block_values: Dict[int, Dict[str, object]] = {}
        for b in (1, 2, 3, 4, 5):
            spec = _BLOCK_SPECS[b]
            slot_count = len(spec.fields)
            sub = layer_action[offset:offset + slot_count]
            offset += slot_count
            layer_block_values[b] = _decode_block_field_values(
                layer_idx=li,
                block_idx=b,
                action_slice=sub,
                max_sfs=max_sfs,
                attn_degree=li_attn_degree,
                gelu_degree=li_gelu_degree,
            )
        per_layer_values.append({f"block{b}": dict(v) for b, v in layer_block_values.items()})

        # Block 1：首层 block1 不安装（用户语义：layer 0 没有上游 FFN2，第一个 HE
        # 配置无损 —— 与 Rescale_optimizer 对齐）。保留 action 向量槽位但不构造 cfg，
        # 这样下游 (bridge.apply / build_optimizer_requests) 自然跳过 layer 0 block1。
        if li == 0:
            pass  # block1_cfgs 故意不写 layer 0，下游用 .get(0) / dict-not-in 判断
        else:
            b1 = _build_block1_action(li, layer_block_values[1], is_first_layer=False)
            block1_cfgs[li] = build_block1_cfg_from_action(
                b1, N=_block_default_N(1, gelu_degree=li_gelu_degree, attn_degree=li_attn_degree),
            )
        # Block 2
        b2 = _build_block2_action(li, layer_block_values[2])
        block2_cfgs[li] = build_block2_cfg_from_action(
            b2, N=_block_default_N(2, gelu_degree=li_gelu_degree, attn_degree=li_attn_degree),
        )
        # Block 3 (degree-aware)
        b3 = _build_block3_action(li, layer_block_values[3], attn_degree=li_attn_degree)
        block3_cfgs[li] = build_block3_cfg_from_action(
            b3, N=_block_default_N(3, gelu_degree=li_gelu_degree, attn_degree=li_attn_degree),
        )
        # Block 4
        b4 = _build_block4_action(li, layer_block_values[4])
        block4_cfgs[li] = build_block4_cfg_from_action(
            b4, N=_block_default_N(4, gelu_degree=li_gelu_degree, attn_degree=li_attn_degree),
        )
        # Block 5 (gelu degree-aware)
        b5 = _build_block5_action(li, layer_block_values[5], gelu_degree=li_gelu_degree)
        block5_cfgs[li] = build_block5_cfg_from_action(
            b5, N=_block_default_N(5, gelu_degree=li_gelu_degree, attn_degree=li_attn_degree),
        )

    # 尾部 first_input_sf：语义已废弃（"第一个 HE 配置无损"，不再注入 layer 0
    # input 端的 fresh 噪声）。保留槽位以维持 policy 网络 shape 与旧 checkpoint
    # 兼容；下游 bridge.apply 完全忽略此字段。``first_input_sf`` 始终回 0 作占位。
    return ActionDecodeResult(
        block1_cfgs=block1_cfgs,
        block2_cfgs=block2_cfgs,
        block3_cfgs=block3_cfgs,
        block4_cfgs=block4_cfgs,
        block5_cfgs=block5_cfgs,
        first_input_sf=0,
        per_layer_field_values=per_layer_values,
    )


def _action_distribution_for_kind(kind: str) -> str:
    return {
        "F": "fresh",
        "W": "encoding",
        "M": "encoding",
        "S": "scalar",
        "R": "rescale",
        "K": "truncation",
    }.get(str(kind), str(kind))


def _action_value_type_for_kind(kind: str) -> str:
    return "truncation_k" if str(kind) == "K" else "scaling_factor"


def _field_level_values(
        *,
        kind: str,
        levels: int,
        max_sf: Optional[int],
        N: int,
        ) -> List[object]:
    if str(kind) == "K":
        return [int(v) for v in K_LEVELS]
    if str(kind) == "R":
        return [
            (
                None if _rescale_sf_from_index(idx, int(max_sf)) is None
                else int(_snap_to_table(_rescale_sf_from_index(idx, int(max_sf)), int(N)))
            )
            for idx in range(int(levels))
        ]
    return [
        int(_snap_to_table(sf_from(idx, int(max_sf), int(levels)), int(N)))
        for idx in range(int(levels))
    ]


def _operation_name(block_idx: int, field_name: str, kind: str) -> str:
    if str(kind) == "K":
        return f"block{int(block_idx)}_output_truncation"
    return _BLOCK_NODE_NAME_BY_FIELD.get(int(block_idx), {}).get(
        str(field_name),
        f"block{int(block_idx)}_{field_name}",
    )


def _short_field_label(field_name: str, kind: str) -> str:
    """Compact field tag for ``slot_label`` (logs / dashboards).

    Goal: minimum-character disambiguation while staying derivable from the
    full field name. See CLAUDE.md "Critical mental model" #3 — every action
    must be locatable in BLB flow without consulting an extra table.
    """
    if str(kind) == "K":
        return ""  # the "K" kind in the label itself already carries this
    f = str(field_name)
    if f.startswith("square_rescale_sf_"):
        return "sq" + f.rsplit("_", 1)[-1]            # block3 softmax square rescales
    if f.startswith("gelu_power_rescale_sf_"):
        return "gp" + f.rsplit("_", 1)[-1]            # block5 GELU power rescales
    if f.startswith("gelu_coeff_mul_rescale_sf_"):
        return "gc" + f.rsplit("_", 1)[-1]            # block5 GELU coeff-mul rescales
    if f.endswith("_rescale_sf"):
        return f[: -len("_rescale_sf")] + "_r"
    if f.endswith("_sf"):
        return f[: -len("_sf")]
    return f


def make_slot_label(
        layer_idx: int,
        block_idx: Optional[int],
        kind: str,
        field_name: str,
        ) -> str:
    """Compact ``L{i}.B{n}.{kind}[.{short}]`` label for a single action slot.

    Examples:
      ``L0.B1.F.gelu_out`` — layer 0, block 1, fresh slot for GELU output
      ``L3.B5.R.gp2``      — layer 3, block 5, GELU power-rescale slot 2
      ``L11.B4.K``         — layer 11, block 4, output truncation k
      ``L0.first_input.F`` — layer-0 first-input fresh (not in any block)
    """
    short = _short_field_label(field_name, kind)
    if block_idx is None:
        # First-input fresh: outside all 5 blocks.
        return f"L{int(layer_idx)}.first_input.{str(kind)}"
    base = f"L{int(layer_idx)}.B{int(block_idx)}.{str(kind)}"
    return base if not short else f"{base}.{short}"


def _is_action_field_effective(
        *,
        layer_idx: int,
        block_idx: int,
        field_name: str,
        attn_degree: int,
        gelu_degree: int,
        ) -> Tuple[bool, str]:
    # Layer 0 has no upstream FFN2 → block 1 noise is *not* installed at all,
    # even though the action vector reserves slots for it (so the policy net
    # shape stays uniform across layers). Mark every block-1 slot at layer 0
    # as ineffective so logs / candidate descriptions reflect reality, and so
    # downstream tooling (bridge.apply, build_optimizer_requests) can filter.
    if int(layer_idx) == 0 and int(block_idx) == 1:
        return False, (
            "layer 0 has no upstream FFN2; the first HE config is treated as "
            "lossless so block1 noise is not installed (aligned with Rescale_optimizer)"
        )
    if int(block_idx) == 3 and str(field_name).startswith("square_rescale_sf_"):
        try:
            slot = int(str(field_name).rsplit("_", 1)[-1])
        except Exception:
            slot = 0
        if slot >= max(1, int(attn_degree)):
            return False, f"softmax degree {int(attn_degree)} does not use this square-rescale slot"
    if int(block_idx) == 5 and str(field_name).startswith("gelu_power_rescale_sf_"):
        try:
            slot = int(str(field_name).rsplit("_", 1)[-1])
        except Exception:
            slot = 0
        if slot >= max(0, int(gelu_degree) - 1):
            return False, f"GELU degree {int(gelu_degree)} does not use this power-rescale slot"
    if int(block_idx) == 5 and str(field_name).startswith("gelu_coeff_mul_rescale_sf_"):
        try:
            slot = int(str(field_name).rsplit("_", 1)[-1])
        except Exception:
            slot = 0
        if slot >= int(gelu_degree):
            return False, f"GELU degree {int(gelu_degree)} does not use this coefficient-rescale slot"
    return True, ""


def describe_action_vector(
        action_vec: np.ndarray,
        *,
        max_sfs: MaxSFsTable,
        num_layers: int,
        gelu_degree: object = 4,
        attn_degree: object = 4,
        profile: str = "default",
        ) -> Dict[str, Any]:
    """Return a readable per-action-slot description for logs and artifacts.

    The returned records preserve the exact global action index while naming
    the model location, BLB block, field, operation/noise point, action index,
    decoded value, scaling-factor table N, and whether the slot is effective
    for the layer's polynomial degree.
    """
    arr = np.asarray(action_vec, dtype=int).reshape(-1)
    expected_dim = len(action_dims_for_config(num_layers))
    if arr.size != expected_dim:
        raise ValueError(
            f"action_vec length {arr.size} != expected {expected_dim} (num_layers={num_layers})"
        )

    fields = per_layer_field_offsets()
    layer_dim = len(fields)
    records: List[Dict[str, Any]] = []
    for li in range(int(num_layers)):
        li_gelu_degree = _degree_for_layer(
            gelu_degree, li, num_layers, default=4, name="gelu_degree",
        )
        li_attn_degree = _degree_for_layer(
            attn_degree, li, num_layers, default=4, name="attn_degree",
        )
        for field_offset, (block_idx, field_name, kind) in enumerate(fields):
            global_index = int(li * layer_dim + field_offset)
            action_index = int(arr[global_index])
            levels = int(NUM_LEVELS_PER_DIM_BY_BLOCK_KIND[kind])
            N = int(_block_default_N(block_idx, gelu_degree=li_gelu_degree, attn_degree=li_attn_degree))
            max_sf = None
            if kind == "K":
                value = int(K_LEVELS[action_index])
            elif kind == "R":
                max_sf = int(max_sfs.get(block_idx, field_name, layer_idx=li))
                raw_value = _rescale_sf_from_index(action_index, max_sf)
                value = None if raw_value is None else int(_snap_to_table(raw_value, N))
            else:
                max_sf = int(max_sfs.get(block_idx, field_name, layer_idx=li))
                value = int(_snap_to_table(sf_from(action_index, max_sf, levels), N))
            effective, note = _is_action_field_effective(
                layer_idx=li,
                block_idx=block_idx,
                field_name=field_name,
                attn_degree=li_attn_degree,
                gelu_degree=li_gelu_degree,
            )
            effective_value = value if effective else None
            operation = _operation_name(block_idx, field_name, kind)
            block_label = f"layer{li}.block{block_idx}"
            record = {
                "global_index": global_index,
                "layer": int(li),
                "layer_label": f"layer{li}",
                "block": f"block{int(block_idx)}",
                "block_index": int(block_idx),
                "block_label": block_label,
                "field": str(field_name),
                "kind": str(kind),
                "distribution": _action_distribution_for_kind(kind),
                "operation": operation,
                "location": f"{block_label}.{field_name}",
                "slot_label": make_slot_label(li, block_idx, kind, field_name),
                "config_name": make_config_name(str(profile), block_idx, li),
                "action_index": action_index,
                "num_levels": levels,
                "level_values": _field_level_values(kind=kind, levels=levels, max_sf=max_sf, N=N),
                "value_type": _action_value_type_for_kind(kind),
                "value": value,
                "effective": bool(effective),
                "effective_value": effective_value,
                "N": N,
                "max_sf": max_sf,
                "gelu_degree": int(li_gelu_degree),
                "attn_degree": int(li_attn_degree),
            }
            if note:
                record["note"] = note
            records.append(record)

    first_idx = int(arr[-1])
    first_value = int(_decode_first_input_sf(first_idx, max_sfs))
    # NOTE: first_input fresh 噪声在新语义下不再注入（"第一个 HE 配置无损"）。
    # 保留 slot 描述方便审阅旧 checkpoint / candidate；effective=False 表明它
    # 不影响 cost / 模型 forward。
    records.append({
        "global_index": int(arr.size - 1),
        "layer": 0,
        "layer_label": "layer0",
        "block": "first_input",
        "block_index": None,
        "block_label": "first_input",
        "field": "first_input_sf",
        "kind": "F",
        "distribution": "fresh",
        "operation": "first_input_fresh",
        "location": "first_input.layer0",
        "slot_label": make_slot_label(0, None, "F", "first_input_sf"),
        "config_name": "first_input_L0",
        "action_index": first_idx,
        "num_levels": int(LEVELS_FIRST_INPUT),
        "level_values": _field_level_values(
            kind="F", levels=LEVELS_FIRST_INPUT, max_sf=30, N=BLB_FIRST_INPUT_N,
        ),
        "value_type": "scaling_factor",
        "value": first_value,
        "effective": False,
        "effective_value": None,
        "N": int(BLB_FIRST_INPUT_N),
        "max_sf": 30,
        "gelu_degree": None,
        "attn_degree": None,
        "note": (
            "first_input fresh noise deprecated; the first HE config is treated "
            "as lossless. Slot kept for action-vector backward compatibility."
        ),
    })

    truncation_count = sum(1 for r in records if r.get("value_type") == "truncation_k")
    sf_count = sum(1 for r in records if r.get("value_type") == "scaling_factor")
    ineffective_count = sum(1 for r in records if not r.get("effective", True))
    return {
        "schema": "blb_action_description_v1",
        "profile": str(profile),
        "num_layers": int(num_layers),
        "action_length": int(arr.size),
        "k_levels": [int(v) for v in K_LEVELS],
        "first_input_N": int(BLB_FIRST_INPUT_N),
        "summary": {
            "record_count": int(len(records)),
            "scaling_factor_count": int(sf_count),
            "truncation_count": int(truncation_count),
            "ineffective_slot_count": int(ineffective_count),
        },
        "records": records,
    }


# ---------------------------------------------------------------------------
# 全 max-action / 全 min-action helper（baseline / sanity）
# ---------------------------------------------------------------------------
def make_all_max_action_vector(num_layers: int) -> np.ndarray:
    """生成"全 max" 动作向量：SF 字段取最高档，K 取数值最大的 truncation。

    用于 §6.3 的 baseline + §11 验证清单中"action 全 max → reward = 0"。
    """
    dims = action_dims_for_config(int(num_layers))
    arr = np.array(dims, dtype=int) - 1
    k_idx = int(K_LEVELS.index(max(K_LEVELS)))
    fields = per_layer_field_offsets()
    layer_dim = len(fields)
    for li in range(int(num_layers)):
        for field_offset, (_block_idx, _field_name, kind) in enumerate(fields):
            if kind == "K":
                arr[li * layer_dim + field_offset] = k_idx
    return arr


def make_all_min_action_vector(num_layers: int) -> np.ndarray:
    """生成"全 min" 动作向量：每个分量取 0。"""
    dims = action_dims_for_config(int(num_layers))
    return np.zeros(len(dims), dtype=int)


def avg_truncation_k_in_action(
        action_vec: np.ndarray,
        num_layers: int,
        ) -> float:
    """从 action 向量里抽出每个 block 的 K 选择，返回平均 k。

    用于 reward 中的 ``k_drop = baseline.avg_k - avg_k``。
    """
    arr = np.asarray(action_vec, dtype=int).reshape(-1)
    dims_list = layer_dims()
    layer_dim = len(dims_list)
    # 单层 block_dims + cumulative offset 构造 K 槽位的全局位置
    k_positions: List[int] = []
    cursor_in_layer = 0
    for b in (1, 2, 3, 4, 5):
        spec = _BLOCK_SPECS[b]
        # K 字段必为 spec.fields 最后一个
        for fname, kind, _max in spec.fields:
            if kind == "K":
                k_positions.append(cursor_in_layer)
            cursor_in_layer += 1
    # 遍历每层
    ks: List[float] = []
    for li in range(int(num_layers)):
        # 首层 Block 1 K 被强制 None；其它 block 的 K 仍生效
        for j, p in enumerate(k_positions):
            if li == 0 and j == 0:
                continue
            slot = li * layer_dim + p
            idx = int(arr[slot])
            ks.append(float(K_LEVELS[idx]))
    if not ks:
        return 0.0
    return float(np.mean(ks))


# ---------------------------------------------------------------------------
# config_name <-> (block_idx, layer_idx) 编解码
# ---------------------------------------------------------------------------
def make_config_name(profile: str, block_idx: int, layer_idx: int, cfg: object = None) -> str:
    """Build the layered Rescale_optimizer config key.

    Block 1/2 are dataset-profile graphs (``block1_mrpc``), while Block 3/5
    are degree-profile graphs (``block3_exp_n4`` / ``block5_n4``) and Block 4
    is shared as ``block4``. The ``_L<i>`` suffix is kept for per-layer
    accounting; ``RescaleOptimizerBridge`` strips it before invoking the
    underlying optimizer.
    """
    block_idx = int(block_idx)
    if block_idx in (1, 2):
        graph_key = f"block{block_idx}_{str(profile)}"
    elif block_idx == 3:
        degree = int(getattr(cfg, "degree", 4) or 4)
        graph_key = f"block3_exp_n{degree}"
    elif block_idx == 4:
        graph_key = "block4"
    elif block_idx == 5:
        degree = int(getattr(cfg, "gelu_degree", 4) or 4)
        graph_key = f"block5_n{degree}"
    else:
        graph_key = f"block{block_idx}_{str(profile)}"
    return f"{graph_key}_L{int(layer_idx)}"


def parse_config_name(config_name: str) -> Tuple[int, str, int]:
    """``"blockN_<profile>_L<i>"`` → ``(N, profile, i)``。"""
    name = str(config_name)
    if not name.startswith("block"):
        raise ValueError(f"unexpected config_name: {name}")
    rest = name[len("block"):]
    block_str, sep, tail = rest.partition("_")
    block_idx = int(block_str)
    if not tail:
        raise ValueError(f"unexpected config_name: {name}")
    if tail.startswith("L") and tail[1:].isdigit():
        return block_idx, "", int(tail[1:])
    if "_L" not in tail:
        return block_idx, tail, -1
    profile, _, layer_str = tail.rpartition("_L")
    return block_idx, profile, int(layer_str)


def build_optimizer_requests(
        profile: str,
        cfgs_dict: Mapping[str, Mapping[int, object]],
        ) -> Dict[str, Tuple[str, object]]:
    """``cfgs_dict["block1"][i]`` → ``{config_name: (block_name, cfg)}``。

    NOTE: 不会发送 ``(block=1, layer=0)`` 给 Rescale_optimizer —— 该位置
    的 block1 噪声整体不安装（语义：layer 0 没有上游 FFN2，第一个 HE 配置
    无损）。``action_vector_to_cfgs`` 也不会把 layer 0 写入 ``block1_cfgs``，
    所以这里的过滤是双保险。
    """
    out: Dict[str, Tuple[str, object]] = {}
    for block_name, layer_cfgs in cfgs_dict.items():
        if not str(block_name).startswith("block"):
            continue
        try:
            block_idx = int(str(block_name)[5:])
        except ValueError:
            continue
        for layer_idx, cfg in layer_cfgs.items():
            if int(block_idx) == 1 and int(layer_idx) == 0:
                # 语义对齐：layer-0 block1 不发给 RO。
                continue
            cn = make_config_name(profile, block_idx, int(layer_idx), cfg=cfg)
            out[cn] = (str(block_name), cfg)
    return out
