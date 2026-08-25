"""Full-vector Stage-2 action schema and model-config materialization.

The layerwise policy projects its compact fusion/precision matrix into this
stable vector. Inactive compatibility slots keep fixed offsets for persisted
artifacts but are never installed or charged by the optimizer.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from rfr.common.json_utils import read_json_file
from rfr.preparation.fusion.count_map import FUSION_CONFIG_ROOT
from rfr.preparation.rescale import RESCALE_CONFIG_ROOT
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

from .layerwise_action import truncation_k_summary_from_full_action
from .truncation_levels import K_LEVELS, LEVELS_K, baseline_k_index


LEVELS_W = 15
LEVELS_MS = 15
LEVELS_R = 15
LEVELS_F = 15

LEVELS_FIRST_INPUT = 5
BLB_FIRST_INPUT_N = 8192


BASELINE_K_BY_BLOCK: Dict[int, int] = {1: 13, 2: 13, 3: 13, 4: 13, 5: 13}


def _baseline_k_index_for_block(block_idx: int) -> int:
    """Map a block's baseline K value to the fixed action index."""
    target = int(BASELINE_K_BY_BLOCK.get(int(block_idx), max(K_LEVELS)))
    return int(baseline_k_index(K_LEVELS, baseline_k=target))


NUM_LEVELS_PER_DIM_BY_BLOCK_KIND = {
    "F": LEVELS_F,
    "W": LEVELS_W,
    "M": LEVELS_MS,
    "S": LEVELS_MS,
    "R": LEVELS_R,
    "K": LEVELS_K,
}


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


_BLOCK1_FIELDS = _BlockFieldSpec(
    fields=(
        ("gelu_out_sf",        "F", 30),
        ("wffn2_sf",           "W", 22),
        ("mean_inv_d_sf",      "S", 22),
        ("var_inv_d_sf",       "S", 22),
        ("mean_rescale_sf",    "R", 22),
        ("var_rescale_sf",     "R", 22),

        ("wffn2_rescale_sf",   "R", 22),
        ("square_rescale_sf",  "R", 22),
        ("output_truncation_k","K", 13),
    ),
)


_BLOCK2_FIELDS = _BlockFieldSpec(
    fields=(
        ("inv_std_fresh_sf",            "F", 30),
        ("x_centered_fresh_sf",         "F", 30),
        ("gamma_sf",                    "M", 22),
        ("wk_sf",                       "W", 22),
        ("wv_sf",                       "W", 22),
        ("kt_mask1_sf",                 "M", 22),
        ("kt_mask2_sf",                 "M", 22),
        ("qkt_merge_mask_sf",           "M", 22),
        ("gamma_rescale_sf",            "R", 22),
        ("kt_mask2_rescale_sf",         "R", 22),
        ("qkt_merge_mask_rescale_sf",   "R", 22),

        ("wq_sf",                       "W", 22),
        ("q_mask1_sf",                  "M", 22),
        ("q_mask2_sf",                  "M", 22),

        ("normalize_rescale_sf",        "R", 22),
        ("wk_rescale_sf",               "R", 22),
        ("wq_rescale_sf",               "R", 22),
        ("wv_rescale_sf",               "R", 22),
        ("kt_mask1_rescale_sf",         "R", 22),
        ("q_mask1_rescale_sf",          "R", 22),
        ("q_mask2_rescale_sf",          "R", 22),
        ("qkt_matmul_rescale_sf",       "R", 22),
        ("output_truncation_k",         "K", 13),
    ),
)


_BLOCK3_R_SLOTS = 4
_BLOCK3_FIELDS = _BlockFieldSpec(
    fields=(
        ("x_fresh_sf",              "F", 30),
        ("inv_2n_sf",               "S", 22),
        ("square_rescale_sf_0",     "R", 22),
        ("square_rescale_sf_1",     "R", 22),
        ("square_rescale_sf_2",     "R", 22),
        ("square_rescale_sf_3",     "R", 22),

        ("x_inv_2n_rescale_sf",     "R", 22),
        ("output_truncation_k",     "K", 13),
    ),
)


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
        ("softmax_v_matmul_rescale_sf",     "R", 22),
        ("ln_mean_rescale_sf",              "R", 22),
        ("ln_var_rescale_sf",               "R", 22),

        ("softmax_out_mask_rescale_sf",     "R", 22),
        ("v_mask_rescale_sf",               "R", 22),
        ("softmax_v_mask_rescale_sf",       "R", 22),
        ("wo_rescale_sf",                   "R", 22),
        ("ln_square_rescale_sf",            "R", 22),
        ("output_truncation_k",             "K", 13),
    ),
)


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
        ("gelu_power_rescale_sf_0",         "R", 22),

        ("gelu_power_rescale_sf_1",         "R", 22),
        ("gelu_power_rescale_sf_2",         "R", 22),
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


_BLOCK_NODE_NAME_BY_FIELD: Dict[int, Dict[str, str]] = {
    1: {
        "gelu_out_sf":          "ctpt_gelu_out",
        "wffn2_sf":             "ctpt_ffn2",
        "mean_inv_d_sf":        "ctpt_inv_d_1",
        "var_inv_d_sf":         "ctpt_inv_d_2",
        "mean_rescale_sf":      "ctct_mean_rescale",
        "var_rescale_sf":       "ctct_var_rescale",
    },
    2: {
        "inv_std_fresh_sf":            "ctpt_inv_std",
        "x_centered_fresh_sf":         "ctpt_x_centered",
        "gamma_sf":                    "ctpt_gamma",

        "wk_sf":                       "ctpt_wq_wk",


        "wq_sf":                       "ctpt_wq_wk",
        "q_mask1_sf":                  "ctpt_rotKT_mask1",
        "q_mask2_sf":                  "ctpt_rotKT_mask2",
        "q_mask2_rescale_sf":          "ctct_kt_mask2_rescale",

        "wv_sf":                       "ctpt_wv",
        "kt_mask1_sf":                 "ctpt_rotKT_mask1",
        "kt_mask2_sf":                 "ctpt_rotKT_mask2",
        "qkt_merge_mask_sf":           "ctpt_mask",
        "gamma_rescale_sf":            "ctct_gamma_rescale",
        "kt_mask2_rescale_sf":         "ctct_kt_mask2_rescale",
        "qkt_merge_mask_rescale_sf":   "ctct_qkt_merge_mask_rescale",
    },
    3: {
        "x_fresh_sf":               "ctpt_softmax_x",
        "inv_2n_sf":                "ctpt_softmax_inv_2n",
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
        "softmax_v_matmul_rescale_sf":  "ctct_softmax_v_matmul_rescale",
        "ln_mean_rescale_sf":           "ctct_attn_mean_rescale",
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


        "gelu_coeff_mul_rescale_sf_0":      "ctct_gelu_coeff_rescale",
    },
}


def _block_default_N(block_idx: int, gelu_degree: int = 4, attn_degree: int = 4) -> int:
    """噪声表 N（决定 noise variance 查表 + snap 范围）。

    All production action slots use the N=16384 noise table. Rescale graph N
    remains owned by the optimizer.
    """
    return 16384


def sf_from(idx: int, max_sf: int, levels: int) -> int:
    """Uniform step-1 downward sweep from ``max_sf`` (= the slot's BASELINE SF,
    not the table max).

    ``idx levels-1..0 → max_sf, -1, -2, …, -(levels-1)`` spans the full integer
    range. For the canonical
    15-level slot, baseline 30 → 30,29,28,…,16 (reach baseline-14).

    May return a value below the noise-table minimum for a low-baseline-SF slot;
    callers wrap with ``_snap_to_table`` which floors it at the table min (SF=10).
    """
    idx = int(idx)
    levels = int(levels)
    if idx < 0 or idx >= levels:
        raise ValueError(f"action idx {idx} out of [0, {levels})")
    dist = (levels - 1) - idx
    return int(max_sf) - int(dist)


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
        candidate_nodes: List[str] = []
        if node is not None:
            candidate_nodes.append(node)
        if str(field_name) not in candidate_nodes:
            candidate_nodes.append(str(field_name))
        for cand in candidate_nodes:
            if layer_idx is not None:
                v = self.by_layer_block_node.get((int(layer_idx), int(block_idx), cand))
                if v is not None:
                    return int(v)
            v = self.by_block_node.get((int(block_idx), cand))
            if v is not None:
                return int(v)

        for fname, kind, default_max_sf in _BLOCK_SPECS[int(block_idx)].fields:
            if fname == field_name:
                return int(default_max_sf)
        return 22


def load_max_sfs(profile: str, search_paths: Optional[Sequence[str]] = None) -> MaxSFsTable:
    """从 ``configs/preparation/fusion/max_sfs/<profile>.json`` 加载 max SF 表。

    JSON 结构（完全可选；缺字段或文件不存在都允许）：

        {
            "block1": {"ctpt_ffn2": 30, "ctpt_inv_d_1": 22, ...},
            "block2": {...},
            "block3": {...},
            "block4": {...},
            "block5": {...}
        }

    若想从 ``configs/preparation/rescale/<profile>/static_skeletons_<profile>.json``
    自动生成，参见 ``docs/BLB_stage2_rl_spec.md`` §4.4。
    """
    profile = str(profile or "default")
    table = MaxSFsTable(by_block_node={})

    candidates: List[str] = []
    if search_paths:
        candidates.extend(search_paths)
    candidates.append(str(FUSION_CONFIG_ROOT / "max_sfs" / f"{profile}.json"))
    candidates.append(str(FUSION_CONFIG_ROOT / "max_sfs" / "default.json"))

    for path in candidates:
        if not path:
            continue
        if not os.path.isfile(path):
            continue
        try:
            payload = read_json_file(path)
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

        break

    return table


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


@lru_cache(maxsize=None)
def _action_dims_array(num_layers: int) -> np.ndarray:
    """Return the immutable per-slot domain used by full-vector consumers."""
    dims = np.asarray(action_dims_for_config(int(num_layers)), dtype=np.int64)
    dims.setflags(write=False)
    return dims


def validate_action_vector(
        action_vec: Sequence[int] | np.ndarray,
        num_layers: int,
        ) -> np.ndarray:
    """Return a 1D integer action after lossless categorical validation."""
    original = action_vec
    if not isinstance(action_vec, np.ndarray):
        try:
            original = tuple(action_vec)
        except TypeError:
            pass
    raw = np.asarray(original)
    if raw.ndim != 1:
        raise ValueError(
            f"action_vec must be one-dimensional, got shape {raw.shape}"
        )

    dims = _action_dims_array(int(num_layers))
    if raw.size != dims.size:
        raise ValueError(
            f"action_vec length {raw.size} != expected {dims.size} "
            f"(num_layers={int(num_layers)})"
        )

    original_has_boolean = (
        any(isinstance(value, (bool, np.bool_)) for value in original)
        if isinstance(original, tuple)
        else raw.dtype == object
        and any(isinstance(value, (bool, np.bool_)) for value in raw.flat)
    )
    integer_dtype = np.issubdtype(raw.dtype, np.integer)
    boolean_dtype = np.issubdtype(raw.dtype, np.bool_) or original_has_boolean
    if not integer_dtype or boolean_dtype:
        if np.issubdtype(raw.dtype, np.floating):
            integer_valued = np.isfinite(raw) & (raw == np.trunc(raw))
        else:
            integer_valued = np.zeros(raw.shape, dtype=bool)
        non_integer_positions = np.flatnonzero(~integer_valued)
        if non_integer_positions.size:
            position = int(non_integer_positions[0])
            raise ValueError(
                "action_vec must contain integer categorical indices; "
                f"position {position}={raw[position]!r}"
            )

    invalid_positions = np.flatnonzero((raw < 0) | (raw >= dims))
    if invalid_positions.size:
        position = int(invalid_positions[0])
        raise ValueError(
            f"action index at position {position}={int(raw[position])} "
            f"out of range [0,{int(dims[position])})"
        )
    return raw.astype(np.int64, copy=False)


def per_layer_field_offsets() -> List[Tuple[int, str, str]]:
    """返回单层动作向量内每个分量的 ``(block_idx, field_name, kind)`` 三元组。"""
    out: List[Tuple[int, str, str]] = []
    for b in (1, 2, 3, 4, 5):
        for fname, kind, _max in _BLOCK_SPECS[b].fields:
            out.append((b, fname, kind))
    return out


@dataclass(frozen=True)
class BlockStepSpec:
    """Description of one (layer, block) decision step in the sequential episode.

    Attributes:
        step_idx:        0-based index within the episode (0 .. horizon-1).
        layer_idx:       0-based transformer layer.
        block_idx:       1..5.
        slot_dims:       num_levels for each slot decided at this step.
        slot_field_names: corresponding _BLOCK{N}_FIELDS field names.
        slot_kinds:      "F"/"W"/"M"/"S"/"R"/"K".
        full_vec_offsets: index into the full action vector where
                          each slot's value should be written.
        graph_key:       e.g. "block2_mrpc" -- the Rescale_optimizer graph that
                         scores this block (independent of layer).
        terminal:        True for the last step in the episode.
    """
    step_idx: int
    layer_idx: int
    block_idx: int
    slot_dims: Tuple[int, ...]
    slot_field_names: Tuple[str, ...]
    slot_kinds: Tuple[str, ...]
    full_vec_offsets: Tuple[int, ...]
    graph_key_suffix: str
    terminal: bool


_LAYER0_BLOCK_ORDER: Tuple[int, ...] = (2, 4, 5)
_LAYER_GE_1_BLOCK_ORDER: Tuple[int, ...] = (1, 2, 4, 5)

_BLOCK_GRAPH_KEY_TEMPLATE = {
    1: "block1_{profile}",
    2: "block2_{profile}",
    3: "block3_exp_n{attn_degree}",
    4: "block4",
    5: "block5_n{gelu_degree}",
}


def horizon_for_num_layers(num_layers: int) -> int:
    """Sequential episode horizon with Block 3 excluded from the schedule.

    Block 3 is no longer a decided step, so:
      - layer 0 has 3 steps (B2, B4, B5)
      - layers 1..L-1 have 4 steps each (B1, B2, B4, B5)
    giving horizon = 3 + (L-1) * 4. For L=12 this is 47 steps.
    The persisted first_input tail slot is not a decision; it remains
    frozen at the baseline placeholder and is ignored by model installation.
    """
    L = int(num_layers)
    if L < 1:
        raise ValueError(f"num_layers must be >= 1, got {L}")
    return 3 + (L - 1) * 4


def _full_vec_offset_for_block(num_layers: int, layer_idx: int, block_idx: int) -> int:
    """Return start offset (into the current full action vector) of
    the (layer_idx, block_idx) slot range.

    The full vec is one categorical index per slot per layer, in block order
    1..5; first_input fresh sits at the very end. For BERT-base, this vector
    contains 73 slots per layer and 877 entries in total.
    """
    per_layer_width = len(layer_dims())
    base = int(layer_idx) * per_layer_width
    cursor = 0
    for b in (1, 2, 3, 4, 5):
        if b == int(block_idx):
            return base + cursor
        cursor += len(block_dims(b))
    raise ValueError(f"unknown block_idx={block_idx}")


def _full_vec_first_input_offset(num_layers: int) -> int:
    """The first_input fresh slot lives at the very end of the action vector."""
    return int(num_layers) * len(layer_dims())


def step_schedule(
        num_layers: int,
        *,
        profile: str = "mrpc",
        attn_degree_per_layer: Optional[Sequence[int]] = None,
        gelu_degree_per_layer: Optional[Sequence[int]] = None,
        ) -> List[BlockStepSpec]:
    """Build the per-block decision schedule.

    Args:
        num_layers:               number of transformer layers (e.g. 12 for BERT-base).
        profile:                  Rescale_optimizer profile (e.g. "mrpc"); used to
                                  compose graph keys for blocks 1, 2, 4. Graph 3
                                  uses ``block3_exp_n{attn_degree}``, graph 5 uses
                                  ``block5_n{gelu_degree}`` and so depend on the
                                  Stage-1 degree per layer.
        attn_degree_per_layer:    Stage-1 softmax/attn polynomial degree for each
                                  layer (default 4 if None).
        gelu_degree_per_layer:    Stage-1 GELU degree per layer (default 4 if None).
    """
    L = int(num_layers)
    horizon = horizon_for_num_layers(L)
    out: List[BlockStepSpec] = []
    step_idx = 0
    for layer_idx in range(L):
        block_order = _LAYER0_BLOCK_ORDER if layer_idx == 0 else _LAYER_GE_1_BLOCK_ORDER
        for b in block_order:
            spec = _BLOCK_SPECS[b]
            slot_dims: List[int] = []
            slot_field_names: List[str] = []
            slot_kinds: List[str] = []
            full_vec_offsets: List[int] = []
            block_base = _full_vec_offset_for_block(L, layer_idx, b)
            for slot_local_idx, (fname, kind, _max) in enumerate(spec.fields):
                slot_dims.append(NUM_LEVELS_PER_DIM_BY_BLOCK_KIND[kind])
                slot_field_names.append(fname)
                slot_kinds.append(kind)
                full_vec_offsets.append(block_base + slot_local_idx)

            if b == 3:
                deg = (
                    int(attn_degree_per_layer[layer_idx])
                    if attn_degree_per_layer is not None and layer_idx < len(attn_degree_per_layer)
                    else 4
                )
                gk = f"block3_exp_n{deg}"
            elif b == 5:
                deg = (
                    int(gelu_degree_per_layer[layer_idx])
                    if gelu_degree_per_layer is not None and layer_idx < len(gelu_degree_per_layer)
                    else 4
                )
                gk = f"block5_n{deg}"
            elif b == 4:
                gk = "block4"
            else:
                gk = _BLOCK_GRAPH_KEY_TEMPLATE[b].format(profile=str(profile))
            out.append(BlockStepSpec(
                step_idx=step_idx,
                layer_idx=int(layer_idx),
                block_idx=int(b),
                slot_dims=tuple(slot_dims),
                slot_field_names=tuple(slot_field_names),
                slot_kinds=tuple(slot_kinds),
                full_vec_offsets=tuple(full_vec_offsets),
                graph_key_suffix=gk,
                terminal=(step_idx == horizon - 1),
            ))
            step_idx += 1
    if step_idx != horizon:
        raise RuntimeError(
            f"step_schedule built {step_idx} specs but horizon expected {horizon}"
        )
    return out


def step_schedule_max_dim(num_layers: int) -> int:
    """Max number of slots decided at any single step in the schedule.

    Used by the policy network to size a single shared MultiDiscrete head;
    per-step masks zero out unused slots.
    """
    sched = step_schedule(int(num_layers))
    return max(len(s.slot_dims) for s in sched)


def splice_step_action_into_full_vec(
        full_vec: np.ndarray,
        step: BlockStepSpec,
        step_action: Sequence[int],
        ) -> np.ndarray:
    """Write the per-step action's slot values into the current full action vector.

    ``step_action`` length must equal ``len(step.slot_dims)``. Returns
    ``full_vec`` for chaining.
    """
    arr = np.asarray(step_action, dtype=int).reshape(-1)
    if arr.size != len(step.full_vec_offsets):
        raise ValueError(
            f"step {step.step_idx} expects {len(step.full_vec_offsets)} slots, got {arr.size}"
        )
    for offset, val in zip(step.full_vec_offsets, arr):
        full_vec[int(offset)] = int(val)
    return full_vec


def empty_full_action_vec(num_layers: int) -> np.ndarray:
    """Zero-initialised full action vector matching ``action_dims_for_config`` width."""
    return np.zeros(len(action_dims_for_config(int(num_layers))), dtype=np.int64)


@dataclass(frozen=True)
class FusionStepSpec:
    """One (layer, block) step in the fusion-count episode.

    The per-step action is ``(policy_fusion_index, k_index)``:
      * slot 0 = policy-local fusion option, ``fusion_num_options`` levels;
        ``map_option_ids`` resolves that local index to the real map option ID.
        Block2/5 expose one local choice fixed to fusion_count=1, while Block4
        keeps its full map domain;
      * slot 1 = K, ``k_num_levels`` (== LEVELS_K) levels.
    ``block_full_vec_offsets`` are offsets into the current full action vector
    for this block's slots (the fusion map's expanded block vector is spliced
    there).
    """
    step_idx: int
    layer_idx: int
    block_idx: int
    graph_key_suffix: str
    fusion_num_options: int
    map_option_ids: Tuple[int, ...]
    k_num_levels: int
    k_slot_index: int
    block_num_slots: int
    block_full_vec_offsets: Tuple[int, ...]
    terminal: bool


def fusion_step_schedule(
        num_layers: int,
        fusion_map: Any,
        *,
        profile: str = "mrpc",
        attn_degree_per_layer: Optional[Sequence[int]] = None,
        gelu_degree_per_layer: Optional[Sequence[int]] = None,
        ) -> List[FusionStepSpec]:
    """Build the fusion-count decision schedule by annotating ``step_schedule``
    with per-step fusion-map geometry. ``fusion_map`` is a
    :class:`rfr.preparation.fusion.count_map.FusionCountMap` (duck-typed)."""
    base = step_schedule(
        int(num_layers), profile=str(profile),
        attn_degree_per_layer=attn_degree_per_layer,
        gelu_degree_per_layer=gelu_degree_per_layer,
    )
    out: List[FusionStepSpec] = []
    for s in base:
        gk = s.graph_key_suffix
        if gk not in fusion_map.graphs:
            raise KeyError(
                f"fusion map has no graph '{gk}' (step {s.step_idx}, layer {s.layer_idx}, "
                f"block {s.block_idx}); built graphs: {sorted(fusion_map.graphs)}"
            )
        block_num_slots = int(fusion_map.graphs[gk].block_num_slots)


        block_offsets = tuple(int(o) for o in s.full_vec_offsets[:block_num_slots])
        if len(block_offsets) != block_num_slots:
            raise RuntimeError(
                f"step {s.step_idx} block {s.block_idx}: {len(block_offsets)} offsets != "
                f"map block_num_slots {block_num_slots}"
            )
        map_options = fusion_map.options(gk)
        if int(s.block_idx) in (2, 5):
            map_option_ids = tuple(
                int(option.option_id)
                for option in map_options
                if int(option.fusion_count) == 1
            )
            if len(map_option_ids) != 1:
                raise ValueError(
                    f"{gk}: block{s.block_idx} requires exactly one fusion_count=1 "
                    f"option, found {list(map_option_ids)}"
                )
        else:
            map_option_ids = tuple(int(option.option_id) for option in map_options)
            if not map_option_ids:
                raise ValueError(f"{gk}: fusion map has no options")
        out.append(FusionStepSpec(
            step_idx=s.step_idx,
            layer_idx=s.layer_idx,
            block_idx=s.block_idx,
            graph_key_suffix=gk,
            fusion_num_options=len(map_option_ids),
            map_option_ids=map_option_ids,
            k_num_levels=int(LEVELS_K),
            k_slot_index=int(fusion_map.k_slot_index(gk)),
            block_num_slots=block_num_slots,
            block_full_vec_offsets=block_offsets,
            terminal=bool(s.terminal),
        ))
    return out


def fusion_step_schedule_dims(fusion_map: Any) -> Tuple[int, int]:
    """``(max_step_dim, max_num_levels)`` for instantiating the policy in fusion
    mode: 2 slots per step (fusion option, K); the shared level grid must fit both
    the widest option list and the K levels."""
    return 2, max(int(fusion_map.max_num_options()), int(LEVELS_K))


def fusion_step_slot_levels(spec: FusionStepSpec) -> List[int]:
    """Per-slot legal level counts for this step: [fusion options, K levels]."""
    return [int(spec.fusion_num_options), int(spec.k_num_levels)]


def resolve_fusion_map_option_id(
        spec: FusionStepSpec,
        policy_option_index: int,
        ) -> int:
    """Resolve a policy-local fusion index to the real map option ID."""
    idx = int(policy_option_index)
    if idx < 0 or idx >= len(spec.map_option_ids):
        raise ValueError(
            f"step {spec.step_idx} policy fusion option {idx} out of range "
            f"[0, {len(spec.map_option_ids)})"
        )
    return int(spec.map_option_ids[idx])


def resolve_fusion_policy_option_index(
        spec: FusionStepSpec,
        map_option_id: int,
        ) -> int:
    """Resolve a real map option ID to the policy-local fusion index."""
    option_id = int(map_option_id)
    try:
        return int(spec.map_option_ids.index(option_id))
    except ValueError as exc:
        raise ValueError(
            f"step {spec.step_idx} map fusion option {option_id} is not selectable; "
            f"selectable map options={list(spec.map_option_ids)}"
        ) from exc


def expand_fusion_step_action(
        spec: FusionStepSpec,
        fusion_map: Any,
        option_id: int,
        k_index: int,
        ) -> np.ndarray:
    """Resolve ``(policy_fusion_index, k_index)`` to this block's full SF slot
    vector, with the separately-decided K spliced in."""
    map_option_id = resolve_fusion_map_option_id(spec, int(option_id))
    return fusion_map.expand(spec.graph_key_suffix, map_option_id, int(k_index))


def splice_fusion_step_into_full_vec(
        full_vec: np.ndarray,
        spec: FusionStepSpec,
        expanded_block_vec: Sequence[int],
        ) -> np.ndarray:
    """Write an expanded per-block SF vector into the current full action vector
    at this block's offsets. Returns ``full_vec`` for chaining."""
    arr = np.asarray(expanded_block_vec, dtype=int).reshape(-1)
    if arr.size != len(spec.block_full_vec_offsets):
        raise ValueError(
            f"step {spec.step_idx} expects {len(spec.block_full_vec_offsets)} block slots, got {arr.size}"
        )
    for offset, val in zip(spec.block_full_vec_offsets, arr):
        full_vec[int(offset)] = int(val)
    return full_vec


@dataclass
class ActionDecodeResult:
    """``action_vector_to_cfgs`` 的返回值。"""
    block1_cfgs: Dict[int, Block1NoiseConfig]
    block2_cfgs: Dict[int, Block2NoiseConfig]
    block3_cfgs: Dict[int, Block3NoiseConfig]
    block4_cfgs: Dict[int, Block4NoiseConfig]
    block5_cfgs: Dict[int, Block5NoiseConfig]
    first_input_sf: int

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

    ``is_first_layer`` 仅保留为调用兼容参数；所有层的 truncation K 都生效。
    layer 0 是否注入 Block 1 噪声由最终 cfg 的 ``noise_enabled`` 独立控制。

    精简后只保留 6 个 RL 槽：``gelu_out_sf / wffn2_sf / mean_inv_d_sf /
    var_inv_d_sf / mean_rescale_sf / var_rescale_sf``，外加每层都生效的
    ``output_truncation_k``。被删除的 ``wffn2_rescale_sf`` 和
    ``square_rescale_sf`` 对应 cfg 字段固定为 None（不安装该处 rescale 噪声）。
    """
    del layer_idx, is_first_layer
    return Block1ActionSpec(
        gelu_out_sf=int(layer_field_values["gelu_out_sf"]),
        wffn2_sf=int(layer_field_values["wffn2_sf"]),
        mean_inv_d_sf=int(layer_field_values["mean_inv_d_sf"]),
        var_inv_d_sf=int(layer_field_values["var_inv_d_sf"]),
        mean_rescale_sf=_optional_int(layer_field_values["mean_rescale_sf"]),
        var_rescale_sf=_optional_int(layer_field_values["var_rescale_sf"]),
        output_truncation_k=int(layer_field_values["output_truncation_k"]),
    )


def _build_block2_action(
        layer_idx: int,
        layer_field_values: Dict[str, object],
        profile: str = "mrpc",
        ) -> Block2ActionSpec:
    """精简后的 Block 2 动作构造。

    Q 侧动作（wq / q_mask1 / q_mask2）被删，cfg 上 Q 侧三个 encode 字段由 K 侧
    的同名动作绑定填入（K 侧选什么 SF，Q 侧用同一个 SF）。

    Wv 不是 RL 动作：Rescale_optimizer 的 block2
    计算图里没有 ``ctpt_wv`` 节点，wv 选什么 SF 都不影响 modulus chain；模型噪声
    侧用一个固定的 SF（``_BLOCK2_FIXED_WV_SF``）安装 Wv 噪声即可。

    ``x_centered_fresh_sf`` 绑定到 ``inv_std_fresh_sf``。
    Rescale_optimizer 的 ``ctct_x_mean_over_std`` 是 "x2" 旁节点，语义要求
    x_centered 和 inv_std 两个 fresh ciphertext 的 SF 严格相等；二者拆成
    两个独立动作只会让 optimizer 的 "x2" 假设在某些组合下失效。所以
    ``x_centered_fresh_sf`` 进 ``_COMPAT_EXTRA_FIELDS[2]``，cfg 上由
    ``inv_std_fresh_sf`` 一同填入。

    slot 留在 action vector 里但 ``_COMPAT_EXTRA_FIELDS[2]`` 已经把它们标记成
    effective=False。其余 8 个被删的 rescale 字段在 cfg 上保留为 None。
    """
    inv_std_fresh_sf = int(layer_field_values["inv_std_fresh_sf"])
    wk_sf = int(layer_field_values["wk_sf"])
    kt_mask1_sf = int(layer_field_values["kt_mask1_sf"])
    kt_mask2_sf = int(layer_field_values["kt_mask2_sf"])


    active = active_rescale_rl_fields(2, profile=profile)

    def _rsc(name: str) -> Optional[int]:
        return _optional_int(layer_field_values[name]) if name in active else None

    kt_mask1_r = _rsc("kt_mask1_rescale_sf")
    kt_mask2_r = _rsc("kt_mask2_rescale_sf")
    return Block2ActionSpec(
        inv_std_fresh_sf=inv_std_fresh_sf,

        x_centered_fresh_sf=inv_std_fresh_sf,
        gamma_sf=int(layer_field_values["gamma_sf"]),

        wk_sf=wk_sf,
        wq_sf=wk_sf,

        wv_sf=_BLOCK2_FIXED_WV_SF,
        kt_mask1_sf=kt_mask1_sf,
        q_mask1_sf=kt_mask1_sf,
        kt_mask2_sf=kt_mask2_sf,
        q_mask2_sf=kt_mask2_sf,
        qkt_merge_mask_sf=int(layer_field_values["qkt_merge_mask_sf"]),

        gamma_rescale_sf=_rsc("gamma_rescale_sf"),
        normalize_rescale_sf=_rsc("normalize_rescale_sf"),
        wk_rescale_sf=_rsc("wk_rescale_sf"),
        kt_mask1_rescale_sf=kt_mask1_r,
        q_mask1_rescale_sf=(kt_mask1_r if "q_mask1_rescale_sf" in active else None),
        kt_mask2_rescale_sf=kt_mask2_r,
        q_mask2_rescale_sf=(kt_mask2_r if "q_mask2_rescale_sf" in active else None),
        qkt_matmul_rescale_sf=_rsc("qkt_matmul_rescale_sf"),
        qkt_merge_mask_rescale_sf=_rsc("qkt_merge_mask_rescale_sf"),
        output_truncation_k=int(layer_field_values["output_truncation_k"]),
    )


_BLOCK2_FIXED_WV_SF: int = 22


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


        square_rescale_sfs = tuple(
            square_rescale_base + [square_rescale_base[-1]] * (deg - len(square_rescale_base))
        )
    return Block3ActionSpec(
        degree=deg,
        x_fresh_sf=int(layer_field_values["x_fresh_sf"]),
        inv_2n_sf=int(layer_field_values["inv_2n_sf"]),
        square_rescale_sfs=square_rescale_sfs,
        output_truncation_k=int(layer_field_values["output_truncation_k"]),
    )


def _build_block4_action(
        layer_idx: int,
        layer_field_values: Dict[str, object],
        profile: str = "mrpc",
        ) -> Block4ActionSpec:
    """精简后的 Block 4 动作构造。

    ``softmax_out_mask_sf`` 和 ``v_mask_sf`` 在 RO 计算
    图里对应同一个 ``ctpt_mask2`` 节点（softmax_out × mask 与 v × mask 共享
    mask2 输入）。RL 只用 ``softmax_out_mask_sf`` 一个 slot 表达 mask2 的 SF；
    ``v_mask_sf`` 仍然在 action vector 里（compat-extra），cfg 上直接绑定到
    softmax_out_mask 的 SF。这样模型在 V × mask 这一步安装的噪声 SF 与
    softmax_out × mask 一致，optimizer 算的 ctct_rot_softmax_mul_v.delta
    （由 ``default_block4_cfg_to_delta`` 动态计算成 ``SF(v_fresh) + SF(v_mask)``）
    也对得上 baseline。``_COMPAT_EXTRA_FIELDS[4]`` 把 ``v_mask_sf`` 标成 inactive。
    """
    shared_mask2_sf = int(layer_field_values["softmax_out_mask_sf"])


    active = active_rescale_rl_fields(4, profile=profile)

    def _rsc(name: str) -> Optional[int]:
        return _optional_int(layer_field_values[name]) if name in active else None

    return Block4ActionSpec(
        softmax_out_fresh_sf=int(layer_field_values["softmax_out_fresh_sf"]),
        softmax_out_mask_sf=shared_mask2_sf,
        v_fresh_sf=int(layer_field_values["v_fresh_sf"]),

        v_mask_sf=shared_mask2_sf,
        softmax_v_mask_sf=int(layer_field_values["softmax_v_mask_sf"]),
        wo_sf=int(layer_field_values["wo_sf"]),
        ln_mean_inv_d_sf=int(layer_field_values["ln_mean_inv_d_sf"]),
        ln_var_inv_d_sf=int(layer_field_values["ln_var_inv_d_sf"]),
        softmax_v_matmul_rescale_sf=_rsc("softmax_v_matmul_rescale_sf"),
        ln_mean_rescale_sf=_rsc("ln_mean_rescale_sf"),
        ln_var_rescale_sf=_rsc("ln_var_rescale_sf"),
        ln_square_rescale_sf=_rsc("ln_square_rescale_sf"),
        output_truncation_k=int(layer_field_values["output_truncation_k"]),
    )


def _build_block5_action(
        layer_idx: int,
        layer_field_values: Dict[str, object],
        gelu_degree: int,
        profile: str = "mrpc",
        ) -> Block5ActionSpec:
    """Block 5 动作构造。

    当前映射约束：
    * ``inv_std_fresh_sf`` 绑定到 ``x_centered_fresh_sf``（mrpc graph 的
      ``ctct_xmean_over_std`` 是 "x2" 旁节点，两个 fresh 必须 SF 相同）。
      二者合并成一个动作（由 x_centered_fresh_sf 主导）。
    * ``gelu_coeff_mul_rescale_sf_0`` 升级为 active，驱动
      ``cfg.gelu_coeff_mul_rescales[-1]``（DEFAULT_CFG_TO_T_NEW_MAP 里所有
      block5_n* 的最后一个 entry 都读 [-1]）。slot 名带 "_0"，因为
      cfg 的 ``gelu_coeff_mul_rescales`` 是 length=deg 的 tuple，但 mrpc
      graph 实际把整条 coeff·x^k rescale 合并成一个 ``ctpt_gelu_coeff`` 节点，
      所以只有 [-1] 位置真正进 optimizer。
    """
    deg = int(gelu_degree)


    if deg not in (0, 1, 2, 4):
        deg = 4 if deg >= 4 else (2 if deg >= 2 else 1)


    power_sf_0 = _optional_int(layer_field_values.get("gelu_power_rescale_sf_0")) if deg >= 2 else None
    gelu_power_rescale_sfs: Tuple[Optional[int], ...] = (
        () if deg <= 1 else tuple([power_sf_0] + [None] * (deg - 2))
    )


    coeff_rescale_sf = _optional_int(layer_field_values.get("gelu_coeff_mul_rescale_sf_0"))
    if deg <= 0:
        gelu_coeff_mul_rescale_sfs: Tuple[Optional[int], ...] = ()
    else:
        gelu_coeff_mul_rescale_sfs = tuple([None] * (deg - 1) + [coeff_rescale_sf])

    x_centered_fresh_sf = int(layer_field_values["x_centered_fresh_sf"])


    active = active_rescale_rl_fields(5, gelu_degree=deg, profile=profile)

    def _rsc(name: str) -> Optional[int]:
        return _optional_int(layer_field_values[name]) if name in active else None

    return Block5ActionSpec(
        gelu_degree=deg,

        inv_std_fresh_sf=x_centered_fresh_sf,
        x_centered_fresh_sf=x_centered_fresh_sf,
        gamma_sf=int(layer_field_values["gamma_sf"]),
        wffn1_sf=int(layer_field_values["wffn1_sf"]),
        gelu_coeff_sf=int(layer_field_values["gelu_coeff_sf"]),
        normalize_rescale_sf=_rsc("normalize_rescale_sf"),
        gamma_rescale_sf=_rsc("gamma_rescale_sf"),
        wffn1_rescale_sf=_rsc("wffn1_rescale_sf"),
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
        only: Optional[Tuple[int, int]] = None,
        ) -> ActionDecodeResult:
    """``MultiDiscrete`` 风格动作向量 → 每层 5 个 BLB block cfg + first_input SF。

    Args:
        action_vec:    1D ndarray，长度 == ``sum(action_dims_for_config(num_layers))``
        max_sfs:       ``load_max_sfs(profile)`` 加载的 max SF 表
        num_layers:    模型层数 L
        only:          可选 ``(layer_idx, block_idx)``。给定时只解码/构造该层该
                       block 的 cfg（每个 (layer, block) 的解码彼此独立，结果与
                       全量解码中的同一项逐位一致）——sequential per-block replan
                       与 fusion 图枚举的热路径只消费一个 cfg，全量解码（12 层 ×
                       全 block）是它们的主要耗时。None = 原全量行为，逐位不变。
        gelu_degree:   Block 5 GELU 多项式 degree (1/2/4)；首层并不影响（每层独立）
        attn_degree:   Block 3 softmax 多项式 degree (1..6)

    Returns:
        ``ActionDecodeResult``
    """
    arr = validate_action_vector(action_vec, num_layers)

    layer_dim_list = layer_dims()
    layer_dim = len(layer_dim_list)

    only_layer: Optional[int] = None
    only_block: Optional[int] = None
    if only is not None:
        only_layer, only_block = int(only[0]), int(only[1])
        if not (0 <= only_layer < int(num_layers)) or only_block not in (1, 2, 3, 4, 5):
            raise ValueError(f"only={only!r} out of range (num_layers={num_layers})")

    block1_cfgs: Dict[int, Block1NoiseConfig] = {}
    block2_cfgs: Dict[int, Block2NoiseConfig] = {}
    block3_cfgs: Dict[int, Block3NoiseConfig] = {}
    block4_cfgs: Dict[int, Block4NoiseConfig] = {}
    block5_cfgs: Dict[int, Block5NoiseConfig] = {}
    per_layer_values: List[Dict[str, object]] = []

    for li in (range(int(num_layers)) if only_layer is None else (only_layer,)):
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


        offset = 0
        layer_block_values: Dict[int, Dict[str, object]] = {}
        for b in (1, 2, 3, 4, 5):
            spec = _BLOCK_SPECS[b]
            slot_count = len(spec.fields)
            sub = layer_action[offset:offset + slot_count]
            offset += slot_count
            if only_block is not None and b != only_block:
                continue
            layer_block_values[b] = _decode_block_field_values(
                layer_idx=li,
                block_idx=b,
                action_slice=sub,
                max_sfs=max_sfs,
                attn_degree=li_attn_degree,
                gelu_degree=li_gelu_degree,
            )
        per_layer_values.append({f"block{b}": dict(v) for b, v in layer_block_values.items()})


        if only_block in (None, 1):
            b1 = _build_block1_action(
                li,
                layer_block_values[1],
                is_first_layer=(li == 0),
            )
            block1_cfgs[li] = build_block1_cfg_from_action(
                b1,
                N=_block_default_N(
                    1,
                    gelu_degree=li_gelu_degree,
                    attn_degree=li_attn_degree,
                ),
                noise_enabled=(li != 0),
            )

        if only_block in (None, 2):
            b2 = _build_block2_action(li, layer_block_values[2])
            block2_cfgs[li] = build_block2_cfg_from_action(
                b2, N=_block_default_N(2, gelu_degree=li_gelu_degree, attn_degree=li_attn_degree),
            )

        if only_block in (None, 3):
            b3 = _build_block3_action(li, layer_block_values[3], attn_degree=li_attn_degree)
            block3_cfgs[li] = build_block3_cfg_from_action(
                b3, N=_block_default_N(3, gelu_degree=li_gelu_degree, attn_degree=li_attn_degree),
            )

        if only_block in (None, 4):
            b4 = _build_block4_action(li, layer_block_values[4])
            block4_cfgs[li] = build_block4_cfg_from_action(
                b4, N=_block_default_N(4, gelu_degree=li_gelu_degree, attn_degree=li_attn_degree),
            )

        if only_block in (None, 5):
            b5 = _build_block5_action(li, layer_block_values[5], gelu_degree=li_gelu_degree)
            block5_cfgs[li] = build_block5_cfg_from_action(
                b5, N=_block_default_N(5, gelu_degree=li_gelu_degree, attn_degree=li_attn_degree),
            )


    return ActionDecodeResult(
        block1_cfgs=block1_cfgs,
        block2_cfgs=block2_cfgs,
        block3_cfgs=block3_cfgs,
        block4_cfgs=block4_cfgs,
        block5_cfgs=block5_cfgs,
        first_input_sf=0,
        per_layer_field_values=per_layer_values,
    )


def build_block_cfg_from_field_values(
        block_idx: int,
        layer_idx: int,
        field_values: Dict[str, object],
        *,
        N: int,
        gelu_degree: int = 4,
        attn_degree: int = 4,
        ) -> object:
    """SF-direct block-cfg builder (bypasses the down-sweep grid).

    ``action_vector_to_cfgs`` decodes action *indices* → field_values (via the
    baseline-anchored grid, which cannot represent above-baseline SF) → cfg. The
    precision-boost ("加大精度") produces *above-baseline* explicit SFs, so its
    boosted options are stored as explicit ``field_values`` and built here,
    reusing the SAME ``_build_block{N}_action`` + ``build_block{N}_cfg_from_action``
    as the index path (so an in-grid field_values yields a byte-identical cfg —
    asserted server-side). ``field_values`` must carry every key the block's
    ``_build_block{N}_action`` reads (i.e. a full per-block decoded field_values).
    """
    fv = dict(field_values)
    b = int(block_idx)
    if b == 1:
        spec = _build_block1_action(int(layer_idx), fv, is_first_layer=(int(layer_idx) == 0))
        return build_block1_cfg_from_action(
            spec,
            N=_block_default_N(
                1,
                gelu_degree=gelu_degree,
                attn_degree=attn_degree,
            ) if N is None else int(N),
            noise_enabled=(int(layer_idx) != 0),
        )
    if b == 2:
        spec = _build_block2_action(int(layer_idx), fv)
        return build_block2_cfg_from_action(spec, N=int(N))
    if b == 4:
        spec = _build_block4_action(int(layer_idx), fv)
        return build_block4_cfg_from_action(spec, N=int(N))
    if b == 5:
        spec = _build_block5_action(int(layer_idx), fv, gelu_degree=int(gelu_degree))
        return build_block5_cfg_from_action(spec, N=int(N))
    raise ValueError(f"build_block_cfg_from_field_values: unsupported block_idx {block_idx}")


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


def distinct_sf_level_indices(
        *,
        kind: str,
        levels: int,
        max_sf: Optional[int],
        N: int,
        ) -> List[int]:
    """Enumeration acceleration: one action index per DISTINCT decoded value.

    Under the hybrid sweep, a low-baseline slot's deep levels all snap to the
    noise-table minimum (SF=10) and decode to IDENTICAL cfgs → identical
    replans. The fusion builder skips those duplicates pre-enumeration instead
    of replanning each and deduping post-eval. Result-equivalent by
    construction: for each duplicated value the LOWEST index is kept, which is
    exactly the lex-min representative the post-eval installed-signature dedup
    (min bits, then lex-min action vector) would have selected — the emitted
    option set, including its ``action_indices``, matches a full enumeration;
    only the ``valid_configs`` build diagnostic shrinks. The action space
    itself is untouched (every slot keeps its fixed 10 indices). For R slots,
    idx 0 (= None = drop the rescale) is excluded as always (mental-model
    item 2). Returns ascending.
    """
    vals = _field_level_values(kind=str(kind), levels=int(levels), max_sf=max_sf, N=int(N))
    kept: List[int] = []
    seen: set = set()
    for idx in range(int(levels)):
        v = vals[idx]
        if v is None:
            continue
        if v in seen:
            continue
        seen.add(v)
        kept.append(int(idx))
    return kept


def _operation_name(block_idx: int, field_name: str, kind: str) -> str:
    if str(kind) == "K":
        return f"block{int(block_idx)}_output_truncation"
    return _BLOCK_NODE_NAME_BY_FIELD.get(int(block_idx), {}).get(
        str(field_name),
        f"block{int(block_idx)}_{field_name}",
    )


def _short_field_label(field_name: str, kind: str) -> str:
    """Return a compact, deterministic field tag for action reports."""
    if str(kind) == "K":
        return ""
    f = str(field_name)
    if f.startswith("square_rescale_sf_"):
        return "sq" + f.rsplit("_", 1)[-1]
    if f.startswith("gelu_power_rescale_sf_"):
        return "gp" + f.rsplit("_", 1)[-1]
    if f.startswith("gelu_coeff_mul_rescale_sf_"):
        return "gc" + f.rsplit("_", 1)[-1]
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

        return f"L{int(layer_idx)}.first_input.{str(kind)}"
    base = f"L{int(layer_idx)}.B{int(block_idx)}.{str(kind)}"
    return base if not short else f"{base}.{short}"


_COMPAT_EXTRA_FIELDS: Dict[int, frozenset] = {


    1: frozenset(),
    2: frozenset({


        "wq_sf", "q_mask1_sf", "q_mask2_sf", "wv_sf", "x_centered_fresh_sf",


        "wq_rescale_sf", "wv_rescale_sf", "q_mask1_rescale_sf", "q_mask2_rescale_sf",
    }),
    3: frozenset(),
    4: frozenset({

        "v_mask_sf", "v_mask_rescale_sf",
    }),
    5: frozenset({

        "inv_std_fresh_sf",
    }),
}


_ACTIVE_RESCALE_SETS_CACHE: Optional[Dict[str, frozenset]] = None


def _load_active_rescale_sets() -> Dict[str, frozenset]:
    """``{graph_key: frozenset(active rescale RL fields)}`` derived from the real
    ``static_skeletons`` via :mod:`skeleton_stage_map`. Cached. Empty on any
    failure (a working run loads the same archive for cost/baseline, so an empty
    map here only happens when the run would already have aborted upstream)."""
    global _ACTIVE_RESCALE_SETS_CACHE
    if _ACTIVE_RESCALE_SETS_CACHE is None:
        out: Dict[str, frozenset] = {}
        try:
            from . import skeleton_stage_map as _ssm
            arch_path = (
                RESCALE_CONFIG_ROOT / "mrpc" / "static_skeletons_mrpc.json"
            )
            archive = read_json_file(arch_path)
            for gk, plan in _ssm.build_stage_plans_from_archive(archive).items():
                out[gk] = frozenset(plan.active_rescale_rl_fields)
        except Exception:
            out = {}
        _ACTIVE_RESCALE_SETS_CACHE = out
    return _ACTIVE_RESCALE_SETS_CACHE


def _graph_key_for(block_idx: int, gelu_degree: int = 4, attn_degree: int = 4,
                   profile: str = "mrpc") -> str:
    b = int(block_idx)
    if b == 1:
        return f"block1_{profile}"
    if b == 2:
        return f"block2_{profile}"
    if b == 3:
        return f"block3_exp_n{int(attn_degree)}"
    if b == 4:
        return "block4"
    if b == 5:
        return f"block5_n{int(gelu_degree)}"
    return ""


def active_rescale_rl_fields(block_idx: int, gelu_degree: int = 4, attn_degree: int = 4,
                             profile: str = "mrpc") -> frozenset:
    """Active rescale RL slots for the graph this (block, degree) maps to."""
    sets = _load_active_rescale_sets()
    gk = _graph_key_for(block_idx, gelu_degree, attn_degree, profile)
    if gk in sets:
        return sets[gk]


    return sets.get(_graph_key_for(block_idx, gelu_degree, attn_degree, "mrpc"), frozenset())


def _field_kind(block_idx: int, field_name: str) -> Optional[str]:
    """RL action-field kind ("F"/"W"/"M"/"S"/"R"/"K") from the block field spec."""
    spec = _BLOCK_SPECS.get(int(block_idx))
    if spec is None:
        return None
    for fname, kind, _default in spec.fields:
        if fname == str(field_name):
            return kind
    return None


def _is_action_field_effective(
        *,
        layer_idx: int,
        block_idx: int,
        field_name: str,
        attn_degree: int,
        gelu_degree: int,
        profile: str = "mrpc",
        ) -> Tuple[bool, str]:


    if int(layer_idx) == 0 and int(block_idx) == 1:
        if str(field_name) == "output_truncation_k":
            return True, ""
        return False, (
            "layer 0 Block1 SF/noise fields remain inactive; only its "
            "output_truncation_k is installed"
        )
    if str(field_name) in _COMPAT_EXTRA_FIELDS.get(int(block_idx), frozenset()):
        return False, (
            "compat-extra slot retained for action-vector back-compat; cfg field "
            "is forced None / bound elsewhere so this action value has no effect"
        )


    if _field_kind(int(block_idx), str(field_name)) == "R":
        active = active_rescale_rl_fields(
            int(block_idx), gelu_degree=int(gelu_degree),
            attn_degree=int(attn_degree), profile=str(profile))
        if str(field_name) not in active:
            return False, "not a rescale stage on the current skeleton"


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
    arr = validate_action_vector(action_vec, num_layers)

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
                profile=str(profile),
            )


            if (
                    int(li) == 0
                    and int(block_idx) == 1
                    and str(field_name) != "output_truncation_k"
            ):
                value = None
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


def make_all_max_action_vector(num_layers: int) -> np.ndarray:
    """生成 baseline 动作向量：SF 字段取最高档，K 取该 block 的 baseline K 值。

    Per-block baseline K：
        Block 1 → K=13    Block 2 → K=10    Block 3 → K=13
        Block 4 → K=10    Block 5 → K=13

    用于 §6.3 的 baseline + §11 验证清单中"action 全 max → reward = 0"。
    注：函数名仍叫 ``all_max``（语义上"每个 slot 取其各自的 baseline 最大档"）；
    truncation K 的"最大"含义已按 BASELINE_K_BY_BLOCK 差异化。
    """
    dims = action_dims_for_config(int(num_layers))
    arr = np.array(dims, dtype=int) - 1
    fields = per_layer_field_offsets()
    layer_dim = len(fields)
    for li in range(int(num_layers)):
        for field_offset, (block_idx, _field_name, kind) in enumerate(fields):
            if kind == "K":
                arr[li * layer_dim + field_offset] = _baseline_k_index_for_block(int(block_idx))
    return arr


def make_all_min_action_vector(num_layers: int) -> np.ndarray:
    """生成"全 min" 动作向量：每个分量取 0。"""
    dims = action_dims_for_config(int(num_layers))
    return np.zeros(len(dims), dtype=int)


_K_POSITIONS_IN_LAYER: Optional[Tuple[int, ...]] = None


def _k_positions_in_layer() -> Tuple[int, ...]:
    global _K_POSITIONS_IN_LAYER
    cached = _K_POSITIONS_IN_LAYER
    if cached is not None:
        return cached
    k_positions: List[int] = []
    cursor_in_layer = 0
    for b in (1, 2, 3, 4, 5):
        spec = _BLOCK_SPECS[b]
        for _fname, kind, _max in spec.fields:
            if kind == "K":
                k_positions.append(cursor_in_layer)
            cursor_in_layer += 1
    _K_POSITIONS_IN_LAYER = tuple(k_positions)
    return _K_POSITIONS_IN_LAYER


def _sum_count_effective_k_values_in_action(
        action_vec: np.ndarray,
        num_layers: int,
        ) -> Tuple[int, int]:
    arr = validate_action_vector(action_vec, num_layers)
    total, count, _average = truncation_k_summary_from_full_action(
        arr,
        num_layers,
        k_levels=K_LEVELS,
    )
    return total, count


def _gather_effective_k_values_in_action(
        action_vec: np.ndarray,
        num_layers: int,
        ) -> List[int]:
    arr = validate_action_vector(action_vec, num_layers)
    layer_dim = len(layer_dims())
    ks: List[int] = []
    for li in range(int(num_layers)):
        for p in _k_positions_in_layer():
            slot = li * layer_dim + p
            idx = int(arr[slot])
            ks.append(int(K_LEVELS[idx]))
    return ks


def avg_truncation_k_in_action(
        action_vec: np.ndarray,
        num_layers: int,
        ) -> float:
    """从 action 向量里抽出每个 block 的 K 选择，返回平均 k。

    用于 reward 中的 ``k_drop = baseline.avg_k - avg_k``。
    """
    total, count = _sum_count_effective_k_values_in_action(action_vec, num_layers)
    if count <= 0:
        return 0.0
    return float(total / count)


def sum_truncation_k_in_action(
        action_vec: np.ndarray,
        num_layers: int,
        ) -> int:
    """Sum of decoded truncation-k values across all effective K slots.

    Lets cost-matched random samplers do an integer equality pre-filter
    before paying for a Rescale_optimizer call (mean comparisons would
    need an explicit tolerance).
    """
    total, _count = _sum_count_effective_k_values_in_action(action_vec, num_layers)
    return int(total)


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
        raw_degree = getattr(cfg, "degree", 4)
        degree = 4 if raw_degree is None else int(raw_degree)
        graph_key = f"block3_exp_n{degree}"
    elif block_idx == 4:
        graph_key = "block4"
    elif block_idx == 5:
        raw_degree = getattr(cfg, "gelu_degree", 4)
        degree = 4 if raw_degree is None else int(raw_degree)
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

    NOTE: 不会发送 ``(block=1, layer=0)`` 给 Rescale_optimizer，因为该位置
    没有可调 SF/fusion replan。它仍会出现在 ``block1_cfgs`` 中并由 bridge
    安装为 K-only cfg；truncation K 不依赖 Rescale_optimizer。
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

                continue
            cn = make_config_name(profile, block_idx, int(layer_idx), cfg=cfg)
            out[cn] = (str(block_name), cfg)
    return out
