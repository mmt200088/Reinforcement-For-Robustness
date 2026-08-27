"""Full-vector Stage-2 action schema and model-config materialization.

The layerwise policy projects its compact fusion/precision matrix into this
stable vector. Inactive compatibility slots keep fixed offsets for persisted
artifacts but are never installed or charged by the optimizer.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from rfr.common.json_utils import read_json_file
from rfr.preparation.fusion.count_map import FUSION_CONFIG_ROOT
from rfr.preparation.rescale import RESCALE_CONFIG_ROOT
from rfr.search.runtime.blb_bridge import (
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
from rfr.search.runtime.model_handler import (
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
    """Discrete action fields for one BLB block in one encoder layer."""
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
    """Choose the default ring degree for a block and approximation degree."""
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
    """Clamp an SF to the nearest supported noise-table key at or below it."""
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
    """Cached maximum scaling factors by block and node name."""
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
    """Load a profile-specific maximum-SF table with safe defaults."""
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
    """Return discrete dimension sizes for one block."""
    spec = _BLOCK_SPECS[int(block_idx)]
    return [NUM_LEVELS_PER_DIM_BY_BLOCK_KIND[kind] for _name, kind, _max in spec.fields]


def block_field_names(block_idx: int) -> Tuple[str, ...]:
    """Return persisted full-vector field names for one block."""
    try:
        spec = _BLOCK_SPECS[int(block_idx)]
    except KeyError as exc:
        raise ValueError(f"unknown block_idx={block_idx}") from exc
    return tuple(str(name) for name, _kind, _maximum in spec.fields)


def layer_dims() -> List[int]:
    """Return discrete dimension sizes for all five blocks in one layer."""
    out: List[int] = []
    for b in (1, 2, 3, 4, 5):
        out.extend(block_dims(b))
    return out


def action_dims_for_config(num_layers: int) -> List[int]:
    """Return the full MultiDiscrete shape for a model."""
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
    """Return each per-layer action component and its block field."""
    out: List[Tuple[int, str, str]] = []
    for b in (1, 2, 3, 4, 5):
        for fname, kind, _max in _BLOCK_SPECS[b].fields:
            out.append((b, fname, kind))
    return out


@dataclass
class ActionDecodeResult:
    """Materialized block configurations and first-input scale for one action."""
    block1_cfgs: Dict[int, Block1NoiseConfig]
    block2_cfgs: Dict[int, Block2NoiseConfig]
    block3_cfgs: Dict[int, Block3NoiseConfig]
    block4_cfgs: Dict[int, Block4NoiseConfig]
    block5_cfgs: Dict[int, Block5NoiseConfig]
    first_input_sf: int

    per_layer_field_values: List[Dict[str, object]] = field(default_factory=list)

    def cfgs_dict(self) -> Dict[str, Dict[int, object]]:
        """Return block configurations in optimizer signal-map form."""
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
    """Build a Block 1 action while keeping K active in every layer."""
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
    """Build a Block 2 action with the required Q/K and fresh-scale bindings."""
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
    """Build a Block 4 action with its shared mask scale."""
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
    """Build a Block 5 action with bound normalization scales and active GELU output scale."""
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
    """Decode one block action segment into field values."""
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
    """Decode the five-level first-input fresh scale."""

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
    """Materialize a full or single-block action vector into BLB configurations."""
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
    """Materialize one block directly from explicit field values."""
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
    """Build the all-maximum baseline action, including compatibility slots."""
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
    """Build the all-minimum action vector."""
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
    """Return the mean simulated truncation K across active layer blocks."""
    total, count = _sum_count_effective_k_values_in_action(action_vec, num_layers)
    if count <= 0:
        return 0.0
    return float(total / count)


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
    """Build per-block Rescale optimizer requests from decoded configurations."""
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
