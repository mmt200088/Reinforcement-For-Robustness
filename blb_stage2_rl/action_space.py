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

注：2026-05-14 一度把 25 个 "mrpc baseline 不覆盖的" 槽（block1 wffn2/square_rescale；
block2 Q-side encodes + 8 个 rescale；block3 x_inv_2n_rescale；block4 5 个 rescale；
block5 6 个 rescale）从字段表里删除，但 CLAUDE.md 的指引是 "classify as compat-extra /
inactive rather than deleting"。所以这些槽现在以 *compat-extra* 形式回到 _BLOCK*_FIELDS：
槽存在于 action vector / registry / describe 输出里，但 ``_is_action_field_effective``
把它们标记成 effective=False，``build_block*_cfg_from_action`` 强制把对应 cfg 字段写成
None（即不安装该处噪声），所以 RL 选这些槽的值不会改变 cfg 也不会改变 optimizer cost。

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
# 2026-06-04: all SF kinds use a uniform 10-level HYBRID downward sweep anchored at
# the slot's BASELINE SF (= calibrated max_sf, NOT the noise-table max=46): the top 5
# levels step by 2 (coarse, near baseline), the bottom 5 step by 1 (fine), reaching
# baseline-14. e.g. baseline 30 → 30,28,26,24,22,20,19,18,17,16 (see ``sf_from``).
# ``_snap_to_table`` floors any decoded SF below the noise table (min SF=10); a slot
# whose baseline SF is low gets duplicate SF=10 levels — the fusion builder skips
# them pre-enumeration via ``distinct_sf_level_indices`` (result-equivalent: a
# duplicate-value level decodes to the identical cfg; the lowest index per value is
# kept, matching the lex-min representative the post-eval signature dedup keeps) and
# still dedups by installed-noise signature as before. K stays ``K_LEVELS``. The
# noise variance comes from the N=16384 column for every slot (``_block_default_N``).
# (2026-06-10: a uniform step-2 / floor-12 grid was tried and REVERTED the same day
# per user decision — every slot keeps the fixed 10-level hybrid sweep.)
LEVELS_W = 10        # weight encode
LEVELS_MS = 10       # mask / scalar encode
LEVELS_R = 10        # rescale (idx0=None; idx1..9 sweep SF, hybrid step)
LEVELS_F = 10        # fresh

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

# 每个 block 在 baseline / 全 max action 下的 truncation K。
# 历史上短暂存在过 per-block 差异化（2026-05-15 实验 B2/B4=10），但相应改动让
# 全 max 动作的 avg_k 偏离 baseline，造成 reward 不为零 / contract test 失败；
# 现在统一回归 K=13（与 noise_std_table N=16384 主测档对齐）。
# 这是 RL warmstart 锚点 + reward.k_drop 的基准；RL 训练时可以选其它 K 值，但 cost
# reward 会以这个 baseline 计算 k_drop。
BASELINE_K_BY_BLOCK: Dict[int, int] = {1: 13, 2: 13, 3: 13, 4: 13, 5: 13}


def _baseline_k_index_for_block(block_idx: int) -> int:
    """Map per-block baseline K value -> action index in K_LEVELS.

    Falls back to ``K_LEVELS.index(max(K_LEVELS))`` if the configured baseline
    K isn't in the K_LEVELS table (defensive — the BLB_TRUNCATION_K_LEVELS env
    var override could remove 10 or 13).
    """
    target = int(BASELINE_K_BY_BLOCK.get(int(block_idx), max(K_LEVELS)))
    if target in K_LEVELS:
        return int(K_LEVELS.index(target))
    return int(K_LEVELS.index(max(K_LEVELS)))


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
# 2026-05-14 一度删过 ``wffn2_rescale_sf`` / ``square_rescale_sf`` 两个 rescale 槽，
# 但 CLAUDE.md 的指引是“classify discrepancies as compat-extra/inactive rather than
# deleting”，所以现在按 compat-extra 复位 —— 槽保留在 action vector 里、参与 registry
# 计数、describe_action_vector 仍会汇报；但它们不在 mrpc baseline skeleton 上、
# Rescale_optimizer 不会选用，``build_block1_cfg_from_action`` 强制把对应 cfg 字段
# 写成 None（即不安装该处 rescale 噪声）。``_is_action_field_effective`` 会把它们
# 标记成 effective=False。
_BLOCK1_FIELDS = _BlockFieldSpec(
    fields=(
        ("gelu_out_sf",        "F", 30),
        ("wffn2_sf",           "W", 22),
        ("mean_inv_d_sf",      "S", 22),
        ("var_inv_d_sf",       "S", 22),
        ("mean_rescale_sf",    "R", 22),
        ("var_rescale_sf",     "R", 22),
        # compat-extra (mrpc baseline 不上这两处 rescale；cfg 字段固定 None)
        ("wffn2_rescale_sf",   "R", 22),
        ("square_rescale_sf",  "R", 22),
        ("output_truncation_k","K", 13),
    ),
)


# Block 2
# 2026-05-14 曾删过 11 个 RL 槽（3 个 Q-side encode + 8 个 rescale），现在按
# compat-extra 全部复位 ——
#   * Q-side encodes (``wq_sf`` / ``q_mask1_sf`` / ``q_mask2_sf``) 与 K-side 绑定，
#     ``_build_block2_action`` 把 K-side action value 拷贝到 cfg 的 Q-side 字段。
#     RL 选这三个槽的 action 不会影响 cfg（cfg 用 K-side 的值）。
#   * 8 个 rescale (``normalize/wk/wq/wv/kt_mask1/q_mask1/q_mask2/qkt_matmul``) 不在
#     mrpc baseline skeleton 上，``build_block2_cfg_from_action`` 强制写 None。
# ``wv_sf`` 仍然驱动模型 V 路径噪声（虽不入 optimizer cost）。
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
        # compat-extra encodes (Q-side bound to K-side via _build_block2_action)
        ("wq_sf",                       "W", 22),
        ("q_mask1_sf",                  "M", 22),
        ("q_mask2_sf",                  "M", 22),
        # compat-extra rescales (mrpc baseline 不上这些点；cfg 字段固定 None)
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


# Block 3：Softmax 近似中的 fresh / encode / 多次 squaring。
# 2026-05-14 删过 ``x_inv_2n_rescale_sf``，现在按 compat-extra 复位 ——
# mrpc baseline 不上 ``ctct_x_inv_2n_rescale``，cfg 字段固定 None。
# ``square_rescale_sf_0..3``（按 degree 截短）通过 t_new 进 optimizer。
_BLOCK3_R_SLOTS = 4   # square_rescale_sf_0..3 (max degree=4)
_BLOCK3_FIELDS = _BlockFieldSpec(
    fields=(
        ("x_fresh_sf",              "F", 30),
        ("inv_2n_sf",               "S", 22),
        ("square_rescale_sf_0",     "R", 22),
        ("square_rescale_sf_1",     "R", 22),
        ("square_rescale_sf_2",     "R", 22),
        ("square_rescale_sf_3",     "R", 22),
        # compat-extra (mrpc baseline 不上这个 rescale；cfg 字段固定 None)
        ("x_inv_2n_rescale_sf",     "R", 22),
        ("output_truncation_k",     "K", 13),
    ),
)


# Block 4
# 2026-05-14 删过 5 个 rescale 槽 (``softmax_out_mask_rescale_sf`` /
# ``v_mask_rescale_sf`` / ``softmax_v_mask_rescale_sf`` / ``wo_rescale_sf`` /
# ``ln_square_rescale_sf``)，现在按 compat-extra 复位 —— 它们不在 mrpc baseline
# skeleton 上，``build_block4_cfg_from_action`` 强制 cfg 字段 None。
# V 侧 ``v_fresh_sf`` / ``v_mask_sf`` 仍驱动模型噪声（虽不入 optimizer cost）。
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
        # compat-extra rescales (mrpc baseline 不上这些点；cfg 字段固定 None)
        ("softmax_out_mask_rescale_sf",     "R", 22),
        ("v_mask_rescale_sf",               "R", 22),
        ("softmax_v_mask_rescale_sf",       "R", 22),
        ("wo_rescale_sf",                   "R", 22),
        ("ln_square_rescale_sf",            "R", 22),
        ("output_truncation_k",             "K", 13),
    ),
)


# Block 5（GELU degree-aware）
# 2026-05-14 删过 6 个 rescale 槽 (gelu_power_rescale_sf_1/2 + 4 个
# gelu_coeff_mul_rescale_sf_*)，现在按 compat-extra 复位 ——
# mrpc graph 里 x³ 折进 x⁴、ctpt_gelu_coeff 单节点替代了 4 段 coeff·x^k rescale，
# 所以这 6 个槽不在 skeleton 上，cfg 字段固定 None。
# ``gelu_power_rescale_sf_0`` (x²) 在 degree>=2 时通过 t_new 进 optimizer。
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
        ("gelu_power_rescale_sf_0",         "R", 22),  # x²；degree>=2 时启用
        # compat-extra rescales (x³ 折进 x⁴；coeff_mul 链合到 ctpt_gelu_coeff)
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


# Per-block "节点名" → block 字段名（用于 max_sfs JSON 反查）。
# 与 ``rescale_optimizer_bridge.default_block*_cfg_to_delta`` 的命名约定保持一致；
# 实际外部 ``static_skeletons_<profile>.json`` 节点名以用户最终使用的 config 为准。
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
        # ``wk_sf`` 同时控制 Q/K 两侧 encode（bridge -> ctpt_wq_wk）
        "wk_sf":                       "ctpt_wq_wk",
        # 2026-05-21 user spec：Q 侧四个 compat-extra 字段（wq / q_mask1 /
        # q_mask2 / q_mask2_r）也 mirror 到对应的 q/k 共享 graph 节点。
        # baseline 抽取在 _RO_*_NODE_TO_RL_FIELD[2] 里写两份；max_sfs 校准
        # 时两个 RL 字段查同一个节点 key，互相一致。
        "wq_sf":                       "ctpt_wq_wk",
        "q_mask1_sf":                  "ctpt_rotKT_mask1",
        "q_mask2_sf":                  "ctpt_rotKT_mask2",
        "q_mask2_rescale_sf":          "ctct_kt_mask2_rescale",
        # ``wv_sf`` 只控制模型噪声（mrpc graph 无对应节点）
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
        # 2026-05-21: gelu_coeff_mul_rescale_sf_0 drives the rescale after
        # ctpt_gelu_coeff (coeff × x^k). All degrees have this rescale in the
        # baseline (n=1: sf_post=30; n=2/n=4: sf_post=31). Node name follows
        # the ``ctct_*_rescale`` convention used by neighbouring rescales.
        "gelu_coeff_mul_rescale_sf_0":      "ctct_gelu_coeff_rescale",
    },
}


# ---------------------------------------------------------------------------
# Block N 选取（与 ``make_block*_default_config`` 推荐一致）
# ---------------------------------------------------------------------------
def _block_default_N(block_idx: int, gelu_degree: int = 4, attn_degree: int = 4) -> int:
    """噪声表 N（决定 noise variance 查表 + snap 范围）。

    2026-06-04（用户指定）：N=8192 暂时不用，所有 block 的所有槽都统一查 N=16384 的噪声表。
    （只影响安装的 noise variance / snap；Rescale_optimizer 的 modulus-chain N 仍由其图定义，
    不受此影响。）原先 block1 / block5_n0,n1 / block3@attn2 用 8192——如需恢复差异化 N，改这里。
    """
    return 16384


# ---------------------------------------------------------------------------
# action index ↔ scaling factor 转换
# ---------------------------------------------------------------------------
def sf_from(idx: int, max_sf: int, levels: int) -> int:
    """Hybrid downward sweep from ``max_sf`` (= the slot's BASELINE SF, not the table max).

    The top half of the levels step by 2 (coarse, near baseline); the bottom half step
    by 1 (fine, far from baseline). For the canonical 10-level slot this gives
    ``idx 9..0 → max_sf, -2, -4, -6, -8, -10, -11, -12, -13, -14`` (e.g. max_sf=30 →
    30,28,26,24,22,20,19,18,17,16).

    May return a value below the noise-table minimum for a low-baseline-SF slot;
    callers wrap with ``_snap_to_table`` which floors it at the table min (SF=10).
    """
    idx = int(idx)
    levels = int(levels)
    if idx < 0 or idx >= levels:
        raise ValueError(f"action idx {idx} out of [0, {levels})")
    n_fine = (levels - 1) // 2          # step-1 intervals at the bottom
    n_coarse = (levels - 1) - n_fine    # step-2 intervals at the top
    dist = (levels - 1) - idx           # 0 at idx = max (baseline SF)
    offset = 2 * dist if dist <= n_coarse else 2 * n_coarse + (dist - n_coarse)
    return int(max_sf) - int(offset)


def _rescale_sf_from_index(idx: int, max_sf: int) -> Optional[int]:
    # idx 0 = None (rescale dropped — never selected by RL / the fusion builder, per
    # CLAUDE.md item 2). idx 1..LEVELS_R-1 sweep SF via the hybrid decode; _snap_to_table
    # floors at SF=10 downstream.
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
        # Look up via registered RO graph-node name first; if there is no entry
        # for the field, fall back to using the field name itself as the node
        # key. The fallback path is what ``static_skeletons_baseline_to_action``
        # writes when a baseline provides an SF for a field that has no entry
        # in ``_BLOCK_NODE_NAME_BY_FIELD`` (e.g. baseline-injected compat-extra
        # rescales like ``wo_rescale_sf``).
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
# Per-block sequential step schedule (for the sequential RL formulation).
#
# Episode order (C, 2026-05-30: block 3 excluded from the decided schedule):
#   step 1:  layer 0, block 2  (+ first_input fresh)   -- layer 0 has no block 1
#   step 2:  layer 0, block 4                           -- block 3 skipped
#   step 3:  layer 0, block 5
#   step 4:  layer 1, block 1
#   step 5:  layer 1, block 2
#   step 6:  layer 1, block 4                           -- block 3 skipped
#   ...
#   step 3 + (L-1)*4: layer L-1, block 5
#
# Total horizon for L layers = 3 + (L-1)*4
# For L=12 -> horizon = 47 (was 59 when block 3 was decided).
# Block 3's slots remain in the legacy full action vector (frozen at the
# static_skeletons baseline); only the decided schedule drops them.
#
# first_input fresh (legacy "tail" slot in the action vector) is decided
# alongside layer 0 block 2 to avoid an extra horizon=0 step. It still occupies
# its same position at the end of the full action vector.
# ---------------------------------------------------------------------------

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
        full_vec_offsets: index into the legacy 577-dim action vector where
                          each slot's value should be written.
        includes_first_input: True for step_idx==0 only (folded into layer-0
                              block-2). When True, the LAST entry of slot_dims
                              etc. is the first_input fresh slot.
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
    includes_first_input: bool
    graph_key_suffix: str          # e.g. "block2_mrpc" -- profile filled in by env
    terminal: bool


# layer 0 lives without block 1 -- the layer-0 input goes straight into block 2.
# C (2026-05-30): block 3 (softmax exp approx) is no longer an RL decision, so it is
# excluded from the decided schedule (no block_idx==3 step). The legacy full action
# vector still KEEPS block 3's slots, frozen at the static_skeletons baseline;
# _BLOCK3_FIELDS / build_block3_cfg_from_action stay defined but unused by the schedule.
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
    """Sequential episode horizon (C, 2026-05-30: block 3 excluded from schedule).

    Block 3 is no longer a decided step, so:
      - layer 0 has 3 steps (B2, B4, B5)
      - layers 1..L-1 have 4 steps each (B1, B2, B4, B5)
    giving horizon = 3 + (L-1) * 4. For L=12 this is 47 steps (was 59 with block 3).
    The first step (layer 0, block 2) also writes first_input fresh.
    """
    L = int(num_layers)
    if L < 1:
        raise ValueError(f"num_layers must be >= 1, got {L}")
    return 3 + (L - 1) * 4


def _full_vec_offset_for_block(num_layers: int, layer_idx: int, block_idx: int) -> int:
    """Return start offset (into the legacy 577-dim full action vector) of
    the (layer_idx, block_idx) slot range.

    The full vec is one categorical index per slot per layer, in block order
    1..5; first_input fresh sits at the very end.
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
    fi_offset = _full_vec_first_input_offset(L)
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
            includes_first = (step_idx == 0)
            if includes_first:
                slot_dims.append(LEVELS_FIRST_INPUT)
                slot_field_names.append("__first_input_sf__")
                slot_kinds.append("F")
                full_vec_offsets.append(fi_offset)
            # graph key suffix
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
                includes_first_input=includes_first,
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
    """Write the per-step action's slot values into the legacy 577-dim vec.

    ``step_action`` length must equal ``len(step.slot_dims)``. Returns
    ``full_vec`` for chaining.
    """
    arr = np.asarray(step_action, dtype=int).reshape(-1)
    if arr.size != len(step.full_vec_offsets):
        raise ValueError(
            f"step {step.step_idx} expects {len(step.full_vec_offsets)} slots, got {arr.size}"
        )
    for offset, val in zip(step.full_vec_offsets, arr.tolist()):
        full_vec[int(offset)] = int(val)
    return full_vec


def empty_full_action_vec(num_layers: int) -> np.ndarray:
    """Zero-initialised full action vector matching ``action_dims_for_config`` width."""
    return np.zeros(len(action_dims_for_config(int(num_layers))), dtype=np.int64)


# ---------------------------------------------------------------------------
# Fusion-count action schedule (opt-in; see fusion_count_map / spec 2026-06-03)
#
# Same (layer, block) order as ``step_schedule`` (block 3 already excluded), but
# each step decides just two categoricals: a fusion-count OPTION (resolved to a
# full per-block SF vector via the offline fusion map) and a separate K. The
# policy is the SAME GTrXL instantiated with max_step_dim=2.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FusionStepSpec:
    """One (layer, block) step in the fusion-count episode.

    The per-step action is ``(fusion_option_id, k_index)``:
      * slot 0 = fusion option, ``fusion_num_options`` levels (1 for a degenerate
        block-type like block1/block4 that can only reach fusion 0);
      * slot 1 = K, ``k_num_levels`` (== LEVELS_K) levels.
    ``block_full_vec_offsets`` are the legacy-577-vec offsets of this block's
    slots (the fusion map's expanded block vector is spliced there).
    """
    step_idx: int
    layer_idx: int
    block_idx: int
    graph_key_suffix: str          # fusion-map key: block1_mrpc / block2_mrpc / block4 / block5_n{deg}
    fusion_num_options: int
    k_num_levels: int
    k_slot_index: int              # within-block index of the K slot
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
    :class:`blb_stage2_rl.fusion_count_map.FusionCountMap` (duck-typed)."""
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
        # s.full_vec_offsets lists this block's contiguous slot offsets first, then
        # (step 0 only) the deprecated first_input fresh slot — which fusion mode
        # leaves at its baseline. Take the block's own slots.
        block_offsets = tuple(int(o) for o in s.full_vec_offsets[:block_num_slots])
        if len(block_offsets) != block_num_slots:
            raise RuntimeError(
                f"step {s.step_idx} block {s.block_idx}: {len(block_offsets)} offsets != "
                f"map block_num_slots {block_num_slots}"
            )
        out.append(FusionStepSpec(
            step_idx=s.step_idx,
            layer_idx=s.layer_idx,
            block_idx=s.block_idx,
            graph_key_suffix=gk,
            fusion_num_options=int(fusion_map.num_options(gk)),
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


def expand_fusion_step_action(
        spec: FusionStepSpec,
        fusion_map: Any,
        option_id: int,
        k_index: int,
        ) -> np.ndarray:
    """Resolve ``(fusion_option_id, k_index)`` to this block's full SF slot vector
    (length ``block_num_slots``), with the separately-decided K spliced in."""
    return fusion_map.expand(spec.graph_key_suffix, int(option_id), int(k_index))


def splice_fusion_step_into_full_vec(
        full_vec: np.ndarray,
        spec: FusionStepSpec,
        expanded_block_vec: Sequence[int],
        ) -> np.ndarray:
    """Write an expanded per-block SF vector into the legacy 577-dim vec at this
    block's offsets. Returns ``full_vec`` for chaining."""
    arr = np.asarray(expanded_block_vec, dtype=int).reshape(-1)
    if arr.size != len(spec.block_full_vec_offsets):
        raise ValueError(
            f"step {spec.step_idx} expects {len(spec.block_full_vec_offsets)} block slots, got {arr.size}"
        )
    for offset, val in zip(spec.block_full_vec_offsets, arr.tolist()):
        full_vec[int(offset)] = int(val)
    return full_vec


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

    精简后只保留 6 个 RL 槽：``gelu_out_sf / wffn2_sf / mean_inv_d_sf /
    var_inv_d_sf / mean_rescale_sf / var_rescale_sf``，外加首层依赖的
    ``output_truncation_k``。被删除的 ``wffn2_rescale_sf`` 和
    ``square_rescale_sf`` 对应 cfg 字段固定为 None（不安装该处 rescale 噪声）。
    """
    return Block1ActionSpec(
        gelu_out_sf=int(layer_field_values["gelu_out_sf"]),
        wffn2_sf=int(layer_field_values["wffn2_sf"]),
        mean_inv_d_sf=int(layer_field_values["mean_inv_d_sf"]),
        var_inv_d_sf=int(layer_field_values["var_inv_d_sf"]),
        mean_rescale_sf=_optional_int(layer_field_values["mean_rescale_sf"]),
        var_rescale_sf=_optional_int(layer_field_values["var_rescale_sf"]),
        output_truncation_k=(None if is_first_layer else int(layer_field_values["output_truncation_k"])),
    )


def _build_block2_action(
        layer_idx: int,
        layer_field_values: Dict[str, object],
        profile: str = "mrpc",
        ) -> Block2ActionSpec:
    """精简后的 Block 2 动作构造。

    Q 侧动作（wq / q_mask1 / q_mask2）被删，cfg 上 Q 侧三个 encode 字段由 K 侧
    的同名动作绑定填入（K 侧选什么 SF，Q 侧用同一个 SF）。

    2026-05-20 user spec：Wv 也不再有 RL 动作 —— Rescale_optimizer 的 block2
    计算图里没有 ``ctpt_wv`` 节点，wv 选什么 SF 都不影响 modulus chain；模型噪声
    侧用一个固定的 SF（``_BLOCK2_FIXED_WV_SF``）安装 Wv 噪声即可。

    2026-05-21 user spec：``x_centered_fresh_sf`` 绑定到 ``inv_std_fresh_sf``。
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
    # Which rescale points the CURRENT block2 skeleton selected (auto via SSOT).
    # Q-side rescales are bound equal to their K-side action when active.
    active = active_rescale_rl_fields(2, profile=profile)

    def _rsc(name: str) -> Optional[int]:
        return _optional_int(layer_field_values[name]) if name in active else None

    kt_mask1_r = _rsc("kt_mask1_rescale_sf")
    kt_mask2_r = _rsc("kt_mask2_rescale_sf")
    return Block2ActionSpec(
        inv_std_fresh_sf=inv_std_fresh_sf,
        # x_centered_fresh 绑定到 inv_std_fresh（一同决定）。
        x_centered_fresh_sf=inv_std_fresh_sf,
        gamma_sf=int(layer_field_values["gamma_sf"]),
        # Q/K 绑定：wq_sf / q_mask{1,2}_sf 不再独立选，等于 K 侧
        wk_sf=wk_sf,
        wq_sf=wk_sf,
        # Wv 不再由 RL 控制；cfg 上用固定 SF 安装 Wv 噪声（默认 22 = baseline）。
        wv_sf=_BLOCK2_FIXED_WV_SF,
        kt_mask1_sf=kt_mask1_sf,
        q_mask1_sf=kt_mask1_sf,
        kt_mask2_sf=kt_mask2_sf,
        q_mask2_sf=kt_mask2_sf,
        qkt_merge_mask_sf=int(layer_field_values["qkt_merge_mask_sf"]),
        # Rescales: set iff on the current skeleton (skeleton_stage_map active set).
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


# 2026-05-20 user spec: Block 2 Wv encode no longer has an RL action. The model
# still installs noise at the Wv multiplication point — we just use a constant
# SF rather than letting RL drift it. 22 matches the mrpc max_sfs.ctpt_wv default.
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
        # The historical action vector exposes four square-rescale slots.
        # Degree-5/6 softmax configs reuse the last slot for the extra powers.
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

    2026-05-20 user spec：``softmax_out_mask_sf`` 和 ``v_mask_sf`` 在 RO 计算
    图里对应同一个 ``ctpt_mask2`` 节点（softmax_out × mask 与 v × mask 共享
    mask2 输入）。RL 只用 ``softmax_out_mask_sf`` 一个 slot 表达 mask2 的 SF；
    ``v_mask_sf`` 仍然在 action vector 里（compat-extra），cfg 上直接绑定到
    softmax_out_mask 的 SF。这样模型在 V × mask 这一步安装的噪声 SF 与
    softmax_out × mask 一致，optimizer 算的 ctct_rot_softmax_mul_v.delta
    （由 ``default_block4_cfg_to_delta`` 动态计算成 ``SF(v_fresh) + SF(v_mask)``）
    也对得上 baseline。``_COMPAT_EXTRA_FIELDS[4]`` 把 ``v_mask_sf`` 标成 inactive。
    """
    shared_mask2_sf = int(layer_field_values["softmax_out_mask_sf"])
    # Rescales follow the current block4 skeleton (auto via SSOT): the 2026 regen
    # put the 3rd LN-tail rescale on ctct_square ((X−μ)²) instead of ln_var.
    active = active_rescale_rl_fields(4, profile=profile)

    def _rsc(name: str) -> Optional[int]:
        return _optional_int(layer_field_values[name]) if name in active else None

    return Block4ActionSpec(
        softmax_out_fresh_sf=int(layer_field_values["softmax_out_fresh_sf"]),
        softmax_out_mask_sf=shared_mask2_sf,
        v_fresh_sf=int(layer_field_values["v_fresh_sf"]),
        # mask2 绑定：v_mask cfg 字段总是用与 softmax_out_mask 相同的 SF。
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

    2026-05-21 user spec：
    * ``inv_std_fresh_sf`` 绑定到 ``x_centered_fresh_sf``（mrpc graph 的
      ``ctct_xmean_over_std`` 是 "x2" 旁节点，两个 fresh 必须 SF 相同）。
      二者合并成一个动作（由 x_centered_fresh_sf 主导）。
    * ``gelu_coeff_mul_rescale_sf_0`` 升级为 active，驱动
      ``cfg.gelu_coeff_mul_rescales[-1]``（DEFAULT_CFG_TO_T_NEW_MAP 里所有
      block5_n* 的最后一个 entry 都读 [-1]）。slot 名带 "_0" 是历史遗留：
      cfg 的 ``gelu_coeff_mul_rescales`` 是 length=deg 的 tuple，但 mrpc
      graph 实际把整条 coeff·x^k rescale 合并成一个 ``ctpt_gelu_coeff`` 节点，
      所以只有 [-1] 位置真正进 optimizer。
    """
    deg = int(gelu_degree)
    # block5 GELU degree 支持 {0, 1, 2, 4}（0=ReLU→block5_n0，无多项式 GELU 噪声）。
    # 其它杂散值钳到最近的合法 degree（>=4→4, ==3→2, <=0 留 0 让 ReLU 分支生效）。
    if deg not in (0, 1, 2, 4):
        deg = 4 if deg >= 4 else (2 if deg >= 2 else 1)
    # RL 只控制 ``gelu_power_rescale_sf_0``（x²，degree>=2 时启用）；x³/x⁴ 在
    # mrpc graph 里被折掉，不上 skeleton。
    power_sf_0 = _optional_int(layer_field_values["gelu_power_rescale_sf_0"]) if deg >= 2 else None
    gelu_power_rescale_sfs: Tuple[Optional[int], ...] = (
        () if deg <= 1 else tuple([power_sf_0] + [None] * (deg - 2))
    )
    # gelu_coeff_mul_rescale_sf_0 → cfg.gelu_coeff_mul_rescales[-1]
    # （tuple 其它位置固定 None；optimizer 只读 [-1]）
    coeff_rescale_sf = _optional_int(layer_field_values["gelu_coeff_mul_rescale_sf_0"])
    if deg <= 0:
        gelu_coeff_mul_rescale_sfs: Tuple[Optional[int], ...] = ()
    else:
        gelu_coeff_mul_rescale_sfs = tuple([None] * (deg - 1) + [coeff_rescale_sf])
    # x_centered_fresh / inv_std_fresh 绑定 —— x_centered 主导（block5 SOURCE）
    x_centered_fresh_sf = int(layer_field_values["x_centered_fresh_sf"])
    # LN-tail rescales (normalize / gamma / wffn1) follow this degree's skeleton
    # (auto via SSOT): e.g. block5_n1 has no wffn1 rescale, gamma is never a
    # rescale on any block5 skeleton. gelu power/coeff stay degree-driven above.
    active = active_rescale_rl_fields(5, gelu_degree=deg, profile=profile)

    def _rsc(name: str) -> Optional[int]:
        return _optional_int(layer_field_values[name]) if name in active else None

    return Block5ActionSpec(
        gelu_degree=deg,
        # inv_std_fresh 绑定 = x_centered_fresh
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
    arr = np.asarray(action_vec, dtype=int).reshape(-1)

    expected_dim = len(action_dims_for_config(num_layers))
    if arr.size != expected_dim:
        raise ValueError(
            f"action_vec length {arr.size} != expected {expected_dim} (num_layers={num_layers})"
        )

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

        # 切出每个 block 的 action 子段
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

        # Block 1：首层 block1 不安装（用户语义：layer 0 没有上游 FFN2，第一个 HE
        # 配置无损 —— 与 Rescale_optimizer 对齐）。保留 action 向量槽位但不构造 cfg，
        # 这样下游 (bridge.apply / build_optimizer_requests) 自然跳过 layer 0 block1。
        if only_block in (None, 1):
            if li == 0:
                pass  # block1_cfgs 故意不写 layer 0，下游用 .get(0) / dict-not-in 判断
            else:
                b1 = _build_block1_action(li, layer_block_values[1], is_first_layer=False)
                block1_cfgs[li] = build_block1_cfg_from_action(
                    b1, N=_block_default_N(1, gelu_degree=li_gelu_degree, attn_degree=li_attn_degree),
                )
        # Block 2
        if only_block in (None, 2):
            b2 = _build_block2_action(li, layer_block_values[2])
            block2_cfgs[li] = build_block2_cfg_from_action(
                b2, N=_block_default_N(2, gelu_degree=li_gelu_degree, attn_degree=li_attn_degree),
            )
        # Block 3 (degree-aware)
        if only_block in (None, 3):
            b3 = _build_block3_action(li, layer_block_values[3], attn_degree=li_attn_degree)
            block3_cfgs[li] = build_block3_cfg_from_action(
                b3, N=_block_default_N(3, gelu_degree=li_gelu_degree, attn_degree=li_attn_degree),
            )
        # Block 4
        if only_block in (None, 4):
            b4 = _build_block4_action(li, layer_block_values[4])
            block4_cfgs[li] = build_block4_cfg_from_action(
                b4, N=_block_default_N(4, gelu_degree=li_gelu_degree, attn_degree=li_attn_degree),
            )
        # Block 5 (gelu degree-aware)
        if only_block in (None, 5):
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
            continue  # R idx0 = drop — never enumerable
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


# Compat-extra slots: kept in the action vector for registry/back-compat but the
# cfg path forces these fields to None (or binds them to another slot), so the
# RL action value at these slots does NOT affect the noise installed in the
# model nor the optimizer's modulus-chain math. ``describe_action_vector`` /
# ``effective_action_hash`` use this to mark them inactive.
_COMPAT_EXTRA_FIELDS: Dict[int, frozenset] = {
    # BINDINGS + BOUND rescale slots only. Whether a FREE rescale slot is
    # effective now follows the live skeleton (skeleton_stage_map active set, via
    # the "R"-kind check in _is_action_field_effective), so free rescale slots
    # are no longer hard-listed here and a skeleton regen auto-updates them.
    1: frozenset(),
    2: frozenset({
        # Q-side encodes bound to K-side; Wv has no RO node; x_centered bound to
        # inv_std ("x2" side of ctct_x_mean_over_std).
        "wq_sf", "q_mask1_sf", "q_mask2_sf", "wv_sf", "x_centered_fresh_sf",
        # Q-side / Wq / Wv rescales are bound to K-side or non-existent — never a
        # free slot regardless of skeleton (the q_mask*_r values mirror kt_mask*_r).
        "wq_rescale_sf", "wv_rescale_sf", "q_mask1_rescale_sf", "q_mask2_rescale_sf",
    }),
    3: frozenset(),
    4: frozenset({
        # v_mask is bound to softmax_out_mask (one shared mask2 node) — encode + rescale.
        "v_mask_sf", "v_mask_rescale_sf",
    }),
    5: frozenset({
        # inv_std_fresh bound to x_centered_fresh ("x2" side of ctct_xmean_over_std).
        "inv_std_fresh_sf",
    }),
}


# ---------------------------------------------------------------------------
# Skeleton-driven active rescale slots (consumes skeleton_stage_map SSOT so a
# skeleton regen auto-changes which rescale slots are active / installed).
# ---------------------------------------------------------------------------
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
            import os
            import json
            from . import skeleton_stage_map as _ssm
            ro_root = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "Rescale_optimizer",
            )
            arch_path = os.path.join(ro_root, "configs", "mrpc", "static_skeletons_mrpc.json")
            with open(arch_path, "r", encoding="utf-8") as f:
                archive = json.load(f)
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
    # Only mrpc has skeleton data; block1/2 graph keys embed the profile, so a
    # display-time profile like "default" falls back to the mrpc graph.
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
    if str(field_name) in _COMPAT_EXTRA_FIELDS.get(int(block_idx), frozenset()):
        return False, (
            "compat-extra slot retained for action-vector back-compat; cfg field "
            "is forced None / bound elsewhere so this action value has no effect"
        )
    # Rescale "R" slots: effective iff the CURRENT skeleton selects this rescale
    # point (skeleton_stage_map active set). This replaces the old per-block
    # degree gates (block3 square / block5 power / block5 coeff-mul) with one
    # skeleton-driven rule, so report effectiveness auto-follows a regen. Bound
    # rescale slots are already filtered above by _COMPAT_EXTRA_FIELDS.
    if _field_kind(int(block_idx), str(field_name)) == "R":
        active = active_rescale_rl_fields(
            int(block_idx), gelu_degree=int(gelu_degree),
            attn_degree=int(attn_degree), profile=str(profile))
        if str(field_name) not in active:
            return False, "not a rescale stage on the current skeleton"
    # degree 0 = ReLU：block5_n0 graph 无 GELU 多项式系数 encode（ctpt_gelu_coeff）。
    # 该 slot 的动作既不进模型噪声（ReLU 跳过 GELU 安装）也不进 optimizer cost，故无效。
    if int(block_idx) == 5 and str(field_name) == "gelu_coeff_sf" and int(gelu_degree) == 0:
        return False, "GELU degree 0 (ReLU) has no polynomial coefficient encode"
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
                profile=str(profile),
            )
            # Layer-0 block-1 noise is *not installed* (the first HE config is
            # treated as lossless). The decoded ``value`` for these slots is a
            # meaningless artifact of the default max-SF table — clear it so
            # value-based queries (e.g. baseline-decode tests that build a
            # ``{(layer, block, field): value}`` map and assert "L0B1 is not
            # present") agree with the install-time semantics.
            if int(li) == 0 and int(block_idx) == 1:
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
    """生成 baseline 动作向量：SF 字段取最高档，K 取该 block 的 baseline K 值。

    Per-block baseline K（user-tuned 2026-05-15）：
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


def _gather_effective_k_values_in_action(
        action_vec: np.ndarray,
        num_layers: int,
        ) -> List[int]:
    arr = np.asarray(action_vec, dtype=int).reshape(-1)
    layer_dim = len(layer_dims())
    k_positions: List[int] = []
    cursor_in_layer = 0
    for b in (1, 2, 3, 4, 5):
        spec = _BLOCK_SPECS[b]
        for _fname, kind, _max in spec.fields:
            if kind == "K":
                k_positions.append(cursor_in_layer)
            cursor_in_layer += 1
    ks: List[int] = []
    for li in range(int(num_layers)):
        # 首层 Block 1 K 被强制 None；其它 block 的 K 仍生效
        for j, p in enumerate(k_positions):
            if li == 0 and j == 0:
                continue
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
    ks = _gather_effective_k_values_in_action(action_vec, num_layers)
    if not ks:
        return 0.0
    return float(np.mean(ks))


def sum_truncation_k_in_action(
        action_vec: np.ndarray,
        num_layers: int,
        ) -> int:
    """Sum of decoded truncation-k values across all effective K slots.

    Lets cost-matched random samplers do an integer equality pre-filter
    before paying for a Rescale_optimizer call (mean comparisons would
    need an explicit tolerance).
    """
    return int(sum(_gather_effective_k_values_in_action(action_vec, num_layers)))


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
        if int(block_idx) == 3:
            # block 3 (softmax exp approx) is frozen and fully excluded (2026-06-03):
            # not RL-decided, not installed (bridge skips it), and not sent to
            # Rescale_optimizer so the baseline (which also drops block 3) and the
            # action cost stay consistent without it.
            continue
        for layer_idx, cfg in layer_cfgs.items():
            if int(block_idx) == 1 and int(layer_idx) == 0:
                # 语义对齐：layer-0 block1 不发给 RO。
                continue
            cn = make_config_name(profile, block_idx, int(layer_idx), cfg=cfg)
            out[cn] = (str(block_name), cfg)
    return out
