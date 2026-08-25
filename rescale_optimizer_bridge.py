"""Materialize BLB configs through the in-process Rescale optimizer.

The bridge owns conversion to ``t_new`` and ``delta_overrides`` and returns
fusion count, total modulus bits, and chain validity. Reward construction stays
with the Stage-2 environment.
"""
from __future__ import annotations

import copy
import json
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable

from function_handler import (
    Block1NoiseConfig,
    Block2NoiseConfig,
    Block3NoiseConfig,
    Block4NoiseConfig,
    Block5NoiseConfig,
)


@dataclass
class RescaleOptimizerOutput:
    """Rescale_optimizer 单个 block 的解析输出（RL reward 侧用）。

    三个核心字段直接来自原始 JSON：
      * ``fusion_count``   = raw["fusion_count"]
      * ``total_bits``     = raw["result"]["chain"]["total_bits"]（链合法时；
                             链不合法时取 raw["result"].get("invalid_chain",{}).get("total_bits", 0)
                             或 0，由 ``_parse`` 决定）
      * ``invalid_chain``  = raw["result"]["invalid_chain"]
                             None ⇒ 链合法；dict ⇒ 不合法 + 原因
      * ``valid``          = (invalid_chain is None) and raw.get("valid", True)

    ``raw`` 保留完整 JSON，便于业务侧拿其它字段做更复杂 reward。
    """
    config_name: str
    fusion_count: int
    total_bits: int
    invalid_chain: Optional[dict]
    valid: bool
    raw: dict

    def to_signal_dict(self) -> dict:
        """打包成"基础奖励信号 dict"，让业务侧组合成最终 reward。"""
        return {
            "config_name": self.config_name,
            "valid": self.valid,
            "fusion_count": int(self.fusion_count),
            "total_bits": int(self.total_bits),
            "invalid_chain": self.invalid_chain,
        }


def _parse_optimizer_raw(raw: dict, *, config_name: str) -> RescaleOptimizerOutput:
    """Extract optimizer signals and fail closed on malformed output."""
    if not isinstance(raw, Mapping):
        return RescaleOptimizerOutput(
            config_name=config_name, fusion_count=0, total_bits=0,
            invalid_chain={"reason": "raw_not_dict", "raw_type": type(raw).__name__},
            valid=False, raw={"_invalid": True},
        )

    fusion_count = int(raw.get("fusion_count", 0))
    result = raw.get("result") or {}
    invalid_chain = result.get("invalid_chain")
    valid = (invalid_chain is None) and bool(raw.get("valid", True)) and bool(result.get("valid", True))

    chain = result.get("chain") or {}
    total_bits = chain.get("total_bits")
    if total_bits is None:

        if isinstance(invalid_chain, Mapping):
            total_bits = invalid_chain.get("total_bits", 0)
        else:
            total_bits = 0
    total_bits = int(total_bits)

    return RescaleOptimizerOutput(
        config_name=str(config_name),
        fusion_count=fusion_count,
        total_bits=total_bits,
        invalid_chain=invalid_chain if isinstance(invalid_chain, Mapping) or invalid_chain is None else {"raw": invalid_chain},
        valid=valid,
        raw=dict(raw),
    )


@runtime_checkable
class RescaleOptimizerInvoker(Protocol):
    """Invoke one named graph with Rescale override inputs."""
    def __call__(self, config_name: str, delta_overrides: dict) -> dict: ...


_BASELINE_ARCHIVE_CACHE: Dict[
    Tuple[str, int, int],
    Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]],
] = {}


def _clone_optimizer_payload(value: Any) -> Any:
    """Clone JSON-like optimizer output without generic deepcopy bookkeeping."""
    if isinstance(value, dict):
        return {key: _clone_optimizer_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_optimizer_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_optimizer_payload(item) for item in value)
    if isinstance(value, set):
        return {_clone_optimizer_payload(item) for item in value}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return copy.deepcopy(value)


def _clone_baseline_archive(
        cached: Mapping[str, Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]],
) -> Dict[str, Tuple[List[int], List[int], List[int]]]:
    return {
        str(name): (list(skeleton), list(t_baseline), list(q_bits_baseline))
        for name, (skeleton, t_baseline, q_bits_baseline) in cached.items()
    }


def load_baseline_archive(path: str) -> Dict[str, Tuple[List[int], List[int], List[int]]]:
    """读 ``static_skeletons_<profile>.json`` → ``{config_name: (skeleton, t_baseline, q_bits_baseline)}``。

    schema v2 (``cut_point_sf`` / ``modulus_chain.drop_order``)：
      * skeleton: ``entry["skeleton"]``
      * t_baseline: 按 skeleton 取 cut_point_sf 的 ``sf_post`` (rescale 点) 或 ``sf`` (source/普通点)
      * q_bits_baseline: ``modulus_chain.drop_order[1:-1]``（去掉首尾 head/tail prime）
    """
    abs_path = os.path.abspath(str(path))
    stat = os.stat(abs_path)
    cache_key = (abs_path, int(stat.st_mtime_ns), int(stat.st_size))
    cached = _BASELINE_ARCHIVE_CACHE.get(cache_key)
    if cached is not None:
        return _clone_baseline_archive(cached)

    with open(abs_path, "r", encoding="utf-8") as f:
        archive = json.load(f)
    out: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]] = {}
    for entry in archive.get("results", []):
        if not entry.get("success"):
            continue
        cname = str(entry["config_name"])
        skel = [int(x) for x in entry.get("skeleton", [])]

        t_for_idx: Dict[int, int] = {}
        for row in entry.get("cut_point_sf", []):
            i = int(row["i"])
            if "sf_post" in row:
                t_for_idx[i] = int(row["sf_post"])
            elif "sf" in row:
                t_for_idx[i] = int(row["sf"])
        t_base = [t_for_idx[i] for i in skel if i in t_for_idx]

        mc = entry.get("modulus_chain", {}) or {}
        drop_order = list(mc.get("drop_order", []))
        q_base = [int(x) for x in drop_order[1:-1]] if len(drop_order) >= 2 else []
        out[cname] = (tuple(skel), tuple(t_base), tuple(q_base))
    _BASELINE_ARCHIVE_CACHE[cache_key] = out
    return _clone_baseline_archive(out)


class InProcessInvoker:
    """Real in-process Rescale_optimizer invoker.

    Thin adapter around ``rescale_optimizer.ReplanSession``: preserves this
    bridge's invoker surface (``__call__(graph_key, payload) -> dict``,
    ``baselines`` property in tuple form, ``from_profile`` classmethod) while
    delegating all graph / baseline / replan work to ``ReplanSession``.

    Usage:

        inv = InProcessInvoker.from_profile(
            rescale_optimizer_root="Rescale_optimizer",
            profile="mrpc",
        )
        bridge = RescaleOptimizerBridge(invoker=inv)

    Manual config map:

        inv = InProcessInvoker(
            configs={
                "block1_mrpc": "Rescale_optimizer/configs/mrpc/block1_mrpc.json",
                ...,
            },
            baseline_archive="Rescale_optimizer/configs/mrpc/static_skeletons_mrpc.json",
            rescale_optimizer_root="Rescale_optimizer",
        )
    """

    def __init__(
            self,
            *,
            configs: Mapping[str, str],
            baseline_archive: str,
            rescale_optimizer_root: Optional[str] = None,
            ):
        if rescale_optimizer_root:
            root = os.path.abspath(rescale_optimizer_root)
            if root not in sys.path:
                sys.path.insert(0, root)


        from rescale_optimizer import ReplanSession

        self._session = ReplanSession(
            configs={str(k): str(v) for k, v in configs.items()},
            baseline_archive=str(baseline_archive),
        )

    @classmethod
    def from_profile(
            cls,
            *,
            rescale_optimizer_root: str,
            profile: str,
            configs_dir: Optional[str] = None,
            baseline_archive: Optional[str] = None,
            include: Optional[List[str]] = None,
            ) -> "InProcessInvoker":
        """Scan ``configs/<profile>/*.json`` and register every graph with a
        successful baseline entry."""
        root = os.path.abspath(str(rescale_optimizer_root))
        if root not in sys.path:
            sys.path.insert(0, root)
        from rescale_optimizer import ReplanSession

        inv = cls.__new__(cls)
        inv._session = ReplanSession.from_profile(
            profile=str(profile),
            root=root,
            configs_dir=configs_dir,
            baseline_archive=baseline_archive,
            include=include,
        )
        return inv

    @property
    def session(self):
        """Underlying ``ReplanSession`` — exposed for tools that need richer API."""
        return self._session

    @property
    def baselines(self) -> Dict[str, Tuple[List[int], List[int], List[int]]]:
        """``{graph_key: (skeleton, t_baseline, q_bits_baseline)}`` — tuple form
        for backward compatibility with bridge code that iterates baselines."""
        return {
            k: (list(rec.skeleton), list(rec.t_baseline), list(rec.q_bits_baseline))
            for k, rec in self._session.baselines.items()
        }

    @property
    def archive_entries(self) -> Dict[str, dict]:
        """``{graph_key: archive_entry}`` — the raw static_skeletons entry per
        graph (carries ``cut_point_sf`` with names + sf_post). Lets the bridge
        derive t_new from the actual skeleton via ``skeleton_stage_map``."""
        return {
            k: dict(rec.archive_entry)
            for k, rec in self._session.baselines.items()
            if getattr(rec, "archive_entry", None) is not None
        }

    def __call__(self, config_name: str, payload: Any) -> dict:
        return self._session(str(config_name), payload)

    def replan_variables(
            self,
            config_name: str,
            *,
            t_new: Optional[Sequence[int]] = None,
            delta_overrides: Optional[Mapping[str, Any]] = None,
            ) -> dict:
        """Direct variable API for the RL hot path.

        ``__call__`` stays as the compatibility surface for callers that still
        build a JSON-shaped payload.  Sequential RL already runs in-process, so
        this method avoids the extra payload construction/splitting layer and
        hands the decoded action variables straight to ``ReplanSession``.
        """
        return self._session.replan(
            str(config_name),
            t_new=([int(x) for x in t_new] if t_new is not None else None),
            delta_overrides=delta_overrides,
            return_dict=True,
        )


def build_rescale_invoker(
        *,
        root: str,
        profile: str,
        baseline_archive: Optional[str] = None,
        ) -> InProcessInvoker:
    """Build the only supported production Rescale invoker."""
    invoker = InProcessInvoker.from_profile(
        rescale_optimizer_root=os.fspath(root),
        profile=str(profile),
        baseline_archive=baseline_archive,
    )
    if not invoker.baselines:
        raise RuntimeError(f"no Rescale baselines loaded for profile={profile!r}")
    return invoker


def default_block1_cfg_to_delta(cfg: Block1NoiseConfig) -> Dict[str, Union[int, str]]:
    """Block 1 默认映射 —— 与 ``configs/<profile>/block1_<profile>.json`` 节点对齐。

    实际节点（schema v2）：
      * ctpt_ffn2 (CTPT_MUL)        ← Wffn2·X
      * ctpt_inv_d_1 (CTPT_MUL)     ← μ 的 1/D
      * ctct_ext_square (CTCT_MUL)  ← (X−μ)²，固定 "x2"
      * ctpt_inv_d_2 (CTPT_MUL)     ← var 的 1/D
    """
    return {
        "ctpt_ffn2":      int(cfg.wffn2_encode.scaling_factor),
        "ctpt_inv_d_1":   int(cfg.mean_inv_d_encode.scaling_factor),
        "ctct_ext_square": "x2",
        "ctpt_inv_d_2":   int(cfg.var_inv_d_encode.scaling_factor),
    }


def default_block2_cfg_to_delta(cfg: Block2NoiseConfig) -> Dict[str, Union[int, str]]:
    """Block 2 默认映射 —— 与 ``block2_<profile>.json`` 节点对齐。

    实际节点：
      * ctct_x_mean_over_std (CTCT_MUL) ← (X−μ)·(1/std)，固定 "x2"
      * ctpt_gama1 (CTPT_MUL)           ← γ
      * ctpt_wq_wk (CTPT_MUL)           ← **Q/K 共享**一个节点
      * ctpt_rotKT_mask1 (CTPT_MUL)     ← K^T BSGS mask 1
      * ctpt_rotKT_mask2 (CTPT_MUL)     ← K^T BSGS mask 2
      * ctct_preprocess_qkt (CTCT_MUL)  ← Q·K^T，固定 "x2"
      * ctpt_mask (CTPT_MUL)            ← 合并 Q,K mask

    2026-05-14 起 BLB Stage-2 RL 的 Q 侧动作（wq_sf / q_mask1_sf / q_mask2_sf）
    与 K 侧绑定 —— ``_build_block2_action`` 用 K 侧的 SF 同时填 cfg 的 Q/K 字段，
    所以这里 ``ctpt_wq_wk`` 直接读 ``cfg.wk_encode.scaling_factor``（语义上"由 K
    侧控制"）。``wk_encode == wq_encode`` 始终成立。BLB cfg 的 ``wv_encode``
    在该 graph 没有对应节点，被丢弃（仅影响模型噪声）。
    """
    return {
        "ctct_x_mean_over_std": "x2",
        "ctpt_gama1":          int(cfg.gamma_encode.scaling_factor),
        "ctpt_wq_wk":          int(cfg.wk_encode.scaling_factor),
        "ctpt_rotKT_mask1":    int(cfg.kt_mask1_encode.scaling_factor),
        "ctpt_rotKT_mask2":    int(cfg.kt_mask2_encode.scaling_factor),
        "ctct_preprocess_qkt": "x2",
        "ctpt_mask":           int(cfg.qkt_merge_mask_encode.scaling_factor),
    }


def default_block3_cfg_to_delta(cfg: Block3NoiseConfig) -> Dict[str, Union[int, str]]:
    """Block 3 默认映射 —— 与 ``block3_exp_n<degree>.json`` 节点对齐。

    实际节点：
      * ctpt_inv_2n (CTPT_MUL)             ← 1/2^n
      * ctct_square_1 .. ctct_square_<n> (CTCT_MUL) ← 迭代平方，固定 "x2"
    """
    deltas: Dict[str, Union[int, str]] = {
        "ctpt_inv_2n": int(cfg.inv_2n_encode.scaling_factor),
    }
    for k in range(int(cfg.degree)):
        deltas[f"ctct_square_{k+1}"] = "x2"
    return deltas


def default_block4_cfg_to_delta(cfg: Block4NoiseConfig) -> Dict[str, Union[int, str]]:
    """Block 4 默认映射 —— 与 ``block4.json`` 节点对齐。

    实际节点（注意只有 2 个 mask 节点；BLB cfg 的 3 个 mask encode 中
    ``softmax_out_mask`` / ``v_mask`` 被 graph 合成一个 ``ctpt_mask2``，
    我们取 ``softmax_out_mask`` 的 SF）：
      * ctpt_mask2 (CTPT_MUL)               ← softmax 输出 / V 路径上的 mask
      * ctct_rot_softmax_mul_v (CTCT_MUL)   ← softmax×V matmul，delta = SF(v) + SF(mask2)
      * ctpt_mask (CTPT_MUL)                ← 合并 softmax×V 后的 mask
      * ctpt_wo_attnout (CTPT_MUL)          ← Wo
      * ctpt_inv_d_1 (CTPT_MUL)             ← post-attn LN μ 的 1/D
      * ctct_square (CTCT_MUL)              ← post-attn LN (X−μ)²，固定 "x2"
      * ctpt_inv_d_2 (CTPT_MUL)             ← post-attn LN var 的 1/D

    2026-05-20 user spec：``ctct_rot_softmax_mul_v`` 的 delta 之前是硬编码 39
    （mrpc baseline 值），现在根据 cfg 动态计算成
    ``SF(v_fresh) + SF(v_mask_encode)``。CKKS 里两个密文相乘的累积 SF 是各自
    SF 之和，所以 v * mask2 这一步的 SF 就是 SF(v) + SF(mask2)。``v_mask_encode``
    已经被 ``_build_block4_action`` 绑定到 ``softmax_out_mask_encode``（同一个
    mask2），所以无论用哪个读出来 SF 都一样；为了和 baseline_bootstrap 里
    ``v_fresh.sf = ctct_rot_softmax_mul_v.delta - ctpt_mask2.delta`` 的反推公式
    精确对称，我们在这里用 ``v_mask_encode`` 来对应 baseline 里的 mask2。
    """
    return {
        "ctpt_mask2":             int(cfg.softmax_out_mask_encode.scaling_factor),
        "ctct_rot_softmax_mul_v": (
            int(cfg.v_fresh.scaling_factor)
            + int(cfg.v_mask_encode.scaling_factor)
        ),
        "ctpt_mask":              int(cfg.softmax_v_mask_encode.scaling_factor),
        "ctpt_wo_attnout":        int(cfg.wo_encode.scaling_factor),
        "ctpt_inv_d_1":           int(cfg.ln_mean_inv_d_encode.scaling_factor),
        "ctct_square":            "x2",
        "ctpt_inv_d_2":           int(cfg.ln_var_inv_d_encode.scaling_factor),
    }


def default_block5_cfg_to_delta(cfg: Block5NoiseConfig) -> Dict[str, Union[int, str]]:
    """Block 5 默认映射 —— 与 ``block5_n<degree>.json`` 节点对齐。

    实际节点（按 GELU degree 不同，graph 含的 ctct_gelu_* 也不同）：
      * ctct_xmean_over_std (CTCT_MUL) ← post-attn (X−μ)·(1/std)
      * ctpt_gamal (CTPT_MUL)          ← γ
      * ctpt_wffn1 (CTPT_MUL)          ← W_ffn1·X
      * (degree==1：无 ctct_gelu_*)
      * (degree==2：ctct_gelu_x2)
      * (degree==4：ctct_gelu_x2 + ctct_gelu_x4；**不含 ctct_gelu_x3**，
        因为 graph 把 x^3 直接折进 x^4)
      * ctpt_gelu_coeff (CTPT_MUL)     ← 多项式系数
    """
    deltas: Dict[str, Union[int, str]] = {
        "ctct_xmean_over_std": "x2",
        "ctpt_gamal":          int(cfg.gamma_encode.scaling_factor),
        "ctpt_wffn1":          int(cfg.wffn1_encode.scaling_factor),
    }
    if cfg.gelu_degree >= 2:
        deltas["ctct_gelu_x2"] = "x2"
    if cfg.gelu_degree >= 4:
        deltas["ctct_gelu_x4"] = "x2"
    if int(cfg.gelu_degree) >= 1:


        deltas["ctpt_gelu_coeff"] = int(cfg.gelu_coeff_encode.scaling_factor)
    return deltas


GRAPH_NODE_TO_CFG_ATTR: Dict[int, Dict[str, str]] = {
    1: {
        "ctpt_ffn2":    "wffn2_encode",
        "ctpt_inv_d_1": "mean_inv_d_encode",
        "ctpt_inv_d_2": "var_inv_d_encode",
    },
    2: {
        "ctpt_gama1":       "gamma_encode",

        "ctpt_wq_wk":       "wk_encode",
        "ctpt_rotKT_mask1": "kt_mask1_encode",
        "ctpt_rotKT_mask2": "kt_mask2_encode",
        "ctpt_mask":        "qkt_merge_mask_encode",
    },
    3: {
        "ctpt_inv_2n": "inv_2n_encode",
    },
    4: {
        "ctpt_mask2":      "softmax_out_mask_encode",
        "ctpt_mask":       "softmax_v_mask_encode",
        "ctpt_wo_attnout": "wo_encode",
        "ctpt_inv_d_1":    "ln_mean_inv_d_encode",
        "ctpt_inv_d_2":    "ln_var_inv_d_encode",
    },
    5: {
        "ctpt_gamal":      "gamma_encode",
        "ctpt_wffn1":      "wffn1_encode",
        "ctpt_gelu_coeff": "gelu_coeff_encode",
    },
}


_DEFAULT_BLOCK_MAPPERS: Dict[str, CfgToDeltaFn] = {
    "block1": default_block1_cfg_to_delta,
    "block2": default_block2_cfg_to_delta,
    "block3": default_block3_cfg_to_delta,
    "block4": default_block4_cfg_to_delta,
    "block5": default_block5_cfg_to_delta,
}


@dataclass(frozen=True)
class _SkelEntry:
    """skeleton 上某个 stage 对应到 cfg 哪个字段。

    Args:
        cfg_field:   cfg 上的属性名（必须是 NoisePoint 或 Optional[NoisePoint]
                     或 Tuple[Optional[NoisePoint]]）
        tuple_index: 若 cfg_field 是 tuple，取第 N 项；None=该字段是单一 NoisePoint。
                     支持负索引（如 -1 表示 last 项）。
    """
    cfg_field: str
    tuple_index: Optional[int] = None


DEFAULT_CFG_TO_T_NEW_MAP: Dict[str, Tuple[_SkelEntry, ...]] = {


    "block1_mrpc": (
        _SkelEntry("gelu_out_fresh"),
        _SkelEntry("mean_result_rescale"),
        _SkelEntry("var_result_rescale"),
    ),


    "block2_mrpc": (
        _SkelEntry("inv_std_fresh"),
        _SkelEntry("gamma_result_rescale"),
        _SkelEntry("kt_mask1_result_rescale"),
        _SkelEntry("qkt_matmul_result_rescale"),
    ),


    "block3_exp_n2": (
        _SkelEntry("x_fresh"),
        _SkelEntry("square_rescales", 0),
        _SkelEntry("square_rescales", 1),
    ),
    "block3_exp_n3": (
        _SkelEntry("x_fresh"),
        _SkelEntry("square_rescales", 0),
        _SkelEntry("square_rescales", 1),
        _SkelEntry("square_rescales", 2),
    ),
    "block3_exp_n4": (
        _SkelEntry("x_fresh"),
        _SkelEntry("square_rescales", 0),
        _SkelEntry("square_rescales", 1),
        _SkelEntry("square_rescales", 2),
        _SkelEntry("square_rescales", 3),
    ),
    "block3_exp_n5": (
        _SkelEntry("x_fresh"),
        _SkelEntry("square_rescales", 0),
        _SkelEntry("square_rescales", 1),
        _SkelEntry("square_rescales", 2),
        _SkelEntry("square_rescales", 3),


        _SkelEntry("square_rescales", 3),
    ),
    "block3_exp_n6": (
        _SkelEntry("x_fresh"),
        _SkelEntry("square_rescales", 0),
        _SkelEntry("square_rescales", 1),
        _SkelEntry("square_rescales", 2),
        _SkelEntry("square_rescales", 3),
        _SkelEntry("square_rescales", 3),
        _SkelEntry("square_rescales", 3),
    ),


    "block4": (
        _SkelEntry("softmax_out_fresh"),
        _SkelEntry("softmax_v_matmul_rescale"),
        _SkelEntry("ln_mean_result_rescale"),
        _SkelEntry("ln_square_result_rescale"),
    ),


    "block5_n1": (
        _SkelEntry("x_centered_fresh"),
        _SkelEntry("normalize_result_rescale"),
        _SkelEntry("gelu_coeff_mul_rescales", -1),
    ),


    "block5_n2": (
        _SkelEntry("x_centered_fresh"),
        _SkelEntry("normalize_result_rescale"),
        _SkelEntry("wffn1_result_rescale"),
        _SkelEntry("gelu_coeff_mul_rescales", -1),
    ),


    "block5_n4": (
        _SkelEntry("x_centered_fresh"),
        _SkelEntry("normalize_result_rescale"),
        _SkelEntry("wffn1_result_rescale"),
        _SkelEntry("gelu_power_rescales", 0),
        _SkelEntry("gelu_coeff_mul_rescales", -1),
    ),
}


def _strip_layer_suffix(config_name: str) -> Tuple[str, Optional[int]]:
    """``"block1_mrpc_L0"`` → ``("block1_mrpc", 0)``；``"block1_mrpc"`` → ``("block1_mrpc", None)``。

    BLB Stage 2 RL 的 env 端会把每层独立编号成 ``"<graph_key>_L<i>"``；invoker
    端只关心 graph_key（每个 (block, profile) 共享一份 graph + baseline）。
    """
    name = str(config_name)
    if "_L" in name:
        head, _, tail = name.rpartition("_L")
        try:
            return head, int(tail)
        except ValueError:
            return name, None
    return name, None


def _extract_sf_from_cfg(cfg: Any, entry: _SkelEntry) -> Optional[int]:
    """按 ``_SkelEntry`` 从 cfg 里抽出 NoisePoint 的 ``scaling_factor``。

    None ⇒ 字段不存在 / 值为 None / tuple 越界。
    """
    attr = getattr(cfg, entry.cfg_field, None)
    if attr is None:
        return None
    if entry.tuple_index is not None:
        try:
            attr = attr[entry.tuple_index]
        except (IndexError, TypeError, KeyError):
            return None
        if attr is None:
            return None
    sf = getattr(attr, "scaling_factor", None)
    if sf is None:
        return None
    try:
        return int(sf)
    except (TypeError, ValueError):
        return None


def cfg_to_t_new_from_table(
        graph_key: str,
        cfg: Any,
        *,
        baseline_t_new: Optional[Sequence[int]] = None,
        table: Optional[Mapping[str, Sequence[_SkelEntry]]] = None,
        ) -> Optional[List[int]]:
    """从 cfg 自动派生 ``t_new``。

    返回 None ⇒ 当前 ``graph_key`` 在表里没有映射（caller 应让 invoker fallback
    到 baseline）。

    返回 list[int] ⇒ 全部 stage 都成功取到 SF（或 baseline_t_new 提供了缺位的
    fallback）。

    规则：
      * 如果某 stage 的 cfg 字段是 None（RL 没启用该 rescale）且 baseline_t_new
        里有对应位置 ⇒ 用 baseline_t_new[r] 填位。
      * 如果某 stage 的 cfg 字段是 None 且没有 baseline ⇒ 整体放弃，返回 None
        （让 invoker 用 baseline）。
    """
    tbl = table or DEFAULT_CFG_TO_T_NEW_MAP
    entries = tbl.get(str(graph_key))
    if not entries:
        return None
    out: List[int] = []
    for r, ent in enumerate(entries):
        sf = _extract_sf_from_cfg(cfg, ent)
        if sf is None:
            if baseline_t_new is not None and r < len(baseline_t_new):
                out.append(int(baseline_t_new[r]))
            else:
                return None
        else:
            out.append(int(sf))
    return out


def _derive_t_new_table_from_invoker(invoker: Any) -> Dict[str, Tuple[_SkelEntry, ...]]:
    """Derive ``{graph_key: (_SkelEntry, ...)}`` from the invoker's REAL skeletons.

    Uses ``invoker.archive_entries`` (the static_skeletons cut_point_sf per graph)
    + ``skeleton_stage_map`` so the t_new ordering follows whatever the current
    skeleton selects. Returns ``{}`` when the invoker has no archive entries.
    Graphs with an unmapped rescale node are skipped
    (fall back to the static table for those).
    """
    try:
        archive = invoker.archive_entries
    except Exception:
        archive = None
    if not archive:
        return {}
    try:
        from blb_stage2_rl import skeleton_stage_map as _ssm
    except Exception:
        return {}
    out: Dict[str, Tuple[_SkelEntry, ...]] = {}
    for gk, plan in _ssm.build_stage_plans(dict(archive)).items():
        if plan.unmapped_rescale_nodes:
            continue
        entries = tuple(
            _SkelEntry(cf, ti) for (cf, ti) in plan.t_new_entries if cf is not None
        )
        if len(entries) == len(plan.t_new_entries):
            out[gk] = entries
    return out


@dataclass
class _BlockRequest:
    config_name: str
    cfg: Any
    block_name: str


class RescaleOptimizerBridge:
    """Convert BLB configs, invoke each graph, and parse optimizer output."""

    def __init__(
            self,
            invoker: RescaleOptimizerInvoker,
            *,
            cfg_to_delta_overrides: Optional[Mapping[str, CfgToDeltaFn]] = None,
            cfg_to_t_new_overrides: Optional[Mapping[str, Sequence[_SkelEntry]]] = None,
            auto_t_new_from_cfg: bool = True,
            cache_max_entries: int = 50000,
            ):
        """构造 bridge。

        Args:
            invoker:                    in-process ``RescaleOptimizerInvoker``
            cfg_to_delta_overrides:     ``{block_name: fn(cfg) -> delta_overrides_dict}``，
                                        覆盖默认 ``default_block{1..5}_cfg_to_delta``
            cfg_to_t_new_overrides:     ``{graph_key: tuple[_SkelEntry, ...]}``，扩展或覆盖
                                        ``DEFAULT_CFG_TO_T_NEW_MAP``。可用于支持 mrpc 之外的
                                        profile key.
            auto_t_new_from_cfg:        默认 ``True`` ⇒ 当 ``evaluate(t_new=None)`` 时
                                        自动从 cfg 派生 t_new；False ⇒ 保持旧行为
                                        （t_new=None ⇒ invoker fallback 到 baseline）。
            cache_max_entries:          LRU cache size for deterministic optimizer
                                        calls. Sequential RL repeats many per-block
                                        action tuples; caching avoids recomputing the
                                        same ReplanSession result dozens of times.
        """
        self.invoker = invoker

        self._cfg_mappers: Dict[str, CfgToDeltaFn] = dict(_DEFAULT_BLOCK_MAPPERS)
        if cfg_to_delta_overrides:
            for k, fn in cfg_to_delta_overrides.items():
                self._cfg_mappers[str(k)] = fn


        self._cfg_to_t_new_table: Dict[str, Tuple[_SkelEntry, ...]] = {
            k: tuple(v) for k, v in DEFAULT_CFG_TO_T_NEW_MAP.items()
        }
        for gk, entries in _derive_t_new_table_from_invoker(invoker).items():
            self._cfg_to_t_new_table[gk] = entries
        if cfg_to_t_new_overrides:
            for k, v in cfg_to_t_new_overrides.items():
                self._cfg_to_t_new_table[str(k)] = tuple(v)
        self.auto_t_new_from_cfg = bool(auto_t_new_from_cfg)
        self.cache_max_entries = max(0, int(cache_max_entries))
        self._eval_cache: "OrderedDict[Tuple[Any, ...], dict]" = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0

    def register_cfg_to_delta_overrides(self, block_name: str, fn: CfgToDeltaFn) -> None:
        """业务侧动态注册 / 覆盖某个 block 的 cfg → delta 转换函数。"""
        self._cfg_mappers[str(block_name)] = fn

    def register_cfg_to_t_new(self, graph_key: str, entries: Sequence[_SkelEntry]) -> None:
        """业务侧动态注册 / 覆盖某个 ``graph_key`` 的 skeleton→cfg 字段映射。"""
        self._cfg_to_t_new_table[str(graph_key)] = tuple(entries)

    def _lookup_baseline_t_new(self, graph_key: str) -> Optional[List[int]]:
        """如果 invoker 暴露 ``baselines`` 属性（``InProcessInvoker`` 有），就读
        baseline t_new 用于 cfg-derived t_new 的 fallback；否则返回 None。"""
        baselines = getattr(self.invoker, "baselines", None)
        if not isinstance(baselines, Mapping):
            return None
        entry = baselines.get(str(graph_key))
        if not entry:
            return None

        try:
            return list(entry[1]) if entry[1] is not None else None
        except (IndexError, TypeError):
            return None

    def cfg_to_delta_overrides(self, block_name: str, cfg: Any) -> Dict[str, Union[int, str]]:
        """单独把 cfg 翻译成 delta_overrides；不调用优化器，仅做翻译。"""
        block_name = str(block_name)
        if block_name not in self._cfg_mappers:
            raise KeyError(
                f"未注册 block_name={block_name!r} 的 cfg→delta 映射；"
                f"已知 = {sorted(self._cfg_mappers.keys())}"
            )
        return dict(self._cfg_mappers[block_name](cfg))

    def _invoke_optimizer(
            self,
            *,
            graph_key: str,
            payload: Any,
            t_new: Optional[Sequence[int]],
            delta_overrides: Mapping[str, Union[int, str]],
            ) -> dict:
        direct = getattr(self.invoker, "replan_variables", None)
        if callable(direct):
            return direct(
                str(graph_key),
                t_new=(list(t_new) if t_new is not None else None),
                delta_overrides=dict(delta_overrides),
            )
        return self.invoker(str(graph_key), payload)

    def evaluate(
            self,
            *,
            config_name: str,
            block_name: str,
            cfg: Any,
            t_new: Optional[List[int]] = None,
            extra_overrides: Optional[Mapping[str, Union[int, str]]] = None,
            _borrow_cached_payload: bool = False,
            ) -> RescaleOptimizerOutput:
        """对单个 (config, block, cfg) 三元组跑一次优化器。

        Args:
            config_name: 配置名。可以是 baseline 的原始名（``"block1_mrpc"``、
                         ``"block3_exp_n4"``）或 RL 端按层加 ``_L<i>`` 后缀的形式
                         （``"block1_mrpc_L0"``）。后缀会被自动剥掉用作 graph key。
            block_name:  ``"block1"`` … ``"block5"``，决定 cfg→delta 用哪个 mapper。
            cfg:         ``Block{N}NoiseConfig`` 实例。
            t_new:       per-stage 新 SF（length = R+1，与 baseline skeleton 对齐）。
                         * **None + auto_t_new_from_cfg=True**（默认）⇒ 从 cfg 自动
                           派生 t_new（按 ``DEFAULT_CFG_TO_T_NEW_MAP``）。
                         * **None + auto_t_new_from_cfg=False** 或表里没有该 graph_key
                           ⇒ invoker 内部用 baseline ``t_baseline``。
            extra_overrides: 在默认 cfg→delta 翻译之上叠加 / 覆盖的节点 deltas。

        Returns:
            ``RescaleOptimizerOutput``（``config_name`` 字段保留 RL 端的原始值，
            含 ``_L<i>`` 后缀）。
        """

        graph_key, _layer_idx = _strip_layer_suffix(config_name)


        deltas = self.cfg_to_delta_overrides(block_name, cfg)
        if extra_overrides:
            deltas.update({str(k): v for k, v in extra_overrides.items()})


        effective_t_new: Optional[List[int]] = None
        if t_new is not None:
            effective_t_new = [int(x) for x in t_new]
        elif self.auto_t_new_from_cfg:
            baseline_t = self._lookup_baseline_t_new(graph_key)
            effective_t_new = cfg_to_t_new_from_table(
                graph_key, cfg,
                baseline_t_new=baseline_t,
                table=self._cfg_to_t_new_table,
            )


        if effective_t_new is not None:
            payload: Any = {
                "t_new": list(effective_t_new),
                "delta_overrides": deltas,
            }
        else:
            payload = deltas

        cache_key = (
            str(graph_key),
            str(block_name),
            tuple(int(x) for x in effective_t_new) if effective_t_new is not None else None,
            tuple(sorted((str(k), str(v)) for k, v in deltas.items())),
        )
        raw: dict
        if self.cache_max_entries > 0 and cache_key in self._eval_cache:
            self.cache_hits += 1
            cached = self._eval_cache.pop(cache_key)
            self._eval_cache[cache_key] = cached
            raw = (
                dict(cached)
                if _borrow_cached_payload
                else _clone_optimizer_payload(cached)
            )
            raw["_optimizer_cache_hit"] = True
        else:
            self.cache_misses += 1

            raw = self._invoke_optimizer(
                graph_key=graph_key,
                payload=payload,
                t_new=effective_t_new,
                delta_overrides=deltas,
            )
            if self.cache_max_entries > 0 and isinstance(raw, dict):
                self._eval_cache[cache_key] = _clone_optimizer_payload(raw)
                while len(self._eval_cache) > self.cache_max_entries:
                    self._eval_cache.popitem(last=False)
            if isinstance(raw, dict):
                raw = dict(raw)
                raw["_optimizer_cache_hit"] = False


        try:
            if isinstance(raw, dict) and effective_t_new is not None and "_t_new_used" not in raw:
                raw["_t_new_used"] = list(effective_t_new)
                raw["_t_new_source"] = (
                    "user_provided" if t_new is not None
                    else ("cfg_derived" if graph_key in self._cfg_to_t_new_table else "baseline")
                )
                raw["_graph_key"] = graph_key
            if isinstance(raw, dict):
                raw["_optimizer_cache_hits"] = int(self.cache_hits)
                raw["_optimizer_cache_misses"] = int(self.cache_misses)
        except Exception:
            pass


        return _parse_optimizer_raw(raw, config_name=config_name)

    def evaluate_readonly(
            self,
            *,
            config_name: str,
            block_name: str,
            cfg: Any,
            t_new: Optional[List[int]] = None,
            extra_overrides: Optional[Mapping[str, Union[int, str]]] = None,
            ) -> RescaleOptimizerOutput:
        """Evaluate for an internal consumer that treats ``output.raw`` as read-only.

        Public :meth:`evaluate` returns a recursively isolated cache payload
        because callers may mutate nested diagnostics. Canonical Stage-2
        materialization only reads those diagnostics, so it can avoid cloning
        the same large JSON tree on every cache hit. A fresh top-level mapping
        still carries the exact per-call cache counters and hit marker.
        """
        return self.evaluate(
            config_name=config_name,
            block_name=block_name,
            cfg=cfg,
            t_new=t_new,
            extra_overrides=extra_overrides,
            _borrow_cached_payload=True,
        )

    def evaluate_baseline(
            self,
            *,
            config_name: str,
            ) -> RescaleOptimizerOutput:
        graph_key, _layer_idx = _strip_layer_suffix(config_name)
        raw = self.invoker(graph_key, {})
        if isinstance(raw, dict):
            raw = dict(raw)
            raw.setdefault("_t_new_source", "optimizer_baseline")
            raw.setdefault("_graph_key", graph_key)
        return _parse_optimizer_raw(raw, config_name=config_name)

    def evaluate_baseline_blocks(
            self,
            requests: Mapping[str, Tuple[str, Any]],
            ) -> Dict[str, RescaleOptimizerOutput]:
        """Evaluate an all-max baseline through the cfg-derived action path.

        The optimizer's empty-payload baseline can differ from the BLB action
        decoder's all-max cfg.  Candidate ranking and reward calibration must
        not mix those conventions, so this batch helper now delegates to the
        same ``evaluate_blocks`` path used by ordinary actions.  The lower
        level ``evaluate_baseline`` method remains available for explicit
        diagnostic comparisons against the optimizer-native baseline.
        """
        return self.evaluate_blocks(requests)

    def evaluate_blocks(
            self,
            requests: Mapping[str, Tuple[str, Any]],
            *,
            t_new_per_config: Optional[Mapping[str, List[int]]] = None,
            extra_overrides: Optional[Mapping[str, Mapping[str, Union[int, str]]]] = None,
            ) -> Dict[str, RescaleOptimizerOutput]:
        """一次跑多个 config。

        Args:
            requests: ``{config_name: (block_name, cfg)}``，比如
                      ``{"block1_mrpc": ("block1", block1_cfg), ...}``
            t_new_per_config: ``{config_name: [int, ...]}``，可选；不传则对应
                              config 走 baseline t_new。
            extra_overrides: ``{config_name: {node: delta, ...}}``，可选

        Returns:
            ``{config_name: RescaleOptimizerOutput}``
        """
        outputs: Dict[str, RescaleOptimizerOutput] = {}
        for config_name, (block_name, cfg) in requests.items():
            xtra = (extra_overrides or {}).get(config_name)
            t_new = (t_new_per_config or {}).get(config_name)
            outputs[config_name] = self.evaluate(
                config_name=config_name,
                block_name=block_name,
                cfg=cfg,
                t_new=t_new,
                extra_overrides=xtra,
            )
        return outputs

    def evaluate_blocks_readonly(
            self,
            requests: Mapping[str, Tuple[str, Any]],
            *,
            t_new_per_config: Optional[Mapping[str, List[int]]] = None,
            extra_overrides: Optional[Mapping[str, Mapping[str, Union[int, str]]]] = None,
            ) -> Dict[str, RescaleOptimizerOutput]:
        """Read-only counterpart of :meth:`evaluate_blocks` for materialization."""
        outputs: Dict[str, RescaleOptimizerOutput] = {}
        for config_name, (block_name, cfg) in requests.items():
            xtra = (extra_overrides or {}).get(config_name)
            t_new = (t_new_per_config or {}).get(config_name)
            outputs[config_name] = self.evaluate_readonly(
                config_name=config_name,
                block_name=block_name,
                cfg=cfg,
                t_new=t_new,
                extra_overrides=xtra,
            )
        return outputs


@dataclass
class OptimizerRewardSignals:
    """Optimizer signals aggregated across one materialized action."""
    total_fusion_count: int
    total_bits_sum: int
    any_invalid: bool
    valid_block_count: int
    invalid_block_count: int
    per_config: Dict[str, dict] = field(default_factory=dict)
    invalid_chains: Dict[str, dict] = field(default_factory=dict)


def apply_rotation_flags_to_cfg(cfg: Any, rotation_flag_names) -> None:
    """把"开启的 rotation 候选点列表"应用到 cfg 上。

    cfg 上所有 ``rotation_after_*`` bool 字段：在 ``rotation_flag_names`` 里出现
    的置 True，其余统一置 False。这是和 Rescale_optimizer 输出对接的最小钩子 ──
    上层 canonical materialization 先用本模块的 per-block 权威名字表，把优化器
    输出的 ``effective_rotations`` 转换成 BLB 命名空间的 flag 名。

    Args:
        cfg:                Block*NoiseConfig 实例（任意 block）
        rotation_flag_names: iterable[str]，要置 True 的 ``rotation_after_*`` 字段名
    """
    enable = {str(n) for n in rotation_flag_names}
    for name in vars(cfg).keys():
        if name.startswith("rotation_after_"):
            setattr(cfg, name, name in enable)


DEFAULT_ROTATION_NAME_MAP_BY_BLOCK: Mapping[
    int, Mapping[str, Tuple[str, ...]]
] = {
    1: {
        "bs_rot_in_mul": ("rotation_after_gelu_out_fresh",),
        "gs_rot_in_mul": ("rotation_after_wffn2_rescale_a",),
        "rot_sum1": ("rotation_after_wffn2_rescale_b",),
        "rot_sum2": ("rotation_after_square_rescale",),
    },
    2: {
        "bs_rot": ("rotation_after_gamma_rescale",),
        "gs_rot": (
            "rotation_after_wq_rescale",
            "rotation_after_wk_rescale",
            "rotation_after_wv_rescale",
        ),
        "bs_rot_step1": (
            "rotation_after_q_mask1_rescale",
            "rotation_after_kt_mask1_rescale",
        ),
        "gs_rot_step1": (
            "rotation_after_q_mask2_rescale",
            "rotation_after_kt_mask2_rescale",
        ),
        "gs_rot_step3": ("rotation_after_qkt_matmul_rescale",),
    },
    3: {},
    4: {
        "rot_gs_step2": (
            "rotation_after_softmax_out_mask_rescale",
            "rotation_after_v_mask_rescale",
        ),
        "rot_st3": ("rotation_after_softmax_v_matmul_rescale",),
        "rot_ct_wo": ("rotation_after_softmax_v_mask_rescale",),
        "rot_pre_ctpt_invd_1": ("rotation_after_wo_rescale",),
        "rot_pre_ctpt_invd_2": ("rotation_after_ln_square_rescale",),
    },
    5: {
        "rot_bs_wffn1": ("rotation_after_gamma_rescale",),
        "rot_gs_wffn1": ("rotation_after_wffn1_rescale",),
        "rot_gs_after_wffn1": ("rotation_after_wffn1_rescale",),
    },
}


def default_rotation_name_map(
        block_idx: int,
        ) -> Mapping[str, Tuple[str, ...]]:
    """Return the canonical optimizer-node to model-flag mapping for a block."""
    return DEFAULT_ROTATION_NAME_MAP_BY_BLOCK.get(int(block_idx), {})


_BLOCK2_QK_BINDING_PAIRS = (
    ("wq_encode", "wk_encode"),
    ("q_mask1_encode", "kt_mask1_encode"),
    ("q_mask2_encode", "kt_mask2_encode"),
)


def sync_block2_qk_binding(cfg: Any) -> List[CfgOverrideEntry]:
    """Mirror K-side encode SFs onto their bound Q-side counterparts.

    Call this on a Block2NoiseConfig immediately after every cfg mutation that
    might have touched ``wk_encode`` / ``kt_mask{1,2}_encode`` (typically right
    after :func:`apply_optimizer_output_to_cfg`). Returns the override entries
    actually applied so callers can include them in the same diagnostic record
    as the optimizer-driven overrides.
    """
    overrides: List[CfgOverrideEntry] = []
    for q_name, k_name in _BLOCK2_QK_BINDING_PAIRS:
        q_point = getattr(cfg, q_name, None)
        k_point = getattr(cfg, k_name, None)
        if q_point is None or k_point is None:
            continue
        new_sf = int(getattr(k_point, "scaling_factor", 0))
        old_sf = int(getattr(q_point, "scaling_factor", 0))
        if old_sf == new_sf:
            continue
        q_point.scaling_factor = new_sf
        overrides.append(CfgOverrideEntry(
            cfg_attr=f"{q_name}.scaling_factor",
            graph_node=None,
            source="qk_binding_sync",
            old_value=old_sf,
            new_value=new_sf,
        ))
    return overrides


_BLOCK4_MASK2_BINDING_PAIRS = (
    ("v_mask_encode", "softmax_out_mask_encode"),
)


def sync_block4_v_mask_binding(cfg: Any) -> List[CfgOverrideEntry]:
    """Mirror softmax_out_mask_encode onto v_mask_encode (mask2 binding).

    Call this on a Block4NoiseConfig immediately after every cfg mutation that
    might have touched ``softmax_out_mask_encode`` (typically right after
    :func:`apply_optimizer_output_to_cfg`). The two encodes represent the same
    ``ctpt_mask2`` graph node and must stay synchronized for the model-side V
    noise install and the ``ctct_rot_softmax_mul_v`` delta computation to
    agree with the RL action.
    """
    overrides: List[CfgOverrideEntry] = []
    for dst_name, src_name in _BLOCK4_MASK2_BINDING_PAIRS:
        src_point = getattr(cfg, src_name, None)
        dst_point = getattr(cfg, dst_name, None)
        if src_point is None or dst_point is None:
            continue
        new_sf = int(getattr(src_point, "scaling_factor", 0))
        old_sf = int(getattr(dst_point, "scaling_factor", 0))
        if old_sf == new_sf:
            continue
        dst_point.scaling_factor = new_sf
        overrides.append(CfgOverrideEntry(
            cfg_attr=f"{dst_name}.scaling_factor",
            graph_node=None,
            source="v_mask_binding_sync",
            old_value=old_sf,
            new_value=new_sf,
        ))
    return overrides


_BLOCK2_AUX_FRESH_BINDING_PAIRS = (
    ("x_centered_fresh", "inv_std_fresh"),
)
_BLOCK5_AUX_FRESH_BINDING_PAIRS = (
    ("inv_std_fresh", "x_centered_fresh"),
)


def _mirror_noise_point_binding(
        cfg: Any,
        pairs: Sequence[Tuple[str, str]],
        source_label: str,
        ) -> List[CfgOverrideEntry]:
    """Generic ``cfg.<dst>.sf = cfg.<src>.sf`` mirror helper used by the
    aux-fresh / mask binding sync calls. Returns the override entries that
    were actually applied (skips the no-op same-SF case).
    """
    overrides: List[CfgOverrideEntry] = []
    for dst_name, src_name in pairs:
        src_point = getattr(cfg, src_name, None)
        dst_point = getattr(cfg, dst_name, None)
        if src_point is None or dst_point is None:
            continue
        new_sf = int(getattr(src_point, "scaling_factor", 0))
        old_sf = int(getattr(dst_point, "scaling_factor", 0))
        if old_sf == new_sf:
            continue
        dst_point.scaling_factor = new_sf
        overrides.append(CfgOverrideEntry(
            cfg_attr=f"{dst_name}.scaling_factor",
            graph_node=None,
            source=source_label,
            old_value=old_sf,
            new_value=new_sf,
        ))
    return overrides


def sync_block2_aux_fresh_binding(cfg: Any) -> List[CfgOverrideEntry]:
    """Mirror inv_std_fresh.sf onto x_centered_fresh.sf (block 2 "x2" binding).

    Call after :func:`apply_optimizer_output_to_cfg` for any block-2 cfg.
    """
    return _mirror_noise_point_binding(
        cfg, _BLOCK2_AUX_FRESH_BINDING_PAIRS, "aux_fresh_binding_sync",
    )


def sync_block5_aux_fresh_binding(cfg: Any) -> List[CfgOverrideEntry]:
    """Mirror x_centered_fresh.sf onto inv_std_fresh.sf (block 5 "x2" binding).

    Call after :func:`apply_optimizer_output_to_cfg` for any block-5 cfg.
    """
    return _mirror_noise_point_binding(
        cfg, _BLOCK5_AUX_FRESH_BINDING_PAIRS, "aux_fresh_binding_sync",
    )


@dataclass(frozen=True)
class CfgOverrideEntry:
    """Single change applied to one cfg attribute, for diagnostics / HTML report."""
    cfg_attr: str
    graph_node: Optional[str]
    source: str
    old_value: Any
    new_value: Any


def _get_cfg_field(cfg: Any, name: str) -> Any:
    return getattr(cfg, name, None)


def _set_noise_point_sf(cfg: Any, field_name: str, new_sf: int) -> Tuple[Any, Any]:
    """Set ``cfg.<field_name>.scaling_factor = new_sf``. Returns (old_sf, new_sf)."""
    point = getattr(cfg, field_name, None)
    if point is None:
        return (None, None)
    old = int(getattr(point, "scaling_factor", 0))
    point.scaling_factor = int(new_sf)
    return (old, int(new_sf))


def _set_noise_point_tuple_sf(
        cfg: Any, field_name: str, tuple_index: int, new_sf: Optional[int],
        ) -> Tuple[Any, Any]:
    """For a tuple-typed cfg field (e.g. ``square_rescales``) update one slot."""
    seq = getattr(cfg, field_name, None)
    if seq is None:
        return (None, None)
    items = list(seq)
    idx = int(tuple_index)
    if idx < 0:
        idx = len(items) + idx
    if not (0 <= idx < len(items)):
        return (None, None)
    old_point = items[idx]
    old_sf = int(getattr(old_point, "scaling_factor", 0)) if old_point is not None else None
    if new_sf is None:
        items[idx] = None
    else:
        if old_point is None:

            return (old_sf, None)
        from function_handler import NoisePoint
        items[idx] = NoisePoint(
            distribution=str(old_point.distribution),
            scaling_factor=int(new_sf),
            N=int(old_point.N),
        )
    setattr(cfg, field_name, tuple(items))
    return (old_sf, new_sf)


def apply_optimizer_output_to_cfg(
        cfg: Any,
        *,
        output_raw: Mapping[str, Any],
        block_idx: int,
        graph_key: str,
        baseline_skeleton: Sequence[int],
        cfg_to_t_new_table: Optional[Mapping[str, Sequence[_SkelEntry]]] = None,
        rotation_name_map: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
        ) -> List[CfgOverrideEntry]:
    """Rewrite ``cfg`` in place to match the optimizer's replan output.

    Args:
        cfg:                 action-decoded ``Block{N}NoiseConfig`` instance.
        output_raw:          raw output dict from one invoker call (the dict
                             ReplanSession returns; equivalently
                             ``RescaleOptimizerOutput.raw``).
        block_idx:           1..5.
        graph_key:           e.g. "block1_mrpc" / "block3_exp_n4".
        baseline_skeleton:   baseline skeleton list for this graph (as found
                             in static_skeletons_<profile>.json; supplies the
                             baseline node-id sequence so we can detect fused
                             positions).
        cfg_to_t_new_table:  per-graph skel-position -> _SkelEntry mapping;
                             defaults to ``DEFAULT_CFG_TO_T_NEW_MAP``.
        rotation_name_map:   optional overrides for the canonical
                             ``{optimizer_rotation: cfg flag(s)}`` mapping.

    Returns:
        Ordered list of ``CfgOverrideEntry`` describing every change made.
        Empty list if the result was invalid or no compact config is present.
    """
    overrides: List[CfgOverrideEntry] = []

    compact = output_raw.get("new_compact_config") if isinstance(output_raw, Mapping) else None
    result = output_raw.get("result") if isinstance(output_raw, Mapping) else None
    if not compact or not isinstance(compact, Mapping):
        return overrides
    if isinstance(result, Mapping) and not bool(result.get("valid", True)):
        return overrides

    table = cfg_to_t_new_table if cfg_to_t_new_table is not None else DEFAULT_CFG_TO_T_NEW_MAP
    skel_entries = list(table.get(str(graph_key), ()))
    if not skel_entries:

        skel_entries = []


    cut_points: Dict[int, Dict[str, Any]] = {}
    for entry in compact.get("cut_point_sf", []) or []:
        if not isinstance(entry, Mapping) or "i" not in entry:
            continue
        cut_points[int(entry["i"])] = dict(entry)


    for r, skel_entry in enumerate(skel_entries):
        if r >= len(baseline_skeleton):
            break
        node_id = int(baseline_skeleton[r])
        cpt = cut_points.get(node_id)
        cfg_field = skel_entry.cfg_field
        tuple_index = skel_entry.tuple_index

        if r == 0:
            if cpt and "sf" in cpt:
                if tuple_index is None:
                    old, new = _set_noise_point_sf(cfg, cfg_field, int(cpt["sf"]))
                    if new is not None and old != new:
                        overrides.append(CfgOverrideEntry(
                            cfg_attr=f"{cfg_field}.scaling_factor",
                            graph_node=str(cpt.get("name", "")),
                            source="fresh",
                            old_value=old, new_value=new,
                        ))
            continue


        if cpt is None or cpt.get("sf_post") is None:

            if tuple_index is None:
                old_val = getattr(cfg, cfg_field, None)
                if old_val is not None:
                    setattr(cfg, cfg_field, None)
                    overrides.append(CfgOverrideEntry(
                        cfg_attr=cfg_field,
                        graph_node=None,
                        source="rescale_fused_away",
                        old_value=getattr(old_val, "scaling_factor", None),
                        new_value=None,
                    ))
            else:
                old, _ = _set_noise_point_tuple_sf(cfg, cfg_field, tuple_index, None)
                if old is not None:
                    overrides.append(CfgOverrideEntry(
                        cfg_attr=f"{cfg_field}[{tuple_index}]",
                        graph_node=None,
                        source="rescale_fused_away",
                        old_value=old, new_value=None,
                    ))
            continue

        if "sf_post" in cpt:
            new_sf = int(cpt["sf_post"])
            if tuple_index is None:
                old, new = _set_noise_point_sf(cfg, cfg_field, new_sf)
                if new is not None and old != new:
                    overrides.append(CfgOverrideEntry(
                        cfg_attr=f"{cfg_field}.scaling_factor",
                        graph_node=str(cpt.get("name", "")),
                        source="rescale_post",
                        old_value=old, new_value=new,
                    ))
            else:
                old, new = _set_noise_point_tuple_sf(cfg, cfg_field, tuple_index, new_sf)
                if new is not None and old != new:
                    overrides.append(CfgOverrideEntry(
                        cfg_attr=f"{cfg_field}[{tuple_index}].scaling_factor",
                        graph_node=str(cpt.get("name", "")),
                        source="rescale_post",
                        old_value=old, new_value=new,
                    ))


    node_to_attr = GRAPH_NODE_TO_CFG_ATTR.get(int(block_idx), {})
    for entry in compact.get("propagation_deltas", []) or []:
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("name", ""))
        delta = entry.get("delta")
        if not isinstance(delta, int):

            continue
        cfg_field = node_to_attr.get(name)
        if cfg_field is None:
            continue
        old, new = _set_noise_point_sf(cfg, cfg_field, int(delta))
        if new is not None and old != new:
            overrides.append(CfgOverrideEntry(
                cfg_attr=f"{cfg_field}.scaling_factor",
                graph_node=name,
                source="propagation_delta",
                old_value=old, new_value=new,
            ))


    eff_rotations = compact.get("effective_rotations", []) or []
    enabled_flags: List[str] = []
    repeat_counts: Dict[str, int] = {}
    resolved_rotation_map: Dict[str, Union[str, Sequence[str]]] = dict(
        default_rotation_name_map(int(block_idx))
    )
    if rotation_name_map:
        resolved_rotation_map.update(rotation_name_map)

    for entry in eff_rotations:
        if not isinstance(entry, Mapping):
            raise ValueError(f"malformed effective rotation entry: {entry!r}")
        src = str(entry.get("name", ""))
        mapped = resolved_rotation_map.get(src)
        if mapped is None:
            raise ValueError(
                f"unmapped effective rotation {src!r} for block {int(block_idx)}"
            )
        raw_count = entry.get("count", 1)
        if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count <= 0:
            raise ValueError(
                f"invalid effective rotation count for {src!r}: {raw_count!r}"
            )
        flags = (mapped,) if isinstance(mapped, str) else tuple(mapped)
        if not flags:
            raise ValueError(
                f"effective rotation {src!r} maps to no model flag"
            )
        for flag in flags:
            flag_name = str(flag)
            if not flag_name.startswith("rotation_after_") or not hasattr(cfg, flag_name):
                raise ValueError(
                    f"effective rotation {src!r} maps to invalid cfg flag "
                    f"{flag_name!r}"
                )
            enabled_flags.append(flag_name)
            repeat_counts[flag_name] = (
                repeat_counts.get(flag_name, 0) + int(raw_count)
            )


    pre_flags = {
        n: bool(getattr(cfg, n))
        for n in vars(cfg).keys()
        if n.startswith("rotation_after_")
    }
    pre_repeat_counts = dict(getattr(cfg, "rotation_repeat_counts", {}) or {})
    apply_rotation_flags_to_cfg(cfg, enabled_flags)
    setattr(cfg, "rotation_repeat_counts", dict(sorted(repeat_counts.items())))
    post_flags = {
        n: bool(getattr(cfg, n))
        for n in vars(cfg).keys()
        if n.startswith("rotation_after_")
    }
    for flag_name in sorted(set(pre_flags) | set(post_flags)):
        before = pre_flags.get(flag_name, False)
        after = post_flags.get(flag_name, False)
        if before != after:
            overrides.append(CfgOverrideEntry(
                cfg_attr=flag_name,
                graph_node=None,
                source="rotation_flag",
                old_value=before, new_value=after,
            ))
    post_repeat_counts = dict(getattr(cfg, "rotation_repeat_counts", {}) or {})
    for flag_name in sorted(set(pre_repeat_counts) | set(post_repeat_counts)):
        before = int(pre_repeat_counts.get(flag_name, 0))
        after = int(post_repeat_counts.get(flag_name, 0))
        if before != after:
            overrides.append(CfgOverrideEntry(
                cfg_attr=f"rotation_repeat_counts.{flag_name}",
                graph_node=None,
                source="rotation_count",
                old_value=before,
                new_value=after,
            ))

    return overrides


def aggregate_optimizer_signals(
        outputs: Mapping[str, RescaleOptimizerOutput],
        ) -> OptimizerRewardSignals:
    """跨 block 聚合 fusion_count / total_bits / invalid_chain。"""
    fusion_total = 0
    bits_total = 0
    valid_n = 0
    invalid_n = 0
    per_config: Dict[str, dict] = {}
    invalid_chains: Dict[str, dict] = {}
    for cname, out in outputs.items():
        fusion_total += int(out.fusion_count)
        bits_total += int(out.total_bits)
        if out.valid:
            valid_n += 1
        else:
            invalid_n += 1
            if out.invalid_chain is not None:
                invalid_chains[cname] = dict(out.invalid_chain)
        per_config[cname] = out.to_signal_dict()

    return OptimizerRewardSignals(
        total_fusion_count=fusion_total,
        total_bits_sum=bits_total,
        any_invalid=(invalid_n > 0),
        valid_block_count=valid_n,
        invalid_block_count=invalid_n,
        per_config=per_config,
        invalid_chains=invalid_chains,
    )
