"""Skeleton-driven stage mapping — the single source of truth (SSOT).

Why this exists
---------------
Rescale_optimizer owns a *complete* computation chain per graph (every node that
could ever appear) in ``configs/<profile>/<graph>.json`` under ``stages``. From a
Stage-1 config it picks, via its SNR rule + modulus-chain optimisation, *which*
cut-points become rescale points + their scaling factors — that selection is the
``static_skeletons`` baseline (our RL baseline action) and the ``replan_actions``
interface. So a skeleton is just a *subset* of the complete chain, and a regen
can move which cut-points carry ``sf_post`` (the 2026 regen moved block2's
rescales to ``rotKT_mask1`` / ``preprocess_qkt`` and block5_n1's middle rescale
to ``ctct_xmean_over_std``).

Previously three places hard-coded order + which nodes (baseline extraction,
``rescale_optimizer_bridge.DEFAULT_CFG_TO_T_NEW_MAP``, action active/compat
slots), so a regen drifted them silently. This module instead maps EVERY node of
the COMPLETE chain to its cfg + RL fields ONCE (stable domain knowledge — node
names + meanings don't change), and derives the ordered t_new stages + active
rescale slots from whatever the *current* skeleton selects. Any skeleton subset
— current or future — is therefore handled automatically, and
:func:`unmapped_full_chain_nodes` flags a brand-new RO node loudly.

Torch-free.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from json_utils import read_json_file


_SOURCE_CFG_FIELD: Dict[int, str] = {
    1: "gelu_out_fresh",
    2: "inv_std_fresh",
    3: "x_fresh",
    4: "softmax_out_fresh",
    5: "x_centered_fresh",
}
_SOURCE_RL_FIELD: Dict[int, str] = {
    1: "gelu_out_sf",
    2: "inv_std_fresh_sf",
    3: "x_fresh_sf",
    4: "softmax_out_fresh_sf",
    5: "x_centered_fresh_sf",
}


@dataclass(frozen=True)
class NodeBinding:
    """How one RO chain cut-point maps onto cfg + RL action slots.

    A cut-point plays up to two roles:
      * ENCODE — for ``CTPT_MUL`` nodes (ct × plaintext weight): the plaintext is
        encoded, driving an ``*_encode`` cfg field + an encode RL slot. ``CTCT_MUL``
        nodes (ct × ct: the symmetric "x2" squarings / asymmetric externals) have
        no encode slot, so ``encode_*`` stay None.
      * RESCALE — *any* cut-point can be selected by the optimizer as a rescale
        stage (``sf_post``), driving a ``*_result_rescale`` cfg field + a rescale
        RL slot.

    ``*_bound_rl_fields`` are extra RL slots kept equal to the primary (block2
    Q/K shared chain, block4 softmax_out_mask ↔ v_mask).
    """
    encode_cfg_field: Optional[str] = None
    encode_rl_field: Optional[str] = None
    encode_bound_rl_fields: Tuple[str, ...] = ()
    rescale_cfg_field: Optional[str] = None
    rescale_tuple_index: Optional[int] = None
    rescale_rl_field: Optional[str] = None
    rescale_bound_rl_fields: Tuple[str, ...] = ()


def _enc(cfg: str, rl: str, *bound: str) -> Dict[str, Any]:
    return {"encode_cfg_field": cfg, "encode_rl_field": rl, "encode_bound_rl_fields": tuple(bound)}


def _rsc(cfg: str, rl: str, *bound: str, tuple_index: Optional[int] = None) -> Dict[str, Any]:
    return {"rescale_cfg_field": cfg, "rescale_tuple_index": tuple_index,
            "rescale_rl_field": rl, "rescale_bound_rl_fields": tuple(bound)}


def _node(**kw: Any) -> NodeBinding:
    return NodeBinding(**kw)


_NODE_MAP: Dict[int, Dict[str, NodeBinding]] = {
    1: {
        "ctpt_ffn2":       _node(**_enc("wffn2_encode", "wffn2_sf"),
                                  **_rsc("wffn2_result_rescale", "wffn2_rescale_sf")),
        "ctpt_inv_d_1":    _node(**_enc("mean_inv_d_encode", "mean_inv_d_sf"),
                                  **_rsc("mean_result_rescale", "mean_rescale_sf")),
        "ctct_ext_square": _node(**_rsc("square_result_rescale", "square_rescale_sf")),
        "ctpt_inv_d_2":    _node(**_enc("var_inv_d_encode", "var_inv_d_sf"),
                                  **_rsc("var_result_rescale", "var_rescale_sf")),
    },
    2: {
        "ctct_x_mean_over_std": _node(**_rsc("normalize_result_rescale", "normalize_rescale_sf")),
        "ctpt_gama1":           _node(**_enc("gamma_encode", "gamma_sf"),
                                       **_rsc("gamma_result_rescale", "gamma_rescale_sf")),

        "ctpt_wq_wk":           _node(**_enc("wk_encode", "wk_sf", "wq_sf"),
                                       **_rsc("wk_result_rescale", "wk_rescale_sf", "wq_rescale_sf")),
        "ctpt_rotKT_mask1":     _node(**_enc("kt_mask1_encode", "kt_mask1_sf", "q_mask1_sf"),
                                       **_rsc("kt_mask1_result_rescale", "kt_mask1_rescale_sf", "q_mask1_rescale_sf")),
        "ctpt_rotKT_mask2":     _node(**_enc("kt_mask2_encode", "kt_mask2_sf", "q_mask2_sf"),
                                       **_rsc("kt_mask2_result_rescale", "kt_mask2_rescale_sf", "q_mask2_rescale_sf")),
        "ctct_preprocess_qkt":  _node(**_rsc("qkt_matmul_result_rescale", "qkt_matmul_rescale_sf")),
        "ctpt_mask":            _node(**_enc("qkt_merge_mask_encode", "qkt_merge_mask_sf"),
                                       **_rsc("qkt_merge_mask_result_rescale", "qkt_merge_mask_rescale_sf")),
    },
    3: {
        "ctpt_inv_2n": _node(**_enc("inv_2n_encode", "inv_2n_sf"),
                             **_rsc("x_inv_2n_result_rescale", "x_inv_2n_rescale_sf")),

    },
    4: {

        "ctpt_mask2":             _node(**_enc("softmax_out_mask_encode", "softmax_out_mask_sf", "v_mask_sf"),
                                        **_rsc("softmax_out_mask_rescale", "softmax_out_mask_rescale_sf", "v_mask_rescale_sf")),
        "ctct_rot_softmax_mul_v": _node(**_rsc("softmax_v_matmul_rescale", "softmax_v_matmul_rescale_sf")),
        "ctpt_mask":              _node(**_enc("softmax_v_mask_encode", "softmax_v_mask_sf"),
                                        **_rsc("softmax_v_mask_rescale", "softmax_v_mask_rescale_sf")),
        "ctpt_wo_attnout":        _node(**_enc("wo_encode", "wo_sf"),
                                        **_rsc("wo_result_rescale", "wo_rescale_sf")),
        "ctpt_inv_d_1":           _node(**_enc("ln_mean_inv_d_encode", "ln_mean_inv_d_sf"),
                                        **_rsc("ln_mean_result_rescale", "ln_mean_rescale_sf")),
        "ctct_square":            _node(**_rsc("ln_square_result_rescale", "ln_square_rescale_sf")),
        "ctpt_inv_d_2":           _node(**_enc("ln_var_inv_d_encode", "ln_var_inv_d_sf"),
                                        **_rsc("ln_var_result_rescale", "ln_var_rescale_sf")),
    },
    5: {
        "ctct_xmean_over_std": _node(**_rsc("normalize_result_rescale", "normalize_rescale_sf")),
        "ctpt_gamal":          _node(**_enc("gamma_encode", "gamma_sf"),
                                     **_rsc("gamma_result_rescale", "gamma_rescale_sf")),
        "ctpt_wffn1":          _node(**_enc("wffn1_encode", "wffn1_sf"),
                                     **_rsc("wffn1_result_rescale", "wffn1_rescale_sf")),
        "ctct_gelu_x2":        _node(**_rsc("gelu_power_rescales", "gelu_power_rescale_sf_0", tuple_index=0)),
        "ctct_gelu_x4":        _node(**_rsc("gelu_power_rescales", "gelu_power_rescale_sf_1", tuple_index=1)),

        "ctpt_gelu_coeff":     _node(**_enc("gelu_coeff_encode", "gelu_coeff_sf"),
                                     **_rsc("gelu_coeff_mul_rescales", "gelu_coeff_mul_rescale_sf_0", tuple_index=-1)),
    },
}


_BLOCK3_MAX_SQUARE_SLOT = 3


def _block3_square_binding(square_k: int) -> NodeBinding:
    idx = max(0, min(int(square_k) - 1, _BLOCK3_MAX_SQUARE_SLOT))
    return _node(**_rsc("square_rescales", f"square_rescale_sf_{idx}", tuple_index=idx))


def _node_binding(block_idx: int, node_name: str) -> Optional[NodeBinding]:
    name = str(node_name)
    if int(block_idx) == 3 and name.startswith("ctct_square_"):
        try:
            return _block3_square_binding(int(name.rsplit("_", 1)[-1]))
        except ValueError:
            return None
    return _NODE_MAP.get(int(block_idx), {}).get(name)


@dataclass
class GraphStagePlan:
    """Skeleton-derived plan for one graph (one (block, degree) config)."""
    graph_key: str
    block_idx: int


    t_new_entries: List[Tuple[Optional[str], Optional[int]]] = field(default_factory=list)


    active_rescale_rl_fields: List[str] = field(default_factory=list)


    rescale_stage_bindings: List[Tuple[str, NodeBinding]] = field(default_factory=list)
    unmapped_rescale_nodes: List[str] = field(default_factory=list)


def _cut_points(archive_entry: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    return list(archive_entry.get("cut_point_sf", []) or [])


def derive_stage_plan(
        block_idx: int,
        graph_key: str,
        archive_entry: Mapping[str, Any],
        ) -> GraphStagePlan:
    """Derive t_new ordering + active rescale slots from one graph's skeleton.

    ``cut_point_sf``: entry 0 (type SOURCE) is the fresh stage; every entry
    carrying ``sf_post`` is a rescale stage, in order.
    """
    block_idx = int(block_idx)
    plan = GraphStagePlan(graph_key=str(graph_key), block_idx=block_idx)

    src_cfg = _SOURCE_CFG_FIELD.get(block_idx)
    if src_cfg is None:
        raise ValueError(f"no SOURCE cfg field for block {block_idx}")
    plan.t_new_entries.append((src_cfg, None))

    for cp in _cut_points(archive_entry):
        if cp.get("sf_post") is None:
            continue
        name = str(cp.get("name") or "")
        binding = _node_binding(block_idx, name)
        if binding is None or binding.rescale_cfg_field is None:
            plan.unmapped_rescale_nodes.append(name)
            plan.t_new_entries.append((None, None))
            continue
        plan.t_new_entries.append((binding.rescale_cfg_field, binding.rescale_tuple_index))
        plan.rescale_stage_bindings.append((name, binding))
        if binding.rescale_rl_field:
            plan.active_rescale_rl_fields.append(binding.rescale_rl_field)
        plan.active_rescale_rl_fields.extend(binding.rescale_bound_rl_fields)
    return plan


def _archive_entries_from_results(results: Any) -> Dict[str, Mapping[str, Any]]:
    out: Dict[str, Mapping[str, Any]] = {}
    for entry in list(results or []):
        if not isinstance(entry, Mapping) or not entry.get("success"):
            continue
        cname = str(entry.get("config_name") or "").strip()
        if cname:
            out[cname] = entry
    return out


def _block_idx_for_graph(graph_key: str) -> Optional[int]:
    gk = str(graph_key)
    for b in (1, 2, 3, 4, 5):
        if gk.startswith(f"block{b}"):
            return b
    return None


def build_stage_plans(
        archive_entries: Mapping[str, Mapping[str, Any]],
        ) -> Dict[str, GraphStagePlan]:
    """Derive a :class:`GraphStagePlan` for every graph in ``archive_entries``
    (``{graph_key: archive_entry}``, each entry carrying ``cut_point_sf``)."""
    plans: Dict[str, GraphStagePlan] = {}
    for gk, entry in archive_entries.items():
        bidx = _block_idx_for_graph(gk)
        if bidx is None:
            continue
        plans[str(gk)] = derive_stage_plan(bidx, gk, entry)
    return plans


def build_stage_plans_from_archive(archive: Mapping[str, Any]) -> Dict[str, GraphStagePlan]:
    """Derive plans straight from a loaded ``static_skeletons_<profile>.json`` dict."""
    return build_stage_plans(_archive_entries_from_results(archive.get("results")))


def full_chain_cut_point_names(config: Mapping[str, Any]) -> List[str]:
    """All cut-point node names of a graph's COMPLETE chain (``configs/<g>.json``)."""
    names: List[str] = []
    for st in config.get("stages", []) or []:
        cp = st.get("cut_point") if isinstance(st, Mapping) else None
        if isinstance(cp, Mapping) and cp.get("name"):
            names.append(str(cp["name"]))
    return names


def unmapped_full_chain_nodes(block_idx: int, config: Mapping[str, Any]) -> List[str]:
    """Cut-points of the COMPLETE chain that the node map does not cover.

    The completeness guard: if Rescale_optimizer adds a node to a chain, this
    returns it so the SSOT can be extended (rather than silently mis-mapping a
    future skeleton that selects that node).
    """
    missing: List[str] = []
    for name in full_chain_cut_point_names(config):
        if _node_binding(int(block_idx), name) is None:
            missing.append(name)
    return missing


def source_rl_field(block_idx: int) -> Optional[str]:
    """RL fresh action slot for a block's SOURCE (cut_point_sf[0]), name-agnostic."""
    return _SOURCE_RL_FIELD.get(int(block_idx))


def rescale_rl_fields(block_idx: int, node_name: str) -> Tuple[str, ...]:
    """RL rescale slots a cut-point drives when selected (primary + bound). ()=unmapped."""
    b = _node_binding(int(block_idx), str(node_name))
    if b is None or b.rescale_rl_field is None:
        return ()
    return (b.rescale_rl_field,) + tuple(b.rescale_bound_rl_fields)


def encode_rl_fields(block_idx: int, node_name: str) -> Tuple[str, ...]:
    """RL encode slots a CTPT_MUL cut-point drives (primary + bound). ()=none/unmapped."""
    b = _node_binding(int(block_idx), str(node_name))
    if b is None or b.encode_rl_field is None:
        return ()
    return (b.encode_rl_field,) + tuple(b.encode_bound_rl_fields)


def active_rescale_rl_fields(block_idx: int, archive_entry: Mapping[str, Any]) -> frozenset:
    """Set of RL rescale slots that are active stages on this graph's skeleton."""
    return frozenset(derive_stage_plan(int(block_idx), "", archive_entry).active_rescale_rl_fields)


def load_profile_configs(rescale_optimizer_root: str, profile: str) -> Dict[str, Mapping[str, Any]]:
    """Load every ``configs/<profile>/<graph>.json`` (excluding static_skeletons)."""
    cfg_dir = os.path.join(os.path.abspath(str(rescale_optimizer_root)), "configs", str(profile))
    out: Dict[str, Mapping[str, Any]] = {}
    if not os.path.isdir(cfg_dir):
        return out
    entries = []
    with os.scandir(cfg_dir) as it:
        for entry in it:
            fn = entry.name
            if not fn.endswith(".json") or fn.startswith("static_skeletons"):
                continue
            if not entry.is_file():
                continue
            entries.append((fn, entry.path))
    for fn, path in sorted(entries):
        out[fn[:-5]] = read_json_file(path)
    return out
