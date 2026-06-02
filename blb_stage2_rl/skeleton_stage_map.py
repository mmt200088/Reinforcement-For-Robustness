"""Skeleton-driven stage mapping — the single source of truth (SSOT).

Why this exists
---------------
When the Rescale_optimizer team regenerates ``static_skeletons_<profile>.json``,
*which* cut-points are rescale stages (carry ``sf_post``) can change — e.g. the
2026 regen moved block2's rescales from ``[gama1, rotKT_mask2, mask]`` to
``[gama1, rotKT_mask1, preprocess_qkt]`` and block5_n1's middle rescale from
``gamma`` to ``ctct_xmean_over_std`` (normalize). Previously three places
hard-coded the *order + which nodes*:

  * ``baseline_bootstrap`` baseline extraction,
  * ``rescale_optimizer_bridge.DEFAULT_CFG_TO_T_NEW_MAP`` (t_new derivation),
  * ``action_space`` active-vs-compat rescale slots.

Hard-coding drifts silently when the skeleton changes. This module derives the
ordered stages + the active rescale slots **from the actual skeleton** using a
*stable* node-name → field table (node names and their semantics do not change;
only which nodes appear as rescale stages does). All three consumers read from
here, so a skeleton change auto-propagates.

Data source: the ``cut_point_sf`` list inside each graph's archive entry
(``ReplanSession.baselines[graph_key].archive_entry`` or the raw
``static_skeletons`` ``results`` entry). The replan_actions_*.json files are
regenerated alongside the skeleton, so we never need to read them separately.

Torch-free.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

# ---------------------------------------------------------------------------
# Stable node-name → field tables (the only domain knowledge)
# ---------------------------------------------------------------------------
# SOURCE (cut_point_sf[0]) → cfg fresh field + RL fresh action field, per block.
# block5's SOURCE is named "x_mean" (n1/n2/n4) or "inv_std" (n0); both map to the
# same x_centered fresh (the two block-5 fresh operands are bound equal).
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
class RescaleBinding:
    """How one RO rescale node maps onto cfg + RL action slots.

    Args:
        cfg_field:    Block*NoiseConfig attribute that receives the rescale SF.
        tuple_index:  if cfg_field is a tuple (block3 square_rescales, block5
                      gelu_*), the index into it; None for a scalar field.
        rl_field:     the primary RL action slot driving this rescale.
        bound_rl_fields: extra RL slots bound equal to ``rl_field`` (block2 Q/K
                      shared chain). They become active together with rl_field.
    """
    cfg_field: str
    tuple_index: Optional[int]
    rl_field: str
    bound_rl_fields: Tuple[str, ...] = ()


# RO rescale node-name → binding, per block. Lists every node that *could* be a
# rescale stage (current + historically-seen), so the derivation stays robust
# across skeleton regens. A node is only consulted when it actually carries
# ``sf_post`` on the loaded skeleton; nodes absent here are reported as unmapped.
# block3's ``ctct_square_<k>`` is handled specially (index parsed from the name).
_RESCALE_NODE: Dict[int, Dict[str, RescaleBinding]] = {
    1: {
        "ctpt_inv_d_1": RescaleBinding("mean_result_rescale", None, "mean_rescale_sf"),
        "ctpt_inv_d_2": RescaleBinding("var_result_rescale", None, "var_rescale_sf"),
    },
    2: {
        "ctpt_gama1":          RescaleBinding("gamma_result_rescale", None, "gamma_rescale_sf"),
        # Q/K shared chain: kt_* rescale is bound equal to its q_* counterpart.
        "ctpt_rotKT_mask1":    RescaleBinding("kt_mask1_result_rescale", None,
                                              "kt_mask1_rescale_sf", ("q_mask1_rescale_sf",)),
        "ctpt_rotKT_mask2":    RescaleBinding("kt_mask2_result_rescale", None,
                                              "kt_mask2_rescale_sf", ("q_mask2_rescale_sf",)),
        # q×kᵀ result rescale (2026 regen put the rescale here instead of ctpt_mask).
        "ctct_preprocess_qkt": RescaleBinding("qkt_matmul_result_rescale", None, "qkt_matmul_rescale_sf"),
        "ctpt_mask":           RescaleBinding("qkt_merge_mask_result_rescale", None, "qkt_merge_mask_rescale_sf"),
    },
    3: {
        # ctct_square_<k> handled by _block3_square_binding(k).
    },
    4: {
        "ctct_rot_softmax_mul_v": RescaleBinding("softmax_v_matmul_rescale", None, "softmax_v_matmul_rescale_sf"),
        "ctpt_inv_d_1":           RescaleBinding("ln_mean_result_rescale", None, "ln_mean_rescale_sf"),
        "ctpt_inv_d_2":           RescaleBinding("ln_var_result_rescale", None, "ln_var_rescale_sf"),
        # (X−μ)² square rescale (2026 regen).
        "ctct_square":            RescaleBinding("ln_square_result_rescale", None, "ln_square_rescale_sf"),
    },
    5: {
        "ctct_xmean_over_std": RescaleBinding("normalize_result_rescale", None, "normalize_rescale_sf"),
        "ctpt_gamal":          RescaleBinding("gamma_result_rescale", None, "gamma_rescale_sf"),
        "ctpt_wffn1":          RescaleBinding("wffn1_result_rescale", None, "wffn1_rescale_sf"),
        "ctct_gelu_x2":        RescaleBinding("gelu_power_rescales", 0, "gelu_power_rescale_sf_0"),
        # ctpt_gelu_coeff → cfg.gelu_coeff_mul_rescales[-1] (graph merges the
        # coeff·x^k chain into one node; only the last tuple slot is read).
        "ctpt_gelu_coeff":     RescaleBinding("gelu_coeff_mul_rescales", -1, "gelu_coeff_mul_rescale_sf_0"),
    },
}

# block3 cfg has only 4 RL square slots (square_rescale_sf_0..3); n5/n6 squares
# beyond that reuse the last slot (matches the old hard-coded table behaviour).
_BLOCK3_MAX_SQUARE_SLOT = 3


def _block3_square_binding(square_k: int) -> RescaleBinding:
    """``ctct_square_<k>`` → cfg.square_rescales[k-1] / square_rescale_sf_<k-1>.

    k is 1-based in the node name. Slots beyond 4 clamp to the last (index 3).
    """
    idx = min(int(square_k) - 1, _BLOCK3_MAX_SQUARE_SLOT)
    idx = max(idx, 0)
    return RescaleBinding("square_rescales", idx, f"square_rescale_sf_{idx}")


def _rescale_binding(block_idx: int, node_name: str) -> Optional[RescaleBinding]:
    name = str(node_name)
    if int(block_idx) == 3 and name.startswith("ctct_square_"):
        try:
            return _block3_square_binding(int(name.rsplit("_", 1)[-1]))
        except ValueError:
            return None
    return _RESCALE_NODE.get(int(block_idx), {}).get(name)


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------
@dataclass
class GraphStagePlan:
    """Skeleton-derived plan for one graph (one (block, degree) config)."""
    graph_key: str
    block_idx: int
    # ordered t_new stages: list of (cfg_field, tuple_index); stage 0 is the
    # SOURCE (fresh) field, stages 1.. are the rescale fields in skeleton order.
    t_new_entries: List[Tuple[str, Optional[int]]] = field(default_factory=list)
    # RL action slots that are ACTIVE rescale stages on this skeleton
    # (primary + bound). The source fresh RL field is NOT included here.
    active_rescale_rl_fields: List[str] = field(default_factory=list)
    # ordered rescale RL fields (primary only, skeleton order) — for baseline
    # extraction parity / debugging.
    rescale_rl_field_order: List[str] = field(default_factory=list)
    # rescale node names on the skeleton that we could not map (drift signal).
    unmapped_rescale_nodes: List[str] = field(default_factory=list)


def _cut_points(archive_entry: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    return list(archive_entry.get("cut_point_sf", []) or [])


def derive_stage_plan(
        block_idx: int,
        graph_key: str,
        archive_entry: Mapping[str, Any],
        ) -> GraphStagePlan:
    """Derive the t_new ordering + active rescale slots from one graph's skeleton.

    Reads ``cut_point_sf``: the first entry (type SOURCE) is the fresh stage;
    every entry carrying ``sf_post`` is a rescale stage, in order.
    """
    block_idx = int(block_idx)
    plan = GraphStagePlan(graph_key=str(graph_key), block_idx=block_idx)

    src_cfg = _SOURCE_CFG_FIELD.get(block_idx)
    if src_cfg is None:
        raise ValueError(f"no SOURCE cfg field for block {block_idx}")
    plan.t_new_entries.append((src_cfg, None))

    for cp in _cut_points(archive_entry):
        # A rescale stage is a cut-point that actually rescales (has sf_post).
        if cp.get("sf_post") is None:
            continue
        name = str(cp.get("name") or "")
        binding = _rescale_binding(block_idx, name)
        if binding is None:
            plan.unmapped_rescale_nodes.append(name)
            # keep a placeholder so positional length still matches the skeleton
            # (caller can detect via unmapped_rescale_nodes and fall back).
            plan.t_new_entries.append((None, None))  # type: ignore[arg-type]
            continue
        plan.t_new_entries.append((binding.cfg_field, binding.tuple_index))
        plan.rescale_rl_field_order.append(binding.rl_field)
        plan.active_rescale_rl_fields.append(binding.rl_field)
        plan.active_rescale_rl_fields.extend(binding.bound_rl_fields)
    return plan


def _archive_entries_from_results(results: Any) -> Dict[str, Mapping[str, Any]]:
    """``static_skeletons['results']`` list → ``{config_name: entry}`` (success only)."""
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
    if gk.startswith("block1"):
        return 1
    if gk.startswith("block2"):
        return 2
    if gk.startswith("block3"):
        return 3
    if gk == "block4" or gk.startswith("block4"):
        return 4
    if gk.startswith("block5"):
        return 5
    return None


def build_stage_plans(
        archive_entries: Mapping[str, Mapping[str, Any]],
        ) -> Dict[str, GraphStagePlan]:
    """Derive a :class:`GraphStagePlan` for every graph in ``archive_entries``.

    ``archive_entries`` is ``{graph_key: archive_entry}`` (each entry has
    ``cut_point_sf``). Accepts the dict form from ``ReplanSession`` baselines or
    from :func:`build_stage_plans_from_archive`.
    """
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
