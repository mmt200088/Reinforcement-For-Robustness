"""
rescale_optimizer/config_loader.py

JSON-driven builder for the rescale optimizer.

Everything that used to be hard-coded in Python (graph topology,
amplitude profiles, SNR requirements, noise table, amplitude budgets,
plaintext-leaf scaling factors, cost / optimization params, …) can now
be supplied as a single JSON file and realized via
:func:`load_graph_from_json`.

High-level JSON schema
----------------------

.. code-block:: json

    {
      "global": {
        "h_sf": 10,
        "q_legal_min": 30,
        "q_legal_max": 60
      },

      "optimization": {
        "cost_params": {
            "lambda_0": 1.0, "lambda_1": 0.1,
            "alpha": 1.0,   "beta": 0.05
        },
        "q_head_bits": 60,
        "q_tail_bits": 60,
        "max_best_first_expansions": 64
      },

      "noise_table": {
        "rescale": { "20": 1.0, "25": 0.3162, "30": 0.1, ... },
        "ctpt":    { "20": 1.0, ... }
      },

      "defaults": {
        "amplitude_profile": {
            "percentiles": [0.1, 0.5, 0.9],
            "values":      [1e-4, 1.0, 1e3]
        },
        "snr_requirement": {
            "percentile": 0.8,
            "max_relative_error": 0.01
        },
        "op_type": "rescale"
      },

      // All three top-level vectors below are OPTIONAL and have length
      // M+1 (one entry per real cut point c_0..c_M).  When present
      // they take precedence over per-cut-point fields and defaults.
      // Use them to make it visually obvious that these quantities are
      // per-cut-point (not a single global scalar).
      //
      //   amplitude_budgets   : A^{budget}_j    for j=0..M
      //   amplitude_profiles  : AmplitudeProfile_j (CDF of |x_j|) per cut point
      //   snr_requirements    : SNRRequirement_j per cut point
      "amplitude_budgets":  [15, 15, 15, 15, 15, 15, 15],
      "amplitude_profiles": [ { "percentiles": [...], "values": [...] }, ... ],
      "snr_requirements":   [ { "percentile": 0.8, "max_relative_error": 0.01 }, ... ],

      "source": {
        "name": "source",
        "scale_delta_bits": 0,
        "amplitude_budget_bits": 15
        // may also override amplitude_profile / snr_requirement / op_type
      },

      "stages": [
        {
          "nodes": [
            { "name": "rot_s0_0", "type": "ROTATION",
              "count": 1, "cost_slope": 0.5, "cost_intercept": 6.0 }
            // "type" may also be PT_OP or PT.  PT leaves may carry a
            //  metadata field "delta_bits" (informational only — the
            //  main-path Δ is written on the CTPT_MUL that consumes it).
          ],
          "cut_point": {
            "name": "ctct_qk",
            "type": "CTCT_MUL",
            "scale_delta_bits": 0,
            "count": 1,
            "cost_slope": 1.0,
            "cost_intercept": 12.0,
            "amplitude_budget_bits": 15
            // optional: amplitude_profile / snr_requirement / op_type
            //
            // For CTCT_MUL nodes:
            //   * scale_delta_bits in the config is IGNORED
            //     (loader warns and forces it to 0 if non-zero);
            //   * by default propagate_scale uses symmetric
            //     squaring  s → 2·s  (both operands at the current
            //     working scale);
            //   * if the other ct operand enters at a known external
            //     scale, set
            //         "other_ct_scale_bits": 30
            //     and propagate_scale will use  s → s + 30  instead
            //     (asymmetric CTCT).
          }
        },
        ...
      ],

      "dummy_sink": {
        "name": "dummy_sink",
        "amplitude_budget_bits": 0
      }
    }

Conventions honored by the loader
---------------------------------

* ``stages[k].nodes`` are the non-cut-point nodes in segment
  ``(c_k, c_{k+1}]`` **excluding** the endpoint itself; their
  ``stage_anchor`` is set to ``k`` automatically.
* ``stages[k].cut_point`` becomes ``c_{k+1}`` (a multiplication).  The
  first element of ``stages`` therefore defines ``c_1``.
* ``source`` defines ``c_0`` and has ``NodeType.SOURCE`` implicitly;
  ``type`` need not be provided.
* ``dummy_sink`` is generated automatically; providing the section is
  optional (only used to tweak its budget / name).
* Any cut-point field that is omitted falls back to ``defaults``.
* Plaintext leaves (``type: "PT"``) have ``scale_delta_bits = 0`` on the
  propagation path — the ct×pt's scale Δ is carried by the consuming
  ``CTPT_MUL``'s ``scale_delta_bits``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .graph import (
    AmplitudeProfile,
    ComputeNode,
    CostParams,
    CutPoint,
    NodeType,
    NoiseLookupTable,
    RescaleGraph,
    SNRRequirement,
)
from .optimizer import OptimizationConfig

logger = logging.getLogger("rescale_optimizer")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_graph_from_json(
    path: Union[str, Path],
) -> Tuple[RescaleGraph, OptimizationConfig, Optional[List[int]]]:
    """
    Load a full optimizer setup from a JSON file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the JSON config.

    Returns
    -------
    graph : RescaleGraph
        Fully assembled graph (nodes, cut points, noise table, global
        Q_legal, h_sf).  The feasibility DAG has **not** been built yet
        — pass the graph into ``optimize_rescale`` to do so.
    opt_config : OptimizationConfig
        Parsed from ``optimization`` (or defaults if absent).
    amplitude_budgets : list[int] | None
        Convenience list of per-cut-point budgets (``c_0..c_M``).  Since
        the loader already writes these onto ``graph.cut_points``, you
        do **not** need to pass this back into ``optimize_rescale`` —
        it is returned mostly for inspection / tests.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    if not isinstance(cfg, Mapping):
        raise ValueError(f"Config root must be a JSON object, got {type(cfg)}")

    return build_from_dict(cfg)


def build_from_dict(
    cfg: Mapping[str, Any],
) -> Tuple[RescaleGraph, OptimizationConfig, Optional[List[int]]]:
    """Like :func:`load_graph_from_json` but takes a pre-parsed dict."""

    global_cfg = cfg.get("global", {}) or {}
    opt_cfg_raw = cfg.get("optimization", {}) or {}
    noise_cfg = cfg.get("noise_table", {}) or {}
    defaults = cfg.get("defaults", {}) or {}
    source_cfg = cfg.get("source")
    stages_cfg = cfg.get("stages")
    dummy_cfg = cfg.get("dummy_sink", {}) or {}
    amp_budget_vec = cfg.get("amplitude_budgets")
    amp_profile_vec = cfg.get("amplitude_profiles")
    snr_req_vec = cfg.get("snr_requirements")

    if source_cfg is None:
        raise ValueError("Config must contain a 'source' section (c_0).")
    if not stages_cfg:
        raise ValueError(
            "Config must contain a non-empty 'stages' list (c_1..c_M)."
        )

    # --- noise table & defaults ---------------------------------------
    noise_table = _build_noise_table(noise_cfg)
    default_amp = _build_amp(defaults.get("amplitude_profile"))
    default_snr = _build_snr(defaults.get("snr_requirement"))
    default_op_type = str(defaults.get("op_type", "rescale"))
    # Scalar fallback for A^{budget}_j when neither the top-level vector
    # nor the per-cut-point field is set.  Kept for backward
    # compatibility; prefer the vector ``amplitude_budgets``.
    default_budget = int(defaults.get("amplitude_budget_bits", 15))

    # Validate any optional top-level per-cut-point vectors.  Length
    # must be M + 1 = len(stages) + 1 (c_0..c_M; dummy sink c_{M+1}
    # has its own ``dummy_sink`` section).
    expected_len = len(stages_cfg) + 1

    def _check_vec_length(name: str, vec: Any) -> None:
        if not isinstance(vec, Sequence) or isinstance(vec, (str, bytes)):
            raise ValueError(
                f"'{name}' must be a JSON array of length "
                f"M+1 = {expected_len} (one entry per real cut point "
                f"c_0..c_M)."
            )
        if len(vec) != expected_len:
            raise ValueError(
                f"'{name}' has length {len(vec)} but expected M+1 = "
                f"{expected_len}."
            )

    if amp_budget_vec is not None:
        _check_vec_length("amplitude_budgets", amp_budget_vec)
        amp_budget_vec = [int(x) for x in amp_budget_vec]

    if amp_profile_vec is not None:
        _check_vec_length("amplitude_profiles", amp_profile_vec)
        amp_profile_vec = [_build_amp(x) for x in amp_profile_vec]

    if snr_req_vec is not None:
        _check_vec_length("snr_requirements", snr_req_vec)
        snr_req_vec = [_build_snr(x) for x in snr_req_vec]

    # --- assemble nodes in topological order --------------------------
    nodes: List[ComputeNode] = []
    cut_point_nodes: List[ComputeNode] = []
    topo_counter = [0]

    def _new_node(**kwargs: Any) -> ComputeNode:
        n = ComputeNode(
            node_id=len(nodes),
            topo_order=topo_counter[0],
            **kwargs,
        )
        topo_counter[0] += 1
        nodes.append(n)
        return n

    # c_0: source -----------------------------------------------------
    src_node = _new_node(
        name=str(source_cfg.get("name", "source")),
        node_type=NodeType.SOURCE,
        stage_anchor=0,
        scale_delta_bits=int(source_cfg.get("scale_delta_bits", 0)),
        count=int(source_cfg.get("count", 1)),
        cost_slope=float(source_cfg.get("cost_slope", 0.0)),
        cost_intercept=float(source_cfg.get("cost_intercept", 0.0)),
    )
    cut_point_nodes.append(src_node)

    # c_1..c_M via stages ---------------------------------------------
    for k, stage in enumerate(stages_cfg):
        if not isinstance(stage, Mapping):
            raise ValueError(
                f"stages[{k}] must be an object with 'nodes' and 'cut_point'."
            )

        # non-cut-point nodes in (c_k, c_{k+1})
        stage_nodes_list = stage.get("nodes", []) or []
        for i, node_spec in enumerate(stage_nodes_list):
            ntype_str = str(node_spec.get("type", "ROTATION"))
            ntype = _parse_non_cp_type(ntype_str, where=f"stages[{k}].nodes[{i}]")
            # PT leaves: ignore any 'delta_bits' metadata and force 0.
            scale_delta = int(node_spec.get("scale_delta_bits", 0))
            if ntype == NodeType.PT:
                if "delta_bits" in node_spec and scale_delta == 0:
                    logger.debug(
                        "PT leaf '%s' has metadata delta_bits=%s (informational; "
                        "main-path Δ is carried by the consuming CTPT_MUL).",
                        node_spec.get("name", "<pt>"),
                        node_spec["delta_bits"],
                    )
                if scale_delta != 0:
                    logger.warning(
                        "PT leaf '%s' has scale_delta_bits=%d != 0; forcing 0 "
                        "(move Δ onto the CTPT_MUL that consumes this pt).",
                        node_spec.get("name", "<pt>"), scale_delta,
                    )
                    scale_delta = 0
            _new_node(
                name=str(node_spec.get("name", f"stage{k}_node{i}")),
                node_type=ntype,
                stage_anchor=k,
                scale_delta_bits=scale_delta,
                count=int(node_spec.get("count", 1)),
                cost_slope=float(node_spec.get("cost_slope", 0.0)),
                cost_intercept=float(node_spec.get("cost_intercept", 0.0)),
            )

        # cut point c_{k+1}
        cp_spec = stage.get("cut_point")
        if cp_spec is None:
            raise ValueError(f"stages[{k}] is missing its 'cut_point' entry.")
        cp_type = _parse_cp_type(
            str(cp_spec.get("type", "CTCT_MUL")),
            where=f"stages[{k}].cut_point",
        )
        cp_scale_delta = int(cp_spec.get("scale_delta_bits", 0))
        cp_other_ct = cp_spec.get("other_ct_scale_bits")
        if cp_other_ct is not None:
            if cp_type != NodeType.CTCT_MUL:
                raise ValueError(
                    f"stages[{k}].cut_point '{cp_spec.get('name', '?')}': "
                    f"'other_ct_scale_bits' is only valid on CTCT_MUL nodes "
                    f"(got {cp_type.name})."
                )
            cp_other_ct = int(cp_other_ct)
        if cp_type == NodeType.CTCT_MUL and cp_scale_delta != 0:
            logger.warning(
                "stages[%d].cut_point '%s' (CTCT_MUL) has scale_delta_bits=%d; "
                "it is ignored by propagate_scale (CTCT uses s -> 2*s, "
                "or s -> s + other_ct_scale_bits when set). Forcing to 0.",
                k, cp_spec.get("name", "?"), cp_scale_delta,
            )
            cp_scale_delta = 0
        cp_node = _new_node(
            name=str(cp_spec.get("name", f"cp_{k + 1}")),
            node_type=cp_type,
            stage_anchor=k + 1,
            scale_delta_bits=cp_scale_delta,
            count=int(cp_spec.get("count", 1)),
            cost_slope=float(cp_spec.get("cost_slope", 0.0)),
            cost_intercept=float(cp_spec.get("cost_intercept", 0.0)),
            other_ct_scale_bits=cp_other_ct,
        )
        cut_point_nodes.append(cp_node)

    # c_{M+1}: dummy sink ---------------------------------------------
    dummy_node = _new_node(
        name=str(dummy_cfg.get("name", "dummy_sink")),
        node_type=NodeType.DUMMY_SINK,
        stage_anchor=len(cut_point_nodes) - 1,
        scale_delta_bits=0,
        count=0,
    )

    # --- cut points ---------------------------------------------------
    cut_points: List[CutPoint] = []

    def _budget_for(j: int, spec: Mapping[str, Any]) -> int:
        """Precedence: top-level vector ``amplitude_budgets[j]``
        > per-CP spec field > scalar default."""
        if amp_budget_vec is not None:
            return amp_budget_vec[j]
        if "amplitude_budget_bits" in spec:
            return int(spec["amplitude_budget_bits"])
        return default_budget

    def _amp_for(j: int, spec: Mapping[str, Any]) -> AmplitudeProfile:
        """Precedence: top-level vector ``amplitude_profiles[j]``
        > per-CP spec field > defaults > empty profile."""
        if amp_profile_vec is not None:
            return amp_profile_vec[j]
        if "amplitude_profile" in spec:
            return _build_amp(spec["amplitude_profile"])
        return default_amp

    def _snr_for(j: int, spec: Mapping[str, Any]) -> SNRRequirement:
        """Precedence: top-level vector ``snr_requirements[j]``
        > per-CP spec field > defaults > built-in default."""
        if snr_req_vec is not None:
            return snr_req_vec[j]
        if "snr_requirement" in spec:
            return _build_snr(spec["snr_requirement"])
        return default_snr

    src_cp_spec = source_cfg
    cut_points.append(_make_cut_point(
        index=0, node=src_node, spec=src_cp_spec,
        amp=_amp_for(0, src_cp_spec),
        snr=_snr_for(0, src_cp_spec),
        default_op_type=default_op_type,
        budget_bits=_budget_for(0, src_cp_spec),
    ))

    for k, stage in enumerate(stages_cfg):
        cp_spec = stage["cut_point"]
        cut_points.append(_make_cut_point(
            index=k + 1, node=cut_point_nodes[k + 1], spec=cp_spec,
            amp=_amp_for(k + 1, cp_spec),
            snr=_snr_for(k + 1, cp_spec),
            default_op_type=default_op_type,
            budget_bits=_budget_for(k + 1, cp_spec),
        ))

    # dummy sink cut point (virtual)
    cut_points.append(CutPoint(
        index=len(cut_points),
        node=dummy_node,
        amplitude_profile=AmplitudeProfile(),
        snr_requirement=default_snr,
        op_type=default_op_type,
        amplitude_budget_bits=int(dummy_cfg.get("amplitude_budget_bits", 0)),
    ))

    # --- graph --------------------------------------------------------
    graph = RescaleGraph(
        nodes=nodes,
        cut_points=cut_points,
        noise_table=noise_table,
        h_sf=int(global_cfg.get("h_sf", 10)),
        q_legal_min=int(global_cfg.get("q_legal_min", 30)),
        q_legal_max=int(global_cfg.get("q_legal_max", 60)),
    )

    # --- optimization config ------------------------------------------
    opt_config = _build_opt_config(opt_cfg_raw)

    amplitude_budgets = [
        int(cp.amplitude_budget_bits) for cp in cut_points[:-1]
    ]

    return graph, opt_config, amplitude_budgets


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_amp(spec: Optional[Mapping[str, Any]]) -> AmplitudeProfile:
    if not spec:
        return AmplitudeProfile()
    pct = [float(x) for x in spec.get("percentiles", []) or []]
    vals = [float(x) for x in spec.get("values", []) or []]
    if len(pct) != len(vals):
        raise ValueError(
            "amplitude_profile.percentiles and .values must have equal length "
            f"(got {len(pct)} vs {len(vals)})."
        )
    return AmplitudeProfile(percentiles=pct, values=vals)


def _build_snr(spec: Optional[Mapping[str, Any]]) -> SNRRequirement:
    if not spec:
        return SNRRequirement()
    return SNRRequirement(
        percentile=float(spec.get("percentile", 0.8)),
        max_relative_error=float(spec.get("max_relative_error", 0.01)),
    )


def _build_noise_table(
    spec: Mapping[str, Mapping[str, float]],
) -> NoiseLookupTable:
    if not spec:
        return NoiseLookupTable()
    table: Dict[str, Dict[int, float]] = {}
    for op_type, entries in spec.items():
        if not isinstance(entries, Mapping):
            raise ValueError(
                f"noise_table['{op_type}'] must be an object of sf_bits→noise."
            )
        inner: Dict[int, float] = {}
        for sf_key, noise in entries.items():
            try:
                sf_int = int(sf_key)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"noise_table['{op_type}'] key '{sf_key}' is not an integer."
                ) from exc
            inner[sf_int] = float(noise)
        table[str(op_type)] = inner
    return NoiseLookupTable(table=table)


def _parse_cp_type(s: str, *, where: str) -> NodeType:
    s_up = s.strip().upper()
    allowed = {"CTCT_MUL", "CTPT_MUL"}
    if s_up not in allowed:
        raise ValueError(
            f"{where}: cut-point 'type' must be one of {sorted(allowed)} "
            f"(got '{s}').  SOURCE/DUMMY_SINK are inferred automatically."
        )
    return NodeType[s_up]


def _parse_non_cp_type(s: str, *, where: str) -> NodeType:
    s_up = s.strip().upper()
    allowed = {"ROTATION", "PT_OP", "PT"}
    if s_up not in allowed:
        raise ValueError(
            f"{where}: non-cut-point 'type' must be one of {sorted(allowed)} "
            f"(got '{s}')."
        )
    return NodeType[s_up]


def _make_cut_point(
    *,
    index: int,
    node: ComputeNode,
    spec: Mapping[str, Any],
    amp: AmplitudeProfile,
    snr: SNRRequirement,
    default_op_type: str,
    budget_bits: int,
) -> CutPoint:
    op_type = str(spec.get("op_type", default_op_type))
    return CutPoint(
        index=index,
        node=node,
        amplitude_profile=amp,
        snr_requirement=snr,
        op_type=op_type,
        amplitude_budget_bits=int(budget_bits),
    )


def _build_opt_config(spec: Mapping[str, Any]) -> OptimizationConfig:
    cost_spec = spec.get("cost_params", {}) or {}
    cost = CostParams(
        lambda_0=float(cost_spec.get("lambda_0", 1.0)),
        lambda_1=float(cost_spec.get("lambda_1", 0.0)),
        alpha=float(cost_spec.get("alpha", 1.0)),
        beta=float(cost_spec.get("beta", 0.1)),
    )
    return OptimizationConfig(
        cost_params=cost,
        q_head_bits=int(spec.get("q_head_bits", 60)),
        q_tail_bits=int(spec.get("q_tail_bits", 60)),
        max_best_first_expansions=int(spec.get("max_best_first_expansions", 64)),
    )
