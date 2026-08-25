"""High-level in-process replan interface.

The module exposes the JSON/CLI replan operation through ordinary Python inputs
and return values. It is intended for the
BLB Stage-2 RL fast path: preload graph configs and static baselines once, then
call replan with ``t_new`` and ``delta_overrides`` variables in a tight loop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .config_loader import load_graph_from_json
from .feasibility import build_feasibility_dag
from .graph import ComputeNode, NodeType, RescaleGraph, propagate_scale
from .replan import (
    ReplanInputs,
    ReplanResult,
    _PreparedFusionPairs,
    _prepare_allowed_fusion_pairs,
    replan_with_user_actions,
)

DeltaValue = Union[int, str]
FusionPair = Tuple[int, int]
DEFAULT_FUSION_POLICY = "default"
SUPPORTED_GELU_GRAPH_DEGREES = {0, 1, 2, 4}

_DEFAULT_ALLOWED_FUSION_PAIRS: Dict[str, List[FusionPair]] = {
    "block1_mrpc": [],
    "block2_mrpc": [(1, 2)],
    "block4": [(1, 2)],
    "block5_n1": [(1, 2)],
    "block5_n2": [(1, 2)],
    "block5_n4": [(1, 2)],
}


@dataclass(frozen=True)
class BaselineRecord:
    """Static-skeleton baseline data for one graph config."""

    config_name: str
    skeleton: List[int]
    t_baseline: List[int]
    q_bits_baseline: List[int]
    archive_entry: Dict[str, Any]


@dataclass(frozen=True)
class CompactReplanResult:
    """Minimal repeated-replan result consumed by fusion enumeration."""

    valid: bool
    fusion_count: int
    total_bits: int
    compact_config: Optional[Dict[str, Any]]


def _parse_delta_value(value: Any) -> DeltaValue:
    if value == "x2":
        return "x2"
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"delta must be int or 'x2', got {value!r}") from exc


def _normalize_delta_overrides(
    delta_overrides: Optional[Mapping[str, Any]],
) -> Dict[str, DeltaValue]:
    if not delta_overrides:
        return {}
    if type(delta_overrides) is dict:
        for key, value in delta_overrides.items():
            if type(key) is not str or not (
                type(value) is int
                or (type(value) is str and value == "x2")
            ):
                break
        else:
            return delta_overrides
    return {str(k): _parse_delta_value(v) for k, v in delta_overrides.items()}


def _normalize_allowed_fusion_pairs(raw: Any) -> Optional[List[FusionPair]]:
    if raw is None or raw == "all":
        return None
    if raw == DEFAULT_FUSION_POLICY:
        raise ValueError("omit allowed_fusion_pairs to use the graph default policy")
    if raw == "none":
        return []
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(
            "allowed_fusion_pairs must be a list of [stage_a, stage_b] pairs, "
            "'all', 'none', or omitted for the default policy"
        )

    out: List[FusionPair] = []
    for pair in raw:
        if not isinstance(pair, Sequence) or isinstance(pair, (str, bytes)) or len(pair) != 2:
            raise ValueError(f"allowed fusion pair must be [stage_a, stage_b], got {pair!r}")
        a, b = int(pair[0]), int(pair[1])
        if a == b:
            raise ValueError(f"allowed fusion pair cannot fuse a stage with itself: {pair!r}")
        if a > b:
            a, b = b, a
        out.append((a, b))
    return out


def default_allowed_fusion_pairs_for_graph(graph_key: str) -> Optional[List[FusionPair]]:
    key = str(graph_key)
    if key in _DEFAULT_ALLOWED_FUSION_PAIRS:
        return list(_DEFAULT_ALLOWED_FUSION_PAIRS[key])
    return None


def resolve_allowed_fusion_pairs(
    graph_key: str,
    allowed_fusion_pairs: Any = DEFAULT_FUSION_POLICY,
) -> Optional[List[FusionPair]]:
    if allowed_fusion_pairs == DEFAULT_FUSION_POLICY:
        return default_allowed_fusion_pairs_for_graph(graph_key)
    return _normalize_allowed_fusion_pairs(allowed_fusion_pairs)


def _fusion_pairs_to_json(allowed_fusion_pairs: Optional[Sequence[FusionPair]]) -> Optional[List[List[int]]]:
    if allowed_fusion_pairs is None:
        return None
    return [[int(a), int(b)] for a, b in allowed_fusion_pairs]


def split_replan_payload(
    payload: Any,
    *,
    include_allowed_fusion_pairs: bool = False,
):
    """Split an invoker payload into ``(t_new, delta_overrides)``.

    Supports two shapes:

    * ``{"t_new": [...], "delta_overrides": {...}}`` for the variable API.
    * ``{"node_name": delta, ...}`` for the bare-delta shorthand.
    * optionally ``{"allowed_fusion_pairs": [[1, 2], ...]}`` for legal
      rescale-fusion boundaries; omit it to use the graph default policy.
    """

    if payload is None:
        base = (None, {})
        return (*base, DEFAULT_FUSION_POLICY) if include_allowed_fusion_pairs else base
    if not isinstance(payload, Mapping):
        raise TypeError(f"payload must be a mapping, got {type(payload).__name__}")

    keys = set(payload.keys())
    structured_keys = {
        "t_new",
        "delta_overrides",
        "propagation_deltas",
        "allowed_fusion_pairs",
        "fusion_pairs",
    }
    if keys & structured_keys:
        t_raw = payload.get("t_new")
        t_new = [int(x) for x in t_raw] if t_raw is not None else None
        deltas: Dict[str, Any] = {}
        if payload.get("delta_overrides"):
            raw = payload["delta_overrides"]
            if not isinstance(raw, Mapping):
                raise ValueError("payload['delta_overrides'] must be a mapping")
            deltas.update(raw)
        if payload.get("propagation_deltas"):
            rows = payload["propagation_deltas"]
            if not isinstance(rows, list):
                raise ValueError("payload['propagation_deltas'] must be a list")
            for row in rows:
                if not isinstance(row, Mapping) or "name" not in row or "delta" not in row:
                    raise ValueError("each propagation_deltas item must have {name, delta}")
                deltas[str(row["name"])] = row["delta"]
        allowed = DEFAULT_FUSION_POLICY
        if "allowed_fusion_pairs" in payload:
            allowed = _normalize_allowed_fusion_pairs(payload.get("allowed_fusion_pairs"))
        elif "fusion_pairs" in payload:
            allowed = _normalize_allowed_fusion_pairs(payload.get("fusion_pairs"))
        base = (t_new, _normalize_delta_overrides(deltas))
        return (*base, allowed) if include_allowed_fusion_pairs else base

    base = (None, _normalize_delta_overrides(payload))
    return (*base, DEFAULT_FUSION_POLICY) if include_allowed_fusion_pairs else base


def load_static_skeleton_baselines(path: Union[str, Path]) -> Dict[str, BaselineRecord]:
    """Load ``static_skeletons_<profile>.json`` into baseline records."""

    archive_path = Path(path)
    with archive_path.open("r", encoding="utf-8") as f:
        doc = json.load(f)

    out: Dict[str, BaselineRecord] = {}
    for entry in doc.get("results", []):
        if not isinstance(entry, Mapping) or not entry.get("success", False):
            continue
        config_name = str(entry["config_name"])
        skeleton = [int(x) for x in entry.get("skeleton", [])]

        if "cut_point_sf" in entry and "modulus_chain" in entry:
            t_for_idx: Dict[int, int] = {}
            for row in entry.get("cut_point_sf", []):
                i = int(row["i"])
                if "sf_post" in row:
                    t_for_idx[i] = int(row["sf_post"])
                elif "sf" in row:
                    t_for_idx[i] = int(row["sf"])
            t_baseline = [t_for_idx[i] for i in skeleton if i in t_for_idx]
            drop_order = list((entry.get("modulus_chain") or {}).get("drop_order", []))
            q_bits = [int(x) for x in drop_order[1:-1]] if len(drop_order) >= 2 else []
        else:
            t_baseline = [int(x) for x in entry.get("t_per_stage", [])]
            q_bits = [
                int(x)
                for x in (
                    entry.get("dp_drop_bits")
                    or entry.get("drop_bits_per_stage")
                    or []
                )
            ]

        out[config_name] = BaselineRecord(
            config_name=config_name,
            skeleton=skeleton,
            t_baseline=t_baseline,
            q_bits_baseline=q_bits,
            archive_entry=dict(entry),
        )
    return out


def _snapshot_graph_delta_state(
    graph: RescaleGraph,
) -> List[Tuple[int, Optional[int]]]:
    return [
        (
            int(getattr(node, "scale_delta_bits", 0)),
            getattr(node, "other_ct_scale_bits", None),
        )
        for node in graph.nodes
    ]


def _restore_graph_delta_state(
    graph: RescaleGraph,
    state: Sequence[Tuple[int, Optional[int]]],
) -> None:
    for node, (scale_delta_bits, other_ct_scale_bits) in zip(graph.nodes, state):
        node.scale_delta_bits = int(scale_delta_bits)
        node.other_ct_scale_bits = other_ct_scale_bits


def _extract_current_propagation_deltas(graph: RescaleGraph) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for node in graph.nodes:
        if node.node_type == NodeType.CTPT_MUL:
            rows.append({
                "node_id": int(node.node_id),
                "name": node.name,
                "type": "CTPT_MUL",
                "delta": int(node.scale_delta_bits),
            })
        elif node.node_type == NodeType.CTCT_MUL:
            rows.append({
                "node_id": int(node.node_id),
                "name": node.name,
                "type": "CTCT_MUL",
                "delta": (
                    "x2"
                    if node.other_ct_scale_bits is None
                    else int(node.other_ct_scale_bits)
                ),
            })
    return rows


def build_new_compact_config(
    graph: RescaleGraph,
    config_name: str,
    result: ReplanResult,
) -> Optional[Dict[str, Any]]:
    """Build a compact deployable config from a valid replan result."""

    if not result.valid or result.chain is None:
        return None

    chain = result.chain
    skeleton = [int(x) for x in result.skeleton]
    t_vec = [int(x) for x in result.t_final]
    M = graph.M
    R = chain.R

    rescale_index_at: Dict[int, int] = {skeleton[r]: r for r in range(1, R + 1)}
    cut_point_sf: List[Dict[str, Any]] = []
    current_scale = int(t_vec[0])

    for i in range(M + 1):
        cp = graph.cut_points[i]
        type_name = cp.node.node_type.name

        if i == skeleton[0]:
            cut_point_sf.append({
                "i": i,
                "name": cp.node.name,
                "type": type_name,
                "sf": current_scale,
            })
            continue

        current_scale = int(
            propagate_scale(current_scale, graph.stage_node_lists[i - 1])
        )
        if i in rescale_index_at:
            r = rescale_index_at[i]
            cut_point_sf.append({
                "i": i,
                "name": cp.node.name,
                "type": type_name,
                "sf_pre": current_scale,
                "sf_post": int(t_vec[r]),
                "drop": int(chain.q_bits[r - 1]),
            })
            current_scale = int(t_vec[r])
            continue

        cut_point_sf.append({
            "i": i,
            "name": cp.node.name,
            "type": type_name,
            "sf": current_scale,
        })

    drop_order = [
        int(chain.q_head_bits),
        *[int(b) for b in chain.q_bits],
        int(chain.q_tail_bits),
    ]
    seal_order = [
        int(chain.q_head_bits),
        *reversed([int(b) for b in chain.q_bits]),
        int(chain.q_tail_bits),
    ]

    skel_full = list(skeleton)
    if skel_full[-1] != graph.dummy_sink_index:
        skel_full.append(graph.dummy_sink_index)
    effective_index_at: Dict[int, int] = {
        skel_full[r]: r for r in range(1, len(skel_full) - 1)
    }
    effective_rotations: List[Dict[str, Any]] = []
    for node in graph.nodes:
        if node.node_type != NodeType.ROTATION:
            continue
        k = int(node.stage_anchor)
        if k not in effective_index_at:
            continue
        r = effective_index_at[k]
        effective_rotations.append({
            "node_id": int(node.node_id),
            "name": node.name,
            "after_cut_point": k,
            "sf": int(t_vec[r]),
            "count": int(node.count),
        })

    return {
        "config_name": config_name,
        "success": True,
        "skeleton": [int(x) for x in skeleton],
        "cut_point_sf": cut_point_sf,
        "propagation_deltas": _extract_current_propagation_deltas(graph),
        "modulus_chain": {
            "drop_order": drop_order,
            "seal_order": seal_order,
            "total_bits": int(chain.total_bits),
        },
        "effective_rotations": effective_rotations,
    }


def _chain_to_dict(chain: Any) -> Optional[Dict[str, Any]]:
    if chain is None:
        return None
    out = {
        "q_head_bits": int(chain.q_head_bits),
        "q_bits": [int(x) for x in chain.q_bits],
        "q_tail_bits": int(chain.q_tail_bits),
    }
    if hasattr(chain, "total_bits"):
        out["total_bits"] = int(chain.total_bits)
    if hasattr(chain, "R"):
        out["R"] = int(chain.R)
    return out


def build_replan_output_dict(
    *,
    graph: RescaleGraph,
    graph_key: str,
    result: ReplanResult,
    skeleton: Sequence[int],
    t_baseline: Optional[Sequence[int]],
    q_bits_baseline: Optional[Sequence[int]],
    t_new: Sequence[int],
    delta_overrides: Optional[Mapping[str, Any]] = None,
    allowed_fusion_pairs: Optional[Sequence[FusionPair]] = None,
    config_path: Optional[Union[str, Path]] = None,
    output_config_name: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    include_compact: bool = True,
) -> Dict[str, Any]:
    """Convert a ``ReplanResult`` to the canonical JSON result shape."""

    deltas = _normalize_delta_overrides(delta_overrides)
    doc: Dict[str, Any] = {
        "config_name": output_config_name or graph_key,
        "graph_key": graph_key,
        "valid": bool(result.valid),
        "fusion_count": int(result.fusion_count),
        "baseline": {
            "skeleton": [int(x) for x in skeleton],
            "t_baseline": None if t_baseline is None else [int(x) for x in t_baseline],
            "q_bits_baseline": (
                None if q_bits_baseline is None else [int(x) for x in q_bits_baseline]
            ),
        },
        "t_new": [int(x) for x in t_new],
        "delta_overrides": dict(deltas),
        "allowed_fusion_pairs": _fusion_pairs_to_json(allowed_fusion_pairs),
        "result": {
            "valid": bool(result.valid),
            "message": str(result.message),
            "fusion_count": int(result.fusion_count),
            "skeleton": [int(x) for x in result.skeleton],
            "q_initial": [int(x) for x in result.q_initial],
            "q_final": [int(x) for x in result.q_final],
            "t_final": [int(x) for x in result.t_final],
            "delta_q_vs_baseline": [int(x) for x in result.delta_q_vs_baseline],
            "applied_delta_overrides": dict(result.applied_delta_overrides),
            "fusions": [
                {
                    "fused_position": int(ev.fused_position),
                    "fused_into": ev.fused_into,
                    "small_q": int(ev.small_q),
                    "neighbour_q_before": int(ev.neighbour_q_before),
                    "neighbour_q_after": int(ev.neighbour_q_after),
                }
                for ev in result.fusions
            ],
            "chain": _chain_to_dict(result.chain),
            "invalid_chain": _chain_to_dict(result.invalid_chain),
        },
    }
    if config_path is not None:
        doc["config_path"] = str(config_path)
    if metadata:
        doc.update(dict(metadata))

    if include_compact:
        compact = build_new_compact_config(graph, graph_key, result)
        if compact is not None:
            compact["fusion_count"] = int(result.fusion_count)
            doc["new_compact_config"] = compact
    return doc


def replan_from_variables(
    *,
    config_path: Union[str, Path],
    t_new: Sequence[int],
    skeleton: Optional[Sequence[int]] = None,
    t_baseline: Optional[Sequence[int]] = None,
    q_bits_baseline: Optional[Sequence[int]] = None,
    baseline_archive: Optional[Union[str, Path]] = None,
    config_name: Optional[str] = None,
    delta_overrides: Optional[Mapping[str, Any]] = None,
    allowed_fusion_pairs: Any = DEFAULT_FUSION_POLICY,
    return_dict: bool = True,
    include_compact: bool = True,
    output_config_name: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Union[Dict[str, Any], ReplanResult]:
    """Single-shot variable API.

    Either pass ``skeleton`` directly, or pass ``baseline_archive`` plus
    ``config_name`` so the baseline can be looked up in a static-skeletons file.
    """

    cfg_path = Path(config_path)
    graph_key = config_name or cfg_path.stem
    graph, _opt_cfg, _amp = load_graph_from_json(cfg_path)
    build_feasibility_dag(graph)

    baseline_record: Optional[BaselineRecord] = None
    if baseline_archive is not None:
        baselines = load_static_skeleton_baselines(baseline_archive)
        if graph_key not in baselines:
            raise KeyError(f"baseline archive has no successful entry for {graph_key!r}")
        baseline_record = baselines[graph_key]

    skel = list(skeleton) if skeleton is not None else (
        list(baseline_record.skeleton) if baseline_record is not None else None
    )
    if not skel:
        raise ValueError("skeleton is required when baseline_archive is not provided")
    if skel[-1] != graph.dummy_sink_index:
        skel.append(graph.dummy_sink_index)

    t_base = (
        [int(x) for x in t_baseline]
        if t_baseline is not None
        else (list(baseline_record.t_baseline) if baseline_record is not None else None)
    )
    q_base = (
        [int(x) for x in q_bits_baseline]
        if q_bits_baseline is not None
        else (list(baseline_record.q_bits_baseline) if baseline_record is not None else None)
    )
    t_new_eff = [int(x) for x in t_new]
    deltas = _normalize_delta_overrides(delta_overrides)
    fusion_pairs = resolve_allowed_fusion_pairs(graph_key, allowed_fusion_pairs)

    result = replan_with_user_actions(
        graph,
        ReplanInputs(
            skeleton=skel,
            t_baseline=t_base,
            t_new=t_new_eff,
            delta_overrides=(deltas or None),
            allowed_fusion_pairs=fusion_pairs,
        ),
        baseline_q_bits=q_base,
    )
    if not return_dict:
        return result
    return build_replan_output_dict(
        graph=graph,
        graph_key=graph_key,
        result=result,
        skeleton=skel,
        t_baseline=t_base,
        q_bits_baseline=q_base,
        t_new=t_new_eff,
        delta_overrides=deltas,
        allowed_fusion_pairs=fusion_pairs,
        config_path=cfg_path,
        output_config_name=output_config_name,
        metadata=metadata,
        include_compact=include_compact,
    )


class ReplanSession:
    """Preloaded in-process replan session for repeated variable calls."""

    def __init__(
        self,
        *,
        configs: Mapping[str, Union[str, Path]],
        baseline_archive: Union[str, Path],
        baselines: Optional[Mapping[str, BaselineRecord]] = None,
    ) -> None:
        self.configs = {str(k): Path(v) for k, v in configs.items()}
        self.baseline_archive = Path(baseline_archive)
        self.baselines = (
            dict(baselines)
            if baselines is not None
            else load_static_skeleton_baselines(self.baseline_archive)
        )

        self._graphs: Dict[str, RescaleGraph] = {}
        self._delta_baselines: Dict[str, List[Tuple[int, Optional[int]]]] = {}
        self._stage_paths: Dict[str, Tuple[Tuple[ComputeNode, ...], ...]] = {}
        self._delta_nodes: Dict[str, Dict[str, ComputeNode]] = {}
        self._delta_state_clean: Dict[str, bool] = {}
        self._default_fusion_policies: Dict[
            str,
            Tuple[Optional[List[FusionPair]], _PreparedFusionPairs],
        ] = {}
        for graph_key, path in self.configs.items():
            graph, _opt_cfg, _amp = load_graph_from_json(path)
            build_feasibility_dag(graph)
            self._graphs[graph_key] = graph
            self._delta_baselines[graph_key] = _snapshot_graph_delta_state(graph)
            self._delta_state_clean[graph_key] = True
            self._delta_nodes[graph_key] = {
                node.name: node
                for node in graph.nodes
                if node.node_type in (NodeType.CTPT_MUL, NodeType.CTCT_MUL)
            }
            default_fusion_pairs = resolve_allowed_fusion_pairs(graph_key)
            self._default_fusion_policies[graph_key] = (
                default_fusion_pairs,
                _prepare_allowed_fusion_pairs(default_fusion_pairs),
            )
            baseline = self.baselines.get(graph_key)
            if baseline is not None:
                skeleton = list(baseline.skeleton)
                if skeleton and skeleton[-1] != graph.dummy_sink_index:
                    skeleton.append(graph.dummy_sink_index)
                self._stage_paths[graph_key] = tuple(
                    tuple(graph.nodes_between(skeleton[r - 1], skeleton[r]))
                    for r in range(1, len(skeleton) - 1)
                )

    @classmethod
    def from_profile(
        cls,
        *,
        profile: str,
        root: Optional[Union[str, Path]] = None,
        configs_dir: Optional[Union[str, Path]] = None,
        baseline_archive: Optional[Union[str, Path]] = None,
        include: Optional[Iterable[str]] = None,
    ) -> "ReplanSession":
        """Create a session by scanning one profile under the config root."""

        if root is None:
            from .. import RESCALE_CONFIG_ROOT

            config_root = RESCALE_CONFIG_ROOT
        else:
            config_root = Path(root)
        cfg_dir = Path(configs_dir) if configs_dir is not None else config_root / profile
        archive = (
            Path(baseline_archive)
            if baseline_archive is not None
            else cfg_dir / f"static_skeletons_{profile}.json"
        )
        include_set = None if include is None else {str(x) for x in include}
        baselines = load_static_skeleton_baselines(archive)
        configs: Dict[str, Path] = {}
        for graph_key in baselines:
            if include_set is not None and graph_key not in include_set:
                continue
            path = cfg_dir / f"{graph_key}.json"
            if path.exists():
                configs[graph_key] = path
        return cls(configs=configs, baseline_archive=archive, baselines=baselines)

    @property
    def graph_keys(self) -> List[str]:
        return sorted(self._graphs.keys())

    def replan(
        self,
        graph_key: str,
        *,
        t_new: Optional[Sequence[int]] = None,
        delta_overrides: Optional[Mapping[str, Any]] = None,
        allowed_fusion_pairs: Any = DEFAULT_FUSION_POLICY,
        return_dict: bool = True,
        include_compact: bool = True,
        output_config_name: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        _compact_result: bool = False,
    ) -> Union[Dict[str, Any], ReplanResult, CompactReplanResult]:
        """Run one replan call from variable inputs.

        If ``t_new`` is omitted, the baseline ``t_baseline`` is reused. This is
        useful when the caller only wants to evaluate propagation delta changes.
        """

        key = str(graph_key)
        if key not in self._graphs:
            raise KeyError(f"unknown graph_key={key!r}; available={self.graph_keys}")
        if key not in self.baselines:
            raise KeyError(f"baseline archive has no successful entry for {key!r}")

        graph = self._graphs[key]
        baseline = self.baselines[key]
        if not self._delta_state_clean[key]:
            _restore_graph_delta_state(graph, self._delta_baselines[key])
        self._delta_state_clean[key] = False

        skeleton = list(baseline.skeleton)
        if skeleton[-1] != graph.dummy_sink_index:
            skeleton.append(graph.dummy_sink_index)
        t_eff = [int(x) for x in (t_new if t_new is not None else baseline.t_baseline)]
        deltas = _normalize_delta_overrides(delta_overrides)
        if allowed_fusion_pairs == DEFAULT_FUSION_POLICY:
            fusion_pairs, replan_fusion_pairs = self._default_fusion_policies[key]
        else:
            fusion_pairs = resolve_allowed_fusion_pairs(key, allowed_fusion_pairs)
            replan_fusion_pairs = fusion_pairs

        result = replan_with_user_actions(
            graph,
            ReplanInputs(
                skeleton=skeleton,
                t_baseline=list(baseline.t_baseline),
                t_new=t_eff,
                delta_overrides=(deltas or None),
                allowed_fusion_pairs=replan_fusion_pairs,
            ),
            baseline_q_bits=(
                None if _compact_result else list(baseline.q_bits_baseline)
            ),
            stage_paths=self._stage_paths.get(key),
            delta_nodes=self._delta_nodes.get(key),
            record_applied_delta_overrides=not _compact_result,
        )

        if _compact_result:
            compact = build_new_compact_config(graph, key, result)
            if compact is not None:
                compact["fusion_count"] = int(result.fusion_count)
            output = CompactReplanResult(
                valid=bool(result.valid and result.invalid_chain is None),
                fusion_count=int(result.fusion_count),
                total_bits=(
                    int(result.chain.total_bits) if result.chain is not None else 0
                ),
                compact_config=compact,
            )
            _restore_graph_delta_state(graph, self._delta_baselines[key])
            self._delta_state_clean[key] = True
            return output

        if not return_dict:
            _restore_graph_delta_state(graph, self._delta_baselines[key])
            self._delta_state_clean[key] = True
            return result

        doc = build_replan_output_dict(
            graph=graph,
            graph_key=key,
            result=result,
            skeleton=skeleton,
            t_baseline=baseline.t_baseline,
            q_bits_baseline=baseline.q_bits_baseline,
            t_new=t_eff,
            delta_overrides=deltas,
            allowed_fusion_pairs=fusion_pairs,
            config_path=self.configs.get(key),
            output_config_name=output_config_name,
            metadata=metadata,
            include_compact=include_compact,
        )
        _restore_graph_delta_state(graph, self._delta_baselines[key])
        self._delta_state_clean[key] = True
        return doc

    def replan_compact(
        self,
        graph_key: str,
        *,
        t_new: Optional[Sequence[int]] = None,
        delta_overrides: Optional[Mapping[str, Any]] = None,
        allowed_fusion_pairs: Any = DEFAULT_FUSION_POLICY,
    ) -> CompactReplanResult:
        """Run replan without expanding the compatibility JSON document."""

        return self.replan(  # type: ignore[return-value]
            graph_key,
            t_new=t_new,
            delta_overrides=delta_overrides,
            allowed_fusion_pairs=allowed_fusion_pairs,
            _compact_result=True,
        )

    def __call__(self, graph_key: str, payload: Any) -> Dict[str, Any]:
        """Compatibility invoker: ``session(graph_key, payload) -> dict``."""

        t_new, delta_overrides, allowed_fusion_pairs = split_replan_payload(
            payload, include_allowed_fusion_pairs=True
        )
        return self.replan(
            graph_key,
            t_new=t_new,
            delta_overrides=delta_overrides,
            allowed_fusion_pairs=allowed_fusion_pairs,
            return_dict=True,
        )


def graph_key_for_stage1(
    *,
    dataset: str,
    block: int,
    layer: int,
    stage1_config: Mapping[str, Sequence[int]],
) -> str:
    """Map Stage-1 per-layer degrees to a Rescale graph key."""

    def _degrees(long_key: str, short_key: str) -> Sequence[int]:
        values = stage1_config.get(long_key)
        if values is None:
            values = stage1_config.get(short_key)
        if values is None:
            raise KeyError(f"stage1_config missing {long_key} / {short_key}")
        return values

    ds = str(dataset)
    if block == 1:
        return f"block1_{ds}"
    if block == 2:
        return f"block2_{ds}"
    if block == 3:
        degrees = _degrees("softmax_degree_per_layer", "softmax")
        return f"block3_exp_n{int(degrees[layer])}"
    if block == 4:
        return "block4"
    if block == 5:
        degrees = _degrees("gelu_degree_per_layer", "gelu")
        degree = int(degrees[layer])
        if degree not in SUPPORTED_GELU_GRAPH_DEGREES:
            raise ValueError(
                f"unsupported block5 GELU degree {degree}; "
                f"supported={sorted(SUPPORTED_GELU_GRAPH_DEGREES)}"
            )
        return f"block5_n{degree}"
    raise ValueError(f"block must be 1..5, got {block}")


def iter_stage2_graph_targets(
    *,
    dataset: str,
    num_layers: int,
    stage1_config: Mapping[str, Sequence[int]],
    skip_block1_layer0: bool = True,
) -> List[Dict[str, Any]]:
    """Return the per-(block, layer) graph selection implied by Stage-1."""

    targets: List[Dict[str, Any]] = []
    for layer in range(int(num_layers)):
        for block in range(1, 6):
            if skip_block1_layer0 and block == 1 and layer == 0:
                continue
            graph_key = graph_key_for_stage1(
                dataset=dataset,
                block=block,
                layer=layer,
                stage1_config=stage1_config,
            )
            targets.append({
                "config_name": f"{graph_key}_L{layer}",
                "graph_key": graph_key,
                "block": block,
                "layer": layer,
            })
    return targets


__all__ = [
    "BaselineRecord",
    "DEFAULT_FUSION_POLICY",
    "DeltaValue",
    "FusionPair",
    "ReplanSession",
    "SUPPORTED_GELU_GRAPH_DEGREES",
    "build_new_compact_config",
    "build_replan_output_dict",
    "default_allowed_fusion_pairs_for_graph",
    "graph_key_for_stage1",
    "iter_stage2_graph_targets",
    "load_static_skeleton_baselines",
    "replan_from_variables",
    "resolve_allowed_fusion_pairs",
    "split_replan_payload",
]
