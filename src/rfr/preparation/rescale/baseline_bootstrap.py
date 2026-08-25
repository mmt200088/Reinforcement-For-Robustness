"""Build the Stage-2 baseline from checked-in static Rescale skeletons."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from blb_stage2_rl import skeleton_stage_map as _ssm
from blb_stage2_rl.reward import BaselineCostStats


ALLOWED_GELU_DEGREES = (1, 2, 4)
ALLOWED_SOFTMAX_DEGREES = (2, 3, 4, 5, 6)


def resolve_stage2_model_type(
        model_type: str,
        *,
        num_layers: int,
        ) -> str:
    """Return the canonical BERT model identity for a Stage-2 depth."""
    layers = int(num_layers)
    expected = {
        12: "bert-base",
        24: "bert-large",
    }.get(layers)
    if expected is None:
        raise ValueError(
            f"Stage-2 currently supports 12 or 24 Transformer layers, got {layers}"
        )

    raw_model = str(model_type or "").strip().lower()
    normalized = raw_model.replace("_", "-").replace(" ", "-")
    if not normalized:
        return expected
    if "large" in normalized and layers != 24:
        raise ValueError(
            f"inconsistent Stage-2 model/depth: model_type={model_type!r}, "
            f"num_layers={layers}"
        )
    if "base" in normalized and layers != 12:
        raise ValueError(
            f"inconsistent Stage-2 model/depth: model_type={model_type!r}, "
            f"num_layers={layers}"
        )
    if "large" in normalized or "base" in normalized:
        return expected
    return normalized


def resolve_stage2_profile(
        dataset: str,
        *,
        model_type: str,
        num_layers: int,
        ) -> str:
    """Resolve the model-aware RO/fusion-map profile used by Stage-2."""
    raw_dataset = str(dataset or "").strip().lower()
    if not raw_dataset:
        raise ValueError("dataset must be nonempty")
    layers = int(num_layers)
    resolve_stage2_model_type(model_type, num_layers=layers)

    dataset_declares_large = raw_dataset.endswith("_large")
    if dataset_declares_large and layers != 24:
        raise ValueError(
            f"inconsistent Stage-2 profile/depth: dataset={dataset!r}, "
            f"num_layers={layers}"
        )
    base_dataset = (
        raw_dataset[:-len("_large")]
        if dataset_declares_large else raw_dataset
    )
    return f"{base_dataset}_large" if layers == 24 else base_dataset


def _validate_stage1(
        gelu_per_layer: Sequence[int],
        softmax_per_layer: Sequence[int],
        num_layers: int,
        ) -> None:
    if len(gelu_per_layer) != int(num_layers):
        raise ValueError(
            f"gelu_degree_per_layer length {len(gelu_per_layer)} != num_layers {num_layers}"
        )
    if len(softmax_per_layer) != int(num_layers):
        raise ValueError(
            f"softmax_degree_per_layer length {len(softmax_per_layer)} != num_layers {num_layers}"
        )
    bad_gelu = [int(d) for d in gelu_per_layer if int(d) not in ALLOWED_GELU_DEGREES]
    if bad_gelu:
        raise ValueError(
            f"gelu_degree_per_layer contains values outside "
            f"{ALLOWED_GELU_DEGREES}: {bad_gelu}"
        )
    bad_sm = [int(d) for d in softmax_per_layer if int(d) not in ALLOWED_SOFTMAX_DEGREES]
    if bad_sm:
        raise ValueError(
            f"softmax_degree_per_layer contains values outside {ALLOWED_SOFTMAX_DEGREES}: {bad_sm}"
        )



def static_skeletons_archive_path(rescale_optimizer_root: str, dataset: str) -> str:
    """Return the profile archive under the Rescale configuration root."""
    return os.path.join(
        os.path.abspath(str(rescale_optimizer_root)),
        str(dataset),
        f"static_skeletons_{dataset}.json",
    )


def load_static_skeletons_archive(path: str) -> Dict[str, Dict[str, Any]]:
    """读 ``static_skeletons_<dataset>.json``，返回 ``{config_name: entry}`` 索引。

    只保留 ``success=True`` 的 entry。
    """
    with open(str(path), "r", encoding="utf-8") as f:
        archive = json.load(f)
    out: Dict[str, Dict[str, Any]] = {}
    for entry in archive.get("results", []) or []:
        if not entry.get("success"):
            continue
        cname = str(entry.get("config_name") or "").strip()
        if cname:
            out[cname] = dict(entry)
    return out


def static_skeletons_graph_key(
        block_idx: int,
        dataset: str,
        gelu_degree: int,
        softmax_degree: int,
        ) -> str:
    """对照 ``action_space.make_config_name`` 的命名规则给 (block, dataset, deg) → graph_key。"""
    block_idx = int(block_idx)
    if block_idx == 1:
        return f"block1_{dataset}"
    if block_idx == 2:
        return f"block2_{dataset}"
    if block_idx == 3:
        return f"block3_exp_n{int(softmax_degree)}"
    if block_idx == 4:
        return "block4"
    if block_idx == 5:
        return f"block5_n{int(gelu_degree)}"
    raise ValueError(f"invalid block_idx {block_idx}")


_RO_X2_AUX_FRESH_FIELD: Dict[int, Dict[str, str]] = {
    2: {"ctct_x_mean_over_std": "x_centered_fresh_sf"},
    5: {"ctct_xmean_over_std":  "inv_std_fresh_sf"},
}


@dataclass
class StaticSkeletonsLayerBlock:
    """单个 (block, layer) 的 RO baseline 抽取结果。"""
    block_idx: int
    layer_idx: int
    graph_key: str

    field_baseline_sfs: Dict[str, int] = field(default_factory=dict)

    field_kind_in_ro: Dict[str, str] = field(default_factory=dict)

    total_bits: int = 0
    fusion_count: int = 0
    drop_order: List[int] = field(default_factory=list)

    effective_rotations: List[Dict[str, Any]] = field(default_factory=list)

    unmapped_propagation_nodes: List[str] = field(default_factory=list)
    unmapped_rescale_nodes: List[str] = field(default_factory=list)


@dataclass
class StaticSkeletonsBaseline:
    """整个模型按 Stage-1 配置抽出的 BLB baseline。"""
    dataset: str
    num_layers: int
    gelu_per_layer: List[int]
    softmax_per_layer: List[int]
    archive_path: str
    per_block_layer: Dict[Tuple[int, int], StaticSkeletonsLayerBlock] = field(default_factory=dict)
    aggregate_total_bits: int = 0
    aggregate_fusion_count: int = 0
    aggregate_valid_block_count: int = 0
    aggregate_invalid_block_count: int = 0

    missing_block_layer: List[Tuple[int, int]] = field(default_factory=list)


@dataclass(frozen=True)
class Stage2CalibratedActionContext:
    """Exact Stage-2 baseline inputs shared by training and replay surfaces."""

    baseline: StaticSkeletonsBaseline
    baseline_action_vec: Any
    max_sfs: Any
    cost_stats: BaselineCostStats
    diagnostics: Mapping[str, Any]
    provenance: Mapping[str, Any]


def _extract_one_block_layer(
        entry: Mapping[str, Any],
        block_idx: int,
        layer_idx: int,
        graph_key: str,
        *,
        gelu_degree: int,
        ) -> StaticSkeletonsLayerBlock:
    """从一条 archive entry 抽出该 (block, layer) 的 RL 字段 baseline。"""
    out = StaticSkeletonsLayerBlock(
        block_idx=int(block_idx),
        layer_idx=int(layer_idx),
        graph_key=str(graph_key),
    )


    cps = entry.get("cut_point_sf") or []
    if not isinstance(cps, list) or not cps:
        return out


    source_entry = cps[0] if isinstance(cps[0], Mapping) else {}
    source_sf: Optional[int] = None
    if str(source_entry.get("type", "")) == "SOURCE":


        rl_field = _ssm.source_rl_field(int(block_idx))
        sf = source_entry.get("sf")
        if sf is not None:
            try:
                source_sf = int(sf)
            except (TypeError, ValueError):
                source_sf = None
        if rl_field and source_sf is not None:
            out.field_baseline_sfs[rl_field] = source_sf
            out.field_kind_in_ro[rl_field] = "fresh"


    for cp in cps[1:]:
        if not isinstance(cp, Mapping):
            continue
        if "sf_post" not in cp:
            continue
        name = str(cp.get("name", ""))
        sf_post = cp.get("sf_post")
        if sf_post is None:
            continue


        rl_fields = _ssm.rescale_rl_fields(int(block_idx), name)
        if not rl_fields:
            out.unmapped_rescale_nodes.append(name)
            continue
        for rl_field in rl_fields:
            out.field_baseline_sfs[rl_field] = int(sf_post)
            out.field_kind_in_ro[rl_field] = "rescale"


    pd_delta_by_name: Dict[str, Any] = {}
    for pd in entry.get("propagation_deltas") or []:
        if not isinstance(pd, Mapping):
            continue
        name = str(pd.get("name", ""))
        delta = pd.get("delta")
        if name:
            pd_delta_by_name[name] = delta
        if not isinstance(delta, (int, float)):
            continue

        rl_fields = _ssm.encode_rl_fields(int(block_idx), name)
        if not rl_fields:
            out.unmapped_propagation_nodes.append(name)
            continue
        for rl_field in rl_fields:
            out.field_baseline_sfs[rl_field] = int(delta)
            out.field_kind_in_ro[rl_field] = "encode"


    if int(block_idx) == 4:
        mulv_delta = pd_delta_by_name.get("ctct_rot_softmax_mul_v")
        mask2_delta = pd_delta_by_name.get("ctpt_mask2")
        if isinstance(mulv_delta, (int, float)) and isinstance(mask2_delta, (int, float)):
            v_fresh_sf = int(mulv_delta) - int(mask2_delta)
            out.field_baseline_sfs["v_fresh_sf"] = int(v_fresh_sf)
            out.field_kind_in_ro["v_fresh_sf"] = "fresh"


    if source_sf is not None:
        aux_map = _RO_X2_AUX_FRESH_FIELD.get(int(block_idx), {})
        for side_name, aux_field in aux_map.items():
            if str(pd_delta_by_name.get(side_name)) == "x2":
                out.field_baseline_sfs[aux_field] = int(source_sf)
                out.field_kind_in_ro[aux_field] = "fresh"


    mc = entry.get("modulus_chain") or {}
    if isinstance(mc, Mapping):
        out.total_bits = int(mc.get("total_bits", 0))
        out.drop_order = [int(x) for x in (mc.get("drop_order") or [])]


    out.fusion_count = int(entry.get("fusion_count", 0))


    er = entry.get("effective_rotations") or []
    if isinstance(er, list):
        out.effective_rotations = [dict(x) for x in er if isinstance(x, Mapping)]

    return out


def load_static_skeletons_baseline(
        rescale_optimizer_root: str,
        dataset: str,
        num_layers: int,
        gelu_per_layer: Sequence[int],
        softmax_per_layer: Sequence[int],
        *,
        archive_path: Optional[str] = None,
        ) -> StaticSkeletonsBaseline:
    """从 ``static_skeletons_<dataset>.json`` 抽出 BLB Stage-2 RL baseline。

    Args:
        rescale_optimizer_root: Rescale configuration root
        dataset:                supported GLUE task name
        num_layers:             模型 encoder 层数
        gelu_per_layer:         长度 num_layers，元素 ∈ {1, 2, 4}
        softmax_per_layer:      长度 num_layers，元素 ∈ {2, 3, 4, 5, 6}
        archive_path:           手动指定 archive 路径；缺省自动拼

    Returns:
        ``StaticSkeletonsBaseline``。``per_block_layer`` 不包含 (1, 0)
        —— layer-0 Block1 没有 SF/fusion baseline；其 K-only model cfg 由
        action decoder 独立物化。

    Raises:
        FileNotFoundError: archive 路径不存在
        BaselineHandoverError: archive schema 不对，或某层的 graph_key 不在 archive
    """
    _validate_stage1(gelu_per_layer, softmax_per_layer, int(num_layers))
    path = archive_path or static_skeletons_archive_path(rescale_optimizer_root, dataset)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"static_skeletons archive not found: {path} "
            f"(expected profile directory {dataset}/ under the configuration root)"
        )
    archive = load_static_skeletons_archive(path)

    out = StaticSkeletonsBaseline(
        dataset=str(dataset),
        num_layers=int(num_layers),
        gelu_per_layer=[int(d) for d in gelu_per_layer],
        softmax_per_layer=[int(d) for d in softmax_per_layer],
        archive_path=str(path),
    )

    for layer_idx in range(int(num_layers)):
        gelu_deg = int(gelu_per_layer[layer_idx])
        softmax_deg = int(softmax_per_layer[layer_idx])


        for block_idx in (1, 2, 3, 4, 5):

            if int(layer_idx) == 0 and int(block_idx) == 1:
                continue
            graph_key = static_skeletons_graph_key(
                block_idx, str(dataset), gelu_deg, softmax_deg,
            )
            entry = archive.get(graph_key)
            if entry is None:
                out.missing_block_layer.append((int(block_idx), int(layer_idx)))
                continue
            lb = _extract_one_block_layer(
                entry, block_idx, layer_idx, graph_key, gelu_degree=gelu_deg,
            )
            out.per_block_layer[(int(block_idx), int(layer_idx))] = lb
            out.aggregate_total_bits += int(lb.total_bits)
            out.aggregate_fusion_count += int(lb.fusion_count)
            out.aggregate_valid_block_count += 1

    if out.missing_block_layer:

        raise BaselineHandoverError(
            f"static_skeletons archive {path} 缺少以下 graph_key："
            + ", ".join(
                static_skeletons_graph_key(
                    b, dataset, gelu_per_layer[l], softmax_per_layer[l],
                ) + f"@layer={l}"
                for b, l in out.missing_block_layer
            )
        )

    return out


def static_skeletons_baseline_to_action(
        baseline: StaticSkeletonsBaseline,
        *,
        base_max_sfs: Optional["MaxSFsTable"] = None,
        snap_sf_to_noise_table: bool = True,
        ) -> Tuple[np.ndarray, "MaxSFsTable", BaselineCostStats, Dict[str, Any]]:
    """把 ``StaticSkeletonsBaseline`` 转换成 RL 可直接消费的三元组：

      * ``action_vec``: ``np.ndarray[int]``，长度 ``sum(action_dims_for_config(num_layers))``。
                         所有 slot 都取 max idx；对于 baseline 里有 JSON SF 的 slot，
                         max_sf 被校准为 baseline SF —— 即 max-idx ↔ baseline。
      * ``max_sfs``:    校准过的 ``MaxSFsTable``。可直接喂给 ``BLBStage2Env``/``action_vector_to_cfgs``。
      * ``cost_stats``: ``BaselineCostStats``（total_bits / fusion / avg_k）。
      * ``diagnostics``: 报告每层每 block 的 active / inactive slot、unmapped 节点等。

    Args:
        baseline:                  ``load_static_skeletons_baseline`` 输出
        base_max_sfs:              基础 max_sfs（缺省时用 ``load_max_sfs(dataset)``）。
                                   未被 JSON 覆盖的 slot 保留 base_max_sfs 取值。
        snap_sf_to_noise_table:    True ⇒ 把每个 calibrated max_sf 钳到 noise table
                                   允许的最近合法值（保证 RL 动作落到 noise table 里）。
                                   False ⇒ 原样使用 JSON SF。
    """

    from blb_stage2_rl.action_space import (
        K_LEVELS, MaxSFsTable, NOISE_TABLE_ALLOWED_SCALING_FACTORS_BY_N,
        _BLOCK_NODE_NAME_BY_FIELD, _BLOCK_SPECS,
        _block_default_N, load_max_sfs, make_all_max_action_vector,
        per_layer_field_offsets,
    )

    L = int(baseline.num_layers)
    base = base_max_sfs if base_max_sfs is not None else load_max_sfs(baseline.dataset)

    calibrated = MaxSFsTable(
        by_block_node=dict(base.by_block_node),
        by_layer_block_node=dict(getattr(base, "by_layer_block_node", {}) or {}),
    )


    field_to_node: Dict[Tuple[int, str], str] = {}
    for b, dct in _BLOCK_NODE_NAME_BY_FIELD.items():
        for fname, node in dct.items():
            field_to_node[(int(b), str(fname))] = str(node)

    diagnostics: Dict[str, Any] = {
        "active_slot_count": 0,
        "fresh_slot_count": 0,
        "encode_slot_count": 0,
        "rescale_slot_count": 0,
        "inactive_rescale_slots": [],
        "unmapped_nodes": {"propagation": [], "rescale": []},
        "calibrated_max_sfs": {},
    }


    for (block_idx, layer_idx), lb in baseline.per_block_layer.items():

        for field_name, sf in lb.field_baseline_sfs.items():
            target_node = field_to_node.get((int(block_idx), str(field_name)))
            if target_node is None:


                target_node = str(field_name)
            calibrated_sf = int(sf)
            if snap_sf_to_noise_table:


                N = _block_default_N(
                    int(block_idx),
                    gelu_degree=baseline.gelu_per_layer[layer_idx],
                    attn_degree=baseline.softmax_per_layer[layer_idx],
                )
                allowed = list(NOISE_TABLE_ALLOWED_SCALING_FACTORS_BY_N.get(int(N), ()))
                if allowed and calibrated_sf not in allowed:
                    le = [v for v in allowed if v <= calibrated_sf]
                    calibrated_sf = max(le) if le else min(allowed)
            calibrated.by_layer_block_node[
                (int(layer_idx), int(block_idx), target_node)
            ] = int(calibrated_sf)
            diagnostics["calibrated_max_sfs"][
                f"L{layer_idx}.block{block_idx}.{field_name}"
            ] = int(calibrated_sf)
            diagnostics["active_slot_count"] += 1
            kind = lb.field_kind_in_ro.get(field_name, "")
            if kind == "fresh":
                diagnostics["fresh_slot_count"] += 1
            elif kind == "encode":
                diagnostics["encode_slot_count"] += 1
            elif kind == "rescale":
                diagnostics["rescale_slot_count"] += 1

        for nd in lb.unmapped_propagation_nodes:
            diagnostics["unmapped_nodes"]["propagation"].append(
                f"block{block_idx}.{nd}@L{layer_idx}"
            )
        for nd in lb.unmapped_rescale_nodes:
            diagnostics["unmapped_nodes"]["rescale"].append(
                f"block{block_idx}.{nd}@L{layer_idx}"
            )


    action_vec = make_all_max_action_vector(L)
    fields = per_layer_field_offsets()
    layer_dim = len(fields)
    active_rescale_slots = {
        (int(layer_idx), int(block_idx), str(field_name))
        for (block_idx, layer_idx), lb in baseline.per_block_layer.items()
        for field_name, kind in lb.field_kind_in_ro.items()
        if str(kind) == "rescale"
    }


    all_rescale_fields_per_block: Dict[int, List[str]] = {b: [] for b in (1, 2, 3, 4, 5)}
    for b, spec in _BLOCK_SPECS.items():
        for fname, kind, _max in spec.fields:
            if str(kind) == "R":
                all_rescale_fields_per_block[int(b)].append(fname)
    for (block_idx, layer_idx), lb in baseline.per_block_layer.items():
        for fname in all_rescale_fields_per_block.get(int(block_idx), []):
            if fname not in lb.field_baseline_sfs:
                diagnostics["inactive_rescale_slots"].append(
                    f"block{block_idx}.{fname}@L{layer_idx}"
                )
    for li in range(L):
        for field_offset, (block_idx, field_name, kind) in enumerate(fields):
            if str(kind) != "R":
                continue
            if (li, int(block_idx), str(field_name)) not in active_rescale_slots:
                action_vec[int(li * layer_dim + field_offset)] = 0


    from blb_stage2_rl.action_space import BASELINE_K_BY_BLOCK
    k_sum = 0.0
    k_count = 0
    for li in range(L):
        for b in (1, 2, 3, 4, 5):
            k_sum += float(BASELINE_K_BY_BLOCK.get(int(b), max(K_LEVELS)))
            k_count += 1
    baseline_avg_k = (k_sum / max(k_count, 1)) if k_count else float(max(K_LEVELS))
    cost_stats = BaselineCostStats(
        total_bits_sum=int(baseline.aggregate_total_bits),
        total_fusion_count=int(baseline.aggregate_fusion_count),
        avg_k=float(baseline_avg_k),
        typical_bits_drop=1.0,
        typical_fusion_count=1.0,
        typical_k_drop=1.0,
    )

    return action_vec, calibrated, cost_stats, diagnostics


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_calibrated_stage2_action_context(
        *,
        rescale_optimizer_root: str,
        dataset: str,
        num_layers: int,
        gelu_per_layer: Sequence[int],
        softmax_per_layer: Sequence[int],
        snap_sf_to_noise_table: bool = False,
        ) -> Stage2CalibratedActionContext:
    """Load the one static-skeleton-calibrated action context for Stage-2.

    Exact replay deliberately defaults to ``snap_sf_to_noise_table=False``,
    matching the production Stage-2 runner. Callers must pass this context's
    ``max_sfs`` through every action decode instead of reloading the generic
    profile-only table.
    """
    layer_count = int(num_layers)
    gelu = tuple(int(value) for value in gelu_per_layer)
    softmax = tuple(int(value) for value in softmax_per_layer)
    baseline = load_static_skeletons_baseline(
        rescale_optimizer_root=str(rescale_optimizer_root),
        dataset=str(dataset),
        num_layers=layer_count,
        gelu_per_layer=gelu,
        softmax_per_layer=softmax,
    )
    action_vec, max_sfs, cost_stats, diagnostics = (
        static_skeletons_baseline_to_action(
            baseline,
            snap_sf_to_noise_table=bool(snap_sf_to_noise_table),
        )
    )
    archive_path = os.path.abspath(str(baseline.archive_path))
    provenance = {
        "schema_version": "stage2_calibrated_action_context_v1",
        "dataset": str(dataset),
        "num_layers": layer_count,
        "gelu_per_layer": list(gelu),
        "softmax_per_layer": list(softmax),
        "rescale_optimizer_root": os.path.abspath(str(rescale_optimizer_root)),
        "archive_path": archive_path,
        "archive_sha256": _sha256_file(archive_path),
        "snap_sf_to_noise_table": bool(snap_sf_to_noise_table),
    }
    return Stage2CalibratedActionContext(
        baseline=baseline,
        baseline_action_vec=action_vec,
        max_sfs=max_sfs,
        cost_stats=cost_stats,
        diagnostics=diagnostics,
        provenance=provenance,
    )


def validate_calibrated_stage2_action_context(
        context: Stage2CalibratedActionContext,
        *,
        dataset: str,
        num_layers: int,
        gelu_per_layer: Sequence[int],
        softmax_per_layer: Sequence[int],
        snap_sf_to_noise_table: bool = False,
        ) -> None:
    """Reject a calibrated context that does not match the requested replay."""
    provenance = dict(context.provenance)
    expected = {
        "schema_version": "stage2_calibrated_action_context_v1",
        "dataset": str(dataset),
        "num_layers": int(num_layers),
        "gelu_per_layer": [int(value) for value in gelu_per_layer],
        "softmax_per_layer": [int(value) for value in softmax_per_layer],
        "snap_sf_to_noise_table": bool(snap_sf_to_noise_table),
    }
    mismatches = {
        key: {"expected": value, "actual": provenance.get(key)}
        for key, value in expected.items()
        if provenance.get(key) != value
    }
    archive_path = os.path.abspath(str(provenance.get("archive_path") or ""))
    if not archive_path or not os.path.isfile(archive_path):
        mismatches["archive_path"] = {
            "expected": "existing calibrated static-skeleton archive",
            "actual": archive_path,
        }
    else:
        current_sha256 = _sha256_file(archive_path)
        if current_sha256 != provenance.get("archive_sha256"):
            mismatches["archive_sha256"] = {
                "expected": provenance.get("archive_sha256"),
                "actual": current_sha256,
            }
    if mismatches:
        raise ValueError(
            "calibrated Stage-2 action context mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
