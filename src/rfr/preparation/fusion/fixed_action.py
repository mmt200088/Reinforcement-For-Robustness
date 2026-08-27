"""Reconstruct per-step fusion ``(option, K)`` selections from a full action.

The flat action stores grid indices, while a selected fusion option may carry
above-grid precision values. Matching each block slice to the committed map
restores that option so strict evaluation installs the exact searched config.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np


def match_option_id(
        *,
        action_slice: Sequence[int],
        graph: Any,
        graph_key: str = "",
        ) -> int:
    """Return the unique fusion ``option_id`` whose ``action_indices`` equal
    ``action_slice`` on every non-K slot.

    ``graph`` is a :class:`rfr.preparation.fusion.count_map.BlockTypeFusionMap`
    (duck-typed: needs ``.k_slot_index`` and ``.options[*].{action_indices,option_id}``).
    The K slot is ignored because K is decided independently of the option.

    Raises ``ValueError`` on no match or ambiguity — it must NEVER silently pick a
    wrong option, since that would install the wrong (possibly un-boosted) config.
    """
    k_slot = int(graph.k_slot_index)
    arr = np.asarray(action_slice, dtype=int).reshape(-1)
    matches: List[int] = []
    for option in graph.options:
        opt_vec = np.asarray(option.action_indices, dtype=int).reshape(-1)
        if opt_vec.size != arr.size:
            continue
        same_non_k = all(
            int(opt_vec[i]) == int(arr[i])
            for i in range(arr.size)
            if i != k_slot
        )
        if same_non_k:
            matches.append(int(option.option_id))
    if not matches:
        raise ValueError(
            f"could not match fusion option for graph={graph_key!r} slice={arr.tolist()}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"ambiguous fusion option match for graph={graph_key!r}: {matches}"
        )
    return int(matches[0])


def reconstruct_fusion_group(
        action_vec: Sequence[int],
        *,
        fusion_map: Any,
        num_layers: int,
        profile: str,
        gelu: Sequence[int],
        softmax: Sequence[int],
        ) -> Dict[str, Any]:
    """Recover persisted fusion selections from the full action vector.

    Returns ``{"option_by_step", "choices_by_step", "summary"}``. ``option_by_step``
    feeds ``BLBActionFinalEvaluationModule._decode_fusion_count_fixed_action`` and
    the GLUE decode so both replay the boosted config. The K value per step is read
    straight from the flat vector (left exactly as the RL search encoded it)."""
    from rfr.search.common.action_space import K_LEVELS
    from rfr.search.common.layerwise_action import fusion_materialization_blocks

    del softmax
    action_arr = np.asarray(action_vec, dtype=int).reshape(-1)
    blocks = fusion_materialization_blocks(
        int(num_layers),
        profile=str(profile),
        gelu_degrees=[int(x) for x in gelu],
    )

    option_by_step: Dict[str, int] = {}
    choices: List[Dict[str, Any]] = []
    total_fusion = 0
    k_values: List[int] = []
    boosted_count = 0

    for block in blocks:
        graph_key = str(block.graph_key)
        graph = fusion_map.graphs.get(graph_key)
        if graph is None:
            if int(block.block_idx) == 1:
                continue
            raise KeyError(f"fusion map missing graph {graph_key!r}")
        action_slice = action_arr[list(block.full_vec_offsets)]
        option_id = match_option_id(
            action_slice=action_slice,
            graph=graph,
            graph_key=graph_key,
        )
        option = next(o for o in graph.options if int(o.option_id) == int(option_id))
        k_index = int(action_slice[int(graph.k_slot_index)])
        if not (0 <= k_index < len(K_LEVELS)):
            raise ValueError(
                f"block {block.artifact_index} graph={graph_key} has invalid K index {k_index}"
            )
        k_value = int(K_LEVELS[k_index])
        option_by_step[str(int(block.artifact_index))] = int(option_id)
        total_fusion += int(option.fusion_count)
        k_values.append(k_value)
        boosted_count += int(bool(getattr(option, "boosted", False)))
        choices.append({
            "step_idx": int(block.artifact_index),
            "layer": int(block.layer_idx),
            "block": int(block.block_idx),
            "graph_key": graph_key,
            "option_id": int(option_id),
            "fusion_count": int(option.fusion_count),
            "boosted": bool(getattr(option, "boosted", False)),
            "k_index": int(k_index),
            "k_value": int(k_value),
        })

    return {
        "option_by_step": option_by_step,
        "choices_by_step": choices,
        "summary": {
            "step_count": int(len(blocks)),
            "total_fusion_count": int(total_fusion),
            "boosted_option_count": int(boosted_count),
            "avg_k": float(sum(k_values) / len(k_values)) if k_values else 0.0,
            "k_values": k_values,
        },
    }


def build_fusion_fixed_config(
        action_vec: Sequence[int],
        *,
        profile: str,
        num_layers: int,
        gelu: Sequence[int],
        softmax: Sequence[int],
        fusion_map: Any = None,
        source: str = "",
        source_path: str = "",
        ) -> Dict[str, Any]:
    """Full ``fusion_count_fixed_action_v1`` payload reconstructed from a flat
    best-action vector. Carries ``gelu_degree`` / ``attn_degree`` so the JSON is
    directly consumable by the GLUE / final-eval paths without separate Stage-1
    args, and ``group.option_by_step`` so the boost is replayed."""
    if fusion_map is None:
        from .count_map import FusionCountMap
        fusion_map = FusionCountMap.load(str(profile))

    group = reconstruct_fusion_group(
        action_vec,
        fusion_map=fusion_map,
        num_layers=int(num_layers),
        profile=str(profile),
        gelu=gelu,
        softmax=softmax,
    )
    action_arr = np.asarray(action_vec, dtype=int).reshape(-1)
    return {
        "schema_version": "fusion_count_fixed_action_v1",
        "profile": str(profile),
        "num_layers": int(num_layers),
        "source": source or "reconstructed_from_flat_best_action",
        "source_path": str(source_path),
        "action_vec": [int(x) for x in action_arr.tolist()],


        "gelu_degree": [int(x) for x in gelu],
        "attn_degree": [int(x) for x in softmax],
        "group": {
            "option_by_step": group["option_by_step"],
            "choices_by_step": group["choices_by_step"],
            "inference_rule": (
                "match map option.action_indices against flat action_vec, "
                "ignoring each graph K slot"
            ),
        },
        "summary": group["summary"],
    }
