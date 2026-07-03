"""Reconstruct the per-step fusion ``(option, K)`` selection from a flat Stage-2
best-action vector + the committed fusion-count map.

WHY THIS EXISTS (the boost-handoff fix, 2026-06-25)
---------------------------------------------------
Stage-2 fusion-count RL decides ``(fusion_option, K)`` per block. The training
terminal probe installs the EXACT chosen config — including the precision boost
("加大精度") whose above-grid SFs live in ``FusionOption.explicit_field_values`` —
via the env's ``_boosted_overrides`` (SF-direct cfg rebuild).

But the persisted ``best_action_vec`` is the legacy per-slot *grid index* vector.
A boosted option's ``action_indices`` are the in-grid base SFs; the boost is NOT
representable as indices. So the flat vector alone loses the boost, and the
standalone consumers (validation-set final eval, GLUE submission) that decode it
directly install PRE-boost (noisier) noise — a config the RL search never
selected.

This module recovers ``group.option_by_step`` by matching each block slice of the
flat vector against the map's options (ignoring the K slot, which is decided
independently). With that, the consumers replay the same boosted option the RL
search chose (``_decode_fusion_count_fixed_action`` then uses the option's
``explicit_field_values``). Match is unambiguous because the map keeps exactly one
option per realized fusion_count (boost is applied in place, not as a sibling).

TORCH NOTES
-----------
:func:`match_option_id` is pure and torch-free (operates on a
``BlockTypeFusionMap`` + a numpy slice), so it is unit-tested on a torch-free box.
:func:`reconstruct_fusion_group` / :func:`build_fusion_fixed_config` call
``action_space.step_schedule`` which pulls torch, so they run in a torch context
(the runner / GLUE / final-eval, all of which already import torch).
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


def match_option_id(
        *,
        action_slice: Sequence[int],
        graph: Any,
        graph_key: str = "",
        slot_dims: Sequence[int] | None = None,
        ) -> int:
    """Return the unique fusion ``option_id`` whose ``action_indices`` equal
    ``action_slice`` on every non-K slot.

    ``graph`` is a :class:`blb_stage2_rl.fusion_count_map.BlockTypeFusionMap`
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
        if slot_dims is not None:
            dims = np.asarray(slot_dims, dtype=int).reshape(-1)
            if dims.size == arr.size:
                legacy_all_max = all(
                    int(arr[i]) == int(dims[i]) - 1
                    for i in range(arr.size)
                    if i != k_slot
                )
                if legacy_all_max:
                    baseline = [
                        int(option.option_id)
                        for option in graph.options
                        if int(option.option_id) == 0
                    ]
                    if len(baseline) == 1:
                        return 0
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
    """Walk the Stage-2 step schedule and recover the per-step fusion selection.

    Returns ``{"option_by_step", "choices_by_step", "summary"}``. ``option_by_step``
    feeds ``BLBActionFinalEvaluationModule._decode_fusion_count_fixed_action`` and
    the GLUE decode so both replay the boosted config. The K value per step is read
    straight from the flat vector (left exactly as the RL search encoded it)."""
    try:  # torch-free test lane (blb_stage2_rl on sys.path)
        from action_space import K_LEVELS, step_schedule
    except ImportError:  # package context
        from .action_space import K_LEVELS, step_schedule

    action_arr = np.asarray(action_vec, dtype=int).reshape(-1)
    schedule = step_schedule(
        int(num_layers),
        profile=str(profile),
        attn_degree_per_layer=[int(x) for x in softmax],
        gelu_degree_per_layer=[int(x) for x in gelu],
    )

    option_by_step: Dict[str, int] = {}
    choices: List[Dict[str, Any]] = []
    total_fusion = 0
    k_values: List[int] = []
    boosted_count = 0

    for step in schedule:
        graph_key = str(step.graph_key_suffix)
        graph = fusion_map.graphs.get(graph_key)
        if graph is None:
            raise KeyError(f"fusion map missing graph {graph_key!r}")
        action_slice = action_arr[list(step.full_vec_offsets)]
        option_id = match_option_id(
            action_slice=action_slice,
            graph=graph,
            graph_key=graph_key,
            slot_dims=getattr(step, "slot_dims", None),
        )
        option = next(o for o in graph.options if int(o.option_id) == int(option_id))
        k_index = int(action_slice[int(graph.k_slot_index)])
        if not (0 <= k_index < len(K_LEVELS)):
            raise ValueError(
                f"step {step.step_idx} graph={graph_key} has invalid K index {k_index}"
            )
        k_value = int(K_LEVELS[k_index])
        option_by_step[str(int(step.step_idx))] = int(option_id)
        total_fusion += int(option.fusion_count)
        k_values.append(k_value)
        boosted_count += int(bool(getattr(option, "boosted", False)))
        choices.append({
            "step_idx": int(step.step_idx),
            "layer": int(step.layer_idx),
            "block": int(step.block_idx),
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
            "step_count": int(len(schedule)),
            "total_fusion_count": int(total_fusion),
            "boosted_option_count": int(boosted_count),
            "avg_k": float(sum(k_values) / len(k_values)) if k_values else 0.0,
            "k_values": k_values,
        },
    }


def build_boosted_overrides_from_group(
        action_vec: Sequence[int],
        *,
        group: Mapping[str, Any],
        fusion_map: Any,
        num_layers: int,
        profile: str,
        gelu: Sequence[int],
        softmax: Sequence[int],
        ) -> Dict[Tuple[int, int], Dict[str, int]]:
    """Return terminal-probe SF-direct overrides for boosted fusion options.

    The legacy full action vector carries map ``action_indices`` and the selected
    K index, but it cannot carry above-baseline precision-boost SFs. This helper
    is the shared handoff for any path that wants to replay a fusion-count action
    through :func:`optimizer_cost.evaluate_action_for_cost`: it recovers the
    chosen option per step from ``group``, inserts the K value selected in
    ``action_vec`` into that option's explicit field values, and returns the
    ``{(block, layer): field_values}`` override map consumed by the canonical cost
    / replan / model-install path.
    """
    if not isinstance(group, Mapping):
        raise ValueError("build_boosted_overrides_from_group requires group metadata")
    raw_option_by_graph = group.get("option_by_graph")
    raw_option_by_step = group.get("option_by_step")
    if not isinstance(raw_option_by_graph, Mapping) and not isinstance(raw_option_by_step, Mapping):
        raise ValueError("fusion group requires option_by_step or option_by_graph")

    try:  # torch-free test lane (blb_stage2_rl on sys.path)
        from action_space import K_LEVELS, step_schedule
    except ImportError:  # package context
        from .action_space import K_LEVELS, step_schedule

    action_arr = np.asarray(action_vec, dtype=int).reshape(-1)
    gelu_arr = np.asarray(gelu, dtype=int).reshape(-1)
    softmax_arr = np.asarray(softmax, dtype=int).reshape(-1)
    option_by_graph = {
        str(k): int(v)
        for k, v in dict(raw_option_by_graph or {}).items()
    }
    option_by_step = {
        str(k): int(v)
        for k, v in dict(raw_option_by_step or {}).items()
    }
    schedule = step_schedule(
        int(num_layers),
        profile=str(profile),
        attn_degree_per_layer=softmax_arr.tolist(),
        gelu_degree_per_layer=gelu_arr.tolist(),
    )

    overrides: Dict[Tuple[int, int], Dict[str, int]] = {}
    for step in schedule:
        graph_key = str(step.graph_key_suffix)
        step_key = str(int(step.step_idx))
        if step_key in option_by_step:
            option_id = int(option_by_step[step_key])
        elif graph_key in option_by_graph:
            option_id = int(option_by_graph[graph_key])
        else:
            continue
        graph = fusion_map.graphs.get(graph_key)
        if graph is None:
            raise KeyError(f"fusion map missing graph {graph_key!r}")
        option = None
        for candidate in graph.options:
            if int(candidate.option_id) == option_id:
                option = candidate
                break
        if option is None:
            raise KeyError(f"fusion map graph {graph_key!r} has no option {option_id}")
        if not (bool(getattr(option, "boosted", False)) and option.explicit_field_values):
            continue

        action_slice = action_arr[list(step.full_vec_offsets)]
        k_slot = int(graph.k_slot_index)
        if not (0 <= k_slot < action_slice.size):
            raise ValueError(f"graph {graph_key!r} K slot {k_slot} out of action slice")
        k_index = int(action_slice[k_slot])
        if not (0 <= k_index < len(K_LEVELS)):
            raise ValueError(f"graph {graph_key!r} has invalid K index {k_index}")
        if not (0 <= k_slot < len(step.slot_field_names)):
            raise ValueError(f"graph {graph_key!r} K slot {k_slot} has no field name")

        field_values = {
            str(k): int(v)
            for k, v in dict(option.explicit_field_values).items()
        }
        field_values[str(step.slot_field_names[k_slot])] = int(K_LEVELS[k_index])
        overrides[(int(step.block_idx), int(step.layer_idx))] = field_values
    return overrides


def select_fusion_eval_metadata(
        *,
        action_vec: Sequence[int],
        base_action: Sequence[int] | None,
        existing_metadata: Any,
        fusion_group: Any,
        fusion_count_action: bool,
        profile: str,
        num_layers: int,
        gelu: Sequence[int],
        softmax: Sequence[int],
        fusion_map: Any = None,
        ) -> Dict[str, Any]:
    """Decide the decode metadata for one final-eval candidate so the trained best
    replays its boosted fusion config.

    Rules (in order):

    * an explicit ``fusion_count_fixed_action_v1`` metadata wins (user-supplied
      config) — returned unchanged;
    * a per-slot run (not fusion, no group) is returned unchanged;
    * ONLY the trained best (``action_vec == base_action``) gets boost-replay; any
      other candidate (cost-matched random / range-mutated) is an arbitrary vector
      that is NOT a map option, so it keeps the default index decode;
    * for the best, attach the persisted ``fusion_group`` if present, else
      reconstruct it from the vector + the committed map.
    """
    md = dict(existing_metadata or {})
    if str(md.get("schema_version", "")) == "fusion_count_fixed_action_v1":
        return md
    if not bool(fusion_count_action) and fusion_group is None:
        return md
    if base_action is None:
        return md
    a = np.asarray(action_vec, dtype=int).reshape(-1)
    b = np.asarray(base_action, dtype=int).reshape(-1)
    if a.size != b.size or not np.array_equal(a, b):
        return md
    group = fusion_group
    if group is None:
        cfg = build_fusion_fixed_config(
            a, profile=str(profile), num_layers=int(num_layers),
            gelu=gelu, softmax=softmax, fusion_map=fusion_map,
        )
        group = cfg["group"]
    md["schema_version"] = "fusion_count_fixed_action_v1"
    md["group"] = group
    return md


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
        try:  # torch-free test lane
            from fusion_count_map import FusionCountMap
        except ImportError:  # package context
            from .fusion_count_map import FusionCountMap
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
        # Stage-1 ladder travels with the action so a single JSON fully specifies
        # the deployed config (boost replayed via group, degrees via these).
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
