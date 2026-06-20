"""Canonical BLB action cost evaluation helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from rescale_optimizer_bridge import aggregate_optimizer_signals

from .action_space import (
    ActionDecodeResult,
    MaxSFsTable,
    action_vector_to_cfgs,
    build_optimizer_requests,
)


@dataclass
class ActionCostEvaluation:
    action_indices: Sequence[int]
    decoded: ActionDecodeResult
    cfgs_dict: Dict[str, Mapping[int, object]]
    requests: Dict[str, Tuple[str, object]]
    outputs: Mapping[str, Any]
    signals: Any
    optimizer_eval_mode: str = "evaluate_blocks_cfg_path"


def evaluate_action_for_cost(
        action_vec: Sequence[int],
        *,
        profile: str,
        num_layers: int,
        max_sfs: MaxSFsTable,
        rescale_bridge: Any,
        gelu_degree: Any = 4,
        attn_degree: Any = 4,
        boosted_overrides: "Mapping[Tuple[int, int], Mapping[str, int]] | None" = None,
        ) -> ActionCostEvaluation:
    """Evaluate every action through the same cfg-derived optimizer path.

    This is the canonical convention for candidate ranking, reward baseline,
    F0 scans, and RL env cost comparison.  Even the all-max baseline uses
    ``action_vector_to_cfgs -> build_optimizer_requests -> evaluate_blocks``.
    Optimizer-native empty-payload baselines remain diagnostic-only because
    they may use a different Rescale_optimizer convention.

    ``boosted_overrides`` (加大精度): ``{(block_idx, layer_idx): {field: sf}}`` of
    explicit boosted SFs (above-baseline, no action index). After the grid decode,
    the listed (block, layer) cfgs are rebuilt SF-direct so the ENTIRE downstream
    path — replan cost signals, optimizer override, AND the model noise install —
    uses the boosted action group rather than the index-decoded pre-boost one.
    """
    action_arr = np.asarray(action_vec, dtype=int).reshape(-1)
    decoded = action_vector_to_cfgs(
        action_arr,
        max_sfs,
        num_layers=int(num_layers),
        gelu_degree=gelu_degree,
        attn_degree=attn_degree,
    )
    if boosted_overrides:
        from .action_space import (
            _block_default_N,
            _degree_for_layer,
            build_block_cfg_from_field_values,
        )
        for (block_idx, layer_idx), field_values in boosted_overrides.items():
            bi, li = int(block_idx), int(layer_idx)
            deg_g = _degree_for_layer(gelu_degree, li, int(num_layers), default=4, name="gelu_degree")
            deg_a = _degree_for_layer(attn_degree, li, int(num_layers), default=4, name="attn_degree")
            cfg = build_block_cfg_from_field_values(
                bi, li, dict(field_values),
                N=int(_block_default_N(bi, gelu_degree=deg_g, attn_degree=deg_a)),
                gelu_degree=deg_g, attn_degree=deg_a,
            )
            getattr(decoded, f"block{bi}_cfgs")[li] = cfg
    cfgs_dict = decoded.cfgs_dict()
    requests = build_optimizer_requests(profile, cfgs_dict)

    outputs = rescale_bridge.evaluate_blocks(requests)

    return ActionCostEvaluation(
        action_indices=[int(x) for x in action_arr.tolist()],
        decoded=decoded,
        cfgs_dict=cfgs_dict,
        requests=requests,
        outputs=outputs,
        signals=aggregate_optimizer_signals(outputs),
    )
