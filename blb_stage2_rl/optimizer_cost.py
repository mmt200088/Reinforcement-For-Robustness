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
        ) -> ActionCostEvaluation:
    """Evaluate every action through the same cfg-derived optimizer path.

    This is the canonical convention for candidate ranking, reward baseline,
    F0 scans, and RL env cost comparison.  Even the all-max baseline uses
    ``action_vector_to_cfgs -> build_optimizer_requests -> evaluate_blocks``.
    Optimizer-native empty-payload baselines remain diagnostic-only because
    they may use a different Rescale_optimizer convention.
    """
    action_arr = np.asarray(action_vec, dtype=int).reshape(-1)
    decoded = action_vector_to_cfgs(
        action_arr,
        max_sfs,
        num_layers=int(num_layers),
        gelu_degree=gelu_degree,
        attn_degree=attn_degree,
    )
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
