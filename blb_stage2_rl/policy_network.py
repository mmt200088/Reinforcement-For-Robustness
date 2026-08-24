"""Fixed identity contract for the production Stage-2 policy network."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


POLICY_NETWORK_ID = "shared_gtrxl_small_v1"
POLICY_RL_VARIANT = "blb_v3_layerwise_robust_shared_gtrxl_small_v1"
POLICY_DESCRIPTION = "Small shared GTrXL actor-critic for fresh Stage-2 runs"
POLICY_ARCHITECTURE = {
    "d_model": 128,
    "n_heads": 4,
    "n_layers": 2,
    "d_ff": 256,
}


def policy_network_architecture() -> dict[str, int]:
    return dict(POLICY_ARCHITECTURE)


def bind_policy_network_contract(
    base_contract: Mapping[str, Any],
    *,
    policy_shape: Mapping[str, Any],
) -> dict[str, Any]:
    mismatches = {
        key: (policy_shape.get(key), expected)
        for key, expected in POLICY_ARCHITECTURE.items()
        if int(policy_shape.get(key, -1)) != expected
    }
    if mismatches:
        raise ValueError(
            "production policy shape does not match its architecture: "
            f"{mismatches}"
        )
    bound = deepcopy(dict(base_contract))
    bound["rl_variant"] = POLICY_RL_VARIANT
    bound["policy_network"] = {
        "rl_variant": POLICY_RL_VARIANT,
        "critic_kind": "shared_gtrxl",
        "shares_actor_trunk": True,
        "description": POLICY_DESCRIPTION,
        "variant": POLICY_NETWORK_ID,
        "architecture": dict(POLICY_ARCHITECTURE),
        "policy_shape": deepcopy(dict(policy_shape)),
    }
    return bound


def validate_checkpoint_policy_network(checkpoint: Mapping[str, Any]) -> None:
    actual = checkpoint.get("policy_network_variant")
    if actual != POLICY_NETWORK_ID:
        raise RuntimeError(
            "layerwise checkpoint policy network "
            f"{actual!r} != {POLICY_NETWORK_ID!r}; start a fresh run"
        )


__all__ = [
    "POLICY_ARCHITECTURE",
    "POLICY_NETWORK_ID",
    "POLICY_RL_VARIANT",
    "bind_policy_network_contract",
    "policy_network_architecture",
    "validate_checkpoint_policy_network",
]
