"""Torch-free identity contract for Stage-2 actor/critic ablations."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping


DEFAULT_POLICY_NETWORK_VARIANT = "shared_gtrxl_v1"
FRESH_POLICY_NETWORK_VARIANT = "shared_gtrxl_small_v1"
LEGACY_SHARED_RL_VARIANT = "blb_v3_layerwise_robust_gtrxl_v1"


@dataclass(frozen=True)
class PolicyNetworkVariantSpec:
    name: str
    rl_variant: str
    critic_kind: str
    shares_actor_trunk: bool
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    description: str

    @property
    def architecture(self) -> Dict[str, int]:
        return {
            "d_model": int(self.d_model),
            "n_heads": int(self.n_heads),
            "n_layers": int(self.n_layers),
            "d_ff": int(self.d_ff),
        }

    def contract_payload(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["variant"] = payload.pop("name")
        for key in ("d_model", "n_heads", "n_layers", "d_ff"):
            payload.pop(key)
        return payload


_VARIANTS = {
    FRESH_POLICY_NETWORK_VARIANT: PolicyNetworkVariantSpec(
        name=FRESH_POLICY_NETWORK_VARIANT,
        rl_variant="blb_v3_layerwise_robust_shared_gtrxl_small_v1",
        critic_kind="shared_gtrxl",
        shares_actor_trunk=True,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        description="Small shared GTrXL actor-critic for fresh Stage-2 runs",
    ),
    DEFAULT_POLICY_NETWORK_VARIANT: PolicyNetworkVariantSpec(
        name=DEFAULT_POLICY_NETWORK_VARIANT,
        rl_variant=LEGACY_SHARED_RL_VARIANT,
        critic_kind="shared_gtrxl",
        shares_actor_trunk=True,
        d_model=256,
        n_heads=8,
        n_layers=4,
        d_ff=512,
        description="Original v10 shared GTrXL actor-critic baseline",
    ),
    "separate_critic_gtrxl_v1": PolicyNetworkVariantSpec(
        name="separate_critic_gtrxl_v1",
        rl_variant="blb_v3_layerwise_robust_separate_critic_gtrxl_v1",
        critic_kind="gtrxl",
        shares_actor_trunk=False,
        d_model=256,
        n_heads=8,
        n_layers=4,
        d_ff=512,
        description="Original actor with an independent isomorphic GTrXL critic",
    ),
    "separate_critic_mlp_v1": PolicyNetworkVariantSpec(
        name="separate_critic_mlp_v1",
        rl_variant="blb_v3_layerwise_robust_separate_critic_mlp_v1",
        critic_kind="mlp",
        shares_actor_trunk=False,
        d_model=256,
        n_heads=8,
        n_layers=4,
        d_ff=512,
        description="Original actor with an independent 512-512-256 value MLP",
    ),
}
SUPPORTED_POLICY_NETWORK_VARIANTS = tuple(_VARIANTS)

_ALIASES = {
    "small": FRESH_POLICY_NETWORK_VARIANT,
    "shared_small": FRESH_POLICY_NETWORK_VARIANT,
    "small_shared_gtrxl": FRESH_POLICY_NETWORK_VARIANT,
    "shared": DEFAULT_POLICY_NETWORK_VARIANT,
    "shared_gtrxl": DEFAULT_POLICY_NETWORK_VARIANT,
    "large": DEFAULT_POLICY_NETWORK_VARIANT,
    "shared_large": DEFAULT_POLICY_NETWORK_VARIANT,
    "separate_gtrxl": "separate_critic_gtrxl_v1",
    "separate_critic_gtrxl": "separate_critic_gtrxl_v1",
    "separate_mlp": "separate_critic_mlp_v1",
    "separate_critic_mlp": "separate_critic_mlp_v1",
}


def normalize_policy_network_variant(value: Any) -> str:
    normalized = str(value or DEFAULT_POLICY_NETWORK_VARIANT).strip().lower()
    normalized = normalized.replace("-", "_")
    normalized = _ALIASES.get(normalized, normalized)
    if normalized not in _VARIANTS:
        supported = ", ".join(SUPPORTED_POLICY_NETWORK_VARIANTS)
        raise ValueError(
            f"unsupported Stage-2 policy network variant {value!r}; "
            f"expected one of: {supported}"
        )
    return normalized


def policy_network_variant_spec(value: Any) -> PolicyNetworkVariantSpec:
    return _VARIANTS[normalize_policy_network_variant(value)]


def policy_network_architecture(value: Any) -> Dict[str, int]:
    return policy_network_variant_spec(value).architecture


def bind_policy_network_contract(
        base_contract: Mapping[str, Any],
        variant: Any,
        *,
        policy_shape: Mapping[str, Any],
        ) -> Dict[str, Any]:
    """Return the algorithm contract for one network arm.

    The original shared arm deliberately remains byte-for-byte compatible with
    the pre-ablation v10 contract so its existing checkpoints can resume.
    """
    bound = deepcopy(dict(base_contract))
    spec = policy_network_variant_spec(variant)
    if spec.name == DEFAULT_POLICY_NETWORK_VARIANT:
        if bound.get("rl_variant") != LEGACY_SHARED_RL_VARIANT:
            raise ValueError(
                "shared_gtrxl_v1 requires the legacy shared rl_variant"
            )
        return bound
    if spec.name == FRESH_POLICY_NETWORK_VARIANT:
        mismatches = {
            key: (policy_shape.get(key), expected)
            for key, expected in spec.architecture.items()
            if int(policy_shape.get(key, -1)) != int(expected)
        }
        if mismatches:
            raise ValueError(
                "shared_gtrxl_small_v1 policy shape does not match its "
                f"registered architecture: {mismatches}"
            )
    network_contract = spec.contract_payload()
    if spec.name == FRESH_POLICY_NETWORK_VARIANT:
        network_contract["architecture"] = spec.architecture
    bound["rl_variant"] = spec.rl_variant
    bound["policy_network"] = {
        **network_contract,
        "policy_shape": deepcopy(dict(policy_shape)),
    }
    return bound


def policy_network_variant_from_checkpoint(checkpoint: Mapping[str, Any]) -> str:
    explicit = checkpoint.get("policy_network_variant")
    if explicit not in (None, ""):
        return normalize_policy_network_variant(explicit)
    if checkpoint.get("rl_variant") == LEGACY_SHARED_RL_VARIANT:
        return DEFAULT_POLICY_NETWORK_VARIANT
    raise RuntimeError(
        "layerwise checkpoint policy network variant is missing and cannot be inferred"
    )


def validate_checkpoint_policy_network_variant(
        checkpoint: Mapping[str, Any],
        expected_variant: Any,
        ) -> None:
    expected = normalize_policy_network_variant(expected_variant)
    actual = policy_network_variant_from_checkpoint(checkpoint)
    if actual != expected:
        raise RuntimeError(
            "layerwise checkpoint policy network variant "
            f"{actual!r} != requested {expected!r}; start a distinct ablation run "
            "or select the matching network variant"
        )
