"""Torch-free truncation-K domain shared by Stage-2 action codecs."""

from __future__ import annotations

import os
from typing import Mapping, Sequence, Tuple

from cli_parse_utils import parse_int_list_text


_ENV_NAME = "BLB_TRUNCATION_K_LEVELS"

DEFAULT_K_LEVELS_LEGACY_COMPAT: Tuple[int, ...] = (
    8,
    9,
    11,
    13,
    10,
    12,
    6,
    7,
)
K_MIN_BITS = 6
K_MAX_BITS = 13
SUPPORTED_K_VALUES = frozenset(range(K_MIN_BITS, K_MAX_BITS + 1))
CHECKPOINT_K_DOMAIN_KEY = "truncation_k_domain"
CHECKPOINT_K_DOMAIN_SCHEMA_VERSION = "stage2_truncation_k_domain_v1"


def load_k_levels(environ: Mapping[str, str] | None = None) -> Tuple[int, ...]:
    """Load the ordered K table, preserving the default when unset."""
    source = os.environ if environ is None else environ
    if _ENV_NAME not in source:
        return DEFAULT_K_LEVELS_LEGACY_COMPAT

    raw = str(source.get(_ENV_NAME, "") or "").strip()
    if not raw:
        return DEFAULT_K_LEVELS_LEGACY_COMPAT
    tokens = raw.replace(";", ",").split(",")
    if any(not token.strip() for token in tokens):
        raise ValueError(
            f"{_ENV_NAME} must be a non-empty ordered list of integers"
        )
    try:
        values = tuple(parse_int_list_text(raw))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{_ENV_NAME} must contain only integers") from exc
    if len(set(values)) != len(values):
        raise ValueError(f"{_ENV_NAME} contains duplicate values: {values}")
    return values


def validate_exact_k_domain(levels: Sequence[int]) -> Tuple[int, ...]:
    """Require every supported K value exactly once, in any order."""
    try:
        values = tuple(int(value) for value in levels)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "K_LEVELS must contain each supported K value exactly once"
        ) from exc
    if len(values) != len(SUPPORTED_K_VALUES) or frozenset(values) != SUPPORTED_K_VALUES:
        raise ValueError(
            "K_LEVELS must contain each supported K value exactly once: "
            f"{sorted(SUPPORTED_K_VALUES)}, got {values}"
        )
    return values


def baseline_k_index(
    levels: Sequence[int],
    baseline_k: int = 13,
) -> int:
    """Return the baseline index, falling back to the largest available K."""
    values = tuple(int(value) for value in levels)
    if not values:
        raise ValueError("K levels must contain at least one value")
    target = int(baseline_k)
    if target not in values:
        target = max(values)
    return values.index(target)


K_LEVELS: Tuple[int, ...] = load_k_levels()
LEVELS_K = len(K_LEVELS)


def checkpoint_k_domain_contract() -> dict:
    """Return the ordered truncation-K identity persisted in new checkpoints."""
    levels = validate_exact_k_domain(K_LEVELS)
    return {
        "schema_version": CHECKPOINT_K_DOMAIN_SCHEMA_VERSION,
        "k_levels": [int(value) for value in levels],
    }


def validate_checkpoint_k_domain(
        checkpoint: Mapping[str, object],
        *,
        context: str = "Stage-2 checkpoint",
        ) -> Tuple[int, ...]:
    """Reject checkpoints whose ordered K domain is missing or incompatible."""
    prefix = str(context or "Stage-2 checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError(
            f"{prefix} is not a mapping and has no K-domain contract; "
            "a fresh run is required"
        )

    raw_contract = checkpoint.get(CHECKPOINT_K_DOMAIN_KEY)
    if not isinstance(raw_contract, Mapping):
        raise RuntimeError(
            f"{prefix} is missing {CHECKPOINT_K_DOMAIN_KEY!r}; "
            "a fresh run is required"
        )
    schema_version = str(raw_contract.get("schema_version", "") or "")
    if schema_version != CHECKPOINT_K_DOMAIN_SCHEMA_VERSION:
        raise RuntimeError(
            f"{prefix} has unsupported K-domain schema {schema_version!r}; "
            f"expected {CHECKPOINT_K_DOMAIN_SCHEMA_VERSION!r}; "
            "a fresh run is required"
        )

    raw_levels = raw_contract.get("k_levels")
    if isinstance(raw_levels, (str, bytes)) or not isinstance(raw_levels, Sequence):
        raise RuntimeError(
            f"{prefix} has an invalid ordered K domain; a fresh run is required"
        )
    try:
        saved_levels = tuple(int(value) for value in raw_levels)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{prefix} has an invalid ordered K domain; a fresh run is required"
        ) from exc

    expected_levels = validate_exact_k_domain(K_LEVELS)
    if saved_levels != expected_levels:
        raise RuntimeError(
            f"{prefix} ordered K domain {saved_levels} does not match "
            f"{expected_levels}; a fresh run is required"
        )
    return saved_levels
