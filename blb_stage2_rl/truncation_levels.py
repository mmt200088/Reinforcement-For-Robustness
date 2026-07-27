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
