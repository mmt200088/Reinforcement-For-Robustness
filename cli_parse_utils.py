"""Shared command-line parsing helpers for small project scripts."""
from __future__ import annotations

import json
from typing import Any, List, Sequence


def split_int_tokens(raw: str | None, *, allow_semicolon: bool = True) -> List[str]:
    """Split a comma-style integer list while preserving caller-specific errors."""
    if raw is None:
        return []
    text = str(raw)
    if allow_semicolon:
        text = text.replace(";", ",")
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_int_list_text(raw: str | None, *, allow_semicolon: bool = True) -> List[int]:
    return [int(item) for item in split_int_tokens(raw, allow_semicolon=allow_semicolon)]


def parse_optional_int_list(raw: str | None, *, allow_semicolon: bool = True) -> List[int] | None:
    if raw is None or str(raw).strip() == "":
        return None
    return parse_int_list_text(raw, allow_semicolon=allow_semicolon)


def parse_json_int_list(raw: str | None, *, default: Sequence[int], name: str) -> List[int]:
    text = str(raw or "").strip()
    if not text:
        return [int(v) for v in default]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{name} must be a JSON list: {exc}") from exc
    if not isinstance(payload, list):
        raise SystemExit(f"{name} must be a JSON list")
    return [int(v) for v in payload]


def parse_exact_json_int_list(raw: str, *, name: str, length: int) -> List[int]:
    payload = json.loads(raw)
    if not isinstance(payload, list) or len(payload) != int(length):
        raise ValueError(f"{name} must be a JSON list with {int(length)} entries")
    return [int(v) for v in payload]


def parse_broadcast_int_vector(
        raw: str | Sequence[int] | None,
        *,
        num_layers: int,
        default: int,
        name: str = "degree vector",
        ) -> List[int]:
    """Parse a one-value or per-layer integer vector.

    Accepts JSON-list strings, comma/semicolon strings, or an existing sequence.
    A single parsed value broadcasts to ``num_layers``.
    """
    layers = int(num_layers)
    if raw is None:
        return [int(default)] * layers
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return [int(default)] * layers
        values = json.loads(text) if text.startswith("[") else split_int_tokens(text)
    else:
        values = list(raw)
    out = [int(v) for v in values]
    if len(out) == 1:
        return out * layers
    if len(out) != layers:
        raise ValueError(f"{name} length {len(out)} must be 1 or num_layers={layers}")
    return out


def parse_degree_config(raw_value: Any) -> List[int] | None:
    """Parse the legacy RL degree-vector flag format.

    Accepts ``None``/empty as missing, Python sequences, JSON-list strings, and
    comma-separated strings.  This intentionally preserves the historical
    ``rl_tune*.py`` command-line behavior.
    """
    if raw_value is None or raw_value == "":
        return None
    if isinstance(raw_value, (list, tuple)):
        return [int(item) for item in raw_value]
    text = str(raw_value).strip()
    if not text:
        return None
    if text.startswith("["):
        return [int(item) for item in json.loads(text)]
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_noise_config(raw_value: Any) -> Any | None:
    """Parse a legacy JSON noise config flag."""
    if raw_value is None or raw_value == "":
        return None
    if isinstance(raw_value, dict):
        return raw_value
    text = str(raw_value).strip()
    if not text:
        return None
    return json.loads(text)


def parse_bool_flag(raw_value: Any, flag_name: str) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    if raw_value is None:
        return False
    text = str(raw_value).strip().lower()
    if text in ("1", "true", "t", "yes", "y", "on"):
        return True
    if text in ("0", "false", "f", "no", "n", "off", ""):
        return False
    raise ValueError(
        f"Invalid boolean value for {flag_name}: {raw_value!r}. "
        "Expected one of: true/false/1/0/yes/no."
    )


def parse_positive_int(raw_value: Any, flag_name: str) -> int:
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Invalid positive integer for {flag_name}: {raw_value!r}."
        ) from None
    if value <= 0:
        raise ValueError(
            f"Invalid positive integer for {flag_name}: {raw_value!r}."
        )
    return value


def parse_optional_positive_int(raw_value: Any, flag_name: str) -> int | None:
    if raw_value is None or raw_value == "":
        return None
    return parse_positive_int(raw_value, flag_name)


def parse_stage1_episode_limit(raw_value: Any, flag_name: str) -> int:
    """Parse Stage-1 episode budget; 0/-1 means unbounded until entropy stop."""
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Invalid integer for {flag_name}: {raw_value!r}."
        ) from None


def parse_optional_positive_float(raw_value: Any, flag_name: str) -> float | None:
    if raw_value in (None, ""):
        return None
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Invalid positive float for {flag_name}: {raw_value!r}."
        ) from None
    if value <= 0:
        raise ValueError(
            f"Invalid positive float for {flag_name}: {raw_value!r}."
        )
    return value
