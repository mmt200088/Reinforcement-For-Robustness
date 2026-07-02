"""Shared command-line parsing helpers for small project scripts."""
from __future__ import annotations

import json
from typing import List, Sequence


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
