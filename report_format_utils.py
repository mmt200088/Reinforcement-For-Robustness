"""Shared helpers for compact metrics and progress output."""
from __future__ import annotations

import math
from typing import Any, Mapping


def format_float(
        value: Any,
        *,
        digits: int = 6,
        none_text: str = "",
        nan_text: str = "nan",
        ) -> str:
    """Format a numeric value for compact diagnostics.

    Non-numeric non-``None`` values are returned as strings so callers can pass
    already-rendered diagnostics without local try/except wrappers.
    """
    if value is None:
        return str(none_text)
    try:
        numeric = float(value)
    except Exception:
        return str(value)
    if math.isnan(numeric):
        return str(nan_text)
    return f"{numeric:.{int(digits)}f}"


def metric_float(mapping: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    """Read ``key`` from a metric mapping as float, returning ``default`` on failure."""
    try:
        return float(mapping.get(key, default))
    except Exception:
        return float(default)


def progress_bar(current: float, total: float, width: int = 30) -> str:
    """Render the compact unicode progress bar used in training logs."""
    ratio = min(float(current) / max(float(total), 1.0), 1.0)
    filled = int(round(ratio * int(width)))
    bar = "\u2588" * filled + "\u2591" * (int(width) - filled)
    return f"[{bar}] {ratio:6.1%}"


def format_elapsed(seconds: float) -> str:
    """Format elapsed seconds as ``XmYYs`` or ``XhYYmZZs``."""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"
