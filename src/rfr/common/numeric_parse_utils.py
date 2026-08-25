"""Small numeric parsing helpers for report scripts."""
from __future__ import annotations

import re

FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")


def parse_first_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = FLOAT_RE.search(str(value))
    if not match:
        return None
    return float(match.group(0))
