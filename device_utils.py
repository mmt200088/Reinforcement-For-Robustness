"""Shared device-list parsing helpers for Stage-1 and Stage-2 runners."""
from __future__ import annotations

from typing import Any, List


def parse_device_ids(spec: Any) -> List[int]:
    """Parse GPU id specs into a clean integer list.

    Accepts ``None``, an int, a list/tuple, comma-separated strings, and Fire's
    parenthesized tuple string form such as ``"(0, 1)"``.
    """
    if spec is None:
        return []
    if isinstance(spec, bool):
        raise ValueError(f"invalid device id {spec!r}; expected comma-separated ints")
    if isinstance(spec, int):
        tokens: List[Any] = [spec]
    elif isinstance(spec, (list, tuple)):
        tokens = list(spec)
    else:
        text = str(spec).strip()
        if not text:
            return []
        if (text.startswith("(") and text.endswith(")")) or (
            text.startswith("[") and text.endswith("]")
        ):
            text = text[1:-1].strip()
        tokens = [tok.strip() for tok in text.split(",") if tok.strip()]

    out: List[int] = []
    for tok in tokens:
        if isinstance(tok, bool):
            raise ValueError(f"invalid device id {tok!r} in spec {spec!r}; expected ints")
        try:
            out.append(int(tok))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid device id {tok!r} in spec {spec!r}; expected comma-separated ints"
            ) from exc
    return out
