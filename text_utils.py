"""Shared text helpers for lightweight report and monitor scripts."""
from __future__ import annotations

from typing import Iterable


def iter_text_lines(text: object) -> Iterable[str]:
    """Yield lines from an in-memory string, preserving trailing ``\n``.

    This mirrors file-handle iteration for callers that already have command
    output in memory.  It intentionally splits only on ``\n`` so legacy parsers
    keep treating bare ``\r`` as ordinary text.
    """
    value = str(text or "")
    start = 0
    while start < len(value):
        end = value.find("\n", start)
        if end < 0:
            yield value[start:]
            return
        yield value[start:end + 1]
        start = end + 1
