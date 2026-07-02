"""Shared JSONL readers for report and diagnostics scripts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Literal

JsonlErrorMode = Literal["skip", "raise"]


def iter_jsonl(
        path: str | Path,
        *,
        errors: JsonlErrorMode = "skip",
        dict_only: bool = True,
        ) -> Iterable[Any]:
    """Yield parsed JSONL rows.

    Blank lines are ignored.  ``errors="skip"`` preserves legacy monitor
    behavior for partially-written logs; ``errors="raise"`` preserves verifier
    behavior that reports the file and line number on malformed JSON.
    """
    jsonl_path = Path(path)
    with jsonl_path.open(encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line or line.isspace():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                if errors == "raise":
                    raise ValueError(f"{jsonl_path}:{line_no}: invalid JSON") from exc
                if errors == "skip":
                    continue
                raise ValueError(f"unsupported JSONL error mode: {errors!r}") from exc
            if dict_only and not isinstance(payload, dict):
                continue
            yield payload


def read_jsonl(
        path: str | Path,
        *,
        errors: JsonlErrorMode = "skip",
        dict_only: bool = True,
        missing_ok: bool = False,
        ) -> list[Any]:
    jsonl_path = Path(path)
    if missing_ok and not jsonl_path.exists():
        return []
    return list(iter_jsonl(jsonl_path, errors=errors, dict_only=dict_only))
