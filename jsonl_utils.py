"""Shared JSONL readers for report and diagnostics scripts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Literal

JsonlErrorMode = Literal["skip", "raise"]


def iter_jsonl_records(
        path: str | Path,
        *,
        errors: JsonlErrorMode = "skip",
        dict_only: bool = True,
        ) -> Iterable[tuple[int, Any]]:
    """Yield ``(line_no, parsed_payload)`` JSONL records.

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
            yield line_no, payload


def iter_jsonl(
        path: str | Path,
        *,
        errors: JsonlErrorMode = "skip",
        dict_only: bool = True,
        ) -> Iterable[Any]:
    """Yield parsed JSONL rows."""
    for _line_no, payload in iter_jsonl_records(path, errors=errors, dict_only=dict_only):
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


def missing_required_fields(payload: dict[str, Any], required_fields: tuple[str, ...]) -> tuple[str, ...] | None:
    missing = tuple(field for field in required_fields if field not in payload)
    return missing or None


def count_jsonl_with_required_fields(
        path: str | Path,
        required_fields: tuple[str, ...],
        *,
        label: str | None = None,
        ) -> tuple[int, list[str]]:
    row_count = 0
    failures: list[str] = []
    missing_examples: list[str] = []
    missing_row_count = 0
    row_label = label or Path(path).name
    for line_no, payload in iter_jsonl_records(path, errors="raise", dict_only=False):
        row_count += 1
        if not isinstance(payload, dict):
            failures.append(f"{row_label}:{line_no} is not a JSON object")
            continue
        missing = missing_required_fields(payload, required_fields)
        if missing:
            missing_row_count += 1
            if len(missing_examples) < 3:
                missing_examples.append(f"line {line_no}: {', '.join(missing)}")
    if missing_row_count:
        failures.append(
            f"{row_label} missing required fields in {missing_row_count} rows "
            f"({'; '.join(missing_examples)})"
        )
    return row_count, failures
