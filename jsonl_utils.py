"""Shared JSONL readers for report and diagnostics scripts."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterable, Literal, TextIO

from json_utils import json_default, to_jsonable

JsonlErrorMode = Literal["skip", "raise"]


def resolve_jsonl_path(path: str | Path, *, gzip_fallback: bool = False) -> Path:
    jsonl_path = Path(path)
    if jsonl_path.exists() or not gzip_fallback:
        return jsonl_path
    gz_path = Path(str(jsonl_path) + ".gz")
    return gz_path if gz_path.exists() else jsonl_path


def open_jsonl(path: str | Path, *, gzip_fallback: bool = False) -> TextIO:
    jsonl_path = resolve_jsonl_path(path, gzip_fallback=gzip_fallback)
    if jsonl_path.suffix == ".gz":
        return gzip.open(jsonl_path, "rt", encoding="utf-8", errors="replace")
    return jsonl_path.open(encoding="utf-8", errors="replace")


def iter_jsonl_records(
        path: str | Path,
        *,
        errors: JsonlErrorMode = "skip",
        dict_only: bool = True,
        gzip_fallback: bool = False,
        ) -> Iterable[tuple[int, Any]]:
    """Yield ``(line_no, parsed_payload)`` JSONL records.

    Blank lines are ignored.  ``errors="skip"`` preserves legacy monitor
    behavior for partially-written logs; ``errors="raise"`` preserves verifier
    behavior that reports the file and line number on malformed JSON.
    """
    jsonl_path = resolve_jsonl_path(path, gzip_fallback=gzip_fallback)
    with open_jsonl(jsonl_path) as handle:
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
        gzip_fallback: bool = False,
        ) -> Iterable[Any]:
    """Yield parsed JSONL rows."""
    for _line_no, payload in iter_jsonl_records(
        path,
        errors=errors,
        dict_only=dict_only,
        gzip_fallback=gzip_fallback,
    ):
        yield payload


def read_jsonl(
        path: str | Path,
        *,
        errors: JsonlErrorMode = "skip",
        dict_only: bool = True,
        missing_ok: bool = False,
        gzip_fallback: bool = False,
        ) -> list[Any]:
    jsonl_path = resolve_jsonl_path(path, gzip_fallback=gzip_fallback)
    if missing_ok and not jsonl_path.exists():
        return []
    return list(iter_jsonl(jsonl_path, errors=errors, dict_only=dict_only))


def write_jsonl_rows(
        path: str | Path,
        rows: Iterable[Any],
        *,
        ensure_ascii: bool = False,
        sort_keys: bool = False,
        ) -> Path:
    """Write a finite JSONL artifact using shared JSON normalization.

    This helper is for report/diagnostic scripts that write bounded row sets.
    High-throughput append-only training logs should keep using their buffered
    writers so flush cadence and open file reuse remain explicit.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    encoder = json.JSONEncoder(
        ensure_ascii=bool(ensure_ascii),
        sort_keys=bool(sort_keys),
        default=json_default,
    )
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.writelines(encoder.iterencode(to_jsonable(row, preserve_native=True)))
            handle.write("\n")
    return out_path


def read_jsonl_fields(
        path: str | Path,
        fields: tuple[str, ...] | list[str] | None = None,
        *,
        errors: JsonlErrorMode = "skip",
        missing_ok: bool = True,
        gzip_fallback: bool = False,
        ) -> list[dict[str, Any]]:
    jsonl_path = resolve_jsonl_path(path, gzip_fallback=gzip_fallback)
    if missing_ok and not jsonl_path.exists():
        return []
    wanted = tuple(fields or ())
    rows: list[dict[str, Any]] = []
    for row in iter_jsonl(jsonl_path, errors=errors, dict_only=True):
        if wanted:
            rows.append({key: row[key] for key in wanted if key in row})
        else:
            rows.append(dict(row))
    return rows


def read_jsonl_xy(
        path: str | Path,
        x_field: str,
        y_field: str,
        *,
        errors: JsonlErrorMode = "skip",
        missing_ok: bool = True,
        gzip_fallback: bool = False,
        ) -> tuple[list[float], list[float]]:
    jsonl_path = resolve_jsonl_path(path, gzip_fallback=gzip_fallback)
    if missing_ok and not jsonl_path.exists():
        return [], []
    xs: list[float] = []
    ys: list[float] = []
    for row in iter_jsonl(jsonl_path, errors=errors, dict_only=True):
        xs.append(float(row.get(x_field, 0.0)))
        ys.append(float(row.get(y_field, 0.0)))
    return xs, ys


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
