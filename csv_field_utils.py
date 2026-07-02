"""Helpers for tolerant CSV/header field lookup in report scripts."""
from __future__ import annotations

import csv
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


def normalize_field_name(fieldname: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(fieldname).strip().lower()).strip("_")


def normalized_row(row: Mapping[str, str]) -> dict[str, str]:
    return {normalize_field_name(key): value for key, value in row.items()}


def first_present(row: Mapping[str, str], keys: Sequence[str]) -> str | None:
    for key in keys:
        if key in row:
            return row[key]
    return None


def normalized_field_lookup(fieldnames: Sequence[str | None] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for fieldname in fieldnames or ():
        if fieldname is None:
            continue
        out[normalize_field_name(fieldname)] = fieldname
    return out


def first_present_by_lookup(
        row: Mapping[str, str],
        field_lookup: Mapping[str, str],
        keys: Sequence[str],
        ) -> str | None:
    for key in keys:
        fieldname = field_lookup.get(key)
        if fieldname is not None and fieldname in row:
            return row[fieldname]
    return None


def normalized_field_index(
        fieldnames: Sequence[str | None] | None,
        *,
        keep_first: bool = False,
        ) -> dict[str, int]:
    out: dict[str, int] = {}
    for idx, fieldname in enumerate(fieldnames or ()):
        if fieldname is None:
            continue
        normalized = normalize_field_name(fieldname)
        if not normalized:
            continue
        if keep_first:
            out.setdefault(normalized, int(idx))
        else:
            out[normalized] = int(idx)
    return out


def first_present_by_index(
        row: Sequence[str],
        field_index: Mapping[str, int],
        keys: Sequence[str],
        ) -> str | None:
    for key in keys:
        idx = field_index.get(key)
        if idx is not None and 0 <= int(idx) < len(row):
            return row[int(idx)]
    return None


def write_csv_rows(
        path: str | Path,
        rows: Sequence[Mapping[str, Any]],
        fieldnames: Sequence[str],
        *,
        default: Any = "",
        ) -> Path:
    """Write a finite CSV artifact after projecting rows to ``fieldnames``."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, default) for field in fields})
    return out_path
