"""Shared helpers for lightweight HTML/metric report scripts."""
from __future__ import annotations

import html
import math
from typing import Any, Iterable, Mapping, Sequence


def format_float(
        value: Any,
        *,
        digits: int = 6,
        none_text: str = "",
        nan_text: str = "nan",
        ) -> str:
    """Format a numeric value for compact HTML tables.

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


def html_table(
        headers: Sequence[Any],
        rows: Iterable[Sequence[Any]],
        *,
        allow_html_cells: bool = False,
        row_classes: Sequence[str] | None = None,
        table_attrs: str = "",
        ) -> str:
    """Render a small escaped HTML table.

    ``allow_html_cells`` preserves the existing report convention where a cell
    starting with ``<`` is intentionally pre-rendered HTML. Keep it disabled for
    untrusted rows.
    """
    attrs = f" {table_attrs.strip()}" if str(table_attrs or "").strip() else ""
    parts = [f"<table{attrs}><thead><tr>"]
    for header in headers:
        parts.append(f"<th>{html.escape(str(header))}</th>")
    parts.append("</tr></thead><tbody>")
    for idx, row in enumerate(rows):
        row_class = ""
        if row_classes is not None and idx < len(row_classes):
            row_class_value = str(row_classes[idx] or "").strip()
            if row_class_value:
                row_class = f' class="{html.escape(row_class_value, quote=True)}"'
        parts.append(f"<tr{row_class}>")
        for cell in row:
            if allow_html_cells and isinstance(cell, str) and cell.startswith("<"):
                parts.append(f"<td>{cell}</td>")
            else:
                parts.append(f"<td>{html.escape(str(cell))}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "\n".join(parts)


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
