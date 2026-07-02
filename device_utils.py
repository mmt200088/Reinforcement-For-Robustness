"""Shared device-list parsing helpers for Stage-1 and Stage-2 runners."""
from __future__ import annotations

from typing import Any, Iterable, List, Sequence


def split_device_spec_tokens(
        spec: Any,
        *,
        disabled_tokens: Iterable[str] = (),
        ) -> List[str]:
    """Split CLI/CUDA-visible device specs into non-empty string tokens.

    This deliberately does not normalize logical ids to ``cuda:N`` because
    CUDA_VISIBLE_DEVICES can contain physical ids or UUIDs. Callers that need
    logical ``cuda:N`` names should normalize these tokens separately.
    """
    if spec is None:
        return []
    disabled = {str(token).lower() for token in disabled_tokens}
    if isinstance(spec, str):
        text = spec.strip()
        if not text or text.lower() in disabled:
            return []
        if (text.startswith("(") and text.endswith(")")) or (
            text.startswith("[") and text.endswith("]")
        ):
            text = text[1:-1].strip()
        raw_items: Iterable[Any] = text.split(",")
    elif isinstance(spec, Sequence) and not isinstance(spec, (bytes, bytearray)):
        raw_items = spec
    else:
        text = str(spec).strip()
        if not text or text.lower() in disabled:
            return []
        raw_items = [text]

    out: List[str] = []
    for item in raw_items:
        token = str(item).strip()
        if not token or token.lower() in disabled:
            continue
        out.append(token)
    return out


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
        tokens = split_device_spec_tokens(spec)

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
