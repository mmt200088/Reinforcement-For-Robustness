"""Shared JSON normalization helpers for reports and RL artifacts."""
from __future__ import annotations

import hashlib
import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

_TORCH_TENSOR_TYPE: Any = None
_TORCH_TENSOR_TYPE_RESOLVED = False
_RAISE = object()
_STABLE_JSON_ENCODER = json.JSONEncoder(
    ensure_ascii=True,
    sort_keys=True,
    separators=(",", ":"),
)


def _torch_tensor_type() -> Any:
    global _TORCH_TENSOR_TYPE, _TORCH_TENSOR_TYPE_RESOLVED
    if not _TORCH_TENSOR_TYPE_RESOLVED:
        try:
            import torch

            _TORCH_TENSOR_TYPE = torch.Tensor
        except Exception:
            _TORCH_TENSOR_TYPE = ()
        _TORCH_TENSOR_TYPE_RESOLVED = True
    return _TORCH_TENSOR_TYPE


def to_jsonable(
        value: Any,
        *,
        stringify_unknown: bool = False,
        preserve_native: bool = False,
        ) -> Any:
    """Convert common project values into JSON-serializable objects.

    ``preserve_native`` lets hot report builders keep already-JSON-native
    branches by identity while still converting numpy/dataclass/path/tensor
    leaves. ``stringify_unknown`` lets diagnostic writers preserve otherwise
    unsupported objects through a plain ``json.dumps`` call.
    """
    if isinstance(value, dict):
        converted_dict: dict[str, Any] | None = None
        for key, item in value.items():
            out_key = key if isinstance(key, str) else str(key)
            converted = to_jsonable(
                item,
                stringify_unknown=stringify_unknown,
                preserve_native=preserve_native,
            )
            if preserve_native and converted_dict is None and out_key is key and converted is item:
                continue
            if converted_dict is None:
                converted_dict = {}
                for prefix_key, prefix_item in value.items():
                    if prefix_key == key:
                        break
                    converted_dict[str(prefix_key)] = prefix_item
            converted_dict[str(out_key)] = converted
        return value if preserve_native and converted_dict is None else (converted_dict or {})
    if isinstance(value, Mapping):
        return {
            str(k): to_jsonable(
                v,
                stringify_unknown=stringify_unknown,
                preserve_native=preserve_native,
            )
            for k, v in value.items()
        }
    if isinstance(value, list):
        converted_list: list[Any] | None = None
        for idx, item in enumerate(value):
            converted = to_jsonable(
                item,
                stringify_unknown=stringify_unknown,
                preserve_native=preserve_native,
            )
            if preserve_native and converted_list is None and converted is item:
                continue
            if converted_list is None:
                converted_list = value[:idx]
            converted_list.append(converted)
        return value if preserve_native and converted_list is None else (converted_list or [])
    if isinstance(value, tuple):
        return [
            to_jsonable(v, stringify_unknown=stringify_unknown, preserve_native=preserve_native)
            for v in value
        ]
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: to_jsonable(
                getattr(value, item.name),
                stringify_unknown=stringify_unknown,
                preserve_native=preserve_native,
            )
            for item in fields(value)
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return to_jsonable(
            value.tolist(),
            stringify_unknown=stringify_unknown,
            preserve_native=preserve_native,
        )
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    tensor_type = _torch_tensor_type()
    if tensor_type and isinstance(value, tensor_type):
        return to_jsonable(
            value.detach().cpu().tolist(),
            stringify_unknown=stringify_unknown,
            preserve_native=preserve_native,
        )
    if stringify_unknown:
        return str(value)
    return value


def json_default(value: Any) -> Any:
    """``json.dump(s, default=...)`` adapter for project scalar/container values."""
    converted = to_jsonable(value)
    if converted is value:
        raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")
    return converted


def stable_json_key(value: Any) -> str:
    return _STABLE_JSON_ENCODER.encode(to_jsonable(value, preserve_native=True))


def stable_json_hash(value: Any) -> str:
    h = hashlib.sha256()
    for chunk in _STABLE_JSON_ENCODER.iterencode(to_jsonable(value, preserve_native=True)):
        h.update(chunk.encode("utf-8"))
    return h.hexdigest()


def bounded_stable_json_hash(value: Any) -> str:
    """Hash a known-small payload without per-token ``sha256.update`` calls."""
    return hashlib.sha256(stable_json_key(value).encode("utf-8")).hexdigest()


def write_json_file(
        path: str | Path,
        payload: Any,
        *,
        ensure_ascii: bool = False,
        indent: int | None = 2,
        sort_keys: bool = False,
        trailing_newline: bool = True,
        ) -> Path:
    """Write one JSON artifact using the repository's shared normalization.

    It creates the parent directory and normalizes numpy /
    dataclass / Path / optional torch values through :func:`to_jsonable`, and
    returns the written path for callers that want to record it.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(
            to_jsonable(payload, preserve_native=True),
            handle,
            ensure_ascii=bool(ensure_ascii),
            indent=indent,
            sort_keys=bool(sort_keys),
            default=json_default,
        )
        if trailing_newline:
            handle.write("\n")
    return out_path


def read_json_file(
        path: str | Path,
        *,
        encoding: str = "utf-8",
        default: Any = _RAISE,
        ) -> Any:
    """Read a JSON artifact through the repository's shared file seam.

    ``default`` is for optional sidecar artifacts in reports. Missing or invalid
    JSON returns the supplied default; callers that need strict reads should omit
    it and let the underlying exception surface.
    """
    try:
        with Path(path).open(encoding=encoding) as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        if default is not _RAISE:
            return default
        raise
