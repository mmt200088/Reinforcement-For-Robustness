"""Shared JSON normalization helpers for reports and RL artifacts."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

_TORCH_TENSOR_TYPE: Any = None
_TORCH_TENSOR_TYPE_RESOLVED = False


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
    leaves. ``stringify_unknown`` matches older script helpers that needed
    unknown objects to survive a plain ``json.dumps`` call.
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
        return to_jsonable(
            asdict(value),
            stringify_unknown=stringify_unknown,
            preserve_native=preserve_native,
        )
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
