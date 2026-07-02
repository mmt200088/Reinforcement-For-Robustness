"""Shared JSON normalization helpers for reports and RL artifacts."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

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


def to_jsonable(value: Any) -> Any:
    """Convert common project values into JSON-serializable objects."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    tensor_type = _torch_tensor_type()
    if tensor_type and isinstance(value, tensor_type):
        return to_jsonable(value.detach().cpu().tolist())
    return value
