"""Torch-light control primitives for elastic RL GPU execution.

This module is safe to import before CUDA initialization. It deliberately uses
only the Python standard library plus the project's string-only device parser.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional

from rfr.search.runtime.device_utils import split_device_spec_tokens


ELASTIC_GPU_RESTART_EXIT_CODE = 75
ELASTIC_GPU_FAILURE_FILENAME = "elastic_gpu_failure.json"
ELASTIC_GPU_RESTART_REQUEST_ENV = "RFR_ELASTIC_GPU_RESTART_REQUEST"

_RECOVERABLE_ERROR_MARKERS = (
    "all cuda-capable devices are busy or unavailable",
    "cuda-capable device is busy or unavailable",
    "cuda error: initialization error",
    "cuda error: unknown error",
    "cuda error: unspecified launch failure",
    "cuda driver error",
    "device is lost",
    "driver shutting down",
    "exited with code",
    "gpu has fallen off the bus",
    "gpu requires reset",
    "process died",
    "process exited",
    "requires reset",
    "xid",
)
_FATAL_ERROR_MARKERS = (
    "assert",
    "candidate identity",
    "device-side assert",
    "index out of",
    "invalid shape",
    "mat1 and mat2 shapes",
    "out of memory",
    "seed mismatch",
    "shape mismatch",
    "size mismatch",
    "trial seed",
)
_RECOVERABLE_EXCEPTION_TYPES = (
    BrokenPipeError,
    ChildProcessError,
    ConnectionError,
    EOFError,
    TimeoutError,
)


def _exception_chain(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        next_exc = current.__cause__
        if next_exc is None and not current.__suppress_context__:
            next_exc = current.__context__
        current = next_exc


def is_recoverable_gpu_failure(exc: BaseException) -> bool:
    """Return True only for device/process infrastructure failures.

    Scientific contract errors and capacity/configuration failures remain fatal
    even if their text also mentions CUDA.
    """
    chain = tuple(_exception_chain(exc))
    if any(isinstance(item, _RECOVERABLE_EXCEPTION_TYPES) for item in chain):
        return True
    message = "\n".join(
        f"{type(item).__name__}: {item}" for item in chain
    ).lower()
    if any(marker in message for marker in _FATAL_ERROR_MARKERS):
        return False
    return any(marker in message for marker in _RECOVERABLE_ERROR_MARKERS)


def _logical_cuda_index(device: object) -> int:
    text = str(device).strip().lower()
    if not text.startswith("cuda:"):
        raise ValueError(f"expected a logical cuda:N device, got {device!r}")
    suffix = text.split("cuda:", 1)[1].strip()
    if not suffix.isdigit():
        raise ValueError(f"expected a logical cuda:N device, got {device!r}")
    return int(suffix)


def physical_token_for_logical_device(
    device: object,
    cuda_visible_devices: Optional[str] = None,
) -> str:
    """Map logical ``cuda:N`` to the physical index/UUID visible at launch."""
    logical_index = _logical_cuda_index(device)
    visibility_was_explicit = cuda_visible_devices is not None
    visibility = (
        cuda_visible_devices
        if visibility_was_explicit
        else os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    if visibility is None:
        return str(logical_index)
    tokens = split_device_spec_tokens(visibility)
    if logical_index >= len(tokens):
        raise ValueError(
            f"logical device cuda:{logical_index} is outside "
            f"CUDA_VISIBLE_DEVICES={visibility!r}"
        )
    return str(tokens[logical_index])


class ElasticGPUFailure(RuntimeError):
    """A typed learner/worker GPU failure that permits exact restart."""

    def __init__(
        self,
        *,
        device: object,
        role: str,
        operation: str,
        cause: BaseException,
        cuda_visible_devices: Optional[str] = None,
    ) -> None:
        self.device = str(device)
        self.role = str(role)
        self.operation = str(operation)
        self.cause = cause
        self.cuda_visible_devices = (
            os.environ.get("CUDA_VISIBLE_DEVICES")
            if cuda_visible_devices is None
            else str(cuda_visible_devices)
        )
        try:
            self.physical_device = physical_token_for_logical_device(
                self.device,
                self.cuda_visible_devices,
            )
        except ValueError:
            self.physical_device = ""
        super().__init__(
            f"{self.role} GPU {self.device} failed during "
            f"{self.operation}: {cause}"
        )

    def to_record(self) -> Dict[str, Any]:
        return {
            "record_type": "elastic_gpu_failure_v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "device": self.device,
            "physical_device": self.physical_device,
            "cuda_visible_devices": self.cuda_visible_devices,
            "role": self.role,
            "operation": self.operation,
            "cause_type": type(self.cause).__name__,
            "cause": str(self.cause),
            "message": str(self),
            "exit_code": ELASTIC_GPU_RESTART_EXIT_CODE,
        }


class ElasticGPURestartRequested(RuntimeError):
    """A checkpoint-boundary restart requested to admit recovered devices."""

    def __init__(
        self,
        *,
        reason: str,
        physical_devices: Iterable[object] = (),
        payload: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.reason = str(reason)
        self.physical_devices = tuple(
            str(device) for device in physical_devices
        )
        self.payload = dict(payload or {})
        super().__init__(
            f"elastic GPU restart requested: {self.reason}; "
            f"devices={list(self.physical_devices)}"
        )

    def to_record(self) -> Dict[str, Any]:
        return {
            "record_type": "elastic_gpu_restart_request_v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "reason": self.reason,
            "physical_devices": list(self.physical_devices),
            "payload": dict(self.payload),
            "message": str(self),
            "exit_code": ELASTIC_GPU_RESTART_EXIT_CODE,
        }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(
            dict(payload),
            handle,
            ensure_ascii=True,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)
    return path


def _restart_request_path() -> Optional[Path]:
    raw = str(os.environ.get(ELASTIC_GPU_RESTART_REQUEST_ENV, "")).strip()
    return Path(raw).expanduser() if raw else None


def request_elastic_gpu_restart(
    *,
    reason: str,
    physical_devices: Iterable[object] = (),
    payload: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Atomically request a restart at the next committed PPO boundary."""
    path = _restart_request_path()
    if path is None:
        raise RuntimeError(
            f"{ELASTIC_GPU_RESTART_REQUEST_ENV} is not configured"
        )
    record = {
        "record_type": "elastic_gpu_restart_request_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reason": str(reason),
        "physical_devices": [
            str(device) for device in physical_devices
        ],
        "payload": dict(payload or {}),
    }
    return _atomic_write_json(path, record)


def consume_elastic_gpu_restart_request() -> Optional[Dict[str, Any]]:
    """Read and remove one complete supervisor restart request."""
    path = _restart_request_path()
    if path is None or not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"elastic restart request must be a JSON object: {path}")
    path.unlink()
    return payload


def raise_if_elastic_gpu_restart_requested(
        *,
        work_remaining: bool = True,
        ) -> None:
    """Raise after a committed PPO transaction only when training will continue."""
    payload = consume_elastic_gpu_restart_request()
    if payload is None or not bool(work_remaining):
        return
    raise ElasticGPURestartRequested(
        reason=str(payload.get("reason", "device_recovery")),
        physical_devices=payload.get("physical_devices") or (),
        payload=payload.get("payload") or {},
    )


def write_elastic_gpu_failure_record(
    run_output_dir: Optional[str],
    exc: ElasticGPUFailure | ElasticGPURestartRequested,
) -> Optional[Path]:
    if not str(run_output_dir or "").strip():
        return None
    path = (
        Path(str(run_output_dir)).expanduser()
        / "logs"
        / ELASTIC_GPU_FAILURE_FILENAME
    )
    return _atomic_write_json(path, exc.to_record())
