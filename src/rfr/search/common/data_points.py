"""Structured RL training data-point persistence.

The training curves saved as PNG/NPZ are useful for quick inspection, but paper
figures need the raw points. This module provides a small JSON/JSONL writer used
by RL stages to mirror every important training point into a stable directory
under the repository root.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Dict, Optional, TextIO

from rfr.common.json_utils import json_default, read_json_file
from rfr.common.jsonl_utils import recover_jsonl_file


def _safe_slug(raw: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(raw).strip())
    slug = slug.strip("._-")
    return slug or "run"


def _strict_jsonable(value: Any) -> Any:
    """Normalize project values and replace non-finite floats with JSON null."""
    if isinstance(value, dict):
        return {str(key): _strict_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        normalized = json_default(value)
    except TypeError:
        return value
    return _strict_jsonable(normalized)


def _strict_json_default(value: Any) -> Any:
    return _strict_jsonable(json_default(value))


def write_strict_json_file(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(
            _strict_jsonable(payload),
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=_strict_json_default,
            allow_nan=False,
        )
        handle.write("\n")
    os.replace(tmp_path, path)


def write_dataset_protocol(
    root_dir: str | Path,
    payload: Dict[str, Any],
) -> Path:
    path = Path(root_dir) / "dataset_protocol.json"
    write_strict_json_file(path, payload)
    return path


def make_unique_run_id(
    base_run_id: str,
    *,
    started_at: Any = None,
    pid: Optional[int] = None,
) -> str:
    """Return a readable run id that is unique per process invocation."""
    if started_at is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    elif isinstance(started_at, datetime):
        dt = started_at.astimezone(timezone.utc) if started_at.tzinfo else started_at
        stamp = dt.strftime("%Y%m%dT%H%M%S%fZ")
    else:
        stamp = re.sub(r"[^A-Za-z0-9]+", "", str(started_at).strip()) or "time"
    proc = os.getpid() if pid is None else int(pid)
    return f"{_safe_slug(base_run_id)}__{stamp}__pid{proc}"


class RLDataPointWriter:
    """Append-only JSONL writer for one RL run."""

    _CHECKPOINT_JSONL_NAMES = (
        "steps.jsonl",
        "episodes.jsonl",
        "ppo_updates.jsonl",
    )

    def __init__(
        self,
        *,
        root_dir: str | Path,
        run_id: str,
        stage: str,
        model_type: str,
        dataset: str,
        jsonl_buffer_size: int = 1024 * 1024,
        jsonl_flush_interval: int = 64,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.stage = str(stage)
        self.model_type = str(model_type)
        self.dataset = str(dataset)
        self._jsonl_buffer_size = max(1, int(jsonl_buffer_size))
        self._jsonl_flush_interval = max(1, int(jsonl_flush_interval))
        self._jsonl_encoder = json.JSONEncoder(
            ensure_ascii=False,
            sort_keys=True,
            default=_strict_json_default,
            allow_nan=False,
        )
        self.run_id = _safe_slug(run_id)
        self.run_dir = (
            self.root_dir
            / self.stage
            / _safe_slug(self.model_type)
            / _safe_slug(self.dataset)
            / self.run_id
        )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._files: Dict[str, TextIO] = {}
        self._line_counts: Dict[str, int] = {}

    def write_manifest(self, payload: Dict[str, Any]) -> None:
        path = self.run_dir / "manifest.json"
        doc: Dict[str, Any] = {}
        if path.is_file():
            try:
                loaded = read_json_file(path)
                if isinstance(loaded, dict):
                    doc.update(loaded)
            except Exception:
                doc = {}
        doc.update(dict(payload))
        doc.update(
            {
                "stage": self.stage,
                "model_type": self.model_type,
                "dataset": self.dataset,
                "run_id": self.run_id,
                "run_dir": str(self.run_dir),
            }
        )
        write_strict_json_file(path, doc)

    def write_step(self, payload: Dict[str, Any]) -> None:
        self._write_jsonl("steps.jsonl", payload)

    def write_episode(self, payload: Dict[str, Any]) -> None:
        self._write_jsonl("episodes.jsonl", payload)

    def write_ppo_update(self, payload: Dict[str, Any]) -> None:
        self._write_jsonl("ppo_updates.jsonl", payload)

    def write_summary(self, payload: Dict[str, Any]) -> None:
        write_strict_json_file(self.run_dir / "summary.json", payload)

    def jsonl_path(self, name: str) -> Path:
        return self.run_dir / str(name)

    def committed_jsonl_sizes(self) -> Dict[str, int]:
        """Flush and return byte boundaries for checkpoint-coupled JSONL."""
        self.flush()
        return {
            name: (self.jsonl_path(name).stat().st_size if self.jsonl_path(name).exists() else 0)
            for name in self._CHECKPOINT_JSONL_NAMES
        }

    def recover_jsonl_files(self, committed_sizes: Optional[Dict[str, Any]]) -> None:
        """Repair mirrored records before opening append handles on resume."""
        if self._files:
            raise RuntimeError("cannot recover structured JSONL after opening writers")
        sizes = dict(committed_sizes or {})
        for name in self._CHECKPOINT_JSONL_NAMES:
            recover_jsonl_file(
                self.jsonl_path(name),
                committed_size=(sizes[name] if name in sizes else None),
            )

    def close(self) -> None:
        self.flush()
        for fh in self._files.values():
            fh.close()
        self._files.clear()
        self._line_counts.clear()

    def flush(self) -> None:
        for fh in self._files.values():
            fh.flush()

    def _write_jsonl(self, name: str, payload: Dict[str, Any]) -> None:
        fh = self._files.get(name)
        if fh is None:
            fh = (
                self.run_dir / name
            ).open("a", encoding="utf-8", buffering=self._jsonl_buffer_size)
            self._files[name] = fh
            self._line_counts[name] = 0
        fh.writelines(self._jsonl_encoder.iterencode(_strict_jsonable(payload)))
        fh.write("\n")
        self._line_counts[name] = self._line_counts.get(name, 0) + 1
        if self._line_counts[name] % self._jsonl_flush_interval == 0:
            fh.flush()

    def __enter__(self) -> "RLDataPointWriter":
        return self

    def __exit__(self, exc_type: Optional[type], exc: Optional[BaseException], tb: Any) -> None:
        self.close()
