"""Structured RL training data-point persistence.

The training curves saved as PNG/NPZ are useful for quick inspection, but paper
figures need the raw points. This module provides a small JSON/JSONL writer used
by RL stages to mirror every important training point into a stable directory
under the repository root.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, Optional, TextIO

from json_utils import json_default, read_json_file, to_jsonable, write_json_file


def _safe_slug(raw: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(raw).strip())
    slug = slug.strip("._-")
    return slug or "run"


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
            default=json_default,
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
        write_json_file(path, doc, sort_keys=True)

    def write_step(self, payload: Dict[str, Any]) -> None:
        self._write_jsonl("steps.jsonl", payload)

    def write_episode(self, payload: Dict[str, Any]) -> None:
        self._write_jsonl("episodes.jsonl", payload)

    def write_ppo_update(self, payload: Dict[str, Any]) -> None:
        self._write_jsonl("ppo_updates.jsonl", payload)

    def write_summary(self, payload: Dict[str, Any]) -> None:
        write_json_file(self.run_dir / "summary.json", payload, sort_keys=True)

    def close(self) -> None:
        for fh in self._files.values():
            fh.flush()
            fh.close()
        self._files.clear()
        self._line_counts.clear()

    def _write_jsonl(self, name: str, payload: Dict[str, Any]) -> None:
        fh = self._files.get(name)
        if fh is None:
            fh = (
                self.run_dir / name
            ).open("a", encoding="utf-8", buffering=self._jsonl_buffer_size)
            self._files[name] = fh
            self._line_counts[name] = 0
        fh.writelines(self._jsonl_encoder.iterencode(to_jsonable(payload, preserve_native=True)))
        fh.write("\n")
        self._line_counts[name] = self._line_counts.get(name, 0) + 1
        if self._line_counts[name] % self._jsonl_flush_interval == 0:
            fh.flush()

    def __enter__(self) -> "RLDataPointWriter":
        return self

    def __exit__(self, exc_type: Optional[type], exc: Optional[BaseException], tb: Any) -> None:
        self.close()
