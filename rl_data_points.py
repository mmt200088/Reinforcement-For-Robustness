"""Structured RL training data-point persistence.

The training curves saved as PNG/NPZ are useful for quick inspection, but paper
figures need the raw points. This module provides a small JSON/JSONL writer used
by RL stages to mirror every important training point into a stable directory
under the repository root.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, TextIO

import numpy as np


def to_jsonable(value: Any) -> Any:
    """Convert common training values into JSON-serializable objects."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return to_jsonable(value.detach().cpu().tolist())
    except Exception:
        pass
    return value


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
    ) -> None:
        self.root_dir = Path(root_dir)
        self.stage = str(stage)
        self.model_type = str(model_type)
        self.dataset = str(dataset)
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

    def write_manifest(self, payload: Dict[str, Any]) -> None:
        doc = dict(payload)
        doc.update(
            {
                "stage": self.stage,
                "model_type": self.model_type,
                "dataset": self.dataset,
                "run_id": self.run_id,
                "run_dir": str(self.run_dir),
            }
        )
        (self.run_dir / "manifest.json").write_text(
            json.dumps(to_jsonable(doc), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )

    def write_step(self, payload: Dict[str, Any]) -> None:
        self._write_jsonl("steps.jsonl", payload)

    def write_episode(self, payload: Dict[str, Any]) -> None:
        self._write_jsonl("episodes.jsonl", payload)

    def write_ppo_update(self, payload: Dict[str, Any]) -> None:
        self._write_jsonl("ppo_updates.jsonl", payload)

    def write_summary(self, payload: Dict[str, Any]) -> None:
        (self.run_dir / "summary.json").write_text(
            json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )

    def close(self) -> None:
        for fh in self._files.values():
            fh.close()
        self._files.clear()

    def _write_jsonl(self, name: str, payload: Dict[str, Any]) -> None:
        fh = self._files.get(name)
        if fh is None:
            fh = (self.run_dir / name).open("a", encoding="utf-8", buffering=1)
            self._files[name] = fh
        fh.write(json.dumps(to_jsonable(payload), ensure_ascii=False, sort_keys=True) + "\n")

    def __enter__(self) -> "RLDataPointWriter":
        return self

    def __exit__(self, exc_type: Optional[type], exc: Optional[BaseException], tb: Any) -> None:
        self.close()
