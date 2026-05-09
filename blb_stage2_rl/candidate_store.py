"""Candidate cache helpers for the BLB Stage-2 optimization playbook.

The store is intentionally small: JSONL on disk, stable action hashes, explicit
fidelity ordering, and the hard-priority rank key used by the playbook.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


FIDELITY_ORDER = {
    "F0": 0,
    "F1": 1,
    "F2": 2,
    "F3": 3,
    "F4": 4,
}


def normalize_action_indices(action_indices: Any) -> List[int]:
    if hasattr(action_indices, "tolist"):
        action_indices = action_indices.tolist()
    if isinstance(action_indices, str):
        action_indices = json.loads(action_indices)
    if not isinstance(action_indices, Iterable):
        raise TypeError("action_indices must be an iterable of integers")
    out: List[int] = []
    for item in action_indices:
        if isinstance(item, (list, tuple)):
            out.extend(normalize_action_indices(item))
        else:
            out.append(int(item))
    return out


def action_hash(action_indices: Any) -> str:
    payload = json.dumps(
        normalize_action_indices(action_indices),
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fidelity_rank(fidelity: str) -> int:
    key = str(fidelity or "F0").upper()
    return int(FIDELITY_ORDER.get(key, -1))


def _finite_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def candidate_rank_key(record: Mapping[str, Any]) -> Tuple[float, float, float, float]:
    """Hard-priority ordering: validity, accuracy, stability, then cost."""
    valid = bool(record.get("valid", not bool(record.get("invalid", False))))
    invalid_flag = 0.0 if valid else 1.0
    accuracy_violation = _finite_float(record.get("acc_violation", 0.0), 1.0e9)
    stability_violation = _finite_float(record.get("stability_violation", 0.0), 1.0e9)
    normalized_cost = _finite_float(
        record.get(
            "normalized_cost",
            record.get("cost_normalized", record.get("total_bits_norm", 1.0e9)),
        ),
        1.0e9,
    )
    return (
        invalid_flag,
        max(0.0, accuracy_violation),
        max(0.0, stability_violation),
        max(0.0, normalized_cost),
    )


class CandidateStore:
    """Append-only JSONL store keyed by stable action hash."""

    def __init__(self, path: os.PathLike[str] | str):
        self.path = Path(path)

    def read_all(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        records: List[Dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                records.append(json.loads(text))
        return records

    def append(self, record: Mapping[str, Any]) -> Dict[str, Any]:
        payload = dict(record)
        if "action_indices" not in payload:
            raise ValueError("candidate record requires action_indices")
        payload["action_indices"] = normalize_action_indices(payload["action_indices"])
        payload.setdefault("action_hash", action_hash(payload["action_indices"]))
        payload.setdefault("created_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        payload.setdefault("rank_key", list(candidate_rank_key(payload)))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
        return payload

    def best_for_action(self, action_indices: Any) -> Optional[Dict[str, Any]]:
        wanted = action_hash(action_indices)
        matches = [r for r in self.read_all() if r.get("action_hash") == wanted]
        if not matches:
            return None
        return sorted(
            matches,
            key=lambda r: (-fidelity_rank(str(r.get("fidelity", "F0"))), candidate_rank_key(r)),
        )[0]

    def should_evaluate(self, action_indices: Any, fidelity: str) -> bool:
        existing = self.best_for_action(action_indices)
        if existing is None:
            return True
        return fidelity_rank(str(existing.get("fidelity", "F0"))) < fidelity_rank(fidelity)
