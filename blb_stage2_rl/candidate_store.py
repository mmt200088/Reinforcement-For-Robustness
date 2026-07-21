"""Candidate cache helpers for the BLB Stage-2 optimization playbook.

The store is intentionally small: JSONL on disk, stable action hashes, explicit
fidelity ordering, and the hard-priority rank key used by the playbook.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from json_utils import stable_json_hash, to_jsonable
from jsonl_utils import iter_jsonl

try:
    from .statistical_constraints import TrialSeries
except ImportError:  # Standalone diagnostics/tests load this module from its path.
    from statistical_constraints import TrialSeries

FIDELITY_ORDER = {
    "F0": 0,
    "F1": 1,
    "F4": 2,
}
_CANDIDATE_JSONL_ENCODER = json.JSONEncoder(ensure_ascii=True, sort_keys=True)
_RECOVERY_RECORD_TYPE = "candidate_store_recovery_v1"
# Note: F2 / F3 were intermediate tiers in the original spec; deprecated and
# removed 2026-05-16. The active ladder is F0 (optimizer-only, no model
# forward) → F1 (small probe + few MC trials during training) → F4 (full
# validation_full final eval with real BLB install). Old JSONL records with
# ``fidelity="F2"`` / ``"F3"`` get rank ``-1`` from ``fidelity_rank`` and
# surface as legacy entries — they are not lost, just not promotable.


def normalize_action_indices(action_indices: Any) -> List[int]:
    if (
            not isinstance(action_indices, (str, bytes))
            and hasattr(action_indices, "reshape")
            and hasattr(action_indices, "shape")
    ):
        return [int(item) for item in action_indices.reshape(-1)]
    if hasattr(action_indices, "tolist"):
        action_indices = action_indices.tolist()
    if isinstance(action_indices, str):
        action_indices = json.loads(action_indices)
    if not isinstance(action_indices, Iterable):
        raise TypeError("action_indices must be an iterable of integers")
    try:
        return [int(item) for item in action_indices]
    except TypeError:
        pass
    out: List[int] = []
    for item in action_indices:
        if isinstance(item, (list, tuple)):
            out.extend(normalize_action_indices(item))
        else:
            out.append(int(item))
    return out


@lru_cache(maxsize=8192)
def _action_hash_from_tuple(action_indices: Tuple[int, ...]) -> str:
    h = hashlib.sha256()
    h.update(b"[")
    for idx, value in enumerate(action_indices):
        if idx:
            h.update(b",")
        h.update(str(int(value)).encode("ascii"))
    h.update(b"]")
    return h.hexdigest()


def action_hash(action_indices: Any) -> str:
    return _action_hash_from_tuple(tuple(normalize_action_indices(action_indices)))


def raw_action_hash(action_indices: Any) -> str:
    return action_hash(action_indices)


def sha256_json(value: Any) -> str:
    return stable_json_hash(value)


def _records_from_registry(registry_or_description: Any) -> List[Mapping[str, Any]]:
    if registry_or_description is None:
        return []
    if isinstance(registry_or_description, Mapping):
        for key in ("records", "slot_registry_full", "slots"):
            rows = registry_or_description.get(key)
            if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
                return [r for r in rows if isinstance(r, Mapping)]
        if "global_index" in registry_or_description:
            return [registry_or_description]
        return []
    if isinstance(registry_or_description, Sequence) and not isinstance(registry_or_description, (str, bytes)):
        return [r for r in registry_or_description if isinstance(r, Mapping)]
    return []


def _record_effective(record: Mapping[str, Any]) -> bool:
    if "effective" in record:
        return bool(record.get("effective"))
    if "is_effective" in record:
        return bool(record.get("is_effective"))
    if "is_required" in record:
        return bool(record.get("is_required"))
    return True


def effective_action_vector(
        action_indices: Any,
        registry_or_description: Any = None,
        baseline_action: Any = None,
        ) -> List[int]:
    raw = normalize_action_indices(action_indices)
    if registry_or_description is None:
        return list(raw)
    if baseline_action is None:
        raise ValueError("baseline_action is required when registry_or_description is provided")
    baseline = normalize_action_indices(baseline_action)
    if len(raw) != len(baseline):
        raise ValueError(f"action width {len(raw)} != baseline width {len(baseline)}")
    out = list(raw)
    for record in _records_from_registry(registry_or_description):
        if _record_effective(record):
            continue
        if "global_index" not in record:
            continue
        idx = int(record["global_index"])
        if 0 <= idx < len(out):
            out[idx] = int(baseline[idx])
    return out


def effective_action_hash(
        action_indices: Any,
        registry_or_description: Any = None,
        baseline_action: Any = None,
        ) -> str:
    return action_hash(
        effective_action_vector(
            action_indices,
            registry_or_description=registry_or_description,
            baseline_action=baseline_action,
        )
    )


def build_candidate_identity_context(
        *,
        action_space_version: str,
        registry_hash: str,
        max_sfs_hash: str,
        stage1_hash: str | None = None,
        stage1_degrees: Any | None = None,
        stage1_config_content_hash: str | None = None,
        stage1_gelu_degrees: Any | None = None,
        stage1_softmax_degrees: Any | None = None,
        profile: str,
        rescale_optimizer_mode: str,
        rescale_optimizer_root: str,
        rescale_optimizer_hash: str | None = None,
        rescale_optimizer_canonical_hash: str | None = None,
        decode_version: str,
        dataset: str,
        model: str,
        metric_policy_version: str,
        threshold_policy_hash: str,
        fidelity: str | None = None,
        mask_schedule_hash: str | None = None,
        ) -> Dict[str, Any]:
    """Build the context that makes an action-index candidate comparable."""
    if stage1_config_content_hash is None:
        stage1_config_content_hash = stage1_hash
    if stage1_gelu_degrees is None and isinstance(stage1_degrees, Mapping):
        stage1_gelu_degrees = stage1_degrees.get("gelu")
    if stage1_softmax_degrees is None and isinstance(stage1_degrees, Mapping):
        stage1_softmax_degrees = stage1_degrees.get("softmax")
    if rescale_optimizer_canonical_hash is None:
        rescale_optimizer_canonical_hash = rescale_optimizer_hash
    return {
        "action_space_version": str(action_space_version),
        "registry_hash": str(registry_hash),
        "max_sfs_hash": str(max_sfs_hash),
        "stage1_hash": str(stage1_hash or stage1_config_content_hash or ""),
        "stage1_degrees": stage1_degrees,
        "stage1_config_content_hash": str(stage1_config_content_hash or ""),
        "stage1_gelu_degrees": stage1_gelu_degrees,
        "stage1_softmax_degrees": stage1_softmax_degrees,
        "profile": str(profile),
        "rescale_optimizer_mode": str(rescale_optimizer_mode),
        "rescale_optimizer_root": str(rescale_optimizer_root),
        "rescale_optimizer_hash": str(rescale_optimizer_hash or rescale_optimizer_canonical_hash or ""),
        "rescale_optimizer_canonical_hash": str(rescale_optimizer_canonical_hash or ""),
        "decode_version": str(decode_version),
        "dataset": str(dataset),
        "model": str(model),
        "metric_policy_version": str(metric_policy_version),
        "threshold_policy_hash": str(threshold_policy_hash),
        "fidelity": None if fidelity is None else str(fidelity),
        "mask_schedule_hash": None if mask_schedule_hash is None else str(mask_schedule_hash),
    }


def candidate_key(
        action_indices: Any,
        identity_context: Mapping[str, Any],
        *,
        effective_action_indices: Any | None = None,
        effective_action_hash_value: str | None = None,
        ) -> str:
    effective_hash = (
        str(effective_action_hash_value)
        if effective_action_hash_value is not None
        else action_hash(effective_action_indices if effective_action_indices is not None else action_indices)
    )
    payload = {
        "candidate_key_basis": "effective_action_hash + identity_context",
        "effective_action_hash": effective_hash,
        "identity_context": dict(identity_context),
    }
    return sha256_json(payload)


def is_legacy_record(record: Mapping[str, Any]) -> bool:
    return not bool(record.get("candidate_key"))


def rescale_cost_rank_key(record: Mapping[str, Any]) -> Tuple[float, float]:
    cost = record.get("rescale_cost") if isinstance(record, Mapping) else None
    if isinstance(cost, Mapping):
        rank_key = cost.get("rank_key")
        if isinstance(rank_key, Sequence) and len(rank_key) >= 2 and not isinstance(rank_key, (str, bytes)):
            return (
                _finite_float(rank_key[0], 1.0e9),
                _finite_float(rank_key[1], 1.0e9),
            )
        terms = cost.get("optimizer_cost_terms")
        if isinstance(terms, Mapping):
            return (
                _finite_float(terms.get("total_bits_sum"), 1.0e9),
                _finite_float(terms.get("fusion_count"), 1.0e9),
            )
    optimizer = record.get("optimizer") if isinstance(record, Mapping) else None
    if isinstance(optimizer, Mapping):
        return (
            _finite_float(optimizer.get("total_bits_sum"), 1.0e9),
            _finite_float(
                optimizer.get("fusion_count", optimizer.get("total_fusion_count")),
                1.0e9,
            ),
        )
    return (1.0e9, 1.0e9)


def f0_sort_key(record: Mapping[str, Any]) -> Tuple[float, float, float]:
    """Optimizer-only F0 ordering: validity gate, then bits, then fusion."""
    valid = bool(record.get("valid", not bool(record.get("invalid", False))))
    optimizer_valid = record.get("optimizer_valid")
    if optimizer_valid is not None:
        valid = bool(optimizer_valid)
    invalid_flag = 0.0 if valid else 1.0
    bits, fusion = rescale_cost_rank_key(record)
    return (invalid_flag, max(0.0, bits), max(0.0, fusion))


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


def candidate_rank_key(record: Mapping[str, Any]) -> Tuple[float, ...]:
    """Hard-priority ordering: accuracy, stability, optimizer validity, then cost."""
    valid = bool(record.get("valid", not bool(record.get("invalid", False))))
    invalid_flag = 0.0 if valid else 1.0
    accuracy_violation = _finite_float(record.get("acc_violation", 0.0), 1.0e9)
    stability_violation = _finite_float(
        record.get("stability_violation", record.get("terminal_stab_violation", 0.0)),
        1.0e9,
    )
    priority_value = record.get("terminal_priority")
    if priority_value is not None:
        try:
            priority = int(priority_value)
        except (TypeError, ValueError):
            priority = 0
        invalid_steps = int(_finite_float(record.get("invalid_steps", 0), 0.0))
        terminal_reward = _finite_float(record.get("terminal_reward", 0.0), 0.0)
        total_reward = _finite_float(record.get("total_reward", 0.0), 0.0)
        metric1 = _finite_float(record.get("terminal_metric1_mean", 0.0), 0.0)
        metric2 = _finite_float(record.get("terminal_metric2_mean", 0.0), 0.0)
        if priority == 3 and valid and invalid_steps == 0:
            return (
                0.0,
                -terminal_reward,
                -total_reward,
                -max(0.0, _finite_float(record.get("terminal_cost_rank_score", 0.0), 0.0)),
                -_finite_float(record.get("terminal_fusion_gain", 0.0), 0.0),
                -_finite_float(record.get("terminal_k_gain", 0.0), 0.0),
                -_finite_float(record.get("terminal_bits_gain", 0.0), 0.0),
            )
        if priority == 2:
            return (
                1.0,
                max(0.0, stability_violation),
                -metric1,
                -metric2,
                -terminal_reward,
                -total_reward,
            )
        if priority == 1:
            return (
                2.0,
                invalid_flag,
                max(0.0, accuracy_violation),
                -metric1,
                -metric2,
                -terminal_reward,
                -total_reward,
            )
        return (3.0, invalid_flag, -terminal_reward, -total_reward)

    cost_value = record.get("normalized_cost", record.get("cost_normalized", record.get("total_bits_norm")))
    if cost_value is not None:
        normalized_cost = _finite_float(cost_value, 1.0e9)
        return (
            max(0.0, accuracy_violation),
            max(0.0, stability_violation),
            invalid_flag,
            max(0.0, normalized_cost),
        )
    bits, fusion = rescale_cost_rank_key(record)
    return (
        max(0.0, accuracy_violation),
        max(0.0, stability_violation),
        invalid_flag,
        max(0.0, bits),
        max(0.0, fusion),
    )


@dataclass(frozen=True)
class CandidateTrialEvidence:
    """Pooled robust trial groups for one canonical candidate identity."""

    candidate_key: str
    action_indices: Tuple[int, ...]
    trials: TrialSeries
    groups: Tuple[Mapping[str, Any], ...]
    promotion_attempted: bool = False
    promotion_status: str = ""

    @property
    def trial_count(self) -> int:
        return len(self.trials.loss)

    @property
    def promoted(self) -> bool:
        return self.promotion_status == "promoted"


class CandidateStore:
    """Append-only JSONL store keyed by stable action hash."""

    def __init__(self, path: os.PathLike[str] | str):
        self.path = Path(path)
        self._recovery_layout_size: Optional[int] = None
        self._recovery_markers: Tuple[Tuple[int, int, int, int], ...] = ()
        self._active_spans: Tuple[Tuple[int, int], ...] = ()
        self._logical_generation = 0
        self._trial_offsets_by_candidate_key: Optional[
            Dict[str, List[int]]
        ] = None
        self._trial_seeds_by_candidate_key: Optional[
            Dict[str, set[int]]
        ] = None
        self._promotion_state_by_candidate_key: Optional[
            Dict[str, Tuple[bool, str]]
        ] = None
        self._latest_promotion_by_candidate_key: Optional[
            Dict[str, Tuple[str, Dict[str, Any]]]
        ] = None

    def _reset_trial_indices(self) -> None:
        self._trial_offsets_by_candidate_key = None
        self._trial_seeds_by_candidate_key = None
        self._promotion_state_by_candidate_key = None
        self._latest_promotion_by_candidate_key = None

    def _invalidate_recovery_layout(self) -> None:
        self._recovery_layout_size = None
        self._recovery_markers = ()
        self._active_spans = ()

    @staticmethod
    def _resolve_active_spans(
            markers: Sequence[Tuple[int, int, int, int]],
            file_size: int,
            ) -> Tuple[Tuple[int, int], ...]:
        spans: List[Tuple[int, int]] = []
        end = int(file_size)
        marker_index = len(markers) - 1
        while end > 0:
            while marker_index >= 0 and markers[marker_index][0] >= end:
                marker_index -= 1
            if marker_index < 0:
                spans.append((0, end))
                break
            _marker_start, marker_end, checkpoint_size, _generation = (
                markers[marker_index]
            )
            if marker_end < end:
                spans.append((marker_end, end))
            end = checkpoint_size
            marker_index -= 1
        spans.reverse()
        return tuple(spans)

    def _extend_active_tail(self, row_offset: int, row_end: int) -> None:
        if self._recovery_layout_size != row_offset:
            self._invalidate_recovery_layout()
            return
        spans = list(self._active_spans)
        if spans and spans[-1][1] == row_offset:
            spans[-1] = (spans[-1][0], row_end)
        else:
            spans.append((row_offset, row_end))
        self._active_spans = tuple(spans)
        self._recovery_layout_size = row_end

    @staticmethod
    def _decode_jsonl_row(
            line: bytes,
            *,
            path: Path,
            offset: int,
            ) -> Optional[Dict[str, Any]]:
        if not line or line.isspace():
            return None
        try:
            payload = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path}: byte {offset}: invalid JSON") from exc
        if not isinstance(payload, dict):
            return None
        return payload

    def _load_recovery_layout(self) -> None:
        self._repair_unterminated_tail()
        file_size = self.path.stat().st_size if self.path.exists() else 0
        if self._recovery_layout_size == file_size:
            return

        markers: List[Tuple[int, int, int, int]] = []
        if file_size:
            with self.path.open("rb") as handle:
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    if _RECOVERY_RECORD_TYPE.encode("ascii") not in line:
                        continue
                    payload = self._decode_jsonl_row(
                        line, path=self.path, offset=offset,
                    )
                    if payload is None or payload.get("record_type") != _RECOVERY_RECORD_TYPE:
                        continue
                    checkpoint_size = payload.get("checkpoint_size")
                    generation = payload.get("logical_generation")
                    if (
                            isinstance(checkpoint_size, bool)
                            or not isinstance(checkpoint_size, int)
                            or checkpoint_size < 0
                            or checkpoint_size > offset
                    ):
                        raise ValueError("invalid candidate-store recovery checkpoint")
                    if (
                            isinstance(generation, bool)
                            or not isinstance(generation, int)
                            or generation <= 0
                    ):
                        raise ValueError("invalid candidate-store logical generation")
                    markers.append(
                        (offset, handle.tell(), checkpoint_size, generation)
                    )

        # The newest marker selects a checkpointed logical prefix plus its own
        # tail. Repeating that rule through older markers excludes every
        # abandoned branch without rewriting any complete physical row.
        self._recovery_layout_size = file_size
        self._recovery_markers = tuple(markers)
        self._active_spans = self._resolve_active_spans(markers, file_size)
        self._logical_generation = max(
            (marker[3] for marker in markers),
            default=0,
        )

    def _iter_active_records(self) -> Iterator[Tuple[int, Dict[str, Any]]]:
        self._load_recovery_layout()
        if not self.path.exists():
            return
        with self.path.open("rb") as handle:
            for span_start, span_end in self._active_spans:
                handle.seek(span_start)
                while handle.tell() < span_end:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    payload = self._decode_jsonl_row(
                        line, path=self.path, offset=offset,
                    )
                    if payload is None or payload.get("record_type") == _RECOVERY_RECORD_TYPE:
                        continue
                    yield offset, payload

    def recover_to_checkpoint_size(self, committed_size: int) -> None:
        """Hide complete post-checkpoint rows without rewriting append-only history."""
        if isinstance(committed_size, bool):
            raise TypeError("committed_size must be a non-negative integer")
        size = int(committed_size)
        if size < 0:
            raise ValueError("committed_size must be non-negative")
        self._repair_unterminated_tail()
        current = self.path.stat().st_size if self.path.exists() else 0
        if size > current:
            raise ValueError(
                f"candidate store is shorter than checkpoint: {current} < {size}"
            )
        if size and self.path.exists():
            with self.path.open("rb") as handle:
                handle.seek(size - 1)
                if handle.read(1) != b"\n":
                    raise ValueError("checkpoint candidate-store size is not a JSONL boundary")
        if current > size:
            self._load_recovery_layout()
            next_generation = self._logical_generation + 1
            marker = {
                "record_type": _RECOVERY_RECORD_TYPE,
                "checkpoint_size": size,
                "logical_generation": next_generation,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(_CANDIDATE_JSONL_ENCODER.encode(marker) + "\n")
            marker_end = self.path.stat().st_size
            markers = self._recovery_markers + (
                (current, marker_end, size, next_generation),
            )
            self._recovery_layout_size = marker_end
            self._recovery_markers = markers
            self._active_spans = self._resolve_active_spans(markers, marker_end)
            self._logical_generation = next_generation
        self._reset_trial_indices()

    def _repair_unterminated_tail(self) -> None:
        """Recover only the final row when an append was interrupted."""
        if not self.path.exists() or self.path.stat().st_size == 0:
            return
        with self.path.open("r+b") as handle:
            handle.seek(0, os.SEEK_END)
            end = handle.tell()
            handle.seek(end - 1)
            if handle.read(1) == b"\n":
                return

            cursor = end
            row_start = 0
            while cursor > 0:
                chunk_size = min(64 * 1024, cursor)
                cursor -= chunk_size
                handle.seek(cursor)
                chunk = handle.read(chunk_size)
                newline = chunk.rfind(b"\n")
                if newline >= 0:
                    row_start = cursor + newline + 1
                    break
            handle.seek(row_start)
            tail = handle.read(end - row_start)
            try:
                payload = json.loads(tail.decode("utf-8"))
                if not isinstance(payload, Mapping):
                    raise ValueError("candidate JSONL rows must be objects")
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                handle.truncate(row_start)
                self._invalidate_recovery_layout()
            else:
                handle.seek(end)
                handle.write(b"\n")
                self._invalidate_recovery_layout()

    def read_all(self) -> List[Dict[str, Any]]:
        return list(self.iter_active_records())

    def iter_active_records(self) -> Iterator[Dict[str, Any]]:
        """Stream the current logical generation without materializing JSONL."""
        if not self.path.exists():
            return
        self._load_recovery_layout()
        payloads: Iterable[Dict[str, Any]]
        if self._recovery_markers:
            payloads = (payload for _offset, payload in self._iter_active_records())
        else:
            payloads = iter_jsonl(self.path, errors="raise")
        for payload in payloads:
            payload.setdefault("legacy_record", is_legacy_record(payload))
            yield payload

    def append(self, record: Mapping[str, Any]) -> Dict[str, Any]:
        payload = dict(record)
        if "action_indices" not in payload:
            raise ValueError("candidate record requires action_indices")
        payload["action_indices"] = normalize_action_indices(payload["action_indices"])
        payload.setdefault("raw_action_indices", list(payload["action_indices"]))
        payload["raw_action_indices"] = normalize_action_indices(payload["raw_action_indices"])
        payload.setdefault("raw_action_hash", action_hash(payload["raw_action_indices"]))
        payload.setdefault("action_hash", payload["raw_action_hash"])
        payload.setdefault("action_vector_hash", payload["action_hash"])
        if "effective_action_indices" in payload:
            payload["effective_action_indices"] = normalize_action_indices(payload["effective_action_indices"])
        else:
            payload["effective_action_indices"] = list(payload["action_indices"])
        payload.setdefault("effective_action_hash", action_hash(payload["effective_action_indices"]))
        payload.setdefault("candidate_key_basis", "effective_action_hash + identity_context")
        identity_context = payload.get("identity_context")
        if isinstance(identity_context, Mapping):
            payload.setdefault(
                "candidate_key",
                candidate_key(
                    payload["action_indices"],
                    identity_context,
                    effective_action_indices=payload["effective_action_indices"],
                    effective_action_hash_value=payload["effective_action_hash"],
                ),
            )
            payload.setdefault("identity_context_hash", sha256_json(dict(identity_context)))
        payload.setdefault("legacy_record", is_legacy_record(payload))
        payload.setdefault("created_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        payload.setdefault("rank_key", list(candidate_rank_key(payload)))
        self._load_recovery_layout()
        payload.setdefault("logical_generation", self._logical_generation)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._repair_unterminated_tail()
        row_offset = self.path.stat().st_size if self.path.exists() else 0
        with self.path.open("a", encoding="utf-8") as f:
            f.write(_CANDIDATE_JSONL_ENCODER.encode(payload) + "\n")
        if self.path.exists():
            self._extend_active_tail(row_offset, self.path.stat().st_size)
        else:
            self._invalidate_recovery_layout()
        if (
                self._trial_offsets_by_candidate_key is not None
                and payload.get("record_type") in (
                    "candidate_trial_group_v1", "candidate_promotion_status_v1",
                )
                and payload.get("candidate_key")
        ):
            self._index_trial_record(payload, offset=row_offset)
        return payload

    @staticmethod
    def _trial_group_from_record(
            record: Mapping[str, Any],
            ) -> Optional[TrialSeries]:
        group = record.get("trial_group")
        if not isinstance(group, Mapping):
            return None
        return TrialSeries(
            loss=group.get("loss", ()),
            metric1=group.get("metric1", ()),
            metric2=group.get("metric2", ()),
            seeds=group.get("seeds", ()),
        )

    def _index_trial_record(
            self,
            record: Mapping[str, Any],
            *,
            offset: int,
            ) -> None:
        if (
                self._trial_offsets_by_candidate_key is None
                or self._trial_seeds_by_candidate_key is None
                or self._promotion_state_by_candidate_key is None
                or self._latest_promotion_by_candidate_key is None
        ):
            return
        key = str(record.get("candidate_key", ""))
        if not key:
            return
        record_type = str(record.get("record_type", ""))
        attempted, status = self._promotion_state_by_candidate_key.get(
            key, (False, ""),
        )
        if record_type == "candidate_promotion_status_v1":
            metadata = record.get("promotion_metadata")
            self._latest_promotion_by_candidate_key[key] = (
                str(record.get("promotion_status", status)),
                dict(metadata) if isinstance(metadata, Mapping) else {},
            )
            self._promotion_state_by_candidate_key[key] = (
                True, str(record.get("promotion_status", status)),
            )
            return
        if record_type != "candidate_trial_group_v1":
            return
        trials = self._trial_group_from_record(record)
        if trials is None:
            return
        seed_index = self._trial_seeds_by_candidate_key.setdefault(key, set())
        for seed in trials.seeds:
            if int(seed) in seed_index:
                raise ValueError("stored candidate trial evidence contains duplicate seeds")
            seed_index.add(int(seed))
        self._trial_offsets_by_candidate_key.setdefault(key, []).append(int(offset))
        metadata = record.get("trial_group_metadata")
        metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
        marker = str(metadata.get("promotion_marker", ""))
        metadata_status = str(metadata.get("promotion_status", ""))
        if marker in ("fresh_top_up", "promoted") or metadata_status:
            attempted = True
        if metadata_status == "promoted" or marker == "promoted":
            status = "promoted"
        self._promotion_state_by_candidate_key[key] = (attempted, status)

    def _ensure_trial_indices(self) -> None:
        if (
                self._trial_offsets_by_candidate_key is not None
                and self._trial_seeds_by_candidate_key is not None
                and self._promotion_state_by_candidate_key is not None
                and self._latest_promotion_by_candidate_key is not None
        ):
            return
        self._trial_offsets_by_candidate_key = {}
        self._trial_seeds_by_candidate_key = {}
        self._promotion_state_by_candidate_key = {}
        self._latest_promotion_by_candidate_key = {}
        for offset, record in self._iter_active_records():
            if record.get("record_type") not in (
                    "candidate_trial_group_v1", "candidate_promotion_status_v1",
            ):
                continue
            key = record.get("candidate_key")
            if not key:
                continue
            self._index_trial_record(record, offset=offset)

    def _trial_records_for_candidate_key(
            self,
            candidate_key_value: str,
            ) -> Iterator[Mapping[str, Any]]:
        self._ensure_trial_indices()
        offsets = self._trial_offsets_by_candidate_key.get(
            str(candidate_key_value), (),
        )
        if not offsets or not self.path.exists():
            return
        with self.path.open("rb") as handle:
            for offset in offsets:
                handle.seek(offset)
                record = self._decode_jsonl_row(
                    handle.readline(), path=self.path, offset=offset,
                )
                if record is not None:
                    yield record

    def append_trial_group(
            self,
            action_indices: Any,
            trials: TrialSeries,
            metadata: Mapping[str, Any],
            ) -> Dict[str, Any]:
        """Append one aligned raw robust-evidence group for a candidate."""
        if not isinstance(trials, TrialSeries):
            raise TypeError("trials must be a TrialSeries")
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        identity_context = metadata.get("identity_context")
        if not isinstance(identity_context, Mapping):
            raise ValueError("trial group metadata requires identity_context")
        if not trials.seeds or len(trials.seeds) != len(trials.loss):
            raise ValueError("robust trial evidence requires nonempty aligned seeds")
        if len(set(trials.seeds)) != len(trials.seeds):
            raise ValueError("duplicate trial seeds within trial group")

        normalized_action = normalize_action_indices(action_indices)
        wanted_key = candidate_key(normalized_action, identity_context)
        self._ensure_trial_indices()
        existing_seeds = self._trial_seeds_by_candidate_key.get(wanted_key, set())
        overlap = sorted(
            int(seed) for seed in trials.seeds if int(seed) in existing_seeds
        )
        metadata_payload = to_jsonable(dict(metadata), stringify_unknown=True)
        if overlap:
            if len(overlap) == len(trials.seeds):
                replay_values = tuple(
                    (
                        float(trials.loss[idx]),
                        float(trials.metric1[idx]),
                        float(trials.metric2[idx]),
                    )
                    for idx in range(len(trials.seeds))
                )
                exact_match = next((
                    (record, stored)
                    for record in self._trial_records_for_candidate_key(wanted_key)
                    if (
                        (stored := self._trial_group_from_record(record)) is not None
                        and tuple(stored.seeds) == tuple(trials.seeds)
                    )
                ), None)
                if exact_match is not None:
                    exact_record, stored = exact_match
                    existing_values = tuple(
                        (
                            float(stored.loss[idx]),
                            float(stored.metric1[idx]),
                            float(stored.metric2[idx]),
                        )
                        for idx in range(len(stored.seeds))
                    )
                    existing_metadata = exact_record.get("trial_group_metadata")
                    if (
                            replay_values == existing_values
                            and existing_metadata == metadata_payload
                    ):
                        return {
                            "record_type": "candidate_trial_group_v1",
                            "candidate_key": wanted_key,
                            "action_indices": normalized_action,
                            "idempotent_replay": True,
                        }
                    if replay_values == existing_values:
                        raise ValueError(
                            "duplicate trial seeds replayed with different metadata"
                        )
            raise ValueError(f"duplicate trial seeds for candidate identity: {overlap}")

        return self.append({
            "record_type": "candidate_trial_group_v1",
            "action_indices": normalized_action,
            "effective_action_indices": normalized_action,
            "identity_context": dict(identity_context),
            "fidelity": str(metadata.get("fidelity", "F1")),
            "valid": bool(metadata.get("valid", True)),
            "trial_group": {
                "loss": [float(value) for value in trials.loss],
                "metric1": [float(value) for value in trials.metric1],
                "metric2": [float(value) for value in trials.metric2],
                "seeds": [int(value) for value in trials.seeds],
            },
            "trial_group_metadata": metadata_payload,
        })

    def trial_evidence_for_action(
            self,
            action_indices: Any,
            identity_context: Mapping[str, Any],
            *,
            max_trials: Optional[int] = None,
            ) -> Optional[CandidateTrialEvidence]:
        """Pool all raw groups matching canonical action plus run context."""
        if not isinstance(identity_context, Mapping):
            raise TypeError("identity_context must be a mapping")
        normalized_action = normalize_action_indices(action_indices)
        wanted_key = candidate_key(normalized_action, identity_context)
        trial_limit = None
        if max_trials is not None:
            if isinstance(max_trials, bool) or int(max_trials) <= 0:
                raise ValueError("max_trials must be a positive integer")
            trial_limit = int(max_trials)
        loss: List[float] = []
        metric1: List[float] = []
        metric2: List[float] = []
        seeds: List[int] = []
        groups: List[Mapping[str, Any]] = []
        promotion_attempted = False
        promotion_status = ""

        for record in self._trial_records_for_candidate_key(wanted_key):
            record_type = str(record.get("record_type", ""))
            if record_type != "candidate_trial_group_v1":
                continue
            trial_group = self._trial_group_from_record(record)
            if trial_group is None:
                continue
            take = len(trial_group.loss)
            if trial_limit is not None:
                take = min(take, trial_limit - len(loss))
            if take <= 0:
                break
            loss.extend(trial_group.loss[:take])
            metric1.extend(trial_group.metric1[:take])
            metric2.extend(trial_group.metric2[:take])
            seeds.extend(trial_group.seeds[:take])
            group_metadata = record.get("trial_group_metadata")
            metadata_dict = dict(group_metadata) if isinstance(group_metadata, Mapping) else {}
            groups.append(MappingProxyType(metadata_dict))
            if trial_limit is not None and len(loss) >= trial_limit:
                break

        if not loss:
            return None
        if len(set(seeds)) != len(seeds):
            raise ValueError("stored candidate trial evidence contains duplicate seeds")
        self._ensure_trial_indices()
        promotion_attempted, promotion_status = (
            self._promotion_state_by_candidate_key.get(wanted_key, (False, ""))
        )
        return CandidateTrialEvidence(
            candidate_key=wanted_key,
            action_indices=tuple(normalized_action),
            trials=TrialSeries(
                loss=loss, metric1=metric1, metric2=metric2, seeds=seeds,
            ),
            groups=tuple(groups),
            promotion_attempted=promotion_attempted,
            promotion_status=promotion_status,
        )

    def trial_count_for_action(
            self,
            action_indices: Any,
            identity_context: Mapping[str, Any],
            ) -> int:
        """Return the complete raw-trial count without materializing evidence."""
        if not isinstance(identity_context, Mapping):
            raise TypeError("identity_context must be a mapping")
        normalized_action = normalize_action_indices(action_indices)
        wanted_key = candidate_key(normalized_action, identity_context)
        self._ensure_trial_indices()
        return len(self._trial_seeds_by_candidate_key.get(wanted_key, {}))

    def latest_promotion_status_for_action(
            self,
            action_indices: Any,
            identity_context: Mapping[str, Any],
            ) -> Tuple[str, Dict[str, Any]]:
        """Return the latest explicit promotion status from the candidate index."""
        if not isinstance(identity_context, Mapping):
            raise TypeError("identity_context must be a mapping")
        normalized_action = normalize_action_indices(action_indices)
        wanted_key = candidate_key(normalized_action, identity_context)
        self._ensure_trial_indices()
        status, metadata = self._latest_promotion_by_candidate_key.get(
            wanted_key, ("", {}),
        )
        return status, dict(metadata)

    def best_for_action(
            self,
            action_indices: Any,
            *,
            identity_context: Mapping[str, Any] | None = None,
            effective_action_indices: Any | None = None,
            registry_or_description: Any = None,
            baseline_action: Any = None,
            allow_legacy: bool = False,
            ) -> Optional[Dict[str, Any]]:
        wanted = action_hash(action_indices)
        records = self.read_all()
        if identity_context is not None:
            effective_indices = effective_action_indices
            if effective_indices is None and registry_or_description is not None:
                effective_indices = effective_action_vector(
                    action_indices,
                    registry_or_description=registry_or_description,
                    baseline_action=baseline_action,
                )
            wanted_key = candidate_key(
                action_indices,
                identity_context,
                effective_action_indices=effective_indices,
            )
            matches = [
                r for r in records
                if r.get("candidate_key") == wanted_key
            ]
            if allow_legacy:
                matches.extend([
                    r for r in records
                    if r.get("action_hash") == wanted and bool(r.get("legacy_record", False))
                ])
        else:
            matches = [r for r in records if r.get("action_hash") == wanted]
        if not matches:
            return None
        return sorted(
            matches,
            key=lambda r: (-fidelity_rank(str(r.get("fidelity", "F0"))), candidate_rank_key(r)),
        )[0]

    def should_evaluate(
            self,
            action_indices: Any,
            fidelity: str,
            *,
            identity_context: Mapping[str, Any] | None = None,
            effective_action_indices: Any | None = None,
            registry_or_description: Any = None,
            baseline_action: Any = None,
            allow_legacy: bool = False,
            ) -> bool:
        existing = self.best_for_action(
            action_indices,
            identity_context=identity_context,
            effective_action_indices=effective_action_indices,
            registry_or_description=registry_or_description,
            baseline_action=baseline_action,
            allow_legacy=allow_legacy if identity_context is None else False,
        )
        if existing is None:
            return True
        return fidelity_rank(str(existing.get("fidelity", "F0"))) < fidelity_rank(fidelity)
