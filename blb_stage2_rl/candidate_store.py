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
    "F4": 2,
}
# Note: F2 / F3 were intermediate tiers in the original spec; deprecated and
# removed 2026-05-16. The active ladder is F0 (optimizer-only, no model
# forward) → F1 (small probe + few MC trials during training) → F4 (full
# validation_full final eval with real BLB install). Old JSONL records with
# ``fidelity="F2"`` / ``"F3"`` get rank ``-1`` from ``fidelity_rank`` and
# surface as legacy entries — they are not lost, just not promotable.


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


def raw_action_hash(action_indices: Any) -> str:
    return action_hash(action_indices)


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


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
    raw_hash = action_hash(action_indices)
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
                -max(0.0, _finite_float(record.get("terminal_cost_rank_score", 0.0), 0.0)),
                -_finite_float(record.get("terminal_fusion_gain", 0.0), 0.0),
                -_finite_float(record.get("terminal_k_gain", 0.0), 0.0),
                -_finite_float(record.get("terminal_bits_gain", 0.0), 0.0),
                -terminal_reward,
                -total_reward,
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
                payload = json.loads(text)
                payload.setdefault("legacy_record", is_legacy_record(payload))
                records.append(payload)
        return records

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
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
        return payload

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
                r for r in self.read_all()
                if r.get("candidate_key") == wanted_key
            ]
            if allow_legacy:
                matches.extend([
                    r for r in self.read_all()
                    if r.get("action_hash") == wanted and bool(r.get("legacy_record", False))
                ])
        else:
            matches = [r for r in self.read_all() if r.get("action_hash") == wanted]
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
