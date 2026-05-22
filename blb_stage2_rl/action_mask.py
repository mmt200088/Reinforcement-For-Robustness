"""Action-mask helpers for BLB Stage-2 Trust-0/Phase-1 diagnostics.

Two related-but-distinct mask abstractions live here:

  1. Per-slot allow-list (legacy ``build_action_mask`` / ``ensure_action_allowed``)
     — static, pre-computed list of which level indices a slot may sample.
     Used by Paean's "near_baseline" / "baseline_only" sweeps.

  2. ``ForbiddenActionMask`` (added 2026-05-17) — dynamic, per-(layer, block)
     blacklist of *full step-action tuples* that triggered invalid_chain at
     runtime. Used by the sequential RL loop to rejection-sample around
     known-bad tuples so the policy never sees them again. The two abstractions
     are orthogonal: a slot allow-list constrains the action *space*; the
     forbidden-action set constrains specific *tuples* within that space.

  3. ``StaticInvalidLevelMask`` (added 2026-05-22) — pre-RL projection from
     a baseline-prefix optimizer feasibility scan back to per-slot level masks.
     This is intentionally more aggressive than the runtime empirical mask:
     a level that is invalid in the baseline-prefix one-slot scan is hidden
     before PPO starts, while preserving protected baseline levels.

  4. ``EmpiricalInvalidLevelMask`` (added 2026-05-22) — guarded projection from
     repeated invalid tuples back to per-slot level masks. It only closes a
     level after enough invalid observations and no contradictory valid
     evidence, and always preserves protected baseline levels.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np

from .action_space import (
    K_LEVELS,
    action_dims_for_config,
    describe_action_vector,
    load_max_sfs,
    make_all_max_action_vector,
)


def _degree_vector(raw: Any, *, num_layers: int, default: int) -> Sequence[int]:
    if raw is None:
        return [int(default)] * int(num_layers)
    if isinstance(raw, (int, np.integer)):
        return [int(raw)] * int(num_layers)
    values = list(raw)
    if len(values) == 1:
        return [int(values[0])] * int(num_layers)
    if len(values) != int(num_layers):
        raise ValueError(f"degree vector length {len(values)} must be 1 or {num_layers}")
    return [int(v) for v in values]


def _baseline_only(dim: int, baseline_idx: int) -> np.ndarray:
    mask = np.zeros(int(dim), dtype=bool)
    mask[int(baseline_idx)] = True
    return mask


def _stable_mask_payload(mask: Sequence[Sequence[bool]]) -> str:
    rows = [[bool(x) for x in row] for row in mask]
    return json.dumps(rows, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def action_mask_hash(mask: Sequence[Sequence[bool]] | None) -> str:
    if mask is None:
        return ""
    return hashlib.sha256(_stable_mask_payload(mask).encode("utf-8")).hexdigest()


def build_baseline_action_bias(
        *,
        action_dims: Sequence[int],
        baseline_action: Sequence[int],
        baseline_logit_bonus: float,
        ) -> List[np.ndarray] | None:
    bonus = float(baseline_logit_bonus)
    if bonus == 0.0:
        return None
    if len(action_dims) != len(baseline_action):
        raise ValueError(
            f"action_dims length {len(action_dims)} != baseline length {len(baseline_action)}"
        )
    out: List[np.ndarray] = []
    for slot_idx, (dim, baseline_idx) in enumerate(zip(action_dims, baseline_action)):
        dim = int(dim)
        baseline_idx = int(baseline_idx)
        if baseline_idx < 0 or baseline_idx >= dim:
            raise ValueError(f"baseline action slot {slot_idx} index {baseline_idx} out of width {dim}")
        row = np.zeros(dim, dtype=np.float32)
        row[baseline_idx] = bonus
        out.append(row)
    return out


def ensure_action_allowed(
        action: Sequence[int],
        action_mask: Sequence[Sequence[bool]] | None,
        *,
        label: str = "action",
        ) -> None:
    if action_mask is None:
        return
    if len(action) != len(action_mask):
        raise ValueError(f"{label} width {len(action)} != mask width {len(action_mask)}")
    for slot_idx, (idx, slot_mask) in enumerate(zip(action, action_mask)):
        arr = np.asarray(slot_mask, dtype=bool).reshape(-1)
        idx = int(idx)
        if idx < 0 or idx >= arr.size or not bool(arr[idx]):
            raise ValueError(f"{label} slot {slot_idx} index {idx} is not allowed by action_mask")


def action_allowed(action: Sequence[int], action_mask: Sequence[Sequence[bool]] | None) -> bool:
    try:
        ensure_action_allowed(action, action_mask)
    except ValueError:
        return False
    return True


def load_action_mask_file(
        path: str | Path,
        *,
        expected_width: int,
        baseline_action: Sequence[int],
        action_dims: Sequence[int] | None = None,
        slot_records: Sequence[Mapping[str, Any]] | None = None,
        ) -> Tuple[List[np.ndarray], Mapping[str, Any]]:
    mask_path = Path(path)
    payload = json.loads(mask_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"action mask file must contain a JSON object: {mask_path}")
    slots = payload.get("slots")
    if not isinstance(slots, list):
        raise ValueError("action mask file requires a list field: slots")
    if int(payload.get("action_width", len(slots))) != int(expected_width):
        raise ValueError(
            f"action mask action_width {payload.get('action_width')} != expected {int(expected_width)}"
        )
    if len(slots) != int(expected_width):
        raise ValueError(f"action mask slots length {len(slots)} != expected {int(expected_width)}")
    if len(baseline_action) != int(expected_width):
        raise ValueError(
            f"baseline action length {len(baseline_action)} != expected {int(expected_width)}"
        )
    out: List[np.ndarray] = []
    if action_dims is not None and len(action_dims) != int(expected_width):
        raise ValueError(f"action_dims length {len(action_dims)} != expected {int(expected_width)}")
    records_by_index = {}
    if slot_records is not None:
        for record in slot_records:
            if isinstance(record, Mapping) and "global_index" in record:
                records_by_index[int(record["global_index"])] = record

    for slot_idx, slot in enumerate(slots):
        if not isinstance(slot, Mapping):
            raise ValueError(f"action mask slot {slot_idx} must be an object")
        baseline_idx = int(slot.get("baseline_index", baseline_action[slot_idx]))
        if baseline_idx != int(baseline_action[slot_idx]):
            raise ValueError(
                f"action mask slot {slot_idx} baseline {baseline_idx} != current baseline {int(baseline_action[slot_idx])}"
            )
        allowed = [int(x) for x in slot.get("allowed_indices", [])]
        if not allowed:
            raise ValueError(f"action mask slot {slot_idx} has no allowed_indices")
        width = int(action_dims[slot_idx]) if action_dims is not None else max(max(allowed), baseline_idx) + 1
        row = np.zeros(width, dtype=bool)
        for idx in allowed:
            if idx < 0 or idx >= width:
                raise ValueError(f"action mask slot {slot_idx} index {idx} out of width {width}")
            row[idx] = True
        if baseline_idx >= row.size or not bool(row[baseline_idx]):
            raise ValueError(f"action mask slot {slot_idx} does not allow baseline index {baseline_idx}")
        record = records_by_index.get(int(slot_idx))
        if record is not None and not bool(record.get("effective", record.get("is_effective", True))):
            non_baseline_allowed = [
                int(idx) for idx in np.flatnonzero(row).tolist()
                if int(idx) != int(baseline_idx)
            ]
            if non_baseline_allowed:
                raise ValueError(
                    "action mask opens ineffective slot "
                    f"{slot_idx} to non-baseline indices {non_baseline_allowed}; "
                    "ineffective compatibility slots must be baseline-only"
                )
        out.append(row)
    return out, payload


def _near_baseline(dim: int, baseline_idx: int, *, kind: str) -> np.ndarray:
    mask = np.zeros(int(dim), dtype=bool)
    if str(kind) == "K":
        top_k_values = sorted([int(v) for v in K_LEVELS], reverse=True)[:3]
        for idx, value in enumerate(K_LEVELS):
            if int(value) in top_k_values and idx < int(dim):
                mask[int(idx)] = True
    else:
        lo = max(0, int(baseline_idx) - 1)
        for idx in range(lo, int(baseline_idx) + 1):
            if idx < int(dim):
                mask[int(idx)] = True
    mask[int(baseline_idx)] = True
    return mask


def build_action_mask(
        *,
        num_layers: int,
        mode: str,
        gelu_degree: Any = 4,
        attn_degree: Any = 4,
        profile: str = "mrpc",
        baseline_action: Sequence[int] | None = None,
        max_sfs: Any = None,
        ) -> List[np.ndarray]:
    """Return one boolean mask per action slot without changing action length."""
    dims = action_dims_for_config(int(num_layers))
    baseline = (
        np.asarray(baseline_action, dtype=int).reshape(-1)
        if baseline_action is not None
        else make_all_max_action_vector(int(num_layers))
    )
    if baseline.shape[0] != len(dims):
        raise ValueError(f"baseline action length {baseline.shape[0]} != expected {len(dims)}")
    gelu = _degree_vector(gelu_degree, num_layers=int(num_layers), default=4)
    attn = _degree_vector(attn_degree, num_layers=int(num_layers), default=4)
    desc = describe_action_vector(
        baseline,
        max_sfs=max_sfs if max_sfs is not None else load_max_sfs(profile),
        num_layers=int(num_layers),
        gelu_degree=gelu,
        attn_degree=attn,
        profile=str(profile),
    )
    records = sorted(desc["records"], key=lambda r: int(r["global_index"]))
    if len(records) != len(dims):
        raise RuntimeError(f"action registry length {len(records)} != dims length {len(dims)}")

    out: List[np.ndarray] = []
    mode_key = str(mode).strip().lower().replace("-", "_")
    for dim, idx, record in zip(dims, baseline.tolist(), records):
        if mode_key == "baseline_only":
            out.append(_baseline_only(dim, idx))
            continue
        if mode_key == "near_baseline":
            if not bool(record.get("effective", True)):
                out.append(_baseline_only(dim, idx))
            else:
                out.append(_near_baseline(dim, idx, kind=str(record.get("kind", ""))))
            continue
        raise ValueError(f"unknown action-mask mode: {mode}")
    return out


# ===========================================================================
# Runtime invalid-action blacklist (added 2026-05-17)
# ===========================================================================
# Each block at each layer accumulates its own forbidden set. Across episodes
# the policy can re-sample any non-forbidden action freely; a sample that
# matches a prior failure is dropped before reaching the optimizer.
_ForbiddenKey = Tuple[int, int]            # (layer_idx, block_idx)
_ForbiddenActionTuple = Tuple[int, ...]    # length = len(spec.slot_dims)


@dataclass
class ForbiddenActionMask:
    """Mutable per-(layer, block) blacklist of failed per-step action tuples.

    Use :meth:`add` after :meth:`is_forbidden` was False but env.evaluate_step
    came back invalid. Use :meth:`is_forbidden` to gate every fresh sample.

    Design intent (user's spec, 2026-05-17):

      > 选取invalid chain的这个动作根本不进入模型，直接完全从动作空间排除。
      > 这样修改之后，我希望达到一个效果，就是模型只能看到所有valid的动作，
      > 在这些valid的动作中学习找到最优。

    Kept intentionally cheap: a plain dict of ``(layer, block) → set[tuple]``.
    JSON-roundtrip-able via :meth:`to_json_records` / :meth:`from_json_records`
    so it can ride along inside the PPO checkpoint and survive resume.
    """

    by_key: Dict[_ForbiddenKey, Set[_ForbiddenActionTuple]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------
    def add(self, layer_idx: int, block_idx: int, action_tuple: Sequence[int]) -> bool:
        """Add an action tuple to the (layer, block) forbidden set.

        Returns True iff the tuple was newly added.
        """
        key = (int(layer_idx), int(block_idx))
        tup = tuple(int(x) for x in action_tuple)
        bucket = self.by_key.setdefault(key, set())
        if tup in bucket:
            return False
        bucket.add(tup)
        return True

    def is_forbidden(self, layer_idx: int, block_idx: int, action_tuple: Sequence[int]) -> bool:
        bucket = self.by_key.get((int(layer_idx), int(block_idx)))
        if not bucket:
            return False
        return tuple(int(x) for x in action_tuple) in bucket

    def count(self, layer_idx: int, block_idx: int) -> int:
        return len(self.by_key.get((int(layer_idx), int(block_idx)), ()))

    def total(self) -> int:
        return sum(len(v) for v in self.by_key.values())

    # ------------------------------------------------------------------
    # Persistence (checkpoint-safe)
    # ------------------------------------------------------------------
    def to_json_records(self) -> List[Dict[str, object]]:
        """``[{"layer", "block", "count", "tuples"}, ...]`` — JSON-safe."""
        records: List[Dict[str, object]] = []
        for (li, bi), bucket in sorted(self.by_key.items()):
            records.append({
                "layer": int(li),
                "block": int(bi),
                "count": int(len(bucket)),
                "tuples": sorted(list(t) for t in bucket),
            })
        return records

    @classmethod
    def from_json_records(cls, records: Iterable[Mapping[str, object]]) -> "ForbiddenActionMask":
        out = cls()
        for r in records or ():
            li = int(r["layer"])         # type: ignore[arg-type]
            bi = int(r["block"])         # type: ignore[arg-type]
            for t in r.get("tuples", []) or []:                   # type: ignore[union-attr]
                out.add(li, bi, tuple(int(x) for x in t))
        return out

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def summary(self, top_n: int = 5) -> str:
        if not self.by_key:
            return "forbidden_action_mask=empty"
        rows = sorted(
            ((k, len(v)) for k, v in self.by_key.items() if v),
            key=lambda kv: -kv[1],
        )
        head = rows[: max(1, int(top_n))]
        body = "; ".join(f"L{li:02d}-B{bi}={n}" for (li, bi), n in head)
        rest = sum(n for (_, n) in rows[len(head):])
        return (
            f"forbidden_action_mask total={self.total()} "
            f"(top {len(head)}: {body}"
            + (f"; +{rest} in {len(rows) - len(head)} more)" if rest else ")")
        )


@dataclass
class StaticInvalidLevelMask:
    """Pre-RL per-slot level mask from optimizer feasibility scanning.

    Inspired by COINN-style pre-search pruning: shrink the candidate space
    before the global optimizer starts, using cheap local feasibility checks.
    Here the check is intentionally local: each candidate changes one slot on a
    baseline-prefix action and asks the real ``Rescale_optimizer`` whether that
    block remains chain-valid. Levels that fail are hidden from PPO sampling.

    This can discard a level that might have become valid under a different
    prefix. That is the requested tradeoff for this run: fewer invalid-chain
    retries and a smaller effective action space, while keeping baseline and
    current proposal levels protected.
    """

    by_key: Dict[_ForbiddenKey, Dict[Tuple[int, int], Dict[str, object]]] = field(default_factory=dict)

    def add_invalid(
            self,
            layer_idx: int,
            block_idx: int,
            slot_idx: int,
            level_idx: int,
            *,
            reason: str = "",
            config_name: str = "",
            ) -> bool:
        bucket = self.by_key.setdefault((int(layer_idx), int(block_idx)), {})
        key = (int(slot_idx), int(level_idx))
        if key in bucket:
            return False
        bucket[key] = {
            "reason": str(reason or ""),
            "config_name": str(config_name or ""),
        }
        return True

    def disabled_levels(self, layer_idx: int, block_idx: int) -> Set[Tuple[int, int]]:
        return set(self.by_key.get((int(layer_idx), int(block_idx)), {}).keys())

    def apply(
            self,
            layer_idx: int,
            block_idx: int,
            action_level_mask: Sequence[Sequence[bool]],
            *,
            protected_actions: Sequence[Sequence[int]] = (),
            ) -> np.ndarray:
        mask = np.asarray(action_level_mask, dtype=bool).copy()
        if mask.ndim != 2:
            raise ValueError("action_level_mask must be a 2-D boolean array")
        protected: Set[Tuple[int, int]] = set()
        for action in protected_actions or ():
            for slot_idx, level_idx in enumerate(action):
                protected.add((int(slot_idx), int(level_idx)))
        for slot_idx, level_idx in self.disabled_levels(layer_idx, block_idx):
            if (slot_idx, level_idx) in protected:
                continue
            if 0 <= slot_idx < mask.shape[0] and 0 <= level_idx < mask.shape[1]:
                mask[slot_idx, level_idx] = False
        for slot_idx, level_idx in protected:
            if 0 <= slot_idx < mask.shape[0] and 0 <= level_idx < mask.shape[1]:
                mask[slot_idx, level_idx] = True
        for slot_idx in range(mask.shape[0]):
            if not bool(mask[slot_idx].any()):
                for protected_slot, protected_level in protected:
                    if protected_slot == slot_idx and 0 <= protected_level < mask.shape[1]:
                        mask[slot_idx, protected_level] = True
                        break
        return mask

    def total_disabled(self) -> int:
        return sum(len(bucket) for bucket in self.by_key.values())

    def to_json_records(self) -> List[Dict[str, object]]:
        records: List[Dict[str, object]] = []
        for (li, bi), bucket in sorted(self.by_key.items()):
            cells = []
            for (slot_idx, level_idx), meta in sorted(bucket.items()):
                cells.append({
                    "slot": int(slot_idx),
                    "level": int(level_idx),
                    "reason": str(meta.get("reason", "") if isinstance(meta, Mapping) else ""),
                    "config_name": str(meta.get("config_name", "") if isinstance(meta, Mapping) else ""),
                })
            records.append({
                "layer": int(li),
                "block": int(bi),
                "cells": cells,
            })
        return records

    @classmethod
    def from_json_records(cls, records: Iterable[Mapping[str, object]]) -> "StaticInvalidLevelMask":
        out = cls()
        for row in records or ():
            li = int(row["layer"])        # type: ignore[arg-type]
            bi = int(row["block"])        # type: ignore[arg-type]
            for cell in row.get("cells", []) or []:              # type: ignore[union-attr]
                if not isinstance(cell, Mapping):
                    continue
                out.add_invalid(
                    li,
                    bi,
                    int(cell["slot"]),
                    int(cell["level"]),
                    reason=str(cell.get("reason", "") or ""),
                    config_name=str(cell.get("config_name", "") or ""),
                )
        return out

    def summary(self, top_n: int = 5) -> str:
        rows = sorted(
            ((k, len(v)) for k, v in self.by_key.items() if v),
            key=lambda kv: -kv[1],
        )
        if not rows:
            return "static_invalid_level_mask=empty"
        head = rows[: max(1, int(top_n))]
        body = "; ".join(f"L{li:02d}-B{bi}={n}" for (li, bi), n in head)
        rest = sum(n for (_, n) in rows[len(head):])
        return (
            f"static_invalid_level_mask disabled={self.total_disabled()} "
            f"(top {len(head)}: {body}"
            + (f"; +{rest} in {len(rows) - len(head)} more)" if rest else ")")
        )


@dataclass
class EmpiricalInvalidLevelMask:
    """Learn high-risk per-slot levels from repeated invalid-chain samples.

    ``ForbiddenActionMask`` removes exact tuples, but it still pays sampling
    and sometimes optimizer cost before a bad region is learned. This helper
    projects strong empirical evidence back onto the per-step level mask:
    if a specific ``(layer, block, slot, level)`` has repeatedly appeared in
    invalid tuples and has not appeared in any valid committed tuple, that
    level can be hidden from future sampling for the same layer/block/slot.

    The rule is deliberately conservative. It never masks a protected level
    (normally the static-skeleton baseline, plus the current frontier/base
    proposal if supplied), and it never leaves a slot with no legal level.
    """

    min_invalid_samples: int = 3
    max_valid_samples: int = 0
    min_invalid_rate: float = 0.80
    by_key: Dict[_ForbiddenKey, Dict[Tuple[int, int], Dict[str, int]]] = field(default_factory=dict)

    def record(
            self,
            layer_idx: int,
            block_idx: int,
            action_tuple: Sequence[int],
            *,
            valid: bool,
            ) -> None:
        bucket = self.by_key.setdefault((int(layer_idx), int(block_idx)), {})
        field = "valid" if bool(valid) else "invalid"
        for slot_idx, level_idx in enumerate(action_tuple):
            key = (int(slot_idx), int(level_idx))
            stats = bucket.setdefault(key, {"valid": 0, "invalid": 0})
            stats[field] = int(stats.get(field, 0)) + 1

    def record_valid(self, layer_idx: int, block_idx: int, action_tuple: Sequence[int]) -> None:
        self.record(layer_idx, block_idx, action_tuple, valid=True)

    def record_invalid(self, layer_idx: int, block_idx: int, action_tuple: Sequence[int]) -> None:
        self.record(layer_idx, block_idx, action_tuple, valid=False)

    def should_mask_level(self, layer_idx: int, block_idx: int, slot_idx: int, level_idx: int) -> bool:
        bucket = self.by_key.get((int(layer_idx), int(block_idx)), {})
        stats = bucket.get((int(slot_idx), int(level_idx)), {})
        invalid = int(stats.get("invalid", 0) or 0)
        valid = int(stats.get("valid", 0) or 0)
        total = invalid + valid
        if invalid < int(self.min_invalid_samples):
            return False
        if valid > int(self.max_valid_samples):
            return False
        rate = float(invalid) / float(max(1, total))
        return rate >= float(self.min_invalid_rate)

    def disabled_levels(self, layer_idx: int, block_idx: int) -> Set[Tuple[int, int]]:
        bucket = self.by_key.get((int(layer_idx), int(block_idx)), {})
        out: Set[Tuple[int, int]] = set()
        for slot_idx, level_idx in bucket:
            if self.should_mask_level(layer_idx, block_idx, slot_idx, level_idx):
                out.add((int(slot_idx), int(level_idx)))
        return out

    def apply(
            self,
            layer_idx: int,
            block_idx: int,
            action_level_mask: Sequence[Sequence[bool]],
            *,
            protected_actions: Sequence[Sequence[int]] = (),
            ) -> np.ndarray:
        mask = np.asarray(action_level_mask, dtype=bool).copy()
        if mask.ndim != 2:
            raise ValueError("action_level_mask must be a 2-D boolean array")
        protected: Set[Tuple[int, int]] = set()
        for action in protected_actions or ():
            for slot_idx, level_idx in enumerate(action):
                protected.add((int(slot_idx), int(level_idx)))
        for slot_idx, level_idx in self.disabled_levels(layer_idx, block_idx):
            if (slot_idx, level_idx) in protected:
                continue
            if 0 <= slot_idx < mask.shape[0] and 0 <= level_idx < mask.shape[1]:
                mask[slot_idx, level_idx] = False
        for slot_idx, level_idx in protected:
            if 0 <= slot_idx < mask.shape[0] and 0 <= level_idx < mask.shape[1]:
                mask[slot_idx, level_idx] = True
        for slot_idx in range(mask.shape[0]):
            if not bool(mask[slot_idx].any()):
                for protected_slot, protected_level in protected:
                    if protected_slot == slot_idx and 0 <= protected_level < mask.shape[1]:
                        mask[slot_idx, protected_level] = True
                        break
        return mask

    def total_disabled(self) -> int:
        return sum(len(self.disabled_levels(li, bi)) for li, bi in self.by_key)

    def to_json_records(self) -> List[Dict[str, object]]:
        records: List[Dict[str, object]] = []
        for (li, bi), bucket in sorted(self.by_key.items()):
            cells = []
            for (slot_idx, level_idx), stats in sorted(bucket.items()):
                cells.append({
                    "slot": int(slot_idx),
                    "level": int(level_idx),
                    "valid": int(stats.get("valid", 0) or 0),
                    "invalid": int(stats.get("invalid", 0) or 0),
                })
            records.append({
                "layer": int(li),
                "block": int(bi),
                "cells": cells,
            })
        return records

    @classmethod
    def from_json_records(cls, records: Iterable[Mapping[str, object]]) -> "EmpiricalInvalidLevelMask":
        out = cls()
        for row in records or ():
            li = int(row["layer"])        # type: ignore[arg-type]
            bi = int(row["block"])        # type: ignore[arg-type]
            bucket = out.by_key.setdefault((li, bi), {})
            for cell in row.get("cells", []) or []:              # type: ignore[union-attr]
                if not isinstance(cell, Mapping):
                    continue
                key = (int(cell["slot"]), int(cell["level"]))
                bucket[key] = {
                    "valid": int(cell.get("valid", 0) or 0),
                    "invalid": int(cell.get("invalid", 0) or 0),
                }
        return out

    def summary(self, top_n: int = 5) -> str:
        disabled = []
        for (li, bi), bucket in sorted(self.by_key.items()):
            count = sum(
                1 for slot_idx, level_idx in bucket
                if self.should_mask_level(li, bi, slot_idx, level_idx)
            )
            if count:
                disabled.append(((li, bi), count))
        if not disabled:
            return "empirical_invalid_level_mask=empty"
        disabled.sort(key=lambda kv: -kv[1])
        head = disabled[: max(1, int(top_n))]
        body = "; ".join(f"L{li:02d}-B{bi}={n}" for (li, bi), n in head)
        rest = sum(n for (_, n) in disabled[len(head):])
        return (
            f"empirical_invalid_level_mask disabled={self.total_disabled()} "
            f"(top {len(head)}: {body}"
            + (f"; +{rest} in {len(disabled) - len(head)} more)" if rest else ")")
        )
