"""Action-mask helpers for BLB Stage-2 Trust-0/Phase-1 diagnostics."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, List, Mapping, Sequence, Tuple

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
