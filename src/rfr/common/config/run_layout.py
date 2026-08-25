"""Deterministic working and record paths for independent search stages."""

from __future__ import annotations

import datetime
import json
import os
import shutil
from typing import Dict, Iterable, List, Optional, Tuple

from rfr.preparation.data.protocol import validate_supported_profile
from rfr.common.config.paths import (
    COMPLETED_MARKER_FILENAME,
    RECORD_SUBDIR,
    RL_RESULTS_ROOT,
    STAGE1_SUBDIR,
    STAGE2_SUBDIR,
)


DEFAULT_ROOT: str = RL_RESULTS_ROOT


CONSTRAINT_KEYS: Tuple[str, ...] = (
    "stage1_accuracy_tolerance",
    "stage2_limit_tolerance",
    "stage2_stability_tolerance",
)


def normalize_stage(stage) -> int:
    """Normalize common Stage 1 and Stage 2 labels to ``1`` or ``2``."""
    s = str(stage).strip().lower()
    if s in ("1", "stage1", "stage1-only", "stage1_only"):
        return 1
    if s in ("2", "stage2", "stage2-only", "stage2_only"):
        return 2
    raise ValueError(f"未知 stage：{stage!r}（只支持 stage1 / stage2）")


def stage_subdir(stage) -> str:
    return STAGE1_SUBDIR if normalize_stage(stage) == 1 else STAGE2_SUBDIR


def combo_name(model_type: str, dataset: str) -> str:
    """Return the stable human-readable model/task directory name."""
    model_family = str(model_type).strip().lower()
    ds = str(dataset).strip().lower()
    validate_supported_profile(model_family, ds)
    mt = model_family.replace("-", " ")

    return " ".join(f"{mt} {ds}".split())


def stage_root(stage, root: str = DEFAULT_ROOT) -> str:
    """``<root>/stage{1,2}``。"""
    return os.path.join(root, stage_subdir(stage))


def stage_working_dir(stage, model_type: str, dataset: str, root: str = DEFAULT_ROOT) -> str:
    """Return the flat working directory ``<root>/stage{1,2}/{combo}``."""
    return os.path.join(stage_root(stage, root), combo_name(model_type, dataset))


def stage_record_root(stage, root: str = DEFAULT_ROOT) -> str:
    """``<root>/stage{1,2}/record``。"""
    return os.path.join(stage_root(stage, root), RECORD_SUBDIR)


def _today_yyyymmdd() -> str:
    return datetime.datetime.now().strftime("%Y%m%d")


def _coerce_date(timestamp: Optional[str]) -> str:
    """Normalize a date or timestamp to ``YYYYMMDD``; use today for ``None``."""
    if timestamp is None or timestamp == "":
        return _today_yyyymmdd()
    s = str(timestamp)
    head = s[:8]
    if head.isdigit() and len(head) == 8:
        return head
    return s


def run_id(model_type: str, dataset: str, n: int, timestamp: Optional[str] = None) -> str:
    """``"bert base rte 1 20260530"``。"""
    return f"{combo_name(model_type, dataset)} {int(n)} {_coerce_date(timestamp)}"


def _run_number_for_combo(name: str, combo: str) -> Optional[int]:
    """Parse a combo's run number from a record name, or return ``None``.

    A literal ``combo + " "`` prefix avoids ambiguity when the combo itself
    contains digits, as in ``sst2``. The suffix must be ``"{N} {YYYYMMDD}"``.
    """
    prefix = combo + " "
    if not name.startswith(prefix):
        return None
    rest = name[len(prefix):].strip()
    parts = rest.split(" ")
    if len(parts) != 2:
        return None
    n_str, date_str = parts
    if not (n_str.isdigit() and date_str.isdigit() and len(date_str) == 8):
        return None
    return int(n_str)


def existing_run_numbers(stage, model_type: str, dataset: str, root: str = DEFAULT_ROOT) -> List[int]:
    combo = combo_name(model_type, dataset)
    rroot = stage_record_root(stage, root)
    if not os.path.isdir(rroot):
        return []
    out: List[int] = []
    for name in os.listdir(rroot):
        n = _run_number_for_combo(name, combo)
        if n is not None:
            out.append(n)
    return sorted(out)


def next_run_number(stage, model_type: str, dataset: str, root: str = DEFAULT_ROOT) -> int:
    nums = existing_run_numbers(stage, model_type, dataset, root)
    return (max(nums) + 1) if nums else 1


def latest_record_dir_in_root(
    record_root: str, combo: str, run_id_name: Optional[str] = None
) -> Optional[str]:
    """Find the latest combo record, or an exact ``run_id_name``, under a root.

    The explicit root and combo keep this lookup independent of Stage 2 output
    path parsing. Returns ``None`` when no matching record exists.
    """
    if run_id_name:
        cand = os.path.join(record_root, run_id_name)
        return cand if os.path.isdir(cand) else None
    if not os.path.isdir(record_root):
        return None
    best_n = 0
    best: Optional[str] = None
    for name in sorted(os.listdir(record_root)):
        n = _run_number_for_combo(name, combo)
        if n is not None and n >= best_n:
            best_n = n
            best = os.path.join(record_root, name)
    return best


def find_record_dir(
    stage,
    model_type: str,
    dataset: str,
    n: Optional[int] = None,
    run_id_name: Optional[str] = None,
    root: str = DEFAULT_ROOT,
) -> Optional[str]:
    """Locate an existing record directory.

    ``run_id_name`` selects an exact record. Otherwise ``n`` selects the
    combo's numbered run, defaulting to the largest available number. Returns
    ``None`` when no match exists.
    """
    rroot = stage_record_root(stage, root)
    if not os.path.isdir(rroot):
        return None
    if run_id_name:
        cand = os.path.join(rroot, run_id_name)
        return cand if os.path.isdir(cand) else None
    combo = combo_name(model_type, dataset)
    target = n if n is not None else (max(existing_run_numbers(stage, model_type, dataset, root) or [0]) or None)
    if not target:
        return None
    for name in sorted(os.listdir(rroot)):
        if _run_number_for_combo(name, combo) == target:
            return os.path.join(rroot, name)
    return None


def completed_marker_path(working_dir: str) -> str:
    return os.path.join(working_dir, COMPLETED_MARKER_FILENAME)


def is_completed(working_dir: str) -> bool:
    return os.path.isfile(completed_marker_path(working_dir))


def mark_completed(working_dir: str, info: Optional[Dict] = None) -> str:
    os.makedirs(working_dir, exist_ok=True)
    p = completed_marker_path(working_dir)
    payload = {"completed": True, "marked_at": datetime.datetime.now().isoformat()}
    if info:
        payload.update(info)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return p


def clear_completed(working_dir: str) -> None:
    p = completed_marker_path(working_dir)
    if os.path.isfile(p):
        os.remove(p)


def make_record_dir(
    stage,
    model_type: str,
    dataset: str,
    n: Optional[int] = None,
    timestamp: Optional[str] = None,
    root: str = DEFAULT_ROOT,
) -> Tuple[str, str, int]:
    """Create a record directory and return ``(path, run_id_name, n)``.

    When ``n`` is omitted, the next available run number is used.
    """
    if n is None:
        n = next_run_number(stage, model_type, dataset, root)
    rid = run_id(model_type, dataset, n, timestamp)
    rdir = os.path.join(stage_record_root(stage, root), rid)
    os.makedirs(rdir, exist_ok=True)
    return rdir, rid, n


def copy_into_record(record_dir: str, src_paths: Iterable[str]) -> List[str]:
    """Copy existing files into a record directory and return their destinations."""
    copied: List[str] = []
    os.makedirs(record_dir, exist_ok=True)
    for src in src_paths:
        if src and os.path.isfile(src):
            dst = os.path.join(record_dir, os.path.basename(src))
            shutil.copy2(src, dst)
            copied.append(dst)
    return copied


def write_json_into_record(record_dir: str, filename: str, payload) -> str:
    os.makedirs(record_dir, exist_ok=True)
    dst = os.path.join(record_dir, filename)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return dst


def next_run_number_in_root(record_root: str, combo: str) -> int:
    """Return the next run number for a combo under an explicit record root."""
    if not os.path.isdir(record_root):
        return 1
    nums: List[int] = []
    for name in os.listdir(record_root):
        n = _run_number_for_combo(name, combo)
        if n is not None:
            nums.append(n)
    return (max(nums) + 1) if nums else 1


def snapshot_decoupled_record(
    stage,
    combo: str,
    working_dir: str,
    *,
    final_config: Dict,
    final_eval: Dict,
    metadata: Optional[Dict] = None,
    curve_paths: Iterable[str] = (),
    report_md: str = "",
    root: str = DEFAULT_ROOT,
    timestamp: Optional[str] = None,
) -> Tuple[str, str, int]:
    """Archive a decoupled run and mark its working directory complete.

    The archive contains the final configuration, evaluation, metadata,
    report, and available curves under ``stage{1,2}/record/{combo N date}``.
    Returns ``(record_dir, run_id, n)``. Archiving is best effort so a failed
    copy does not terminate training.
    """
    rec_root = stage_record_root(stage, root)
    n = next_run_number_in_root(rec_root, combo)
    rid = f"{combo} {n} {_coerce_date(timestamp)}"
    rdir = os.path.join(rec_root, rid)
    os.makedirs(rdir, exist_ok=True)
    write_json_into_record(rdir, "final_config.json", final_config)
    write_json_into_record(rdir, "final_eval.json", final_eval)
    if metadata is not None:
        write_json_into_record(rdir, "metadata.json", metadata)
    if report_md:
        with open(os.path.join(rdir, "report.md"), "w", encoding="utf-8") as f:
            f.write(report_md)
    copy_into_record(rdir, curve_paths)
    mark_completed(
        working_dir,
        {"record_dir": rdir, "run_id": rid, "stage": normalize_stage(stage), "run_number": n},
    )
    return rdir, rid, n


def _num_eq(a, b) -> bool:
    """Compare numeric values while tolerating string representations."""
    if a is None or b is None:
        return a is b
    try:
        return abs(float(a) - float(b)) <= 1e-9
    except (TypeError, ValueError):
        return str(a) == str(b)


def constraints_from(meta: Dict) -> Dict[str, object]:
    return {k: meta[k] for k in CONSTRAINT_KEYS if k in meta and meta[k] is not None}


def constraint_mismatch(persisted_meta: Dict, current: Dict) -> Optional[str]:
    """Describe resume-constraint mismatches, or return ``None`` when aligned.

    Only keys present in both mappings are compared.
    """
    msgs: List[str] = []
    for k in CONSTRAINT_KEYS:
        pv = persisted_meta.get(k)
        cv = current.get(k)
        if pv is None or cv is None:
            continue
        if not _num_eq(pv, cv):
            msgs.append(f"{k}: 已持久化={pv} 当前={cv}")
    return "; ".join(msgs) if msgs else None
