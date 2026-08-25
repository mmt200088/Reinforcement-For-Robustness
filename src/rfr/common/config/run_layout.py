"""Deterministic working and record paths for independent search stages."""

from __future__ import annotations

import datetime
import json
import os
import shutil
from typing import Dict, Iterable, List, Optional, Tuple

from glue_data_protocol import validate_supported_profile
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
    """把 ``1`` / ``"1"`` / ``"stage1"`` / ``"stage1-only"`` 归一成 ``1`` 或 ``2``。"""
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
    """每个 combo 的扁平工作目录：``<root>/stage{1,2}/{combo}``。"""
    return os.path.join(stage_root(stage, root), combo_name(model_type, dataset))


def stage_record_root(stage, root: str = DEFAULT_ROOT) -> str:
    """``<root>/stage{1,2}/record``。"""
    return os.path.join(stage_root(stage, root), RECORD_SUBDIR)


def _today_yyyymmdd() -> str:
    return datetime.datetime.now().strftime("%Y%m%d")


def _coerce_date(timestamp: Optional[str]) -> str:
    """接受 ``None``（今天）/ ``"20260530"`` / ``"20260530_141500"`` -> ``"20260530"``。"""
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
    """从 record 条目名里解析该 combo 的运行序号；不匹配返回 ``None``。

    用 ``startswith(combo + " ")`` 而不是宽松正则，避免 combo 自身含数字（如
    ``sst2``）时的歧义。剩余部分必须是 ``"{N} {8位日期}"``。
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
    """在给定 record 根目录下，按 combo 找最大 N 的 record 目录（或精确 ``run_id_name``）。

    与 ``find_record_dir`` 类似，但用显式 ``record_root`` + ``combo`` 字符串（解耦
    Stage-2 从 ``run_output_dir`` 解析出 combo 时用）。找不到返回 ``None``。
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
    """定位一个已存在的 record 目录。

    - ``run_id_name`` 给定（如 ``"bert base mrpc 2 20260530"``）→ 精确匹配。
    - 否则按 ``n``（缺省取最大 N）匹配该 combo 序号为 ``n`` 的条目。
    找不到返回 ``None``。
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
    """建立（并返回）一个新的 record 目录。

    返回 ``(record_dir, run_id_name, n)``。``n`` 缺省时取 ``next_run_number``。
    """
    if n is None:
        n = next_run_number(stage, model_type, dataset, root)
    rid = run_id(model_type, dataset, n, timestamp)
    rdir = os.path.join(stage_record_root(stage, root), rid)
    os.makedirs(rdir, exist_ok=True)
    return rdir, rid, n


def copy_into_record(record_dir: str, src_paths: Iterable[str]) -> List[str]:
    """把存在的文件按 basename 拷进 record_dir；不存在的跳过。返回拷成功的目标路径。"""
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
    """combo 版的 ``next_run_number``：直接给 record 根目录 + combo 字符串。"""
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
    """完成时把一次解耦运行归档进 ``stage{1,2}/record/{combo N date}/`` 并打 COMPLETED。

    ``combo`` 直接给（解耦时由 ``run_output_dir`` 的 basename 得到）。写
    ``final_config.json`` / ``final_eval.json`` / ``metadata.json`` / ``report.md``，
    拷贝 ``curve_paths``，并在 ``working_dir`` 打 ``COMPLETED`` 标记。
    返回 ``(record_dir, run_id, n)``。best-effort：任一子步骤失败不应让训练崩溃，
    由调用方决定是否吞掉异常。
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
    """容忍 ``0.005`` vs ``"0.005"`` 这类数值/字符串格式差异。"""
    if a is None or b is None:
        return a is b
    try:
        return abs(float(a) - float(b)) <= 1e-9
    except (TypeError, ValueError):
        return str(a) == str(b)


def constraints_from(meta: Dict) -> Dict[str, object]:
    return {k: meta[k] for k in CONSTRAINT_KEYS if k in meta and meta[k] is not None}


def constraint_mismatch(persisted_meta: Dict, current: Dict) -> Optional[str]:
    """续训时校验约束。返回人类可读的不一致信息；一致返回 ``None``。

    只比较两边都给出的键；任一侧缺失则跳过。
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
