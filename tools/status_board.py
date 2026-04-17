"""status_board — RL / GA / General-RL / RL-vs-GA 任务总板

用法：

    # 终端打印
    python tools/status_board.py

    # 同时刷新 docs/STATUS.md 快照
    python tools/status_board.py --write-md

    # 只看某一类
    python tools/status_board.py --only rl
    python tools/status_board.py --only compare

    # 指定其他项目根（默认就是当前工作目录）
    python tools/status_board.py --root /some/other/project

扫描范围：
- ``rl_results/persistent/rl/{model}/{dataset}/{slug}/metadata.json``   (单任务 RL)
- ``rl_results/persistent/ga/{model}/{dataset}/{slug}/metadata.json``   (单任务 GA)
- ``rl_results/persistent/general-rl/{model}/{taskset}/{slug}/...``     (通用策略)
- ``rl_results/runs/compare/rl_vs_ga/{dataset}/{run_name}/meta/*.json`` (对比)
- ``experiment_results/*/``                                             (一次性实验)

本脚本**不修改**任何结果目录内容，只读。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _force_utf8_stdout() -> None:
    """Windows 默认 GBK 终端下，用 Unicode 边框 / ASCII 图标会炸；尝试切 UTF-8。"""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_force_utf8_stdout()


# 避免依赖整条训练 pipeline 的引入；直接用相对路径常量
RL_RESULTS_ROOT = "rl_results"
PERSISTENT_SUBDIR = "persistent"
RUNS_SUBDIR = "runs"
COMPARE_SUBDIR_REL = os.path.join("compare", "rl_vs_ga")
EXPERIMENT_RESULTS_ROOT = "experiment_results"

PERSISTENT_RL_BRANCH = "rl"
PERSISTENT_GA_BRANCH = "ga"
PERSISTENT_GENERAL_RL_BRANCH = "general-rl"

METADATA_FILENAME = "metadata.json"
COMPARE_FINAL_STATUS = "compare_final_status.json"
COMPARE_STATUS = "compare_status.json"
COMPARE_METADATA = "compare_metadata.json"

STAGE_KEYS = (
    "stage1_search",
    "stage1_final_eval",
    "stage2_search",
    "stage2_final_eval",
)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class RunRecord:
    algorithm: str
    model_type: str
    task: str
    slug: str
    path: Path
    stage_status: Dict[str, str] = field(default_factory=dict)
    last_updated: Optional[str] = None
    run_count: Optional[int] = None
    latest_pid: Optional[str] = None
    latest_run_dir: Optional[str] = None
    alive: Optional[bool] = None
    error: Optional[str] = None


@dataclass
class CompareRecord:
    dataset: str
    run_name: str
    path: Path
    mode: Optional[str] = None
    state_rl: Optional[str] = None
    state_ga: Optional[str] = None
    rl_stage1_ready: Optional[bool] = None
    rl_stage2_ready: Optional[bool] = None
    ga_stage1_ready: Optional[bool] = None
    ga_stage2_ready: Optional[bool] = None
    updated_at: Optional[str] = None
    stage1_report: Optional[str] = None
    stage2_report: Optional[str] = None
    error: Optional[str] = None


@dataclass
class ExperimentRecord:
    name: str
    path: Path
    has_run_log: bool
    last_mtime: Optional[str] = None


# ---------------------------------------------------------------------------
# 读取辅助
# ---------------------------------------------------------------------------


def _read_json(p: Path) -> Optional[dict]:
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _read_text_single(p: Path) -> Optional[str]:
    try:
        return p.read_text(encoding="utf-8").strip() or None
    except Exception:
        return None


def _find_first(base: Path, *patterns: str, recursive: bool = False) -> Optional[Path]:
    if not base.exists():
        return None
    for pattern in patterns:
        finder = base.rglob if recursive else base.glob
        for p in sorted(finder(pattern)):
            if p.is_file():
                return p
    return None


def _find_all(base: Path, pattern: str, recursive: bool = False) -> List[Path]:
    if not base.exists():
        return []
    finder = base.rglob if recursive else base.glob
    return sorted(p for p in finder(pattern) if p.is_file())


def _pid_alive(pid: Optional[str]) -> Optional[bool]:
    """检查 PID 是否还在运行。Windows / POSIX 分别处理；失败就返回 None。"""
    if not pid or not pid.strip().isdigit():
        return None
    p = int(pid)
    try:
        if os.name == "nt":
            # Windows：用 tasklist 查
            import ctypes
            PROCESS_QUERY_INFORMATION = 0x0400
            h = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_INFORMATION, 0, p)
            if not h:
                return False
            ctypes.windll.kernel32.CloseHandle(h)
            return True
        else:
            os.kill(p, 0)
            return True
    except (OSError, PermissionError):
        return False
    except Exception:
        return None


def _format_time(raw: Optional[str]) -> str:
    if not raw:
        return "-"
    # 允许多种 ISO / naive 格式，直接返回字符串即可
    return raw.split(".")[0].replace("T", " ")


def _format_float(v: Any, digits: int = 4) -> str:
    if not isinstance(v, (int, float)):
        return "-"
    return f"{float(v):.{digits}f}"


def _format_cost(v: Any) -> str:
    if not isinstance(v, (int, float)):
        return "-"
    return f"{float(v):.2f}"


def _format_speedup(v: Any) -> str:
    if not isinstance(v, (int, float)):
        return "-"
    return f"{float(v):.2f}x"


def _format_delta(v: Any) -> str:
    if not isinstance(v, (int, float)):
        return "-"
    return f"{float(v):+.4f}"


def _normalize_stage_status(status: Optional[str]) -> str:
    if not status:
        return "unknown"
    if status == "in_progress":
        return "running"
    return status


# ---------------------------------------------------------------------------
# 扫描：persistent RL / GA
# ---------------------------------------------------------------------------


def _scan_persistent_runs(
    root: Path,
    algorithm_branch: str,
    algorithm_label: str,
) -> List[RunRecord]:
    records: List[RunRecord] = []
    branch_root = root / RL_RESULTS_ROOT / PERSISTENT_SUBDIR / algorithm_branch
    if not branch_root.is_dir():
        return records

    for model_dir in sorted(branch_root.iterdir()):
        if not model_dir.is_dir():
            continue
        for task_dir in sorted(model_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            latest_pid = _read_text_single(task_dir / "LATEST_PID")
            latest_run_dir = _read_text_single(task_dir / "LATEST_RUN_DIR")

            slug_dirs = [d for d in task_dir.iterdir() if d.is_dir()]
            if not slug_dirs:
                continue
            for slug_dir in sorted(slug_dirs):
                meta_path = slug_dir / METADATA_FILENAME
                meta = _read_json(meta_path) or {}
                stage_status = {
                    str(k): _normalize_stage_status(v)
                    for k, v in dict(meta.get("stage_status") or {}).items()
                }
                for k in STAGE_KEYS:
                    stage_status.setdefault(k, "unknown")
                rec = RunRecord(
                    algorithm=algorithm_label,
                    model_type=str(meta.get("model_type") or model_dir.name),
                    task=str(meta.get("dataset") or task_dir.name),
                    slug=slug_dir.name,
                    path=slug_dir,
                    stage_status=stage_status,
                    last_updated=meta.get("last_updated_at") or meta.get("created_at"),
                    run_count=meta.get("run_count"),
                    latest_pid=latest_pid,
                    latest_run_dir=latest_run_dir,
                    alive=_pid_alive(latest_pid),
                    error=None if meta else f"无法读取 {meta_path.name}",
                )
                records.append(rec)
    return records


# ---------------------------------------------------------------------------
# 扫描：compare rl_vs_ga
# ---------------------------------------------------------------------------


def _scan_compare_runs(root: Path) -> List[CompareRecord]:
    records: List[CompareRecord] = []
    compare_root = root / RL_RESULTS_ROOT / RUNS_SUBDIR / COMPARE_SUBDIR_REL
    if not compare_root.is_dir():
        return records

    for dataset_dir in sorted(compare_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for run_dir in sorted(dataset_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            meta_dir = run_dir / "meta"
            reports_dir = run_dir / "reports"
            final = _read_json(meta_dir / COMPARE_FINAL_STATUS) or {}
            status = _read_json(meta_dir / COMPARE_STATUS) or {}
            chosen = final or status
            metadata = _read_json(meta_dir / COMPARE_METADATA) or {}
            if not (final or status or metadata):
                continue
            rl_side = chosen.get("rl", {}) if isinstance(chosen, dict) else {}
            ga_side = chosen.get("ga", {}) if isinstance(chosen, dict) else {}

            # 自动寻找 reports
            stage1_report = stage2_report = None
            if reports_dir.is_dir():
                for p in sorted(reports_dir.glob("stage1_compare_report*.md")):
                    stage1_report = p.name
                    break
                for p in sorted(reports_dir.glob("stage2_compare_report*.md")):
                    stage2_report = p.name
                    break

            rec = CompareRecord(
                dataset=str(metadata.get("dataset") or dataset_dir.name),
                run_name=run_dir.name,
                path=run_dir,
                mode=chosen.get("mode") if isinstance(chosen, dict) else None,
                state_rl=rl_side.get("state"),
                state_ga=ga_side.get("state"),
                rl_stage1_ready=rl_side.get("stage1_final_eval_ready"),
                rl_stage2_ready=rl_side.get("stage2_final_eval_ready"),
                ga_stage1_ready=ga_side.get("stage1_final_eval_ready"),
                ga_stage2_ready=ga_side.get("stage2_final_eval_ready"),
                updated_at=chosen.get("updated_at") if isinstance(chosen, dict) else None,
                stage1_report=stage1_report,
                stage2_report=stage2_report,
            )
            records.append(rec)
    return records


# ---------------------------------------------------------------------------
# 扫描：experiment_results
# ---------------------------------------------------------------------------


def _scan_experiments(root: Path) -> List[ExperimentRecord]:
    records: List[ExperimentRecord] = []
    exp_root = root / EXPERIMENT_RESULTS_ROOT
    if not exp_root.is_dir():
        return records
    for d in sorted(exp_root.iterdir()):
        if not d.is_dir():
            continue
        has_log = (d / "run.log").exists()
        try:
            mtime = datetime.fromtimestamp(d.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            mtime = None
        records.append(ExperimentRecord(name=d.name, path=d, has_run_log=has_log, last_mtime=mtime))
    return records


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------


_STAGE_ICON = {
    "completed": "✓",
    "not_started": "·",
    "skipped": "→",
    "running": "*",
    "in_progress": "*",
    "failed": "×",
    "unknown": "?",
}

_STAGE_LABEL = {
    "completed": "完成",
    "not_started": "未开始",
    "skipped": "跳过",
    "running": "进行中",
    "in_progress": "进行中",
    "failed": "失败",
    "unknown": "未知",
}

_PROCESS_LABEL = {
    "completed": "完成",
    "running": "进行中",
    "failed": "失败",
    "stopped": "已停止",
    "cancelled": "已取消",
    "unknown": "未知",
}

_RL_MEAN_SCORE_RE = re.compile(
    r"回合[^\d]*(?P<step>\d+).*?mean_score=(?P<score>-?\d+(?:\.\d+)?).*?cost=(?P<cost>-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_RL_SELECTION_SCORE_RE = re.compile(
    r"episode[^\d]*(?P<step>\d+).*?cost[^0-9]*(?P<cost>-?\d+(?:\.\d+)?).*?final_selection_score\)?=?(?P<score>-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_GA_INFLIGHT_RE = re.compile(
    r"\[Stage(?P<stage>\d)\]\[Gen (?P<current>\d+)/(?P<total>\d+)\].*?incumbent=(?P<score>-?\d+(?:\.\d+)?)\(cost=(?P<cost>-?\d+(?:\.\d+)?)\)",
    re.IGNORECASE,
)


def _stage_cell(status: str) -> str:
    status = _normalize_stage_status(status)
    icon = _STAGE_ICON.get(status, "?")
    return f"{icon} {status}"


def _stage_row(record: RunRecord) -> str:
    parts = [_stage_cell(record.stage_status.get(k, "unknown")) for k in STAGE_KEYS]
    return " | ".join(parts)


def _bool_cell(v: Optional[bool]) -> str:
    if v is True:
        return "✓"
    if v is False:
        return "×"
    return "?"


def _alive_cell(record: RunRecord) -> str:
    if record.latest_pid is None:
        return "-"
    if record.alive is True:
        return f"alive({record.latest_pid})"
    if record.alive is False:
        return f"dead({record.latest_pid})"
    return f"?({record.latest_pid})"


def _stage_label(status: Optional[str]) -> str:
    return _STAGE_LABEL.get(_normalize_stage_status(status), "未知")


def _process_label(status: Optional[str]) -> str:
    return _PROCESS_LABEL.get(status or "unknown", "未知")


def _bool_mark(v: Optional[bool]) -> str:
    if v is True:
        return "✓"
    if v is False:
        return "·"
    return "?"


def _run_title(record: RunRecord) -> str:
    base = f"{record.model_type} / {record.task}"
    if record.slug:
        return f"{base} [{_truncate(record.slug, 24)}]"
    return base


def _stage_hint_from_path(path: Path) -> str:
    name = path.as_posix().lower()
    return "S2" if "stage2" in name else "S1"


def _format_selected_summary(selected: Optional[dict]) -> Optional[str]:
    if not isinstance(selected, dict):
        return None
    parts: List[str] = []
    if selected.get("p") is not None:
        parts.append(f"主={_format_float(selected.get('p'))}")
    if selected.get("s") is not None:
        parts.append(f"次={_format_float(selected.get('s'))}")
    if selected.get("tot_c") is not None and not selected.get("show_cost_as_na"):
        parts.append(f"cost={_format_cost(selected.get('tot_c'))}")
    if selected.get("tot_spd") is not None:
        parts.append(_format_speedup(selected.get("tot_spd")))
    if selected.get("feasible") is False:
        parts.append("不可行")
    return "，".join(parts) if parts else None


def _summarize_eval_json(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    data = _read_json(path) or {}
    selected = data.get("selected") or data.get("selected_single")
    summary = _format_selected_summary(selected)
    if not summary:
        return None
    return f"{_stage_hint_from_path(path)} 终评 {summary}"


def _summarize_rl_search_log(path: Optional[Path]) -> Optional[str]:
    if path is None or not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None

    best: Optional[re.Match[str]] = None
    for line in text.splitlines():
        match = _RL_MEAN_SCORE_RE.search(line) or _RL_SELECTION_SCORE_RE.search(line)
        if match:
            best = match
    if not best:
        return None
    return (
        f"{_stage_hint_from_path(path)} 搜索 "
        f"score={_format_float(float(best.group('score')))}，"
        f"cost={_format_cost(float(best.group('cost')))}，"
        f"ep{best.group('step')}"
    )


def _summarize_ga_search_log(path: Optional[Path]) -> Optional[str]:
    if path is None or not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None

    latest: Optional[re.Match[str]] = None
    for line in text.splitlines():
        match = _GA_INFLIGHT_RE.search(line)
        if match:
            latest = match
    if not latest:
        return None

    stage = f"S{latest.group('stage')}"
    return (
        f"{stage} 搜索 "
        f"score={_format_float(float(latest.group('score')))}，"
        f"cost={_format_cost(float(latest.group('cost')))}，"
        f"gen{latest.group('current')}/{latest.group('total')}"
    )


def _stage_artifact_notes(record: RunRecord) -> List[str]:
    notes: List[str] = []
    stage1_eval = _find_first(record.path / "stage1_final_eval", "final_eval_results_*.json")
    stage2_eval = _find_first(record.path / "stage2_noise_final_eval", "noise_final_eval_results_*.json")
    stage1_log = record.path / "stage1" / ("ga_search_log.txt" if record.algorithm == "GA" else "pruning_search_log.txt")
    stage2_log = record.path / "stage2_noise" / ("ga_noise_search_log.txt" if record.algorithm == "GA" else "pruning_search_log.txt")

    if record.stage_status.get("stage1_final_eval") != "completed" and stage1_eval:
        notes.append("S1 已有终评结果")
    if record.stage_status.get("stage2_final_eval") != "completed" and stage2_eval:
        notes.append("S2 已有终评结果")
    if record.stage_status.get("stage1_search") in {"unknown", "not_started"} and stage1_log.exists():
        notes.append("S1 已有搜索日志")
    if record.stage_status.get("stage2_search") in {"unknown", "not_started"} and stage2_log.exists():
        notes.append("S2 已有搜索日志")
    return notes


def _run_progress_summary(record: RunRecord) -> str:
    summary = (
        f"S1 {_stage_label(record.stage_status.get('stage1_search'))}/{_stage_label(record.stage_status.get('stage1_final_eval'))}；"
        f"S2 {_stage_label(record.stage_status.get('stage2_search'))}/{_stage_label(record.stage_status.get('stage2_final_eval'))}"
    )
    notes = _stage_artifact_notes(record)
    if notes:
        summary += "；" + "，".join(notes)
    return summary


def _run_best_result_summary(record: RunRecord) -> str:
    stage2_eval = _summarize_eval_json(
        _find_first(record.path / "stage2_noise_final_eval", "noise_final_eval_results_*.json")
    )
    if stage2_eval:
        return stage2_eval

    if record.algorithm == "GA":
        stage2_search = _summarize_ga_search_log(record.path / "stage2_noise" / "ga_noise_search_log.txt")
    else:
        stage2_search = _summarize_rl_search_log(record.path / "stage2_noise" / "pruning_search_log.txt")
    if stage2_search:
        return stage2_search

    stage1_eval = _summarize_eval_json(
        _find_first(record.path / "stage1_final_eval", "final_eval_results_*.json")
    )
    if stage1_eval:
        return stage1_eval

    if record.algorithm == "GA":
        stage1_search = _summarize_ga_search_log(record.path / "stage1" / "ga_search_log.txt")
    else:
        stage1_search = _summarize_rl_search_log(record.path / "stage1" / "pruning_search_log.txt")
    if stage1_search:
        return stage1_search

    return "暂无可用结果"


def _compare_progress_summary(record: CompareRecord) -> str:
    return (
        f"RL {_process_label(record.state_rl)}，GA {_process_label(record.state_ga)}；"
        f"终评 S1 {_bool_mark(record.rl_stage1_ready)}/{_bool_mark(record.ga_stage1_ready)}，"
        f"S2 {_bool_mark(record.rl_stage2_ready)}/{_bool_mark(record.ga_stage2_ready)}"
    )


def _summarize_compare_stage(path: Optional[Path], stage_label: str) -> Optional[str]:
    data = _read_json(path) or {}
    sides = data.get("sides") or {}
    rl_summary = _format_selected_summary((sides.get("rl") or {}).get("selected"))
    ga_summary = _format_selected_summary((sides.get("ga") or {}).get("selected"))
    if not rl_summary and not ga_summary:
        return None
    return f"{stage_label} RL({rl_summary or '暂无'})；GA({ga_summary or '暂无'})"


def _compare_best_result_summary(record: CompareRecord) -> str:
    reports_dir = record.path / "reports"
    stage2 = _summarize_compare_stage(
        _find_first(reports_dir, "stage2_compare_summary*.json"),
        "S2",
    )
    if stage2:
        return stage2
    stage1 = _summarize_compare_stage(
        _find_first(reports_dir, "stage1_compare_summary*.json"),
        "S1",
    )
    if stage1:
        return stage1
    return "暂无可用结果"


def _experiment_progress_summary(record: ExperimentRecord) -> str:
    pid = (
        _read_text_single(record.path / "pid.txt")
        or _read_text_single(record.path / "run.pid")
        or _read_text_single(record.path / "LATEST_PID")
    )
    alive = _pid_alive(pid)
    if alive is True:
        return "进行中"
    if _find_first(record.path, "*.json", recursive=True) or _find_first(record.path, "*.png", recursive=True):
        return "已完成"
    if record.has_run_log:
        return "已有日志"
    return "仅目录"


def _extract_dataset_from_name(path: Path) -> str:
    match = re.search(r"_([A-Za-z0-9\-]+)\.json$", path.name)
    if match:
        return match.group(1)
    return path.stem


def _summarize_eval_collection(paths: List[Path]) -> Optional[str]:
    records: List[Tuple[int, float, float, str, dict]] = []
    for p in paths:
        data = _read_json(p) or {}
        selected = data.get("selected") or data.get("selected_single")
        if not isinstance(selected, dict):
            continue
        dataset = str(data.get("dataset") or _extract_dataset_from_name(p))
        feasible_rank = 1 if selected.get("feasible") is not False else 0
        primary = float(selected.get("p")) if isinstance(selected.get("p"), (int, float)) else float("-inf")
        speed = float(selected.get("tot_spd")) if isinstance(selected.get("tot_spd"), (int, float)) else float("-inf")
        records.append((feasible_rank, primary, speed, dataset, selected))
    if not records:
        return None
    _, _, _, dataset, selected = max(records, key=lambda item: (item[0], item[1], item[2]))
    summary = _format_selected_summary(selected)
    if summary is None:
        return None
    if len(records) == 1:
        return f"{dataset}：{summary}"
    return f"共 {len(records)} 项；当前最佳 {dataset}：{summary}"


def _summarize_single_layer(path: Path) -> Optional[str]:
    data = _read_json(path / "single_layer_all_results.json")
    if not isinstance(data, list) or not data:
        return None
    best: Optional[Tuple[float, str, str, float]] = None
    for entry in data:
        if not isinstance(entry, dict):
            continue
        metric = str(entry.get("primary_metric") or "metric")
        baseline = (entry.get("baseline") or {}).get(metric)
        if not isinstance(baseline, (int, float)):
            continue
        values: List[float] = []
        for group_name in ("gelu_degradation", "softmax_degradation"):
            for point in entry.get(group_name, []) or []:
                value = (point or {}).get(metric)
                if isinstance(value, (int, float)):
                    values.append(float(value))
        if not values:
            continue
        best_value = max(values)
        diff = best_value - float(baseline)
        candidate = (diff, str(entry.get("task") or "?"), metric, best_value)
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        return None
    diff, task, metric, value = best
    return f"{len(data)} 个任务；最佳 {task} {metric}={_format_float(value)}（较 baseline {_format_delta(diff)}）"


def _summarize_stepwise(path: Path) -> Optional[str]:
    files = _find_all(path, "stepwise_*.json")
    if not files:
        return None
    best_peak: Optional[Tuple[float, str, str]] = None
    best_final: Optional[Tuple[float, str, str]] = None
    for p in files:
        data = _read_json(p) or {}
        metric = str(data.get("primary_metric") or "metric")
        task = str(data.get("task") or _extract_dataset_from_name(p))
        peaks: List[float] = []
        finals: List[float] = []
        for trial in data.get("trials") or []:
            if not isinstance(trial, list) or not trial:
                continue
            for step in trial:
                value = ((step or {}).get("metrics") or {}).get(metric)
                if isinstance(value, (int, float)):
                    peaks.append(float(value))
            final_value = ((trial[-1] or {}).get("metrics") or {}).get(metric)
            if isinstance(final_value, (int, float)):
                finals.append(float(final_value))
        if peaks:
            candidate = (max(peaks), task, metric)
            if best_peak is None or candidate[0] > best_peak[0]:
                best_peak = candidate
        if finals:
            candidate = (max(finals), task, metric)
            if best_final is None or candidate[0] > best_final[0]:
                best_final = candidate
    parts = [f"{len(files)} 个任务"]
    if best_peak is not None:
        parts.append(
            f"最佳峰值 {best_peak[1]} {best_peak[2]}={_format_float(best_peak[0])}"
        )
    if best_final is not None:
        parts.append(
            f"最佳终局 {best_final[1]} {best_final[2]}={_format_float(best_final[0])}"
        )
    return "；".join(parts) if len(parts) > 1 else None


def _summarize_block1(path: Path) -> Optional[str]:
    data = _read_json(path / "block1_summary.json") or {}
    summary = data.get("summary") or []
    details = data.get("details") or {}
    if not isinstance(summary, list) or not summary:
        return None
    top = max(summary, key=lambda item: int((item or {}).get("anomalous_pairs") or 0))
    best_diff: Optional[Tuple[float, str]] = None
    for task, detail in (details or {}).items():
        if not isinstance(detail, dict):
            continue
        for pair in detail.get("all_pairs_summary") or []:
            diff = (pair or {}).get("diff")
            if isinstance(diff, (int, float)):
                candidate = (float(diff), str(task))
                if best_diff is None or candidate[0] > best_diff[0]:
                    best_diff = candidate
    parts = [
        f"异常对最多 {top.get('task')} {top.get('anomalous_pairs')}/{top.get('total_pairs')}"
    ]
    if best_diff is not None:
        parts.append(f"最大正向增益 {best_diff[1]} {_format_delta(best_diff[0])}")
    return "；".join(parts)


def _summarize_block2(path: Path) -> Optional[str]:
    data = _read_json(path / "block2_all_results.json") or {}
    if not isinstance(data, dict) or not data:
        return None
    significant_count = 0
    best: Optional[Tuple[float, str, int, int, str]] = None
    for task, payload in data.items():
        for pair in (payload or {}).get("pair_results") or []:
            layer_i = int(pair.get("layer_i", -1))
            layer_j = int(pair.get("layer_j", -1))
            for group_name, comp in ((pair.get("baseline_comparisons") or {}).items()):
                if not isinstance(comp, dict):
                    continue
                if comp.get("significant") is True:
                    significant_count += 1
                    diff = comp.get("diff")
                    if isinstance(diff, (int, float)):
                        candidate = (float(diff), str(task), layer_i, layer_j, str(group_name))
                        if best is None or candidate[0] > best[0]:
                            best = candidate
    if best is None:
        return f"{len(data)} 个任务；无显著提升"
    return (
        f"显著结果 {significant_count} 项；"
        f"最佳 {best[1]} L{best[2]}/L{best[3]} {best[4]} {_format_delta(best[0])}"
    )


def _summarize_block3(path: Path) -> Optional[str]:
    data = _read_json(path / "block3_results.json") or {}
    low_pairs = data.get("low_consistency_pairs") or []
    tasks = data.get("tasks") or []
    if not isinstance(low_pairs, list) or not low_pairs:
        return None
    lowest = min(
        (pair for pair in low_pairs if isinstance((pair or {}).get("rho"), (int, float))),
        key=lambda pair: float(pair.get("rho")),
        default=None,
    )
    if lowest is None:
        return None
    return (
        f"{len(tasks)} 个任务；最低一致性 "
        f"{lowest.get('task_i')}/{lowest.get('task_j')} rho={_format_float(lowest.get('rho'))}"
    )


def _summarize_noise_scaling(path: Path) -> Optional[str]:
    files = _find_all(path, "noise_scaling_sweep_*.json")
    if not files:
        return None
    best: Optional[Tuple[float, float, str, str, float]] = None
    best_scaling: Optional[Any] = None
    for p in files:
        data = _read_json(p) or {}
        dataset = str(data.get("dataset") or _extract_dataset_from_name(p))
        for target_name, target in (data.get("targets") or {}).items():
            if not isinstance(target, dict):
                continue
            for record in target.get("records") or []:
                primary_mean = ((record or {}).get("primary_metric") or {}).get("mean")
                loss_mean = ((record or {}).get("loss") or {}).get("mean")
                if not isinstance(primary_mean, (int, float)):
                    continue
                cost = (record or {}).get("simulated_total_cost")
                scaling = (record or {}).get("scaling_factor")
                candidate = (
                    float(primary_mean),
                    -float(loss_mean) if isinstance(loss_mean, (int, float)) else float("-inf"),
                    dataset,
                    str(target_name),
                    -float(cost) if isinstance(cost, (int, float)) else float("-inf"),
                )
                if best is None or candidate > best:
                    best = candidate
                    best_scaling = scaling
    if best is None:
        return None
    primary, _, dataset, target_name, neg_cost = best
    scaling_text = f"，factor={best_scaling}" if best_scaling is not None else ""
    return (
        f"{len(files)} 个任务；最佳 {dataset}/{target_name} "
        f"主={_format_float(primary)}，cost={_format_cost(-neg_cost)}{scaling_text}"
    )


def _summarize_layer_importance_runs(path: Path) -> Optional[str]:
    return _summarize_eval_collection(_find_all(path, "noise_final_eval_results_*.json", recursive=True))


def _fallback_experiment_summary(path: Path) -> str:
    json_count = len(_find_all(path, "*.json", recursive=True))
    png_count = len(_find_all(path, "*.png", recursive=True))
    if json_count or png_count:
        return f"已产出 {json_count} 个 JSON / {png_count} 张图"
    return "暂无可用结果"


def _experiment_best_result_summary(record: ExperimentRecord) -> str:
    name = record.name
    path = record.path
    summary = None
    if name == "final_evaluation":
        summary = _summarize_eval_collection(_find_all(path, "final_eval_results_*.json"))
    elif name == "noise_final_evaluation":
        summary = _summarize_eval_collection(_find_all(path, "noise_final_eval_results_*.json"))
    elif name == "layer_importance_runs":
        summary = _summarize_layer_importance_runs(path)
    elif name == "single_layer":
        summary = _summarize_single_layer(path)
    elif name == "stepwise":
        summary = _summarize_stepwise(path)
    elif name == "block1":
        summary = _summarize_block1(path)
    elif name == "block2":
        summary = _summarize_block2(path)
    elif name == "block3":
        summary = _summarize_block3(path)
    elif name == "noise_scaling_sweep":
        summary = _summarize_noise_scaling(path)
    return summary or _fallback_experiment_summary(path)


_NOISE_CONFIG_KEY_MAP = {
    "input_noise_scaling_factors": "x",
    "wq_noise_scaling_factors": "wq",
    "wk_noise_scaling_factors": "wk",
    "wv_noise_scaling_factors": "wv",
    "wo_noise_scaling_factors": "wo",
    "wffn1_noise_scaling_factors": "wffn1",
    "wffn2_noise_scaling_factors": "wffn2",
    "x": "x",
    "wq": "wq",
    "wk": "wk",
    "wv": "wv",
    "wo": "wo",
    "wffn1": "wffn1",
    "wffn2": "wffn2",
}
_NOISE_CONFIG_ORDER = ("x", "wq", "wk", "wv", "wo", "wffn1", "wffn2")
_DETAIL_RANGE_RE = re.compile(r"_(?P<start>\d+)-(?P<end>\d+)\.txt$")
_RL_NEW_BEST_RE = re.compile(r"回合\s*(?P<episode>\d+).*训练过程新高")
_RL_CONFIRM_HEADER_RE = re.compile(r"挑战者确认.*回合.*?(?P<episode>\d+)")
_RL_CONFIRM_N_RE = re.compile(r"N=(?P<n>\d+)")
_RL_CONFIRM_LOSS_RE = re.compile(
    r"loss=(?P<loss>-?\d+(?:\.\d+)?)\+/-?(?P<loss_std>\d+(?:\.\d+)?)\s*,?\s*m1=(?P<m1>-?\d+(?:\.\d+)?)\+/-?(?P<m1_std>\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_RL_CONFIRM_BOOL_RE = re.compile(r"=(?P<value>True|False)")
_RL_CONFIRM_VERDICT_RE = re.compile(r"\b(?P<value>PASS|FAIL)\b")


def _parse_int_list(raw: Any) -> Optional[List[int]]:
    if isinstance(raw, list):
        values: List[int] = []
        for item in raw:
            if not isinstance(item, (int, float)):
                return None
            values.append(int(item))
        return values
    if not isinstance(raw, str):
        return None
    values = [int(part) for part in re.findall(r"-?\d+", raw)]
    return values or None


def _normalize_stage1_config(config: Optional[dict]) -> Optional[Dict[str, List[int]]]:
    if not isinstance(config, dict):
        return None
    gelu = _parse_int_list(config.get("gelu"))
    softmax = _parse_int_list(config.get("softmax"))
    if not gelu and not softmax:
        return None
    out: Dict[str, List[int]] = {}
    if gelu:
        out["gelu"] = gelu
    if softmax:
        out["softmax"] = softmax
    return out or None


def _normalize_noise_config(config: Optional[dict]) -> Optional[Dict[str, List[int]]]:
    if not isinstance(config, dict):
        return None
    out: Dict[str, List[int]] = {}
    for key, alias in _NOISE_CONFIG_KEY_MAP.items():
        values = _parse_int_list(config.get(key))
        if values:
            out[alias] = values
    return out or None


def _extract_stage1_config(data: Optional[dict]) -> Optional[Dict[str, List[int]]]:
    if not isinstance(data, dict):
        return None
    for candidate in (
        data.get("fixed_stage1_config"),
        data.get("stage1_selected_config"),
        {"gelu": data.get("fixed_gelu"), "softmax": data.get("fixed_softmax")},
        data.get("selected"),
        data.get("selected_single"),
        data.get("best_config"),
    ):
        normalized = _normalize_stage1_config(candidate)
        if normalized:
            return normalized
    return None


def _extract_noise_config(data: Optional[dict]) -> Optional[Dict[str, List[int]]]:
    if not isinstance(data, dict):
        return None
    for candidate in (
        (data.get("selected") or {}).get("noise_config"),
        (data.get("selected_single") or {}).get("noise_config"),
        data.get("best_noise_config"),
        data.get("stable_search_best_noise_config"),
        data.get("stable_joint_best_noise_config"),
    ):
        normalized = _normalize_noise_config(candidate)
        if normalized:
            return normalized
    return None


def _stats_from_selected(selected: Optional[dict]) -> Optional[dict]:
    if not isinstance(selected, dict):
        return None
    payload = {
        "n": selected.get("evaluation_n"),
        "loss_mean": selected.get("loss"),
        "loss_std": selected.get("loss_std"),
        "p_mean": selected.get("p"),
        "p_std": selected.get("p_std"),
        "s_mean": selected.get("s"),
        "s_std": selected.get("s_std"),
        "time_mean_ms": selected.get("time_ms"),
        "time_std_ms": selected.get("time_std_ms"),
    }
    if any(payload.get(key) is not None for key in ("loss_mean", "p_mean", "s_mean")):
        return payload
    return None


def _extract_eval_stats(data: Optional[dict]) -> Optional[dict]:
    if not isinstance(data, dict):
        return None
    repeat = data.get("repeat_evaluation") or {}
    stats = repeat.get("stats")
    if isinstance(stats, dict):
        return stats
    return _stats_from_selected(data.get("selected") or data.get("selected_single"))


def _format_token_line(tokens: Iterable[str]) -> str:
    filtered = [f"`{token}`" for token in tokens if token]
    return " · ".join(filtered)


def _selected_metric_tokens(selected: Optional[dict]) -> List[str]:
    if not isinstance(selected, dict):
        return []
    tokens: List[str] = []
    if isinstance(selected.get("score"), (int, float)):
        tokens.append(f"score={_format_float(selected.get('score'))}")
    if isinstance(selected.get("loss"), (int, float)):
        tokens.append(f"loss={_format_float(selected.get('loss'))}")
    if isinstance(selected.get("p"), (int, float)):
        tokens.append(f"主={_format_float(selected.get('p'))}")
    elif isinstance(selected.get("metric1"), (int, float)):
        tokens.append(f"m1={_format_float(selected.get('metric1'))}")
    if isinstance(selected.get("s"), (int, float)):
        tokens.append(f"次={_format_float(selected.get('s'))}")
    elif isinstance(selected.get("metric2"), (int, float)):
        tokens.append(f"m2={_format_float(selected.get('metric2'))}")
    if isinstance(selected.get("tot_c"), (int, float)) and not selected.get("show_cost_as_na"):
        tokens.append(f"cost={_format_cost(selected.get('tot_c'))}")
    elif isinstance(selected.get("cost"), (int, float)):
        tokens.append(f"cost={_format_cost(selected.get('cost'))}")
    if isinstance(selected.get("tot_spd"), (int, float)):
        tokens.append(f"speed={_format_speedup(selected.get('tot_spd'))}")
    if selected.get("feasible") is False or selected.get("qualification_passed") is False:
        tokens.append("不可行")
    elif selected.get("feasible") is True or selected.get("qualification_passed") is True:
        tokens.append("可行")
    return tokens


def _stats_tokens(stats: Optional[dict]) -> List[str]:
    if not isinstance(stats, dict):
        return []
    tokens: List[str] = []
    if stats.get("n") is not None:
        tokens.append(f"n={stats.get('n')}")
    if isinstance(stats.get("loss_mean"), (int, float)):
        loss_std = stats.get("loss_std")
        if isinstance(loss_std, (int, float)):
            tokens.append(f"loss={_format_float(stats.get('loss_mean'))}±{_format_float(loss_std)}")
        else:
            tokens.append(f"loss={_format_float(stats.get('loss_mean'))}")
    if isinstance(stats.get("p_mean"), (int, float)):
        p_std = stats.get("p_std")
        if isinstance(p_std, (int, float)):
            tokens.append(f"主={_format_float(stats.get('p_mean'))}±{_format_float(p_std)}")
        else:
            tokens.append(f"主={_format_float(stats.get('p_mean'))}")
    if isinstance(stats.get("s_mean"), (int, float)):
        s_std = stats.get("s_std")
        if isinstance(s_std, (int, float)):
            tokens.append(f"次={_format_float(stats.get('s_mean'))}±{_format_float(s_std)}")
        else:
            tokens.append(f"次={_format_float(stats.get('s_mean'))}")
    if isinstance(stats.get("time_mean_ms"), (int, float)):
        time_std = stats.get("time_std_ms")
        if isinstance(time_std, (int, float)):
            tokens.append(f"time={_format_float(stats.get('time_mean_ms'))}±{_format_float(time_std)}ms")
        else:
            tokens.append(f"time={_format_float(stats.get('time_mean_ms'))}ms")
    return tokens


def _format_config_entries(entries: List[Tuple[str, List[int]]]) -> List[str]:
    if not entries:
        return []
    label_width = max(len(label) for label, _ in entries)
    lines: List[str] = []
    for label, values in entries:
        if not values:
            continue
        vector_text = "[" + ", ".join(str(v) for v in values) + "]"
        lines.append(f"{label:<{label_width}} {vector_text}")
    return lines


def _format_stage1_config_lines(config: Optional[Dict[str, List[int]]], prefix: str = "") -> List[str]:
    if not config:
        return []
    label_prefix = f"{prefix}." if prefix else ""
    entries: List[Tuple[str, List[int]]] = []
    if config.get("gelu"):
        entries.append((f"{label_prefix}gelu", config["gelu"]))
    if config.get("softmax"):
        entries.append((f"{label_prefix}softmax", config["softmax"]))
    return _format_config_entries(entries)


def _format_noise_config_lines(config: Optional[Dict[str, List[int]]], prefix: str = "") -> List[str]:
    if not config:
        return []
    label_prefix = f"{prefix}." if prefix else ""
    entries: List[Tuple[str, List[int]]] = []
    for key in _NOISE_CONFIG_ORDER:
        values = config.get(key)
        if values:
            entries.append((f"{label_prefix}{key}", values))
    return _format_config_entries(entries)


def _combine_config_lines(
    stage1_config: Optional[Dict[str, List[int]]] = None,
    noise_config: Optional[Dict[str, List[int]]] = None,
    prefix: str = "",
) -> Optional[str]:
    lines: List[str] = []
    stage1_lines = _format_stage1_config_lines(stage1_config, prefix)
    noise_lines = _format_noise_config_lines(noise_config, prefix)
    if stage1_lines:
        lines.append("[阶段1]")
        lines.extend(stage1_lines)
    if stage1_lines and noise_lines:
        lines.append("")
    if noise_lines:
        lines.append("[阶段2]")
        lines.extend(noise_lines)
    if not lines:
        return None
    return "\n".join(lines)


def _md_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "<br>")


def _append_markdown_card(
    lines: List[str],
    title: str,
    rows: List[Tuple[str, str]],
    *,
    config_blocks: Optional[List[Tuple[str, str]]] = None,
) -> None:
    lines.append(f"### `{title}`")
    lines.append("")
    lines.append("| 项目 | 内容 |")
    lines.append("|---|---|")
    for label, value in rows:
        if not value:
            continue
        lines.append(f"| {label} | {_md_cell(value)} |")
    lines.append("")
    for block_title, block_body in config_blocks or []:
        if not block_body:
            continue
        lines.append(f"**{block_title}**")
        lines.append("")
        lines.append("```text")
        lines.extend(block_body.splitlines())
        lines.append("```")
        lines.append("")
    lines.append("---")
    lines.append("")


def _parse_rl_stage1_config_from_stage2_log(log_path: Path) -> Optional[Dict[str, List[int]]]:
    if not log_path.exists():
        return None
    gelu: Optional[List[int]] = None
    softmax: Optional[List[int]] = None
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return None
    for line in lines[:120]:
        if gelu is None and "GELU" in line and "[" in line and ("离散阶数向量" in line or "[Selected" in line):
            gelu = _parse_int_list(line)
        if softmax is None and "Softmax" in line and "[" in line and ("离散阶数向量" in line or "[Selected" in line):
            softmax = _parse_int_list(line)
        if gelu and softmax:
            break
    return _normalize_stage1_config({"gelu": gelu, "softmax": softmax})


def _parse_rl_latest_config_from_log(log_path: Path) -> Optional[Dict[str, List[int]]]:
    if not log_path.exists():
        return None
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return None
    latest: Optional[Dict[str, List[int]]] = None
    current: Optional[Dict[str, List[int]]] = None
    for line in lines:
        if _RL_NEW_BEST_RE.search(line):
            if current:
                latest = current
            current = {}
            continue
        if current is None:
            continue
        match = re.search(r"(?P<key>[A-Za-z0-9_]+):\s*(?P<vals>\[.*\])", line)
        if not match:
            if current:
                latest = current
                current = None
            continue
        alias = _NOISE_CONFIG_KEY_MAP.get(match.group("key"))
        values = _parse_int_list(match.group("vals"))
        if alias and values:
            current[alias] = values
    if current:
        latest = current
    return latest or None


def _parse_rl_confirmation_blocks(log_path: Path) -> Dict[int, dict]:
    blocks: Dict[int, dict] = {}
    if not log_path.exists():
        return blocks
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return blocks
    i = 0
    while i < len(lines):
        header_match = _RL_CONFIRM_HEADER_RE.search(lines[i])
        if not header_match:
            i += 1
            continue
        episode = int(header_match.group("episode"))
        payload: Dict[str, Any] = {"episode": episode}
        j = i + 1
        while j < len(lines):
            line = lines[j]
            if _RL_CONFIRM_HEADER_RE.search(line):
                break
            n_match = _RL_CONFIRM_N_RE.search(line)
            if n_match:
                payload["n"] = int(n_match.group("n"))
            loss_match = _RL_CONFIRM_LOSS_RE.search(line)
            if loss_match:
                payload["loss"] = float(loss_match.group("loss"))
                payload["loss_std"] = float(loss_match.group("loss_std"))
                payload["m1"] = float(loss_match.group("m1"))
                payload["m1_std"] = float(loss_match.group("m1_std"))
            if "std_check=" in line:
                bool_match = _RL_CONFIRM_BOOL_RE.search(line)
                if bool_match:
                    payload["std_check"] = (bool_match.group("value") == "True")
            if "constraint=" in line:
                bool_match = _RL_CONFIRM_BOOL_RE.search(line)
                if bool_match:
                    payload["constraint"] = (bool_match.group("value") == "True")
            if "裁定" in line:
                verdict_match = _RL_CONFIRM_VERDICT_RE.search(line)
                if verdict_match:
                    payload["verdict"] = verdict_match.group("value")
            j += 1
        blocks[episode] = payload
        i = j
    return blocks


def _parse_rl_latest_incumbent(log_path: Path) -> Optional[dict]:
    if not log_path.exists():
        return None
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return None
    latest: Optional[dict] = None
    for line in lines:
        match = _RL_MEAN_SCORE_RE.search(line) or _RL_SELECTION_SCORE_RE.search(line)
        if match:
            latest = {
                "episode": int(match.group("step")),
                "score": float(match.group("score")),
                "cost": float(match.group("cost")),
            }
    return latest


def _parse_rl_episode_noise_config(details_dir: Path, episode: int) -> Optional[Dict[str, List[int]]]:
    if not details_dir.is_dir():
        return None
    target_file: Optional[Path] = None
    for path in sorted(details_dir.glob("noise_ppo_step_info_*.txt")):
        match = _DETAIL_RANGE_RE.search(path.name)
        if not match:
            continue
        start = int(match.group("start"))
        end = int(match.group("end"))
        if start <= episode <= end:
            target_file = path
            break
    if target_file is None:
        return None
    try:
        lines = target_file.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return None
    collecting = False
    out: Dict[str, List[int]] = {key: [] for key in _NOISE_CONFIG_ORDER}
    for line in lines:
        header_match = re.search(r"episode\s+(?P<episode>\d+)", line)
        if header_match:
            current_episode = int(header_match.group("episode"))
            if current_episode == episode:
                collecting = True
                continue
            if collecting:
                break
        if not collecting:
            continue
        for key, alias in (
            ("curr_input_noise_scaling_factor", "x"),
            ("curr_wq_noise_scaling_factor", "wq"),
            ("curr_wk_noise_scaling_factor", "wk"),
            ("curr_wv_noise_scaling_factor", "wv"),
            ("curr_wo_noise_scaling_factor", "wo"),
            ("curr_wffn1_noise_scaling_factor", "wffn1"),
            ("curr_wffn2_noise_scaling_factor", "wffn2"),
        ):
            match = re.search(rf"{key}:\s*(?P<value>-?\d+)", line)
            if match:
                out[alias].append(int(match.group("value")))
    if not any(out.values()):
        return None
    return {key: value for key, value in out.items() if value}


def _parse_rl_stage2_bundle(run_path: Path) -> Dict[str, Any]:
    bundle: Dict[str, Any] = {}
    log_path = run_path / "stage2_noise" / "pruning_search_log.txt"
    if not log_path.exists():
        return bundle
    bundle["stage1_config"] = _parse_rl_stage1_config_from_stage2_log(log_path)
    incumbent = _parse_rl_latest_incumbent(log_path)
    if incumbent:
        bundle["best"] = incumbent
        confirm_blocks = _parse_rl_confirmation_blocks(log_path)
        confirmation = confirm_blocks.get(int(incumbent["episode"]))
        if confirmation:
            bundle["confirmation"] = confirmation
        config = _parse_rl_episode_noise_config(run_path / "stage2_noise" / "details", int(incumbent["episode"]))
        if config:
            bundle["noise_config"] = config
    if "noise_config" not in bundle:
        config = _parse_rl_latest_config_from_log(log_path)
        if config:
            bundle["noise_config"] = config
    return bundle


def _format_rl_search_line(bundle: Optional[dict]) -> str:
    if not isinstance(bundle, dict):
        return ""
    best = bundle.get("best") or {}
    confirmation = bundle.get("confirmation") or {}
    tokens: List[str] = []
    if best.get("episode") is not None:
        tokens.append(f"ep={best.get('episode')}")
    if isinstance(best.get("score"), (int, float)):
        tokens.append(f"score={_format_float(best.get('score'))}")
    if isinstance(best.get("cost"), (int, float)):
        tokens.append(f"cost={_format_cost(best.get('cost'))}")
    if confirmation.get("n") is not None:
        tokens.append(f"N={confirmation.get('n')}")
    if isinstance(confirmation.get("loss"), (int, float)):
        if isinstance(confirmation.get("loss_std"), (int, float)):
            tokens.append(
                f"loss={_format_float(confirmation.get('loss'))}±{_format_float(confirmation.get('loss_std'))}"
            )
        else:
            tokens.append(f"loss={_format_float(confirmation.get('loss'))}")
    if isinstance(confirmation.get("m1"), (int, float)):
        if isinstance(confirmation.get("m1_std"), (int, float)):
            tokens.append(
                f"m1={_format_float(confirmation.get('m1'))}±{_format_float(confirmation.get('m1_std'))}"
            )
        else:
            tokens.append(f"m1={_format_float(confirmation.get('m1'))}")
    if confirmation.get("verdict"):
        tokens.append(str(confirmation.get("verdict")))
    return _format_token_line(tokens)


def _choose_ga_noise_search_best(data: Optional[dict]) -> Optional[dict]:
    if not isinstance(data, dict):
        return None
    for key in ("stable_joint_best_noise_config", "stable_search_best_noise_config", "best_noise_config"):
        candidate = data.get(key)
        if isinstance(candidate, dict):
            return candidate
    return None


def _format_ga_noise_search_line(data: Optional[dict]) -> str:
    best = _choose_ga_noise_search_best(data)
    if not isinstance(best, dict):
        return ""
    tokens: List[str] = []
    if data and data.get("best_generation") is not None:
        tokens.append(f"gen={data.get('best_generation')}")
    if isinstance(best.get("search_score_mean"), (int, float)):
        tokens.append(f"score={_format_float(best.get('search_score_mean'))}")
    elif isinstance(best.get("score"), (int, float)):
        tokens.append(f"score={_format_float(best.get('score'))}")
    if isinstance(best.get("cost"), (int, float)):
        tokens.append(f"cost={_format_cost(best.get('cost'))}")
    stats = best.get("stats") if isinstance(best.get("stats"), dict) else None
    tokens.extend(_stats_tokens(stats))
    return _format_token_line(tokens)


def _format_ga_stage1_search_line(data: Optional[dict]) -> str:
    if not isinstance(data, dict):
        return ""
    best = data.get("best_config")
    if not isinstance(best, dict):
        return ""
    tokens: List[str] = []
    if data.get("best_generation") is not None:
        tokens.append(f"gen={data.get('best_generation')}")
    tokens.extend(_selected_metric_tokens(best))
    return _format_token_line(tokens)


def _select_best_eval(paths: List[Path]) -> Optional[Tuple[Path, dict]]:
    best: Optional[Tuple[int, float, float, Path, dict]] = None
    for path in paths:
        data = _read_json(path) or {}
        selected = data.get("selected") or data.get("selected_single")
        if not isinstance(selected, dict):
            continue
        feasible_rank = 1 if selected.get("feasible") is not False else 0
        primary = float(selected.get("p")) if isinstance(selected.get("p"), (int, float)) else float("-inf")
        speed = float(selected.get("tot_spd")) if isinstance(selected.get("tot_spd"), (int, float)) else float("-inf")
        candidate = (feasible_rank, primary, speed, path, data)
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    if best is None:
        return None
    return best[3], best[4]


def _build_run_markdown_card(record: RunRecord) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    rows: List[Tuple[str, str]] = [("进度", _format_token_line([_run_progress_summary(record)]))]
    config_blocks: List[Tuple[str, str]] = []
    stage1_eval_path = _find_first(record.path / "stage1_final_eval", "final_eval_results_*.json")
    stage2_eval_path = _find_first(record.path / "stage2_noise_final_eval", "noise_final_eval_results_*.json")
    stage1_eval = _read_json(stage1_eval_path) if stage1_eval_path else None
    stage2_eval = _read_json(stage2_eval_path) if stage2_eval_path else None
    ga_stage1_search = _read_json(record.path / "stage1" / "ga_search_results.json")
    ga_stage2_search = _read_json(record.path / "stage2_noise" / "noise_ga_search_results.json")
    rl_stage2_bundle = _parse_rl_stage2_bundle(record.path) if record.algorithm == "RL" else {}

    stage1_config = None
    noise_config = None

    if stage2_eval:
        selected = stage2_eval.get("selected") or stage2_eval.get("selected_single")
        rows.append(("当前最优", _format_token_line(["S2 终评"] + _selected_metric_tokens(selected))))
        eval_stats = _extract_eval_stats(stage2_eval)
        if eval_stats:
            rows.append(("终评测试", _format_token_line(_stats_tokens(eval_stats))))
        if record.algorithm == "GA" and ga_stage2_search:
            search_line = _format_ga_noise_search_line(ga_stage2_search)
            if search_line:
                rows.append(("搜索验证", search_line))
        elif record.algorithm == "RL":
            search_line = _format_rl_search_line(rl_stage2_bundle)
            if search_line:
                rows.append(("搜索确认", search_line))
        stage1_config = _extract_stage1_config(stage2_eval) or _extract_stage1_config(ga_stage2_search) or rl_stage2_bundle.get("stage1_config")
        noise_config = _extract_noise_config(stage2_eval) or _extract_noise_config(ga_stage2_search) or rl_stage2_bundle.get("noise_config")
    elif record.algorithm == "GA" and ga_stage2_search:
        rows.append(("当前最优", _format_token_line(["S2 搜索"] + [token.strip("`") for token in _format_ga_noise_search_line(ga_stage2_search).replace("`", "").split(" · ")])))
        stage1_config = _extract_stage1_config(ga_stage2_search)
        noise_config = _extract_noise_config(ga_stage2_search)
    elif record.algorithm == "RL" and rl_stage2_bundle.get("best"):
        best = rl_stage2_bundle.get("best") or {}
        tokens = ["S2 搜索"]
        if best.get("episode") is not None:
            tokens.append(f"ep={best.get('episode')}")
        if isinstance(best.get("score"), (int, float)):
            tokens.append(f"score={_format_float(best.get('score'))}")
        if isinstance(best.get("cost"), (int, float)):
            tokens.append(f"cost={_format_cost(best.get('cost'))}")
        rows.append(("当前最优", _format_token_line(tokens)))
        search_line = _format_rl_search_line(rl_stage2_bundle)
        if search_line:
            rows.append(("搜索确认", search_line))
        stage1_config = rl_stage2_bundle.get("stage1_config")
        noise_config = rl_stage2_bundle.get("noise_config")
    elif stage1_eval:
        selected = stage1_eval.get("selected") or stage1_eval.get("selected_single")
        rows.append(("当前最优", _format_token_line(["S1 终评"] + _selected_metric_tokens(selected))))
        eval_stats = _extract_eval_stats(stage1_eval)
        if eval_stats:
            rows.append(("终评测试", _format_token_line(_stats_tokens(eval_stats))))
        if record.algorithm == "GA" and ga_stage1_search:
            search_line = _format_ga_stage1_search_line(ga_stage1_search)
            if search_line:
                rows.append(("搜索验证", search_line))
        stage1_config = _extract_stage1_config(stage1_eval) or _extract_stage1_config(ga_stage1_search)
    elif record.algorithm == "GA" and ga_stage1_search:
        rows.append(("当前最优", _format_token_line(["S1 搜索"] + [token.strip("`") for token in _format_ga_stage1_search_line(ga_stage1_search).replace("`", "").split(" · ")])))
        stage1_config = _extract_stage1_config(ga_stage1_search)
    else:
        rows.append(("当前最优", _format_token_line([_run_best_result_summary(record)])))

    config_text = _combine_config_lines(stage1_config, noise_config)
    if config_text:
        config_blocks.append(("最优配置", config_text))
    return rows, config_blocks


def _build_compare_markdown_card(record: CompareRecord) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    rows: List[Tuple[str, str]] = [("进度", _format_token_line([_compare_progress_summary(record)]))]
    config_blocks: List[Tuple[str, str]] = []
    reports_dir = record.path / "reports"
    stage2_summary_path = _find_first(reports_dir, "stage2_compare_summary*.json")
    stage1_summary_path = _find_first(reports_dir, "stage1_compare_summary*.json")
    stage_label = "S2" if stage2_summary_path else "S1"
    summary = _read_json(stage2_summary_path or stage1_summary_path) or {}
    rows.append(("当前阶段", _format_token_line([stage_label])))

    for side_key, side_label in (("rl", "RL"), ("ga", "GA")):
        side = (summary.get("sides") or {}).get(side_key) or {}
        selected = side.get("selected")
        if isinstance(selected, dict):
            rows.append((f"{side_label} 终评", _format_token_line(_selected_metric_tokens(selected))))
            stats = _stats_from_selected(selected)
            if stats:
                rows.append((f"{side_label} 终评测试", _format_token_line(_stats_tokens(stats))))
        search_line = ""
        if stage_label == "S2":
            if side_key == "ga":
                search_line = _format_ga_noise_search_line(
                    _read_json(record.path / "children" / "ga" / "stage2_noise" / "noise_ga_search_results.json")
                )
            else:
                search_line = _format_rl_search_line(_parse_rl_stage2_bundle(record.path / "children" / "rl"))
        elif stage_label == "S1" and side_key == "ga":
            search_line = _format_ga_stage1_search_line(
                _read_json(record.path / "children" / "ga" / "stage1" / "ga_search_results.json")
            )
        if search_line:
            rows.append((f"{side_label} 搜索验证", search_line))

        stage1_config = _extract_stage1_config(side)
        noise_config = _extract_noise_config(side)
        block = _combine_config_lines(stage1_config, noise_config)
        if block:
            config_blocks.append((f"{side_label} 最优配置", block))

    return rows, config_blocks


def _build_eval_experiment_card(record: ExperimentRecord, eval_paths: List[Path]) -> Optional[Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]]:
    chosen = _select_best_eval(eval_paths)
    if chosen is None:
        return None
    best_path, data = chosen
    selected = data.get("selected") or data.get("selected_single")
    rows: List[Tuple[str, str]] = [
        ("进度", _format_token_line([_experiment_progress_summary(record)])),
        ("最佳来源", _format_token_line([str(best_path.parent.relative_to(record.path)) if best_path.parent != record.path else best_path.name])),
        ("当前最优", _format_token_line(_selected_metric_tokens(selected))),
    ]
    stats = _extract_eval_stats(data)
    if stats:
        rows.append(("终评测试", _format_token_line(_stats_tokens(stats))))
    config_text = _combine_config_lines(_extract_stage1_config(data), _extract_noise_config(data))
    config_blocks = [("最优配置", config_text)] if config_text else []
    return rows, config_blocks


def _build_experiment_markdown_card(record: ExperimentRecord) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    rows: List[Tuple[str, str]] = [("进度", _format_token_line([_experiment_progress_summary(record)]))]
    config_blocks: List[Tuple[str, str]] = []

    if record.name == "final_evaluation":
        card = _build_eval_experiment_card(record, _find_all(record.path, "final_eval_results_*.json"))
        if card:
            return card
    elif record.name == "noise_final_evaluation":
        card = _build_eval_experiment_card(record, _find_all(record.path, "noise_final_eval_results_*.json"))
        if card:
            return card
    elif record.name == "layer_importance_runs":
        card = _build_eval_experiment_card(record, _find_all(record.path, "noise_final_eval_results_*.json", recursive=True))
        if card:
            return card

    rows.append(("结果摘要", _format_token_line([_experiment_best_result_summary(record)])))
    return rows, config_blocks


def render_markdown(
    rl_runs: List[RunRecord],
    ga_runs: List[RunRecord],
    general_runs: List[RunRecord],
    compare_runs: List[CompareRecord],
    experiments: List[ExperimentRecord],
    root: Path,
    generated_at: str,
) -> str:
    lines: List[str] = []
    lines.append("# 任务总板 / STATUS")
    lines.append("")
    lines.append("> 聚焦任务进度、当前最优结果、最优配置，以及训练/终评阶段已经产出的测试摘要。")
    lines.append("")

    def _emit_run_section(title: str, runs: List[RunRecord]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        if not runs:
            lines.append("_无记录_")
            lines.append("")
            return
        for r in runs:
            rows, config_blocks = _build_run_markdown_card(r)
            _append_markdown_card(lines, _run_title(r), rows, config_blocks=config_blocks)

    _emit_run_section("1. 单任务 RL（rl_results/persistent/rl/）", rl_runs)
    _emit_run_section("2. 单任务 GA（rl_results/persistent/ga/）", ga_runs)
    _emit_run_section("3. 通用策略 General-RL（rl_results/persistent/general-rl/）", general_runs)

    # ---- compare ----
    lines.append("## 4. RL vs GA 对比（rl_results/runs/compare/rl_vs_ga/）")
    lines.append("")
    if not compare_runs:
        lines.append("_无记录_")
        lines.append("")
    else:
        for c in compare_runs:
            rows, config_blocks = _build_compare_markdown_card(c)
            _append_markdown_card(lines, f"{c.dataset} / {c.run_name}", rows, config_blocks=config_blocks)

    # ---- experiments ----
    lines.append("## 5. 一次性实验（experiment_results/）")
    lines.append("")
    if not experiments:
        lines.append("_无记录_")
        lines.append("")
    else:
        for e in experiments:
            rows, config_blocks = _build_experiment_markdown_card(e)
            _append_markdown_card(lines, e.name, rows, config_blocks=config_blocks)

    if len(lines) < 2 or lines[-2] != "---":
        lines.append("---")
        lines.append("")
    lines.append("- `S1/S2` 进度格式：`搜索/终评`")
    lines.append("- compare 的 `终评 S1/S2` 格式：`RL/GA`")
    lines.append("- 结果展示优先级：`S2 终评 > S2 搜索 > S1 终评 > S1 搜索`")
    lines.append("- 自动生成：`tools/status_board.py`")
    lines.append("")
    return "\n".join(lines)


def render_terminal(
    rl_runs: List[RunRecord],
    ga_runs: List[RunRecord],
    general_runs: List[RunRecord],
    compare_runs: List[CompareRecord],
    experiments: List[ExperimentRecord],
    root: Path,
    generated_at: str,
) -> str:
    def _line(ch: str = "─", n: int = 80) -> str:
        return ch * n

    out: List[str] = []
    out.append(_line("━"))
    out.append(f" 任务总板  root={root}  生成时间={generated_at}")
    out.append(_line("━"))

    def _emit_runs(title: str, runs: List[RunRecord]) -> None:
        out.append("")
        out.append(f"▌ {title}  ({len(runs)} 条)")
        out.append(_line())
        if not runs:
            out.append("  （无记录）")
            return
        for r in runs:
            out.append(f"  - {_run_title(r)}")
            out.append(f"    进度: {_run_progress_summary(r)}")
            out.append(f"    当前最优: {_run_best_result_summary(r)}")

    _emit_runs("① 单任务 RL", rl_runs)
    _emit_runs("② 单任务 GA", ga_runs)
    _emit_runs("③ 通用策略 General-RL", general_runs)

    # compare
    out.append("")
    out.append(f"▌ ④ RL vs GA 对比  ({len(compare_runs)} 条)")
    out.append(_line())
    if not compare_runs:
        out.append("  （无记录）")
    else:
        for c in compare_runs:
            out.append(f"  - {c.dataset} / {c.run_name}")
            out.append(f"    进度: {_compare_progress_summary(c)}")
            out.append(f"    当前结果: {_compare_best_result_summary(c)}")

    # experiments
    out.append("")
    out.append(f"▌ ⑤ 一次性实验  ({len(experiments)} 条)")
    out.append(_line())
    if not experiments:
        out.append("  （无记录）")
    else:
        for e in experiments:
            out.append(f"  - {e.name}")
            out.append(f"    进度: {_experiment_progress_summary(e)}")
            out.append(f"    当前结果: {_experiment_best_result_summary(e)}")

    out.append("")
    out.append("  注: S1/S2 进度格式=搜索/终评；compare 的终评格式=RL/GA。")
    out.append(_line("━"))
    return "\n".join(out)


def _truncate(s: Optional[str], n: int) -> str:
    s = "" if s is None else str(s)
    if len(s) <= n:
        return s
    if n <= 1:
        return s[:n]
    return s[: n - 1] + "…"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def collect_all(root: Path) -> Tuple[List[RunRecord], List[RunRecord], List[RunRecord], List[CompareRecord], List[ExperimentRecord]]:
    rl_runs = _scan_persistent_runs(root, PERSISTENT_RL_BRANCH, "RL")
    ga_runs = _scan_persistent_runs(root, PERSISTENT_GA_BRANCH, "GA")
    general_runs = _scan_persistent_runs(root, PERSISTENT_GENERAL_RL_BRANCH, "GeneralRL")
    compare_runs = _scan_compare_runs(root)
    experiments = _scan_experiments(root)
    return rl_runs, ga_runs, general_runs, compare_runs, experiments


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument(
        "--root",
        default=os.getcwd(),
        help="项目根目录（默认：当前工作目录）",
    )
    parser.add_argument(
        "--only",
        choices=("rl", "ga", "general", "compare", "experiments"),
        default=None,
        help="只打印某一类。默认全部。",
    )
    parser.add_argument(
        "--write-md",
        action="store_true",
        help="同时写 docs/STATUS.md 快照。",
    )
    parser.add_argument(
        "--md-path",
        default=os.path.join("docs", "STATUS.md"),
        help="STATUS.md 输出路径（默认 docs/STATUS.md）",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 形式输出到 stdout（方便脚本消费，终端友好格式会被抑制）",
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rl_runs, ga_runs, general_runs, compare_runs, experiments = collect_all(root)

    # --only 过滤
    if args.only == "rl":
        ga_runs, general_runs, compare_runs, experiments = [], [], [], []
    elif args.only == "ga":
        rl_runs, general_runs, compare_runs, experiments = [], [], [], []
    elif args.only == "general":
        rl_runs, ga_runs, compare_runs, experiments = [], [], [], []
    elif args.only == "compare":
        rl_runs, ga_runs, general_runs, experiments = [], [], [], []
    elif args.only == "experiments":
        rl_runs, ga_runs, general_runs, compare_runs = [], [], [], []

    if args.json:
        payload = {
            "generated_at": generated_at,
            "root": str(root),
            "rl_runs": [_record_to_dict(r) for r in rl_runs],
            "ga_runs": [_record_to_dict(r) for r in ga_runs],
            "general_runs": [_record_to_dict(r) for r in general_runs],
            "compare_runs": [_compare_to_dict(c) for c in compare_runs],
            "experiments": [_experiment_to_dict(e) for e in experiments],
        }
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return 0

    # 终端打印
    text = render_terminal(rl_runs, ga_runs, general_runs, compare_runs, experiments, root, generated_at)
    print(text)

    if args.write_md:
        md_text = render_markdown(rl_runs, ga_runs, general_runs, compare_runs, experiments, root, generated_at)
        md_path = Path(args.md_path)
        if not md_path.is_absolute():
            md_path = root / md_path
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md_text, encoding="utf-8")
        print(f"\n[status_board] 已写入 {md_path}")

    return 0


def _record_to_dict(r: RunRecord) -> dict:
    return {
        "algorithm": r.algorithm,
        "model_type": r.model_type,
        "task": r.task,
        "slug": r.slug,
        "path": str(r.path),
        "stage_status": r.stage_status,
        "last_updated": r.last_updated,
        "run_count": r.run_count,
        "latest_pid": r.latest_pid,
        "latest_run_dir": r.latest_run_dir,
        "alive": r.alive,
        "error": r.error,
    }


def _compare_to_dict(c: CompareRecord) -> dict:
    return {
        "dataset": c.dataset,
        "run_name": c.run_name,
        "path": str(c.path),
        "mode": c.mode,
        "state_rl": c.state_rl,
        "state_ga": c.state_ga,
        "rl_stage1_ready": c.rl_stage1_ready,
        "rl_stage2_ready": c.rl_stage2_ready,
        "ga_stage1_ready": c.ga_stage1_ready,
        "ga_stage2_ready": c.ga_stage2_ready,
        "updated_at": c.updated_at,
        "stage1_report": c.stage1_report,
        "stage2_report": c.stage2_report,
    }


def _experiment_to_dict(e: ExperimentRecord) -> dict:
    return {
        "name": e.name,
        "path": str(e.path),
        "has_run_log": e.has_run_log,
        "last_mtime": e.last_mtime,
    }


if __name__ == "__main__":
    raise SystemExit(main())
