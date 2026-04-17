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
    lines.append("> 只保留任务进度和当前最优结果，省略更新时间、PID 等运维字段。")
    lines.append("")

    def _emit_run_section(title: str, runs: List[RunRecord]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        if not runs:
            lines.append("_无记录_")
            lines.append("")
            return
        for r in runs:
            lines.append(
                f"- `{_run_title(r)}`：进度 `{_run_progress_summary(r)}`；"
                f"当前最优 `{_run_best_result_summary(r)}`"
            )
        lines.append("")

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
            lines.append(
                f"- `{c.dataset} / {c.run_name}`：进度 `{_compare_progress_summary(c)}`；"
                f"当前结果 `{_compare_best_result_summary(c)}`"
            )
        lines.append("")

    # ---- experiments ----
    lines.append("## 5. 一次性实验（experiment_results/）")
    lines.append("")
    if not experiments:
        lines.append("_无记录_")
        lines.append("")
    else:
        for e in experiments:
            lines.append(
                f"- `{e.name}`：进度 `{_experiment_progress_summary(e)}`；"
                f"当前结果 `{_experiment_best_result_summary(e)}`"
            )
        lines.append("")

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
