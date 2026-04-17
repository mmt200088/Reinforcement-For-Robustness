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
                stage_status = dict(meta.get("stage_status") or {})
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
    "failed": "×",
    "unknown": "?",
}


def _stage_cell(status: str) -> str:
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
    lines.append("# 任务进度总板 / STATUS")
    lines.append("")
    lines.append(f"- 项目根目录：`{root}`")
    lines.append(f"- 生成时间：`{generated_at}`")
    lines.append(f"- 由 `tools/status_board.py` 生成（只读扫描，不修改任何结果目录）")
    lines.append("")
    lines.append("图例：`✓ completed`, `· not_started`, `→ skipped`, `* running`, `× failed`, `? unknown`")
    lines.append("")

    def _emit_run_table(title: str, runs: List[RunRecord]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        if not runs:
            lines.append("_无记录_")
            lines.append("")
            return
        lines.append(
            "| model | dataset | slug | stage1_search | stage1_eval | stage2_search | stage2_eval | runs | 最近更新 | PID |"
        )
        lines.append(
            "|---|---|---|---|---|---|---|---:|---|---|"
        )
        for r in runs:
            ss = r.stage_status
            lines.append(
                "| {model} | {task} | {slug} | {s1s} | {s1e} | {s2s} | {s2e} | {cnt} | {ts} | {pid} |".format(
                    model=r.model_type,
                    task=r.task,
                    slug=r.slug,
                    s1s=_stage_cell(ss.get("stage1_search", "unknown")),
                    s1e=_stage_cell(ss.get("stage1_final_eval", "unknown")),
                    s2s=_stage_cell(ss.get("stage2_search", "unknown")),
                    s2e=_stage_cell(ss.get("stage2_final_eval", "unknown")),
                    cnt=("-" if r.run_count is None else r.run_count),
                    ts=_format_time(r.last_updated),
                    pid=_alive_cell(r),
                )
            )
        lines.append("")

    _emit_run_table("1. 单任务 RL（rl_results/persistent/rl/）", rl_runs)
    _emit_run_table("2. 单任务 GA（rl_results/persistent/ga/）", ga_runs)
    _emit_run_table("3. 通用策略 General-RL（rl_results/persistent/general-rl/）", general_runs)

    # ---- compare ----
    lines.append("## 4. RL vs GA 对比（rl_results/runs/compare/rl_vs_ga/）")
    lines.append("")
    if not compare_runs:
        lines.append("_无记录_")
        lines.append("")
    else:
        lines.append(
            "| dataset | 运行名 | mode | RL状态 | GA状态 | RL S1/S2 eval | GA S1/S2 eval | 最近更新 | 报告 |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for c in compare_runs:
            rl_eval = f"{_bool_cell(c.rl_stage1_ready)}/{_bool_cell(c.rl_stage2_ready)}"
            ga_eval = f"{_bool_cell(c.ga_stage1_ready)}/{_bool_cell(c.ga_stage2_ready)}"
            reports: List[str] = []
            if c.stage1_report:
                reports.append(f"[{c.stage1_report}]({c.path.as_posix()}/reports/{c.stage1_report})")
            if c.stage2_report:
                reports.append(f"[{c.stage2_report}]({c.path.as_posix()}/reports/{c.stage2_report})")
            lines.append(
                "| {ds} | {name} | {mode} | {rl} | {ga} | {re} | {ge} | {ts} | {rep} |".format(
                    ds=c.dataset,
                    name=c.run_name,
                    mode=c.mode or "-",
                    rl=c.state_rl or "?",
                    ga=c.state_ga or "?",
                    re=rl_eval,
                    ge=ga_eval,
                    ts=_format_time(c.updated_at),
                    rep=" / ".join(reports) if reports else "-",
                )
            )
        lines.append("")

    # ---- experiments ----
    lines.append("## 5. 一次性实验（experiment_results/）")
    lines.append("")
    if not experiments:
        lines.append("_无记录_")
        lines.append("")
    else:
        lines.append("| 名称 | 有 run.log | 最近变动 |")
        lines.append("|---|---|---|")
        for e in experiments:
            lines.append(f"| {e.name} | {'✓' if e.has_run_log else '-'} | {e.last_mtime or '-'} |")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 字段说明")
    lines.append("")
    lines.append("- **stage1_search / stage1_eval**：Stage-1（GELU/Softmax）搜索与最终评估进度，取自各 `metadata.json` 的 `stage_status` 字典")
    lines.append("- **stage2_search / stage2_eval**：Stage-2（噪声）搜索与最终评估进度")
    lines.append("- **runs**：该 slug 累计跑过多少轮（续训次数）")
    lines.append("- **PID alive/dead**：根据 `LATEST_PID` 文件探测该训练进程是否还活着")
    lines.append("- **RL S1/S2 eval**：compare 实验里 RL 侧两阶段最终评估是否就绪；GA 侧类推")
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
    out.append(f" 任务进度总板  root={root}  生成时间={generated_at}")
    out.append(_line("━"))

    def _emit_runs(title: str, runs: List[RunRecord]) -> None:
        out.append("")
        out.append(f"▌ {title}  ({len(runs)} 条)")
        out.append(_line())
        if not runs:
            out.append("  （无记录）")
            return
        header = "  {:<10} {:<8} {:<32} {:<45} {:>5} {:<20} {}".format(
            "model", "task", "slug", "stage_status", "runs", "last_updated", "pid"
        )
        out.append(header)
        out.append("  " + "-" * (len(header) - 2))
        for r in runs:
            ss = r.stage_status
            status_str = "{} {} {} {}".format(
                _stage_cell(ss.get("stage1_search", "unknown")),
                _stage_cell(ss.get("stage1_final_eval", "unknown")),
                _stage_cell(ss.get("stage2_search", "unknown")),
                _stage_cell(ss.get("stage2_final_eval", "unknown")),
            )
            out.append(
                "  {model:<10} {task:<8} {slug:<32} {status:<45} {cnt:>5} {ts:<20} {pid}".format(
                    model=_truncate(r.model_type, 10),
                    task=_truncate(r.task, 8),
                    slug=_truncate(r.slug, 32),
                    status=_truncate(status_str, 45),
                    cnt=("-" if r.run_count is None else r.run_count),
                    ts=_truncate(_format_time(r.last_updated), 20),
                    pid=_alive_cell(r),
                )
            )

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
        out.append(
            "  {:<8} {:<38} {:<16} {:<10} {:<10} {:<9} {:<9} {:<20}".format(
                "dataset", "run_name", "mode", "rl_state", "ga_state", "rl_eval", "ga_eval", "updated_at"
            )
        )
        out.append("  " + "-" * 130)
        for c in compare_runs:
            out.append(
                "  {ds:<8} {name:<38} {mode:<16} {rl:<10} {ga:<10} {re:<9} {ge:<9} {ts:<20}".format(
                    ds=_truncate(c.dataset, 8),
                    name=_truncate(c.run_name, 38),
                    mode=_truncate(c.mode or "-", 16),
                    rl=_truncate(c.state_rl or "?", 10),
                    ga=_truncate(c.state_ga or "?", 10),
                    re=f"{_bool_cell(c.rl_stage1_ready)}/{_bool_cell(c.rl_stage2_ready)}",
                    ge=f"{_bool_cell(c.ga_stage1_ready)}/{_bool_cell(c.ga_stage2_ready)}",
                    ts=_truncate(_format_time(c.updated_at), 20),
                )
            )

    # experiments
    out.append("")
    out.append(f"▌ ⑤ 一次性实验  ({len(experiments)} 条)")
    out.append(_line())
    if not experiments:
        out.append("  （无记录）")
    else:
        for e in experiments:
            out.append(f"  {e.name:<40}  run.log={'✓' if e.has_run_log else '-'}  mtime={e.last_mtime or '-'}")

    out.append("")
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
