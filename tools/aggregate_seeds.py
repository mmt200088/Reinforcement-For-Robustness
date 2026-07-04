"""Aggregate metrics across multiple seeds of a BLB Stage-2 RL multi-seed run.

Reads each seed's persistent dir (located via the launcher's
``--run-tag <RUN>_s<SEED>`` slug suffix), extracts training-best reward and
final-eval metrics from the JSON / Markdown artifacts, and produces a single
``seed_summary.md`` plus a ``seed_summary.json`` with mean ± std and
per-seed rows.

Designed to be **robust to missing seeds**: if a run crashed or hadn't
written its final report yet, the row shows ``status=incomplete`` and the
aggregate is computed over the seeds that did finish.

Usage (called automatically by ``tools/run_multi_seed.sh``)::

    python3 tools/aggregate_seeds.py \\
        --run-name myrun \\
        --seed-list experiments/multi_seed/myrun/seed_list.txt \\
        --output-dir experiments/multi_seed/myrun

The seed-list file is two columns: ``<seed> <run_tag>`` per line.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import AbstractSet, Any, Dict, Iterable, List, Optional, Tuple

PERSISTENT_ROOT = os.path.join("Parting Chapter", "persistent")
_INVALID_RATE_LAST50_RE = re.compile(r"最近 50 回合 mean invalid 子步数: \*\*([0-9.]+)\*\*")
_DIAGNOSTICS_SCAN_CHUNK_SIZE = 1 << 20


@dataclass
class SeedRow:
    seed: int
    run_tag: str
    persistent_dir: str = ""
    status: str = "unknown"
    completed_episodes: Optional[int] = None
    best_reward: Optional[float] = None
    final_eval_loss: Optional[float] = None
    final_eval_metric1: Optional[float] = None   # accuracy / pearson
    final_eval_metric2: Optional[float] = None   # f1 / spearman (task-dep.)
    total_bits_sum: Optional[int] = None
    fusion_count: Optional[int] = None
    avg_truncation_k: Optional[float] = None
    invalid_rate_last50: Optional[float] = None
    error_msg: Optional[str] = None


def _iter_child_dirs(path: str):
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                try:
                    if entry.is_dir():
                        yield entry
                except OSError:
                    continue
    except OSError:
        return


def _build_persistent_dir_index(
    root: str = PERSISTENT_ROOT,
    requested_run_tags: Optional[AbstractSet[str]] = None,
) -> Dict[str, str]:
    """Index legacy persistent dirs keyed by run_tag suffix.

    Matches the old ``Parting Chapter/persistent/*/*/*/*__<run_tag>`` shape
    while scanning the tree once for a whole multi-seed aggregation.
    """
    if requested_run_tags is not None and not requested_run_tags:
        return {}

    best: Dict[str, Tuple[float, str]] = {}
    for level1 in _iter_child_dirs(root):
        for level2 in _iter_child_dirs(level1.path):
            for level3 in _iter_child_dirs(level2.path):
                for entry in _iter_child_dirs(level3.path):
                    if "__" not in entry.name:
                        continue
                    run_tag = entry.name.rsplit("__", 1)[1]
                    if not run_tag:
                        continue
                    if requested_run_tags is not None and run_tag not in requested_run_tags:
                        continue
                    try:
                        mtime = entry.stat().st_mtime
                    except OSError:
                        continue
                    candidate = (mtime, entry.path)
                    if run_tag not in best or candidate > best[run_tag]:
                        best[run_tag] = candidate
    return {run_tag: path for run_tag, (_, path) in best.items()}


def _find_persistent_dir(run_tag: str) -> Optional[str]:
    """Walk the persistent root looking for a slug ending in ``__<run_tag>``.

    注：这是给旧 ``persistent/`` 多 seed sweep（``--run-tag``）用的。解耦后新布局
    （2026-06-01）每个 combo 只一个工作目录，多次运行归档到
    ``Parting Chapter/stage{1,2}/record/{combo} N date/``（按 N 编号，不是 run_tag
    seed），不走这条 legacy discovery；如需聚合解耦 record，请按 combo + N 直接读
    record 目录。
    """
    return _build_persistent_dir_index().get(run_tag)


def _read_status(progress_dir: str) -> Dict[str, Any]:
    path = os.path.join(progress_dir, "blb_stage2_status.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _read_best_action_full(progress_dir: str) -> Dict[str, Any]:
    path = os.path.join(progress_dir, "blb_stage2_best_action_full.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _read_final_eval_results(persistent_dir: str) -> Dict[str, Any]:
    """Look for Paean's blb_action_final_eval_results_*.json under
    final_eval/ in the run directory."""
    latest_path = None
    latest_mtime = None
    for dirpath, _, filenames in os.walk(persistent_dir):
        for filename in filenames:
            if not filename.startswith("blb_action_final_eval_results_"):
                continue
            if not filename.endswith(".json"):
                continue
            path = os.path.join(dirpath, filename)
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            if latest_mtime is None or mtime > latest_mtime:
                latest_path = path
                latest_mtime = mtime

    if latest_path is None:
        return {}

    try:
        with open(latest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _extract_final_eval_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Pull the headline candidate's metrics from Paean's results.json."""
    out: Dict[str, Any] = {}
    cands = results.get("candidate_results") or []
    if not cands:
        return out
    # First candidate is the BLB best action (or the user-pinned one).
    head = cands[0]
    out["final_eval_loss"] = head.get("loss")
    out["final_eval_metric1"] = head.get("p")
    out["final_eval_metric2"] = head.get("s")
    out["total_bits_sum"] = head.get("total_bits_sum")
    out["fusion_count"] = head.get("total_fusion_count")
    out["avg_truncation_k"] = head.get("avg_truncation_k")
    return out


def _read_invalid_rate_last50(diag_summary: str) -> Optional[float]:
    if not os.path.isfile(diag_summary):
        return None
    try:
        with open(diag_summary, "r", encoding="utf-8") as f:
            tail = ""
            while True:
                chunk = f.read(_DIAGNOSTICS_SCAN_CHUNK_SIZE)
                if not chunk:
                    return None
                text = tail + chunk
                match = _INVALID_RATE_LAST50_RE.search(text)
                if match:
                    return float(match.group(1)) / 59.0
                tail = text[-256:]
    except Exception:
        return None


def _gather_one_seed(seed: int, run_tag: str, persistent_index: Optional[Dict[str, str]] = None) -> SeedRow:
    row = SeedRow(seed=seed, run_tag=run_tag)
    if persistent_index is None:
        persistent_dir = _find_persistent_dir(run_tag)
    else:
        persistent_dir = persistent_index.get(run_tag)
    if persistent_dir is None:
        row.status = "missing"
        row.error_msg = f"no persistent dir matching __${run_tag} under Parting Chapter/persistent/"
        return row
    row.persistent_dir = persistent_dir

    progress_dir = os.path.join(persistent_dir, "blb_stage2", "progress")
    if not os.path.isdir(progress_dir):
        row.status = "incomplete"
        row.error_msg = f"blb_stage2/progress/ not found under {persistent_dir}"
        return row

    status = _read_status(progress_dir)
    row.completed_episodes = status.get("completed_episodes")
    best = status.get("best") or {}
    row.best_reward = best.get("reward")
    breakdown = (status.get("last_breakdown") or {})
    if isinstance(breakdown, dict):
        row.total_bits_sum = breakdown.get("total_bits_sum")
        row.fusion_count = breakdown.get("fusion_count")

    final_eval = _read_final_eval_results(persistent_dir)
    if final_eval:
        for k, v in _extract_final_eval_metrics(final_eval).items():
            setattr(row, k, v)

    # Pull invalid_rate from diagnostics summary if present.
    diag_summary = os.path.join(progress_dir, "diagnostics", "diagnostics_summary.md")
    invalid_rate = _read_invalid_rate_last50(diag_summary)
    if invalid_rate is not None:
        row.invalid_rate_last50 = invalid_rate

    if row.best_reward is None:
        row.status = "incomplete"
    elif row.final_eval_loss is None:
        row.status = "training_only"  # ran, no final eval landed
    else:
        row.status = "complete"
    return row


def _mean_std_str(values: List[float], fmt: str = "+.4f") -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:{fmt}} (n=1)"
    n = len(values)
    m = float(math.fsum(values)) / float(n)
    variance = math.fsum((float(value) - m) ** 2 for value in values) / float(n - 1)
    s = math.sqrt(variance)
    return f"{m:{fmt}} ± {s:{fmt}} (n={len(values)})"


def _iter_summary_md_lines(run_name: str, rows: List[SeedRow]) -> Iterable[str]:
    yield f"# Multi-seed summary · `{run_name}`"
    yield ""
    yield f"- 总 seed 数：{len(rows)}"
    n_complete = sum(1 for r in rows if r.status == "complete")
    n_training = sum(1 for r in rows if r.status == "training_only")
    n_incomplete = sum(1 for r in rows if r.status in ("incomplete", "missing"))
    yield f"- 完成 (训练 + final-eval)：**{n_complete}**"
    yield f"- 仅训练完（无 final-eval）：{n_training}"
    yield f"- 未完成 / 缺失：{n_incomplete}"
    yield ""

    # Aggregate
    yield "## 1. 聚合指标（mean ± std，跨 complete seeds）"
    yield ""
    complete = [r for r in rows if r.status == "complete"]
    yield "| 指标 | 值 |"
    yield "|------|------|"
    yield f"| Training best reward | {_mean_std_str([float(r.best_reward) for r in complete if r.best_reward is not None])} |"
    yield f"| Final-eval loss | {_mean_std_str([float(r.final_eval_loss) for r in complete if r.final_eval_loss is not None])} |"
    yield f"| Final-eval metric1 | {_mean_std_str([float(r.final_eval_metric1) for r in complete if r.final_eval_metric1 is not None])} |"
    yield f"| Final-eval metric2 | {_mean_std_str([float(r.final_eval_metric2) for r in complete if r.final_eval_metric2 is not None])} |"
    yield f"| Total bits sum | {_mean_std_str([float(r.total_bits_sum) for r in complete if r.total_bits_sum is not None], fmt='.0f')} |"
    yield f"| Fusion count | {_mean_std_str([float(r.fusion_count) for r in complete if r.fusion_count is not None], fmt='.1f')} |"
    yield f"| Avg truncation K | {_mean_std_str([float(r.avg_truncation_k) for r in complete if r.avg_truncation_k is not None], fmt='.2f')} |"
    yield f"| Invalid rate (last 50 ep) | {_mean_std_str([float(r.invalid_rate_last50) * 100 for r in complete if r.invalid_rate_last50 is not None], fmt='.1f')} % |"
    yield ""

    # Per-seed rows
    yield "## 2. Per-seed 明细"
    yield ""
    yield "| Seed | Status | Completed eps | Best reward | Final loss | Final metric1 | Final metric2 | Bits | Fusion | avg_k | Persistent dir |"
    yield "|----:|:------|--------------:|------------:|-----------:|--------------:|--------------:|-----:|-------:|------:|:----------------|"
    for r in sorted(rows, key=lambda x: x.seed):
        dir_short = ""
        if r.persistent_dir:
            dir_short = r.persistent_dir.split("/")[-1]
        yield (
            f"| {r.seed} | {r.status} | "
            f"{r.completed_episodes if r.completed_episodes is not None else ''} | "
            f"{(f'{r.best_reward:+.4f}' if r.best_reward is not None else '')} | "
            f"{(f'{r.final_eval_loss:.4f}' if r.final_eval_loss is not None else '')} | "
            f"{(f'{r.final_eval_metric1:.4f}' if r.final_eval_metric1 is not None else '')} | "
            f"{(f'{r.final_eval_metric2:.4f}' if r.final_eval_metric2 is not None else '')} | "
            f"{(r.total_bits_sum if r.total_bits_sum is not None else '')} | "
            f"{(r.fusion_count if r.fusion_count is not None else '')} | "
            f"{(f'{r.avg_truncation_k:.2f}' if r.avg_truncation_k is not None else '')} | "
            f"`{dir_short}` |"
        )
    yield ""

    # Errors
    errs = [r for r in rows if r.error_msg]
    if errs:
        yield "## 3. 错误日志"
        yield ""
        for r in errs:
            yield f"- seed={r.seed} run_tag=`{r.run_tag}` → {r.error_msg}"
        yield ""

    # Significance hint (paired bootstrap suggestion)
    if n_complete >= 3:
        yield "## 4. 统计建议"
        yield ""
        yield "- 至少有 3 个 complete seed → 可以做 paired bootstrap 跟某个外部 baseline 比。"
        yield "- 论文报数最少 5 seeds；目前 complete=%d，建议补到 5 以上后再 freeze 数字。" % n_complete
        yield ""


def _build_summary_md(run_name: str, rows: List[SeedRow]) -> str:
    return "\n".join(_iter_summary_md_lines(run_name, rows))


def _write_summary_md(path: str, run_name: str, rows: List[SeedRow]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        first = True
        for line in _iter_summary_md_lines(run_name, rows):
            if first:
                first = False
            else:
                f.write("\n")
            f.write(line)


def _seed_row_json_dict(row: SeedRow) -> Dict[str, Any]:
    return {name: getattr(row, name) for name in row.__dataclass_fields__}


def _write_summary_json(path: str, rows: List[SeedRow]) -> None:
    encoder = json.JSONEncoder(indent=2, ensure_ascii=False)
    with open(path, "w", encoding="utf-8") as f:
        f.write("[")
        first = True
        for row in rows:
            if first:
                first = False
                f.write("\n")
            else:
                f.write(",\n")
            for chunk in encoder.iterencode(_seed_row_json_dict(row)):
                f.write(chunk)
        if first:
            f.write("]")
        else:
            f.write("\n]")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--seed-list", required=True,
                    help="Text file, each line '<seed> <run_tag>'")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args(argv)

    seed_specs: List[Tuple[int, str]] = []
    with open(args.seed_list, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            seed = int(parts[0])
            run_tag = parts[1]
            seed_specs.append((seed, run_tag))

    persistent_index = _build_persistent_dir_index(
        requested_run_tags={run_tag for _, run_tag in seed_specs}
    )
    seed_rows = [
        _gather_one_seed(seed, run_tag, persistent_index=persistent_index)
        for seed, run_tag in seed_specs
    ]

    os.makedirs(args.output_dir, exist_ok=True)
    md_path = os.path.join(args.output_dir, "seed_summary.md")
    _write_summary_md(md_path, args.run_name, seed_rows)

    json_path = os.path.join(args.output_dir, "seed_summary.json")
    _write_summary_json(json_path, seed_rows)

    print(f"Wrote: {md_path}")
    print(f"Wrote: {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
