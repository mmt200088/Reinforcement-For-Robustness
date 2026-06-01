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
import glob
import json
import os
import statistics
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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


def _find_persistent_dir(run_tag: str) -> Optional[str]:
    """Walk the persistent root looking for a slug ending in ``__<run_tag>``.

    注：这是给旧 ``persistent/`` 多 seed sweep（``--run-tag``）用的。解耦后新布局
    （2026-06-01）每个 combo 只一个工作目录，多次运行归档到
    ``Parting Chapter/stage{1,2}/record/{combo} N date/``（按 N 编号，不是 run_tag
    seed），不走这条 glob；如需聚合解耦 record，请按 combo + N 直接读 record 目录。
    """
    root = "Parting Chapter/persistent"
    if not os.path.isdir(root):
        return None
    matches = glob.glob(f"{root}/*/*/*/*__{run_tag}")
    if not matches:
        return None
    # If multiple matches, prefer the most recently modified one.
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return matches[0]


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
    candidates = glob.glob(f"{persistent_dir}/**/blb_action_final_eval_results_*.json", recursive=True)
    if not candidates:
        return {}
    # Most recent one.
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    try:
        with open(candidates[0], "r", encoding="utf-8") as f:
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


def _gather_one_seed(seed: int, run_tag: str) -> SeedRow:
    row = SeedRow(seed=seed, run_tag=run_tag)
    persistent_dir = _find_persistent_dir(run_tag)
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
    if os.path.isfile(diag_summary):
        try:
            with open(diag_summary, "r", encoding="utf-8") as f:
                txt = f.read()
            # Look for "最近 50 回合 mean invalid 子步数: **N.NN**" pattern.
            import re
            m = re.search(r"最近 50 回合 mean invalid 子步数: \*\*([0-9.]+)\*\*", txt)
            if m:
                row.invalid_rate_last50 = float(m.group(1)) / 59.0  # normalize to fraction
        except Exception:
            pass

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
    m = statistics.mean(values)
    s = statistics.stdev(values)
    return f"{m:{fmt}} ± {s:{fmt}} (n={len(values)})"


def _build_summary_md(run_name: str, rows: List[SeedRow]) -> str:
    lines: List[str] = []
    lines.append(f"# Multi-seed summary · `{run_name}`")
    lines.append("")
    lines.append(f"- 总 seed 数：{len(rows)}")
    n_complete = sum(1 for r in rows if r.status == "complete")
    n_training = sum(1 for r in rows if r.status == "training_only")
    n_incomplete = sum(1 for r in rows if r.status in ("incomplete", "missing"))
    lines.append(f"- 完成 (训练 + final-eval)：**{n_complete}**")
    lines.append(f"- 仅训练完（无 final-eval）：{n_training}")
    lines.append(f"- 未完成 / 缺失：{n_incomplete}")
    lines.append("")

    # Aggregate
    lines.append("## 1. 聚合指标（mean ± std，跨 complete seeds）")
    lines.append("")
    complete = [r for r in rows if r.status == "complete"]
    lines.append("| 指标 | 值 |")
    lines.append("|------|------|")
    lines.append(f"| Training best reward | {_mean_std_str([float(r.best_reward) for r in complete if r.best_reward is not None])} |")
    lines.append(f"| Final-eval loss | {_mean_std_str([float(r.final_eval_loss) for r in complete if r.final_eval_loss is not None])} |")
    lines.append(f"| Final-eval metric1 | {_mean_std_str([float(r.final_eval_metric1) for r in complete if r.final_eval_metric1 is not None])} |")
    lines.append(f"| Final-eval metric2 | {_mean_std_str([float(r.final_eval_metric2) for r in complete if r.final_eval_metric2 is not None])} |")
    lines.append(f"| Total bits sum | {_mean_std_str([float(r.total_bits_sum) for r in complete if r.total_bits_sum is not None], fmt='.0f')} |")
    lines.append(f"| Fusion count | {_mean_std_str([float(r.fusion_count) for r in complete if r.fusion_count is not None], fmt='.1f')} |")
    lines.append(f"| Avg truncation K | {_mean_std_str([float(r.avg_truncation_k) for r in complete if r.avg_truncation_k is not None], fmt='.2f')} |")
    lines.append(f"| Invalid rate (last 50 ep) | {_mean_std_str([float(r.invalid_rate_last50) * 100 for r in complete if r.invalid_rate_last50 is not None], fmt='.1f')} % |")
    lines.append("")

    # Per-seed rows
    lines.append("## 2. Per-seed 明细")
    lines.append("")
    lines.append("| Seed | Status | Completed eps | Best reward | Final loss | Final metric1 | Final metric2 | Bits | Fusion | avg_k | Persistent dir |")
    lines.append("|----:|:------|--------------:|------------:|-----------:|--------------:|--------------:|-----:|-------:|------:|:----------------|")
    for r in sorted(rows, key=lambda x: x.seed):
        dir_short = ""
        if r.persistent_dir:
            dir_short = r.persistent_dir.split("/")[-1]
        lines.append(
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
    lines.append("")

    # Errors
    errs = [r for r in rows if r.error_msg]
    if errs:
        lines.append("## 3. 错误日志")
        lines.append("")
        for r in errs:
            lines.append(f"- seed={r.seed} run_tag=`{r.run_tag}` → {r.error_msg}")
        lines.append("")

    # Significance hint (paired bootstrap suggestion)
    if n_complete >= 3:
        lines.append("## 4. 统计建议")
        lines.append("")
        lines.append(
            "- 至少有 3 个 complete seed → 可以做 paired bootstrap 跟某个外部 baseline 比。"
        )
        lines.append(
            "- 论文报数最少 5 seeds；目前 complete=%d，建议补到 5 以上后再 freeze 数字。" % n_complete
        )
        lines.append("")

    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--seed-list", required=True,
                    help="Text file, each line '<seed> <run_tag>'")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args(argv)

    seed_rows: List[SeedRow] = []
    with open(args.seed_list, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            seed = int(parts[0])
            run_tag = parts[1]
            seed_rows.append(_gather_one_seed(seed, run_tag))

    os.makedirs(args.output_dir, exist_ok=True)
    md = _build_summary_md(args.run_name, seed_rows)
    md_path = os.path.join(args.output_dir, "seed_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    json_path = os.path.join(args.output_dir, "seed_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in seed_rows], f, indent=2, ensure_ascii=False)

    print(f"Wrote: {md_path}")
    print(f"Wrote: {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
