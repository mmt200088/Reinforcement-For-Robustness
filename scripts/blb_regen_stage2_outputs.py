#!/usr/bin/env python3
"""离线再生 BLB Stage-2 RL 的「图片 / 检测报告」产物（torch-free 边车工具）。

用途
----
1. 在一个**已完成（或进行中）的 Stage-2 run** 上，按对齐 Stage-1 的新版式重新生成
   训练曲线 + 熵曲线 + 局部最优检测报告——无需重训、无需 torch、无需服务器，
   本地即可肉眼核对。
2. 回填历史 run（这些产物以前要么是旧版式、要么根本没有）。
3. 作为 ``persistence.write_training_curves`` / ``rl_local_optimum`` 的端到端验证手段。

只读输入：``diagnostics/episodes.jsonl(.gz)``、``diagnostics/ppo_updates.jsonl``、
``blb_stage2_status.json``、``blb_stage2_report.md``（取 baseline 参考线，可缺）。
写出（到 ``--out-dir``，默认就是 progress 目录本身）：
``blb_stage2_training_curve.png/.npz``、``blb_stage2_reward_paper.png/.pdf``、
``blb_stage2_entropy_curve.png``、``blb_stage2_search_log.txt``。

用法
----
    python scripts/blb_regen_stage2_outputs.py "Parting Chapter/stage2/bert base mrpc/progress"
    python scripts/blb_regen_stage2_outputs.py <combo_dir>            # 自动找 progress/
    python scripts/blb_regen_stage2_outputs.py <dir> --out-dir /tmp/preview --metric1-name accuracy
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
import re
import sys

# rl_local_optimum 只依赖 numpy（torch-free），直接 import。
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import rl_local_optimum  # noqa: E402


def _load_persistence_module():
    """加载 blb_stage2_rl/persistence.py，但**绕过** ``blb_stage2_rl/__init__``
    （后者 import runner → torch）。这样无 torch 的机器也能出图。"""
    path = os.path.join(_REPO_ROOT, "blb_stage2_rl", "persistence.py")
    spec = importlib.util.spec_from_file_location("blb_persistence_standalone", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _resolve_progress_dir(path: str) -> str:
    """接受 progress 目录、combo 目录，或它们的父目录；返回 progress 目录。"""
    path = os.path.abspath(path)
    cands = [
        path,
        os.path.join(path, "progress"),
    ]
    for c in cands:
        if os.path.isdir(os.path.join(c, "diagnostics")) or os.path.isfile(
            os.path.join(c, "blb_stage2_status.json")
        ):
            return c
    # last resort: a *.../progress under path
    for root, dirs, _files in os.walk(path):
        if os.path.basename(root) == "progress" and os.path.isdir(
            os.path.join(root, "diagnostics")
        ):
            return root
    raise FileNotFoundError(
        f"找不到 Stage-2 progress 目录（需含 diagnostics/ 或 blb_stage2_status.json）：{path}"
    )


def _open_jsonl(progress_dir: str, name: str):
    """优先 .jsonl，其次 .jsonl.gz；都没有返回 None。"""
    plain = os.path.join(progress_dir, "diagnostics", name)
    if os.path.isfile(plain):
        return open(plain, "r", encoding="utf-8")
    gz = plain + ".gz"
    if os.path.isfile(gz):
        return gzip.open(gz, "rt", encoding="utf-8")
    return None


def _read_episodes(progress_dir: str):
    """返回每回合并列序列 dict。total_reward = per_step_sum + terminal_reward。"""
    f = _open_jsonl(progress_dir, "episodes.jsonl")
    series = {k: [] for k in (
        "returns", "losses", "metric1s", "metric2s", "fusion", "k_gain", "priority",
    )}
    if f is None:
        return series
    with f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            total = float(d.get("per_step_sum", 0.0) or 0.0) + float(
                d.get("terminal_reward", 0.0) or 0.0
            )
            series["returns"].append(total)
            series["losses"].append(float(d.get("terminal_loss_mean", 0.0) or 0.0))
            series["metric1s"].append(float(d.get("terminal_metric1_mean", 0.0) or 0.0))
            series["metric2s"].append(float(d.get("terminal_metric2_mean", 0.0) or 0.0))
            series["fusion"].append(float(d.get("fusion_count", 0) or 0))
            series["k_gain"].append(float(d.get("terminal_k_gain", 0.0) or 0.0))
            series["priority"].append(int(d.get("terminal_priority", 0) or 0))
    return series


def _read_entropy(progress_dir: str):
    """ppo_updates.jsonl → (entropy_series, completed_episodes)。"""
    f = _open_jsonl(progress_dir, "ppo_updates.jsonl")
    ent, eps = [], []
    if f is None:
        return ent, eps
    with f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("entropy") is None:
                continue
            ent.append(float(d["entropy"]))
            eps.append(float(d.get("completed_episodes", len(ent) * 1) or len(ent)))
    return ent, eps


_BASELINE_KEYS = ("loss_mean", "metric1_mean", "metric2_mean", "avg_k")


def _parse_baselines(progress_dir: str):
    """从 blb_stage2_report.md §3 baseline 表解析参考线值（缺失则返回部分/空）。"""
    out = {}
    report = os.path.join(progress_dir, "blb_stage2_report.md")
    if os.path.isfile(report):
        try:
            text = open(report, "r", encoding="utf-8").read()
            for key in _BASELINE_KEYS:
                m = re.search(rf"`{re.escape(key)}`\s*\|\s*([-\d.eE+]+)", text)
                if m:
                    out[key] = float(m.group(1))
        except Exception:
            pass
    # diagnostics_summary.md 兜底 avg_k
    if "avg_k" not in out:
        summ = os.path.join(progress_dir, "diagnostics", "diagnostics_summary.md")
        if os.path.isfile(summ):
            try:
                m = re.search(r"baseline avg_k.*?\*\*([\d.]+)\*\*",
                              open(summ, "r", encoding="utf-8").read())
                if m:
                    out["avg_k"] = float(m.group(1))
            except Exception:
                pass
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Regenerate Stage-1-aligned Stage-2 RL outputs (torch-free).")
    ap.add_argument("progress_dir", help="Stage-2 progress/ dir, or a combo dir containing progress/.")
    ap.add_argument("--out-dir", default=None, help="Where to write artifacts (default: the progress dir itself).")
    ap.add_argument("--metric1-name", default="metric1", help="Label for the metric1 panel (e.g. accuracy).")
    ap.add_argument("--metric2-name", default="metric2", help="Label for the metric2 panel (e.g. f1).")
    ap.add_argument("--ma-window", type=int, default=None, help="Moving-average window (default: auto).")
    args = ap.parse_args(argv)

    progress_dir = _resolve_progress_dir(args.progress_dir)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else progress_dir
    os.makedirs(out_dir, exist_ok=True)
    persistence = _load_persistence_module()

    print(f"[regen] progress dir : {progress_dir}")
    print(f"[regen] output  dir : {out_dir}")

    ep = _read_episodes(progress_dir)
    n = len(ep["returns"])
    if n == 0:
        print("[regen][ERROR] episodes.jsonl 为空或缺失，无法出图。")
        return 2
    ent, ent_eps = _read_entropy(progress_dir)
    baselines = _parse_baselines(progress_dir)
    base_avg_k = float(baselines.get("avg_k", 13.0))
    avg_ks = [base_avg_k - kg for kg in ep["k_gain"]]
    print(f"[regen] episodes={n}  ppo_updates(entropy)={len(ent)}  baselines={baselines}")

    curve_paths = persistence.write_training_curves(
        out_dir,
        episode_returns=ep["returns"],
        episode_losses=ep["losses"],
        episode_metric1s=ep["metric1s"],
        episode_metric2s=ep["metric2s"],
        episode_fusion_counts=ep["fusion"],
        episode_avg_ks=avg_ks,
        baselines={
            "loss": baselines.get("loss_mean"),
            "metric1": baselines.get("metric1_mean"),
            "metric2": baselines.get("metric2_mean"),
            "avg_k": base_avg_k,
        },
        metric1_name=args.metric1_name,
        metric2_name=args.metric2_name,
        entropy_series=ent or None,
        entropy_episodes=ent_eps or None,
        ma_window=args.ma_window,
        log_fn=print,
    )
    for k, v in curve_paths.items():
        if v:
            print(f"[regen]   {k:11s} → {v}  ({os.path.getsize(v)} bytes)")

    # 局部最优 / 健康检测报告（Stage-1 同款版式）。
    report_path = rl_local_optimum.write_local_optimum_report(
        os.path.join(out_dir, persistence.BLB_SEARCH_LOG_TXT),
        episode_returns=ep["returns"],
        episode_entropies=ent or None,
        best_score_history=None,
        completed_episodes=n,
        title="BLB Stage-2 RL",
        extra_lines=[
            "",
            "--- 优先级分布（priority histogram）---",
            f"  P1(acc): {sum(1 for p in ep['priority'] if p == 1)}",
            f"  P2(stab): {sum(1 for p in ep['priority'] if p == 2)}",
            f"  P3(cost): {sum(1 for p in ep['priority'] if p == 3)}",
        ],
        log_fn=print,
    )
    if report_path:
        print(f"[regen]   search_log  → {report_path}")
    print("[regen] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
