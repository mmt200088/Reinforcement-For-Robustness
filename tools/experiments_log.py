"""Single source of truth for all RL / GA / search runs across this project.

Maintains two artifacts at `experiments/`:

    experiments/registry.jsonl     -- append-only, one record per run
    experiments/index.md           -- auto-regenerated index (last N runs,
                                       grouped by date, with best metric +
                                       links to artifacts)

Three CLI subcommands:

    register     -- append a new run record (called automatically at end
                    of training from sequential_runner.py)
    rebuild      -- regenerate `index.md` from current `registry.jsonl`
    query        -- filter / list runs from CLI (e.g. "all completed
                    runs on mrpc with best_reward > 0.5")

Schema (one JSONL row)::

    {
      "run_id":          "20260516_022031_pid12345",  // unique
      "registered_at":   "2026-05-16T02:21:08",
      "git_commit":      "abc1234",
      "git_dirty":       false,
      "dataset":         "mrpc",
      "model_type":      "bert-base",
      "algorithm":       "rl",                         // rl / ga / greedy / general-rl
      "preset":          "mrpc-blb-stage2-rl",
      "rl_variant":      "blb_v3_sequential",
      "seed":            42,
      "status":          "complete",                   // complete / training_only / crashed
      "elapsed_sec":     6312.5,
      "completed_episodes": 6000,
      "total_episodes_planned": 6000,
      "best_reward":     0.4521,
      "final_eval": {
        "loss":          0.3812,
        "metric1":       0.8623,
        "metric2":       0.9012,
        "total_bits":    14110,
        "fusion_count":  158,
        "avg_truncation_k": 11.34
      },
      "persistent_dir":  "Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.005..._myrun_s42",
      "artifact_paths": {
        "best_action_full_json": "...",
        "best_action_full_md":   "...",
        "report_md":             "...",
        "diagnostics_summary":   "..."
      },
      "notes":           ""                              // free-form, user-editable
    }

Design notes
------------
* Records are append-only. A run that gets re-run with the same seed produces
  a NEW row (old row stays); the index shows the latest by run_id timestamp.
* `git_dirty=true` means uncommitted changes existed at register time —
  important provenance for paper claims.
* `notes` is user-editable: open `registry.jsonl` and add a sentence later.
* The CLI ``query`` subcommand supports basic filters; for complex analysis,
  read the JSONL directly with jq / pandas.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


REGISTRY_REL = "experiments/registry.jsonl"
INDEX_REL = "experiments/index.md"


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _git_info() -> Dict[str, Any]:
    """Best-effort git commit + dirty flag. Empty on failure."""
    out: Dict[str, Any] = {"git_commit": "", "git_dirty": False}
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()
        out["git_commit"] = sha
    except Exception:
        pass
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode()
        out["git_dirty"] = bool(status.strip())
    except Exception:
        pass
    return out


def _load_records(registry_path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(registry_path):
        return []
    out: List[Dict[str, Any]] = []
    with open(registry_path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            try:
                out.append(json.loads(t))
            except Exception:
                pass
    return out


def _append_record(registry_path: str, record: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(registry_path) or ".", exist_ok=True)
    with open(registry_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# register: called from sequential_runner.py (or manually).
# ---------------------------------------------------------------------------

def register(
        *,
        run_id: str,
        dataset: str,
        model_type: str,
        algorithm: str,
        preset: str = "",
        rl_variant: str = "",
        seed: Optional[int] = None,
        status: str = "complete",
        elapsed_sec: float = 0.0,
        completed_episodes: Optional[int] = None,
        total_episodes_planned: Optional[int] = None,
        best_reward: Optional[float] = None,
        final_eval: Optional[Mapping[str, Any]] = None,
        persistent_dir: str = "",
        record_dir: str = "",
        artifact_paths: Optional[Mapping[str, str]] = None,
        notes: str = "",
        registry_path: str = REGISTRY_REL,
        ) -> Dict[str, Any]:
    """Append a record to the registry and rebuild the index.

    Idempotent in the sense that re-calling with the same run_id appends a
    NEW row (old row preserved); the index shows the most-recent row per
    run_id.
    """
    record: Dict[str, Any] = {
        "run_id": str(run_id),
        "registered_at": _now_iso(),
        **_git_info(),
        "dataset": str(dataset),
        "model_type": str(model_type),
        "algorithm": str(algorithm),
        "preset": str(preset),
        "rl_variant": str(rl_variant),
        "seed": (int(seed) if seed is not None else None),
        "status": str(status),
        "elapsed_sec": float(elapsed_sec),
        "completed_episodes": (int(completed_episodes) if completed_episodes is not None else None),
        "total_episodes_planned": (int(total_episodes_planned) if total_episodes_planned is not None else None),
        "best_reward": (float(best_reward) if best_reward is not None else None),
        "final_eval": dict(final_eval) if final_eval else None,
        "persistent_dir": str(persistent_dir),
        # 解耦布局（2026-06-01）：完成时归档的 record 目录（stage{1,2}/record/{combo N date}）。
        "record_dir": str(record_dir),
        "artifact_paths": dict(artifact_paths) if artifact_paths else {},
        "notes": str(notes),
    }
    _append_record(registry_path, record)
    try:
        _rebuild_index(registry_path)
    except Exception as exc:
        sys.stderr.write(f"[experiments_log] index rebuild failed: {exc}\n")
    return record


# ---------------------------------------------------------------------------
# rebuild: regenerate index.md from current registry.jsonl
# ---------------------------------------------------------------------------

def _latest_per_run_id(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for r in records:
        rid = str(r.get("run_id", ""))
        if not rid:
            continue
        prev = by_id.get(rid)
        if prev is None:
            by_id[rid] = dict(r)
            continue
        # keep the later-registered one
        if str(r.get("registered_at", "")) >= str(prev.get("registered_at", "")):
            by_id[rid] = dict(r)
    return list(by_id.values())


def _md_row(r: Mapping[str, Any]) -> str:
    final = r.get("final_eval") or {}
    best_r = r.get("best_reward")
    final_loss = final.get("loss") if isinstance(final, dict) else None
    final_m1 = final.get("metric1") if isinstance(final, dict) else None
    dir_short = ""
    if r.get("record_dir"):
        dir_short = str(r["record_dir"]).rstrip("/").split("/")[-1]
    elif r.get("persistent_dir"):
        dir_short = str(r["persistent_dir"]).split("/")[-1]
    elapsed_h = (float(r.get("elapsed_sec", 0) or 0) / 3600.0)
    git_flag = "⚠dirty" if r.get("git_dirty") else r.get("git_commit", "")
    return (
        f"| {r.get('run_id','')[:19]} | {r.get('dataset','')} | "
        f"{r.get('algorithm','')} | {r.get('preset','')} | "
        f"{r.get('seed') if r.get('seed') is not None else ''} | "
        f"{r.get('status','')} | "
        f"{elapsed_h:.2f}h | "
        f"{(f'{best_r:+.4f}' if isinstance(best_r,(int,float)) else '')} | "
        f"{(f'{final_loss:.4f}' if isinstance(final_loss,(int,float)) else '')} | "
        f"{(f'{final_m1:.4f}' if isinstance(final_m1,(int,float)) else '')} | "
        f"`{git_flag}` | "
        f"`{dir_short}` |"
    )


def _rebuild_index(registry_path: str = REGISTRY_REL, index_path: str = INDEX_REL) -> str:
    records = _load_records(registry_path)
    latest = _latest_per_run_id(records)
    latest.sort(key=lambda r: str(r.get("registered_at", "")), reverse=True)

    lines: List[str] = []
    lines.append("# Experiments index")
    lines.append("")
    lines.append(
        f"_Auto-generated from `{registry_path}` on {_now_iso()}. "
        "Edit `notes` field in registry.jsonl to annotate a run; rerun "
        "`python3 tools/experiments_log.py rebuild` to refresh._"
    )
    lines.append("")
    lines.append(f"- Total registered run_ids: **{len(latest)}**")
    by_status: Dict[str, int] = {}
    for r in latest:
        by_status[str(r.get("status", "unknown"))] = by_status.get(str(r.get("status", "unknown")), 0) + 1
    if by_status:
        lines.append(f"- By status: " + ", ".join(f"{k}={v}" for k, v in sorted(by_status.items())))
    by_dataset: Dict[str, int] = {}
    for r in latest:
        by_dataset[str(r.get("dataset", ""))] = by_dataset.get(str(r.get("dataset", "")), 0) + 1
    if by_dataset:
        lines.append(f"- By dataset: " + ", ".join(f"{k}={v}" for k, v in sorted(by_dataset.items())))
    lines.append("")

    # Best-by-dataset summary
    lines.append("## Best so far (per dataset)")
    lines.append("")
    lines.append("| Dataset | Best reward | Final loss | Final metric1 | Run ID |")
    lines.append("|---|---:|---:|---:|---|")
    by_ds: Dict[str, List[Dict[str, Any]]] = {}
    for r in latest:
        if r.get("status") not in ("complete", "training_only"):
            continue
        by_ds.setdefault(str(r.get("dataset", "")), []).append(r)
    for ds, items in sorted(by_ds.items()):
        items.sort(key=lambda r: float(r.get("best_reward") or float("-inf")), reverse=True)
        if not items:
            continue
        top = items[0]
        final = top.get("final_eval") or {}
        best_r_val = top.get("best_reward")
        final_loss_val = final.get("loss") if isinstance(final, dict) else None
        final_m1_val = final.get("metric1") if isinstance(final, dict) else None
        best_r_str = f"{best_r_val:+.4f}" if isinstance(best_r_val, (int, float)) else ""
        final_loss_str = f"{final_loss_val:.4f}" if isinstance(final_loss_val, (int, float)) else ""
        final_m1_str = f"{final_m1_val:.4f}" if isinstance(final_m1_val, (int, float)) else ""
        top_run_id = str(top.get("run_id", ""))[:19]
        lines.append(
            f"| {ds} | {best_r_str} | {final_loss_str} | {final_m1_str} | `{top_run_id}` |"
        )
    lines.append("")

    # All runs (most recent first)
    lines.append("## All runs (most recent first)")
    lines.append("")
    lines.append("| Run ID | Dataset | Algo | Preset | Seed | Status | Time | Best | Loss | Metric1 | Git | Persistent |")
    lines.append("|---|---|---|---|---:|---|---:|---:|---:|---:|---|---|")
    for r in latest:
        lines.append(_md_row(r))
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("**How to use this file**:")
    lines.append("")
    lines.append("- 想看某个具体 run 的细节：去 `persistent` 列对应的目录，看 `blb_stage2_best_action_full.md` / `diagnostics/diagnostics_summary.md`。")
    lines.append("- 想做 cross-run 对比：用 `python3 tools/experiments_log.py query --dataset mrpc --min-reward 0.4`。")
    lines.append("- 想给某个 run 加注释：直接编辑 `registry.jsonl` 那一行的 `notes` 字段，然后 `python3 tools/experiments_log.py rebuild`。")

    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return index_path


# ---------------------------------------------------------------------------
# query: filter records from CLI
# ---------------------------------------------------------------------------

def _query(
        *,
        dataset: Optional[str] = None,
        algorithm: Optional[str] = None,
        preset_substr: Optional[str] = None,
        status: Optional[str] = None,
        min_reward: Optional[float] = None,
        last_n: Optional[int] = None,
        registry_path: str = REGISTRY_REL,
        ) -> List[Dict[str, Any]]:
    records = _latest_per_run_id(_load_records(registry_path))
    records.sort(key=lambda r: str(r.get("registered_at", "")), reverse=True)
    out: List[Dict[str, Any]] = []
    for r in records:
        if dataset and r.get("dataset") != dataset:
            continue
        if algorithm and r.get("algorithm") != algorithm:
            continue
        if preset_substr and preset_substr.lower() not in str(r.get("preset", "")).lower():
            continue
        if status and r.get("status") != status:
            continue
        if min_reward is not None:
            br = r.get("best_reward")
            if not isinstance(br, (int, float)) or float(br) < float(min_reward):
                continue
        out.append(r)
        if last_n and len(out) >= int(last_n):
            break
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    # register
    ap_reg = sub.add_parser("register", help="Append a new run record")
    ap_reg.add_argument("--run-id", required=True)
    ap_reg.add_argument("--dataset", required=True)
    ap_reg.add_argument("--model-type", default="bert-base")
    ap_reg.add_argument("--algorithm", default="rl")
    ap_reg.add_argument("--preset", default="")
    ap_reg.add_argument("--rl-variant", default="")
    ap_reg.add_argument("--seed", type=int, default=None)
    ap_reg.add_argument("--status", default="complete")
    ap_reg.add_argument("--elapsed-sec", type=float, default=0.0)
    ap_reg.add_argument("--completed-episodes", type=int, default=None)
    ap_reg.add_argument("--total-episodes-planned", type=int, default=None)
    ap_reg.add_argument("--best-reward", type=float, default=None)
    ap_reg.add_argument("--final-eval-json", default="",
                        help="Inline JSON with final_eval metrics")
    ap_reg.add_argument("--persistent-dir", default="")
    ap_reg.add_argument("--artifact-paths-json", default="",
                        help="Inline JSON dict of artifact_paths")
    ap_reg.add_argument("--notes", default="")
    ap_reg.add_argument("--registry-path", default=REGISTRY_REL)

    # rebuild
    ap_re = sub.add_parser("rebuild", help="Regenerate index.md from registry.jsonl")
    ap_re.add_argument("--registry-path", default=REGISTRY_REL)
    ap_re.add_argument("--index-path", default=INDEX_REL)

    # query
    ap_q = sub.add_parser("query", help="Filter records and print as table")
    ap_q.add_argument("--dataset", default=None)
    ap_q.add_argument("--algorithm", default=None)
    ap_q.add_argument("--preset-substr", default=None)
    ap_q.add_argument("--status", default=None)
    ap_q.add_argument("--min-reward", type=float, default=None)
    ap_q.add_argument("--last-n", type=int, default=None)
    ap_q.add_argument("--format", choices=("md", "json", "tsv"), default="md")
    ap_q.add_argument("--registry-path", default=REGISTRY_REL)

    args = ap.parse_args(argv)

    if args.cmd == "register":
        final = json.loads(args.final_eval_json) if args.final_eval_json else None
        artifacts = json.loads(args.artifact_paths_json) if args.artifact_paths_json else None
        rec = register(
            run_id=args.run_id,
            dataset=args.dataset,
            model_type=args.model_type,
            algorithm=args.algorithm,
            preset=args.preset,
            rl_variant=args.rl_variant,
            seed=args.seed,
            status=args.status,
            elapsed_sec=args.elapsed_sec,
            completed_episodes=args.completed_episodes,
            total_episodes_planned=args.total_episodes_planned,
            best_reward=args.best_reward,
            final_eval=final,
            persistent_dir=args.persistent_dir,
            artifact_paths=artifacts,
            notes=args.notes,
            registry_path=args.registry_path,
        )
        print(json.dumps(rec, indent=2, ensure_ascii=False))
        return 0

    if args.cmd == "rebuild":
        path = _rebuild_index(args.registry_path, args.index_path)
        print(f"Wrote: {path}")
        return 0

    if args.cmd == "query":
        rows = _query(
            dataset=args.dataset,
            algorithm=args.algorithm,
            preset_substr=args.preset_substr,
            status=args.status,
            min_reward=args.min_reward,
            last_n=args.last_n,
            registry_path=args.registry_path,
        )
        if args.format == "json":
            print(json.dumps(rows, indent=2, ensure_ascii=False))
        elif args.format == "tsv":
            keys = ("run_id", "dataset", "algorithm", "preset", "seed", "status", "best_reward")
            print("\t".join(keys))
            for r in rows:
                print("\t".join(str(r.get(k, "")) for k in keys))
        else:
            if not rows:
                print("(no matching runs)")
                return 0
            print("| Run ID | Dataset | Algo | Preset | Seed | Status | Time | Best | Loss | Metric1 |")
            print("|---|---|---|---|---:|---|---:|---:|---:|---:|")
            for r in rows:
                final = r.get("final_eval") or {}
                elapsed_h = (float(r.get("elapsed_sec", 0) or 0) / 3600.0)
                best_r = r.get("best_reward")
                final_loss = final.get("loss") if isinstance(final, dict) else None
                final_m1 = final.get("metric1") if isinstance(final, dict) else None
                print(
                    f"| {r.get('run_id','')[:19]} | {r.get('dataset','')} | "
                    f"{r.get('algorithm','')} | {r.get('preset','')} | "
                    f"{r.get('seed') if r.get('seed') is not None else ''} | "
                    f"{r.get('status','')} | "
                    f"{elapsed_h:.2f}h | "
                    f"{(f'{best_r:+.4f}' if isinstance(best_r,(int,float)) else '')} | "
                    f"{(f'{final_loss:.4f}' if isinstance(final_loss,(int,float)) else '')} | "
                    f"{(f'{final_m1:.4f}' if isinstance(final_m1,(int,float)) else '')} |"
                )
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
