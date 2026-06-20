"""End-to-end KV-cache A/B: compare two fusion-run ``episodes.jsonl`` (OFF vs ON).

The production benchmark (``blb_kvcache_benchmark.py``) proves the rollout FORWARD
is equivalent + faster in isolation. This script confirms it END-TO-END on a real
short fusion run: the KV-cache must (a) not change the search quality (same seed →
near-identical reward / priority / fusion distributions, since the path is
float-equivalent not bit-identical) and (b) measurably cut the per-episode rollout
wall-time. Reads the two runs' diagnostics ``episodes.jsonl`` and prints a verdict.

Usage:
  python scripts/blb_kvcache_ab_compare.py --off OFF/episodes.jsonl --on ON/episodes.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from typing import Dict, List


def _load(path: str) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def _col(rows: List[dict], key: str) -> List[float]:
    out = []
    for r in rows:
        v = r.get(key)
        if v is not None:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                pass
    return out


def _stat(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {"n": 0, "mean": float("nan"), "median": float("nan")}
    return {"n": len(xs), "mean": st.fmean(xs), "median": st.median(xs)}


def _frac(rows: List[dict], key: str, val: int) -> float:
    xs = [int(r[key]) for r in rows if r.get(key) is not None]
    return (sum(1 for x in xs if x == val) / len(xs)) if xs else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", required=True, help="episodes.jsonl from the KV-cache OFF run")
    ap.add_argument("--on", required=True, help="episodes.jsonl from the KV-cache ON run")
    ap.add_argument("--reward-tol", type=float, default=0.05,
                    help="max |mean reward| relative diff to call quality MATCHED")
    args = ap.parse_args()

    off, on = _load(args.off), _load(args.on)
    print(f"[ab] OFF episodes={len(off)}  ON episodes={len(on)}")

    # --- speed: per-episode rollout wall-time (the only thing KV-cache changes) ---
    off_roll = _stat(_col(off, "policy_rollout_wall_seconds"))
    on_roll = _stat(_col(on, "policy_rollout_wall_seconds"))
    print("\n[ab] policy_rollout_wall_seconds (the rollout the KV-cache speeds up):")
    print(f"     OFF median={off_roll['median']:.4f}s mean={off_roll['mean']:.4f}s (n={off_roll['n']})")
    print(f"     ON  median={on_roll['median']:.4f}s mean={on_roll['mean']:.4f}s (n={on_roll['n']})")
    roll_speedup = (off_roll["median"] / on_roll["median"]) if on_roll["median"] else float("nan")
    print(f"     rollout speedup (OFF/ON median) = {roll_speedup:.2f}x")

    # --- end-to-end per-episode wall (if logged) ---
    for k in ("episode_wall_seconds", "terminal_probe_install_wall_seconds"):
        o, n = _stat(_col(off, k)), _stat(_col(on, k))
        if o["n"] and n["n"]:
            sp = (o["median"] / n["median"]) if n["median"] else float("nan")
            print(f"[ab] {k}: OFF median={o['median']:.4f}s ON median={n['median']:.4f}s ({sp:.2f}x)")

    # --- quality: reward / priority / fusion distributions must match (same seed) ---
    print("\n[ab] quality (same seed → distributions should match; path is float-equivalent):")
    off_r, on_r = _stat(_col(off, "terminal_reward")), _stat(_col(on, "terminal_reward"))
    print(f"     terminal_reward: OFF mean={off_r['mean']:.4f}  ON mean={on_r['mean']:.4f}")
    for p in (1, 2, 3):
        print(f"     priority=={p} frac: OFF={_frac(off,'priority',p):.3f}  ON={_frac(on,'priority',p):.3f}")
    off_f, on_f = _stat(_col(off, "terminal_fusion_count")), _stat(_col(on, "terminal_fusion_count"))
    if off_f["n"] and on_f["n"]:
        print(f"     fusion_count: OFF mean={off_f['mean']:.3f}  ON mean={on_f['mean']:.3f}")

    # verdict
    denom = abs(off_r["mean"]) if off_r["mean"] else 1.0
    reward_rel = abs(off_r["mean"] - on_r["mean"]) / max(denom, 1e-9)
    quality_ok = reward_rel <= args.reward_tol
    print("\n[ab] ==== VERDICT ====")
    print(f"     quality {'MATCHED' if quality_ok else 'DIVERGED'} "
          f"(|Δmean reward|/|OFF|={reward_rel:.3f}, tol={args.reward_tol})")
    if roll_speedup >= 1.2:
        sv = f"EFFECTIVE ({roll_speedup:.2f}x rollout)"
    elif roll_speedup > 1.02:
        sv = f"MARGINAL ({roll_speedup:.2f}x)"
    else:
        sv = f"NOT EFFECTIVE ({roll_speedup:.2f}x)"
    print(f"     speedup {sv}")
    return 0 if quality_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
