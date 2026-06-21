"""Generic Stage-2 rollout A/B comparator for two ``episodes.jsonl`` files.

The filename is kept for command compatibility with the older KV-cache checks,
but the comparator is now rollout-feature agnostic. It judges speed from
end-to-end episode throughput, not from ``policy_rollout_wall_seconds`` alone.
That policy timer is still useful diagnostics, but when rollout profiling uses
``torch.cuda.synchronize()`` on a shared GPU it can include sibling worker
terminal-probe work and must not be the final verdict metric.

Usage:
  python3 scripts/blb_kvcache_ab_compare.py --off OFF/episodes.jsonl --on ON/episodes.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence


def is_nan(value: float) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _load(path: str) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def _float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(out):
        return out
    return None


def _int(value) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first(row: dict, keys: Sequence[str]):
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    return None


def _col(rows: Iterable[dict], keys: Sequence[str]) -> List[float]:
    out = []
    for row in rows:
        value = _float(_first(row, keys))
        if value is not None:
            out.append(value)
    return out


def _stat(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {"n": 0, "sum": 0.0, "mean": float("nan"), "median": float("nan")}
    return {
        "n": len(xs),
        "sum": float(sum(xs)),
        "mean": float(st.fmean(xs)),
        "median": float(st.median(xs)),
    }


def _wall_seconds(rows: List[dict], explicit_wall_seconds: Optional[float]) -> float:
    explicit = _float(explicit_wall_seconds)
    if explicit is not None and explicit > 0:
        return explicit

    timestamps = _col(rows, ("timestamp",))
    if len(timestamps) >= 2:
        span = max(timestamps) - min(timestamps)
        if span > 0:
            return float(span)

    episode_walls = _col(rows, ("episode_wall_seconds",))
    if episode_walls:
        total = sum(episode_walls)
        if total > 0:
            return float(total)
    return float("nan")


def _priority_fracs(rows: List[dict]) -> Dict[int, float]:
    priorities = [
        value for value in (
            _int(_first(row, ("terminal_priority", "priority"))) for row in rows
        )
        if value is not None
    ]
    if not priorities:
        return {1: float("nan"), 2: float("nan"), 3: float("nan")}
    denom = float(len(priorities))
    return {p: sum(1 for value in priorities if value == p) / denom for p in (1, 2, 3)}


def _probe_device_totals(rows: List[dict]) -> Dict[str, float]:
    totals: Dict[str, float] = defaultdict(float)
    for row in rows:
        wall = _float(row.get("terminal_probe_wall_seconds"))
        if wall is None or wall <= 0:
            continue
        devices = row.get("terminal_probe_devices")
        counts = row.get("terminal_probe_trial_counts")
        if (
            isinstance(devices, list)
            and isinstance(counts, list)
            and len(devices) == len(counts)
            and devices
        ):
            numeric_counts = [_float(v) or 0.0 for v in counts]
            total_count = sum(numeric_counts)
            if total_count > 0:
                for device, count in zip(devices, numeric_counts):
                    totals[str(device)] += wall * count / total_count
                continue
        totals["unknown"] += wall
    return dict(totals)


def _speedup_time(off_stat: Dict[str, float], on_stat: Dict[str, float]) -> float:
    off_med = off_stat["median"]
    on_med = on_stat["median"]
    if is_nan(off_med) or is_nan(on_med) or on_med <= 0:
        return float("nan")
    return float(off_med / on_med)


def summarize(
        rows: List[dict],
        label: str,
        *,
        wall_seconds: Optional[float] = None,
        ) -> Dict[str, object]:
    wall = _wall_seconds(rows, wall_seconds)
    episodes = len(rows)
    episodes_per_hour = (
        float(episodes) * 3600.0 / wall
        if episodes > 0 and not is_nan(wall) and wall > 0
        else float("nan")
    )
    probe_totals = _probe_device_totals(rows)
    return {
        "label": label,
        "episodes": episodes,
        "wall_seconds": wall,
        "episodes_per_hour": episodes_per_hour,
        "reward": _stat(_col(rows, ("terminal_reward", "total_reward"))),
        "fusion": _stat(_col(rows, ("fusion_count", "terminal_fusion_count"))),
        "priority_frac": _priority_fracs(rows),
        "policy_rollout": _stat(_col(rows, ("policy_rollout_wall_seconds",))),
        "terminal_probe": _stat(_col(rows, ("terminal_probe_wall_seconds",))),
        "terminal_probe_install": _stat(_col(rows, ("terminal_probe_install_wall_seconds",))),
        "per_step_optimizer": _stat(_col(rows, ("per_step_optimizer_wall_seconds",))),
        "terminal_probe_device_seconds": probe_totals,
        "terminal_probe_critical_path_seconds": (
            max(probe_totals.values()) if probe_totals else 0.0
        ),
    }


def _relative_delta(a: float, b: float, *, floor: float = 1.0) -> float:
    if is_nan(a) or is_nan(b):
        return float("nan")
    return abs(a - b) / max(abs(a), floor)


def compare_summaries(
        off: Dict[str, object],
        on: Dict[str, object],
        *,
        reward_tol: float = 0.05,
        priority_tol: float = 0.05,
        fusion_tol: float = 0.10,
        speedup_effective: float = 1.20,
        speedup_marginal: float = 1.10,
        ) -> Dict[str, object]:
    off_reward = off["reward"]["mean"]
    on_reward = on["reward"]["mean"]
    reward_rel = _relative_delta(off_reward, on_reward, floor=1e-9)

    off_priority = off["priority_frac"]
    on_priority = on["priority_frac"]
    priority_deltas = [
        abs(float(off_priority[p]) - float(on_priority[p]))
        for p in (1, 2, 3)
        if not is_nan(float(off_priority[p])) and not is_nan(float(on_priority[p]))
    ]
    priority_delta = max(priority_deltas) if priority_deltas else float("nan")

    off_fusion = off["fusion"]["mean"]
    on_fusion = on["fusion"]["mean"]
    fusion_rel = _relative_delta(off_fusion, on_fusion, floor=1.0)

    quality_ok = (
        (is_nan(reward_rel) or reward_rel <= reward_tol)
        and (is_nan(priority_delta) or priority_delta <= priority_tol)
        and (is_nan(fusion_rel) or fusion_rel <= fusion_tol)
    )

    off_eph = float(off["episodes_per_hour"])
    on_eph = float(on["episodes_per_hour"])
    if is_nan(off_eph) or is_nan(on_eph) or off_eph <= 0:
        end_to_end_speedup = float("nan")
        speed = "UNKNOWN"
    else:
        end_to_end_speedup = on_eph / off_eph
        if end_to_end_speedup >= speedup_effective:
            speed = "EFFECTIVE"
        elif end_to_end_speedup >= speedup_marginal:
            speed = "MARGINAL"
        else:
            speed = "NOT EFFECTIVE"

    return {
        "quality": "MATCHED" if quality_ok else "DIVERGED",
        "reward_rel_delta": reward_rel,
        "priority_max_abs_delta": priority_delta,
        "fusion_rel_delta": fusion_rel,
        "speed": speed,
        "end_to_end_speedup": end_to_end_speedup,
        "policy_rollout_speedup": _speedup_time(off["policy_rollout"], on["policy_rollout"]),
    }


def _fmt(value: float, suffix: str = "") -> str:
    if is_nan(value):
        return "nan"
    return f"{value:.4f}{suffix}"


def _print_arm(summary: Dict[str, object]) -> None:
    print(f"[ab] {summary['label']}:")
    print(
        f"     episodes={summary['episodes']}  "
        f"wall={_fmt(float(summary['wall_seconds']), 's')}  "
        f"episodes/hour={_fmt(float(summary['episodes_per_hour']))}"
    )
    print(
        "     terminal_reward "
        f"mean={_fmt(summary['reward']['mean'])} median={_fmt(summary['reward']['median'])}"
    )
    pr = summary["priority_frac"]
    print(
        "     terminal_priority frac "
        f"P1={_fmt(float(pr[1]))} P2={_fmt(float(pr[2]))} P3={_fmt(float(pr[3]))}"
    )
    print(
        "     fusion_count "
        f"mean={_fmt(summary['fusion']['mean'])} median={_fmt(summary['fusion']['median'])}"
    )
    print(
        "     terminal_probe critical-path bound="
        f"{_fmt(float(summary['terminal_probe_critical_path_seconds']), 's')}"
    )


def _print_component(
        name: str,
        off: Dict[str, object],
        on: Dict[str, object],
        ) -> None:
    o = off[name]
    n = on[name]
    speedup = _speedup_time(o, n)
    print(
        f"     {name:24s} "
        f"OFF sum={_fmt(o['sum'], 's')} mean={_fmt(o['mean'], 's')} med={_fmt(o['median'], 's')} | "
        f"ON sum={_fmt(n['sum'], 's')} mean={_fmt(n['mean'], 's')} med={_fmt(n['median'], 's')} | "
        f"time speedup={_fmt(speedup, 'x')}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", required=True, help="episodes.jsonl from the feature OFF run")
    ap.add_argument("--on", required=True, help="episodes.jsonl from the feature ON run")
    ap.add_argument("--off-wall-seconds", type=float, default=None,
                    help="wrapper wall seconds for OFF; overrides timestamp span")
    ap.add_argument("--on-wall-seconds", type=float, default=None,
                    help="wrapper wall seconds for ON; overrides timestamp span")
    ap.add_argument("--reward-tol", type=float, default=0.05,
                    help="max relative mean terminal_reward delta for quality MATCHED")
    ap.add_argument("--priority-tol", type=float, default=0.05,
                    help="max absolute P1/P2/P3 fraction delta for quality MATCHED")
    ap.add_argument("--fusion-tol", type=float, default=0.10,
                    help="max relative mean fusion_count delta for quality MATCHED")
    ap.add_argument("--speedup-effective", type=float, default=1.20,
                    help="end-to-end throughput speedup needed for EFFECTIVE")
    ap.add_argument("--speedup-marginal", type=float, default=1.10,
                    help="end-to-end throughput speedup needed for MARGINAL")
    args = ap.parse_args()

    off = summarize(_load(args.off), "OFF", wall_seconds=args.off_wall_seconds)
    on = summarize(_load(args.on), "ON", wall_seconds=args.on_wall_seconds)
    verdict = compare_summaries(
        off,
        on,
        reward_tol=args.reward_tol,
        priority_tol=args.priority_tol,
        fusion_tol=args.fusion_tol,
        speedup_effective=args.speedup_effective,
        speedup_marginal=args.speedup_marginal,
    )

    print("[ab] Stage-2 rollout A/B comparator")
    print("[ab] speed verdict uses end-to-end episodes/hour, not policy_rollout_wall_seconds.")
    print(
        "[ab] profile note: cuda.synchronize() can wait for sibling worker terminal "
        "probes on the same GPU, so policy_rollout_wall_seconds is diagnostic only."
    )
    print("")
    _print_arm(off)
    _print_arm(on)

    print("\n[ab] runtime decomposition from episodes.jsonl:")
    for name in (
            "policy_rollout",
            "per_step_optimizer",
            "terminal_probe",
            "terminal_probe_install",
    ):
        _print_component(name, off, on)

    print("\n[ab] ==== VERDICT ====")
    print(
        f"     quality {verdict['quality']} "
        f"(reward_rel={_fmt(verdict['reward_rel_delta'])}, "
        f"priority_abs={_fmt(verdict['priority_max_abs_delta'])}, "
        f"fusion_rel={_fmt(verdict['fusion_rel_delta'])})"
    )
    print(
        f"     speed {verdict['speed']} "
        f"(end-to-end throughput ON/OFF={_fmt(verdict['end_to_end_speedup'], 'x')}, "
        f"policy timer OFF/ON={_fmt(verdict['policy_rollout_speedup'], 'x')})"
    )
    print(
        "     decision: keep batched rollout OFF for 60k unless quality is MATCHED "
        "and end-to-end throughput is >= requested threshold."
    )
    return 0 if verdict["quality"] == "MATCHED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
