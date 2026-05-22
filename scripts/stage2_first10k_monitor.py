#!/usr/bin/env python3
"""Monitor/report helper for the first-10k Stage-2 sequential RL run."""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _is_nonfinite(value: Any) -> bool:
    try:
        return not math.isfinite(float(value))
    except Exception:
        return False


def _window(values: List[float], size: int) -> Optional[Dict[str, float]]:
    if len(values) < size:
        return None
    tail = values[-size:]
    ordered = sorted(tail)
    def pct(q: float) -> float:
        if not ordered:
            return 0.0
        idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
        return float(ordered[idx])
    return {
        "size": int(size),
        "mean": float(statistics.mean(tail)),
        "min": float(min(tail)),
        "p05": pct(0.05),
        "p50": pct(0.50),
        "p95": pct(0.95),
        "max": float(max(tail)),
        "slope": float(tail[-1] - tail[0]) / float(max(1, len(tail) - 1)),
    }


def _max_consecutive(items: Iterable[bool]) -> int:
    best = 0
    cur = 0
    for item in items:
        if item:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _gpu_stats(path: Path) -> Dict[str, Any]:
    rows = []
    if path.exists():
        with path.open(newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                try:
                    rows.append({
                        "idx": str(row.get("gpu_idx", "")).strip(),
                        "util": float(str(row.get("util_pct", "0")).strip()),
                        "mem": float(str(row.get("mem_used_mib", "0")).strip()),
                    })
                except Exception:
                    continue
    by_gpu: Dict[str, List[Dict[str, float]]] = {}
    for row in rows:
        by_gpu.setdefault(row["idx"], []).append(row)
    summary: Dict[str, Any] = {"samples": len(rows), "by_gpu": {}}
    for idx, gpu_rows in sorted(by_gpu.items()):
        utils = [r["util"] for r in gpu_rows]
        mems = [r["mem"] for r in gpu_rows]
        summary["by_gpu"][idx] = {
            "max_util": max(utils) if utils else 0.0,
            "p50_util": statistics.median(utils) if utils else 0.0,
            "active_sample_rate": (
                sum(1 for value in utils if value > 0.0) / float(len(utils))
                if utils else 0.0
            ),
            "max_mem_mib": max(mems) if mems else 0.0,
        }
    return summary


def _episode_priority(row: Dict[str, Any]) -> int:
    if "terminal_priority" in row:
        try:
            return int(row.get("terminal_priority") or 0)
        except Exception:
            return 0
    return 1 if int(row.get("invalid_steps", 0) or 0) > 0 else 3


def _parse_expected_devices(spec: str) -> List[str]:
    if not spec:
        return []
    out: List[str] = []
    for item in str(spec).replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        if item.startswith("cuda:"):
            out.append(item)
        else:
            try:
                out.append(f"cuda:{int(item)}")
            except Exception:
                out.append(item)
    return out


def _expected_trial_split(k_trials: int, device_count: int) -> List[int]:
    if device_count <= 0:
        return []
    base = int(k_trials) // int(device_count)
    rem = int(k_trials) % int(device_count)
    return [base + (1 if idx < rem else 0) for idx in range(device_count)]


def build_summary(args: argparse.Namespace) -> Dict[str, Any]:
    artifact = Path(args.artifact_dir)
    stage2_noise = Path(args.stage2_noise)
    episodes = _read_jsonl(artifact / "episodes.jsonl")
    if not episodes:
        episodes = _read_jsonl(stage2_noise / "progress" / "diagnostics" / "episodes.jsonl")
    ppo = _read_jsonl(artifact / "ppo_updates.jsonl")
    if not ppo:
        ppo = _read_jsonl(stage2_noise / "progress" / "diagnostics" / "ppo_updates.jsonl")

    returns = [_finite(e.get("total_reward")) for e in episodes]
    terminal_rewards = [_finite(e.get("terminal_reward")) for e in episodes]
    priorities = [_episode_priority(e) for e in episodes]
    losses = [_finite(e.get("terminal_loss_mean")) for e in episodes if "terminal_loss_mean" in e]
    metric1 = [_finite(e.get("terminal_metric1_mean")) for e in episodes if "terminal_metric1_mean" in e]
    metric2 = [_finite(e.get("terminal_metric2_mean")) for e in episodes if "terminal_metric2_mean" in e]
    metric1_std = [_finite(e.get("terminal_metric1_std")) for e in episodes if "terminal_metric1_std" in e]
    metric2_std = [_finite(e.get("terminal_metric2_std")) for e in episodes if "terminal_metric2_std" in e]
    stab_violation = [
        _finite(e.get("terminal_stab_violation"))
        for e in episodes if "terminal_stab_violation" in e
    ]
    safe = [bool(e.get("safe_neighbor_active", False)) for e in episodes]
    bits = [_finite(e.get("total_bits")) for e in episodes]
    invalid_steps = [int(e.get("invalid_steps", 0) or 0) for e in episodes]

    post = [e for e in episodes if int(e.get("episode", -1)) >= int(args.anchor)]
    post_priorities = [_episode_priority(e) for e in post]
    post_losses = [_finite(e.get("terminal_loss_mean")) for e in post if "terminal_loss_mean" in e]
    post_returns = [_finite(e.get("total_reward")) for e in post]
    post_safe_rows = [e for e in post if "safe_neighbor_active" in e]
    post_safe = [bool(e.get("safe_neighbor_active", False)) for e in post_safe_rows]
    guarded_rows = [e for e in episodes if "guarded_radius2_active" in e]
    guarded_active = [e for e in guarded_rows if bool(e.get("guarded_radius2_active", False))]
    guarded_failures = [
        e for e in guarded_active
        if _episode_priority(e) in (1, 2)
        or int(e.get("invalid_steps", 0) or 0) > 0
        or _finite(e.get("terminal_stab_violation")) > 0.0
        or _finite(e.get("terminal_loss_mean")) >= 99.0
    ]
    guarded_expansions = [
        e for e in guarded_active
        if str(e.get("terminal_pareto_event_kind", "")) == "frontier_expansion"
    ]

    best_reward = max(returns) if returns else None
    best_episode = None
    if returns:
        best_idx = max(range(len(returns)), key=lambda i: returns[i])
        best_episode = int(episodes[best_idx].get("episode", best_idx))
    completed = len(episodes)
    episodes_since_best = (
        max(0, int(episodes[-1].get("episode", completed - 1)) - int(best_episode))
        if episodes and best_episode is not None else None
    )

    rolling = {
        str(size): _window(returns, size)
        for size in (60, 300, 1000)
    }
    terminal_rolling = {
        str(size): _window(terminal_rewards, size)
        for size in (60, 300, 1000)
    }
    ppo_recent = ppo[-5:]
    ppo_entropy_recent = [
        _finite(row.get("entropy"))
        for row in ppo_recent
    ]
    ppo_clip_recent = [
        _finite(row.get("clip_fraction"))
        for row in ppo_recent
    ]
    ppo_kl_recent = [
        _finite(row.get("approx_kl"))
        for row in ppo_recent
        if "approx_kl" in row
    ]
    ppo_lr_scale_recent = [
        _finite(row.get("lr_scale"))
        for row in ppo_recent
        if "lr_scale" in row
    ]
    ppo_entropy_recovery_recent = [
        _finite(row.get("entropy_recovery_delta"))
        for row in ppo_recent
        if "entropy_recovery_delta" in row
    ]

    hard_failures: List[str] = []
    warnings: List[str] = []

    if completed == 0:
        warnings.append("No episodes observed yet.")
    if args.phase == "final" and completed < int(args.planned):
        hard_failures.append(
            f"Run ended before planned episodes: completed {completed} < planned {int(args.planned)}."
        )
    post_loss_caps = [
        _finite(e.get("terminal_loss_mean"))
        for e in post
        if "terminal_loss_mean" in e and _finite(e.get("terminal_loss_mean")) >= 99.0
    ]
    post_loss_cap_flags = [
        "terminal_loss_mean" in e and _finite(e.get("terminal_loss_mean")) >= 99.0
        for e in post
    ]
    if _max_consecutive(post_loss_cap_flags) >= 2:
        hard_failures.append("Repeated terminal_loss_mean collapse cap: >=2 consecutive post-anchor episodes.")
    # Sparse loss-cap spikes are expected under the current exploratory policy.
    # Treat them as hard failures only when they form a burst or become frequent.
    if len(post_loss_cap_flags) >= 100 and sum(1 for flag in post_loss_cap_flags[-100:] if flag) >= 5:
        hard_failures.append("Repeated terminal_loss_mean collapse cap: >=5 episodes in latest 100 post-anchor episodes.")
    if post_loss_caps and not any("terminal_loss_mean collapse cap" in item for item in hard_failures):
        warnings.append(f"Observed {len(post_loss_caps)} isolated post-anchor terminal_loss_mean collapse-cap episode(s).")
    if any(_is_nonfinite(e.get("total_reward")) for e in episodes):
        hard_failures.append("Non-finite total_reward observed.")
    if _max_consecutive(p == 1 for p in post_priorities) >= 3:
        hard_failures.append("Sustained P1(acc): >=3 consecutive post-anchor P1 episodes.")
    if len(post_priorities) >= 100:
        recent100 = post_priorities[-100:]
        if sum(1 for p in recent100 if p == 1) >= 5:
            hard_failures.append("Sustained P1(acc): >=5 P1 episodes in latest 100 post-anchor episodes.")
    if rolling["60"] and rolling["60"]["mean"] < 20.0:
        hard_failures.append("rolling60 mean return fell below 20.")
    if rolling["300"] and rolling["300"]["mean"] < 35.0:
        hard_failures.append("rolling300 mean return fell below 35.")
    if completed > int(args.anchor) and post_safe_rows and not any(post_safe):
        hard_failures.append("No post-anchor safe-neighbor active episodes observed.")
    if sum(invalid_steps) > 0:
        hard_failures.append("Invalid steps reappeared in structured episode diagnostics.")
    expected_samples = int(args.rollout) * int(args.horizon)
    for row in ppo:
        if int(row.get("n_samples", expected_samples) or expected_samples) != expected_samples:
            hard_failures.append(f"PPO update {row.get('update')} n_samples != {expected_samples}.")
            break
        if _is_nonfinite(row.get("policy_loss")) or _is_nonfinite(row.get("value_loss")):
            hard_failures.append(f"PPO update {row.get('update')} has non-finite loss.")
            break

    if completed > 1000 and episodes_since_best is not None and episodes_since_best > 2000:
        warnings.append("No new best for more than 2000 episodes.")
    if (
            completed > 1000
            and ppo_entropy_recent
            and statistics.mean(ppo_entropy_recent) < 1e-4
            and episodes_since_best is not None
            and episodes_since_best > 1000
            ):
        warnings.append("Entropy is near zero and best reward has not improved for >1000 episodes.")
    if len(ppo_clip_recent) >= 2 and all(value > 0.5 for value in ppo_clip_recent[-2:]):
        warnings.append("Recent PPO clip_fraction stayed above 0.5 for two updates.")

    gpu = _gpu_stats(Path(args.nvidia_log))
    expected_devices = _parse_expected_devices(str(getattr(args, "expected_reward_devices", "") or ""))
    expected_gpu_indices = [
        dev.split("cuda:", 1)[1]
        for dev in expected_devices
        if dev.startswith("cuda:") and dev.split("cuda:", 1)[1].isdigit()
    ]
    probe_rows = [
        e for e in episodes
        if e.get("terminal_probe_devices") or e.get("terminal_probe_trial_counts")
    ]
    observed_device_sets = sorted({
        tuple(str(x) for x in (e.get("terminal_probe_devices") or []))
        for e in probe_rows
        if e.get("terminal_probe_devices")
    })
    observed_trial_splits = sorted({
        tuple(int(x) for x in (e.get("terminal_probe_trial_counts") or []))
        for e in probe_rows
        if e.get("terminal_probe_trial_counts")
    })
    if args.phase == "final":
        by_gpu = gpu.get("by_gpu", {})
        required_gpu_indices = expected_gpu_indices if expected_gpu_indices else ["0", "1"]
        if len(by_gpu) < len(required_gpu_indices):
            hard_failures.append(
                f"Fewer than expected GPUs observed in nvidia-smi samples: "
                f"{len(by_gpu)} < {len(required_gpu_indices)}."
            )
        for idx in required_gpu_indices:
            info = by_gpu.get(idx, {})
            if _finite(info.get("max_util")) <= 0:
                hard_failures.append(f"GPU {idx} never showed nonzero utilization.")
            if _finite(info.get("active_sample_rate")) < 0.05:
                warnings.append(f"GPU {idx} active_sample_rate below 5%.")
        if expected_devices:
            expected_set = set(expected_devices)
            if not any(expected_set.issubset(set(devices)) for devices in observed_device_sets):
                hard_failures.append(
                    f"Reward probe devices never included expected set {sorted(expected_set)}."
                )
            expected_split = tuple(_expected_trial_split(int(args.k_trials), len(expected_devices)))
            if expected_split and expected_split not in observed_trial_splits:
                hard_failures.append(
                    f"Reward probe trial split never matched expected {list(expected_split)}."
                )

    status = "FAIL" if hard_failures else ("WARN" if warnings else "PASS")
    return {
        "status": status,
        "hard_failures": hard_failures,
        "warnings": warnings,
        "artifact_dir": str(artifact),
        "stage2_noise": str(stage2_noise),
        "planned_episodes": int(args.planned),
        "completed_episodes": completed,
        "anchor_episodes": int(args.anchor),
        "rollout_size": int(args.rollout),
        "horizon": int(args.horizon),
        "k_trials": int(args.k_trials),
        "probe_size": int(args.probe_size),
        "reward_probe": {
            "expected_devices": expected_devices,
            "observed_device_sets": [list(x) for x in observed_device_sets],
            "observed_trial_splits": [list(x) for x in observed_trial_splits],
        },
        "reward": {
            "best_reward": best_reward,
            "best_episode": best_episode,
            "episodes_since_best": episodes_since_best,
            "mean": statistics.mean(returns) if returns else None,
            "post_anchor_mean": statistics.mean(post_returns) if post_returns else None,
            "rolling": rolling,
        },
        "terminal_reward": {"rolling": terminal_rolling},
        "terminal_metrics": {
            "loss_mean_min": min(losses) if losses else None,
            "loss_mean_max": max(losses) if losses else None,
            "post_anchor_loss_mean_max": max(post_losses) if post_losses else None,
            "post_anchor_loss_cap_count": len(post_loss_caps),
            "post_anchor_loss_cap_max_consecutive": _max_consecutive(post_loss_cap_flags),
            "metric1_min": min(metric1) if metric1 else None,
            "metric1_max": max(metric1) if metric1 else None,
            "metric2_min": min(metric2) if metric2 else None,
            "metric2_max": max(metric2) if metric2 else None,
            "metric1_std_max": max(metric1_std) if metric1_std else None,
            "metric2_std_max": max(metric2_std) if metric2_std else None,
            "terminal_stab_violation_max": max(stab_violation) if stab_violation else None,
        },
        "priority": {
            "p1_count": sum(1 for p in priorities if p == 1),
            "p2_count": sum(1 for p in priorities if p == 2),
            "p3_count": sum(1 for p in priorities if p == 3),
            "post_anchor_p1_count": sum(1 for p in post_priorities if p == 1),
            "post_anchor_p2_count": sum(1 for p in post_priorities if p == 2),
            "post_anchor_max_consecutive_p1": _max_consecutive(p == 1 for p in post_priorities),
        },
        "validity": {
            "invalid_steps_total": sum(invalid_steps),
            "early_terminated_count": sum(1 for e in episodes if bool(e.get("early_terminated", False))),
            "valid_steps_min": min((int(e.get("valid_steps", 0) or 0) for e in episodes), default=None),
        },
        "safe_neighbor": {
            "post_anchor_active_count": sum(1 for value in post_safe if value),
            "post_anchor_active_rate": (
                sum(1 for value in post_safe if value) / float(len(post_safe))
                if post_safe else None
            ),
            "last_mutation_count": (
                int(episodes[-1].get("safe_neighbor_mutation_count", 0) or 0)
                if episodes else None
            ),
            "last_radius": (
                int(episodes[-1].get("safe_neighbor_radius", 0) or 0)
                if episodes else None
            ),
        },
        "guarded_radius2": {
            "active_count": len(guarded_active),
            "failure_count": len(guarded_failures),
            "frontier_expansion_count": len(guarded_expansions),
            "last_cooldown_remaining": (
                int(episodes[-1].get("guarded_radius2_cooldown_remaining", 0) or 0)
                if episodes else None
            ),
            "last_safe_offset_count": (
                int(episodes[-1].get("guarded_radius2_safe_offset_count", 0) or 0)
                if episodes else None
            ),
        },
        "cost": {
            "total_bits_min": min(bits) if bits else None,
            "total_bits_max": max(bits) if bits else None,
            "total_bits_last": bits[-1] if bits else None,
        },
        "ppo": {
            "updates_seen": len(ppo),
            "last_update": ppo[-1] if ppo else None,
            "recent_entropy_mean": statistics.mean(ppo_entropy_recent) if ppo_entropy_recent else None,
            "recent_clip_fraction_mean": statistics.mean(ppo_clip_recent) if ppo_clip_recent else None,
            "recent_approx_kl_mean": statistics.mean(ppo_kl_recent) if ppo_kl_recent else None,
            "recent_lr_scale_mean": statistics.mean(ppo_lr_scale_recent) if ppo_lr_scale_recent else None,
            "recent_entropy_recovery_mean": (
                statistics.mean(ppo_entropy_recovery_recent)
                if ppo_entropy_recovery_recent else None
            ),
        },
        "gpu": gpu,
        "updated_at": time.time(),
    }


def write_window_csv(path: Path, episodes: List[Dict[str, Any]]) -> None:
    rows = []
    returns = [_finite(e.get("total_reward")) for e in episodes]
    for idx in range(len(episodes)):
        record = {"episode": int(episodes[idx].get("episode", idx))}
        for size in (60, 300, 1000):
            win = _window(returns[: idx + 1], size)
            record[f"rolling{size}_mean"] = "" if win is None else f"{win['mean']:.8f}"
            record[f"rolling{size}_min"] = "" if win is None else f"{win['min']:.8f}"
            record[f"rolling{size}_max"] = "" if win is None else f"{win['max']:.8f}"
        rows.append(record)
    if not rows:
        path.write_text("episode\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_health_csv(path: Path, episodes: List[Dict[str, Any]]) -> None:
    fields = [
        "episode", "total_reward", "terminal_reward", "terminal_priority",
        "terminal_loss_mean", "terminal_loss_std",
        "terminal_metric1_mean", "terminal_metric2_mean",
        "terminal_metric1_std", "terminal_metric2_std",
        "terminal_stab_excess_m1", "terminal_stab_excess_m2",
        "terminal_stab_excess_loss", "terminal_stab_violation",
        "valid_steps", "invalid_steps", "total_bits",
        "safe_neighbor_active", "safe_neighbor_mutation_count", "safe_neighbor_radius",
        "exploration_mode", "guarded_radius2_active",
        "guarded_radius2_recent_frontier_expansions",
        "guarded_radius2_recent_duplicate_rate",
        "guarded_radius2_recent_dominated_rate",
        "guarded_radius2_cooldown_remaining",
        "guarded_radius2_safe_offset_count",
        "baseline_prior_scale",
        "base_action_source",
        "proposal_direction",
        "empirical_offset_success_rate",
        "empirical_offset_failure_rate",
        "frontier_seed_episode",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, row in enumerate(episodes):
            writer.writerow({field: row.get(field, idx if field == "episode" else "") for field in fields})


def write_report(path: Path, summary: Dict[str, Any]) -> None:
    reward = summary.get("reward", {})
    terminal = summary.get("terminal_metrics", {})
    ppo = summary.get("ppo", {})
    guarded = summary.get("guarded_radius2", {})
    gpu = summary.get("gpu", {}).get("by_gpu", {})
    reward_probe = summary.get("reward_probe", {})
    rows = [
        ("status", summary.get("status")),
        ("completed_episodes", summary.get("completed_episodes")),
        ("best_reward", reward.get("best_reward")),
        ("best_episode", reward.get("best_episode")),
        ("post_anchor_mean", reward.get("post_anchor_mean")),
        ("post_anchor_p1_count", summary.get("priority", {}).get("post_anchor_p1_count")),
        ("post_anchor_p2_count", summary.get("priority", {}).get("post_anchor_p2_count")),
        ("post_anchor_loss_mean_max", terminal.get("post_anchor_loss_mean_max")),
        ("metric1_range", (terminal.get("metric1_min"), terminal.get("metric1_max"))),
        ("metric2_range", (terminal.get("metric2_min"), terminal.get("metric2_max"))),
        ("metric_std_max", (terminal.get("metric1_std_max"), terminal.get("metric2_std_max"))),
        ("terminal_stab_violation_max", terminal.get("terminal_stab_violation_max")),
        ("invalid_steps_total", summary.get("validity", {}).get("invalid_steps_total")),
        ("safe_neighbor_active_rate", summary.get("safe_neighbor", {}).get("post_anchor_active_rate")),
        ("last_mutation_count", summary.get("safe_neighbor", {}).get("last_mutation_count")),
        ("last_radius", summary.get("safe_neighbor", {}).get("last_radius")),
        ("guarded_radius2_active_count", guarded.get("active_count")),
        ("guarded_radius2_failure_count", guarded.get("failure_count")),
        ("guarded_radius2_frontier_expansion_count", guarded.get("frontier_expansion_count")),
        ("guarded_radius2_last_cooldown", guarded.get("last_cooldown_remaining")),
        ("guarded_radius2_last_safe_offset_count", guarded.get("last_safe_offset_count")),
        ("ppo_updates_seen", ppo.get("updates_seen")),
        ("recent_entropy_mean", ppo.get("recent_entropy_mean")),
        ("recent_clip_fraction_mean", ppo.get("recent_clip_fraction_mean")),
        ("recent_approx_kl_mean", ppo.get("recent_approx_kl_mean")),
        ("recent_lr_scale_mean", ppo.get("recent_lr_scale_mean")),
        ("recent_entropy_recovery_mean", ppo.get("recent_entropy_recovery_mean")),
        ("reward_probe", json.dumps(reward_probe, ensure_ascii=False, indent=2)),
        ("gpu", json.dumps(gpu, ensure_ascii=False, indent=2)),
    ]
    table = "".join(
        f"<tr><th>{html.escape(str(k))}</th><td><pre>{html.escape(str(v))}</pre></td></tr>"
        for k, v in rows
    )
    failures = summary.get("hard_failures") or []
    warnings = summary.get("warnings") or []
    failure_html = "".join(f"<li>{html.escape(str(x))}</li>" for x in failures) or "<li>None</li>"
    warning_html = "".join(f"<li>{html.escape(str(x))}</li>" for x in warnings) or "<li>None</li>"
    path.write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<title>Stage2 RL first-10k monitor</title>"
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;"
        "line-height:1.45;margin:28px;color:#202124}table{border-collapse:collapse;width:100%;"
        "margin-top:16px}th,td{border:1px solid #ddd;padding:8px;vertical-align:top}"
        "th{text-align:left;background:#f6f6f6;width:260px}pre{white-space:pre-wrap;margin:0}"
        ".pass{color:#137333}.fail{color:#a50e0e}.warn{color:#8a5a00}</style>"
        f"<h1>Stage2 RL first-10k monitor: {html.escape(str(summary.get('status')))}</h1>"
        "<h2>Hard Failures</h2><ul>" + failure_html + "</ul>"
        "<h2>Warnings</h2><ul>" + warning_html + "</ul>"
        "<h2>Summary</h2><table>" + table + "</table>",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("live", "final"), required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--stage2-noise", required=True)
    parser.add_argument("--nvidia-log", required=True)
    parser.add_argument("--planned", type=int, default=10000)
    parser.add_argument("--anchor", type=int, default=120)
    parser.add_argument("--rollout", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=59)
    parser.add_argument("--k-trials", type=int, default=5)
    parser.add_argument("--probe-size", type=int, default=256)
    parser.add_argument("--expected-reward-devices", default="")
    args = parser.parse_args()

    artifact = Path(args.artifact_dir)
    artifact.mkdir(parents=True, exist_ok=True)
    summary = build_summary(args)
    (artifact / "monitor_live.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (artifact / "monitor_events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "phase": args.phase,
            "status": summary["status"],
            "completed_episodes": summary["completed_episodes"],
            "hard_failure_count": len(summary["hard_failures"]),
            "warning_count": len(summary["warnings"]),
            "updated_at": summary["updated_at"],
        }, ensure_ascii=False) + "\n")

    if args.phase == "final":
        stage2_noise = Path(args.stage2_noise)
        episodes = _read_jsonl(artifact / "episodes.jsonl")
        if not episodes:
            episodes = _read_jsonl(stage2_noise / "progress" / "diagnostics" / "episodes.jsonl")
        (artifact / "monitor_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_window_csv(artifact / "reward_windows.csv", episodes)
        write_health_csv(artifact / "episode_health_windows.csv", episodes)
        write_report(artifact / "server_monitor_report.html", summary)
    return 2 if summary["hard_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
