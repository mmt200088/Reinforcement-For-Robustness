#!/usr/bin/env python3
"""A/B compare two fusion-mode Stage-2 RL runs (curriculum ON vs OFF).

Torch-free. Reads each run's ``diagnostics/episodes.jsonl`` and emits a side-by-side
HTML report: reward / priority-mix / fusion_count / collapse-sentinel curves plus a
verdict on whether the block-granularity safe-neighbor curriculum prevents the
post-anchor collapse (and whether it costs final-config quality).

Usage:
    python scripts/blb_fusion_ab_compare.py \
        --run-a <dir_with_diagnostics> --label-a "curriculum ON" \
        --run-b <dir_with_diagnostics> --label-b "curriculum OFF" \
        --anchor 80 --window 200 --out report.html
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

LOSS_CAP = 100.0  # terminal_loss_mean sentinel for an accuracy-collapse episode


def _find_episodes_jsonl(run_dir: str) -> Optional[str]:
    cands = [
        os.path.join(run_dir, "diagnostics", "episodes.jsonl"),
        os.path.join(run_dir, "progress", "diagnostics", "episodes.jsonl"),
        os.path.join(run_dir, "episodes.jsonl"),
    ]
    for c in cands:
        if os.path.isfile(c):
            return c
    for root, _dirs, files in os.walk(run_dir):
        if "episodes.jsonl" in files:
            return os.path.join(root, "episodes.jsonl")
    return None


def load_episodes(run_dir: str) -> List[Dict[str, Any]]:
    path = _find_episodes_jsonl(run_dir)
    if not path:
        raise FileNotFoundError(f"no episodes.jsonl under {run_dir}")
    out: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    out.sort(key=lambda r: int(r.get("episode", 0) or 0))
    return out


def _mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _frac(flags: List[bool]) -> float:
    return float(sum(1 for x in flags if x) / len(flags)) if flags else 0.0


def window_stats(eps: List[Dict[str, Any]], window: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i in range(0, len(eps), window):
        chunk = eps[i:i + window]
        if not chunk:
            continue
        pr = [int(r.get("terminal_priority", 0) or 0) for r in chunk]
        rows.append({
            "ep_lo": int(chunk[0].get("episode", i)),
            "ep_hi": int(chunk[-1].get("episode", i + len(chunk) - 1)),
            "n": len(chunk),
            "reward": _mean([float(r.get("total_reward", 0.0) or 0.0) for r in chunk]),
            "p1": _frac([p == 1 for p in pr]),
            "p2": _frac([p == 2 for p in pr]),
            "p3": _frac([p == 3 for p in pr]),
            "fusion": _mean([float(r.get("fusion_count", 0) or 0) for r in chunk]),
            "loss_cap": _frac([float(r.get("terminal_loss_mean", 0.0) or 0.0) >= LOSS_CAP for r in chunk]),
            "m1": _mean([float(r.get("terminal_metric1_mean", 0.0) or 0.0) for r in chunk]),
            "sn_active": _frac([bool(r.get("safe_neighbor_active", False)) for r in chunk]),
            "sn_radius": _mean([float(r.get("safe_neighbor_radius", 0) or 0) for r in chunk]),
            "invalid": _mean([float(r.get("invalid_steps", 0) or 0) for r in chunk]),
        })
    return rows


def summarize(eps: List[Dict[str, Any]], anchor: int) -> Dict[str, Any]:
    post = [r for r in eps if int(r.get("episode", 0) or 0) >= anchor]
    pr = [int(r.get("terminal_priority", 0) or 0) for r in post]
    rewards = [float(r.get("total_reward", 0.0) or 0.0) for r in eps]
    tail = post[-max(1, len(post) // 5):] if post else []  # final ~20% post-anchor
    tail_pr = [int(r.get("terminal_priority", 0) or 0) for r in tail]
    # Search progress: the project's goal is the best hard-priority P3 config,
    # so track the best P3 (valid) episode reward AND where it was found —
    # a run whose best all sits in the first third and whose tail fusion≈0
    # has stopped searching, no matter how nice its tail mean reward looks.
    p3_rows = [
        (float(r.get("total_reward", 0.0) or 0.0), int(r.get("episode", 0) or 0))
        for r in post  # post-anchor only: the forced-baseline anchor is not "found"
        if int(r.get("terminal_priority", 0) or 0) == 3
        and int(r.get("invalid_steps", 0) or 0) == 0
    ]
    best_p3_reward, best_p3_episode = (max(p3_rows) if p3_rows else (0.0, -1))
    return {
        "n_total": len(eps),
        "n_post": len(post),
        "best_reward": max(rewards) if rewards else 0.0,
        "best_p3_reward": float(best_p3_reward),
        "best_p3_episode": int(best_p3_episode),
        "post_p1": _frac([p == 1 for p in pr]),
        "post_p2": _frac([p == 2 for p in pr]),
        "post_p3": _frac([p == 3 for p in pr]),
        "post_loss_cap": _frac([float(r.get("terminal_loss_mean", 0.0) or 0.0) >= LOSS_CAP for r in post]),
        "post_mean_reward": _mean([float(r.get("total_reward", 0.0) or 0.0) for r in post]),
        "post_mean_fusion": _mean([float(r.get("fusion_count", 0) or 0) for r in post]),
        "tail_p1": _frac([p == 1 for p in tail_pr]),
        "tail_p2": _frac([p == 2 for p in tail_pr]),
        "tail_p3": _frac([p == 3 for p in tail_pr]),
        "tail_mean_reward": _mean([float(r.get("total_reward", 0.0) or 0.0) for r in tail]),
        "tail_mean_fusion": _mean([float(r.get("fusion_count", 0) or 0) for r in tail]),
        "tail_mean_m1": _mean([float(r.get("terminal_metric1_mean", 0.0) or 0.0) for r in tail]),
    }


def _load_best_action(run_dir: str) -> Optional[Dict[str, Any]]:
    for root, _dirs, files in os.walk(run_dir):
        if "blb_stage2_best_action_full.json" in files:
            try:
                with open(os.path.join(root, "blb_stage2_best_action_full.json"), encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
    return None


def _try_plots(out_dir: str, wa, wb, la, lb) -> List[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    pngs: List[str] = []
    panels = [
        ("reward", "mean episode reward"),
        ("p1", "P1(acc-collapse) fraction"),
        ("fusion", "mean fusion_count / episode"),
        ("p3", "P3(cost) fraction"),
    ]
    for key, title in panels:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.plot([r["ep_hi"] for r in wa], [r[key] for r in wa], label=la, color="#1f77b4")
        ax.plot([r["ep_hi"] for r in wb], [r[key] for r in wb], label=lb, color="#d62728")
        ax.set_title(title)
        ax.set_xlabel("episode")
        ax.legend()
        ax.grid(alpha=0.3)
        p = os.path.join(out_dir, f"ab_{key}.png")
        fig.tight_layout()
        fig.savefig(p, dpi=110)
        plt.close(fig)
        pngs.append(os.path.basename(p))
    return pngs


def _verdict(sa: Dict[str, Any], sb: Dict[str, Any], la: str, lb: str) -> str:
    # B (curriculum off) is expected to collapse: high post-anchor P1, loss-cap.
    a_healthy = sa["tail_p1"] < 0.25 and sa["tail_p3"] > 0.4
    b_collapsed = sb["tail_p1"] > 0.5 or sb["post_loss_cap"] > 0.4
    if a_healthy and b_collapsed:
        return (f"✅ Curriculum helps: <b>{la}</b> stays healthy (tail P1={sa['tail_p1']:.0%}, "
                f"P3={sa['tail_p3']:.0%}) while <b>{lb}</b> collapses (tail P1={sb['tail_p1']:.0%}, "
                f"loss-cap={sb['post_loss_cap']:.0%}).")
    if a_healthy and not b_collapsed:
        # Both avoided ACCURACY collapse → judge by SEARCH PROGRESS, not tail
        # mean reward. The goal is the best hard-priority P3 config; a tail
        # parked at fusion≈0 means the policy retreated to baseline
        # (exploration collapse) — that *raises* tail mean reward (no P1 tax)
        # while ending the search. The 2026-06-10 6k A/B is the canonical
        # example: OFF had the better tail mean but found all its best
        # candidates before ep 1000 and adopted 0 fusions thereafter, while
        # ON kept improving through the end with ~15 fusions/episode.
        a_best = float(sa.get("best_p3_reward", 0.0))
        b_best = float(sb.get("best_p3_reward", 0.0))
        a_ep = int(sa.get("best_p3_episode", -1))
        b_ep = int(sb.get("best_p3_episode", -1))
        winner_label = la if a_best >= b_best else lb
        notes = []
        for s, lbl in ((sa, la), (sb, lb)):
            n_tot = max(1, int(s.get("n_total", 0)))
            stalled = (
                float(s.get("tail_mean_fusion", 0.0)) < 0.5
                and 0 <= int(s.get("best_p3_episode", -1)) < n_tot // 3
            )
            if stalled:
                notes.append(
                    f"<b>{lbl}</b> shows exploration collapse: tail fusion≈0 and its best "
                    f"P3 was found in the first third (ep {int(s.get('best_p3_episode', -1))}) "
                    f"— the higher tail mean is the safety of parking at baseline, not progress"
                )
        return (
            f"🔎 Both runs avoided accuracy collapse → verdict by SEARCH PROGRESS "
            f"(best P3 reward + when it was found; tail mean reward is only a safety metric): "
            f"<b>{winner_label}</b> wins — best P3 {a_best:.2f}@ep{a_ep} ({la}) vs "
            f"{b_best:.2f}@ep{b_ep} ({lb}); tail fusion {la}={sa['tail_mean_fusion']:.1f} / "
            f"{lb}={sb['tail_mean_fusion']:.1f}; tail P2 {la}={sa.get('tail_p2', 0.0):.0%} / "
            f"{lb}={sb.get('tail_p2', 0.0):.0%}."
            + ((" " + "; ".join(notes) + ".") if notes else "")
        )
    if not a_healthy and b_collapsed:
        return (f"❌ Curriculum did not fully prevent collapse (A tail P1={sa['tail_p1']:.0%}). "
                f"Needs a longer ramp or tighter radius schedule.")
    return (f"❓ Inconclusive — A tail P1={sa['tail_p1']:.0%}/P3={sa['tail_p3']:.0%}, "
            f"B tail P1={sb['tail_p1']:.0%}. Inspect curves.")


def _table(rows: List[Dict[str, Any]]) -> str:
    head = ("<tr><th>ep</th><th>n</th><th>reward</th><th>P1</th><th>P2</th><th>P3</th>"
            "<th>fusion</th><th>loss_cap</th><th>m1</th><th>sn_act</th><th>sn_r</th></tr>")
    body = "".join(
        f"<tr><td>{r['ep_lo']}-{r['ep_hi']}</td><td>{r['n']}</td><td>{r['reward']:.2f}</td>"
        f"<td>{r['p1']:.0%}</td><td>{r['p2']:.0%}</td><td>{r['p3']:.0%}</td>"
        f"<td>{r['fusion']:.1f}</td><td>{r['loss_cap']:.0%}</td><td>{r['m1']:.3f}</td>"
        f"<td>{r['sn_active']:.0%}</td><td>{r['sn_radius']:.1f}</td></tr>"
        for r in rows
    )
    return f"<table>{head}{body}</table>"


def render_html(la, lb, sa, sb, wa, wb, pngs, ba, bb) -> str:
    def _sum_row(name, key, fmt="{:.3f}"):
        return f"<tr><td>{name}</td><td>{fmt.format(sa[key])}</td><td>{fmt.format(sb[key])}</td></tr>"
    summary = (
        "<table><tr><th>metric</th><th>" + la + "</th><th>" + lb + "</th></tr>"
        + _sum_row("episodes", "n_total", "{:d}")
        + _sum_row("best reward", "best_reward", "{:.2f}")
        + _sum_row("best P3 reward (valid)", "best_p3_reward", "{:.2f}")
        + _sum_row("best P3 found @ episode", "best_p3_episode", "{:d}")
        + _sum_row("post-anchor mean reward", "post_mean_reward", "{:.2f}")
        + _sum_row("post-anchor P1(acc) rate", "post_p1", "{:.1%}")
        + _sum_row("post-anchor P2(stab) rate", "post_p2", "{:.1%}")
        + _sum_row("post-anchor P3(cost) rate", "post_p3", "{:.1%}")
        + _sum_row("post-anchor loss-cap rate", "post_loss_cap", "{:.1%}")
        + _sum_row("post-anchor mean fusion", "post_mean_fusion", "{:.2f}")
        + _sum_row("final-20% P1 rate", "tail_p1", "{:.1%}")
        + _sum_row("final-20% P2 rate", "tail_p2", "{:.1%}")
        + _sum_row("final-20% P3 rate", "tail_p3", "{:.1%}")
        + _sum_row("final-20% mean reward", "tail_mean_reward", "{:.2f}")
        + _sum_row("final-20% mean metric1", "tail_mean_m1", "{:.3f}")
        + _sum_row("final-20% mean fusion", "tail_mean_fusion", "{:.2f}")
        + "</table>"
    )
    plots = "".join(f'<img src="{p}" style="max-width:48%;margin:4px">' for p in pngs)

    def _best_block(label, b):
        if not b:
            return f"<p><b>{label}</b>: (no best_action_full.json found)</p>"
        n = len(b.get("slots", b.get("records", [])) or [])
        return f"<p><b>{label} best action</b>: {n} slots recorded (see run dir for full table).</p>"

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Fusion Stage-2 RL A/B — safe-neighbor curriculum</title>
<style>body{{font-family:system-ui,Arial;margin:24px;color:#222}}
table{{border-collapse:collapse;margin:8px 0;font-size:13px}}
td,th{{border:1px solid #ccc;padding:3px 7px;text-align:right}}
th{{background:#f3f3f3}} td:first-child,th:first-child{{text-align:left}}
.verdict{{padding:10px 14px;background:#f7f7ff;border-left:4px solid #4453c4;margin:12px 0}}</style>
</head><body>
<h1>Fusion-count Stage-2 RL — A/B: block-granularity safe-neighbor curriculum</h1>
<div class="verdict">{_verdict(sa, sb, la, lb)}</div>
<h2>Summary</h2>{summary}
<h2>Curves</h2>{plots or '<p>(matplotlib unavailable — see tables)</p>'}
<h2>Best action</h2>{_best_block(la, ba)}{_best_block(lb, bb)}
<h2>{la} — per-window</h2>{_table(wa)}
<h2>{lb} — per-window</h2>{_table(wb)}
</body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", required=True)
    ap.add_argument("--run-b", required=True)
    ap.add_argument("--label-a", default="curriculum ON")
    ap.add_argument("--label-b", default="curriculum OFF")
    ap.add_argument("--anchor", type=int, default=80)
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ea, eb = load_episodes(args.run_a), load_episodes(args.run_b)
    sa, sb = summarize(ea, args.anchor), summarize(eb, args.anchor)
    wa, wb = window_stats(ea, args.window), window_stats(eb, args.window)
    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(out_dir, exist_ok=True)
    pngs = _try_plots(out_dir, wa, wb, args.label_a, args.label_b)
    ba, bb = _load_best_action(args.run_a), _load_best_action(args.run_b)
    html = render_html(args.label_a, args.label_b, sa, sb, wa, wb, pngs, ba, bb)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    # Also drop a compact JSON for machine consumption.
    with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump({"label_a": args.label_a, "label_b": args.label_b,
                   "summary_a": sa, "summary_b": sb}, f, indent=2)
    print(f"[ab-compare] wrote {args.out}")
    print(f"[ab-compare] {args.label_a}: tail P1={sa['tail_p1']:.0%} P3={sa['tail_p3']:.0%} "
          f"reward={sa['tail_mean_reward']:.2f}")
    print(f"[ab-compare] {args.label_b}: tail P1={sb['tail_p1']:.0%} P3={sb['tail_p3']:.0%} "
          f"reward={sb['tail_mean_reward']:.2f}")
    print("[ab-compare] verdict:", _verdict(sa, sb, args.label_a, args.label_b))


if __name__ == "__main__":
    main()
