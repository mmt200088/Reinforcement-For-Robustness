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
import sys
from typing import Any, Dict, List

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from jsonl_utils import iter_jsonl  # noqa: E402
from report_format_utils import html_table  # noqa: E402
from stats_utils import fraction_true, mean_from_total, mean_or_default, ratio_or_default  # noqa: E402

LOSS_CAP = 100.0  # terminal_loss_mean sentinel for an accuracy-collapse episode


def _find_episodes_jsonl(run_dir: str) -> str | None:
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
    for row in _iter_episode_rows(path):
        out.append(row)
    out.sort(key=lambda r: int(r.get("episode", 0) or 0))
    return out


def _iter_episode_rows(path: str):
    yield from iter_jsonl(path, errors="skip")


def _scan_and_window_ordered_path(path: str, anchor: int, window: int) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    prev_episode: int | None = None
    ordered = True
    n_total = 0
    n_post = 0
    windows: List[Dict[str, Any]] = []
    chunk: List[Dict[str, Any]] = []
    chunk_offset = 0
    for row in _iter_episode_rows(path):
        episode = int(row.get("episode", 0) or 0)
        if prev_episode is not None and episode < prev_episode:
            ordered = False
        prev_episode = episode
        if not chunk:
            chunk_offset = n_total
        chunk.append(row)
        n_total += 1
        if episode >= anchor:
            n_post += 1
        if len(chunk) >= window:
            windows.append(_window_stats_chunk(chunk, chunk_offset))
            chunk = []
    if chunk:
        windows.append(_window_stats_chunk(chunk, chunk_offset))
    return {"ordered": ordered, "n_total": n_total, "n_post": n_post}, windows


def window_stats(eps: List[Dict[str, Any]], window: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i in range(0, len(eps), window):
        chunk = eps[i:i + window]
        if not chunk:
            continue
        rows.append(_window_stats_chunk(chunk, i))
    return rows


def _window_stats_chunk(chunk: List[Dict[str, Any]], fallback_offset: int) -> Dict[str, Any]:
    n = len(chunk)
    p1 = p2 = p3 = 0
    loss_cap = 0
    sn_active = 0
    reward_sum = 0.0
    fusion_sum = 0.0
    m1_sum = 0.0
    sn_radius_sum = 0.0
    invalid_sum = 0.0
    for row in chunk:
        priority = int(row.get("terminal_priority", 0) or 0)
        if priority == 1:
            p1 += 1
        elif priority == 2:
            p2 += 1
        elif priority == 3:
            p3 += 1
        reward_sum += float(row.get("total_reward", 0.0) or 0.0)
        fusion_sum += float(row.get("fusion_count", 0) or 0)
        loss = float(row.get("terminal_loss_mean", 0.0) or 0.0)
        if loss >= LOSS_CAP:
            loss_cap += 1
        m1_sum += float(row.get("terminal_metric1_mean", 0.0) or 0.0)
        if bool(row.get("safe_neighbor_active", False)):
            sn_active += 1
        sn_radius_sum += float(row.get("safe_neighbor_radius", 0) or 0)
        invalid_sum += float(row.get("invalid_steps", 0) or 0)
    return {
        "ep_lo": int(chunk[0].get("episode", fallback_offset)),
        "ep_hi": int(chunk[-1].get("episode", fallback_offset + len(chunk) - 1)),
        "n": n,
        "reward": mean_from_total(reward_sum, n),
        "p1": ratio_or_default(p1, n),
        "p2": ratio_or_default(p2, n),
        "p3": ratio_or_default(p3, n),
        "fusion": mean_from_total(fusion_sum, n),
        "loss_cap": ratio_or_default(loss_cap, n),
        "m1": mean_from_total(m1_sum, n),
        "sn_active": ratio_or_default(sn_active, n),
        "sn_radius": mean_from_total(sn_radius_sum, n),
        "invalid": mean_from_total(invalid_sum, n),
    }


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
        "post_p1": fraction_true(p == 1 for p in pr),
        "post_p2": fraction_true(p == 2 for p in pr),
        "post_p3": fraction_true(p == 3 for p in pr),
        "post_loss_cap": fraction_true(float(r.get("terminal_loss_mean", 0.0) or 0.0) >= LOSS_CAP for r in post),
        "post_mean_reward": mean_or_default(float(r.get("total_reward", 0.0) or 0.0) for r in post),
        "post_mean_fusion": mean_or_default(float(r.get("fusion_count", 0) or 0) for r in post),
        "tail_p1": fraction_true(p == 1 for p in tail_pr),
        "tail_p2": fraction_true(p == 2 for p in tail_pr),
        "tail_p3": fraction_true(p == 3 for p in tail_pr),
        "tail_mean_reward": mean_or_default(float(r.get("total_reward", 0.0) or 0.0) for r in tail),
        "tail_mean_fusion": mean_or_default(float(r.get("fusion_count", 0) or 0) for r in tail),
        "tail_mean_m1": mean_or_default(float(r.get("terminal_metric1_mean", 0.0) or 0.0) for r in tail),
    }


def _summarize_ordered_path(path: str, anchor: int, n_post: int) -> Dict[str, Any]:
    tail_keep = max(1, n_post // 5) if n_post else 0
    tail_start = n_post - tail_keep

    n_total = 0
    best_reward: float | None = None
    best_p3: tuple[float, int] | None = None

    post_seen = 0
    post_p1 = post_p2 = post_p3 = 0
    post_loss_cap = 0
    post_reward_sum = 0.0
    post_fusion_sum = 0.0

    tail_n = 0
    tail_p1 = tail_p2 = tail_p3 = 0
    tail_reward_sum = 0.0
    tail_fusion_sum = 0.0
    tail_m1_sum = 0.0

    for row in _iter_episode_rows(path):
        episode = int(row.get("episode", 0) or 0)
        priority = int(row.get("terminal_priority", 0) or 0)
        reward = float(row.get("total_reward", 0.0) or 0.0)
        fusion = float(row.get("fusion_count", 0) or 0)
        loss = float(row.get("terminal_loss_mean", 0.0) or 0.0)
        metric1 = float(row.get("terminal_metric1_mean", 0.0) or 0.0)
        invalid_steps = int(row.get("invalid_steps", 0) or 0)

        n_total += 1
        best_reward = reward if best_reward is None else max(best_reward, reward)
        if episode < anchor:
            continue

        if priority == 1:
            post_p1 += 1
        elif priority == 2:
            post_p2 += 1
        elif priority == 3:
            post_p3 += 1
        if loss >= LOSS_CAP:
            post_loss_cap += 1
        post_reward_sum += reward
        post_fusion_sum += fusion
        if priority == 3 and invalid_steps == 0:
            candidate = (reward, episode)
            best_p3 = candidate if best_p3 is None else max(best_p3, candidate)

        if post_seen >= tail_start:
            tail_n += 1
            if priority == 1:
                tail_p1 += 1
            elif priority == 2:
                tail_p2 += 1
            elif priority == 3:
                tail_p3 += 1
            tail_reward_sum += reward
            tail_fusion_sum += fusion
            tail_m1_sum += metric1
        post_seen += 1

    best_p3_reward, best_p3_episode = best_p3 if best_p3 is not None else (0.0, -1)
    return {
        "n_total": n_total,
        "n_post": n_post,
        "best_reward": best_reward if best_reward is not None else 0.0,
        "best_p3_reward": float(best_p3_reward),
        "best_p3_episode": int(best_p3_episode),
        "post_p1": ratio_or_default(post_p1, n_post),
        "post_p2": ratio_or_default(post_p2, n_post),
        "post_p3": ratio_or_default(post_p3, n_post),
        "post_loss_cap": ratio_or_default(post_loss_cap, n_post),
        "post_mean_reward": mean_from_total(post_reward_sum, n_post),
        "post_mean_fusion": mean_from_total(post_fusion_sum, n_post),
        "tail_p1": ratio_or_default(tail_p1, tail_n),
        "tail_p2": ratio_or_default(tail_p2, tail_n),
        "tail_p3": ratio_or_default(tail_p3, tail_n),
        "tail_mean_reward": mean_from_total(tail_reward_sum, tail_n),
        "tail_mean_fusion": mean_from_total(tail_fusion_sum, tail_n),
        "tail_mean_m1": mean_from_total(tail_m1_sum, tail_n),
    }


def analyze_episodes(run_dir: str, anchor: int, window: int) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    path = _find_episodes_jsonl(run_dir)
    if not path:
        raise FileNotFoundError(f"no episodes.jsonl under {run_dir}")
    scan, windows = _scan_and_window_ordered_path(path, anchor, window)
    if not scan["ordered"]:
        eps = load_episodes(run_dir)
        return summarize(eps, anchor), window_stats(eps, window)
    return (
        _summarize_ordered_path(path, anchor, int(scan["n_post"])),
        windows,
    )


def _load_best_action(run_dir: str) -> Dict[str, Any] | None:
    direct_candidates = [
        os.path.join(run_dir, "blb_stage2_best_action_full.json"),
        os.path.join(run_dir, "progress", "blb_stage2_best_action_full.json"),
        os.path.join(run_dir, "blb_stage2", "progress", "blb_stage2_best_action_full.json"),
    ]
    for path in direct_candidates:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
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


def _window_table(rows: List[Dict[str, Any]]) -> str:
    return html_table(
        ["ep", "n", "reward", "P1", "P2", "P3", "fusion", "loss_cap", "m1", "sn_act", "sn_r"],
        [
            [
                f"{r['ep_lo']}-{r['ep_hi']}",
                r["n"],
                f"{r['reward']:.2f}",
                f"{r['p1']:.0%}",
                f"{r['p2']:.0%}",
                f"{r['p3']:.0%}",
                f"{r['fusion']:.1f}",
                f"{r['loss_cap']:.0%}",
                f"{r['m1']:.3f}",
                f"{r['sn_active']:.0%}",
                f"{r['sn_radius']:.1f}",
            ]
            for r in rows
        ],
    )


def render_html(la, lb, sa, sb, wa, wb, pngs, ba, bb) -> str:
    summary_specs = [
        ("episodes", "n_total", "{:d}"),
        ("best reward", "best_reward", "{:.2f}"),
        ("best P3 reward (valid)", "best_p3_reward", "{:.2f}"),
        ("best P3 found @ episode", "best_p3_episode", "{:d}"),
        ("post-anchor mean reward", "post_mean_reward", "{:.2f}"),
        ("post-anchor P1(acc) rate", "post_p1", "{:.1%}"),
        ("post-anchor P2(stab) rate", "post_p2", "{:.1%}"),
        ("post-anchor P3(cost) rate", "post_p3", "{:.1%}"),
        ("post-anchor loss-cap rate", "post_loss_cap", "{:.1%}"),
        ("post-anchor mean fusion", "post_mean_fusion", "{:.2f}"),
        ("final-20% P1 rate", "tail_p1", "{:.1%}"),
        ("final-20% P2 rate", "tail_p2", "{:.1%}"),
        ("final-20% P3 rate", "tail_p3", "{:.1%}"),
        ("final-20% mean reward", "tail_mean_reward", "{:.2f}"),
        ("final-20% mean metric1", "tail_mean_m1", "{:.3f}"),
        ("final-20% mean fusion", "tail_mean_fusion", "{:.2f}"),
    ]
    summary = html_table(
        ["metric", la, lb],
        [[name, fmt.format(sa[key]), fmt.format(sb[key])] for name, key, fmt in summary_specs],
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
<h2>{la} — per-window</h2>{_window_table(wa)}
<h2>{lb} — per-window</h2>{_window_table(wb)}
</body></html>"""


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", required=True)
    ap.add_argument("--run-b", required=True)
    ap.add_argument("--label-a", default="curriculum ON")
    ap.add_argument("--label-b", default="curriculum OFF")
    ap.add_argument("--anchor", type=int, default=80)
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    sa, wa = analyze_episodes(args.run_a, args.anchor, args.window)
    sb, wb = analyze_episodes(args.run_b, args.anchor, args.window)
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
