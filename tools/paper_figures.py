"""Paper-style figure generator for BLB Stage-2 RL runs.

Read one or more run directories (each a ``Parting Chapter/persistent/...``
sub-tree) and produce publication-friendly PNG + PDF figures with consistent
styling.

Supported figure types
----------------------

1. ``training_curves``  -- per-run reward curve over episodes; if multiple
                           runs of the **same group** (e.g. multi-seed),
                           shows mean ± std band instead.
2. ``invalid_heatmap``  -- (layer, block) → frequency of first-invalid
                           events from ``diagnostics/first_invalid_counts.json``.
3. ``best_vs_baseline`` -- bar chart of SF deltas (best - baseline) per
                           slot, grouped by block.
4. ``action_histogram`` -- per-slot policy convergence heatmap (which
                           action_index each slot ended up selecting most).
5. ``cost_vs_accuracy`` -- scatter of top-K candidates in (cost, accuracy)
                           plane, one cross per run.
6. ``ppo_dynamics``     -- 2×2 panel of policy_loss / value_loss /
                           entropy / clip_fraction over PPO updates.

CLI
---

::

    # All figures for one run
    python3 tools/paper_figures.py \\
        --runs "Parting Chapter/persistent/rl/bert-base/mrpc/<slug>" \\
        --out figures/run1 --formats png pdf

    # Multi-seed group (auto mean ± std band on training_curves)
    python3 tools/paper_figures.py \\
        --runs "Parting Chapter/persistent/.../mrpc/<slug>__myrun_s1" \\
                "Parting Chapter/persistent/.../mrpc/<slug>__myrun_s2" \\
                "Parting Chapter/persistent/.../mrpc/<slug>__myrun_s3" \\
        --group-label "myrun (5 seeds)" \\
        --out figures/myrun_multiseed

    # Cross-run comparison (different presets / algorithms)
    python3 tools/paper_figures.py \\
        --runs <run_A> <run_B> \\
        --labels "Sequential" "Single-shot" \\
        --out figures/compare_ablation

Style
-----

- Times-like serif fonts (`DejaVu Serif` fallback if Times not installed)
- LaTeX-friendly: 9pt labels, 7pt tick labels, 1.5 line width
- Color palette: colorblind-safe (Wong 2011)
- Output sizes: 3.5" wide single-column / 7.0" wide double-column
- Saves both PNG (300 DPI) and PDF (vector)
- LaTeX table output: if `--latex-tables` flag, emits a `.tex` table next to each figure
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import itertools
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from jsonl_utils import read_jsonl_fields, read_jsonl_float_field, read_jsonl_xy
from json_utils import read_json_file

# ---------------------------------------------------------------------------
# Style setup
# ---------------------------------------------------------------------------

# Wong 2011 colorblind-safe palette.
PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # vermillion
    "#CC79A7",  # purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#999999",  # gray
]

PAPER_FIG_DPI = 300


def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 8,
        "lines.linewidth": 1.4,
        "axes.linewidth": 0.8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "figure.dpi": PAPER_FIG_DPI,
        "savefig.dpi": PAPER_FIG_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "pdf.fonttype": 42,   # TrueType embed for LaTeX-friendliness
        "ps.fonttype": 42,
    })
    return plt


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

@dataclass
class RunData:
    run_dir: str
    label: str
    progress_dir: str
    episodes: List[float]
    ppo_updates: List[Dict[str, Any]]
    best_action_vec: List[int]
    best_slots: List[Dict[str, Any]]
    baseline_slots: List[Dict[str, Any]]
    diff_vs_baseline: List[Dict[str, Any]]
    first_invalid_counts: Dict[str, int]
    action_histogram: Optional[np.ndarray]   # shape (num_slots, max_levels)


def _json_list_or_empty(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if not value:
        return []
    return list(value)


def _json_dict_or_empty(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    return dict(value)


def _reward_series_for_plot(values: Sequence[float]) -> Sequence[float]:
    if isinstance(values, list):
        return values
    return [float(value) for value in values]


def _reward_seed_matrix(series: Sequence[Sequence[float]], min_len: int) -> np.ndarray:
    width = int(min_len)
    arr = np.empty((len(series), width), dtype=float)
    for row_idx, values in enumerate(series):
        arr[row_idx] = np.fromiter(
            itertools.islice(values, width),
            dtype=float,
            count=width,
        )
    return arr


def load_run(
        run_dir: str,
        label: str = "",
        *,
        include_episodes: bool = True,
        include_ppo_updates: bool = True,
        include_best_action: bool = True,
        include_baseline_action: bool = True,
        include_first_invalid: bool = True,
        include_action_histogram: bool = True,
) -> RunData:
    """Read a run's persistent dir into a RunData."""
    run_dir = str(run_dir).rstrip("/")
    # The blb_stage2/progress/ subtree is where everything lives.
    progress_dir = os.path.join(run_dir, "blb_stage2", "progress")
    if not os.path.isdir(progress_dir):
        # 候选顺序：解耦扁平布局 stage2/{combo}/progress/ → 旧 stage2_noise/progress/
        # → 最后回退到 run_dir 本身（很老的 run）。
        for _cand in (
            os.path.join(run_dir, "progress"),
            os.path.join(run_dir, "stage2_noise", "progress"),
        ):
            if os.path.isdir(_cand):
                progress_dir = _cand
                break
        else:
            progress_dir = run_dir

    diag = os.path.join(progress_dir, "diagnostics")
    episodes = (
        read_jsonl_float_field(os.path.join(diag, "episodes.jsonl"), "total_reward")
        if include_episodes else []
    )
    ppo_updates = read_jsonl_fields(
        os.path.join(diag, "ppo_updates.jsonl"),
        fields=("policy_loss", "value_loss", "entropy", "clip_fraction"),
    ) if include_ppo_updates else []
    best_blob = (
        read_json_file(os.path.join(diag, "best_action_vec.json"), default={})
        if include_best_action else {}
    )
    baseline_blob = (
        read_json_file(os.path.join(diag, "baseline_action_vec.json"), default={})
        if include_baseline_action else {}
    )
    first_inv = (
        read_json_file(os.path.join(diag, "first_invalid_counts.json"), default={})
        if include_first_invalid else {}
    )
    hist = None
    npz_path = os.path.join(diag, "action_histogram.npz")
    if include_action_histogram and os.path.isfile(npz_path):
        try:
            hist = np.load(npz_path)["counts"]
        except Exception:
            hist = None

    return RunData(
        run_dir=run_dir,
        label=label or os.path.basename(run_dir),
        progress_dir=progress_dir,
        episodes=episodes,
        ppo_updates=ppo_updates,
        best_action_vec=_json_list_or_empty(best_blob.get("action_vec")),
        best_slots=_json_list_or_empty(best_blob.get("slots")),
        baseline_slots=_json_list_or_empty(baseline_blob.get("slots")),
        diff_vs_baseline=_json_list_or_empty(best_blob.get("diff_vs_baseline")),
        first_invalid_counts=_json_dict_or_empty(first_inv),
        action_histogram=hist,
    )


# ---------------------------------------------------------------------------
# Figure 1: training curves
# ---------------------------------------------------------------------------

def fig_training_curves(
        runs: Sequence[RunData],
        *,
        group_label: str = "",
        out_path_no_ext: str = "training_curves",
        formats: Sequence[str] = ("png", "pdf"),
        single_column: bool = True,
        ) -> List[str]:
    """Plot per-episode reward curves. If `group_label` given, treat all
    `runs` as seeds of the same group and plot mean ± std band."""
    plt = _setup_matplotlib()
    fig_w = 3.5 if single_column else 7.0
    fig, ax = plt.subplots(figsize=(fig_w, 2.1))

    if group_label:
        # mean ± std across runs (truncate to shortest)
        series: List[Sequence[float]] = []
        for r in runs:
            ep_rewards = _reward_series_for_plot(r.episodes)
            if ep_rewards:
                series.append(ep_rewards)
        if not series:
            ax.text(0.5, 0.5, "(no data)", ha="center", va="center", transform=ax.transAxes)
        else:
            min_len = min(len(s) for s in series)
            arr = _reward_seed_matrix(series, min_len)
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            x = np.arange(1, min_len + 1)
            ax.plot(x, mean, color=PALETTE[0], label=group_label)
            ax.fill_between(x, mean - std, mean + std, color=PALETTE[0], alpha=0.20,
                             label=f"± std (n={len(series)})")
    else:
        for i, r in enumerate(runs):
            ep_rewards = _reward_series_for_plot(r.episodes)
            if not ep_rewards:
                continue
            x = np.arange(1, len(ep_rewards) + 1)
            ax.plot(x, ep_rewards, color=PALETTE[i % len(PALETTE)], label=r.label, alpha=0.85)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Per-episode training reward")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(loc="lower right", frameon=False)
    return _save_fig(fig, out_path_no_ext, formats)


# ---------------------------------------------------------------------------
# Figure 2: first-invalid (layer, block) heatmap
# ---------------------------------------------------------------------------

def fig_invalid_heatmap(
        runs: Sequence[RunData],
        *,
        out_path_no_ext: str = "invalid_heatmap",
        formats: Sequence[str] = ("png", "pdf"),
        num_layers: int = 12,
        ) -> List[str]:
    """For each run, render a (num_layers × 5 blocks) heatmap of first-invalid
    frequencies. If multiple runs, panel them side by side."""
    plt = _setup_matplotlib()
    n = max(1, len(runs))
    fig_w = 1.6 * n + 0.5
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 2.4), squeeze=False)
    blocks = [1, 2, 3, 4, 5]
    for ax_i, run in enumerate(runs):
        ax = axes[0, ax_i]
        mat = np.zeros((num_layers, len(blocks)), dtype=float)
        total = sum(int(v) for v in run.first_invalid_counts.values()) or 1
        for k, v in run.first_invalid_counts.items():
            # k is "L05-B3" form
            try:
                li = int(k[1:3])
                bi = int(k[-1])
                if 0 <= li < num_layers and bi in blocks:
                    mat[li, blocks.index(bi)] = float(v) / float(total) * 100.0
            except Exception:
                pass
        im = ax.imshow(mat, cmap="Reds", aspect="auto", vmin=0)
        ax.set_xticks(range(len(blocks)))
        ax.set_xticklabels([f"B{b}" for b in blocks])
        ax.set_yticks(range(num_layers))
        ax.set_yticklabels([f"L{i}" for i in range(num_layers)])
        ax.set_title(run.label, fontsize=8)
        for li in range(num_layers):
            for bj in range(len(blocks)):
                val = mat[li, bj]
                if val > 0.5:
                    ax.text(bj, li, f"{val:.0f}", ha="center", va="center",
                             fontsize=6,
                             color=("white" if val > 30 else "black"))
        if ax_i == 0:
            ax.set_ylabel("Layer")
        if ax_i == len(runs) - 1:
            cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
            cbar.set_label("% of invalid episodes", fontsize=7)
            cbar.ax.tick_params(labelsize=6)
    fig.suptitle("First-invalid frequency by (layer, block)", fontsize=10, y=1.02)
    return _save_fig(fig, out_path_no_ext, formats)


# ---------------------------------------------------------------------------
# Figure 3: best vs baseline (SF deltas per block)
# ---------------------------------------------------------------------------

def fig_best_vs_baseline(
        runs: Sequence[RunData],
        *,
        out_path_no_ext: str = "best_vs_baseline",
        formats: Sequence[str] = ("png", "pdf"),
        ) -> List[str]:
    """Bar chart: for each block, count how many slots had SF↓ / SF↑ / off→on
    / on→off / K↑ / K↓ in best vs baseline."""
    plt = _setup_matplotlib()
    n = max(1, len(runs))
    fig_w = 3.5 * (1 if n == 1 else 2)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 2.4), squeeze=False, sharey=True)
    for ax_i, run in enumerate(runs):
        ax = axes[0, ax_i]
        per_block_sf_down = {b: 0 for b in (1, 2, 3, 4, 5)}
        per_block_sf_up = {b: 0 for b in (1, 2, 3, 4, 5)}
        per_block_k_down = {b: 0 for b in (1, 2, 3, 4, 5)}
        per_block_k_up = {b: 0 for b in (1, 2, 3, 4, 5)}
        for d in run.diff_vs_baseline:
            label = str(d.get("label", ""))
            # Parse block index from "L05.B3.K" / "L05.B3.W.wffn1"
            try:
                bi = int(label.split(".", 2)[1][1:])
            except Exception:
                continue
            kind = str(d.get("kind", ""))
            if kind == "K":
                delta = d.get("delta")
                if isinstance(delta, (int, float)) and delta < 0:
                    per_block_k_down[bi] = per_block_k_down.get(bi, 0) + 1
                elif isinstance(delta, (int, float)) and delta > 0:
                    per_block_k_up[bi] = per_block_k_up.get(bi, 0) + 1
            else:
                delta = d.get("delta")
                if delta is None:
                    continue
                if isinstance(delta, (int, float)):
                    if float(delta) < 0:
                        per_block_sf_down[bi] = per_block_sf_down.get(bi, 0) + 1
                    elif float(delta) > 0:
                        per_block_sf_up[bi] = per_block_sf_up.get(bi, 0) + 1
        blocks = [1, 2, 3, 4, 5]
        x = np.arange(len(blocks))
        w = 0.2
        ax.bar(x - 1.5 * w, [per_block_sf_down[b] for b in blocks], w,
                label="SF ↓", color=PALETTE[0])
        ax.bar(x - 0.5 * w, [per_block_sf_up[b] for b in blocks], w,
                label="SF ↑", color=PALETTE[1])
        ax.bar(x + 0.5 * w, [per_block_k_down[b] for b in blocks], w,
                label="K ↓", color=PALETTE[2])
        ax.bar(x + 1.5 * w, [per_block_k_up[b] for b in blocks], w,
                label="K ↑", color=PALETTE[3])
        ax.set_xticks(x)
        ax.set_xticklabels([f"B{b}" for b in blocks])
        ax.set_title(run.label, fontsize=8)
        if ax_i == 0:
            ax.set_ylabel("Slot count")
        ax.set_xlabel("Block")
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
        if ax_i == n - 1:
            ax.legend(loc="upper right", frameon=False, fontsize=7)
    fig.suptitle("Best vs baseline: per-block slot changes", fontsize=10, y=1.04)
    return _save_fig(fig, out_path_no_ext, formats)


# ---------------------------------------------------------------------------
# Figure 4: action histogram (slot convergence heatmap)
# ---------------------------------------------------------------------------

def fig_action_histogram(
        runs: Sequence[RunData],
        *,
        out_path_no_ext: str = "action_histogram",
        formats: Sequence[str] = ("png", "pdf"),
        ) -> List[str]:
    """Heatmap of [slot_idx × action_idx] counts (normalized per slot)."""
    plt = _setup_matplotlib()
    n = max(1, len(runs))
    fig, axes = plt.subplots(n, 1, figsize=(7.0, 1.4 * n + 0.6), squeeze=False)
    for ax_i, run in enumerate(runs):
        ax = axes[ax_i, 0]
        h = run.action_histogram
        if h is None:
            ax.text(0.5, 0.5, "(no action_histogram.npz)",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        # Normalize per slot
        row_sum = h.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        norm = h / row_sum
        im = ax.imshow(norm.T, cmap="viridis", aspect="auto", vmin=0, vmax=1)
        ax.set_xlabel("Slot index")
        ax.set_ylabel("action_idx")
        ax.set_title(run.label, fontsize=8)
        if ax_i == n - 1:
            cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.18)
            cbar.set_label("Selection probability", fontsize=7)
            cbar.ax.tick_params(labelsize=6)
    fig.suptitle("Per-slot action selection distribution", fontsize=10, y=1.02)
    return _save_fig(fig, out_path_no_ext, formats)


# ---------------------------------------------------------------------------
# Figure 5: PPO learning dynamics
# ---------------------------------------------------------------------------

def fig_ppo_dynamics(
        runs: Sequence[RunData],
        *,
        out_path_no_ext: str = "ppo_dynamics",
        formats: Sequence[str] = ("png", "pdf"),
        ) -> List[str]:
    """2×2 panel: policy_loss / value_loss / entropy / clip_fraction."""
    plt = _setup_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.0), sharex=True)
    metrics = [
        ("policy_loss", "Policy loss"),
        ("value_loss", "Value loss"),
        ("entropy", "Entropy"),
        ("clip_fraction", "Clip fraction"),
    ]
    for idx, (key, title) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        for i, run in enumerate(runs):
            ys = [float(u.get(key, 0.0)) for u in run.ppo_updates]
            if not ys:
                continue
            x = np.arange(1, len(ys) + 1)
            ax.plot(x, ys, color=PALETTE[i % len(PALETTE)],
                    label=run.label if idx == 0 else None, alpha=0.85)
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        if idx >= 2:
            ax.set_xlabel("PPO update")
    if any(r.ppo_updates for r in runs):
        axes[0, 0].legend(loc="upper right", frameon=False, fontsize=7)
    return _save_fig(fig, out_path_no_ext, formats)


# ---------------------------------------------------------------------------
# Figure 6: cost vs accuracy scatter (top-K candidates per run)
# ---------------------------------------------------------------------------

def fig_cost_vs_accuracy(
        runs: Sequence[RunData],
        *,
        out_path_no_ext: str = "cost_vs_accuracy",
        formats: Sequence[str] = ("png", "pdf"),
        single_column: bool = True,
        ) -> List[str]:
    """Scatter: x=total_bits, y=accuracy-proxy(reward); one cross per
    Top-K candidate. If diagnostics/top_candidates.jsonl missing, use the
    best episode only."""
    plt = _setup_matplotlib()
    fig_w = 3.5 if single_column else 7.0
    fig, ax = plt.subplots(figsize=(fig_w, 2.4))
    has_points = False
    for i, run in enumerate(runs):
        top_path = os.path.join(run.progress_dir, "diagnostics", "top_candidates.jsonl")
        xs, ys = read_jsonl_xy(top_path, "total_bits", "total_reward")
        if not xs:
            continue
        has_points = True
        ax.scatter(xs, ys, color=PALETTE[i % len(PALETTE)],
                    label=run.label, alpha=0.7, s=22, edgecolors="none")
    ax.set_xlabel("Total bits")
    ax.set_ylabel("Training reward")
    ax.set_title("Top-K candidates: cost vs reward")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    if has_points:
        ax.legend(loc="best", frameon=False, fontsize=7)
    return _save_fig(fig, out_path_no_ext, formats)


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save_fig(fig, out_path_no_ext: str, formats: Sequence[str]) -> List[str]:
    os.makedirs(os.path.dirname(out_path_no_ext) or ".", exist_ok=True)
    written: List[str] = []
    for fmt in formats:
        path = f"{out_path_no_ext}.{fmt}"
        try:
            fig.savefig(path)
            written.append(path)
        except Exception as exc:
            sys.stderr.write(f"[paper_figures] save {path} failed: {exc}\n")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return written


# ---------------------------------------------------------------------------
# LaTeX table writer
# ---------------------------------------------------------------------------

def write_latex_summary_table(
        runs: Sequence[RunData],
        out_path: str,
        *,
        caption: str = "BLB Stage-2 RL: training summary",
        label: str = "tab:blb_stage2_summary",
        ) -> str:
    """Tiny LaTeX `booktabs`-style table summarizing each run."""
    rows: List[str] = []
    rows.append(r"\begin{table}[t]")
    rows.append(r"\centering")
    rows.append(r"\caption{" + caption + r"}")
    rows.append(r"\label{" + label + r"}")
    rows.append(r"\small")
    rows.append(r"\begin{tabular}{lrrrr}")
    rows.append(r"\toprule")
    rows.append(r"Run & Best reward & \# episodes & Bits & avg\_k \\")
    rows.append(r"\midrule")
    for run in runs:
        if not run.episodes:
            continue
        best_reward = max(float(value) for value in run.episodes)
        n_ep = len(run.episodes)
        bits = ""
        avg_k = ""
        # If best_action_vec.json carries summary, pull from there.
        for slot in run.best_slots:
            pass  # placeholder; per-slot summary not aggregated here yet
        # Try to read from diagnostics/best_action_vec.json
        # (best_slots already loaded; we'd need cost; leave blank)
        label_escaped = run.label.replace("_", r"\_")
        rows.append(
            f"{label_escaped} & ${best_reward:+.4f}$ & {n_ep} & "
            f"{bits} & {avg_k} \\\\"
        )
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    rows.append(r"\end{table}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows))
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ALL_FIGS = {
    "training_curves": fig_training_curves,
    "invalid_heatmap": fig_invalid_heatmap,
    "best_vs_baseline": fig_best_vs_baseline,
    "action_histogram": fig_action_histogram,
    "ppo_dynamics": fig_ppo_dynamics,
    "cost_vs_accuracy": fig_cost_vs_accuracy,
}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", nargs="+", required=True,
                    help="One or more run directories (persistent-dir level, i.e. the one CONTAINING blb_stage2/)")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="Per-run labels (must match --runs length); defaults to basenames")
    ap.add_argument("--group-label", default="",
                    help="If set, treat all --runs as seeds and plot mean ± std on training_curves")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--formats", nargs="+", default=["png", "pdf"])
    ap.add_argument("--figs", nargs="+", default=list(ALL_FIGS.keys()),
                    help=f"Which figures to render; default = all. Options: {list(ALL_FIGS.keys())}")
    ap.add_argument("--single-column", action="store_true",
                    help="Render at single-column width (3.5\") instead of double (7\")")
    ap.add_argument("--latex-tables", action="store_true",
                    help="Also emit a LaTeX summary table .tex file")
    ap.add_argument("--num-layers", type=int, default=12)
    args = ap.parse_args(argv)

    labels = args.labels
    if labels and len(labels) != len(args.runs):
        ap.error(f"--labels has {len(labels)} entries, --runs has {len(args.runs)}")
    if not labels:
        labels = [os.path.basename(r.rstrip("/")) for r in args.runs]

    selected_figs = set(args.figs)
    need_episodes = bool(args.latex_tables or "training_curves" in selected_figs)
    need_ppo_updates = "ppo_dynamics" in selected_figs
    need_best_action = bool(args.latex_tables or "best_vs_baseline" in selected_figs)
    need_baseline_action = "best_vs_baseline" in selected_figs
    need_first_invalid = "invalid_heatmap" in selected_figs
    need_action_histogram = "action_histogram" in selected_figs
    runs: List[RunData] = []
    for idx, run_path in enumerate(args.runs):
        runs.append(load_run(
            run_path,
            label=labels[idx],
            include_episodes=need_episodes,
            include_ppo_updates=need_ppo_updates,
            include_best_action=need_best_action,
            include_baseline_action=need_baseline_action,
            include_first_invalid=need_first_invalid,
            include_action_histogram=need_action_histogram,
        ))
    os.makedirs(args.out, exist_ok=True)

    written: List[str] = []

    if "training_curves" in args.figs:
        written += fig_training_curves(
            runs, group_label=args.group_label,
            out_path_no_ext=os.path.join(args.out, "training_curves"),
            formats=args.formats, single_column=args.single_column,
        )
    if "invalid_heatmap" in args.figs:
        written += fig_invalid_heatmap(
            runs, out_path_no_ext=os.path.join(args.out, "invalid_heatmap"),
            formats=args.formats, num_layers=args.num_layers,
        )
    if "best_vs_baseline" in args.figs:
        written += fig_best_vs_baseline(
            runs, out_path_no_ext=os.path.join(args.out, "best_vs_baseline"),
            formats=args.formats,
        )
    if "action_histogram" in args.figs:
        written += fig_action_histogram(
            runs, out_path_no_ext=os.path.join(args.out, "action_histogram"),
            formats=args.formats,
        )
    if "ppo_dynamics" in args.figs:
        written += fig_ppo_dynamics(
            runs, out_path_no_ext=os.path.join(args.out, "ppo_dynamics"),
            formats=args.formats,
        )
    if "cost_vs_accuracy" in args.figs:
        written += fig_cost_vs_accuracy(
            runs, out_path_no_ext=os.path.join(args.out, "cost_vs_accuracy"),
            formats=args.formats, single_column=args.single_column,
        )

    if args.latex_tables:
        tex_path = os.path.join(args.out, "summary_table.tex")
        write_latex_summary_table(runs, tex_path)
        written.append(tex_path)

    print(f"Wrote {len(written)} file(s) to {args.out}:")
    for w in written:
        print(f"  {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
