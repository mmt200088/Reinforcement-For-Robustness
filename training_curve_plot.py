"""Shared Stage-1-style PPO training curve renderer.

This module is intentionally torch-free.  Stage-1 and Stage-2 persistence code
can import it without pulling model or evaluation dependencies into lightweight
artifact-generation tests.
"""
from __future__ import annotations

from typing import Optional, Sequence


def _moving_average(values, window: int):
    import numpy as np

    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return np.array([], dtype=float)
    w = int(max(1, min(window, arr.size)))
    if arr.size < w:
        return arr
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(arr, kernel, mode="valid")


def _stage1_window(moving_average_window: Optional[int]) -> int:
    if moving_average_window is not None:
        return int(max(1, moving_average_window))
    # Mirrors layer_importance_evaluator: min(max(5, PPO_UPDATE_INTERVAL // 5), 50)
    return 24


def save_stage1_style_training_curve(
    *,
    out_path: str,
    reward: Sequence[float],
    loss: Optional[Sequence[float]] = None,
    metric1: Optional[Sequence[float]] = None,
    metric2: Optional[Sequence[float]] = None,
    baseline_loss: Optional[float] = None,
    baseline_metric1: Optional[float] = None,
    baseline_metric2: Optional[float] = None,
    metric1_name: str = "metric1",
    metric2_name: Optional[str] = None,
    title_suffix: str = "",
    moving_average_window: Optional[int] = None,
) -> str:
    """Save the same raw+moving-average curve layout used by Stage-1 PPO.

    The full two-metric case uses Stage-1's 2x2 layout:
    reward / loss / metric1 / metric2.  The one-metric case uses Stage-1's
    1x3 layout: reward / loss / metric1.  When only reward is provided, a
    single-panel fallback is written for legacy callers.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rewards = np.asarray(list(reward), dtype=float)
    episodes = np.arange(1, rewards.size + 1)
    window = _stage1_window(moving_average_window)

    def _to_arr(seq):
        if seq is None:
            return None
        arr = np.asarray(list(seq), dtype=float)
        return arr if arr.size else None

    losses = _to_arr(loss)
    metric1s = _to_arr(metric1)
    metric2s = _to_arr(metric2)

    def _plot_raw_ma(ax, values, *, label, color, ma_color, ylabel, title,
                     baseline=None):
        vals = np.asarray(values, dtype=float)
        xs = np.arange(1, vals.size + 1)
        ax.plot(xs, vals, label=label, alpha=0.6, color=color)
        ma = _moving_average(vals, window)
        ma_x = xs[window - 1:] if vals.size >= window else xs
        ax.plot(
            ma_x,
            ma,
            label=f"Moving Avg ({window})",
            linewidth=2,
            color=ma_color,
        )
        if baseline is not None:
            ax.axhline(
                y=float(baseline),
                color="gray",
                linestyle="--",
                linewidth=1,
                alpha=0.7,
                label="Baseline",
            )
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

    has_one_metric = losses is not None and metric1s is not None and metric2s is None
    has_two_metrics = losses is not None and metric1s is not None and metric2s is not None

    if has_one_metric:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(
            f"PPO Training Curves{title_suffix}",
            fontsize=14,
            fontweight="bold",
        )
        metric1_label = metric1_name or "metric1"
        panel_specs = (
            (axes[0], rewards, "Episode Reward", "blue", "darkblue",
             "Reward", "Episode Reward", None),
            (axes[1], losses, "Loss", "red", "darkred",
             "Loss", "Loss (lower is better)", baseline_loss),
            (axes[2], metric1s, metric1_label, "green", "darkgreen",
             metric1_label, f"{metric1_label} (higher is better)",
             baseline_metric1),
        )
    elif has_two_metrics:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"PPO Training Curves{title_suffix}",
            fontsize=14,
            fontweight="bold",
        )
        metric1_label = metric1_name or "metric1"
        metric2_label = metric2_name or "metric2"
        panel_specs = (
            (axes[0, 0], rewards, "Episode Reward", "blue", "darkblue",
             "Reward", "Episode Reward", None),
            (axes[0, 1], losses, "Loss", "red", "darkred",
             "Loss", "Loss (lower is better)", baseline_loss),
            (axes[1, 0], metric1s, metric1_label, "green", "darkgreen",
             metric1_label, f"{metric1_label} (higher is better)",
             baseline_metric1),
            (axes[1, 1], metric2s, metric2_label, "purple", "darkviolet",
             metric2_label, f"{metric2_label} (higher is better)",
             baseline_metric2),
        )
    else:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        fig.suptitle(
            f"PPO Training Curves{title_suffix}",
            fontsize=14,
            fontweight="bold",
        )
        panel_specs = (
            (ax, rewards, "Episode Reward", "blue", "darkblue",
             "Reward", "Episode Reward", None),
        )

    for spec in panel_specs:
        _plot_raw_ma(
            spec[0],
            spec[1],
            label=spec[2],
            color=spec[3],
            ma_color=spec[4],
            ylabel=spec[5],
            title=spec[6],
            baseline=spec[7],
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
