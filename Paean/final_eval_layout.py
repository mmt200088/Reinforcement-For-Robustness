"""Torch-free primitives for the decoupled standalone final-eval tools.

Holds the layout/numbering, the sorted-bar plot-data shaping, and the Stage-1
same-cost peer sampler. Kept torch-free (imports only ``config.constants`` +
``config.run_layout`` + numpy/stdlib) so the units in
``2026-05-30-decoupled-standalone-final-eval-design.md`` §10 are testable
without torch. The torch-importing Stage-1/Stage-2 final-eval modules import
from here.

(The Stage-1 sampler lives here rather than ``Paean/action_grid.py`` as the spec
suggested, because ``action_grid`` imports ``blb_stage2_rl.action_space`` whose
package ``__init__`` pulls torch — which would defeat the torch-free test.)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from config import run_layout
from config.constants import GELU_COST, SOFTMAX_COST


RL_GELU_CHOICES: Tuple[int, ...] = (0, 1, 2, 4)


def paean_stage_run_dir(
    stage, model_type: str, dataset: str, paean_root: str, *, n: int, timestamp=None
) -> str:
    """``{paean_root}/stage{1,2}/{combo} {n} {YYYYMMDD}`` (one flat FE run dir)."""
    sub = run_layout.stage_subdir(stage)
    rid = run_layout.run_id(model_type, dataset, n, timestamp)
    return os.path.join(paean_root, sub, rid)


def next_final_eval_number(stage, model_type: str, dataset: str, paean_root: str) -> int:
    """Next independent final-eval sequence number for this combo under
    ``{paean_root}/stage{1,2}/`` (scans the flat FE dirs, max+1, 1 if none)."""
    sub = run_layout.stage_subdir(stage)
    combo = run_layout.combo_name(model_type, dataset)
    return run_layout.next_run_number_in_root(os.path.join(paean_root, sub), combo)


@dataclass
class SortedBars:
    sorted_values: List[float]
    sorted_labels: List[str]
    selected_position: int
    rank: int
    total: int


def sorted_bar_highlight(
    values: Sequence[float], labels: Sequence[str], selected_idx: int, ascending: bool = True
) -> SortedBars:
    """Sort ``(values, labels)`` and track where ``selected_idx`` lands.

    ``ascending=True`` => lower is rank 1 (loss); ``ascending=False`` => higher
    is rank 1 (metric). Stable sort keeps ties in original order.
    """
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i], reverse=not ascending)
    sorted_values = [values[i] for i in order]
    sorted_labels = [labels[i] for i in order]
    selected_position = order.index(int(selected_idx))
    return SortedBars(sorted_values, sorted_labels, selected_position, selected_position + 1, n)


def save_sorted_bar_plot(
    values: Sequence[float], labels: Sequence[str], selected_idx: int, *,
    out_path: str, title: str, ylabel: str, ascending: bool = True,
) -> str:
    """Render a sorted bar chart with the selected config highlighted. Best-effort
    (matplotlib imported lazily; ASCII title to stay GBK-safe). Returns out_path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shaped = sorted_bar_highlight(values, labels, selected_idx, ascending=ascending)
    colors = ["#d62728" if i == shaped.selected_position else "#7f9cc0"
              for i in range(shaped.total)]
    fig, ax = plt.subplots(figsize=(max(6, shaped.total * 0.22), 4))
    ax.bar(range(shaped.total), shaped.sorted_values, color=colors)
    ax.set_title(f"{title} (selected rank {shaped.rank}/{shaped.total})")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("configs sorted by metric (selected highlighted)")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _cost_q(gelu: Sequence[int]) -> int:
    """Quantize total GELU cost to an int (costs are multiples of 0.5) so exact
    equality is float-safe."""
    return int(round(sum(GELU_COST[int(g)] for g in gelu) * 2))


def _gelu_choice_costs_q(gelu_choices: Sequence[int]) -> np.ndarray:
    """Return per-choice GELU costs in the same quantized units as ``_cost_q``."""
    return np.asarray(
        [int(round(float(GELU_COST[int(choice)]) * 2)) for choice in gelu_choices],
        dtype=np.int64,
    )


def _unique_extreme_cost_has_no_peer(
    target_q: int,
    num_layers: int,
    choice_costs_q: np.ndarray,
) -> bool:
    min_cost_q = int(choice_costs_q.min())
    max_cost_q = int(choice_costs_q.max())
    if int(target_q) == int(num_layers) * min_cost_q:
        return int(np.count_nonzero(choice_costs_q == min_cost_q)) == 1
    if int(target_q) == int(num_layers) * max_cost_q:
        return int(np.count_nonzero(choice_costs_q == max_cost_q)) == 1
    return False


def build_cost_matched_stage1_configs(
    selected_gelu: Sequence[int],
    selected_softmax: Sequence[int],
    num_layers: int,
    *,
    count: int = 50,
    max_attempts: int = 20000,
    seed: int = 42,
    gelu_choices: Sequence[int] = RL_GELU_CHOICES,
) -> Tuple[List[Tuple[List[int], List[int]]], int]:
    """Sample up to ``count`` distinct gelu-degree vectors whose **total Stage-1
    cost exactly equals** the selected config, with softmax **held fixed**.

    Gelu-only: softmax cost is constant across the domain, so exact total-cost
    match reduces to exact total-gelu-cost match. Degrees are drawn from
    ``gelu_choices`` (RL-selectable {0,1,2,4}, incl. ReLU=0). The selected vector
    is excluded. Returns ``(peers, shortfall)`` where ``shortfall = max(0,
    count - len(peers))`` (large near a cost extreme). Deterministic per ``seed``.
    """
    rng = np.random.default_rng(int(seed))
    sel_gelu = [int(g) for g in selected_gelu]
    sel_softmax = [int(s) for s in selected_softmax]
    target_q = _cost_q(sel_gelu)
    choices = np.asarray([int(c) for c in gelu_choices], dtype=np.int64)
    choice_costs_q = _gelu_choice_costs_q(choices)
    if _unique_extreme_cost_has_no_peer(target_q, int(num_layers), choice_costs_q):
        return [], max(0, int(count))

    accepted: List[Tuple[List[int], List[int]]] = []
    seen = {tuple(sel_gelu)}
    attempts = 0
    while len(accepted) < int(count) and attempts < int(max_attempts):
        attempts += 1
        idx = rng.integers(0, choices.size, size=int(num_layers))
        if int(choice_costs_q[idx].sum()) != target_q:
            continue
        vec = [int(choice) for choice in choices[idx]]
        key = tuple(vec)
        if key in seen:
            continue
        seen.add(key)
        accepted.append((vec, list(sel_softmax)))

    return accepted, max(0, int(count) - len(accepted))
