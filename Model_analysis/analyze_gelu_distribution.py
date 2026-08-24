#!/usr/bin/env python
"""
Analyze GELU(x) input distributions for the six supported BERT/GLUE profiles.

For each dataset and each of the 12 layers, this script:
  1. Collects GELU inputs on the shared 256-example training probe
  2. Plots fine-grained histograms with boundary lines at -2.7, 0, 2.7
  3. Computes and plots statistics for 4 intervals:
     x < -2.7 | -2.7 <= x < 0 | 0 <= x <= 2.7 | x > 2.7

Usage:
    python analyze_gelu_distribution.py --output_dir gelu_analysis
    python analyze_gelu_distribution.py --tasks sst2 mrpc
"""

# 运行全部数据集 bash run_gelu_analysis.sh

import os
import argparse
import queue
import sys
import threading
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
for _path in (_PARENT_DIR, _THIS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from analyze_all_distribution_new import (  # noqa: E402
    TASK_REGISTRY,
    _prepare_bert_data,
    write_profile_protocol,
)

NUM_LAYERS = 12
BOUNDARIES = [-2.7, 0.0, 2.7]
INTERVAL_LABELS = ['x < -2.7', '-2.7 ≤ x < 0', '0 ≤ x ≤ 2.7', 'x > 2.7']
INTERVAL_COLORS = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']

HIST_MIN, HIST_MAX, HIST_BINS = -15.0, 15.0, 300


# ==================== Statistics Collector ====================

class GELUInputCollector:
    """Incrementally accumulates histogram and interval statistics for GELU inputs.
    Also accumulates per-position (per dimension) basic stats (mean, std, min, max)
    for each dimension in the input vector."""

    def __init__(self, num_layers=12, hist_min=-15.0, hist_max=15.0, hist_bins=300):
        self.num_layers = num_layers
        self.bin_edges = np.linspace(hist_min, hist_max, hist_bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2

        self.histograms = [np.zeros(hist_bins, dtype=np.int64) for _ in range(num_layers)]
        self.interval_counts = [np.zeros(4, dtype=np.int64) for _ in range(num_layers)]
        self.total_count = [0] * num_layers
        self.sum_val = [0.0] * num_layers
        self.sum_sq = [0.0] * num_layers
        self.min_val = [float('inf')] * num_layers
        self.max_val = [float('-inf')] * num_layers

        # Per-position stats: layer_idx -> dict with (D,) arrays; D set on first update
        self._per_pos = [None] * num_layers

    def _init_per_pos(self, layer_idx, D):
        if self._per_pos[layer_idx] is not None:
            return
        self._per_pos[layer_idx] = {
            'count': np.zeros(D, dtype=np.int64),
            'sum_val': np.zeros(D, dtype=np.float64),
            'sum_sq': np.zeros(D, dtype=np.float64),
            'min_val': np.full(D, np.inf, dtype=np.float64),
            'max_val': np.full(D, -np.inf, dtype=np.float64),
            'interval_counts': np.zeros((D, 4), dtype=np.int64),
        }

    def merge_batch_stats(self, layer_idx, batch_stats):
        """Merge pre-computed batch statistics (small numpy arrays) into accumulator."""
        self.histograms[layer_idx] += batch_stats['hist']
        self.interval_counts[layer_idx] += batch_stats['ic']
        self.total_count[layer_idx] += batch_stats['n']
        self.sum_val[layer_idx] += batch_stats['s']
        self.sum_sq[layer_idx] += batch_stats['sq']
        self.min_val[layer_idx] = min(self.min_val[layer_idx], batch_stats['mi'])
        self.max_val[layer_idx] = max(self.max_val[layer_idx], batch_stats['ma'])

        D = len(batch_stats['pp_sum'])
        self._init_per_pos(layer_idx, D)
        pp = self._per_pos[layer_idx]
        pp['count'] += batch_stats['pp_n']
        pp['sum_val'] += batch_stats['pp_sum']
        pp['sum_sq'] += batch_stats['pp_sq']
        np.minimum(pp['min_val'], batch_stats['pp_min'], out=pp['min_val'])
        np.maximum(pp['max_val'], batch_stats['pp_max'], out=pp['max_val'])
        pp['interval_counts'] += batch_stats['pp_ic']

    def get_stats(self, layer_idx):
        n = self.total_count[layer_idx]
        if n == 0:
            return None
        mean = self.sum_val[layer_idx] / n
        var = self.sum_sq[layer_idx] / n - mean ** 2
        std = np.sqrt(max(var, 0))
        interval_pct = self.interval_counts[layer_idx] / n * 100
        return {
            'count': n,
            'mean': mean,
            'std': std,
            'min': self.min_val[layer_idx],
            'max': self.max_val[layer_idx],
            'interval_counts': self.interval_counts[layer_idx].tolist(),
            'interval_pct': interval_pct.tolist(),
        }

    def get_per_position_stats(self, layer_idx):
        """Return per-position stats for a layer: mean (D,), std (D,), min (D,), max (D,)."""
        if self._per_pos[layer_idx] is None:
            return None
        pp = self._per_pos[layer_idx]
        cnt = pp['count']
        if cnt.sum() == 0:
            return None
        mean = np.where(cnt > 0, pp['sum_val'] / cnt, 0.0)
        var = np.where(cnt > 0, pp['sum_sq'] / cnt - mean ** 2, 0.0)
        std = np.sqrt(np.maximum(var, 0))
        interval_pct = np.where(cnt[:, None] > 0,
                                pp['interval_counts'] / cnt[:, None] * 100, 0.0)
        return {
            'num_positions': len(cnt),
            'count': cnt,
            'mean': mean,
            'std': std,
            'min': pp['min_val'],
            'max': pp['max_val'],
            'interval_pct': interval_pct,
        }


# ==================== Hook Installation ====================


class GELUHookWrapper(nn.Module):
    """Compute all statistics on GPU, enqueue only tiny result dicts (~220KB)."""

    def __init__(self, orig_act_fn, layer_idx, stats_queue,
                 hist_bins=300, hist_min=-15.0, hist_max=15.0):
        super().__init__()
        self.layer_idx = layer_idx
        self.stats_queue = stats_queue
        self.hist_bins = hist_bins
        self.hist_min = hist_min
        self.hist_max = hist_max
        if isinstance(orig_act_fn, nn.Module):
            self.orig_act_fn = orig_act_fn
        else:
            self.orig_act_fn = orig_act_fn

    def forward(self, x):
        self._collect_stats(x.detach())
        return self.orig_act_fn(x)

    @torch.no_grad()
    def _collect_stats(self, x):
        flat = x.reshape(-1)
        x_2d = x.reshape(-1, x.shape[-1])  # (N, D)

        hist = torch.histc(flat.float(), bins=self.hist_bins,
                           min=self.hist_min, max=self.hist_max)

        ic = torch.stack([
            (flat < -2.7).sum(),
            ((flat >= -2.7) & (flat < 0)).sum(),
            ((flat >= 0) & (flat <= 2.7)).sum(),
            (flat > 2.7).sum(),
        ])

        n = flat.numel()
        flat_d = flat.double()
        s = flat_d.sum()
        sq = (flat_d ** 2).sum()
        mi = flat.min()
        ma = flat.max()

        x_d = x_2d.double()
        pp_sum = x_d.sum(dim=0)
        pp_sq = (x_d ** 2).sum(dim=0)
        pp_min = x_2d.min(dim=0).values.float()
        pp_max = x_2d.max(dim=0).values.float()
        pp_ic = torch.stack([
            (x_2d < -2.7).sum(dim=0),
            ((x_2d >= -2.7) & (x_2d < 0)).sum(dim=0),
            ((x_2d >= 0) & (x_2d <= 2.7)).sum(dim=0),
            (x_2d > 2.7).sum(dim=0),
        ], dim=1)  # (D, 4)

        batch_stats = {
            'hist': hist.long().cpu().numpy(),
            'ic': ic.long().cpu().numpy(),
            'n': n,
            's': s.cpu().item(),
            'sq': sq.cpu().item(),
            'mi': mi.cpu().item(),
            'ma': ma.cpu().item(),
            'pp_n': x_2d.shape[0],
            'pp_sum': pp_sum.cpu().numpy(),
            'pp_sq': pp_sq.cpu().numpy(),
            'pp_min': pp_min.cpu().numpy().astype(np.float64),
            'pp_max': pp_max.cpu().numpy().astype(np.float64),
            'pp_ic': pp_ic.long().cpu().numpy(),
        }
        self.stats_queue.put((self.layer_idx, batch_stats))


def _stats_worker(stats_queue, collector, lock):
    """Merge pre-computed tiny stats dicts into collector — trivially fast."""
    while True:
        item = stats_queue.get()
        if item is None:
            stats_queue.task_done()
            break
        layer_idx, batch_stats = item
        with lock:
            collector.merge_batch_stats(layer_idx, batch_stats)
        stats_queue.task_done()


def install_gelu_hooks(model, collector, stats_queue):
    """Wrap each layer's GELU with a hook that enqueues input x for async collection."""
    original_fns = []
    for i, layer in enumerate(model.bert.encoder.layer):
        orig = layer.intermediate.intermediate_act_fn
        original_fns.append(orig)
        layer.intermediate.intermediate_act_fn = GELUHookWrapper(orig, i, stats_queue)
    return original_fns


def restore_gelu(model, original_fns):
    for i, layer in enumerate(model.bert.encoder.layer):
        layer.intermediate.intermediate_act_fn = original_fns[i]


# ==================== Plotting ====================

def plot_fine_grained_histograms(collector, task_name, output_dir):
    """Grid of per-layer histograms, colored by interval boundaries."""
    ncols = 3
    nrows = (NUM_LAYERS + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 5.5))
    fig.suptitle(f'GELU Input Distribution — {task_name.upper()} (Training Set)',
                 fontsize=18, fontweight='bold', y=0.98)

    for layer_idx in range(NUM_LAYERS):
        row, col = divmod(layer_idx, 3)
        ax = axes[row][col]

        hist = collector.histograms[layer_idx].astype(np.float64)
        centers = collector.bin_centers

        colors = []
        for c in centers:
            if c < -2.7:
                colors.append(INTERVAL_COLORS[0])
            elif c < 0:
                colors.append(INTERVAL_COLORS[1])
            elif c <= 2.7:
                colors.append(INTERVAL_COLORS[2])
            else:
                colors.append(INTERVAL_COLORS[3])

        bin_w = centers[1] - centers[0]
        ax.bar(centers, hist, width=bin_w, color=colors, alpha=0.85, edgecolor='none')

        for b in BOUNDARIES:
            ax.axvline(x=b, color='black', linestyle='--', linewidth=1.5, alpha=0.8)

        ymax = ax.get_ylim()[1]
        ax.text(-2.7, ymax * 0.92, 'x=-2.7', fontsize=7, ha='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        ax.text(0, ymax * 0.92, 'x=0', fontsize=7, ha='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        ax.text(2.7, ymax * 0.92, 'x=2.7', fontsize=7, ha='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

        stats = collector.get_stats(layer_idx)
        title = f'Layer {layer_idx}'
        if stats:
            title += f'  (μ={stats["mean"]:.3f}, σ={stats["std"]:.3f})'
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('x (input to GELU)', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.set_xlim(-10, 10)

        if stats:
            pct = stats['interval_pct']
            text_lines = (f'<-2.7: {pct[0]:.1f}%  |  [-2.7,0): {pct[1]:.1f}%\n'
                          f'[0,2.7]: {pct[2]:.1f}%  |  >2.7: {pct[3]:.1f}%')
            ax.text(0.98, 0.95, text_lines, transform=ax.transAxes, fontsize=7,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

    for layer_idx in range(NUM_LAYERS, nrows * ncols):
        row, col = divmod(layer_idx, ncols)
        axes[row][col].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    filepath = os.path.join(output_dir, f'{task_name}_gelu_distribution.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {filepath}')


def plot_interval_distribution(collector, task_name, output_dir):
    """Two-panel figure: grouped bar chart (top) and stacked bar chart (bottom)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    fig.suptitle(f'GELU Input Interval Distribution — {task_name.upper()} (Training Set)',
                 fontsize=16, fontweight='bold', y=0.98)

    x_pos = np.arange(NUM_LAYERS)
    bar_w = 0.18

    # ---- Top: Grouped bar chart ----
    for iv in range(4):
        pcts = []
        for li in range(NUM_LAYERS):
            s = collector.get_stats(li)
            pcts.append(s['interval_pct'][iv] if s else 0)
        bars = ax1.bar(x_pos + iv * bar_w, pcts, bar_w,
                       label=INTERVAL_LABELS[iv], color=INTERVAL_COLORS[iv], alpha=0.88)
        for bar, val in zip(bars, pcts):
            if val > 1.5:
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                         f'{val:.1f}', ha='center', va='bottom', fontsize=6.5, rotation=90)

    ax1.set_xlabel('Layer', fontsize=11)
    ax1.set_ylabel('Percentage (%)', fontsize=11)
    ax1.set_title('Grouped Bar Chart — Interval Distribution by Layer', fontsize=12)
    ax1.set_xticks(x_pos + 1.5 * bar_w)
    ax1.set_xticklabels([f'L{i}' for i in range(NUM_LAYERS)])
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    # ---- Bottom: Stacked bar chart ----
    bottoms = np.zeros(NUM_LAYERS)
    for iv in range(4):
        pcts = np.array([
            (collector.get_stats(li) or {}).get('interval_pct', [0, 0, 0, 0])[iv]
            for li in range(NUM_LAYERS)
        ])
        ax2.bar(x_pos, pcts, 0.6, bottom=bottoms,
                label=INTERVAL_LABELS[iv], color=INTERVAL_COLORS[iv], alpha=0.88)
        for j in range(NUM_LAYERS):
            if pcts[j] > 4:
                ax2.text(j, bottoms[j] + pcts[j] / 2, f'{pcts[j]:.1f}%',
                         ha='center', va='center', fontsize=7, fontweight='bold')
        bottoms += pcts

    ax2.set_xlabel('Layer', fontsize=11)
    ax2.set_ylabel('Percentage (%)', fontsize=11)
    ax2.set_title('Stacked Bar Chart — Interval Distribution by Layer', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'L{i}' for i in range(NUM_LAYERS)])
    ax2.legend(loc='upper right', fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    filepath = os.path.join(output_dir, f'{task_name}_gelu_intervals.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {filepath}')


def plot_per_position_distribution(collector, task_name, output_dir):
    """Per-position mean ± std across 12 layers."""
    fig, axes = plt.subplots(4, 3, figsize=(20, 22))
    fig.suptitle(f'GELU Input Per-Position Statistics — {task_name.upper()} (Training Set)',
                 fontsize=18, fontweight='bold', y=0.98)

    for layer_idx in range(NUM_LAYERS):
        row, col = divmod(layer_idx, 3)
        ax = axes[row][col]
        stats = collector.get_per_position_stats(layer_idx)
        if stats is None:
            ax.set_title(f'Layer {layer_idx} (no data)')
            continue
        D = stats['num_positions']
        x_pos = np.arange(D)
        mean = stats['mean']
        std = stats['std']
        ax.plot(x_pos, mean, color='#2c3e50', linewidth=0.5, label='mean')
        ax.fill_between(x_pos, mean - std, mean + std,
                        color='#3498db', alpha=0.3, label='mean ± std')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.6)
        ax.set_xlabel('Position (dimension index)', fontsize=9)
        ax.set_ylabel('Value', fontsize=9)
        ax.set_title(f'Layer {layer_idx} — {D} dims  '
                     f'(μ̄={mean.mean():.3f}, σ̄={std.mean():.3f})',
                     fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    filepath = os.path.join(output_dir, f'{task_name}_gelu_per_position.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {filepath}')


def plot_per_position_interval_heatmap(collector, task_name, output_dir):
    """Heatmap: x=position (D), y=4 intervals, color=percentage. One subplot per layer."""
    fig, axes = plt.subplots(NUM_LAYERS, 1, figsize=(20, NUM_LAYERS * 2.2))
    fig.suptitle(f'GELU Input Per-Position Interval % (Heatmap) — {task_name.upper()}',
                 fontsize=16, fontweight='bold', y=1.0)

    for layer_idx in range(NUM_LAYERS):
        ax = axes[layer_idx]
        stats = collector.get_per_position_stats(layer_idx)
        if stats is None:
            ax.set_title(f'Layer {layer_idx} (no data)')
            continue
        pct = stats['interval_pct']  # (D, 4)
        im = ax.imshow(pct.T, aspect='auto', cmap='YlOrRd',
                       vmin=0, vmax=100, interpolation='nearest')
        ax.set_yticks(range(4))
        ax.set_yticklabels(INTERVAL_LABELS, fontsize=8)
        ax.set_ylabel(f'L{layer_idx}', fontsize=10, fontweight='bold', rotation=0,
                      labelpad=25, va='center')
        if layer_idx == NUM_LAYERS - 1:
            ax.set_xlabel('Position (dimension index)', fontsize=10)
        else:
            ax.set_xticklabels([])
        cbar = fig.colorbar(im, ax=ax, fraction=0.01, pad=0.005)
        cbar.ax.tick_params(labelsize=7)
        if layer_idx == 0:
            cbar.set_label('%', fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    filepath = os.path.join(output_dir, f'{task_name}_gelu_per_position_intervals.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {filepath}')


def save_per_position_stats(collector, task_name, output_dir):
    """Save per-position stats to .npz (mean, std, min, max, count per layer)."""
    data = {}
    for layer_idx in range(NUM_LAYERS):
        stats = collector.get_per_position_stats(layer_idx)
        if stats is None:
            continue
        prefix = f'layer{layer_idx}'
        data[f'{prefix}_mean'] = stats['mean']
        data[f'{prefix}_std'] = stats['std']
        data[f'{prefix}_min'] = stats['min']
        data[f'{prefix}_max'] = stats['max']
        data[f'{prefix}_count'] = stats['count']
        data[f'{prefix}_interval_pct'] = stats['interval_pct']
    if not data:
        return
    filepath = os.path.join(output_dir, f'{task_name}_gelu_per_position.npz')
    np.savez_compressed(filepath, **data)
    print(f'  Saved: {filepath}')


# ==================== Per-Task Processing ====================

def process_task(task_name, task_config, output_dir, device,
                 max_length=128, batch_size=32, max_samples=0):
    global NUM_LAYERS
    NUM_LAYERS = int(task_config['num_layers'])
    print(f'\n{"=" * 70}')
    print(f'  Task: {task_name.upper()}')
    print(f'{"=" * 70}')

    model_name = task_config['model_name']
    print(f'  Model : {model_name}')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=task_config['num_labels'],
        pad_token_id=tokenizer.pad_token_id,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    dataloader, protocol_payload = _prepare_bert_data(
        task_config,
        tokenizer,
        max_length,
        batch_size,
        max_samples,
    )
    write_profile_protocol(output_dir, task_name, protocol_payload)

    collector = GELUInputCollector(NUM_LAYERS, HIST_MIN, HIST_MAX, HIST_BINS)
    stats_queue = queue.Queue(maxsize=128)
    lock = threading.Lock()
    worker = threading.Thread(target=_stats_worker, args=(stats_queue, collector, lock),
                              daemon=True)
    worker.start()
    original_fns = install_gelu_hooks(model, collector, stats_queue)

    print(f'  Running forward pass …')
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f'  {task_name}'):
            batch = {k: v.to(device) for k, v in batch.items()}
            model(**batch)

    stats_queue.put(None)
    stats_queue.join()
    worker.join()
    restore_gelu(model, original_fns)

    # ---- Print statistics ----
    header = (f'  {"Layer":<8} {"Count":>14} {"Mean":>9} {"Std":>9} '
              f'{"Min":>9} {"Max":>9}  '
              f'{"<-2.7":>8} {"[-2.7,0)":>9} {"[0,2.7]":>9} {">2.7":>8}')
    print(f'\n{header}')
    print(f'  {"-" * 108}')

    stats_lines = [header, '  ' + '-' * 108]
    for i in range(NUM_LAYERS):
        s = collector.get_stats(i)
        if s:
            p = s['interval_pct']
            line = (f'  Layer {i:<3} {s["count"]:>14,} {s["mean"]:>9.4f} {s["std"]:>9.4f} '
                    f'{s["min"]:>9.3f} {s["max"]:>9.3f}  '
                    f'{p[0]:>7.2f}% {p[1]:>8.2f}% {p[2]:>8.2f}% {p[3]:>7.2f}%')
            print(line)
            stats_lines.append(line)

    stats_path = os.path.join(output_dir, f'{task_name}_gelu_stats.txt')
    with open(stats_path, 'w') as f:
        f.write(f'GELU Input Distribution Statistics — {task_name.upper()}\n')
        f.write(f'Boundary points: -2.7, 0, 2.7\n')
        f.write(f'Intervals: x<-2.7 | -2.7<=x<0 | 0<=x<=2.7 | x>2.7\n')
        f.write(f'Per-position (per dimension) stats: see {task_name}_gelu_per_position.npz and .png\n\n')
        for line in stats_lines:
            f.write(line + '\n')
        f.write('\n')
        for i in range(NUM_LAYERS):
            s = collector.get_stats(i)
            if s:
                f.write(f'Layer {i}: interval_counts = {s["interval_counts"]}\n')
    print(f'  Saved: {stats_path}')

    # ---- Plot ----
    plot_fine_grained_histograms(collector, task_name, output_dir)
    plot_interval_distribution(collector, task_name, output_dir)
    plot_per_position_distribution(collector, task_name, output_dir)
    plot_per_position_interval_heatmap(collector, task_name, output_dir)
    save_per_position_stats(collector, task_name, output_dir)

    del model, collector
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(
        description='Analyze GELU(x) input distribution on GLUE training sets'
    )
    parser.add_argument('--output_dir', type=str, default='gelu_analysis',
                        help='Output directory (default: gelu_analysis)')
    parser.add_argument('--tasks', type=str, nargs='+', default=None,
                        help='Tasks to analyze (default: all supported profiles)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (default: cuda)')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Max sequence length (default: 128)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--max_samples', type=int, default=0,
                        help='Formal probe size; accepted values are 0 and 256')
    args = parser.parse_args()

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print('[Warning] CUDA not available, falling back to CPU')
        device = 'cpu'

    os.makedirs(args.output_dir, exist_ok=True)

    tasks = args.tasks or list(TASK_REGISTRY.keys())
    print(f'Tasks       : {tasks}')
    print(f'Output dir  : {args.output_dir}')
    print(f'Device      : {device}')
    print(f'Batch size  : {args.batch_size}')
    print(f'Max samples : {"all" if args.max_samples == 0 else args.max_samples}')

    for task_name in tasks:
        if task_name not in TASK_REGISTRY:
            print(f'\n[Warning] Unknown task "{task_name}", skipping')
            continue
        process_task(task_name, TASK_REGISTRY[task_name], args.output_dir,
                     device, args.max_length, args.batch_size, args.max_samples)

    print(f'\n{"=" * 70}')
    print(f'  All done! Results saved to {args.output_dir}/')
    print(f'{"=" * 70}')


if __name__ == '__main__':
    main()
