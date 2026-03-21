#!/usr/bin/env python
"""
Supplementary Test 1: Single-Layer Degradation Panorama

For each GLUE dataset, degrade one layer at a time (GELU->1 or Softmax->2)
while keeping all other layers at full precision. Plot per-dataset results
and a summary heatmap.

Usage:
    python experiment_single_layer_degradation.py --device cuda --output_dir results/single_layer
    python experiment_single_layer_degradation.py --tasks sst2 mrpc --device cuda
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']

from experiment_core import (
    TASK_REGISTRY, ALL_TASKS, NUM_LAYERS,
    GELU_FULL, GELU_LOW, SOFTMAX_FULL, SOFTMAX_LOW, BASELINE_CONFIG,
    load_model_and_data, evaluate_config, get_primary_metric,
)


def run_single_layer_degradation(task_name, device='cuda', max_length=128, batch_size=16):
    """
    For a given task, evaluate:
      - Full precision baseline
      - 12 configs with GELU layer i -> 1 (others 4)
      - 12 configs with Softmax layer i -> 2 (others 6)
    Returns dict with baseline and per-layer metrics.
    """
    print(f"\n{'='*60}")
    print(f"  Single-Layer Degradation: {task_name.upper()}")
    print(f"{'='*60}")

    model, handler, layers_attr, dataloader, labels, task_cfg = load_model_and_data(
        task_name, device, max_length, batch_size
    )
    primary = get_primary_metric(task_name)
    all_metric_keys = task_cfg['all_metrics']

    baseline_metrics = evaluate_config(
        model, handler, layers_attr, dataloader, labels, task_name,
        BASELINE_CONFIG['gelu'], BASELINE_CONFIG['softmax'], device
    )
    print(f"  Baseline: {baseline_metrics}")

    gelu_results = []
    for layer_idx in range(NUM_LAYERS):
        gelu_deg = [GELU_FULL] * NUM_LAYERS
        gelu_deg[layer_idx] = GELU_LOW
        softmax_deg = [SOFTMAX_FULL] * NUM_LAYERS

        m = evaluate_config(
            model, handler, layers_attr, dataloader, labels, task_name,
            gelu_deg, softmax_deg, device
        )
        gelu_results.append(m)
        print(f"  GELU Layer {layer_idx} -> {GELU_LOW}: {m[primary]:.4f}")

    softmax_results = []
    for layer_idx in range(NUM_LAYERS):
        gelu_deg = [GELU_FULL] * NUM_LAYERS
        softmax_deg = [SOFTMAX_FULL] * NUM_LAYERS
        softmax_deg[layer_idx] = SOFTMAX_LOW

        m = evaluate_config(
            model, handler, layers_attr, dataloader, labels, task_name,
            gelu_deg, softmax_deg, device
        )
        softmax_results.append(m)
        print(f"  Softmax Layer {layer_idx} -> {SOFTMAX_LOW}: {m[primary]:.4f}")

    import gc, torch as _torch
    del model, handler, dataloader
    gc.collect()
    if _torch.cuda.is_available():
        _torch.cuda.empty_cache()

    return {
        'task': task_name,
        'primary_metric': primary,
        'all_metrics': all_metric_keys,
        'metric_names': task_cfg['metric_names'],
        'baseline': baseline_metrics,
        'gelu_degradation': gelu_results,
        'softmax_degradation': softmax_results,
    }


def plot_single_task(result, output_dir):
    """Plot single-layer degradation for one task."""
    task = result['task']
    primary = result['primary_metric']
    baseline_val = result['baseline'][primary]
    metric_label = result['metric_names'][0]

    gelu_vals = [r[primary] for r in result['gelu_degradation']]
    softmax_vals = [r[primary] for r in result['softmax_degradation']]
    layers = list(range(NUM_LAYERS))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, gelu_vals, 'o-', color='#e74c3c', label=f'GELU -> {GELU_LOW}', linewidth=2, markersize=6)
    ax.plot(layers, softmax_vals, 's-', color='#3498db', label=f'Softmax -> {SOFTMAX_LOW}', linewidth=2, markersize=6)
    ax.axhline(y=baseline_val, color='#2ecc71', linestyle='--', linewidth=2, label=f'Baseline ({baseline_val:.4f})')

    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_title(f'{task.upper()} - Single Layer Degradation ({metric_label})', fontsize=14)
    ax.set_xticks(layers)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, f'single_layer_degradation_{task}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    all_metrics = result['all_metrics']
    if len(all_metrics) > 1:
        for mk, mn in zip(all_metrics, result['metric_names']):
            if mk == primary:
                continue
            g_vals = [r[mk] for r in result['gelu_degradation']]
            s_vals = [r[mk] for r in result['softmax_degradation']]
            b_val = result['baseline'][mk]

            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.plot(layers, g_vals, 'o-', color='#e74c3c', label=f'GELU -> {GELU_LOW}', linewidth=2)
            ax2.plot(layers, s_vals, 's-', color='#3498db', label=f'Softmax -> {SOFTMAX_LOW}', linewidth=2)
            ax2.axhline(y=b_val, color='#2ecc71', linestyle='--', linewidth=2, label=f'Baseline ({b_val:.4f})')
            ax2.set_xlabel('Layer Index', fontsize=12)
            ax2.set_ylabel(mn, fontsize=12)
            ax2.set_title(f'{task.upper()} - Single Layer Degradation ({mn})', fontsize=14)
            ax2.set_xticks(layers)
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
            fig2.tight_layout()
            fig2.savefig(os.path.join(output_dir, f'single_layer_degradation_{task}_{mk}.png'), dpi=150)
            plt.close(fig2)


def plot_summary_heatmap(all_results, output_dir):
    """Plot summary heatmap: relative change from baseline for all tasks and layers."""
    tasks = [r['task'] for r in all_results]
    n_tasks = len(tasks)

    gelu_matrix = np.zeros((n_tasks, NUM_LAYERS))
    softmax_matrix = np.zeros((n_tasks, NUM_LAYERS))

    for i, r in enumerate(all_results):
        primary = r['primary_metric']
        baseline = r['baseline'][primary]
        if abs(baseline) < 1e-10:
            baseline = 1e-10
        for j in range(NUM_LAYERS):
            gelu_matrix[i, j] = (r['gelu_degradation'][j][primary] - baseline) / abs(baseline) * 100
            softmax_matrix[i, j] = (r['softmax_degradation'][j][primary] - baseline) / abs(baseline) * 100

    fig, axes = plt.subplots(1, 2, figsize=(20, max(4, n_tasks * 0.7 + 2)))

    vmin = min(gelu_matrix.min(), softmax_matrix.min())
    vmax = max(gelu_matrix.max(), softmax_matrix.max())
    abs_max = max(abs(vmin), abs(vmax))

    for ax, matrix, title in [
        (axes[0], gelu_matrix, f'GELU Layer Degradation (degree {GELU_FULL}->{GELU_LOW})'),
        (axes[1], softmax_matrix, f'Softmax Layer Degradation (degree {SOFTMAX_FULL}->{SOFTMAX_LOW})'),
    ]:
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=-abs_max, vmax=abs_max)
        ax.set_xticks(range(NUM_LAYERS))
        ax.set_xticklabels([str(i) for i in range(NUM_LAYERS)])
        ax.set_yticks(range(n_tasks))
        ax.set_yticklabels([t.upper() for t in tasks])
        ax.set_xlabel('Layer Index')
        ax.set_title(title, fontsize=12)

        for yi in range(n_tasks):
            for xi in range(NUM_LAYERS):
                val = matrix[yi, xi]
                ax.text(xi, yi, f'{val:.1f}%', ha='center', va='center', fontsize=7,
                        color='white' if abs(val) > abs_max * 0.6 else 'black')

    fig.colorbar(im, ax=axes, label='Relative Change from Baseline (%)', shrink=0.8)
    fig.suptitle('Single-Layer Degradation Impact (% change in primary metric)', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'single_layer_degradation_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {os.path.join(output_dir, 'single_layer_degradation_heatmap.png')}")


def main():
    parser = argparse.ArgumentParser(description="Single-layer degradation test for all GLUE tasks")
    parser.add_argument("--tasks", nargs='+', default=None, help="Tasks to run (default: all)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="results/single_layer")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tasks = args.tasks if args.tasks else ALL_TASKS

    all_results = []
    for task in tasks:
        if task not in TASK_REGISTRY:
            print(f"[Warning] Unknown task '{task}', skipping")
            continue
        result = run_single_layer_degradation(task, args.device, args.max_length, args.batch_size)
        all_results.append(result)

        plot_single_task(result, args.output_dir)

        result_serializable = {
            'task': result['task'],
            'primary_metric': result['primary_metric'],
            'baseline': result['baseline'],
            'gelu_degradation': result['gelu_degradation'],
            'softmax_degradation': result['softmax_degradation'],
        }
        with open(os.path.join(args.output_dir, f'single_layer_{task}.json'), 'w') as f:
            json.dump(result_serializable, f, indent=2)

    if len(all_results) > 1:
        plot_summary_heatmap(all_results, args.output_dir)

    with open(os.path.join(args.output_dir, 'single_layer_all_results.json'), 'w') as f:
        serializable = []
        for r in all_results:
            serializable.append({
                'task': r['task'],
                'primary_metric': r['primary_metric'],
                'baseline': r['baseline'],
                'gelu_degradation': r['gelu_degradation'],
                'softmax_degradation': r['softmax_degradation'],
            })
        json.dump(serializable, f, indent=2)

    print(f"\nAll single-layer degradation tests completed. Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
