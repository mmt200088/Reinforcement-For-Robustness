#!/usr/bin/env python
"""
Block 3: Cross-Task Differential Verification & Robustness Analysis

Groups GLUE tasks by type, evaluates validated configurations from Block 1/2
across all 8 datasets, and performs Spearman rank correlation analysis to
test whether approximation error effects are consistent across tasks.

Usage:
    python experiment_block3_cross_task.py --device cuda --output_dir results/block3
    python experiment_block3_cross_task.py --input_dir results/ --output_dir results/block3
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
from scipy.stats import spearmanr
from itertools import combinations

from experiment_core import (
    TASK_REGISTRY, ALL_TASKS, NUM_LAYERS, TASK_GROUPS,
    GELU_FULL, GELU_LOW, SOFTMAX_FULL, SOFTMAX_LOW, BASELINE_CONFIG,
    load_model_and_data, evaluate_config, get_primary_metric,
)


def get_test_configurations():
    """
    Define a fixed set of controlled configurations for cross-task testing.
    Includes baseline, industry standard, and systematic degradation combos.
    """
    configs = []

    configs.append({
        'name': 'C0_baseline',
        'description': 'Full precision (GELU=4, Softmax=6 all layers)',
        'gelu': [GELU_FULL] * NUM_LAYERS,
        'softmax': [SOFTMAX_FULL] * NUM_LAYERS,
    })

    for layer in [0, 3, 6, 11]:
        g = [GELU_FULL] * NUM_LAYERS
        s = [SOFTMAX_FULL] * NUM_LAYERS
        g[layer] = GELU_LOW
        configs.append({
            'name': f'C_gelu_L{layer}',
            'description': f'GELU L{layer} degraded to {GELU_LOW}',
            'gelu': g, 'softmax': s,
        })

    for layer in [1, 5, 8, 10]:
        g = [GELU_FULL] * NUM_LAYERS
        s = [SOFTMAX_FULL] * NUM_LAYERS
        s[layer] = SOFTMAX_LOW
        configs.append({
            'name': f'C_sm_L{layer}',
            'description': f'Softmax L{layer} degraded to {SOFTMAX_LOW}',
            'gelu': g, 'softmax': s,
        })

    for li, lj in [(1, 5), (3, 8), (0, 11)]:
        g = [GELU_FULL] * NUM_LAYERS
        s = [SOFTMAX_FULL] * NUM_LAYERS
        g[li] = GELU_LOW
        s[lj] = SOFTMAX_LOW
        configs.append({
            'name': f'C_joint_G{li}_S{lj}',
            'description': f'GELU L{li} + Softmax L{lj} degraded',
            'gelu': g, 'softmax': s,
        })

    g_all_low = [GELU_LOW] * NUM_LAYERS
    s_all_low = [SOFTMAX_LOW] * NUM_LAYERS
    configs.append({
        'name': 'C_all_low',
        'description': f'All layers GELU={GELU_LOW}, Softmax={SOFTMAX_LOW}',
        'gelu': g_all_low, 'softmax': s_all_low,
    })

    g_mid = [2] * NUM_LAYERS
    s_mid = [4] * NUM_LAYERS
    configs.append({
        'name': 'C_mid_approx',
        'description': 'All layers GELU=2, Softmax=4',
        'gelu': g_mid, 'softmax': s_mid,
    })

    return configs


def evaluate_configs_across_tasks(configs, tasks, device='cuda', max_length=128, batch_size=16):
    """
    Evaluate all configurations on all tasks.
    Returns performance_matrix[config_idx][task_name] = primary_metric_value.
    """
    import gc, torch

    performance = {cfg['name']: {} for cfg in configs}

    for task in tasks:
        if task not in TASK_REGISTRY:
            continue

        print(f"\n  Evaluating task: {task.upper()}")
        model, handler, layers_attr, dataloader, labels, task_cfg = load_model_and_data(
            task, device, max_length, batch_size
        )
        primary = get_primary_metric(task)

        for cfg in configs:
            metrics = evaluate_config(
                model, handler, layers_attr, dataloader, labels, task,
                cfg['gelu'], cfg['softmax'], device
            )
            performance[cfg['name']][task] = metrics[primary]
            print(f"    {cfg['name']:30s}: {primary}={metrics[primary]:.4f}")

        del model, handler, dataloader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return performance


def compute_spearman_matrix(performance, configs, tasks):
    """
    Compute Spearman rank correlation between task pairs based on config ranking.
    Returns (rho_matrix, p_matrix, task_names).
    """
    n_tasks = len(tasks)
    rho_matrix = np.ones((n_tasks, n_tasks))
    p_matrix = np.zeros((n_tasks, n_tasks))

    config_names = [c['name'] for c in configs]

    task_rankings = {}
    for task in tasks:
        values = [performance[cn][task] for cn in config_names]
        task_rankings[task] = values

    for i, task_i in enumerate(tasks):
        for j, task_j in enumerate(tasks):
            if i == j:
                continue
            rho, p = spearmanr(task_rankings[task_i], task_rankings[task_j])
            rho_matrix[i, j] = rho
            p_matrix[i, j] = p

    return rho_matrix, p_matrix


def plot_spearman_heatmap(rho_matrix, p_matrix, tasks, output_dir, suffix=""):
    """Plot Spearman rho heatmap with significance markers."""
    n = len(tasks)
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(rho_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')

    for i in range(n):
        for j in range(n):
            if i == j:
                text = "1.00"
                color = 'white'
            else:
                rho = rho_matrix[i, j]
                p = p_matrix[i, j]
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                text = f"{rho:.2f}{sig}"
                color = 'white' if abs(rho) > 0.5 else 'black'
            ax.text(j, i, text, ha='center', va='center', fontsize=8, color=color)

    task_labels = [t.upper() for t in tasks]
    ax.set_xticks(range(n))
    ax.set_xticklabels(task_labels, fontsize=10, rotation=45, ha='right')
    ax.set_yticks(range(n))
    ax.set_yticklabels(task_labels, fontsize=10)

    cbar = fig.colorbar(im, ax=ax, label='Spearman rho', shrink=0.8)
    ax.set_title(f'Cross-Task Config Ranking Consistency (Spearman rho){suffix}', fontsize=13)

    group_colors = {'single_sentence': '#e74c3c', 'similarity_paraphrase': '#3498db', 'nli': '#2ecc71'}
    for i, task in enumerate(tasks):
        for group_name, group_tasks in TASK_GROUPS.items():
            if task in group_tasks:
                ax.get_yticklabels()[i].set_color(group_colors.get(group_name, 'black'))
                ax.get_xticklabels()[i].set_color(group_colors.get(group_name, 'black'))

    fig.tight_layout()
    fname = f'block3_spearman_heatmap{suffix}.png'
    fig.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {os.path.join(output_dir, fname)}")


def plot_ranking_profiles(performance, configs, tasks, output_dir):
    """Plot config performance profiles across tasks."""
    config_names = [c['name'] for c in configs]

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, len(config_names)))

    for ci, cn in enumerate(config_names):
        values = [performance[cn].get(t, 0) for t in tasks]
        if cn == 'C0_baseline':
            ax.plot(range(len(tasks)), values, 'k-', linewidth=2.5, marker='D', markersize=8,
                    label='Baseline', zorder=10)
        else:
            ax.plot(range(len(tasks)), values, '-', color=colors[ci], linewidth=1.2,
                    marker='o', markersize=5, alpha=0.7, label=cn)

    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([t.upper() for t in tasks], fontsize=10, rotation=45, ha='right')
    ax.set_ylabel('Primary Metric Value', fontsize=12)
    ax.set_title('Configuration Performance Profiles Across GLUE Tasks', fontsize=14)
    ax.legend(fontsize=7, ncol=3, loc='lower left')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'block3_ranking_profiles.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_group_analysis(rho_matrix, p_matrix, tasks, output_dir):
    """Analyze within-group vs between-group consistency."""
    within_rhos, between_rhos = [], []

    for (i, ti), (j, tj) in combinations(enumerate(tasks), 2):
        same_group = False
        for group_tasks in TASK_GROUPS.values():
            if ti in group_tasks and tj in group_tasks:
                same_group = True
                break

        rho = rho_matrix[i, j]
        if same_group:
            within_rhos.append(rho)
        else:
            between_rhos.append(rho)

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot([within_rhos, between_rhos],
                     labels=['Within-Group', 'Between-Group'],
                     patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][1].set_facecolor('#e74c3c')

    ax.set_ylabel('Spearman rho', fontsize=12)
    ax.set_title('Ranking Consistency: Within-Group vs Between-Group', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')

    ax.text(1, max(within_rhos) + 0.03 if within_rhos else 0.5,
            f'n={len(within_rhos)}, mean={np.mean(within_rhos):.3f}' if within_rhos else '',
            ha='center', fontsize=9)
    ax.text(2, max(between_rhos) + 0.03 if between_rhos else 0.5,
            f'n={len(between_rhos)}, mean={np.mean(between_rhos):.3f}' if between_rhos else '',
            ha='center', fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'block3_group_analysis.png'), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Block 3: Cross-task verification & robustness")
    parser.add_argument("--tasks", nargs='+', default=None, help="Tasks (default: all 8)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="results/block3")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tasks = args.tasks if args.tasks else ALL_TASKS

    configs = get_test_configurations()
    print(f"Testing {len(configs)} configurations across {len(tasks)} tasks")

    performance = evaluate_configs_across_tasks(
        configs, tasks, args.device, args.max_length, args.batch_size
    )

    rho_matrix, p_matrix = compute_spearman_matrix(performance, configs, tasks)

    print(f"\n{'='*60}")
    print(f"  Spearman Rank Correlation Matrix")
    print(f"{'='*60}")
    print(f"  {'':>8}", end='')
    for t in tasks:
        print(f"  {t.upper():>8}", end='')
    print()
    for i, ti in enumerate(tasks):
        print(f"  {ti.upper():>8}", end='')
        for j, tj in enumerate(tasks):
            if i == j:
                print(f"  {'1.00':>8}", end='')
            else:
                sig = "*" if p_matrix[i, j] < 0.05 else " "
                print(f"  {rho_matrix[i,j]:>7.3f}{sig}", end='')
        print()

    plot_spearman_heatmap(rho_matrix, p_matrix, tasks, args.output_dir)
    plot_ranking_profiles(performance, configs, tasks, args.output_dir)
    plot_group_analysis(rho_matrix, p_matrix, tasks, args.output_dir)

    low_rho_pairs = []
    for (i, ti), (j, tj) in combinations(enumerate(tasks), 2):
        rho = rho_matrix[i, j]
        p = p_matrix[i, j]
        if rho < 0.5 or p > 0.05:
            low_rho_pairs.append({
                'task_i': ti, 'task_j': tj,
                'rho': float(rho), 'p': float(p),
                'significant': bool(p < 0.05),
            })

    result = {
        'configs': [{'name': c['name'], 'description': c['description'],
                     'gelu': c['gelu'], 'softmax': c['softmax']} for c in configs],
        'performance': performance,
        'spearman_rho': rho_matrix.tolist(),
        'spearman_p': p_matrix.tolist(),
        'tasks': tasks,
        'low_consistency_pairs': low_rho_pairs,
        'task_groups': TASK_GROUPS,
    }

    with open(os.path.join(args.output_dir, 'block3_results.json'), 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  BLOCK 3 KEY FINDINGS")
    print(f"{'='*60}")
    print(f"  Low-consistency task pairs (rho < 0.5 or p > 0.05):")
    for lp in low_rho_pairs:
        print(f"    {lp['task_i'].upper()} vs {lp['task_j'].upper()}: "
              f"rho={lp['rho']:.3f}, p={lp['p']:.4f}")

    if low_rho_pairs:
        print(f"\n  CONCLUSION: {len(low_rho_pairs)} task pairs show inconsistent config rankings,")
        print(f"  providing evidence that approximation error effects are task-dependent")
        print(f"  and NOT simply accumulative across all contexts.")
    else:
        print(f"\n  All task pairs show consistent rankings (rho >= 0.5, p < 0.05).")

    print(f"\nBlock 3 completed. Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
