#!/usr/bin/env python
"""
Block 2: Two-Way ANOVA Interaction Effect Analysis

Performs 2x2 factorial design: Factor A (GELU of layer i, 2 levels: full/low),
Factor B (Softmax of layer j, 2 levels: full/low). Tests for significant
A*B interaction via two-way ANOVA, interaction plots, and baseline comparison.

Usage:
    python -m experiment.scripts.blocks.block2_anova --device cuda
    python -m experiment.scripts.blocks.block2_anova --tasks sst2 mrpc --n_bootstrap 100
"""

import os
import json
import argparse
import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
from scipy import stats
import warnings

from experiment.core.experiment_core import (
    TASK_REGISTRY, ALL_TASKS, NUM_LAYERS,
    GELU_FULL, GELU_LOW, SOFTMAX_FULL, SOFTMAX_LOW, BASELINE_CONFIG,
    load_model_and_data, get_logits_for_config, compute_metrics,
    bootstrap_metric, get_primary_metric,
)

try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    warnings.warn("statsmodels not installed. Will use scipy-based ANOVA fallback.")


LAYER_PAIRS = [
    (1, 5), (3, 8), (0, 11), (6, 10),
]


def build_factorial_configs(layer_i, layer_j):
    """
    Build 2x2 factorial configs for GELU(layer_i) x Softmax(layer_j).
    All other layers at full precision.
    Returns list of 4 config dicts with factor labels.
    """
    configs = []
    for gelu_level, gelu_label in [(GELU_FULL, 'full'), (GELU_LOW, 'low')]:
        for sm_level, sm_label in [(SOFTMAX_FULL, 'full'), (SOFTMAX_LOW, 'low')]:
            g = [GELU_FULL] * NUM_LAYERS
            s = [SOFTMAX_FULL] * NUM_LAYERS
            g[layer_i] = gelu_level
            s[layer_j] = sm_level
            configs.append({
                'gelu': g, 'softmax': s,
                'factor_a': gelu_label,
                'factor_b': sm_label,
                'label': f'GELU={gelu_label},SM={sm_label}',
            })
    return configs


def run_anova_for_pair(layer_i, layer_j, model, handler, layers_attr, dataloader,
                        labels, task_name, device, n_bootstrap=100):
    """
    Run 2x2 factorial experiment for one (layer_i, layer_j) pair.
    Returns ANOVA results and group statistics.
    """
    primary = get_primary_metric(task_name)
    configs = build_factorial_configs(layer_i, layer_j)

    industry_logits, _ = get_logits_for_config(
        model, handler, layers_attr, dataloader,
        BASELINE_CONFIG['gelu'], BASELINE_CONFIG['softmax'], device
    )

    group_data = {}
    for cfg in configs:
        logits, _ = get_logits_for_config(
            model, handler, layers_attr, dataloader,
            cfg['gelu'], cfg['softmax'], device
        )
        boot = bootstrap_metric_from_logits(logits, labels, task_name, n_bootstrap, primary)
        group_data[cfg['label']] = {
            'factor_a': cfg['factor_a'],
            'factor_b': cfg['factor_b'],
            'boot_samples': boot,
            'mean': float(np.mean(boot)),
            'std': float(np.std(boot)),
            'single_metric': float(compute_metrics(logits, labels, task_name)[primary]),
        }

    industry_boot = bootstrap_metric_from_logits(industry_logits, labels, task_name, n_bootstrap, primary)
    industry_stats = {
        'mean': float(np.mean(industry_boot)),
        'std': float(np.std(industry_boot)),
        'single_metric': float(compute_metrics(industry_logits, labels, task_name)[primary]),
        'boot_samples': industry_boot,
    }

    anova_result = perform_two_way_anova(group_data, n_bootstrap)

    baseline_comparisons = {}
    for label, gd in group_data.items():
        t_stat, p_val = stats.ttest_ind(gd['boot_samples'], industry_boot)
        baseline_comparisons[label] = {
            'group_mean': gd['mean'],
            'baseline_mean': industry_stats['mean'],
            'diff': gd['mean'] - industry_stats['mean'],
            't_stat': float(t_stat),
            'p_value': float(p_val),
            'significant': bool(p_val < 0.05),
        }

    return {
        'layer_i': layer_i, 'layer_j': layer_j,
        'group_data': {k: {kk: vv for kk, vv in v.items() if kk != 'boot_samples'}
                       for k, v in group_data.items()},
        'group_boot_samples': {k: v['boot_samples'].tolist() for k, v in group_data.items()},
        'industry_baseline': {k: v for k, v in industry_stats.items() if k != 'boot_samples'},
        'anova': anova_result,
        'baseline_comparisons': baseline_comparisons,
    }


def bootstrap_metric_from_logits(logits, labels, task_name, n_bootstrap, primary_metric):
    """Bootstrap resample and compute a single metric."""
    n = len(labels)
    rng = np.random.RandomState(42)
    samples = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        m = compute_metrics(logits[idx], labels[idx], task_name)
        samples.append(m[primary_metric])
    return np.array(samples)


def perform_two_way_anova(group_data, n_bootstrap):
    """
    Perform two-way ANOVA on the factorial data.
    Uses statsmodels if available, otherwise scipy fallback.
    """
    import pandas as pd

    rows = []
    for label, gd in group_data.items():
        for val in gd['boot_samples']:
            rows.append({
                'factor_a': 1 if gd['factor_a'] == 'low' else 0,
                'factor_b': 1 if gd['factor_b'] == 'low' else 0,
                'metric': val,
            })
    df = pd.DataFrame(rows)

    if HAS_STATSMODELS:
        df['A'] = df['factor_a'].astype('category')
        df['B'] = df['factor_b'].astype('category')
        model = ols('metric ~ C(A) * C(B)', data=df).fit()
        anova_table = anova_lm(model, typ=2)

        result = {
            'method': 'statsmodels_type2_anova',
            'factor_a_f': float(anova_table.loc['C(A)', 'F']),
            'factor_a_p': float(anova_table.loc['C(A)', 'PR(>F)']),
            'factor_b_f': float(anova_table.loc['C(B)', 'F']),
            'factor_b_p': float(anova_table.loc['C(B)', 'PR(>F)']),
            'interaction_f': float(anova_table.loc['C(A):C(B)', 'F']),
            'interaction_p': float(anova_table.loc['C(A):C(B)', 'PR(>F)']),
        }

        ss_total = anova_table['sum_sq'].sum()
        for effect, key in [('C(A)', 'factor_a'), ('C(B)', 'factor_b'), ('C(A):C(B)', 'interaction')]:
            result[f'{key}_eta_sq'] = float(anova_table.loc[effect, 'sum_sq'] / ss_total)

        result['factor_a_sig'] = bool(result['factor_a_p'] < 0.05)
        result['factor_b_sig'] = bool(result['factor_b_p'] < 0.05)
        result['interaction_sig'] = bool(result['interaction_p'] < 0.05)

    else:
        g00 = df[(df['factor_a'] == 0) & (df['factor_b'] == 0)]['metric'].values
        g10 = df[(df['factor_a'] == 1) & (df['factor_b'] == 0)]['metric'].values
        g01 = df[(df['factor_a'] == 0) & (df['factor_b'] == 1)]['metric'].values
        g11 = df[(df['factor_a'] == 1) & (df['factor_b'] == 1)]['metric'].values

        grand_mean = df['metric'].mean()
        n_per = len(g00)

        a0_mean = np.mean(np.concatenate([g00, g01]))
        a1_mean = np.mean(np.concatenate([g10, g11]))
        b0_mean = np.mean(np.concatenate([g00, g10]))
        b1_mean = np.mean(np.concatenate([g01, g11]))

        ss_a = 2 * n_per * ((a0_mean - grand_mean)**2 + (a1_mean - grand_mean)**2)
        ss_b = 2 * n_per * ((b0_mean - grand_mean)**2 + (b1_mean - grand_mean)**2)

        cell_means = [np.mean(g00), np.mean(g10), np.mean(g01), np.mean(g11)]
        a_levels = [0, 1, 0, 1]
        b_levels = [0, 0, 1, 1]
        a_means = [a0_mean, a1_mean]
        b_means = [b0_mean, b1_mean]
        ss_ab = n_per * sum((cm - a_means[a] - b_means[b] + grand_mean)**2
                            for cm, a, b in zip(cell_means, a_levels, b_levels))

        ss_within = sum(np.sum((g - np.mean(g))**2) for g in [g00, g10, g01, g11])
        df_a, df_b, df_ab = 1, 1, 1
        df_within = 4 * (n_per - 1)

        ms_a = ss_a / df_a
        ms_b = ss_b / df_b
        ms_ab = ss_ab / df_ab
        ms_within = ss_within / df_within if df_within > 0 else 1e-10

        f_a = ms_a / ms_within
        f_b = ms_b / ms_within
        f_ab = ms_ab / ms_within

        p_a = 1 - stats.f.cdf(f_a, df_a, df_within)
        p_b = 1 - stats.f.cdf(f_b, df_b, df_within)
        p_ab = 1 - stats.f.cdf(f_ab, df_ab, df_within)

        ss_total = ss_a + ss_b + ss_ab + ss_within
        result = {
            'method': 'scipy_manual_anova',
            'factor_a_f': float(f_a), 'factor_a_p': float(p_a),
            'factor_b_f': float(f_b), 'factor_b_p': float(p_b),
            'interaction_f': float(f_ab), 'interaction_p': float(p_ab),
            'factor_a_eta_sq': float(ss_a / ss_total),
            'factor_b_eta_sq': float(ss_b / ss_total),
            'interaction_eta_sq': float(ss_ab / ss_total),
            'factor_a_sig': bool(p_a < 0.05),
            'factor_b_sig': bool(p_b < 0.05),
            'interaction_sig': bool(p_ab < 0.05),
        }

    return result


def plot_interaction(pair_result, task_name, output_dir):
    """Plot interaction effect diagram for one layer pair."""
    layer_i = pair_result['layer_i']
    layer_j = pair_result['layer_j']
    gd = pair_result['group_data']
    anova = pair_result['anova']
    primary = get_primary_metric(task_name)

    means = {}
    for label, data in gd.items():
        means[(data['factor_a'], data['factor_b'])] = data['mean']

    fig, ax = plt.subplots(figsize=(8, 6))

    x = [0, 1]
    x_labels = ['Full Precision', f'Low (degree {SOFTMAX_LOW})']

    for gelu_level, color, marker in [('full', '#2ecc71', 'o'), ('low', '#e74c3c', 's')]:
        y = [means[(gelu_level, 'full')], means[(gelu_level, 'low')]]
        gelu_label = f'GELU=Full (deg {GELU_FULL})' if gelu_level == 'full' else f'GELU=Low (deg {GELU_LOW})'
        ax.plot(x, y, f'{marker}-', color=color, linewidth=2.5, markersize=10, label=gelu_label)

    if pair_result['industry_baseline']:
        ax.axhline(y=pair_result['industry_baseline']['mean'], color='gray',
                   linestyle=':', linewidth=1.5, alpha=0.7,
                   label=f"Industry Baseline ({pair_result['industry_baseline']['mean']:.4f})")

    ax.set_xticks(x)
    ax.set_xticklabels([f'Softmax Full\n(degree {SOFTMAX_FULL})',
                        f'Softmax Low\n(degree {SOFTMAX_LOW})'])
    ax.set_ylabel(f'{primary}', fontsize=12)

    interaction_str = (f"A*B: F={anova['interaction_f']:.2f}, p={anova['interaction_p']:.4f}"
                       + (" ***" if anova['interaction_p'] < 0.001
                          else " **" if anova['interaction_p'] < 0.01
                          else " *" if anova['interaction_p'] < 0.05 else " n.s."))

    ax.set_title(f'{task_name.upper()} - Interaction: GELU(L{layer_i}) x Softmax(L{layer_j})\n{interaction_str}',
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, f'block2_interaction_{task_name}_L{layer_i}_L{layer_j}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_anova_summary(all_pair_results, task_name, output_dir):
    """Plot summary of ANOVA results across all layer pairs."""
    n = len(all_pair_results)
    if n == 0:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    labels_list = [f"L{r['layer_i']}-L{r['layer_j']}" for r in all_pair_results]

    for ax, effect, title in [
        (axes[0], 'factor_a', 'Factor A (GELU) Main Effect'),
        (axes[1], 'factor_b', 'Factor B (Softmax) Main Effect'),
        (axes[2], 'interaction', 'A x B Interaction Effect'),
    ]:
        p_values = [r['anova'][f'{effect}_p'] for r in all_pair_results]
        eta_sq = [r['anova'][f'{effect}_eta_sq'] for r in all_pair_results]

        colors = ['#e74c3c' if p < 0.05 else '#95a5a6' for p in p_values]
        bars = ax.bar(range(n), eta_sq, color=colors, edgecolor='white', linewidth=0.5)

        for i, (p, e) in enumerate(zip(p_values, eta_sq)):
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            ax.text(i, e + 0.002, f'p={p:.3f}{sig}', ha='center', fontsize=7, rotation=45)

        ax.set_xticks(range(n))
        ax.set_xticklabels(labels_list, fontsize=9)
        ax.set_ylabel('Eta-squared', fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.axhline(y=0.01, color='gray', linestyle=':', alpha=0.5, label='Small effect')
        ax.axhline(y=0.06, color='gray', linestyle='--', alpha=0.5, label='Medium effect')
        ax.legend(fontsize=7)

    fig.suptitle(f'{task_name.upper()} - Two-Way ANOVA Effect Sizes', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'block2_anova_summary_{task_name}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def run_block2(task_name, layer_pairs=None, n_bootstrap=100, device='cuda',
               max_length=128, batch_size=16):
    """Run full Block 2 analysis for a single task."""
    print(f"\n{'='*60}")
    print(f"  Block 2 - ANOVA Interaction Effect: {task_name.upper()}")
    print(f"{'='*60}")

    if layer_pairs is None:
        layer_pairs = LAYER_PAIRS

    model, handler, layers_attr, dataloader, labels, task_cfg = load_model_and_data(
        task_name, device, max_length, batch_size
    )

    all_pair_results = []
    for layer_i, layer_j in layer_pairs:
        print(f"\n  --- Layer pair: GELU(L{layer_i}) x Softmax(L{layer_j}) ---")
        pair_result = run_anova_for_pair(
            layer_i, layer_j, model, handler, layers_attr, dataloader,
            labels, task_name, device, n_bootstrap
        )
        all_pair_results.append(pair_result)

        anova = pair_result['anova']
        print(f"    Factor A (GELU):     F={anova['factor_a_f']:.2f}, p={anova['factor_a_p']:.4f}")
        print(f"    Factor B (Softmax):  F={anova['factor_b_f']:.2f}, p={anova['factor_b_p']:.4f}")
        print(f"    A*B Interaction:     F={anova['interaction_f']:.2f}, p={anova['interaction_p']:.4f}")
        print(f"    Interaction eta^2:   {anova['interaction_eta_sq']:.4f}")

    import gc, torch as _torch
    del model, handler, dataloader
    gc.collect()
    if _torch.cuda.is_available():
        _torch.cuda.empty_cache()

    return {
        'task': task_name,
        'primary_metric': get_primary_metric(task_name),
        'layer_pairs': [(li, lj) for li, lj in layer_pairs],
        'pair_results': all_pair_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Block 2: ANOVA interaction effect analysis")
    parser.add_argument("--tasks", nargs='+', default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiment/outputs/blocks/block2",
    )
    parser.add_argument("--n_bootstrap", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tasks = args.tasks if args.tasks else ALL_TASKS

    all_task_results = {}

    for task in tasks:
        if task not in TASK_REGISTRY:
            print(f"[Warning] Unknown task '{task}', skipping")
            continue

        result = run_block2(task, LAYER_PAIRS, args.n_bootstrap, args.device,
                            args.max_length, args.batch_size)
        all_task_results[task] = result

        for pr in result['pair_results']:
            plot_interaction(pr, task, args.output_dir)
        plot_anova_summary(result['pair_results'], task, args.output_dir)

        serializable = {
            'task': result['task'],
            'primary_metric': result['primary_metric'],
            'pair_results': [{
                'layer_i': pr['layer_i'],
                'layer_j': pr['layer_j'],
                'group_data': pr['group_data'],
                'industry_baseline': pr['industry_baseline'],
                'anova': pr['anova'],
                'baseline_comparisons': pr['baseline_comparisons'],
            } for pr in result['pair_results']],
        }
        with open(os.path.join(args.output_dir, f'block2_{task}.json'), 'w') as f:
            json.dump(serializable, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  BLOCK 2 SUMMARY - Interaction Effects")
    print(f"{'='*70}")
    print(f"  {'Task':<8} {'Pair':<10} {'A*B F':>8} {'A*B p':>10} {'eta^2':>8} {'Sig':>5}")
    print(f"  {'-'*55}")
    for task, result in all_task_results.items():
        for pr in result['pair_results']:
            a = pr['anova']
            sig = "*" if a['interaction_sig'] else ""
            print(f"  {task:<8} L{pr['layer_i']}-L{pr['layer_j']:<6} "
                  f"{a['interaction_f']:>8.2f} {a['interaction_p']:>10.4f} "
                  f"{a['interaction_eta_sq']:>8.4f} {sig:>5}")

    with open(os.path.join(args.output_dir, 'block2_all_results.json'), 'w') as f:
        all_serializable = {}
        for task, result in all_task_results.items():
            all_serializable[task] = {
                'primary_metric': result['primary_metric'],
                'pair_results': [{
                    'layer_i': pr['layer_i'],
                    'layer_j': pr['layer_j'],
                    'group_data': pr['group_data'],
                    'industry_baseline': pr['industry_baseline'],
                    'anova': pr['anova'],
                    'baseline_comparisons': pr['baseline_comparisons'],
                } for pr in result['pair_results']],
            }
        json.dump(all_serializable, f, indent=2)

    print(f"\nBlock 2 completed. Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
