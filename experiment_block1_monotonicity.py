#!/usr/bin/env python
"""
Block 1: Non-Monotonicity Statistical Test

Establishes partial order on approximation configs, constructs directed degradation
pairs (C_A > C_B in precision), and tests whether M(C_B) > M(C_A) (counter-monotonic).
Uses Bootstrap resampling + paired t-test / Wilcoxon signed-rank test.

Usage:
    python experiment_block1_monotonicity.py --device cuda --output_dir results/block1
    python experiment_block1_monotonicity.py --tasks sst2 mrpc --n_bootstrap 200
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

from experiment_core import (
    TASK_REGISTRY, ALL_TASKS, NUM_LAYERS,
    GELU_FULL, GELU_LOW, SOFTMAX_FULL, SOFTMAX_LOW, BASELINE_CONFIG,
    load_model_and_data, get_logits_for_config, compute_metrics,
    bootstrap_metric, get_primary_metric,
)


def generate_degradation_pairs(n_pairs=30, seed=42):
    """
    Generate directed degradation pairs (C_A, C_B) where C_A > C_B.
    C_A: single-layer degradation; C_B: multi-layer degradation (superset of C_A).
    """
    rng = np.random.RandomState(seed)
    pairs = []
    layers = list(range(NUM_LAYERS))

    for _ in range(n_pairs * 3):
        n_degrade = rng.choice([2, 3])
        selected = sorted(rng.choice(layers, size=n_degrade, replace=False).tolist())

        anchor_idx = rng.choice(range(len(selected)))
        anchor_layer = selected[anchor_idx]
        extra_layers = [l for l in selected if l != anchor_layer]

        degrade_gelu = rng.random() > 0.5
        degrade_softmax = rng.random() > 0.5
        if not degrade_gelu and not degrade_softmax:
            degrade_gelu = True

        gelu_a = [GELU_FULL] * NUM_LAYERS
        softmax_a = [SOFTMAX_FULL] * NUM_LAYERS
        gelu_b = [GELU_FULL] * NUM_LAYERS
        softmax_b = [SOFTMAX_FULL] * NUM_LAYERS

        if degrade_gelu:
            gelu_a[anchor_layer] = GELU_LOW
            gelu_b[anchor_layer] = GELU_LOW
            for l in extra_layers:
                gelu_b[l] = GELU_LOW

        if degrade_softmax:
            softmax_a[anchor_layer] = SOFTMAX_LOW
            softmax_b[anchor_layer] = SOFTMAX_LOW
            for l in extra_layers:
                softmax_b[l] = SOFTMAX_LOW

        pair_key = (tuple(gelu_a), tuple(softmax_a), tuple(gelu_b), tuple(softmax_b))
        if pair_key not in {(tuple(p['gelu_a']), tuple(p['softmax_a']),
                              tuple(p['gelu_b']), tuple(p['softmax_b'])) for p in pairs}:
            description = f"anchor=L{anchor_layer}"
            if degrade_gelu:
                description += f" GELU->{GELU_LOW}"
            if degrade_softmax:
                description += f" SM->{SOFTMAX_LOW}"
            description += f" | extra={extra_layers}"

            pairs.append({
                'gelu_a': gelu_a, 'softmax_a': softmax_a,
                'gelu_b': gelu_b, 'softmax_b': softmax_b,
                'description': description,
                'anchor': anchor_layer,
                'extra_layers': extra_layers,
            })

        if len(pairs) >= n_pairs:
            break

    return pairs[:n_pairs]


def screen_candidate_pairs(pairs, model, handler, layers_attr, dataloader, labels,
                            task_name, device):
    """
    Quick single-pass screen: find pairs where M(C_B) > M(C_A) on full validation set.
    """
    primary = get_primary_metric(task_name)
    candidates = []

    for i, pair in enumerate(pairs):
        logits_a, _ = get_logits_for_config(
            model, handler, layers_attr, dataloader,
            pair['gelu_a'], pair['softmax_a'], device
        )
        m_a = compute_metrics(logits_a, labels, task_name)

        logits_b, _ = get_logits_for_config(
            model, handler, layers_attr, dataloader,
            pair['gelu_b'], pair['softmax_b'], device
        )
        m_b = compute_metrics(logits_b, labels, task_name)

        diff = m_b[primary] - m_a[primary]
        pair['metric_a'] = m_a[primary]
        pair['metric_b'] = m_b[primary]
        pair['diff'] = diff
        pair['logits_a'] = logits_a
        pair['logits_b'] = logits_b

        status = "ANOMALOUS" if diff > 0 else "expected"
        print(f"  Pair {i}: M(C_A)={m_a[primary]:.4f} M(C_B)={m_b[primary]:.4f} "
              f"diff={diff:+.4f} [{status}] -- {pair['description']}")

        if diff > 0:
            candidates.append(pair)

    return candidates


def run_bootstrap_test(pair, labels, task_name, n_bootstrap=100, alpha=0.05):
    """
    Bootstrap resampling + statistical test for one candidate pair.
    H0: M(C_B) <= M(C_A) (monotonicity holds)
    """
    primary = get_primary_metric(task_name)
    n = len(labels)

    boot_a, boot_b, boot_diff = [], [], []
    rng = np.random.RandomState(42)

    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        la = pair['logits_a'][indices]
        lb = pair['logits_b'][indices]
        bl = labels[indices]
        ma = compute_metrics(la, bl, task_name)[primary]
        mb = compute_metrics(lb, bl, task_name)[primary]
        boot_a.append(ma)
        boot_b.append(mb)
        boot_diff.append(mb - ma)

    boot_a = np.array(boot_a)
    boot_b = np.array(boot_b)
    boot_diff = np.array(boot_diff)

    _, shapiro_p = stats.shapiro(boot_diff) if len(boot_diff) >= 8 else (0, 0)
    is_normal = shapiro_p > 0.05

    if is_normal:
        t_stat, p_two = stats.ttest_rel(boot_b, boot_a)
        p_value = p_two / 2 if t_stat > 0 else 1 - p_two / 2
        test_name = "paired t-test"
    else:
        stat, p_two = stats.wilcoxon(boot_diff, alternative='greater')
        p_value = p_two
        test_name = "Wilcoxon signed-rank"

    mean_diff = np.mean(boot_diff)
    std_diff = np.std(boot_diff)
    cohens_d = mean_diff / std_diff if std_diff > 1e-10 else 0.0

    reject = p_value < alpha
    return {
        'test_name': test_name,
        'is_normal': bool(is_normal),
        'shapiro_p': float(shapiro_p),
        'p_value': float(p_value),
        'reject_h0': bool(reject),
        'mean_diff': float(mean_diff),
        'std_diff': float(std_diff),
        'cohens_d': float(cohens_d),
        'boot_a_mean': float(np.mean(boot_a)),
        'boot_b_mean': float(np.mean(boot_b)),
        'boot_a_std': float(np.std(boot_a)),
        'boot_b_std': float(np.std(boot_b)),
        'boot_diff': boot_diff.tolist(),
        'n_bootstrap': n_bootstrap,
    }


def plot_bootstrap_results(task_name, tested_pairs, output_dir):
    """Plot bootstrap distributions for tested pairs."""
    primary = get_primary_metric(task_name)
    n = len(tested_pairs)
    if n == 0:
        return

    fig, axes = plt.subplots(1, min(n, 5), figsize=(5 * min(n, 5), 5))
    if n == 1:
        axes = [axes]

    for i, pair in enumerate(tested_pairs[:5]):
        ax = axes[i]
        test = pair['test_result']
        diffs = np.array(test['boot_diff'])

        ax.hist(diffs, bins=30, color='#3498db', alpha=0.7, edgecolor='white')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero (monotonicity)')
        ax.axvline(x=test['mean_diff'], color='green', linestyle='-', linewidth=2,
                   label=f'Mean={test["mean_diff"]:.4f}')

        sig_marker = "***" if test['p_value'] < 0.001 else "**" if test['p_value'] < 0.01 else "*" if test['p_value'] < 0.05 else "n.s."
        ax.set_title(f"Pair {i+1}: p={test['p_value']:.4f} {sig_marker}\n{pair['description'][:40]}",
                     fontsize=9)
        ax.set_xlabel(f'M(C_B) - M(C_A) [{primary}]', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.legend(fontsize=8)

    fig.suptitle(f'{task_name.upper()} - Bootstrap Distribution of M(C_B)-M(C_A)', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'block1_bootstrap_{task_name}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def run_block1(task_name, n_pairs=30, n_bootstrap=100, device='cuda',
               max_length=128, batch_size=16):
    """Run full Block 1 analysis for a single task."""
    print(f"\n{'='*60}")
    print(f"  Block 1 - Non-Monotonicity Test: {task_name.upper()}")
    print(f"{'='*60}")

    model, handler, layers_attr, dataloader, labels, task_cfg = load_model_and_data(
        task_name, device, max_length, batch_size
    )
    primary = get_primary_metric(task_name)

    print(f"\n  Step A: Generating {n_pairs} degradation pairs...")
    pairs = generate_degradation_pairs(n_pairs, seed=42)

    print(f"\n  Step B: Screening for anomalous pairs (M(C_B) > M(C_A))...")
    candidates = screen_candidate_pairs(pairs, model, handler, layers_attr,
                                         dataloader, labels, task_name, device)

    print(f"\n  Found {len(candidates)} anomalous candidate pairs out of {len(pairs)}")

    tested_pairs = []
    if candidates:
        top_candidates = sorted(candidates, key=lambda x: x['diff'], reverse=True)[:5]

        print(f"\n  Step C: Bootstrap testing top {len(top_candidates)} candidates...")
        for i, pair in enumerate(top_candidates):
            print(f"\n    Testing pair {i+1}: {pair['description']}")
            test_result = run_bootstrap_test(pair, labels, task_name, n_bootstrap)
            pair['test_result'] = test_result
            tested_pairs.append(pair)

            sig = "SIGNIFICANT" if test_result['reject_h0'] else "not significant"
            print(f"      {test_result['test_name']}: p={test_result['p_value']:.6f} ({sig})")
            print(f"      Mean diff: {test_result['mean_diff']:.4f} +/- {test_result['std_diff']:.4f}")
            print(f"      Cohen's d: {test_result['cohens_d']:.4f}")

    import gc, torch as _torch
    del model, handler, dataloader
    gc.collect()
    if _torch.cuda.is_available():
        _torch.cuda.empty_cache()

    result = {
        'task': task_name,
        'primary_metric': primary,
        'n_pairs_generated': len(pairs),
        'n_anomalous': len(candidates),
        'all_pairs_summary': [{
            'description': p['description'],
            'metric_a': float(p['metric_a']),
            'metric_b': float(p['metric_b']),
            'diff': float(p['diff']),
            'gelu_a': p['gelu_a'],
            'softmax_a': p['softmax_a'],
            'gelu_b': p['gelu_b'],
            'softmax_b': p['softmax_b'],
        } for p in pairs if 'metric_a' in p],
        'tested_pairs': [{
            'description': p['description'],
            'gelu_a': p['gelu_a'],
            'softmax_a': p['softmax_a'],
            'gelu_b': p['gelu_b'],
            'softmax_b': p['softmax_b'],
            'metric_a': float(p['metric_a']),
            'metric_b': float(p['metric_b']),
            'diff': float(p['diff']),
            'test_result': {k: v for k, v in p['test_result'].items() if k != 'boot_diff'},
        } for p in tested_pairs],
    }

    return result, tested_pairs


def main():
    parser = argparse.ArgumentParser(description="Block 1: Non-monotonicity statistical test")
    parser.add_argument("--tasks", nargs='+', default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="results/block1")
    parser.add_argument("--n_pairs", type=int, default=30)
    parser.add_argument("--n_bootstrap", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tasks = args.tasks if args.tasks else ALL_TASKS

    summary_table = []
    all_results = {}

    for task in tasks:
        if task not in TASK_REGISTRY:
            print(f"[Warning] Unknown task '{task}', skipping")
            continue

        result, tested_pairs = run_block1(
            task, args.n_pairs, args.n_bootstrap, args.device,
            args.max_length, args.batch_size
        )
        all_results[task] = result

        plot_bootstrap_results(task, tested_pairs, args.output_dir)

        with open(os.path.join(args.output_dir, f'block1_{task}.json'), 'w') as f:
            json.dump(result, f, indent=2)

        summary_table.append({
            'task': task,
            'anomalous_pairs': result['n_anomalous'],
            'total_pairs': result['n_pairs_generated'],
            'significant_pairs': sum(1 for p in result['tested_pairs'] if p['test_result']['reject_h0']),
            'tested_pairs': len(result['tested_pairs']),
        })

    print(f"\n{'='*70}")
    print(f"  BLOCK 1 SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Task':<8} {'Anomalous':>10} {'Tested':>8} {'Significant':>12} {'Total':>8}")
    print(f"  {'-'*50}")
    for row in summary_table:
        print(f"  {row['task']:<8} {row['anomalous_pairs']:>10} {row['tested_pairs']:>8} "
              f"{row['significant_pairs']:>12} {row['total_pairs']:>8}")

    with open(os.path.join(args.output_dir, 'block1_summary.json'), 'w') as f:
        json.dump({'summary': summary_table, 'details': all_results}, f, indent=2)

    print(f"\nBlock 1 completed. Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
