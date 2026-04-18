#!/usr/bin/env python
"""
Supplementary Test 2: Stepwise Degradation Curve

Starting from full precision (GELU=4, Softmax=6 all layers), degrade one
position at a time in random order until reaching the PPO-optimal config.
Repeat 5 times with different random orderings per dataset.

Usage:
    python -m experiment.scripts.degradation.stepwise_degradation --device cuda
    python -m experiment.scripts.degradation.stepwise_degradation --tasks sst2 mrpc --n_trials 5
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

from experiment.core.experiment_core import (
    TASK_REGISTRY, ALL_TASKS, NUM_LAYERS,
    GELU_FULL, SOFTMAX_FULL, BASELINE_CONFIG,
    load_model_and_data, evaluate_config, get_primary_metric, load_ppo_configs,
)


def compute_degradation_steps(baseline_gelu, baseline_softmax, target_gelu, target_softmax):
    """
    Compute all positions that differ between baseline and target.
    Returns list of (module_type, layer_idx, target_value) tuples.
    module_type is 'gelu' or 'softmax'.
    """
    steps = []
    for i in range(NUM_LAYERS):
        if baseline_gelu[i] != target_gelu[i]:
            steps.append(('gelu', i, target_gelu[i]))
        if baseline_softmax[i] != target_softmax[i]:
            steps.append(('softmax', i, target_softmax[i]))
    return steps


def run_stepwise_degradation(task_name, ppo_config, n_trials=5, device='cuda',
                              max_length=128, batch_size=16):
    """
    Run stepwise degradation from full precision to PPO-optimal config.
    Returns dict with per-trial degradation curves.
    """
    print(f"\n{'='*60}")
    print(f"  Stepwise Degradation: {task_name.upper()}")
    print(f"{'='*60}")

    target_gelu = ppo_config['gelu']
    target_softmax = ppo_config['softmax']

    print(f"  Target GELU:    {target_gelu}")
    print(f"  Target Softmax: {target_softmax}")

    steps = compute_degradation_steps(
        BASELINE_CONFIG['gelu'], BASELINE_CONFIG['softmax'],
        target_gelu, target_softmax
    )
    n_steps = len(steps)
    print(f"  Total degradation steps: {n_steps}")

    if n_steps == 0:
        print("  No difference between baseline and target. Skipping.")
        return None

    model, handler, layers_attr, dataloader, labels, task_cfg = load_model_and_data(
        task_name, device, max_length, batch_size
    )
    primary = get_primary_metric(task_name)
    all_metric_keys = task_cfg['all_metrics']

    all_trials = []

    for trial in range(n_trials):
        print(f"\n  --- Trial {trial+1}/{n_trials} ---")
        rng = np.random.RandomState(seed=42 + trial)
        order = rng.permutation(n_steps).tolist()

        current_gelu = list(BASELINE_CONFIG['gelu'])
        current_softmax = list(BASELINE_CONFIG['softmax'])

        baseline_m = evaluate_config(
            model, handler, layers_attr, dataloader, labels, task_name,
            current_gelu, current_softmax, device
        )

        curve = [{
            'step': 0,
            'metrics': baseline_m,
            'action': 'baseline',
            'gelu': list(current_gelu),
            'softmax': list(current_softmax),
        }]
        print(f"    Step 0 (baseline): {primary}={baseline_m[primary]:.4f}")

        for step_num, step_idx in enumerate(order):
            module_type, layer_idx, target_val = steps[step_idx]

            if module_type == 'gelu':
                current_gelu[layer_idx] = target_val
            else:
                current_softmax[layer_idx] = target_val

            m = evaluate_config(
                model, handler, layers_attr, dataloader, labels, task_name,
                current_gelu, current_softmax, device
            )

            action_str = f"{module_type}_L{layer_idx}->{target_val}"
            curve.append({
                'step': step_num + 1,
                'metrics': m,
                'action': action_str,
                'gelu': list(current_gelu),
                'softmax': list(current_softmax),
            })

            if (step_num + 1) % max(1, n_steps // 10) == 0 or step_num == n_steps - 1:
                print(f"    Step {step_num+1}/{n_steps} [{action_str}]: {primary}={m[primary]:.4f}")

        all_trials.append(curve)

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
        'n_steps': n_steps,
        'target_gelu': target_gelu,
        'target_softmax': target_softmax,
        'trials': all_trials,
    }


def plot_stepwise_curve(result, output_dir):
    """Plot degradation curves for one task."""
    task = result['task']
    primary = result['primary_metric']
    n_steps = result['n_steps']
    trials = result['trials']
    metric_label = result['metric_names'][0]

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(trials)))

    all_curves = []
    for trial_idx, curve in enumerate(trials):
        steps = [pt['step'] for pt in curve]
        vals = [pt['metrics'][primary] for pt in curve]
        all_curves.append(vals)
        ax.plot(steps, vals, '-', color=colors[trial_idx], alpha=0.4, linewidth=1,
                label=f'Trial {trial_idx+1}')

    mean_curve = np.mean(all_curves, axis=0)
    std_curve = np.std(all_curves, axis=0)
    step_range = list(range(n_steps + 1))
    ax.plot(step_range, mean_curve, 'k-', linewidth=2.5, label='Mean')
    ax.fill_between(step_range, mean_curve - std_curve, mean_curve + std_curve,
                     color='gray', alpha=0.2, label='Mean +/- Std')

    ax.axhline(y=all_curves[0][0], color='#2ecc71', linestyle='--', linewidth=1.5, alpha=0.7,
               label=f'Full Precision ({all_curves[0][0]:.4f})')
    ax.axhline(y=mean_curve[-1], color='#e74c3c', linestyle=':', linewidth=1.5, alpha=0.7,
               label=f'PPO Optimal ({mean_curve[-1]:.4f})')

    ax.set_xlabel('Degradation Step', fontsize=12)
    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_title(f'{task.upper()} - Stepwise Degradation to PPO Optimal ({metric_label})', fontsize=14)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, f'stepwise_degradation_{task}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    all_metrics = result['all_metrics']
    if len(all_metrics) > 1:
        for mk, mn in zip(all_metrics, result['metric_names']):
            if mk == primary:
                continue
            fig2, ax2 = plt.subplots(figsize=(12, 5))
            for trial_idx, curve in enumerate(trials):
                steps = [pt['step'] for pt in curve]
                vals = [pt['metrics'][mk] for pt in curve]
                ax2.plot(steps, vals, '-', color=colors[trial_idx], alpha=0.4, linewidth=1)
            mc = np.mean([[pt['metrics'][mk] for pt in c] for c in trials], axis=0)
            ax2.plot(step_range, mc, 'k-', linewidth=2.5, label='Mean')
            ax2.set_xlabel('Degradation Step', fontsize=12)
            ax2.set_ylabel(mn, fontsize=12)
            ax2.set_title(f'{task.upper()} - Stepwise Degradation ({mn})', fontsize=14)
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)
            fig2.tight_layout()
            fig2.savefig(os.path.join(output_dir, f'stepwise_degradation_{task}_{mk}.png'), dpi=150)
            plt.close(fig2)


def main():
    parser = argparse.ArgumentParser(description="Stepwise degradation from full precision to PPO optimal")
    parser.add_argument("--tasks", nargs='+', default=None, help="Tasks to run (default: all)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiment/outputs/degradation/stepwise",
    )
    parser.add_argument("--ppo_config", type=str, default="glue_configs_best_ppo.json")
    parser.add_argument("--n_trials", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ppo_configs = load_ppo_configs(args.ppo_config)
    tasks = args.tasks if args.tasks else ALL_TASKS

    all_results = []
    for task in tasks:
        if task not in TASK_REGISTRY:
            print(f"[Warning] Unknown task '{task}', skipping")
            continue
        if task not in ppo_configs:
            print(f"[Warning] No PPO config for task '{task}', skipping")
            continue

        result = run_stepwise_degradation(
            task, ppo_configs[task], args.n_trials, args.device,
            args.max_length, args.batch_size
        )
        if result is None:
            continue

        all_results.append(result)
        plot_stepwise_curve(result, args.output_dir)

        serializable = {
            'task': result['task'],
            'primary_metric': result['primary_metric'],
            'n_steps': result['n_steps'],
            'target_gelu': result['target_gelu'],
            'target_softmax': result['target_softmax'],
            'trials': result['trials'],
        }
        with open(os.path.join(args.output_dir, f'stepwise_{task}.json'), 'w') as f:
            json.dump(serializable, f, indent=2)

    print(f"\nAll stepwise degradation tests completed. Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
