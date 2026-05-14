# BERT MRPC Layer Noise Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reproducible BERT-base MRPC experiments that measure how Gaussian noise at Transformer layer outputs affects accuracy and F1, then render publication-style figures.

**Architecture:** Keep the experiment independent from the BLB/RL launcher. A single script loads the HuggingFace MRPC model and dataset, injects Gaussian noise through encoder-layer forward hooks, writes JSON/CSV results, and calls plotting helpers. Pure utility functions are unit-tested without requiring torch.

**Tech Stack:** Python, PyTorch, HuggingFace Transformers/Datasets, scikit-learn, Matplotlib, unittest.

---

### Task 1: Utility Tests

**Files:**
- Create: `tests/test_transformer_layer_noise_experiment.py`
- Create later: `scripts/bert_mrpc_layer_noise_experiment.py`

- [ ] **Step 1: Write the failing tests**

```python
import unittest

from scripts.bert_mrpc_layer_noise_experiment import (
    aggregate_metric_trials,
    build_sigma_grid,
    inject_noise_into_layer_output,
    select_mild_drop_sigma,
)


class TransformerLayerNoiseExperimentTests(unittest.TestCase):
    def test_build_sigma_grid_is_sorted_unique_and_includes_endpoints(self):
        grid = build_sigma_grid(1e-10, 1e-1)
        self.assertEqual(grid[0], 1e-10)
        self.assertEqual(grid[-1], 1e-1)
        self.assertEqual(grid, sorted(set(grid)))
        self.assertIn(1e-4, grid)
        self.assertIn(2e-4, grid)
        self.assertIn(9e-4, grid)

    def test_select_mild_drop_sigma_targets_small_f1_drop(self):
        rows = [
            {"sigma": 1e-5, "f1_mean": 0.910, "acc_mean": 0.880},
            {"sigma": 1e-4, "f1_mean": 0.905, "acc_mean": 0.879},
            {"sigma": 1e-3, "f1_mean": 0.890, "acc_mean": 0.868},
            {"sigma": 1e-2, "f1_mean": 0.810, "acc_mean": 0.790},
        ]
        chosen = select_mild_drop_sigma(rows, baseline_f1=0.910, baseline_acc=0.880, target_drop=0.02)
        self.assertEqual(chosen, 1e-3)

    def test_aggregate_metric_trials_reports_mean_and_sample_std(self):
        summary = aggregate_metric_trials([
            {"acc": 0.80, "f1": 0.90},
            {"acc": 0.84, "f1": 0.86},
            {"acc": 0.82, "f1": 0.88},
        ])
        self.assertAlmostEqual(summary["acc_mean"], 0.82)
        self.assertAlmostEqual(summary["f1_mean"], 0.88)
        self.assertAlmostEqual(summary["acc_std"], 0.02)
        self.assertAlmostEqual(summary["f1_std"], 0.02)

    def test_inject_noise_preserves_bert_tuple_structure(self):
        output = ("hidden", "attention")
        result = inject_noise_into_layer_output(output, lambda x: f"noisy:{x}")
        self.assertEqual(result, ("noisy:hidden", "attention"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 tests/test_transformer_layer_noise_experiment.py`
Expected: FAIL with `ModuleNotFoundError` because `scripts.bert_mrpc_layer_noise_experiment` does not exist yet.

### Task 2: Experiment and Plot Script

**Files:**
- Create: `scripts/bert_mrpc_layer_noise_experiment.py`
- Test: `tests/test_transformer_layer_noise_experiment.py`

- [ ] **Step 1: Implement pure utilities**

Implement:
- `build_sigma_grid(start, stop)`: log endpoints plus dense `1..9 * 10^k` points from `1e-4` through `1e-1`.
- `inject_noise_into_layer_output(output, add_noise)`: preserve tuple/list output shape and only perturb the first tensor.
- `aggregate_metric_trials(trials)`: mean and sample std for `acc` and `f1`.
- `select_mild_drop_sigma(rows, baseline_f1, baseline_acc, target_drop)`: choose the sigma whose max metric drop is closest to `target_drop`.

- [ ] **Step 2: Run utility tests**

Run: `python3 tests/test_transformer_layer_noise_experiment.py`
Expected: PASS.

- [ ] **Step 3: Implement model/dataset evaluation**

Add CLI options for model, split, batch size, max samples, repeats, seed, device, output dir, sigma schedule, and optional fixed layer sigma.
Load `textattack/bert-base-uncased-MRPC` and `glue/mrpc` validation by default.
Evaluate clean baseline, all-layer noise for each sigma, and one-layer-at-a-time noise for each layer.

- [ ] **Step 4: Implement plotting**

Render:
- `noise_magnitude_sensitivity.pdf/png`: semilog line plot of ACC and F1 versus standard deviation.
- `layer_position_sensitivity.pdf/png`: grouped bar plot of ACC and F1 versus perturbed layer.
- `bert_mrpc_noise_sensitivity_combined.pdf/png`: two-panel figure for direct insertion.

Use English labels, Times New Roman family with serif fallback, colorblind-safe colors, light grid, PDF output, and 600 DPI PNG.

### Task 3: Verification, Commit, Push

**Files:**
- `scripts/bert_mrpc_layer_noise_experiment.py`
- `tests/test_transformer_layer_noise_experiment.py`
- `docs/superpowers/plans/2026-05-14-bert-mrpc-layer-noise-experiments.md`
- `reports/transformer_noise_mrpc/*` generated outputs

- [ ] **Step 1: Run unit tests**

Run: `python3 tests/test_transformer_layer_noise_experiment.py`
Expected: PASS.

- [ ] **Step 2: Run experiment**

Run: `python3 scripts/bert_mrpc_layer_noise_experiment.py --output-dir reports/transformer_noise_mrpc`
Expected: JSON, CSV, PDF, and PNG outputs are written under `reports/transformer_noise_mrpc/`.

- [ ] **Step 3: Inspect generated files**

Run: `find reports/transformer_noise_mrpc -maxdepth 1 -type f -print`
Expected: includes result JSON/CSV plus the three figure pairs.

- [ ] **Step 4: Commit only task files**

Run: `git add docs/superpowers/plans/2026-05-14-bert-mrpc-layer-noise-experiments.md tests/test_transformer_layer_noise_experiment.py scripts/bert_mrpc_layer_noise_experiment.py reports/transformer_noise_mrpc`
Run: `git commit -m "add bert mrpc layer noise experiments"`

- [ ] **Step 5: Push current branch**

Run: `git push origin jk_standard_rl`
Expected: push succeeds without staging unrelated pre-existing worktree changes.
