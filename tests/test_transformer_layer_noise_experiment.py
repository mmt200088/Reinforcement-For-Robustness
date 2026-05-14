import unittest
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.bert_mrpc_layer_noise_experiment import (
    accuracy_and_weighted_f1,
    aggregate_metric_trials,
    build_sigma_grid,
    inject_noise_into_layer_output,
    select_mild_drop_sigma,
    temporary_layer_output_noise,
)


class FakeTensor:
    def __init__(self, value):
        self.value = float(value)

    def __add__(self, other):
        other_value = other.value if isinstance(other, FakeTensor) else other
        return FakeTensor(self.value + other_value)

    def __mul__(self, other):
        other_value = other.value if isinstance(other, FakeTensor) else other
        return FakeTensor(self.value * other_value)


class FakeTorch:
    @staticmethod
    def randn_like(_tensor):
        return FakeTensor(10.0)


class FakeHandle:
    def __init__(self, layer, hook):
        self.layer = layer
        self.hook = hook

    def remove(self):
        self.layer.hooks.remove(self.hook)


class FakeLayer:
    def __init__(self):
        self.hooks = []

    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        return FakeHandle(self, hook)

    def __call__(self, hidden):
        output = (hidden + FakeTensor(1.0), "attention")
        for hook in list(self.hooks):
            output = hook(self, (hidden,), output)
        return output[0]


class FakeEncoder:
    def __init__(self, layers):
        self.layer = layers


class FakeBert:
    def __init__(self, layers):
        self.encoder = FakeEncoder(layers)


class FakeModel:
    def __init__(self, layer_count):
        self.bert = FakeBert([FakeLayer() for _ in range(layer_count)])

    def forward_hidden(self, value):
        hidden = FakeTensor(value)
        for layer in self.bert.encoder.layer:
            hidden = layer(hidden)
        return hidden.value


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
        chosen = select_mild_drop_sigma(
            rows,
            baseline_f1=0.910,
            baseline_acc=0.880,
            target_drop=0.02,
        )
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

    def test_accuracy_and_f1_uses_weighted_average(self):
        labels = [0, 0, 0, 0, 1]
        preds = [0, 0, 1, 1, 1]
        metrics = accuracy_and_weighted_f1(labels, preds)
        self.assertAlmostEqual(metrics["acc"], 0.6)
        self.assertAlmostEqual(metrics["f1"], 19 / 30)

    def test_inject_noise_preserves_bert_tuple_structure(self):
        output = ("hidden", "attention")
        result = inject_noise_into_layer_output(output, lambda x: f"noisy:{x}")
        self.assertEqual(result, ("noisy:hidden", "attention"))

    def test_temporary_layer_output_noise_changes_only_selected_forward_path(self):
        model = FakeModel(layer_count=3)
        self.assertEqual(model.forward_hidden(0.0), 3.0)

        with temporary_layer_output_noise(
            model=model,
            layer_indices=[1],
            sigma=0.5,
            torch_module=FakeTorch,
            layers_attr="bert.encoder.layer",
        ):
            self.assertEqual(model.forward_hidden(0.0), 8.0)

        self.assertEqual(model.forward_hidden(0.0), 3.0)

    def test_temporary_layer_output_noise_can_attach_to_all_layers(self):
        model = FakeModel(layer_count=3)

        with temporary_layer_output_noise(
            model=model,
            layer_indices=[0, 1, 2],
            sigma=0.5,
            torch_module=FakeTorch,
            layers_attr="bert.encoder.layer",
        ):
            self.assertEqual(model.forward_hidden(0.0), 18.0)


if __name__ == "__main__":
    unittest.main()
