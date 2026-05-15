import unittest
from pathlib import Path
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.bert_mrpc_layer_noise_experiment import (
    accuracy_metric,
    aggregate_metric_trials,
    build_arg_parser,
    build_sigma_grid,
    inject_noise_into_layer_output,
    layer_position_accuracy_bars,
    log_tick_positions,
    noise_magnitude_accuracy_curve,
    remove_stale_figure_outputs,
    select_mild_drop_sigma,
    stretched_log_positions,
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
        grid = build_sigma_grid()
        self.assertEqual(grid[0], 1e-10)
        self.assertEqual(grid[-1], 10.0)
        self.assertEqual(grid, sorted(set(grid)))
        self.assertIn(1e-4, grid)
        self.assertIn(2e-4, grid)
        self.assertIn(9e-4, grid)
        self.assertIn(1.0, grid)
        self.assertIn(10.0, grid)

    def test_select_mild_drop_sigma_targets_small_acc_drop(self):
        rows = [
            {"sigma": 1e-5, "acc_mean": 0.880},
            {"sigma": 1e-4, "acc_mean": 0.879},
            {"sigma": 1e-3, "acc_mean": 0.868},
            {"sigma": 1e-2, "acc_mean": 0.790},
        ]
        chosen = select_mild_drop_sigma(
            rows,
            baseline_acc=0.880,
            target_drop=0.02,
        )
        self.assertEqual(chosen, 1e-3)

    def test_aggregate_metric_trials_reports_accuracy_mean_only(self):
        summary = aggregate_metric_trials([
            {"acc": 0.80},
            {"acc": 0.84},
            {"acc": 0.82},
        ])
        self.assertAlmostEqual(summary["acc_mean"], 0.82)
        self.assertEqual(sorted(summary), ["acc_mean"])

    def test_accuracy_metric_reports_accuracy_only(self):
        labels = [0, 0, 0, 0, 1]
        preds = [0, 0, 1, 1, 1]
        metrics = accuracy_metric(labels, preds)
        self.assertAlmostEqual(metrics["acc"], 0.6)
        self.assertEqual(sorted(metrics), ["acc"])

    def test_default_cli_uses_accuracy_only_experiment_settings(self):
        args = build_arg_parser().parse_args([])
        self.assertEqual(args.repeats, 50)
        self.assertEqual(args.layer_sigma, "0.6")
        self.assertIsNone(args.sigmas)

    def test_stretched_log_positions_center_degradation_pivot(self):
        sigmas = [1e-10, 1e-6, 1e-3, 1e-1, 1.0, 10.0]
        positions = stretched_log_positions(sigmas)
        self.assertAlmostEqual(positions[0], 0.0)
        self.assertAlmostEqual(positions[3], 0.5)
        self.assertAlmostEqual(positions[-1], 1.0)
        self.assertGreater(positions[4] - positions[3], positions[3] - positions[2])

    def test_log_tick_positions_keep_sparse_labels(self):
        _, labels = log_tick_positions([1e-10, 1e-6, 1e-3, 1e-1, 1.0, 10.0])
        self.assertEqual(labels, [
            "$10^{-10}$",
            "$10^{-6}$",
            "$10^{-3}$",
            "$10^{-1}$",
            "$10^{0}$",
            "$10^{1}$",
        ])

    def test_noise_magnitude_accuracy_curve_starts_with_zero_clean_baseline(self):
        x_positions, values, tick_positions, tick_labels = noise_magnitude_accuracy_curve({
            "baseline": {"acc": 0.90},
            "experiment1": [
                {"sigma": 1e-10, "acc_mean": 0.89},
                {"sigma": 1e-1, "acc_mean": 0.80},
                {"sigma": 10.0, "acc_mean": 0.70},
            ],
        })
        self.assertEqual(x_positions[0], 0.0)
        self.assertGreater(x_positions[1], 0.0)
        self.assertEqual(values[0], 90.0)
        self.assertEqual(tick_positions[0], 0.0)
        self.assertEqual(tick_labels[0], "0")
        self.assertIn("$10^{-1}$", tick_labels)

    def test_layer_position_accuracy_bars_keep_layers_without_clean_group(self):
        labels, values = layer_position_accuracy_bars({
            "baseline": {"acc": 0.90},
            "experiment2": {
                "rows": [
                    {"layer": 0, "acc_mean": 0.80},
                    {"layer": 1, "acc_mean": 0.70},
                ],
            },
        })
        self.assertEqual(labels, ["0", "1"])
        self.assertEqual(values, [80.0, 70.0])

    def test_remove_stale_figure_outputs_deletes_old_f1_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            stale = output_dir / "noise_magnitude_f1.png"
            keep = output_dir / "noise_magnitude_accuracy.png"
            stale.write_text("old", encoding="utf-8")
            keep.write_text("new", encoding="utf-8")
            remove_stale_figure_outputs(output_dir)
            self.assertFalse(stale.exists())
            self.assertTrue(keep.exists())

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
