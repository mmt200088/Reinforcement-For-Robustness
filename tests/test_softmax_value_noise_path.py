import unittest

import numpy as np
import torch
from transformers import BertConfig, BertForSequenceClassification

from experiment.scripts.noise.softmax_v_noise_sweep import evaluate_grid_point
from function_handler import (
    BertSelfAttentionWithAproximation,
    ReversibleLayerHandler,
    SOFTMAX_VALUE_NOISE_ALLOWED_SCALING_FACTORS,
)
from layer_importance_evaluator import LayerImportanceEvaluator


class SoftmaxValueNoisePathTests(unittest.TestCase):
    def test_full_scaling_factor_map_is_scanned_by_default(self):
        self.assertEqual(
            SOFTMAX_VALUE_NOISE_ALLOWED_SCALING_FACTORS,
            tuple(range(10, 50, 2)),
        )

    def test_grid_point_evaluates_same_scaling_pair_five_times(self):
        class FakeEvaluator:
            total_layers = 3

            def __init__(self):
                self.calls = []

            def evaluate_model_with_softmax_value_noise(
                    self,
                    gelu_degrees,
                    softmax_degrees,
                    softmax_noise_scaling_factors,
                    value_noise_scaling_factors,
                    use_train=True,
                    split=None,
                    **base_noise,
            ):
                self.calls.append(
                    {
                        "gelu": np.asarray(gelu_degrees).copy(),
                        "softmax": np.asarray(softmax_degrees).copy(),
                        "softmax_noise": np.asarray(
                            softmax_noise_scaling_factors
                        ).copy(),
                        "value_noise": np.asarray(value_noise_scaling_factors).copy(),
                        "use_train": use_train,
                        "split": split,
                        "base_noise": {
                            key: np.asarray(value).copy()
                            for key, value in base_noise.items()
                        },
                    }
                )
                trial = len(self.calls)
                return float(trial), 0.5 + 0.01 * trial, 0.25 + 0.02 * trial, 7.0

        evaluator = FakeEvaluator()
        record = evaluate_grid_point(
            evaluator=evaluator,
            fixed_gelu=np.full(evaluator.total_layers, 4, dtype=int),
            fixed_softmax=np.full(evaluator.total_layers, 6, dtype=int),
            base_noise={
                "input_noise_scaling_factors": np.full(
                    evaluator.total_layers, 30, dtype=int
                )
            },
            softmax_factor=12,
            value_factor=14,
            repeat_n=5,
            split_name="validation_full",
            dataset_idx=0,
            softmax_idx=1,
            value_idx=2,
            seed=42,
        )

        self.assertEqual(len(evaluator.calls), 5)
        self.assertEqual(record["loss"]["trial_count"], 5)
        self.assertEqual(record["loss"]["raw_values"], [1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(record["loss"]["mean"], 3.0)
        self.assertAlmostEqual(record["loss"]["variance"], 2.0)
        for call in evaluator.calls:
            np.testing.assert_array_equal(call["gelu"], [4, 4, 4])
            np.testing.assert_array_equal(call["softmax"], [6, 6, 6])
            np.testing.assert_array_equal(call["softmax_noise"], [12, 12, 12])
            np.testing.assert_array_equal(call["value_noise"], [14, 14, 14])
            np.testing.assert_array_equal(
                call["base_noise"]["input_noise_scaling_factors"], [30, 30, 30]
            )
            self.assertFalse(call["use_train"])
            self.assertEqual(call["split"], "validation_full")

    def test_attention_product_noise_changes_context_output(self):
        config = BertConfig(
            hidden_size=32,
            num_attention_heads=4,
            intermediate_size=64,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )
        torch.manual_seed(123)
        attn = BertSelfAttentionWithAproximation(config, degree=6, lower_bound=-13)
        attn.eval()
        hidden = torch.randn(2, 5, config.hidden_size)

        with torch.no_grad():
            torch.manual_seed(999)
            baseline = attn(hidden)[0]
            attn._softmax_value_noise_state = {
                "softmax_scaling_factor": 10,
                "value_scaling_factor": 10,
                "distribution": "fresh",
            }
            torch.manual_seed(999)
            noisy = attn(hidden)[0]

        self.assertFalse(torch.allclose(baseline, noisy))

    def test_evaluator_apply_writes_noise_state_to_attention_modules(self):
        config = BertConfig(
            num_hidden_layers=2,
            hidden_size=32,
            num_attention_heads=4,
            intermediate_size=64,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )
        model = BertForSequenceClassification(config)

        evaluator = LayerImportanceEvaluator.__new__(LayerImportanceEvaluator)
        evaluator.total_layers = 2
        evaluator.layers_attribute = "bert.encoder.layer"
        evaluator.reversible_handler = ReversibleLayerHandler(model)

        evaluator.apply_configuration(
            np.full(2, 4, dtype=int),
            np.full(2, 6, dtype=int),
        )
        evaluator.apply_softmax_value_noise_configuration(
            np.asarray([10, 12], dtype=int),
            np.asarray([14, 16], dtype=int),
        )

        first_state = model.bert.encoder.layer[0].attention.self._softmax_value_noise_state
        second_state = model.bert.encoder.layer[1].attention.self._softmax_value_noise_state
        self.assertEqual(first_state["softmax_scaling_factor"], 10)
        self.assertEqual(first_state["value_scaling_factor"], 14)
        self.assertEqual(second_state["softmax_scaling_factor"], 12)
        self.assertEqual(second_state["value_scaling_factor"], 16)

        evaluator.clear_softmax_value_noise_configuration()
        self.assertIsNone(
            model.bert.encoder.layer[0].attention.self._softmax_value_noise_state
        )
        self.assertIsNone(
            model.bert.encoder.layer[1].attention.self._softmax_value_noise_state
        )


if __name__ == "__main__":
    unittest.main()
