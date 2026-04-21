import unittest

import numpy as np
import torch
from transformers import BertConfig, BertForSequenceClassification

from function_handler import BertSelfAttentionWithAproximation, ReversibleLayerHandler
from layer_importance_evaluator import LayerImportanceEvaluator


class SoftmaxValueNoisePathTests(unittest.TestCase):
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
