import unittest

import layer_importance_evaluator as evaluator_module
from rfr.search.runtime.model_handler import (
    INPUT_NOISE_ALLOWED_SCALING_FACTORS,
    WEIGHT_NOISE_ALLOWED_SCALING_FACTORS,
    WFFN1_NOISE_ALLOWED_SCALING_FACTORS,
)


class LayerImportanceRuntimeCostMapTests(unittest.TestCase):
    def test_active_noise_cost_maps_cover_every_runtime_scaling_factor(self):
        expected_input = {
            value: value * 0.025
            for value in INPUT_NOISE_ALLOWED_SCALING_FACTORS
        }
        expected_weight = {
            value: value * 0.025
            for value in WEIGHT_NOISE_ALLOWED_SCALING_FACTORS
        }
        expected_wffn1 = {
            value: value * 0.025
            for value in WFFN1_NOISE_ALLOWED_SCALING_FACTORS
        }

        self.assertEqual(evaluator_module.INPUT_NOISE_COST_MAP, expected_input)
        self.assertEqual(evaluator_module.WEIGHT_NOISE_COST_MAP, expected_weight)
        self.assertEqual(evaluator_module.WFFN1_NOISE_COST_MAP, expected_wffn1)


if __name__ == "__main__":
    unittest.main()
