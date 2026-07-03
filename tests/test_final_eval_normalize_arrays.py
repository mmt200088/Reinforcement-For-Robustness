import builtins
import unittest
from unittest import mock

import numpy as np

from final_evaluation_module import UnifiedFinalEvaluationModule


class _DummyEvaluator:
    def __init__(self):
        self.messages = []

    def log(self, message):
        self.messages.append(message)


def _module():
    module = UnifiedFinalEvaluationModule.__new__(UnifiedFinalEvaluationModule)
    module.evaluator = _DummyEvaluator()
    module.input_noise_allowed = [20, 22, 24]
    module.weight_noise_allowed = [30, 32, 34]
    module.wffn1_noise_allowed = [28, 30, 32]
    return module


class FinalEvalNormalizeArrayTests(unittest.TestCase):
    def test_normalize_config_array_avoids_list_materialization_for_ndarray(self):
        module = _module()
        values = np.array([[1, 2], [4, 1]], dtype=np.int64)

        with mock.patch.object(
            builtins,
            "list",
            side_effect=AssertionError("ndarray config normalization should not call list()"),
        ):
            arr = module._normalize_config_array(
                values,
                total_layers=4,
                default_degree=4,
                allowed=[1, 2, 4],
                label="manual_gelu",
            )

        self.assertEqual(arr.tolist(), [1, 2, 4, 1])

    def test_normalize_noise_array_avoids_list_materialization_for_ndarray(self):
        module = _module()
        values = np.array([[30, 32], [34, 30]], dtype=np.int64)

        with mock.patch.object(
            builtins,
            "list",
            side_effect=AssertionError("ndarray noise normalization should not call list()"),
        ):
            arr = module._normalize_noise_array(
                values,
                total_layers=4,
                label="wq_noise_scaling_factors",
            )

        self.assertEqual(arr.tolist(), [30, 32, 34, 30])


if __name__ == "__main__":
    unittest.main()
