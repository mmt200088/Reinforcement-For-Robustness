import builtins
import inspect
import unittest
from unittest import mock

import numpy as np

from final_evaluation_module import NOISE_SCALING_FACTOR_KEYS, UnifiedFinalEvaluationModule


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
    def test_final_evaluator_uses_resolved_validation_split_for_every_forward(self):
        run_source = inspect.getsource(UnifiedFinalEvaluationModule.run)

        self.assertIn("split=self.final_eval_split", run_source)
        self.assertNotIn('split="validation_full"', run_source)

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

    def test_invalid_value_checks_scan_arrays_without_tolist_materialization(self):
        for method in (
            UnifiedFinalEvaluationModule._normalize_config_array,
            UnifiedFinalEvaluationModule._normalize_noise_array,
        ):
            source = inspect.getsource(method)
            self.assertIn("_unsupported_int_values(", source)
            self.assertNotIn("arr.tolist()", source)

    def test_full_signature_avoids_tolist_materialization(self):
        source = inspect.getsource(UnifiedFinalEvaluationModule._full_signature)
        self.assertIn("_int_tuple", source)
        self.assertNotIn(".tolist()", source)

        run_source = inspect.getsource(UnifiedFinalEvaluationModule.run)
        noise_region = run_source.split("def _noise_eval(", 1)[1].split(
            "if sig in eval_cache", 1
        )[0]
        self.assertIn("self._full_signature(", noise_region)
        self.assertNotIn(".tolist()", noise_region)

        noise_cfg = {
            key: np.array([idx, idx + 1], dtype=np.int64)
            for idx, key in enumerate(NOISE_SCALING_FACTOR_KEYS)
        }
        sig = UnifiedFinalEvaluationModule._full_signature(
            np.array([1, 2], dtype=np.int64),
            np.array([6, 6], dtype=np.int64),
            noise_cfg,
        )

        self.assertEqual(sig[0], (1, 2))
        self.assertEqual(sig[1], (6, 6))
        self.assertEqual(sig[2][0], (0, 1))
        self.assertEqual(sig[2][-1], (6, 7))

    def test_stage2_cost_matched_array_tracks_cost_incrementally(self):
        source = inspect.getsource(UnifiedFinalEvaluationModule._stage2_cost_matched_array)
        update_loop = source.split("for _ in range(500):", 1)[1]

        self.assertIn("curr_cost", source)
        self.assertNotIn("sum(", update_loop)


if __name__ == "__main__":
    unittest.main()
