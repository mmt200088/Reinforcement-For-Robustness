"""Contracts for allocation-light BLB inference noise helpers."""
from __future__ import annotations

import importlib.util
from unittest import mock
import unittest


@unittest.skipIf(importlib.util.find_spec("torch") is None, "torch unavailable")
class BLBInferenceNoiseFastPathTest(unittest.TestCase):
    def test_inference_add_reuses_sample_storage_but_training_path_does_not(self):
        import torch

        import function_handler as handler

        self.assertTrue(
            hasattr(handler, "_sample_and_add_gaussian_for_point"),
        )
        helper = handler._sample_and_add_gaussian_for_point
        point = handler.NoisePoint("fresh", 30, 16384)
        reference = torch.tensor([1.25, -2.5, 4.0], dtype=torch.float32)

        inference_sample = torch.tensor(
            [0.5, -0.25, 0.125], dtype=torch.float32,
        )
        expected = reference + inference_sample
        with (
            mock.patch.object(
                handler,
                "_sample_gaussian_for_point",
                return_value=inference_sample,
            ),
            mock.patch.object(
                handler,
                "_BLB_INFERENCE_NOISE_ADD_ENABLED",
                True,
            ),
            torch.inference_mode(),
        ):
            actual = helper(reference, point)

        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(actual.data_ptr(), inference_sample.data_ptr())
        self.assertTrue(
            torch.equal(
                reference,
                torch.tensor([1.25, -2.5, 4.0], dtype=torch.float32),
            )
        )

        training_sample = torch.tensor(
            [0.5, -0.25, 0.125], dtype=torch.float32,
        )
        with (
            mock.patch.object(
                handler,
                "_sample_gaussian_for_point",
                return_value=training_sample,
            ),
            mock.patch.object(
                handler,
                "_BLB_INFERENCE_NOISE_ADD_ENABLED",
                True,
            ),
        ):
            training_result = helper(reference, point)

        self.assertTrue(torch.equal(training_result, expected))
        self.assertNotEqual(
            training_result.data_ptr(),
            training_sample.data_ptr(),
        )

    def test_cached_noise_std_preserves_value_and_skips_repeated_lookup(self):
        import function_handler as handler

        self.assertTrue(hasattr(handler, "_noise_std_for_values"))
        helper = handler._noise_std_for_values
        helper.cache_clear()
        with mock.patch.object(
            handler,
            "get_input_noise_variance_by_N",
            return_value=0.25,
        ) as lookup:
            first = helper("fresh", 30, 16384)
            second = helper("fresh", 30, 16384)

        self.assertEqual(first, 0.5)
        self.assertEqual(second, 0.5)
        lookup.assert_called_once_with(
            scaling_factor=30,
            distribution="fresh",
            N=16384,
        )

    def test_cuda_output_and_next_noise_draw_match_legacy_bitwise(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")

        import function_handler as handler

        self.assertTrue(
            hasattr(handler, "_sample_and_add_gaussian_for_point"),
        )
        point = handler.NoisePoint("rescale", 20, 16384)
        reference = torch.linspace(
            -2.0,
            2.0,
            steps=64 * 17,
            device="cuda",
            dtype=torch.float32,
        ).reshape(64, 17)

        for seed in (0, 1, 987654, 2147483647):
            handler.reseed_noise_rng_for_device(reference.device, seed)
            with torch.inference_mode():
                expected = (
                    reference
                    + handler._sample_gaussian_for_point(reference, point)
                )
                expected_next = handler._sample_independent_gaussian(
                    torch.empty(257, device=reference.device),
                    0.25,
                )

            handler.reseed_noise_rng_for_device(reference.device, seed)
            with (
                mock.patch.object(
                    handler,
                    "_BLB_INFERENCE_NOISE_ADD_ENABLED",
                    True,
                ),
                torch.inference_mode(),
            ):
                actual = handler._sample_and_add_gaussian_for_point(
                    reference,
                    point,
                )
                actual_next = handler._sample_independent_gaussian(
                    torch.empty(257, device=reference.device),
                    0.25,
                )

            self.assertTrue(torch.equal(actual, expected), seed)
            self.assertTrue(
                torch.equal(actual_next, expected_next),
                ("RNG state mismatch", seed),
            )


if __name__ == "__main__":
    unittest.main()
