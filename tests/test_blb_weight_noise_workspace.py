"""Bitwise and RNG contracts for reusable CUDA noisy-weight storage."""
from __future__ import annotations

import importlib.util
import unittest


@unittest.skipIf(importlib.util.find_spec("torch") is None, "torch unavailable")
class BLBWeightNoiseWorkspaceTest(unittest.TestCase):
    def test_workspace_reuses_and_grows_storage_on_one_stream(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")

        import function_handler as handler

        self.assertTrue(hasattr(handler, "_get_blb_noisy_weight_workspace"))
        reference = torch.empty((4, 7), device="cuda", dtype=torch.float32)
        try:
            first = handler._get_blb_noisy_weight_workspace(reference)
            second = handler._get_blb_noisy_weight_workspace(reference)
            self.assertEqual(first.data_ptr(), second.data_ptr())

            larger = torch.empty((9, 11), device="cuda", dtype=torch.float32)
            grown = handler._get_blb_noisy_weight_workspace(larger)
            reused = handler._get_blb_noisy_weight_workspace(reference)
            self.assertEqual(grown.data_ptr(), reused.data_ptr())
            self.assertEqual(tuple(reused.shape), tuple(reference.shape))
        finally:
            handler._BLB_NOISY_WEIGHT_WORKSPACES.clear()

    def test_noisy_weight_and_next_rng_draw_match_eager_bitwise(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")

        import function_handler as handler

        self.assertTrue(hasattr(handler, "_noisy_weight_for_point"))
        point = handler.NoisePoint("encoding", 20, 16384)
        weight = torch.linspace(
            -0.75, 0.75, steps=35, device="cuda", dtype=torch.float32,
        ).reshape(5, 7)
        activations = torch.linspace(
            -1.0, 1.0, steps=21, device="cuda", dtype=torch.float32,
        ).reshape(3, 7)

        for seed in (0, 1, 987654, 2147483647):
            handler.reseed_noise_rng_for_device(weight.device, seed)
            with torch.inference_mode():
                expected_weight = (
                    weight + handler._sample_gaussian_for_point(weight, point)
                )
                expected_output = torch.nn.functional.linear(
                    activations, expected_weight,
                )
                expected_next = handler._sample_independent_gaussian(
                    torch.empty(257, device=weight.device), 0.25,
                )

            handler.reseed_noise_rng_for_device(weight.device, seed)
            handler._BLB_NOISY_WEIGHT_WORKSPACES.clear()
            with torch.inference_mode():
                actual_weight = handler._noisy_weight_for_point(
                    weight,
                    point,
                    reference=activations,
                )
                actual_output = torch.nn.functional.linear(
                    activations, actual_weight,
                )
                actual_next = handler._sample_independent_gaussian(
                    torch.empty(257, device=weight.device), 0.25,
                )

            self.assertTrue(torch.equal(actual_weight, expected_weight), seed)
            self.assertTrue(torch.equal(actual_output, expected_output), seed)
            self.assertTrue(
                torch.equal(actual_next, expected_next),
                ("RNG state mismatch", seed),
            )


if __name__ == "__main__":
    unittest.main()
