"""Bitwise and RNG contracts for the optional Block3 CUDA hot path."""
from __future__ import annotations

import importlib.util
from unittest import mock
import unittest


@unittest.skipIf(importlib.util.find_spec("torch") is None, "torch unavailable")
class Block3CudaFusionTest(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "torch unavailable",
    )
    def test_degree4_cuda_fast_path_matches_eager_and_rng_state_bitwise(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")

        import function_handler as handler

        self.assertTrue(hasattr(handler, "_try_block3_fused_cuda"))
        cfg = handler.make_block3_default_config(
            degree=4,
            N=16384,
            x_fresh_sf=28,
            inv_2n_sf=15,
            square_rescale_sfs=(31, 31, 31, 31),
            output_truncation_k=9,
        )
        forward = handler._make_block3_approximation_exponential(cfg)
        real_fast_path = handler._try_block3_fused_cuda

        for shape in ((1, 2, 5, 7), (3, 4, 9, 11)):
            x = torch.linspace(
                -3.0,
                0.0,
                steps=int(torch.tensor(shape).prod().item()),
                device="cuda",
                dtype=torch.float32,
            ).reshape(shape)
            for seed in (0, 1, 987654, 2147483647):
                handler.reseed_noise_rng_for_device(x.device, seed)
                with mock.patch.object(
                    handler,
                    "_try_block3_fused_cuda",
                    return_value=None,
                ):
                    expected = forward(x)
                    expected_next = handler._sample_independent_gaussian(
                        torch.empty(257, device=x.device), 0.25,
                    )

                used_fast_path = []

                def tracked_fast_path(*args, **kwargs):
                    result = real_fast_path(*args, **kwargs)
                    used_fast_path.append(result is not None)
                    return result

                handler.reseed_noise_rng_for_device(x.device, seed)
                with mock.patch.object(
                    handler,
                    "_try_block3_fused_cuda",
                    side_effect=tracked_fast_path,
                ):
                    actual = forward(x)
                    actual_next = handler._sample_independent_gaussian(
                        torch.empty(257, device=x.device), 0.25,
                    )

                self.assertEqual(used_fast_path, [True])
                self.assertTrue(torch.equal(actual, expected), (shape, seed))
                self.assertTrue(
                    torch.equal(actual_next, expected_next),
                    ("RNG state mismatch", shape, seed),
                )


if __name__ == "__main__":
    unittest.main()
