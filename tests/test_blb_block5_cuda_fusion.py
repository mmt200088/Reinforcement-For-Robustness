"""Bitwise and RNG contracts for the optional Block5 CUDA hot path."""
from __future__ import annotations

import importlib.util
from unittest import mock
import unittest


@unittest.skipIf(importlib.util.find_spec("torch") is None, "torch unavailable")
class Block5CudaFusionTest(unittest.TestCase):
    def test_degree4_fuses_piece_setup_and_selection_into_accumulation(self):
        from rfr.search.runtime.cuda import block5_fused_cuda

        self.assertFalse(hasattr(block5_fused_cuda, "_initialize_piece_kernel"))
        self.assertFalse(hasattr(block5_fused_cuda, "_select_piece_kernel"))

    def test_degree4_computes_both_polynomial_pieces_in_one_kernel(self):
        from rfr.search.runtime.cuda import block5_fused_cuda

        self.assertFalse(hasattr(block5_fused_cuda, "_power_kernel"))
        self.assertFalse(hasattr(block5_fused_cuda, "_powers_kernel"))
        self.assertFalse(hasattr(block5_fused_cuda, "_accumulate_piece_kernel"))
        self.assertFalse(
            hasattr(block5_fused_cuda, "_accumulate_and_select_piece_kernel")
        )
        self.assertFalse(hasattr(block5_fused_cuda, "_polynomial_piece_kernel"))
        self.assertFalse(
            hasattr(block5_fused_cuda, "_polynomial_piece_and_select_kernel")
        )
        self.assertTrue(hasattr(block5_fused_cuda, "_piecewise_polynomial_kernel"))

    def test_noise_workspace_reuses_storage_on_the_same_cuda_stream(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")

        from rfr.search.runtime import model_handler as handler

        self.assertTrue(
            hasattr(handler, "_get_block5_fused_cuda_noise_workspace")
        )
        x = torch.empty((2, 3, 17), device="cuda", dtype=torch.float32)
        try:
            first = handler._get_block5_fused_cuda_noise_workspace(x, 21)
            second = handler._get_block5_fused_cuda_noise_workspace(x, 21)
            self.assertEqual(first.data_ptr(), second.data_ptr())
            self.assertEqual(tuple(second.shape), (21, *x.shape))
        finally:
            handler._BLOCK5_FUSED_CUDA_WORKSPACES.clear()

    def test_degree4_cuda_fast_path_matches_eager_and_rng_state_bitwise(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")

        from rfr.search.runtime import model_handler as handler

        self.assertTrue(hasattr(handler, "_try_block5_fused_cuda"))
        original_gelu = handler.PolynomialGELU(degree=4)
        real_fast_path = handler._try_block5_fused_cuda
        rescale_patterns = (
            ((31, None, 31), (31, None, 31, 31)),
            ((31, 31, 31), (31, 31, 31, 31)),
        )

        for power_sfs, coefficient_sfs in rescale_patterns:
            cfg = handler.make_block5_default_config(
                gelu_degree=4,
                N=16384,
                gelu_coeff_sf=31,
                gelu_power_rescale_sfs=power_sfs,
                gelu_coeff_mul_rescale_sfs=coefficient_sfs,
                output_truncation_k=9,
            )
            forward = handler._make_block5_gelu_forward(original_gelu, cfg)

            for shape in ((2, 3, 17), (1, 4, 31)):
                x = torch.linspace(
                    -2.5,
                    2.5,
                    steps=int(torch.tensor(shape).prod().item()),
                    device="cuda",
                    dtype=torch.float32,
                ).reshape(shape)
                for seed in (0, 987654):
                    handler.reseed_noise_rng_for_device(x.device, seed)
                    handler._BLOCK5_FUSED_CUDA_WORKSPACES.clear()
                    with mock.patch.object(
                        handler,
                        "_try_block5_fused_cuda",
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
                        "_try_block5_fused_cuda",
                        side_effect=tracked_fast_path,
                    ), mock.patch.object(
                        handler,
                        "_apply_truncation",
                        side_effect=AssertionError(
                            "Block5 fast path launched standalone truncation"
                        ),
                    ):
                        actual = forward(x)
                        actual_next = handler._sample_independent_gaussian(
                            torch.empty(257, device=x.device), 0.25,
                        )

                    self.assertEqual(
                        used_fast_path,
                        [True],
                        (power_sfs, coefficient_sfs, shape, seed),
                    )
                    self.assertEqual(
                        [workspace.numel() for workspace in
                         handler._BLOCK5_FUSED_CUDA_WORKSPACES.values()],
                        [21 * x.numel()],
                        ("workspace footprint", power_sfs, coefficient_sfs),
                    )
                    self.assertTrue(
                        torch.equal(actual, expected),
                        (power_sfs, coefficient_sfs, shape, seed),
                    )
                    self.assertTrue(
                        torch.equal(actual_next, expected_next),
                        ("RNG state mismatch", power_sfs, coefficient_sfs, shape, seed),
                    )


if __name__ == "__main__":
    unittest.main()
