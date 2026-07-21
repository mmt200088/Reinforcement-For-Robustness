"""Numerical and RNG contracts for Stage-2 truncation backends."""
from __future__ import annotations

from types import SimpleNamespace
import unittest

try:
    import torch
    from function_handler import (
        _apply_truncation,
        _make_block3_approximation_exponential,
        _make_block5_gelu_forward,
        _make_block2_qkt_merge_hook,
        _sample_independent_gaussian,
        NoisyBlock1LayerNorm,
        NoisyBlock4LayerNorm,
        make_block1_default_config,
        make_block2_default_config,
        make_block3_default_config,
        make_block4_default_config,
        make_block5_default_config,
        reseed_noise_rng_for_device,
    )
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - local macOS may be torch-free.
    torch = None  # type: ignore
    _apply_truncation = None  # type: ignore
    _make_block3_approximation_exponential = None  # type: ignore
    _make_block5_gelu_forward = None  # type: ignore
    _make_block2_qkt_merge_hook = None  # type: ignore
    _sample_independent_gaussian = None  # type: ignore
    NoisyBlock1LayerNorm = None  # type: ignore
    NoisyBlock4LayerNorm = None  # type: ignore
    make_block1_default_config = None  # type: ignore
    make_block2_default_config = None  # type: ignore
    make_block3_default_config = None  # type: ignore
    make_block4_default_config = None  # type: ignore
    make_block5_default_config = None  # type: ignore
    reseed_noise_rng_for_device = None  # type: ignore
    _IMPORT_ERROR = exc


@unittest.skipUnless(_IMPORT_ERROR is None, f"torch runtime unavailable: {_IMPORT_ERROR!r}")
class TruncationBackendTests(unittest.TestCase):
    def _without_gaussian_noise(self, callback):
        import function_handler as fh

        original_sampler = fh._sample_gaussian_for_point
        fh._sample_gaussian_for_point = (
            lambda reference, _point: torch.zeros_like(reference)
        )
        try:
            return callback()
        finally:
            fh._sample_gaussian_for_point = original_sampler

    def test_legacy_binary_is_bit_for_bit_unchanged_for_signed_values(self):
        x = torch.tensor([-1.2345, -0.0001, 0.0001, 1.2345], dtype=torch.float64)
        expected = torch.trunc(x * (2 ** 8)) / (2 ** 8)
        actual = _apply_truncation(x, 8, "binary")
        self.assertTrue(torch.equal(actual, expected))

    def test_rotation_repeat_count_executes_independent_noise_for_every_rotation(self):
        import function_handler as fh

        source = make_block1_default_config().gelu_out_fresh
        original_sampler = fh._sample_gaussian_for_point
        calls = []

        def unit_noise(reference, point):
            calls.append(point)
            return torch.ones_like(reference)

        fh._sample_gaussian_for_point = unit_noise
        try:
            actual = fh._apply_rotation_noise(
                torch.zeros(4, dtype=torch.float64),
                source,
                repeat_count=3,
            )
        finally:
            fh._sample_gaussian_for_point = original_sampler

        self.assertTrue(torch.equal(actual, torch.full((4,), 3.0, dtype=torch.float64)))
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(point.distribution == "rotation" for point in calls))

    def test_none_k_is_identity_for_every_backend(self):
        x = torch.tensor([-1.25, 0.75], dtype=torch.float64)
        self.assertIs(_apply_truncation(x, None, "binary"), x)
        self.assertIs(_apply_truncation(x, None, "stochastic_ring"), x)

    def test_block2_hook_executes_k_from_the_materialized_cfg(self):
        cfg = make_block2_default_config(output_truncation_k=4)
        hook = _make_block2_qkt_merge_hook(
            None,
            cfg.qkt_merge_mask_encode,
            None,
            truncation_cfg=cfg,
        )
        x = torch.tensor([-1.2345, 1.2345], dtype=torch.float64)
        actual = self._without_gaussian_noise(lambda: hook(x))
        expected = torch.trunc(x * (2 ** 4)) / (2 ** 4)
        self.assertTrue(torch.equal(actual, expected))

    def test_block3_executes_k_after_the_polynomial(self):
        cfg = make_block3_default_config(
            degree=2,
            square_rescale_sfs=(None, None),
            output_truncation_k=4,
        )
        forward = _make_block3_approximation_exponential(cfg)
        x = torch.tensor([-0.28137, 0.17391], dtype=torch.float64)

        actual = self._without_gaussian_noise(lambda: forward(x))
        polynomial = (1.0 + x / 4.0) ** 4
        expected = torch.trunc(polynomial * (2 ** 4)) / (2 ** 4)
        self.assertTrue(torch.equal(actual, expected))

    def test_block1_executes_k_on_variance_before_rsqrt(self):
        original_ln = torch.nn.LayerNorm(4, eps=1e-5, dtype=torch.float64)
        cfg = make_block1_default_config(output_truncation_k=4)
        wrapped = NoisyBlock1LayerNorm(original_ln, cfg=cfg)
        x = torch.tensor([[[1.2345, -0.7654, 0.4567, 2.3456]]], dtype=torch.float64)

        actual = self._without_gaussian_noise(lambda: wrapped(x))
        centered = x - x.mean(dim=-1, keepdim=True)
        variance = (centered * centered).mean(dim=-1, keepdim=True)
        quantized = torch.trunc(variance * (2 ** 4)) / (2 ** 4)
        expected = centered * torch.rsqrt(quantized + original_ln.eps)
        self.assertTrue(torch.equal(actual, expected.expand_as(actual)))

    def test_block4_executes_k_on_variance_before_rsqrt(self):
        original_ln = torch.nn.LayerNorm(4, eps=1e-5, dtype=torch.float64)
        cfg = make_block4_default_config(output_truncation_k=4)
        wrapped = NoisyBlock4LayerNorm(original_ln, cfg4=cfg)
        x = torch.tensor([[[1.2345, -0.7654, 0.4567, 2.3456]]], dtype=torch.float64)

        actual = self._without_gaussian_noise(lambda: wrapped(x))
        centered = x - x.mean(dim=-1, keepdim=True)
        variance = (centered * centered).mean(dim=-1, keepdim=True)
        quantized = torch.trunc(variance * (2 ** 4)) / (2 ** 4)
        expected = centered * torch.rsqrt(quantized + original_ln.eps)
        self.assertTrue(torch.equal(actual, expected.expand_as(actual)))

    def test_block5_executes_k_after_polynomial_gelu(self):
        cfg = make_block5_default_config(
            gelu_degree=1,
            output_truncation_k=4,
        )
        original_gelu = SimpleNamespace(
            degree=1,
            coeff={0: [0.0, 1.0], 1: [0.0, 1.0]},
        )
        forward = _make_block5_gelu_forward(original_gelu, cfg)
        x = torch.tensor([-1.2345, 1.2345], dtype=torch.float64)

        actual = self._without_gaussian_noise(lambda: forward(x))
        expected = torch.trunc(x * (2 ** 4)) / (2 ** 4)
        self.assertTrue(torch.equal(actual, expected))

    def test_stochastic_ring_outputs_adjacent_k_grid_points(self):
        x = torch.full((4096,), -1.2345, dtype=torch.float64)
        reseed_noise_rng_for_device(x.device, 123)
        out = _apply_truncation(
            x, 4, "stochastic_ring", ring_bits=43,
            source_fractional_bits=24,
        )
        scaled = out * (2 ** 4)
        self.assertTrue(torch.equal(scaled, torch.round(scaled)))
        lower = torch.floor(x[0] * (2 ** 4)) / (2 ** 4)
        upper = lower + 1.0 / (2 ** 4)
        self.assertTrue(bool(torch.all((out == lower) | (out == upper))))

    def test_stochastic_ring_is_empirically_unbiased_for_both_signs(self):
        x = torch.cat((
            torch.full((50000,), 1.2345, dtype=torch.float64),
            torch.full((50000,), -1.2345, dtype=torch.float64),
        ))
        reseed_noise_rng_for_device(x.device, 456)
        out = _apply_truncation(
            x, 8, "stochastic_ring", ring_bits=43,
            source_fractional_bits=24,
        )
        self.assertAlmostEqual(float(out[:50000].mean()), 1.2345, delta=8e-5)
        self.assertAlmostEqual(float(out[50000:].mean()), -1.2345, delta=8e-5)

    def test_two_complement_ring_wrap_is_applied_before_truncation(self):
        x = torch.tensor([8.5], dtype=torch.float64)
        out = _apply_truncation(
            x, 4, "stochastic_ring", ring_bits=8,
            source_fractional_bits=4,
        )
        self.assertEqual(float(out.item()), -7.5)

    def test_truncation_rng_is_reproducible_and_isolated_from_gaussian_rng(self):
        x = torch.full((4096,), 0.12345, dtype=torch.float64)
        reference = torch.zeros(64, dtype=torch.float64)

        reseed_noise_rng_for_device(x.device, 999)
        expected_gaussian = _sample_independent_gaussian(reference, 1.0)

        reseed_noise_rng_for_device(x.device, 999)
        first = _apply_truncation(
            x, 8, "stochastic_ring", ring_bits=43,
            source_fractional_bits=24,
        )
        actual_gaussian = _sample_independent_gaussian(reference, 1.0)

        reseed_noise_rng_for_device(x.device, 999)
        second = _apply_truncation(
            x, 8, "stochastic_ring", ring_bits=43,
            source_fractional_bits=24,
        )
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(expected_gaussian, actual_gaussian))

        reseed_noise_rng_for_device(x.device, 1000)
        different = _apply_truncation(
            x, 8, "stochastic_ring", ring_bits=43,
            source_fractional_bits=24,
        )
        self.assertFalse(torch.equal(first, different))

    def test_invalid_backend_and_invalid_parameters_fail_loudly(self):
        x = torch.tensor([1.0])
        with self.assertRaisesRegex(ValueError, "unsupported truncation mode"):
            _apply_truncation(x, 8, "unknown")
        with self.assertRaisesRegex(ValueError, "source_fractional_bits"):
            _apply_truncation(
                x, 25, "stochastic_ring", ring_bits=43,
                source_fractional_bits=24,
            )
        with self.assertRaisesRegex(ValueError, "smaller than ring_bits"):
            _apply_truncation(
                x, 8, "stochastic_ring", ring_bits=24,
                source_fractional_bits=24,
            )


if __name__ == "__main__":
    unittest.main()
