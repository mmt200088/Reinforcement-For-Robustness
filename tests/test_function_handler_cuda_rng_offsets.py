import unittest

try:
    import torch

    from function_handler import (
        noise_rng_offsets_for_device,
        reseed_noise_rng_for_device,
        set_noise_rng_offsets_for_device,
    )

    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - dependency-light local lane
    torch = None  # type: ignore
    _IMPORT_ERROR = exc


@unittest.skipUnless(
    _IMPORT_ERROR is None and torch is not None and torch.cuda.is_available(),
    f"CUDA function handler unavailable: {_IMPORT_ERROR!r}",
)
class CudaNoiseRngOffsetTest(unittest.TestCase):
    def test_noise_and_truncation_offsets_round_trip(self):
        device = torch.device("cuda:0")
        reseed_noise_rng_for_device(device, 12345)

        self.assertEqual(noise_rng_offsets_for_device(device), (0, 0))

        set_noise_rng_offsets_for_device(
            device,
            noise_offset=28,
            truncation_offset=12,
        )
        self.assertEqual(noise_rng_offsets_for_device(device), (28, 12))


if __name__ == "__main__":
    unittest.main()
