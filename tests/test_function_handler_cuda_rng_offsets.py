import unittest
from types import SimpleNamespace

try:
    import torch
    import function_handler
    from blb_stage2_rl.inference_eval import (
        finalize_probe_batch_contributions,
    )
    from blb_stage2_rl.probe_runner import ProbeWorker

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
        noise_rng_offsets_for_device = (
            function_handler.noise_rng_offsets_for_device
        )
        reseed_noise_rng_for_device = (
            function_handler.reseed_noise_rng_for_device
        )
        set_noise_rng_offsets_for_device = (
            function_handler.set_noise_rng_offsets_for_device
        )
        device = torch.device("cuda:0")
        reseed_noise_rng_for_device(device, 12345)

        self.assertEqual(noise_rng_offsets_for_device(device), (0, 0))

        set_noise_rng_offsets_for_device(
            device,
            noise_offset=28,
            truncation_offset=12,
        )
        self.assertEqual(noise_rng_offsets_for_device(device), (28, 12))

    def test_probe_batch_offset_replay_matches_complete_trial(self):
        class NoisyClassifier(torch.nn.Module):
            def forward(
                    self,
                    input_ids,
                    attention_mask,
                    token_type_ids=None,
                    ):
                values = input_ids.float()
                noisy = values + function_handler._sample_independent_gaussian(
                    values,
                    0.125,
                )
                truncated = function_handler._apply_truncation(
                    noisy,
                    2,
                    "stochastic_ring",
                    ring_bits=43,
                    source_fractional_bits=4,
                )
                score = truncated.sum(dim=1)
                return SimpleNamespace(
                    logits=torch.stack((-score, score), dim=1),
                )

        device = torch.device("cuda:0")
        batches = [
            SimpleNamespace(
                input_ids=torch.tensor(
                    [[1, 2], [-1, -2]],
                    device=device,
                ),
                attention_mask=torch.ones(
                    (2, 2),
                    dtype=torch.long,
                    device=device,
                ),
                token_type_ids=None,
                labels=torch.tensor([1, 0], device=device),
            ),
            SimpleNamespace(
                input_ids=torch.tensor(
                    [[3, -1], [-3, 1]],
                    device=device,
                ),
                attention_mask=torch.ones(
                    (2, 2),
                    dtype=torch.long,
                    device=device,
                ),
                token_type_ids=None,
                labels=torch.tensor([1, 0], device=device),
            ),
        ]
        worker = ProbeWorker(
            device=device,
            model=NoisyClassifier().to(device),
            handler=None,
            bridge=None,
            probe_batches=batches,
            is_regression=False,
            metric_profile="mrpc",
        )
        trial_index = 2
        base_seed = 9182

        complete = worker.run_trial(trial_index, base_seed)
        complete_offsets = function_handler.noise_rng_offsets_for_device(
            device,
        )
        complete_next_noise = torch.empty(
            16,
            device=device,
        ).normal_(
            generator=function_handler._get_noise_generator(device),
        )
        complete_next_truncation = torch.empty(
            16,
            device=device,
        ).uniform_(
            generator=function_handler._get_truncation_generator(device),
        )

        deltas = worker.calibrate_batch_rng_offsets("F1")
        contributions = [
            worker.run_trial_batch(
                trial_index,
                base_seed,
                batch_index=batch_index,
                noise_offset=batch_index * deltas[0],
                truncation_offset=batch_index * deltas[1],
                expected_offset_deltas=deltas,
                batch_set_key="F1",
            )
            for batch_index in range(2)
        ]
        sharded = finalize_probe_batch_contributions(
            contributions,
            expected_trial_index=trial_index,
            expected_batch_count=2,
            metric_profile="mrpc",
            is_regression=False,
        )
        sharded_offsets = function_handler.noise_rng_offsets_for_device(device)
        sharded_next_noise = torch.empty(
            16,
            device=device,
        ).normal_(
            generator=function_handler._get_noise_generator(device),
        )
        sharded_next_truncation = torch.empty(
            16,
            device=device,
        ).uniform_(
            generator=function_handler._get_truncation_generator(device),
        )

        self.assertEqual(sharded, complete)
        self.assertEqual(sharded_offsets, complete_offsets)
        self.assertTrue(torch.equal(sharded_next_noise, complete_next_noise))
        self.assertTrue(torch.equal(
            sharded_next_truncation,
            complete_next_truncation,
        ))

    def test_probe_batch_rng_plan_uses_full_resident_batch_shapes(self):
        class ShapeDependentNoiseClassifier(torch.nn.Module):
            def forward(
                    self,
                    input_ids,
                    attention_mask,
                    token_type_ids=None,
                    ):
                del attention_mask, token_type_ids
                values = input_ids.float()
                noisy = values + function_handler._sample_independent_gaussian(
                    values,
                    0.125,
                )
                truncated = function_handler._apply_truncation(
                    noisy,
                    2,
                    "stochastic_ring",
                    ring_bits=43,
                    source_fractional_bits=4,
                )
                score = truncated[:, :8].sum(dim=1)
                return SimpleNamespace(
                    logits=torch.stack((-score, score), dim=1),
                )

        device = torch.device("cuda:0")

        def make_batch(batch_size):
            shape = (int(batch_size), 131072)
            return SimpleNamespace(
                input_ids=torch.ones(shape, device=device),
                attention_mask=torch.ones(
                    shape,
                    dtype=torch.long,
                    device=device,
                ),
                token_type_ids=None,
                labels=torch.ones(
                    int(batch_size),
                    dtype=torch.long,
                    device=device,
                ),
            )

        batches = [make_batch(8), make_batch(2), make_batch(8)]
        worker = ProbeWorker(
            device=device,
            model=ShapeDependentNoiseClassifier().to(device),
            handler=None,
            bridge=None,
            probe_batches=batches,
            is_regression=False,
            metric_profile="sst2",
        )

        plan = worker.calibrate_batch_rng_plan("F1")
        expected = []
        for batch in batches:
            function_handler.reseed_noise_rng_for_device(device, 0)
            start_offsets = function_handler.noise_rng_offsets_for_device(
                device,
            )
            with torch.inference_mode():
                worker.model(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                )
            end_offsets = function_handler.noise_rng_offsets_for_device(device)
            expected.append((
                end_offsets[0] - start_offsets[0],
                end_offsets[1] - start_offsets[1],
            ))

        self.assertEqual(plan, tuple(expected))
        self.assertEqual(plan[0], plan[2])
        self.assertNotEqual(plan[0], plan[1])


if __name__ == "__main__":
    unittest.main()
