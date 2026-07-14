import threading
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - local macOS env may be torch-free.
    torch = None


@unittest.skipIf(torch is None, "torch is required for deterministic probe tests")
class DeterministicProbeLockTests(unittest.TestCase):
    @staticmethod
    def _make_probe_worker():
        from blb_stage2_rl.probe_runner import ProbeWorker
        from function_handler import _sample_independent_gaussian

        class FakeNoisyModel(torch.nn.Module):
            def forward(
                    self,
                    input_ids,
                    attention_mask=None,
                    labels=None,
                    token_type_ids=None,
                    ):
                reference = torch.zeros(
                    int(input_ids.shape[0]), 2,
                    device=input_ids.device,
                    dtype=torch.float32,
                )
                return SimpleNamespace(
                    logits=_sample_independent_gaussian(reference, 1.0),
                )

        device = torch.device("cpu")
        batch = SimpleNamespace(
            input_ids=torch.zeros(8, 3, device=device, dtype=torch.long),
            attention_mask=torch.ones(8, 3, device=device, dtype=torch.long),
            labels=torch.tensor(
                [0, 1, 0, 1, 0, 1, 0, 1],
                device=device,
                dtype=torch.long,
            ),
            token_type_ids=None,
        )
        return ProbeWorker(
            device=device,
            model=FakeNoisyModel(),
            handler=None,
            bridge=None,
            probe_batches=[batch],
            is_regression=False,
            metric_profile="mrpc",
        )

    def test_probe_batch_metrics_are_sample_weighted_with_tail_batch(self):
        from blb_stage2_rl.eval_metrics import weighted_probe_batch_means

        loss, m1, m2 = weighted_probe_batch_means(
            losses=[0.0, 10.0],
            m1s=[1.0, 0.0],
            m2s=[1.0, 0.0],
            counts=[4, 1],
        )
        self.assertAlmostEqual(loss, 2.0)
        self.assertAlmostEqual(m1, 0.8)
        self.assertAlmostEqual(m2, 0.8)

    def test_mrpc_metric2_is_weighted_f1_not_accuracy(self):
        from blb_stage2_rl.eval_metrics import finalize_probe_trial_metrics
        from blb_stage2_rl.inference_eval import run_installed_probe_trial
        import blb_stage2_rl.probe_runner as probe_mod

        kwargs = dict(
            losses=[0.2, 0.4],
            m1s=[1.0, 1.0 / 3.0],
            m2s=[1.0, 1.0 / 3.0],
            counts=[2, 3],
            preds=[[0, 1], [0, 0, 1]],
            labels=[[0, 1], [1, 1, 1]],
            is_regression=False,
        )
        loss, m1, m2 = finalize_probe_trial_metrics(
            **kwargs,
            metric_profile="mrpc",
        )
        self.assertAlmostEqual(loss, 0.32)
        self.assertAlmostEqual(m1, 0.6)
        self.assertAlmostEqual(m2, 0.6333333333333333)

        _loss, _m1, large_m2 = finalize_probe_trial_metrics(
            **kwargs,
            metric_profile="mrpc_large",
        )
        self.assertAlmostEqual(large_m2, m2)

        _loss, _m1, sst2_m2 = finalize_probe_trial_metrics(
            **kwargs,
            metric_profile="sst2",
        )
        self.assertAlmostEqual(sst2_m2, 0.6)
        self.assertIs(probe_mod.run_installed_probe_trial, run_installed_probe_trial)

    def _make_env(self, *, seed: int, lock, scope=None, device=None, use_stream=False):
        import blb_stage2_rl.env as env_mod
        from function_handler import _sample_independent_gaussian

        class FakeNoisyModel(torch.nn.Module):
            def forward(self, input_ids, attention_mask=None, labels=None, token_type_ids=None):
                reference = torch.zeros(
                    int(input_ids.shape[0]), 2,
                    device=input_ids.device,
                    dtype=torch.float32,
                )
                return SimpleNamespace(logits=_sample_independent_gaussian(reference, 1.0))

        device = torch.device("cpu") if device is None else torch.device(device)
        batches = []
        for offset in (0, 10):
            labels = torch.tensor([0, 1, 0, 1], device=device, dtype=torch.long)
            batches.append(env_mod.ProbeBatch(
                input_ids=torch.full((4, 3), int(offset), device=device, dtype=torch.long),
                attention_mask=torch.ones((4, 3), device=device, dtype=torch.long),
                labels=labels,
                token_type_ids=None,
            ))

        probe_env = env_mod.BLBStage2Env.__new__(env_mod.BLBStage2Env)
        probe_env.probe_noise_seed = int(seed)
        probe_env.probe_device_lock = lock
        probe_env.probe_noise_scope = scope
        probe_env.probe_device_lock_requires_sync = False
        probe_env.probe_cuda_stream = (
            torch.cuda.Stream(device=device)
            if bool(use_stream) and device.type == "cuda" else None
        )
        probe_env._device = device
        probe_env.model = FakeNoisyModel()
        probe_env.probe_batches = batches
        probe_env.is_regression = False
        return probe_env

    @staticmethod
    def _metrics_tuple(metrics):
        return (
            metrics.loss_mean,
            metrics.loss_std,
            metrics.metric1_mean,
            metrics.metric2_mean,
            metrics.metric1_std,
            metrics.metric2_std,
            metrics.loss_max,
            metrics.metric1_min,
            metrics.metric2_min,
        )

    def test_probe_metrics_are_computed_outside_device_lock(self):
        import blb_stage2_rl.inference_eval as inference_eval

        lock = threading.Lock()
        probe_env = self._make_env(seed=123, lock=lock)
        original = inference_eval.tensor_scalar_sequences_to_float_lists
        calls = []

        def checked_metric_sync(*args, **kwargs):
            self.assertFalse(lock.locked(), "metric sync should not run under probe_device_lock")
            calls.append(True)
            return original(*args, **kwargs)

        with mock.patch.object(
                inference_eval,
                "tensor_scalar_sequences_to_float_lists",
                side_effect=checked_metric_sync,
        ):
            probe_env._eval_on_probe_deterministic(3)
        self.assertGreater(len(calls), 0)

    def test_returned_metrics_and_diagnostics_reuse_shared_trial_seeds(self):
        import blb_stage2_rl.seed_utils as seed_utils

        probe_env = self._make_env(seed=123, lock=threading.Lock())
        expected = (9001, 9002, 9003)

        with mock.patch.object(
                seed_utils,
                "derive_probe_trial_seed",
                side_effect=lambda _base, trial_idx: expected[int(trial_idx)],
        ) as derive_mock:
            metrics = probe_env._eval_on_probe_deterministic(3)

        self.assertEqual(metrics.trial_seeds, expected)
        self.assertEqual(
            probe_env._last_probe_diagnostics["per_worker_trial_seeds"],
            [list(expected)],
        )
        self.assertEqual(derive_mock.call_count, 3)

    def test_probe_worker_replays_independent_noise_for_same_trial_seed(self):
        worker = self._make_probe_worker()

        first = worker.run_trial(trial_idx=3, base_seed=12345)
        second = worker.run_trial(trial_idx=3, base_seed=12345)

        self.assertEqual(second, first)

    def test_probe_worker_does_not_mutate_global_rng_streams(self):
        worker = self._make_probe_worker()
        torch_state = torch.get_rng_state()
        numpy_state = np.random.get_state()
        try:
            torch.manual_seed(2468)
            expected_torch = torch.rand(8)
            np.random.seed(1357)
            expected_numpy = np.random.random(8)

            torch.manual_seed(2468)
            np.random.seed(1357)
            worker.run_trial(trial_idx=2, base_seed=67890)
            actual_torch = torch.rand(8)
            actual_numpy = np.random.random(8)
        finally:
            torch.set_rng_state(torch_state)
            np.random.set_state(numpy_state)

        self.assertTrue(torch.equal(actual_torch, expected_torch))
        np.testing.assert_array_equal(actual_numpy, expected_numpy)

    def test_noise_rng_scope_isolates_and_restores_thread_local_generator(self):
        from function_handler import (
            _noise_generator_key,
            _sample_independent_gaussian,
            noise_rng_scope,
            reseed_noise_rng,
            reseed_noise_rng_for_device,
        )

        reference = torch.zeros(8, device=torch.device("cpu"), dtype=torch.float32)
        self.assertEqual(_noise_generator_key(reference.device), str(reference.device))
        with noise_rng_scope("scope-a"):
            self.assertEqual(
                _noise_generator_key(reference.device),
                f"{reference.device}|scope=scope-a",
            )
            reseed_noise_rng_for_device(reference.device, 123)
            a1 = _sample_independent_gaussian(reference, 1.0)
            with noise_rng_scope("scope-inner"):
                self.assertEqual(
                    _noise_generator_key(reference.device),
                    f"{reference.device}|scope=scope-inner",
                )
            self.assertEqual(
                _noise_generator_key(reference.device),
                f"{reference.device}|scope=scope-a",
            )
        with noise_rng_scope("scope-b"):
            reseed_noise_rng_for_device(reference.device, 456)
            _ = _sample_independent_gaussian(reference, 1.0)
        with noise_rng_scope("scope-a"):
            reseed_noise_rng_for_device(reference.device, 123)
            a2 = _sample_independent_gaussian(reference, 1.0)

        reseed_noise_rng_for_device(reference.device, 123)
        unscoped = _sample_independent_gaussian(reference, 1.0)

        self.assertEqual(_noise_generator_key(reference.device), str(reference.device))
        self.assertTrue(torch.equal(a1, a2))
        self.assertTrue(torch.equal(a1, unscoped))

    def test_global_noise_reseed_reaches_scoped_generators(self):
        from function_handler import (
            _sample_independent_gaussian,
            noise_rng_scope,
            reseed_noise_rng,
        )

        reference = torch.zeros(8, device=torch.device("cpu"), dtype=torch.float32)
        with noise_rng_scope("scope-a"):
            reseed_noise_rng(777)
            fixed_a1 = _sample_independent_gaussian(reference, 1.0)
            reseed_noise_rng(777)
            fixed_a2 = _sample_independent_gaussian(reference, 1.0)
            reseed_noise_rng(None)
            os_seeded = _sample_independent_gaussian(reference, 1.0)

        reseed_noise_rng(777)
        unscoped_fixed = _sample_independent_gaussian(reference, 1.0)

        self.assertTrue(torch.equal(fixed_a1, fixed_a2))
        self.assertTrue(torch.equal(fixed_a1, unscoped_fixed))
        self.assertFalse(torch.equal(fixed_a1, os_seeded))

    def test_eval_model_does_not_recurse_eval_every_probe(self):
        lock = threading.Lock()
        probe_env = self._make_env(seed=123, lock=lock)
        probe_env.model.eval()
        eval_calls = []
        original_eval = probe_env.model.eval

        def counted_eval():
            eval_calls.append(True)
            return original_eval()

        probe_env.model.eval = counted_eval
        probe_env._eval_on_probe_deterministic(2)
        self.assertEqual(eval_calls, [])

    def test_concurrent_same_device_probe_matches_serial_metrics(self):
        lock = threading.Lock()
        serial_a = self._metrics_tuple(self._make_env(seed=101, lock=lock)._eval_on_probe_deterministic(4))
        serial_b = self._metrics_tuple(self._make_env(seed=202, lock=lock)._eval_on_probe_deterministic(4))

        concurrent_lock = threading.Lock()
        results = {}

        def run(name, seed):
            results[name] = self._metrics_tuple(
                self._make_env(seed=seed, lock=concurrent_lock)._eval_on_probe_deterministic(4)
            )

        threads = [
            threading.Thread(target=run, args=("a", 101)),
            threading.Thread(target=run, args=("b", 202)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(results["a"], serial_a)
        self.assertEqual(results["b"], serial_b)

    def test_scoped_same_device_probe_skips_lock_and_matches_serial_metrics(self):
        class ForbiddenLock:
            def __enter__(self):
                raise AssertionError("scoped probe should not acquire shared lock")

            def __exit__(self, exc_type, exc, tb):
                return False

        serial_a = self._metrics_tuple(
            self._make_env(
                seed=101,
                lock=ForbiddenLock(),
                scope="worker-a",
            )._eval_on_probe_deterministic(4)
        )
        serial_b = self._metrics_tuple(
            self._make_env(
                seed=202,
                lock=ForbiddenLock(),
                scope="worker-b",
            )._eval_on_probe_deterministic(4)
        )

        results = {}

        def run(name, seed, scope):
            results[name] = self._metrics_tuple(
                self._make_env(
                    seed=seed,
                    lock=ForbiddenLock(),
                    scope=scope,
                )._eval_on_probe_deterministic(4)
            )

        threads = [
            threading.Thread(target=run, args=("a", 101, "worker-a")),
            threading.Thread(target=run, args=("b", 202, "worker-b")),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(results["a"], serial_a)
        self.assertEqual(results["b"], serial_b)

    @unittest.skipIf(
        torch is None or not torch.cuda.is_available(),
        "CUDA is required for scoped same-device probe concurrency test",
    )
    def test_cuda_scoped_same_device_probe_matches_serial_without_sync(self):
        class ForbiddenLock:
            def __enter__(self):
                raise AssertionError("scoped CUDA probe should not acquire shared lock")

            def __exit__(self, exc_type, exc, tb):
                return False

        device = torch.device("cuda:0")
        serial_a = self._metrics_tuple(
            self._make_env(
                seed=303,
                lock=ForbiddenLock(),
                scope="cuda-worker-a",
                device=device,
                use_stream=True,
            )._eval_on_probe_deterministic(3)
        )
        serial_b = self._metrics_tuple(
            self._make_env(
                seed=404,
                lock=ForbiddenLock(),
                scope="cuda-worker-b",
                device=device,
                use_stream=True,
            )._eval_on_probe_deterministic(3)
        )

        results = {}

        def run(name, seed, scope):
            results[name] = self._metrics_tuple(
                self._make_env(
                    seed=seed,
                    lock=ForbiddenLock(),
                    scope=scope,
                    device=device,
                    use_stream=True,
                )._eval_on_probe_deterministic(3)
            )

        with mock.patch("torch.cuda.synchronize") as sync_mock:
            threads = [
                threading.Thread(target=run, args=("a", 303, "cuda-worker-a")),
                threading.Thread(target=run, args=("b", 404, "cuda-worker-b")),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(sync_mock.call_count, 0)

        self.assertEqual(results["a"], serial_a)
        self.assertEqual(results["b"], serial_b)

    def test_cuda_sync_only_when_shared_device_lock_requires_it(self):
        lock = threading.Lock()
        probe_env = self._make_env(seed=123, lock=lock)
        probe_env._device = torch.device("cuda:0")

        with (
            mock.patch("function_handler.reseed_noise_rng_for_device"),
            mock.patch("torch.cuda.synchronize") as sync_mock,
        ):
            probe_env.probe_device_lock_requires_sync = False
            probe_env._eval_on_probe_deterministic(2)
            self.assertEqual(sync_mock.call_count, 0)

            probe_env.probe_device_lock_requires_sync = True
            probe_env._eval_on_probe_deterministic(3)
            self.assertEqual(sync_mock.call_count, 3)


if __name__ == "__main__":
    unittest.main()
