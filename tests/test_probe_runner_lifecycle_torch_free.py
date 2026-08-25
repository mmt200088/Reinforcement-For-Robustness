"""Torch-free contracts for ordinary deterministic ProbeRunner execution."""

from __future__ import annotations

import importlib.util
import inspect
import math
from pathlib import Path
import sys
import types
import unittest

_MISSING_MODULE = object()


class _FakeDevice:
    def __init__(self, name):
        self.name = str(name)

    def __str__(self):
        return self.name


class _Worker:
    def __init__(self, name):
        self.device = _FakeDevice(name)
        self.probe_batches = ("probe",)
        self.installed = None
        self.calls = []

    def install(self, decoded):
        self.installed = decoded

    def run_trial(self, trial_index, base_seed, batch_set_key="F1"):
        self.calls.append((int(trial_index), int(base_seed), str(batch_set_key)))
        return (float(trial_index), float(base_seed), 3.0)


class _RemoteWorker:
    def __init__(self, name, *, partial_then_fail=False):
        self.device = _FakeDevice(name)
        self.partial_then_fail = bool(partial_then_fail)
        self.pending = None
        self.closed = False
        self.submissions = []

    def submit(self, operation, payload):
        self.pending = (str(operation), payload)
        self.submissions.append((str(operation), payload))

    def receive(self, operation, result_handler=None):
        if self.pending is None or self.pending[0] != operation:
            raise AssertionError(f"unexpected pending operation {self.pending!r}")
        _, payload = self.pending
        self.pending = None
        if self.partial_then_fail:
            self.partial_then_fail = False
            if result_handler is None:
                raise AssertionError("partial results require a result handler")
            if operation == "run_trials":
                trial_index = int(payload["trial_indices"][0])
                result_handler(
                    {
                        "trial_index": trial_index,
                        "result": (
                            float(trial_index),
                            float(payload["base_seed"]),
                            4.0,
                        ),
                    }
                )
            else:
                group = payload["action_groups"][0]
                result_handler(
                    {
                        "action_index": int(group["action_index"]),
                        "trial_index": int(group["trial_indices"][0]),
                        "result": (
                            float(group["trial_indices"][0]),
                            float(group["base_seed"]),
                            4.0,
                        ),
                    }
                )
            raise BrokenPipeError("injected partial child crash")
        if operation == "run_trials":
            return {
                "results": [
                    (
                        int(trial_index),
                        (
                            float(trial_index),
                            float(payload["base_seed"]),
                            4.0,
                        ),
                    )
                    for trial_index in payload["trial_indices"]
                ],
                "wall_seconds": 0.01,
            }
        results = []
        for group in payload["action_groups"]:
            for trial_index in group["trial_indices"]:
                results.append(
                    (
                        int(group["action_index"]),
                        int(trial_index),
                        (
                            float(trial_index),
                            float(group["base_seed"]),
                            4.0,
                        ),
                    )
                )
        return {"results": results, "wall_seconds": 0.01}

    def close(self):
        self.closed = True


class ProbeRunnerDeterministicTorchFreeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._previous_modules = {}
        package_name = "_probe_runner_deterministic_testpkg"
        source_root = Path(__file__).resolve().parents[1]

        package = types.ModuleType(package_name)
        package.__path__ = [str(source_root / "blb_stage2_rl")]
        cls._install_module(package_name, package)

        action_space = types.ModuleType("rfr.search.common.action_space")
        action_space.ActionDecodeResult = object
        cls._install_module("rfr.search.common.action_space", action_space)

        inference_eval = types.ModuleType(f"{package_name}.inference_eval")
        inference_eval.run_installed_probe_trial = lambda *args, **kwargs: None
        cls._install_module(f"{package_name}.inference_eval", inference_eval)

        seed_utils = types.ModuleType("blb_stage2_rl.seed_utils")
        seed_utils.derive_probe_trial_seed = lambda base_seed, trial_index: (
            int(base_seed) ^ (int(trial_index) * 2654435761)
        )
        cls._install_module("blb_stage2_rl.seed_utils", seed_utils)

        if "torch" not in sys.modules:
            torch = types.ModuleType("torch")
            torch.device = _FakeDevice
            torch.Tensor = object
            torch.cuda = types.SimpleNamespace()
            cls._install_module("torch", torch)
        if "torch.nn" not in sys.modules:
            torch_nn = types.ModuleType("torch.nn")
            torch_nn.Module = object
            cls._install_module("torch.nn", torch_nn)

        function_handler = types.ModuleType("rfr.search.runtime.model_handler")
        function_handler.ReversibleLayerHandler = object
        function_handler.reseed_noise_rng_for_device = lambda *args, **kwargs: None
        cls._install_module("rfr.search.runtime.model_handler", function_handler)

        module_name = f"{package_name}.probe_runner"
        spec = importlib.util.spec_from_file_location(
            module_name,
            source_root / "src/rfr/search/runtime/probe_runner.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("unable to load probe_runner.py")
        module = importlib.util.module_from_spec(spec)
        cls._install_module(module_name, module)
        spec.loader.exec_module(module)
        cls.probe_runner = module

    @classmethod
    def _install_module(cls, name, module):
        if name not in cls._previous_modules:
            cls._previous_modules[name] = sys.modules.get(
                name,
                _MISSING_MODULE,
            )
        sys.modules[name] = module

    @classmethod
    def tearDownClass(cls):
        for name, previous in reversed(tuple(cls._previous_modules.items())):
            if previous is _MISSING_MODULE:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous

    def test_public_trial_apis_do_not_expose_attempt_lifecycle_parameters(self):
        for method_name in (
            "run_trials_at_indices",
            "run_action_trial_groups",
            "run_action_trial_groups_at_indices",
        ):
            parameters = inspect.signature(getattr(self.probe_runner.ProbeRunner, method_name)).parameters
            self.assertNotIn("lifecycle_handler", parameters)
            self.assertNotIn("retry_of_by_trial", parameters)
            self.assertNotIn("retry_of_by_task", parameters)

    def test_trial_result_acceptors_preserve_none_loss_for_later_cap(self):
        direct_results = {3: None}
        self.probe_runner.ProbeRunner._accept_trial_result(
            trial_index=3,
            raw_result=(None, 0.9, 0.8),
            expected_indices=(3,),
            results=direct_results,
        )
        grouped_results = [[None]]
        self.probe_runner.ProbeRunner._accept_grouped_result(
            action_index=0,
            trial_index=3,
            raw_result=(None, 0.9, 0.8),
            expected_tasks=((0, 3),),
            results=grouped_results,
            position_by_trial={3: 0},
        )

        self.assertTrue(math.isnan(direct_results[3][0]))
        self.assertEqual(direct_results[3][1:], (0.9, 0.8))
        self.assertTrue(math.isnan(grouped_results[0][0][0]))
        self.assertEqual(grouped_results[0][0][1:], (0.9, 0.8))

    def test_run_trials_is_a_thin_exact_index_wrapper(self):
        runner = self.probe_runner.ProbeRunner([_Worker("cuda:0")])
        calls = []

        def run_exact(**kwargs):
            calls.append(kwargs)
            return [(float(index), 70.0, 3.0) for index in kwargs["trial_indices"]]

        runner.run_trials_at_indices = run_exact

        results = runner.run_trials(3, base_seed=70)

        self.assertEqual(results, [(0.0, 70.0, 3.0), (1.0, 70.0, 3.0), (2.0, 70.0, 3.0)])
        self.assertEqual(
            calls,
            [
                {
                    "trial_indices": range(0, 3),
                    "base_seed": 70,
                    "batch_set_key": "F1",
                }
            ],
        )

    def test_explicit_indices_keep_caller_order_and_deterministic_seeds(self):
        runner = self.probe_runner.ProbeRunner(
            [
                _Worker("cuda:0"),
                _Worker("cuda:1"),
            ]
        )

        results = runner.run_trials_at_indices(
            trial_indices=[3, 7, 9],
            base_seed=70,
        )

        self.assertEqual(results, [(3.0, 70.0, 3.0), (7.0, 70.0, 3.0), (9.0, 70.0, 3.0)])
        self.assertEqual(
            runner.last_diagnostics.per_worker_trial_indices,
            [[3, 9], [7]],
        )
        self.assertEqual(
            runner.last_diagnostics.per_worker_trial_seeds,
            [
                [70 ^ (3 * 2654435761), 70 ^ (9 * 2654435761)],
                [70 ^ (7 * 2654435761)],
            ],
        )

    def test_process_partial_result_retries_only_missing_trial(self):
        failed = _RemoteWorker("cuda:2", partial_then_fail=True)
        runner = self.probe_runner.ProbeRunner(
            [_Worker("cuda:0")],
            process_workers=[_RemoteWorker("cuda:1"), failed],
        )

        results = runner.run_trials_at_indices(
            trial_indices=[0, 1, 2, 3, 4, 5],
            base_seed=70,
        )

        self.assertEqual(results[2], (2.0, 70.0, 4.0))
        self.assertEqual(runner.last_diagnostics.retried_trial_indices, [5])
        self.assertTrue(failed.closed)
        operation, payload = failed.submissions[0]
        self.assertEqual(operation, "run_trials")
        self.assertNotIn("physical_attempt_ids", payload)

    def test_grouped_partial_result_retries_only_missing_task(self):
        failed = _RemoteWorker("cuda:2", partial_then_fail=True)
        runner = self.probe_runner.ProbeRunner(
            [_Worker("cuda:0")],
            process_workers=[_RemoteWorker("cuda:1"), failed],
        )

        results = runner.run_action_trial_groups(
            [object(), object()],
            base_seeds=[70, 80],
            k=3,
        )

        self.assertEqual(results[0][2], (2.0, 70.0, 4.0))
        self.assertEqual(
            runner.last_diagnostics.retried_action_trial_indices,
            [(1, 2)],
        )
        self.assertTrue(failed.closed)
        operation, payload = failed.submissions[0]
        self.assertEqual(operation, "run_action_trial_groups")
        self.assertTrue(all("physical_attempt_ids" not in group for group in payload["action_groups"]))

    def test_view_forwards_exact_indices_without_extra_protocol(self):
        runner = self.probe_runner.ProbeRunner([_Worker("cuda:0")])
        view = runner.view("F1")

        results = view.run_trials_at_indices(
            trial_indices=[4],
            base_seed=70,
        )

        self.assertEqual(results, [(4.0, 70.0, 3.0)])


class ProbeRunnerModuleRestorationTest(unittest.TestCase):
    def test_stub_cleanup_restores_preexisting_module(self):
        module_name = "_probe_runner_preexisting_module_sentinel"
        original = types.ModuleType(module_name)
        replacement = types.ModuleType(module_name)
        sys.modules[module_name] = original

        class Harness:
            _inserted_modules = []
            _previous_modules = {}

        install = ProbeRunnerDeterministicTorchFreeTest._install_module.__func__
        teardown = ProbeRunnerDeterministicTorchFreeTest.tearDownClass.__func__
        try:
            install(Harness, module_name, replacement)
            self.assertIs(sys.modules[module_name], replacement)
            teardown(Harness)
            self.assertIs(sys.modules[module_name], original)
        finally:
            sys.modules[module_name] = original


if __name__ == "__main__":
    unittest.main()
