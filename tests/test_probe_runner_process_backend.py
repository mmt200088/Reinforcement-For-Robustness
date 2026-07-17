"""Contracts for the persistent-process Stage-2 reward-probe backend."""
from __future__ import annotations

import os
import unittest
from unittest import mock

try:
    import torch

    from blb_stage2_rl import probe_runner as _probe_runner

    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - dependency-light local checkout
    torch = None  # type: ignore
    _probe_runner = None  # type: ignore
    _IMPORT_ERROR = exc

ProbeRunner = (
    _probe_runner.ProbeRunner if _probe_runner is not None else None
)
resolve_probe_backend = (
    getattr(_probe_runner, "resolve_probe_backend", None)
    if _probe_runner is not None else None
)


@unittest.skipUnless(_IMPORT_ERROR is None, f"probe runner unavailable: {_IMPORT_ERROR!r}")
class ProbeRunnerProcessBackendTest(unittest.TestCase):
    class _LocalWorker:
        def __init__(self, events):
            self.device = torch.device("cuda:0")
            self.events = events
            self.probe_batches = ("online-probe",)
            self.batch_sets = {"F1": self.probe_batches}

        def register_batch_set(self, key, batches):
            normalized = str(key)
            if normalized in self.batch_sets:
                raise ValueError(f"duplicate batch set {normalized}")
            self.batch_sets[normalized] = tuple(batches)
            self.events.append(("local-register", normalized, tuple(batches)))

        def install(self, decoded):
            self.events.append(("local-install", decoded))

        def clear(self):
            self.events.append(("local-clear", None))

        def run_trial(self, trial_idx, base_seed, batch_set_key="F1"):
            self.events.append(
                ("local-run", trial_idx, base_seed, str(batch_set_key))
            )
            return (float(trial_idx), float(base_seed), -1.0)

    class _RemoteWorker:
        def __init__(
                self,
                events,
                *,
                device_id=1,
                fail_submit=False,
                fail_receive=False,
                ):
            self.device = torch.device(f"cuda:{int(device_id)}")
            self.events = events
            self.fail_submit = fail_submit
            self.fail_receive = fail_receive
            self.pending = None
            self.closed = False
            self.close_count = 0

        def submit(self, operation, payload):
            self.events.append(("remote-submit", operation, payload))
            if self.fail_submit:
                raise RuntimeError("child submit marker")
            self.pending = (operation, payload)

        def receive(self, operation):
            self.events.append(("remote-receive", operation))
            if self.fail_receive:
                raise RuntimeError("child traceback marker")
            self.assert_pending(operation)
            _, payload = self.pending
            self.pending = None
            if operation == "run_trials":
                return {
                    "results": [
                        (int(idx), (float(idx), float(payload["base_seed"]), 1.0))
                        for idx in payload["trial_indices"]
                    ],
                    "wall_seconds": 0.25,
                }
            if operation == "run_action_trial":
                idx = int(payload["trial_idx"])
                return {
                    "trial_idx": idx,
                    "result": (float(idx), float(payload["base_seed"]), 2.0),
                    "wall_seconds": 0.20,
                }
            return {"wall_seconds": 0.01}

        def assert_pending(self, operation):
            if self.pending is None or self.pending[0] != operation:
                raise AssertionError(
                    f"expected pending {operation!r}, got {self.pending!r}"
                )

        def close(self):
            self.closed = True
            self.close_count += 1
            self.events.append(("remote-close", None))

    def _runner_with_process_count(
            self,
            events,
            *,
            process_count,
            fail_submit_device=None,
            fail_receive=False,
            fail_receive_device=None,
            ):
        local = self._LocalWorker(events)
        remotes = [
            self._RemoteWorker(
                events,
                device_id=device_id,
                fail_submit=(device_id == fail_submit_device),
                fail_receive=(
                    fail_receive or device_id == fail_receive_device
                ),
            )
            for device_id in range(1, int(process_count) + 1)
        ]
        return ProbeRunner([local], process_workers=remotes), remotes

    def _runner(self, events, *, fail_receive=False):
        runner, remotes = self._runner_with_process_count(
            events,
            process_count=1,
            fail_receive=fail_receive,
        )
        return runner, remotes[0]

    def test_remote_trials_are_submitted_before_local_gpu_runs(self):
        events = []
        runner, _remote = self._runner(events)

        results = runner.run_trials(k=2, base_seed=41)

        self.assertEqual(results, [(0.0, 41.0, -1.0), (1.0, 41.0, 1.0)])
        event_names = [event[0] for event in events]
        self.assertEqual(
            event_names,
            ["remote-submit", "local-run", "remote-receive"],
        )
        submitted = events[0]
        self.assertEqual(submitted[1], "run_trials")
        self.assertEqual(submitted[2]["trial_indices"], [1])
        diag = runner.last_diagnostics
        self.assertEqual(diag.per_worker_trial_counts, [1, 1])
        self.assertEqual(diag.per_worker_trial_indices, [[0], [1]])
        self.assertEqual(diag.devices, ["cuda:0", "cuda:1"])

    def test_multi_action_remote_work_overlaps_primary_worker(self):
        events = []
        runner, _remote = self._runner(events)
        action0 = object()
        action1 = object()

        results = runner.run_action_trials_once([action0, action1], base_seed=73)

        self.assertEqual(results, [(0.0, 73.0, -1.0), (1.0, 73.0, 2.0)])
        event_names = [event[0] for event in events]
        self.assertEqual(
            event_names,
            ["remote-submit", "local-install", "local-run", "remote-receive"],
        )
        self.assertIs(events[0][2]["decoded"], action1)
        self.assertTrue(runner.last_diagnostics.multi_action)

    def test_install_submits_remote_before_installing_primary(self):
        events = []
        runner, _remote = self._runner(events)
        decoded = object()

        runner.install_action(decoded)

        self.assertEqual(
            [event[0] for event in events],
            ["remote-submit", "local-install", "remote-receive"],
        )

    def test_remote_failure_names_worker_and_device(self):
        events = []
        runner, _remote = self._runner(events, fail_receive=True)

        with self.assertRaisesRegex(
            RuntimeError,
            r"worker 1 .*cuda:1.*child traceback marker",
        ):
            runner.run_trials(k=2, base_seed=1)

    def test_close_reaps_remote_worker(self):
        events = []
        runner, remote = self._runner(events)

        runner.close()

        self.assertTrue(remote.closed)
        self.assertEqual(events, [("remote-close", None)])

    def test_views_route_f1_and_f4_through_one_five_device_pool(self):
        events = []
        owner, remotes = self._runner_with_process_count(
            events, process_count=4,
        )
        owner.register_batch_set("F4", ["validation-full"])
        f1 = owner.view("F1")
        f4 = owner.view("F4")

        f1_results = f1.run_trials(k=5, base_seed=41)
        f4_results = f4.run_trials(k=5, base_seed=43)

        self.assertEqual(owner.num_workers, 5)
        self.assertEqual(
            [str(device) for device in owner.devices],
            [f"cuda:{device_id}" for device_id in range(5)],
        )
        self.assertEqual(len(remotes), 4)
        self.assertEqual(f1.pool_id, owner.pool_id)
        self.assertEqual(f4.pool_id, owner.pool_id)
        self.assertEqual(f1.backend, "process")
        self.assertEqual(f1.num_workers, owner.num_workers)
        self.assertEqual(f1.devices, owner.devices)
        self.assertEqual(len(f1_results), 5)
        self.assertEqual(len(f4_results), 5)
        registration_submissions = [
            event
            for event in events
            if event[:2] == ("remote-submit", "register_batch_set")
        ]
        self.assertEqual(len(registration_submissions), 4)
        run_submissions = [
            event
            for event in events
            if event[:2] == ("remote-submit", "run_trials")
        ]
        self.assertEqual(len(run_submissions), 8)
        self.assertEqual(
            [event[2]["batch_set_key"] for event in run_submissions],
            ["F1"] * 4 + ["F4"] * 4,
        )
        self.assertIs(f4.last_diagnostics, owner.last_diagnostics)

    def test_view_delegates_actions_with_its_batch_set_key(self):
        events = []
        owner, _remote = self._runner(events)
        owner.register_batch_set("F4", ["validation-full"])
        view = owner.view("F4")
        decoded = object()
        action0 = object()
        action1 = object()

        view.install_action(decoded)
        view.clear()
        results = view.run_action_trials_once([action0, action1], base_seed=73)

        self.assertEqual(len(results), 2)
        action_submission = next(
            event
            for event in events
            if event[:2] == ("remote-submit", "run_action_trial")
        )
        self.assertEqual(action_submission[2]["batch_set_key"], "F4")
        self.assertIs(view.last_diagnostics, owner.last_diagnostics)

    def test_view_close_is_noop_and_owner_close_is_idempotent(self):
        events = []
        owner, remotes = self._runner_with_process_count(
            events, process_count=4,
        )

        owner.view("F1").close()
        owner.view("F4").close()

        self.assertFalse(any(remote.closed for remote in remotes))
        owner.close()
        owner.close()
        self.assertTrue(all(remote.closed for remote in remotes))
        self.assertEqual([remote.close_count for remote in remotes], [1] * 4)
        self.assertEqual(events.count(("remote-close", None)), 4)

    def test_batch_set_keys_must_be_nonempty_and_unique(self):
        events = []
        owner, _remote = self._runner(events)

        with self.assertRaises(ValueError):
            owner.register_batch_set("", ["invalid"])
        owner.register_batch_set("F4", ["validation-full"])
        with self.assertRaises(ValueError):
            owner.register_batch_set("F4", ["duplicate"])

    def _assert_registration_failure_poisoned_pool(
            self, *, fail_submit_device=None, fail_receive_device=None,
            ):
        events = []
        owner, remotes = self._runner_with_process_count(
            events,
            process_count=4,
            fail_submit_device=fail_submit_device,
            fail_receive_device=fail_receive_device,
        )

        with self.assertRaisesRegex(RuntimeError, "register_batch_set"):
            owner.register_batch_set("F4", ["validation-full"])

        self.assertTrue(all(remote.closed for remote in remotes))
        self.assertEqual([remote.close_count for remote in remotes], [1] * 4)
        event_count_after_failure = len(events)
        with self.assertRaisesRegex(RuntimeError, "closed|poison"):
            owner.run_trials(k=1, base_seed=41)
        with self.assertRaisesRegex(RuntimeError, "closed|poison"):
            owner.register_batch_set("F5", ["final-validation"])
        with self.assertRaisesRegex(RuntimeError, "closed|poison"):
            owner.view("F1").run_trials(k=1, base_seed=43)
        self.assertEqual(len(events), event_count_after_failure)

        owner.close()
        self.assertEqual([remote.close_count for remote in remotes], [1] * 4)

    def test_batch_registration_submit_failure_closes_and_poisons_pool(self):
        self._assert_registration_failure_poisoned_pool(
            fail_submit_device=2,
        )

    def test_batch_registration_receive_failure_closes_and_poisons_pool(self):
        self._assert_registration_failure_poisoned_pool(
            fail_receive_device=2,
        )

    def test_probe_worker_uses_an_immutable_keyed_batch_set(self):
        model = object()
        worker = _probe_runner.ProbeWorker(
            device=torch.device("cuda:0"),
            model=model,
            handler=object(),
            bridge=object(),
            probe_batches=["online-probe"],
            is_regression=False,
            metric_profile="mrpc",
        )
        validation_batches = ["validation-full"]
        worker.register_batch_set("F4", validation_batches)
        validation_batches.append("late-mutation")

        with mock.patch.object(_probe_runner.torch.cuda, "device"):
            with mock.patch.object(
                _probe_runner, "reseed_noise_rng_for_device",
            ):
                with mock.patch.object(
                    _probe_runner,
                    "run_installed_probe_trial",
                    return_value=(0.3, 0.9, 0.8),
                ) as run_probe:
                    result = worker.run_trial(0, 41, batch_set_key="F4")

        self.assertEqual(result, (0.3, 0.9, 0.8))
        self.assertEqual(run_probe.call_args.args[1], ("validation-full",))
        self.assertEqual(run_probe.call_args.kwargs["metric_profile"], "mrpc")


@unittest.skipUnless(_IMPORT_ERROR is None, f"probe runner unavailable: {_IMPORT_ERROR!r}")
class ProbeBackendSelectionTest(unittest.TestCase):
    def test_process_is_default_and_thread_is_explicit_fallback(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_probe_backend(), "process")
        self.assertEqual(resolve_probe_backend("PROCESS"), "process")
        self.assertEqual(resolve_probe_backend("thread"), "thread")

    def test_invalid_backend_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "BLB_STAGE2_PROBE_BACKEND"):
            resolve_probe_backend("asyncio")


if __name__ == "__main__":
    unittest.main()
