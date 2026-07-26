"""Contracts for the persistent-process Stage-2 reward-probe backend."""
from __future__ import annotations

import os
import signal
import threading
import time
import types
import unittest
from unittest import mock

from elastic_gpu import ElasticGPUFailure

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


_GPU_INTEGRATION_READY = (
    os.environ.get("BLB_STAGE2_RUN_GPU_INTEGRATION") == "1"
    and _IMPORT_ERROR is None
    and torch is not None
    and torch.cuda.device_count() >= 5
)


@unittest.skipUnless(_IMPORT_ERROR is None, f"probe runner unavailable: {_IMPORT_ERROR!r}")
class ProbeRunnerProcessBackendTest(unittest.TestCase):
    class _LocalWorker:
        def __init__(self, events):
            self.device = torch.device("cuda:0")
            self.events = events
            self.probe_batches = ("online-probe",)
            self.batch_sets = {"F1": self.probe_batches}
            self.installed = None

        def register_batch_set(self, key, batches):
            normalized = str(key)
            if normalized in self.batch_sets:
                raise ValueError(f"duplicate batch set {normalized}")
            self.batch_sets[normalized] = tuple(batches)
            self.events.append(("local-register", normalized, tuple(batches)))

        def install(self, decoded):
            self.installed = decoded
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
                fail_receive_once=False,
                receive_error=None,
                duplicate_result=False,
                ):
            self.device = torch.device(f"cuda:{int(device_id)}")
            self.events = events
            self.fail_submit = fail_submit
            self.fail_receive = fail_receive
            self.fail_receive_once = fail_receive_once
            self.receive_error = receive_error
            self.duplicate_result = duplicate_result
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
            if self.fail_receive_once:
                self.fail_receive_once = False
                raise BrokenPipeError("injected replica pipe loss")
            if self.receive_error is not None:
                raise self.receive_error
            self.assert_pending(operation)
            _, payload = self.pending
            self.pending = None
            if operation == "run_trials":
                results = [
                    (int(idx), (float(idx), float(payload["base_seed"]), 1.0))
                    for idx in payload["trial_indices"]
                ]
                if self.duplicate_result and results:
                    results.append(results[0])
                return {
                    "results": results,
                    "wall_seconds": 0.25,
                }
            if operation == "run_action_trial":
                idx = int(payload["trial_idx"])
                return {
                    "trial_idx": idx,
                    "result": (float(idx), float(payload["base_seed"]), 2.0),
                    "wall_seconds": 0.20,
                }
            if operation == "run_action_trial_groups":
                results = []
                for group in payload["action_groups"]:
                    for idx in group["trial_indices"]:
                        results.append((
                            int(group["action_index"]),
                            int(idx),
                            (float(idx), float(group["base_seed"]), 2.0),
                        ))
                return {"results": results, "wall_seconds": 0.50}
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

    def test_grouped_action_trials_balance_four_actions_by_five_trials(self):
        assignments = _probe_runner._split_action_trial_tasks_cached(4, 5, 4)

        self.assertEqual([len(tasks) for tasks in assignments], [5, 5, 5, 5])
        self.assertEqual(
            assignments[0],
            ((0, 0), (0, 4), (1, 3), (2, 2), (3, 1)),
        )
        self.assertEqual(
            sorted(task for tasks in assignments for task in tasks),
            [(action_idx, trial_idx) for action_idx in range(4) for trial_idx in range(5)],
        )

    def test_grouped_action_trials_preserve_action_and_trial_order(self):
        events = []
        runner, _remote = self._runner(events)
        actions = [object(), object()]

        results = runner.run_action_trial_groups(
            actions,
            base_seeds=[70, 80],
            k=3,
        )

        self.assertEqual(results, [
            [(0.0, 70.0, -1.0), (1.0, 70.0, 2.0), (2.0, 70.0, -1.0)],
            [(0.0, 80.0, 2.0), (1.0, 80.0, -1.0), (2.0, 80.0, 2.0)],
        ])
        self.assertEqual(events[0][:2], ("remote-submit", "run_action_trial_groups"))
        self.assertEqual(
            events[0][2]["action_groups"],
            [
                {
                    "action_index": 0,
                    "decoded": actions[0],
                    "base_seed": 70,
                    "trial_indices": [1],
                },
                {
                    "action_index": 1,
                    "decoded": actions[1],
                    "base_seed": 80,
                    "trial_indices": [0, 2],
                },
            ],
        )
        self.assertEqual(events[-1], ("remote-receive", "run_action_trial_groups"))
        diag = runner.last_diagnostics
        self.assertTrue(diag.multi_action)
        self.assertEqual(diag.action_count, 2)
        self.assertEqual(diag.trials_per_action, 3)
        self.assertEqual(
            diag.per_worker_action_trial_indices,
            [[(0, 0), (0, 2), (1, 1)], [(0, 1), (1, 0), (1, 2)]],
        )

    def test_grouped_action_trials_preserve_explicit_indices_on_process_backend(self):
        events = []
        runner, _remote = self._runner(events)
        actions = [object(), object()]

        results = runner.run_action_trial_groups_at_indices(
            actions,
            base_seeds=[70, 80],
            trial_indices=[1, 3, 4],
        )

        self.assertEqual(results, [
            [(1.0, 70.0, -1.0), (3.0, 70.0, 2.0), (4.0, 70.0, -1.0)],
            [(1.0, 80.0, 2.0), (3.0, 80.0, -1.0), (4.0, 80.0, 2.0)],
        ])
        self.assertEqual(
            runner.last_diagnostics.per_worker_action_trial_indices,
            [[(0, 1), (0, 4), (1, 3)], [(0, 3), (1, 1), (1, 4)]],
        )
        self.assertEqual(runner.last_diagnostics.trials_per_action, 3)

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

    def test_recoverable_replica_failure_retries_only_missing_trials(self):
        events = []
        local = self._LocalWorker(events)

        def canonical_local(
                worker, trial_idx, base_seed, batch_set_key="F1",
                ):
            worker.events.append(
                ("local-run", trial_idx, base_seed, str(batch_set_key))
            )
            return (float(trial_idx), float(base_seed), 1.0)

        local.run_trial = types.MethodType(canonical_local, local)
        healthy = self._RemoteWorker(events, device_id=1)
        failed = self._RemoteWorker(
            events,
            device_id=2,
            fail_receive_once=True,
        )
        runner = ProbeRunner(
            [local],
            process_workers=[healthy, failed],
        )

        results = runner.run_trials(k=6, base_seed=41)

        self.assertEqual(
            results,
            [
                (float(index), 41.0, 1.0)
                for index in range(6)
            ],
        )
        self.assertEqual(runner.num_workers, 2)
        self.assertEqual(runner.pool_generation, 1)
        self.assertEqual(
            runner.last_diagnostics.retried_trial_indices,
            [2, 5],
        )
        self.assertNotIn(
            1,
            runner.last_diagnostics.retried_trial_indices,
        )
        self.assertNotIn(
            4,
            runner.last_diagnostics.retried_trial_indices,
        )
        self.assertTrue(failed.closed)

        runner.close()
        runner.close()
        self.assertEqual(failed.close_count, 1)
        self.assertEqual(healthy.close_count, 1)

    def test_remote_shape_error_remains_fatal(self):
        events = []
        remote = self._RemoteWorker(
            events,
            receive_error=RuntimeError("shape mismatch"),
        )
        runner = ProbeRunner(
            [self._LocalWorker(events)],
            process_workers=[remote],
        )

        with self.assertRaisesRegex(RuntimeError, "shape mismatch"):
            runner.run_trials(k=2, base_seed=41)

        self.assertEqual(runner.pool_generation, 0)
        self.assertFalse(remote.closed)

    def test_primary_transport_failure_requests_supervisor_restart(self):
        events = []
        local = self._LocalWorker(events)

        def fail_primary(
                worker, trial_idx, base_seed, batch_set_key="F1",
                ):
            raise BrokenPipeError("learner CUDA context lost")

        local.run_trial = types.MethodType(fail_primary, local)
        runner = ProbeRunner(
            [local],
            process_workers=[self._RemoteWorker(events)],
        )

        with self.assertRaises(ElasticGPUFailure) as raised:
            runner.run_trials(k=2, base_seed=41)

        self.assertEqual(raised.exception.role, "learner-primary")
        self.assertEqual(runner.pool_generation, 0)

    def test_duplicate_trial_identity_fails_closed(self):
        events = []
        remote = self._RemoteWorker(events, duplicate_result=True)
        runner = ProbeRunner(
            [self._LocalWorker(events)],
            process_workers=[remote],
        )

        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            runner.run_trials(k=2, base_seed=41)

        self.assertEqual(runner.pool_generation, 0)
        self.assertFalse(remote.closed)

    def test_close_reaps_remote_worker(self):
        events = []
        runner, remote = self._runner(events)

        runner.close()

        self.assertTrue(remote.closed)
        self.assertEqual(events, [("remote-close", None)])

    def test_pending_child_close_terminates_before_first_join(self):
        events = []

        class _FakeProcess:
            def __init__(self):
                self.alive = True

            def is_alive(self):
                return self.alive

            def join(self, timeout=None):
                events.append(("join", timeout))

            def terminate(self):
                events.append(("terminate", None))
                self.alive = False

        class _FakeConnection:
            def __init__(self):
                self.closed = False

            def send(self, message):
                events.append(("send", message.get("operation")))

            def close(self):
                self.closed = True
                events.append(("connection-close", None))

        process = _FakeProcess()
        connection = _FakeConnection()
        worker = _probe_runner._ProcessProbeWorker(
            device=torch.device("cuda:1"),
            connection=connection,
            process=process,
        )
        worker._pending_operation = "register_batch_set"

        worker.close()

        event_names = [event[0] for event in events]
        self.assertLess(
            event_names.index("terminate"), event_names.index("join")
        )
        self.assertFalse(process.is_alive())
        self.assertTrue(connection.closed)

    def test_pending_child_close_kills_if_terminate_does_not_stop_process(self):
        events = []

        class _FakeProcess:
            def __init__(self):
                self.alive = True

            def is_alive(self):
                return self.alive

            def join(self, timeout=None):
                events.append(("join", timeout))

            def terminate(self):
                events.append(("terminate", None))

            def kill(self):
                events.append(("kill", None))
                self.alive = False

        class _FakeConnection:
            def __init__(self):
                self.closed = False

            def send(self, message):
                events.append(("send", message.get("operation")))

            def close(self):
                self.closed = True
                events.append(("connection-close", None))

        process = _FakeProcess()
        connection = _FakeConnection()
        worker = _probe_runner._ProcessProbeWorker(
            device=torch.device("cuda:1"),
            connection=connection,
            process=process,
        )
        worker._pending_operation = "register_batch_set"

        worker.close()

        lifecycle = [
            event_name
            for event_name, _payload in events
            if event_name in {"terminate", "join", "kill"}
        ]
        self.assertEqual(
            lifecycle, ["terminate", "join", "kill", "join"]
        )
        self.assertFalse(process.is_alive())
        self.assertTrue(connection.closed)

    def test_pending_close_uses_os_sigkill_without_process_kill_method(self):
        events = []

        class _FakeProcess:
            def __init__(self):
                self.pid = 4321
                self.alive = True

            def is_alive(self):
                return self.alive

            def join(self, timeout=None):
                events.append(("join", timeout))

            def terminate(self):
                events.append(("terminate", None))

        class _FakeConnection:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True
                events.append(("connection-close", None))

        process = _FakeProcess()
        connection = _FakeConnection()
        worker = _probe_runner._ProcessProbeWorker(
            device=torch.device("cuda:1"),
            connection=connection,
            process=process,
        )
        worker._pending_operation = "register_batch_set"

        def fake_os_kill(pid, sig):
            events.append(("os-kill", pid, sig))
            process.alive = False

        with mock.patch.object(
                _probe_runner.os, "kill", side_effect=fake_os_kill,
                ) as os_kill:
            worker.close()

        os_kill.assert_called_once_with(process.pid, signal.SIGKILL)
        lifecycle = [
            event[0]
            for event in events
            if event[0] in {"terminate", "join", "os-kill"}
        ]
        self.assertEqual(
            lifecycle, ["terminate", "join", "os-kill", "join"]
        )
        self.assertFalse(process.is_alive())
        self.assertTrue(connection.closed)

    def test_graceful_close_kills_if_terminate_does_not_stop_process(self):
        events = []

        class _FakeProcess:
            def __init__(self):
                self.alive = True

            def is_alive(self):
                return self.alive

            def join(self, timeout=None):
                events.append(("join", timeout))

            def terminate(self):
                events.append(("terminate", None))

            def kill(self):
                events.append(("kill", None))
                self.alive = False

        class _FakeConnection:
            def __init__(self):
                self.closed = False

            def send(self, message):
                events.append(("send", message.get("operation")))

            def close(self):
                self.closed = True
                events.append(("connection-close", None))

        process = _FakeProcess()
        connection = _FakeConnection()
        worker = _probe_runner._ProcessProbeWorker(
            device=torch.device("cuda:1"),
            connection=connection,
            process=process,
        )

        def fail_close_receive(operation, timeout=None):
            events.append(("receive", operation, timeout))
            worker._pending_operation = None
            raise TimeoutError("close receive timeout marker")

        with mock.patch.object(
                worker, "receive", side_effect=fail_close_receive,
                ) as receive:
            worker.close()

        receive.assert_called_once_with("close", timeout=5.0)
        self.assertIn(("send", "close"), events)
        lifecycle = [
            event_name
            for event_name, *_payload in events
            if event_name in {"terminate", "join", "kill"}
        ]
        terminate_index = lifecycle.index("terminate")
        self.assertEqual(
            lifecycle[terminate_index:],
            ["terminate", "join", "kill", "join"],
        )
        self.assertFalse(process.is_alive())
        self.assertTrue(connection.closed)

    def test_graceful_send_failure_still_reaps_process(self):
        for error_type in (BrokenPipeError, OSError):
            with self.subTest(error_type=error_type.__name__):
                events = []

                class _FakeProcess:
                    def __init__(self):
                        self.alive = True

                    def is_alive(self):
                        return self.alive

                    def join(self, timeout=None):
                        events.append(("join", timeout))

                    def terminate(self):
                        events.append(("terminate", None))

                    def kill(self):
                        events.append(("kill", None))
                        self.alive = False

                class _FailingConnection:
                    def __init__(self):
                        self.closed = False

                    def send(self, message):
                        events.append(("send", message.get("operation")))
                        raise error_type("close send marker")

                    def close(self):
                        self.closed = True
                        events.append(("connection-close", None))

                process = _FakeProcess()
                connection = _FailingConnection()
                worker = _probe_runner._ProcessProbeWorker(
                    device=torch.device("cuda:1"),
                    connection=connection,
                    process=process,
                )

                worker.close()

                self.assertIn(("send", "close"), events)
                lifecycle = [
                    event[0]
                    for event in events
                    if event[0] in {"terminate", "join", "kill"}
                ]
                self.assertEqual(
                    lifecycle,
                    ["join", "terminate", "join", "kill", "join"],
                )
                self.assertFalse(process.is_alive())
                self.assertTrue(connection.closed)

    def test_owner_closes_all_remote_workers_concurrently(self):
        barrier = threading.Barrier(4)
        state_lock = threading.Lock()
        arrivals = []
        completions = []

        class _BarrierRemote:
            def __init__(self, device_id):
                self.device = torch.device(f"cuda:{device_id}")
                self.close_calls = 0

            def close(self):
                with state_lock:
                    self.close_calls += 1
                    arrivals.append(str(self.device))
                barrier.wait(timeout=2.0)
                with state_lock:
                    completions.append(str(self.device))

        remotes = [_BarrierRemote(device_id) for device_id in range(1, 5)]
        owner = ProbeRunner(
            [self._LocalWorker([])], process_workers=remotes,
        )

        owner.close()

        expected_devices = [f"cuda:{device_id}" for device_id in range(1, 5)]
        self.assertEqual(sorted(arrivals), expected_devices)
        self.assertEqual(sorted(completions), expected_devices)
        self.assertEqual([remote.close_calls for remote in remotes], [1] * 4)

    def test_view_rejects_an_unregistered_batch_set(self):
        owner, _remote = self._runner([])

        try:
            with self.assertRaises((KeyError, ValueError)):
                owner.view("F4")
        finally:
            owner.close()

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
        owner.register_batch_set("F4", ["validation-full"])

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

    def test_build_startup_failure_reaps_all_handles_via_shared_helper(self):
        created_handles = []

        class _FakeConnection:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        class _FakeProcess:
            def __init__(self, pid):
                self.pid = pid
                self.started = False

            def start(self):
                self.started = True

        class _FakeContext:
            def __init__(self):
                self.processes = []

            def Pipe(self, duplex=True):
                self.assert_duplex(duplex)
                return _FakeConnection(), _FakeConnection()

            def Process(self, **_kwargs):
                process = _FakeProcess(pid=1000 + len(self.processes))
                self.processes.append(process)
                return process

            @staticmethod
            def assert_duplex(duplex):
                if duplex is not True:
                    raise AssertionError("probe process pipe must be duplex")

        class _FakeHandle:
            def __init__(self, *, device, process):
                self.device = device
                self.process = process
                self.close_calls = 0

            def wait_until_ready(self):
                if str(self.device) == "cuda:2":
                    raise RuntimeError("startup marker")

            def close(self):
                self.close_calls += 1
                if str(self.device) == "cuda:2":
                    raise RuntimeError("close marker")

        def make_handle(*, device, connection, process):
            del connection
            handle = _FakeHandle(device=device, process=process)
            created_handles.append(handle)
            return handle

        fake_context = _FakeContext()
        primary_model = torch.nn.Linear(2, 2)
        primary_batch = types.SimpleNamespace(
            input_ids=torch.ones((1, 2), dtype=torch.long),
            attention_mask=torch.ones((1, 2), dtype=torch.long),
            labels=torch.zeros((1,), dtype=torch.long),
            token_type_ids=None,
        )
        real_close_helper = ProbeRunner._close_worker_handles

        with mock.patch.object(
                _probe_runner, "BLBNoiseRLBridge", object(),
                ), mock.patch.object(
                    _probe_runner, "resolve_probe_backend", return_value="process",
                ), mock.patch.object(
                    _probe_runner, "enable_cuda_reward_probe_fast_math",
                ), mock.patch.object(
                    _probe_runner.torch.cuda, "empty_cache",
                ), mock.patch.object(
                    _probe_runner.mp, "get_context", return_value=fake_context,
                ), mock.patch.object(
                    _probe_runner, "_ProcessProbeWorker", side_effect=make_handle,
                ), mock.patch.object(
                    ProbeRunner,
                    "_close_worker_handles",
                    wraps=real_close_helper,
                ) as close_helper:
            try:
                _probe_runner.build_probe_runner(
                    primary_model=primary_model,
                    primary_handler=object(),
                    primary_bridge=object(),
                    primary_probe_batches=[primary_batch],
                    layers_attribute="unused.layers",
                    is_regression=False,
                    device_ids=[0, 1, 2, 3, 4],
                    metric_profile="startup-failure",
                )
            except RuntimeError as exc:
                failure = exc
            else:
                self.fail("startup failure must propagate as RuntimeError")

        self.assertEqual(len(created_handles), 4)
        self.assertEqual(close_helper.call_count, 1)
        helper_handles = list(close_helper.call_args.args[0])
        self.assertEqual(helper_handles, created_handles)
        self.assertEqual(
            [handle.close_calls for handle in created_handles], [1] * 4
        )
        self.assertIn("failed to start persistent probe processes", str(failure))
        self.assertIn("startup marker", str(failure))

    def test_started_process_is_tracked_before_child_pipe_close(self):
        created_handles = []

        class _FakeParentConnection:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        class _FakeProcess:
            def __init__(self):
                self.pid = 2001
                self.started = False
                self.reaped = False

            def start(self):
                self.started = True

        class _FailingChildConnection:
            def __init__(self, context):
                self.context = context

            def close(self):
                if self.context.process is None:
                    raise AssertionError("child process was not constructed")
                if not self.context.process.started:
                    raise AssertionError("child process was not started")
                raise RuntimeError("child connection close marker")

        class _FakeContext:
            def __init__(self):
                self.process = None

            def Pipe(self, duplex=True):
                if duplex is not True:
                    raise AssertionError("probe process pipe must be duplex")
                return _FakeParentConnection(), _FailingChildConnection(self)

            def Process(self, **_kwargs):
                self.process = _FakeProcess()
                return self.process

        class _FakeHandle:
            def __init__(self, *, device, connection, process):
                self.device = device
                self.connection = connection
                self.process = process
                self.close_calls = 0

            def close(self):
                self.close_calls += 1
                self.process.reaped = True
                self.connection.close()

        def make_handle(*, device, connection, process):
            handle = _FakeHandle(
                device=device,
                connection=connection,
                process=process,
            )
            created_handles.append(handle)
            return handle

        fake_context = _FakeContext()
        primary_model = torch.nn.Linear(2, 2)
        primary_batch = types.SimpleNamespace(
            input_ids=torch.ones((1, 2), dtype=torch.long),
            attention_mask=torch.ones((1, 2), dtype=torch.long),
            labels=torch.zeros((1,), dtype=torch.long),
            token_type_ids=None,
        )
        real_close_helper = ProbeRunner._close_worker_handles

        with mock.patch.object(
                _probe_runner, "BLBNoiseRLBridge", object(),
                ), mock.patch.object(
                    _probe_runner, "resolve_probe_backend", return_value="process",
                ), mock.patch.object(
                    _probe_runner, "enable_cuda_reward_probe_fast_math",
                ), mock.patch.object(
                    _probe_runner.torch.cuda, "empty_cache",
                ), mock.patch.object(
                    _probe_runner.mp, "get_context", return_value=fake_context,
                ), mock.patch.object(
                    _probe_runner, "_ProcessProbeWorker", side_effect=make_handle,
                ), mock.patch.object(
                    ProbeRunner,
                    "_close_worker_handles",
                    wraps=real_close_helper,
                ) as close_helper:
            try:
                _probe_runner.build_probe_runner(
                    primary_model=primary_model,
                    primary_handler=object(),
                    primary_bridge=object(),
                    primary_probe_batches=[primary_batch],
                    layers_attribute="unused.layers",
                    is_regression=False,
                    device_ids=[0, 1],
                    metric_profile="start-tracking",
                )
            except RuntimeError as exc:
                failure = exc
            else:
                self.fail("child connection close failure must propagate")

        self.assertIsNotNone(fake_context.process)
        self.assertTrue(fake_context.process.started)
        self.assertEqual(len(created_handles), 1)
        self.assertEqual(close_helper.call_count, 1)
        helper_handles = list(close_helper.call_args.args[0])
        self.assertEqual(helper_handles, created_handles)
        self.assertEqual(created_handles[0].close_calls, 1)
        self.assertTrue(fake_context.process.reaped)
        self.assertIn("failed to start persistent probe processes", str(failure))
        self.assertIn("child connection close marker", str(failure))

    def test_hard_interrupts_reap_handles_without_wrapping(self):
        scenarios = (
            ("child-close", KeyboardInterrupt, 1),
            ("wait-ready", SystemExit, 2),
        )
        for failure_point, error_type, expected_handle_count in scenarios:
            with self.subTest(
                    failure_point=failure_point,
                    error_type=error_type.__name__,
                    ):
                created_handles = []

                class _FakeParentConnection:
                    def __init__(self):
                        self.closed = False

                    def close(self):
                        self.closed = True

                class _FakeChildConnection:
                    def __init__(self):
                        self.closed = False

                    def close(self):
                        self.closed = True
                        if failure_point == "child-close":
                            raise error_type(f"{failure_point} marker")

                class _FakeProcess:
                    def __init__(self, pid):
                        self.pid = pid
                        self.started = False
                        self.reaped = False

                    def start(self):
                        self.started = True

                class _FakeContext:
                    def __init__(self):
                        self.processes = []

                    def Pipe(self, duplex=True):
                        if duplex is not True:
                            raise AssertionError(
                                "probe process pipe must be duplex"
                            )
                        return _FakeParentConnection(), _FakeChildConnection()

                    def Process(self, **_kwargs):
                        process = _FakeProcess(
                            pid=3000 + len(self.processes)
                        )
                        self.processes.append(process)
                        return process

                class _FakeHandle:
                    def __init__(self, *, device, connection, process):
                        self.device = device
                        self.connection = connection
                        self.process = process
                        self.close_calls = 0

                    def wait_until_ready(self):
                        if (
                                failure_point == "wait-ready"
                                and str(self.device) == "cuda:2"
                        ):
                            raise error_type(f"{failure_point} marker")

                    def close(self):
                        self.close_calls += 1
                        self.process.reaped = True
                        self.connection.close()

                def make_handle(*, device, connection, process):
                    handle = _FakeHandle(
                        device=device,
                        connection=connection,
                        process=process,
                    )
                    created_handles.append(handle)
                    return handle

                fake_context = _FakeContext()
                primary_model = torch.nn.Linear(2, 2)
                primary_batch = types.SimpleNamespace(
                    input_ids=torch.ones((1, 2), dtype=torch.long),
                    attention_mask=torch.ones((1, 2), dtype=torch.long),
                    labels=torch.zeros((1,), dtype=torch.long),
                    token_type_ids=None,
                )
                real_close_helper = ProbeRunner._close_worker_handles

                with mock.patch.object(
                        _probe_runner, "BLBNoiseRLBridge", object(),
                        ), mock.patch.object(
                            _probe_runner,
                            "resolve_probe_backend",
                            return_value="process",
                        ), mock.patch.object(
                            _probe_runner,
                            "enable_cuda_reward_probe_fast_math",
                        ), mock.patch.object(
                            _probe_runner.torch.cuda, "empty_cache",
                        ), mock.patch.object(
                            _probe_runner.mp,
                            "get_context",
                            return_value=fake_context,
                        ), mock.patch.object(
                            _probe_runner,
                            "_ProcessProbeWorker",
                            side_effect=make_handle,
                        ), mock.patch.object(
                            ProbeRunner,
                            "_close_worker_handles",
                            wraps=real_close_helper,
                        ) as close_helper:
                    with self.assertRaises(error_type) as raised:
                        _probe_runner.build_probe_runner(
                            primary_model=primary_model,
                            primary_handler=object(),
                            primary_bridge=object(),
                            primary_probe_batches=[primary_batch],
                            layers_attribute="unused.layers",
                            is_regression=False,
                            device_ids=[0, 1, 2],
                            metric_profile="hard-interrupt",
                        )

                self.assertIs(type(raised.exception), error_type)
                self.assertIn(f"{failure_point} marker", str(raised.exception))
                self.assertEqual(
                    len(created_handles), expected_handle_count
                )
                self.assertEqual(close_helper.call_count, 1)
                helper_handles = list(close_helper.call_args.args[0])
                self.assertEqual(helper_handles, created_handles)
                self.assertEqual(
                    [handle.close_calls for handle in created_handles],
                    [1] * expected_handle_count,
                )
                self.assertTrue(
                    all(handle.process.reaped for handle in created_handles)
                )


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


@unittest.skipUnless(
    _GPU_INTEGRATION_READY,
    "requires BLB_STAGE2_RUN_GPU_INTEGRATION=1 and at least five CUDA devices",
)
class ProbeRunnerFiveGpuIntegrationTest(unittest.TestCase):
    @staticmethod
    def _probe_batch(device, offset):
        return types.SimpleNamespace(
            input_ids=torch.tensor(
                [[offset, offset + 1, offset + 2, offset + 3]],
                dtype=torch.long,
                device=device,
            ),
            attention_mask=torch.ones(
                (1, 4), dtype=torch.long, device=device,
            ),
            labels=torch.tensor([1], dtype=torch.long, device=device),
            token_type_ids=torch.zeros(
                (1, 4), dtype=torch.long, device=device,
            ),
        )

    def test_real_five_gpu_registration_failure_cleans_pending_children(self):
        primary_device = torch.device("cuda:0")
        primary_model = torch.nn.Linear(4, 2).to(primary_device).eval()
        primary_handler = _probe_runner.ReversibleLayerHandler(primary_model)
        primary_bridge = _probe_runner.BLBNoiseRLBridge(
            primary_handler, layers_attribute="unused.layers",
        )
        f1_batch = self._probe_batch(primary_device, offset=1)

        with mock.patch.dict(
                os.environ, {"BLB_STAGE2_PROBE_BACKEND": "process"},
                ):
            owner = _probe_runner.build_probe_runner(
                primary_model=primary_model,
                primary_handler=primary_handler,
                primary_bridge=primary_bridge,
                primary_probe_batches=[f1_batch],
                layers_attribute="unused.layers",
                is_regression=False,
                device_ids=[0, 1, 2, 3, 4],
                metric_profile="integration",
            )

        try:
            self.assertEqual(owner.backend, "process")
            self.assertEqual(owner.num_workers, 5)
            remotes = list(owner._process_workers)
            self.assertEqual(len(remotes), 4)
            child_pids = [remote.process.pid for remote in remotes]
            self.assertTrue(all(pid is not None for pid in child_pids))
            self.assertEqual(len(set(child_pids)), 4)
            self.assertTrue(all(remote.process.is_alive() for remote in remotes))

            f4_batch = self._probe_batch(primary_device, offset=11)
            owner.register_batch_set("F4", [f4_batch])
            self.assertEqual(f4_batch.input_ids.device, primary_device)

            failed_remote = remotes[1]
            self.assertEqual(str(failed_remote.device), "cuda:2")
            failed_remote.process.terminate()
            failed_remote.process.join(timeout=10.0)
            self.assertFalse(failed_remote.process.is_alive())

            f5_batch = self._probe_batch(primary_device, offset=21)
            started = time.monotonic()
            with self.assertRaisesRegex(RuntimeError, "register_batch_set"):
                owner.register_batch_set("F5", [f5_batch])
            cleanup_seconds = time.monotonic() - started

            self.assertLess(cleanup_seconds, 4.0)
            self.assertFalse(
                any(remote.process.is_alive() for remote in remotes)
            )
            with self.assertRaisesRegex(RuntimeError, "poison"):
                owner.view("F1")
        finally:
            owner.close()


if __name__ == "__main__":
    unittest.main()
