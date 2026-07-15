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

        def install(self, decoded):
            self.events.append(("local-install", decoded))

        def clear(self):
            self.events.append(("local-clear", None))

        def run_trial(self, trial_idx, base_seed):
            self.events.append(("local-run", trial_idx, base_seed))
            return (float(trial_idx), float(base_seed), -1.0)

    class _RemoteWorker:
        def __init__(self, events, *, fail_receive=False):
            self.device = torch.device("cuda:1")
            self.events = events
            self.fail_receive = fail_receive
            self.pending = None
            self.closed = False

        def submit(self, operation, payload):
            self.pending = (operation, payload)
            self.events.append(("remote-submit", operation, payload))

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
            self.events.append(("remote-close", None))

    def _runner(self, events, *, fail_receive=False):
        local = self._LocalWorker(events)
        remote = self._RemoteWorker(events, fail_receive=fail_receive)
        return ProbeRunner([local], process_workers=[remote]), remote

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
