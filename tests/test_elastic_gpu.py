from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from elastic_gpu import (
    ELASTIC_GPU_RESTART_EXIT_CODE,
    ElasticGPUFailure,
    ElasticGPURestartRequested,
    consume_elastic_gpu_restart_request,
    is_recoverable_gpu_failure,
    physical_token_for_logical_device,
    raise_if_elastic_gpu_restart_requested,
    request_elastic_gpu_restart,
)
from rfr.common.runtime_error_reporter import run_fire_entrypoint


class ElasticGPUFailureContractTest(unittest.TestCase):
    def test_transport_and_device_loss_failures_are_recoverable(self):
        recoverable = (
            BrokenPipeError("pipe closed"),
            EOFError("worker pipe ended"),
            TimeoutError("probe child timed out after 3600.0s"),
            RuntimeError("CUDA error: unspecified launch failure"),
            RuntimeError("GPU requires reset"),
            RuntimeError("CUDA-capable device is busy or unavailable"),
            RuntimeError("probe child cuda:2 exited with code 9"),
        )

        for error in recoverable:
            with self.subTest(error=repr(error)):
                self.assertTrue(is_recoverable_gpu_failure(error))

    def test_scientific_and_capacity_failures_are_not_recoverable(self):
        fatal = (
            RuntimeError("mat1 and mat2 shapes cannot be multiplied"),
            RuntimeError("CUDA error: device-side assert triggered"),
            RuntimeError("CUDA out of memory"),
            ValueError("trial seed mismatch"),
            AssertionError("candidate identity changed"),
        )

        for error in fatal:
            with self.subTest(error=repr(error)):
                self.assertFalse(is_recoverable_gpu_failure(error))

    def test_logical_device_maps_through_visibility(self):
        self.assertEqual(
            physical_token_for_logical_device("cuda:2", "0,1,4,7"),
            "4",
        )
        self.assertEqual(
            physical_token_for_logical_device(
                "cuda:1",
                "GPU-a,GPU-b,GPU-c",
            ),
            "GPU-b",
        )

    def test_mapping_rejects_cpu_malformed_and_out_of_range_devices(self):
        for device in ("cpu", "cuda", "cuda:x", "cuda:4"):
            with self.subTest(device=device):
                with self.assertRaises(ValueError):
                    physical_token_for_logical_device(device, "0,1,2,3")

    def test_failure_record_contains_physical_device_mapping(self):
        failure = ElasticGPUFailure(
            device="cuda:2",
            role="probe-replica",
            operation="run_action_trial_groups",
            cause=BrokenPipeError("pipe closed"),
            cuda_visible_devices="0,1,4,7",
        )

        self.assertEqual(failure.physical_device, "4")
        self.assertEqual(failure.device, "cuda:2")
        self.assertEqual(failure.role, "probe-replica")
        self.assertEqual(failure.operation, "run_action_trial_groups")
        self.assertIn("pipe closed", str(failure))


class ElasticGPURestartRequestTest(unittest.TestCase):
    def test_request_is_atomic_consumable_and_raises_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            request_path = Path(tmp) / "restart-request.json"
            with mock.patch.dict(
                os.environ,
                {"RFR_ELASTIC_GPU_RESTART_REQUEST": str(request_path)},
                clear=False,
            ):
                request_elastic_gpu_restart(
                    reason="recovered_device",
                    physical_devices=["3"],
                )
                payload = json.loads(request_path.read_text(encoding="utf-8"))
                self.assertEqual(payload["reason"], "recovered_device")
                self.assertEqual(payload["physical_devices"], ["3"])

                with self.assertRaises(ElasticGPURestartRequested) as raised:
                    raise_if_elastic_gpu_restart_requested()
                self.assertEqual(raised.exception.reason, "recovered_device")
                self.assertEqual(raised.exception.physical_devices, ("3",))
                self.assertFalse(request_path.exists())

                self.assertIsNone(consume_elastic_gpu_restart_request())
                raise_if_elastic_gpu_restart_requested()

    def test_final_boundary_consumes_request_without_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            request_path = Path(tmp) / "restart-request.json"
            with mock.patch.dict(
                os.environ,
                {"RFR_ELASTIC_GPU_RESTART_REQUEST": str(request_path)},
                clear=False,
            ):
                request_elastic_gpu_restart(
                    reason="recovered_device",
                    physical_devices=["4"],
                )

                raise_if_elastic_gpu_restart_requested(work_remaining=False)

                self.assertFalse(request_path.exists())
                self.assertIsNone(consume_elastic_gpu_restart_request())


class RuntimeErrorReporterElasticExitTest(unittest.TestCase):
    class _Fire:
        error = None

        @classmethod
        def Fire(cls, _target):
            raise cls.error

    def test_typed_gpu_failure_uses_reserved_exit_and_writes_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            self._Fire.error = ElasticGPUFailure(
                device="cuda:1",
                role="learner-primary",
                operation="policy_forward",
                cause=RuntimeError("GPU requires reset"),
                cuda_visible_devices="0,4",
            )
            argv = [
                "rl_tune.py",
                "--output_dir",
                str(output_dir),
            ]
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit) as raised:
                    run_fire_entrypoint(
                        self._Fire,
                        object(),
                        program_name="rl_tune.py",
                    )

            self.assertEqual(
                raised.exception.code,
                ELASTIC_GPU_RESTART_EXIT_CODE,
            )
            record_path = output_dir / "logs" / "elastic_gpu_failure.json"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            self.assertEqual(record["device"], "cuda:1")
            self.assertEqual(record["physical_device"], "4")
            self.assertEqual(record["role"], "learner-primary")
            self.assertEqual(record["operation"], "policy_forward")
            self.assertEqual(record["exit_code"], ELASTIC_GPU_RESTART_EXIT_CODE)

    def test_generic_failure_keeps_existing_exception_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._Fire.error = RuntimeError("metric contract marker")
            argv = ["rl_tune.py", "--output_dir", tmp]
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "metric contract marker",
                ):
                    run_fire_entrypoint(
                        self._Fire,
                        object(),
                        program_name="rl_tune.py",
                    )


if __name__ == "__main__":
    unittest.main()
