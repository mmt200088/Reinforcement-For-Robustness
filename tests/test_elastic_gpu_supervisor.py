import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from elastic_gpu import ELASTIC_GPU_RESTART_EXIT_CODE
from scripts.elastic_gpu_supervisor import (
    RecoveryMonitor,
    build_child_command,
    parse_nvidia_smi_csv,
    resolve_health_snapshot,
    run_child_foreground,
    run_supervised,
)


GPU_CSV = """\
0, GPU-a, None
1, GPU-b, None
2, GPU-c, None
3, GPU-d, Reset
4, GPU-e, None
"""

ALL_HEALTHY_GPU_CSV = """\
0, GPU-a, None
1, GPU-b, None
2, GPU-c, None
3, GPU-d, None
4, GPU-e, None
"""


class HealthResolverTest(unittest.TestCase):
    def test_all_five_healthy_devices_are_preserved_in_candidate_order(self):
        resolved = resolve_health_snapshot(
            parse_nvidia_smi_csv(ALL_HEALTHY_GPU_CSV),
            candidate_tokens=["0", "1", "2", "3", "4"],
        )

        self.assertEqual(resolved.healthy_tokens, ("0", "1", "2", "3", "4"))
        self.assertEqual(resolved.quarantined_tokens, ())
        self.assertEqual(resolved.logical_device_spec, "0,1,2,3,4")
        self.assertEqual(
            resolved.healthy_visibility_tokens,
            ("GPU-a", "GPU-b", "GPU-c", "GPU-d", "GPU-e"),
        )

    def test_reset_device_is_removed_and_logical_ids_are_dense(self):
        resolved = resolve_health_snapshot(
            parse_nvidia_smi_csv(GPU_CSV),
            candidate_tokens=["0", "1", "2", "3", "4"],
        )

        self.assertEqual(resolved.healthy_tokens, ("0", "1", "2", "4"))
        self.assertEqual(resolved.quarantined_tokens, ("3",))
        self.assertEqual(resolved.logical_device_spec, "0,1,2,3")
        self.assertEqual(
            resolved.healthy_visibility_tokens,
            ("GPU-a", "GPU-b", "GPU-c", "GPU-e"),
        )

    def test_explicit_index_subset_preserves_requested_order(self):
        resolved = resolve_health_snapshot(
            parse_nvidia_smi_csv(GPU_CSV),
            candidate_tokens=["4", "3", "1"],
        )

        self.assertEqual(resolved.healthy_tokens, ("4", "1"))
        self.assertEqual(resolved.quarantined_tokens, ("3",))
        self.assertEqual(resolved.logical_device_spec, "0,1")

    def test_uuid_subset_is_supported_without_changing_token_form(self):
        resolved = resolve_health_snapshot(
            parse_nvidia_smi_csv(GPU_CSV),
            candidate_tokens=["GPU-e", "GPU-d", "GPU-a"],
        )

        self.assertEqual(
            resolved.healthy_tokens,
            ("GPU-e", "GPU-a"),
        )
        self.assertEqual(resolved.quarantined_tokens, ("GPU-d",))

    def test_unknown_and_all_unhealthy_candidates_fail_closed(self):
        records = parse_nvidia_smi_csv(GPU_CSV)
        with self.assertRaisesRegex(ValueError, "not present"):
            resolve_health_snapshot(records, candidate_tokens=["GPU-missing"])
        with self.assertRaisesRegex(RuntimeError, "no healthy GPU"):
            resolve_health_snapshot(records, candidate_tokens=["3"])


class ChildCommandTest(unittest.TestCase):
    def test_script_entrypoint_imports_project_modules(self):
        repo_root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts" / "elastic_gpu_supervisor.py"),
                "--help",
            ],
            cwd="/",
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_foreground_child_preserves_inherited_stage2_lock_fd(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock_path = Path(tmp) / "stage2.lock"
            lock_fd = os.open(
                lock_path,
                os.O_CREAT | os.O_APPEND | os.O_RDWR,
                0o600,
            )
            try:
                env = os.environ.copy()
                env["BLB_STAGE2_RUN_LOCK_FD"] = str(lock_fd)
                child_code = (
                    "import os, sys; "
                    "fd = int(os.environ['BLB_STAGE2_RUN_LOCK_FD']); "
                    "held = os.fstat(fd); expected = os.stat(sys.argv[1]); "
                    "sys.exit(0 if (held.st_dev, held.st_ino) == "
                    "(expected.st_dev, expected.st_ino) else 7)"
                )

                completed = run_child_foreground(
                    [sys.executable, "-c", child_code, str(lock_path)],
                    env=env,
                    check=False,
                )
            finally:
                os.close(lock_fd)

        self.assertEqual(completed.returncode, 0)

    def test_auto_device_flags_are_rewritten_to_dense_logical_ids(self):
        command = [
            "python",
            "rl_tune.py",
            "--stage1_rl_devices",
            "auto",
            "--blb_v3_reward_devices=auto",
            "--unrelated",
            "auto",
        ]

        rewritten = build_child_command(
            command,
            logical_device_spec="0,1,2,3",
            resume_run_dir=None,
        )

        self.assertEqual(
            rewritten,
            [
                "python",
                "rl_tune.py",
                "--stage1_rl_devices",
                "0,1,2,3",
                "--blb_v3_reward_devices=0,1,2,3",
                "--unrelated",
                "auto",
            ],
        )

    def test_restart_replaces_or_appends_resume_run_dir(self):
        replaced = build_child_command(
            ["python", "rl_tune.py", "--resume_run_dir", "old"],
            logical_device_spec="0",
            resume_run_dir="/run/current",
        )
        appended = build_child_command(
            ["python", "rl_tune.py"],
            logical_device_spec="0",
            resume_run_dir="/run/current",
        )

        self.assertEqual(replaced[-2:], ["--resume_run_dir", "/run/current"])
        self.assertEqual(appended[-2:], ["--resume_run_dir", "/run/current"])
        self.assertNotIn("old", replaced)

    def test_explicit_devices_follow_the_compressed_visibility_map(self):
        rewritten = build_child_command(
            [
                "python",
                "rl_tune.py",
                "--blb_v3_reward_devices",
                "2,4",
            ],
            logical_device_spec="0,1,2,3",
            resume_run_dir=None,
            original_to_current_logical={
                "0": "0",
                "1": "1",
                "2": "2",
                "4": "3",
            },
        )

        self.assertEqual(rewritten[-1], "2,3")

    def test_explicit_devices_drop_only_quarantined_members(self):
        rewritten = build_child_command(
            [
                "python",
                "rl_tune.py",
                "--blb_v3_reward_devices",
                "2,4",
            ],
            logical_device_spec="0,1,2",
            resume_run_dir=None,
            original_to_current_logical={
                "0": "0",
                "1": "1",
                "4": "2",
            },
        )

        self.assertEqual(rewritten[-1], "2")


class RecoveryMonitorTest(unittest.TestCase):
    def test_only_quarantined_eligible_devices_reach_canary(self):
        records = parse_nvidia_smi_csv(GPU_CSV)
        canary_calls = []
        recovered = []
        monitor = RecoveryMonitor(
            quarantined_tokens=("3", "4"),
            query_records=lambda: records,
            canary=lambda token: canary_calls.append(token) or True,
            on_recovered=lambda tokens: recovered.extend(tokens),
            interval_seconds=60.0,
        )

        monitor.poll_once()

        self.assertEqual(canary_calls, ["GPU-e"])
        self.assertEqual(recovered, ["4"])


class SupervisorRestartTest(unittest.TestCase):
    def test_reserved_exit_quarantines_failed_device_and_resumes(self):
        records = parse_nvidia_smi_csv(GPU_CSV)
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            logs = run_dir / "logs"
            logs.mkdir()
            launches = []
            results = iter(
                [
                    subprocess.CompletedProcess([], ELASTIC_GPU_RESTART_EXIT_CODE),
                    subprocess.CompletedProcess([], 0),
                ]
            )

            def fake_run(command, *, env, check):
                launches.append((list(command), dict(env), check))
                result = next(results)
                if result.returncode == ELASTIC_GPU_RESTART_EXIT_CODE:
                    (logs / "elastic_gpu_failure.json").write_text(
                        '{"record_type":"elastic_gpu_failure_v1",'
                        '"physical_device":"GPU-b"}\n',
                        encoding="utf-8",
                    )
                return result

            rc = run_supervised(
                child_command=[
                    "python",
                    "rl_tune.py",
                    "--blb_v3_reward_devices",
                    "auto",
                ],
                run_dir=run_dir,
                candidate_tokens=("0", "1", "2", "4"),
                query_records=lambda: records,
                process_runner=fake_run,
                max_restarts=2,
                recovery_interval=0.0,
            )

        self.assertEqual(rc, 0)
        self.assertEqual(len(launches), 2)
        self.assertEqual(
            launches[0][1]["CUDA_VISIBLE_DEVICES"],
            "GPU-a,GPU-b,GPU-c,GPU-e",
        )
        self.assertEqual(
            launches[1][1]["CUDA_VISIBLE_DEVICES"],
            "GPU-a,GPU-c,GPU-e",
        )
        self.assertEqual(
            launches[1][0][-2:],
            ["--resume_run_dir", str(run_dir)],
        )

    def test_restart_budget_is_bounded(self):
        records = parse_nvidia_smi_csv(GPU_CSV)
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            logs = run_dir / "logs"
            logs.mkdir()

            def always_restart(command, *, env, check):
                (logs / "elastic_gpu_failure.json").write_text(
                    '{"record_type":"elastic_gpu_failure_v1",'
                    '"physical_device":"1"}\n',
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(
                    command,
                    ELASTIC_GPU_RESTART_EXIT_CODE,
                )

            rc = run_supervised(
                child_command=["python", "rl_tune.py"],
                run_dir=run_dir,
                candidate_tokens=("0", "1", "2"),
                query_records=lambda: records,
                process_runner=always_restart,
                max_restarts=1,
                recovery_interval=0.0,
            )

        self.assertEqual(rc, ELASTIC_GPU_RESTART_EXIT_CODE)

    def test_non_reserved_exit_is_never_restarted(self):
        records = parse_nvidia_smi_csv(GPU_CSV)
        launches = []

        def fail_once(command, *, env, check):
            launches.append(list(command))
            return subprocess.CompletedProcess(command, 2)

        with tempfile.TemporaryDirectory() as tmp:
            rc = run_supervised(
                child_command=["python", "rl_tune.py"],
                run_dir=Path(tmp),
                candidate_tokens=("0", "1"),
                query_records=lambda: records,
                process_runner=fail_once,
                max_restarts=8,
                recovery_interval=0.0,
            )

        self.assertEqual(rc, 2)
        self.assertEqual(len(launches), 1)


if __name__ == "__main__":
    unittest.main()
