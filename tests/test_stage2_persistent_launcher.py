from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]


class Stage2PersistentLauncherTest(unittest.TestCase):
    def test_stage2_launcher_warns_when_visible_gpus_are_not_forwarded_to_gpu_flags(self):
        with tempfile.TemporaryDirectory(prefix="stage2_gpu_audit_") as td:
            tmp = Path(td)
            capture = tmp / "python_argv.nul"
            fakebin = tmp / "fakebin"
            fakebin.mkdir()
            fake_python = fakebin / "python"
            fake_python.write_text(
                textwrap.dedent(
                    f"""\
                    #!/usr/bin/env bash
                    printf '%s\\0' "$@" > {str(capture)!r}
                    exit 0
                    """
                ),
                encoding="utf-8",
            )
            fake_python.chmod(0o755)

            env = os.environ.copy()
            env["PATH"] = f"{fakebin}{os.pathsep}{env.get('PATH', '')}"
            env["CUDA_VISIBLE_DEVICES"] = "0,1"

            cmd = [
                "bash",
                "llama_7B_LayerImportance.sh",
                "run",
                "rl",
                "--preset",
                "mrpc-blb-stage2-rl",
                "--mode",
                "stage2-only",
                "--persistent-root",
                str(tmp / "persistent"),
                "--stage2-search-episodes",
                "170",
                "--stage2-fixed-config-source",
                "json",
                "--stage2-fixed-config",
                "glue_final_configs_best_ppo.json",
                "--fresh",
            ]
            result = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(
                result.returncode,
                0,
                msg=result.stdout + "\n" + result.stderr,
            )
            combined = result.stdout + "\n" + result.stderr
            self.assertIn("[gpu-audit][WARN]", combined)
            self.assertIn("--blb-v3-reward-devices 0,1", combined)

    def test_stage2_rl_launches_inside_constraint_persistent_dir(self):
        with tempfile.TemporaryDirectory(prefix="stage2_persist_launcher_") as td:
            tmp = Path(td)
            capture = tmp / "python_argv.nul"
            fakebin = tmp / "fakebin"
            fakebin.mkdir()
            fake_python = fakebin / "python"
            fake_python.write_text(
                textwrap.dedent(
                    f"""\
                    #!/usr/bin/env bash
                    printf '%s\\0' "$@" > {str(capture)!r}
                    exit 0
                    """
                ),
                encoding="utf-8",
            )
            fake_python.chmod(0o755)

            persistent_root = tmp / "persistent"
            expected_output_dir = (
                persistent_root
                / "rl"
                / "bert-base"
                / "mrpc"
                / "s1t0.001_s2t0.001_s2st3.0"
            )

            env = os.environ.copy()
            env["PATH"] = f"{fakebin}{os.pathsep}{env.get('PATH', '')}"
            env["CUDA_VISIBLE_DEVICES"] = ""
            env["CAPTURE"] = str(capture)

            cmd = [
                "bash",
                "llama_7B_LayerImportance.sh",
                "run",
                "rl",
                "--preset",
                "mrpc-blb-stage2-rl",
                "--mode",
                "stage2-only",
                "--persistent-root",
                str(persistent_root),
                "--stage1-accuracy-tolerance",
                "0.001",
                "--stage2-limit-tolerance",
                "0.001",
                "--stage2-stability-tolerance",
                "3.0",
                "--stage2-search-episodes",
                "170",
                "--stage2-fixed-config-source",
                "json",
                "--stage2-fixed-config",
                "glue_final_configs_best_ppo.json",
                "--fresh",
            ]
            result = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(
                result.returncode,
                0,
                msg=result.stdout + "\n" + result.stderr,
            )
            for _ in range(50):
                if capture.is_file():
                    break
                import time
                time.sleep(0.1)
            self.assertTrue(capture.is_file(), msg="launcher did not invoke python")
            argv = [
                part.decode("utf-8")
                for part in capture.read_bytes().split(b"\0")
                if part
            ]
            latest_run_dir = expected_output_dir.parent / "LATEST_RUN_DIR"
            self.assertTrue(latest_run_dir.is_file())
            self.assertEqual(latest_run_dir.read_text(encoding="utf-8").strip(), str(expected_output_dir))

        self.assertIn("--output_dir", argv)
        self.assertEqual(argv[argv.index("--output_dir") + 1], str(expected_output_dir))
        self.assertIn("--decoupled_layout", argv)
        self.assertEqual(argv[argv.index("--decoupled_layout") + 1], "false")

    def test_stage2_run_tag_creates_separate_persistent_slug(self):
        with tempfile.TemporaryDirectory(prefix="stage2_persist_tag_") as td:
            tmp = Path(td)
            capture = tmp / "python_argv.nul"
            fakebin = tmp / "fakebin"
            fakebin.mkdir()
            fake_python = fakebin / "python"
            fake_python.write_text(
                textwrap.dedent(
                    f"""\
                    #!/usr/bin/env bash
                    printf '%s\\0' "$@" > {str(capture)!r}
                    exit 0
                    """
                ),
                encoding="utf-8",
            )
            fake_python.chmod(0o755)

            persistent_root = tmp / "persistent"
            expected_output_dir = (
                persistent_root
                / "rl"
                / "bert-base"
                / "mrpc"
                / "s1t0.001_s2t0.001_s2st3.0__gate_gN_20260625"
            )

            env = os.environ.copy()
            env["PATH"] = f"{fakebin}{os.pathsep}{env.get('PATH', '')}"
            env["CUDA_VISIBLE_DEVICES"] = ""
            env["CAPTURE"] = str(capture)

            cmd = [
                "bash",
                "llama_7B_LayerImportance.sh",
                "run",
                "rl",
                "--preset",
                "mrpc-blb-stage2-rl",
                "--mode",
                "stage2-only",
                "--persistent-root",
                str(persistent_root),
                "--run-tag",
                "gate_gN_20260625",
                "--stage1-accuracy-tolerance",
                "0.001",
                "--stage2-limit-tolerance",
                "0.001",
                "--stage2-stability-tolerance",
                "3.0",
                "--stage2-search-episodes",
                "170",
                "--stage2-fixed-config-source",
                "json",
                "--stage2-fixed-config",
                "glue_final_configs_best_ppo.json",
                "--fresh",
            ]
            result = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(
                result.returncode,
                0,
                msg=result.stdout + "\n" + result.stderr,
            )
            for _ in range(50):
                if capture.is_file():
                    break
                import time
                time.sleep(0.1)
            self.assertTrue(capture.is_file(), msg="launcher did not invoke python")
            argv = [
                part.decode("utf-8")
                for part in capture.read_bytes().split(b"\0")
                if part
            ]

        self.assertIn("--output_dir", argv)
        self.assertEqual(argv[argv.index("--output_dir") + 1], str(expected_output_dir))

    def test_stage2_preset_defaults_to_current_formal_constraints(self):
        with tempfile.TemporaryDirectory(prefix="stage2_persist_preset_") as td:
            tmp = Path(td)
            capture = tmp / "python_argv.nul"
            fakebin = tmp / "fakebin"
            fakebin.mkdir()
            fake_python = fakebin / "python"
            fake_python.write_text(
                textwrap.dedent(
                    f"""\
                    #!/usr/bin/env bash
                    printf '%s\\0' "$@" > {str(capture)!r}
                    exit 0
                    """
                ),
                encoding="utf-8",
            )
            fake_python.chmod(0o755)

            persistent_root = tmp / "persistent"
            expected_output_dir = (
                persistent_root
                / "rl"
                / "bert-base"
                / "mrpc"
                / "s1t0.001_s2t0.001_s2st3.0"
            )

            env = os.environ.copy()
            env["PATH"] = f"{fakebin}{os.pathsep}{env.get('PATH', '')}"
            env["CUDA_VISIBLE_DEVICES"] = ""

            cmd = [
                "bash",
                "llama_7B_LayerImportance.sh",
                "run",
                "rl",
                "--preset",
                "mrpc-blb-stage2-rl",
                "--persistent-root",
                str(persistent_root),
                "--stage2-search-episodes",
                "170",
                "--stage2-fixed-config-source",
                "json",
                "--stage2-fixed-config",
                "glue_final_configs_best_ppo.json",
                "--fresh",
            ]
            result = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(
                result.returncode,
                0,
                msg=result.stdout + "\n" + result.stderr,
            )
            for _ in range(50):
                if capture.is_file():
                    break
                import time
                time.sleep(0.1)
            self.assertTrue(capture.is_file(), msg="launcher did not invoke python")
            argv = [
                part.decode("utf-8")
                for part in capture.read_bytes().split(b"\0")
                if part
            ]

        self.assertEqual(argv[argv.index("--output_dir") + 1], str(expected_output_dir))
        self.assertEqual(argv[argv.index("--stage1_accuracy_tolerance") + 1], "0.001")
        self.assertEqual(argv[argv.index("--stage2_limit_tolerance") + 1], "0.001")
        self.assertEqual(argv[argv.index("--stage2_stability_tolerance") + 1], "3.0")

    def test_server_command_short_rl_runs_use_tagged_persistent_dirs(self):
        source = (REPO_ROOT / "SERVER_COMMAND.md").read_text(encoding="utf-8")
        self.assertIn("--run-tag \"ab_${tag}_${TS}\"", source)
        self.assertIn("--run-tag \"gate_${tag}_${TS}\"", source)
        self.assertIn("${tag}_persistent_verify.txt", source)
        self.assertIn("--min-episodes \"$EPISODES_AB\"", source)

    def test_stage2_training_loop_flushes_status_on_ppo_update(self):
        source = (REPO_ROOT / "blb_stage2_rl" / "sequential_runner.py").read_text(
            encoding="utf-8",
        )
        callback_start = source.index("def _ppo_update_end_callback(")
        callback_end = source.index("    t_start = time.time()", callback_start)
        callback_source = source[callback_start:callback_end]
        self.assertIn("status.update_after_ppo_update", callback_source)

    def test_stage2_training_loop_refreshes_live_curves_after_ppo_update(self):
        source = (REPO_ROOT / "blb_stage2_rl" / "sequential_runner.py").read_text(
            encoding="utf-8",
        )
        callback_start = source.index("def _ppo_update_end_callback(")
        callback_end = source.index("    t_start = time.time()", callback_start)
        callback_source = source[callback_start:callback_end]
        self.assertIn("_write_live_training_curves", callback_source)
        self.assertIn("live_curve_refresh", source)
        self.assertIn("write_training_curves(", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
