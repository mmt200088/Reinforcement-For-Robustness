import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest


_REPO = pathlib.Path(__file__).resolve().parents[1]


class Stage2NgpuSpeedTargetedFirstTests(unittest.TestCase):
    def test_targeted_first_defaults_to_validated_fast_candidate_only(self):
        script = (_REPO / "scripts" / "stage2_ngpu_speed_targeted_first.sh").read_text(
            encoding="utf-8"
        )
        expected = (
            'TARGET_CANDIDATES="${TARGET_CANDIDATES:-'
            '1:worker:1}"'
        )

        self.assertIn(expected, script)
        self.assertIn('TARGET_MIN_SPEEDUP="${TARGET_MIN_SPEEDUP:-3.4}"', script)
        self.assertIn('FALLBACK_AUTOTUNE="${FALLBACK_AUTOTUNE:-0}"', script)
        self.assertIn('SWEEP_MANY_WPD_LIST="${SWEEP_MANY_WPD_LIST:-1}"', script)

    def test_ab_and_grid_defaults_use_validated_fast_config(self):
        ab = (_REPO / "scripts" / "stage2_ngpu_speed_ab.sh").read_text(
            encoding="utf-8"
        )
        sweep = (_REPO / "scripts" / "stage2_ngpu_speed_sweep.sh").read_text(
            encoding="utf-8"
        )
        autotune = (_REPO / "scripts" / "stage2_ngpu_speed_autotune.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn('MANY_WORKERS_PER_DEVICE="${MANY_WORKERS_PER_DEVICE:-1}"', ab)
        self.assertIn('POLICY_DEVICE="${POLICY_DEVICE:-worker}"', ab)
        self.assertIn('DYNAMIC_ASSIGNMENT="${DYNAMIC_ASSIGNMENT:-1}"', ab)
        self.assertIn('MIN_SPEEDUP="${MIN_SPEEDUP:-3.4}"', ab)
        for script in (sweep, autotune):
            self.assertIn('SWEEP_MANY_WPD_LIST="${SWEEP_MANY_WPD_LIST:-1}"', script)
            self.assertIn(
                'SWEEP_POLICY_DEVICE_LIST="${SWEEP_POLICY_DEVICE_LIST:-worker}"',
                script,
            )
            self.assertIn('SWEEP_DYNAMIC_LIST="${SWEEP_DYNAMIC_LIST:-1}"', script)

    def test_server_command_defaults_ngpu_gate_and_60k_to_one_worker_per_device(self):
        command = (_REPO / "SERVER_COMMAND.md").read_text(encoding="utf-8")

        self.assertIn('run_gate gN "$DEVS" "$DEVS" 1', command)
        self.assertIn("--stage2-workers-per-device 1", command)
        self.assertNotIn('run_gate gN "$DEVS" "$DEVS" 2', command)

    def test_ab_runner_prefers_run_local_stage2_artifacts_before_canonical_fallback(self):
        script = (_REPO / "scripts" / "stage2_ngpu_speed_ab.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn('RUN_STAGE2="${RUN_STAGE2:-${ARTIFACT_DIR}/stage2}"', script)
        self.assertIn('"${RUN_STAGE2}/LATEST_PID"', script)
        self.assertIn('"${RUN_STAGE2}/bert base mrpc/run.pid"', script)
        self.assertIn('cat "${RUN_STAGE2}/LATEST_RUN_DIR"', script)
        self.assertIn(
            'STAGE1_RECORD_SOURCE="${STAGE1_RECORD_SOURCE:-Parting Chapter/stage1/record}"',
            script,
        )
        self.assertIn('ln -s "$source_abs" "$target"', script)

    def test_ab_runner_requires_sampled_activity_on_every_requested_gpu(self):
        script = (_REPO / "scripts" / "stage2_ngpu_speed_ab.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("scripts/gpu_utilization_report.py", script)
        self.assertIn('--nvidia-smi-csv "$gpu_sample_file"', script)
        self.assertIn('--visible-devices "$devices"', script)
        self.assertIn("--require-all-visible-sampled-active", script)

    def test_ab_prints_layerwise_production_commands_without_touching_gpus(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            nvidia_smi_marker = root / "nvidia-smi-called"
            fake_nvidia_smi = fake_bin / "nvidia-smi"
            fake_nvidia_smi.write_text(
                "#!/usr/bin/env bash\n"
                f"touch {nvidia_smi_marker!s}\n"
                "exit 99\n",
                encoding="utf-8",
            )
            os.chmod(fake_nvidia_smi, 0o755)

            env = os.environ.copy()
            env.update({
                "ARTIFACT_DIR": str(root / "out"),
                "PRINT_EFFECTIVE_COMMANDS": "1",
                "ONE_DEVS": "3",
                "MANY_DEVS": "2,4",
                "ONE_WORKERS_PER_DEVICE": "2",
                "MANY_WORKERS_PER_DEVICE": "3",
                "POLICY_DEVICE": "cpu",
                "DYNAMIC_ASSIGNMENT": "0",
                "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            })
            proc = subprocess.run(
                ["bash", "scripts/stage2_ngpu_speed_ab.sh"],
                cwd=str(_REPO),
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            nvidia_smi_called = nvidia_smi_marker.exists()

        output = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 0, output)
        self.assertFalse(nvidia_smi_called, output)
        command_lines = [
            line for line in output.splitlines()
            if line.startswith("[ab][effective]")
        ]
        self.assertEqual(len(command_lines), 2, output)
        one_command = next(line for line in command_lines if " one " in line)
        many_command = next(
            line for line in command_lines if " many " in line
        ).replace("\\,", ",")
        self.assertIn("CUDA_VISIBLE_DEVICES=3", one_command)
        self.assertIn("BLB_STAGE2_POLICY_DEVICE=cpu", one_command)
        self.assertIn("BLB_STAGE2_DYNAMIC_ASSIGNMENT=0", one_command)
        self.assertIn("--blb-v3-reward-devices 0", one_command)
        self.assertIn("--stage2-workers-per-device 2", one_command)
        self.assertIn("CUDA_VISIBLE_DEVICES=2,4", many_command)
        self.assertIn("BLB_STAGE2_POLICY_DEVICE=cpu", many_command)
        self.assertIn("BLB_STAGE2_DYNAMIC_ASSIGNMENT=0", many_command)
        self.assertIn("--blb-v3-reward-devices 0,1", many_command)
        self.assertIn("--stage2-workers-per-device 3", many_command)
        for expected in (
            "--stage2-fixed-config-source all4",
            "--blb-v3-decision-granularity layer",
            "--blb-v3-reward-design robust_constrained",
            "--batch-size 64",
            "--stage2-rollout-size 120",
            "--stage2-search-lr 5e-5",
            "--stage2-limit-tolerance 0.001",
            "--stage2-stability-tolerance 1.2",
            "--stage2-stability-multiplier 2.0",
            "--stage2-calibrate-baseline-samples 8",
            "--blb-v3-online-k-trials 5",
            "--blb-v3-baseline-groups 5",
            "--blb-v3-baseline-trials-per-group 5",
            "--blb-v3-constraint-bootstrap-samples 4096",
            "--blb-v3-online-constraint-probability 0.50",
            "--blb-v3-promotion-constraint-probability 0.80",
            "--blb-v3-final-constraint-probability 0.95",
            "--stage2-save-interval 200",
            "--stage2-eval-interval 100",
            "--random-seed 42",
        ):
            self.assertIn(expected, output)
        for stale in (
            "--stage2-rl-devices",
            "--blb-v3-fusion-neighbor-curriculum",
            "--blb-v3-warmstart-anchor-episodes",
            "--blb-v3-fusion-probe-interval",
            "--blb-v3-fusion-exploration-epsilon",
            "--batch-size 512",
            "--stage2-limit-tolerance 0.005",
            "--stage2-stability-tolerance 5.0",
        ):
            self.assertNotIn(stale, output)

    def test_rejects_malformed_candidate_before_ab_runner(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            scripts = root / "scripts"
            scripts.mkdir()
            shutil.copy2(
                _REPO / "scripts" / "stage2_ngpu_speed_targeted_first.sh",
                scripts / "stage2_ngpu_speed_targeted_first.sh",
            )
            (scripts / "stage2_ngpu_speed_ab.sh").write_text(
                "#!/usr/bin/env bash\n"
                "echo '[stub][FATAL] expensive A/B runner should not be reached'\n"
                "exit 99\n",
                encoding="utf-8",
            )
            os.chmod(scripts / "stage2_ngpu_speed_ab.sh", 0o755)

            env = os.environ.copy()
            env.update({
                "TARGET_ROOT": str(root / "out"),
                "TARGET_CANDIDATES": "badtuple",
                "FALLBACK_AUTOTUNE": "0",
            })
            proc = subprocess.run(
                ["bash", "scripts/stage2_ngpu_speed_targeted_first.sh"],
                cwd=str(root),
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

        output = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 24, output)
        self.assertIn("malformed TARGET_CANDIDATES entry", output)
        self.assertNotIn("expensive A/B runner should not be reached", output)


if __name__ == "__main__":
    unittest.main()
