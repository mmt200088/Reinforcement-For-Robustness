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
