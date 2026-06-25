from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_minimal_progress(progress_dir: Path, *, include_live_summary: bool = True) -> None:
    progress_dir.mkdir(parents=True, exist_ok=True)
    (progress_dir / "diagnostics").mkdir()
    (progress_dir / "details").mkdir()

    (progress_dir / "blb_stage2_status.json").write_text(
        json.dumps(
            {
                "schema": "blb_stage2_status_v1",
                "completed_episodes": 3,
                "total_episodes": 170,
                "ppo_update_count": 1,
                "phase": "PPO training",
            }
        ),
        encoding="utf-8",
    )
    if include_live_summary:
        (progress_dir / "blb_stage2_live_summary.md").write_text(
            "# BLB Stage-2 RL Live Summary\n\n- Episode: 3 / 170\n",
            encoding="utf-8",
        )
    (progress_dir / "blb_stage2_training_curve.npz").write_bytes(b"npz")
    (progress_dir / "blb_stage2_training_curve.png").write_bytes(b"png")
    (progress_dir / "blb_stage2_entropy_curve.png").write_bytes(b"png")
    (progress_dir / "diagnostics" / "episodes.jsonl").write_text(
        "\n".join(json.dumps({"episode": i, "total_reward": float(i)}) for i in range(3)) + "\n",
        encoding="utf-8",
    )
    (progress_dir / "diagnostics" / "ppo_updates.jsonl").write_text(
        json.dumps({"update": 1, "completed_episodes": 3, "entropy": 1.0}) + "\n",
        encoding="utf-8",
    )
    (progress_dir / "diagnostics" / "diagnostics_summary.md").write_text(
        "# diagnostics\n",
        encoding="utf-8",
    )
    (progress_dir / "details" / "noise_ppo_step_info_1-3.txt").write_text(
        "episode details\n",
        encoding="utf-8",
    )


class Stage2PersistentOutputVerifierTest(unittest.TestCase):
    def test_verifier_accepts_complete_persistent_run_dir(self):
        with tempfile.TemporaryDirectory(prefix="stage2_verify_ok_") as td:
            run_dir = Path(td) / "persistent" / "rl" / "bert-base" / "mrpc" / "s1t0.001_s2t0.001_s2st3.0__smoke"
            _write_minimal_progress(run_dir / "stage2_noise" / "progress")

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/verify_stage2_persistent_outputs.py",
                    "--run-dir",
                    str(run_dir),
                    "--min-episodes",
                    "3",
                    "--min-ppo-updates",
                    "1",
                    "--require-png",
                ],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn("VERIFY_OK", result.stdout)
        self.assertIn("blb_stage2_live_summary.md", result.stdout)

    def test_verifier_fails_when_live_summary_is_missing(self):
        with tempfile.TemporaryDirectory(prefix="stage2_verify_bad_") as td:
            run_dir = Path(td) / "persistent" / "rl" / "bert-base" / "mrpc" / "s1t0.001_s2t0.001_s2st3.0__smoke"
            _write_minimal_progress(
                run_dir / "stage2_noise" / "progress",
                include_live_summary=False,
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/verify_stage2_persistent_outputs.py",
                    "--run-dir",
                    str(run_dir),
                    "--min-episodes",
                    "3",
                ],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("blb_stage2_live_summary.md", result.stdout + result.stderr)
        self.assertIn("VERIFY_FAIL", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
