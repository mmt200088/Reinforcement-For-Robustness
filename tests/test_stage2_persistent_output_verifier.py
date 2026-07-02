from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from scripts import verify_stage2_persistent_outputs as verifier

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_minimal_progress(
        progress_dir: Path,
        *,
        include_live_summary: bool = True,
        details_dir: Path | None = None,
) -> None:
    progress_dir.mkdir(parents=True, exist_ok=True)
    (progress_dir / "diagnostics").mkdir()
    details_dir = details_dir or (progress_dir / "details")
    details_dir.mkdir(parents=True, exist_ok=True)

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
        "\n".join(
            json.dumps(
                {
                    "episode": i,
                    "total_reward": float(i),
                    "terminal_reward": float(i) / 2.0,
                    "valid_steps": 47,
                    "invalid_steps": 0,
                    "total_bits": 123,
                    "fusion_count": 1,
                    "terminal_priority": 3,
                    "terminal_loss_mean": 0.35,
                    "terminal_metric1_mean": 0.88,
                    "terminal_metric2_mean": 0.87,
                    "policy_rollout_wall_seconds": 0.2,
                    "per_step_optimizer_wall_seconds": 0.1,
                }
            )
            for i in range(3)
        )
        + "\n",
        encoding="utf-8",
    )
    (progress_dir / "diagnostics" / "ppo_updates.jsonl").write_text(
        json.dumps(
            {
                "update": 1,
                "completed_episodes": 3,
                "policy_loss": 0.1,
                "value_loss": 0.2,
                "entropy": 1.0,
                "clip_fraction": 0.01,
                "n_samples": 141,
                "window_mean_return": 1.0,
                "best_reward_so_far": 2.0,
                "elapsed_sec": 12.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (progress_dir / "diagnostics" / "diagnostics_summary.md").write_text(
        "# diagnostics\n",
        encoding="utf-8",
    )
    (details_dir / "noise_ppo_step_info_1-3.txt").write_text(
        "episode details\n",
        encoding="utf-8",
    )


class Stage2PersistentOutputVerifierTest(unittest.TestCase):
    def test_missing_required_fields_returns_none_for_complete_rows(self):
        payload = {
            "episode": 0,
            "total_reward": 1.0,
            "terminal_reward": 0.5,
        }

        self.assertIsNone(
            verifier._missing_required_fields(
                payload,
                ("episode", "total_reward", "terminal_reward"),
            )
        )
        self.assertEqual(
            verifier._missing_required_fields(
                payload,
                ("episode", "total_reward", "missing_a", "missing_b"),
            ),
            ("missing_a", "missing_b"),
        )

    def test_required_field_counter_skips_blank_lines_without_strip_copy(self):
        class NoStripLine(str):
            def strip(self, *_args, **_kwargs):
                raise AssertionError("JSONL counter should not allocate strip() copies")

        class FakeHandle:
            def __init__(self):
                self.lines = [
                    NoStripLine('{"episode": 0, "total_reward": 1.0}\n'),
                    NoStripLine("   \n"),
                    NoStripLine('{"episode": 1, "total_reward": 2.0}\n'),
                ]

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def __iter__(self):
                return iter(self.lines)

        class FakePath:
            def open(self, *_args, **_kwargs):
                return FakeHandle()

        count, failures = verifier._count_jsonl_with_required_fields(
            FakePath(),
            ("episode", "total_reward"),
            "episodes.jsonl",
        )

        self.assertEqual(count, 2)
        self.assertEqual(failures, [])

    def test_detail_file_count_does_not_sort_or_materialize_files(self):
        with tempfile.TemporaryDirectory(prefix="stage2_verify_details_") as td:
            progress_dir = Path(td) / "progress"
            details_dir = progress_dir / "details"
            details_dir.mkdir(parents=True)
            for idx in range(5):
                (details_dir / f"batch_{idx}.txt").write_text("details\n", encoding="utf-8")

            with mock.patch(
                "builtins.sorted",
                side_effect=AssertionError("detail check should count files without sorting"),
            ):
                count, detail_dirs = verifier._latest_detail_files(
                    progress_dir,
                    Namespace(run_dir=""),
                )

        self.assertEqual(count, 5)
        self.assertEqual(detail_dirs, [details_dir])

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

    def test_verifier_accepts_details_sibling_of_progress_dir(self):
        with tempfile.TemporaryDirectory(prefix="stage2_verify_real_layout_") as td:
            run_dir = Path(td) / "persistent" / "rl" / "bert-base" / "mrpc" / "s1t0.001_s2t0.001_s2st3.0__smoke"
            progress_dir = run_dir / "stage2_noise" / "progress"
            _write_minimal_progress(
                progress_dir,
                details_dir=run_dir / "stage2_noise" / "details",
            )

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
        self.assertIn("details files=1", result.stdout)

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

    def test_verifier_fails_when_episode_required_fields_are_missing(self):
        with tempfile.TemporaryDirectory(prefix="stage2_verify_missing_fields_") as td:
            run_dir = Path(td) / "persistent" / "rl" / "bert-base" / "mrpc" / "s1t0.001_s2t0.001_s2st3.0__smoke"
            progress_dir = run_dir / "stage2_noise" / "progress"
            _write_minimal_progress(progress_dir)
            (progress_dir / "diagnostics" / "episodes.jsonl").write_text(
                json.dumps({"episode": 0, "total_reward": 1.0}) + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/verify_stage2_persistent_outputs.py",
                    "--run-dir",
                    str(run_dir),
                    "--min-episodes",
                    "1",
                ],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("episodes.jsonl missing required fields", result.stdout + result.stderr)
        self.assertIn("terminal_reward", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
