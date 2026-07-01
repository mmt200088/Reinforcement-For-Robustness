import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = REPO_ROOT / "scripts" / "optimization_evidence_bundle.py"


def _load_bundle_module():
    spec = importlib.util.spec_from_file_location("optimization_evidence_bundle", BUNDLE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_stage2_progress(progress: Path) -> None:
    diagnostics = progress / "diagnostics"
    details = progress.parent / "details"
    diagnostics.mkdir(parents=True, exist_ok=True)
    details.mkdir(parents=True, exist_ok=True)
    _write_text(
        progress / "blb_stage2_status.json",
        json.dumps(
            {
                "schema": "blb_stage2_status_v1",
                "completed_episodes": 1,
                "ppo_update_count": 1,
            }
        )
        + "\n",
    )
    _write_text(progress / "blb_stage2_live_summary.md", "# live\n")
    (progress / "blb_stage2_training_curve.npz").write_bytes(b"npz")
    _write_text(progress / "diagnostics" / "diagnostics_summary.md", "# diag\n")
    _write_text(details / "noise_ppo_step_info_1-1.txt", "detail\n")
    episode = {
        "episode": 0,
        "total_reward": 1.0,
        "terminal_reward": 0.5,
        "valid_steps": 47,
        "invalid_steps": 0,
        "total_bits": 123,
        "fusion_count": 1,
        "terminal_priority": 3,
        "terminal_loss_mean": 0.35,
        "terminal_metric1_mean": 0.88,
        "terminal_metric2_mean": 0.87,
        "terminal_probe_devices": ["cuda:0", "cuda:1"],
        "terminal_probe_trial_counts": [1, 1],
        "terminal_probe_wall_seconds": 1.0,
        "policy_rollout_wall_seconds": 0.2,
        "per_step_optimizer_wall_seconds": 0.1,
    }
    _write_text(diagnostics / "episodes.jsonl", json.dumps(episode) + "\n")
    update = {
        "update": 1,
        "completed_episodes": 1,
        "policy_loss": 0.1,
        "value_loss": 0.2,
        "entropy": 1.0,
        "clip_fraction": 0.01,
        "n_samples": 47,
        "window_mean_return": 1.0,
        "best_reward_so_far": 1.0,
        "elapsed_sec": 5.0,
    }
    _write_text(diagnostics / "ppo_updates.jsonl", json.dumps(update) + "\n")


class OptimizationEvidenceBundleTest(unittest.TestCase):
    def test_cli_generates_bundle_manifest_and_reports(self):
        bundle = _load_bundle_module()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out_dir = root / "bundle"
            stage1_log = root / "stage1.log"
            _write_text(
                stage1_log,
                "  [stage1-rollout] window=0 eps_per_worker=1  "
                "devices=[cuda:0, cuda:1] counts=[1, 1]  "
                "wall=2.000s worker_seconds=[2.000, 2.000]  speedup=2.00x\n"
                "  [stage1-rollout] window=0 eval_cache hits=1 misses=1 "
                "distinct=1 hit_rate=50.0%\n"
                "  [stage1-rollout-total] window=0 episodes=2 total=4.000s "
                "collect=2.000s replay=0.500s detail=0.500s "
                "ppo_update=0.500s other=0.500s throughput=1800.0ep/h\n",
            )
            progress = root / "run" / "stage2_noise" / "progress"
            _write_stage2_progress(progress)

            rc = bundle.main(
                [
                    "--root",
                    str(root),
                    "--out-dir",
                    str(out_dir),
                    "--stage1-log",
                    str(stage1_log),
                    "--stage2-episodes",
                    str(progress / "diagnostics" / "episodes.jsonl"),
                    "--visible-devices",
                    "0,1",
                    "--stage2-progress-dir",
                    str(progress),
                    "--min-episodes",
                    "1",
                    "--min-ppo-updates",
                    "1",
                ]
            )

            self.assertEqual(rc, 0)
            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "ok")
            self.assertIn("server_resource_snapshot", manifest)
            self.assertEqual(manifest["stage2_persistent_verify"]["returncode"], 0)
            self.assertTrue((out_dir / "server_resource_snapshot.json").is_file())
            self.assertTrue((out_dir / "project_optimization_audit.json").is_file())
            self.assertTrue((out_dir / "stage1_parallel_report.json").is_file())
            self.assertTrue((out_dir / "stage2_gpu_utilization_report.json").is_file())
            index = (out_dir / "index.md").read_text(encoding="utf-8")
            self.assertIn("# Optimization Evidence Bundle", index)
            self.assertIn("server_resource_snapshot.md", index)
            self.assertIn("stage1_parallel_report.md", index)
            self.assertIn("stage2_gpu_utilization_report.md", index)


if __name__ == "__main__":
    unittest.main()
