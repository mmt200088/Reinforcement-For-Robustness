from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - local macOS may be torch-free.
    torch = None


@unittest.skipIf(torch is None, "torch is required for checkpoint comparison")
class ElasticRLScalingABTests(unittest.TestCase):
    @staticmethod
    def _write_jsonl(path: Path, rows) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    def _write_run(
        self,
        root: Path,
        data_root: Path,
        *,
        run_id: str,
        reward: float = 1.25,
        weight: float = 3.0,
        device: str = "cuda:0",
        timestamp: float = 1.0,
    ) -> None:
        diagnostics = root / "progress" / "diagnostics"
        self._write_jsonl(
            diagnostics / "episodes.jsonl",
            [{
                "episode": 0,
                "total_reward": reward,
                "terminal_probe_wall_seconds": timestamp,
                "terminal_probe_devices": [device],
                "terminal_probe_trial_indices": [[0, 1, 2, 3, 4]],
                "fresh_trials": {
                    "seeds": [11, 12, 13, 14, 15],
                    "loss": [0.1, 0.2, 0.3, 0.4, 0.5],
                },
            }],
        )
        self._write_jsonl(
            diagnostics / "ppo_updates.jsonl",
            [{
                "update": 1,
                "policy_loss": 0.75,
                "elapsed_sec": timestamp,
                "timestamp": timestamp,
            }],
        )
        self._write_jsonl(
            root / "progress" / "candidate_store.jsonl",
            [{
                "record_type": "candidate_trial_group_v1",
                "candidate_hash": "same-candidate",
                "trial_seeds": [11, 12, 13, 14, 15],
                "trial_values": [0.1, 0.2, 0.3, 0.4, 0.5],
                "created_at": f"timestamp-{timestamp}",
                "logical_generation": int(timestamp),
            }],
        )
        checkpoint = {
            "episode": 1,
            "policy": {"weight": torch.tensor([weight])},
            "optimizer": {"state": {0: {"step": torch.tensor(1.0)}}},
            "structured_run_id": run_id,
            "diagnostics_jsonl_sizes": {"episodes.jsonl": int(timestamp * 10)},
            "candidate_store_size": int(timestamp * 20),
            "store_file_fingerprints": {"candidate_store.jsonl": str(timestamp)},
            "cuda_rng_state_all": [torch.tensor([int(timestamp)])],
            "cuda_rng_state_by_role": [torch.tensor([int(timestamp)])],
            "cuda_rng_active_role_count": int(timestamp),
            "torch_rng_state": torch.tensor([9], dtype=torch.uint8),
            "numpy_rng_state": ("MT19937", [1, 2, 3], 0, 0, 0.0),
            "python_rng_state": (3, (1, 2, 3), None),
        }
        torch.save(
            checkpoint,
            root / "progress" / "blb_stage2_rl_checkpoint_live.pt",
        )

        structured = data_root / "stage2" / "bert-base" / "mrpc" / run_id
        self._write_jsonl(
            structured / "steps.jsonl",
            [{
                "episode": 0,
                "step": 0,
                "action": 2,
                "run_id": run_id,
                "device": device,
                "timestamp": timestamp,
            }],
        )
        self._write_jsonl(
            structured / "episodes.jsonl",
            [{
                "episode": 0,
                "reward": reward,
                "run_id": run_id,
                "pool_generation": int(timestamp),
                "retry_count": int(timestamp),
            }],
        )
        self._write_jsonl(
            structured / "ppo_updates.jsonl",
            [{
                "update": 1,
                "policy_loss": 0.75,
                "run_id": run_id,
                "elapsed_seconds": timestamp,
            }],
        )

    def test_strict_comparison_ignores_only_efficiency_telemetry(self):
        from scripts.elastic_rl_scaling_ab import compare_runs

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            data_root = base / "rl_training_data_points"
            control = base / "control"
            candidate = base / "candidate"
            self._write_run(
                control,
                data_root,
                run_id="control-run",
                device="cuda:0",
                timestamp=1.0,
            )
            self._write_run(
                candidate,
                data_root,
                run_id="candidate-run",
                device="cuda:3",
                timestamp=4.0,
            )

            result = compare_runs(
                control,
                candidate,
                stage="stage2",
                data_points_root=data_root,
            )

        self.assertTrue(result.equal, result.diffs)
        self.assertEqual(result.compared["diagnostic_episodes"], 1)
        self.assertEqual(result.compared["structured_steps"], 1)
        self.assertEqual(result.compared["candidate_records"], 1)
        self.assertTrue(result.compared["checkpoint"])

    def test_scientific_episode_change_fails(self):
        from scripts.elastic_rl_scaling_ab import compare_runs

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            data_root = base / "rl_training_data_points"
            control = base / "control"
            candidate = base / "candidate"
            self._write_run(control, data_root, run_id="control-run")
            self._write_run(
                candidate,
                data_root,
                run_id="candidate-run",
                reward=1.5,
                timestamp=2.0,
            )

            result = compare_runs(
                control,
                candidate,
                stage="stage2",
                data_points_root=data_root,
            )

        self.assertFalse(result.equal)
        self.assertTrue(
            any("total_reward" in diff or ".reward" in diff for diff in result.diffs),
            result.diffs,
        )

    def test_recursive_checkpoint_change_fails(self):
        from scripts.elastic_rl_scaling_ab import compare_runs

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            data_root = base / "rl_training_data_points"
            control = base / "control"
            candidate = base / "candidate"
            self._write_run(control, data_root, run_id="control-run")
            self._write_run(
                candidate,
                data_root,
                run_id="candidate-run",
                weight=3.5,
                timestamp=2.0,
            )

            result = compare_runs(
                control,
                candidate,
                stage="stage2",
                data_points_root=data_root,
            )

        self.assertFalse(result.equal)
        self.assertTrue(
            any("checkpoint.policy.weight" in diff for diff in result.diffs),
            result.diffs,
        )


if __name__ == "__main__":
    unittest.main()
