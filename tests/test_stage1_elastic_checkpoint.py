from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - local macOS may be torch-free.
    torch = None


@unittest.skipIf(torch is None, "torch is required for Stage-1 checkpoint tests")
class Stage1ElasticCheckpointTests(unittest.TestCase):
    def test_detail_files_recover_to_committed_boundary(self):
        from noise_rl_module_v2 import (
            recover_stage1_detail_files,
            stage1_detail_file_sizes,
        )

        with tempfile.TemporaryDirectory() as td:
            details = Path(td)
            first = details / "ppo_step_info_1-360.txt"
            first.write_bytes(b"committed")
            committed = stage1_detail_file_sizes(details)

            first.write_bytes(b"committed-uncommitted")
            extra = details / "ppo_step_info_361-720.txt"
            extra.write_bytes(b"uncommitted")
            unrelated = details / "notes.txt"
            unrelated.write_bytes(b"preserve")

            recover_stage1_detail_files(details, committed)

            self.assertEqual(first.read_bytes(), b"committed")
            self.assertFalse(extra.exists())
            self.assertEqual(unrelated.read_bytes(), b"preserve")

    def test_checkpoint_round_trip_preserves_artifact_transaction(self):
        from noise_rl_module_v2 import (
            load_stage1_rl_checkpoint,
            save_stage1_rl_checkpoint,
        )

        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "stage1.pt")
            model = torch.nn.Linear(2, 2)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            save_stage1_rl_checkpoint(
                path=path,
                gtrxl_net=model,
                optimizer=optimizer,
                episode=119,
                gtrxl_ppo_update_count=1,
                episode_rewards=[1.0],
                episode_losses=[0.5],
                episode_metric1s=[0.8],
                episode_metric2s=[0.7],
                episode_entropies=[0.6],
                best_reward=1.0,
                best_cost=2.0,
                best_config=None,
                search_best_config=None,
                global_best_config=None,
                window_best_reward=float("-inf"),
                window_best_cost=float("inf"),
                window_best_config=None,
                ev_runtime_state={},
                stage1_prev_avg_reward=1.0,
                stage1_warnings=[],
                structured_run_id="stage1-run",
                structured_jsonl_sizes={
                    "steps.jsonl": 101,
                    "episodes.jsonl": 202,
                    "ppo_updates.jsonl": 303,
                },
                detail_file_sizes={"ppo_step_info_1-360.txt": 404},
            )

            restored_model = torch.nn.Linear(2, 2)
            restored_optimizer = torch.optim.Adam(
                restored_model.parameters(),
                lr=1e-3,
            )
            checkpoint = load_stage1_rl_checkpoint(
                path,
                restored_model,
                restored_optimizer,
                device="cpu",
            )

        self.assertEqual(checkpoint["version"], 2)
        self.assertEqual(checkpoint["structured_run_id"], "stage1-run")
        self.assertEqual(
            checkpoint["structured_jsonl_sizes"],
            {
                "steps.jsonl": 101,
                "episodes.jsonl": 202,
                "ppo_updates.jsonl": 303,
            },
        )
        self.assertEqual(
            checkpoint["detail_file_sizes"],
            {"ppo_step_info_1-360.txt": 404},
        )


if __name__ == "__main__":
    unittest.main()
