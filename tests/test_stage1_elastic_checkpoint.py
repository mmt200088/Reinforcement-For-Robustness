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
    @staticmethod
    def _training_source():
        return (
            Path(__file__).resolve().parents[1]
            / "layer_importance_evaluator.py"
        ).read_text(encoding="utf-8")

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

    def test_resume_recovers_artifacts_before_first_episode(self):
        source = self._training_source()
        resume_load = source.index("ckpt = load_stage1_rl_checkpoint(")
        recover_jsonl = source.index(
            "stage1_data_writer.recover_jsonl_files(",
            resume_load,
        )
        recover_details = source.index(
            "recover_stage1_detail_files(",
            resume_load,
        )
        episode_loop = source.index("for episode in _stage1_episode_iter:")

        self.assertIn('ckpt.get("structured_run_id")', source)
        self.assertLess(resume_load, recover_jsonl)
        self.assertLess(resume_load, recover_details)
        self.assertLess(recover_jsonl, episode_loop)
        self.assertLess(recover_details, episode_loop)

    def test_full_ppo_checkpoint_precedes_elastic_restart(self):
        source = self._training_source()
        boundary = source.index("# 保存 Stage-1 checkpoint（断点续训用）")
        jsonl_sizes = source.index(
            "stage1_data_writer.committed_jsonl_sizes()",
            boundary,
        )
        detail_sizes = source.index(
            "stage1_detail_file_sizes(step_info_details_dir)",
            boundary,
        )
        save = source.index("save_stage1_rl_checkpoint(", jsonl_sizes)
        deferred_failure = source.index(
            "_stage1_parallel_runner.pop_deferred_gpu_failure()",
            save,
        )
        recovery_restart = source.index(
            "raise_if_elastic_gpu_restart_requested()",
            save,
        )

        self.assertLess(jsonl_sizes, save)
        self.assertLess(detail_sizes, save)
        self.assertLess(save, deferred_failure)
        self.assertLess(deferred_failure, recovery_restart)

    def test_ppo_cuda_failure_is_promoted_to_primary_restart(self):
        source = self._training_source()
        ppo_call = source.index(
            "policy_loss, value_loss, entropy = self.ppo_update_gtrxl("
        )
        recoverable_check = source.index(
            "is_recoverable_gpu_failure(",
            ppo_call,
        )
        primary_failure = source.index(
            'role="learner-primary"',
            recoverable_check,
        )

        self.assertLess(ppo_call, recoverable_check)
        self.assertLess(recoverable_check, primary_failure)


if __name__ == "__main__":
    unittest.main()
