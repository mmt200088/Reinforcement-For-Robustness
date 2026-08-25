from __future__ import annotations

import ast
import random
from pathlib import Path
import tempfile
import unittest

import numpy as np

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
        from rfr.search.rl.stage1.checkpoint import (
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
        from rfr.search.rl.stage1.checkpoint import (
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
                dataset_protocol_hash="probe-a",
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
                expected_dataset_protocol_hash="probe-a",
            )

        self.assertEqual(checkpoint["version"], 2)
        self.assertEqual(
            checkpoint["dataset_protocol_schema"],
            "glue_train_probe_protocol_v1",
        )
        self.assertEqual(checkpoint["dataset_protocol_hash"], "probe-a")
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

    def test_checkpoint_round_trip_restores_all_rng_roles(self):
        from rfr.search.rl.stage1.checkpoint import (
            load_stage1_rl_checkpoint,
            save_stage1_rl_checkpoint,
        )

        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "stage1.pt")
            model = torch.nn.Linear(2, 2)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            random.seed(1101)
            np.random.seed(2202)
            torch.manual_seed(3303)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(4404)

            expected_python = random.getstate()
            expected_numpy = np.random.get_state()
            expected_torch = torch.get_rng_state().clone()
            expected_cuda = (
                [state.clone() for state in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available()
                else []
            )

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
                dataset_protocol_hash="probe-a",
            )

            random.random()
            np.random.random()
            torch.rand(1)
            if torch.cuda.is_available():
                for device_index in range(torch.cuda.device_count()):
                    torch.rand(1, device=f"cuda:{device_index}")

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
                expected_dataset_protocol_hash="probe-a",
            )

            self.assertEqual(checkpoint["python_rng_state"], expected_python)
            self.assertEqual(checkpoint["numpy_rng_state"][0], expected_numpy[0])
            np.testing.assert_array_equal(
                checkpoint["numpy_rng_state"][1],
                expected_numpy[1],
            )
            self.assertTrue(torch.equal(checkpoint["torch_rng_state"], expected_torch))
            self.assertEqual(
                checkpoint["cuda_rng_active_role_count"],
                len(expected_cuda),
            )
            self.assertEqual(
                len(checkpoint["cuda_rng_state_by_role"]),
                len(expected_cuda),
            )
            self.assertEqual(random.getstate(), expected_python)
            current_numpy = np.random.get_state()
            self.assertEqual(current_numpy[0], expected_numpy[0])
            np.testing.assert_array_equal(current_numpy[1], expected_numpy[1])
            self.assertTrue(torch.equal(torch.get_rng_state(), expected_torch))
            if expected_cuda:
                for actual, expected in zip(
                    torch.cuda.get_rng_state_all(),
                    expected_cuda,
                ):
                    self.assertTrue(torch.equal(actual, expected))

    def test_checkpoint_protocol_guard_precedes_weight_and_rng_restore(self):
        source = (
            Path(__file__).resolve().parents[1] / "src/rfr/search/rl/stage1/checkpoint.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        node = next(
            item
            for item in tree.body
            if isinstance(item, ast.FunctionDef)
            and item.name == "load_stage1_rl_checkpoint"
        )
        method = ast.get_source_segment(source, node)
        self.assertIsNotNone(method)

        validation = method.index("validate_dataset_protocol_binding(")
        self.assertLess(validation, method.index("gtrxl_net.load_state_dict("))
        self.assertLess(validation, method.index("optimizer.load_state_dict("))
        self.assertLess(validation, method.index("torch.set_rng_state("))

    def test_cuda_rng_registry_retains_temporarily_absent_roles(self):
        from rfr.search.rl.stage1.checkpoint import (
            merge_stage1_cuda_rng_role_registry,
            resolve_stage1_cuda_rng_role_registry,
        )

        checkpoint = {
            "cuda_rng_role_registry_version": 1,
            "cuda_rng_state_by_role": ["r0", "r1", "r2", "r3", "r4"],
            "cuda_rng_active_role_count": 5,
        }
        registry, active = resolve_stage1_cuda_rng_role_registry(
            checkpoint,
            active_role_count=4,
            new_role_state_factory=lambda index: f"new-{index}",
        )
        self.assertEqual(active, ["r0", "r1", "r2", "r3"])

        registry = merge_stage1_cuda_rng_role_registry(
            registry,
            ["u0", "u1", "u2", "u3"],
        )
        self.assertEqual(registry, ["u0", "u1", "u2", "u3", "r4"])

        resumed_registry, resumed_active = resolve_stage1_cuda_rng_role_registry(
            {
                "cuda_rng_role_registry_version": 1,
                "cuda_rng_state_by_role": registry,
                "cuda_rng_active_role_count": 4,
            },
            active_role_count=5,
            new_role_state_factory=lambda index: f"new-{index}",
        )
        self.assertEqual(resumed_registry, ["u0", "u1", "u2", "u3", "r4"])
        self.assertEqual(resumed_active, ["u0", "u1", "u2", "u3", "r4"])

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
        boundary = 0
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
