from __future__ import annotations

from collections import Counter
import threading
from types import SimpleNamespace
import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - local macOS may be torch-free.
    torch = None


@unittest.skipIf(torch is None, "torch is required for Stage-1 elastic runner tests")
class Stage1ElasticParallelRunnerTests(unittest.TestCase):
    @staticmethod
    def _worker(worker_idx: int, *, role: str):
        from stage1_rl.parallel_runner import Stage1RolloutWorker

        return Stage1RolloutWorker(
            worker_idx=worker_idx,
            device=torch.device("cpu"),
            model=torch.nn.Linear(1, 1),
            handler=None,
            evaluator=SimpleNamespace(),
            env=None,
            eval_split_name="validation_full",
            role=role,
        )

    @staticmethod
    def _rollout(seed: int):
        from stage1_rl.parallel_runner import EpisodeRollout

        return EpisodeRollout(
            cont_features=[],
            layer_indices=[],
            prev_g_actions=[],
            actions_g=[int(seed)],
            logprobs=[],
            rewards=[float(seed)],
            values=[],
            dones=[],
            gelu_masks=[],
            episode_reward=float(seed),
            episode_loss=0.0,
            episode_metric1=0.0,
            episode_metric2=0.0,
            episode_cost=0.0,
            gelu_config=[],
            softmax_config=[],
        )

    def test_replica_failure_retries_only_missing_episode_in_global_order(self):
        from stage1_rl.parallel_runner import Stage1ParallelRunner
        from stage1_rl.seed_utils import derive_episode_seed

        first_call_barrier = threading.Barrier(2)
        first_call_workers = set()
        state_lock = threading.Lock()
        failed_once = False
        failed_seed = None
        calls = Counter()

        def collect_episode(*, worker, episode_seed):
            nonlocal failed_once, failed_seed
            with state_lock:
                first_for_worker = worker.worker_idx not in first_call_workers
                first_call_workers.add(worker.worker_idx)
                calls[int(episode_seed)] += 1
            if first_for_worker:
                first_call_barrier.wait(timeout=2.0)
            if worker.role == "replica" and not failed_once:
                with state_lock:
                    if not failed_once:
                        failed_once = True
                        failed_seed = int(episode_seed)
                        raise RuntimeError("CUDA error: unknown error")
            return self._rollout(int(episode_seed))

        runner = Stage1ParallelRunner(
            workers=[
                self._worker(0, role="primary"),
                self._worker(1, role="replica"),
            ],
            primary_device=torch.device("cpu"),
            collect_episode_fn=collect_episode,
        )
        rollouts = runner.run_window(
            gtrxl_net=torch.nn.Linear(1, 1),
            total_episodes=6,
            window_idx=3,
            base_seed=42,
        )

        expected_seeds = [
            derive_episode_seed(42, 3, global_episode)
            for global_episode in range(6)
        ]
        self.assertEqual(
            [rollout.actions_g[0] for rollout in rollouts],
            expected_seeds,
        )
        self.assertEqual(runner.num_workers, 1)
        self.assertEqual(runner.pool_generation, 1)
        self.assertEqual(len(runner.quarantine_events), 1)
        self.assertEqual(calls[failed_seed], 2)
        for seed in expected_seeds:
            self.assertEqual(calls[seed], 2 if seed == failed_seed else 1)

        deferred = runner.pop_deferred_gpu_failure()
        self.assertIsNotNone(deferred)
        self.assertEqual(deferred.role, "rollout-replica")
        self.assertIsNone(runner.pop_deferred_gpu_failure())

    def test_primary_failure_requests_checkpoint_restart(self):
        from elastic_gpu import ElasticGPUFailure
        from stage1_rl.parallel_runner import Stage1ParallelRunner

        def collect_episode(**_kwargs):
            raise RuntimeError("CUDA error: device is lost")

        runner = Stage1ParallelRunner(
            workers=[self._worker(0, role="primary")],
            primary_device=torch.device("cpu"),
            collect_episode_fn=collect_episode,
        )

        with self.assertRaises(ElasticGPUFailure) as raised:
            runner.run_window(
                gtrxl_net=torch.nn.Linear(1, 1),
                total_episodes=1,
                window_idx=0,
                base_seed=42,
            )
        self.assertEqual(raised.exception.role, "learner-primary")

    def test_scientific_worker_error_remains_fatal(self):
        from stage1_rl.parallel_runner import Stage1ParallelRunner

        original = ValueError("trial seed mismatch")

        def collect_episode(**_kwargs):
            raise original

        runner = Stage1ParallelRunner(
            workers=[
                self._worker(0, role="primary"),
                self._worker(1, role="replica"),
            ],
            primary_device=torch.device("cpu"),
            collect_episode_fn=collect_episode,
        )

        with self.assertRaises(ValueError) as raised:
            runner.run_window(
                gtrxl_net=torch.nn.Linear(1, 1),
                total_episodes=2,
                window_idx=0,
                base_seed=42,
            )
        self.assertIs(raised.exception, original)


if __name__ == "__main__":
    unittest.main()
