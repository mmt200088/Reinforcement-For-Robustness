import os
import unittest
import time
from types import SimpleNamespace

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - local macOS env may be torch-free.
    torch = None


@unittest.skipIf(torch is None, "torch is required for Stage-2 parallel runner tests")
class Stage2ParallelRunnerPolicySyncTests(unittest.TestCase):
    def test_policy_replicas_are_reused_and_aux_state_is_synced(self):
        from blb_stage2_rl.parallel_runner import (
            Stage2FusionWorker,
            Stage2ParallelRunner,
        )
        from blb_stage2_rl.sequential_policy import (
            BLBStage2SequentialPolicy,
            SequentialPolicyConfig,
        )

        cfg = SequentialPolicyConfig(
            state_dim=16,
            max_step_dim=2,
            max_num_levels=3,
            d_model=16,
            n_heads=2,
            n_layers=1,
            d_ff=32,
            step_embed_dim=4,
            layer_embed_dim=4,
            block_embed_dim=2,
            prev_action_embed_dim=2,
            cont_proj_dim=8,
            actor_dim=8,
            critic_dim=8,
            horizon=4,
            num_layers=2,
        )
        policy = BLBStage2SequentialPolicy(cfg)
        policy.return_normalizer.mean = 1.25
        policy.return_normalizer.var = 2.5
        policy.return_normalizer.count = 7.0
        policy._ppo_lr_scale = 0.75
        policy._ppo_last_avg_kl = 0.125
        policy.default_prior_scale = 0.5
        policy.apply_preferred_per_step_bias([1, 2], gain=0.5)

        runner = Stage2ParallelRunner(
            workers=[
                Stage2FusionWorker(device=torch.device("cpu"), seq_env=None),
                Stage2FusionWorker(
                    device=torch.device("cpu"),
                    seq_env=None,
                    role="replica",
                ),
            ]
        )

        runner._sync_policy_replicas(policy)
        first_ids = [id(w.policy_replica) for w in runner.workers]
        self.assertIs(runner.workers[0].policy_replica, policy)

        with torch.no_grad():
            for param in policy.parameters():
                param.add_(0.01)
            policy.return_normalizer.mean = 3.0
            policy.return_normalizer.var = 4.0
            policy.return_normalizer.count = 9.0
            policy._ppo_lr_scale = 0.33
            policy._ppo_last_avg_kl = 0.44
            policy.default_prior_scale = 0.25
            policy.apply_preferred_per_step_bias([2, 1], gain=0.25)

        runner._sync_policy_replicas(policy)
        second_ids = [id(w.policy_replica) for w in runner.workers]

        self.assertEqual(first_ids, second_ids)
        self.assertIs(runner.workers[0].policy_replica, policy)
        for worker in runner.workers:
            replica = worker.policy_replica
            self.assertIsNotNone(replica)
            self.assertFalse(replica.training)
            for key, tensor in policy.state_dict().items():
                self.assertTrue(torch.equal(replica.state_dict()[key], tensor), key)
            self.assertEqual(replica.return_normalizer.mean, 3.0)
            self.assertEqual(replica.return_normalizer.var, 4.0)
            self.assertEqual(replica.return_normalizer.count, 9.0)
            self.assertEqual(replica._ppo_lr_scale, 0.33)
            self.assertEqual(replica._ppo_last_avg_kl, 0.44)
            self.assertEqual(replica.default_prior_scale, 0.25)
            self.assertEqual(replica._preferred_per_slot_idx.tolist(), [2, 1])

    def test_policy_device_can_be_decoupled_from_worker_probe_device(self):
        from blb_stage2_rl.parallel_runner import (
            Stage2FusionWorker,
            Stage2ParallelRunner,
        )
        from blb_stage2_rl.sequential_policy import (
            BLBStage2SequentialPolicy,
            SequentialPolicyConfig,
        )

        cfg = SequentialPolicyConfig(
            state_dim=8,
            max_step_dim=2,
            max_num_levels=3,
            d_model=8,
            n_heads=2,
            n_layers=1,
            d_ff=16,
            step_embed_dim=2,
            layer_embed_dim=2,
            block_embed_dim=2,
            prev_action_embed_dim=2,
            cont_proj_dim=4,
            actor_dim=4,
            critic_dim=4,
            horizon=2,
            num_layers=1,
        )
        policy = BLBStage2SequentialPolicy(cfg)
        runner = Stage2ParallelRunner(
            workers=[
                Stage2FusionWorker(
                    device=torch.device("cuda:0"),
                    policy_device=torch.device("cpu"),
                    seq_env=None,
                    role="primary",
                ),
                Stage2FusionWorker(
                    device=torch.device("cuda:1"),
                    policy_device=torch.device("cpu"),
                    seq_env=None,
                    role="replica",
                ),
            ]
        )

        runner._sync_policy_replicas(policy)

        for worker in runner.workers:
            replica = worker.policy_replica
            self.assertIsNotNone(replica)
            self.assertEqual(next(replica.parameters()).device.type, "cpu")

    def test_record_full_vec_prefers_parallel_outcome_over_primary_env(self):
        from blb_stage2_rl.sequential_runner import (
            EpisodeRecord,
            _attach_pending_full_vec_for_callback,
            _record_full_vec_for_callback,
        )

        record = EpisodeRecord(
            episode_idx=0,
            total_reward=0.0,
            terminal_reward=0.0,
            per_step_reward_sum=0.0,
            invalid_steps=0,
            early_terminated=False,
            steps_taken=0,
        )
        primary_env = SimpleNamespace(
            _pending_full_vec=np.asarray([9, 9, 9], dtype=np.int64),
        )
        worker_vec = np.asarray([1, 2, 3], dtype=np.int64)

        _attach_pending_full_vec_for_callback(record, worker_vec)
        worker_vec[:] = 0
        resolved = _record_full_vec_for_callback(record, primary_env)

        self.assertEqual(resolved.tolist(), [1, 2, 3])
        self.assertFalse(np.shares_memory(resolved, worker_vec))

    def test_record_full_vec_falls_back_to_primary_env_for_serial_path(self):
        from blb_stage2_rl.sequential_runner import (
            EpisodeRecord,
            _record_full_vec_for_callback,
        )

        record = EpisodeRecord(
            episode_idx=0,
            total_reward=0.0,
            terminal_reward=0.0,
            per_step_reward_sum=0.0,
            invalid_steps=0,
            early_terminated=False,
            steps_taken=0,
        )
        primary_vec = np.asarray([4, 5, 6], dtype=np.int64)
        primary_env = SimpleNamespace(_pending_full_vec=primary_vec)

        resolved = _record_full_vec_for_callback(record, primary_env)
        primary_vec[:] = 0

        self.assertEqual(resolved.tolist(), [4, 5, 6])
        self.assertFalse(np.shares_memory(resolved, primary_vec))

    def test_run_window_uses_interleaved_episode_assignment(self):
        from blb_stage2_rl import parallel_runner as pr
        from blb_stage2_rl.parallel_runner import (
            FusionEpisodeOutcome,
            Stage2FusionWorker,
            Stage2ParallelRunner,
        )
        from blb_stage2_rl.sequential_runner import EpisodeRecord

        seen = []
        original_collect = pr.collect_fusion_episode

        def fake_collect(**kwargs):
            worker_name = kwargs["seq_env"].name
            rel_ep = int(kwargs["rel_ep"])
            seen.append((worker_name, rel_ep))
            return FusionEpisodeOutcome(
                rel_ep=rel_ep,
                absolute_ep=int(kwargs["absolute_ep"]),
                transitions=[],
                record=EpisodeRecord(
                    episode_idx=rel_ep,
                    total_reward=0.0,
                    terminal_reward=0.0,
                    per_step_reward_sum=0.0,
                    invalid_steps=0,
                    early_terminated=False,
                    steps_taken=0,
                ),
                pending_full_vec=None,
            )

        policy = torch.nn.Linear(1, 1)
        logs = []
        runner = Stage2ParallelRunner(
            workers=[
                Stage2FusionWorker(device=torch.device("cpu"), seq_env=SimpleNamespace(name="w0")),
                Stage2FusionWorker(device=torch.device("cpu"), seq_env=SimpleNamespace(name="w1"), role="replica"),
                Stage2FusionWorker(device=torch.device("cpu"), seq_env=SimpleNamespace(name="w2"), role="replica"),
            ],
            log_fn=logs.append,
            emit_rollout_signature=False,
            dynamic_assignment=False,
        )
        pr.collect_fusion_episode = fake_collect
        try:
            outcomes = runner.run_window(
                policy=policy,
                train_cfg=SimpleNamespace(seed=1),
                window_rel_start=10,
                num_episodes=8,
                absolute_episode_start=1000,
                base_seed=1,
                baseline_action_vec=np.zeros(1, dtype=np.int64),
                force_baseline_episodes=0,
                forbidden_mask=None,
            )
        finally:
            pr.collect_fusion_episode = original_collect

        by_worker = {}
        for worker_name, rel_ep in seen:
            by_worker.setdefault(worker_name, []).append(rel_ep)
        self.assertEqual(by_worker["w0"], [10, 13, 16])
        self.assertEqual(by_worker["w1"], [11, 14, 17])
        self.assertEqual(by_worker["w2"], [12, 15])
        self.assertEqual([oc.rel_ep for oc in outcomes], list(range(10, 18)))
        self.assertTrue(any("rollout_sig=disabled" in line for line in logs))

    def test_parallel_runner_defaults_to_dynamic_assignment(self):
        from blb_stage2_rl.parallel_runner import (
            Stage2FusionWorker,
            Stage2ParallelRunner,
        )

        old = os.environ.pop("BLB_STAGE2_DYNAMIC_ASSIGNMENT", None)
        try:
            runner = Stage2ParallelRunner(
                workers=[
                    Stage2FusionWorker(device=torch.device("cpu"), seq_env=None),
                    Stage2FusionWorker(device=torch.device("cpu"), seq_env=None, role="replica"),
                ],
                emit_rollout_signature=False,
            )
        finally:
            if old is not None:
                os.environ["BLB_STAGE2_DYNAMIC_ASSIGNMENT"] = old

        self.assertTrue(runner.dynamic_assignment)

    def test_run_window_dynamic_assignment_keeps_global_order(self):
        from blb_stage2_rl import parallel_runner as pr
        from blb_stage2_rl.parallel_runner import (
            FusionEpisodeOutcome,
            Stage2FusionWorker,
            Stage2ParallelRunner,
        )
        from blb_stage2_rl.sequential_runner import EpisodeRecord

        seen = []
        original_collect = pr.collect_fusion_episode

        def fake_collect(**kwargs):
            worker_name = kwargs["seq_env"].name
            rel_ep = int(kwargs["rel_ep"])
            seen.append((worker_name, rel_ep))
            time.sleep(0.002)
            return FusionEpisodeOutcome(
                rel_ep=rel_ep,
                absolute_ep=int(kwargs["absolute_ep"]),
                transitions=[],
                record=EpisodeRecord(
                    episode_idx=rel_ep,
                    total_reward=0.0,
                    terminal_reward=0.0,
                    per_step_reward_sum=0.0,
                    invalid_steps=0,
                    early_terminated=False,
                    steps_taken=0,
                ),
                pending_full_vec=None,
            )

        logs = []
        runner = Stage2ParallelRunner(
            workers=[
                Stage2FusionWorker(device=torch.device("cpu"), seq_env=SimpleNamespace(name="w0")),
                Stage2FusionWorker(device=torch.device("cpu"), seq_env=SimpleNamespace(name="w1"), role="replica"),
                Stage2FusionWorker(device=torch.device("cpu"), seq_env=SimpleNamespace(name="w2"), role="replica"),
            ],
            log_fn=logs.append,
            emit_rollout_signature=False,
            dynamic_assignment=True,
        )
        pr.collect_fusion_episode = fake_collect
        try:
            outcomes = runner.run_window(
                policy=torch.nn.Linear(1, 1),
                train_cfg=SimpleNamespace(seed=1),
                window_rel_start=20,
                num_episodes=9,
                absolute_episode_start=1000,
                base_seed=1,
                baseline_action_vec=np.zeros(1, dtype=np.int64),
                force_baseline_episodes=0,
                forbidden_mask=None,
            )
        finally:
            pr.collect_fusion_episode = original_collect

        self.assertEqual(sorted(rel_ep for _worker, rel_ep in seen), list(range(20, 29)))
        self.assertGreater(len({worker for worker, _rel_ep in seen}), 1)
        self.assertEqual([oc.rel_ep for oc in outcomes], list(range(20, 29)))
        self.assertTrue(any("assignment=dynamic_queue" in line for line in logs))

    def test_rollout_signature_can_be_enabled_for_debugging(self):
        from blb_stage2_rl import parallel_runner as pr
        from blb_stage2_rl.parallel_runner import (
            FusionEpisodeOutcome,
            Stage2FusionWorker,
            Stage2ParallelRunner,
        )
        from blb_stage2_rl.sequential_runner import EpisodeRecord

        original_collect = pr.collect_fusion_episode

        def fake_collect(**kwargs):
            rel_ep = int(kwargs["rel_ep"])
            return FusionEpisodeOutcome(
                rel_ep=rel_ep,
                absolute_ep=int(kwargs["absolute_ep"]),
                transitions=[{
                    "action": np.asarray([rel_ep, 0], dtype=np.int64),
                    "reward": float(rel_ep),
                }],
                record=EpisodeRecord(
                    episode_idx=rel_ep,
                    total_reward=0.0,
                    terminal_reward=float(rel_ep),
                    per_step_reward_sum=0.0,
                    invalid_steps=0,
                    early_terminated=False,
                    steps_taken=0,
                    terminal_priority=3,
                    terminal_loss_mean=0.1,
                    terminal_metric1_mean=0.9,
                ),
                pending_full_vec=None,
            )

        logs = []
        runner = Stage2ParallelRunner(
            workers=[
                Stage2FusionWorker(device=torch.device("cpu"), seq_env=SimpleNamespace(name="w0")),
            ],
            log_fn=logs.append,
            emit_rollout_signature=True,
        )
        pr.collect_fusion_episode = fake_collect
        try:
            runner.run_window(
                policy=torch.nn.Linear(1, 1),
                train_cfg=SimpleNamespace(seed=1),
                window_rel_start=0,
                num_episodes=2,
                absolute_episode_start=1000,
                base_seed=1,
                baseline_action_vec=np.zeros(1, dtype=np.int64),
                force_baseline_episodes=0,
                forbidden_mask=None,
            )
        finally:
            pr.collect_fusion_episode = original_collect

        self.assertTrue(any("rollout_sig=" in line and "disabled" not in line for line in logs))

    def test_terminal_snapshot_preserves_fusion_action_steps(self):
        from blb_stage2_rl.parallel_runner import (
            _default_terminal_snapshot,
            _update_terminal_snapshot,
        )

        snapshot = _default_terminal_snapshot()
        action_steps = [
            {
                "step_idx": 0,
                "layer_idx": 3,
                "block_idx": 2,
                "graph_key": "block2_mrpc",
                "option_id": 1,
                "fusion_count": 1,
                "k_index": 2,
                "k_value": 11,
                "valid": True,
            },
            "ignore-non-mapping",
        ]

        _update_terminal_snapshot(
            snapshot,
            {"terminal_info": {"fusion_action_steps": action_steps}},
        )
        action_steps[0]["option_id"] = 99

        self.assertEqual(len(snapshot["fusion_action_steps"]), 1)
        self.assertEqual(snapshot["fusion_action_steps"][0]["option_id"], 1)
        self.assertEqual(snapshot["fusion_action_steps"][0]["k_value"], 11)


if __name__ == "__main__":
    unittest.main()
